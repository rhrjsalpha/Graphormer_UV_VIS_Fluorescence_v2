import math
import torch
import torch.nn as nn
import time


def init_params(module, n_layers):
    """
    Initialize parameters for Linear and Embedding layers.
    """
    if isinstance(module, nn.Linear):
        # print(f"Initializing Linear: weight {module.weight.shape}, bias {module.bias.shape if module.bias is not None else 'None'}")
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        # print(f"Initializing Embedding: weight {module.weight.shape}")
        module.weight.data.normal_(mean=0.0, std=0.02)
        # print(f"Weight after initialization: {module.weight.shape}")

def _meta_to_slices(meta, total_dim: int):
    """
    meta 포맷을 유연하게 받아 (start, end) 슬라이스 리스트로 변환.
    허용 포맷 예:
      - {'groups': [{'name':'atomic_num','start':0,'end':35}, ...]}
      - {'slices': [(0,35),(35,42),...]}
      - {'dims': [35,7,5,2,4,...]}
      - [(0,35),(35,42),...]
      - [35,7,5,2,4,...]  (dims)
    """
    if meta is None:
        return None
    # dict
    if isinstance(meta, dict):
        if 'groups' in meta and isinstance(meta['groups'], (list, tuple)):
            slices = []
            for g in meta['groups']:
                s, e = int(g['start']), int(g['end'])
                slices.append((s, e))
            return slices
        if 'slices' in meta:
            return [(int(s), int(e)) for s, e in meta['slices']]
        if 'dims' in meta:
            dims = [int(d) for d in meta['dims']]
            s = 0
            out = []
            for d in dims:
                out.append((s, s + d))
                s += d
            assert s == total_dim, f"sum(dims)={s} != total_dim={total_dim}"
            return out
        if 'sizes' in meta and 'offsets' in meta:
            sizes = [int(d) for d in meta['sizes']]
            offsets = [int(o) for o in meta['offsets']]
            # offsets가 누적 시작점이면 (start, end)로 환산
            if len(offsets) == len(sizes):
                return [(offsets[i], offsets[i] + sizes[i]) for i in range(len(sizes))]
    # list/tuple
    if isinstance(meta, (list, tuple)) and len(meta) > 0:
        if isinstance(meta[0], (list, tuple)) and len(meta[0]) == 2:
            # [(s,e), ...]
            return [(int(s), int(e)) for s, e in meta]
        else:
            # [d0, d1, ...]
            dims = [int(d) for d in meta]
            s = 0
            out = []
            for d in dims:
                out.append((s, s + d))
                s += d
            assert s == total_dim, f"sum(dims)={s} != total_dim={total_dim}"
            return out
    return None
class GraphNodeFeature(nn.Module):
    def __init__(
        self,
        num_heads,
        num_atoms,
        num_in_degree,
        num_out_degree,
        hidden_dim,
        n_layers,
        global_cat_dim=0,
        global_cont_dim=0,
        num_categorical_features=7,
        num_continuous_features=2,
        mode="cls_only",
    ):
        super().__init__()
        self.mode  = mode
        self.num_heads  = num_heads
        self.hidden_dim = hidden_dim

        # === 간단화: x_cat_onehot → 단일 Linear ===
        self.cat_proj = nn.Linear(num_categorical_features, hidden_dim, bias=False)

        # === 연속형 ===
        print("GraphNodeFeature num_continuous_features, hidden_dim", num_continuous_features, hidden_dim)
        self.continuous_proj = nn.Linear(num_continuous_features, hidden_dim)

        # === degree embedding (안전 클램프/폴백 지원)
        self.in_degree_encoder  = nn.Embedding(num_in_degree,  hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

        # === 글로벌 (cls_global_model 전용)
        self.global_cat_dim  = global_cat_dim
        self.global_cont_dim = global_cont_dim
        self.global_cat_proj  = nn.Linear(global_cat_dim,  hidden_dim, bias=False) if global_cat_dim  > 0 else None
        self.global_cont_proj = nn.Linear(global_cont_dim, hidden_dim)             if global_cont_dim > 0 else None

        # === 결합 MLP (cat + in + out + cont)
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.graph_token = nn.Embedding(1, hidden_dim)
        if self.mode == "cls_global_model":
            self.global_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda m: init_params(m, n_layers=n_layers))

    def _safe_degree_feats(self, in_degree, out_degree):
        # 범위 클램프 + 폴백(임베딩 실패 시 원핫→Linear)
        max_in  = self.in_degree_encoder.num_embeddings - 1
        max_out = self.out_degree_encoder.num_embeddings - 1
        in_degree  = in_degree.clamp_(0, max_in)
        out_degree = out_degree.clamp_(0, max_out)
        try:
            in_deg_feat  = self.in_degree_encoder(in_degree)
            out_deg_feat = self.out_degree_encoder(out_degree)
            return in_deg_feat, out_deg_feat
        except RuntimeError:
            # 폴백: one-hot -> Linear
            if not hasattr(self, "_deg_in_linear"):
                self._deg_in_linear  = nn.Linear(max_in+1,  self.hidden_dim, bias=False).to(in_degree.device)
                self._deg_out_linear = nn.Linear(max_out+1, self.hidden_dim, bias=False).to(out_degree.device)
            import torch.nn.functional as F
            in_oh  = F.one_hot(in_degree,  num_classes=max_in+1).float()
            out_oh = F.one_hot(out_degree, num_classes=max_out+1).float()
            in_deg_feat  = self._deg_in_linear(in_oh)
            out_deg_feat = self._deg_out_linear(out_oh)
            return in_deg_feat, out_deg_feat

    def _safe_continuous_proj(self, x_cont, replace_mode: str = "zero"):
        """
        x_cont: (B, N, F)
        replace_mode:
          - "zero" : nan/inf -> 0.0
          - "clip" : nan -> 0.0, +inf -> 1e6, -inf -> -1e6
          - "colmean" : 각 feature(col)별 유효값 평균으로 대체
        """
        B, N, F = x_cont.shape
        bad_mask = ~torch.isfinite(x_cont)

        if bad_mask.any():
            # 너무 시끄럽지 않게 '한 번만' 경고
            if not getattr(self, "_cont_warned", False):
                bad_cols = bad_mask.any(dim=(0, 1)).nonzero(as_tuple=False).view(-1)
                print(
                    f"[WARN] x_cont had non-finite values in columns {bad_cols.tolist()} - replaced ({replace_mode}).")
                self._cont_warned = True

            if replace_mode == "zero":
                # 전부 0으로 대체
                x_cont = torch.nan_to_num(x_cont, nan=0.0, posinf=0.0, neginf=0.0)

            elif replace_mode == "clip":
                # 큰 절댓값으로 클립
                x_cont = torch.nan_to_num(x_cont, nan=0.0, posinf=1e6, neginf=-1e6)

            elif replace_mode == "colmean":
                # 열 평균(유효값 기준)으로 대체
                finite = torch.where(bad_mask, torch.zeros_like(x_cont), x_cont)
                counts = (~bad_mask).sum(dim=(0, 1)).clamp_min(1)  # (F,)
                col_mean = finite.sum(dim=(0, 1)) / counts  # (F,)
                x_cont = torch.where(
                    bad_mask,
                    col_mean.view(1, 1, F).expand_as(x_cont),
                    x_cont
                )
            else:
                # 알 수 없는 모드면 안전하게 zero 모드
                x_cont = torch.nan_to_num(x_cont, nan=0.0, posinf=0.0, neginf=0.0)

        # (B,N,F) -> (B*N,F) 후 선형사상
        x_flat = x_cont.contiguous().view(B * N, F)
        W, b = self.continuous_proj.weight, self.continuous_proj.bias
        out_flat = x_flat.matmul(W.t())
        if b is not None:
            out_flat = out_flat + b
        return out_flat.view(B, N, self.hidden_dim)

    def forward(self, batched_data):
        x_cat      = batched_data["x_cat_onehot"].float()   # (B,N,F_cat)
        x_cont     = batched_data["x_cont"].float()         # (B,N,F_cont)
        in_degree  = batched_data["in_degree"].long()       # (B,N)
        out_degree = batched_data["out_degree"].long()      # (B,N)

        # ⬇⬇ 추가: 임베딩 크기에 맞춰 clamp
        max_in_idx = self.in_degree_encoder.num_embeddings - 1  # == config["num_in_degree"] - 1
        max_out_idx = self.out_degree_encoder.num_embeddings - 1  # == config["num_out_degree"] - 1
        in_degree = in_degree.clamp(min=0, max=max_in_idx)
        out_degree = out_degree.clamp(min=0, max=max_out_idx)

        categorical_feat = self.cat_proj(x_cat)                            # (B,N,H)
        in_deg_feat, out_deg_feat = self._safe_degree_feats(in_degree, out_degree)
        continuous_feat  = self._safe_continuous_proj(x_cont)              # (B,N,H)

        node_feature = torch.cat([categorical_feat, in_deg_feat, out_deg_feat, continuous_feat], dim=-1)
        node_feature = self.feature_mlp(node_feature)                      # (B,N,H)

        B = x_cat.size(0)
        cls_tok = self.graph_token.weight.unsqueeze(0).expand(B, -1, -1)   # (B,1,H)

        if self.mode == "cls_only":
            return torch.cat([cls_tok, node_feature], dim=1)

        elif self.mode == "cls_global_data":
            # 데이터에 글로벌 노드 이미 포함
            return torch.cat([cls_tok, node_feature], dim=1)

        elif self.mode == "cls_global_model":
            global_feats = []
            if self.global_cat_dim > 0 and "global_features_cat" in batched_data:
                gcat = batched_data["global_features_cat"].float()         # (B,Fg_cat)
                if self.global_cat_proj is not None:
                    global_feats.append(self.global_cat_proj(gcat))        # (B,H)
            if self.global_cont_dim > 0 and "global_features_cont" in batched_data:
                gcont = batched_data["global_features_cont"].float()       # (B,Fg_cont)
                global_feats.append(self.global_cont_proj(gcont))          # (B,H)
            gtok = (sum(global_feats).unsqueeze(1) if global_feats
                    else self.global_token.weight.unsqueeze(0).expand(B, -1, -1))
            return torch.cat([cls_tok, gtok, node_feature], dim=1)

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

class GraphAttnBias(nn.Module):
    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
        mode="cls_only",
        spatial_pos_pad_val: float = 510.0,
    ):
        super().__init__()
        self.mode = mode
        self.num_heads = num_heads
        self.num_atoms = num_atoms
        self.multi_hop_max_dist = multi_hop_max_dist
        self.spatial_pos_pad_val = spatial_pos_pad_val

        # 간단화: 엣지 onehot → 단일 Linear
        self.edge_encoder = nn.Linear(num_edges, num_heads, bias=False)
        self.spatial_pos_encoder = nn.Linear(1, num_heads)

        if edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis, num_heads * num_heads)
        self.disable_direct_1hop = True

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        if self.mode == "cls_global_model":
            self.global_node_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def _proj_edge_onehot(self, X: torch.Tensor):
        # X: [B,N,N,K] 또는 [B,N,N,D,K]
        X = X.contiguous()
        out = self.edge_encoder(X)  # 4D→[B,N,N,H], 5D→[B,N,N,D,H]
        if X.dim() == 4:
            return out.permute(0, 3, 1, 2).contiguous()  # [B,H,N,N]
        else:
            return out  # [B,N,N,D,H]

    def forward(self, batched_data):
        attn_bias   = batched_data["attn_bias"]         # [B,N,N]
        spatial_pos = batched_data["spatial_pos"]       # [B,N,N]
        x_cat       = batched_data["x_cat_onehot"]      # [B,N,F_cat]
        edge_input  = batched_data["edge_input"]        # [B,N,N,D,K]
        attn_edge_type = batched_data["attn_edge_type"] # [B,N,N,K]

        n_graph, n_node, _ = x_cat.size()
        graph_attn_bias = attn_bias.clone().unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [B,H,N,N]

        # 공간 거리 bias
        spatial_pos_processed = spatial_pos.clone()
        spatial_pos_processed[torch.isinf(spatial_pos_processed)] = 0
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos_processed.unsqueeze(-1)).permute(0, 3, 1, 2).contiguous()
        pad_mask = (spatial_pos >= self.spatial_pos_pad_val).unsqueeze(1)
        spatial_pos_bias = spatial_pos_bias.masked_fill(pad_mask, float("-inf")) #  AMP16 문제로 인해 1e-9 -> -inf

        graph_attn_bias = graph_attn_bias + spatial_pos_bias

        # 엣지 원핫 직접/멀티홉
        if self.disable_direct_1hop == True:
            edge_bias_direct = self._proj_edge_onehot(attn_edge_type.float())  # [B,H,N,N]
        edge_hop_feat    = self._proj_edge_onehot(edge_input.float())      # [B,N,N,D,H]

        if hasattr(self, "edge_dis_encoder"):
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_ = torch.where(torch.isinf(spatial_pos_), torch.tensor(0.0, device=spatial_pos_.device), spatial_pos_)
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)

            B, N, _, D, H = edge_hop_feat.shape
            feat_flat = edge_hop_feat.permute(3, 0, 1, 2, 4).reshape(D, -1, H)  # (D, BN², H)
            W = self.edge_dis_encoder.weight[:D].view(D, H, H)
            feat_w = torch.bmm(feat_flat, W)  # (D, BN², H)
            feat_w = feat_w.reshape(D, B, N, N, H).permute(1, 4, 2, 3, 0).contiguous()  # [B,H,N,N,D]
            denom = spatial_pos_.float().clamp(min=1.0).unsqueeze(1)  # [B,1,N,N]
            edge_bias_hop = (feat_w.sum(dim=-1)) / denom              # [B,H,N,N]
        else:
            # 멀티홉을 쓰지 않는 설정이면 hop 바이어스는 0
            edge_bias_hop = torch.zeros_like(graph_attn_bias)

        # CLS/Global 포함한 새 bias
        if self.mode == "cls_only":
            total_virtual = 1
        elif self.mode == "cls_global_data":
            total_virtual = 1
        elif self.mode == "cls_global_model":
            total_virtual = 2
        else:
            raise ValueError(f"Invalid GraphAttnBias mode: {self.mode}")

        B = graph_attn_bias.size(0)
        new_bias = torch.full(
            (B, self.num_heads, n_node + total_virtual, n_node + total_virtual),
            -1e9, device=graph_attn_bias.device
        )
        if not self.disable_direct_1hop and edge_bias_direct is not None:
            # direct(1-hop) + multi-hop 평균 둘 다 사용
            new_bias[:, :, total_virtual:, total_virtual:] = graph_attn_bias + edge_bias_direct + edge_bias_hop
        else:
            # 원래 Graphormer/IR처럼: multi-hop 평균만 사용 (1-hop은 평균의 특수 케이스)
            new_bias[:, :, total_virtual:, total_virtual:] = graph_attn_bias + edge_bias_hop

        t_cls = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        new_bias[:, :, total_virtual:, 0] = t_cls
        new_bias[:, :, 0, total_virtual:] = t_cls

        if self.mode == "cls_global_model":
            t_global = self.global_node_virtual_distance.weight.view(1, self.num_heads, 1)
            new_bias[:, :, total_virtual:, 1] = t_global
            new_bias[:, :, 1, total_virtual:] = t_global
            new_bias[:, :, 0, 1] = t_global.squeeze(-1)
            new_bias[:, :, 1, 0] = t_global.squeeze(-1)

        return new_bias
