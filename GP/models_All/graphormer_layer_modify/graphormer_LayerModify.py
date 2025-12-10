# graphormer_LayerModify.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from GP.modules_All.graphormer_graph_encoder import GraphormerGraphEncoder


# --- 기존 초기화 함수 유지 ---
def init_graphormer_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))
    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def _make_activation(name: str) -> nn.Module:
    """문자열로부터 활성화 모듈 생성."""
    if name is None:
        return nn.Identity()
    name = str(name).lower()
    if name in ("none", "identity"):
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "softplus":
        return nn.Softplus()
    if name == "softmax":
        # 스펙트럼 벡터의 마지막 차원에 대해 정규화
        return nn.Softmax(dim=-1)
    raise ValueError(f"Unknown activation: {name}")


def _init_linear_identity(linear: nn.Linear):
    """nn.Linear에 직교/아이덴티티에 가까운 초기화(직사각 행렬도 허용)."""
    with torch.no_grad():
        linear.weight.zero_()
        n = min(linear.out_features, linear.in_features)
        linear.weight[:n, :n].copy_(torch.eye(n, device=linear.weight.device))
        if linear.bias is not None:
            linear.bias.zero_()


def _init_linear_constant(linear: nn.Linear, value: float = 0.0):
    with torch.no_grad():
        linear.weight.fill_(value)
        if linear.bias is not None:
            linear.bias.fill_(0.0)


class GraphormerModel(nn.Module):
    """
    출력 헤드(마지막 레이어) 구성 가능 버전.
    config에서 제어할 키(예):

    # 공통
    embedding_dim: int
    num_attention_heads: int
    mode: "cls_only" | "cls_global_data" | "cls_global_model"
    output_size: int = 601

    # 출력 헤드 제어
    out_num_layers: int = 1
    out_hidden_dims: list[int] = []      # 예: [512, 512]
    out_activation: str = "relu"         # 중간 레이어 활성화
    out_final_activation: str = "softplus"  # 마지막에 적용할 활성화(예: "softplus", "softmax", "none")
    out_bias: bool = True
    out_dropout: float = 0.0

    # 생성 타이밍 및 학습/초기화
    out_build_in_forward: bool = False   # True면 __init__에서 만들지 않고 forward에서 최초 생성
    out_freeze: bool = False             # True면 출력 헤드 파라미터 학습 안 함
    out_init: str = "random"             # "random" | "identity" | "constant"
    out_const_value: float = 0.0         # out_init == "constant"일 때 가중치 상수값

    # 나머지 GraphormerGraphEncoder 필요한 키들은 기존 그대로
    """
    def __init__(self, config, target_type: str = "default", mode: str = "cls_only"):
        super().__init__()

        H = config["embedding_dim"]
        nhead = config["num_attention_heads"]
        assert isinstance(H, int) and H > 0
        assert H % nhead == 0, f"embedding_dim({H}) must be divisible by num_attention_heads({nhead})"
        print(f"[DEBUG] embedding_dim={H}, num_heads={nhead}")

        self.target_type = target_type
        self.embedding_dim = H
        self.mode = config.get("mode", mode)
        print("GraphormerModel self.mode", self.mode)

        # ===== Encoder =====
        self.encoder = GraphormerGraphEncoder(
            num_atoms=config["num_atoms"],
            num_in_degree=config["num_in_degree"],
            num_out_degree=config["num_out_degree"],
            num_edges=config["num_edges"],
            num_spatial=config["num_spatial"],
            num_edge_dis=config["num_edge_dis"],
            edge_type=config["edge_type"],
            multi_hop_max_dist=config["multi_hop_max_dist"],
            num_encoder_layers=config["num_encoder_layers"],
            embedding_dim=config["embedding_dim"],
            ffn_embedding_dim=config["ffn_embedding_dim"],
            num_attention_heads=config["num_attention_heads"],
            dropout=config["dropout"],
            attention_dropout=config["attention_dropout"],
            activation_dropout=config["activation_dropout"],
            activation_fn=config["activation_fn"],
            pre_layernorm=config.get("pre_layernorm", False),
            q_noise=config.get("q_noise", 0.0),
            qn_block_size=config.get("qn_block_size", 8),
            global_cat_dim=config.get("global_cat_dim", 0),
            global_cont_dim=config.get("global_cont_dim", 0),
            num_categorical_features=config.get("num_categorical_features", 7),
            num_continuous_features=config.get("num_continuous_features", 2),
            mode=self.mode,
        )

        # ===== Output head 설정값 저장 =====
        self.output_size = int(config.get("output_size", 601))

        self.out_num_layers = int(config.get("out_num_layers", 1))
        # 정수 하나만 주면 리스트로 변환
        ohd = config.get("out_hidden_dims", [])
        if isinstance(ohd, int):
            ohd = [ohd]
        self.out_hidden_dims = ohd

        self.out_activation = config.get("out_activation", "relu")
        print("self.out_activation",self.out_activation)
        self.out_final_activation = config.get("out_final_activation", None)
        self.out_bias = bool(config.get("out_bias", True))
        self.out_dropout = float(config.get("out_dropout", 0.0))

        self.out_build_in_forward = bool(config.get("out_build_in_forward", False))
        self.out_freeze = bool(config.get("out_freeze", False))
        self.out_init = str(config.get("out_init", "random")).lower()  # random | identity | constant
        self.out_const_value = float(config.get("out_const_value", 0.0))

        # ===== Output head 생성 =====
        # forward에서 만들고 싶으면 None으로 두기
        if self.out_build_in_forward:
            self.output_layer = None
            print("[INFO] output head will be built at first forward() "
                  f"(freeze={self.out_freeze}, init={self.out_init}).")
        else:
            self.output_layer = self._build_output_head(
                in_dim=self.embedding_dim,
                out_dim=self.output_size,
                trainable=not self.out_freeze,
                init_scheme=self.out_init,
                const_value=self.out_const_value,
            )

        # 공용 초기화
        self.apply(init_graphormer_params)

        # ===== 출력 헤드만 원하는 방식으로 다시 초기화 =====
        def _reinit_output_head(head: nn.Sequential, scheme: str, const_value: float):
            # head 내부에서 마지막 Linear를 찾음
            final_linear = None
            for mod in reversed(head):
                if isinstance(mod, nn.Linear):
                    final_linear = mod
                    break
            if final_linear is None:
                return

            if scheme == "identity":
                _init_linear_identity(final_linear)
            elif scheme == "constant":
                _init_linear_constant(final_linear, value=const_value)
            # bias는 0으로 두는게 보통 안전
            if final_linear.bias is not None:
                with torch.no_grad():
                    final_linear.bias.zero_()

        if self.output_layer is not None:
            _reinit_output_head(
                self.output_layer,
                scheme=self.out_init,  # "identity" 또는 "constant"
                const_value=self.out_const_value
            )

        nf = self.encoder.graph_node_feature
        print("[DEBUG] NodeFeature.hidden_dim =", getattr(nf, "hidden_dim", None))
        if hasattr(nf, "continuous_proj"):
            print("[DEBUG] continuous_proj.weight.shape =", tuple(nf.continuous_proj.weight.shape))

    # === 출력 헤드 빌더 ===
    def _build_output_head(
        self,
        in_dim: int,
        out_dim: int,
        trainable: bool,
        init_scheme: str = "random",
        const_value: float = 0.0,
    ) -> nn.Sequential:
        layers = []
        last = in_dim

        # 히든 레이어 구성
        num_hidden_layers = max(0, self.out_num_layers - 1)
        hidden_dims = self.out_hidden_dims
        if len(hidden_dims) < num_hidden_layers:
            # hidden 차원 명시가 부족하면 embedding_dim으로 채움
            hidden_dims = hidden_dims + [self.embedding_dim] * (num_hidden_layers - len(hidden_dims))

        for i in range(num_hidden_layers):
            hd = int(hidden_dims[i])
            layers.append(nn.Linear(last, hd, bias=self.out_bias))
            if self.out_dropout > 1e-7:
                layers.append(nn.Dropout(self.out_dropout))
            layers.append(_make_activation(self.out_activation))
            last = hd

        # 최종 Linear
        final_linear = nn.Linear(last, out_dim, bias=self.out_bias)
        layers.append(final_linear)

        # 최종 활성화(softplus/softmax 등)
        if self.out_final_activation not in (None, "none"):
            layers.append(_make_activation(self.out_final_activation))

        head = nn.Sequential(*layers)

        # 초기화 스킴 적용
        if init_scheme == "identity":
            # 아이덴티티는 '단일 선형 헤드'일 때 의미가 명확함
            if self.out_num_layers == 1:
                _init_linear_identity(final_linear)
            else:
                print("[WARN] identity init requested but out_num_layers>1. "
                      "Applying identity only to the final Linear block.")
                _init_linear_identity(final_linear)
        elif init_scheme == "constant":
            _init_linear_constant(final_linear, value=const_value)
        elif init_scheme == "random":
            pass  # 기본 init_graphormer_params에 의해 초기화됨
        else:
            print(f"[WARN] unknown out_init={init_scheme}; using default init.")

        # freeze 옵션
        if not trainable:
            for p in head.parameters():
                p.requires_grad_(False)

        return head

    def forward(self, batched_data, targets=None, target_type=None):
        target_type = target_type if target_type is not None else self.target_type

        node_embeddings, _ = self.encoder(batched_data)
        if node_embeddings.dim() == 3:
            # [T, B, H] → CLS([0])
            cls_repr = node_embeddings[0]
        else:
            cls_repr = node_embeddings  # [B, H]

        # 필요 시 forward 시점에서 출력 헤드 생성
        if self.output_layer is None:
            if (targets is None) and self.out_freeze:
                # freeze 헤드라면 targets 없어도 생성 가능
                pass
            # 경고: trainable=True로 forward에서 생성하면 optimizer에 등록되지 않음
            trainable = not self.out_freeze
            if trainable:
                print("[WARN] output head is being created during forward() with trainable=True. "
                      "Unless you recreate the optimizer afterwards, these params won't be optimized.")
            self.output_layer = self._build_output_head(
                in_dim=self.embedding_dim,
                out_dim=self.output_size,
                trainable=trainable,
                init_scheme=self.out_init,
                const_value=self.out_const_value,
            ).to(cls_repr.device)

        # 출력
        output = self.output_layer(cls_repr)

        # ex_prob 모드면 [B, n_pairs, 2]로 변형 (output_size가 2의 배수여야 함)
        if (target_type == "ex_prob") and (output.dim() == 2):
            if self.output_size % 2 != 0:
                raise ValueError(f"output_size({self.output_size}) must be divisible by 2 for 'ex_prob' mode.")
            output = output.view(output.size(0), -1, 2)

        return output

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    def main():
        # 최소 구성의 데모용 config
        cfg = {
            "embedding_dim": 8,
            "num_attention_heads": 2,

            # GraphormerGraphEncoder가 요구하는 최소 키들(작은 값으로 채움)
            "num_atoms": 10,
            "num_in_degree": 5,
            "num_out_degree": 5,
            "num_edges": 10,
            "num_spatial": 10,
            "num_edge_dis": 10,
            "edge_type": 4,
            "multi_hop_max_dist": 5,
            "num_encoder_layers": 1,
            "ffn_embedding_dim": 32,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "activation_dropout": 0.0,
            "activation_fn": "relu",
            "pre_layernorm": False,
            "q_noise": 0.0,
            "qn_block_size": 8,
            "global_cat_dim": 0,
            "global_cont_dim": 0,
            "num_categorical_features": 7,
            "num_continuous_features": 2,
            "mode": "cls_only",

            # 출력 헤드 설정: 단일 선형 + identity 초기화
            "output_size": 8,
            "out_num_layers": 1,
            "out_hidden_dims": [],
            "out_activation": "relu",
            "out_final_activation": "none",
            "out_bias": True,
            "out_dropout": 0.0,
            "out_build_in_forward": False,
            "out_freeze": False,
            "out_init": "identity",
            "out_const_value": 0.0,
        }

        # 모델 생성(헤드는 __init__에서 곧바로 만들어짐)
        model = GraphormerModel(cfg, target_type="exp_spectrum", mode="cls_only")

        # 최종 Linear 모듈 찾기(Sequential의 마지막 Linear)
        head = model.output_layer
        final_linear = None
        for mod in reversed(head):
            if isinstance(mod, nn.Linear):
                final_linear = mod
                break

        if final_linear is None:
            print("[ERROR] Final Linear layer not found in output head.")
            return

        W = final_linear.weight.detach().cpu()
        b = final_linear.bias.detach().cpu() if final_linear.bias is not None else None

        print("[TEST] Final Linear weight shape:", tuple(W.shape))
        print(W)
        if b is not None:
            print("[TEST] Final Linear bias shape:", tuple(b.shape))
            print(b)

        # 좌상단 m×m 블록이 단위행렬인지 확인
        m = min(final_linear.out_features, final_linear.in_features)
        eye = torch.eye(m)
        is_identity = torch.allclose(W[:m, :m], eye, atol=1e-6)
        print("[TEST] Top-left m×m block is identity:", bool(is_identity))

    main()
