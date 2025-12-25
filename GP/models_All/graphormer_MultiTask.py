# graphormer_MultiBranch.py
import copy
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn

from GP.modules_All.graphormer_graph_encoder import GraphormerGraphEncoder


# -------------------------
# utils
# -------------------------
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


def _make_activation(name: Optional[str]) -> nn.Module:
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
        return nn.Softmax(dim=-1)
    raise ValueError(f"Unknown activation: {name}")


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Optional[List[int]] = None,
    activation: str = "relu",
    final_activation: Optional[str] = None,
    dropout: float = 0.0,
    bias: bool = True,
) -> nn.Sequential:
    hidden_dims = hidden_dims or []
    layers: List[nn.Module] = []
    last = in_dim
    for hd in hidden_dims:
        layers.append(nn.Linear(last, int(hd), bias=bias))
        if dropout and dropout > 1e-7:
            layers.append(nn.Dropout(dropout))
        layers.append(_make_activation(activation))
        last = int(hd)

    layers.append(nn.Linear(last, out_dim, bias=bias))
    if final_activation not in (None, "none"):
        layers.append(_make_activation(final_activation))
    return nn.Sequential(*layers) # * = 리스트 언패킹


def _replace_first_arg(args, x_new):
    """positional call: args[0] is x. Replace it safely."""
    if len(args) == 0:
        return args
    new_args = (x_new,) + tuple(args[1:])
    return new_args


def _call_layer_replay(layer: nn.Module, x_new: torch.Tensor, args, kwargs):
    """
    Graphormer layer는 보통 layer(x, attn_bias, padding_mask, ...) 형태.
    우리는 hook으로 캡처한 args/kwargs를 그대로 쓰되 첫 인자(x)만 교체.
    """
    if len(args) > 0:
        new_args = _replace_first_arg(args, x_new)
        out = layer(*new_args, **kwargs)
    else:
        new_kwargs = dict(kwargs)
        for key in ("x", "input", "hidden_states"):
            if key in new_kwargs:
                new_kwargs[key] = x_new
                out = layer(**new_kwargs)
                break
        else:
            out = layer(x_new, **kwargs)

    # ✅ 중요: GraphormerGraphEncoderLayer는 (x, attn) 튜플을 반환할 수 있음
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


# -------------------------
# main model
# -------------------------
class GraphormerMultiBranchModel(nn.Module):
    """
    구조:
      shared_layers (n)
        ├─ spectrum branch: spec_layers (k_spec) + spec_mlp (-> spectrum_dim)
        └─ point branch: point_trunk_layers (k_point)
              ├─ abs_lambda : prop_layers[j] + head
              ├─ em_lambda  : prop_layers[j] + head
              ├─ lifetime   : prop_layers[j] + head
              └─ qy         : prop_layers[j] + head
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # ------------- required basic config -------------
        H = int(config["embedding_dim"])
        nhead = int(config["num_attention_heads"])
        assert H % nhead == 0, f"embedding_dim({H}) must be divisible by num_attention_heads({nhead})"

        self.embedding_dim = H
        self.mode = config.get("mode", "cls_only")

        # ------------- split hyperparams -------------
        self.shared_layers = int(config.get("shared_layers", config.get("num_encoder_layers", 6)))
        self.spec_layers = int(config.get("spec_layers", 2))
        self.point_trunk_layers = int(config.get("point_trunk_layers", 2))

        # property별 tail layer 수 (j)
        # 예: {"abs_lambda": 1, "em_lambda": 1, "lifetime": 1, "qy": 1}
        self.prop_layers: Dict[str, int] = dict(config.get("prop_layers", {}))
        if not self.prop_layers:
            # 기본값
            self.prop_layers = {"abs_lambda": 1, "em_lambda": 1, "lifetime": 1, "qy": 1}

        # ------------- output dims -------------
        self.spectrum_dim = int(config.get("spectrum_output_size", config.get("output_size", 601)))
        self.prop_output_dims: Dict[str, int] = dict(config.get("prop_output_dims", {}))
        # 기본: 모두 1 (회귀 스칼라)
        for k in self.prop_layers.keys():
            self.prop_output_dims.setdefault(k, 1)

        # ------------- encoder (shared only) -------------
        # shared까지만 "진짜 encoder"로 돌리고,
        # 나머지 branch 레이어는 encoder.layer[0] 템플릿을 deepcopy해서 우리가 따로 만든다.
        self.encoder = GraphormerGraphEncoder(
            num_atoms=config["num_atoms"],
            num_in_degree=config["num_in_degree"],
            num_out_degree=config["num_out_degree"],
            num_edges=config["num_edges"],
            num_spatial=config["num_spatial"],
            num_edge_dis=config["num_edge_dis"],
            edge_type=config["edge_type"],
            multi_hop_max_dist=config["multi_hop_max_dist"],
            num_encoder_layers=self.shared_layers,
            embedding_dim=H,
            ffn_embedding_dim=config["ffn_embedding_dim"],
            num_attention_heads=nhead,
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

        # ------------- template layer 확보 -------------
        # encoder.layers[0]를 템플릿으로 사용 (GraphormerGraphEncoder가 layers를 갖는 구조라고 가정)
        if not hasattr(self.encoder, "layers"):
            raise AttributeError(
                "GraphormerGraphEncoder에 'layers'가 없습니다. "
                "현재 encoder 구현에서 레이어 리스트 이름이 다르면 여기서 맞춰줘야 합니다."
            )
        if len(self.encoder.layers) < 1:
            raise RuntimeError("encoder.layers가 비어 있습니다. num_encoder_layers 확인 필요.")

        self._layer_template = self.encoder.layers[0]

        # ------------- branch layers 구성 -------------
        self.spec_branch_layers = nn.ModuleList([copy.deepcopy(self._layer_template) for _ in range(self.spec_layers)])
        self.point_trunk = nn.ModuleList([copy.deepcopy(self._layer_template) for _ in range(self.point_trunk_layers)])

        self.prop_tails = nn.ModuleDict()
        for prop, j in self.prop_layers.items():
            self.prop_tails[prop] = nn.ModuleList([copy.deepcopy(self._layer_template) for _ in range(int(j))])

        # ------------- heads -------------
        # spectrum head (MLP)
        self.spec_head = build_mlp(
            in_dim=H,
            out_dim=self.spectrum_dim,
            hidden_dims=config.get("spec_head_hidden_dims", []),
            activation=config.get("spec_head_activation", "relu"),
            final_activation=config.get("spec_head_final_activation", None),
            dropout=float(config.get("spec_head_dropout", 0.0)),
            bias=bool(config.get("spec_head_bias", True)),
        )

        # property heads
        self.prop_heads = nn.ModuleDict()
        for prop, out_dim in self.prop_output_dims.items():
            self.prop_heads[prop] = build_mlp(
                in_dim=H,
                out_dim=int(out_dim),
                hidden_dims=config.get("prop_head_hidden_dims", []),
                activation=config.get("prop_head_activation", "relu"),
                final_activation=config.get("prop_head_final_activation", None),
                dropout=float(config.get("prop_head_dropout", 0.0)),
                bias=bool(config.get("prop_head_bias", True)),
            )

        # ------------- init -------------
        self.apply(init_graphormer_params)

        # ------------- hook storage -------------
        self._last_layer_args = None
        self._last_layer_kwargs = None
        self._hook_handle = None
        self._register_capture_hook()

        # ------------- freeze config -------------
        self._spec_frozen = False

    # -------------------------
    # hook: capture per-forward signature/tensors : encoder.layers[0]가 호출되기 직전에, 그 레이어에 들어갈 입력 인자(args/kwargs) 를 그대로 저장
    # -------------------------
    def _register_capture_hook(self):
        """
        1) 첫 레이어 pre_hook: args/kwargs 캡처 (attn_bias/padmask 등)
        2) 마지막 레이어 forward_hook: shared encoder의 output token([T,B,H]) 캡처
        """

        def _capture_pre(module, args, kwargs):
            self._last_layer_args = args
            self._last_layer_kwargs = kwargs

        def _capture_last_out(module, args, output):
            # GraphormerGraphEncoderLayer forward가 (x, attn) 또는 x만 반환할 수 있음
            if isinstance(output, (tuple, list)):
                x = output[0]
            else:
                x = output

            # 반드시 [T,B,H]여야 함
            self._shared_out_tokens = x

        # 첫 레이어: 입력 시그니처 캡처
        self.encoder.layers[0].register_forward_pre_hook(_capture_pre, with_kwargs=True)

        # ✅ 마지막 shared 레이어: output 토큰 캡처
        last_idx = len(self.encoder.layers) - 1
        self.encoder.layers[last_idx].register_forward_hook(_capture_last_out)

    # -------------------------
    # freeze/unfreeze helpers
    # -------------------------
    def freeze_spectrum_branch(self, freeze_graph_layers: bool = False, freeze_head: bool = True):
        """
        스펙트럼 branch를 freeze:
          - freeze_head=True면 spec_head 고정
          - freeze_graph_layers=True면 spec_branch_layers까지 고정
        """
        if freeze_head:
            for p in self.spec_head.parameters():
                p.requires_grad_(False)
        if freeze_graph_layers:
            for p in self.spec_branch_layers.parameters():
                p.requires_grad_(False)
        self._spec_frozen = True

    def unfreeze_spectrum_branch(self, unfreeze_graph_layers: bool = False, unfreeze_head: bool = True):
        if unfreeze_head:
            for p in self.spec_head.parameters():
                p.requires_grad_(True)
        if unfreeze_graph_layers:
            for p in self.spec_branch_layers.parameters():
                p.requires_grad_(True)
        self._spec_frozen = False

    # -------------------------
    # forward
    # -------------------------
    def forward(self, batched_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        return:
          {
            "spectrum": [B, spectrum_dim],
            "abs_lambda": [B, d],
            "em_lambda":  [B, d],
            "lifetime":   [B, d],
            "qy":         [B, d],
          }
        """

        # hook에서 이 배치의 호출 args/kwargs 확보
        # 1) shared encoder 실행 (hooks가 내부에서 args/kwargs + 마지막 토큰 출력 캡처)
        _ = self.encoder(batched_data)

        if self._last_layer_args is None:
            raise RuntimeError("capture failed: encoder.layers[0] pre_hook not triggered.")
        if self._shared_out_tokens is None:
            raise RuntimeError("capture failed: encoder last layer output hook not triggered.")

        x_shared = self._shared_out_tokens  # ✅ [T,B,H] 여기서 확보!

        args = self._last_layer_args
        kwargs = self._last_layer_kwargs

        #print("[DEBUG] x_shared:", tuple(x_shared.shape))  # (T,B,H)에서 T가 1이면 아직 문제
        #print("[DEBUG] attn_bias:", tuple(kwargs["attn_bias"].shape) if "attn_bias" in kwargs else None)

        # 2) spectrum branch
        x_spec = x_shared
        for layer in self.spec_branch_layers:
            x_spec = _call_layer_replay(layer, x_spec, args, kwargs)

        cls_spec = x_spec[0]  # [B,H]
        y_spec = self.spec_head(cls_spec)  # [B, spectrum_dim]

        # 3) point trunk
        x_point = x_shared
        for layer in self.point_trunk:
            x_point = _call_layer_replay(layer, x_point, args, kwargs)

        cls_point = x_point[0]  # [B,H]

        # 4) per-property tails
        outputs: Dict[str, torch.Tensor] = {"spectrum": y_spec}
        for prop, tail_layers in self.prop_tails.items():
            x_prop = x_point
            for layer in tail_layers:
                x_prop = _call_layer_replay(layer, x_prop, args, kwargs)
            cls_prop = x_prop[0]
            outputs[prop] = self.prop_heads[prop](cls_prop)

        return outputs
