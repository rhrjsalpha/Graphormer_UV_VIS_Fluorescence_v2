# Graphormer/G P3/modules_base_attention/multihead_attention.py
# 재작성 2025-07-22  ★ 변경 범위 전체

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

# ----------(★) quant_noise 불필요 시 graceful fallback ----------
try:
    from GP.modules_All.quant_noise import quant_noise
except ModuleNotFoundError:              #
    def quant_noise(layer, *_, **__) -> nn.Module:  #
        return layer                                #


class MultiheadAttention(nn.Module):
    r"""PyTorch-native multi-head attention (+ optional bias-kv)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        self_attention: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:
        super().__init__()

        # ---------- 기본 하이퍼파라미터 ----------
        self.embed_dim = embed_dim
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.self_attention = self_attention

        # ---------- 차원 체크 ----------
        assert embed_dim % num_heads == 0, "`embed_dim` must be divisible by `num_heads`"
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5            #

        # ---------- Q/K/V/Out 프로젝션 ----------
        self.q_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.v_proj = quant_noise(nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size)
        self.out_proj = quant_noise(nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()          #

    # ------------------------------------------------------------------ #
    # (★) Xavier 초기화 – 4개 프로젝션 모두
    # ------------------------------------------------------------------ #
    def reset_parameters(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(proj.weight, gain=1 / math.sqrt(2))
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        query: Tensor,                          # (T, B, E)
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        *,
        attn_bias: Optional[Tensor] = None,     # (B, H, T, S)
        key_padding_mask: Optional[Tensor] = None,  # (B, S)
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,     # (T, S)
    ) -> Tuple[Tensor, Optional[Tensor]]:

        # ---- Q / K / V 준비 ------------------------------------------------
        if self.self_attention or key is None:
            key = value = query
        T, B, _ = query.size()
        S = key.size(0)

        q = self.q_proj(query) * self.scale       # scaling 먼저
        k = self.k_proj(key)
        v = self.v_proj(value)

        # ---- (B*H, len, head_dim) 로 reshape ------------------------------
        def _reshape(x: Tensor) -> Tensor:        # helper 함수
            return x.contiguous().view(-1, B * self.num_heads, self.head_dim).transpose(0, 1)

        q, k, v = map(_reshape, (q, k, v))        # (B*H, T/S, head_dim)

        # ---- Scaled Dot-Product Attention ---------------------------------
        attn_weights = torch.bmm(q, k.transpose(1, 2))   # (B*H, T, S)

        if attn_bias is not None:                         # add-bias
            attn_weights += attn_bias.view(B * self.num_heads, T, S)

        if attn_mask is not None:                         # attn_mask
            attn_weights += attn_mask.unsqueeze(0)

        if key_padding_mask is not None:                  # padding mask
            pad = key_padding_mask[:, None, :].expand(-1, self.num_heads, -1)
            attn_weights = attn_weights.masked_fill(
                pad.reshape(B * self.num_heads, 1, S),
                float("-inf"),
            )

        attn_probs = self.dropout(torch.softmax(attn_weights, dim=-1, dtype=torch.float32))
        attn_output = torch.bmm(attn_probs, v)            # (B*H, T, head_dim)

        # ---- Merge heads ---------------------------------------------------
        attn_output = (
            attn_output.transpose(0, 1)
            .contiguous()
            .view(T, B, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        if need_weights:
            attn_probs = attn_probs.view(B, self.num_heads, T, S)  # ★ 반환용 reshape
            return attn_output, attn_probs
        return attn_output, None

# forward 인자 설명 #
"""
1. query: Tensor
형태: (T, B, E)
설명: Query 텐서로, Attention을 수행할 때 기본이 되는 입력 텐서입니다. 여기서,
T: 타겟 시퀀스의 길이 (보통은 노드의 개수)
B: 배치 크기
E: 임베딩 차원

2. key: Optional[Tensor]
형태: (S, B, E) (기본값: None)
설명: Key 텐서로, Attention을 계산할 때 query와 곱해져 유사도 점수를 생성하는 데 사용됩니다.
S: 소스 시퀀스의 길이 (query와 동일하면 self-attention, 다르면 cross-attention)
기본값으로는 self-attention을 수행하므로 key=None일 때는 query가 key로 자동 지정됩니다.

3. value: Optional[Tensor]
형태: (S, B, E) (기본값: None)
설명: Value 텐서로, Attention 점수에 따라 가중 평균되어 최종 출력 값을 구성합니다.
기본값으로는 self-attention을 수행하므로 value=None일 때는 query가 value로 자동 지정됩니다.

4. attn_bias: Optional[Tensor]
형태: (B, H, T, S) (기본값: None)
설명: Attention 점수에 더해지는 편향값(bias)으로, 각 head와 각 query-key 쌍에 대한 추가적인 정보를 제공합니다. 
예를 들어, 특정 노드 간의 연결 관계나 거리를 반영하여 attention 점수를 조정할 때 사용됩니다.
H: 헤드(head)의 개수 (멀티헤드 어텐션에서)

5. key_padding_mask: Optional[Tensor]
형태: (B, S) (기본값: None)
설명: Key 텐서에 있는 패딩된 위치를 표시하는 마스크입니다. 패딩된 위치는 attention 계산에서 제외되며, 
해당 위치의 attention 점수는 매우 큰 음수(-inf)로 설정되어 softmax를 적용할 때 거의 0이 됩니다.
패딩된 위치 : 여러 분자는 서로 다른 노드 개수 -> 가장 원자 많은 분자 기준으로 크기를 맞춤
-> 이때 생기는 가짜 원자 노드를 가리는 역할 = 패딩

6. need_weights: bool
기본값: False
설명: Attention 계산 후, attention 가중치(weight) 행렬을 반환할지 여부를 나타냅니다. 
True로 설정하면 Attention 계산에서 얻은 가중치 행렬을 반환합니다.

7. attn_mask: Optional[Tensor]
형태: (T, S) (기본값: None)

설명: Attention의 특정 위치를 마스킹하기 위한 텐서로, 예를 들어 특정 시간 단계 이후의 정보를 무시하고자 할 때 활용됩니다. 
마스크된 위치는 매우 큰 음수(-inf)로 설정되어 attention 계산에서 제외됩니다.
"""