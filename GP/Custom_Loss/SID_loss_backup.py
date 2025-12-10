### From ChemProp ##
# Modified #
import torch
import torch.nn.functional as F

import torch

### From ChemProp ##
# Modified #
import torch
import torch.nn as nn

import torch
import torch.nn.functional as F

#def sid_loss(
#    model_spectra: torch.Tensor,   # (B, L)  ← 모델 "생(raw) 출력"(활성화 X)
#    target_spectra: torch.Tensor,  # (B, L)
#    mask: torch.Tensor,            # (B, L) bool 또는 0/1
#    eps: float = 1e-8,
#    reduction: str = "mean_valid", # "mean_valid" | "mean" | "sum" | "none"
#    temperature: float = 1.0,      # softmax 온도(초기 1.0, 필요시 1.5~2.0)
#) -> torch.Tensor:
#    """
#    SID = KL(p||q) + KL(q||p)
#    - p: 예측 분포 = masked softmax((logits)/temperature)  [마스크 밖은 -inf, dim=-1]
#    - q: 타깃 분포 = 마스크 내부 합=1 정규화(softmax 아님)
#    """
#    device = model_spectra.device
#    mask_bool = mask.bool().to(device)
#
#    #print("mask_bool",mask_bool.shape, mask_bool.sum(dim=-2))
#
#    # --- (1) 예측 분포 p: 마스크드 소프트맥스 (양수+합=1을 한 번에, 배치 섞임 방지) ---
#    logits = (model_spectra / temperature).masked_fill(~mask_bool, float("-inf"))
#    # 행 전체가 False(유효 파장 없음)인 경우 NaN 방지
#    has_valid = mask_bool.any(dim=-1, keepdim=True)
#    logits = torch.where(has_valid, logits, torch.zeros_like(logits))
#    p = F.softmax(logits, dim=-1)                 # (B, L), 마스크 밖은 exp(-inf)=0
#    p = p * mask_bool                             # 혹시 모를 수치오차 방지용
#
#    # --- (2) 타깃 분포 q: 마스크 내부 합=1 정규화(원형 보존) ---
#    q_raw = torch.clamp(target_spectra, min=0) * mask_bool
#    q_sum = q_raw.sum(dim=-1, keepdim=True).clamp_min(eps)
#    q = q_raw / q_sum
#
#    # --- (3) SID 원식: KL(p||q)+KL(q||p) ---
#    sid_elem = p * (torch.log(p + eps) - torch.log(q + eps)) \
#             + q * (torch.log(q + eps) - torch.log(p + eps))
#    sid_elem = sid_elem * mask_bool  # 마스크 밖 0
#
#    # --- (4) reduction ---
#    if reduction == "none":
#        return sid_elem
#    if reduction == "sum":
#        return sid_elem.sum()
#    if reduction == "mean":
#        B, L = sid_elem.shape
#        return sid_elem.sum() / (B * L)
#    if reduction == "mean_valid":
#        valid_counts = mask_bool.sum(dim=-1).clamp_min(1)
#        per_sample = sid_elem.sum(dim=-1) / valid_counts
#        return per_sample.mean()
#
#    raise ValueError(f"Unknown reduction: {reduction}")

def sid_loss(
    model_spectra: torch.Tensor,   # (B, L)
    target_spectra: torch.Tensor,  # (B, L)
    mask: torch.Tensor,            # (B, L) bool or 0/1
    eps: float = 1e-8,
    reduction: str = "mean_valid", # "mean_valid" | "mean" | "sum" | "none"
) -> torch.Tensor:
    """
    SID with both p,q normalized *inside mask* and masked-out terms removed.
    - p = model / sum(model*mask)
    - q = target / sum(target*mask)
    - sid_i = sum_j [ p_j (log p_j - log q_j) + q_j (log q_j - log p_j) ] over masked j
    - reduction:
        "mean_valid": (per-sample sum / valid_count) -> batch mean   [추천: [GRAD] SID와 일치]
        "mean":       전체(B*L) 평균(마스크 밖은 0)                   [chemprop의 .mean()에 가까움]
        "sum":        합만 반환
        "none":       (B, L) 위치별 항 반환
    """
    device = model_spectra.device
    # print(model_spectra.shape, target_spectra.shape, mask.shape)
    if model_spectra.dim() == 2:
        model_spectra = model_spectra.unsqueeze(-1)
    if target_spectra.dim() == 2:
        target_spectra = target_spectra.unsqueeze(-1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    # mask -> bool
    mask = mask.bool().to(device)
    # print(mask.shape,mask.sum(dim=1))

    # 마스크 안쪽만 사용, log 안전을 위해 eps 바닥
    p_raw = (model_spectra.clamp_min(eps)) * mask
    q_raw = (target_spectra.clamp_min(eps)) * mask

    # 마스크 합으로 각 분포 정규화
    p_sum = p_raw.sum(dim=1, keepdim=True).clamp_min(eps)
    q_sum = q_raw.sum(dim=1, keepdim=True).clamp_min(eps)
    p = p_raw / p_sum
    q = q_raw / q_sum

    # SID 원식: p*log(p/q) + q*log(q/p)
    logp = p.clamp_min(eps).log()
    logq = q.clamp_min(eps).log()
    sid_elem = p * (logp - logq) + q * (logq - logp)  # (B, L)

    if reduction == "none":
        return sid_elem * mask  # 위치별 항

    if reduction == "sum":
        return (sid_elem * mask).sum()

    if reduction == "mean":
        # 전체 포인트(B*L) 평균(마스크 바깥은 0)
        B, L = sid_elem.shape
        return (sid_elem * mask).sum() / (B * L)

    if reduction == "mean_valid":
        # 각 샘플의 유효 길이로 나눈 뒤 배치 평균 → [GRAD] SID와 동일 스케일
        valid_counts = mask.sum(dim=1).clamp_min(1)
        per_sample = (sid_elem * mask).sum(dim=1) / valid_counts
        return per_sample.mean()

    raise ValueError(f"Unknown reduction: {reduction}")