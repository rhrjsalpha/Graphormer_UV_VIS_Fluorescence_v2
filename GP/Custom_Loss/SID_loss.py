
import torch
import torch.nn.functional as F

import torch
import torch

def sid_loss(
    model_spectra: torch.Tensor,   # (B, L)
    target_spectra: torch.Tensor,  # (B, L)  (각 스펙트럼 합=1을 '가정'하지만 안전하게 재정규화함)
    mask: torch.Tensor,            # (B, L)  bool
    threshold: float = 1e-12,      # 값 하한(모델/타깃 모두)
    eps: float = 1e-12,
    reduction: str = "mean_valid",
) -> torch.Tensor:
    device = model_spectra.device
    m = mask.bool().to(device)

    # 1) NaN/Inf 정리
    model = torch.nan_to_num(model_spectra, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target_spectra, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) 마스크 밖은 중립값으로 처리(손실 0이 되도록)
    zero = torch.zeros_like(model, device=device)
    one  = torch.ones_like(model, device=device)

    # 모델 정규화는 마스크 안에서만
    model_in = torch.where(m, model, zero)
    # 분모 0 방지
    denom = model_in.sum(dim=1, keepdim=True).clamp_min(eps)
    p = (model_in / denom).clamp_min(eps)

    # 타깃도 마스크 안에서만 사용, 너무 작은 값은 threshold로 끌어올림
    target_in = torch.where(m, target, zero)
    target_in = torch.clamp(target_in, min=threshold)
    q_denom = target_in.sum(dim=1, keepdim=True).clamp_min(eps)
    q = (target_in / q_denom).clamp_min(eps)

    # 마스크 밖은 중립값 1로 세팅 → 손실 0
    p = torch.where(m, p, one)
    q = torch.where(m, q, one)

    # 3) SID = KL(p||q) + KL(q||p)
    logp = torch.log(p)
    logq = torch.log(q)
    loss = p * (logp - logq) + q * (logq - logp)  # (B, L)

    # ---- reduction ----
    if reduction == "none":
        return loss
    if reduction == "sum":
        return (loss * m).sum()
    if reduction == "mean":
        B, L = loss.shape
        return (loss * m).sum() / (B * L)
    if reduction == "mean_valid":
        valid_counts = m.sum(dim=1).clamp_min(1)  # (B,)
        per_sample = (loss * m).sum(dim=1) / valid_counts
        # all-False 샘플은 평균에서 제외
        weights = (m.sum(dim=1) > 0).float()
        return (per_sample * weights).sum() / weights.sum().clamp_min(1)
    raise ValueError(f"unknown reduction: {reduction}")

    #return loss  # 필요하면 .mean()/.sum() 등 reduction 추가

#def sid_loss(
#    model_spectra: torch.Tensor,   # (B, L) or (B, L, 1)
#    target_spectra: torch.Tensor,  # (B, L) or (B, L, 1)
#    mask: torch.Tensor,            # (B, L) or (B, L, 1)
#    eps: float = 1e-8,
#    reduction: str = "mean_valid", # "mean_valid" | "mean" | "sum" | "none"
#    use_softmax: bool = False,     # True면 softmax(파장축), False면 L1 정규화(권장)
#    temperature: float = 1.0,      # use_softmax=True일 때만 사용
#) -> torch.Tensor:
#    print(model_spectra.shape, target_spectra.shape, mask.shape)
#    print(print(mask.shape,mask.sum(dim=1)))
#    # ---- 모양 통일: (B, L, 1) 허용, (B, L)도 자동 처리
#    if model_spectra.dim() == 2: model_spectra = model_spectra.unsqueeze(-1)
#    if target_spectra.dim() == 2: target_spectra = target_spectra.unsqueeze(-1)
#    if mask.dim() == 2:           mask = mask.unsqueeze(-1)
#    # 혹시 (B, L, 1, 1) 들어오면 꼬리 1 제거
#    while mask.dim() > 3 and mask.size(-1) == 1:
#        mask = mask.squeeze(-1)
#
#    device = model_spectra.device
#    mask = mask.bool().to(device)
#
#    # ---- 파장 축(L) = dim=-2 (채널=1이 꼬리니까)
#    axis = -2
#
#    # ---- p, q 계산 (마스크 내부에서만 정규화)
#    if use_softmax:
#        logits = (model_spectra / temperature).masked_fill(~mask, float("-inf"))
#        # 샘플에 유효 파장이 하나도 없으면 0으로
#        logits = torch.where(mask.any(dim=axis, keepdim=True), logits, torch.zeros_like(logits))
#        p = torch.softmax(logits, dim=axis) * mask
#    else:
#        p_raw = torch.clamp(model_spectra, min=eps) * mask
#        p = p_raw / p_raw.sum(dim=axis, keepdim=True).clamp_min(eps)
#
#    q_raw = torch.clamp(target_spectra, min=eps) * mask
#    q = q_raw / q_raw.sum(dim=axis, keepdim=True).clamp_min(eps)
#
#    # ---- SID
#    sid_elem = p * (torch.log(p + eps) - torch.log(q + eps)) \
#             + q * (torch.log(q + eps) - torch.log(p + eps))
#    sid_elem = sid_elem * mask  # 마스크 밖 제거
#
#    # ---- reduction
#    if reduction == "none":
#        return sid_elem
#    if reduction == "sum":
#        return sid_elem.sum()
#    if reduction == "mean":
#        return sid_elem.mean()
#    if reduction == "mean_valid":
#        valid_counts = mask.sum(dim=axis, keepdim=True).clamp_min(1)
#        per_sample = sid_elem.sum(dim=axis, keepdim=True) / valid_counts  # (B,1,1)
#        return per_sample.mean()
#
#    raise ValueError(f"Unknown reduction: {reduction}")

