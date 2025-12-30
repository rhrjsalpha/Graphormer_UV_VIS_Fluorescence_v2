# graphromer_Multitask_Train.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import csv
import time
import math
import json
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ======= your project imports =======
from GP.Custom_Loss.SID_loss import sid_loss
from GP.Custom_Loss.soft_dtw_cuda import SoftDTW
# MultiTask dataset / collate
from GP.data_prepare.MultiTaskDataset import MultiTaskSMILESDataset, multitask_collate_fn

# Multi-branch multitask model
from GP.models_All.graphormer_MultiTask import GraphormerMultiBranchModel

# config/vocab generator (프로젝트에서 쓰던 것)
try:
    from Pre_Defined_Vocab_Generator import generate_graphormer_config
except Exception:
    # 경로가 다르면 여기만 맞춰줘
    from GP.data_prepare.Pre_Defined_Vocab_Generator import generate_graphormer_config

# -------------------------
# utils
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """batch 내부의 Tensor들을 device로 이동 (dict 안 dict 포함)."""
    def _move(x):
        if torch.is_tensor(x):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: _move(v) for k, v in x.items()}
        return x

    return _move(batch)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _to_jsonable(obj):
    """Convert obj to something json can serialize."""
    import numpy as np
    import torch

    # 기본형
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy scalar
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    # torch dtype/device
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)

    # class/type (여기가 너 에러의 핵심!)
    if isinstance(obj, type):
        return obj.__name__  # 또는 str(obj)

    # tensor -> list
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()

    # dict
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]

    # 그 외: 문자열로 강제
    return str(obj)


def save_json(path: str, obj: Any):
    obj2 = _to_jsonable(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj2, f, indent=2, ensure_ascii=False)


# -------------------------
# losses
# -------------------------


def build_loss_fn(name: str) -> nn.Module:
    nm = str(name).upper()
    if nm == "MAE":
        return nn.L1Loss(reduction="none")
    if nm == "MSE":
        return nn.MSELoss(reduction="none")
    if nm in ("HUBER", "SMOOTH_L1"):
        return nn.SmoothL1Loss(reduction="none")
    raise ValueError(f"Unknown loss: {name}")

def build_softdtw(device, gamma=1.0, normalize=True, bandwidth=None):
    use_cuda = (device.type == "cuda")
    return SoftDTW(use_cuda=use_cuda, gamma=gamma, normalize=normalize, bandwidth=bandwidth)

@torch.no_grad()
def compute_sid_scalar(
    yhat: torch.Tensor,  # (B,L)
    y: torch.Tensor,     # (B,L)
    mB: torch.Tensor,    # (B,) bool
) -> torch.Tensor:
    """
    sid_loss를 네 train loss 계산과 동일한 방식으로(마스크 포함) scalar로 반환.
    """
    mB = mB.bool()
    if mB.sum() == 0:
        return yhat.new_tensor(float("nan"))

    mBL = torch.isfinite(y) & torch.isfinite(yhat)
    mBL = mBL & mB.unsqueeze(1)

    sid = sid_loss(
        model_spectra=yhat,
        target_spectra=y,
        mask=mBL,
        reduction="mean_valid",
    )
    return torch.nan_to_num(sid, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))


@torch.no_grad()
def sis_from_sid(sid: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    SIS = 1 / (1 + SID)
    """
    if torch.isnan(sid):
        return sid
    sid = sid.clamp_min(0.0)
    return 1.0 / (1.0 + sid + eps)

@torch.no_grad()
def compute_spectrum_metrics(
    pred_spec: torch.Tensor,   # (B, L)
    true_spec: torch.Tensor,   # (B, L)
    mB: torch.Tensor,          # (B,) bool  <-- 추가
    names: List[str],
    softdtw: Optional[SoftDTW] = None,
):
    out = {}
    names_u = [str(x).upper().replace("_", "") for x in names]  # "SoftDTW"도 처리

    for n in names_u:
        if n == "MAE":
            # sample mask 적용해서 평균
            err = (pred_spec - true_spec).abs().view(pred_spec.size(0), -1).mean(dim=1)
            out["MAE"] = err[mB].mean() if mB.any() else pred_spec.new_tensor(float("nan"))

        elif n == "RMSE":
            err2 = ((pred_spec - true_spec) ** 2).view(pred_spec.size(0), -1).mean(dim=1)
            rmse = torch.sqrt(err2 + 1e-12)
            out["RMSE"] = rmse[mB].mean() if mB.any() else pred_spec.new_tensor(float("nan"))

        elif n == "SID":
            out["SID"] = compute_sid_scalar(pred_spec, true_spec, mB)

        elif n == "SIS":
            sid_v = compute_sid_scalar(pred_spec, true_spec, mB)
            out["SIS"] = sis_from_sid(sid_v)

        elif n == "SOFTDTW":
            if softdtw is None:
                raise RuntimeError("SoftDTW metric requested but softdtw is None")
            if not mB.any():
                out["SoftDTW"] = pred_spec.new_tensor(float("nan"))
            else:
                X = torch.nan_to_num(pred_spec, nan=0.0, posinf=0.0, neginf=0.0)[mB].unsqueeze(-1)  # (Bv,L,1)
                Y = torch.nan_to_num(true_spec, nan=0.0, posinf=0.0, neginf=0.0)[mB].unsqueeze(-1)
                out["SoftDTW"] = softdtw(X, Y).mean()

        else:
            raise ValueError(f"Unknown spectrum term: {n}")

    return out

def masked_reduce(loss_per_elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    loss_per_elem:
      - (B, ...)  elementwise loss (reduction=none)
    mask:
      - (B,) bool : 샘플 단위 마스크
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    if loss_per_elem.dim() == 1:
        # (B,)
        per_sample = loss_per_elem
    else:
        # (B, D...) -> per-sample mean
        per_sample = loss_per_elem.view(loss_per_elem.size(0), -1).mean(dim=-1)

    if mask.sum() == 0:
        return per_sample.new_tensor(0.0)

    return per_sample[mask].mean()

def soft_argmax_nm(
    spec: torch.Tensor,          # (B, L)
    nm_start: int,
    nm_end: int,
    tau: float = 0.05,
) -> torch.Tensor:
    """
    Differentiable peak position (λmax) estimator.
    Returns: (B,) float nm
    """
    B, L = spec.shape
    # nm grid: (L,)
    nm_grid = torch.arange(nm_start, nm_end + 1, device=spec.device, dtype=spec.dtype)
    # softmax over wavelength axis
    p = torch.softmax(spec / tau, dim=1)  # (B, L)
    # expectation of nm
    return (p * nm_grid.unsqueeze(0)).sum(dim=1)  # (B,)

# -------------------------
# metrics
# -------------------------
@torch.no_grad()
def masked_mae_rmse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum() == 0:
        return (float("nan"), float("nan"))
    p = pred[mask]
    t = true[mask]
    err = (p - t).view(p.size(0), -1)
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()
    return mae, rmse


# -------------------------
# config dataclass (선택)
# -------------------------
@dataclass
class TrainCfg:
    exp_dir: str = "./exp_multitask"
    seed: int = 42

    # data
    batch_size: int = 16
    num_workers: int = 0
    max_nodes: int = 128
    multi_hop_max_dist: int = 5
    intensity_range: Tuple[int, int] = (200, 800)

    # train
    epochs: int = 200
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    use_amp: bool = True

    # milestones
    save_milestones: Tuple[int, ...] = (50, 100, 150, 200)

    # freeze schedule (optional)
    freeze_spec_head_epochs: int = 0          # 예: 20
    freeze_spec_layers_epochs: int = 0        # 예: 20  (spec_branch_layers까지 freeze하려면)

    # losses
    # spectrum: MAE/MSE/HUBER/SID (SID는 별도 함수)
    spectrum_loss: Union[str, List[str]] = "MAE"
    prop_loss: str = "MAE"  # max_nm/life/qy 공통 (원하면 task별 dict로 바꿔도 됨)
    peak_consistency_loss: str = "MAE"   # "MAE" or "MSE"
    peak_tau: float = 0.05               # soft-argmax temperature

    # loss weights
    w_spectrum: float = 1.0
    w_max_nm: float = 1.0
    w_life: float = 1.0
    w_qy: float = 1.0
    w_peak_consistency: float = 1.0

    # ---- metrics schedule
    metrics_mode: str = "every"       # "every" | "milestone" | "last" | "none"
    metrics_every: int = 10            # metrics_mode=="every"일 때 n epoch마다
    metrics_milestones: Tuple[int, ...] = ()  # metrics_mode=="milestone"일 때

    # ---- metrics options
    metrics_compute_train: bool = True
    metrics_compute_val: bool = True
    metrics_compute_softdtw: bool = True

    # ---- best 모델 모니터링 (metrics를 기반으로)
    monitor_key: str = "val_metric_spectrum_mae"   # 예시
    monitor_mode: str = "min"                      # "min" or "max"

    # ===== model (Graphormer + heads) =====
    # encoder depth split
    shared_layers: int = 12
    spec_layers: int = 2
    point_trunk_layers: int = 2

    # graphormer core
    embedding_dim: int = 512
    num_attention_heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    encoder_normalize_before: bool = True  # pre-LN
    layernorm_embedding: bool = True

    # graphormer graph-specific
    max_nodes: int = 128  # 이미 있음
    multi_hop_max_dist: int = 5  # 이미 있음
    # num_encoder_layers: int = 12  # gcfg override용(원하면 gcfg 무시)

    # output heads depth
    prop_layers: Dict[str, int] = None  # None이면 기본 {"max_nm":1,"life":1,"qy":1} 같은 식

def should_run_metrics(epoch: int, cfg: TrainCfg) -> bool:
    m = str(cfg.metrics_mode).lower()
    if m == "none":
        return False
    if m == "every":
        return (cfg.metrics_every <= 1) or (epoch % cfg.metrics_every == 0)
    if m == "milestone":
        ms = cfg.metrics_milestones if len(cfg.metrics_milestones) > 0 else cfg.save_milestones
        return epoch in set(ms)
    if m == "last":
        return epoch == cfg.epochs
    raise ValueError(f"Unknown metrics_mode: {cfg.metrics_mode}")

@torch.no_grad()
def compute_metrics_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    softdtw_fn: Optional[SoftDTW] = None,
    prefix: str = "val_metric_",   # "train_metric_" / "val_metric_"
) -> Dict[str, float]:
    model.eval()
    t0 = time.time()

    # ---------- Spectrum accum (pointwise MAE/RMSE) ----------
    spec_abs_sum = 0.0
    spec_sq_sum = 0.0
    spec_cnt = 0

    # batch-scalar 평균용 (SID/SoftDTW)
    spec_sid_sum = 0.0
    spec_sid_n = 0
    spec_softdtw_sum = 0.0
    spec_softdtw_n = 0

    # ---------- Point tasks accum ----------
    point_tasks = ["max_nm", "life", "qy"]
    pt = {k: {"n": 0, "abs_sum": 0.0, "sq_sum": 0.0, "y_sum": 0.0, "y2_sum": 0.0, "res2_sum": 0.0}
          for k in point_tasks}

    for batch in loader:
        if batch is None:
            continue
        batch = to_device_batch(batch, device)
        targets: Dict[str, torch.Tensor] = batch["targets"]
        masks: Dict[str, torch.Tensor] = batch["target_masks"]

        outputs = model(batch)

        # ===== Spectrum metrics =====
        if "spectrum" in outputs and "spectrum" in targets:
            mB = masks["spectrum"].bool()
            if mB.any():
                yhat = outputs["spectrum"]   # (B,L)
                y = targets["spectrum"]      # (B,L)

                # MAE/RMSE (전체 point 기준)
                err = (yhat - y)[mB]  # (Bv,L)
                e = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
                spec_abs_sum += float(e.abs().sum().item())
                spec_sq_sum += float((e * e).sum().item())
                spec_cnt += int(e.numel())

                # SID (mask 포함 scalar)
                sid_v = compute_sid_scalar(yhat, y, mB)
                if torch.isfinite(sid_v):
                    spec_sid_sum += float(sid_v.item())
                    spec_sid_n += 1

                # SoftDTW (scalar)
                if softdtw_fn is not None:
                    yh = torch.nan_to_num(yhat, nan=0.0, posinf=0.0, neginf=0.0)[mB].unsqueeze(-1)
                    yt = torch.nan_to_num(y,    nan=0.0, posinf=0.0, neginf=0.0)[mB].unsqueeze(-1)
                    if yh.size(0) > 0:
                        dtw_v = softdtw_fn(yh, yt).mean()
                        if torch.isfinite(dtw_v):
                            spec_softdtw_sum += float(dtw_v.item())
                            spec_softdtw_n += 1

        # ===== Point metrics =====
        for k in point_tasks:
            if k in outputs and k in targets:
                m = masks[k].bool()
                if m.any():
                    p = outputs[k][m].view(-1)
                    t = targets[k][m].view(-1)

                    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

                    diff = p - t
                    n = int(t.numel())
                    pt[k]["n"] += n
                    pt[k]["abs_sum"] += float(diff.abs().sum().item())
                    pt[k]["sq_sum"]  += float((diff * diff).sum().item())

                    pt[k]["y_sum"]   += float(t.sum().item())
                    pt[k]["y2_sum"]  += float((t * t).sum().item())
                    pt[k]["res2_sum"] += float((diff * diff).sum().item())

    out: Dict[str, float] = {}

    # ----- spectrum finalize -----
    if spec_cnt > 0:
        out[prefix + "spectrum_mae"] = spec_abs_sum / spec_cnt
        out[prefix + "spectrum_rmse"] = math.sqrt(spec_sq_sum / spec_cnt)
    else:
        out[prefix + "spectrum_mae"] = float("nan")
        out[prefix + "spectrum_rmse"] = float("nan")

    sid_mean = (spec_sid_sum / spec_sid_n) if spec_sid_n > 0 else float("nan")
    out[prefix + "spectrum_sid"] = sid_mean
    out[prefix + "spectrum_softdtw"] = (spec_softdtw_sum / spec_softdtw_n) if spec_softdtw_n > 0 else float("nan")
    out[prefix + "spectrum_sis"] = (1.0 / (1.0 + sid_mean + 1e-12)) if math.isfinite(sid_mean) else float("nan")

    # ----- point finalize -----
    for k in point_tasks:
        n = pt[k]["n"]
        if n > 0:
            mae = pt[k]["abs_sum"] / n
            rmse = math.sqrt(pt[k]["sq_sum"] / n)

            mean_y = pt[k]["y_sum"] / n
            ss_tot = pt[k]["y2_sum"] - n * (mean_y ** 2)
            if ss_tot <= 1e-12:
                r2 = float("nan")
            else:
                r2 = 1.0 - (pt[k]["res2_sum"] / ss_tot)

            out[prefix + f"{k}_mae"] = mae
            out[prefix + f"{k}_rmse"] = rmse
            out[prefix + f"{k}_r2"] = r2
        else:
            out[prefix + f"{k}_mae"] = float("nan")
            out[prefix + f"{k}_rmse"] = float("nan")
            out[prefix + f"{k}_r2"] = float("nan")

    out[prefix + "sec"] = time.time() - t0
    return out

# -------------------------
# core: train/eval one epoch
# -------------------------
def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    cfg: TrainCfg,
    loss_fns_spec: Dict[str, Optional[nn.Module]],  # {"MAE": L1Loss, "SID": None, ...}
    loss_fn_prop: nn.Module,
    init_losses: Dict[str, torch.Tensor],
    update_init: bool = False,
    eps: float = 1e-12,
    softdtw_fn: Optional[SoftDTW] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:

    raw_losses: Dict[str, torch.Tensor] = {}   # 정규화 전 scalar loss들

    # -----------------
    # 1) spectrum losses (multiple)
    # -----------------
    if "spectrum" in outputs and "spectrum" in targets:
        mB = masks["spectrum"].bool()  # (B,)
        if mB.any():
            yhat = outputs["spectrum"]  # (B,L)
            y = targets["spectrum"]  # (B,L)

            for nm, fn in loss_fns_spec.items():
                key = f"spectrum_{nm}"

                if nm == "SID":
                    mBL = torch.isfinite(y) & torch.isfinite(yhat)
                    mBL = mBL & mB.unsqueeze(1)

                    ls = sid_loss(
                        model_spectra=yhat,
                        target_spectra=y,
                        mask=mBL,
                        reduction="mean_valid",
                    )
                    ls = torch.nan_to_num(ls, nan=0.0, posinf=0.0, neginf=0.0)

                elif nm == "SOFTDTW":
                    if softdtw_fn is None:
                        raise RuntimeError("SOFTDTW requested but softdtw_fn is None")

                    yh = torch.nan_to_num(yhat, nan=0.0, posinf=0.0, neginf=0.0)
                    yt = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

                    # sample mask만 적용 (Bv, L, 1)
                    yh = yh[mB].unsqueeze(-1)
                    yt = yt[mB].unsqueeze(-1)

                    if yh.size(0) == 0:
                        ls = yhat.new_tensor(0.0)
                    else:
                        ls = softdtw_fn(yh, yt).mean()

                else:
                    per_elem = fn(yhat, y)  # (B,L)
                    ls = masked_reduce(per_elem, mB)

                raw_losses[key] = ls

    # -----------------
    # 1.5) peak consistency loss (spectrum λmax vs point max_nm)
    # -----------------
    if ("spectrum" in outputs) and ("max_nm" in outputs) and ("max_nm" in targets) and ("max_nm" in masks):
        spec_pred = outputs["spectrum"]  # (B, L)

        peak_nm = soft_argmax_nm(
            spec_pred,
            nm_start=int(cfg.intensity_range[0]),
            nm_end=int(cfg.intensity_range[1]),
            tau=float(getattr(cfg, "peak_tau", 0.05)),
        )  # (B,)

        max_nm_pred = outputs["max_nm"].view(-1)  # (B,)
        max_nm_true = targets["max_nm"].view(-1)  # (B,)
        m = masks["max_nm"].bool().view(-1)  # (B,)

        # peak-consistency는 "point가 존재하는 샘플"만 계산
        if m.any():
            d = peak_nm[m] - max_nm_pred[m]

            peak_loss_type = str(getattr(cfg, "peak_consistency_loss", "MAE")).upper()
            if peak_loss_type == "MSE":
                raw_losses["peak_consistency"] = (d * d).mean()
            else:
                raw_losses["peak_consistency"] = d.abs().mean()

    # -----------------
    # 2) property losses
    # -----------------
    prop_specs = [
        ("max_nm", float(cfg.w_max_nm)),
        ("life", float(cfg.w_life)),
        ("qy", float(cfg.w_qy)),
    ]

    # spectrum weight는 spectrum 계열 공통으로 cfg.w_spectrum 사용
    task_weight = {}
    task_weight.update({
        "max_nm": float(cfg.w_max_nm),
        "life": float(cfg.w_life),
        "qy": float(cfg.w_qy),
    })
    task_weight.update({"max_nm": float(cfg.w_max_nm), "life": float(cfg.w_life), "qy": float(cfg.w_qy)})
    task_weight.update({"peak_consistency": float(cfg.w_peak_consistency)})

    for k, _w in prop_specs:
        if k in outputs and k in targets:
            m = masks[k].bool()
            if m.sum() > 0:
                per_elem = loss_fn_prop(outputs[k], targets[k])  # (B,1) or (B,)
                ls = masked_reduce(per_elem, m)
                raw_losses[k] = ls
    #print("raw_losses", raw_losses)

    # -----------------
    # 3) init_losses 업데이트(최초 등장 시점)
    # -----------------
    # (mask=0인 task는 raw_losses에 없으므로 자동 제외됨)
    for k, ls in raw_losses.items():
        if (k not in init_losses) and update_init:
            init_losses[k] = ls.detach().float().clamp_min(eps)

    # -----------------
    # 4) 정규화: (loss / init_loss) / N
    # -----------------
    active_keys = [k for k in raw_losses.keys() if k in init_losses]  # init 없는 건 제외
    if len(active_keys) == 0:
        total = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        return total, {"loss_total": 0.0}, init_losses

    N = float(len(active_keys))

    norm_losses: Dict[str, torch.Tensor] = {}
    for k in active_keys:
        ratio = raw_losses[k] / init_losses[k]
        norm = ratio / N
        w = float(task_weight.get(k, 1.0))
        norm_losses[k] = norm * w

    if len(norm_losses) > 0:
        zero = next(iter(norm_losses.values())).new_tensor(0.0)
    else:
        zero = torch.tensor(0.0, device=next(iter(outputs.values())).device)
    #print("norm_losses", norm_losses)

    total = sum(norm_losses.values(), start=zero)
    #print("total:", total)

    # -----------------
    # 5) logging (원하면 raw도 같이 찍기)
    # -----------------
    log = {}

    # (A) raw도 저장
    for k, v in raw_losses.items():
        log[f"raw_{k}"] = v.detach().item()

    # raw_total / raw_spectrum 도 같이
    zero_raw = next(iter(raw_losses.values())).new_tensor(0.0) if len(raw_losses) > 0 else total.new_tensor(0.0)
    raw_total = sum(raw_losses.values(), start=zero_raw)
    log["raw_total"] = raw_total.detach().item()

    raw_spec_sum = sum((v for k, v in raw_losses.items() if k.startswith("spectrum_")), start=zero_raw)
    log["raw_spectrum"] = raw_spec_sum.detach().item()

    for pk in ["max_nm","life","qy"]:
        if pk in raw_losses:
            log[f"raw_{pk}"] = raw_losses[pk].detach().item()

    # (B) 기존 정규화된(norm_losses) 저장(너 코드 그대로)
    for k, v in norm_losses.items():
        log[f"loss_{k}"] = v.detach().item()

    log["loss_total"] = total.detach().item()

    # spec_sum은 int 방지 위해 start=zero
    zero = next(iter(norm_losses.values())).new_tensor(0.0)
    spec_sum = sum((v for k, v in norm_losses.items() if k.startswith("spectrum_")), start=zero)
    log["loss_spectrum"] = spec_sum.detach().item()

    for pk in ["max_nm","life","qy"]:
        if pk in norm_losses:
            log[f"loss_{pk}"] = norm_losses[pk].detach().item()

    return total, log, init_losses



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainCfg,
    scaler: Optional[torch.cuda.amp.GradScaler],
    loss_fns_spec: Dict[str, Optional[nn.Module]],
    loss_fn_prop: nn.Module,
    init_losses: Dict[str, torch.Tensor],
    update_init: bool = False,
    softdtw_fn: Optional[SoftDTW] = None,
) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    model.train()
    t0 = time.time()

    sum_logs: Dict[str, float] = {}
    n_steps = 0

    for batch in loader:
        if batch is None:
            continue
        batch = to_device_batch(batch, device)

        targets: Dict[str, torch.Tensor] = batch["targets"]
        masks: Dict[str, torch.Tensor] = batch["target_masks"]

        optimizer.zero_grad(set_to_none=True)

        use_amp = (cfg.use_amp and scaler is not None and device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch)  # dict
            total_loss, log, init_losses = compute_multitask_loss(
                outputs=outputs,
                targets=targets,
                masks=masks,
                cfg=cfg,
                loss_fns_spec=loss_fns_spec,
                loss_fn_prop=loss_fn_prop,
                init_losses=init_losses,
                update_init=update_init,
                softdtw_fn=softdtw_fn,
            )

        if use_amp:
            scaler.scale(total_loss).backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        for k, v in log.items():
            sum_logs[k] = sum_logs.get(k, 0.0) + float(v)
        n_steps += 1

    # average
    for k in list(sum_logs.keys()):
        sum_logs[k] /= max(n_steps, 1)

    sum_logs["sec"] = time.time() - t0
    sum_logs["steps"] = n_steps
    return sum_logs, init_losses


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainCfg,
    loss_fns_spec: Dict[str, Optional[nn.Module]],
    loss_fn_prop: nn.Module,
    init_losses: Dict[str, torch.Tensor],
    softdtw_fn: Optional[SoftDTW] = None,
) -> Dict[str, float]:
    model.eval()
    t0 = time.time()

    sum_logs: Dict[str, float] = {}
    n_steps = 0

    # -------------------------
    # (A) Spectrum metrics accum (전체 데이터 기준)
    # -------------------------
    spec_abs_sum = 0.0
    spec_sq_sum = 0.0
    spec_cnt = 0

    spec_sid_sum = 0.0
    spec_sid_n = 0
    spec_softdtw_sum = 0.0
    spec_softdtw_n = 0

    # -------------------------
    # (B) Point metrics accum (R2/MAE/RMSE를 “전체 데이터 기준”으로)
    # -------------------------
    point_tasks = ["lam_abs", "lam_emi", "life", "qy"]
    pt = {
        k: {"n": 0, "abs_sum": 0.0, "sq_sum": 0.0, "y_sum": 0.0, "y2_sum": 0.0, "res2_sum": 0.0}
        for k in point_tasks
    }

    for batch in loader:
        if batch is None:
            continue
        batch = to_device_batch(batch, device)
        targets: Dict[str, torch.Tensor] = batch["targets"]
        masks: Dict[str, torch.Tensor] = batch["target_masks"]

        outputs = model(batch)

        # ---------- 1) loss logging(기존 sum_logs 누적) ----------
        total_loss, log, _ = compute_multitask_loss(
            outputs, targets, masks, cfg,
            loss_fns_spec, loss_fn_prop, init_losses,
            softdtw_fn=softdtw_fn,
        )
        for k, v in log.items():
            sum_logs[k] = sum_logs.get(k, 0.0) + float(v)
        n_steps += 1

        # ---------- 2) Spectrum metrics ----------
        if "spectrum" in outputs and "spectrum" in targets:
            mB = masks["spectrum"].bool()
            if mB.any():
                yhat = outputs["spectrum"]   # (B,L)
                y = targets["spectrum"]      # (B,L)

                # MAE/RMSE (전체 point 기준 누적)
                err = (yhat - y)[mB]  # (Bv,L)
                e = torch.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)
                spec_abs_sum += float(e.abs().sum().item())
                spec_sq_sum += float((e * e).sum().item())
                spec_cnt += int(e.numel())

                # SID (배치 scalar 평균 누적)
                sid_v = compute_sid_scalar(yhat, y, mB)  # scalar
                if torch.isfinite(sid_v):
                    spec_sid_sum += float(sid_v.item())
                    spec_sid_n += 1

                # SoftDTW (배치 scalar 평균 누적)
                if softdtw_fn is not None:
                    yh = torch.nan_to_num(yhat, nan=0.0, posinf=0.0, neginf=0.0)[mB].unsqueeze(-1)
                    yt = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)[mB].unsqueeze(-1)
                    if yh.size(0) > 0:
                        dtw_v = softdtw_fn(yh, yt).mean()
                        if torch.isfinite(dtw_v):
                            spec_softdtw_sum += float(dtw_v.item())
                            spec_softdtw_n += 1

        # ---------- 3) Point metrics (R2/MAE/RMSE) ----------
        for k in point_tasks:
            if k in outputs and k in targets:
                m = masks[k].bool()
                if m.any():
                    p = outputs[k][m].view(-1)
                    t = targets[k][m].view(-1)

                    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

                    diff = p - t
                    pt[k]["n"] += int(t.numel())
                    pt[k]["abs_sum"] += float(diff.abs().sum().item())
                    pt[k]["sq_sum"] += float((diff * diff).sum().item())

                    pt[k]["y_sum"] += float(t.sum().item())
                    pt[k]["y2_sum"] += float((t * t).sum().item())
                    pt[k]["res2_sum"] += float((diff * diff).sum().item())

    # -------------------------
    # (C) 평균 loss (기존 방식 유지)
    # -------------------------
    for k in list(sum_logs.keys()):
        sum_logs[k] /= max(n_steps, 1)

    sum_logs["sec"] = time.time() - t0
    sum_logs["steps"] = n_steps

    # -------------------------
    # (D) Spectrum metrics finalize
    # -------------------------
    if spec_cnt > 0:
        sum_logs["spectrum_mae"] = spec_abs_sum / spec_cnt
        sum_logs["spectrum_rmse"] = math.sqrt(spec_sq_sum / spec_cnt)
    else:
        sum_logs["spectrum_mae"] = float("nan")
        sum_logs["spectrum_rmse"] = float("nan")

    sum_logs["spectrum_sid"] = (spec_sid_sum / spec_sid_n) if spec_sid_n > 0 else float("nan")
    sum_logs["spectrum_softdtw"] = (spec_softdtw_sum / spec_softdtw_n) if spec_softdtw_n > 0 else float("nan")
    # (원하면) SIS도 여기서 계산 가능: SIS = 1/(1+SID)
    sum_logs["spectrum_sis"] = (1.0 / (1.0 + sum_logs["spectrum_sid"] + 1e-12)) if math.isfinite(sum_logs["spectrum_sid"]) else float("nan")

    # -------------------------
    # (E) Point metrics finalize
    # -------------------------
    for k in point_tasks:
        n = pt[k]["n"]
        if n > 0:
            mae = pt[k]["abs_sum"] / n
            rmse = math.sqrt(pt[k]["sq_sum"] / n)

            y_sum = pt[k]["y_sum"]
            y2_sum = pt[k]["y2_sum"]
            res2 = pt[k]["res2_sum"]
            mean_y = y_sum / n
            ss_tot = y2_sum - n * (mean_y ** 2)

            if ss_tot <= 1e-12:
                r2 = float("nan")  # 타겟이 거의 상수면 R2 정의가 불안정
            else:
                r2 = 1.0 - (res2 / ss_tot)

            sum_logs[f"{k}_mae"] = mae
            sum_logs[f"{k}_rmse"] = rmse
            sum_logs[f"{k}_r2"] = r2
        else:
            sum_logs[f"{k}_mae"] = float("nan")
            sum_logs[f"{k}_rmse"] = float("nan")
            sum_logs[f"{k}_r2"] = float("nan")

    return sum_logs



# -------------------------
# checkpoint / milestone logging
# -------------------------
def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg: TrainCfg, extra: Dict[str, Any] | None = None):
    obj = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg.__dict__,
    }
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def append_csv_row(path: str, fieldnames: List[str], row: Dict[str, Any]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


# -------------------------
# main train function
# -------------------------
def run_train_multitask(
    *,
    cfg: TrainCfg,
    # train/val csv_info_list
    train_csv_info_list: List[Dict[str, Any]],
    val_csv_info_list: Optional[List[Dict[str, Any]]] = None,
):
    ensure_dir(cfg.exp_dir)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # ---- build graphormer config/vocab using all involved CSVs (권장: train+val 같이 넣기)
    dataset_path_list = [d["path"] for d in train_csv_info_list]
    if val_csv_info_list:
        dataset_path_list += [d["path"] for d in val_csv_info_list]

    gcfg = generate_graphormer_config(
        dataset_path_list=dataset_path_list,
        mode="cls_global_data",  # MultiTaskDataset이 기본 global 사용 전제
        mol_col="solute_mol",    # MultiTaskDataset 내부 통일 컬럼명
        target_type="exp_spectrum",
        intensity_range=cfg.intensity_range,
        # 아래 global 관련은 네 프로젝트 설정에 맞게 바꿔도 됨
        # global_feature_order / global_multihot_cols / continuous_feature_names 는
        # generate_graphormer_config 구현에 맞춰 넣으면 됨.
    )
    print("num_encoder_layers:", gcfg.get("num_encoder_layers"))
    print("shared_layers:", gcfg.get("shared_layers"))
    print("spec_layers:", gcfg.get("spec_layers"))
    print("point_trunk_layers:", gcfg.get("point_trunk_layers"))

    # ---- dataset objects
    train_ds = MultiTaskSMILESDataset(
        csv_info_list=train_csv_info_list,
        nominal_feature_vocab=gcfg["nominal_feature_vocab"],
        continuous_feature_names=gcfg.get("continuous_feature_names", []),
        global_cat_dim=gcfg.get("global_cat_dim", 0),
        global_cont_dim=gcfg.get("global_cont_dim", 0),
        ATOM_FEATURES_VOCAB=gcfg["ATOM_FEATURES_VOCAB"],
        float_feature_keys=gcfg["ATOM_FLOAT_FEATURE_KEYS"],
        BOND_FEATURES_VOCAB=gcfg["BOND_FEATURES_VOCAB"],
        GLOBAL_FEATURE_VOCABS_dict=gcfg["GLOBAL_FEATURE_VOCABS_dict"],
        x_cat_mode=gcfg.get("x_cat_mode", "onehot"),
        global_cat_mode=gcfg.get("global_cat_mode", "onehot"),
        max_nodes=cfg.max_nodes,
        multi_hop_max_dist=cfg.multi_hop_max_dist,
        intensity_range=cfg.intensity_range,
        mode="cls_global_data",
    )

    if val_csv_info_list:
        val_ds = MultiTaskSMILESDataset(
            csv_info_list=val_csv_info_list,
            nominal_feature_vocab=gcfg["nominal_feature_vocab"],
            continuous_feature_names=gcfg.get("continuous_feature_names", []),
            global_cat_dim=gcfg.get("global_cat_dim", 0),
            global_cont_dim=gcfg.get("global_cont_dim", 0),
            ATOM_FEATURES_VOCAB=gcfg["ATOM_FEATURES_VOCAB"],
            float_feature_keys=gcfg["ATOM_FLOAT_FEATURE_KEYS"],
            BOND_FEATURES_VOCAB=gcfg["BOND_FEATURES_VOCAB"],
            GLOBAL_FEATURE_VOCABS_dict=gcfg["GLOBAL_FEATURE_VOCABS_dict"],
            x_cat_mode=gcfg.get("x_cat_mode", "onehot"),
            global_cat_mode=gcfg.get("global_cat_mode", "onehot"),
            max_nodes=cfg.max_nodes,
            multi_hop_max_dist=cfg.multi_hop_max_dist,
            intensity_range=cfg.intensity_range,
            mode="cls_global_data",
        )
    else:
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: multitask_collate_fn(b, train_ds),
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=lambda b: multitask_collate_fn(b, val_ds),
            drop_last=False,
        )

    #g0, _, _, _ = train_ds[0]  # g_processed
#
    #aet = g0["attn_edge_type"]
    #if isinstance(aet, dict):
    #    # GraphAttnBias가 보통 'bond' 같은 키를 씀. 없으면 첫 키로.
    #    key = "bond" if "bond" in aet else next(iter(aet.keys()))
    #    edge_dim = int(aet[key].shape[-1])
    #    print(f"[Infer] attn_edge_type key='{key}', edge_dim={edge_dim}")
    #else:
    #    edge_dim = int(aet.shape[-1])
    #    print(f"[Infer] attn_edge_type tensor edge_dim={edge_dim}")


    # ---- build model config
    # 중요한 포인트: dataset 타겟 키에 맞춰 prop_layers/prop_output_dims를 설정해야 outputs 키가 일치함
    model_cfg = dict(gcfg)
    print("model_cfg",model_cfg)
    model_cfg["num_edges"] = model_cfg["num_edges"] + 1 # is_global 차원 추가
    print("model_cfg",model_cfg)
    model_cfg.update({
        "mode": "cls_global_data",
        "output_size": (cfg.intensity_range[1] - cfg.intensity_range[0] + 1),  # spectrum_dim
        "spectrum_output_size": (cfg.intensity_range[1] - cfg.intensity_range[0] + 1),

        # multitask heads keys: dataset targets keys와 동일하게!
        "prop_layers": {"max_nm": 1, "life": 1, "qy": 1},
        "prop_output_dims": {"max_nm": 1, "life": 1, "qy": 1},

        # branch split (원하면 조절)
        "shared_layers": 1, # model_cfg.get("num_encoder_layers", 1),
        "spec_layers": 1,
        "point_trunk_layers": 1,
    })
    print("model_cfg", model_cfg)


    model = GraphormerMultiBranchModel(model_cfg).to(device)

    # ---- optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    # ---- loss fns
    spec_losses = cfg.spectrum_loss
    if isinstance(spec_losses, str):
        spec_losses = [spec_losses]
    spec_losses = cfg.spectrum_loss
    if isinstance(spec_losses, str):
        spec_losses = [spec_losses]

    # "SoftDTW", "SOFT_DTW" 같은 입력도 허용
    spec_losses = [str(x).upper().replace("_", "") for x in spec_losses]  # -> "SOFTDTW"

    loss_fns_spec: Dict[str, Optional[nn.Module]] = {}
    for nm in spec_losses:
        if nm in ("SID", "SOFTDTW"):
            loss_fns_spec[nm] = None
        else:
            loss_fns_spec[nm] = build_loss_fn(nm)

    # ---- SoftDTW (loss용)
    softdtw_loss_fn = None
    if "SOFTDTW" in loss_fns_spec:
        softdtw_loss_fn = build_softdtw(device, gamma=1.0, normalize=True, bandwidth=None)

    # ---- SoftDTW (metric용: loss에 없더라도 metrics에서 쓰면 생성)
    softdtw_metric_fn = None
    if cfg.metrics_compute_softdtw:
        softdtw_metric_fn = build_softdtw(device, gamma=1.0, normalize=True, bandwidth=None)

    loss_fn_prop = build_loss_fn(cfg.prop_loss)

    # ---- save config snapshot
    save_json(os.path.join(cfg.exp_dir, "train_cfg.json"), cfg.__dict__)
    save_json(os.path.join(cfg.exp_dir, "graphormer_cfg.json"), model_cfg)

    # ---- optional freeze schedule
    # epoch 0에 바로 freeze 적용
    if cfg.freeze_spec_head_epochs > 0 or cfg.freeze_spec_layers_epochs > 0:
        freeze_graph_layers = (cfg.freeze_spec_layers_epochs > 0)
        model.freeze_spectrum_branch(
            freeze_graph_layers=freeze_graph_layers,
            freeze_head=(cfg.freeze_spec_head_epochs > 0),
        )
        print(f"[Freeze] spectrum branch frozen at start | head={cfg.freeze_spec_head_epochs>0}, layers={freeze_graph_layers}")

    # ---- logging
    loss_csv = os.path.join(cfg.exp_dir, "loss.csv")

    # ✅ spectrum per-loss 컬럼을 "항상" 만들 고정 세트
    SPEC_LOSS_COLUMNS = ["MAE", "MSE", "SID", "SOFTDTW"]  # 필요하면 "HUBER"도 추가 가능

    loss_fieldnames = [
        "epoch",
        "train_loss_total",
        "train_loss_spectrum",
        "train_loss_max_nm",
        "train_loss_life",
        "train_loss_qy",
        "train_raw_total",
        "train_raw_spectrum",
        "sec_train",
    ]

    # 개별 spectrum loss 컬럼을 항상 추가
    for nm in SPEC_LOSS_COLUMNS:
        loss_fieldnames.append(f"train_loss_spectrum_{nm}")
    for nm in SPEC_LOSS_COLUMNS:
        loss_fieldnames.append(f"train_raw_spectrum_{nm}")

    metrics_csv = os.path.join(cfg.exp_dir, "metrics.csv")
    metrics_fieldnames = [
        "epoch",

        # train metrics
        "train_metric_spectrum_mae",
        "train_metric_spectrum_rmse",
        "train_metric_spectrum_sid",
        "train_metric_spectrum_softdtw",
        "train_metric_spectrum_sis",

        # point: max_nm / life / qy
        "train_metric_max_nm_mae", "train_metric_max_nm_rmse", "train_metric_max_nm_r2",
        "train_metric_life_mae", "train_metric_life_rmse", "train_metric_life_r2",
        "train_metric_qy_mae", "train_metric_qy_rmse", "train_metric_qy_r2",

        "train_metric_sec",

        # val metrics
        "val_metric_spectrum_mae",
        "val_metric_spectrum_rmse",
        "val_metric_spectrum_sid",
        "val_metric_spectrum_softdtw",
        "val_metric_spectrum_sis",

        # point: max_nm / life / qy
        "val_metric_max_nm_mae", "val_metric_max_nm_rmse", "val_metric_max_nm_r2",
        "val_metric_life_mae", "val_metric_life_rmse", "val_metric_life_r2",
        "val_metric_qy_mae", "val_metric_qy_rmse", "val_metric_qy_r2",

        "val_metric_sec",
    ]

    # ---- train loop

    best_val = float("inf") if cfg.monitor_mode == "min" else -float("inf")
    best_path = os.path.join(cfg.exp_dir, "best.pt")
    init_losses = {}

    for epoch in range(1, cfg.epochs + 1):

        # ---- unfreeze schedule
        if cfg.freeze_spec_head_epochs > 0 and epoch == (cfg.freeze_spec_head_epochs + 1):
            model.unfreeze_spectrum_branch(unfreeze_graph_layers=False, unfreeze_head=True)

        if cfg.freeze_spec_layers_epochs > 0 and epoch == (cfg.freeze_spec_layers_epochs + 1):
            model.unfreeze_spectrum_branch(unfreeze_graph_layers=True, unfreeze_head=False)

        update_init = (len(init_losses) == 0)

        # ========= 1) TRAIN LOSS (매 epoch) =========
        tr, init_losses = train_one_epoch(
            model, train_loader, optimizer, device, cfg, scaler,
            loss_fns_spec, loss_fn_prop, init_losses,
            update_init=update_init,
            softdtw_fn=softdtw_loss_fn,
        )

        metrics_softdtw = softdtw_metric_fn if cfg.metrics_compute_softdtw else None

        m_tr = compute_metrics_one_epoch(
            model=model, loader=train_loader, device=device,
            softdtw_fn=metrics_softdtw,
            prefix="train_metric_",
        )

        m_va = compute_metrics_one_epoch(
            model=model, loader=val_loader, device=device,
            softdtw_fn=metrics_softdtw,
            prefix="val_metric_",
        )

        # loss.csv 저장 (항상)
        loss_row = {
            "epoch": epoch,
            "train_loss_total": tr.get("loss_total", float("nan")),
            "train_loss_spectrum": tr.get("loss_spectrum", float("nan")),
            "train_loss_max_nm": tr.get("loss_max_nm", 'nan'),
            "train_loss_life": tr.get("loss_life", float("nan")),
            "train_loss_qy": tr.get("loss_qy", float("nan")),
            "train_raw_total": tr.get("raw_total", float("nan")),
            "train_raw_spectrum": tr.get("raw_spectrum", float("nan")),
            "sec_train": tr.get("sec", 0.0),
        }

        # ✅ cfg.spectrum_loss에 없으면 tr에 키가 없으니 -> get(...)이 nan으로 들어감
        for nm in SPEC_LOSS_COLUMNS:
            loss_row[f"train_loss_spectrum_{nm}"] = tr.get(f"loss_spectrum_{nm}", float("nan"))
        for nm in SPEC_LOSS_COLUMNS:
            loss_row[f"train_raw_spectrum_{nm}"] = tr.get(f"raw_spectrum_{nm}", float("nan"))

        append_csv_row(loss_csv, loss_fieldnames, loss_row)

        # ========= 2) METRICS (선택 epoch) =========
        do_metrics = should_run_metrics(epoch, cfg)
        metrics_row = {"epoch": epoch}

        if do_metrics:

            # train metrics
            if cfg.metrics_compute_train:
                m_tr = compute_metrics_one_epoch(
                    model=model, loader=train_loader, device=device,
                    softdtw_fn=metrics_softdtw,
                    prefix="train_metric_",
                )
                metrics_row.update(m_tr)

            # val metrics
            if cfg.metrics_compute_val and (val_loader is not None):
                m_va = compute_metrics_one_epoch(
                    model=model, loader=val_loader, device=device,
                    softdtw_fn=metrics_softdtw,
                    prefix="val_metric_",
                )
                metrics_row.update(m_va)

            append_csv_row(metrics_csv, metrics_fieldnames, metrics_row)

            # ========= 3) BEST CHECKPOINT (metrics 기반) =========
            monitor_key = cfg.monitor_key
            monitor_val = metrics_row.get(monitor_key, float("nan"))

            if math.isfinite(monitor_val):
                improved = (monitor_val < best_val) if (cfg.monitor_mode == "min") else (monitor_val > best_val)
                if improved:
                    best_val = monitor_val
                    save_checkpoint(best_path, model, optimizer, epoch, cfg,
                                    extra={"best_val": best_val, "monitor_key": monitor_key})
                    print(f"[Best] saved: {best_path} ({monitor_key}={best_val:.6f})")

        # ---- milestone saving
        if epoch in set(cfg.save_milestones):
            ckpt_path = os.path.join(cfg.exp_dir, f"milestone_epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch, cfg, extra={"best_val": best_val})
            print(f"[Milestone] saved: {ckpt_path}")

        # ---- print (metrics 없을 땐 loss만)
        def _fmt(v):
            if v is None:
                return "nan"
            if isinstance(v, float) and (not math.isfinite(v)):
                return "nan"
            return f"{float(v):.4g}"

        # loss_fns_spec keys는 "MAE","SID","SOFTDTW" 같은 것들
        spec_keys = [f"loss_spectrum_{nm}" for nm in loss_fns_spec.keys()]  # ex) loss_spectrum_MAE, loss_spectrum_SID
        prop_keys = ["loss_max_nm", "loss_life", "loss_qy"]

        spec_str = " ".join([f"{k.split('_')[-1]}={_fmt(tr.get(k, float('nan')))}" for k in spec_keys])
        prop_str = " ".join([f"{k.split('_', 1)[1]}={_fmt(tr.get(k, float('nan')))}" for k in prop_keys])

        if do_metrics:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | "
                f"total={_fmt(tr.get('loss_total'))} (raw {tr.get('raw_total', float('nan')):.4g}) | "
                f"spec[{spec_str}] | prop[{prop_str}] | "
                f"{cfg.monitor_key}={_fmt(metrics_row.get(cfg.monitor_key, float('nan')))} | "
                f"{tr.get('sec', 0):.1f}s",
                flush=True
            )
        else:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | "
                f"total={_fmt(tr.get('loss_total'))} (raw {tr.get('raw_total', float('nan')):.4g}) | "
                f"spec[{spec_str}] | prop[{prop_str}] | "
                f"{tr.get('sec', 0):.1f}s",
                flush=True
            )


# -------------------------
# example usage
# -------------------------
if __name__ == "__main__":
    cfg = TrainCfg(
        exp_dir=r"./exp_multitask_run1",
        seed=42,
        batch_size=16,
        num_workers=0,
        epochs=200,
        lr=3e-4,
        weight_decay=1e-2,
        use_amp=True,
        save_milestones=(50, 100, 150, 200),

        # freeze 예시 (원하면 0으로)
        freeze_spec_head_epochs=0,
        freeze_spec_layers_epochs=0,

        spectrum_loss=["MAE", "SID"],  # 또는 "SID"
        prop_loss="MAE",
        w_spectrum=1.0,
        w_max_nm=1.0,
        w_life=1.0,
        w_qy=1.0,
    )

    # ====== 여기를 네 데이터 경로/컬럼에 맞게 세팅 ======
    # csv_info_list 원소 예시:
    # {
    #   "path": "C:/.../train_full.csv",
    #   "kind": "full" or "chromo",
    #   "mol_col": "smiles" or "InChI",
    #   "solvent_col": "solvent_smiles"  (없으면 None 가능)
    # }
    train_csv_info_list = [
        {
            "path": r"C:/Users/kogun/PycharmProjects/Graphormer_UV_VIS_Fluorescence_v2/Data/EM_with_solvent_smiles_test_sample100.csv",
            "kind": "full",
            "mol_col": "InChI",
            "solvent_col": "solvent_smiles",
        },
        {
            "path": r"C:/Users/kogun/PycharmProjects/Graphormer_UV_VIS_Fluorescence_v2/Data/DB for chromophore_Sci_Data_rev02_sample100.csv",
            "kind": "chromo",
            "mol_col": "Chromophore",
            "solvent_col": "Solvent",
        },
    ]

    val_csv_info_list = [
        {
            "path": r"C:/Users/kogun/PycharmProjects/Graphormer_UV_VIS_Fluorescence_v2/Data/EM_with_solvent_smiles_test_sample100.csv",
            "kind": "full",
            "mol_col": "InChI",
            "solvent_col": "solvent_smiles",
        },
        {
            "path": r"C:/Users/kogun/PycharmProjects/Graphormer_UV_VIS_Fluorescence_v2/Data/DB for chromophore_Sci_Data_rev02_sample100.csv",
            "kind": "chromo",
            "mol_col": "Chromophore",
            "solvent_col": "Solvent",
        },
    ]

    run_train_multitask(
        cfg=cfg,
        train_csv_info_list=train_csv_info_list,
        val_csv_info_list=val_csv_info_list,
    )
