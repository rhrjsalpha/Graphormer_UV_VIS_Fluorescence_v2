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
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ======= your project imports =======
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
def sid_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Spectral Information Divergence (SID)
    pred/target: (B, L) non-negative recommended
    """
    p = torch.clamp(pred, min=0.0) + eps
    q = torch.clamp(target, min=0.0) + eps
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (p * (p / q).log() + q * (q / p).log()).sum(dim=-1).mean()


def build_loss_fn(name: str) -> nn.Module:
    nm = str(name).upper()
    if nm == "MAE":
        return nn.L1Loss(reduction="none")
    if nm == "MSE":
        return nn.MSELoss(reduction="none")
    if nm in ("HUBER", "SMOOTH_L1"):
        return nn.SmoothL1Loss(reduction="none")
    raise ValueError(f"Unknown loss: {name}")


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
    spectrum_loss: str = "MAE"
    prop_loss: str = "MAE"  # lam_abs/lam_emi/life/qy 공통 (원하면 task별 dict로 바꿔도 됨)

    # loss weights
    w_spectrum: float = 1.0
    w_lam_abs: float = 1.0
    w_lam_emi: float = 1.0
    w_life: float = 1.0
    w_qy: float = 1.0


# -------------------------
# core: train/eval one epoch
# -------------------------
def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    cfg: TrainCfg,
    loss_fn_spec: Optional[nn.Module],
    loss_fn_prop: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    outputs keys should match dataset keys:
      spectrum, lam_abs, lam_emi, life, qy
    """
    losses: Dict[str, torch.Tensor] = {}

    # ---- spectrum ----
    if "spectrum" in outputs and "spectrum" in targets:
        m = masks["spectrum"]
        yhat = outputs["spectrum"]
        y = targets["spectrum"]

        if str(cfg.spectrum_loss).upper() == "SID":
            # SID는 sample mask 적용을 "샘플 단위로"만 한다 (원하면 nm mask까지 확장 가능)
            if m.sum() == 0:
                ls = yhat.new_tensor(0.0)
            else:
                ls = sid_loss(yhat[m], y[m])
        else:
            per_elem = loss_fn_spec(yhat, y)  # (B,L)
            ls = masked_reduce(per_elem, m)

        losses["loss_spectrum"] = ls * float(cfg.w_spectrum)

    # ---- props ----
    for k, w in [
        ("lam_abs", cfg.w_lam_abs),
        ("lam_emi", cfg.w_lam_emi),
        ("life", cfg.w_life),
        ("qy", cfg.w_qy),
    ]:
        if k in outputs and k in targets:
            m = masks[k]
            per_elem = loss_fn_prop(outputs[k], targets[k])  # (B,1) elementwise
            ls = masked_reduce(per_elem, m)
            losses[f"loss_{k}"] = ls * float(w)

    total = sum(losses.values()) if len(losses) else torch.tensor(0.0, device=next(iter(outputs.values())).device)

    # logging
    log = {k: v.detach().item() for k, v in losses.items()}
    log["loss_total"] = total.detach().item()
    return total, log


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: TrainCfg,
    scaler: Optional[torch.cuda.amp.GradScaler],
    loss_fn_spec: Optional[nn.Module],
    loss_fn_prop: nn.Module,
) -> Dict[str, float]:
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
            total_loss, log = compute_multitask_loss(
                outputs=outputs,
                targets=targets,
                masks=masks,
                cfg=cfg,
                loss_fn_spec=loss_fn_spec,
                loss_fn_prop=loss_fn_prop,
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
    return sum_logs


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainCfg,
    loss_fn_spec: Optional[nn.Module],
    loss_fn_prop: nn.Module,
) -> Dict[str, float]:
    model.eval()
    t0 = time.time()

    sum_logs: Dict[str, float] = {}
    n_steps = 0

    # metrics accum
    # (loss와 별개로 mae/rmse도 per task)
    mae_rmse_sum = {
        "spectrum_mae": 0.0, "spectrum_rmse": 0.0, "spectrum_n": 0,
        "lam_abs_mae": 0.0, "lam_abs_rmse": 0.0, "lam_abs_n": 0,
        "lam_emi_mae": 0.0, "lam_emi_rmse": 0.0, "lam_emi_n": 0,
        "life_mae": 0.0, "life_rmse": 0.0, "life_n": 0,
        "qy_mae": 0.0, "qy_rmse": 0.0, "qy_n": 0,
    }

    for batch in loader:
        if batch is None:
            continue
        batch = to_device_batch(batch, device)

        targets: Dict[str, torch.Tensor] = batch["targets"]
        masks: Dict[str, torch.Tensor] = batch["target_masks"]

        outputs = model(batch)
        total_loss, log = compute_multitask_loss(outputs, targets, masks, cfg, loss_fn_spec, loss_fn_prop)

        for k, v in log.items():
            sum_logs[k] = sum_logs.get(k, 0.0) + float(v)
        n_steps += 1

        # metrics (masked)
        for task in ["spectrum", "lam_abs", "lam_emi", "life", "qy"]:
            if task in outputs and task in targets:
                m = masks[task]
                mae, rmse = masked_mae_rmse(outputs[task], targets[task], m)
                if not (math.isnan(mae) or math.isnan(rmse)):
                    mae_rmse_sum[f"{task}_mae"] += mae
                    mae_rmse_sum[f"{task}_rmse"] += rmse
                    mae_rmse_sum[f"{task}_n"] += 1

    # average losses
    for k in list(sum_logs.keys()):
        sum_logs[k] /= max(n_steps, 1)
    sum_logs["sec"] = time.time() - t0
    sum_logs["steps"] = n_steps

    # average metrics
    for task in ["spectrum", "lam_abs", "lam_emi", "life", "qy"]:
        n = mae_rmse_sum[f"{task}_n"]
        if n > 0:
            sum_logs[f"{task}_mae"] = mae_rmse_sum[f"{task}_mae"] / n
            sum_logs[f"{task}_rmse"] = mae_rmse_sum[f"{task}_rmse"] / n
        else:
            sum_logs[f"{task}_mae"] = float("nan")
            sum_logs[f"{task}_rmse"] = float("nan")

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
        "prop_layers": {"lam_abs": 1, "lam_emi": 1, "life": 1, "qy": 1},
        "prop_output_dims": {"lam_abs": 1, "lam_emi": 1, "life": 1, "qy": 1},

        # branch split (원하면 조절)
        "shared_layers": model_cfg.get("num_encoder_layers", 6),
        "spec_layers": 2,
        "point_trunk_layers": 2,
    })


    model = GraphormerMultiBranchModel(model_cfg).to(device)

    # ---- optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    # ---- loss fns
    loss_fn_prop = build_loss_fn(cfg.prop_loss)

    if str(cfg.spectrum_loss).upper() == "SID":
        loss_fn_spec = None
    else:
        loss_fn_spec = build_loss_fn(cfg.spectrum_loss)

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
    log_csv = os.path.join(cfg.exp_dir, "metrics.csv")
    fieldnames = [
        "epoch",
        "train_loss_total", "val_loss_total",
        "train_loss_spectrum", "val_loss_spectrum",
        "train_loss_lam_abs", "val_loss_lam_abs",
        "train_loss_lam_emi", "val_loss_lam_emi",
        "train_loss_life", "val_loss_life",
        "train_loss_qy", "val_loss_qy",
        "val_spectrum_mae", "val_spectrum_rmse",
        "val_lam_abs_mae", "val_lam_abs_rmse",
        "val_lam_emi_mae", "val_lam_emi_rmse",
        "val_life_mae", "val_life_rmse",
        "val_qy_mae", "val_qy_rmse",
        "sec_train", "sec_val",
    ]

    # ---- train loop
    best_val = float("inf")
    best_path = os.path.join(cfg.exp_dir, "best.pt")

    for epoch in range(1, cfg.epochs + 1):
        # ---- unfreeze schedule
        if cfg.freeze_spec_head_epochs > 0 and epoch == (cfg.freeze_spec_head_epochs + 1):
            model.unfreeze_spectrum_branch(unfreeze_graph_layers=False, unfreeze_head=True)
            print(f"[Unfreeze] spectrum head at epoch {epoch}")

        if cfg.freeze_spec_layers_epochs > 0 and epoch == (cfg.freeze_spec_layers_epochs + 1):
            model.unfreeze_spectrum_branch(unfreeze_graph_layers=True, unfreeze_head=False)
            print(f"[Unfreeze] spectrum layers at epoch {epoch}")

        tr = train_one_epoch(model, train_loader, optimizer, device, cfg, scaler, loss_fn_spec, loss_fn_prop)

        if val_loader is not None:
            va = eval_one_epoch(model, val_loader, device, cfg, loss_fn_spec, loss_fn_prop)
        else:
            va = {"loss_total": float("nan"), "sec": 0.0, "steps": 0}

        # ---- print
        msg = (
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train total {tr.get('loss_total',0):.4f} | "
            f"val total {va.get('loss_total',float('nan')):.4f} | "
            f"train spec {tr.get('loss_spectrum',0):.4f} | "
            f"train abs {tr.get('loss_lam_abs',0):.4f} | "
            f"train emi {tr.get('loss_lam_emi',0):.4f} | "
            f"train life {tr.get('loss_life',0):.4f} | "
            f"train qy {tr.get('loss_qy',0):.4f} | "
            f"{tr.get('sec',0):.1f}s"
        )
        print(msg, flush=True)

        # ---- csv log
        row = {
            "epoch": epoch,
            "train_loss_total": tr.get("loss_total", float("nan")),
            "val_loss_total": va.get("loss_total", float("nan")),
            "train_loss_spectrum": tr.get("loss_spectrum", float("nan")),
            "val_loss_spectrum": va.get("loss_spectrum", float("nan")),
            "train_loss_lam_abs": tr.get("loss_lam_abs", float("nan")),
            "val_loss_lam_abs": va.get("loss_lam_abs", float("nan")),
            "train_loss_lam_emi": tr.get("loss_lam_emi", float("nan")),
            "val_loss_lam_emi": va.get("loss_lam_emi", float("nan")),
            "train_loss_life": tr.get("loss_life", float("nan")),
            "val_loss_life": va.get("loss_life", float("nan")),
            "train_loss_qy": tr.get("loss_qy", float("nan")),
            "val_loss_qy": va.get("loss_qy", float("nan")),
            "val_spectrum_mae": va.get("spectrum_mae", float("nan")),
            "val_spectrum_rmse": va.get("spectrum_rmse", float("nan")),
            "val_lam_abs_mae": va.get("lam_abs_mae", float("nan")),
            "val_lam_abs_rmse": va.get("lam_abs_rmse", float("nan")),
            "val_lam_emi_mae": va.get("lam_emi_mae", float("nan")),
            "val_lam_emi_rmse": va.get("lam_emi_rmse", float("nan")),
            "val_life_mae": va.get("life_mae", float("nan")),
            "val_life_rmse": va.get("life_rmse", float("nan")),
            "val_qy_mae": va.get("qy_mae", float("nan")),
            "val_qy_rmse": va.get("qy_rmse", float("nan")),
            "sec_train": tr.get("sec", 0.0),
            "sec_val": va.get("sec", 0.0),
        }
        append_csv_row(log_csv, fieldnames, row)

        # ---- best checkpoint (valobust: val_loader 없으면 train 기준)
        monitor = va.get("loss_total", float("nan"))
        if math.isnan(monitor):
            monitor = tr.get("loss_total", float("inf"))

        if monitor < best_val:
            best_val = monitor
            save_checkpoint(best_path, model, optimizer, epoch, cfg, extra={"best_val": best_val})
            print(f"[Best] saved: {best_path} (best_val={best_val:.6f})")

        # ---- milestone saving
        if epoch in set(cfg.save_milestones):
            ckpt_path = os.path.join(cfg.exp_dir, f"milestone_epoch_{epoch}.pt")
            save_checkpoint(ckpt_path, model, optimizer, epoch, cfg, extra={"best_val": best_val})
            print(f"[Milestone] saved: {ckpt_path}")

    print("[Done] training finished.")


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

        spectrum_loss="MAE",  # 또는 "SID"
        prop_loss="MAE",
        w_spectrum=1.0,
        w_lam_abs=1.0,
        w_lam_emi=1.0,
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
