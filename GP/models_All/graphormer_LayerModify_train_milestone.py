from __future__ import annotations

# ===== 기본 패키지 =====
import os
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
from functools import partial

# ===== Graphormer / 사용자 정의 모듈 =====
from GP.data_prepare.DataLoader_QMData_All import (
    UnifiedSMILESDataset,
    get_global_feature_info,
)
from GP.data_prepare.DataLoader_QMData_All import collate_fn as graph_collate_fn

from GP.models_All.graphormer_layer_modify.graphormer_LayerModify import GraphormerModel
from GP.Custom_Loss.soft_dtw_cuda import SoftDTW
from GP.Custom_Loss.GradNorm import GradNorm
from GP.data_prepare.DataLoader_Test import show_batch_shapes, peek_one_batch  # 유틸 가져오기
from GP.data_prepare.Pre_Defined_Vocab_Generator import build_vocabs_from_df
from GP.models_All.evaluate_metrics import evaluate_model_metrics

#from chemprop.train.loss_functions import sid_loss
from GP.Custom_Loss.SID_loss import sid_loss
from rdkit import Chem
import hashlib
from typing import Optional, Dict, Any
from tqdm import tqdm
import torch.distributed as dist

# ==== Milestone & Eval helpers (paste once) ===================================
import os, csv, time
from pathlib import Path
import torch
from GP.After_Training_Module.Caculate_Visualize_Each_Molecule_new import evaluate_split_per_sample

# === Resume helpers ===========================================================
import re
import glob

# ==== Resume helpers (legacy ckpt 호환) =======================================
import re, glob, json, torch
import os, csv, time
from pathlib import Path

def parse_epoch_from_path(path: str) -> int | None:
    """파일명에 들어있는 _epoch{N}.pt 에서 N을 파싱."""
    m = re.search(r"_epoch(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None

def find_latest_checkpoint(ckpt_dir: str | Path) -> str | None:
    """
    ckpt_dir 안에서 '..._epoch{N}.pt' 패턴의 가장 큰 N을 찾아 경로를 반환.
    없으면 None.
    """
    ckpt_dir = str(ckpt_dir)
    cands = glob.glob(os.path.join(ckpt_dir, "*.pt"))
    best = None
    best_ep = -1
    for p in cands:
        m = re.search(r"_epoch(\d+)\.pt$", os.path.basename(p))
        if not m:
            continue
        ep = int(m.group(1))
        if ep > best_ep:
            best_ep = ep
            best = p
    return best

def load_legacy_checkpoint(
    ckpt_path: str,
    model,
    optimizer=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
):
    """
    예전 형식(모델 state_dict만 있는 경우 포함)도 로드 가능하게.
    반환: (start_epoch:int|None, had_optimizer:bool, had_scheduler:bool)
    """
    blob = torch.load(ckpt_path, map_location=map_location)

    # 1) 모델 가중치 추출
    if isinstance(blob, dict) and "model" in blob:
        state_dict = blob["model"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        state_dict = blob["state_dict"]
    elif isinstance(blob, dict):
        # 키들이 파라미터처럼 보이면 바로 사용
        state_dict = blob
    else:
        raise ValueError(f"알 수 없는 체크포인트 형식: {type(blob)}")

    (model.module if hasattr(model, "module") else model).load_state_dict(state_dict, strict=False)

    # 2) 옵티마이저/스케줄러 (있을 때만)
    had_opt = False
    had_sch = False
    if isinstance(blob, dict) and "optimizer" in blob and blob["optimizer"] and optimizer is not None:
        try:
            optimizer.load_state_dict(blob["optimizer"])
            had_opt = True
        except Exception:
            pass
    if isinstance(blob, dict) and "scheduler" in blob and blob["scheduler"] and scheduler is not None:
        try:
            scheduler.load_state_dict(blob["scheduler"])
            had_sch = True
        except Exception:
            pass

    # 3) 시작 epoch 추정
    if isinstance(blob, dict) and "epoch" in blob:
        start_epoch = int(blob["epoch"]) + 1  # 보통 다음 epoch부터
    else:
        # 파일명에서 파싱
        ep = parse_epoch_from_path(ckpt_path)
        start_epoch = (ep + 1) if (ep is not None) else None

    return start_epoch, had_opt, had_sch

# ==== 작은 유틸들 ==============================================================
def is_main_process() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0

def _safe_makedirs(p: str | None):
    if p and not os.path.exists(p):
        Path(p).mkdir(parents=True, exist_ok=True)

def save_milestone_checkpoint(
    save_dir: str,
    run_id: str,
    fold_idx: int,
    epoch: int,
    model,
    optimizer=None,
    scheduler=None,
    extra_state: dict | None = None,
):
    """마일스톤용 체크포인트 저장(DDP면 rank0만)."""
    if not is_main_process():
        return None
    _safe_makedirs(save_dir)
    state = {
        "epoch": epoch,
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "optimizer": (optimizer.state_dict() if optimizer else None),
        "scheduler": (scheduler.state_dict() if scheduler else None),
        "extra_state": extra_state or {},
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(save_dir, f"{run_id}_fold{fold_idx}_epoch{epoch}.pt")
    torch.save(state, path)
    return path

def append_metrics_csv(csv_path: str, row: dict):
    """CSV에 한 줄 append(헤더 자동). DDP면 rank0만."""
    if not is_main_process():
        return
    _safe_makedirs(os.path.dirname(csv_path))
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def _move_tensors_to_device(batch: dict, device: torch.device) -> dict:
    """딕셔너리에서 텐서만 device로 이동."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def _assert_required_meta_once(batch: dict, config: dict, *, where: str = "train"):
    """
    데이터셋이 생성해주는 메타 텐서가 있는지 1회 확인.
    config의 플래그에 따라 필요한 키만 강제합니다.
    """
    def need(name: str, default: bool) -> bool:
        return bool(config.get(name, default))

    if need("mode_onehot_embedding_split", False):
        assert "x_cat_onehot_meta" in batch, f"[{where}] missing x_cat_onehot_meta"
    if need("global_onehot_embedding_split", False):
        assert "global_features_cat_meta" in batch, f"[{where}] missing global_features_cat_meta"
    if need("edge_onehot_embedding_split", True):  # 기본적으로 엣지 onehot 메타는 필요
        assert "edge_onehot_meta" in batch, f"[{where}] missing edge_onehot_meta"

# ==== output layer 제어 유틸 ===================================================
def _unwrap(m):
    return m.module if hasattr(m, "module") else m

def _get_output_layer(model):
    m = _unwrap(model)
    if hasattr(m, "output_layer"):
        return m.output_layer
    # fallback: 이름으로 탐색
    for name, mod in m.named_modules():
        if name.endswith("output_layer") or name == "head" or "output_layer" in name:
            return mod
    raise AttributeError("GraphormerModel에서 output_layer 모듈을 찾지 못했습니다.")

def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def _add_params_to_optimizer(optimizer, params, lr=None, weight_decay=None):
    existing = {id(p) for g in optimizer.param_groups for p in g["params"]}
    new_params = [p for p in params if id(p) not in existing]
    if new_params:
        group = {"params": new_params}
        if lr is not None: group["lr"] = lr
        if weight_decay is not None: group["weight_decay"] = weight_decay
        optimizer.add_param_group(group)
        return True
    return False

# ==== 마일스톤 평가 ============================================================
@torch.no_grad()
def _evaluate_for_milestone(
    model,
    loader: DataLoader,
    device: torch.device,
    target_type: str = "exp_spectrum",
    split: str = "train",   # "train" | "val" | "test"
):
    model.eval()
    soft_dtw = SoftDTW(use_cuda=(device.type == "cuda"), gamma=0.2, bandwidth=None, normalize=True)

    # 평가 실행
    if split == "train":
        is_val, is_cv = False, False
    elif split == "val":
        is_val, is_cv = True,  True
    elif split == "test":
        is_val, is_cv = True,  False
    else:
        raise ValueError(f"unknown split: {split}")

    res = evaluate_model_metrics(
        model, loader, target_type, device, soft_dtw, sid_loss,
        is_cv=is_cv, best_epoch=None, is_val=is_val
    )

    def pick(resdict: dict, base: str, split: str) -> float:
        if split == "train":
            order = ["train", "tr", "in", "train_cv"]
        elif split == "val":
            order = ["val", "validation", "valid", "val_cv", "cv", "test", "eval"]
        else:  # test
            order = ["test", "te", "eval", "out", "val", "cv"]
        for suf in order:
            key = f"{base}_{suf}"
            if key in resdict and resdict[key] is not None:
                return float(resdict[key])
        if base in resdict and resdict[base] is not None:
            return float(resdict[base])
        for k, v in resdict.items():
            if k.startswith(base + "_") and v is not None:
                return float(v)
        return 0.0

    return {
        "r2":      pick(res, "r2_avg", split),
        "mae":     pick(res, "mae_avg", split),
        "rmse":    pick(res, "rmse_avg", split),
        "sid":     pick(res, "sid_avg", split),
        "softdtw": pick(res, "softdtw_avg", split),
    }

# --------------------------------------------
# 메인 학습 함수 (체크포인트 + 마일스톤 로깅 + 단계적 unfreeze)
# --------------------------------------------
def train_model_ex_porb(
    *,
    config: Dict,
    save_milestones: list[int] | None = None,       # 예: [200,300,400]
    milestone_metrics_csv: str | None = None,       # 생략 시 exp_dir/milestone_metrics.csv
    run_id: str = "run",
    fold_idx: int = -1,
    target_type: str = "exp_spectrum",

    # 손실 구성
    loss_function_full_spectrum: List[str] = ("MSE",),
    loss_function_ex: List[str] = ("SoftDTW",),
    loss_function_prob: List[str] = ("SoftDTW",),

    # 하이퍼파라미터
    num_epochs: int = 10,
    batch_size: int = 64,
    n_pairs: int = 50,
    learning_rate: float = 1e-4,

    # 데이터 소스
    dataset_path: str | None = None,
    test_dataset_path: str | None = None,
    DATASET=None,
    TEST_VAL_DATASET=None,

    # 부가 옵션
    alpha: float = 0.12,
    cv_context: Optional[Dict[str, Any]] = None,
    is_cv: bool = False,
    nominal_feature_vocab=None,
    continuous_feature_names=None,
    global_cat_dim=0,
    global_cont_dim=0,
    global_feature_names: List[str] | None = None,
    ex_normalize: str | None = "ex",
    prob_normalize: str | None = "prob",
    patience: int | None = None,
    num_workers: int = 0,
    debug: bool = False,
    use_gradnorm: bool = True,

    # 그림 저장 옵션
    draw_overlays_on_milestone: bool = False,
    overlay_topk: int = 8,
    overlay_save_all: bool = False,
    figure_milestones: list[int] | None = None,

    # 전이학습/가중치 로드
    init_from_ckpt: str | None = None,
    load_strict: bool | None = None,
    ignore_prefixes: tuple[str, ...] = ("output_layer",),
    resume_from_ckpt: str | None = None,
    resume_auto: bool = False,
    resume_force_start_epoch: int | None = None,
) -> Tuple[Dict, str]:

    # ---------------- 공통 설정 ----------------
    def set_seed(seed: int = 42) -> None:
        import random
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    cv_context = (cv_context or {}).copy()
    cv_context.setdefault("is_cv", bool(is_cv))
    is_cv_effective = bool(cv_context["is_cv"])

    # ---------------- 데이터 로딩 ----------------
    if DATASET is None:
        dataset = UnifiedSMILESDataset(
            csv_file=dataset_path,
            mol_col=config["mol_col"],
            nominal_feature_vocab=config["nominal_feature_vocab"],
            continuous_feature_names=config.get("continuous_feature_names", []),
            global_cat_dim=config.get("global_cat_dim", 0),
            global_cont_dim=config.get("global_cont_dim", 0),
            ATOM_FEATURES_VOCAB=config["ATOM_FEATURES_VOCAB"],
            float_feature_keys=config["float_feature_keys"],
            BOND_FEATURES_VOCAB=config["BOND_FEATURES_VOCAB"],
            GLOBAL_FEATURE_VOCABS_dict=config.get("GLOBAL_FEATURE_VOCABS_dict", None),
            mode=config["mode"],
            max_nodes=config.get("max_nodes", 128),
            multi_hop_max_dist=config.get("multi_hop_max_dist", 5),
            target_type=config.get("target_type", "default"),
            attn_bias_w=config.get("attn_bias_w", 1.0),
            ex_normalize=config.get("ex_normalize", None),
            prob_normalize=config.get("prob_normalize", None),
            nm_dist_mode=config.get("nm_dist_mode", "hist"),
            nm_gauss_sigma=config.get("nm_gauss_sigma", 10.0),
            intensity_normalize=config.get("intensity_normalize", "min_max"),
            intensity_range=config.get("intensity_range", (200, 800)),
            x_cat_mode=config.get("x_cat_mode", "onehot"),
            global_cat_mode=config.get("global_cat_mode", "onehot"),
        )
    else:
        dataset = DATASET

    # 학습 로더
    train_collate = partial(graph_collate_fn, ds=dataset)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=train_collate,
    )

    # 평가용 train 로더
    train_eval_collate = partial(graph_collate_fn, ds=dataset)
    train_eval_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=train_eval_collate,
    )

    # 메타 필수 체크 1회
    for _b in train_eval_loader:
        _assert_required_meta_once(_b, config, where="eval")
        break

    # CV면 val, 아니면 test 데이터셋 구성
    if TEST_VAL_DATASET is not None:
        dataset_eval2 = TEST_VAL_DATASET
    else:
        assert test_dataset_path is not None, "test_dataset_path가 필요합니다."
        dataset_eval2 = UnifiedSMILESDataset(
            csv_file=test_dataset_path,
            mol_col=config["mol_col"],
            nominal_feature_vocab=config["nominal_feature_vocab"],
            continuous_feature_names=config.get("continuous_feature_names", []),
            global_cat_dim=config.get("global_cat_dim", 0),
            global_cont_dim=config.get("global_cont_dim", 0),
            ATOM_FEATURES_VOCAB=config["ATOM_FEATURES_VOCAB"],
            float_feature_keys=config["float_feature_keys"],
            BOND_FEATURES_VOCAB=config["BOND_FEATURES_VOCAB"],
            GLOBAL_FEATURE_VOCABS_dict=config.get("GLOBAL_FEATURE_VOCABS_dict", None),
            mode=config["mode"],
            max_nodes=config.get("max_nodes", 128),
            multi_hop_max_dist=config.get("multi_hop_max_dist", 5),
            target_type=config.get("target_type", "default"),
            attn_bias_w=config.get("attn_bias_w", 1.0),
            ex_normalize=config.get("ex_normalize", None),
            prob_normalize=config.get("prob_normalize", None),
            nm_dist_mode=config.get("nm_dist_mode", "hist"),
            nm_gauss_sigma=config.get("nm_gauss_sigma", 10.0),
            intensity_normalize=config.get("intensity_normalize", "min_max"),
            intensity_range=config.get("intensity_range", (200, 800)),
            x_cat_mode=config.get("x_cat_mode", "onehot"),
            global_cat_mode=config.get("global_cat_mode", "onehot"),
        )

    eval2_collate = partial(graph_collate_fn, ds=dataset_eval2)
    eval2_loader = DataLoader(
        dataset_eval2, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=eval2_collate,
    )
    for _b in eval2_loader:
        _assert_required_meta_once(_b, config, where=("val" if is_cv_effective else "test"))
        break

    # ---------------- 경로/메타 ----------------
    def _slug_from_config(cfg: Dict[str, Any], target_type: str) -> str:
        arch = f"ed{cfg.get('embedding_dim','x')}-L{cfg.get('num_layers','x')}-H{cfg.get('num_heads','x')}"
        data = f"ir{tuple(cfg.get('intensity_range',(200,800)))}-ex{cfg.get('ex_normalize')}-pb{cfg.get('prob_normalize')}"
        loss = f"lf{'+'.join(sorted(cfg.get('loss_function_full_spectrum', [])))}"
        gn_tag = "GN" if cfg.get("use_gradnorm", True) else "NoGN"
        loss = f"{loss}-{gn_tag}"
        key_subset = {k: cfg.get(k) for k in [
            "mode","x_cat_mode","global_cat_mode","multi_hop_max_dist","attn_bias_w",
            "float_feature_keys","global_feature_order","global_multihot_cols"
        ]}
        h = hashlib.sha1(repr(sorted(key_subset.items())).encode()).hexdigest()[:8]
        tag = cfg.get("grid_tag")
        return f"{target_type}_{arch}_{data}_{loss}_{h}" + (f"_{tag}" if tag else "")

    save_root = config.get("save_root", "runs")
    config.setdefault("loss_function_full_spectrum", list(loss_function_full_spectrum))
    config["use_gradnorm"] = bool(use_gradnorm)

    if is_cv_effective:
        fold_idx_disp = int(cv_context.get("fold", fold_idx))
        n_splits = int(cv_context.get("n_splits", 0) or 0)
        subdir = f"cvF{fold_idx_disp:02d}-of-{n_splits:02d}" if (fold_idx_disp >= 0 and n_splits > 0) else "cv"
        split2 = "val"
    else:
        fold_idx_disp = -1
        n_splits = None
        subdir = "final"
        split2 = "test"

    exp_slug = _slug_from_config(config, target_type)
    exp_dir = Path(save_root) / exp_slug / subdir
    exp_dir.mkdir(parents=True, exist_ok=True)

    if milestone_metrics_csv is None:
        milestone_metrics_csv = str(exp_dir / "milestone_metrics.csv")

    if save_milestones is None:
        metric_ms = {200, 300, 400}
    else:
        metric_ms = set(int(x) for x in save_milestones)

    if figure_milestones is None:
        fig_ms: set[int] = set(metric_ms) if draw_overlays_on_milestone else set()
    else:
        fig_ms = set(int(x) for x in figure_milestones)

    best_model_path = str(exp_dir / "best_model.pth")
    best_combined = math.inf
    best_epoch = 0
    no_improve = 0

    # ---------------- 모델/옵티마이저/손실 ----------------
    model = GraphormerModel(config).to(device)

    # --- 모델 생성 직후, 옵티마이저 만들기 전에 ---
    model = GraphormerModel(config).to(device)

    backbone_wd = float(config.get("base_weight_decay", 0.0))  # 언프리즈 전(=백본) WD
    head_wd = config.get("out_unfreeze_weight_decay", None)  # 언프리즈 후 헤드 WD(없으면 백본 WD)

    if config.get("out_freeze", False):
        try:
            head = _get_output_layer(model)
            _set_requires_grad(head, False)  # ★ 헤드 미리 동결
        except Exception:
            pass

    # 이제 trainable_params에는 헤드가 빠짐
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(
        trainable_params,
        lr=learning_rate,
        weight_decay=backbone_wd,
    )

    # ── 기존 동작 유지: out_freeze=False일 때만 더미 forward로 head를 먼저 생성 ──
    if config.get("out_build_in_forward", False) and not config.get("out_freeze", False):
        with torch.no_grad():
            _tmp_loader = DataLoader(
                dataset, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=(device.type == "cuda"),
                collate_fn=partial(graph_collate_fn, ds=dataset),
            )
            _tmp_batch = next(iter(_tmp_loader))
            _tmp_batch = _move_tensors_to_device(_tmp_batch, device)
            _ = model(
                {k: v for k, v in _tmp_batch.items() if k != "targets"},
                targets=_tmp_batch["targets"],
                target_type=target_type
            )

    # --- 동결/해제 스케줄 파라미터
    out_freeze: bool = bool(config.get("out_freeze", False))
    out_unfreeze_epoch: int | None = config.get("out_unfreeze_epoch", None)
    out_unfreeze_lr: float | None = config.get("out_unfreeze_lr", None)
    out_unfreeze_wd: float | None = config.get("out_unfreeze_weight_decay", None)

    if is_main_process():
        n_all = sum(p.numel() for p in model.parameters())
        n_tr  = sum(p.numel() for p in trainable_params)
        print(f"[Param] total={n_all:,} | trainable={n_tr:,} | frozen={n_all - n_tr:,}")

    # (선택) transfer init
    if init_from_ckpt:
        state = torch.load(init_from_ckpt, map_location=device)
        def _try_load(_state, _strict: bool):
            missing, unexpected = model.load_state_dict(_state, strict=_strict)
            if isinstance(missing, (list, tuple)) and isinstance(unexpected, (list, tuple)):
                return missing, unexpected
            return [], []
        ok = False
        if load_strict is None:
            try:
                _try_load(state, True); ok = True
                print(f"[Transfer] loaded strictly from {init_from_ckpt}")
            except Exception:
                ok = False
        if not ok:
            filt = {k: v for k, v in state.items()
                    if not any(k.startswith(pfx) for pfx in ignore_prefixes)}
            _try_load(filt, False)
            print(f"[Transfer] loaded with ignore_prefixes={ignore_prefixes} (strict=False) from {init_from_ckpt}")

    def build_loss_fns(loss_names: List[str], soft_dtw_fn: SoftDTW):
        fns: "OrderedDict[str, callable]" = {}
        for name in loss_names:
            nm = name.upper()
            if nm == "MSE":
                fns[nm] = nn.MSELoss(reduction="mean")
            elif nm == "MAE":
                fns[nm] = nn.L1Loss(reduction="mean")
            elif nm == "HUBER":
                fns[nm] = nn.SmoothL1Loss(reduction="mean")
            elif nm == "SOFTDTW":
                fns[nm] = soft_dtw_fn
            elif nm == "SID":
                def _sid_wrapper(pred, tgt):
                    m = torch.ones_like(pred, dtype=torch.bool)
                    return sid_loss(pred + 1e-8, tgt + 1e-8, m, 1e-8, reduction="mean_valid").mean()
                fns[nm] = _sid_wrapper
            else:
                raise ValueError(f"Unsupported loss name: {name}")
        return fns

    soft_dtw_fn = SoftDTW(use_cuda=(device.type == "cuda"), gamma=0.2, bandwidth=None, normalize=True)
    if target_type == "exp_spectrum":
        loss_fns = build_loss_fns(list(loss_function_full_spectrum), soft_dtw_fn)
    elif target_type == "ex_prob":
        loss_fns = build_loss_fns(list(set(loss_function_ex) | set(loss_function_prob)), soft_dtw_fn)
    else:
        loss_fns = build_loss_fns(["MSE"], soft_dtw_fn)

    loss_names = list(loss_fns.keys())
    num_losses = len(loss_names)
    if use_gradnorm:
        gradnorm = GradNorm(num_losses=num_losses, alpha=alpha)

    history: Dict[str, List[float]] = {"total_loss": [], "normalized_total_loss": []}  # ← 추가
    for nm in loss_names:
        history[f"loss_{nm}"] = []
        history[f"normalized_loss_{nm}"] = []
        history[f"weight_{nm}"] = []
    first_losses_vec: torch.Tensor | None = None

    # ---- Resume from checkpoint (legacy 지원) --------------------------------
    start_epoch = 1
    ckpt_dir = Path(save_root) / _slug_from_config(config, target_type) / ("cv" if is_cv_effective else "final") / "checkpoints"
    target_resume = None

    if resume_from_ckpt:
        target_resume = resume_from_ckpt
    elif resume_auto:
        target_resume = find_latest_checkpoint(ckpt_dir)

    if target_resume:
        print(f"[Resume] load: {target_resume}")
        se, had_opt, had_sch = load_legacy_checkpoint(
            target_resume, model, optimizer=(None if resume_force_start_epoch else optimizer), scheduler=None, map_location=device
        )
        if resume_force_start_epoch is not None:
            start_epoch = int(resume_force_start_epoch)
        elif se is not None:
            start_epoch = int(se)
        else:
            start_epoch = 1

        best_combined = math.inf
        best_epoch = 0
        no_improve = 0
        first_losses_vec = None
        history = {k: [] for k in history.keys()} if isinstance(history, dict) else {"total_loss": []}
        print(f"[Resume] start_epoch={start_epoch} (optimizer_loaded={had_opt})")

    # ---- Resume 시 이미 unfreeze 시점이 지났다면 바로 unfreeze (head가 있다면)
    if out_freeze and out_unfreeze_epoch is not None and start_epoch > out_unfreeze_epoch:
        # head가 생성되어 있다면 즉시 편입
        try:
            head = _get_output_layer(model)
            _set_requires_grad(head, True)
            added = _add_params_to_optimizer(
                optimizer,
                list(head.parameters()),
                lr=out_unfreeze_lr or learning_rate,  # ◀ 헤드만 다른 LR
                weight_decay=head_wd if head_wd is not None else backbone_wd  # ◀ 헤드 WD
            )
            if is_main_process():
                print(f"[Resume-Unfreeze] output_layer unfrozen immediately (start_epoch={start_epoch} > unfreeze={out_unfreeze_epoch})."
                      f"{' (added new param group)' if added else ''}")
        except Exception:
            # 아직 head가 생성되지 않았다면, 첫 forward 후 스케줄 훅에서 다 처리됨
            pass

    # ---------------- 학습 루프 ----------------
    head_frozen_applied = False  # out_freeze=True인 경우, 첫 forward 직후 한 번 freeze 적용 여부

    for epoch in range(start_epoch, num_epochs + 1):
        # --- epoch 시작 Hook: 지정 epoch에 도달하면 unfreeze 시도
        if out_freeze and out_unfreeze_epoch is not None and (epoch == out_unfreeze_epoch):
            try:
                head = _get_output_layer(model)  # 생성되어 있어야 함(첫 배치 이후에는 보장)
                _set_requires_grad(head, True)
                added = _add_params_to_optimizer(
                    optimizer, list(head.parameters()),
                    lr=out_unfreeze_lr or learning_rate,
                    weight_decay=out_unfreeze_wd
                )
                if is_main_process():
                    print(f"[Unfreeze@epoch {epoch}] output_layer UNFROZEN."
                          f"{' (added new param group)' if added else ''} "
                          f"lr={out_unfreeze_lr or learning_rate}, wd={out_unfreeze_wd}")
            except Exception:
                # 아직 head가 생성 전이라면, 아래 첫 배치 이후에도 다시 시도됨
                pass

        t0 = time.time()
        model.train()
        epoch_total = 0.0

        running_losses = torch.zeros(num_losses, device=device)
        running_norms = torch.zeros(num_losses, device=device)
        running_weights = torch.zeros(num_losses, device=device)
        batches = 0
        did_check_meta = False

        for batch in dataloader:
            batches += 1
            if not did_check_meta:
                _assert_required_meta_once(batch, config, where="train"); did_check_meta = True

            batch_t = _move_tensors_to_device(batch, device)
            batch_data = {k: v for k, v in batch_t.items() if k != "targets"}
            targets = batch_t["targets"]

            # ------- 첫 forward 직후: head 생성 시점에 freeze 적용(초기 out_freeze=True인 경우)
            if out_freeze and not head_frozen_applied:
                # 아직 head가 없을 수도 있으므로 먼저 forward로 생성
                outputs = model(batch_data, targets=targets, target_type=target_type)

                # 생성되었으면 즉시 requires_grad=False로 묶어두고, optimizer엔 이미 포함 안 됨
                try:
                    head = _get_output_layer(model)
                    _set_requires_grad(head, False)
                    head_frozen_applied = True
                    if is_main_process():
                        print("[Init-Freeze] output_layer is FROZEN right after first forward.")
                except Exception:
                    # head가 이 시점에도 없다면 모델 구조상 head가 별도로 없을 수 있음
                    pass
            else:
                outputs = model(batch_data, targets=targets, target_type=target_type)

            if target_type == "exp_spectrum":
                mask_batch = batch_t["masks"].bool()  # (B, L)
                y_pred = outputs  # (B, L)
                y_true = targets  # (B, L)

                avg_batch_losses = torch.zeros(num_losses, device=device)

                def _masked_mean_per_sample(x, m):
                    per = (x * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1)
                    return per.mean()

                m2 = mask_batch.squeeze(-1) if mask_batch.dim() == 3 else mask_batch  # (B,L) bool

                for j, nm in enumerate(loss_names):
                    if nm == "SOFTDTW":
                        vals = []
                        for i in range(y_pred.size(0)):
                            valid_idx = m2[i]
                            if not valid_idx.any():
                                continue
                            yp_i = y_pred[i, valid_idx].unsqueeze(0).unsqueeze(-1)  # (1, Li, 1)
                            yt_i = y_true[i, valid_idx].unsqueeze(0).unsqueeze(-1)
                            vals.append(loss_fns[nm](yp_i, yt_i).mean())
                        avg_batch_losses[j] = torch.stack(vals).mean() if len(vals) else torch.tensor(0., device=device)

                    elif nm == "SID":
                        sid_map = sid_loss(
                            y_pred, y_true, m2,
                            threshold=1e-12, eps=1e-12, reduction="mean_valid"
                        )
                        if sid_map.shape != m2.shape and sid_map.t().shape == m2.shape:
                            sid_map = sid_map.t()
                        valid_counts = m2.sum(dim=1).clamp_min(1)
                        per_sample = (sid_map * m2).sum(dim=1) / valid_counts
                        avg_batch_losses[j] = per_sample.mean()

                    elif nm == "MSE":
                        point = (y_pred - y_true) ** 2
                        avg_batch_losses[j] = _masked_mean_per_sample(point, m2)

                    elif nm == "MAE":
                        point = (y_pred - y_true).abs()
                        avg_batch_losses[j] = _masked_mean_per_sample(point, m2)

                    elif nm == "HUBER":
                        point = torch.nn.functional.smooth_l1_loss(y_pred, y_true, reduction="none")
                        avg_batch_losses[j] = _masked_mean_per_sample(point, m2)

                    else:
                        raise ValueError(f"Unsupported loss name: {nm}")

            else:
                raise NotImplementedError("현재 exp_spectrum 경로만 지원")

            # --- 첫 에폭 모드 여부
            first_epoch_mode = (first_losses_vec is None)

            if first_epoch_mode:
                # 에폭 1: 정규화/GradNorm 없이 동일 가중치로 합산
                weights = torch.ones((num_losses,), device=device, dtype=avg_batch_losses.dtype)
                total_loss = torch.sum(avg_batch_losses)
            else:
                # 에폭 2~: 첫 에폭 평균으로 정규화 + (옵션) GradNorm
                norm_losses = avg_batch_losses / first_losses_vec
                if use_gradnorm:
                    weights = gradnorm.compute_weights(norm_losses, model)
                    if not torch.is_tensor(weights):
                        weights = torch.tensor(weights, device=device, dtype=norm_losses.dtype)
                    weights = weights.to(device=device, dtype=norm_losses.dtype)
                else:
                    weights = torch.ones((num_losses,), device=device, dtype=norm_losses.dtype)
                total_loss = torch.sum(weights * norm_losses)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total += float(total_loss.item())
            running_losses += avg_batch_losses.detach()
            if not first_epoch_mode:
                running_norms += (avg_batch_losses / first_losses_vec).detach()
            running_weights += weights.detach()

            # --- 배치 시작부에서 unfreeze 시점이었는데 head가 없어서 실패했다면,
            #     head가 이제 생성됐으니 여기서도 한 번 더 시도
            if out_freeze and out_unfreeze_epoch is not None and (epoch == out_unfreeze_epoch):
                try:
                    head = _get_output_layer(model)
                    if any(p.requires_grad is False for p in head.parameters()):
                        _set_requires_grad(head, True)
                        added = _add_params_to_optimizer(
                            optimizer, list(head.parameters()),
                            lr=out_unfreeze_lr or learning_rate,
                            weight_decay=out_unfreeze_wd
                        )
                        if is_main_process():
                            print(f"[Unfreeze@epoch {epoch}][late] output_layer UNFROZEN."
                                  f"{' (added new param group)' if added else ''}")
                except Exception:
                    pass

        if batches == 0:
            break

        epoch_total /= batches
        mean_losses = running_losses / batches

        if first_losses_vec is None:
            # 첫 에폭: 정규화 기준이 아직 없으므로 1.0으로 고정
            first_losses_vec = mean_losses.detach().clamp_min(1e-12)
            mean_norms = torch.ones_like(mean_losses)
        else:
            mean_norms = running_norms / batches

        mean_weights = running_weights / batches

        # ▶ 추가: 정규화 합(각 loss를 첫 에폭 평균으로 나눈 값의 합)
        norm_total = float(mean_norms.sum().item())
        raw_total = float(epoch_total)

        history["total_loss"].append(raw_total)
        history["normalized_total_loss"].append(norm_total)  # ← 추가 기록
        for j, nm in enumerate(loss_names):
            history[f"loss_{nm}"].append(float(mean_losses[j].item()))
            history[f"normalized_loss_{nm}"].append(float(mean_norms[j].item()))
            history[f"weight_{nm}"].append(float(mean_weights[j].item()))

        elapsed = time.time() - t0
        msg_losses = " | ".join(
            f"{nm}: {history[f'loss_{nm}'][-1]:.4f} (norm {history[f'normalized_loss_{nm}'][-1]:.4f})"
            for nm in loss_names
        )

        # ▶ 출력 변경: raw total 과 norm_sum 둘 다 보여주기
        display_total = norm_total
        print(
            f"Epoch {epoch:03d}/{num_epochs} | total {display_total:.4f} | "
            f"{msg_losses} | w {mean_weights.tolist()} | {elapsed:.1f}s",
            flush=True,
        )

        # ---------------- 성능 마일스톤: ckpt 저장 + train & (val|test) 평가/기록 ----------------
        if (epoch in metric_ms) and is_main_process():
            ckpt_dir = str(exp_dir / "checkpoints")
            ckpt_path = save_milestone_checkpoint(
                save_dir=ckpt_dir,
                run_id=run_id, fold_idx=fold_idx, epoch=epoch,
                model=model, optimizer=optimizer, scheduler=None,
                extra_state={
                    "target_type": target_type,
                    "best_combined": best_combined,
                    "best_epoch": best_epoch,
                    "no_improve": no_improve,
                    "first_losses_vec": (first_losses_vec.detach().cpu() if (first_losses_vec is not None) else None),
                    "history": history,
                    "config_hash": hashlib.sha1(repr(sorted(config.items())).encode()).hexdigest()[:10],
                },
            )

            base_meta = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_id": run_id,
                "is_cv": int(is_cv_effective),
                "fold": fold_idx_disp,
                "n_splits": n_splits,
                "epoch": epoch,
                "ckpt_path": ckpt_path or "",
            }

            # (a) train
            train_metrics = _evaluate_for_milestone(
                model=(_unwrap(model)),
                loader=train_eval_loader, device=device, target_type=target_type, split="train"
            )
            append_metrics_csv(milestone_metrics_csv, {
                **base_meta, "split": "train", "loss": float(epoch_total), **train_metrics
            })

            # (b) val/test
            eval2_metrics = _evaluate_for_milestone(
                model=(_unwrap(model)),
                loader=eval2_loader, device=device, target_type=target_type, split=split2
            )
            append_metrics_csv(milestone_metrics_csv, {
                **base_meta, "split": split2, "loss": None, **eval2_metrics
            })

        # ---------------- 그림 마일스톤: 스펙트럼 오버레이 저장만 별도로 ----------------
        if draw_overlays_on_milestone and (epoch in fig_ms) and is_main_process():
            mdl = _unwrap(model)
            mdl.eval()

            out_dir = Path(exp_dir) / "milestone_figs" / f"epoch{epoch:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            train_csv_for_plot = dataset_path or getattr(dataset, "csv_file", None)
            eval2_csv_for_plot = (
                (test_dataset_path or getattr(dataset_eval2, "csv_file", None))
                if not is_cv_effective else
                (cv_context.get("val_csv") or getattr(TEST_VAL_DATASET, "csv_file", None))
            )

            if train_csv_for_plot:
                evaluate_split_per_sample(
                    mdl, dict(config, out_of_training=True),
                    csv_path=train_csv_for_plot,
                    batch_size=batch_size, num_workers=num_workers, n_pairs=n_pairs,
                    split_name="train", out_dir=str(out_dir),
                    topk_overlay=overlay_topk, save_all_overlays=overlay_save_all
                )

            if eval2_csv_for_plot:
                evaluate_split_per_sample(
                    mdl, dict(config, out_of_training=True),
                    csv_path=eval2_csv_for_plot,
                    batch_size=batch_size, num_workers=num_workers, n_pairs=n_pairs,
                    split_name=("val" if is_cv_effective else "test"), out_dir=str(out_dir),
                    topk_overlay=overlay_topk, save_all_overlays=overlay_save_all
                )

        # ---------------- 베스트/얼리 스톱 ----------------
        if epoch_total + 1e-12 < best_combined:
            best_combined = epoch_total
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            no_improve = 0
        else:
            no_improve += 1
            if patience is not None and no_improve >= patience:
                print(f"[EarlyStop] {patience} epochs no improvement. Stop at {epoch}.")
                break

    # ---------------- 학습 후 평가 (Train & (Val|Test)) ----------------
    results: Dict = {}
    config_out = dict(config); config_out["out_of_training"] = True
    model_eval = GraphormerModel(config_out).to(device)
    model_eval.load_state_dict(torch.load(best_model_path, map_location=device))

    # in-sample (train)
    results.update(
        evaluate_model_metrics(
            model_eval, train_eval_loader, target_type, device,
            SoftDTW(use_cuda=(device.type == "cuda"), gamma=0.2, bandwidth=None, normalize=True),
            sid_loss, is_cv=False, best_epoch=best_epoch, is_val=False
        )
    )

    # eval2: CV면 val, 아니면 test
    results.update(
        evaluate_model_metrics(
            model_eval, eval2_loader, target_type, device,
            SoftDTW(use_cuda=(device.type == "cuda"), gamma=0.2, bandwidth=None, normalize=True),
            sid_loss, is_cv=is_cv_effective, best_epoch=best_epoch, is_val=True
        )
    )

    for k, v in history.items():
        results[f"{k}_history"] = v

    return results, best_model_path

# ====== TRAIN-ONLY MAIN ======
from pathlib import Path
import time
import pandas as pd  # (원하면 안 써도 됨. 결과 저장 안 할 거면 import 제거 가능)
from GP.data_prepare.Pre_Defined_Vocab_Generator import generate_graphormer_config

# ── Silence RDKit + Numba warnings (put this at the very top) ──
import os, warnings
os.environ["RDKIT_LOG_LEVEL"] = "ERROR"
from rdkit import RDLogger, rdBase
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")
rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.info")
try:
    from numba.core.errors import NumbaPerformanceWarning
except Exception:
    from numba import NumbaPerformanceWarning
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings(
    "ignore",
    message=r".*Grid size .* will likely result in GPU under-utilization.*",
    category=NumbaPerformanceWarning,
)

# ▶ import: 같은 파일에 train_model_ex_porb가 있다면 불필요하지만,
#   모듈로 분리되어 있다면 아래 임포트를 유지하세요.
# from GP.models_All.graphormer_layer_modify.train_fn_freeze_unfreeze import train_model_ex_porb
def main():
    """
    하나의 main()에서 ⓐ단일 학습(train-only, = final) ⓑK-Fold CV+Final 두 가지를 모두 지원합니다.
    - Transfer(전이학습) 관련 설정은 완전히 제거했습니다.
    - RUN_MODE == "train" (final)일 때만 config를 저장하고,
      RUN_MODE == "cv+final"일 때는 config를 저장하지 않습니다.
    """
    # ========================== [0] 실행 모드 ==========================
    RUN_MODE = "train"          # "train" = 단일 학습(=final) / "cv+final" = KFold CV 후 Full-Train

    # ========================== [1] 데이터 경로 ==========================
    TRAIN_CSV = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\ABS_stratified_train_clustered_resplit_with_mu_eps_fillZero_fold_not0.csv"
    TEST_CSV  = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\ABS_stratified_train_clustered_resplit_with_mu_eps_fillZero_fold0.csv"
    TRAIN_CSV = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv"
    TEST_CSV = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
    # (scratch config 생성을 위한 스캔 목록)
    CONFIG_LIST = [
        r"Z:\data\학부과정\형광데이터\Full_Spectral_Data_ALL\Complete\final_유전율/ABS_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv",
        r"Z:\data\학부과정\형광데이터\Full_Spectral_Data_ALL\Complete\final_유전율/ABS_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv",
        r"Z:\data\학부과정\형광데이터\Full_Spectral_Data_ALL\Complete\final_유전율/EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv",
        r"Z:\data\학부과정\형광데이터\Full_Spectral_Data_ALL\Complete\final_유전율/EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv",
    ]

    # ========================== [2] 타겟/전역피처 설정 ==========================
    MODE         = "cls_global_model"
    TARGET_TYPE  = "exp_spectrum"
    INT_RANGE    = (200, 800)
    EX_NORM      = "ex_min_max"
    PROB_NORM    = "prob_min_max"
    INT_NORM     = "min_max"
    GLOBAL_ORDER = ["pH_label", "type"]
    CONTINUOUS_GLOBAL_OVERRIDE = ["dielectric_constant_avg"]
    GLOBAL_MH    = {"pH_label", "type"}

    # ========================== [3] 모델/헤드/학습 하이퍼파라미터 ==========================
    # —— Transfer 코드에서 쓰던 값을 기본으로 사용 ——
    H               = 768
    EMBED_DIM       = H
    NUM_HEADS       = 32
    NUM_LAYERS      = 6
    DROPOUT         = 0.1
    ATTN_DROPOUT    = 0.1
    ACT_DROPOUT     = 0.1
    ACT_FN          = "gelu"
    FFN_MULTIPLIER  = 1
    FFN_EMBED_DIM   = FFN_MULTIPLIER * H

    OUT_NUM_LAYERS       = 1
    OUT_HIDDEN_DIMS      = []
    OUT_ACTIVATION       = "relu"
    OUT_FINAL_ACTIVATION = "softplus"
    OUT_BIAS             = True
    OUT_DROPOUT          = 0.0
    OUT_BUILD_IN_FORWARD = False
    OUT_INIT             = "random"
    OUT_CONST_VALUE      = 0.0

    N_SPLITS   = 5               # CV 전용
    NUM_EPOCHS = 10
    BATCH_SIZE = 50
    N_PAIRS    = 50
    BASE_LR    = 1e-4
    BASE_WD    = 1e-5
    USE_GRADNORM   = False
    ALPHA_GRADNORM = 0.12
    NUM_WORKERS    = 0
    DEBUG_MODE     = False

    # ========================== [4] Freeze→Unfreeze 스케줄 ==========================
    FREEZE_UNTIL_EPOCH = 2000     # 0이면 동결X
    HEAD_LR_AFTER      = 1e-4
    HEAD_WD_AFTER      = 1e-5

    # ========================== [5] 손실/마일스톤/출력 디렉토리 ==========================
    LOSS_FUNCTIONS_FULL = ["SID", "MAE"]
    LOSS_FN_EX   = ["MSE", "MAE", "SOFTDTW", "SID"]
    LOSS_FN_PROB = ["MSE", "MAE", "SOFTDTW", "SID"]

    METRIC_MILESTONES = []                 # [200,300,...] 등
    FIGURE_MILESTONES = []                 # 그림 저장 시점
    DRAW_OVERLAYS_ON_MILESTONE = False
    OVERLAY_TOPK     = 8
    OVERLAY_SAVE_ALL = True
    WORK_DIR = "transfer_runs"

    # ========================== [6] Config 생성 ==========================
    from GP.data_prepare.Pre_Defined_Vocab_Generator import generate_graphormer_config
    cfg = generate_graphormer_config(
        dataset_path_list=CONFIG_LIST,
        mode=MODE,
        target_type=TARGET_TYPE,
        intensity_range=INT_RANGE,
        ex_normalize=EX_NORM,
        prob_normalize=PROB_NORM,
        global_feature_order=GLOBAL_ORDER,
        global_multihot_cols=GLOBAL_MH,
        continuous_feature_names=CONTINUOUS_GLOBAL_OVERRIDE,
    )

    # 안전 보정
    if "float_feature_keys" not in cfg:
        cfg["float_feature_keys"] = cfg.get("ATOM_FLOAT_FEATURE_KEYS", [])
    if "output_size" not in cfg and "intensity_range" in cfg:
        s, e = cfg["intensity_range"]; cfg["output_size"] = int(e - s) + 1

    # 연속형 전역피처 보강
    cont_set = set(cfg.get("continuous_feature_names", []))
    cont_set.update(CONTINUOUS_GLOBAL_OVERRIDE)
    cfg["continuous_feature_names"] = sorted(cont_set)

    # 1-batch probe로 실제 입력 차원 확정
    from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset, collate_fn as graph_collate_fn
    from torch.utils.data import DataLoader
    from functools import partial
    try:
        probe_ds = UnifiedSMILESDataset(
            csv_file=TRAIN_CSV,
            mol_col=cfg["mol_col"],
            nominal_feature_vocab=cfg["nominal_feature_vocab"],
            continuous_feature_names=cfg.get("continuous_feature_names", []),
            global_cat_dim=cfg.get("global_cat_dim", 0),
            global_cont_dim=cfg.get("global_cont_dim", 0),
            ATOM_FEATURES_VOCAB=cfg["ATOM_FEATURES_VOCAB"],
            float_feature_keys=cfg.get("float_feature_keys", cfg.get("ATOM_FLOAT_FEATURE_KEYS", [])),
            BOND_FEATURES_VOCAB=cfg["BOND_FEATURES_VOCAB"],
            GLOBAL_FEATURE_VOCABS_dict=cfg.get("GLOBAL_FEATURE_VOCABS_dict"),
            mode=cfg["mode"],
            max_nodes=cfg.get("max_nodes", 128),
            multi_hop_max_dist=cfg.get("multi_hop_max_dist", 5),
            target_type=cfg.get("target_type", TARGET_TYPE),
            attn_bias_w=cfg.get("attn_bias_w", 1.0),
            ex_normalize=cfg.get("ex_normalize"),
            prob_normalize=cfg.get("prob_normalize"),
            nm_dist_mode=cfg.get("nm_dist_mode", "hist"),
            nm_gauss_sigma=cfg.get("nm_gauss_sigma", 10.0),
            intensity_normalize=cfg.get("intensity_normalize", INT_NORM),
            intensity_range=cfg.get("intensity_range", INT_RANGE),
            x_cat_mode=cfg.get("x_cat_mode", "onehot"),
            global_cat_mode=cfg.get("global_cat_mode", "onehot"),
        )
        probe_loader = DataLoader(probe_ds, batch_size=1, shuffle=False,
                                  collate_fn=partial(graph_collate_fn, ds=probe_ds))
        b0 = next(iter(probe_loader))
        F_cat = int(getattr(b0.get("x_cat_onehot", None), "shape", [0, 0])[-1]) if "x_cat_onehot" in b0 else int(cfg.get("num_categorical_features", 0))
        F_cont = int(getattr(b0.get("x_cont", None), "shape", [0, 0])[-1]) if "x_cont" in b0 else int(cfg.get("num_continuous_features", 0))
        num_edges = int(getattr(b0.get("attn_edge_type", None), "shape", [0, 0])[-1]) if "attn_edge_type" in b0 else int(cfg.get("num_edges", 14))
        cfg.update({
            "num_categorical_features": F_cat,
            "num_continuous_features":  F_cont,
            "num_edges": num_edges,
            "num_in_degree": 6,
            "num_out_degree": 6,
            "num_spatial": int(cfg.get("num_spatial", 512)),
            "deg_clip_max": 6,
        })
        print(f"[Probe] F_cat={F_cat} | F_cont={F_cont} | edges={num_edges}")
    except Exception as e:
        print(f"[WARN] Probe 실패: {e}")

    # 최종 덮어쓰기(모델/헤드/학습/스케줄/손실)
    cfg.update({
        # 데이터/타겟
        "mode": MODE,
        "target_type": TARGET_TYPE,
        "intensity_range": INT_RANGE,
        "ex_normalize": EX_NORM,
        "prob_normalize": PROB_NORM,
        "intensity_normalize": INT_NORM,
        "multi_hop_max_dist": 5,
        "attn_bias_w": 1.0,
        "max_nodes": 128,

        # 모델 구조
        "embedding_dim": EMBED_DIM,
        "num_attention_heads": NUM_HEADS,
        "num_encoder_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "attention_dropout": ATTN_DROPOUT,
        "activation_dropout": ACT_DROPOUT,
        "activation_fn": ACT_FN,
        "ffn_multiplier": FFN_MULTIPLIER,
        "ffn_embedding_dim": FFN_EMBED_DIM,

        # 출력 헤드
        "out_num_layers": OUT_NUM_LAYERS,
        "out_hidden_dims": OUT_HIDDEN_DIMS,
        "out_activation": OUT_ACTIVATION,
        "out_final_activation": OUT_FINAL_ACTIVATION,
        "out_bias": OUT_BIAS,
        "out_dropout": OUT_DROPOUT,
        "out_build_in_forward": OUT_BUILD_IN_FORWARD,
        "out_init": OUT_INIT,
        "out_const_value": OUT_CONST_VALUE,

        # 손실/GradNorm
        "loss_function_full_spectrum": list(LOSS_FUNCTIONS_FULL),
        "use_gradnorm": bool(USE_GRADNORM),
        "alpha_gradnorm": ALPHA_GRADNORM,

        # 로깅용(실제 옵티마이저는 run 인자에서 적용)
        "base_learning_rate": BASE_LR,
        "base_weight_decay": BASE_WD,

        # Freeze→Unfreeze(트레이너가 인지)
        "out_freeze": bool(FREEZE_UNTIL_EPOCH and FREEZE_UNTIL_EPOCH > 0),
        "out_unfreeze_epoch": int(FREEZE_UNTIL_EPOCH) if (FREEZE_UNTIL_EPOCH and FREEZE_UNTIL_EPOCH > 0) else None,
        "out_unfreeze_lr": float(HEAD_LR_AFTER) if (FREEZE_UNTIL_EPOCH and FREEZE_UNTIL_EPOCH > 0) else None,
        "out_unfreeze_weight_decay": float(HEAD_WD_AFTER) if (FREEZE_UNTIL_EPOCH and FREEZE_UNTIL_EPOCH > 0) else None,
    })

    # ========================== [7] 실행 분기 ==========================
    if RUN_MODE == "cv+final":
        # ── KFold CV + Full-Train ── (config 저장 안 함)
        from GP.models_All.graphormer_layer_modify.graphormer_LayerModify_CV_milestone import (
            run_cv_and_final_training_milestone,
        )
        run_cv_and_final_training_milestone(
            config=cfg,
            train_csv=TRAIN_CSV,
            test_csv=TEST_CSV,
            n_splits=N_SPLITS,
            save_path=os.path.join(WORK_DIR, "cv_results_exp_spectrum_milestone.csv"),
            loss_functions_full=list(LOSS_FUNCTIONS_FULL),

            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            n_pairs=N_PAIRS,
            alpha=ALPHA_GRADNORM,
            num_workers=NUM_WORKERS,
            debug=DEBUG_MODE,
            use_gradnorm=USE_GRADNORM,

            base_learning_rate=BASE_LR,
            base_weight_decay=BASE_WD,

            freeze_head_until_epoch=FREEZE_UNTIL_EPOCH,
            head_lr_after_unfreeze=HEAD_LR_AFTER,
            head_wd_after_unfreeze=HEAD_WD_AFTER,

            save_milestones=METRIC_MILESTONES,
            figure_milestones=FIGURE_MILESTONES,
            draw_overlays_on_milestone=DRAW_OVERLAYS_ON_MILESTONE,
            overlay_topk=OVERLAY_TOPK,
            overlay_save_all=OVERLAY_SAVE_ALL,

            init_from_ckpt=None,            # ← 전이학습 제거
            load_strict=None,
            ignore_prefixes=("output_layer",),
        )

    else:
        # ── 단일 학습(train-only = final) ── (config 저장 함)
        results, best_path = train_model_ex_porb(
            config=cfg,
            target_type=TARGET_TYPE,
            loss_function_full_spectrum=list(LOSS_FUNCTIONS_FULL),
            loss_function_ex=list(LOSS_FN_EX),
            loss_function_prob=list(LOSS_FN_PROB),

            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            n_pairs=N_PAIRS,
            learning_rate=BASE_LR,

            dataset_path=TRAIN_CSV,
            test_dataset_path=TEST_CSV,

            alpha=ALPHA_GRADNORM,
            patience=None,
            num_workers=NUM_WORKERS,
            debug=DEBUG_MODE,
            use_gradnorm=USE_GRADNORM,

            save_milestones=METRIC_MILESTONES,
            milestone_metrics_csv=None,
            draw_overlays_on_milestone=DRAW_OVERLAYS_ON_MILESTONE,
            overlay_topk=OVERLAY_TOPK,
            overlay_save_all=OVERLAY_SAVE_ALL,
            figure_milestones=FIGURE_MILESTONES,

            init_from_ckpt=None,            # ← 전이학습 제거
            load_strict=None,
            ignore_prefixes=("output_layer",),

            resume_from_ckpt=None,
            resume_auto=False,
            resume_force_start_epoch=None,

            is_cv=False,
            cv_context={"is_cv": False},
            run_id=f"final_{time.strftime('%Y%m%d-%H%M%S')}",
            fold_idx=-1,
        )

        # --- (final 전용) config 저장 ---
        try:
            import json
            out_dir = Path(best_path).parent
            with open(out_dir / "config_train.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] config_train.json → {out_dir}")
        except Exception as e:
            print(f"[WARN] config 저장 실패: {e}")

        # 결과 단일 CSV 저장
        import json as _json, numpy as _np, pandas as _pd
        def _cell(v):
            if isinstance(v, _np.generic): return v.item()
            try:
                import torch as _torch
                if isinstance(v, _torch.Tensor):
                    return _json.dumps(v.detach().cpu().tolist(), ensure_ascii=False)
            except Exception:
                pass
            if isinstance(v, (list, dict, tuple)):
                return _json.dumps(v, ensure_ascii=False)
            return v

        out_csv = Path(best_path).parent / "results_one_row.csv"
        row = {k: _cell(v) for k, v in results.items()}
        df = _pd.DataFrame([row])
        if out_csv.exists():
            df.to_csv(out_csv, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_csv}")
        print("Best model:", best_path)
        print("Milestone metrics CSV:", (Path(best_path).parent / "milestone_metrics.csv"))


if __name__ == "__main__":
    main()
