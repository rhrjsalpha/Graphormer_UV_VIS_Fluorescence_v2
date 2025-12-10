# -*- coding: utf-8 -*-
"""
Per-sample evaluation utilities for Graphormer UV/Vis models (UPDATED).

핵심 변경 사항 (현재 코드베이스와 호환):
1) 모델/배치 처리
   - 배치에서 "텐서만" device로 이동하고(meta dict는 그대로 유지) 모델에 전달
   - collate_fn(batch, dataset) 서명 유지
   - GraphormerModel.forward(batched_data, targets=None, target_type=...) 사용

2) 출력 레이어 로딩 안정화
   - 학습 시 output_layer가 존재하므로, 평가 시에도 로드 전에 output_size를 명시해서
     out_of_training=True로 모델을 초기화 → load_state_dict()가 안전하게 통과되도록 함
   - output_size는 train/test CSV 중 하나를 미리 한 배치 읽어 자동 추론

3) 타깃 타입 및 마스킹 처리
   - 현재 주력인 target_type="exp_spectrum" 경로를 안정화 (마스크 적용 후 per-sample 메트릭)
   - nm/wavelength 축은 config["intensity_range"] 및 길이 L에 맞춰 자동 생성

4) 손실/지표
   - SoftDTW: GP.Custom_Loss.soft_dtw_cuda.SoftDTW (normalize=True)
   - SID: GP.Custom_Loss.SID_loss.sid_loss(reduction='mean_valid')에 맞춰 호출
   - per-sample R2 / RMSE / MAE / SID / SoftDTW 계산 및 CSV+히스토그램 저장
   - overlay: 기본은 RMSE 상위 k개만, 옵션(save_all_overlays=True)로 전부 저장 가능

사용 예)
from GP.After_Training_Module.Caculate_Visualize_Each_Molecule import evaluate_train_and_test_per_sample

csv_train, csv_test, out_dir = evaluate_train_and_test_per_sample(
    best_model_path=best_path,
    config=config,
    dataset_train_path=dataset_train_path,
    dataset_test_path=dataset_test_path,
    batch_size=50,
    num_workers=config.get("num_workers", 0),
    n_pairs=50,
    out_dir=None,  # None이면 eval_outputs_타임스탬프 폴더 자동 생성
    topk_overlay=8,
    save_all_overlays=False,
)
"""
from __future__ import annotations

import os
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===== 프로젝트 모듈 =====
from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset, collate_fn
from GP.models_All.graphormer_layer_modify.graphormer_LayerModify import GraphormerModel
from GP.Custom_Loss.soft_dtw_cuda import SoftDTW
from GP.Custom_Loss.SID_loss import sid_loss  # ← 커스텀 SID
import textwrap
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from functools import partial
from GP.data_prepare.DataLoader_QMData_All import collate_fn as graph_collate_fn

__all__ = [
    "evaluate_split_per_sample",
    "evaluate_train_and_test_per_sample",
]

def _keep_longest_run(mask_np: np.ndarray, min_run: int = 5) -> np.ndarray:
    """True가 띄엄띄엄 있으면 꼬리처럼 보인다.
    가장 긴 연속 구간만 남기고 나머지 True는 False로 만든다."""
    idx = np.flatnonzero(mask_np)
    if idx.size == 0:
        return mask_np
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    best = max(runs, key=len)
    out = np.zeros_like(mask_np, dtype=bool)
    if len(best) >= min_run:
        out[best] = True
    return out

def _refine_mask(mask_np: np.ndarray, y_true_np: np.ndarray, eps: float = 1e-8, min_run: int = 5) -> np.ndarray:
    m = mask_np.astype(bool).copy()
    m &= (np.asarray(y_true_np) > eps)  # 거의 0인 값은 제외(선택)
    return _keep_longest_run(m, min_run=min_run)

def _ser_array(x, decimals: int = 6) -> str:
    """np array / torch tensor를 CSV에 넣기 위한 compact JSON 직렬화."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    # float로 저장하고 싶다면 아래 줄 유지, 아니면 상황에 따라 제거 가능
    arr = arr.astype(np.float64, copy=False)

    if decimals is not None:
        # np.round 의 키워드는 'decimals' 입니다!
        arr = np.round(arr, decimals=decimals)

    return json.dumps(arr.tolist(), ensure_ascii=False, separators=(",", ":"))

# ----------------------------- helpers -----------------------------
def _make_dataset_from_config(config: Dict, csv_path: str) -> UnifiedSMILESDataset:
    """config 기반으로 Dataset을 생성 (train/test 공통)."""
    return UnifiedSMILESDataset(
        csv_file=csv_path,
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
        target_type=config.get("target_type", "exp_spectrum"),
        attn_bias_w=config.get("attn_bias_w", 1.0),
        ex_normalize=config.get("ex_normalize", None),
        prob_normalize=config.get("prob_normalize", None),
        nm_dist_mode=config.get("nm_dist_mode", "hist"),
        nm_gauss_sigma=config.get("nm_gauss_sigma", 10.0),
        intensity_normalize=config.get("intensity_normalize", "min_max"),
        intensity_range=config.get("intensity_range", (200, 800)),
        # (데이터 준비 모드 지정이 있는 경우)
        x_cat_mode=config.get("x_cat_mode", "onehot"),
        global_cat_mode=config.get("global_cat_mode", "onehot"),
    )


def _make_loader(dataset: UnifiedSMILESDataset, batch_size: int, num_workers: int,
                 n_pairs: int = 50, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=partial(graph_collate_fn, ds=dataset),
        #collate_fn=lambda b, ds=dataset: collate_fn(b, ds),
    )


def _safe_r2(y_true_np: np.ndarray, y_pred_np: np.ndarray) -> float:
    """분산 0 케이스에서 정의 가능한 R^2 (per-sample)."""
    ss_res = float(np.sum((y_pred_np - y_true_np) ** 2))
    mu = float(np.mean(y_true_np))
    ss_tot = float(np.sum((y_true_np - mu) ** 2))
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - (ss_res / ss_tot)


def _build_nm_grid(config: Dict, L: int) -> np.ndarray:
    """시각화용 nm 그리드 생성. config의 intensity_range/nm_step와 길이 L을 일치시킴."""
    start_nm, end_nm = config["intensity_range"]
    step = float(config.get("nm_step", 1))
    n_bins = int(round((end_nm - start_nm) / step)) + 1
    if n_bins == L:
        return np.linspace(start_nm, end_nm, L, dtype=float)
    return np.linspace(start_nm, end_nm, L, dtype=float)


def _sid_per_sample(pred_1d: torch.Tensor, true_1d: torch.Tensor, device: torch.device) -> float:
    """
    커스텀 SID 손실을 per‑sample로 계산.
    - pred/true: 1D (Li,)
    - 내부에서 (1, Li, 1) 형태로 맞춤
    """
    yp = pred_1d.view(1, -1, 1).to(device)
    yt = true_1d.view(1, -1, 1).to(device)
    mask = torch.ones_like(yp, dtype=torch.bool, device=device)
    return sid_loss(yp + 1e-8, yt + 1e-8, mask, 1e-8, reduction="mean_valid").mean().item()


def _softdtw_per_sample(softdtw, pred_1d: torch.Tensor, true_1d: torch.Tensor) -> float:
    """
    pred_1d/true_1d: shape (L,)
    SoftDTW가 기대하는 (B=1, L, 1)로 재배치. 텐서의 기존 device를 그대로 사용.
    """
    yp = pred_1d.view(1, -1, 1)  # (1, L, 1)
    yt = true_1d.view(1, -1, 1)  # (1, L, 1)
    return softdtw(yp, yt).item()


def _per_sample_metrics(
    yp_1d: torch.Tensor,
    yt_1d: torch.Tensor,
    softdtw: SoftDTW,
    sid_fn,
    eps: float = 1e-8,
) -> Dict[str, float]:
    # (L,) 보장
    yp_1d = yp_1d.view(-1)
    yt_1d = yt_1d.view(-1)

    # 스칼라 메트릭
    mae = torch.mean(torch.abs(yp_1d - yt_1d)).item()
    rmse = torch.sqrt(torch.mean((yp_1d - yt_1d) ** 2)).item()
    sdtw = _softdtw_per_sample(softdtw, yp_1d, yt_1d)

    # SID (마스크 전부 유효)
    m = torch.ones_like(yp_1d, dtype=torch.bool).view(1, -1, 1)
    sid = sid_fn(
        yp_1d.view(1, -1, 1) + eps,
        yt_1d.view(1, -1, 1) + eps,
        m,
        eps,
        # reduction="mean_valid",  # 우리 SID 구현과 맞춤
    ).mean().item()

    r2 = _safe_r2(yt_1d.detach().cpu().numpy(), yp_1d.detach().cpu().numpy())
    return dict(R2=r2, RMSE=rmse, MAE=mae, SID=sid, SoftDTW=sdtw)


def _peek_output_size_for_model(config: Dict, csv_path: str, batch_size: int, num_workers: int) -> int:
    """
    out_of_training=True 로 모델을 만들기 위해 output_size를 미리 추정한다.
    - exp_spectrum: targets.size(-1)
    - ex_prob: n_pairs * 2
    - nm_distribution: targets.size(-1)
    """
    ds = _make_dataset_from_config(config, csv_path)
    dl = _make_loader(ds, max(1, batch_size), num_workers, shuffle=False)
    batch = next(iter(dl))
    targets = batch["targets"]
    tt = config.get("target_type", "exp_spectrum")
    if tt == "ex_prob":
        return targets.size(1) * 2
    else:
        return targets.size(-1)


def _move_tensors_to_device(batch: dict, device: torch.device) -> dict:
    """dict에서 텐서만 device로 이동(meta dict는 그대로 유지)."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def _mol_from_strings(smiles: str | None, inchi: str | None):
    m = None
    if inchi and isinstance(inchi, str) and inchi.strip():
        try: m = Chem.MolFromInchi(inchi)
        except Exception: m = None
    if m is None and smiles and isinstance(smiles, str) and smiles.strip():
        try: m = Chem.MolFromSmiles(smiles)
        except Exception: m = None
    return m

def _wrap(s: str | None, width: int = 38) -> str:
    if not s: return ""
    return "\n".join(textwrap.wrap(str(s), width=width))

def _save_overlay_with_sidepanel(
    out_path: str,
    nm_grid: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    info: dict,
    title: str,
):
    """왼쪽: 스펙트럼, 오른쪽: 분자 그림 + 텍스트 패널
    - True: 마스크 구간만(실선)
    - Pred: 마스크 구간(실선) + 비마스크 구간(점선/옅은색) 모두 시각화
    """
    fig = plt.figure(figsize=(14, 8), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 2.8])
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.12,
        wspace=0.35
    )
    ax = fig.add_subplot(gs[0, 0])

    mk = mask.astype(bool)

    # --- True 곡선: 마스크 구간만 실선 ---
    y_true_masked = np.where(mk, y_true, np.nan)
    line_true, = ax.plot(nm_grid, y_true_masked, linewidth=1.6, label="True (masked)")

    # --- Pred 곡선: 마스크 구간(실선) + 비마스크 구간(점선) ---
    y_pred_masked   = np.where(mk,  y_pred, np.nan)
    y_pred_unmasked = np.where(~mk, y_pred, np.nan)

    # 먼저 마스크 구간을 그려 색상을 받아옵니다(두 구간 색을 통일).
    line_pred_masked, = ax.plot(nm_grid, y_pred_masked,
                                linewidth=1.6, linestyle="-", label="Pred (masked)")
    pred_color = line_pred_masked.get_color()

    # 비마스크 구간은 같은 색으로 점선/옅은 투명도
    ax.plot(nm_grid, y_pred_unmasked,
            linewidth=1.2, linestyle="--", alpha=0.7, color=pred_color, label="Pred (unmasked)")

    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (norm)")
    ax.legend(loc="best")

    # ---- 오른쪽 사이드 패널(분자/텍스트) ----
    gs_right = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.2])
    ax_mol  = fig.add_subplot(gs_right[0, 0])
    ax_txt  = fig.add_subplot(gs_right[1, 0])
    ax_mol.axis("off"); ax_txt.axis("off")

    mol = _mol_from_strings(info.get("SMILES"), info.get("InChI"))
    if mol is not None:
        img = Draw.MolToImage(mol, size=(260, 220))
        ax_mol.imshow(img)
        ax_mol.set_xticks([]); ax_mol.set_yticks([])
    else:
        ax_mol.text(0.5, 0.5, "No molecule", ha="center", va="center")

    # 텍스트 구성
    lines = []
    if info.get("InChI"):  lines += ["InChI:", _wrap(info["InChI"], 38), ""]
    elif info.get("SMILES"): lines += ["SMILES:", _wrap(info["SMILES"], 38), ""]
    for k, lab in (("pH_label","pH"), ("Solvent","Solvent"), ("type","Type")):
        if info.get(k) is not None:
            lines.append(f"{lab}: {info[k]}")

    metrics_order = ("R2", "RMSE", "MAE", "SID", "SoftDTW")
    have_any_metric = False
    for k in metrics_order:
        if k in info and info[k] is not None:
            have_any_metric = True
            try:
                lines.append(f"{k}: {float(info[k]):.4f}")
            except Exception:
                lines.append(f"{k}: {info[k]}")

    if have_any_metric:
        N = info.get("N")
        lines.append("")
        lines.append("Ranks:")
        if "R2_rank" in info:
            lines.append(f"  R2: #{info['R2_rank']}" + (f"/{N}" if N else ""))
        for k in ("RMSE", "MAE", "SID", "SoftDTW"):
            rk = info.get(f"{k}_rank")
            if rk is not None:
                lines.append(f"  {k}: #{rk}" + (f"/{N}" if N else ""))
        if "rank_sum" in info:
            rs = int(info["rank_sum"])
            rsr = info.get("rank_sum_rank")
            tail = f"  (#{rsr}/{N})" if (rsr and N) else ""
            lines += ["", f"Rank Sum: {rs}{tail}"]

    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
    ax_txt.text(0.02, 0.98, "\n".join(lines),
                va="top", ha="left", family="monospace", fontsize=9, wrap=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------- main APIs -----------------------------
@torch.no_grad()
def evaluate_split_per_sample(
    model: nn.Module,
    config: Dict,
    csv_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    n_pairs: int = 50,
    split_name: str = "train",
    out_dir: str = "eval_outputs",
    topk_overlay: int = 8,
    save_all_overlays: bool = False,
) -> str:
    import pandas as pd
    """
    주어진 split(csv_path)에 대해 per-row 메트릭을 계산/저장하고, 히스토그램/overlay 이미지를 저장.
    Returns
    -------
    csv_out: str
        저장된 per-sample metrics CSV 경로
    """
    device = next(model.parameters()).device
    os.makedirs(out_dir, exist_ok=True)

    dataset = _make_dataset_from_config(config, csv_path)

    df_meta = pd.read_csv(csv_path)
    all_info = []

    loader = _make_loader(dataset, batch_size, num_workers, n_pairs, shuffle=False)

    softdtw = SoftDTW(use_cuda=(device.type == "cuda"), gamma=0.2, bandwidth=None, normalize=True)

    rows: List[Dict] = []
    all_true, all_pred, all_mask, all_row_id = [], [], [], []

    def _pick_head(x):
        if isinstance(x, dict):
            for key in ("exp_spectrum", "exp", "spectrum"):
                if key in x:
                    return x[key]
            return next(iter(x.values()))
        if isinstance(x, (tuple, list)):
            return x[0]
        return x

    def _ensure_BL(t: torch.Tensor) -> torch.Tensor:
        """
        (L,)   -> (1, L)
        (B,L)  -> (B, L)
        (B,L,1)-> (B, L)
        (1,L)  -> (1, L)
        """
        if t.ndim == 1:
            t = t.unsqueeze(0)
        if t.ndim == 3 and t.size(-1) == 1:
            t = t.squeeze(-1)
        assert t.ndim == 2, f"Expect 2D (B,L), got shape {tuple(t.shape)}"
        return t

    model.eval()
    running_row = 0
    for batch in loader:
        # 텐서만 device로 이동 (메타 dict/리스트는 그대로 유지)
        batch_data = {}
        for k, v in batch.items():
            if k in ("targets", "masks"):
                continue
            if torch.is_tensor(v):
                batch_data[k] = v.to(device, non_blocking=True)
            else:
                batch_data[k] = v

        targets = batch["targets"].to(device)              # (B, L) 혹은 (L,)
        masks   = batch["masks"].to(device)                # (B, L) 0/1 또는 bool

        # 모델 추론: 새 Graphormer는 targets를 주면 출력 크기를 확정함
        raw   = model(batch_data, targets=targets, target_type=config["target_type"])
        preds = _ensure_BL(_pick_head(raw))
        targets = _ensure_BL(targets)
        masks   = _ensure_BL(masks)

        # 안전장치: 길이 맞추기
        if preds.size(1) != targets.size(1):
            L = min(preds.size(1), targets.size(1))
            preds   = preds[:, :L]
            targets = targets[:, :L]
            masks   = masks[:, :L]

        # 마스크를 bool로 확실히
        if masks.dtype != torch.bool:
            masks = masks != 0

        B, L = preds.shape
        nm_grid = _build_nm_grid(config, L)

        for i in range(B):
            mask_i = masks[i]  # (L,) bool
            if mask_i.sum().item() == 0:
                continue

            # 2D 전제 인덱싱
            yp_i = preds[i, mask_i]  # (Li,)
            yt_i = targets[i, mask_i]  # (Li,)

            metrics = _per_sample_metrics(yp_i, yt_i, softdtw, sid_loss)

            # row 식별자
            rid = None
            for k in ("row_idx", "idx", "uid", "smiles", "InChI"):
                if k in batch:
                    val = batch[k][i]
                    rid = val.item() if torch.is_tensor(val) and val.dim() == 0 else (
                        val if isinstance(val, (str, int)) else None
                    )
                    if rid is not None:
                        break
            if rid is None:
                rid = running_row

            row_meta = None
            try:
                rid_int = int(rid)
                if 0 <= rid_int < len(df_meta):
                    row_meta = df_meta.iloc[rid_int]  # 행 전체(Series)
            except Exception:
                row_meta = None

            # 메타 수집(기존 info 로직 그대로 사용)
            def _get(col):
                # 1) 배치 안에서 먼저 찾기
                if col in batch:
                    v = batch[col][i]
                    if torch.is_tensor(v):
                        v = v.item() if v.dim() == 0 else None
                    # 문자열은 공백 제거 후 유효할 때만, 숫자는 그대로
                    if isinstance(v, str):
                        if v.strip():
                            return v
                    elif isinstance(v, (int, float)):
                        return v

                # 2) CSV 메타에서 보조로 찾기
                if row_meta is not None and col in df_meta.columns:
                    return row_meta[col]
                # row_id가 인덱스로 안 맞을 수도 있으니, 동일 배치 위치 i로도 한번 더 시도
                if col in df_meta.columns and 0 <= i < len(df_meta):
                    try:
                        return df_meta.iloc[i][col]
                    except Exception:
                        pass
                return None

            info = dict(
                row_id=rid,
                SMILES=_get("SMILES") or _get("smiles"),
                InChI=_get("InChI") or _get("inchi"),
                pH_label=_get("pH_label"),
                Solvent=_get("Solvent"),
                type=_get("type"),
            )
            all_info.append(info)

            # ----- 곡선 직렬화: 마스크 적용 구간 + (옵션) 전체 곡선 -----
            nm_masked = _build_nm_grid(config, preds.size(1))[mask_i.detach().cpu().numpy()]
            row_dict = dict(
                split=split_name,
                row_id=rid,

                # 메타데이터
                SMILES=info["SMILES"],
                InChI=info["InChI"],
                pH_label=info["pH_label"],
                Solvent=info["Solvent"],
                type=info["type"],

                # 길이 정보
                L=int(preds.size(1)),
                Li=int(mask_i.sum().item()),

                # 그래프(마스크 적용)
                nm=_ser_array(nm_masked),
                y_true=_ser_array(yt_i),
                y_pred=_ser_array(yp_i),
                mask_idx=_ser_array(np.where(mask_i.detach().cpu().numpy())[0]),

                # 지표
                **metrics,
            )

            # (옵션) 전체 곡선도 함께 저장하려면 아래 3줄 주석 해제
            row_dict["nm_full"] = _ser_array(_build_nm_grid(config, preds.size(1)))
            row_dict["y_true_full"] = _ser_array(targets[i])
            row_dict["y_pred_full"] = _ser_array(preds[i])

            rows.append(row_dict)

            # overlay 저장용(원본 유지)
            all_true.append(targets[i].detach().cpu().numpy())
            all_pred.append(preds[i].detach().cpu().numpy())
            all_mask.append(mask_i.detach().cpu().numpy())
            all_row_id.append(rid)

            running_row += 1

    # ---------- 저장: CSV ----------
    import pandas as pd
    df = pd.DataFrame(rows)
    # ★ 순위 계산 (split별)
    N = len(df)
    by_row = {}  # row_id -> dict(지표/순위)
    if N > 0:
        df["R2_rank"] = df["R2"].rank(ascending=False, method="min").astype(int)
        df["RMSE_rank"] = df["RMSE"].rank(ascending=True, method="min").astype(int)
        df["MAE_rank"] = df["MAE"].rank(ascending=True, method="min").astype(int)
        df["SID_rank"] = df["SID"].rank(ascending=True, method="min").astype(int)
        df["SoftDTW_rank"] = df["SoftDTW"].rank(ascending=True, method="min").astype(int)
        rank_cols = ["R2_rank", "RMSE_rank", "MAE_rank", "SID_rank", "SoftDTW_rank"]
        df["rank_sum"] = df[rank_cols].sum(axis=1)
        df["rank_sum_rank"] = df["rank_sum"].rank(ascending=True, method="min").astype(int)
        # row_id -> 지표/순위 매핑 (overlay에서 빠르게 접근용)
        by_row = df.set_index("row_id").to_dict(orient="index")

    csv_out = os.path.join(out_dir, f"per_sample_metrics_{split_name}.csv")
    df.to_csv(csv_out, index=False)

    # (옵션) overlay 저장
    if save_all_overlays:
        for idx, rid in enumerate(all_row_id):
            yt = all_true[idx];
            yp = all_pred[idx];
            mk = all_mask[idx].astype(bool)
            L = len(yt);
            nm_grid = _build_nm_grid(config, L)

            info = dict(all_info[idx]) if idx < len(all_info) else {"row_id": rid}
            # ★ 지표/순위 주입
            if rid in by_row:
                r = by_row[rid]
                for k in ("R2", "RMSE", "MAE", "SID", "SoftDTW",
                          "R2_rank", "RMSE_rank", "MAE_rank", "SID_rank", "SoftDTW_rank",
                          "rank_sum", "rank_sum_rank"):
                    if k in r: info[k] = r[k]
                info["N"] = N

            # ★ 파일명: overlay_{split}_row_{rank_sum}.png (rank_sum 없으면 -1)
            rs = int(info.get("rank_sum", -1))
            out_png = os.path.join(out_dir, f"overlay_{split_name}_row_{rs}.png")

            _save_overlay_with_sidepanel(out_png, nm_grid, yt, yp, mk, info,
                                         f"{split_name} row_id={rid}")

    # ---------- 저장: 히스토그램 ----------
    metrics_cols = ["R2", "RMSE", "MAE", "SID", "SoftDTW"]
    for m in metrics_cols:
        plt.figure()
        plt.hist(df[m].dropna().values, bins=30)
        plt.title(f"{split_name} {m} distribution (n={len(df)})")
        plt.xlabel(m); plt.ylabel("Count")
        png_out = os.path.join(out_dir, f"hist_{split_name}_{m}.png")
        plt.tight_layout(); plt.savefig(png_out, dpi=150); plt.close()

    # ---------- 저장: RMSE top‑k overlay ----------
    if (not save_all_overlays) and topk_overlay and len(df) > 0:
        # ★ rank_sum 큰 순(나쁜 순) 상위 k개
        df_sorted = df.sort_values("rank_sum", ascending=True).head(topk_overlay)
        for j, (_, r) in enumerate(df_sorted.iterrows(), 1):
            rid = r["row_id"]
            try:
                idx = all_row_id.index(rid)
            except ValueError:
                continue

            yt = all_true[idx];
            yp = all_pred[idx];
            mk = all_mask[idx].astype(bool)
            L = len(yt);
            nm_grid = _build_nm_grid(config, L)

            info = dict(all_info[idx]) if idx < len(all_info) else {"row_id": rid}
            for k in ("R2", "RMSE", "MAE", "SID", "SoftDTW",
                      "R2_rank", "RMSE_rank", "MAE_rank", "SID_rank", "SoftDTW_rank",
                      "rank_sum", "rank_sum_rank"):
                if k in r: info[k] = r[k]
            info["N"] = N

            rs = int(r["rank_sum"])
            out_png = os.path.join(out_dir, f"overlay_{split_name}_row_{rs}.png")

            _save_overlay_with_sidepanel(
                out_png, nm_grid, yt, yp, mk, info,
                f"{split_name} overlay #{j} | row_id={rid} | RankSum={rs} (#{int(r['rank_sum_rank'])}/{N})"
            )

    return csv_out


def _json_safe(obj):
    """config를 JSON으로 저장할 수 있게 안전 변환."""
    import numpy as _np
    from pathlib import Path as _Path
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, _Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, type):
        return obj.__name__
    try:
        return int(obj)
    except Exception:
        return str(obj)


@torch.no_grad()
def evaluate_train_and_test_per_sample(
    best_model_path: str,
    config: Dict,
    dataset_train_path: str,
    dataset_test_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    n_pairs: int = 50,
    out_dir: Optional[str] = None,
    topk_overlay: int = 8,
    save_all_overlays: bool = False,
    debug: bool = False,
) -> Tuple[str, str, str]:
    """
    베스트 모델을 로드하여 train/test 두 split 모두에 대해 per-sample 메트릭/시각화를 생성.
    Returns:
        (csv_train, csv_test, out_dir)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 모델 초기화 준비 (출력 크기 사전 추정) ---
    #   out_of_training=True 상태에서 output_size를 명시해야
    #   load_state_dict()가 output_layer 매개변수를 매칭할 수 있음.
    if config.get("target_type", "exp_spectrum") not in ("exp_spectrum", "ex_prob", "nm_distribution"):
        raise ValueError(f"Unsupported target_type: {config.get('target_type')}")

    # train CSV로 먼저 추정, 실패 시 test CSV로 추정
    try:
        out_size = _peek_output_size_for_model(config, dataset_train_path, batch_size, num_workers)
    except Exception:
        out_size = _peek_output_size_for_model(config, dataset_test_path, batch_size, num_workers)

    config_out = dict(config)
    config_out["out_of_training"] = True
    config_out["output_size"] = int(out_size)
    config_out["out_build_in_forward"] = False
    config_out.setdefault("out_final_activation", "softplus")

    # --- 모델 생성 & 가중치 로드 ---
    model = GraphormerModel(config_out, target_type=config_out.get("target_type", "exp_spectrum")).to(device)
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # --- 출력 폴더 ---
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if out_dir is None:
        out_dir = f"eval_outputs_{stamp}"
    os.makedirs(out_dir, exist_ok=True)

    # 설정 기록(재현성)
    with open(os.path.join(out_dir, "config_eval.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(config_out), f, ensure_ascii=False, indent=2)

    # --- split별 평가 ---
    csv_train = evaluate_split_per_sample(
        model, config_out, dataset_train_path,
        batch_size=batch_size, num_workers=num_workers, n_pairs=n_pairs,
        split_name="train", out_dir=out_dir,
        topk_overlay=topk_overlay, save_all_overlays=save_all_overlays, #debug=debug
    )
    csv_test = evaluate_split_per_sample(
        model, config_out, dataset_test_path,
        batch_size=batch_size, num_workers=num_workers, n_pairs=n_pairs,
        split_name="test", out_dir=out_dir,
        topk_overlay=topk_overlay, save_all_overlays=save_all_overlays, # debug=debug
    )

    print("[Saved]", csv_train)
    print("[Saved]", csv_test)
    print(f"[Saved] figures under: {out_dir}")
    return csv_train, csv_test, out_dir


# ----------------------------- CLI (optional) -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_model", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--config_json", required=True, help="학습에 사용한 config를 JSON으로 저장한 파일")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_pairs", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    evaluate_train_and_test_per_sample(
        best_model_path=args.best_model,
        config=cfg,
        dataset_train_path=args.train_csv,
        dataset_test_path=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_pairs=args.n_pairs,
        out_dir=args.out_dir,
        debug=args.debug,
    )
