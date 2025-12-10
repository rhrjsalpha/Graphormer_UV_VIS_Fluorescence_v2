# Transfer_learning/Trasfer_learning_v3_milestone.py
from __future__ import annotations
import os, json, hashlib
from datetime import datetime
from functools import partial
from typing import List, Dict, Any

import pandas as pd
from torch.utils.data import DataLoader
from rdkit import RDLogger
from numba import NumbaPerformanceWarning
import warnings

# ── quiet noisy logs ──────────────────────────────────────────────────────────
RDLogger.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ── milestone CV+final runner (metrics/figures milestones 분리 지원) ──────────
from GP.models_All.graphormer_CV_expSpectrum_GradNorm_manyloss_nonearlysotp_milestone import (
    run_cv_and_final_training_milestone,
)

from GP.data_prepare.Pre_Defined_Vocab_Generator import generate_graphormer_config
from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset, collate_fn


# ============================= User knobs =====================================

USE_TRANSFER = False  # True → PRETRAINED_* 사용, False → scratch config 생성

# (optional) pretrained 경로
PRETRAINED_CKPT = r"C:\path\to\runs\...\best_model.pth"
PRETRAINED_CFG  = r"C:\path\to\runs\...\config_eval.json"

# config 생성을 위한 데이터(여러 개 OK)
CONFIG_LIST: List[str] = [
    "../graphormer_data/QM_EM_ABS_stratified_config_base.csv",
    #"../graphormer_data/QM_EM_ABS_stratified_train_resplit.csv",
    #"../graphormer_data/QM_EM_ABS_stratified_test_resplit.csv",
]

# 학습/테스트 데이터
TRAIN_CSV = "../graphormer_data/EM_stratified_train_clustered_resplit_fillZero.csv"
TEST_CSV  = "../graphormer_data/EM_stratified_test_clustered_resplit_fillZero.csv"

# CV / 학습 파라미터
N_SPLITS    = 5
NUM_EPOCHS  = 20
BATCH_SIZE  = 50
N_PAIRS     = 50
ALPHA       = 0.12
NUM_WORKERS = 0

# ── Milestones ──
METRIC_MILESTONES = [5, 10, 15, 20]  # 성능 측정 + ckpt 저장
FIGURE_MILESTONES = [10, 20]                  # 스펙트럼 그림 저장 (분리)

DRAW_OVERLAYS_ON_MILESTONE = True  # 그림 저장 on/off
OVERLAY_TOPK   = 8                 # split별 저장 샘플 수
OVERLAY_SAVE_ALL = False           # True면 전 샘플 저장(용량↑)

# =============================================================================


def _make_config_slug_and_sig(cfg: Dict[str, Any]) -> tuple[str, str]:
    keys = [
        "mode","target_type","embedding_dim","num_encoder_layers","num_attention_heads",
        "activation_fn","dropout","attention_dropout","activation_dropout",
        "intensity_normalize","intensity_range","ex_normalize","prob_normalize",
        "multi_hop_max_dist","attn_bias_w","use_gradnorm","loss_function_full_spectrum",
    ]
    parts = [f"{k}-{cfg.get(k)}" for k in keys if k in cfg]
    slug = "_".join(parts)
    sig = hashlib.md5(slug.encode("utf-8")).hexdigest()[:8]
    return slug, sig


def _build_or_load_config() -> Dict[str, Any]:
    """Transfer 여부에 따라 config 로드/생성."""
    if USE_TRANSFER:
        with open(PRETRAINED_CFG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print("[INFO] Loaded pretrained config.")
    else:
        cfg = generate_graphormer_config(
            dataset_path_list=CONFIG_LIST,
            mode="cls_global_model",
            target_type="exp_spectrum",
            intensity_range=(200, 800),
            ex_normalize="ex_min_max",
            prob_normalize="prob_min_max",
            global_feature_order=["pH_label", "type", "Solvent"],
            global_multihot_cols={"Solvent": True},
        )
        print("[INFO] Generated scratch config.")

    # 호환성 패치
    if "float_feature_keys" not in cfg:
        cfg["float_feature_keys"] = cfg.get("ATOM_FLOAT_FEATURE_KEYS", [])
    if "output_size" not in cfg and "intensity_range" in cfg:
        s, e = cfg["intensity_range"]
        cfg["output_size"] = int(e - s) + 1

    # 실제 데이터로 feature 차원 확정(안전)
    probe_ds = UnifiedSMILESDataset(
        csv_file=CONFIG_LIST[0],
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
        target_type=cfg.get("target_type", "exp_spectrum"),
        attn_bias_w=cfg.get("attn_bias_w", 1.0),
        ex_normalize=cfg.get("ex_normalize"),
        prob_normalize=cfg.get("prob_normalize"),
        nm_dist_mode=cfg.get("nm_dist_mode", "hist"),
        nm_gauss_sigma=cfg.get("nm_gauss_sigma", 10.0),
        intensity_normalize=cfg.get("intensity_normalize", "min_max"),
        intensity_range=cfg.get("intensity_range", (200, 800)),
        x_cat_mode=cfg.get("x_cat_mode", "onehot"),
        global_cat_mode=cfg.get("global_cat_mode", "onehot"),
    )
    probe_loader = DataLoader(probe_ds, batch_size=1, shuffle=False,
                              collate_fn=partial(collate_fn, ds=probe_ds))
    batch0 = next(iter(probe_loader))
    cfg.update({
        "num_categorical_features": int(batch0["x_cat_onehot"].shape[-1]),
        "num_continuous_features": int(batch0["x_cont"].shape[-1]),
        "num_spatial": int(cfg.get("num_spatial", 512)),  # 유지
    })

    # 손실/GradNorm 기본값 보정
    lf_full = cfg.get("loss_function_full_spectrum")
    if isinstance(lf_full, str):
        lf_full = [lf_full]
    if not lf_full:
        lf_full = ["MAE", "SID"]
    cfg["loss_function_full_spectrum"] = lf_full
    cfg["use_gradnorm"] = bool(cfg.get("use_gradnorm", True))

    cfg.pop("out_of_training", None)
    return cfg


def main():
    os.makedirs("transfer_runs", exist_ok=True)

    cfg = _build_or_load_config()
    cfg_slug, cfg_sig = _make_config_slug_and_sig(cfg)

    # num_epochs ≥ 모든 마일스톤의 최댓값
    merged = sorted(set(METRIC_MILESTONES) | set(FIGURE_MILESTONES or METRIC_MILESTONES))
    if merged:
        assert NUM_EPOCHS >= merged[-1], \
            f"NUM_EPOCHS({NUM_EPOCHS}) must be >= max(milestones)({merged[-1]})"

    tstamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join("transfer_runs", f"{'transfer' if USE_TRANSFER else 'scratch'}_{tstamp}.csv")

    run_cv_and_final_training_milestone(
        config=cfg,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        n_splits=N_SPLITS,
        save_path=out_csv,
        loss_functions_full=list(cfg["loss_function_full_spectrum"]),
        num_epochs=NUM_EPOCHS,

        # ── milestones (분리) ──
        save_milestones=METRIC_MILESTONES,
        figure_milestones=FIGURE_MILESTONES,
        draw_overlays_on_milestone=DRAW_OVERLAYS_ON_MILESTONE,
        overlay_topk=OVERLAY_TOPK,
        overlay_save_all=OVERLAY_SAVE_ALL,

        batch_size=BATCH_SIZE,
        n_pairs=N_PAIRS,
        alpha=ALPHA,
        num_workers=NUM_WORKERS,
        debug=False,
        use_gradnorm=cfg["use_gradnorm"],
        cfg_slug=cfg_slug,
        cfg_sig=cfg_sig,
        init_from_ckpt=(PRETRAINED_CKPT if USE_TRANSFER else None),
        load_strict=None,
        ignore_prefixes=("output_layer",),
    )

    print(f"[DONE] Results saved → {out_csv}")


if __name__ == "__main__":
    main()

