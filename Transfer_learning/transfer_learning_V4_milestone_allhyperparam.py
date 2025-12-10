# Transfer_learning/Trasfer_learning_v3_milestone.py
from __future__ import annotations
import hashlib
import os, re, json, shutil
from pathlib import Path
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
from GP.models_All.graphormer_layer_modify.graphormer_LayerModify_CV_milestone import (
    run_cv_and_final_training_milestone,
)
from GP.data_prepare.Pre_Defined_Vocab_Generator import generate_graphormer_config
from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset, collate_fn


# ============================= User knobs =====================================
# (A) Transfer vs Scratch
USE_TRANSFER = True  # True → PRETRAINED_* 사용, False → scratch config 생성
PRETRAINED_CKPT = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\Transfer_learning\runs\cls_globla_model_final\best_model.pth"   # USE_TRANSFER=True일 때만 사용
PRETRAINED_CFG  = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\Transfer_learning\runs\cls_globla_model_final\config_eval.json" # USE_TRANSFER=True일 때만 사용

# (B) Config 생성을 위한 데이터(여러 개 OK)
CONFIG_LIST: List[str] = [
    r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\ABS_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv",
    r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\ABS_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv",
    r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv",
    r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv"
]
# ABS_stratified_test_clustered_resplit_with_mu_eps_fillZero_first100
# (C) 학습/테스트 CSV
#TRAIN_CSV = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\ABS_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
#TEST_CSV  = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\ABS_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv"

TRAIN_CSV = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
TEST_CSV  = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv"


# (D) 데이터/타겟 설정
MODE          = "cls_global_model"   # cls_global_model / cls_only / cls_global_data ...
TARGET_TYPE   = "exp_spectrum"
INT_RANGE     = (200, 800)
EX_NORM       = "ex_min_max"
PROB_NORM     = "prob_min_max"
INT_NORM      = "min_max"            # intensity_normalize
GLOBAL_ORDER  = ["pH_label", "type",]
CONTINUOUS_GLOBAL_OVERRIDE = ["dielectric_constant_avg"]
GLOBAL_MH     = {"pH_label", "type",}    # multihot로 처리할 글로벌 카테고리
MAX_NODES     = 128
MULTI_HOP_MAX = 5
ATTN_BIAS_W   = 1.0

# (E) 모델 구조 하이퍼파라미터
H               = 768
EMBED_DIM       = H
NUM_HEADS       = 32
NUM_LAYERS      = 4
DROPOUT         = 0.1
ATTN_DROPOUT    = 0.1
ACT_DROPOUT     = 0.1
ACT_FN          = "gelu"             # "relu" / "gelu"
FFN_MULTIPLIER  = 1                  # ffn_embedding_dim = FFN_MULTIPLIER * H
FFN_EMBED_DIM   = FFN_MULTIPLIER * H # 옵션: 명시적으로 지정

# (F) 출력 헤드(아웃풋층) 설정
OUT_NUM_LAYERS       = 1
OUT_HIDDEN_DIMS      = []            # 예: [H] 또는 []
OUT_ACTIVATION       = "relu"
OUT_FINAL_ACTIVATION = "softplus"     # 강도 예측: softplus 권장 가능, shape는 sigmoid 많이 사용
OUT_BIAS             = True
OUT_DROPOUT          = 0.0
OUT_BUILD_IN_FORWARD = False         # 트레이너 내부 더미 fwd 여부(코드 호환용)
OUT_INIT             = "random"      # "random" / "const"
OUT_CONST_VALUE      = 0.0

# (G) 학습/최적화 하이퍼파라미터
N_SPLITS        = 5
NUM_EPOCHS      = 2500 # 2500
BATCH_SIZE      = 50
N_PAIRS         = 50
BASE_LR         = 1e-4
BASE_WD         = 1e-5
USE_GRADNORM    = False
ALPHA_GRADNORM  = 0.12               # (트레이너가 지원하면 사용)
NUM_WORKERS     = 0
DEBUG_MODE      = False

# (H) Freeze → Unfreeze 스케줄
FREEZE_UNTIL_EPOCH   = 2000          # 0이면 동결 사용 안 함 2000
HEAD_LR_AFTER        = 1e-4
HEAD_WD_AFTER        = 1e-5

# (I) 손실 묶음
LOSS_FUNCTIONS_FULL = ["SID", "MAE"] # 필요시 "SOFTDTW" 추가 가능
LOSS_FN_EX          = ["MSE", "MAE", "SOFTDTW", "SID"]
LOSS_FN_PROB        = ["MSE", "MAE", "SOFTDTW", "SID"]

# (J) 마일스톤(메트릭/그림 분리)
METRIC_MILESTONES = []  # 성능 측정 + ckpt 저장
FIGURE_MILESTONES = []                 # 스펙트럼 그림 저장
DRAW_OVERLAYS_ON_MILESTONE = False
OVERLAY_TOPK       = 8
OVERLAY_SAVE_ALL   = True

# (K) 기타
WORK_DIR = "transfer_runs"
# ============================================================================


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
    """
    CV 코드와 동일한 순서로 cfg를 구성:
    1) generate_graphormer_config로 베이스 생성 (TRAIN/TEST를 함께 스캔)
    2) 안전 보정(출력길이, float_feature_keys, 연속형 전역 피처 합집합 등)
    3) 1-배치 프로빙으로 실제 입력 차원 확정
    4) 출력 헤드/모델/학습 관련 user-knobs 값으로 일괄 덮어쓰기
    """
    # 1) Scratch or Transfer base config
    if USE_TRANSFER:
        with open(PRETRAINED_CFG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print("[INFO] Loaded pretrained config.")
        # Transfer라도 아래 단계(2~4)는 동일하게 적용해 최신 파이프라인으로 정규화
    else:
        cfg = generate_graphormer_config(
            dataset_path_list=CONFIG_LIST,     # << 동일: 학습/테스트 동시 스캔
            mode=MODE,
            target_type=TARGET_TYPE,
            intensity_range=INT_RANGE,
            ex_normalize=EX_NORM,
            prob_normalize=PROB_NORM,
            global_feature_order=GLOBAL_ORDER,          # 명목형 글로벌 피처
            global_multihot_cols=GLOBAL_MH,             # 멀티핫 지정
            # CV 코드처럼 연속형 글로벌 피처를 외부에서 강제 지정하고 싶다면:
            continuous_feature_names=CONTINUOUS_GLOBAL_OVERRIDE,
        )
        print("[INFO] Generated scratch config.")

    # 2) 안전 보정/추가
    if "float_feature_keys" not in cfg:
        cfg["float_feature_keys"] = cfg.get("ATOM_FLOAT_FEATURE_KEYS", [])
    if "output_size" not in cfg and "intensity_range" in cfg:
        s, e = cfg["intensity_range"]
        cfg["output_size"] = int(e - s) + 1

    # 연속형 글로벌 피처(override가 있다면 합집합으로 보강)
    if CONTINUOUS_GLOBAL_OVERRIDE:
        cont = set(cfg.get("continuous_feature_names", []))
        cont.update(CONTINUOUS_GLOBAL_OVERRIDE)
        cfg["continuous_feature_names"] = sorted(cont)

    # 3) 1-배치 프로빙으로 실제 입력 차원/엣지 차원 확정 (TRAIN_CSV 사용)
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
        max_nodes=cfg.get("max_nodes", MAX_NODES),
        multi_hop_max_dist=cfg.get("multi_hop_max_dist", MULTI_HOP_MAX),
        target_type=cfg.get("target_type", TARGET_TYPE),
        attn_bias_w=cfg.get("attn_bias_w", ATTN_BIAS_W),
        ex_normalize=cfg.get("ex_normalize", EX_NORM),
        prob_normalize=cfg.get("prob_normalize", PROB_NORM),
        nm_dist_mode=cfg.get("nm_dist_mode", "hist"),
        nm_gauss_sigma=cfg.get("nm_gauss_sigma", 10.0),
        intensity_normalize=cfg.get("intensity_normalize", INT_NORM),
        intensity_range=cfg.get("intensity_range", INT_RANGE),
        x_cat_mode=cfg.get("x_cat_mode", "onehot"),
        global_cat_mode=cfg.get("global_cat_mode", "onehot"),
    )
    probe_loader = DataLoader(
        probe_ds, batch_size=1, shuffle=False,
        collate_fn=partial(collate_fn, ds=probe_ds)
    )
    batch0 = next(iter(probe_loader))

    # 안전 추출(키 유무에 따라 fallback)
    try:
        F_cat = int(batch0["x_cat_onehot"].shape[-1])
    except Exception:
        F_cat = int(cfg.get("num_categorical_features", 0))
    try:
        F_cont = int(batch0["x_cont"].shape[-1])
    except Exception:
        F_cont = int(cfg.get("num_continuous_features", 0))
    try:
        num_edges = int(batch0["attn_edge_type"].shape[-1])
    except Exception:
        num_edges = int(cfg.get("num_edges", 14))

    cfg.update({
        "num_categorical_features": F_cat,
        "num_continuous_features":  F_cont,
        "num_edges": num_edges,
        "num_in_degree": 6,
        "num_out_degree": 6,
        "num_spatial": int(cfg.get("num_spatial", 512)),
        "deg_clip_max": 6,
    })

    # 4) (핵심) 위쪽 user-knobs로 일괄 덮어쓰기 (CV 코드와 동일)
    cfg.update({
        # 데이터/타겟
        "mode": MODE,
        "target_type": TARGET_TYPE,
        "intensity_range": INT_RANGE,
        "ex_normalize": EX_NORM,
        "prob_normalize": PROB_NORM,
        "intensity_normalize": INT_NORM,
        "multi_hop_max_dist": MULTI_HOP_MAX,
        "attn_bias_w": ATTN_BIAS_W,
        "max_nodes": MAX_NODES,

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

        # 손실/GradNorm (cfg에도 기록)
        "loss_function_full_spectrum": list(LOSS_FUNCTIONS_FULL),
        "use_gradnorm": bool(USE_GRADNORM),
        "alpha_gradnorm": ALPHA_GRADNORM,

        # 로깅용(실제 옵티마이저는 run 인자)
        "base_learning_rate": BASE_LR,
        "base_weight_decay": BASE_WD,
    })

    # 구형 키 정리
    cfg.pop("out_of_training", None)
    return cfg

def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    cfg = _build_or_load_config()
    cfg_slug, cfg_sig = _make_config_slug_and_sig(cfg)

    # num_epochs ≥ 모든 마일스톤의 최댓값
    merged = sorted(set(METRIC_MILESTONES) | set(FIGURE_MILESTONES or METRIC_MILESTONES))
    if merged:
        assert NUM_EPOCHS >= merged[-1], \
            f"NUM_EPOCHS({NUM_EPOCHS}) must be >= max(milestones)({merged[-1]})"

    tstamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = os.path.join(
        WORK_DIR,
        f"{'transfer' if USE_TRANSFER else 'scratch'}_{cfg_sig}_{tstamp}.csv"
    )

    # ── freeze→unfreeze는 run 인자로 넘기면, 내부에서 cfg 키(out_freeze/out_unfreeze_*)로 매핑됨 ──
    run_cv_and_final_training_milestone(
        config=cfg,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        n_splits=N_SPLITS,
        save_path=out_csv,
        loss_functions_full=list(LOSS_FUNCTIONS_FULL),

        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        n_pairs=N_PAIRS,
        alpha=ALPHA_GRADNORM,         # 트레이너가 gradnorm alpha로 쓰면 그대로, 아니면 무시됨
        num_workers=NUM_WORKERS,
        debug=DEBUG_MODE,
        use_gradnorm=USE_GRADNORM,

        # 학습률/WD(실제 옵티마이저 적용)
        base_learning_rate=BASE_LR,
        base_weight_decay=BASE_WD,

        # freeze→unfreeze
        freeze_head_until_epoch=FREEZE_UNTIL_EPOCH,
        head_lr_after_unfreeze=HEAD_LR_AFTER,
        head_wd_after_unfreeze=HEAD_WD_AFTER,

        # milestones(분리)
        save_milestones=METRIC_MILESTONES,
        figure_milestones=FIGURE_MILESTONES,
        draw_overlays_on_milestone=DRAW_OVERLAYS_ON_MILESTONE,
        overlay_topk=OVERLAY_TOPK,
        overlay_save_all=OVERLAY_SAVE_ALL,

        # TL
        cfg_slug=cfg_slug,
        cfg_sig=cfg_sig,
        init_from_ckpt=(PRETRAINED_CKPT if USE_TRANSFER else None),
        load_strict=None,
        ignore_prefixes=("output_layer",),
    )



if __name__ == "__main__":
    main()
