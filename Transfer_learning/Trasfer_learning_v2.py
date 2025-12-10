# graphormer_transfer_runner_v5.py
from __future__ import annotations
import os, json, hashlib
from datetime import datetime
import pandas as pd
from typing import List

from GP.models_All.graphomer_CV_expSpectrum_GradNorm_manyloss_nonearlystop_v3 import run_cv_and_final_training_v3
from GP.data_prepare.Pre_Defined_Vocab_Generator import generate_graphormer_config
from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset, collate_fn
from torch.utils.data import DataLoader
from functools import partial
# 버전 안맞을시 나오는 오류 프린트 안되게 #
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
from numba import NumbaPerformanceWarning
import warnings
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

# ==== 경로/입력 ====
USE_TRANSFER = False  # True → pretrained 불러오기, False → scratch config 생성

PRETRAINED_CKPT = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\Transfer_learning\runs\exp_spectrum_ed768-Lx-Hx_ir(200, 800)-exNone-pbNone_lfMAE-GN_ef5daeb5\final\best_model.pth"
PRETRAINED_CFG  = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\Transfer_learning\runs\exp_spectrum_ed768-Lx-Hx_ir(200, 800)-exNone-pbNone_lfMAE-GN_ef5daeb5\final\eval_20250830-180658\config_eval.json"

# Config 생성을 위해 사용할 여러 데이터셋
CONFIG_LIST: List[str] = [
    "../graphormer_data/QM_EM_ABS_stratified_config_base.csv",
    # "../graphormer_data/another_dataset.csv",   # 필요시 추가
]

# 학습/테스트는 단일 CSV만 사용
TRAIN_CSV = "../graphormer_data/QM_stratified_train_resplit.csv" # ABS_stratified_train_clustered_resplit.csv # QM_stratified_train_resplit.csv
TEST_CSV  = "../graphormer_data/QM_stratified_test_resplit.csv"  #  ABS_stratified_test_clustered_resplit.csv # QM_stratified_test_resplit.csv

# CV/학습 파라미터
N_SPLITS   = 3
NUM_EPOCHS = 3
BATCH_SIZE = 50
N_PAIRS    = 50
ALPHA      = 0.12
NUM_WORKERS= 0


# ---- 유틸 ----
def _make_config_slug_and_sig(cfg: dict) -> tuple[str, str]:
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


def main():
    os.makedirs("transfer_runs", exist_ok=True)

    if USE_TRANSFER:
        # ── pretrained config 불러오기 ──
        with open(PRETRAINED_CFG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(cfg)
        print("[INFO] Transfer Learning 모드")
    else:
        # ── scratch config 생성 ──
        cfg = generate_graphormer_config(
            dataset_path_list=CONFIG_LIST,
            mode="cls_only", # 나중에 바꾸기
            target_type="exp_spectrum",
            intensity_range=(200, 800),
            ex_normalize="ex_min_max",
            prob_normalize="prob_min_max",
            global_feature_order=["pH_label", "type", "Solvent"],
            global_multihot_cols={"Solvent": True},
        )
        print("cfg",cfg)
        ### num_spatial 는 기능적으로 영향을 주지 않음 ###
        print("[INFO] Scratch 모드 (CONFIG_LIST에서 vocab/config 생성)")

    probe_ds = UnifiedSMILESDataset(
        csv_file="../graphormer_data/QM_EM_ABS_stratified_config_base.csv",  # 한 파일이면 충분
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

    probe_loader = DataLoader(probe_ds, batch_size=1, shuffle=False, collate_fn=partial(collate_fn, ds=probe_ds))
    batch0 = next(iter(probe_loader))

    F_cat = batch0["x_cat_onehot"].shape[-1]
    F_cont = batch0["x_cont"].shape[-1]
    # num_in_deg = int(batch0["in_degree"].max().item()) + 1
    # num_out_deg = int(batch0["out_degree"].max().item()) + 1
    # edge 관련도 필요시 측정
    # num_edges = batch0["attn_edge_type"].shape[-1]
    num_spatial = cfg.get("num_spatial", 512)  # 기존 값 유지 or 측정 로직 있으면 교체

    # (3) base_cfg에 확정값 주입
    cfg.update({
        "num_categorical_features": F_cat,
        "num_continuous_features": F_cont,
        # "num_edges": num_edges,
        "num_spatial": num_spatial,
    })
    print(f"[PROBE] F_cat={F_cat}, F_cont={F_cont}")
    print("cfg update", cfg)
    print("config make end")

    cfg.pop("out_of_training", None)

    lf_full = cfg.get("loss_function_full_spectrum")
    if isinstance(lf_full, str):
        lf_full = [lf_full]
    if not lf_full:
        lf_full = ["MAE"]
    cfg["loss_function_full_spectrum"] = lf_full

    use_gn = bool(cfg.get("use_gradnorm", True))
    cfg["use_gradnorm"] = use_gn

    cfg_slug, cfg_sig = _make_config_slug_and_sig(cfg)
    tstamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_csv = os.path.join("transfer_runs", f"{'transfer' if USE_TRANSFER else 'scratch'}_{tstamp}.csv")

    run_cv_and_final_training_v3(
        config=cfg,
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        n_splits=N_SPLITS,
        save_path=save_csv,
        loss_functions_full=list(lf_full),
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        n_pairs=N_PAIRS,
        alpha=ALPHA,
        num_workers=NUM_WORKERS,
        debug=False,
        use_gradnorm=use_gn,
        cfg_slug=cfg_slug,
        cfg_sig=cfg_sig,
        init_from_ckpt=(PRETRAINED_CKPT if USE_TRANSFER else None),
        load_strict=None,
        ignore_prefixes=("output_layer",),
    )

    print(f"[DONE] Results saved → {save_csv}")


if __name__ == "__main__":
    main()
### cfg 미사용 되는 것 ###
#num_spatial : GraphAttnBias에 전달만 되고 로직에서 쓰이지 않음.
#num_atoms : GraphNodeFeature/GraphAttnBias에서 속성으로만 보관, 계산에는 미참여.