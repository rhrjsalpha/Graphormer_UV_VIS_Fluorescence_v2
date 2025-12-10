# graphormer_transfer_runner_v3.py
from __future__ import annotations
import os, json
from datetime import datetime
import hashlib
from GP.models_All.graphomer_CV_expSpectrum_GradNorm_manyloss_nonearlystop_v3 import run_cv_and_final_training_v3


# ==== 경로/입력 ====
PRETRAINED_CKPT = r"C:\Users\kogun\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\GridSearch\runs\exp_spectrum_ed768-Lx-Hx_ir(200, 800)-exNone-pbNone_lfMAE-GN_ef5daeb5\final\best_model.pth"    # 기존 학습된 모델
PRETRAINED_CFG  = r"C:\Users\kogun\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\GridSearch\runs\exp_spectrum_ed768-Lx-Hx_ir(200, 800)-exNone-pbNone_lfMAE-GN_ef5daeb5\final\eval_20250826-233803\config_eval.json"  # 기존 config JSON
NEW_TRAIN_CSV   = r"../graphormer_data/final_split/ABS_stratified_train_plus.csv"
NEW_TEST_CSV    = r"../graphormer_data/final_split/ABS_stratified_test_plus.csv"
# CV/학습 파라미터(필요 시만 수정)
N_SPLITS   = 5
NUM_EPOCHS = 3
BATCH_SIZE = 50
N_PAIRS    = 50
ALPHA      = 0.12
NUM_WORKERS= 0

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
    with open(PRETRAINED_CFG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ── 평가용 플래그는 제거하고, 손실/GradNorm 설정을 우선 주입 ──
    cfg.pop("out_of_training", None)

    lf_full = cfg.get("loss_function_full_spectrum")
    if isinstance(lf_full, str):
        lf_full = [lf_full]
    if not lf_full:
        lf_full = ["MAE"]  # 안전 기본값
    cfg["loss_function_full_spectrum"] = lf_full

    use_gn = bool(cfg.get("use_gradnorm", True))
    cfg["use_gradnorm"] = use_gn

    # 손실/GradNorm까지 포함해 slug 계산
    cfg_slug, cfg_sig = _make_config_slug_and_sig(cfg)

    tstamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_csv = os.path.join("transfer_runs", f"transfer_cv_{tstamp}.csv")

    print(f"[Transfer] using losses={lf_full}, use_gradnorm={use_gn}")

    # CV + Full Train (Transfer)
    run_cv_and_final_training_v3(
        config=cfg,                     # ← cfg 자체에 손실/GradNorm 반영됨
        train_csv=NEW_TRAIN_CSV,
        test_csv=NEW_TEST_CSV,
        n_splits=N_SPLITS,
        save_path=save_csv,
        loss_functions_full=list(lf_full),  # ← config의 손실 사용
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        n_pairs=N_PAIRS,
        alpha=ALPHA,
        num_workers=NUM_WORKERS,
        debug=False,
        use_gradnorm=use_gn,                # ← config의 값 우선
        cfg_slug=cfg_slug,
        cfg_sig=cfg_sig,

        # 사전학습 로드
        init_from_ckpt=PRETRAINED_CKPT,
        load_strict=None,                    # 먼저 strict, 실패 시 출력층 무시
        ignore_prefixes=("output_layer",),
    )

    print(f"[DONE] Transfer CV+Final saved to: {save_csv}")

if __name__ == "__main__":
    main()
