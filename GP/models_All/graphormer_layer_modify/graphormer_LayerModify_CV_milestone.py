# graphormer_CV_expSpectrum_from_milestone_LayerModify.py
from __future__ import annotations

import os
from operator import truediv
from typing import Dict, List, Tuple, Any

import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path
import hashlib
import time

# ✅ 마일스톤 버전 트레이너 (내부에서 Graphormer_LayerModify를 사용하도록 수정되어 있어야 함)
from GP.models_All.graphormer_layer_modify.graphormer_LayerModify_train_milestone import (
    train_model_ex_porb,
)
from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset
from GP.data_prepare.Pre_Defined_Vocab_Generator import generate_graphormer_config
from GP.After_Training_Module.Caculate_Visualize_Each_Molecule_new import (
    evaluate_train_and_test_per_sample
)

# ---------------------------------------------------------------------
# 유틸: config에 대한 짧은 서명(slug, sig) 생성
# ---------------------------------------------------------------------
def _make_config_slug_and_sig(cfg: Dict[str, Any]) -> tuple[str, str]:
    keys = [
        # 모델/인퍼런스 공통
        "mode","target_type","embedding_dim","num_encoder_layers","num_attention_heads",
        "activation_fn","dropout","attention_dropout","activation_dropout",
        "intensity_normalize","intensity_range","ex_normalize","prob_normalize",
        "multi_hop_max_dist","attn_bias_w",
        # 옵티마이저 기본값
        "base_weight_decay",
        "base_learning_rate",
        # 출력 헤드 관련
        "output_size",
        "out_num_layers","out_hidden_dims","out_activation","out_final_activation",
        "out_bias","out_dropout","out_build_in_forward","out_freeze","out_init","out_const_value",
        "out_unfreeze_epoch","out_unfreeze_lr","out_unfreeze_weight_decay",
    ]
    parts = [f"{k}-{cfg.get(k)}" for k in keys if k in cfg]
    slug = "_".join(parts)
    sig = hashlib.md5(slug.encode("utf-8")).hexdigest()[:8]
    return slug, sig


# ---------------------------------------------------------------------
# K-Fold CV + 최종 학습 (마일스톤 저장/로깅 + 그림 마일스톤 분리)
#   - 새 training 코드에 맞춰 freeze→unfreeze를 config로 전달
# ---------------------------------------------------------------------
def run_cv_and_final_training_milestone(
    *,
    config: Dict[str, Any],
    train_csv: str,
    test_csv: str,
    n_splits: int = 5,
    save_path: str = "cv_results_exp_spectrum_milestone.csv",
    loss_functions_full: List[str] = ("SID", "MAE"),
    num_epochs: int = 420,
    save_milestones: List[int] = (200, 300, 400),
    figure_milestones: List[int] | None = None,
    draw_overlays_on_milestone: bool = False,
    overlay_topk: int = 8,
    overlay_save_all: bool = False,

    batch_size: int = 50,
    n_pairs: int = 50,
    alpha: float = 0.12,
    num_workers: int = 0,
    debug: bool = False,
    use_gradnorm: bool = True,
    cfg_slug: str = "",
    cfg_sig: str = "",
    init_from_ckpt: str | None = None,
    load_strict: bool | None = None,
    ignore_prefixes: tuple[str, ...] = ("output_layer",),

    # ▼▼ 새 학습 코드의 freeze→unfreeze 스케줄을 쉽게 쓰기 위한 외부 인자 ▼▼
    freeze_head_until_epoch: int = 0,           # 0이면 동결 사용 안 함
    base_learning_rate: float = 1e-4,
    base_weight_decay: float = 0.0,
    head_lr_after_unfreeze: float | None = None,
    head_wd_after_unfreeze: float | None = None,
) -> None:

    # --- 마일스톤 상한 검증
    _metric_max = max(save_milestones) if save_milestones else 0
    _figure_max = max(figure_milestones) if (draw_overlays_on_milestone and figure_milestones) else _metric_max
    assert num_epochs >= max(_metric_max, _figure_max), \
        f"num_epochs({num_epochs}) must be >= max milestone ({max(_metric_max, _figure_max)})"

    # config 사본에 학습 관련 키 반영
    cfg = dict(config)
    cfg["loss_function_full_spectrum"] = list(loss_functions_full)
    cfg["use_gradnorm"] = bool(use_gradnorm)
    cfg["base_weight_decay"] = float(base_weight_decay)
    cfg["base_learning_rate"] = float(base_learning_rate)

    # 안전 기본값 보정
    if "float_feature_keys" not in cfg:
        cfg["float_feature_keys"] = cfg.get("ATOM_FLOAT_FEATURE_KEYS", [])
    if "output_size" not in cfg and "intensity_range" in cfg:
        s, e = cfg["intensity_range"]
        cfg["output_size"] = int(e - s) + 1

    # ▼ freeze→unfreeze 스케줄을 config 키로 매핑 (새 training 코드가 인지)
    if freeze_head_until_epoch and freeze_head_until_epoch > 0:
        cfg["out_freeze"] = True
        cfg["out_unfreeze_epoch"] = int(freeze_head_until_epoch)
        if head_lr_after_unfreeze is not None:
            cfg["out_unfreeze_lr"] = float(head_lr_after_unfreeze)
        if head_wd_after_unfreeze is not None:
            cfg["out_unfreeze_weight_decay"] = float(head_wd_after_unfreeze)
    else:
        cfg["out_freeze"] = False
        cfg.pop("out_unfreeze_epoch", None)
        cfg.pop("out_unfreeze_lr", None)
        cfg.pop("out_unfreeze_weight_decay", None)

    # slug/sig (입력 안 주면 자동 생성)
    if not cfg_slug or not cfg_sig:
        _slug, _sig = _make_config_slug_and_sig(cfg)
        cfg_slug = cfg_slug or _slug
        cfg_sig  = cfg_sig  or _sig

    # ---- CV 준비
    full_df = pd.read_csv(train_csv)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_results: List[Dict[str, Any]] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(full_df)):
        print(f"\n=== Fold {fold+1}/{n_splits} (exp_spectrum · milestone) ===")
        df_tr = full_df.iloc[tr_idx].reset_index(drop=True)
        df_va = full_df.iloc[va_idx].reset_index(drop=True)

        # 임시 CSV
        tmp_tr = f"__tmp_train_fold{fold}.csv"
        tmp_va = f"__tmp_val_fold{fold}.csv"
        df_tr.to_csv(tmp_tr, index=False)
        df_va.to_csv(tmp_va, index=False)

        # Dataset
        ds_kwargs = dict(
            mol_col=cfg["mol_col"],
            nominal_feature_vocab=cfg["nominal_feature_vocab"],
            continuous_feature_names=cfg.get("continuous_feature_names", []),
            global_cat_dim=cfg.get("global_cat_dim", 0),
            global_cont_dim=cfg.get("global_cont_dim", 0),
            ATOM_FEATURES_VOCAB=cfg["ATOM_FEATURES_VOCAB"],
            float_feature_keys=cfg["float_feature_keys"],
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
        ds_tr = UnifiedSMILESDataset(csv_file=tmp_tr, **ds_kwargs)
        ds_va = UnifiedSMILESDataset(csv_file=tmp_va, **ds_kwargs)

        cv_ctx = {
            "is_cv": True,
            "fold": fold + 1,
            "n_splits": n_splits,
            "config_slug": cfg_slug,
            "config_sig": cfg_sig,
            "val_csv": tmp_va,        # 그림 저장 시 val csv 경로 전달
        }
        run_id = f"{cfg_sig}_cvF{fold+1:02d}"

        # ✅ 마일스톤 트레이너 호출
        res, best_path = train_model_ex_porb(
            config=dict(cfg),
            target_type="exp_spectrum",
            loss_function_full_spectrum=list(loss_functions_full),
            loss_function_ex=["MSE","MAE","SOFTDTW","SID"],
            loss_function_prob=["MSE","MAE","SOFTDTW","SID"],
            num_epochs=num_epochs, batch_size=batch_size, n_pairs=n_pairs, learning_rate=base_learning_rate,
            DATASET=ds_tr, TEST_VAL_DATASET=ds_va,
            alpha=alpha, is_cv=True,
            nominal_feature_vocab=cfg["nominal_feature_vocab"],
            continuous_feature_names=cfg.get("continuous_feature_names", []),
            patience=None, num_workers=num_workers, debug=debug,
            use_gradnorm=use_gradnorm,
            init_from_ckpt=init_from_ckpt, load_strict=load_strict,
            ignore_prefixes=ignore_prefixes,

            save_milestones=list(save_milestones),
            figure_milestones=(list(figure_milestones) if figure_milestones is not None else None),
            draw_overlays_on_milestone=draw_overlays_on_milestone,
            overlay_topk=overlay_topk,
            overlay_save_all=overlay_save_all,

            cv_context=cv_ctx, run_id=run_id, fold_idx=fold + 1,
        )

        res["fold"] = fold + 1
        res["best_model_path"] = best_path
        all_results.append(res)
        pd.DataFrame(all_results).to_csv("__intermediate_cv_results_exp_milestone.csv", index=False)

        # 임시 파일 정리
        for p in (tmp_tr, tmp_va):
            try: os.remove(p)
            except Exception: pass

    # ---------------- 최종 학습 (full train + test)
    print("\n=== Final Training on Full Train (milestone) ===")
    ds_kwargs_final = dict(
        mol_col=cfg["mol_col"],
        nominal_feature_vocab=cfg["nominal_feature_vocab"],
        continuous_feature_names=cfg.get("continuous_feature_names", []),
        global_cat_dim=cfg.get("global_cat_dim", 0),
        global_cont_dim=cfg.get("global_cont_dim", 0),
        ATOM_FEATURES_VOCAB=cfg["ATOM_FEATURES_VOCAB"],
        float_feature_keys=cfg["float_feature_keys"],
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
    ds_tr_final = UnifiedSMILESDataset(csv_file=train_csv, **ds_kwargs_final)
    ds_te_final = UnifiedSMILESDataset(csv_file=test_csv,  **ds_kwargs_final)

    final_ctx = {"is_cv": False, "config_slug": cfg_slug, "config_sig": cfg_sig}
    run_id_final = f"{cfg_sig}_final"

    res_final, best_path_final = train_model_ex_porb(
        config=dict(cfg),
        target_type="exp_spectrum",
        loss_function_full_spectrum=list(loss_functions_full),
        loss_function_ex=["MSE","MAE","SOFTDTW","SID"],
        loss_function_prob=["MSE","MAE","SOFTDTW","SID"],
        num_epochs=num_epochs, batch_size=batch_size, n_pairs=n_pairs, learning_rate=base_learning_rate,
        DATASET=ds_tr_final, TEST_VAL_DATASET=ds_te_final,
        alpha=alpha, is_cv=False,
        nominal_feature_vocab=cfg["nominal_feature_vocab"],
        continuous_feature_names=cfg.get("continuous_feature_names", []),
        patience=None, num_workers=num_workers, debug=debug,
        cv_context=final_ctx, use_gradnorm=use_gradnorm,
        init_from_ckpt=init_from_ckpt, load_strict=load_strict,
        ignore_prefixes=ignore_prefixes,

        save_milestones=list(save_milestones),
        figure_milestones=(list(figure_milestones) if figure_milestones is not None else None),
        draw_overlays_on_milestone=draw_overlays_on_milestone,
        overlay_topk=overlay_topk,
        overlay_save_all=overlay_save_all,

        run_id=run_id_final, fold_idx=-1,
    )

    res_final["fold"] = "FullTrain"
    res_final["best_model_path"] = best_path_final
    all_results.append(res_final)

    pd.DataFrame(all_results).to_csv(save_path, index=False)
    print(f"\n✅ Saved CV + Final results to: {save_path}")
    print(f"Best(full): {best_path_final}")

    # (선택) 최종 모델 샘플별 시각화
    final_dir = Path(res_final.get("exp_dir", Path(best_path_final).parent))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    eval_dir = final_dir / f"eval_{stamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    csv_train, csv_test, _ = evaluate_train_and_test_per_sample(
        best_model_path=best_path_final,
        config=cfg,
        dataset_train_path=train_csv,
        dataset_test_path=test_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        n_pairs=n_pairs,
        out_dir=str(eval_dir),
        save_all_overlays=True,
        topk_overlay=0
    )
    print(f"[EVAL SAVED] {eval_dir}")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    # ── (0) 경로 ───────────────────────────────────────────────────────────────
    train_csv = "../../../graphormer_data/EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"
    test_csv  = "../../../graphormer_data/EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv"

    TARGET_TYPE = "exp_spectrum"
    INT_RANGE   = (200, 800)
    EX_NORM     = "ex_min_max"
    PROB_NORM   = "prob_min_max"
    MODE        = "cls_only"

    NUM_EPOCHS  = 2500
    BATCH_SIZE  = 50
    N_PAIRS     = 50
    ALPHA       = 0.12
    NUM_WORKERS = 0
    N_SPLITS    = 5

    # ── (1) 전역 피처: pH=카테고리, dielectric_constant_avg=연속형 ───────────────
    gf_order      = ["dielectric_constant_avg","pH_label"] # "pH_label", "dielectric_constant_avg"
    gf_mh         = {}
    cont_override = ["dielectric_constant_avg"] # "dielectric_constant_avg"

    # ── (2) config 생성 ─────────────────────────────────────────────────────────
    base_cfg = generate_graphormer_config(
        dataset_path_list=[train_csv, test_csv],
        mode=MODE,
        target_type=TARGET_TYPE,
        intensity_range=INT_RANGE,
        ex_normalize=EX_NORM,
        prob_normalize=PROB_NORM,
        global_feature_order=gf_order,
        global_multihot_cols=gf_mh,
        continuous_feature_names=cont_override,
    )

    # 출력 길이/가드 및 안전 보정
    s, e = base_cfg["intensity_range"]
    base_cfg["output_size"] = int(e - s) + 1
    if "float_feature_keys" not in base_cfg:
        base_cfg["float_feature_keys"] = base_cfg.get("ATOM_FLOAT_FEATURE_KEYS", [])
    cont = set(base_cfg.get("continuous_feature_names", []))
    cont.add("dielectric_constant_avg")
    base_cfg["continuous_feature_names"] = sorted(cont)

    # ── (3) 1-batch probe로 실제 입력 차원 고정(선택/권장) ────────────────────────
    from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset, collate_fn as graph_collate_fn
    from torch.utils.data import DataLoader
    from functools import partial

    try:
        probe_ds = UnifiedSMILESDataset(
            csv_file=train_csv,
            mol_col=base_cfg["mol_col"],
            nominal_feature_vocab=base_cfg["nominal_feature_vocab"],
            continuous_feature_names=base_cfg.get("continuous_feature_names", []),
            global_cat_dim=base_cfg.get("global_cat_dim", 0),
            global_cont_dim=base_cfg.get("global_cont_dim", 0),
            ATOM_FEATURES_VOCAB=base_cfg["ATOM_FEATURES_VOCAB"],
            float_feature_keys=base_cfg.get("float_feature_keys", base_cfg.get("ATOM_FLOAT_FEATURE_KEYS", [])),
            BOND_FEATURES_VOCAB=base_cfg["BOND_FEATURES_VOCAB"],
            GLOBAL_FEATURE_VOCABS_dict=base_cfg.get("GLOBAL_FEATURE_VOCABS_dict"),
            mode=base_cfg["mode"],
            max_nodes=base_cfg.get("max_nodes", 128),
            multi_hop_max_dist=base_cfg.get("multi_hop_max_dist", 5),
            target_type=base_cfg.get("target_type", TARGET_TYPE),
            attn_bias_w=base_cfg.get("attn_bias_w", 1.0),
            ex_normalize=base_cfg.get("ex_normalize"),
            prob_normalize=base_cfg.get("prob_normalize"),
            nm_dist_mode=base_cfg.get("nm_dist_mode", "hist"),
            nm_gauss_sigma=base_cfg.get("nm_gauss_sigma", 10.0),
            intensity_normalize=base_cfg.get("intensity_normalize", "min_max"),
            intensity_range=base_cfg.get("intensity_range", (200, 800)),
            x_cat_mode=base_cfg.get("x_cat_mode", "onehot"),
            global_cat_mode=base_cfg.get("global_cat_mode", "onehot"),
            deg_clip_max=base_cfg.get("deg_clip_max", 5),
        )
        probe_loader = DataLoader(
            probe_ds, batch_size=1, shuffle=False,
            collate_fn=partial(graph_collate_fn, ds=probe_ds)
        )
        batch0 = next(iter(probe_loader))
        try: F_cat = batch0["x_cat_onehot"].shape[-1]
        except: F_cat = base_cfg.get("num_categorical_features", 0)
        try: F_cont = batch0["x_cont"].shape[-1]
        except: F_cont = base_cfg.get("num_continuous_features", 0)
        try: num_edges = batch0["attn_edge_type"].shape[-1]
        except: num_edges = base_cfg.get("num_edges", 14)

        base_cfg.update({
            "num_categorical_features": F_cat,
            "num_continuous_features":  F_cont,
            "num_in_degree": 6,
            "num_out_degree": 6,
            "num_edges": num_edges,
            "num_spatial": base_cfg.get("num_spatial", 512),
            "deg_clip_max": 6,
        })
        print(f"[Probe] F_cat={F_cat}  F_cont={F_cont}  edges={num_edges}")
    except Exception as e:
        print(f"[WARN] Probe 실패: {e}  (CSV 컬럼명/타입 확인)")

    # ── (4) 출력 헤드 설정 ─────────────────────────────────────────────────────
    base_cfg.update({
        "out_num_layers": 1,
        "out_hidden_dims": [],
        "out_activation": "relu",
        "out_final_activation": "softplus",   # 강도 예측이면 softplus 권장, shape은 sigmoid
        "out_bias": True,
        "out_dropout": 0.0,
        "out_build_in_forward": False,        # out_freeze=False일 때만 더미 fwd 선생성(학습 함수 내부)
        "out_freeze": True,                  # 필요 시 아래 FREEZE 옵션으로 덮어씀
        "out_init": "random",
        "out_const_value": 0.0,
    })

    # ── (5) 슬러그/시그 & 실행 ─────────────────────────────────────────────────
    cfg_slug, cfg_sig = _make_config_slug_and_sig(base_cfg)

    save_milestones   = []
    figure_milestones = [2500]
    draw_overlays_on_ms = False

    # ▼▼ 필요 시 freeze→unfreeze 스케줄을 간단히 지정 (없으면 동결 X) ▼▼
    FREEZE_UNTIL_EPOCH   = 2000      # 0이면 동결 사용 안 함
    HEAD_LR_AFTER_UNFREEZE = 5e-5 # 예: 5e-5, None, 1e-4
    HEAD_WD_AFTER_UNFREEZE = 1e-5 # 예: 0.0
    BASE_LR = 5e-5
    BASE_WD = 1e-5

    optimized_hyperparam_apply = True
    H = 768
    if optimized_hyperparam_apply == True:
        base_cfg.update({
            "embedding_dim": H,  # embedding_dim # Base : 768 # bayes_opt : 768
            "num_attention_heads": 32,  # heads BASE : 32 # bayes_opt : 8
            "num_encoder_layers": 8,  # layers BASE : 12 # bayes_opt : 10
            "multi_hop_max_dist": 4,  # hops BASE : 5 # bayes_opt : 3

            "dropout": 0.241,  # dropout # BASE : 0.1 # bayes_opt : 0
            "attention_dropout": 0.208,  # attn_drop BASE : 0.1 # bayes_opt : 0
            "activation_dropout": 0.088,  # act_drop BASE : 0.1 # # bayes_opt : 0.3
            "activation_fn": "gelu",  # act_fn (소문자 권장) BASE : gelu # bayes_opt : relu

            # FFN 배수: bo 코드와 동일 키 사용
            "ffn_multiplier": 4, #
            # 모델이 ffn_embedding_dim을 직접 요구할 수 있으니 함께 넣어줌
            "ffn_embedding_dim": 4 * H, #  Base : 1*H # # bayes_opt : 3*H

            # 로깅/슬러그용(선택) — 트레이너 인자에도 BASE_LR를 넘겨주세요
            "base_learning_rate": BASE_LR,
        })

    run_cv_and_final_training_milestone(
        config=base_cfg,
        train_csv=train_csv,
        test_csv=test_csv,
        n_splits=N_SPLITS,
        save_path="cv_results_exp_spectrum_milestone_epsOnly.csv",
        loss_functions_full=["SID", "MAE",],   # 필요시 "SOFTDTW" 추가 가능
        num_epochs=NUM_EPOCHS,
        save_milestones=save_milestones,
        figure_milestones=figure_milestones,
        draw_overlays_on_milestone=draw_overlays_on_ms,
        overlay_topk=8,
        overlay_save_all=True,
        batch_size=BATCH_SIZE, n_pairs=N_PAIRS,
        alpha=ALPHA, num_workers=NUM_WORKERS, debug=False,
        use_gradnorm=False,                                                      #주의하기
        cfg_slug=cfg_slug, cfg_sig=cfg_sig,
        init_from_ckpt=None, load_strict=None, ignore_prefixes=("output_layer",),

        # freeze→unfreeze 스케줄 (새 training 코드가 config 키로 인식)
        freeze_head_until_epoch=FREEZE_UNTIL_EPOCH,
        head_lr_after_unfreeze=HEAD_LR_AFTER_UNFREEZE,
        head_wd_after_unfreeze=HEAD_WD_AFTER_UNFREEZE,
        base_learning_rate=BASE_LR,
        base_weight_decay=BASE_WD,
    )

if __name__ == "__main__":
    main()