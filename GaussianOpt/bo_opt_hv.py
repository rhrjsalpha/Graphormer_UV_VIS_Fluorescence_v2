# bo_opt_hv.py — all objectives use -ratio vs 0-pred baseline; R2 is logged only
from __future__ import annotations

import csv, os, time, math
import numpy as np
import pandas as pd
import torch

from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
try:
    from botorch.acquisition.multi_objective.monte_carlo import qLogNoisyExpectedHypervolumeImprovement as qNEHVI
except ImportError:
    from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement as qNEHVI
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.models.transforms import Normalize, Standardize
from botorch.utils.multi_objective.pareto import is_non_dominated

from sklearn.model_selection import KFold
from GP.data_prepare.Pre_Defined_Vocab_Generator import generate_graphormer_config
from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset
from GP.models_All.graphormer_layer_modify.graphormer_LayerModify_train_milestone import train_model_ex_porb
from GP.data_prepare.DataLoader_QMData_All import collate_fn
from GP.Custom_Loss.soft_dtw_cuda import SoftDTW

from torch.utils.data import DataLoader
from functools import partial
# ====== 데이터 경로 ======
TRAIN_CSV = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_train_clustered_resplit_with_mu_eps.csv"
TEST_CSV  = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_test_clustered_resplit_with_mu_eps.csv"

# ───────── 고정 옵션 ─────────
FIX_CV_FOLDS = 5
FIX_NUM_EPOCHS = 500
FIX_MODE = "cls_global_data"
FIX_GLOBAL_FEATURE = (["dielectric_constant_avg", "pH_label"], {}, "GF=eps+pH", ["dielectric_constant_avg"])
FIX_USE_GRADNORM = False
FIX_OUT_HEAD = dict(tag="head=1xSoftplus", out_num_layers=1, out_hidden_dims=[], out_activation="relu",
                    out_final_activation="softplus", out_bias=True, out_dropout=0.0,
                    out_build_in_forward=False, out_freeze=True, out_init="random", out_const_value=0.0)
FIX_LOSS_COMBO = ["MAE", "SID"]
FIX_BATCH_SIZE = 50
FIX_N_PAIRS    = 50
FIX_ALPHA      = 0.12
FIX_NUM_WORKERS= 0
FIX_DEBUG      = False

# ───────── 탐색 공간 ─────────
CHOICES_EMB   = [128, 256,512,768] # 512,768
CHOICES_HEADS = [4, 8, 16, 32] #
CHOICES_LAYERS= [2, 4, 6, 8, 10, 12] # 12
CHOICES_HOPS  = [2, 3, 4, 5, 6] # 5
FFN_MUL       = [2, 3, 4,]

CHOICES_LR     = [1e-4, 5e-5, 3e-5, 1e-5]
CHOICES_ACTFN  = ["relu"]


BOUNDS = torch.tensor([
    [0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0],
    [len(CHOICES_EMB)-1, len(CHOICES_HEADS)-1, len(FFN_MUL)-1,
     len(CHOICES_LAYERS)-1, len(CHOICES_HOPS)-1, 0.3, 0.3, 0.3,
     len(CHOICES_LR)-1, len(CHOICES_ACTFN)-1]
], dtype=torch.double)

N_INIT, N_ITER, BATCH_Q = 10, 30, 1
D = int(BOUNDS.shape[1])

# ───────── 유틸 ─────────
def decode(x: torch.Tensor) -> dict:
    x = x.view(-1)
    ei, hi, fi, li, ho = [int(round(v.item())) for v in x[:5]]
    lr_i  = int(round(x[8].item()))
    act_i = int(round(x[9].item()))
    return dict(
        embedding_dim=CHOICES_EMB[ei],
        num_attention_heads=CHOICES_HEADS[hi],
        ffn_multiplier=FFN_MUL[fi],
        num_encoder_layers=CHOICES_LAYERS[li],
        multi_hop_max_dist=CHOICES_HOPS[ho],
        dropout=float(x[5].item()),
        attention_dropout=float(x[6].item()),
        activation_dropout=float(x[7].item()),
        learning_rate=CHOICES_LR[lr_i],
        activation_fn=CHOICES_ACTFN[act_i],
    )

def is_feasible(cfg: dict) -> bool:
    return (cfg["embedding_dim"] % cfg["num_attention_heads"]) == 0

def _safe_pick(resdict: dict, base: str, split: str) -> float:
    def first(*cands):
        for k in cands:
            if k in resdict and resdict[k] is not None: return float(resdict[k])
        for k,v in resdict.items():
            if k.startswith(base + "_") and v is not None: return float(v)
        return 0.0
    return first(f"{base}_avg_val", f"{base}_avg_cv", f"{base}_avg_test", base) if split=="val" else \
           first(f"{base}_avg_test", f"{base}_avg_eval", f"{base}_avg_val", base) if split=="test" else \
           first(f"{base}_avg_train", base)
def compute_ref_point(Y: torch.Tensor, margin: float = 1e-3) -> torch.Tensor:
    mins = Y.min(dim=0).values
    maxs = Y.max(dim=0).values
    slack = 0.05 * (maxs - mins + 1e-6)  # 범위의 5%만큼 더 내려준다
    return (mins - slack - margin).to(dtype=torch.double, device=Y.device)
def compute_zero_baselines_masked_from_loader(val_ds, *, device: torch.device, gamma: float = 0.2, eps: float = 1e-12):
    """
    학습 코드와 동일하게, 배치의 masks로 유효 파장만 사용해 0-예측 베이스라인을 계산.
    SoftDTW는 train에서 쓰던 구현(SoftDTW, normalize=True, gamma=0.2) 그대로 사용.
    """
    from torch.utils.data import DataLoader
    from functools import partial
    # collate_fn, SoftDTW는 이미 상단에서 import 되어 있음

    loader = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        collate_fn=partial(collate_fn, ds=val_ds), num_workers=0
    )
    sdtw = SoftDTW(use_cuda=(device.type == "cuda"), gamma=float(gamma), bandwidth=None, normalize=True)

    mae_rows, rmse_rows, sdtw_rows, sid_rows = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            y = batch["targets"]
            m = batch["masks"]

            # 텐서 → 넘파이
            if isinstance(y, torch.Tensor): y = y.detach().cpu().numpy()
            if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()

            # (B, L, 1) 형태면 마지막 축 제거
            if y.ndim == 3 and y.shape[-1] == 1: y = y[..., 0]
            if m.ndim == 3 and m.shape[-1] == 1: m = m[..., 0]

            # bool로 통일
            m = m.astype(bool)

            B = y.shape[0]
            for i in range(B):
                valid = m[i]
                if not np.any(valid):
                    # 완전 결측이면 안전값
                    mae_rows.append(1.0); rmse_rows.append(1.0); sdtw_rows.append(1.0); sid_rows.append(1.0)
                    continue

                yi = y[i, valid]
                yi = np.nan_to_num(yi, nan=0.0, posinf=0.0, neginf=0.0)
                yi = np.clip(yi, 0.0, None)

                # MAE0 / RMSE0 (per-sample 평균, 이후 배치 평균)
                mae_rows.append(float(np.mean(np.abs(yi))))
                rmse_rows.append(float(np.sqrt(np.mean(yi**2))))

                # SoftDTW0: yi vs zeros, (1, T, 1) 형태로
                yi_t = torch.from_numpy(yi.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(-1)  # [1,T,1]
                zi_t = torch.zeros_like(yi_t)
                v = sdtw(yi_t, zi_t)
                sdtw_rows.append(float(v.item()))

                # SID0 (uniform과의 대칭KL)
                s = float(yi.sum())
                if s <= eps:
                    sid_rows.append(0.0)
                else:
                    P = (yi / s).astype(np.float64)
                    L = P.shape[0]
                    U = np.full(L, 1.0 / L, dtype=np.float64)
                    P = np.clip(P, eps, 1.0); U = np.clip(U, eps, 1.0)
                    kl_pu = float(np.sum(P * (np.log(P) - np.log(U))))
                    kl_up = float(np.sum(U * (np.log(U) - np.log(P))))
                    sid_rows.append(kl_pu + kl_up)

    def _avg(x, default=1.0):
        v = float(np.mean(x)) if len(x) else default
        return v if np.isfinite(v) and v > eps else default

    return dict(
        MAE0=_avg(mae_rows),
        RMSE0=_avg(rmse_rows),
        SoftDTW0=_avg(sdtw_rows),
        SID0=_avg(sid_rows),
    )

# ───────── SoftDTW(간단 구현) ─────────
def _sdtw0_with_training_softdtw(X_np: np.ndarray, gamma: float, device: torch.device) -> float:
    """
    학습에서 쓰는 SoftDTW( :contentReference[oaicite:1]{index=1} )로 y vs 0 의 평균 거리를 계산.
    """
    # 1) NaN/Inf 정리 + 음수 클리핑(학습 파이프라인과 일관)
    X = np.nan_to_num(np.asarray(X_np, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X[X < 0] = 0.0
    if X.size == 0:
        return 1.0
    B, T = X.shape

    # 2) torch 텐서로 변환 (B,T,1), zeros 동일 크기
    x = torch.from_numpy(X).to(device=device, dtype=torch.float32).unsqueeze(-1)  # [B,T,1]
    z = torch.zeros_like(x)                                                      # [B,T,1]

    # 3) 학습과 동일한 설정의 SoftDTW (gamma, normalize 등 유지)  :contentReference[oaicite:2]{index=2}
    sdtw = SoftDTW(use_cuda=(device.type == "cuda"), gamma=float(gamma), bandwidth=None, normalize=True)

    with torch.no_grad():
        v = sdtw(x, z)      # 보통 [B] 또는 scalar
        v = v.mean() if v.ndim > 0 else v
        val = float(v.item())
    return val if np.isfinite(val) else 1.0


def compute_zero_baselines(csv_path: str, start_nm: int, end_nm: int, *,
                           sdtw_gamma: float = 0.2,  # ← 학습과 동일 gamma 기본값
                           eps: float = 1e-12,
                           device: torch.device = torch.device("cpu")):
    want = [str(i) for i in range(start_nm, end_nm+1)]
    header = pd.read_csv(csv_path, nrows=0)
    cols = [c for c in want if c in header.columns]
    if not cols:
        return dict(MAE0=1.0, RMSE0=1.0, SoftDTW0=1.0)

    X = pd.read_csv(csv_path, usecols=cols).to_numpy(dtype=np.float64)
    if X.size == 0:
        return dict(MAE0=1.0, RMSE0=1.0, SoftDTW0=1.0)

    # NaN/Inf → 0, 음수 클리핑
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[X < 0] = 0.0

    # MAE0, RMSE0 (항상 유한값)
    mae0  = float(np.mean(np.mean(np.abs(X), axis=1)))
    rmse0 = float(np.mean(np.sqrt(np.mean(X**2, axis=1))))

    # SoftDTW0: 기존 SoftDTW로 y vs zeros
    sdtw0 = _sdtw0_with_training_softdtw(X, gamma=sdtw_gamma, device=device)

    # 안전 보정
    def _good(v, default=1.0):
        return float(v) if np.isfinite(v) and v > eps else float(default)

    return dict(MAE0=_good(mae0), RMSE0=_good(rmse0), SoftDTW0=_good(sdtw0))

# ───────── SID uniform baseline ─────────
def compute_sid_uniform_baseline(csv_path: str, start_nm: int, end_nm: int, eps: float = 1e-12) -> float:
    want = [str(i) for i in range(start_nm, end_nm+1)]
    head = pd.read_csv(csv_path, nrows=0)
    cols = [c for c in want if c in head.columns]
    if not cols: return 1.0
    X = pd.read_csv(csv_path, usecols=cols).to_numpy(dtype=np.float64)
    if X.size == 0: return 1.0
    X[X<0] = 0.0; X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sums = X.sum(axis=1, keepdims=True); sums[sums<eps] = 1.0
    P = X / sums
    L = P.shape[1]
    U = np.full((1,L), 1.0/L, dtype=np.float64)
    P = np.clip(P, eps, 1.0); U = np.clip(U, eps, 1.0)
    kl_pu = (P * (np.log(P) - np.log(U))).sum(axis=1)
    kl_up = (U * (np.log(U) - np.log(P))).sum(axis=1)
    return float(np.mean(kl_pu + kl_up))

# ───────── CV 실행 ─────────
def run_cv_and_get_metrics(cfg: dict, *, device: torch.device) -> dict:
    gf_order, gf_mh, gf_tag, cont_override = FIX_GLOBAL_FEATURE
    base = generate_graphormer_config(
        dataset_path_list=[TRAIN_CSV, TEST_CSV], mode=FIX_MODE,
        target_type="exp_spectrum", intensity_range=(200, 800),
        ex_normalize="ex_min_max", prob_normalize="prob_min_max",
        global_feature_order=gf_order, global_multihot_cols=gf_mh,
        continuous_feature_names=cont_override,
    )
    H = int(cfg["embedding_dim"])
    base.update(dict(embedding_dim=H, ffn_embedding_dim=int(cfg["ffn_multiplier"]*H),
                     num_attention_heads=int(cfg["num_attention_heads"]),
                     num_encoder_layers=int(cfg["num_encoder_layers"]),
                     multi_hop_max_dist=int(cfg["multi_hop_max_dist"]),
                     num_edge_dis=int(cfg["multi_hop_max_dist"]),
                     num_spatial=int(cfg["multi_hop_max_dist"]),
                     dropout=float(cfg["dropout"]), attention_dropout=float(cfg["attention_dropout"]),
                     activation_dropout=float(cfg["activation_dropout"]),
                     activation_fn=str(cfg["activation_fn"]),
                     mode=FIX_MODE, target_type="exp_spectrum",
                     out_num_layers=FIX_OUT_HEAD["out_num_layers"],
                     out_hidden_dims=list(FIX_OUT_HEAD["out_hidden_dims"]),
                     out_activation=FIX_OUT_HEAD["out_activation"],
                     out_final_activation=FIX_OUT_HEAD["out_final_activation"],
                     out_bias=FIX_OUT_HEAD["out_bias"], out_dropout=FIX_OUT_HEAD["out_dropout"],
                     out_build_in_forward=FIX_OUT_HEAD["out_build_in_forward"],
                     out_freeze=FIX_OUT_HEAD["out_freeze"], out_init=FIX_OUT_HEAD["out_init"],
                     out_const_value=FIX_OUT_HEAD["out_const_value"]))
    s,e = base["intensity_range"]; base["output_size"] = int(e-s)+1
    if "float_feature_keys" not in base:
        base["float_feature_keys"] = base.get("ATOM_FLOAT_FEATURE_KEYS", [])

    full_df = pd.read_csv(TRAIN_CSV)
    kf = KFold(n_splits=FIX_CV_FOLDS, shuffle=True, random_state=42)

    agg = {
        "MAE": [], "RMSE": [], "SoftDTW": [], "SID": [], "R2": [],
        "MAE0": [], "RMSE0": [], "SoftDTW0": [], "SID0": [],
        "MAE_ratio": [], "RMSE_ratio": [], "SoftDTW_ratio": [], "SID_ratio": [],
    }

    for fold,(tr_idx,va_idx) in enumerate(kf.split(full_df)):
        print(f"[CV {fold+1}/{FIX_CV_FOLDS}] start "
              f"(emb={base['embedding_dim']}, heads={base['num_attention_heads']}, "
              f"ffn×={cfg['ffn_multiplier']}, L={cfg['num_encoder_layers']}, "
              f"hop={cfg['multi_hop_max_dist']}, p={cfg['dropout']:.2f})")
        t0 = time.perf_counter()

        df_tr = full_df.iloc[tr_idx].reset_index(drop=True)
        df_va = full_df.iloc[va_idx].reset_index(drop=True)

        tmp_tr = f"__tmp_bo_train_fold{fold}.csv";
        df_tr.to_csv(tmp_tr, index=False)
        tmp_va = f"__tmp_bo_val_fold{fold}.csv";
        df_va.to_csv(tmp_va, index=False)

        # ① ds_kwargs를 먼저 정의
        ds_kwargs = dict(
            mol_col=base["mol_col"],
            nominal_feature_vocab=base["nominal_feature_vocab"],
            continuous_feature_names=base.get("continuous_feature_names", []),
            global_cat_dim=base.get("global_cat_dim", 0),
            global_cont_dim=base.get("global_cont_dim", 0),
            ATOM_FEATURES_VOCAB=base["ATOM_FEATURES_VOCAB"],
            float_feature_keys=base["float_feature_keys"],
            BOND_FEATURES_VOCAB=base["BOND_FEATURES_VOCAB"],
            GLOBAL_FEATURE_VOCABS_dict=base.get("GLOBAL_FEATURE_VOCABS_dict"),
            mode=base["mode"],
            max_nodes=base.get("max_nodes", 128),
            multi_hop_max_dist=base.get("multi_hop_max_dist", 5),
            target_type=base.get("target_type", "exp_spectrum"),
            attn_bias_w=base.get("attn_bias_w", 1.0),
            ex_normalize=base.get("ex_normalize"),
            prob_normalize=base.get("prob_normalize"),
            nm_dist_mode=base.get("nm_dist_mode", "hist"),
            nm_gauss_sigma=base.get("nm_gauss_sigma", 10.0),
            intensity_normalize=base.get("intensity_normalize", "min_max"),
            intensity_range=base.get("intensity_range", (200, 800)),
            x_cat_mode=base.get("x_cat_mode", "onehot"),
            global_cat_mode=base.get("global_cat_mode", "onehot"),
        )

        # ② Dataset 생성 (딱 한 번만)
        ds_tr = UnifiedSMILESDataset(csv_file=tmp_tr, **ds_kwargs)
        ds_va = UnifiedSMILESDataset(csv_file=tmp_va, **ds_kwargs)

        # ③ 마스크 기반 0-예측 베이스라인 (학습과 동일 SoftDTW 설정)
        zero_bls = compute_zero_baselines_masked_from_loader(ds_va, device=device, gamma=0.2, eps=1e-12)
        mae0, rmse0, sdtw0, sid0 = zero_bls["MAE0"], zero_bls["RMSE0"], zero_bls["SoftDTW0"], zero_bls["SID0"]
        print(f"[CV {fold + 1}] MAE0={mae0:.6g}  RMSE0={rmse0:.6g}  SoftDTW0={sdtw0:.6g}")
        print(f"[CV {fold + 1}] SID0(uniform) = {sid0:.6g}")

        # ④ probe (차원 주입은 그대로)
        probe_loader = DataLoader(ds_tr, batch_size=1, shuffle=False,
                                  collate_fn=partial(collate_fn, ds=ds_tr))
        batch0 = next(iter(probe_loader))
        F_cat = int(batch0["x_cat_onehot"].shape[-1])
        F_cont = int(batch0["x_cont"].shape[-1])
        F_edge = int(batch0["attn_edge_type"].shape[-1])
        max_pos = int(batch0["spatial_pos"].max().item()) if hasattr(batch0["spatial_pos"], "max") else 0
        base.update(dict(
            num_categorical_features=F_cat,
            num_continuous_features=F_cont,
            num_edges=F_edge,
            num_edge_dis=min(base.get("multi_hop_max_dist", 5), F_edge),
            num_spatial=max_pos + 1,
        ))

        print(f"[PROBE] F_cat={F_cat}, F_cont={F_cont}, F_edge={F_edge}, num_spatial={max_pos+1}")
        print("[CONFIG] emb={embedding_dim} heads={num_attention_heads} L={num_encoder_layers} "
              f"F_cat={base['num_categorical_features']} F_cont={base['num_continuous_features']} "
              f"F_edge={base['num_edges']} num_spatial={base['num_spatial']} num_edge_dis={base['num_edge_dis']}")

        res, _best = train_model_ex_porb(
            config=dict(base), target_type="exp_spectrum",
            loss_function_full_spectrum=list(FIX_LOSS_COMBO),
            loss_function_ex=["MSE","MAE","SOFTDTW","SID"],
            loss_function_prob=["MSE","MAE","SOFTDTW","SID"],
            num_epochs=int(FIX_NUM_EPOCHS),
            batch_size=int(FIX_BATCH_SIZE),
            n_pairs=int(FIX_N_PAIRS),
            learning_rate=float(cfg["learning_rate"]),
            DATASET=ds_tr, TEST_VAL_DATASET=ds_va, alpha=float(FIX_ALPHA), is_cv=True,
            nominal_feature_vocab=base["nominal_feature_vocab"],
            continuous_feature_names=base.get("continuous_feature_names", []),
            patience=None, num_workers=int(FIX_NUM_WORKERS),
            debug=bool(FIX_DEBUG), use_gradnorm=bool(FIX_USE_GRADNORM),
            init_from_ckpt=None, load_strict=None, ignore_prefixes=("output_layer",),
            save_milestones=[], figure_milestones=[],
            draw_overlays_on_milestone=False, overlay_topk=0, overlay_save_all=False,
            cv_context={"is_cv": True, "fold": fold+1, "n_splits": int(FIX_CV_FOLDS)},
            run_id=f"bo_cvF{fold+1:02d}", fold_idx=fold+1,
        )

        mae  = _safe_pick(res, "mae", "val")
        rmse = _safe_pick(res, "rmse", "val")
        sid  = _safe_pick(res, "sid", "val")
        sdtw = _safe_pick(res, "softdtw", "val")
        r2   = _safe_pick(res, "r2", "val")

        # ratios
        def _safe_ratio(num, den, eps=1e-12, bad=1e6):
            if not np.isfinite(num):
                return float(bad)
            d = den if (np.isfinite(den) and den > eps) else eps
            r = float(num) / float(d)
            return float(r) if np.isfinite(r) else float(bad)

        mae_ratio = _safe_ratio(mae, mae0)
        rmse_ratio = _safe_ratio(rmse, rmse0)
        sdtw_ratio = _safe_ratio(sdtw, sdtw0)
        sid_ratio = _safe_ratio(sid, sid0)

        # aggregate
        agg["MAE"].append(mae);       agg["MAE0"].append(mae0);       agg["MAE_ratio"].append(mae_ratio)
        agg["RMSE"].append(rmse);     agg["RMSE0"].append(rmse0);     agg["RMSE_ratio"].append(rmse_ratio)
        agg["SoftDTW"].append(sdtw);  agg["SoftDTW0"].append(sdtw0);  agg["SoftDTW_ratio"].append(sdtw_ratio)
        agg["SID"].append(sid);       agg["SID0"].append(sid0);       agg["SID_ratio"].append(sid_ratio)
        agg["R2"].append(r2)

        dt = time.perf_counter() - t0
        print(f"[CV {fold+1}/{FIX_CV_FOLDS}] done  "
              f"MAE={mae:.4f}(r={mae_ratio:.4f})  RMSE={rmse:.4f}(r={rmse_ratio:.4f})  "
              f"SID={sid:.4f}(r={sid_ratio:.4f})  sDTW={sdtw:.4f}(r={sdtw_ratio:.4f})  "
              f"R2={r2:.4f}  elapsed={dt/60:.1f} min")

        for p in (tmp_tr,tmp_va):
            try: os.remove(p)
            except: pass

    return {k: float(np.mean(v) if len(v) else 0.0) for k,v in agg.items()}

# ───────── 목적 벡터(최대화) — 4차원 ─────────
#  -MAE_ratio, -RMSE_ratio, -SID_ratio, -SoftDTW_ratio
def objective_vector(metrics: dict) -> torch.Tensor:
    return torch.tensor(
        [-metrics["MAE_ratio"], -metrics["RMSE_ratio"], -metrics["SID_ratio"], -metrics["SoftDTW_ratio"]],
        dtype=torch.double,
    )

def sobol_init(n: int, device: torch.device) -> torch.Tensor:
    X = torch.rand(n, D, dtype=torch.double, device=device)
    mins, maxs = BOUNDS[0].to(device), BOUNDS[1].to(device)
    return mins + (maxs - mins) * X

def make_qnehvi(model, ref_point, train_X, train_Y, sampler):
    try:
        part = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
        return qNEHVI(model=model, ref_point=ref_point, partitioning=part, sampler=sampler, prune_baseline=True)
    except TypeError:
        try:
            return qNEHVI(model=model, ref_point=ref_point, X_baseline=train_X, Y=train_Y, sampler=sampler, prune_baseline=True)
        except TypeError:
            return qNEHVI(model=model, ref_point=ref_point, X_baseline=train_X, sampler=sampler, prune_baseline=True)

def build_model(train_X: torch.Tensor, train_Y: torch.Tensor) -> ModelListGP:
    models = []
    for i in range(train_Y.shape[-1]):
        m = SingleTaskGP(train_X, train_Y[..., i:i+1],
                         input_transform=Normalize(d=D), outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
        models.append(m)
    return ModelListGP(*models)

def make_sampler(n=128):
    try: return SobolQMCNormalSampler(sample_shape=torch.Size([n]))
    except TypeError: return SobolQMCNormalSampler(num_samples=n)

# ───────── 메인 BO 루프 ─────────
def bo_loop(n_init=8, n_iter=10, batch_q=2, device: str = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] device = {device}")

    history, train_X_list, train_Y_list = [], [], []
    print(f"[INIT] sampling {n_init} candidates")
    X0 = sobol_init(n_init, device=device); accepted = 0
    for i, xi in enumerate(X0, 1):
        cfg = decode(xi)
        if not is_feasible(cfg):
            print(f"[INIT {i}/{n_init}] infeasible → skip (H={cfg['embedding_dim']}, A={cfg['num_attention_heads']})")
            continue
        print(f"[INIT {accepted+1}/{n_init}] eval (H={cfg['embedding_dim']}, A={cfg['num_attention_heads']}, "
              f"ffn×={cfg['ffn_multiplier']}, L={cfg['num_encoder_layers']}, "
              f"hop={cfg['multi_hop_max_dist']}, p={cfg['dropout']:.2f})")
        m = run_cv_and_get_metrics(cfg, device=device)
        y = objective_vector(m).to(device)
        train_X_list.append(xi); train_Y_list.append(y)
        history.append({"cfg": cfg, "metrics": m, "obj": y.cpu().tolist()})
        accepted += 1

    if not train_X_list:
        xi = torch.tensor([
            CHOICES_EMB.index(256),
            CHOICES_HEADS.index(8),
            FFN_MUL.index(2),
            CHOICES_LAYERS.index(4),
            CHOICES_HOPS.index(3),
            0.10,  # dropout
            0.10,  # attention_dropout
            0.10,  # activation_dropout
            CHOICES_LR.index(3e-5),
            CHOICES_ACTFN.index("gelu"),
        ], dtype=torch.double, device=device)
        cfg = decode(xi); print("[INIT fallback] injecting one feasible config:", cfg)
        m = run_cv_and_get_metrics(cfg, device=device); y = objective_vector(m).to(device)
        train_X_list.append(xi); train_Y_list.append(y)
        history.append({"cfg": cfg, "metrics": m, "obj": y.cpu().tolist()})

    train_X = torch.stack(train_X_list).to(device)
    train_Y = torch.stack(train_Y_list).to(device)

    # ref_point (4D): baseline과 동등(ratio=1) → objective는 -1
    ref_point = compute_ref_point(train_Y)

    sampler = make_sampler(128)

    for t in range(n_iter):
        ref_point = compute_ref_point(train_Y)
        print(f"\n[BO {t+1}/{n_iter}] build GP & optimize acquisition ...")
        model = build_model(train_X, train_Y).to(device)
        acq = make_qnehvi(model, ref_point, train_X, train_Y, sampler)
        cand, _ = optimize_acqf(acq_function=acq, bounds=BOUNDS.to(device),
                                 q=batch_q, num_restarts=8, raw_samples=256)
        print(f"[BO {t+1}/{n_iter}] suggested {batch_q} candidates")

        new_X, new_Y = [], []
        for j, xi in enumerate(cand, 1):
            cfg = decode(xi)
            if not is_feasible(cfg):
                print(f"[BO {t+1}/{n_iter} · {j}/{batch_q}] infeasible → skip (H={cfg['embedding_dim']}, A={cfg['num_attention_heads']})")
                continue
            print(f"[BO {t+1}/{n_iter} · {j}/{batch_q}] eval (H={cfg['embedding_dim']}, A={cfg['num_attention_heads']}, "
                  f"ffn×={cfg['ffn_multiplier']}, L={cfg['num_encoder_layers']}, "
                  f"hop={cfg['multi_hop_max_dist']}, p={cfg['dropout']:.2f})")
            m = run_cv_and_get_metrics(cfg, device=device)
            y = objective_vector(m).to(device)
            new_X.append(xi); new_Y.append(y)
            history.append({"cfg": cfg, "metrics": m, "obj": y.cpu().tolist()})

        if not new_X:
            print(f"[BO {t+1}/{n_iter}] all suggested candidates infeasible/failed; skip update.")
            continue

        train_X = torch.cat([train_X, torch.stack(new_X).to(device)], dim=0)
        train_Y = torch.cat([train_Y, torch.stack(new_Y).to(device)], dim=0)
        hv = NondominatedPartitioning(ref_point=ref_point, Y=train_Y).compute_hypervolume().item()
        print(f"[BO {t+1}/{n_iter}] HV={hv:.6f} | total_evals={train_X.size(0)}")

    # Pareto 요약 (R²는 참고)
    Y_all = torch.tensor([h["obj"] for h in history], dtype=torch.double)
    nd_mask = is_non_dominated(Y_all)
    pareto = [h for h, keep in zip(history, nd_mask.tolist()) if keep]
    print("\n=== Pareto set (non-dominated) ===")
    for i, h in enumerate(pareto, 1):
        c, m = h["cfg"], h["metrics"]
        print(f"[{i}] R2={m['R2']:.4f}  "
              f"MAE={m['MAE']:.4f}(r={m['MAE_ratio']:.4f})  "
              f"RMSE={m['RMSE']:.4f}(r={m['RMSE_ratio']:.4f})  "
              f"SID={m['SID']:.4f}(r={m['SID_ratio']:.4f})  "
              f"sDTW={m['SoftDTW']:.4f}(r={m['SoftDTW_ratio']:.4f})")
        print("     cfg:", c)

    # CSV 저장 (원지표, baseline, ratio, 목적 4개, R2 기록)
    with open("bo_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "embedding_dim", "heads", "ffn_mul", "layers", "hops",
            "dropout", "attn_drop", "act_drop",
            "lr", "act_fn",
            "MAE", "MAE0", "MAE_ratio",
            "RMSE", "RMSE0", "RMSE_ratio",
            "SID", "SID0", "SID_ratio",
            "SoftDTW", "SoftDTW0", "SoftDTW_ratio",
            "R2",
            "obj1(-MAE_ratio)", "obj2(-RMSE_ratio)", "obj3(-SID_ratio)", "obj4(-SoftDTW_ratio)",
            "is_pareto"
        ])
        for keep, h in zip(nd_mask.tolist(), history):
            c, m, y = h["cfg"], h["metrics"], h["obj"]
            w.writerow([
                c["embedding_dim"], c["num_attention_heads"], c["ffn_multiplier"], c["num_encoder_layers"],
                c["multi_hop_max_dist"], c["dropout"], c["attention_dropout"], c["activation_dropout"],
                c["learning_rate"], c["activation_fn"],
                m["MAE"], m.get("MAE0",0.0), m.get("MAE_ratio",0.0),
                m["RMSE"], m.get("RMSE0",0.0), m.get("RMSE_ratio",0.0),
                m["SID"], m.get("SID0",0.0), m.get("SID_ratio",0.0),
                m["SoftDTW"], m.get("SoftDTW0",0.0), m.get("SoftDTW_ratio",0.0),
                m["R2"],
                *y, int(keep)
            ])
    print("Saved: bo_results.csv")
    return train_X, train_Y

if __name__ == "__main__":
    bo_loop(n_init=N_INIT, n_iter=N_ITER, batch_q=BATCH_Q, device=None)
