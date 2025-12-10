# Load_Model_And_Predict.py
# 기능: Graphormer(exp_spectrum) 로드 → DataLoader → 200..800nm(601) 예측
# 사용: from Load_Model_And_Predict import execute_predict

import os, re, json, tempfile
from typing import List, Dict, Any
import pandas as pd
import torch
from torch.utils.data import DataLoader
from functools import partial
from rdkit import Chem
from rdkit.Chem import inchi as rdInchi
import numpy as np
# ---- 프로젝트 내부 모듈 import (필요시 경로만 조정) ----
from GP.models_All.graphormer_layer_modify.graphormer_LayerModify import GraphormerModel
from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset, collate_fn as graph_collate_fn
from GP.Custom_Loss.SID_loss import sid_loss
from GP.Custom_Loss.soft_dtw_cuda import SoftDTW

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")   # 경고만 끄기
# ------------------ 유틸 ------------------
def to_wsl_path(p: str) -> str:
    """Windows 'C:\\...' → WSL '/mnt/c/...' 로 변환 (WSL에서 실행 시)."""
    if os.name == "posix" and re.match(r"^[A-Za-z]:\\", p):
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p

def _norm_series(val_list, n, default=None, is_cont=False):
    """리스트 길이를 n에 맞춰 자르거나 패딩. None이면 기본값으로 채움."""
    if val_list is None:
        base = (0.0 if is_cont else "Unknown") if default is None else default
        return [base] * n
    vals = list(val_list)
    if len(vals) < n:
        pad = n - len(vals)
        vals += [vals[-1] if len(vals) else (0.0 if is_cont else "Unknown")] * pad
    elif len(vals) > n:
        vals = vals[:n]
    if is_cont:
        vals = [float(x) if x is not None else 0.0 for x in vals]
    else:
        vals = [str(x) if x is not None else "Unknown" for x in vals]
    return vals


# ------------------ 임시 CSV 생성 ------------------
def _make_temp_infer_csv(
    smiles_list: List[str],
    cfg: dict,
    per_globals: Dict[str, List] | None = None,
    global_defaults: Dict[str, Any] | None = None,
) -> str:
    """
    최소 스키마 + 200..800 nm 컬럼(0으로 채움)으로 임시 CSV 생성.
    - cfg['mol_col']이 'InChI'면 가능한 한 SMILES→InChI 변환, 실패 시 smiles 사용.
    - per_globals: {"pH_label":[..], "dielectric_constant_avg":[..], "Solvent":[..], "type":[..]} 등
    """
    n = len(smiles_list)
    per_globals = per_globals or {}
    global_defaults = global_defaults or {}

    mol_col = cfg.get("mol_col", "smiles")
    s_nm, e_nm = cfg.get("intensity_range", [200, 800])
    nm_cols = list(range(int(s_nm), int(e_nm) + 1))

    df = pd.DataFrame({"smiles": smiles_list})

    # InChI 필요 시 변환
    if mol_col.lower() == "inchi":
        inchi_vals = []
        for s in smiles_list:
            try:
                m = Chem.MolFromSmiles(s)
                inchi_vals.append(rdInchi.MolToInchi(m) if m else "")
            except Exception:
                inchi_vals.append("")
        if any(inchi_vals):
            df["InChI"] = inchi_vals
        else:
            mol_col = "smiles"  # 변환 거의 실패 시 smiles로 전환

    # 명목형/연속형 글로벌 피처 처리
    vocab_dict = cfg.get("GLOBAL_FEATURE_VOCABS_dict", {}) or {}
    cont_names = cfg.get("continuous_feature_names", []) or []

    # 명목형
    for col, vocab in vocab_dict.items():
        fallback = global_defaults.get(col)
        if fallback is None:
            fallback = "Unknown" if "Unknown" in vocab else (vocab[0] if vocab else "Unknown")
        if col in per_globals:
            df[col] = _norm_series(per_globals[col], n, fallback, is_cont=False)
        else:
            df[col] = [fallback] * n

    # 연속형
    for col in cont_names:
        fallback = float(global_defaults.get(col, 0.0))
        if col in per_globals:
            df[col] = _norm_series(per_globals[col], n, fallback, is_cont=True)
        else:
            df[col] = [fallback] * n

    # 기타 메타(필요 시)
    for extra in ["DataKind", "type", "Solvent", "DB", "ID", "solvent_phase",
                  "is_gas", "is_liquid", "is_solid", "is_qm", "__strat__"]:
        if extra not in df.columns:
            if extra in ("is_gas", "is_liquid", "is_solid", "is_qm"):
                df[extra] = 0
            elif extra == "type":
                # 기본은 Emission
                if "type" in per_globals:
                    df[extra] = _norm_series(per_globals["type"], n, "Emission", is_cont=False)
                else:
                    df[extra] = "Emission"
            elif extra == "Solvent":
                if "Solvent" in per_globals:
                    df[extra] = _norm_series(per_globals["Solvent"], n, "Water", is_cont=False)
                else:
                    df[extra] = "Water"
            else:
                df[extra] = ""

    # 200..800 컬럼을 한 번에 생성 (fragmentation 방지)
    nm_values = list(range(int(s_nm), int(e_nm) + 1))  # ex) 200..800
    nm_df = pd.DataFrame(
        np.zeros((len(df), len(nm_values)), dtype=np.float32),
        columns=[str(n) for n in nm_values],
        index=df.index,
    )
    df["orig_idx"] = list(range(len(df)))

    # 수평 결합 (주의: pd.concat 사용, df.concat 아님)
    df = pd.concat([df, nm_df], axis=1, copy=False)

    # cfg에 실제 mol_col 반영
    cfg["mol_col"] = mol_col

    tmp = tempfile.NamedTemporaryFile(prefix="infer_", suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    return tmp.name


# ------------------ 실행 함수(웹에서 import) ------------------
def execute_predict(
    smiles_list: List[str],
    ckpt_path: str,
    config_json_path: str,
    dielectric_list: List[float] | None = None,
    phlabel_list: List[str] | None = None,
    solvent_list: List[str] | None = None,
    device: str | None = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    반환: {"x_nm":[200..800], "preds":[{"smiles": s, "y":[601]}, ...]}
    - type은 Emission으로 강제.
    - dielectric_list, phlabel_list, solvent_list는 per-row 적용(길이 자동 보정).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = to_wsl_path(ckpt_path)
    config_json_path = to_wsl_path(config_json_path)

    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    s_nm, e_nm = cfg.get("intensity_range", [200, 800])
    out_size = int(cfg.get("output_size", int(e_nm - s_nm) + 1))
    x_nm = list(range(int(s_nm), int(e_nm) + 1))

    # per-row 글로벌 세팅
    n = len(smiles_list)
    per_globals = {
        "type": ["Emission"] * n,  # ★ Emission 고정
    }
    if phlabel_list is not None:
        per_globals["pH_label"] = phlabel_list
    if dielectric_list is not None:
        per_globals["dielectric_constant_avg"] = dielectric_list
    if solvent_list is not None:
        per_globals["Solvent"] = solvent_list

    # 임시 CSV 생성
    csv_path = _make_temp_infer_csv(
        smiles_list=smiles_list,
        cfg=cfg,
        per_globals=per_globals,
        global_defaults=None,
    )
    csv_path = to_wsl_path(csv_path)

    # Dataset / DataLoader
    ds = UnifiedSMILESDataset(
        csv_file=csv_path,
        nominal_feature_vocab=cfg.get("nominal_feature_vocab", cfg.get("GLOBAL_FEATURE_VOCABS_dict", {})),
        continuous_feature_names=cfg.get("continuous_feature_names", []),
        global_cat_dim=cfg.get("global_cat_dim", 0),
        global_cont_dim=cfg.get("global_cont_dim", 0),
        ATOM_FEATURES_VOCAB=cfg["ATOM_FEATURES_VOCAB"],
        float_feature_keys=cfg.get("float_feature_keys", cfg.get("ATOM_FLOAT_FEATURE_KEYS", [])),
        BOND_FEATURES_VOCAB=cfg["BOND_FEATURES_VOCAB"],
        GLOBAL_FEATURE_VOCABS_dict=cfg.get("GLOBAL_FEATURE_VOCABS_dict", {}),
        x_cat_mode=cfg.get("x_cat_mode", "onehot"),
        global_cat_mode=cfg.get("global_cat_mode", "onehot"),
        mol_col=cfg.get("mol_col", "smiles"),
        mode=cfg.get("mode", "cls_global_data"),
        max_nodes=cfg.get("max_nodes", 128),
        multi_hop_max_dist=cfg.get("multi_hop_max_dist", 5),
        target_type="exp_spectrum",
        intensity_normalize=cfg.get("intensity_normalize", "min_max"),
        intensity_range=(s_nm, e_nm),
        attn_bias_w=cfg.get("attn_bias_w", 1.0),
        ex_normalize=cfg.get("ex_normalize"),
        prob_normalize=cfg.get("prob_normalize"),
        nm_dist_mode=cfg.get("nm_dist_mode", "hist"),
        nm_gauss_sigma=cfg.get("nm_gauss_sigma", 10.0),
        deg_clip_max=cfg.get("deg_clip_max", 5),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=partial(graph_collate_fn, ds=ds))

    if "orig_idx" in ds.data.columns:
        orig_idx_list = ds.data["orig_idx"].astype(int).tolist()
    else:
        # fallback: 예전 코드와 동일하게 0..n-1 가정
        orig_idx_list = list(range(len(smiles_list)))

    # 모델 로드
    model = GraphormerModel(cfg, target_type="exp_spectrum").to(device)
    blob = torch.load(ckpt_path, map_location=device)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=False)
    model.eval()

    # 추론
    preds = []
    pos = 0  # Dataset 내 index 포인터

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            model_in = {k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in batch.items() if k not in ("targets", "masks")}
            out = model(model_in, target_type="exp_spectrum")  # [B, out_size]
            y = out.detach().cpu().numpy()
            B = y.shape[0]

            for b in range(B):
                vec = y[b].tolist()
                if len(vec) != out_size:
                    if len(vec) > out_size:
                        vec = vec[:out_size]
                    else:
                        vec = vec + [0.0] * (out_size - len(vec))

                # ★ Dataset에서 살아 남은 샘플의 원본 인덱스
                orig_i = int(orig_idx_list[pos + b])
                smi = smiles_list[orig_i] if 0 <= orig_i < len(smiles_list) else ""

                preds.append({
                    "orig_idx": orig_i,
                    "smiles": smi,
                    "y": vec,
                })
            pos += B

    return {
        "x_nm": x_nm,
        "preds": preds,
        "orig_idx": orig_idx_list,  # ★ 매핑 정보도 같이 반환
    }

# ------------------ 예측 후 CSV 생성 ------------------
def save_prediction_csvs(
    original_df: pd.DataFrame,
    pred_result: Dict[str, Any],
    out_csv_spectrum: str,
    out_csv_metrics: str,
    device: str | None = None,
):
    """
    original_df : 실험 스펙트럼 + 조건이 들어있는 DataFrame
                  (200~800 nm 컬럼 이름: '200','201',... 형식 가정)
    pred_result : execute_predict()의 반환값
                  {"x_nm":[200..800], "preds":[{"smiles":..., "y":[601]}, ...]}
    out_csv_spectrum : 실험/예측 전체 스펙트럼을 넣을 CSV 경로
    out_csv_metrics  : 분자별 MAE/RMSE/SID/SIS/SoftDTW를 넣을 CSV 경로
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    df = original_df.reset_index(drop=True).copy()

    # --- 1) x축 / 예측 벡터 정리 ---
    x_nm: List[int] = pred_result["x_nm"]
    nm_cols = [str(nm) for nm in x_nm]

    # 실험 스펙트럼 컬럼 존재 여부 체크
    missing = [c for c in nm_cols if c not in df.columns]
    if missing:
        raise ValueError(f"입력 DataFrame에 다음 파장 컬럼이 없습니다: {missing[:10]} ...")

    preds_list = pred_result["preds"]
    orig_idx_list = pred_result.get("orig_idx", None)

    # ★ orig_idx가 있으면, 원본 df에서 해당 행만 골라서 사용
    if orig_idx_list is not None:
        df = df.iloc[orig_idx_list].reset_index(drop=True)
        #if len(orig_idx_list) != len(preds_list):
        #    raise ValueError(f"orig_idx 길이({len(orig_idx_list)})와 예측 개수({len(preds_list)})가 다릅니다.")

    # 최종적으로 길이 다시 확인
    if len(preds_list) != len(df):
        raise ValueError(f"DataFrame row 수({len(df)})와 예측 개수({len(preds_list)})가 다릅니다.")

    # (N, L) 행렬로 변환
    pred_mat = np.array([row["y"] for row in preds_list], dtype=np.float32)
    exp_mat = df[nm_cols].to_numpy(dtype=np.float32)

    # --- 2) 1번 CSV: 실험 + 예측 스펙트럼 전체 저장 ---

    # 메타(분자/조건) 컬럼을 앞으로 배치
    meta_candidates = [
        "Name", "ID", "DB", "DataKind",
        "smiles", "SMILES", "InChI", "inchi",
        "type", "pH_label", "pH",
        "dielectric_constant_avg", "Solvent",
        "solvent_phase", "is_qm", "is_gas", "is_liquid", "is_solid",
    ]
    meta_cols: List[str] = []
    seen = set()
    for c in meta_candidates:
        if c in df.columns and c not in seen:
            meta_cols.append(c)
            seen.add(c)
    # 나머지 non-spectrum 컬럼들도 뒤에 추가
    for c in df.columns:
        if c in nm_cols or c in meta_cols:
            continue
        meta_cols.append(c)

    df_meta = df[meta_cols].copy()

    # 실험/예측 컬럼 이름: exp_200, exp_201, ..., pred_200, pred_201, ...
    exp_cols = [f"exp_{nm}" for nm in x_nm]
    pred_cols = [f"pred_{nm}" for nm in x_nm]

    df_spectrum = df_meta.copy()
    df_spectrum[exp_cols] = exp_mat
    df_spectrum[pred_cols] = pred_mat

    df_spectrum.to_csv(out_csv_spectrum, index=False, encoding="utf-8-sig")

    # --- 3) 2번 CSV: 분자별 metric (MAE, RMSE, SID, SIS, SoftDTW) ---

    # SoftDTW 인스턴스 (안 되면 SoftDTW만 NaN 처리)
    try:
        soft_dtw = SoftDTW(use_cuda=(torch_device.type == "cuda"), gamma=0.2,
                           bandwidth=None, normalize=True,)
    except Exception as e:
        print(f"[WARN] SoftDTW 초기화 실패: {e}")
        soft_dtw = None

    records = []
    for i in range(len(df)):
        y_true = exp_mat[i]    # (L,)
        y_pred = pred_mat[i]   # (L,)

        # --- MAE / RMSE ---
        diff = y_pred - y_true
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        # --- SID / SIS ---
        y_true_t = torch.tensor(y_true, dtype=torch.float32, device=torch_device).unsqueeze(0)  # (1,L)
        y_pred_t = torch.tensor(y_pred, dtype=torch.float32, device=torch_device).unsqueeze(0)  # (1,L)
        mask_t   = torch.ones_like(y_true_t, dtype=torch.bool, device=torch_device)             # (1,L)

        sid_val_t = sid_loss(y_pred_t, y_true_t, mask_t, reduction="sum")
        sid_val = float(sid_val_t.detach().cpu().item())
        # ★ SIS 정의: 1 / (1 + SID)  (0~1 범위의 similarity로 매핑)
        sis_val = float(1.0 / (1.0 + sid_val))

        # --- SoftDTW ---
        if soft_dtw is not None:
            try:
                x = y_pred_t.unsqueeze(-1)  # (1,L,1)
                y = y_true_t.unsqueeze(-1)  # (1,L,1)
                soft_val = float(soft_dtw(x, y).detach().cpu().item())
            except Exception as e:
                print(f"[WARN] SoftDTW 계산 실패 (row {i}): {e}")
                soft_val = float("nan")
        else:
            soft_val = float("nan")

        rec = {}

        # 기본 식별자/조건 정보도 같이 저장
        for col in ["Name", "ID", "DB",
                    "smiles", "SMILES", "InChI", "inchi",
                    "type", "pH_label", "dielectric_constant_avg", "Solvent"]:
            if col in df.columns:
                rec[col] = df.loc[i, col]

        rec.update({
            "mae": mae,
            "rmse": rmse,
            "sid": sid_val,
            "sis": sis_val,
            "softdtw": soft_val,
        })
        records.append(rec)

    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(out_csv_metrics, index=False, encoding="utf-8-sig")

def run_for_split(
    split_name: str,
    exp_csv_path: str,
    ckpt_path: str,
    config_json_path: str,
    out_dir: str,
    device: str | None = None,
    batch_size: int = 32,
):
    """
    한 데이터셋(train/test)에 대해:
      - 실험 CSV 읽기
      - execute_predict로 예측
      - 스펙트럼 CSV + metric CSV 저장
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) 실험 데이터 로드
    exp_csv_path = to_wsl_path(exp_csv_path)
    df_exp = pd.read_csv(exp_csv_path)

    # 2) 분자 식별자 추출 (smiles 우선, 없으면 InChI 사용)
    if "smiles" in df_exp.columns:
        smiles_list = df_exp["smiles"].astype(str).tolist()

    elif "SMILES" in df_exp.columns:
        smiles_list = df_exp["SMILES"].astype(str).tolist()

    elif "InChI" in df_exp.columns or "inchi" in df_exp.columns:
        inchi_col = "InChI" if "InChI" in df_exp.columns else "inchi"
        inchi_series = df_exp[inchi_col].astype(str).fillna("")

        smiles_list = []
        fail_count = 0
        for ich in inchi_series:
            if not ich or ich.strip() == "":
                smiles_list.append("")
                fail_count += 1
                continue
            try:
                mol = rdInchi.MolFromInchi(ich)
                if mol is None:
                    smiles_list.append("")
                    fail_count += 1
                else:
                    smiles_list.append(Chem.MolToSmiles(mol))
            except Exception:
                smiles_list.append("")
                fail_count += 1

        if fail_count > 0:
            print(f"[WARN] InChI → SMILES 변환 실패 {fail_count}개 (빈 문자열로 대체)")

    else:
        raise ValueError("실험 CSV에 'smiles'/'SMILES' 또는 'InChI'/'inchi' 컬럼이 필요합니다.")

    # ★★★ 2.5) 유효하지 않은 SMILES 행 드롭 ("", NaN 등)
    valid_mask = []
    for s in smiles_list:
        if s is None:
            valid_mask.append(False)
        else:
            s_str = str(s).strip()
            # 빈 문자열이거나 "nan" 문자열이면 invalid
            valid_mask.append(bool(s_str) and s_str.lower() != "nan")

    if not all(valid_mask):
        n_before = len(smiles_list)
        smiles_list = [s for s, ok in zip(smiles_list, valid_mask) if ok]
        df_exp = df_exp.loc[valid_mask].reset_index(drop=True)
        print(f"[WARN] SMILES/INCHI 변환 실패 등으로 {n_before - len(smiles_list)}개 행을 제거했습니다.")

    # 3) 조건들 추출
    phlabel_list = df_exp["pH_label"].tolist() if "pH_label" in df_exp.columns else None
    dielectric_list = df_exp["dielectric_constant_avg"].tolist() if "dielectric_constant_avg" in df_exp.columns else None
    solvent_list = df_exp["Solvent"].tolist() if "Solvent" in df_exp.columns else None

    # 4) 예측 수행
    pred_result = execute_predict(
        smiles_list=smiles_list,          # ← InChI만 있던 경우에도 여기로 통일
        ckpt_path=ckpt_path,
        config_json_path=config_json_path,
        phlabel_list=phlabel_list,
        dielectric_list=dielectric_list,
        solvent_list=solvent_list,
        device=device,
        batch_size=batch_size,
    )

    # 5) CSV 경로 설정
    out_csv_spectrum = os.path.join(out_dir, f"{split_name}_spectrum_exp_pred.csv")
    out_csv_metrics  = os.path.join(out_dir, f"{split_name}_metrics_per_molecule.csv")

    # 6) CSV 생성
    save_prediction_csvs(
        original_df=df_exp,
        pred_result=pred_result,
        out_csv_spectrum=out_csv_spectrum,
        out_csv_metrics=out_csv_metrics,
        device=device,
    )

    print(f"[{split_name}] 완료")
    print(" - 스펙트럼 CSV:", out_csv_spectrum)
    print(" - metric CSV :", out_csv_metrics)

def main():
    # 공통 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 checkpoint / config
    ckpt = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\GP\models_All\graphormer_layer_modify\models\best_model_em.pth"
    cfg  = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\GP\models_All\graphormer_layer_modify\models\config_eval_em.json"

    # 평가할 데이터셋들 (train, test)
    # → 여기만 실제 파일 경로에 맞게 수정하면 됨
    datasets = [
        ("train", r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_train_clustered_resplit_with_mu_eps_fillZero.csv"),
        ("test",  r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\EM_stratified_test_clustered_resplit_with_mu_eps_fillZero.csv"),
    ]

    # 결과 저장 폴더
    out_dir = r"C:\Users\analcheminfo\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\GP\models_All\graphormer_layer_modify\models"

    for split_name, exp_csv in datasets:
        run_for_split(
            split_name=split_name,
            exp_csv_path=exp_csv,
            ckpt_path=ckpt,
            config_json_path=cfg,
            out_dir=out_dir,
            device=device,
            batch_size=32,
        )

if __name__ == "__main__":
    main()
