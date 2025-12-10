import pandas as pd
import numpy as np
from typing import Iterable
# 맨 위에 이미 있다면 유지, 없다면 추가
from typing import Optional, List, Tuple, Dict
# from click.formatting import iter_rows
from rdkit import Chem
try:
    from GP.data_prepare.Chem_Graph_Utils import mol_from_text
except Exception:
    mol_from_text = None  # 없으면 fallback 사용

def _max_degree_from_df(
    df: pd.DataFrame,
    mol_col_candidates=("InChI","inchi","smiles","SMILES"),
    sample_rows: Optional[int] = 2000,  # int | None → Optional[int]   # 속도 위해 일부만 스캔
) -> int:
    mol_col = next((c for c in mol_col_candidates if c in df.columns), None)
    if mol_col is None:
        return 4  # 안전 폴백

    series = df[mol_col].astype(str)
    if sample_rows is not None and len(series) > sample_rows:
        series = series.sample(sample_rows, random_state=42)

    dmax = 0
    for txt in series:
        try:
            m = Chem.MolFromInchi(txt) if txt.startswith("InChI=") else Chem.MolFromSmiles(txt)
        except Exception:
            m = None
        if m is None:
            continue
        for a in m.GetAtoms():
            dmax = max(dmax, a.GetDegree())

    return max(dmax, 3) + 1  # 최소 4 보장

def _normalize_token(s: str) -> str:
    s = ("" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)).strip()
    if not s:
        return "Unknown"
    s = " ".join(s.split())
    return s.title()

def _normalize_solvent_cell(cell: str) -> str:
    """
    'ethanol + water + ethanol' -> 'Ethanol + Water' (중복 제거 + 정렬 + 포맷 통일)
    """
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return "Unknown"
    parts = [p for p in map(_normalize_token, str(cell).split("+")) if p and p != "Unknown"]
    if not parts:
        return "Unknown"
    # 중복 제거
    uniq = sorted(set(p.strip() for p in parts if p.strip()))
    return " + ".join(uniq) if uniq else "Unknown"

from typing import Optional, List, Tuple, Dict
def build_vocabs_from_df(
    df: pd.DataFrame,
    nominal_cols: Optional[List[str]] = None,
    continuous_cols: Optional[List[str]] = None,
    plus_split_cols: tuple[str, ...] = ("Solvent",),  # 여기에 있는 칼럼은 한 칼럼 그대로 유지하며 내부 토큰만 처리
):
    """
    - DF를 받아 vocab/차원 계산
    - plus_split_cols에 포함된 칼럼은 '한 칼럼 유지'하되, 셀 내부를 '+'로 분해하여
      1) DF 값은 중복 제거·정렬로 정규화
      2) vocab은 '개별 용매 토큰'의 합집합으로 생성
    - 반환: (정규화된 df, nominal_vocab, continuous_feature_names, global_cat_dim, global_cont_dim)
    """
    df_proc = df.copy()

    # 0) 자동 추론
    if nominal_cols is None:
        nominal_cols = [c for c in df_proc.columns
                        if pd.api.types.is_object_dtype(df_proc[c]) or
                           pd.api.types.is_string_dtype(df_proc[c]) or
                           pd.api.types.is_categorical_dtype(df_proc[c])]
        for c in ["type", "Type", "pH", "pH_label", "Solvent"]:
            if c in df_proc.columns and c not in nominal_cols:
                nominal_cols.append(c)

    if continuous_cols is None:
        continuous_cols = [c for c in df_proc.columns if pd.api.types.is_numeric_dtype(df_proc[c])]

    # 1) plus_split_cols: 한 칼럼 유지 + 셀 내부 정규화
    for col in plus_split_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].apply(_normalize_solvent_cell)

    # 2) nominal vocab 생성
    nominal_feature_vocab: dict[str, list[str]] = {}
    for col in nominal_cols:
        if col not in df_proc.columns:
            continue

        if col in plus_split_cols:
            # 개별 토큰의 합집합으로 vocab 생성
            token_set = set()
            for val in df_proc[col].astype(str):
                if not val or val == "nan":
                    token_set.add("Unknown")
                    continue
                # 정규화된 형태: 'A + B' → 토큰 분리
                tokens = [t.strip() for t in val.split("+")]
                for t in tokens:
                    t = _normalize_token(t)
                    if t:
                        token_set.add(t)
            if not token_set:
                token_set = {"Unknown"}
            nominal_feature_vocab[col] = sorted(token_set)
        else:
            # 일반 명목형: 셀 전체 문자열 기준 vocab
            uniq = sorted({_normalize_token(x) for x in df_proc[col].astype(str).tolist()})
            if not uniq:
                uniq = ["Unknown"]
            nominal_feature_vocab[col] = uniq

    # 3) 연속형 이름 필터링
    continuous_feature_names = [c for c in continuous_cols if c in df_proc.columns]

    # 4) 차원 계산
    global_cat_dim = sum(len(v) for v in nominal_feature_vocab.values())
    global_cont_dim = len(continuous_feature_names)

    return df_proc, nominal_feature_vocab, continuous_feature_names, global_cat_dim, global_cont_dim

def _mol_from_text_fallback(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        if text.startswith("InChI="):
            return Chem.MolFromInchi(text, sanitize=True, treatWarningAsError=False)
        else:
            return Chem.MolFromSmiles(text, sanitize=True)
    except Exception:
        return None

def _read_mol(text: str):
    if mol_from_text is not None:
        return mol_from_text(text)
    return _mol_from_text_fallback(text)

def build_atomic_num_vocab_from_dfs(
    df_all,
    mol_col_candidates: Iterable[str] = ("smiles", "SMILES", "InChI", "inchi"),
    *,
    add_unk_zero: bool = True,
    keep_sorted_unique: bool = True,
):
    """
    df_all 전체에서 분자 컬럼을 찾아 모든 원자번호를 수집한 뒤,
    [0] (UNK) + 등장한 원자번호들로 atomic_num vocab을 만들어 반환합니다.

    return 예: [0, 1, 6, 7, 8]  (데이터에 H, C, N, O만 있을 때)
    """
    # 1) 분자 컬럼 결정
    mol_col = None
    for c in mol_col_candidates:
        if c in df_all.columns:
            mol_col = c
            break
    if mol_col is None:
        raise ValueError(f"[build_atomic_num_vocab_from_dfs] 분자 컬럼을 찾지 못했습니다. 후보: {mol_col_candidates}")

    # 2) 스캔
    active = set()
    for text in df_all[mol_col].astype(str).tolist():
        m = _read_mol(text)
        if m is None:
            continue
        for a in m.GetAtoms():
            z = int(a.GetAtomicNum())
            if z > 0:
                active.add(z)

    if not active:
        # 데이터가 비어있는 경우라도 최소한 1(H)은 포함하도록 방어
        active = {1}

    # 3) vocab 구성
    uniq = sorted(active) if keep_sorted_unique else list(active)
    if add_unk_zero:
        return [0] + uniq
    return uniq
# =======================================================================
import pandas as pd
from typing import List, Tuple, Dict
import pandas as pd
from rdkit import Chem

def generate_graphormer_config(dataset_path_list: List[str],
                                mode: str = "cls_only",
                                embedding_dim: int = 768,
                                ffn_embedding_dim: int = 768,
                                num_heads: int = 32,
                                dropout: float = 0.1,
                                attn_dropout: float = 0.1,
                                act_dropout: float = 0.1,
                                activation_fn: str = "gelu",
                                pre_layernorm: bool = False,
                                num_layers: int = 12,
                                q_noise: float = 0.0,
                                qn_block_size: int = 8,
                                multi_hop_max_dist: int = 5,
                                mol_col: str = "InChI",
                                target_type:str = "exp_spectrum", # "default", "exp_spectrum", "ex_prob", "nm_distribution"
                                intensity_normalize: str = "min_max",
                                intensity_range:tuple = (200, 800),
                                ex_normalize:str = "min_max",
                                prob_normalize:str = "min_max",
                                float_feature_keys: List[str] = ['partial_charge', 'atomic_mass'],
                                global_feature_order: List[str] = ["pH_label", "type", "Solvent"],
                                global_multihot_cols: Dict[str, bool] = {"Solvent": True},
                                continuous_feature_names: List[str] = ['dielectric_constant_avg'],

                               ) -> Tuple[Dict, Dict]:
    """
    Generate Graphormer model config and feature vocabularies from dataset paths.
    Returns: (config_dict, vocab_dict)
    """
    dfs = [pd.read_csv(path, low_memory=False) for path in dataset_path_list]
    df_all = pd.concat(dfs, axis=0).reset_index(drop=True)

    # === Bond vocab (고정)
    BOND_FEATURES_VOCAB = {
        'bond_type': [
            Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
        ],
        'stereo': [
            Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS,
        ],
        'is_conjugated': [0, 1],
        'is_in_ring': [0, 1],
    }

    # === Atom vocab (atomic_num은 동적으로 추출)
    atomic_vocab = build_atomic_num_vocab_from_dfs(
        df_all,
        mol_col_candidates=("smiles", "SMILES", "InChI", "inchi"),
        add_unk_zero=True,
        keep_sorted_unique=True,
    )

    ATOM_FEATURES_VOCAB = {
        'atomic_num': atomic_vocab,
        'formal_charge': list(range(-5, 6)),
        'hybridization': [
            Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.OTHER,
        ],
        'is_aromatic': [0, 1],
        'total_num_hs': list(range(0, 9)),
        'explicit_valence': list(range(0, 8)),
        'total_bonds': list(range(0, 8)),
        'partial_charge': float,
        'atomic_mass': float,
    }

    ATOM_FLOAT_FEATURE_KEYS = float_feature_keys

    # === Global vocab 자동 추출
    #vocab_dict, *_ = build_vocabs_from_df(df_all)
    #print("vocab_dict",vocab_dict)
    #global_cat_dim = vocab_dict.get("global_cat_dim", 0)
    #print("global_cat_dim",global_cat_dim)
    #global_cont_dim = vocab_dict.get("global_cont_dim", 0)
    #print("global_cont_dim",global_cont_dim)

    # === 자동 계산 항목
    num_atoms = sum(len(v) for v in ATOM_FEATURES_VOCAB.values() if isinstance(v, (list, tuple)))
    num_edges = sum(len(v) for v in BOND_FEATURES_VOCAB.values())
    #num_in_degree = vocab_dict.get("num_in_degree", 512)
    #num_out_degree = vocab_dict.get("num_out_degree", 512)
    num_cat_feat = num_atoms
    num_cont_feat = len(ATOM_FLOAT_FEATURE_KEYS)
    num_spatial = num_edge_dis = multi_hop_max_dist # + 1

    # === 3) 차수 임베딩 크기(실데이터로 계산, 최소 4 보장) ===
    deg_cap = _max_degree_from_df(
        df_all,
        mol_col_candidates=("InChI", "inchi", "smiles", "SMILES"),
        sample_rows=2000,  # 전체가 크면 2천개 샘플만 스캔; 전부 보고 싶으면 None
    )
    num_in_degree = deg_cap
    num_out_degree = deg_cap

    config = {
        "num_atoms": num_atoms,
        "num_in_degree": num_in_degree,
        "num_out_degree": num_out_degree,
        "num_edges": num_edges,
        "num_spatial": num_spatial,
        "num_edge_dis": num_edge_dis,
        "edge_type": "multi_hop",
        "multi_hop_max_dist": multi_hop_max_dist,
        "num_encoder_layers": num_layers,
        "embedding_dim": embedding_dim,
        "ffn_embedding_dim": ffn_embedding_dim,
        "num_attention_heads": num_heads,
        "dropout": dropout,
        "attention_dropout": attn_dropout,
        "activation_dropout": act_dropout,
        "activation_fn": activation_fn,
        "pre_layernorm": pre_layernorm,
        "q_noise": q_noise,
        "qn_block_size": qn_block_size,
        "num_categorical_features": num_cat_feat,
        "num_continuous_features": num_cont_feat,
        "intensity_range":intensity_range,
        "mode": mode,
        "mol_col": mol_col,
        "target_type": target_type,
    }

    # 전역 카테고리/연속형 컬럼 정의
    global_feature_order = global_feature_order  # 전역 피처 순서
    config["global_multihot_cols"] = global_multihot_cols

    cont_cols_user = list(continuous_feature_names or [])
    nominal_cols = [c for c in global_feature_order if c not in cont_cols_user]

    # Solvent는 'A + B' 같은 혼합 문자열 유지 (내부 토큰만 정규화)
    _df_proc, nominal_feature_vocab, continuous_feature_names, global_cat_dim, global_cont_dim = build_vocabs_from_df(
        df_all,
        nominal_cols=nominal_cols ,  # 명목형
        continuous_cols=cont_cols_user,  # 연속형
        plus_split_cols=("Solvent",),  # ← 콤마 꼭!
    )

    #GLOBAL_FEATURE_VOCABS_dict = {
    #    name: nominal_feature_vocab[name] for name in global_feature_order
    #}
    GLOBAL_FEATURE_VOCABS_dict = {name: nominal_feature_vocab[name] for name in nominal_cols}

    config.update({
        "nominal_feature_vocab":    nominal_feature_vocab,
        "continuous_feature_names": continuous_feature_names,
        "global_cat_dim":           global_cat_dim,
        "global_cont_dim":          global_cont_dim,
        "GLOBAL_FEATURE_VOCABS_dict":    GLOBAL_FEATURE_VOCABS_dict,
        "global_feature_order":     global_feature_order,
        "ATOM_FEATURES_VOCAB": ATOM_FEATURES_VOCAB,
        "ATOM_FLOAT_FEATURE_KEYS": ATOM_FLOAT_FEATURE_KEYS,
        "BOND_FEATURES_VOCAB": BOND_FEATURES_VOCAB,
        # "GLOBAL_FEATURE_VOCABS": vocab_dict,  # 여기엔 global_cat_vocab 등도 있음
    })

    nm_step = 1  # 필요하면 0.5 등 소수도 가능
    # 2) output_size 자동 계산 (양 끝 포함 그리드)
    start_nm, end_nm = config["intensity_range"]
    n_bins = int(round((end_nm - start_nm) / nm_step)) + 1
    config["output_size"] = n_bins

    return config

if __name__ == "__main__":
    # 샘플 DataFrame 생성
    ".."
    path_list = [r"C:\Users\kogun\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\final_split\ABS_stratified_test_plus.csv",
                 r"C:\Users\kogun\PycharmProjects\Graphormer_UV_VIS_AND_Fluoroscence\graphormer_data\final_split\ABS_stratified_train_plus.csv"]

    config = generate_graphormer_config(dataset_path_list = path_list, mode="cls_global_model")
    print(config)
    for key, value in config.items():
        print(key, value)


