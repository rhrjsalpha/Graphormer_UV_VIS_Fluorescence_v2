# GP/data_prepare/chem_graph_utils.py
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
from typing import List, Optional, Dict, Any, Tuple


# --- (유지) 데이터 오류로 생기는 고립 H 제거(선택적) ---
def _drop_isolated_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    edit = Chem.RWMol(mol)
    to_remove = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1 and a.GetDegree() == 0]
    for idx in sorted(to_remove, reverse=True):
        edit.RemoveAtom(idx)
    return edit.GetMol()


# --- (유지) Normalize만 적용: reionize/uncharge 금지 ---
def _normalize_without_reionize_uncharge(mol: Chem.Mol) -> Chem.Mol | None:
    if mol is None:
        return None
    try:
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
    except Exception:
        pass
    return mol


# --- (신규) 토토머 전체 전개 ---
def enumerate_tautomers_all(mol: Chem.Mol, dedup: bool = True, max_count: Optional[int] = None) -> List[Chem.Mol]:
    """
    RDKit의 TautomerEnumerator로 가능한 토토머를 전개.
    - dedup=True: canonical SMILES로 중복 제거
    - max_count: 너무 많을 때 상한(옵션)
    """
    te = rdMolStandardize.TautomerEnumerator()
    tauto_set = te.Enumerate(mol)
    tautomers = list(tauto_set)

    if dedup:
        seen = set()
        uniq = []
        for m in tautomers:
            s = Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)
            if s not in seen:
                seen.add(s)
                uniq.append(m)
        tautomers = uniq

    if max_count is not None and len(tautomers) > max_count:
        tautomers = tautomers[:max_count]

    return tautomers


# --- 파서: sanitize=True, FixedH/Normalize/고립H만 수동 ---
def mol_from_text(text: str) -> Optional[Chem.Mol]:
    """
    SMILES 또는 InChI를 파싱(sanitize=True).
    - Normalize만 수동 적용(표준화 규칙 기반 정규화)
    - InChI(+/FixedH 여부)는 여기선 단순 파싱만; 토토머 전개는 아래 wrapper에서 처리
    - 고립 수소 제거(옵션)
    """
    if not isinstance(text, str) or not text.strip():
        print(f"[MolFromText] invalid: {text!r}")
        return None

    mol = None
    try:
        if text.startswith("InChI="):
            mol = Chem.MolFromInchi(text, sanitize=True, treatWarningAsError=False)
        else:
            mol = Chem.MolFromSmiles(text, sanitize=True)
    except Exception as e:
        print(f"[MolFromText] parse error: {e}")
        mol = None

    if mol is None:
        # InChI가 Mobile-H 문제면 /FixedH로 한 번 더 시도
        if text.startswith("InChI="):
            try:
                mol = Chem.MolFromInchi(text, options="/FixedH", sanitize=True, treatWarningAsError=False)
            except Exception as e:
                print(f"[MolFromText] parse (/FixedH) error: {e}")
                mol = None

    if mol is None:
        return None

    # 고립 H 정리(옵션)
    try:
        mol = _drop_isolated_hydrogens(mol)
    except Exception:
        pass

    # Normalize만
    mol = _normalize_without_reionize_uncharge(mol)

    # 부분전하(옵션)
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    return mol


# === 토토머-aware 그래프 변환 래퍼 ===
def mols_from_text_with_tautomers(text: str) -> List[Chem.Mol]:
    """
    - InChI이고 '/FixedH'가 명시된 경우: 해당 골격에서 가능한 토토머를 모두 반환
    - 그 외(일반 SMILES 또는 FixedH 미지정 InChI)는 단일 Mol만 반환
    """
    mol = mol_from_text(text)
    if mol is None:
        return []

    if text.startswith("InChI=") and ("/FixedH" in text):
        try:
            return enumerate_tautomers_all(mol, dedup=True)
        except Exception:
            return [mol]
    else:
        return [mol]


# --- 기존 그래프 변환기들을 재사용하기 위해 어댑터 추가 ---
def smiles_or_inchi_to_graphs(
    text: str,
    multi_hop_max_dist: int,
    *,
    ATOM_FEATURES_VOCAB: Dict[str, Any],
    float_feature_keys: List[str],
    BOND_FEATURES_VOCAB: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """
    입력 1개 → 그래프 '여러 개' (토토머 전개 대응)
    """
    mols = mols_from_text_with_tautomers(text)
    graphs = []
    for m in mols:
        g = mol_to_graph_customized(
            m,
            multi_hop_max_dist=multi_hop_max_dist,
            ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
            float_feature_keys=float_feature_keys,
            BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
        )
        if g is not None:
            graphs.append(g)
    return graphs


def smiles_or_inchi_to_graphs_with_global(
    text: str,
    global_cat_idx: List[int],
    global_cont_val: List[float],
    *,
    ATOM_FEATURES_VOCAB: Dict[str, Any],
    float_feature_keys: List[str],
    BOND_FEATURES_VOCAB: Dict[str, List[Any]],
    GLOBAL_FEATURE_VOCABS: List[List[Any]],
    multi_hop_max_dist: int,
) -> List[Dict[str, Any]]:
    """
    입력 1개 → (글로벌 노드 포함) 그래프 '여러 개'
    """
    mols = mols_from_text_with_tautomers(text)
    graphs = []
    for m in mols:
        g = mol_to_graph_with_global(
            m,
            global_cat_idx,
            global_cont_val,
            ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
            float_feature_keys=float_feature_keys,
            BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
            multi_hop_max_dist=multi_hop_max_dist,
            GLOBAL_FEATURE_VOCABS=GLOBAL_FEATURE_VOCABS,
        )
        if g is not None:
            graphs.append(g)
    return graphs


def _get_feature_index(value: Any, vocab: List[Any]) -> int:
    """
    value가 vocab에 있으면 그 인덱스, 없으면 0(UNK) 반환.
    RDKit enum 값도 그대로 비교.
    """
    try:
        return vocab.index(value)
    except ValueError:
        return 0


# --- (추가) 한 소스에서 모든 타겟으로 가는 최단 경로 부모 인덱스 복원 ---
def _bfs_parents(adj: np.ndarray, src: int) -> Tuple[np.ndarray, np.ndarray]:
    n = adj.shape[0]
    prev = -np.ones(n, dtype=np.int32)
    dist = np.full(n, -1, dtype=np.int32)
    q = [src]
    dist[src] = 0
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in np.where(adj[u])[0]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)
    return prev, dist


def _reconstruct_path(prev: np.ndarray, src: int, dst: int) -> List[int]:
    if src == dst or prev[dst] == -1:
        return []  # path 없거나 자기자신
    path = [dst]
    cur = dst
    while cur != src and cur != -1:
        cur = prev[cur]
        path.append(cur)
    if cur != src:
        return []  # 연결 안됨
    path.reverse()
    return path  # [src, ..., dst]


def _compute_shortest_paths(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    dist = np.full((n, n), -1, dtype=int)
    np.fill_diagonal(dist, 0)
    for i in range(n):
        q = [(i, 0)]
        vis = {i}
        head = 0
        while head < len(q):
            u, d = q[head]
            head += 1
            dist[i, u] = d
            for v in np.where(adj[u])[0]:
                if v not in vis:
                    vis.add(v)
                    q.append((v, d + 1))
    return dist


# === NEW: feature-wise index builder for atoms ===
def _atom_feature_indices(atom: Chem.Atom, ATOM_FEATURES_VOCAB: dict) -> List[int]:
    """
    각 원자에 대해 ATOM_FEATURES_VOCAB 순서대로 정수 인덱스 리스트를 반환.
    (연속형은 여기서 처리하지 않음)
    """
    idxs: List[int] = []
    for key, vocab_or_type in ATOM_FEATURES_VOCAB.items():
        if not isinstance(vocab_or_type, list):
            continue  # 연속형은 건너뜀
        # 카테고리 값 추출
        if key == 'atomic_num':
            prop = atom.GetAtomicNum()
        elif key == 'formal_charge':
            prop = atom.GetFormalCharge()
        elif key == 'hybridization':
            prop = atom.GetHybridization()
        elif key == 'is_aromatic':
            prop = int(atom.GetIsAromatic())
        elif key == 'total_num_hs':
            prop = atom.GetTotalNumHs()
        elif key == 'explicit_valence':
            prop = atom.GetExplicitValence()
        elif key == 'total_bonds':
            prop = atom.GetTotalDegree()
        else:
            # 필요시 추가
            continue
        idxs.append(_get_feature_index(prop, vocab_or_type))  # 없으면 0
    return idxs


def mol_to_graph_customized(
    mol: Chem.Mol,
    multi_hop_max_dist: int,
    ATOM_FEATURES_VOCAB: Dict[str, Any],
    float_feature_keys: List[str],
    BOND_FEATURES_VOCAB: Dict[str, List[Any]],
) -> Dict[str, Any] | None:
    """RDKit Mol → Graphormer 입력 그래프(dict)."""
    if mol is None:
        return None

    num_nodes = mol.GetNumAtoms()
    adj = np.zeros((num_nodes, num_nodes), dtype=bool)

    # --- node features ---
    # (1) 카테고리: "정수 인덱스" 행렬 x_cat ∈ [N, F_cat]
    per_atom_indices: List[List[int]] = []
    for atom in mol.GetAtoms():
        per_atom_indices.append(_atom_feature_indices(atom, ATOM_FEATURES_VOCAB))
    x_cat = np.array(per_atom_indices, dtype=np.int64) if per_atom_indices else np.zeros((num_nodes, 0), dtype=np.int64)

    # (2) 연속형: x_cont ∈ [N, F_cont]
    cont_cols: List[np.ndarray] = []
    for key in (float_feature_keys or []):
        if key == 'atomic_mass':
            vals = [a.GetMass() for a in mol.GetAtoms()]
        elif key == 'partial_charge':
            vals = []
            for a in mol.GetAtoms():
                try:
                    charge = float(a.GetProp('_GasteigerCharge'))
                except Exception:
                    charge = 0.0
                vals.append(charge)
        else:
            # 알 수 없는 연속형 키: 0.0으로 채움
            vals = [0.0 for _ in mol.GetAtoms()]
        cont_cols.append(np.asarray(vals, dtype=np.float32))
    if cont_cols:
        x_cont = np.stack(cont_cols, axis=-1)
    else:
        x_cont = np.zeros((num_nodes, 0), dtype=np.float32)

    # --- edge features ---
    attn_edge_type: Dict[str, np.ndarray] = {
        k: np.zeros((num_nodes, num_nodes, len(vocab)), dtype=np.int64)
        for k, vocab in BOND_FEATURES_VOCAB.items()
    }
    edge_indices: List[List[int]] = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = True
        edge_indices.extend([[i, j], [j, i]])

        for key, vocab in BOND_FEATURES_VOCAB.items():
            if key == 'bond_type':
                prop = bond.GetBondType()
            elif key == 'stereo':
                prop = bond.GetStereo()
            elif key == 'is_conjugated':
                prop = int(bond.GetIsConjugated())
            elif key == 'is_in_ring':
                prop = int(bond.IsInRing())
            else:
                continue
            idx = _get_feature_index(prop, vocab)
            attn_edge_type[key][i, j, idx] = 1
            attn_edge_type[key][j, i, idx] = 1

    spatial_pos = _compute_shortest_paths(adj)

    # multi-hop edge_input
    edge_input: Dict[str, np.ndarray] = {
        key: np.zeros((num_nodes, num_nodes, multi_hop_max_dist, len(vocab)), dtype=np.int64)
        for key, vocab in BOND_FEATURES_VOCAB.items()
    }

    # BFS 기반 경로 복원으로 hop별 edge 타입 채우기
    for i in range(num_nodes):
        prev, dist_vec = _bfs_parents(adj, i)
        for j in range(num_nodes):
            d = dist_vec[j]
            if 1 <= d <= multi_hop_max_dist:
                path = _reconstruct_path(prev, i, j)
                if len(path) != d + 1:
                    continue
                for h in range(d):  # h = 0 .. d-1  (d==max_dist이면 마지막 hop까지 채워짐)
                    u, v = path[h], path[h + 1]
                    for key, vocab in BOND_FEATURES_VOCAB.items():
                        edge_input[key][i, j, h, :] = attn_edge_type[key][u, v, :]

    edge_index = np.array(edge_indices, dtype=int).T if edge_indices else np.empty((2, 0), dtype=int)

    return {
        'x_cat': x_cat,          # ✅ 이름은 유지하지만 "정수 인덱스" [N, F_cat]
        'x_cont': x_cont,        # float32 [N, F_cont]
        'adj': adj,              # bool [N, N]
        'edge_index': edge_index,  # int [2, E]
        'attn_edge_type': attn_edge_type,  # dict → [N,N,D_k]
        'spatial_pos': spatial_pos,        # int [N,N], 0=자기자신
        'edge_input': edge_input,          # dict → [N,N,H,D_k]
        'num_nodes': num_nodes,
    }

# 필요: import numpy as np, from typing import List, from rdkit import Chem

def mol_to_graph_with_global(
    mol,
    global_cat_idx: List[int],
    global_cont_val: List[float],
    *,
    ATOM_FEATURES_VOCAB,
    float_feature_keys,
    BOND_FEATURES_VOCAB,
    GLOBAL_FEATURE_VOCABS,
    multi_hop_max_dist: int,
):
    base = mol_to_graph_customized(
        mol,
        multi_hop_max_dist=multi_hop_max_dist,
        ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
        float_feature_keys=float_feature_keys,
        BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
    )
    if base is None:
        return None

    n = base["num_nodes"]
    F_atom = base["x_cat"].shape[1]
    F_cont = base["x_cont"].shape[1] if base["x_cont"].ndim == 2 else 0
    F_glob = len(global_cat_idx)

    # ---------- (A) x_cat 확장 ----------
    x_cat = np.zeros((n + 1, F_atom + F_glob), dtype=np.int64)
    if F_atom > 0:
        x_cat[:n, :F_atom] = base["x_cat"]
    gidx = np.asarray(global_cat_idx, dtype=np.int64)
    x_cat[:n, F_atom:] = gidx[None, :].repeat(n, axis=0)
    x_cat[n, :F_atom] = 0
    x_cat[n, F_atom:] = gidx

    # ---------- (B) x_cont 확장 ----------
    x_cont = np.zeros((n + 1, F_cont), dtype=np.float32)
    if F_cont > 0:
        x_cont[:n, :] = base["x_cont"]

    # ---------- (C) adj / edge_index 확장 ----------
    adj = np.zeros((n + 1, n + 1), dtype=bool)
    adj[:n, :n] = base["adj"]
    adj[n, :n] = True
    adj[:n, n] = True

    edge_index_list = base["edge_index"].T.tolist() if base["edge_index"].size > 0 else []
    extra_edges = [[i, n] for i in range(n)] + [[n, i] for i in range(n)]
    edge_index = (np.array(edge_index_list + extra_edges, dtype=int).T
                  if edge_index_list else np.array(extra_edges, dtype=int).T)

    # ---------- (D) attn_edge_type 확장 ----------
    attn_edge_type = {}
    for key, orig in base["attn_edge_type"].items():
        Dk = orig.shape[-1]
        t = np.zeros((n + 1, n + 1, Dk), dtype=np.int64)
        t[:n, :n, :] = orig  # AA 그대로 복사
        attn_edge_type[key] = t
    # ★ GA의 화학 채널(4그룹) 기본값 세팅은 하지 않는다 → 전부 0 유지

    # --- is_global: 1채널(YES) + pad 1채널로 총합 16 유지 ---
    is_gl = np.zeros((n + 1, n + 1, 1), dtype=np.int64)  # YES만 1채널
    is_gl[n, :n, 0] = 1
    is_gl[:n, n, 0] = 1
    attn_edge_type["is_global"] = is_gl

    # ---------- (E) edge_input 확장 ----------
    edge_input = {}
    for key, orig in base["edge_input"].items():
        K, Dk = orig.shape[2], orig.shape[3]
        e = np.zeros((n + 1, n + 1, K, Dk), dtype=np.int64)
        e[:n, :n, :, :] = orig  # AA 그대로 복사
        edge_input[key] = e
    # ★ GA의 화학 채널 기본값(E-1) 설정은 하지 않는다 → 전부 0 유지

    # --- is_global (edge_input): 1채널(YES), GA는 hop=0만 1 + pad 1채널 ---
    # K 얻기 (edge_input이 항상 존재하긴 하지만, 안전하게 처리)
    if len(edge_input) > 0:
        some_key = next(iter(edge_input))
        K = edge_input[some_key].shape[2]
    else:
        K = multi_hop_max_dist

    is_gl_ei = np.zeros((n + 1, n + 1, K, 1), dtype=np.int64)
    is_gl_ei[n, :n, 0, 0] = 1
    is_gl_ei[:n, n, 0, 0] = 1
    edge_input["is_global"] = is_gl_ei

    # ---------- (F) spatial_pos ----------
    spatial_pos = _compute_shortest_paths(adj)
    spatial_pos[spatial_pos == 0] = 1  # self-loop 1

    return {
        "x_cat": x_cat,
        "x_cont": x_cont,
        "adj": adj,
        "edge_index": edge_index,
        "attn_edge_type": attn_edge_type,
        "spatial_pos": spatial_pos,
        "edge_input": edge_input,
        "num_nodes": n + 1,
        "global_features_cat": gidx,
        "global_features_cont": np.asarray(global_cont_val, dtype=np.float32),
    }

def graph_from_text(
    text: str,
    multi_hop_max_dist: int,
    ATOM_FEATURES_VOCAB: Dict[str, Any],
    float_feature_keys: List[str],
    BOND_FEATURES_VOCAB: Dict[str, List[Any]],
) -> Dict[str, Any] | None:
    mol = mol_from_text(text)
    return mol_to_graph_customized(mol, multi_hop_max_dist, ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB)

def smiles_or_inchi_to_graph(
    text: str,
    multi_hop_max_dist: int,
    *,
    ATOM_FEATURES_VOCAB: Dict[str, Any],
    float_feature_keys: List[str],
    BOND_FEATURES_VOCAB: Dict[str, List[Any]],
) -> Dict[str, Any] | None:
    """
    SMILES 또는 InChI 문자열을 자동 인식하여 Graphormer 그래프로 변환.
    DataLoader에서 기존 smiles2graph_customized 대체 용도로 사용.
    """
    mol = mol_from_text(text)
    return mol_to_graph_customized(
        mol,
        multi_hop_max_dist=multi_hop_max_dist,
        ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
        float_feature_keys=float_feature_keys,
        BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
    )


def smiles_or_inchi_to_graph_with_global(
    text: str,
    global_cat_idx: List[int],
    global_cont_val: List[float],
    *,
    ATOM_FEATURES_VOCAB: Dict[str, Any],
    float_feature_keys: List[str],
    BOND_FEATURES_VOCAB: Dict[str, List[Any]],
    GLOBAL_FEATURE_VOCABS: List[List[Any]],
    multi_hop_max_dist: int,
) -> Dict[str, Any] | None:
    """
    SMILES/InChI → 그래프(+글로벌 노드).
    DataLoader에서 mode == 'cls_global_data'일 때 사용.
    """
    mol = mol_from_text(text)
    return mol_to_graph_with_global(
        mol,
        global_cat_idx,
        global_cont_val,
        ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
        float_feature_keys=float_feature_keys,
        BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
        multi_hop_max_dist=multi_hop_max_dist,
        GLOBAL_FEATURE_VOCABS=GLOBAL_FEATURE_VOCABS,
    )


__all__ = [
    "mol_from_text",
    "mols_from_text_with_tautomers",
    "mol_to_graph_customized",
    "mol_to_graph_with_global",
    "graph_from_text",
    "smiles_or_inchi_to_graph",
    "smiles_or_inchi_to_graphs",
    "smiles_or_inchi_to_graph_with_global",
    "smiles_or_inchi_to_graphs_with_global",
]
