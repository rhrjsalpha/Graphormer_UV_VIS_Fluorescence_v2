import sys
import argparse
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from tqdm import tqdm
from GP.data_prepare.Pre_Defined_Vocab_Generator import build_vocabs_from_df
from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset, collate_fn
from typing import Optional

torch.set_printoptions(edgeitems=1000000, threshold=10_000_000, linewidth=1000)
np.set_printoptions(edgeitems=1000000, threshold=10_000_000, linewidth=1000, suppress=True)

#### 실행 코드 ####
def show_batch_shapes(batch, title="Batch"):
    print(f"  ▶ {title}")
    for k, v in batch.items():
        if isinstance(v, dict):
            print(f"    {k:16s} (dict of tensors)")
            for sub_k, sub_v in v.items():
                if torch.is_tensor(sub_v):
                    print(f"      {sub_k:14s} {tuple(sub_v.shape)}")
        elif torch.is_tensor(v):
            print(f"    {k:16s} {tuple(v.shape)}")


def build_parser():
    p = argparse.ArgumentParser("UnifiedSMILESDataset pipeline")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--mode", type=str, choices=["cls", "cls_global_data", "cls_global_model"], default="cls")
    p.add_argument("--target_type", choices=["default", "ex_prob", "nm_distribution"], default="default")
    p.add_argument("--ex_norm", choices=["ex_min_max", "ex_std", "none"], default="none")
    p.add_argument("--prob_norm", choices=["prob_min_max", "prob_std", "none"], default="none")
    p.add_argument("--nm_dist_mode", choices=["hist", "gauss"], default="hist")
    p.add_argument("--nm_gauss_sigma", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_nodes", type=int, default=128)
    p.add_argument("--multi_hop_max_dist", type=int, default=5)
    p.add_argument("--x_cat_mode", choices=["index", "onehot", "both"], default="both")
    p.add_argument("--global_cat_mode", choices=["index","onehot","both"], default="both")
    return p

def show_all_nodes_onehot(batch, b: int = 0, max_nodes: int = None,
                          use_raw: bool = False, group_filter=None,
                          skip_padding: bool = True):
    """
    배치 b의 모든 노드에 대해 x_cat_onehot을 그룹 단위로 출력.
    - use_raw=True 이면 compact 전(before-merge) 버전을 사용
    - group_filter=['atomic_num','mh_Solvent', ...] 로 특정 그룹만 출력 가능
    - skip_padding=True 이면 완전 0인 패딩 노드는 스킵
    """
    # --- 어떤 버전을 볼지 선택 (compact vs raw) ---
    Xkey   = "x_cat_onehot_raw" if use_raw else "x_cat_onehot"
    Mkey   = "x_cat_onehot_meta_raw" if use_raw else "x_cat_onehot_meta"
    Xoh    = batch[Xkey][b]                # [N, sumC]
    meta   = batch[Mkey]
    names  = list(meta["group_names"])
    sizes  = [int(x) for x in (meta["sizes"].tolist() if torch.is_tensor(meta["sizes"]) else meta["sizes"])]
    offs   = [int(x) for x in (meta["offsets"].tolist() if torch.is_tensor(meta["offsets"]) else meta["offsets"])]

    # --- 출력할 그룹 범위(필터링) ---
    if group_filter:
        gi_list = [i for i, n in enumerate(names) if n in set(group_filter)]
    else:
        gi_list = list(range(len(names)))

    N = Xoh.shape[0]
    if max_nodes is not None:
        N = min(N, max_nodes)

    # (선택) 글로벌 노드 감지용 보조: 원자 그룹 끝 오프셋 잡기
    try:
        atom_end = offs[names.index("total_bonds")] + sizes[names.index("total_bonds")]
    except ValueError:
        atom_end = offs[0]  # 안전장치

    for n in range(N):
        v = Xoh[n]  # [sumC]
        if skip_padding and torch.all(v == 0):
            continue

        # 전역 노드 힌트(원자 그룹 합이 0이면 전역 노드로 추정)
        is_global_like = (v[:atom_end].sum() == 0)

        print(f"\n[node {n}] sum={float(v.sum())}  {'<GLOBAL?>' if is_global_like else ''}")
        for gi in gi_list:
            s, e = offs[gi], offs[gi] + sizes[gi]
            gvec = v[s:e].to(torch.int8)
            active = (gvec > 0).nonzero(as_tuple=True)[0].tolist()
            print(f"  group{gi:02d} {names[gi]:<20s} [{s}:{e}] -> {gvec.tolist()}   active={active}")

def peek_one_batch(batch: dict, head_n: int = None, node_n: int = None, ds=None):
    """
    한 배치(batch)에서 첫 샘플 전체를 프린트.
    head_n, node_n을 None으로 주면 전체 출력
    """
    import torch
    b0 = 0
    keys = batch.keys()

    bond_groups = [("bond_type", 4), ("stereo", 6), ("is_conjugated", 2), ("is_in_ring", 2)]
    onehot_total = sum(s for _, s in bond_groups)

    def _head(x, n=None):
        return x if n is None else x[:n]
    def _shape(x): return tuple(x.shape)

    print("\n[Peek one batch — sample 0 only]")
    print("-" * 60)

    # -------- targets / masks --------
    if "targets" in keys and torch.is_tensor(batch["targets"]):
        t = batch["targets"][b0]
        print(f"targets[b0] shape={_shape(t)}")
        print(t)   # 전체 출력
    if "masks" in keys and torch.is_tensor(batch["masks"]):
        m = batch["masks"][b0]
        print(f"masks[b0] shape={_shape(m)} | valid_count={(m>0).sum().item()}")
        print(m)

    # -------- node categorical / continuous --------
    if "x_cat" in keys:
        x_cat = batch["x_cat"][b0]
        print(f"x_cat[b0] shape={_shape(x_cat)}")
        print("맨 마지막의 n개는 0으로 나올수 있음, 가장 큰 그래프 기준으로 한 padding임")
        print("맨 마지막의 n개 직전은 global node 로써 9번째 열이 global_cat_dim")
        print("첫 7개 컬럼 feature : atomic_num, formal_charge, hybridization, is_aromatic, total_num_hs, explicit_valence, total_bonds")
        print("나머지 n개 글로벌 노드 feature : 여기서는 Solvent, pH_label, type(AB/Em) 등")
        ATOM_FEATURES_VOCAB = {
            'atomic_num': list(range(1, 119)),
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

        print(x_cat)   # 전체 출력
    if "x_cont" in keys:
        x_cont = batch["x_cont"][b0]
        print(f"x_cont[b0] shape={_shape(x_cont)}")
        print(x_cont)

    # -------- degree / adj / spatial_pos / attn_bias --------
    if "in_degree" in keys:
        indeg = batch["in_degree"][b0]
        print(f"in_degree[b0] shape={_shape(indeg)}")
        print(indeg)
    if "out_degree" in keys:
        outdeg = batch["out_degree"][b0]
        print(f"out_degree[b0] shape={_shape(outdeg)}")
        print(outdeg)
    if "adj" in keys:
        adj = batch["adj"][b0]
        print(f"adj[b0] shape={_shape(adj)}")
        print(adj)
    if "spatial_pos" in keys:
        sp = batch["spatial_pos"][b0]
        print(f"spatial_pos[b0] shape={_shape(sp)}")
        print(sp)
    if "attn_bias" in keys:
        ab = batch["attn_bias"]
        ab0 = ab[b0] if ab.ndim == 3 else ab[b0, 0]
        print(f"attn_bias[b0] shape={_shape(ab0)}")
        print(ab0)

    # -------- attn_edge_type / edge_input --------
    if "attn_edge_type" in keys:
        aet = batch["attn_edge_type"][b0]
        print(f"attn_edge_type[b0] shape={_shape(aet)}")
        print("전체 onehot = {bond_type(4), stereo(6), is_conjugated(2), is_in_ring(2), is_global(1)}, 각 부분에서 00001, 10000, 이런식, 1이 각 feature 별 한군데 존재")
        print("기존 분자 그래프 사이의 정보 = {bond_type(4), stereo(6), is_conjugated(2), is_in_ring(2)}, 4로 나옴 is_global은 0으로 채워지고 나머지에는 1인 부분이 1개씩 존재 : one hot에서 1인 부분을 셈")
        print("Globa Node 는 1로 나옴 : is_global =1 이고 나머지는 0으로 채워짐, one hot에서 1인 부분을 셈")
        print(aet.sum(dim=-1))  # 그룹 수 합
    if "edge_input" in keys:
        ein = batch["edge_input"][b0]
        print(f"edge_input[b0] shape={_shape(ein)}")
        print(ein.sum(dim=-1))  # hop별 그룹 수 합

    # -------- Global features --------
    if "global_features_cat" in keys and torch.is_tensor(batch["global_features_cat"]):
        gcat = batch["global_features_cat"][b0]
        print(f"global_features_cat[b0] shape={_shape(gcat)}")
        print(gcat)
    if "global_features_cont" in keys and torch.is_tensor(batch["global_features_cont"]):
        gcont = batch["global_features_cont"][b0]
        print(f"global_features_cont[b0] shape={_shape(gcont)}")
        print(gcont)

    print("-" * 60)


    # -------- Global features --------
    if "global_features_cat" in keys and torch.is_tensor(batch["global_features_cat"]):
        gcat = batch["global_features_cat"][b0]  # [F_global]
        nz = (gcat.view(-1) > 0).nonzero(as_tuple=True)[0]
        print(f"global_features_cat[b0] shape={_shape(gcat)} | nnz={nz.numel()}  (multi-hot; Solvent 혼합물 등)")
        print("  nonzero idx (first 32):", _head(nz, 32))
    else:
        print("(info) global_features_cat not found in batch")

    if "global_features_cont" in keys and torch.is_tensor(batch["global_features_cont"]):
        gcont = batch["global_features_cont"][b0]
        print(f"global_features_cont[b0] shape={_shape(gcont)} (연속형 글로벌 특성)")
        print(_head(gcont))
    else:
        print("(info) global_features_cont not found in batch")

    print("-" * 60)

def show_actual_xcat_onehot(batch, b=0, n=0):
    """
    x_cat_onehot의 실제 0/1 값을 '그룹별 슬라이스'와 '전체 벡터' 둘 다 보여줍니다.
    b: 배치 인덱스, n: 노드 인덱스
    """
    x_oh = batch["x_cat_onehot"][b, n]                 # [sumC]
    meta = batch["x_cat_onehot_meta"]
    names   = meta["group_names"]
    sizes   = meta["sizes"]
    offsets = meta["offsets"]

    print(f"\n[x_cat_onehot — sample b={b}, node={n}] shape={tuple(x_oh.shape)}  sum={float(x_oh.sum())}")
    # 전체 벡터(0/1) 그대로
    print("concat onehot:", x_oh.to(torch.int8).tolist())

    # 그룹별로 잘라서 보기
    for gi, (name, off, sz) in enumerate(zip(names, offsets, sizes)):
        s, e = int(off), int(off + sz)
        vec = x_oh[s:e].to(torch.int8)                 # [sz]
        # 켜진 위치도 같이 표기
        active = (vec > 0).nonzero(as_tuple=True)[0].tolist()
        print(f"  group{gi} {name:<16s} [{s}:{e}] -> {vec.tolist()}   active={active}")

def _find_global_idx_from_xcat(batch, b=0):
    """x_cat_onehot 메타정보로 '원자그룹 합=0'인 노드를 후보로 잡고,
       그 중 연결이 가장 많은 노드를 글로벌 노드로 추정."""
    import torch
    xoh  = batch["x_cat_onehot"][b]  # [N, sumC]
    meta = batch["x_cat_onehot_meta"]
    names   = list(meta["group_names"])
    sizes   = [int(x) for x in (meta["sizes"].tolist() if torch.is_tensor(meta["sizes"]) else meta["sizes"])]
    offsets = [int(x) for x in (meta["offsets"].tolist() if torch.is_tensor(meta["offsets"]) else meta["offsets"])]

    # 원자 특성 구간의 끝(= 'total_bonds' 그룹 끝) 계산
    gi_tb = names.index("total_bonds")
    atom_end = offsets[gi_tb] + sizes[gi_tb]

    # 글로벌 후보: 원자 특성(앞부분) 합계가 0인 노드
    atom_sums = xoh[:, :atom_end].sum(dim=1)
    cand = (atom_sums == 0).nonzero(as_tuple=True)[0]

    # 패딩 제외 겸, 가장 연결 많은 것 선택
    s = batch["attn_edge_type"][b].sum(dim=-1)  # [N,N]
    best, best_deg = None, -1
    for idx in cand.tolist():
        deg = int(s[idx].sum().item() + s[:, idx].sum().item())
        if deg > best_deg:
            best, best_deg = idx, deg
    return int(best) if best is not None else None


def print_one_five_edges(batch, b=0):
    """
    변경 후 규칙:
      - Global ↔ Atom: sum(attn_edge_type) == 1 (is_global만 1)
      - Atom ↔ Atom : sum(attn_edge_type) == 4 (화학 4그룹에서 각 1)
    """
    import torch

    aet = batch["attn_edge_type"][b]  # [N,N,16] (4+6+2+2+1+1)
    ein = batch["edge_input"][b]      # [N,N,H,16]
    s   = aet.sum(dim=-1)             # [N,N]

    N = s.shape[0]

    # 글로벌 노드 찾기 (실패 시 '마지막 노드' 가정)
    def _find_global_idx_from_xcat_safe(batch, b=0):
        try:
            gi = _find_global_idx_from_xcat(batch, b=b)
            if gi is None:
                raise ValueError
            return gi
        except Exception:
            # 보통 마지막이 글로벌 노드
            return int((batch["attn_edge_type"][b].shape[0]) - 1)

    g = _find_global_idx_from_xcat_safe(batch, b=b)

    # 채널 슬라이스(우리 생성 순서: bond_type, stereo, is_conj, is_ring, is_global, pad)
    sl = {
        "bond_type"     : slice(0, 4),
        "stereo"        : slice(4, 10),
        "is_conjugated" : slice(10, 12),
        "is_in_ring"    : slice(12, 14),
        "is_global"     : slice(14, 15),  # YES = 1 No = 0, 1채널
    }

    EXPECT_GA_SUM = 1
    EXPECT_AA_SUM = 4

    def _first_pair(mask_func, expect_sum):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if mask_func(i, j) and s[i, j].item() == expect_sum:
                    return i, j
        return None, None

    # A) Global ↔ Atom
    gi, gj = _first_pair(lambda i, j: (i == g) ^ (j == g), EXPECT_GA_SUM)
    # B) Atom ↔ Atom
    ai, aj = _first_pair(lambda i, j: (i != g) and (j != g), EXPECT_AA_SUM)

    def _print_edge(i, j, tag):
        if i is None:
            print(f"[WARN] {tag}에서 합={EXPECT_GA_SUM if 'Global' in tag else EXPECT_AA_SUM}인 에지를 찾지 못했습니다.")
            return
        v = aet[i, j]         # [16]
        hops = ein[i, j]      # [H,16]
        active_hops = torch.nonzero(hops.sum(dim=-1) > 0, as_tuple=True)[0].tolist()

        print(f"\n[{tag}] edge (i={i}, j={j})  |  sum(attn_edge_type)={int(s[i,j].item())}")
        print("  attn_edge_type onehot (len=16):", v.to(torch.int8).tolist())
        for k, slc in sl.items():
            subv = v[slc].to(torch.int8).tolist()
            print(f"   - {k:<14s}: {subv}")

        if len(active_hops) == 0:
            print("  edge_input: 활성 hop 없음")
        else:
            print("  edge_input 활성 hop (1-base):", [h+1 for h in active_hops])
            for h in active_hops:
                subv = hops[h].to(torch.int8).tolist()
                print(f"   • hop {h+1}: {subv}")

    _print_edge(gi, gj, "Global ↔ Atom")
    _print_edge(ai, aj, "Atom ↔ Atom")

# === index 모드: 노드별 카테고리 인덱스 출력 ===
def show_all_nodes_index(batch, ds, b: int = 0, max_nodes: int = None,
                         group_filter=None, skip_padding: bool = True):
    """
    x_cat(one-hot 아님) 기반으로 한 분자의 모든 노드에서
    그룹별 카테고리 '인덱스'를 그대로 출력한다.
    - ds.x_cat_group_names / ds.x_cat_onehot_sizes 메타를 사용
    - 글로벌 노드는 '원자 그룹'들의 인덱스가 모두 0인 행으로 추정
    """
    import torch

    if "x_cat" not in batch:
        print("[show_all_nodes_index] batch에 x_cat이 없습니다.")
        return

    X = batch["x_cat"][b]                      # [N, C_groups]
    N, C = X.shape
    if max_nodes is not None:
        N = min(N, max_nodes)

    names = list(getattr(ds, "x_cat_group_names", [f"g{i}" for i in range(C)]))
    sizes = list(getattr(ds, "x_cat_onehot_sizes", [0]*C))  # 각 그룹의 |V| (정보용)

    # 원자 그룹(앞 7개) 위치 찾아두기
    ATOM_GROUPS = ["atomic_num", "formal_charge", "hybridization",
                   "is_aromatic", "total_num_hs", "explicit_valence", "total_bonds"]
    atom_col_idx = [names.index(nm) for nm in ATOM_GROUPS if nm in names]

    # 필터링(원하면 특정 그룹만)
    if group_filter:
        allowed = set(group_filter)
        group_indices = [i for i, nm in enumerate(names) if nm in allowed]
    else:
        group_indices = list(range(C))

    for n in range(N):
        row = X[n]                              # [C]
        if skip_padding and torch.all(row == 0):
            # 완전 0인 패딩 노드 스킵
            continue

        is_global_like = all(int(row[i].item()) == 0 for i in atom_col_idx)
        print(f"\n[node {n}] {'<GLOBAL?>' if is_global_like else ''}")

        for gi in group_indices:
            nm  = names[gi]
            idx = int(row[gi].item())
            if nm.startswith("mh_Solvent::"):
                # 멀티핫 비트(0/1): 켜져 있으면 해당 용매 토큰이 포함되었다는 의미
                print(f"  {nm:<20s} -> {idx} ({'ON' if idx > 0 else 'off'})")
            else:
                # 일반 싱글 카테고리: 인덱스 그대로
                # sizes[gi]는 이 그룹의 총 클래스 개수(|V|)
                print(f"  {nm:<20s} idx={idx}  (|V|={sizes[gi]})")

def _find_global_idx_from_index(batch, ds, b=0):
    """
    index 모드에서 '원자 그룹 인덱스가 모두 0'인 노드를 후보로 잡고,
    그 중 연결(차수)이 가장 큰 노드를 글로벌 노드로 선택한다.
    실패 시 마지막 노드를 반환.
    """
    import torch

    X = batch["x_cat"][b]                      # [N, C]
    N, C = X.shape
    names = list(getattr(ds, "x_cat_group_names", [f"g{i}" for i in range(C)]))

    ATOM_GROUPS = ["atomic_num", "formal_charge", "hybridization",
                   "is_aromatic", "total_num_hs", "explicit_valence", "total_bonds"]
    atom_col_idx = [names.index(nm) for nm in ATOM_GROUPS if nm in names]

    # 후보: 원자 그룹이 전부 0인 노드
    cand = []
    for i in range(N):
        row = X[i]
        if all(int(row[j].item()) == 0 for j in atom_col_idx):
            cand.append(i)

    # 연결 수(양방향 합) 가장 큰 것 선택
    s = batch["attn_edge_type"][b].sum(dim=-1)  # [N, N]
    best, best_deg = None, -1
    for i in cand:
        deg = int(s[i].sum().item() + s[:, i].sum().item())
        if deg > best_deg:
            best, best_deg = i, deg

    if best is None:
        # 안전장치: 마지막 노드를 글로벌 취급
        best = N - 1
    return int(best)

def print_one_five_edges_index(batch, ds, b=0):
    """
    index 모드에서 엣지 2개 예시 출력:
      - Global ↔ Atom: sum(attn_edge_type)=1 (is_global 1채널만 켜짐)
      - Atom   ↔ Atom: sum(attn_edge_type)=4 (화학 4그룹에서 각 1개)
    """
    import torch

    aet = batch["attn_edge_type"][b]   # [N, N, 15]  (4+6+2+2+1)
    ein = batch["edge_input"][b]       # [N, N, H, 15]
    s   = aet.sum(dim=-1)              # [N, N]
    N   = s.shape[0]

    g = _find_global_idx_from_index(batch, ds, b=b)

    # 채널 슬라이스 (15채널 고정: bond4 + stereo6 + is_conjugated2 + is_in_ring2 + is_global1)
    sl = {
        "bond_type"     : slice(0, 4),
        "stereo"        : slice(4, 10),
        "is_conjugated" : slice(10, 12),
        "is_in_ring"    : slice(12, 14),
        "is_global"     : slice(14, 15),
    }
    EXPECT_GA_SUM = 1
    EXPECT_AA_SUM = 4

    def _first_pair(mask_func, expect_sum):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if mask_func(i, j) and s[i, j].item() == expect_sum:
                    return i, j
        return None, None

    gi, gj = _first_pair(lambda i, j: (i == g) ^ (j == g), EXPECT_GA_SUM)
    ai, aj = _first_pair(lambda i, j: (i != g) and (j != g), EXPECT_AA_SUM)

    def _print_edge(i, j, tag):
        if i is None:
            print(f"[WARN] {tag}: 조건(sum={EXPECT_GA_SUM if 'Global' in tag else EXPECT_AA_SUM})에 맞는 에지를 못 찾음")
            return
        v    = aet[i, j]  # [15]
        hops = ein[i, j]  # [H, 15]
        active_hops = torch.nonzero(hops.sum(dim=-1) > 0, as_tuple=True)[0].tolist()

        print(f"\n[{tag}] edge (i={i}, j={j})  |  sum(attn_edge_type)={int(s[i,j].item())}")
        print("  attn_edge_type onehot (len=15):", v.to(torch.int8).tolist())
        for k, slc in sl.items():
            subv = v[slc].to(torch.int8).tolist()
            print(f"   - {k:<14s}: {subv}")

        if len(active_hops) == 0:
            print("  edge_input: 활성 hop 없음")
        else:
            print("  edge_input 활성 hop (1-base):", [h+1 for h in active_hops])
            for h in active_hops:
                subv = hops[h].to(torch.int8).tolist()
                print(f"   • hop {h+1}: {subv}")

    _print_edge(gi, gj, "Global ↔ Atom")
    _print_edge(ai, aj, "Atom ↔ Atom")

def debug_print_global_features(batch, ds, b: int = 0, mode: str = None):
    import torch
    mode = mode or getattr(ds, "mode", None)

    # ---- cls_global_model: 전역은 오직 batch["global_features_cat"]에만 존재 ----
    if mode == "cls_global_model":
        if "global_features_cat" not in batch or "global_features_cat_meta" not in batch:
            print("(info) cls_global_model: global_features_cat 없음 — collate_fn에서 합본 onehot을 리턴하는지 확인")
            return
        gvec  = batch["global_features_cat"][b]
        gmeta = batch["global_features_cat_meta"]
        def _as_list(x): return [int(v) for v in (x.view(-1).tolist() if torch.is_tensor(x) else list(x))]
        names, sizes, offs = list(gmeta["names"]), _as_list(gmeta["sizes"]), _as_list(gmeta["offsets"])

        print("\n== global_features_cat (onehot, by column) ==")
        for name, off, sz in zip(names, offs, sizes):
            sl = gvec[off:off+sz]
            idx1 = torch.nonzero(sl, as_tuple=True)[0].tolist()
            print(f"  {name:<12s} [{off}:{off+sz}] -> {sl.to(torch.int8).tolist()}   active={idx1}")
        return

    # ---- cls_global_data: 전역은 그래프에도, batch에도 존재 → 교차 검증 ----
    if mode == "cls_global_data":
        if "global_features_cat" not in batch or "global_features_cat_meta" not in batch:
            print("(info) cls_global_data: global_features_cat 없음 — collate_fn에서 합본 onehot을 리턴하는지 확인")
            return
        gvec  = batch["global_features_cat"][b]
        gmeta = batch["global_features_cat_meta"]
        def _as_list(x): return [int(v) for v in (x.view(-1).tolist() if torch.is_tensor(x) else list(x))]
        names, sizes, offs = list(gmeta["names"]), _as_list(gmeta["sizes"]), _as_list(gmeta["offsets"])

        print("\n== global_features_cat (onehot, by column) ==")
        active_by_col = {}
        for name, off, sz in zip(names, offs, sizes):
            sl = gvec[off:off+sz]
            idx1 = torch.nonzero(sl, as_tuple=True)[0].tolist()
            active_by_col[name] = idx1
            print(f"  {name:<12s} [{off}:{off+sz}] -> {sl.to(torch.int8).tolist()}   active={idx1}")

        # x_cat_onehot의 글로벌 노드 크로스체크
        Xoh   = batch["x_cat_onehot"][b]
        xmeta = batch["x_cat_onehot_meta"]
        xnames, xsizes, xoffs = list(xmeta["group_names"]), _as_list(xmeta["sizes"]), _as_list(xmeta["offsets"])
        gi_tb   = xnames.index("total_bonds")
        atom_end = xoffs[gi_tb] + xsizes[gi_tb]
        atom_sums = Xoh[:, :atom_end].sum(dim=1)
        cand = (atom_sums == 0).nonzero(as_tuple=True)[0]
        g = int(cand[0]) if cand.numel() > 0 else (Xoh.shape[0]-1)
        print(f"\n== global node index (b={b}) => {g}")

        Xg = Xoh[g]
        print("== cross-check: global node in x_cat_onehot ==")
        for name in names:
            if name in xnames:
                i = xnames.index(name); s, e = xoffs[i], xoffs[i]+xsizes[i]
                vec = Xg[s:e]
                idx1 = torch.nonzero(vec, as_tuple=True)[0].tolist()
                print(f"  {name:<12s} [{s}:{e}] -> {vec.to(torch.int8).tolist()}   active={idx1}   (match {active_by_col[name]})")
        return

    # ---- cls: 전역 피처 없음 → 스킵 + (선택) 그래프 내부에 전역이 섞이지 않았는지 확인 ----
    if mode == "cls":
        print("(info) cls mode: global features are not present (as expected).")
        # 원한다면 '그래프에 전역 안 섞였는지' 빠른 검사를 돌리세요:
        _assert_no_global_in_graph(batch, ds, b=b)
        return

    print(f"(info) unknown mode={mode}.")

def _assert_no_global_in_graph(batch, ds, b=0):
    import torch
    # 1) x_cat_onehot 그룹 이름에 mh_Solvent, pH_label, type 등이 없어야 함
    if "x_cat_onehot_meta" in batch:
        names = list(batch["x_cat_onehot_meta"]["group_names"])
        bad = [n for n in names if n.startswith("mh_Solvent") or n in getattr(ds, "global_single_cols", [])]
        if bad:
            print(f"[WARN] global-like groups found in node features (cls/cls_global_model should not have these): {bad}")
        else:
            print("[OK] no global groups in x_cat_onehot groups")

        # 2) 원자 그룹 합이 0인 '가짜 글로벌 노드'가 없어야 함(패딩 제외)
        Xoh   = batch["x_cat_onehot"][b]
        sizes = batch["x_cat_onehot_meta"]["sizes"]
        offs  = batch["x_cat_onehot_meta"]["offsets"]
        xnames= list(batch["x_cat_onehot_meta"]["group_names"])
        def _as_list(x): return [int(v) for v in (x.view(-1).tolist() if torch.is_tensor(x) else list(x))]
        sizes, offs = _as_list(sizes), _as_list(offs)
        gi_tb = xnames.index("total_bonds") if "total_bonds" in xnames else None
        if gi_tb is not None:
            atom_end = offs[gi_tb] + sizes[gi_tb]
            atom_sums = Xoh[:, :atom_end].sum(dim=1)
            # 패딩(완전 0) 제외
            pad_mask = (Xoh.sum(dim=1) == 0)
            global_like = ((atom_sums == 0) & (~pad_mask)).nonzero(as_tuple=True)[0].tolist()
            if global_like:
                print(f"[WARN] global-like nodes detected at indices {global_like}")
            else:
                print("[OK] no global-like nodes (good)")

def run_pipeline(args):
    ATOM_FEATURES_VOCAB = {
        'atomic_num': list(range(1, 119)),
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
    float_feature_keys = ['partial_charge', 'atomic_mass']

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

    # 1) CSV들로부터 vocab만 자동 생성 (데이터셋에는 CSV 경로 그대로 줄 것)
    df_1 = pd.read_csv(
        r"../../graphormer_data/final_split/ABS_stratified_test_plus.csv",
        low_memory=False
    )
    df_2 = pd.read_csv(
        r"../../graphormer_data/final_split/ABS_stratified_train_plus.csv",
        low_memory=False
    )
    df_all = pd.concat([df_1, df_2], axis=0, ignore_index=True)

    # NOTE: 데이터셋에서 raw CSV 값과 정확히 매칭되도록 plus_split_cols 비움
    _df_proc, nominal_feature_vocab, continuous_feature_names, global_cat_dim, global_cont_dim = build_vocabs_from_df(
        df_all,
        nominal_cols=getattr(args, "nominal_cols", None),               # 필요 시 지정
        continuous_cols=getattr(args, "continuous_feature_names", None),
        plus_split_cols=("Solvent",)  # <= 중요한 포인트! (셀 전체 문자열 기준 vocab)
    ) # nominal_feature_vocab에 기반해서 Unified Smiels Dataset 함수에서 one hot encoding이 수행됨
    print("global_cat_dim", global_cat_dim)
    continuous_feature_names = []

    # 2) 안전 체크
    if args.mode not in ["cls", "cls_global_data", "cls_global_model"]:
        raise ValueError("Invalid mode: choose from 'cls', 'cls_global_data', 'cls_global_model'")

    # 원하는 순서
    global_feature_order = ["pH_label", "type", "Solvent"]
    # 1) 딕셔너리: 컬럼명 → 해당 vocab 리스트
    GLOBAL_FEATURE_VOCABS_Dict = {name: nominal_feature_vocab[name] for name in global_feature_order}

    # 3) 데이터셋은 원본 CSV 경로 그대로 사용
    ds = UnifiedSMILESDataset(
        csv_file=args.train_file,                    # <= 원본 CSV 경로
        nominal_feature_vocab=nominal_feature_vocab, # <= 위에서 추출한 vocab만 주입
        continuous_feature_names=continuous_feature_names or [],
        global_cat_dim=global_cat_dim,
        global_cont_dim=global_cont_dim,
        ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
        float_feature_keys=float_feature_keys,
        BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
        GLOBAL_FEATURE_VOCABS_dict=GLOBAL_FEATURE_VOCABS_Dict,
        x_cat_mode=args.x_cat_mode,  # ★ 추가
        global_cat_mode=args.global_cat_mode,  # ★ 추가
        mol_col=getattr(args, "mol_col", "smiles"),
        mode=args.mode,
        max_nodes=args.max_nodes,
        multi_hop_max_dist=args.multi_hop_max_dist,
        target_type=args.target_type,
        ex_normalize=args.ex_norm,
        prob_normalize=args.prob_norm,
        nm_dist_mode=args.nm_dist_mode,
        nm_gauss_sigma=args.nm_gauss_sigma,
        intensity_normalize=getattr(args, "intensity_normalize", "min_max"),
        intensity_range=getattr(args, "intensity_range", (200, 800)),
        attn_bias_w=getattr(args, "attn_bias_w", 0.0),
    )

    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b, _ds=ds: collate_fn(b, _ds)
    )

    for i, batch in enumerate(tqdm(dl, desc="Loading batches")):
        print(f"\n===== Mode: {args.mode} | Batch {i + 1} =====")
        show_batch_shapes(batch, f"Batch {i + 1}")  # 기존 요약

        # 전역 특성 디버그 (b=0 샘플)
        debug_print_global_features(batch, ds, b=0)

        # ---- x_cat 원-핫(옵션) ----
        if "x_cat_onehot" in batch:
            show_all_nodes_onehot(batch, b=0, skip_padding=True)
            show_actual_xcat_onehot(batch, b=0, n=0)
            if i == 0:
                print_one_five_edges(batch, b=0)
            Xoh = batch["x_cat_onehot"]  # [B, N, ΣC]
            meta = batch.get("x_cat_onehot_meta", {})  # {"group_names", "sizes", "offsets"}

            # 텐서일 수도, 리스트일 수도 있으니 리스트로 정규화
            def _as_list(x):
                import torch
                if torch.is_tensor(x):
                    return [int(v) for v in x.view(-1).tolist()]
                return list(x)

            sizes = _as_list(meta.get("sizes", []))  # e.g. [35, 7, 5, ...]
            offsets = _as_list(meta.get("offsets", []))  # e.g. [0, 35, 42, ...]
            names = meta.get("group_names", meta.get("names", []))

            print("x_cat_onehot:", tuple(Xoh.shape), "| sumC =", sum(sizes))

            # 샘플0, 노드0의 각 그룹 활성 인덱스 출력
            b0, n0 = 0, 0
            if len(sizes) > 0 and len(offsets) > 0:
                act = []
                for g, (st, sz) in enumerate(zip(offsets, sizes)):
                    st, sz = int(st), int(sz)
                    sl = Xoh[b0, n0, st:st + sz]
                    idx1 = torch.nonzero(sl, as_tuple=True)[0]
                    name = names[g] if g < len(names) else f"group{g}"
                    act.append(f"{name}:{idx1.tolist()}")
                print("여러 곳중 1 이 존재하는 위치를 보여줌 (예시 00100... 이면 3으로 나옴)")
                print("x_cat_onehot[b0,node0] active:", ", ".join(act))

        else:
            # 새 index 버전 출력
            show_all_nodes_index(batch, ds, b=0, skip_padding=True)
            if i == 0:
                print_one_five_edges_index(batch, ds, b=0)

        # ---- 글로벌 single(옵션: 원-핫) ----
        if "global_single_onehot" in batch:
            Gs = batch["global_single_onehot"]  # [B, ΣFs]
            gmeta = batch.get("global_features_meta", {})  # {"single_sizes", "single_offsets", "single_cols", ...}
            ss = gmeta.get("single_sizes", [])
            so = gmeta.get("single_offsets", [])
            scn = gmeta.get("single_cols", [])
            print("global_single_onehot:", tuple(Gs.shape), "| sumFs =", sum(ss))
            # 샘플0에서 각 컬럼의 활성 위치 확인
            b0 = 0
            act = []
            for g, (st, sz) in enumerate(zip(so, ss)):
                sl = Gs[b0, st:st + sz]
                idx1 = torch.nonzero(sl, as_tuple=True)[0]
                name = scn[g] if g < len(scn) else f"single{g}"
                act.append(f"{name}:{idx1.tolist()}")
            print("global_single_onehot[b0] active:", ", ".join(act))

        # ---- 글로벌 multi(예: Solvent) ----
        # EmbeddingBag 경로: idx/offsets (항상 있을 수 있음)
        if "global_features_meta" in batch:
            gmeta = batch["global_features_meta"]
            for col in gmeta.get("multi_cols", []):
                key_idx = f"global_mh_{col}_idx"
                key_off = f"global_mh_{col}_offsets"
                if key_idx in batch and key_off in batch:
                    idx, off = batch[key_idx], batch[key_off]
                    print(f"[{col}] idx {tuple(idx.shape)} | offsets {tuple(off.shape)}")
                    if off.numel() >= 2:
                        b0_ids = idx[off[0]:off[1]].tolist()
                        print(f"[{col}] sample0 token-ids:", b0_ids)
                # (선택) multi-hot을 collate에서 만들었다면 여기서도 확인
                key_mh = f"global_mh_{col}_multi_hot"
                if key_mh in batch:
                    mh = batch[key_mh]  # [B, |V_col|]
                    b0 = 0
                    nz = torch.nonzero(mh[b0], as_tuple=True)[0].tolist()
                    print(f"[{col}] multi_hot {tuple(mh.shape)} | sample0 active idx:", nz)

        # ---- exp_spectrum 마스크(있다면) ----
        if "masks" in batch:
            print("masks:", tuple(batch["masks"].shape))

        # 샘플 1개 상세 출력 (원하면 유지)
        if i == 0:
            peek_one_batch(batch, head_n=8, node_n=10)
            break

from types import SimpleNamespace
if __name__ == "__main__":
    file_path_1 = r"C:\Users\kogun\PycharmProjects\DiGress\Graphormer\graphormer_data\train_50_with_features.csv"
    file_path_exp = r"../../graphormer_data/final_split/ABS_stratified_train_plus.csv"

    #df = pd.read_csv(file_path_2)
    #df[0:100].to_csv("fake_exp_like_data_from_QM9Snm_1nm_last_withGlobalFeature_100data.csv")
    #print(df.head())
    if len(sys.argv) == 1:
        args = SimpleNamespace(
            train_file= file_path_exp,
            vocab_file=file_path_1,
            mode="cls_global_data", # cls_global_data, cls_global_model, cls
            target_type="exp_spectrum", # nm_distribution, ex_prob, exp_spectrum
            ex_norm="none",
            prob_norm="none",
            nm_dist_mode="hist",
            nm_gauss_sigma=10.0,
            batch_size=10,
            max_nodes=128,
            multi_hop_max_dist=5,
            nominal_cols=["Solvent", "pH_label", "type"],
            continuous_feature_names=None,
            intensity_normalize="min_max",  #
            intensity_range=(200, 800),  # 200 , 800
            attn_bias_w=0.0,
            x_cat_mode="onehot", # "index", "onehot", "both"
            global_cat_mode="onehot", # "index", "onehot", "both"
        )
    else:
        args = build_parser().parse_args()

    run_pipeline(args)

##### 원자 및 글로벌 Node Feature 모드 "index", "onehot" ####
# index 모드의 경우 None type을 정의 하지 않았음 사용하지 말것 #
# 예시 idx 0 = 수소, 1 = 헬륨 .... -> 이렇게 되어서 None type 이 없음 #
# 따라서 None type 에 대한 id 가 없어서 Global Node 에 Feature 로 None을 줄수 없으니 사용하지 말것 #
# one hot 모드의 경우
# 10000 -> 수소, 01000 헬륨, 00000 -> None 임 따라서 사용 가능
