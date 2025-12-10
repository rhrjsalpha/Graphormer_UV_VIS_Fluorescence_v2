import sys
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from functools import partial
from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from rdkit.Chem import AllChem
from tqdm import tqdm
import io
from rdkit.Chem.MolStandardize import rdMolStandardize
from GP.data_prepare.Pre_Defined_Vocab_Generator import build_vocabs_from_df ,_normalize_token, _normalize_solvent_cell
from GP.data_prepare.Chem_Graph_Utils import (
    smiles_or_inchi_to_graph,            # (text, multi_hop_max_dist, ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB) -> dict or None
    smiles_or_inchi_to_graph_with_global, # (text, global_cat_idx, global_cont_val, ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB, multi_hop_max_dist) -> dict or None
    smiles_or_inchi_to_graphs,
    smiles_or_inchi_to_graphs_with_global
)


sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def get_global_feature_info(global_feature_names, PREDEFINED_VOCAB):
    nominal_feature_vocab = {k: v for k, v in PREDEFINED_VOCAB.items() if k in global_feature_names}
    continuous_feature_names_list = [name for name in global_feature_names if name not in nominal_feature_vocab]

    global_cat_dim = 0
    for name in nominal_feature_vocab:  # 명목형
        global_cat_dim += len(nominal_feature_vocab[name])

    global_cont_dim = len(continuous_feature_names_list)  # 수치형

    return nominal_feature_vocab, continuous_feature_names_list, global_cat_dim, global_cont_dim

def clean_primary_component(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    # 1) 구조 표준화
    m = rdMolStandardize.Cleanup(m)
    m = rdMolStandardize.Normalize(m)
    m = rdMolStandardize.Reionize(m)
    m = rdMolStandardize.Uncharger().uncharge(m)
    # 2) 혼합물일 경우 최대 중원자 조각만
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    # 3) 입체/방향 재할당(가능할 때)
    Chem.SanitizeMol(m)
    Chem.AssignStereochemistry(m, force=True, cleanIt=True)
    return Chem.MolToSmiles(m, isomericSmiles=True)

class UnifiedSMILESDataset(Dataset):
    def __init__(
        self,
        csv_file,
        nominal_feature_vocab,
        continuous_feature_names,
        global_cat_dim,
        global_cont_dim,
        ATOM_FEATURES_VOCAB, float_feature_keys, BOND_FEATURES_VOCAB, GLOBAL_FEATURE_VOCABS_dict,
        # global_multihot_cols: dict,
        x_cat_mode: str = "index",  # "index" | "onehot" | "both"
        global_cat_mode: str = "index",  # "index" | "onehot" | "both"
        mol_col: str = 'smiles',
        mode="cls",  # "cls", "cls+global_data", "cls+global_model"
        max_nodes: int = 128,
        multi_hop_max_dist: int = 5,
        target_type: str = "default",
        intensity_normalize: str = "min_max", #
        intensity_range: tuple = (200, 800),  #
        attn_bias_w: float = 0.0,
        ex_normalize: str = None,
        prob_normalize: str = None,
        nm_dist_mode: str = "hist",
        nm_gauss_sigma: float = 10.0,
        deg_clip_max: int = 5,
    ):
        self.mode = mode
        self.is_global = mode in ("cls_global_data", "cls_global_model")
        self.nominal_feature_vocab = nominal_feature_vocab
        self.continuous_feature_names = continuous_feature_names
        print("self.continuous_feature_names",type(self.continuous_feature_names), self.continuous_feature_names)
        self.global_cat_dim = global_cat_dim
        self.global_cont_dim = global_cont_dim
        self.target_type = target_type
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist

        # target_type ex_prob #
        self.ex_normalize = ex_normalize
        self.prob_normalize = prob_normalize
        self.nm_dist_mode = nm_dist_mode
        self.nm_gauss_sigma = nm_gauss_sigma

        # target_type experiment #
        self.intensity_normalize = intensity_normalize
        self.intensity_range = intensity_range

        self.attn_bias_weight = attn_bias_w

        self.mol_col = mol_col
        self.data = pd.read_csv(csv_file, low_memory=False)
        # === (A) 명목형 컬럼 정규화 ===
        for col in self.nominal_feature_vocab.keys():
            if col in self.data.columns:
                if col == "Solvent":
                    # 'ethanol + water + ethanol' -> 'Ethanol + Water' (중복 제거/정렬/타이틀 케이스)
                    self.data[col] = self.data[col].apply(_normalize_solvent_cell)
                else:
                    # 일반 명목형: 공백/대소문자 정리 (Title Case)
                    self.data[col] = self.data[col].apply(_normalize_token)

        # === (B) Solvent 전용: 토큰 리스트 컬럼 및 대표 토큰 컬럼 생성 ===
        if "Solvent" in self.data.columns:
            self.data["Solvent"] = self.data["Solvent"].apply(_normalize_solvent_cell)

            # primary token(첫 토큰) 뽑기: 'Ethanol + Water' -> 'Ethanol'
            def _pick_primary_token(cell: str) -> str:
                if cell is None or not isinstance(cell, str) or not cell.strip():
                    return "Unknown"
                first = cell.split("+")[0].strip()
                return _normalize_token(first)

            self.data["Solvent_primary_token"] = self.data["Solvent"].apply(_pick_primary_token)

        if self.mol_col not in self.data.columns:
            if 'smiles' in self.data.columns:
                self.mol_col = 'smiles'
            elif 'InChI' in self.data.columns or 'inchi' in self.data.columns:
                self.mol_col = 'InChI' if 'InChI' in self.data.columns else 'inchi'
            else:
                raise ValueError(
                    f"No molecular column found among ['smiles','InChI','inchi']. Found: {list(self.data.columns)[:20]}")

        self.nominal_feature_info = self._build_nominal_feature_info()
        # dict 보관 (+ 방어적 copy)
        print("GLOBAL_FEATURE_VOCABS_dict", GLOBAL_FEATURE_VOCABS_dict)
        self.GLOBAL_FEATURE_VOCABS_dict = dict(GLOBAL_FEATURE_VOCABS_dict)

        self.x_cat_mode = x_cat_mode
        self.global_cat_mode = global_cat_mode

        # 모드에 따라 전역 명목형을 single/multi로 분리
        cols = list(self.nominal_feature_vocab.keys())

        if self.global_cat_mode == "index":
            # index 모드: Solvent라도 '+'를 분리하지 않고 통짜 문자열로 단일 클래스 취급
            self._global_multi_cols = []
            self._global_single_cols = cols[:]  # 모두 싱글
        else:
            # onehot 모드
            # 1) '+' 존재 여부로 멀티 후보 탐지
            plus_mask = {
                c: (c in self.data.columns and self.data[c].astype(str).str.contains(r"\+").any())
                for c in cols
            }

            plus_multi = {c for c in cols if plus_mask.get(c, False)}

            # 2) Solvent는 무조건 멀티핫에 포함(플러스 유무와 무관)
            base_multi = {"Solvent"} if "Solvent" in self.GLOBAL_FEATURE_VOCABS_dict else set()

            # 3) 최종 멀티/싱글 목록
            self._global_multi_cols = sorted(base_multi | plus_multi)
            self._global_single_cols = [c for c in cols if c not in self._global_multi_cols]

            # (선택) 디버그 한 번만
            if not hasattr(self, "_printed_global_cols"):
                self._printed_global_cols = True
                print("[global split] multi:", self._global_multi_cols,
                      "| single:", self._global_single_cols)

        # (1) 전역 카테고리 피처 순서/크기/오프셋 메타
        self.global_cat_feature_order = list(self.nominal_feature_vocab.keys())  # 예: ["Solvent","pH_label","type"]
        self.global_cat_feature_sizes = [len(self.nominal_feature_vocab[n]) for n in self.global_cat_feature_order]
        self.global_cat_feature_offsets = np.cumsum([0] + self.global_cat_feature_sizes[:-1]).tolist()
        self.global_cat_total_dim = sum(self.global_cat_feature_sizes)

        assert self.global_cat_dim == self.global_cat_total_dim, \
            f"global_cat_dim mismatch: {self.global_cat_dim} vs {self.global_cat_total_dim}"

        self._validate_columns(csv_file)
        self.data = self.data.loc[:, self._get_all_cols_to_load()]

        # Stastics to Normalize ex_prob
        if target_type in ["ex_prob", "nm_distribution"]:
            ex_data = self.data[[f"ex{i}" for i in range(1, 51)]].values
            prob_data = self.data[[f"prob{i}" for i in range(1, 51)]].values

            self.global_ex_min = float(np.min(ex_data))
            self.global_ex_max = float(np.max(ex_data))
            self.global_ex_mean = float(np.mean(ex_data))
            self.global_ex_std = float(np.std(ex_data))

            self.global_prob_min = float(np.min(prob_data))
            self.global_prob_max = float(np.max(prob_data))
            self.global_prob_mean = float(np.mean(prob_data))
            self.global_prob_std = float(np.std(prob_data))

        elif self.target_type == "exp_spectrum":
            nm_min, nm_max = self.intensity_range
            target_cols = [str(i) for i in range(nm_min, nm_max + 1)]
            existing_cols = [col for col in target_cols if col in self.data.columns]
            missing_cols = set(target_cols) - set(existing_cols)
            for col in missing_cols:
                self.data[col] = 0.0

        else:
            self.global_ex_min = self.global_ex_max = self.global_ex_mean = self.global_ex_std = None
            self.global_prob_min = self.global_prob_max = self.global_prob_mean = self.global_prob_std = None

        self.deg_clip_max = int(deg_clip_max)
        #self.raw_graphs = [g for g in [smiles2graph_customized(s, self.multi_hop_max_dist ,ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB, float_feature_keys=float_feature_keys, BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB) for s in self.data["smiles"]] if g is not None]

        # 1. targets, mask 분리
        if self.target_type == "exp_spectrum":
            spectrum, mask_tensor = self.process_targets()
        else:
            spectrum = self.process_targets()
            mask_tensor = None

        # 2. 그래프 생성 (토토머 전개 지원)
        self.raw_graphs = []
        valid_indices = []

        expanded_targets = []  # targets 복제본 누적
        expanded_rows = []  # self.data의 행 복제본 누적 (DataFrame 재구성용)

        use_global = (self.mode == "cls_global_data")
        self._graphs_already_have_global = use_global

        # 전역 피처 순서(딕셔너리 삽입 순서 유지; 필요하면 원하는 순서로 직접 지정 가능)
        self.global_cat_feature_order = list(self.GLOBAL_FEATURE_VOCABS_dict.keys())
        # dict → list-of-vocabs 복원 (유틸 함수들이 기대하는 형태)
        GLOBAL_FEATURE_VOCABS = [
            self.GLOBAL_FEATURE_VOCABS_dict[name] for name in self.global_cat_feature_order
        ]

        # === [PATCH 1] Solvent 멀티핫 스펙/헬퍼 준비 ===
        # 멀티핫 기능 on/off 토글(원하면 False로 끌 수 있음)
        self._enable_solvent_multihot = (self.mode == "cls_global_data") and (self.x_cat_mode != "index")

        # Solvent vocab(비트 순서 고정용)
        self.SOLVENT_VOCAB = list(self.GLOBAL_FEATURE_VOCABS_dict.get("Solvent", []))

        # 비트별 그룹 메타(각 비트는 0/1 2-클래스 카테고리)
        self.mh_group_names = [f"mh_Solvent::{tok}" for tok in self.SOLVENT_VOCAB]
        self.mh_group_sizes = [2] * len(self.SOLVENT_VOCAB)

        # 'Ethanol + Water' 같은 셀 → ['Ethanol','Water'] 로 표준화 분해
        def _split_multi_solvent_cell(cell: str) -> list[str]:
            if not isinstance(cell, str) or not cell.strip():
                return []
            # 이미 위에서 _normalize_solvent_cell 적용됨(self.data["Solvent"] 라인들)
            # 그래도 안전하게 처리:
            toks = [t.strip() for t in cell.split("+")]
            # Title-Case 정규화와 동일하게 맞추기(혹시 모를 편차 방지)
            return [_normalize_token(t) for t in toks if t]

        # 현재 row(idx)의 Solvent 멀티핫 비트(0/1) 벡터 만들기
        def _multihot_bits_for_row(idx: int) -> np.ndarray:
            if "Solvent" not in self.data.columns or not self.SOLVENT_VOCAB:
                return np.zeros((0,), dtype=np.int64)
            cell = str(self.data.loc[idx, "Solvent"])
            toks = set(_split_multi_solvent_cell(cell))
            bits = np.fromiter((1 if tok in toks else 0 for tok in self.SOLVENT_VOCAB),
                               dtype=np.int64, count=len(self.SOLVENT_VOCAB))
            return bits  # shape=(M,), dtype=int64 (0/1)

        for i, text in enumerate(tqdm(self.data[self.mol_col], desc="Building graphs")):
            try:
                # /FixedH InChI 인지 확인
                is_fixedh_inchi = isinstance(text, str) and text.startswith("InChI=") and ("/FixedH" in text)

                if is_fixedh_inchi:
                    if use_global:
                        graphs = smiles_or_inchi_to_graphs_with_global(
                            text,
                            self._get_global_feature_cat_tensor(i).tolist(),
                            (self._get_global_feature_cont_tensor(i).tolist()
                             if self.continuous_feature_names else []),
                            ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                            float_feature_keys=float_feature_keys,
                            BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
                            GLOBAL_FEATURE_VOCABS=GLOBAL_FEATURE_VOCABS_dict,
                            multi_hop_max_dist=self.multi_hop_max_dist,
                        )
                    else:
                        graphs = smiles_or_inchi_to_graphs(
                            text,
                            multi_hop_max_dist=self.multi_hop_max_dist,
                            ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                            float_feature_keys=float_feature_keys,
                            BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
                        )
                else:
                    # 단일 그래프 경로
                    if use_global:
                        g = smiles_or_inchi_to_graph_with_global(
                            text,
                            self._get_global_feature_cat_tensor(i).tolist(),
                            (self._get_global_feature_cont_tensor(i).tolist()
                             if self.continuous_feature_names else []),
                            multi_hop_max_dist=self.multi_hop_max_dist,
                            ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                            float_feature_keys=float_feature_keys,
                            BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
                            GLOBAL_FEATURE_VOCABS=GLOBAL_FEATURE_VOCABS_dict,
                        )
                    else:
                        g = smiles_or_inchi_to_graph(
                            text,
                            self.multi_hop_max_dist,
                            ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                            float_feature_keys=float_feature_keys,
                            BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
                        )
                    graphs = [g] if g is not None else []

                # 누적 + 타깃/행 복제
                for g in graphs:
                    if g is None:
                        continue
                    self.raw_graphs.append(g)
                    valid_indices.append(i)
                    expanded_targets.append(spectrum[i])  # 타깃 1:1 복제
                    expanded_rows.append(self.data.iloc[i].copy())  # 메타/글로벌 피처 복제

            except Exception:
                # 실패시 해당 레코드 스킵
                continue

        # 3. 대상/데이터 재구성 (토토머 복제로 인덱스 늘어난 상태) 대상 필터링
        self.targets = torch.as_tensor(np.stack(expanded_targets), dtype=torch.float32) \
            if isinstance(spectrum, torch.Tensor) else spectrum[valid_indices]

        if mask_tensor is not None:
            self.masks = mask_tensor[valid_indices]  # ← 필요 시 사용

        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        self.data = pd.DataFrame(expanded_rows).reset_index(drop=True)

        self.graphs = self.raw_graphs
        self._xcat_cols_before_mh = self.graphs[0]['x_cat'].shape[1]

        # === [PATCH 2] 모든 노드의 x_cat 뒤에 Solvent 멀티핫(0/1) 열 붙이기 ===
        if (self.mode == "cls_global_data") and self._enable_solvent_multihot and len(self.SOLVENT_VOCAB) > 0 and "Solvent" in self.data.columns:
            for gi, g in enumerate(self.graphs):
                x_cat = g.get("x_cat", None)
                if x_cat is None:
                    continue  # 방어

                # x_cat은 numpy 배열이어야 함 [N, C]
                if isinstance(x_cat, torch.Tensor):
                    x_cat = x_cat.cpu().numpy()

                bits = _multihot_bits_for_row(gi)  # shape=(M,)
                if bits.size == 0:
                    continue
                # 모든 노드에 동일한 조건 비트 복제
                bits_tile = np.broadcast_to(bits, (x_cat.shape[0], bits.shape[0]))  # [N, M]

                # 정수 타입 유지(카테고리 인덱스)
                bits_tile = bits_tile.astype(np.int64, copy=False)

                # x_cat 뒤에 concat
                g["x_cat"] = np.concatenate([x_cat, bits_tile], axis=1)


        # ------------- 메타 계산: 실제 컬럼 순서 = [원자] + [멀티핫] + [싱글 전역] -------------
        # ------------- 메타 계산: 실제 컬럼 순서 반영 -------------
        # ---- [고정 메타] : vocab 길이로 차원 고정 ----
        # 1) 원자(feature 그룹) 순서/이름
        atom_names = [k for k, v in ATOM_FEATURES_VOCAB.items() if isinstance(v, list)]
        n_atom = len(atom_names)

        # 2) 각 원자 그룹의 클래스 수 = 해당 vocab 길이 (고정)
        atom_sizes = [len(ATOM_FEATURES_VOCAB[name]) for name in atom_names]

        # 3) 멀티핫/전역(single)도 고정 크기 사용
        C_before = self._xcat_cols_before_mh
        C_after = self.graphs[0]['x_cat'].shape[1]

        # (전역 싱글이 그래프에 '이미' 들어간 구간)
        pre_extra_names = []
        pre_extra_sizes = []

        # pre_extra 후보를 아예 ‘싱글 집합’에서만 뽑는 것 : 따라서 Solvent가 multihot 일때(고정됨) singlehot 경로로 가지 않음
        pre_extra_C = max(0, C_before - n_atom)
        expected_global_names_order = list(self.GLOBAL_FEATURE_VOCABS_dict.keys())
        # 항상 Solvent 제외
        expected_no_solvent = [n for n in expected_global_names_order if n != "Solvent"]
        # 필요 개수만큼 앞에서 슬라이스 (Solvent가 절대 포함되지 않도록 보장)
        if len(expected_no_solvent) < pre_extra_C:
            # 안전장치: 부족하면 있는 만큼만 쓰고 나머지는 무시
            pre_extra_names = expected_no_solvent
        else:
            pre_extra_names = expected_no_solvent[:pre_extra_C]
        pre_extra_sizes = [len(self.GLOBAL_FEATURE_VOCABS_dict[n]) for n in pre_extra_names]

        # (멀티핫 구간: Solvent bit 당 2상태 {NO, YES} → 항상 2로 고정)
        mh_C = max(0, C_after - C_before)
        mh_names = [f"mh_Solvent::{tok}" for tok in self.SOLVENT_VOCAB][:mh_C]
        mh_sizes = [2] * len(mh_names)

        # (추가로 뒤에 붙일 전역 싱글 — 이미 들어간 건 제외)
        single_names_all = list(getattr(self, "_global_single_cols", []))
        single_names_to_append = [n for n in single_names_all if n not in pre_extra_names]
        single_sizes_to_append = [
            len(self.GLOBAL_FEATURE_VOCABS_dict[n]) if n in self.GLOBAL_FEATURE_VOCABS_dict
            else len(self.nominal_feature_vocab.get(n, []))
            for n in single_names_to_append
        ]

        # 모드가 cls_only면 전역 관련은 비움
        if self.mode != "cls_global_data":
            mh_names, mh_sizes = [], []
            single_names_to_append, single_sizes_to_append = [], []

        # 최종 메타 (실제 x_cat 열 순서와 일치; ‘고정 크기’만 반영)
        self.x_cat_group_names = atom_names + pre_extra_names + mh_names + single_names_to_append
        self.x_cat_onehot_sizes = atom_sizes + pre_extra_sizes + mh_sizes + single_sizes_to_append
        self.x_cat_onehot_offsets = np.cumsum([0] + self.x_cat_onehot_sizes[:-1]).tolist()

        # (2) 멀티핫을 붙이기 전/후 경계
        C_before = self._xcat_cols_before_mh  # [원자 + (이미 들어온 전역 싱글)]
        C_after = self.graphs[0]['x_cat'].shape[1]  # [위 + 멀티핫]

        # (2-1) 이미 들어와 있던 "전역 싱글" 구간
        pre_extra_C = max(0, C_before - n_atom)
        # with_global()가 사용하는 전역 순서를 그대로 따라감
        expected_global_names_order = list(self.GLOBAL_FEATURE_VOCABS_dict.keys())
        pre_extra_names = expected_global_names_order[:pre_extra_C]
        pre_extra_sizes = [len(self.GLOBAL_FEATURE_VOCABS_dict[n]) for n in pre_extra_names]

        # (2-2) 멀티핫 구간 = [C_before : C_after]
        mh_C = max(0, C_after - C_before)
        mh_names = [f"mh_Solvent::{tok}" for tok in self.SOLVENT_VOCAB][:mh_C]
        mh_sizes = [2] * mh_C

        # (3) 앞으로 __getitem__에서 "추가로" 붙일 싱글(이미 pre_extra에 들어간 건 제외)
        single_names_all = list(getattr(self, "_global_single_cols", []))
        single_names_to_append = [n for n in single_names_all if n not in pre_extra_names]
        single_sizes_to_append = [
            (len(self.GLOBAL_FEATURE_VOCABS_dict[n]) if n in self.GLOBAL_FEATURE_VOCABS_dict
             else len(self.nominal_feature_vocab.get(n, [])))
            for n in single_names_to_append
        ]

        if self.mode != "cls_global_data":
            mh_names, mh_sizes = [], []  # 멀티핫 배제
            single_names_to_append, single_sizes_to_append = [], []  # 싱글 배제

        # (4) 최종 메타 (실제 x_cat 열 순서와 정확히 일치)
        self.x_cat_group_names = atom_names + pre_extra_names + mh_names + single_names_to_append
        self.x_cat_onehot_sizes = atom_sizes + pre_extra_sizes + mh_sizes + single_sizes_to_append
        self.x_cat_onehot_offsets = np.cumsum([0] + self.x_cat_onehot_sizes[:-1]).tolist()

        # self.x_cat_group_names / sizes 계산 바로 아래에 추가
        # 슬라이스 경계
        n_atom = len(atom_names)
        pre_extra_C = len(pre_extra_names)  # 이미 그래프에 들어온 전역 싱글 개수
        mh_C = len(mh_names)
        pre_lo, pre_hi = n_atom, n_atom + pre_extra_C
        mh_lo, mh_hi = C_before, C_after

        def _find_global_row(X, n_atom):
            # "원자 슬롯이 모두 0"인 행을 전역 노드로 간주, 없으면 마지막 노드 사용
            for r in range(X.shape[0]):
                if (X[r, :n_atom] == 0).all():
                    return r
            return X.shape[0] - 1

        if self.mode == "cls_global_data":
            for g in self.graphs:
                X = g["x_cat"]
                # numpy 보장
                if isinstance(X, torch.Tensor):
                    X = X.cpu().numpy()
                N = g["num_nodes"]
                g_row = _find_global_row(X, n_atom)

                mask = np.ones(N, dtype=bool)
                mask[g_row] = False

                # 전역 싱글/멀티핫을 전역 노드 외에는 0으로
                if pre_extra_C > 0:
                    X[mask, pre_lo:pre_hi] = 0
                if mh_C > 0:
                    X[mask, mh_lo:mh_hi] = 0

                g["x_cat"] = X

        print("[META] atoms:", len(atom_names),
              "pre_extra:", len(pre_extra_names),
              "mh:", len(mh_names),
              "append_singles:", len(single_names_to_append))
        print("[META] C_before=", C_before, "C_after=", C_after)
        print("[META] pre_extra:", list(zip(pre_extra_names, pre_extra_sizes))[:5])
        print("[META] mh_names first 5:", mh_names[:5])
        print("[META] append singles:", list(zip(single_names_to_append, single_sizes_to_append)))

        for i, g in enumerate(self.graphs):
            if i == 0:
                print("[DEBUG] raw graph keys:", len(list(g.keys())),list(g.keys()))
            break

        # --- global single(=index 1개짜리) 메타 ---
        self.global_single_cols = getattr(self, "_global_single_cols", [])
        self.global_single_sizes = [len(self.nominal_feature_vocab[c]) for c in self.global_single_cols]
        self.global_single_offsets = np.cumsum([0] + self.global_single_sizes[:-1]).tolist()

        # --- global multi(예: Solvent) 메타 ---
        self.global_multi_cols = getattr(self, "_global_multi_cols", [])
        self.global_multi_sizes = {c: len(self.nominal_feature_vocab[c]) for c in self.global_multi_cols}

        # 안전 가드 추가
        feature_order = list(self.nominal_feature_vocab.keys())

        if ("Solvent" in feature_order) and ("Solvent" in self.data.columns):
            try:
                solvent_col = feature_order.index("Solvent")
                nrows = min(5, len(self.data))
                info = self._build_nominal_feature_info()
                for i in range(nrows):
                    val = self.data.loc[i, "Solvent"]
                    idx = info.get("Solvent", {}).get("value_to_idx", {}).get(val, None)
                    print(f"[DEBUG] row={i}  raw='{val}'  -> idx={idx}")
            except Exception as e:
                print(f"[DEBUG] skip solvent debug: {e}")
        else:
            print("[DEBUG] 'Solvent' not in global features; skip solvent debug.")

        # ---- (기존) 최종 메타 계산 코드 바로 아래에 이어서 추가 ----
        # (기존) self.xcat_meta_raw / self.xcat_meta_compact 둘 다 만들던 코드 대신
        self.xcat_group_names = self.x_cat_group_names  # 실제 컬럼 순서
        self.xcat_sizes = self.x_cat_onehot_sizes
        self.xcat_offsets = self.x_cat_onehot_offsets

        # compact 메타 생성 (mh_Solvent::*를 1비트씩 |V|개로 접은 버전)
        mh_idx = [i for i, n in enumerate(self.xcat_group_names) if str(n).startswith("mh_Solvent::")]

        if mh_idx:
            first_mh, last_mh = min(mh_idx), max(mh_idx)
            compact_group_names = (
                    self.xcat_group_names[:first_mh] +
                    ["mh_Solvent"] +
                    self.xcat_group_names[last_mh + 1:]
            )

            compact_sizes = (
                    self.xcat_sizes[:first_mh] +
                    [len(self.SOLVENT_VOCAB)] +
                    self.xcat_sizes[last_mh + 1:]
            )

            compact_offsets = np.cumsum([0] + compact_sizes[:-1]).tolist()
        else:
            compact_group_names = list(self.xcat_group_names)
            compact_sizes = list(self.xcat_sizes)
            compact_offsets = list(self.xcat_offsets)

        self.xcat_meta = {
            "group_names": compact_group_names,
            "sizes": compact_sizes,
            "offsets": compact_offsets,
            "total_dim": int(sum(compact_sizes)),
        }

    def __getitem__(self, idx):
        tgt = self.targets[idx]
        raw_g = self.raw_graphs[idx]
        g_processed = self.preprocess_graph(raw_g)  # 여기서 x_cat_onehot(compact)까지 이미 생성됨

        # ── 글로벌 연속형
        if self.continuous_feature_names:
            vals = []
            for name in self.continuous_feature_names:
                v = self.data.loc[idx, name]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                vals.append(float(v) if pd.notna(v) else 0.0)
            global_cont = torch.tensor(vals, dtype=torch.float32)
        else:
            global_cont = torch.zeros(0, dtype=torch.float32)

        # ── 글로벌 카테고리 인덱스
        global_cat_idx = self._get_global_feature_cat_tensor(idx).long()  # [Fg]

        # (옵션) x_cat 인덱스 뒤에 글로벌 슬롯을 붙여야 하는 모드라면 여기서 1회 처리
        if (global_cat_idx.numel() > 0) and (not self._graphs_already_have_global) and ("x_cat" in g_processed):
            X = g_processed["x_cat"]  # [N, C_atom]
            N, C_atom = X.shape
            G = global_cat_idx.numel()
            pad = torch.zeros((N, G), dtype=torch.long)
            pad[N - 1] = global_cat_idx  # 글로벌 노드(마지막)에만 세팅
            X = torch.cat([X, pad], dim=1)
            X[N - 1, :C_atom] = 0  # 글로벌 노드의 원자 슬롯 0
            g_processed["x_cat"] = X

        # ── 글로벌 카테고리: index ↔ onehot 변환
        def _idx_to_onehot(idx_vec: torch.Tensor) -> torch.Tensor:
            if idx_vec.numel() == 0:
                return torch.zeros(0, dtype=torch.float32)
            sizes = [len(self.nominal_feature_vocab[n]) for n in self._global_single_cols]
            pieces = []
            for f, nc in enumerate(sizes):
                i = idx_vec[f].clamp(min=0)
                pieces.append(torch.nn.functional.one_hot(i, num_classes=int(nc)).float())
            return torch.cat(pieces, dim=-1) if pieces else torch.zeros(0, dtype=torch.float32)

        global_cat = _idx_to_onehot(global_cat_idx) if (self.global_cat_mode == "onehot") else global_cat_idx

        # ── 모드별 반환
        if self.mode == "cls_global_data":
            if global_cat.numel() > 0:
                g_processed["global_features_cat"] = global_cat
            if global_cont.numel() > 0:
                g_processed["global_features_cont"] = global_cont
            return g_processed, tgt, idx

        elif self.mode == "cls_global_model":
            extras = {}
            if global_cat.numel() > 0:
                extras["global_features_cat"] = global_cat
            if global_cont.numel() > 0:
                extras["global_features_cont"] = global_cont
            return g_processed, tgt, idx, extras

        # cls-only
        return g_processed, tgt, idx

    def __len__(self):
        return len(self.graphs)

    def _preprocess_graph_with_optional_global(self, idx, graph, ATOM_FEATURES_VOCAB, float_feature_keys,
                                               BOND_FEATURES_VOCAB):
        # 이미 global이 들어있는 그래프면 그대로 사용
        if self.mode == "cls_global_data" and (
                "global_features_cat" in graph and "global_features_cont" in graph
        ):
            return graph

        if self.mode == "cls_global_data":
            global_cat = self._get_global_feature_cat_tensor(idx).tolist()
            global_cont = (self._get_global_feature_cont_tensor(idx).tolist()
                           if self.continuous_feature_names else [])
            return smiles_or_inchi_to_graph_with_global(
                self.data.loc[idx, self.mol_col],
                global_cat,
                global_cont,
                multi_hop_max_dist=self.multi_hop_max_dist,
                ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                float_feature_keys=float_feature_keys,
                BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB
            )
        else:
            return graph

    def _build_nominal_feature_info(self):
        return {
            name: {
                'unique_values': vocab,
                'value_to_idx': {val: i for i, val in enumerate(vocab)}
            } for name, vocab in self.nominal_feature_vocab.items()
        }

    def _get_all_cols_to_load(self):
        if self.target_type in ["ex_prob", "nm_distribution"]:
            target_cols = [f"ex{i}" for i in range(1, 51)] + [f"prob{i}" for i in range(1, 51)]
        elif self.target_type == "exp_spectrum":
            nm_min, nm_max = self.intensity_range  # 예: (200, 800)
            all_columns = self.data.columns
            target_cols = []
            for i in range(nm_min, nm_max + 1):
                target_cols.append(str(i))
        else:
            target_cols = []
        required_cols = [self.mol_col] + target_cols

        extra_cols = []

        # 🔧 멀티핫만 쓰는 경우에도 Solvent 원본 셀을 반드시 로드
        if "Solvent" not in self.nominal_feature_vocab:
            extra_cols.append("Solvent")

        if "Solvent" in self.nominal_feature_vocab and "Solvent_primary_token" in self.data.columns:
            extra_cols.append("Solvent_primary_token")

        return required_cols + list(self.nominal_feature_vocab.keys()) + self.continuous_feature_names + extra_cols

    def _validate_columns(self, csv_file):
        for col in self._get_all_cols_to_load():
            #print(self._get_all_cols_to_load())
            if col not in self.data.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_file}")

    def process_targets(self, n_pairs=None):
        if self.target_type == "default":
            arr = self.data.iloc[:, 1:101].values
            return torch.tensor(arr, dtype=torch.float32)

        elif self.target_type == "ex_prob":
            arr = self.data.iloc[:, 1:101].values
            max_pairs = arr.shape[1] // 2
            if n_pairs is None or n_pairs > max_pairs:
                n_pairs = max_pairs
            ex = arr[:, :max_pairs]
            prob = arr[:, max_pairs:]

            # 1. intensity 기준 정렬
            sorted_idx = np.argsort(-prob, axis=1)
            top_idx = sorted_idx[:, :n_pairs]
            ex_top = np.take_along_axis(ex, top_idx, axis=1)
            prob_top = np.take_along_axis(prob, top_idx, axis=1)

            # 2. eV 기준 정렬
            asc_idx = np.argsort(ex_top, axis=1)
            ex_top = np.take_along_axis(ex_top, asc_idx, axis=1)
            prob_top = np.take_along_axis(prob_top, asc_idx, axis=1)

            # 4. ex 정규화
            if self.ex_normalize == "ex_min_max":
                ex_top = (ex_top - self.global_ex_min) / (self.global_ex_max - self.global_ex_min + 1e-8)
            elif self.ex_normalize == "ex_std":
                ex_top = (ex_top - self.global_ex_mean) / (self.global_ex_std + 1e-8)
            elif self.ex_normalize == "none":
                pass
            else:
                raise ValueError(f"Unknown ex_normalize: {self.ex_normalize}")

            # 5. prob 정규화
            if self.prob_normalize == "prob_min_max":
                prob_top = (prob_top - self.global_prob_min) / (self.global_prob_max - self.global_prob_min + 1e-8)
            elif self.prob_normalize == "prob_std":
                prob_top = (prob_top - self.global_prob_mean) / (self.global_prob_std + 1e-8)
            elif self.prob_normalize == "none":
                pass
            else:
                raise ValueError(f"Unknown prob_normalize: {self.prob_normalize}")

            stacked = np.stack((ex_top, prob_top), axis=-1)
            return torch.tensor(stacked, dtype=torch.float32)

        elif self.target_type == "nm_distribution":
            ex = self.data[[f"ex{i}" for i in range(1, 51)]].values
            prob = self.data[[f"prob{i}" for i in range(1, 51)]].values
            nm = (1239.841984 / ex).round().astype(int)

            if self.intensity_normalize == "min_max":
                prob = (prob - self.global_prob_min) / (self.global_prob_max - self.global_prob_min + 1e-8)

            nm_min, nm_max = self.intensity_range
            nm = np.clip(nm, nm_min, nm_max)
            spec_len = nm_max - nm_min + 1
            out = np.zeros((len(self.data), spec_len), dtype=np.float32)

            if self.nm_dist_mode == "hist":
                for i, (row_nm, row_p) in enumerate(zip(nm, prob)):
                    for λ, p in zip(row_nm, row_p):
                        if 150 <= λ <= 600:
                            out[i, λ - 150] += p

            elif self.nm_dist_mode == "gauss":
                bins = np.arange(150, 601)
                σ = self.nm_gauss_sigma
                for i, (row_nm, row_p) in enumerate(zip(nm, prob)):
                    spec = np.zeros_like(bins, dtype=np.float32)
                    for λ, p in zip(row_nm, row_p):
                        if 150 <= λ <= 600 and p > 0:
                            kernel = np.exp(-0.5 * ((bins - λ) / σ) ** 2)
                            kernel /= (kernel.sum() + 1e-8)
                            spec += p * kernel
                    out[i] = spec
            else:
                raise ValueError(f"Unknown nm_dist_mode: {self.nm_dist_mode}, use 'hist' or 'gauss'")
            return torch.tensor(out, dtype=torch.float32)

        elif self.target_type == "exp_spectrum":
            nm_min, nm_max = self.intensity_range
            target_cols = [str(i) for i in range(nm_min, nm_max + 1)]

            # 존재하는 컬럼만 선택
            existing_cols = [col for col in target_cols if col in self.data.columns]
            missing_cols = set(target_cols) - set(existing_cols)
            if missing_cols:
                print(f"[Warning] {len(missing_cols)} missing columns will be filled with zeros")

            # 누락된 컬럼 0으로 채워 넣기
            for col in missing_cols:
                self.data[col] = 0.0

            spectrum = self.data[target_cols].fillna(0.0).values

            # 개별 스펙트럼별 정규화
            normed = []
            masks = []
            for row in spectrum:
                mask = (row != 0)
                if np.sum(mask) == 0:
                    normed.append(np.zeros_like(row))
                else:
                    valid_vals = row[mask]
                    row_min, row_max = np.min(valid_vals), np.max(valid_vals)
                    row_range = row_max - row_min + 1e-8
                    norm_row = np.zeros_like(row)
                    norm_row[mask] = (valid_vals - row_min) / row_range
                    normed.append(norm_row)

                masks.append(mask)

            spectrum = np.stack(normed)
            spectrum = torch.tensor(spectrum, dtype=torch.float32)
            mask_tensor = torch.tensor(np.stack(masks), dtype=torch.bool)  # 또는 dtype=torch.bool
            return spectrum, mask_tensor

        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

    def preprocess_graph(self, graph):
        num_nodes = graph["num_nodes"]

        x_cat = torch.from_numpy(graph['x_cat']).long()
        x_cont = torch.from_numpy(graph['x_cont']).float()

        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        in_deg = torch.bincount(edge_index[1], minlength=num_nodes).long()
        out_deg = torch.bincount(edge_index[0], minlength=num_nodes).long()
        #print(in_deg.shape, out_deg.shape,)

        adj = torch.from_numpy(graph['adj'])

        attn_edge_type_tensor_dict = {
            k: torch.from_numpy(v).long()
            for k, v in graph['attn_edge_type'].items()
        }

        spatial_pos = torch.from_numpy(graph['spatial_pos']).float()
        attn_bias = torch.zeros((num_nodes, num_nodes), dtype=torch.float)  # Placeholder

        edge_input_tensor_dict = {
            k: torch.from_numpy(v).long()
            for k, v in graph['edge_input'].items()
        }

        g = {
            "x_cat": x_cat,
            "x_cont": x_cont,
            "adj": adj,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "attn_edge_type": attn_edge_type_tensor_dict,
            "spatial_pos": spatial_pos,
            "attn_bias": attn_bias,
            "edge_input": edge_input_tensor_dict,
            "num_nodes": num_nodes,
        }

        # 오직 cls+global_data 모드일 때만 포함
        if self.mode == "cls_global_data":
            global_cat = torch.tensor(graph.get("global_features_cat", []), dtype=torch.long)
            global_cont = torch.tensor(graph.get("global_features_cont", []), dtype=torch.float32)
            g["global_features_cat"] = global_cat
            g["global_features_cont"] = global_cont
            v = g.get("global_features_cont", torch.empty(0, dtype=torch.float32))
            if torch.is_tensor(v) and v.numel() > 0:
                # NaN / Inf 방어
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                # 모든 노드에 동일 값으로 타일링
                N = g["num_nodes"]
                v_tiled = v.unsqueeze(0).repeat(N, 1)  # [N, G]
                # 기존 노드 연속형 뒤에 이어 붙이기
                g["x_cont"] = torch.cat([g["x_cont"], v_tiled], dim=1)  # [N, C_node+G]
                # (옵션) 디버깅 플래그
                g["x_cont_has_global"] = True

        #else:
        #    print("cls only OR cls+global_data mode")

        # ======================= [ADD] x_cat → onehot + 게이팅 =======================
        # self.x_cat_onehot_sizes / names / offsets 는 __init__ 에서 이미 계산됨
        if getattr(self, "x_cat_mode", "index") == "onehot":
            X_idx = g["x_cat"]  # [N, C_idx]
            sizes = self.x_cat_onehot_sizes  # raw 기준
            names = self.x_cat_group_names
            offs = self.x_cat_onehot_offsets

            # 1) onehot
            pieces = []
            for c, nc in enumerate(sizes):
                col = X_idx[:, c].clamp(min=0, max=int(nc) - 1)  # 인덱스 범위 고정
                pieces.append(torch.nn.functional.one_hot(col, num_classes=int(nc)).float())
            X_oh_raw = torch.cat(pieces, dim=-1)

            # 2) 게이팅 (원자/전역 채널)
            ATOM_GROUPS = {"atomic_num", "formal_charge", "hybridization",
                           "is_aromatic", "total_num_hs", "explicit_valence", "total_bonds"}
            C_total = offs[-1] + sizes[-1]
            atom_mask = torch.zeros(C_total, dtype=torch.bool)
            for nm, off, sz in zip(names, offs, sizes):
                if nm in ATOM_GROUPS:
                    atom_mask[off:off + sz] = True
            glob_mask = ~atom_mask

            N = X_oh_raw.shape[0]
            g_row = g["num_nodes"] - 1 if g["num_nodes"] > 0 else 0
            if g_row > 0:
                X_oh_raw[:g_row, :][:, glob_mask] = 0.0
            X_oh_raw[g_row, :][atom_mask] = 0.0

            # 3) compact 접기 (mh_Solvent::* 2채널 × |V| → 1채널 × |V|)
            mh_idx = [i for i, n in enumerate(names) if str(n).startswith("mh_Solvent::")]
            if mh_idx:
                first_mh, last_mh = min(mh_idx), max(mh_idx)
                head = X_oh_raw[:, : offs[first_mh]]

                mh_bits = []
                for i in mh_idx:
                    s = offs[i]
                    sz = sizes[i]  # 보통 2
                    yes = X_oh_raw[:, s + 1:s + 2] if sz >= 2 else X_oh_raw[:, s:s + 1]
                    mh_bits.append(yes)
                mh_block = torch.cat(mh_bits, dim=-1)

                tail_start = offs[last_mh] + sizes[last_mh]
                tail = X_oh_raw[:, tail_start:]

                X_oh_compact = torch.cat([head, mh_block, tail], dim=-1)
            else:
                X_oh_compact = X_oh_raw

            # 최종: compact 결과/메타만 저장
            g["x_cat_onehot"] = X_oh_compact
            g["x_cat_onehot_meta"] = {
                "group_names": self.xcat_meta["group_names"],
                "sizes": torch.tensor(self.xcat_meta["sizes"], dtype=torch.long),
                "offsets": torch.tensor(self.xcat_meta["offsets"], dtype=torch.long),
                "total_dim": self.xcat_meta["total_dim"],
            }
            # ============================================================================

        return g

    def _get_global_feature_cat_tensor(self, idx):
        """싱글 전역 카테고리만 인덱스 1개씩 반환"""

        indices = []
        for name in self._global_single_cols:
            val = self.data.loc[idx, name]
            indices.append(self.nominal_feature_info[name]['value_to_idx'].get(val, 0))
        return torch.tensor(indices, dtype=torch.long)

    def _get_global_feature_cont_tensor(self, idx):
        if not self.continuous_feature_names:
            return torch.zeros(0, dtype=torch.float32)
        vals = []
        for name in self.continuous_feature_names:
            v = self.data.loc[idx, name]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            vals.append(float(v) if pd.notna(v) else 0.0)
        return torch.tensor(vals, dtype=torch.float32)


def collate_fn(batch, ds):
    # 1) 유효 샘플만
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch:
        return None



    graphs  = [b[0] for b in batch]
    tgt_idx = [b[2] for b in batch]
    max_nodes = max(g['num_nodes'] for g in graphs) if graphs else 0

    # ───────────────────────── helpers
    def _pad2d(x, total_rows, pad_value=0):
        """x: [N, C] -> pad to [total_rows, C]"""
        pad = total_rows - x.shape[0]
        return torch.nn.functional.pad(x, (0, 0, 0, pad), value=pad_value) if pad > 0 else x

    def _stack_opt(list_of_tensors):
        """빈 텐서 리스트 안전 스택"""
        if len(list_of_tensors) == 0:
            return None
        if list_of_tensors[0].numel() == 0:
            # 빈 벡터들인 경우 [B, 0] 형태로
            return torch.zeros((len(list_of_tensors), 0), dtype=list_of_tensors[0].dtype)
        return torch.stack(list_of_tensors)

    # ───────────────────────── 2) 노드 피처 pad+stack
    collated_x_cat  = _stack_opt([_pad2d(g['x_cat'],  max_nodes, 0) for g in graphs]) if 'x_cat'  in graphs[0] else None
    collated_x_cont = _stack_opt([_pad2d(g['x_cont'], max_nodes, 0.0) for g in graphs]) if 'x_cont' in graphs[0] else None

    # 선택: 이미 그래프에 준비된 onehot이 있다면 그대로 pad+stack (변환/compact 없음)
    if 'x_cat_onehot' in graphs[0]:
        x_cat_oh_batch = []
        for g in graphs:
            Xoh = g['x_cat_onehot']                   # [N, ΣC]
            x_cat_oh_batch.append(_pad2d(Xoh, max_nodes, 0.0))
        collated_x_cat_onehot = torch.stack(x_cat_oh_batch)  # [B, maxN, ΣC]
        x_cat_onehot_meta = graphs[0].get('x_cat_onehot_meta', getattr(ds, 'xcat_meta_raw', None))
    else:
        collated_x_cat_onehot, x_cat_onehot_meta = None, None

    # ───────────────────────── 3) 그래프 텐서 pad+stack
    adj_list, spatial_pos_list, attn_bias_list = [], [], []
    in_degree_list, out_degree_list = [], []
    deg_max = getattr(ds, "deg_clip_max", 5)
    coll_attn_edge_type = {k: [] for k in graphs[0]['attn_edge_type'].keys()}
    coll_edge_input     = {k: [] for k in graphs[0]['edge_input'].keys()}

    for g in graphs:
        pad_len = max_nodes - g['num_nodes']

        adj_list.append(torch.nn.functional.pad(g['adj'], (0, pad_len, 0, pad_len)))
        spatial_pos_list.append(torch.nn.functional.pad(g['spatial_pos'], (0, pad_len, 0, pad_len), value=510))
        attn_bias_list.append(torch.nn.functional.pad(g['attn_bias'], (0, pad_len, 0, pad_len)))

        in_deg = torch.clamp(g['in_degree'], min=0, max=deg_max)
        out_deg = torch.clamp(g['out_degree'], min=0, max=deg_max)

        in_degree_list.append(torch.nn.functional.pad(in_deg , (0, pad_len)))
        out_degree_list.append(torch.nn.functional.pad(out_deg , (0, pad_len)))

        for key, t in g['attn_edge_type'].items():
            D = t.shape[-1]
            pad_t = torch.zeros((max_nodes, max_nodes, D), dtype=torch.long)
            pad_t[:g['num_nodes'], :g['num_nodes'], :] = t
            coll_attn_edge_type[key].append(pad_t)

        for key, t in g['edge_input'].items():
            max_dist, D = t.shape[2], t.shape[-1]
            pad_t = torch.zeros((max_nodes, max_nodes, max_dist, D), dtype=t.dtype)
            pad_t[:g['num_nodes'], :g['num_nodes'], :, :] = t
            coll_edge_input[key].append(pad_t)

    coll_attn_edge_type_tensor = torch.cat(
        [torch.stack(coll_attn_edge_type[k]) for k in coll_attn_edge_type], dim=-1
    )
    coll_edge_input_tensor = torch.cat(
        [torch.stack(coll_edge_input[k]) for k in coll_edge_input], dim=-1
    )

    # ───────────────────────── 4) 글로벌 피처 (이미 index/onehot로 준비됨 → 그대로 stack)
    if ds.mode == "cls_global_model":
        cat_list  = [b[3].get("global_features_cat",  torch.empty(0, dtype=torch.float32)) for b in batch]
        cont_list = [b[3].get("global_features_cont", torch.empty(0, dtype=torch.float32)) for b in batch]
    elif ds.mode == "cls_global_data":
        cat_list  = [g.get("global_features_cat",  torch.empty(0, dtype=torch.float32)) for g in graphs]
        cont_list = [g.get("global_features_cont", torch.empty(0, dtype=torch.float32)) for g in graphs]
    else:
        cat_list, cont_list = [], []

    collated_global_cat  = _stack_opt(cat_list)
    collated_global_cont = _stack_opt(cont_list)

    # ========== 🔴 여기서부터 '싱글 + 멀티'를 합쳐 단일 전역 onehot 벡터로 병합 ==========
    def _split_tokens(cell: str):
        if not isinstance(cell, str) or not cell.strip():
            return []
        # 이미 Dataset에서 normalize 되어 있지만 혹시 몰라 마지막 방어
        toks = [t.strip() for t in cell.split("+") if t.strip()]
        return [_normalize_token(t) for t in toks]  # Title-Case 정규화

    # 멀티 전역 컬럼들(예: ["Solvent"]) → 각자 멀티핫 블록 생성해 리스트에 담기
    multi_blocks = []        # 각 블록 shape: [B, |V_col|]
    multi_sizes  = []        # 각 멀티 컬럼별 |V_col|
    multi_names  = list(getattr(ds, "global_multi_cols", []))  # 예: ["Solvent"]

    for col in multi_names:
        vocab = list(ds.nominal_feature_vocab[col])                 # 토큰 순서 고정
        v2i   = {v: i for i, v in enumerate(vocab)}
        B = len(tgt_idx)
        V = len(vocab)
        block = torch.zeros((B, V), dtype=torch.float32)
        for bi, row_idx in enumerate(tgt_idx):
            cell = str(ds.data.loc[row_idx, col])
            for tok in set(_split_tokens(cell)):
                if tok in v2i:
                    block[bi, v2i[tok]] = 1.0
        multi_blocks.append(block)
        multi_sizes.append(V)

    # 싱글(기존 onehot)과 멀티(방금 만든 멀티핫) 결합
    if collated_global_cat is None:
        if multi_blocks:
            global_cat_all = torch.cat(multi_blocks, dim=-1)
        else:
            global_cat_all = torch.zeros((len(tgt_idx), 0), dtype=torch.float32)
    else:
        global_cat_all = torch.cat([collated_global_cat] + multi_blocks, dim=-1) \
                         if multi_blocks else collated_global_cat

    # 단일 메타 생성 (싱글 + 멀티 모두 포함)
    single_names = list(getattr(ds, "global_single_cols", []))                  # 예: ["pH_label","type"]
    single_sizes = list(getattr(ds, "global_single_sizes", []))                 # 예: [3,1]

    concat_names   = single_names + multi_names                                 # 예: ["pH_label","type","Solvent"]
    concat_sizes   = single_sizes + multi_sizes                                  # 예: [3,1,|V_solvent|]
    concat_offsets = np.cumsum([0] + concat_sizes[:-1]).tolist()
    concat_vocabs  = {n: list(ds.nominal_feature_vocab[n]) for n in concat_names}
    concat_types   = (["single"] * len(single_names)) + (["multi"] * len(multi_names))

    # ───────────────────────── 5) 결과 패킹
    res = {
        "adj": torch.stack(adj_list),
        "spatial_pos": torch.stack(spatial_pos_list),
        "attn_bias": torch.stack(attn_bias_list),
        "in_degree": torch.stack(in_degree_list),
        "out_degree": torch.stack(out_degree_list),
        "attn_edge_type": coll_attn_edge_type_tensor,
        "edge_input": coll_edge_input_tensor,
        "targets": torch.stack([ds.targets[i] for i in tgt_idx]),
    }
    if collated_x_cat is not None:   res["x_cat"] = collated_x_cat
    if collated_x_cont is not None:  res["x_cont"] = collated_x_cont
    if collated_x_cat_onehot is not None:
        res["x_cat_onehot"] = collated_x_cat_onehot
        if x_cat_onehot_meta is not None:
            res["x_cat_onehot_meta"] = x_cat_onehot_meta

    # ✅ 여기서 딱 한 번만 통합 전역 one-hot + 통합 메타를 넣는다
    if ds.mode in ["cls_global_data", "cls_global_model"]:
        res["global_features_cat"] = global_cat_all
        res["global_features_cat_meta"] = {
            "names": concat_names,
            "sizes": torch.tensor(concat_sizes, dtype=torch.long),
            "offsets": torch.tensor(concat_offsets, dtype=torch.long),
            "total_dim": int(sum(concat_sizes)),
            "types": concat_types,  # ["single", ..., "multi", ...]
            "vocabs": concat_vocabs,  # 각 컬럼의 클래스/토큰 리스트
        }
        if (collated_global_cont is not None) and (collated_global_cont.numel() > 0):
            res["global_features_cont"] = collated_global_cont

    if ds.target_type == "exp_spectrum":
        mask_batch = torch.as_tensor(ds.masks[tgt_idx], dtype=torch.bool).unsqueeze(-1)
        res["masks"] = mask_batch

    # collate_fn 안에서 edge 채널 메타 생성 (고정 슬라이스)
    edge_groups = [
        ("bond_type", 4),
        ("stereo", 6),
        ("is_conjugated", 2),
        ("is_in_ring", 2),
        ("is_global", 1),
    ]
    edge_sizes = torch.tensor([sz for _, sz in edge_groups], dtype=torch.long)
    edge_offsets = torch.tensor([0] + list(torch.cumsum(edge_sizes, dim=0)[:-1]), dtype=torch.long)
    res["edge_onehot_meta"] = {
        "group_names": [nm for nm, _ in edge_groups],
        "sizes": edge_sizes,  # tensor([4,6,2,2,1])
        "offsets": edge_offsets,  # tensor([0,4,10,12,14])
    }

    # print("res.keys() collated", len(res.keys()), res.keys())
    return res

##### 원자 및 글로벌 Node Feature 모드 "index", "onehot" ####
# index 모드의 경우 None type을 정의 하지 않았음 사용하지 말것 #
# 예시 idx 0 = 수소, 1 = 헬륨 .... -> 이렇게 되어서 None type 이 없음 #
# 따라서 None type 에 대한 id 가 없어서 Global Node 에 Feature 로 None을 줄수 없으니 사용하지 말것 #
# one hot 모드의 경우
# 10000 -> 수소, 01000 헬륨, 00000 -> None 임 따라서 사용 가능
