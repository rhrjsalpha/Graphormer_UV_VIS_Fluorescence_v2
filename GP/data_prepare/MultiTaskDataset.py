from GP.data_prepare.DataLoader_QMData_All import UnifiedSMILESDataset, _normalize_token, _normalize_solvent_cell
from GP.data_prepare.Chem_Graph_Utils import smiles_or_inchi_to_graph
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
# =========================================
#  Multi-task Dataset (full + chromophore)
# =========================================

class MultiTaskSMILESDataset(UnifiedSMILESDataset):
    """
    여러 CSV를 받아서 하나의 Dataset으로 합치는 멀티태스크 버전.
    - 각 CSV는 kind="full" 또는 "chromo" 로 지정
    - outputs:
        graph dict + targets dict + masks dict + data_kind(int: 0=full,1=chromo)
    """

    def __init__(
        self,
        csv_info_list,              # [{'path':..., 'kind':'full'/'chromo',
                                    #   'mol_col':..., 'solvent_col':...}, ...]
        nominal_feature_vocab,
        continuous_feature_names,
        global_cat_dim,
        global_cont_dim,
        ATOM_FEATURES_VOCAB,
        float_feature_keys,
        BOND_FEATURES_VOCAB,
        GLOBAL_FEATURE_VOCABS_dict,
        x_cat_mode: str = "index",
        global_cat_mode: str = "index",
        max_nodes: int = 128,
        multi_hop_max_dist: int = 5,
        intensity_range: tuple = (200, 800),
        deg_clip_max: int = 5,
        mode: str = "cls_global_data",   # global 사용 전제
    ):
        # ----- 공통 하이퍼파라미터 셋업 (기존 UnifiedSMILESDataset와 동일) -----
        super(UnifiedSMILESDataset, self).__init__()  # Dataset.__init__()

        assert mode in ("cls", "cls_global_data")
        self.mode = mode
        self.is_global = (mode == "cls_global_data")

        self.nominal_feature_vocab = nominal_feature_vocab
        self.continuous_feature_names = continuous_feature_names
        self.global_cat_dim = global_cat_dim
        self.global_cont_dim = global_cont_dim

        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist
        self.intensity_range = intensity_range
        self.deg_clip_max = int(deg_clip_max)

        self.x_cat_mode = x_cat_mode
        self.global_cat_mode = global_cat_mode

        self.GLOBAL_FEATURE_VOCABS_dict = dict(GLOBAL_FEATURE_VOCABS_dict)

        # solute/solvent 컬럼 이름 통일용
        self.solute_mol_col = "solute_mol"
        self.solvent_mol_col = "solvent_mol"

        # ----- 여러 CSV 읽어서 하나의 DataFrame으로 합치기 -----
        frames = []
        csv_sample_counts = {}

        for info in csv_info_list:
            print(info)
            path = info["path"]
            kind = info["kind"]  # "full" or "chromo"
            mol_col = info.get("mol_col", "smiles")
            solv_col = info.get("solvent_col", "solvent_smiles")

            df = pd.read_csv(path, low_memory=False)
            csv_sample_counts[path] = len(df)

            # kind 라벨
            df["data_kind"] = kind

            # 공통 컬럼 이름 만들기
            df[self.solute_mol_col] = df[mol_col]
            df[self.solvent_mol_col] = df[solv_col]

            frames.append(df)

        self.data = pd.concat(frames, ignore_index=True)
        self.csv_sample_counts = csv_sample_counts
        self.num_full_samples = int((self.data["data_kind"] == "full").sum())
        self.num_chromo_samples = int((self.data["data_kind"] == "chromo").sum())

        # ---------- 명목형 컬럼 정규화 (기존과 동일) ----------
        for col in self.nominal_feature_vocab.keys():
            if col in self.data.columns:
                if col == "Solvent":
                    self.data[col] = self.data[col].apply(_normalize_solvent_cell)
                else:
                    self.data[col] = self.data[col].apply(_normalize_token)

        if "Solvent" in self.data.columns:
            self.data["Solvent"] = self.data["Solvent"].apply(_normalize_solvent_cell)

            def _pick_primary_token(cell: str) -> str:
                if cell is None or not isinstance(cell, str) or not cell.strip():
                    return "Unknown"
                first = cell.split("+")[0].strip()
                return _normalize_token(first)

            self.data["Solvent_primary_token"] = self.data["Solvent"].apply(
                _pick_primary_token
            )

        # ----- nominal feature 메타 -----
        self.nominal_feature_info = self._build_nominal_feature_info()

        cols = list(self.nominal_feature_vocab.keys())
        if self.global_cat_mode == "index":
            self._global_multi_cols = []
            self._global_single_cols = cols[:]
        else:
            plus_mask = {
                c: (
                    c in self.data.columns
                    and self.data[c].astype(str).str.contains(r"\+").any()
                )
                for c in cols
            }
            plus_multi = {c for c in cols if plus_mask.get(c, False)}
            base_multi = (
                {"Solvent"} if "Solvent" in self.GLOBAL_FEATURE_VOCABS_dict else set()
            )
            self._global_multi_cols = sorted(base_multi | plus_multi)
            self._global_single_cols = [
                c for c in cols if c not in self._global_multi_cols
            ]

        # global_cat 메타
        self.global_cat_feature_order = list(self.nominal_feature_vocab.keys())
        self.global_cat_feature_sizes = [
            len(self.nominal_feature_vocab[n]) for n in self.global_cat_feature_order
        ]
        self.global_cat_feature_offsets = np.cumsum(
            [0] + self.global_cat_feature_sizes[:-1]
        ).tolist()
        self.global_cat_total_dim = sum(self.global_cat_feature_sizes)
        assert self.global_cat_dim == self.global_cat_total_dim

        # ---------- target (multi-task) ----------
        self._build_multitask_targets()

        # ---------- solute + solvent 그래프 생성 (기존 로직 재사용) ----------
        self.raw_graphs = []
        valid_indices = []
        expanded_rows = []

        n_rows = len(self.data)
        c_ok = 0
        c_fail_solute_graph = 0
        c_special_env = 0
        c_missing_solvent_smiles = 0
        c_fail_solvent_graph = 0
        c_fail_merge = 0

        def _detect_env_kind(solvent_name_l: str, solvent_smile_raw):
            candidates = []
            if isinstance(solvent_name_l, str) and solvent_name_l:
                candidates.append(solvent_name_l.strip().lower())
            if isinstance(solvent_smile_raw, str) and solvent_smile_raw.strip():
                candidates.append(solvent_smile_raw.strip().lower())

            for raw, kind in self.SPECIAL_SOLVENTS.items():
                raw_l = raw.strip().lower()
                if any(c == raw_l for c in candidates):
                    return kind
            return None

        for i in tqdm(range(len(self.data)), desc="Building solute+solvent graphs (multi-task)"):
            try:
                solute_text = self.data.loc[i, self.solute_mol_col]

                solvent_name = (
                    str(self.data.loc[i, "Solvent"])
                    if "Solvent" in self.data.columns
                    else ""
                )
                solvent_name_l = solvent_name.lower().strip()

                solvent_smile = (
                    self.data.loc[i, self.solvent_mol_col]
                    if self.solvent_mol_col in self.data.columns
                    else None
                )

                # 1) solute graph
                g_solute = smiles_or_inchi_to_graph(
                    solute_text,
                    self.multi_hop_max_dist,
                    ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                    float_feature_keys=float_feature_keys,
                    BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
                )
                if g_solute is None:
                    c_fail_solute_graph += 1
                    continue

                # 2) solvent/env graph
                env_kind = _detect_env_kind(solvent_name_l, solvent_smile)

                if env_kind is not None:
                    g_solvent = self._make_virtual_env_graph(env_kind, ATOM_FEATURES_VOCAB)
                    c_special_env += 1
                else:
                    if (
                        solvent_smile is None
                        or not isinstance(solvent_smile, str)
                        or not solvent_smile.strip()
                    ):
                        c_missing_solvent_smiles += 1
                        continue

                    g_solvent = smiles_or_inchi_to_graph(
                        solvent_smile,
                        self.multi_hop_max_dist,
                        ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                        float_feature_keys=float_feature_keys,
                        BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
                    )
                    if g_solvent is None:
                        c_fail_solvent_graph += 1
                        continue

                merged = self._merge_two_graphs_bipartite(g_solute, g_solvent)
                if merged is None:
                    c_fail_merge += 1
                    continue

                self.raw_graphs.append(merged)
                valid_indices.append(i)
                expanded_rows.append(self.data.iloc[i].to_dict())
                c_ok += 1

            except Exception:
                c_fail_merge += 1
                continue

        print("=== MultiTaskSMILESDataset build summary ===")
        print(f" total rows                 : {n_rows}")
        print(f"  OK                        : {c_ok}")
        print(f"  fail solute graph (RDKit) : {c_fail_solute_graph}")
        print(f"  special env (solid/gas/qm): {c_special_env}")
        print(f"  missing solvent_smiles    : {c_missing_solvent_smiles}")
        print(f"  fail solvent graph        : {c_fail_solvent_graph}")
        print(f"  fail merge/others         : {c_fail_merge}")

        if len(valid_indices) == 0:
            raise RuntimeError("No valid graphs built in MultiTaskSMILESDataset")

        # 유효한 인덱스만 타겟/마스크 선택
        self.valid_indices = np.array(valid_indices, dtype=np.int64)
        self.data = pd.DataFrame(expanded_rows).reset_index(drop=True)
        self.graphs = self.raw_graphs

        for key in self.targets.keys():
            self.targets[key] = self.targets[key][self.valid_indices]
            self.target_masks[key] = self.target_masks[key][self.valid_indices]

        # ----- x_cat meta / global meta (기존과 거의 동일) -----
        atom_names = [k for k, v in ATOM_FEATURES_VOCAB.items() if isinstance(v, list)]
        atom_sizes = [len(ATOM_FEATURES_VOCAB[n]) for n in atom_names]

        self.xcat_group_names = atom_names
        self.xcat_sizes = atom_sizes
        self.xcat_offsets = np.cumsum([0] + atom_sizes[:-1]).tolist()
        self.xcat_meta = {
            "group_names": self.xcat_group_names,
            "sizes": self.xcat_sizes,
            "offsets": self.xcat_offsets,
            "total_dim": int(sum(self.xcat_sizes)),
        }

        self.global_single_cols = getattr(self, "_global_single_cols", [])
        self.global_single_sizes = [
            len(self.nominal_feature_vocab[c]) for c in self.global_single_cols
        ]
        self.global_single_offsets = np.cumsum(
            [0] + self.global_single_sizes[:-1]
        ).tolist()

        self.global_multi_cols = getattr(self, "_global_multi_cols", [])
        self.global_multi_sizes = {
            c: len(self.nominal_feature_vocab[c]) for c in self.global_multi_cols
        }

    # ---------- 멀티태스크용 target/mask 생성 ----------
    def _build_multitask_targets(self):
        nm_min, nm_max = self.intensity_range
        nm_grid = np.arange(nm_min, nm_max + 1)
        N = len(self.data)

        spec = np.zeros((N, len(nm_grid)), dtype=np.float32)
        spec_mask = np.zeros((N,), dtype=bool)

        lam_abs = np.zeros((N,), dtype=np.float32)
        lam_abs_mask = np.zeros((N,), dtype=bool)

        lam_emi = np.zeros((N,), dtype=np.float32)
        lam_emi_mask = np.zeros((N,), dtype=bool)

        qy = np.zeros((N,), dtype=np.float32)
        qy_mask = np.zeros((N,), dtype=bool)

        life = np.zeros((N,), dtype=np.float32)
        life_mask = np.zeros((N,), dtype=bool)

        kind_series = self.data["data_kind"].astype(str)

        # --- full 데이터 처리: 스펙트럼 + λmax 계산 ---
        full_idx = kind_series == "full"
        if full_idx.any():
            target_cols = [str(i) for i in nm_grid if str(i) in self.data.columns]
            full_spec_raw = self.data.loc[full_idx, target_cols].fillna(0.0).values

            # 필요하면 normalization 적용 (지금은 min-max)
            normed = []
            for row in full_spec_raw:
                mask = row != 0
                if mask.sum() == 0:
                    normed.append(np.zeros_like(row))
                else:
                    vals = row[mask]
                    r_min, r_max = np.min(vals), np.max(vals)
                    r_range = r_max - r_min + 1e-8
                    tmp = np.zeros_like(row)
                    tmp[mask] = (vals - r_min) / r_range
                    normed.append(tmp)
            normed = np.stack(normed)
            spec[full_idx, : normed.shape[1]] = normed
            spec_mask[full_idx] = True

            # λmax (emission 기준; ABS 따로 있으면 ABS도 가능)
            idx_max = normed.argmax(axis=1)
            lam_emi[full_idx] = nm_grid[idx_max]
            lam_emi_mask[full_idx] = True

        # --- chromophore 데이터 처리: Abs/Em λmax, QY, Lifetime ---
        chromo_idx = kind_series == "chromo"
        if chromo_idx.any():
            if "Absorption max (nm)" in self.data.columns:
                vals = pd.to_numeric(
                    self.data.loc[chromo_idx, "Absorption max (nm)"],
                    errors="coerce",
                )
                mask_valid = vals.notna()
                lam_abs[chromo_idx] = vals.fillna(0).values
                lam_abs_mask[chromo_idx] = mask_valid.values

            if "Emission max (nm)" in self.data.columns:
                vals = pd.to_numeric(
                    self.data.loc[chromo_idx, "Emission max (nm)"],
                    errors="coerce",
                )
                mask_valid = vals.notna()
                lam_emi[chromo_idx] = vals.fillna(0).values
                lam_emi_mask[chromo_idx] = mask_valid.values

            if "Quantum yield" in self.data.columns:
                vals = pd.to_numeric(
                    self.data.loc[chromo_idx, "Quantum yield"],
                    errors="coerce",
                )
                mask_valid = vals.notna()
                qy[chromo_idx] = vals.fillna(0).values
                qy_mask[chromo_idx] = mask_valid.values

            if "Lifetime (ns)" in self.data.columns:
                vals = pd.to_numeric(
                    self.data.loc[chromo_idx, "Lifetime (ns)"],
                    errors="coerce",
                )
                mask_valid = vals.notna()
                life[chromo_idx] = vals.fillna(0).values
                life_mask[chromo_idx] = mask_valid.values

        # torch로 저장
        self.targets = {
            "spectrum": torch.tensor(spec, dtype=torch.float32),
            "lam_abs":  torch.tensor(lam_abs, dtype=torch.float32).unsqueeze(-1),
            "lam_emi":  torch.tensor(lam_emi, dtype=torch.float32).unsqueeze(-1),
            "qy":       torch.tensor(qy, dtype=torch.float32).unsqueeze(-1),
            "life":     torch.tensor(life, dtype=torch.float32).unsqueeze(-1),
        }
        self.target_masks = {
            "spectrum": torch.tensor(spec_mask, dtype=torch.bool),
            "lam_abs":  torch.tensor(lam_abs_mask, dtype=torch.bool),
            "lam_emi":  torch.tensor(lam_emi_mask, dtype=torch.bool),
            "qy":       torch.tensor(qy_mask, dtype=torch.bool),
            "life":     torch.tensor(life_mask, dtype=torch.bool),
        }

    # ---------- Dataset API (getitem / len) ----------
    def __getitem__(self, idx):
        raw_g = self.raw_graphs[idx]
        g_processed = self.preprocess_graph(raw_g)

        # global features (기존 UnifiedSMILESDataset.__getitem__ 참고)
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

        global_cat_idx = self._get_global_feature_cat_tensor(idx).long()

        def _idx_to_onehot(idx_vec: torch.Tensor) -> torch.Tensor:
            if idx_vec.numel() == 0:
                return torch.zeros(0, dtype=torch.float32)
            sizes = [len(self.nominal_feature_vocab[n]) for n in self._global_single_cols]
            pieces = []
            for f, nc in enumerate(sizes):
                i = idx_vec[f].clamp(min=0)
                pieces.append(
                    torch.nn.functional.one_hot(i, num_classes=int(nc)).float()
                )
            return torch.cat(pieces, dim=-1) if pieces else torch.zeros(
                0, dtype=torch.float32
            )

        global_cat = (
            _idx_to_onehot(global_cat_idx)
            if (self.global_cat_mode == "onehot")
            else global_cat_idx
        )

        if self.mode == "cls_global_data":
            if global_cat.numel() > 0:
                g_processed["global_features_cat"] = global_cat
            if global_cont.numel() > 0:
                g_processed["global_features_cont"] = global_cont

        # 멀티타겟
        sample_targets = {k: v[idx] for k, v in self.targets.items()}
        sample_masks   = {k: v[idx] for k, v in self.target_masks.items()}

        # data_kind (0=full,1=chromo)
        kind_str = str(self.data.loc[idx, "data_kind"])
        kind_id = 0 if kind_str == "full" else 1

        return g_processed, sample_targets, sample_masks, kind_id

    def __len__(self):
        return len(self.graphs)

# =========================================
#  Multi-task collate_fn
# =========================================

def multitask_collate_fn(batch, ds: MultiTaskSMILESDataset):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch:
        return None

    graphs = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    masks_list   = [b[2] for b in batch]
    kind_list    = [b[3] for b in batch]   # 0=full,1=chromo

    max_nodes = max(g["num_nodes"] for g in graphs) if graphs else 0

    def _pad2d(x, total_rows, pad_value=0):
        pad = total_rows - x.shape[0]
        return (
            torch.nn.functional.pad(x, (0, 0, 0, pad), value=pad_value)
            if pad > 0
            else x
        )

    def _stack_opt(list_of_tensors):
        if len(list_of_tensors) == 0:
            return None
        if list_of_tensors[0].numel() == 0:
            return torch.zeros(
                (len(list_of_tensors), 0), dtype=list_of_tensors[0].dtype
            )
        return torch.stack(list_of_tensors)

    collated_x_cat = (
        _stack_opt([_pad2d(g["x_cat"], max_nodes, 0) for g in graphs])
        if "x_cat" in graphs[0]
        else None
    )
    collated_x_cont = (
        _stack_opt([_pad2d(g["x_cont"], max_nodes, 0.0) for g in graphs])
        if "x_cont" in graphs[0]
        else None
    )

    if "x_cat_onehot" in graphs[0]:
        x_cat_oh_batch = []
        for g in graphs:
            Xoh = g["x_cat_onehot"]
            x_cat_oh_batch.append(_pad2d(Xoh, max_nodes, 0.0))
        collated_x_cat_onehot = torch.stack(x_cat_oh_batch)
        x_cat_onehot_meta = graphs[0].get("x_cat_onehot_meta", ds.xcat_meta)
    else:
        collated_x_cat_onehot, x_cat_onehot_meta = None, None

    adj_list, spatial_pos_list, attn_bias_list = [], [], []
    in_degree_list, out_degree_list = [], []
    deg_max = getattr(ds, "deg_clip_max", 5)

    coll_attn_edge_type = {k: [] for k in graphs[0]["attn_edge_type"].keys()}
    coll_edge_input = {k: [] for k in graphs[0]["edge_input"].keys()}

    for g in graphs:
        pad_len = max_nodes - g["num_nodes"]

        adj_list.append(
            torch.nn.functional.pad(g["adj"], (0, pad_len, 0, pad_len))
        )
        spatial_pos_list.append(
            torch.nn.functional.pad(
                g["spatial_pos"], (0, pad_len, 0, pad_len), value=510
            )
        )
        attn_bias_list.append(
            torch.nn.functional.pad(g["attn_bias"], (0, pad_len, 0, pad_len))
        )

        in_deg = torch.clamp(g["in_degree"], min=0, max=deg_max)
        out_deg = torch.clamp(g["out_degree"], min=0, max=deg_max)
        in_degree_list.append(torch.nn.functional.pad(in_deg, (0, pad_len)))
        out_degree_list.append(torch.nn.functional.pad(out_deg, (0, pad_len)))

        for key, t in g["attn_edge_type"].items():
            D = t.shape[-1]
            pad_t = torch.zeros((max_nodes, max_nodes, D), dtype=torch.long)
            pad_t[: g["num_nodes"], : g["num_nodes"], :] = t
            coll_attn_edge_type[key].append(pad_t)

        for key, t in g["edge_input"].items():
            max_dist, D = t.shape[2], t.shape[-1]
            pad_t = torch.zeros(
                (max_nodes, max_nodes, max_dist, D), dtype=t.dtype
            )
            pad_t[: g["num_nodes"], : g["num_nodes"], :, :] = t
            coll_edge_input[key].append(pad_t)

    coll_attn_edge_type_tensor = torch.cat(
        [torch.stack(coll_attn_edge_type[k]) for k in coll_attn_edge_type], dim=-1
    )
    coll_edge_input_tensor = torch.cat(
        [torch.stack(coll_edge_input[k]) for k in coll_edge_input], dim=-1
    )

    # ---- global features ----
    if ds.mode == "cls_global_data":
        cat_list = [
            g.get("global_features_cat", torch.empty(0, dtype=torch.float32))
            for g in graphs
        ]
        cont_list = [
            g.get("global_features_cont", torch.empty(0, dtype=torch.float32))
            for g in graphs
        ]
    else:
        cat_list, cont_list = [], []

    collated_global_cat = _stack_opt(cat_list)
    collated_global_cont = _stack_opt(cont_list)

    # multi-hot 처리 (Solvent 등) – 기존 collate_fn과 동일
    def _split_tokens(cell: str):
        if not isinstance(cell, str) or not cell.strip():
            return []
        toks = [t.strip() for t in cell.split("+") if t.strip()]
        return [_normalize_token(t) for t in toks]

    # 여기서는 ds.data 를 직접 쓰지 않고, ds.global_multi_cols 메타만 사용 (간단화)
    multi_blocks = []
    multi_sizes = []
    multi_names = list(getattr(ds, "global_multi_cols", []))
    B = len(batch)

    for col in multi_names:
        vocab = list(ds.nominal_feature_vocab[col])
        v2i = {v: i for i, v in enumerate(vocab)}
        V = len(vocab)
        block = torch.zeros((B, V), dtype=torch.float32)
        # batch 안의 idx 순서대로 ds.data 에 접근하기는 조금 복잡해서,
        # multi-hot global 은 일단 사용하지 않는 방향으로 가도 됨.
        # 필요하면 여기에서 sample 별 row_idx 를 MultiTaskSMILESDataset 에서 넘겨주도록 확장.

        multi_blocks.append(block)
        multi_sizes.append(V)

    if collated_global_cat is None:
        if multi_blocks:
            global_cat_all = torch.cat(multi_blocks, dim=-1)
        else:
            global_cat_all = torch.zeros((B, 0), dtype=torch.float32)
    else:
        global_cat_all = (
            torch.cat([collated_global_cat] + multi_blocks, dim=-1)
            if multi_blocks
            else collated_global_cat
        )

    single_names = list(getattr(ds, "global_single_cols", []))
    single_sizes = list(getattr(ds, "global_single_sizes", []))
    concat_names = single_names + multi_names
    concat_sizes = single_sizes + multi_sizes
    concat_offsets = np.cumsum([0] + concat_sizes[:-1]).tolist()
    concat_vocabs = {n: list(ds.nominal_feature_vocab[n]) for n in concat_names}
    concat_types = (["single"] * len(single_names)) + (["multi"] * len(multi_names))

    # ----- targets & masks 스택 -----
    target_batch = {}
    mask_batch = {}
    for key in targets_list[0].keys():
        target_batch[key] = torch.stack([t[key] for t in targets_list])
        mask_batch[key] = torch.stack([m[key] for m in masks_list])

    kind_tensor = torch.tensor(kind_list, dtype=torch.long)

    res = {
        "adj": torch.stack(adj_list),
        "spatial_pos": torch.stack(spatial_pos_list),
        "attn_bias": torch.stack(attn_bias_list),
        "in_degree": torch.stack(in_degree_list),
        "out_degree": torch.stack(out_degree_list),
        "attn_edge_type": coll_attn_edge_type_tensor,
        "edge_input": coll_edge_input_tensor,
        "targets": target_batch,
        "target_masks": mask_batch,
        "data_kind": kind_tensor,   # 0=full,1=chromo
    }

    if collated_x_cat is not None:
        res["x_cat"] = collated_x_cat
    if collated_x_cont is not None:
        res["x_cont"] = collated_x_cont
    if collated_x_cat_onehot is not None:
        res["x_cat_onehot"] = collated_x_cat_onehot
        if x_cat_onehot_meta is not None:
            res["x_cat_onehot_meta"] = x_cat_onehot_meta

    if ds.mode == "cls_global_data":
        res["global_features_cat"] = global_cat_all
        res["global_features_cat_meta"] = {
            "names": concat_names,
            "sizes": torch.tensor(concat_sizes, dtype=torch.long),
            "offsets": torch.tensor(concat_offsets, dtype=torch.long),
            "total_dim": int(sum(concat_sizes)),
            "types": concat_types,
            "vocabs": concat_vocabs,
        }
        if collated_global_cont is not None and collated_global_cont.numel() > 0:
            res["global_features_cont"] = collated_global_cont

    # edge meta 그대로
    edge_groups = [
        ("bond_type", 4),
        ("stereo", 6),
        ("is_conjugated", 2),
        ("is_in_ring", 2),
        ("is_global", 1),
    ]

    edge_sizes = torch.tensor([sz for _, sz in edge_groups], dtype=torch.long)
    edge_offsets = torch.tensor(
        [0] + list(torch.cumsum(edge_sizes, dim=0)[:-1]), dtype=torch.long
    )
    res["edge_onehot_meta"] = {
        "group_names": [nm for nm, _ in edge_groups],
        "sizes": edge_sizes,
        "offsets": edge_offsets,
    }

    return res

def print_edges_for_sample(batch, sample_idx: int = 0, max_print: int = 40):
    """
    한 배치에서 sample_idx 번째 그래프의 edge 정보 출력
    - adj > 0 인 모든 edge를 찾고
    - edge_onehot_meta / attn_edge_type 를 이용해서
      bond edge 인지, global edge 인지 표시
    """
    print("\n================ EDGE DEBUG ================")
    print(f"Sample index in batch: {sample_idx}")
    print("-------------------------------------------")

    adj  = batch["adj"][sample_idx]                 # (N, N)
    attn = batch["attn_edge_type"][sample_idx]      # (N, N, edge_dim)
    meta = batch["edge_onehot_meta"]

    group_names = meta["group_names"]              # ['bond_type','stereo',...,'is_global']
    sizes       = meta["sizes"]                    # tensor([...])
    offsets     = meta["offsets"]                  # tensor([...])

    if "is_global" not in group_names:
        print("⚠ is_global edge feature not present in this batch")
        glob_off = None
    else:
        global_gidx = group_names.index("is_global")
        glob_off = int(offsets[global_gidx])

    # group index 찾기
    bond_gidx   = group_names.index("bond_type")
    global_gidx = group_names.index("is_global")

    bond_off   = int(offsets[bond_gidx].item())
    bond_size  = int(sizes[bond_gidx].item())
    glob_off   = int(offsets[global_gidx].item())
    # is_global size 는 1 이라서 glob_off 만 있으면 됨

    N = adj.shape[0]
    global_node = N - 1  # 우리 코드에서 마지막 노드가 global

    # adj > 0 인 edge 모두 찾기
    edges = (adj > 0).nonzero(as_tuple=False)      # (E, 2), [src, dst]
    print(f"total edges (adj>0): {edges.shape[0]}")
    print(f"global node index  : {global_node}")

    # 앞에서 몇 개만 출력
    print(f"\nFirst {min(max_print, edges.shape[0])} edges:")
    for e_idx in range(min(max_print, edges.shape[0])):
        src = int(edges[e_idx, 0])
        dst = int(edges[e_idx, 1])

        # bond_type one-hot 벡터
        bond_vec = attn[src, dst, bond_off: bond_off + bond_size]
        is_bond  = bool(bond_vec.sum().item() > 0)

        # is_global flag
        is_glob  = bool(attn[src, dst, glob_off].item() == 1)

        edge_type_str = []
        if is_bond:
            bt_id = int(bond_vec.argmax().item())
            edge_type_str.append(f"bond(type={bt_id})")
        if is_glob:
            edge_type_str.append("global")

        if not edge_type_str:
            edge_type_str.append("virtual/none")

        # global node 관여 여부
        mark = ""
        if src == global_node or dst == global_node:
            mark = "  <-- involves GLOBAL NODE"

        print(f"  ({src:2d} -> {dst:2d}) : {', '.join(edge_type_str)}{mark}")

    print("===========================================\n")

def print_one_sample_from_batch(batch, sample_idx: int = 0):
    print("\n================ SAMPLE DEBUG ================")
    print(f"Sample index in batch: {sample_idx}")
    print("---------------------------------------------")

    # 1) data_kind (0 = full, 1 = chromo)
    dk = batch["data_kind"][sample_idx].item()
    print(f"data_kind         : {dk}  (0=full spectrum, 1=chromophore)")

    # 2) 그래프 기본 정보
    adj = batch["adj"][sample_idx]                 # (N, N)
    sp  = batch["spatial_pos"][sample_idx]        # (N, N)
    indeg = batch["in_degree"][sample_idx]        # (N,)
    outdeg = batch["out_degree"][sample_idx]      # (N,)

    num_nodes = adj.shape[0]
    print(f"num_nodes         : {num_nodes}")
    print(f"in_degree (0:10)  : {indeg[:10].tolist()}")
    print(f"out_degree(0:10)  : {outdeg[:10].tolist()}")

    # 3) 노드 카테고리 / 연속형 feature
    if "x_cat" in batch:
        x_cat = batch["x_cat"][sample_idx]        # (N, F_cat)
        print(f"x_cat shape       : {tuple(x_cat.shape)}")
        print(f"x_cat[0]          : {x_cat[0].tolist()}")

    if "x_cont" in batch:
        x_cont = batch["x_cont"][sample_idx]      # (N, F_cont)
        print(f"x_cont shape      : {tuple(x_cont.shape)}")
        if x_cont.shape[1] > 0:
            print(f"x_cont[0]        : {x_cont[0].tolist()}")

    # 4) global feature
    if "global_features_cat" in batch:
        g_cat = batch["global_features_cat"][sample_idx]
        print(f"global_cat shape  : {tuple(g_cat.shape)}")
        print(f"global_cat (first 10 dims): {g_cat[:10].tolist()}")

    # 5) 타겟들 (멀티태스크)
    print("\n--- Targets ---")
    for name, tensor in batch["targets"].items():
        t = tensor[sample_idx]
        # 너무 길면 앞부분만
        if t.ndim == 1:
            preview = t[:10].tolist()
        else:
            preview = t.view(-1)[:10].tolist()
        print(f"{name:8s} shape={tuple(t.shape)}, sample[:10]={preview}")

    # 6) 마스크들 (어떤 loss를 쓸지 확인)
    print("\n--- Target masks ---")
    for name, tensor in batch["target_masks"].items():
        m = tensor[sample_idx]
        if m.ndim == 1:
            preview = m[:10].tolist()
        else:
            preview = m.view(-1)[:10].tolist()
        print(f"{name:8s} shape={tuple(m.shape)}, mask[:10]={preview}")

    print("==============================================\n")

def make_subset_csv(
    src_csv: str,
    dst_csv: str,
    n_rows: int = 100,
    random: bool = False,
    seed: int = 42,
):
    """
    src_csv에서 n_rows 만큼 잘라 dst_csv로 저장
    """
    df = pd.read_csv(src_csv, low_memory=False)

    if random:
        df = df.sample(n=min(n_rows, len(df)), random_state=seed)
    else:
        df = df.iloc[:n_rows]

    df.to_csv(dst_csv, index=False)
    print(f"[OK] subset csv saved: {dst_csv}  (rows={len(df)})")

if __name__ == "__main__":
    import os
    from Pre_Defined_Vocab_Generator import generate_graphormer_config

    full_train_src  = r"C:\Users\kogun\PycharmProjects\Graphormer_UV_VIS_Fluorescence_v2\Data\EM_with_solvent_smiles_test.csv"
    chromo_csv_src  = r"C:\Users\kogun\PycharmProjects\Graphormer_UV_VIS_Fluorescence_v2\Data\DB for chromophore_Sci_Data_rev02.csv"

    full_train = full_train_src.replace(".csv", "_sample100.csv")
    chromo_csv = chromo_csv_src.replace(".csv", "_sample100.csv")

    # 100 row subset 생성 (이미 있으면 다시 안 만들어도 됨)
    if not os.path.exists(full_train):
        make_subset_csv(full_train_src, full_train, n_rows=100)

    if not os.path.exists(chromo_csv):
        make_subset_csv(chromo_csv_src, chromo_csv, n_rows=100)

    csv_info_list = [
        {
            "path": full_train,
            "kind": "full",
            "mol_col": "InChI",
            "solvent_col": "solvent_smiles",
        },
        {
            "path": chromo_csv,
            "kind": "chromo",
            "mol_col": "Chromophore",
            "solvent_col": "Solvent",   # 이 컬럼이 solvent SMILES 라고 가정
        },
    ]

    # vocab 생성은 예전처럼 full + chromo 둘 다 넣어서
    dataset_path_list = [full_train, chromo_csv]
    config = generate_graphormer_config(
        dataset_path_list=dataset_path_list,
        mode="cls_global_data",
        mol_col="smiles",                # 여기서는 solute 컬럼 이름만 맞추면 됨
        target_type="exp_spectrum",
        intensity_range=(200, 800),
        global_feature_order=["pH_label"],
        global_multihot_cols={},
        continuous_feature_names=[],
    )

    ATOM_FEATURES_VOCAB = config["ATOM_FEATURES_VOCAB"]
    BOND_FEATURES_VOCAB = config["BOND_FEATURES_VOCAB"]
    GLOBAL_FEATURE_VOCABS_dict = config["GLOBAL_FEATURE_VOCABS_dict"]

    nominal_feature_vocab = config["nominal_feature_vocab"]
    continuous_feature_names = config["continuous_feature_names"]
    float_feature_keys = config["ATOM_FLOAT_FEATURE_KEYS"]

    global_cat_dim = config["global_cat_dim"]
    global_cont_dim = config["global_cont_dim"]
    multi_hop_max_dist = config["multi_hop_max_dist"]
    intensity_range = config["intensity_range"]

    ds = MultiTaskSMILESDataset(
        csv_info_list=csv_info_list,
        nominal_feature_vocab=nominal_feature_vocab,
        continuous_feature_names=continuous_feature_names,
        global_cat_dim=global_cat_dim,
        global_cont_dim=global_cont_dim,
        ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
        float_feature_keys=float_feature_keys,
        BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
        GLOBAL_FEATURE_VOCABS_dict=GLOBAL_FEATURE_VOCABS_dict,
        x_cat_mode="index",
        global_cat_mode="onehot",
        max_nodes=128,
        multi_hop_max_dist=multi_hop_max_dist,
        intensity_range=intensity_range,
    )

    print("num_full_samples   :", ds.num_full_samples)
    print("num_chromo_samples :", ds.num_chromo_samples)
    print("csv_sample_counts  :", ds.csv_sample_counts)

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda b: multitask_collate_fn(b, ds),
    )

    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("targets keys:", batch["targets"].keys())
    print("data_kind:", batch["data_kind"])

    print_one_sample_from_batch(batch, sample_idx=0)
    print_edges_for_sample(batch, sample_idx=0, max_print=40)
