import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# ============================================================
#  헬퍼: 토큰/용매 문자열 정규화
# ============================================================

def _normalize_token(x: str) -> str:
    """공백 정리 + Title Case"""
    if x is None or not isinstance(x, str):
        return ""
    x = x.strip()
    if not x:
        return ""
    return x.lower().strip().title()


def _normalize_solvent_cell(cell: str) -> str:
    """
    'ethanol + water + ethanol' → 'Ethanol + Water' (중복 제거 / 정렬 / Title Case)
    """
    if cell is None or not isinstance(cell, str):
        return ""
    cell = cell.strip()
    if not cell:
        return ""

    toks = [t.strip() for t in cell.split("+") if t.strip()]
    norm = sorted(set(_normalize_token(t) for t in toks if t))
    return " + ".join(norm)


# ============================================================
#  Dataset 본체
# ============================================================

class UnifiedSMILESDataset(Dataset):
    """
    - 한 샘플 = (solute_graph + solvent_graph)을 합친 하나의 그래프
    - solvent 가 'solid(neat)', 'gas', 'QM' 이면 가상의 환경 그래프(1 노드) 사용
    - global node 사용 X
    """

    SPECIAL_SOLVENTS = {
        "solid(neat)": "solid",
        "solid": "solid",
        "gas": "gas",
        "qm": "qm",
        "qm calc": "qm",
    }

    def __init__(
        self,
        csv_file,
        nominal_feature_vocab,
        continuous_feature_names,
        global_cat_dim,
        global_cont_dim,
        ATOM_FEATURES_VOCAB,
        float_feature_keys,
        BOND_FEATURES_VOCAB,
        GLOBAL_FEATURE_VOCABS_dict,
        x_cat_mode: str = "index",   # "index" | "onehot"
        global_cat_mode: str = "index",  # "index" | "onehot"
        mol_col: str = "smiles",
        solvent_mol_col: str = "solvent_smiles",
        mode: str = "cls",  # "cls", "cls_global_data", "cls_global_model"
        max_nodes: int = 128,
        multi_hop_max_dist: int = 5,
        target_type: str = "default",
        intensity_normalize: str = "min_max",
        intensity_range: tuple = (200, 800),
        attn_bias_w: float = 0.0,
        ex_normalize: str = None,
        prob_normalize: str = None,
        nm_dist_mode: str = "hist",
        nm_gauss_sigma: float = 10.0,
        deg_clip_max: int = 5,
    ):
        super().__init__()

        self.mode = mode
        self.is_global = mode in ("cls_global_data", "cls_global_model")

        self.nominal_feature_vocab = nominal_feature_vocab
        self.continuous_feature_names = continuous_feature_names
        self.global_cat_dim = global_cat_dim
        self.global_cont_dim = global_cont_dim
        self.target_type = target_type
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist

        self.ex_normalize = ex_normalize
        self.prob_normalize = prob_normalize
        self.nm_dist_mode = nm_dist_mode
        self.nm_gauss_sigma = nm_gauss_sigma

        self.intensity_normalize = intensity_normalize
        self.intensity_range = intensity_range
        self.attn_bias_weight = attn_bias_w

        self.mol_col = mol_col
        self.solvent_mol_col = solvent_mol_col
        self.data = pd.read_csv(csv_file, low_memory=False)

        # ------------- (A) 명목형 컬럼 정규화 -------------
        for col in self.nominal_feature_vocab.keys():
            if col in self.data.columns:
                if col == "Solvent":
                    self.data[col] = self.data[col].apply(_normalize_solvent_cell)
                else:
                    self.data[col] = self.data[col].apply(_normalize_token)

        # Solvent용 보조 컬럼
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

        # mol_col 자동 추론 (없으면)
        if self.mol_col not in self.data.columns:
            if "smiles" in self.data.columns:
                self.mol_col = "smiles"
            elif "InChI" in self.data.columns or "inchi" in self.data.columns:
                self.mol_col = "InChI" if "InChI" in self.data.columns else "inchi"
            else:
                raise ValueError(
                    f"No molecular column found among ['smiles','InChI','inchi']. "
                    f"Found: {list(self.data.columns)[:20]}"
                )

        self.nominal_feature_info = self._build_nominal_feature_info()
        self.GLOBAL_FEATURE_VOCABS_dict = dict(GLOBAL_FEATURE_VOCABS_dict)

        self.x_cat_mode = x_cat_mode
        self.global_cat_mode = global_cat_mode

        # ---------- global_cat: single vs multi (Solvent etc.) ----------
        cols = list(self.nominal_feature_vocab.keys())

        if self.global_cat_mode == "index":
            # index 모드: 모두 single 취급
            self._global_multi_cols = []
            self._global_single_cols = cols[:]
        else:
            # onehot 모드: '+' 포함되면 multi 후보
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

        # global_cat 메타 (index 기준)
        self.global_cat_feature_order = list(self.nominal_feature_vocab.keys())
        self.global_cat_feature_sizes = [
            len(self.nominal_feature_vocab[n]) for n in self.global_cat_feature_order
        ]
        self.global_cat_feature_offsets = np.cumsum(
            [0] + self.global_cat_feature_sizes[:-1]
        ).tolist()
        self.global_cat_total_dim = sum(self.global_cat_feature_sizes)
        assert (
            self.global_cat_dim == self.global_cat_total_dim
        ), f"global_cat_dim mismatch: {self.global_cat_dim} vs {self.global_cat_total_dim}"

        # 필요한 컬럼만 로드
        self._validate_columns(csv_file)
        self.data = self.data.loc[:, self._get_all_cols_to_load()]

        # ------------- target 처리 (ex_prob / nm_distribution / exp_spectrum ...) -------------
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

        elif target_type == "exp_spectrum":
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

        # targets / masks
        if self.target_type == "exp_spectrum":
            spectrum, mask_tensor = self.process_targets()
        else:
            spectrum = self.process_targets()
            mask_tensor = None

        # ------------------------------------------------------------
        #  Solute + Solvent 그래프 생성 + 병합 (global node 없음)
        # ------------------------------------------------------------
        self.raw_graphs = []
        valid_indices = []
        expanded_targets = []
        expanded_rows = []

        for i in tqdm(range(len(self.data)), desc="Building solute+solvent graphs"):
            try:
                solute_text = self.data.loc[i, self.mol_col]

                # ---- Solvent 이름 / SMILES
                solvent_name = str(self.data.loc[i, "Solvent"]) if "Solvent" in self.data.columns else ""
                solvent_name_l = solvent_name.lower().strip()

                solvent_smile = None
                if self.solvent_mol_col in self.data.columns:
                    solvent_smile = self.data.loc[i, self.solvent_mol_col]

                # 1) solute graph
                g_solute = smiles_or_inchi_to_graph(
                    solute_text,
                    self.multi_hop_max_dist,
                    ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                    float_feature_keys=float_feature_keys,
                    BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
                )
                if g_solute is None:
                    continue

                # 2) solvent / 환경 graph
                env_kind = None
                for raw, kind in self.SPECIAL_SOLVENTS.items():
                    if solvent_name_l == raw:
                        env_kind = kind
                        break

                if env_kind is not None:
                    g_solvent = self._make_virtual_env_graph(env_kind, ATOM_FEATURES_VOCAB)
                else:
                    if (
                        solvent_smile is None
                        or not isinstance(solvent_smile, str)
                        or not solvent_smile.strip()
                    ):
                        # solvent SMILES 없으면 스킵(원하면 나중에 다른 처리)
                        continue
                    g_solvent = smiles_or_inchi_to_graph(
                        solvent_smile,
                        self.multi_hop_max_dist,
                        ATOM_FEATURES_VOCAB=ATOM_FEATURES_VOCAB,
                        float_feature_keys=float_feature_keys,
                        BOND_FEATURES_VOCAB=BOND_FEATURES_VOCAB,
                    )
                    if g_solvent is None:
                        continue

                merged = self._merge_two_graphs_bipartite(g_solute, g_solvent)
                if merged is None:
                    continue

                self.raw_graphs.append(merged)
                valid_indices.append(i)
                expanded_targets.append(spectrum[i])
                expanded_rows.append(self.data.iloc[i].copy())

            except Exception:
                # 에러 나면 해당 샘플은 스킵
                continue

        # 대상/데이터 재구성
        self.targets = (
            torch.as_tensor(np.stack(expanded_targets), dtype=torch.float32)
            if isinstance(spectrum, torch.Tensor)
            else spectrum[valid_indices]
        )

        if mask_tensor is not None:
            self.masks = mask_tensor[valid_indices]

        self.data = pd.DataFrame(expanded_rows).reset_index(drop=True)
        self.graphs = self.raw_graphs

        # ------------------------------------------------------------
        # x_cat 메타: 원자 feature 그룹만 (global node 없음)
        # ------------------------------------------------------------
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

        # ---- global single/multi 메타 (collate_fn에서 사용) ----
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

        print(
            "[META] atoms:", len(atom_names),
            "global_single:", self.global_single_cols,
            "global_multi:", self.global_multi_cols,
        )
        if len(self.graphs) > 0:
            print("[DEBUG] example graph keys:", list(self.graphs[0].keys()))

    # ============================================================
    #  Virtual env graph (solid / gas / qm)
    # ============================================================

    def _make_virtual_env_graph(self, env_kind, ATOM_FEATURES_VOCAB):
        """
        env_kind: "solid" / "gas" / "qm"
        1개의 dummy node 를 가지는 '환경 그래프' 생성
        """
        num_nodes = 1

        atom_names = [k for k, v in ATOM_FEATURES_VOCAB.items() if isinstance(v, list)]
        num_cat = len(atom_names)
        x_cat = np.zeros((num_nodes, num_cat), dtype=np.int64)

        env_token_map = {
            "solid": "ENV_SOLID",
            "gas": "ENV_GAS",
            "qm": "ENV_QM",
        }
        token = env_token_map[env_kind]
        atom_vocab = ATOM_FEATURES_VOCAB["atomic_num"]
        if token not in atom_vocab:
            atom_vocab.append(token)
        env_idx = atom_vocab.index(token)

        atomic_num_col = atom_names.index("atomic_num")
        x_cat[0, atomic_num_col] = env_idx

        x_cont = np.zeros((num_nodes, 0), dtype=np.float32)

        edge_index = np.zeros((2, 0), dtype=np.int64)
        adj = np.zeros((num_nodes, num_nodes), dtype=np.int64)

        spatial_pos = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # 가상 그래프에는 attn_edge_type / edge_input 을 비워둔다.
        # merge 시에 전체 zero tensor에서 일부를 채워 넣기 때문에 없어도 됨.
        attn_edge_type = {}
        edge_input = {}

        return {
            "num_nodes": num_nodes,
            "x_cat": x_cat,
            "x_cont": x_cont,
            "edge_index": edge_index,
            "adj": adj,
            "attn_edge_type": attn_edge_type,
            "spatial_pos": spatial_pos,
            "attn_bias": np.zeros((num_nodes, num_nodes), dtype=np.float32),
            "edge_input": edge_input,
        }

    # ============================================================
    #  Solute + Solvent 그래프 병합
    # ============================================================

    def _merge_two_graphs_bipartite(self, g1, g2):
        """
        g1: solute_graph
        g2: solvent_graph or virtual env graph

        두 그래프를 하나로 합치고,
        solute 모든 노드와 solvent 모든 노드 사이에 'is_global=1' edge 를 추가.
        """
        if g1 is None or g2 is None:
            return None

        n1 = g1["num_nodes"]
        n2 = g2["num_nodes"]
        N = n1 + n2

        # ---- 노드 피처
        x_cat = np.concatenate([g1["x_cat"], g2["x_cat"]], axis=0)
        x_cont = np.concatenate(
            [
                g1.get("x_cont", np.zeros((n1, 0), dtype=np.float32)),
                g2.get("x_cont", np.zeros((n2, 0), dtype=np.float32)),
            ],
            axis=0,
        )

        # ---- edge_index / adj
        e1 = g1["edge_index"]
        e2 = g2["edge_index"].copy()
        if e2.size > 0:
            e2 = e2 + n1
            edge_index = np.concatenate([e1, e2], axis=1)
        else:
            edge_index = e1.copy()

        adj = np.zeros((N, N), dtype=np.int64)
        adj[:n1, :n1] = g1["adj"]
        adj[n1:, n1:] = g2["adj"]

        # solute ↔ solvent fully connected (양방향)
        for i in range(n1):
            for j in range(n1, N):
                adj[i, j] = 1
                adj[j, i] = 1
                edge_index = np.concatenate(
                    [edge_index, np.array([[i, j], [j, i]], dtype=np.int64)], axis=1
                )

        # ---- spatial_pos: 기존 값 유지, cross 는 1, diag 0
        sp = np.full((N, N), 510.0, dtype=np.float32)
        sp[:n1, :n1] = g1["spatial_pos"]
        sp[n1:, n1:] = g2["spatial_pos"]
        np.fill_diagonal(sp, 0.0)
        sp[:n1, n1:] = 1.0
        sp[n1:, :n1] = 1.0

        # ---- attn_edge_type 병합
        attn_edge_type = {}
        keys = set(g1.get("attn_edge_type", {}).keys()) | set(
            g2.get("attn_edge_type", {}).keys()
        )
        for key in keys:
            t1 = g1.get("attn_edge_type", {}).get(key, None)
            t2 = g2.get("attn_edge_type", {}).get(key, None)

            if t1 is not None:
                D = t1.shape[-1]
            elif t2 is not None:
                D = t2.shape[-1]
            else:
                continue

            t = np.zeros((N, N, D), dtype=np.int64)
            if t1 is not None:
                t[:n1, :n1, :] = t1
            if t2 is not None:
                t[n1:, n1:, :] = t2

            # solute–solvent cross edge: is_global 채널 1로
            if key == "is_global":
                t[:n1, n1:, 0] = 1
                t[n1:, :n1, 0] = 1

            attn_edge_type[key] = t

        # ---- edge_input 병합 (multi-hop)
        edge_input = {}
        keys = set(g1.get("edge_input", {}).keys()) | set(
            g2.get("edge_input", {}).keys()
        )
        for key in keys:
            t1 = g1.get("edge_input", {}).get(key, None)
            t2 = g2.get("edge_input", {}).get(key, None)

            if t1 is not None:
                max_dist, D = t1.shape[2], t1.shape[-1]
            elif t2 is not None:
                max_dist, D = t2.shape[2], t2.shape[-1]
            else:
                continue

            t = np.zeros((N, N, max_dist, D), dtype=np.int64)
            if t1 is not None:
                t[:n1, :n1, :, :] = t1
            if t2 is not None:
                t[n1:, n1:, :, :] = t2
            # solute–solvent cross 부분은 0 그대로
            edge_input[key] = t

        return {
            "num_nodes": N,
            "x_cat": x_cat,
            "x_cont": x_cont,
            "edge_index": edge_index,
            "adj": adj,
            "attn_edge_type": attn_edge_type,
            "spatial_pos": sp,
            "attn_bias": np.zeros((N, N), dtype=np.float32),
            "edge_input": edge_input,
        }

    # ============================================================
    #   표준 함수들 (target, preprocess, global feature 등)
    # ============================================================

    def _build_nominal_feature_info(self):
        return {
            name: {
                "unique_values": vocab,
                "value_to_idx": {val: i for i, val in enumerate(vocab)},
            }
            for name, vocab in self.nominal_feature_vocab.items()
        }

    def _get_all_cols_to_load(self):
        # target 컬럼
        if self.target_type in ["ex_prob", "nm_distribution"]:
            target_cols = [f"ex{i}" for i in range(1, 51)] + [
                f"prob{i}" for i in range(1, 51)
            ]
        elif self.target_type == "exp_spectrum":
            nm_min, nm_max = self.intensity_range
            target_cols = [str(i) for i in range(nm_min, nm_max + 1)]
        else:
            target_cols = []

        required_cols = [self.mol_col] + target_cols

        extra_cols = []

        # Solvent, solvent_smiles, Solvent_primary_token
        if "Solvent" not in self.nominal_feature_vocab and "Solvent" in self.data.columns:
            extra_cols.append("Solvent")
        if "Solvent_primary_token" in self.data.columns:
            extra_cols.append("Solvent_primary_token")
        if self.solvent_mol_col in self.data.columns:
            extra_cols.append(self.solvent_mol_col)

        return (
            required_cols
            + list(self.nominal_feature_vocab.keys())
            + self.continuous_feature_names
            + extra_cols
        )

    def _validate_columns(self, csv_file):
        for col in self._get_all_cols_to_load():
            if col not in self.data.columns:
                raise ValueError(f"Missing required column '{col}' in {csv_file}")

    # --------------------- target 처리 --------------------------

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

            # intensity 기준 정렬
            sorted_idx = np.argsort(-prob, axis=1)
            top_idx = sorted_idx[:, :n_pairs]
            ex_top = np.take_along_axis(ex, top_idx, axis=1)
            prob_top = np.take_along_axis(prob, top_idx, axis=1)

            # eV 기준 정렬
            asc_idx = np.argsort(ex_top, axis=1)
            ex_top = np.take_along_axis(ex_top, asc_idx, axis=1)
            prob_top = np.take_along_axis(prob_top, asc_idx, axis=1)

            # ex 정규화
            if self.ex_normalize == "ex_min_max":
                ex_top = (ex_top - self.global_ex_min) / (
                    self.global_ex_max - self.global_ex_min + 1e-8
                )
            elif self.ex_normalize == "ex_std":
                ex_top = (ex_top - self.global_ex_mean) / (
                    self.global_ex_std + 1e-8
                )
            elif self.ex_normalize in ["none", None]:
                pass
            else:
                raise ValueError(f"Unknown ex_normalize: {self.ex_normalize}")

            # prob 정규화
            if self.prob_normalize == "prob_min_max":
                prob_top = (prob_top - self.global_prob_min) / (
                    self.global_prob_max - self.global_prob_min + 1e-8
                )
            elif self.prob_normalize == "prob_std":
                prob_top = (prob_top - self.global_prob_mean) / (
                    self.global_prob_std + 1e-8
                )
            elif self.prob_normalize in ["none", None]:
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
                prob = (prob - self.global_prob_min) / (
                    self.global_prob_max - self.global_prob_min + 1e-8
                )

            nm_min, nm_max = self.intensity_range
            nm = np.clip(nm, nm_min, nm_max)
            spec_len = nm_max - nm_min + 1
            out = np.zeros((len(self.data), spec_len), dtype=np.float32)

            if self.nm_dist_mode == "hist":
                for i, (row_nm, row_p) in enumerate(zip(nm, prob)):
                    for lam, p in zip(row_nm, row_p):
                        if nm_min <= lam <= nm_max:
                            out[i, lam - nm_min] += p

            elif self.nm_dist_mode == "gauss":
                bins = np.arange(nm_min, nm_max + 1)
                sigma = self.nm_gauss_sigma
                for i, (row_nm, row_p) in enumerate(zip(nm, prob)):
                    spec = np.zeros_like(bins, dtype=np.float32)
                    for lam, p in zip(row_nm, row_p):
                        if nm_min <= lam <= nm_max and p > 0:
                            kernel = np.exp(-0.5 * ((bins - lam) / sigma) ** 2)
                            kernel /= kernel.sum() + 1e-8
                            spec += p * kernel
                    out[i] = spec
            else:
                raise ValueError(f"Unknown nm_dist_mode: {self.nm_dist_mode}")

            return torch.tensor(out, dtype=torch.float32)

        elif self.target_type == "exp_spectrum":
            nm_min, nm_max = self.intensity_range
            target_cols = [str(i) for i in range(nm_min, nm_max + 1)]
            existing_cols = [col for col in target_cols if col in self.data.columns]
            missing_cols = set(target_cols) - set(existing_cols)
            if missing_cols:
                print(
                    f"[Warning] {len(missing_cols)} missing columns will be filled with zeros"
                )
            for col in missing_cols:
                self.data[col] = 0.0

            spectrum = self.data[target_cols].fillna(0.0).values

            normed = []
            masks = []
            for row in spectrum:
                mask = row != 0
                if np.sum(mask) == 0:
                    normed.append(np.zeros_like(row))
                else:
                    vals = row[mask]
                    r_min, r_max = np.min(vals), np.max(vals)
                    r_range = r_max - r_min + 1e-8
                    norm_row = np.zeros_like(row)
                    norm_row[mask] = (vals - r_min) / r_range
                    normed.append(norm_row)
                masks.append(mask)

            spectrum = np.stack(normed)
            spectrum = torch.tensor(spectrum, dtype=torch.float32)
            mask_tensor = torch.tensor(np.stack(masks), dtype=torch.bool)
            return spectrum, mask_tensor

        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")

    # --------------------- 그래프 전처리 --------------------------

    def preprocess_graph(self, graph):
        num_nodes = graph["num_nodes"]

        x_cat = torch.from_numpy(graph["x_cat"]).long()
        x_cont = torch.from_numpy(graph["x_cont"]).float()

        edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
        in_deg = torch.bincount(edge_index[1], minlength=num_nodes).long()
        out_deg = torch.bincount(edge_index[0], minlength=num_nodes).long()

        adj = torch.from_numpy(graph["adj"])
        spatial_pos = torch.from_numpy(graph["spatial_pos"]).float()
        attn_bias = torch.from_numpy(graph["attn_bias"]).float()

        attn_edge_type = {
            k: torch.from_numpy(v).long() for k, v in graph["attn_edge_type"].items()
        }
        edge_input = {
            k: torch.from_numpy(v).long() for k, v in graph["edge_input"].items()
        }

        g = {
            "x_cat": x_cat,
            "x_cont": x_cont,
            "adj": adj,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "attn_edge_type": attn_edge_type,
            "spatial_pos": spatial_pos,
            "attn_bias": attn_bias,
            "edge_input": edge_input,
            "num_nodes": num_nodes,
        }

        # ---- x_cat → onehot (원자 feature 그룹만) ----
        if getattr(self, "x_cat_mode", "index") == "onehot":
            X_idx = g["x_cat"]
            sizes = self.xcat_sizes
            pieces = []
            for c, nc in enumerate(sizes):
                col = X_idx[:, c].clamp(min=0, max=int(nc) - 1)
                pieces.append(
                    torch.nn.functional.one_hot(col, num_classes=int(nc)).float()
                )
            X_oh = torch.cat(pieces, dim=-1)
            g["x_cat_onehot"] = X_oh
            g["x_cat_onehot_meta"] = {
                "group_names": self.xcat_meta["group_names"],
                "sizes": torch.tensor(self.xcat_meta["sizes"], dtype=torch.long),
                "offsets": torch.tensor(self.xcat_meta["offsets"], dtype=torch.long),
                "total_dim": self.xcat_meta["total_dim"],
            }

        return g

    # --------------------- global feature tensor -------------------

    def _get_global_feature_cat_tensor(self, idx):
        """싱글 전역 카테고리만 인덱스 반환"""
        indices = []
        for name in self._global_single_cols:
            val = self.data.loc[idx, name]
            indices.append(self.nominal_feature_info[name]["value_to_idx"].get(val, 0))
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

    # ============================================================
    #  Dataset API
    # ============================================================

    def __getitem__(self, idx):
        tgt = self.targets[idx]
        raw_g = self.raw_graphs[idx]
        g_processed = self.preprocess_graph(raw_g)

        # global 연속형
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

        # global 카테고리 index
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
            return g_processed, tgt, idx

        elif self.mode == "cls_global_model":
            extras = {}
            if global_cat.numel() > 0:
                extras["global_features_cat"] = global_cat
            if global_cont.numel() > 0:
                extras["global_features_cont"] = global_cont
            return g_processed, tgt, idx, extras

        # cls only
        return g_processed, tgt, idx

    def __len__(self):
        return len(self.graphs)


# ============================================================
#  collate_fn
# ============================================================

def collate_fn(batch, ds: UnifiedSMILESDataset):
    # 유효 샘플만
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch:
        return None

    graphs = [b[0] for b in batch]
    tgt_idx = [b[2] for b in batch]
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

    # ---- node features
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

    # already computed onehot
    if "x_cat_onehot" in graphs[0]:
        x_cat_oh_batch = []
        for g in graphs:
            Xoh = g["x_cat_onehot"]
            x_cat_oh_batch.append(_pad2d(Xoh, max_nodes, 0.0))
        collated_x_cat_onehot = torch.stack(x_cat_oh_batch)
        x_cat_onehot_meta = graphs[0].get("x_cat_onehot_meta", ds.xcat_meta)
    else:
        collated_x_cat_onehot, x_cat_onehot_meta = None, None

    # ---- graph tensors
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

    # ---- global features (sing글 index + multi onehot) ----
    if ds.mode == "cls_global_model":
        cat_list = [
            b[3].get("global_features_cat", torch.empty(0, dtype=torch.float32))
            for b in batch
        ]
        cont_list = [
            b[3].get("global_features_cont", torch.empty(0, dtype=torch.float32))
            for b in batch
        ]
    elif ds.mode == "cls_global_data":
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

    # multi-hot (예: Solvent)
    def _split_tokens(cell: str):
        if not isinstance(cell, str) or not cell.strip():
            return []
        toks = [t.strip() for t in cell.split("+") if t.strip()]
        return [_normalize_token(t) for t in toks]

    multi_blocks = []
    multi_sizes = []
    multi_names = list(getattr(ds, "global_multi_cols", []))

    for col in multi_names:
        vocab = list(ds.nominal_feature_vocab[col])
        v2i = {v: i for i, v in enumerate(vocab)}
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

    if collated_global_cat is None:
        if multi_blocks:
            global_cat_all = torch.cat(multi_blocks, dim=-1)
        else:
            global_cat_all = torch.zeros((len(tgt_idx), 0), dtype=torch.float32)
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

    # ---- 결과 패킹 ----
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

    if collated_x_cat is not None:
        res["x_cat"] = collated_x_cat
    if collated_x_cont is not None:
        res["x_cont"] = collated_x_cont
    if collated_x_cat_onehot is not None:
        res["x_cat_onehot"] = collated_x_cat_onehot
        if x_cat_onehot_meta is not None:
            res["x_cat_onehot_meta"] = x_cat_onehot_meta

    if ds.mode in ["cls_global_data", "cls_global_model"]:
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

    if ds.target_type == "exp_spectrum":
        mask_batch = torch.as_tensor(ds.masks[tgt_idx], dtype=torch.bool).unsqueeze(-1)
        res["masks"] = mask_batch

    # edge feature 메타
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

