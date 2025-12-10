import pandas as pd

def normalize_token(x: str) -> str:
    """Solvent 이름을 normalize(Ethanol, Water 등)"""
    if x is None or not isinstance(x, str):
        return ""
    x = x.strip()
    if not x:
        return ""
    return x.lower().strip().title()

def normalize_solvent_cell(cell: str) -> str:
    """
    'ethanol + water' → 'Ethanol + Water'
    (여기서는 정규화만 하고, 실제 매핑에서는 '+' 있으면 버림)
    """
    if cell is None or not isinstance(cell, str):
        return ""
    cell = cell.strip()
    if not cell:
        return ""
    toks = [t.strip() for t in cell.split("+") if t.strip()]
    norm = sorted(set(normalize_token(t) for t in toks))
    return " + ".join(norm)


def add_solvent_smiles(db_csv_path, solvent_smiles_csv_path, save_path):
    """DB CSV에 solvent_smiles 정보를 추가하여 저장 (+ 포함된 혼합 용매는 매핑 안 함)"""
    # DB 읽기
    df = pd.read_csv(db_csv_path)

    # Solvent-SMILES 매핑 읽기
    solv_df = pd.read_csv(solvent_smiles_csv_path)

    # 정규화하여 mapping key 통일
    solv_df["solvent_norm"] = solv_df["solvent"].apply(normalize_token)
    solv_df = solv_df.set_index("solvent_norm")

    # DB 쪽 solvent 정규화
    df["Solvent_norm"] = df["Solvent"].apply(normalize_solvent_cell)

    # 매핑 함수 정의
    def map_solvent_to_smiles(cell):
        if not isinstance(cell, str) or not cell.strip():
            return ""

        # 혼합 용매(+) 포함되어 있으면 아예 매핑하지 않음
        if "+" in cell:
            return ""

        token = normalize_token(cell)
        if token in solv_df.index:
            return solv_df.loc[token, "smiles"]
        else:
            return ""

    df["solvent_smiles"] = df["Solvent_norm"].apply(map_solvent_to_smiles)

    # norm 컬럼 제거
    df = df.drop(columns=["Solvent_norm"])

    df.to_csv(save_path, index=False)
    print(f"Saved with solvent_smiles → {save_path}")


if __name__ == "__main__":
    DB_CSV   = r"QM_stratified_train_resplit_with_mu_eps.csv"
    SOLV_CSV = r"solvent_smiles.csv"
    SAVE_CSV = r"QM_with_solvent_smiles_train.csv"

    add_solvent_smiles(DB_CSV, SOLV_CSV, SAVE_CSV)
