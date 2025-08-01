import os, csv, requests, pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ───────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────
RAW_DIR, MERGED_DIR, PROCESSED_DIR = [
    f"data_sets/data/{d}" for d in ("raw", "merged", "processed")]

REBUILD_DATASET = False
TARGET_ID   = "CHEMBL392"    # A549
PAGE_LIMIT  = 1000
THRESHOLD_NM = 1_0000

CUSTOM_DATASET_PATH = (
    "/Users/tomaszszostek/PycharmProjects/REFIDD/"
    "data_sets/data/raw/my_compounds.csv"
)

for d in (RAW_DIR, MERGED_DIR, PROCESSED_DIR):
    os.makedirs(d, exist_ok=True)

# ───────────────────────────────────────────────────────────────
# Fetch data from ChEMBL
# ───────────────────────────────────────────────────────────────

def fetch_page(offset: int) -> list[dict]:
    url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
           f"?target_chembl_id={TARGET_ID}&limit={PAGE_LIMIT}&offset={offset}")
    try:
        r = requests.get(url, timeout=30); r.raise_for_status()
        return [{
            "ChEMBL_ID":        a.get("molecule_chembl_id"),
            "canonical_smiles": a.get("canonical_smiles"),
            "standard_type":    a.get("standard_type"),
            "standard_value":   a.get("standard_value"),
            "standard_units":   a.get("standard_units"),
            "assay_description":a.get("assay_description")
        } for a in r.json().get("activities", [])]
    except Exception as exc:
        print("fetch_page", exc); return []

def total_records() -> int:
    url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
           f"?target_chembl_id={TARGET_ID}&limit=1")
    return requests.get(url, timeout=30).json()["page_meta"]["total_count"]

# ───────────────────────────────────────────────────────────────
# MERGE datasets and clean
# ───────────────────────────────────────────────────────────────

def merge_with_thiazolidinones(raw_csv, thiaz_csv, out_csv) -> pd.DataFrame:
    df_act  = pd.read_csv(raw_csv, on_bad_lines="skip")
    df_thia = pd.read_csv(thiaz_csv, sep=";", on_bad_lines="skip")

    df_act  = df_act.rename (columns=lambda c: c.strip().replace(" ", "_"))
    df_thia = df_thia.rename(columns=lambda c: c.strip().replace(" ", "_"))

    merged = pd.merge(df_thia, df_act, on="ChEMBL_ID", how="inner")
    merged.dropna(axis=1, how="all").to_csv(out_csv, index=False)
    return merged

def convert_to_nM(row):
    if row["standard_units"] == "ug.mL-1":
        mw = row["MolWt"]; return (row["standard_value"] / mw * 1e6) if mw else None
    return row["standard_value"]

def add_flag(df):
    df["activity_flag"] = np.where(df["standard_value"] < THRESHOLD_NM,
                                   "active", "inactive")
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", "standard_value"])

    df["Mol"]   = df["canonical_smiles"].map(Chem.MolFromSmiles)
    df["MolWt"] = df["Mol"].map(Descriptors.MolWt)
    df["LogP"]  = df["Mol"].map(Descriptors.MolLogP)
    df = df.dropna(subset=["Mol", "MolWt", "LogP"])
    df = df[(df["MolWt"] < 900) & (df["LogP"] < 8)]

    df["standard_value"] = df.apply(convert_to_nM, axis=1)
    df = df.dropna(subset=["standard_value"])
    df["standard_units"] = "nM"
    df = add_flag(df).drop_duplicates(subset="canonical_smiles")
    return df[["ChEMBL_ID", "canonical_smiles", "activity_flag"]]

# ───────────────────────────────────────────────────────────────
# Own data resources
# ───────────────────────────────────────────────────────────────

def load_custom(path) -> pd.DataFrame:

    if not os.path.isfile(path):
        print(f"ℹ️  CUSTOM: brak {path} – pomijam."); return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep=';', on_bad_lines="skip")
        has_header = set(map(str.lower, df.columns)) & {"smiles", "canonical_smiles"}
    except pd.errors.ParserError:
        has_header = False

    if not has_header:

        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        df = pd.DataFrame(rows, columns=["ID", "canonical_smiles", "activity_flag"])

    df = df.rename(columns={"Smiles":"canonical_smiles",
                            "SMILES":"canonical_smiles",
                            "smiles":"canonical_smiles",
                            "chembl_id":"ID"})
    df = df[["ID","canonical_smiles","activity_flag"]]
    df = df.dropna(subset=["canonical_smiles"])
    return df

def drop_tautomer_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    enum = rdMolStandardize.TautomerEnumerator()

    def canon_taut(s):
        try:
            m = Chem.MolFromSmiles(str(s))
            if m:
                return Chem.MolToSmiles(enum.Canonicalize(m), canonical=True)
        except Exception:
            pass
        return None

    df = df.copy()
    df["taut_can"] = df["canonical_smiles"].apply(canon_taut)
    df = df.dropna(subset=["taut_can"])


    def pick_one(group: pd.DataFrame) -> pd.Series:

        active = group[group["activity_flag"] == "active"]
        sub = active if not active.empty else group


        if "standard_value" in sub.columns and sub["standard_value"].notna().any():
            sub = sub.loc[sub["standard_value"].astype(float).idxmin()]
        else:
            sub = sub.iloc[0]

        return sub

    deduped = (
        df.groupby("taut_can", group_keys=False)
        .apply(pick_one, include_groups=False)
    )

    return deduped


# ───────────────────────────────────────────────────────────────
# MAIN pipeline
# ───────────────────────────────────────────────────────────────

def create_final_dataset():
    raw_csv   = f"{RAW_DIR}/A549_activities.csv"
    final_csv = f"{PROCESSED_DIR}/final_dataset.csv"
    thia_csv  = "data_sets/data/raw/your_molecules.csv"


    if final_csv and Path(final_csv).exists() and not REBUILD_DATASET:
        print(f"ℹ️  {final_csv} already exists – skipping rebuild.")
        return

    if not os.path.exists(raw_csv):
        tot = total_records(); print("💻fetching ", tot, "records")
        offsets = range(0, tot, PAGE_LIMIT)
        with Pool(cpu_count()) as p:
            pages = list(tqdm(p.imap(fetch_page, offsets), total=len(offsets)))
        pd.DataFrame([x for pg in pages for x in pg]).to_csv(raw_csv, index=False)

    merged  = merge_with_thiazolidinones(raw_csv, thia_csv,
                                         f"{MERGED_DIR}/dataset_A549.csv")
    cleaned = clean_dataset(merged[merged["standard_type"] == "IC50"])
    cleaned = cleaned.rename(columns={"ChEMBL_ID":"ID"})

    custom  = load_custom(CUSTOM_DATASET_PATH)
    combined = pd.concat([cleaned, custom], ignore_index=True)
    combined = combined.drop_duplicates(subset="canonical_smiles")
    combined = drop_tautomer_duplicates(combined)

    combined.to_csv(final_csv, index=False, header=True)
    print("💾 saved", final_csv, "→", len(combined), "compounds")

# ────────────────────────────────────
if __name__ == "__main__":
    create_final_dataset()









