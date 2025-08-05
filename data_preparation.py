import os, csv, requests, pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Mapping



# ───────────────────────────────────────────────────────────────
# Fetch data from ChEMBL
# ───────────────────────────────────────────────────────────────

def fetch_page(TARGET_ID, PAGE_LIMIT, offset: int) -> list[dict]:
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

def total_records(TARGET_ID) -> int:
    url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
           f"?target_chembl_id={TARGET_ID}&limit=1")
    return requests.get(url, timeout=30).json()["page_meta"]["total_count"]

# ───────────────────────────────────────────────────────────────
# MERGE datasets and clean
# ───────────────────────────────────────────────────────────────

def merge_with_chemical_space(all_target_compounds, chemical_space_csv, merged_csv) -> pd.DataFrame:
    df_act  = pd.read_csv(all_target_compounds, on_bad_lines="skip")
    df_thia = pd.read_csv(chemical_space_csv, sep=";", on_bad_lines="skip")

    df_act  = df_act.rename (columns=lambda c: c.strip().replace(" ", "_"))
    df_thia = df_thia.rename(columns=lambda c: c.strip().replace(" ", "_"))

    merged = pd.merge(df_thia, df_act, on="ChEMBL_ID", how="inner")
    merged.dropna(axis=1, how="all").to_csv(merged_csv, index=False)
    merged.to_csv(merged_csv, index=False)
    return merged

def convert_to_nM(row):
    if row["standard_units"] == "ug.mL-1":
        mw = row["MolWt"]; return (row["standard_value"] / mw * 1e6) if mw else None
    return row["standard_value"]

def add_flag(df, THRESHOLD_NM):
    df["activity_flag"] = np.where(df["standard_value"] < THRESHOLD_NM,
                                   "active", "inactive")
    return df

def clean_dataset(df: pd.DataFrame, THRESHOLD_NM) -> pd.DataFrame:
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
    df = add_flag(df, THRESHOLD_NM).drop_duplicates(subset="canonical_smiles")
    return df[["ChEMBL_ID", "canonical_smiles", "activity_flag"]]

# ───────────────────────────────────────────────────────────────
# Own data resources
# ───────────────────────────────────────────────────────────────

def load_custom(path) -> pd.DataFrame:

    if not os.path.isfile(path):
        print(f"ℹ️  CUSTOM: none {path} – skipping."); return pd.DataFrame()

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
# MAIN func
# ───────────────────────────────────────────────────────────────

def create_final_dataset(cfg: Mapping, log) -> None:
    # --- All parameters are read from YAML config ---
    target_id    = cfg["TARGET_ID"]
    page_limit   = cfg["PAGE_LIMIT"]
    threshold_nm = cfg["THRESHOLD_NM"]
    paths        = cfg["Paths"]

    all_target_compounds_csv = Path(paths["target_path"])
    merged_csv   = Path(paths["merged_path"])
    final_csv    = Path(paths["final_path"])
    custom_csv   = Path(paths["own_resources_path"])
    chemical_space_csv = Path(paths["chemical_space_path"])

    # --- Create directories if needed ---
    for p in [all_target_compounds_csv.parent, merged_csv.parent, final_csv.parent]:
        p.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Fetch data from ChEMBL if not already fetched ---
    if not all_target_compounds_csv.exists():
        log.info("[1/5] Fetching ChEMBL data for %s", target_id)
        total = total_records(target_id)
        offsets = range(0, total, page_limit)
        with Pool(cpu_count()) as pool:
            pages = list(tqdm(pool.imap_unordered(
                lambda o: fetch_page(target_id, page_limit, o), offsets),
                desc="Downloading"))
        df_raw = pd.DataFrame([r for page in pages for r in page])
        df_raw.to_csv(all_target_compounds_csv, index=False)
        log.info("Saved %d records to: %s", len(df_raw), all_target_compounds_csv)
    else:
        log.info("[1/5] ChEMBL data already present: %s", all_target_compounds_csv)
        df_raw = pd.read_csv(all_target_compounds_csv)

    # --- Step 2: Merge with chemical space ---
    if not merged_csv.exists():
        log.info("[2/5] Merging with chemical_space...")
        df_merged = merge_with_chemical_space(df_raw, chemical_space_csv)
        df_merged.to_csv(merged_csv, index=False)
        log.info("Merged data saved: %s (%d records)", merged_csv, len(df_merged))
    else:
        log.info("[2/5] Merged CSV already exists.")
        df_merged = pd.read_csv(merged_csv)

    # --- Step 3: Clean and flag dataset ---
    log.info("[3/5] Cleaning and flagging dataset...")
    df_ic50 = df_merged[df_merged["standard_type"] == "IC50"]
    df_clean = clean_dataset(df_ic50, threshold_nm)
    log.info("Cleaned dataset: %d compounds", len(df_clean))

    # --- Step 4: Append custom compounds if present ---
    if custom_csv.exists():
        log.info("[4/5] Appending custom.csv")
        df_custom = load_custom(custom_csv)
        df_combined = pd.concat([df_clean, df_custom], ignore_index=True)
    else:
        log.info("[4/5] No custom.csv found, skipping")
        df_combined = df_clean

    # --- Step 5: Tautomer deduplication and save final dataset ---
    log.info("[5/5] Deduplicating tautomers and saving final dataset...")
    df_final = drop_tautomer_duplicates(df_combined)
    df_final.to_csv(final_csv, index=False)
    log.info("Final dataset saved: %s (%d unique compounds)", final_csv, len(df_final))











