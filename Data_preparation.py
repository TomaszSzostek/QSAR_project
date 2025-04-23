import os
import requests
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import Descriptors

RAW_DIR = "data_sets/data/raw"
MERGED_DIR = "data_sets/data/merged"
PROCESSED_DIR = "data_sets/data/processed"
for folder in [RAW_DIR, MERGED_DIR, PROCESSED_DIR]:
    os.makedirs(folder, exist_ok=True)

target_id = "CHEMBL392"
limit = 1000


def fetch_page(offset):
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_id}&limit={limit}&offset={offset}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json().get("activities", [])
        return [{
            "molecule_chembl_id": a.get("molecule_chembl_id"),
            "canonical_smiles": a.get("canonical_smiles"),
            "standard_type": a.get("standard_type"),
            "standard_value": a.get("standard_value"),
            "standard_units": a.get("standard_units"),
            "target_chembl_id": a.get("target_chembl_id"),
            "assay_chembl_id": a.get("assay_chembl_id"),
            "assay_description": a.get("assay_description")
        } for a in data]
    except Exception as e:
        print(f"Błąd przy offset={offset}: {e}")
        return []

def get_total_count():
    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_id}&limit=1"
    r = requests.get(url)
    return r.json().get("page_meta", {}).get("total_count", 0)


def merge_with_thiazolidinones(raw_file, thiazolidinone_file, output_file):
    df_aktywnosci = pd.read_csv(raw_file, on_bad_lines='skip')
    df_thiazolidinony = pd.read_csv(thiazolidinone_file, sep=";", on_bad_lines='skip')
    df_aktywnosci = df_aktywnosci.rename(columns={"molecule_chembl_id": "ChEMBL_ID"})
    df_polaczony = pd.merge(df_thiazolidinony, df_aktywnosci, on="ChEMBL_ID", how="inner")
    df_polaczony = df_polaczony[["ChEMBL_ID", "canonical_smiles", "target_chembl_id", "standard_type", "standard_value", "standard_units", "assay_description"]]
    df_polaczony.to_csv(output_file, index=False)
    return df_polaczony

def process_activity_data(df):
    df_clean = df.dropna(subset=["standard_type", "standard_value"])
    df_filtered = df_clean[df_clean["standard_type"].isin(["IC50", "GI50"])].copy()

    def classify_activity(row):
        if row["standard_value"] < 10000:
            return "active"
        return "inactive"

    df_filtered["activity_flag"] = df_filtered.apply(classify_activity, axis=1)
    return df_filtered

def extract_ic50_dataset(df):
    ic50_df = df[df["standard_type"] == "IC50"].copy()
    selected_cols = ["ChEMBL_ID", "canonical_smiles", "standard_value", "standard_units", "assay_description"]
    return ic50_df[selected_cols]

def label_ic50_activity(df):
    df["activity_flag"] = df["standard_value"].apply(lambda x: "active" if x < 10000 else "inactive")
    return df

def clean_dataset(df):
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.drop_duplicates(subset=["canonical_smiles"])
    df = df.dropna(subset=["canonical_smiles"])
    df["Mol"] = df["canonical_smiles"].apply(lambda x: Chem.MolFromSmiles(x) if pd.notnull(x) else None)
    df = df.dropna(subset=["Mol"])
    df["MolWt"] = df["Mol"].apply(lambda mol: Descriptors.MolWt(mol) if mol else None)
    df["LogP"] = df["Mol"].apply(lambda mol: Descriptors.MolLogP(mol) if mol else None)
    df = df.dropna(subset=["MolWt", "LogP", "standard_value"])
    df = df[(df["MolWt"] < 900) & (df["LogP"] < 8)]

    def convert_to_nM(row):
        if row['standard_units'] == 'ug.mL-1':
            if row['MolWt'] and row['MolWt'] > 0:
                return row['standard_value'] / row['MolWt'] * 1_000_000
            else:
                return None
        else:
            return row['standard_value']

    df['standard_value'] = df.apply(convert_to_nM, axis=1)
    df['standard_units'] = df['standard_units'].replace('ug.mL-1', 'nM')

    def is_bad_inactive(row):
        if str(row.get('activity_flag', '')).lower() == 'inactive':
            description = str(row.get('assay_description', '')).lower()
            if any(keyword in description for keyword in ['24', '48', 'srb']):
                return True
        return False

    df = df[~df.apply(is_bad_inactive, axis=1)]
    return df

def create_final_dataset():
    total = get_total_count()
    print(f"📥 Całkowita liczba rekordów do pobrania: {total}")
    offsets = list(range(0, total, limit))

    with Pool(cpu_count()) as pool:
        all_data = list(tqdm(pool.imap(fetch_page, offsets), total=len(offsets)))

    flat_results = [item for sublist in all_data if sublist for item in sublist]
    raw_path = f"{RAW_DIR}/A549_activities.csv"

    if flat_results:
        df = pd.DataFrame(flat_results)
        df.to_csv(raw_path, index=False)
        print(f"✅ Dane zapisane: {raw_path}")

        merged_df = merge_with_thiazolidinones(raw_path, "baza_4-thiazolidinones.csv", f"{MERGED_DIR}/dataset_A549.csv")
        processed_df = process_activity_data(merged_df)
        ic50_df = extract_ic50_dataset(processed_df)
        labeled_df = label_ic50_activity(ic50_df)

        labeled_df.to_csv(f"{PROCESSED_DIR}/IC50_only_dataset.csv", index=False)
        cleaned = clean_dataset(labeled_df)
        cleaned.to_csv(f"{PROCESSED_DIR}/cleaned_dataset.csv", index=False)
        cleaned[["ChEMBL_ID", "canonical_smiles", "activity_flag"]].to_csv(f"{PROCESSED_DIR}/final_dataset.csv",
                                                                           index=False)
        print("✅ Pipeline zakończony sukcesem.")
    else:
        print("❌ Brak danych do przetworzenia.")

if __name__ == "__main__":
    create_final_dataset()














