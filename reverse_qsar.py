import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, accuracy_score
)
from rdkit import Chem
from rdkit.Chem import Draw, BRICS
from tqdm import tqdm
import os

folders = [
    "results/models",
    "results/plots/qsar",
    "results/plots/logistic",
    "results/new_molecules"
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

def load_and_merge_data(descriptor_file, raw_values_file, smiles_file, labels_file):
    descriptors = pd.read_csv(descriptor_file, sep='\t', header=None)
    raw_values = pd.read_csv(raw_values_file, sep='\t', header=None)
    smiles = pd.read_csv(smiles_file, header=None, names=['SMILES'])
    labels_df = pd.read_csv(labels_file)
    labels = labels_df[['activity_flag']]

    df = pd.concat([smiles, labels, raw_values], axis=1)
    descriptor_names = descriptors.iloc[:, 0].tolist()
    df.columns = ['SMILES', 'Activity'] + descriptor_names

    return df

def prepare_features_labels(df):
    X = df.drop(columns=['SMILES', 'Activity'])
    y = df['Activity'].map({'active': 1, 'inactive': 0})
    smiles = df[['SMILES']]
    return X, y, smiles

def train_qsar_model(X, y, smiles, save_prefix="qsar_model"):
    from sklearn.model_selection import StratifiedKFold

    # Podział danych
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(
        X, y, smiles, stratify=y, test_size=0.2, random_state=42
    )

    # Skalowanie
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LASSO + selekcja cech
    lasso = LassoCV(cv=5, random_state=42, max_iter=20000)
    lasso.fit(X_train_scaled, y_train)
    selector = SelectFromModel(lasso, threshold=1e-5, prefit=True)

    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train_selected, y_train)

    # Predykcja i metryki
    y_pred = rf.predict(X_test_selected)

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=['inactive', 'active'], cmap='Blues'
    )
    plt.title("Confusion Matrix")
    plt.savefig(f"results/plots/qsar/{save_prefix}_confusion_matrix.png")
    plt.clf()

    # Cross-val tylko na RF + wybrane cechy
    X_scaled_all = scaler.transform(X)
    X_selected_all = selector.transform(X_scaled_all)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_selected_all, y, cv=cv, n_jobs=1)

    results = {
        "ROC AUC": roc_auc_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "CV Accuracy Mean": scores.mean(),
        "CV Accuracy Std": scores.std(),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Zapis modelu
    joblib.dump((scaler, selector, rf), f"results/models/{save_prefix}.pkl")
    pd.DataFrame(smiles_test, columns=["SMILES"]).to_csv(f"results/models/{save_prefix}_test_smiles.csv", index=False)

    selected_features = X.columns[selector.get_support()]
    pd.DataFrame(selected_features, columns=["Selected Features"]).to_csv(f"results/models/{save_prefix}_features.csv", index=False)

    return rf, results, X_train

def generate_fragment_matrix(smiles_df):
    smiles_list = smiles_df['SMILES'].tolist()
    fragment_dict = {}
    fragment_names = set()

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        try:
            frags = BRICS.BRICSDecompose(mol)  # zwraca set SMARTS stringów
            if not frags:
                continue

            frag_list = list(frags)
            fragment_dict[i] = frag_list
            fragment_names.update(frag_list)

        except:
            continue

    fragment_names = sorted(fragment_names)
    X_frag = pd.DataFrame(0, index=range(len(smiles_list)), columns=fragment_names)

    for idx, frags in fragment_dict.items():
        for frag in frags:
            if frag in X_frag.columns:
                X_frag.loc[idx, frag] = 1

    return X_frag, fragment_names

def rank_fragments_logistic(X_frag, y, fragment_names):
    print("🔢 Trening regresji logistycznej na macierzy fragmentów z podziałem na train/test...")


    X_train, X_test, y_train, y_test = train_test_split(X_frag, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)


    df = pd.DataFrame({'fragment': fragment_names, 'score': model.coef_[0]})
    df = df.sort_values(by='score', ascending=False)
    df.to_csv("results/new_molecules/fragment_scores_logistic.csv", index=False)


    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("📊 Ewaluacja modelu regresji logistycznej (test set):")
    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ ROC-AUC: {auc:.3f}")
    print("📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🧱 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['inactive', 'active'], cmap='Purples')
    plt.title("Confusion Matrix – Fragment Logistic Model")
    plt.savefig("results/plots/logistic/confusion_matrix_fragments.png")
    plt.clf()

    # SHAP analiza
    print("📈 Analiza SHAP dla regresji logistycznej na fragmentach...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Summary
    fig = plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    fig.savefig("results/plots/logistic/shap_summary_bar_fragments.png");
    plt.close()

    # Waterfall (dla top próbki)
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    fig.savefig("results/plots/logistic/shap_waterfall_sample0.png");
    plt.close()

    # Decision
    fig = plt.figure()
    shap.decision_plot(explainer.expected_value, shap_values.values, X_test, show=False, link="logit",
    feature_order="hclust")
    fig.savefig("results/plots/logistic/shap_decision_fragments.png");
    plt.close()

    return df

def draw_top_fragments_logistic(score_file, top_n, output_file):
    df = pd.read_csv(score_file)
    top_fragments = df.head(top_n)

    mols = []
    labels = []

    for idx, row in top_fragments.iterrows():
        smi = row['fragment']
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            labels.append(f"{row['score']:.2f}")

    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=labels)
        img.save(output_file)
        print(f"Zapisano top {top_n} fragmentów jako {output_file}")
    else:
        print("Nie znaleziono poprawnych fragmentów do rysowania.")

def generate_new_molecules(
        fragment_scores_file,
        linker_smiles_list,
        top_n,
        num_molecules,
        output_file_all,
        output_file_filtered,
        output_image,
        thiazolidinone_core_smarts_list
):
    # Wczytanie top fragmentów z pliku
    df = pd.read_csv(fragment_scores_file)
    all_top_frags = df.nlargest(top_n, 'score')['fragment'].tolist()

    # Konwersja wszystkich fragmentów do obiektów Mol
    rdkit_linkers = [Chem.MolFromSmiles(smi) for smi in linker_smiles_list if Chem.MolFromSmiles(smi)]
    rdkit_fragments = [Chem.MolFromSmiles(smi) for smi in all_top_frags if Chem.MolFromSmiles(smi)]

    # Łączenie linków i fragmentów w jeden zbiór
    building_blocks = rdkit_linkers + rdkit_fragments

    # Konwersja rdzeni thiazolidinonu do obiektów Mol
    thiazolidinone_cores = [Chem.MolFromSmarts(core) for core in thiazolidinone_core_smarts_list]

    print(f"⚙️ Start budowy z {len(building_blocks)} fragmentów (linkery + top {top_n} frag).")

    mols = []  # Lista na unikalne cząsteczki
    generator = BRICS.BRICSBuild(building_blocks)

    # Generowanie cząsteczek
    pbar = tqdm(total=num_molecules, desc="Generating molecules", unit="molecule")

    for mol in generator:
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        if smi not in mols:  # Sprawdzamy, czy molekuła już jest w liście
            mols.append(smi)  # Dodajemy do listy, jeśli jest unikalna
        if len(mols) >= num_molecules:
            break
        pbar.update(1)

    pbar.close()

    # Zapisz wygenerowane cząsteczki do pliku CSV
    df_all = pd.DataFrame(mols, columns=["SMILES"])
    df_all.to_csv(output_file_all, index=False)
    print(f"✅ Zapisano {len(mols)} cząsteczek do {output_file_all}")

    # Filtrowanie cząsteczek, które zawierają rdzeń thiazolidinonu
    filtered_mols = []
    for smi in mols:
        mol_obj = Chem.MolFromSmiles(smi)
        # Sprawdzamy, czy cząsteczka zawiera którykolwiek z rdzeni thiazolidinonu
        if any(mol_obj.HasSubstructMatch(core) for core in thiazolidinone_cores):
            filtered_mols.append(smi)

    # Zapisz tylko cząsteczki z rdzeniem thiazolidinonu do nowego pliku CSV
    df_filtered = pd.DataFrame(filtered_mols, columns=["SMILES"])
    df_filtered.to_csv(output_file_filtered, index=False)
    print(f"✅ Zapisano {len(filtered_mols)} cząsteczek zawierających rdzeń thiazolidinonu do {output_file_filtered}")

    # Wizualizacja 10 przykładowych cząsteczek, które zawierają rdzeń thiazolidinonu
    if filtered_mols:  # Jeśli są jakieś cząsteczki do rysowania
        mols_to_draw = [Chem.MolFromSmiles(s) for s in filtered_mols[:10]]  # Wizualizuj tylko odfiltrowane cząsteczki

        # Tworzenie rysunku cząsteczek z MolDraw2DCairo
        img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=5, subImgSize=(300, 300))

        # Zapisz obrazek
        img.save(output_image)
        print(f"🖼️ Zapisano podgląd jako {output_image}")
    else:
        print("Brak cząsteczek do narysowania (brak rdzenia thiazolidinonu).")


def emphasize_fragments(molecules_list, ):
    df_molecules = pd.DataFrame(molecules_list, columns=["SMILES"])

def run_pipeline_reverse_qsar():
    print("📦 Wczytywanie danych...")
    df = load_and_merge_data("new_descriptors.txt", "new.txt", "new_molecules.txt", "final_dataset.csv")

    print("✅ Przygotowanie danych...")
    X, y, smiles = prepare_features_labels(df)

    print("🤖 Trening modelu QSAR...")
    model, results, X_train = train_qsar_model(X, y, smiles, save_prefix="qsar_model")

    print("📊 Wyniki walidacji:")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("🧩 Tworzenie macierzy fragmentów...")
    X_frag, fragment_names = generate_fragment_matrix(smiles)

    print("📈 Ranking fragmentów...")
    print("🏹 Ewaluacja modelu regresji logicznej...")
    fragment_scores = rank_fragments_logistic(X_frag, y, fragment_names)

    print("🎨 Rysowanie top fragmentów...")
    draw_top_fragments_logistic(score_file= "results/new_molecules/fragment_scores_logistic.csv",
                                top_n=20, output_file="results/new_molecules/top_fragments.png")

    print("🧪 Generowanie nowych molekuł...")
    generate_new_molecules(
        linker_smiles_list=[
            "[*:1]CC[*:2]",
            "[*:1]O[*:2]",
            "[*:1]CCC[*:2]",
            "[*:1]C(C)C[*:2]",
            "[*:1]C(=O)N[*:2]",
            "[*:1]C=C[*:2]",
            "[*:1]N[*:2]",
            "[*:1]C(=O)C[*:2]",
            "[*:1]C(=N)C[*:2]"
            "[*:1]NC[*:2]"

        ],
        fragment_scores_file="results/new_molecules/fragment_scores_logistic.csv",
        top_n=20,
        num_molecules=10000,
        output_file_all="results/new_molecules/all_generated_molecules.csv",
        output_file_filtered="results/new_molecules/filtered_molecules.csv",
        output_image="results/new_molecules/filtered_molecules_image.png",
        thiazolidinone_core_smarts_list=
        ["O=C1N([*])C([*])C([*])S1",
         "O=C1C([*])SC([*])N([*])1",
         "O=C1C([*])C([*])SN([*])1"]

    )
    print("✅ Zakończono działanie pipeline.")

if __name__ == "__main__":
    run_pipeline_reverse_qsar()
