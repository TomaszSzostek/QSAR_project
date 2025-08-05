from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from rdkit import Chem
from rdkit.Chem import Draw, BRICS

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)

# ───────────────────────────────────────────────────────────────
# 0. Data loading
# ───────────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"canonical_smiles", "activity_flag"}
    if req - set(df.columns):
        raise ValueError(f"CSV must contain columns: {req}")
    return df

# ───────────────────────────────────────────────────────────────
# 1. BRICS fragmentation → binary matrix
# ───────────────────────────────────────────────────────────────

def generate_fragment_matrix_brics(
    smiles_df: pd.DataFrame, min_occ: int = 1
) -> Tuple[pd.DataFrame, List[str]]:
    frag_dict: dict[int, List[str]] = {}
    counter: dict[str, int] = {}

    for idx, smi in enumerate(smiles_df["SMILES"]):
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            frag_dict[idx] = []
            continue
        frags = list(BRICS.BRICSDecompose(mol))
        frag_dict[idx] = frags
        for f in frags:
            counter[f] = counter.get(f, 0) + 1

    cols = sorted([f for f, c in counter.items() if c >= min_occ])
    X = pd.DataFrame(0, index=smiles_df.index, columns=cols, dtype=int)
    for i, frags in frag_dict.items():
        for f in frags:
            if f in X.columns:
                X.at[i, f] = 1
    return X, cols

# ───────────────────────────────────────────────────────────────
# 1b. Scientific selection of fragments (80% cumulative for actives)
# ───────────────────────────────────────────────────────────────

def select_fragments_active_only(rank: pd.Series, X: pd.DataFrame, y: pd.Series, threshold=0.80):
    active_mask = y == 1
    active_frags = set(X.loc[active_mask].sum(axis=0)[lambda s: s > 0].index)
    rank_active = rank[rank.index.intersection(active_frags)]
    shap_sorted = rank_active.sort_values(ascending=False)
    cumsum = shap_sorted.cumsum() / shap_sorted.sum()
    n_frags = np.searchsorted(cumsum, threshold) + 1
    return shap_sorted.index[:n_frags].tolist(), shap_sorted, cumsum, n_frags

# ───────────────────────────────────────────────────────────────
# 2. Visualization helpers
# ───────────────────────────────────────────────────────────────

def _ensure_parent(p: Path | str):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def save_cm(y_true, y_pred, labels, cmap, title, out_png: Path):
    out_png = _ensure_parent(out_png)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap=cmap
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def draw_top_fragments_grid(selected: List[str], out_png: Path, molsPerRow, subImgSize, dpi):
    out_png = _ensure_parent(out_png)
    mols = [Chem.MolFromSmiles(s) for s in selected]
    legends = [f"{i+1}" for i in range(len(mols))]
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=molsPerRow, subImgSize=subImgSize)
    img.save(str(out_png))

def plot_top_bar(rank: pd.Series, out_png: Path, title: str):
    out_png = _ensure_parent(out_png)
    n = len(rank)
    vals = rank.values
    fig, ax = plt.subplots(figsize=(max(12, n // 3), 6))
    ax.bar(range(1, n+1), vals)
    ax.set_xlabel("Fragment rank")
    ax.set_ylabel("Importance score")
    ax.set_title(title)
    ax.set_xticks(range(1, n+1))
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_cumulative_cut(
    shap_sorted: pd.Series, cumsum: pd.Series, n_frags: int, threshold: float, out_png: Path, title: str
):
    out_png = _ensure_parent(out_png)
    max_show = max(50, n_frags + 5)
    xvals = np.arange(1, min(len(cumsum), max_show) + 1)
    yvals = cumsum.values[:max_show]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(xvals, yvals, marker='o', color='dodgerblue', linewidth=2.5, markersize=7)
    ax.axhline(threshold, color='red', linestyle='--', lw=2, label=f"{int(threshold*100)}% threshold")
    ax.axvline(n_frags, color='green', linestyle='--', lw=2, label=f"N selected: {n_frags}")

    ax.set_xlim(1, max(xvals))
    ax.set_ylim(0, 1.03)
    ax.grid(True, axis='y', linestyle=':', lw=1.2)
    ax.set_xlabel("Fragment rank", fontsize=15)
    ax.set_ylabel("Cumulative importance", fontsize=15)
    ax.set_title(title, fontsize=15, pad=16)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend(fontsize=13, loc='lower right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def plot_presence(selected: List[str], X: pd.DataFrame, y: pd.Series, out_csv: Path, out_png: Path, title: str):
    out_png = _ensure_parent(out_png)
    df = pd.concat([X[selected], y.rename("y")], axis=1)
    rows = []
    for i, f in enumerate(selected):
        rows.append({
            "fragment": f,
            "rank": i + 1,
            "active_%": df.loc[df.y == 1, f].mean() * 100,
            "inactive_%": df.loc[df.y == 0, f].mean() * 100,
        })
    pres = pd.DataFrame(rows)
    pres.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(1, len(selected) + 1)
    ax.bar(x - 0.17, pres["active_%"], width=0.34, label="Active")
    ax.bar(x + 0.17, pres["inactive_%"], width=0.34, label="Inactive")
    ax.set_xlabel("Fragment rank")
    ax.set_ylabel("Presence (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_stability(selected: List[str], X, y, model_type: str, out_csv: Path, out_png: Path, top_n: int = 20):
    out_png = _ensure_parent(out_png)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    hits = []
    for tr_idx, _ in cv.split(X, y):
        Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
        if model_type == "logistic":
            m = LogisticRegression(max_iter=10000).fit(Xtr, ytr)
            r = pd.Series(m.coef_[0], index=X.columns).sort_values(ascending=False)
        else:
            m = RandomForestClassifier(
                800, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
            ).fit(Xtr, ytr)
            arr = shap.TreeExplainer(m).shap_values(Xtr, check_additivity=False)[1]
            r = pd.Series(np.abs(arr).mean(axis=0), index=X.columns).sort_values(ascending=False)
        hits.extend([frag for frag in r.head(top_n).index if frag in selected])

    counts = pd.Series(hits).value_counts().reset_index()
    counts.columns = ["fragment", "value"]
    counts["rank"] = np.arange(1, len(counts) + 1)
    counts.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(counts["rank"], counts["value"])
    ax.set_xlabel("Fragment rank")
    ax.set_ylabel("Appearances in top 20 (CV)")
    ax.set_title(f"Fragment stability – {model_type}")
    ax.set_xticks(counts["rank"])
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ───────────────────────────────────────────────────────────────
# 5. Training + evaluation
# ───────────────────────────────────────────────────────────────

def train_and_eval_models(X, y, out_dir: Path):
    out_log = out_dir / "Logistic"
    out_rf = out_dir / "RandomForest"
    out_log.mkdir(parents=True, exist_ok=True)
    out_rf.mkdir(parents=True, exist_ok=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=10000).fit(X_tr, y_tr)
    proba_lr = lr.predict_proba(X_te)[:, 1]
    pred_lr = proba_lr > 0.5

    save_cm(y_te, pred_lr, ["inactive", "active"], "BuPu", "Confusion matrix – Logistic",
            out_log / "plots/Confusion_matrix.png")

    expl_lr = shap.Explainer(lr, X_tr)(X_te)
    shap.summary_plot(expl_lr, X_te, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_log / "plots/shap_summary.png")
    plt.close()

    rank_lr = pd.Series(lr.coef_[0], index=X.columns).sort_values(ascending=False)
    rank_lr.to_csv(out_log / "fragments_scores.csv")

    # --- 80% cumulative importance fragments ---
    selected_lr, shap_sorted_lr, cumsum_lr, n_frags_lr = select_fragments_active_only(rank_lr, X, y, threshold=0.80)
    print(f"\n(LogReg) Selected {len(selected_lr)} fragments (80% cumulative SHAP for actives).")
    draw_top_fragments_grid(selected_lr, out_log / "plots/selected_fragments_grid.png", molsPerRow=5, subImgSize=(600,600), dpi= 600)
    plot_top_bar(rank_lr[selected_lr], out_log / "plots/top_bar_selected.png", "SHAP importance – selected fragments")
    plot_cumulative_cut(
        shap_sorted_lr, cumsum_lr, n_frags_lr, 0.80, out_log / "plots/shap_cumulative_cutoff.png",
        "LogReg: Cumulative SHAP importance (actives only, cutoff at 80%)"
    )
    plot_presence(
        selected_lr, X, y,
        out_log / "presence_selected.csv", out_log / "plots/presence_selected.png",
        "LogReg: Fragment presence – selected"
    )
    plot_stability(
        selected_lr, X, y, "logistic",
        out_log / "stability_selected.csv", out_log / "plots/stability_selected.png",
        top_n=min(20, len(selected_lr))
    )
    with open(out_log / "selected_fragments.smi", "w") as fh:
        for frag in selected_lr:
            fh.write(f"{frag}\n")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        800, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
    ).fit(X_tr, y_tr)
    proba_rf = rf.predict_proba(X_te)[:, 1]
    pred_rf = proba_rf > 0.5

    arr_rf = shap.TreeExplainer(rf).shap_values(X_te, check_additivity=False)[1]
    rank_rf = pd.Series(np.abs(arr_rf).mean(axis=0), index=X.columns).sort_values(ascending=False)
    rank_rf.to_csv(out_rf / "fragment_scores.csv")

    selected_rf, shap_sorted_rf, cumsum_rf, n_frags_rf = select_fragments_active_only(rank_rf, X, y, threshold=0.80)
    print(f"\n(RandomForest) Selected {len(selected_rf)} fragments (80% cumulative SHAP for actives).")
    draw_top_fragments_grid(selected_rf, out_rf / "plots/selected_fragments_grid.png", molsPerRow=5, subImgSize=(600,600), dpi=600)
    plot_top_bar(rank_rf[selected_rf], out_rf / "plots/top_bar_selected.png", " SHAP importance – selected fragments")
    plot_cumulative_cut(
        shap_sorted_rf, cumsum_rf, n_frags_rf, 0.80, out_rf / "plots/shap_cumulative_cutoff.png",
        "Cumulative SHAP importance (actives only, cutoff at 80%)"
    )
    plot_presence(
        selected_rf, X, y,
        out_rf / "presence_selected.csv", out_rf / "plots/presence_selected.png",
        "RF: Fragment presence – selected"
    )
    plot_stability(
        selected_rf, X, y, "rf",
        out_rf / "stability_selected.csv", out_rf / "plots/stability_selected.png",
        top_n=min(20, len(selected_rf))
    )
    save_cm(y_te, pred_rf, ["inactive", "active"], "GnBu", "Confusion matrix", out_rf / "plots/Confusion_matrix.png")
    with open(out_rf / "selected_fragments.smi", "w") as fh:
        for frag in selected_rf:
            fh.write(f"{frag}\n")

# ───────────────────────────────────────────────────────────────
# 6. Entry point
# ───────────────────────────────────────────────────────────────

def defragmenttion(cfg: Mapping, log):
    paths = cfg["Paths"]

    final_csv = "data/processed/final_dataset.csv"
    fragments = Path(paths["fragments"])
    fragments.mkdir(exist_ok=True)

    df = load_dataset(final_csv)
    y = df.activity_flag.map({"inactive": 0, "active": 1}).astype(int)
    smiles_df = df[["canonical_smiles"]].rename(columns={"canonical_smiles": "SMILES"})

    print("🔨  BRICS fragmentation...")
    X_frag, cols = generate_fragment_matrix_brics(smiles_df, min_occ=1)
    print(f"   → {X_frag.shape[1]} unique fragments")

    train_and_eval_models(X_frag, y, fragments)
    print("\n✅  Pipeline complete – results saved in:", fragments)


