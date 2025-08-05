from __future__ import annotations
from pathlib import Path
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib import rcParams
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (silhouette_score, pairwise_distances,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import yaml


rcParams.update({"figure.dpi": 300, "font.size": 14,
                 "axes.grid": True, "grid.alpha": .3,
                 "axes.spines.top": False, "axes.spines.right": False})
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", message="overflow encountered.*")

# ───────────────────────────────────────
# 1.  Descriptors
# ───────────────────────────────────────
def rdkit_df(smiles):
    names = [n for n, _ in Descriptors._descList]
    calc  = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    rows  = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append([np.nan]*len(names) if m is None else list(calc.CalcDescriptors(m)))
    return pd.DataFrame(rows, columns=names)

def mordred_numeric(smiles):
    try:
        from mordred import Calculator, descriptors as mordred_desc
    except ImportError:
        warnings.warn("Mordred does not work.")
        return pd.DataFrame(index=range(len(smiles)))
    calc = Calculator(mordred_desc, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    df   = calc.pandas(mols, nproc=1, quiet=True)
    return df.apply(pd.to_numeric, errors="coerce")

def make_matrix(smiles):
    df = pd.concat([rdkit_df(smiles), mordred_numeric(smiles)], axis=1)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# ───────────────────────────────────────
# 2.  Data
# ───────────────────────────────────────
def prepare_Xy(csv_dataset: Path, csv_sel: Path):
    data = pd.read_csv(csv_dataset)
    data["y"] = data["activity_flag"].map({"inactive": 0,
                                           "active":   1}).astype(int)

    sel   = pd.read_csv(csv_sel)["descriptor"].tolist()
    Xfull = make_matrix(data["canonical_smiles"])
    X     = Xfull[sel].copy()

    bad = X.isna().any()
    if bad.any():
        warnings.warn(f"Deleted {bad.sum()} NaN descriptors.")
        X = X.loc[:, ~bad]

    Xs = StandardScaler().fit_transform(X)
    return Xs, data["y"].values

# ───────────────────────────────────────
# 3.  Visuals
# ───────────────────────────────────────
def plsda_visual(Xs, y, out_dir: Path):
    """
    Build PLS-DA similarity plots and statistics.

    Parameters
    ----------
    Xs : ndarray
        Standard-scaled descriptor matrix.
    y  : ndarray
        Binary activity vector (0 = inactive, 1 = active).
    out_dir : Path
        Output directory for all artifacts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Train two-component PLS-DA model ───────────────────────────────
    pls = PLSRegression(n_components=2).fit(Xs, y)
    LV  = pls.x_scores_

    # ── Quality metrics ────────────────────────────────────────────────
    sil   = silhouette_score(LV, y)
    D     = pairwise_distances(LV)
    intra = D[y[:, None] == y].mean()
    inter = D[y[:, None] != y].mean()

    cv     = StratifiedKFold(5, shuffle=True, random_state=0)
    y_pred = cross_val_predict(pls, Xs, y, cv=cv, method="predict").ravel()
    roc    = roc_auc_score(y, y_pred)

    # ── Save raw scores & statistics ───────────────────────────────────
    pd.DataFrame(LV, columns=["LV1", "LV2"]).to_csv(out_dir / "pls_scores.csv",
                                                    index=False)
    with open(out_dir / "plsda_stats.txt", "w") as f:
        f.write(f"Silhouette           = {sil:.3f}\n")
        f.write(f"Intra-class distance = {intra:.3f}\n")
        f.write(f"Inter-class distance = {inter:.3f}\n")
        f.write(f"ROC-AUC (5-fold CV)  = {roc:.3f}\n")

    # ── Scatter plot of LV1 vs LV2 ─────────────────────────────────────
    cmap   = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "inactive", 1: "active"}

    plt.figure(figsize=(6, 5))
    for cls in [0, 1]:
        m = y == cls
        plt.scatter(LV[m, 0], LV[m, 1], s=28, alpha=.8,
                    label=labels[cls], color=cmap[cls])
    plt.xlabel("Latent Variable 1")
    plt.ylabel("Latent Variable 2")
    plt.title(f"PLS-DA   (silhouette = {sil:.2f})")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "plsda_scatter.png")
    plt.close()

    # ── Violin plots for score distributions ───────────────────────────
    plt.figure(figsize=(6, 4))
    parts = plt.violinplot([LV[y == 0, 0], LV[y == 1, 0],
                            LV[y == 0, 1], LV[y == 1, 1]],
                           showmeans=True, widths=.8)
    for pc, col in zip(parts["bodies"],
                       [cmap[0], cmap[1], cmap[0], cmap[1]]):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    plt.xticks([1, 2, 3, 4],
               ["LV1 inactive", "LV1 active",
                "LV2 inactive", "LV2 active"],
               rotation=20)
    plt.ylabel("Score value")
    plt.title("Distributions of LV1 / LV2")
    plt.tight_layout()
    plt.savefig(out_dir / "plsda_violin.png")
    plt.close()

    print("✅  All files saved in:", out_dir.resolve())


# ───────────────────────────────────────
# 4.  MAIN
# ───────────────────────────────────────
def run_plsda_visuals(cfg_or_path="config.yml"):
    """
    Build PLS-DA similarity visuals.
    Accepts either a config-dict or a path to the YAML file.
    """

    # -------- configuration -------------------------------------------
    if isinstance(cfg_or_path, dict):
        cfg = cfg_or_path
    else:
        with open(cfg_or_path) as fh:
            cfg = yaml.safe_load(fh)

    paths      = cfg["Paths"]
    artifacts  = cfg["Artifacts"]

    # helper: absolute path stays absolute; otherwise prepend results_root
    root = Path(paths["results_root"])
    def _p(item):
        p = Path(artifacts[item])
        return p if p.is_absolute() else root / p

    csv_data = Path(paths["final_path"])
    csv_sel  = _p("selected_descriptors")
    out_dir  = Path(paths["visuals_root"])

    # -------- run analysis --------------------------------------------
    Xs, y = prepare_Xy(csv_data, csv_sel)
    plsda_visual(Xs, y, out_dir)




