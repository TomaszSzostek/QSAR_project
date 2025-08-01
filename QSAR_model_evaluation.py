"""
QSAR  –  RDKit + Mordred‑2D  →  Boruta → Balanced‑RandomForest
Split = cliff ≤15%, cluster ≥30%, dissimilar ≥20%, leakage Tc ≥0.85
Artefacts: ROC / PR / Calibration / Reliability + SHAP, bootstrap‑CI, y‑scramble
"""
from __future__ import annotations
import warnings, time, random, math, itertools
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.ML.Cluster import Butina
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import shap, joblib, matplotlib.pyplot as plt
from matplotlib import rcParams, cm
import pickle

# ────────────────────────────────── SETUP ───────────────────────────────────
rcParams.update({
    'figure.dpi':300, 'font.size':14,
    'axes.grid':True, 'grid.alpha':.3,
    'axes.spines.top':False, 'axes.spines.right':False
})
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', message='.*FutureWarning.*|.*overflow encountered.*')
try:
    from imblearn.ensemble import BalancedRandomForestClassifier as BRF
except ModuleNotFoundError:
    BRF = RandomForestClassifier
    print('⚠️ imblearn not found – using RandomForestClassifier')

# ╭────────── CACHE & PATHS ──────────────╮
OUT          = Path("results/Evaluation_qsar_model")
PLOTS_DIR    = OUT / "plots"
METRICS_DIR  = OUT / "model_metrics"


PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42

# descriptors & model artefacts
DESC_PKL    = OUT / "X_full.pkl"
SEL_CSV     = OUT / "selected_descriptors.csv"
MODEL_PKL   = OUT / "rf_model.joblib"


SHAP_BAR    = PLOTS_DIR / "shap_bar_plot.png"
SHAP_BEESWARM= PLOTS_DIR / "shap_beeswarm_plot.png"
ROC_PNG     = PLOTS_DIR / "roc.png"
PR_PNG      = PLOTS_DIR / "pr.png"
CAL_PNG     = PLOTS_DIR / "calibration.png"
REL_PNG     = PLOTS_DIR / "reliability.png"
HIST_PNG    = PLOTS_DIR / "proba_hist.png"
CM_PNG      = PLOTS_DIR / "confusion_matrix.png"
SUM_PNG     = PLOTS_DIR / "metrics_summary.png"


BOOT_TXT    = METRICS_DIR / "bootstrap_stats.txt"
Y_TXT       = METRICS_DIR / "y_scramble.txt"
CV_TXT      = METRICS_DIR / "cv_auc.txt"
# ╰──────────────────────────────────────────╯


def _exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)

# ╭────────── DATA ────────────────────────╮
def load_dataset(csv:str)->pd.DataFrame:
    df=pd.read_csv(csv)
    need={'canonical_smiles','activity_flag'}
    if need-set(df.columns): raise ValueError(f'CSV must contain {need}')
    df=df[['canonical_smiles','activity_flag']].copy()
    df['y']=df['activity_flag'].map({'inactive':0,'active':1}).astype(int)
    return df
# ╰──────────────────────────────────────────╯

# ╭────────── DESCRIPTORS ──────────────────╮
def rdkit_df(smiles):
    names=[n for n,_ in Descriptors._descList]
    calc=MoleculeDescriptors.MolecularDescriptorCalculator(names)
    rows=[[np.nan]*len(names) if (m:=Chem.MolFromSmiles(s)) is None else list(calc.CalcDescriptors(m)) for s in smiles]
    return pd.DataFrame(rows,columns=names)

def mordred_df(smiles):
    try:
        from mordred import Calculator, descriptors as md
    except ImportError:
        warnings.warn('Mordred not installed – using RDKit only')
        return pd.DataFrame(index=range(len(smiles)))
    calc=Calculator(md,ignore_3D=True)
    return calc.pandas([Chem.MolFromSmiles(s) for s in smiles]).apply(pd.to_numeric,errors='coerce')

def compute_descriptors(smiles):
    if DESC_PKL.exists():
        print('📂 Loading descriptors cache')
        return pd.read_pickle(DESC_PKL)
    t0=time.time()
    df=pd.concat([rdkit_df(smiles),mordred_df(smiles)],axis=1)
    df=df.loc[:,~df.columns.duplicated()].replace([np.inf,-np.inf],np.nan).dropna(axis=1)
    low=df.var()<1e-4
    if low.any(): df=df.loc[:,~low]; print(f'🗑️ {low.sum()} near-constant removed')
    corr=df.corr().abs(); up=corr.where(np.triu(np.ones(corr.shape),1).astype(bool))
    drop=[c for c in up.columns if any(up[c]>0.9)]
    if drop: df=df.drop(columns=drop); print(f'🗑️ {len(drop)} correlated removed')
    print(f'📐 Descriptors shape {df.shape} t={time.time()-t0:.1f}s')
    df.to_pickle(DESC_PKL)
    return df
# ╰──────────────────────────────────────────╯

# ╭────────── SPLIT HELPERS ─────────────────╮
def morgan_fp(smi,r=2,bits=2048):
    m=Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m,r,bits) if m else None

def activity_cliffs(fps,y,th=0.8):
    return {i for (i,fi),(j,fj) in itertools.combinations(enumerate(fps),2)
            if DataStructs.TanimotoSimilarity(fi,fj)>=th and y[i]!=y[j]}

def clusters(fps,cut=0.7):
    n=len(fps);d=[]
    for i in range(1,n): d.extend([1-s for s in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])])
    return Butina.ClusterData(d,n,cut,isDistData=True)

import pandas as _pd

def mw_logp(smiles):
    rows=[]
    for s in smiles:
        m=Chem.MolFromSmiles(s)
        rows.append({'mw':Descriptors.MolWt(m) if m else np.nan,
                     'logp':Crippen.MolLogP(m) if m else np.nan})
    return _pd.DataFrame(rows)

import math,sys

def stratified_sampling(idx,props,n,rng):
    b1=_pd.qcut(props.mw,4,labels=False,duplicates='drop')
    b2=_pd.qcut(props.logp,4,labels=False,duplicates='drop')
    strata=(b1.astype(str)+'_'+b2.astype(str)).values
    groups=_pd.DataFrame({'idx':idx,'stratum':strata}).groupby('stratum').idx.apply(list)
    picks=[]
    for g in groups:
        k=min(len(g),math.ceil(n*len(g)/len(idx)))
        picks.extend(rng.choice(g,size=k,replace=False))
    return picks[:n]

def custom_split(smiles,y,frac=0.2,max_cliff=0.15,min_cluster=0.3,min_dissim=0.2,seed=42):
    rng=np.random.default_rng(seed);n=len(smiles);fps=[morgan_fp(s) for s in smiles]
    nt=round(n*frac);reason=dict();test=list();picked={0:0,1:0};target={0:nt//2,1:nt-nt//2}
    can=lambda i: picked[y[i]]<target[y[i]]

    for i in random.sample(activity_cliffs(fps,y),k=min(9999,int(nt*max_cliff))):
        if can(i):test.append(i);reason[i]='cliff';picked[y[i]]+=1
    need_cl,need_di=math.ceil(min_cluster*nt),math.ceil(min_dissim*nt);added=0

    for cl in sorted(clusters(fps),key=len,reverse=True):
        for i in cl:
            if len(test)>=nt-need_di:break
            if i in test or not can(i):continue
            test.append(i);reason[i]='cluster';picked[y[i]]+=1;added+=1
            if added>=need_cl:break
    remain=[i for i in range(n) if i not in test]
    extra=stratified_sampling(remain,mw_logp([smiles[i] for i in remain]),nt-len(test),rng)

    for i in extra:
        if can(i):test.append(i);reason[i]='dissim';picked[y[i]]+=1
    train=[i for i in range(n) if i not in test]
    drop=[i for i in train if max(DataStructs.TanimotoSimilarity(fps[i],fps[j]) for j in test)>=0.85]
    train=[i for i in train if i not in drop]

    if drop: warnings.warn(f'Dropped {len(drop)} duplicates')
    return train,test,reason
# ╰──────────────────────────────────────────╯

# ╭────────── BORUTA + RF ─────────────────────╮
def boruta_select(X,y,seed=42):
    if SEL_CSV.exists():
        return pd.read_csv(SEL_CSV)['descriptor'].tolist()
    rf=BRF(n_estimators=400,class_weight='balanced',n_jobs=-1,random_state=seed)
    bor=BorutaPy(rf,n_estimators='auto',max_iter=20,two_step=True,random_state=seed)
    bor.fit(X.fillna(X.median()).values,y)
    sel=X.columns[bor.support_].tolist()
    pd.Series(sel,name='descriptor').to_csv(SEL_CSV,index=False)
    print(f'✅ Boruta kept {len(sel)}/{X.shape[1]}')
    return sel

def make_rf(seed=42): return BRF(n_estimators=1200,class_weight='balanced',random_state=seed,n_jobs=-1)
# ╰──────────────────────────────────────────╯

# ╭────────── SHAP + VAL ───────────────────────╮
def safe_shap(rf,X,out):
    if SHAP_BAR.exists():return
    try:
        expl=shap.TreeExplainer(rf).shap_values(X)[1]
        shap.summary_plot(expl,X,plot_type='bar',show=False);plt.savefig(SHAP_BAR);plt.close()
        shap.summary_plot(expl,X,max_display=10,show=False);plt.savefig(SHAP_BEESWARM);plt.close()
    except Exception: pass

def plots_val(model,X,y,out):
    if _exists(ROC_PNG,PR_PNG,CAL_PNG,REL_PNG,HIST_PNG):return
    proba=model.predict_proba(X)[:,1];pred=proba>0.5
    roc=roc_auc_score(y,proba);pr=average_precision_score(y,proba)
    RocCurveDisplay.from_predictions(y,proba);plt.title(f'ROC (AUC={roc:.3f})');plt.savefig(ROC_PNG);plt.close()
    PrecisionRecallDisplay.from_predictions(y,proba);plt.title(f'PR (AUC={pr:.3f})');plt.savefig(PR_PNG);plt.close()
    t,p=calibration_curve(y,proba,n_bins=10,strategy='quantile')
    plt.plot(p,t,'o-');plt.plot([0,1],[0,1],'--'); plt.title("Calibration", fontsize=15, weight="bold"); plt.savefig(CAL_PNG);plt.close()
    bins=np.linspace(0,1,11);centers=0.5*(bins[:-1]+bins[1:]);err=[np.nan if not ((m:=(proba>=lo)&(proba<hi))).any() else 1-accuracy_score(y[m],pred[m]) for lo,hi in zip(bins[:-1],bins[1:])]
    plt.plot(centers,err,'s-');plt.savefig(REL_PNG);plt.close()
    plt.hist(proba,bins=20,alpha=0.8);plt.savefig(HIST_PNG);plt.close()
# ╰──────────────────────────────────────────╯

# ╭────────── BOOTSTRAP + Y‑SCRAMBLE ─────────────╮
def bootstrap_auc(rf,X,y,n=1000,seed=42):
    rng=np.random.default_rng(seed)
    stats=[roc_auc_score(y[r],rf.predict_proba(X[r])[:,1]) for r in rng.integers(0,len(y),size=(n,len(y)))]
    return np.median(stats),np.percentile(stats,2.5),np.percentile(stats,97.5)

def y_scramble(model_f,X,y,n=50,seed=42):
    rng=np.random.default_rng(seed);scores=[]
    for _ in range(n):
        mdl=model_f();mdl.fit(X,rng.permutation(y))
        scores.append(roc_auc_score(y,mdl.predict_proba(X)[:,1]))
    return np.mean(scores),np.std(scores)
# ╰──────────────────────────────────────────╯

# ─────────── Confusion‑matrix───────────
def plot_confusion_matrix_pastel(model, X, y, out: Path, thresh: float = 0.5):

    if out.exists():
        return

    cm_raw  = confusion_matrix(y, model.predict_proba(X)[:, 1] > thresh)
    cm_perc = cm_raw / cm_raw.sum(axis=1, keepdims=True)

    # --- figure ---
    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=450, facecolor="white")
    cmap    = cm.get_cmap("Pastel2")

    im = ax.imshow(cm_perc,
                   vmin=0, vmax=1,
                   interpolation="nearest",
                   cmap=cmap)

    # grid lines
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # labels & ticks
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Inactive", "Active"], fontsize=14, weight="bold")
    ax.set_yticklabels(["Inactive", "Active"], fontsize=14, weight="bold")
    ax.set_xlabel("Predicted", fontsize=15, labelpad=10)
    ax.set_ylabel("True",      fontsize=15, labelpad=10)

    # numbers +percentages
    for (i, j), val in np.ndenumerate(cm_raw):
        pct = cm_perc[i, j] * 100
        ax.text(j, i,
                f"{val}\n({pct:4.1f}%)",
                ha="center", va="center",
                fontsize=13, weight="bold",
                color="black" if pct < .6 else "white")

    # colour‑bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("% within true class", rotation=270, labelpad=15)

    ax.set_title("Confusion Matrix", fontsize=18, weight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ─────────── Summary‑figure as neat table ───────────
def summary_figure(metrics: dict, out: Path):

    if out.exists():
        return

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=450, facecolor="#f7f7f7")
    ax.axis("off")

    # table data
    rows = [
        ["ROC‑AUC (boot)",
         f"{metrics['roc_med']:.3f}  "
         f"[{metrics['roc_lo']:.3f}; {metrics['roc_hi']:.3f}]"
         if not np.isnan(metrics['roc_med']) else "N/A"],
        ["PR‑AUC",    f"{metrics['pr']:.3f}"],
        ["Accuracy",  f"{metrics['acc']:.3f}"],
        ["Y‑scramble",
         f"{metrics['ys_mean']:.3f}  ± {metrics['ys_std']:.3f}"]
    ]

    col_labels = ["Metric", "Value"]
    table = ax.table(cellText=rows,
                     colLabels=col_labels,
                     colWidths=[0.45, 0.55],
                     loc="center",
                     cellLoc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.6)

    # header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:                 # header row
            cell.set_text_props(weight="bold", color="#333333")
            cell.set_facecolor("#d1e0f3")
        elif col == 0:               # first column (metric names)
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f5fb")
        else:                        # values column
            cell.set_facecolor("#ffffff")

    ax.set_title("Model Performance Summary",
                 fontsize=16, weight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

# ╰──────────────────────────────────────────╯

# ╭──────────────── MAIN ───────────────────────────╮
def main():
    CSV_PATH = "data_sets/data/processed/final_dataset.csv"
    df = load_dataset(CSV_PATH)
    smiles, y = df.canonical_smiles.tolist(), df.y.values
    X_full = compute_descriptors(smiles)
    tr, te, _ = custom_split(smiles, y, seed=SEED)
    sel = boruta_select(X_full.iloc[tr], y[tr], SEED)
    X_tr = X_full.loc[tr, sel].fillna(X_full.loc[tr, sel].median()).values
    X_te = X_full.loc[te, sel].fillna(X_full.loc[tr, sel].median()).values

    # Model caching / training
    if MODEL_PKL.exists():
        rf = joblib.load(MODEL_PKL)
    else:
        rf = make_rf(SEED)
        rf.fit(X_tr, y[tr])
        joblib.dump(rf, MODEL_PKL)
        with open(OUT / "rf.pkl", "wb") as fh:
            pickle.dump(rf, fh)


    # SHAP & validation plots
    safe_shap(rf, pd.DataFrame(X_tr, columns=sel), OUT)
    plots_val(rf, X_te, y[te], OUT)

    # Bootstrap ROC‑AUC
    if BOOT_TXT.exists():
        roc_med, roc_lo, roc_hi = np.loadtxt(BOOT_TXT)
    else:
        roc_med, roc_lo, roc_hi = bootstrap_auc(rf, X_te, y[te])
        np.savetxt(BOOT_TXT, [roc_med, roc_lo, roc_hi], fmt='%.6f')

    # Y‑scramble
    if Y_TXT.exists():
        ys_mean, ys_std = np.loadtxt(Y_TXT)
    else:
        ys_mean, ys_std = y_scramble(lambda: make_rf(SEED), X_tr, y[tr], n=30)
        np.savetxt(Y_TXT, [ys_mean, ys_std], fmt='%.6f')

    # Beautiful plots: confusion matrix + summary
    plot_confusion_matrix_pastel(rf, X_te, y[te], CM_PNG)
    pr = average_precision_score(y[te], rf.predict_proba(X_te)[:,1])
    acc = accuracy_score(y[te], rf.predict_proba(X_te)[:,1] > 0.5)
    summary_figure({
        'roc_med': roc_med,
        'roc_lo': roc_lo,
        'roc_hi': roc_hi,
        'pr': pr,
        'acc': acc,
        'ys_mean': ys_mean,
        'ys_std': ys_std
    }, SUM_PNG)

    # Terminal output
    print(f"Bootstrap ROC-AUC {roc_med:.3f} [ {roc_lo:.3f} ; {roc_hi:.3f} ]")
    print(f"Y-scramble AUC {ys_mean:.3f} ± {ys_std:.3f}")
    if CV_TXT.exists():
        cv = np.loadtxt(CV_TXT)
    else:
        cv = cross_val_score(
            make_rf(SEED),
            X_full[sel].values,
            y,
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED),
            scoring='roc_auc',
            n_jobs=-1
        )
        np.savetxt(CV_TXT, cv, fmt='%.6f')
    print(f"CV AUC median={np.median(cv):.3f} (min={cv.min():.3f} max={cv.max():.3f})")

if __name__ == '__main__':
    main()



