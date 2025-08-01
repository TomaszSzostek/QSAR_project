"""
Genetic-Island Fragment – Enhanced Diversity Scaffold (QSAR + QED + Exclusion + Diversity)
────────────────────────────────────────────────────────────────────────────────────────
**Purpose**
Generate a diverse library (exactly 100 top hits) of unique and new
small molecules *not present* in a reference file (``EXCLUDE_FILE``) by recombining
BRICS-derived fragments with full wildcard flexibility, hydrogen/halogen capping, and
a triple-objective fitness (QSAR + QED + Diversity).
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import csv, functools, random, warnings, signal, re, os, multiprocessing as mp
from typing import List, Tuple, Dict, Set

import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import (
    Descriptors, Crippen, Lipinski, rdMolDescriptors, FilterCatalog, Draw, rdchem, QED, AllChem
)
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from mordred import Calculator, descriptors as md

# ────────── CONFIG ──────────
FRAG_SMI     = "results/Defragmentation_results/RandomForest/selected_fragments.smi"
DESCR_FILE   = "results/Evaluation_qsar_model/selected_descriptors.csv"
MODEL_FILE   = "results/Evaluation_qsar_model/rf.pkl"
EXCLUDE_FILE = "data_sets/data/processed/final_dataset.bak"
OUT_DIR      = Path("results/new_compounds"); OUT_DIR.mkdir(exist_ok=True)
EXP_FRAG_CSV = OUT_DIR / "fragments_library.csv"
HITS_TSV     = OUT_DIR / "top100_hits.tsv"
GRID_PNG     = OUT_DIR / "first_10_hits.png"

# GA hyperparameters
N_ISLANDS      = 4
POP_PER_ISLAND = 200
GENERATIONS    = 300
ELITE_KEEP     = 0.02
MIGRATE_EPOCH  = 20
P_CROSSOVER    = 0.9
P_MUTATE_NODE  = 0.3
P_SWAP_FRAG     = 0.1
P_INFLATE       = 0.05
P_DEFLATE       = 0.05
P_ADD_FRAG     = 0.15
P_REMOVE_FRAG  = 0.1
MAX_FRAGS      = 8
MIN_FRAGS      = 3

# Fitness weights
W_QSAR = 0.8
W_QED  = 0.1
W_NOV  = 0.1  # tanimoto diversity weight

# Misc
SEED       = 42
QSAR_BATCH = 4096
TOP_HITS   = 100

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", message=".*Descriptors.*")
rng_global = random.Random(SEED)

# ────────── DIVERSITY HELPERS ──────────
@functools.lru_cache(maxsize=32768)
def fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None

@functools.lru_cache(maxsize=32768)
def avg_tanimoto(smi: str, others: Tuple[str, ...]) -> float:
    f1 = fp(smi)
    if not f1 or not others: return 1.0
    sims = [DataStructs.TanimotoSimilarity(f1, fp(o)) for o in others if fp(o)]
    return 1.0 - (sum(sims)/len(sims)) if sims else 1.0

# ────────── CHEMISTRY FILTERS ──────────
_pains = FilterCatalog.FilterCatalog(FilterCatalog.FilterCatalogParams())

def passes_lipinski(m: Chem.Mol) -> bool:
    return (Descriptors.MolWt(m) <= 550 and Crippen.MolLogP(m) <= 5 and
            Lipinski.NumHDonors(m) <= 5 and Lipinski.NumHAcceptors(m) <= 10 and
            rdMolDescriptors.CalcTPSA(m) <= 140 and Lipinski.NumRotatableBonds(m) <= 8 and
            Descriptors.HeavyAtomCount(m) <= 45)

def is_good(m: Chem.Mol) -> bool:
    return bool(m) and passes_lipinski(m) and not _pains.HasMatch(m)

# ────────── QSAR & DESCRIPTORS ──────────
_RD_NAMES = [n for n,_ in Descriptors._descList]
_RD_CALC  = MolecularDescriptorCalculator(_RD_NAMES)
_MD_CALC  = Calculator(md, ignore_3D=True)
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError))

@functools.lru_cache(maxsize=32768)
def vec(smi: str, want: Tuple[str, ...]) -> Tuple[float, ...]:
    m = Chem.MolFromSmiles(smi)
    if m is None: return (float('nan'),)*len(want)
    try:
        signal.alarm(2)
        rdv = dict(zip(_RD_NAMES, _RD_CALC.CalcDescriptors(m)))
        mdv = _MD_CALC(m).asdict(); signal.alarm(0)
    except:
        signal.alarm(0)
        return (float('nan'),)*len(want)
    return tuple(rdv.get(x, mdv.get(x, float('nan'))) for x in want)

class QSARPredictor:
    def __init__(self, model, cols: List[str]):
        self.model, self.want = model, tuple(cols)
    def __call__(self, smiles: List[str]) -> np.ndarray:
        out = np.empty(len(smiles), float)
        for i in range(0, len(smiles), QSAR_BATCH):
            batch = smiles[i:i+QSAR_BATCH]
            mat = np.zeros((len(batch), len(self.want)))
            for j,s in enumerate(batch): mat[j]=vec(s,self.want)
            mat[~np.isfinite(mat)] = 0.0
            out[i:i+len(batch)] = self.model.predict_proba(mat)[:,1]
        return out

GLOBAL_PREDICT: QSARPredictor|None = None

def init_worker():
    global GLOBAL_PREDICT
    import joblib,pickle
    cols=[r.strip() for r in Path(DESCR_FILE).read_text().splitlines() if r.strip()]
    if cols and not cols[0].startswith(('EState','MolLogP','NumH','TPSA')): cols=cols[1:]
    model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else pickle.load(open(MODEL_FILE,'rb'))
    if hasattr(model,'set_params'): model.set_params(n_jobs=1)
    GLOBAL_PREDICT = QSARPredictor(model,cols)

# ────────── WILDCARD & BUILD OPS ──────────
HYDROGEN_FRAGMENT = "[H][*]"
HALOGEN_FRAGMENTS = {"[F][*]","[Cl][*]","[Br][*]","[I][*]"}
STAR_RE = re.compile(r"\[\d+\*]")

def preprocess_fragments(path: str) -> List[str]:
    raw = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
    cleaned = set(re.sub(STAR_RE, "[*]", s) for s in raw)
    cleaned |= {HYDROGEN_FRAGMENT} | HALOGEN_FRAGMENTS
    with open(EXP_FRAG_CSV,'w',newline='') as fh:
        csv.writer(fh).writerows([[s] for s in sorted(cleaned)])
    return sorted(cleaned)

def sample_frag_count(rng: random.Random) -> int:
    mid=(MIN_FRAGS+MAX_FRAGS)/2
    return int(rng.triangular(MIN_FRAGS, mid, MAX_FRAGS))

def find_stars(m: Chem.Mol) -> List[int]: return [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()==0]

def star_to_h(m: Chem.Mol) -> Chem.Mol|None:
    rw=Chem.RWMol(m)
    for idx in find_stars(rw): rw.GetAtomWithIdx(idx).SetAtomicNum(1)
    try: Chem.SanitizeMol(rw); return rw.GetMol()
    except: return None

def connect_pair(m1,m2,rng,tries=20):
    if not m1 or not m2: return None
    n1=m1.GetNumAtoms()
    for _ in range(tries):
        w1,w2=find_stars(m1),find_stars(m2)
        if not w1 or not w2: return None
        i1,i2=rng.choice(w1),rng.choice(w2)
        nei1=m1.GetAtomWithIdx(i1).GetNeighbors()[0].GetIdx()
        nei2=m2.GetAtomWithIdx(i2).GetNeighbors()[0].GetIdx()
        combo=Chem.CombineMols(m1,m2); rw=Chem.RWMol(combo)
        rw.AddBond(nei1,nei2+n1,rdchem.BondType.SINGLE)
        for idx in sorted((i2+n1,i1),reverse=True): rw.RemoveAtom(idx)
        try: Chem.SanitizeMol(rw); return rw.GetMol()
        except: continue
    return None

def custom_build(frags,rng):
    m=Chem.MolFromSmiles(frags[0],sanitize=False)
    if not m: return None
    try: Chem.SanitizeMol(m)
    except: return None
    mol=m
    for smi in frags[1:]:
        frag=Chem.MolFromSmiles(smi,sanitize=False)
        if not frag: return None
        try: Chem.SanitizeMol(frag)
        except: return None
        new=connect_pair(mol,frag,rng)
        if not new: return None
        mol=new
    mol=star_to_h(mol)
    return Chem.MolToSmiles(mol,True) if mol else None

@functools.lru_cache(maxsize=32768)
def qed_score(smi):
    m=Chem.MolFromSmiles(smi)
    return QED.qed(m) if m else 0.0

@dataclass
class Individual:
    fragments:Tuple[str,...]
    smiles:str
    qsar:float
    qed:float
    novelty:float
    fitness:float
    def as_row(self): return (self.smiles,self.fitness,self.qsar,self.qed,self.novelty)

class IslandGA:
    def __init__(self,frags,seed,ref_set):
        self.F=frags; self.rng=random.Random(seed)
        self.predict=GLOBAL_PREDICT; self.qsar_cache={}
        self.ref_set=ref_set; self.pop=[]
    def _random_frag_vector(self):
        n=sample_frag_count(self.rng)
        return tuple(self.rng.sample(self.F,n))
    def _init_population(self):
        pbar=tqdm(total=POP_PER_ISLAND,desc="Init pop",leave=False)
        while len(self.pop)<POP_PER_ISLAND:
            vec=self._random_frag_vector(); smi=custom_build(list(vec),self.rng)
            if smi and smi not in self.ref_set and is_good(Chem.MolFromSmiles(smi)):
                self.pop.append(Individual(vec,smi,0,0,0,0)); pbar.update(1)
        pbar.close()
    def _batch_evaluate(self,inds):
        smiles=[ind.smiles for ind in inds if ind.smiles not in self.qsar_cache]
        if smiles:
            preds=self.predict(smiles)
            for smi,p in zip(smiles,preds): self.qsar_cache[smi]=float(p)
        pop_smiles=tuple(ind.smiles for ind in inds)
        for ind in inds:
            ind.qsar=self.qsar_cache[ind.smiles]
            ind.qed=qed_score(ind.smiles)
            ind.novelty=avg_tanimoto(ind.smiles,pop_smiles)
            ind.fitness=W_QSAR*ind.qsar+W_QED*ind.qed+W_NOV*ind.novelty
    def _tournament(self,k=4): return max(self.rng.sample(self.pop,k),key=lambda i:i.fitness)
    def _crossover(self,p1,p2):
        n1,n2=len(p1.fragments),len(p2.fragments)
        if n1<2 or n2<2: return p1.fragments
        i1,j1=sorted(self.rng.sample(range(1,n1),2))
        i2,j2=sorted(self.rng.sample(range(1,n2),2))
        child=p1.fragments[:i1]+p2.fragments[i2:j2]+p1.fragments[j1:]
        if len(child)<MIN_FRAGS: child+=tuple(self.rng.sample(self.F,MIN_FRAGS-len(child)))
        if len(child)>MAX_FRAGS: child=child[:MAX_FRAGS]
        return child
    def _mutate(self,vec):
        frags=list(vec)
        if self.rng.random()<P_MUTATE_NODE and frags: frags[self.rng.randrange(len(frags))]=self.rng.choice(self.F)
        if self.rng.random()<P_SWAP_FRAG and len(frags)>=2:
            i,j=self.rng.sample(range(len(frags)),2); frags[i],frags[j]=frags[j],frags[i]
        if self.rng.random()<P_ADD_FRAG and len(frags)<MAX_FRAGS:
            frags.insert(self.rng.randrange(len(frags)+1),self.rng.choice(self.F))
        if self.rng.random()<P_INFLATE and len(frags)<MAX_FRAGS-1:
            for _ in range(2): frags.insert(self.rng.randrange(len(frags)+1),self.rng.choice(self.F))
        if self.rng.random()<P_REMOVE_FRAG and len(frags)>MIN_FRAGS: del frags[self.rng.randrange(len(frags))]
        if self.rng.random()<P_DEFLATE and len(frags)>MIN_FRAGS+1:
            for _ in range(2): del frags[self.rng.randrange(len(frags))]
        return tuple(frags)
    def epoch(self):
        self._batch_evaluate(self.pop)
        elite=max(1,int(ELITE_KEEP*POP_PER_ISLAND))
        next_pop=sorted(self.pop,key=lambda i:i.fitness,reverse=True)[:elite]
        pbar=tqdm(total=POP_PER_ISLAND-elite,desc="Gen acc",leave=False);
        added=0
        while added<POP_PER_ISLAND-elite:
            p1,p2=self._tournament(),self._tournament()
            vec=self._crossover(p1,p2) if self.rng.random()<P_CROSSOVER else p1.fragments
            child=self._mutate(vec)
            smi=custom_build(list(child),self.rng)
            if smi and smi not in self.ref_set and is_good(Chem.MolFromSmiles(smi)):
                next_pop.append(Individual(child,smi,0,0,0,0)); added+=1; pbar.update(1)
        pbar.close(); self.pop=next_pop
    def best(self):
        self._batch_evaluate(self.pop); return max(self.pop,key=lambda i:i.fitness)

def run_island(frags,seed,gens,mq,ref):
    isl=IslandGA(frags,seed,ref); isl._init_population()
    for g in range(gens):
        isl.epoch()
        if (g+1)%MIGRATE_EPOCH==0: mq.put(isl.best())
        while not mq.empty():
            mig=mq.get_nowait(); worst=min(isl.pop,key=lambda i:i.fitness)
            isl.pop.remove(worst); isl.pop.append(mig)
    return isl.best(),isl.pop

def load_exclude(path):
    if not path or not os.path.exists(path): return set()
    raw=[]
    if path.lower().endswith('.smi'):
        raw=[l.split()[0] for l in Path(path).read_text().splitlines() if l.strip()]
    else:
        with open(path,newline='') as fh:
            raw=[row[0] for row in csv.reader(fh) if row]
    out=set()
    for s in raw:
        m=Chem.MolFromSmiles(s);
        if m: out.add(Chem.MolToSmiles(m,True))
    return out

# ───────────────────────────────────────────────────────────────
def main():
    if HITS_TSV.exists():
        print("⚡  top100_hits.tsv found – skipping GA evolution.")
        if not GRID_PNG.exists():

            smiles = [row.split("\t", 1)[0]
                      for row in Path(HITS_TSV).read_text().splitlines()[1:TOP_HITS+1]]
            mols = [Chem.MolFromSmiles(s) for s in smiles[:9]]
            grid = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(600, 600))
            grid.save(str(GRID_PNG))
            print("✅  Grid saved →", GRID_PNG)
        else:
            print("✅  All artefacts already exists- nothing to do.")
        return


    fragments = preprocess_fragments(FRAG_SMI)
    exclude   = load_exclude(EXCLUDE_FILE)

    mq   = mp.Manager().Queue()
    args = [(fragments, SEED + i, GENERATIONS, mq, exclude) for i in range(N_ISLANDS)]
    with mp.Pool(N_ISLANDS, initializer=init_worker) as p:
        results = p.starmap(run_island, args)


    pop  = [ind for _, pop in results for ind in pop]
    pop.sort(key=lambda i: i.fitness, reverse=True)

    hits, seen = [], set()
    for ind in pop:
        if ind.smiles in seen or ind.smiles in exclude:
            continue
        hits.append(ind)
        seen.add(ind.smiles)
        if len(hits) == TOP_HITS:
            break

    if len(hits) < TOP_HITS:
        print(f"⚠  Only {len(hits)} novel molecules found.")

    with open(HITS_TSV, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["smiles", "fitness", "qsar", "qed", "novelty"])
        for h in hits:
            w.writerow(h.as_row())
    print("✅  Saved hits →", HITS_TSV)


    mols = [Chem.MolFromSmiles(h.smiles) for h in hits[:10]]
    grid = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(250, 400))
    grid.save(str(GRID_PNG))
    print("✅  Grid saved →", GRID_PNG)
# ───────────────────────────────────────────────────────────────


if __name__=="__main__":
    main()



