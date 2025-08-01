# Reverse-QSAR and Fragment-Driven Design for Anticancer Molecules

<br>[![REFIDD](https://img.shields.io/badge/ReFiDD-From%20fragments%20to%20first--in--class%20hits%20in%20one%20automated%20pipeline-blue?style=for-the-badge&logo=python&logoColor=orange)](#)<br>



---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Screenshots](#screenshots)
4. [Getting Started](#getting-started)
5. [Project Structure](#project-structure)
6. [Pipeline Steps](#pipeline-steps)
7. [Configuration](#configuration)
8. [Results](#results)
9. [License](#license)
10. [Contact](#contact)

---
> [!NOTE]
> Useful information that users should know, even when skimming content.
## Overview

Design and evaluate anticancer molecules with an **explainable Random Forest QSAR** model.  The workflow covers:

* Fetching all target-specific molecules from CheMBL
* 
* descriptor generation (RDKit + Mordred)
* Boruta feature selection
* balanced RF training with bootstrap validation
* SHAP global & local interpretation
* fragment enrichment analysis
* Island‑style GA that recombines BRICS fragments into new 4‑thiazolidinone analogues

---

## Key Features

| Module           | Highlights                                   |
| ---------------- | -------------------------------------------- |
| Modelling        | Balanced RF, Boruta, bootstrap AUC ± CI      |
| Interpretability | SHAP bar / beeswarm / waterfall              |
| Fragmentation    | BRICS & custom SMARTS library                |
| Generation       | 4 parallel islands, QED + QSAR fitness       |
| Robustness       | y‑scrambling, reliability curves             |
| Output           | ready‑to‑dock SDF, publication‑grade figures |

---

## Screenshots

<p align="center">
<img src="results/Evaluation_qsar_model/shap_beeswarm.png" width="45%">
<img src="results/new_compounds/first_10_hits.png"   width="45%">
</p>

---

## Getting Started

### Prerequisites

* macOS 13+ / Linux
* Python 3.10 (Conda recommended)

```bash
conda env create -n qsar-env -f environment.yml
conda activate qsar-env
```

### Quick run

```bash
python main.py --config configs/default.yaml --run-all
```

---

## Project Structure

```text
reverse-qsar-pipeline/
├─ configs/
│   └─ default.yaml
├─ Deskryptory/
│   ├─ Data_preparation.py
│   ├─ Similarity.py
│   └─ results/
│       ├─ Evaluation_qsar_model/
│       ├─ Defragmentation_results/
│       └─ GA_Island_Hits/
├─ environment.yml
├─ requirements.txt
├─ main.py
└─ README.md
```

---

## Pipeline Steps

1. **Data Preparation** – sanitise SMILES, calculate descriptors.
2. **QSAR Training** – Boruta + balanced RF, bootstrap validation.
3. **SHAP Analysis** – global feature importance, per‑sample plots.
4. **Fragment Enrichment** – identify top 20 fragments (odds ratio).
5. **Island GA** – generate & score new molecules (QED + QSAR).
6. **Reporting** – ROC/PR, reliability, top hits, SHAP figures.

---

## Configuration

All parameters live in `configs/default.yaml` – override via CLI:

```bash
python main.py --override model.n_estimators=2000
```

---

## Results

Key outputs (under `results/`):

* `roc.png`, `pr.png` – performance metrics
* `shap_beeswarm.png`, `shap_bar.png` – descriptor impact
* `first_10_hits.png`, `top100_hits.tsv` – generated molecules

---

## License

Research‑only. For commercial use contact the author.

---

## Contact

**Tomasz Szostek** – PhD Candidate, University of Milano‑Bicocca
Email: [tomasz.szostek@example.com](mailto:tomasz.szostek@example.com)

---

*Happy modelling!*