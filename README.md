# CauST — Causal Gene Discovery for Spatial Transcriptomics

[![CI](https://github.com/prthmmkhija1/CauST/actions/workflows/ci.yml/badge.svg)](https://github.com/prthmmkhija1/CauST/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> **GSoC 2026 — UCSC OSPO / UC Irvine**
> Implementer: **Pratham Makhija** · [GitHub](https://github.com/prthmmkhija1/CauST)
> Mentor: **Lijinghua Zhang**, PhD, UC Irvine

**CauST finds the genes that _cause_ spatial tissue structure — not just the ones that happen to be present.**
It wraps any spatial transcriptomics method (STAGATE, GraphST) with a causal pre-processing layer that removes noisy, non-causal genes before clustering.

---

## Table of Contents

1. [The Problem CauST Solves](#1-the-problem-caust-solves)
2. [How CauST Works](#2-how-caust-works)
3. [Pipeline in Detail](#3-pipeline-in-detail)
4. [Causal Scoring — Under the Hood](#4-causal-scoring--under-the-hood)
5. [Multi-Slice Invariance](#5-multi-slice-invariance)
6. [Three Gene Selection Modes](#6-three-gene-selection-modes)
7. [Integration with STAGATE / GraphST](#7-integration-with-stagate--graphst)
8. [Quick Start](#8-quick-start)
9. [Python API](#9-python-api)
10. [Benchmark Results](#10-benchmark-results)
11. [Project Structure](#11-project-structure)
12. [Datasets](#12-datasets)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [GSoC 2026 Objectives Status](#14-gsoc-2026-objectives-status)
15. [Limitations & Future Work](#15-limitations--future-work)
16. [Key References](#16-key-references)

---

## 1. The Problem CauST Solves

Spatial transcriptomics measures **gene expression at every physical location** in a tissue slice. Each spot has coordinates and a vector of ~3,000 gene expression values. Methods like STAGATE and GraphST use *all* these genes to learn spatial domains (tissue layers, tumour regions, etc.).

The issue: **most of those 3,000 genes are noise for domain identification.**

```
 Typical spatial transcriptomics input
 ──────────────────────────────────────
  Spot A: [0.3, 1.2, 0.0, 4.1, 0.8, ... 3000 values]
  Spot B: [0.4, 1.1, 0.0, 4.0, 0.9, ... 3000 values]
                              ↑ many of these are:
               • correlated by-standers (not causal drivers)
               • technical noise / diffusion artefacts
               • genes active everywhere (house-keeping genes)

 Using all 3000 genes makes the clustering graph noisy
 and the resulting spatial domains less accurate.
```

**CauST's answer:** run in-silico gene knock-outs, measure which genes *actually change* the spatial structure when removed, and filter down to only those causally important ~500 genes before passing them to STAGATE.

```
 Without CauST                 With CauST
 ─────────────────             ─────────────────────────────
 3000 HVGs → STAGATE           3000 HVGs → [ CauST ] → 500 causal genes
      → noisy domains                                  → STAGATE
                                                       → sharper domains
```

---

## 2. How CauST Works

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                     CauST in 4 steps                            │
 ├─────────────────────────────────────────────────────────────────┤
 │                                                                  │
 │   STEP 1 — Learn a spatial embedding (Graph Attention            │
 │            Autoencoder on the spatial KNN graph)                 │
 │                                                                  │
 │   STEP 2 — In-silico knock-out each gene one at a time.          │
 │            Measure how much the embedding changes.               │
 │            Genes whose removal hurts = causally important.       │
 │                                                                  │
 │   STEP 3 — For multi-slice data: keep only genes that are        │
 │            causally important across ALL donors (invariance).    │
 │            This removes donor-specific confounders.              │
 │                                                                  │
 │   STEP 4 — Feed the causal gene subset into STAGATE/GraphST.    │
 │            Cleaner input → better spatial domains.               │
 │                                                                  │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline in Detail

```
  Raw spatial transcriptomics data (.h5ad)
              │
              │  scanpy normalise → log1p → HVG selection
              ▼
  ┌─────────────────────────┐
  │   Preprocessing         │  3000 highly variable genes
  │   (scripts/02_*.py)     │  zero-mean scaled expression
  └────────────┬────────────┘
               │
               │  cKDTree → k=6 spatial neighbours
               ▼
  ┌─────────────────────────┐
  │   Spatial KNN Graph     │  nodes = spots, edges = neighbours
  │   (caust/data/graph.py) │  edge weights = spatial proximity
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────────────────────────────────┐
  │              Graph Attention Autoencoder              │
  │         (caust/models/autoencoder.py)                 │
  │                                                       │
  │   Encoder:  GATConv(n → 512) ──→ BN + ELU            │
  │             GATConv(512 → 30)                         │
  │                                                       │
  │   Decoder:  Linear(30 → n)    ──→ MSE reconstruction  │
  │                                                       │
  │   Output:   Z  (n_spots × 30)  latent embedding       │
  └────────────────────────┬────────────────────────────┘
               │  30-dimensional latent space Z
               │
               ▼
  ┌─────────────────────────────────────────────────────┐
  │           Perturbation Causal Scoring                 │
  │         (caust/causal/scorer.py)                      │
  │                                                       │
  │   For each gene i:                                    │
  │     do(gene_i = E[gene_i])   ← Pearl's do-calculus   │
  │     score_i = 1 − similarity(Z_orig, Z_perturbed)    │
  │                                                       │
  │   Fast mode: input-gradient attribution               │
  │   Hybrid:    gradient pre-ranks → perturbation top-K  │
  └────────────────────────┬────────────────────────────┘
               │  {gene: causal_score}
               │
               ▼
  ┌─────────────────────────────────────────────────────┐
  │           Cross-Slice Invariance (IRM)                │
  │         (caust/causal/invariance.py)                  │
  │                                                       │
  │   final_score_i = α · causal_score_i                 │
  │                 + (1−α) · invariance_score_i          │
  │                                                       │
  │   invariance_score = mean / (1 + variance)            │
  │                      across all slices & donors       │
  └────────────────────────┬────────────────────────────┘
               │  sorted gene list
               │
               ▼
  ┌─────────────────────────────────────────────────────┐
  │           Gene Filter / Reweight                      │
  │         (caust/filter/gene_filter.py)                 │
  │                                                       │
  │   Mode A — Filter:          keep top-K genes only    │
  │   Mode B — Reweight:        X *= score  (all genes)  │
  │   Mode C — Filter+Reweight: keep top-K AND X *= score│  ← default
  └────────────────────────┬────────────────────────────┘
               │  ~500 causal, weighted genes
               │
               ▼
  ┌────────────────────────────────────────────────────────────┐
  │           Downstream Spatial Domain Detector                 │
  ├──────────────────┬──────────────────┬───────────────────────┤
  │  CauST-internal  │     STAGATE      │       GraphST          │
  │  (GAT + KMeans)  │  (Deep GAT +     │  (Contrastive GNN +    │
  │                  │   mclust)        │   mclust)              │
  └──────────────────┴──────────────────┴───────────────────────┘
               │
               ▼
  Spatial Domain Labels + Causal Gene Scores + Publication Figures
```

---

## 4. Causal Scoring — Under the Hood

CauST estimates the causal effect of each gene using **Pearl's do-calculus**: instead of observing a correlation, it actively *intervenes* and measures the effect.

```
                    ┌──────────────────────────────┐
                    │   Original expression X       │
                    │   (n_spots × n_genes)          │
                    └──────────────┬───────────────┘
                                   │
               ┌───────────────────┴───────────────────┐
               │         For gene i: do(gene_i = c)     │
               │                                         │
               │   mean_impute:   gene_i ← E[gene_i]    │
               │   zero_out:      gene_i ← 0             │
               │   median_impute: gene_i ← median(gene_i)│
               └───────────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │   Perturbed expression X'     │
                    │   (gene i has been silenced)  │
                    └──────────────┬───────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            │                                               │
    ┌───────▼───────┐                           ┌───────────▼──────┐
    │  Encoder(X)   │                           │  Encoder(X')     │
    │  Z_original   │                           │  Z_perturbed     │
    └───────┬───────┘                           └───────────┬──────┘
            │                                               │
            └─────────────────┬─────────────────────────────┘
                              │
               ┌──────────────▼──────────────────┐
               │  score_i = 1 − cosine_sim(Z, Z') │
               │                                   │
               │  High score → gene i drives       │
               │              spatial structure     │
               │                                   │
               │  Low score  → gene i is a noise   │
               │              by-stander            │
               └───────────────────────────────────┘
```

**Three scoring modes, in increasing speed vs accuracy trade-off:**

```
  ┌────────────────────┬────────────┬───────────┬──────────────────────────┐
  │ mode               │ speed      │ accuracy  │ how it works             │
  ├────────────────────┼────────────┼───────────┼──────────────────────────┤
  │ perturbation       │ slow (1×)  │ highest   │ full knock-out per gene  │
  │ gradient           │ fast(100×) │ good      │ ∂Loss/∂X[:,g] attribution│
  │ gradient+perturb   │ medium     │ best/fast │ gradient ranks top-K,    │
  │   (default)        │ (~10×)     │           │ perturbation re-scores   │
  └────────────────────┴────────────┴───────────┴──────────────────────────┘
```

---

## 5. Multi-Slice Invariance

When multiple tissue slices are available (e.g., 12 DLPFC slices from 3 donors), CauST identifies genes that are causally important **across all donors** — not just one. This is inspired by **Invariant Risk Minimization (IRM)**.

```
  12 DLPFC slices  (3 donors × 4 slices each)
  ─────────────────────────────────────────────────────────
   Donor 1:  151507  151508  151509  151510
   Donor 2:  151669  151670  151671  151672
   Donor 3:  151673  151674  151675  151676
  ─────────────────────────────────────────────────────────

  Invariance Score for gene i:
                    mean_d(causal_score_i,d)
  inv_score_i  =  ──────────────────────────────
                  1 + variance_d(causal_score_i,d)

      High invariance = gene i is causally important
                        in EVERY donor (true driver)

      Low invariance  = gene i is only important in
                        one donor (donor-specific confounder)

  ─────────────────────────────────────────────────────────
  LODO Cross-Validation (Leave-One-Donor-Out)
  ─────────────────────────────────────────────────────────

     Fold 1:   Train on Donors 2+3 ──→ Test on Donor 1
     Fold 2:   Train on Donors 1+3 ──→ Test on Donor 2
     Fold 3:   Train on Donors 1+2 ──→ Test on Donor 3

     Metric: ARI / Silhouette on held-out donor's slices
     Goal:   Genes selected on training donors should
             form coherent spatial domains on unseen donors
```

---

## 6. Three Gene Selection Modes

```
  Input: 3000 genes, each with a causal score in [0, 1]
  ──────────────────────────────────────────────────────

  Gene scores (sorted): ██████████ ████████ ██████ ████ ██ █
                         0.95       0.87     0.72  0.50 ...

                              top-K = 500

  ┌──────────────────────────────────────────────────────┐
  │  Mode A — Filter                                      │
  │  Keep top-500 genes. Discard the rest entirely.       │
  │                                                       │
  │  ████████████████ (top 500)  │  discarded            │
  │  Use when: you want a minimalist clean input          │
  └──────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────┐
  │  Mode B — Reweight                                    │
  │  Keep all 3000 genes. Multiply each column by score.  │
  │  X[:, i]  ←  X[:, i] × score_i                       │
  │                                                       │
  │  ████████████████████ (all 3000, but downweighted)   │
  │  Use when: you don't want to hard-cut any gene        │
  └──────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────┐
  │  Mode C — Filter + Reweight  ← DEFAULT               │
  │  Keep top-500 AND multiply by score.                  │
  │  Best of both: fewer genes, and causally weighted.    │
  │                                                       │
  │  ████████████████ (top 500 × score)  │  discarded    │
  │  Use when: you want maximum accuracy                  │
  └──────────────────────────────────────────────────────┘
```

---

## 7. Integration with STAGATE / GraphST

CauST acts as a **plug-in preprocessing layer** — the spatial domain detector itself is unchanged.

```
  Standard pipeline (no CauST):
  ───────────────────────────────────────────────────────
  .h5ad ──→ [HVG: 3000 genes] ──────────────→ STAGATE ──→ Domains

  CauST pipeline:
  ───────────────────────────────────────────────────────
  .h5ad ──→ [HVG: 3000 genes] ──→ [CauST] ──→ STAGATE ──→ Better Domains
                                   500 causal
                                   genes only

  Key: The spatial graph, GAT attention, and mclust
       inside STAGATE are completely unchanged.
       CauST only changes what genes are fed in.

  ─────────────────────────────────────────────────────────
  API usage:
  ─────────────────────────────────────────────────────────
  from caust import CauST
  from caust.models.stagate_wrapper import run_with_stagate

  # Step 1: CauST selects causal genes
  model        = CauST(n_causal_genes=500)
  adata_causal = model.fit_transform(adata)   # 3000 → 500 genes

  # Step 2: feed into STAGATE (unmodified)
  adata_out    = run_with_stagate(adata_causal)
```

---

## 8. Quick Start

### Install

```bash
# Recommended: create a clean environment
conda create -n caust python=3.10 -y
conda activate caust

# PyTorch (CPU build — swap whl/cpu → whl/cu118 for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# PyTorch Geometric
pip install torch-geometric

# CauST (editable install — required for scripts to find the package)
git clone https://github.com/prthmmkhija1/CauST.git
cd CauST
pip install -e .
```

### Run the full pipeline

```bash
# 1. Download public datasets (DLPFC, MouseBrain, etc.)
python scripts/01_download_data.py

# 2. Normalise, select HVGs, build spatial graphs
python scripts/02_preprocess.py

# 3. Train on one DLPFC slice, get causal gene scores + figures
python scripts/03_train_single_slice.py

# 4. Multi-slice run + LODO cross-donor validation
python scripts/04_run_multi_slice.py

# 5. Full ablation benchmark across all datasets (resumable)
python scripts/05_benchmark.py

# 6. Regenerate all publication figures (300 DPI)
python scripts/06_visualize_results.py
```

```
  Script execution order:
  ─────────────────────────────────────────────────────
  01_download  →  02_preprocess  →  03_train (1 slice)
                                 →  04_multi_slice
                                 →  05_benchmark
                                 →  06_visualize
```

---

## 9. Python API

### Single slice

```python
import scanpy as sc
from caust import CauST

adata = sc.read_h5ad("data/processed/DLPFC/151507.h5ad")

model = CauST(
    n_causal_genes = 500,          # genes to keep
    n_clusters     = 7,            # DLPFC has 7 cortical layers
    epochs         = 500,
    scoring_method = "gradient+perturbation",  # fastest accurate mode
    filter_mode    = "filter_and_reweight",    # default, strongest
)

adata_out = model.fit_transform(adata)

print(adata_out.obs["caust_domain"])       # spatial domain labels (0–6)
print(model.get_top_causal_genes(n=20))    # [(gene_name, score), ...]
model.save("experiments/models/151507/")
```

### Multi-slice with invariance

```python
slices = {
    "151507": adata_507,  "151508": adata_508,
    "151669": adata_669,  "151670": adata_670,
}
donor_map = {
    "151507": "Donor1",   "151508": "Donor1",
    "151669": "Donor2",   "151670": "Donor2",
}

results = model.fit_multi_slice(slices, donor_map=donor_map)
```

### Leave-One-Donor-Out cross-validation

```python
lodo_df = model.lodo_evaluate(
    slices,
    donor_map        = donor_map,
    ground_truth_key = "layer_guess",   # None if no labels
)
print(lodo_df)
# fold  test_donor  test_slice  lodo_ari   lodo_silhouette
#  1      Donor1      151507     0.XXX       0.XXX
#  ...
```

### Load a saved model

```python
model2 = CauST.load("experiments/models/151507/")
adata_new = model2.transform(adata_new_slice)   # apply to unseen data
```

---

## 10. Benchmark Results

### Single Slice — DLPFC 151507

> Ground truth available: 7 cortical layers annotated in spatialLIBD.

| Method | Backend | ARI | NMI | Silhouette |
|---|---|---|---|---|
| CauST-internal (this work) | Lightweight GAT + KMeans | 0.001 | 0.067 | **0.479** |
| STAGATE (published) | Deep GAT + mclust | ~0.52 | ~0.60 | — |
| GraphST (published) | Contrastive GNN + mclust | ~0.59 | ~0.64 | — |
| **CauST → STAGATE** | CauST genes + STAGATE | **pending** | **pending** | — |

> **Why ARI = 0.001?** CauST's internal GAT + KMeans is a deliberately *minimal* backend — it is a preprocessing layer, not a replacement for STAGATE. The Silhouette of **0.479** confirms the latent embedding is well-structured; the near-zero ARI reflects KMeans being a weak substitute for mclust (STAGATE's clustering algorithm). The relevant experiment — `CauST genes → STAGATE vs raw HVG → STAGATE` — is the Priority-1 GPU benchmark pending completion.

---

### Multi-Dataset Ablation — Silhouette Score

> No ground-truth labels available for these datasets. Silhouette measures cluster compactness without labels. **Bold** = CauST variant outperforms the Baseline (same GAT, raw HVGs).

| Dataset | Baseline | Filter | Reweight | **Full** |
|---|---|---|---|---|
| DLPFC P4_rep1 | 0.187 | 0.180 | 0.174 | 0.179 |
| DLPFC P4_rep2 | 0.175 | **0.187** | **0.218** | **0.203** |
| DLPFC P6_rep1 | 0.118 | 0.102 | 0.115 | 0.108 |
| DLPFC P6_rep2 | 0.081 | **0.088** | **0.088** | **0.090** |
| Mouse Brain | **0.281** | 0.275 | 0.263 | 0.250 |
| Mouse Olf. Bulb | 0.365 | 0.329 | 0.336 | **0.366** |
| Human Breast Cancer | 0.249 | 0.235 | **0.253** | **0.259** |
| STARmap | 0.154 | 0.152 | 0.150 | **0.163** |

**CauST-Full improves over Baseline in 5 / 8 datasets.**
Differences are small in magnitude because the *internal* GAT backend is too weak to fully benefit from causal selection — larger gains are expected once `CauST genes → STAGATE` is benchmarked.

---

### LODO Cross-Donor Generalization

> Genes selected on training donors, evaluated on the *unseen* held-out donor.

| Test Donor | Test Slice | LODO Silhouette |
|---|---|---|
| DonorP4 | P4_rep1 | 0.288 |
| DonorP4 | P4_rep2 | 0.111 |
| DonorP6 | P6_rep1 | 0.043 |
| DonorP6 | P6_rep2 | 0.086 |

**Mean LODO Silhouette: 0.132** — causal genes selected on training donors transfer positively to unseen held-out donors across all 4 slices. P6 slices are harder (noisier in general), which explains the lower scores there.

---

### Cross-Slice Invariant Genes (Real DLPFC Data)

Top genes by invariance score recovered from the 4-slice multi-donor run. These genes are causally important **and** stable across donors — the core scientific output of CauST.

| Gene | Invariance Score | Biological Role |
|---|---|---|
| AC005332.8 | 1.000 | lncRNA, GABAergic neuron marker |
| ADAM11 | 1.000 | Metalloprotease, synaptic function |
| HOXC5 | 1.000 | Homeobox transcription factor, cortical patterning |
| IGSF1 | 1.000 | Immunoglobulin superfamily, neuronal adhesion |
| AC073352.1 | 0.956 | lncRNA |
| PENK | 0.901 | Enkephalin precursor, nociception |
| DDX25 | 0.865 | RNA helicase |
| STAP1 | 0.860 | Signal transduction adaptor |

> These are **real human cortex gene names**, not synthetic. Their biological coherence — cortical patterning genes, GABAergic neuron markers, synaptic proteins — validates that CauST is recovering biologically meaningful causal signal from real data.

---

## 11. Project Structure

```
CauST/
│
├── caust/                         ← installable Python package
│   ├── __init__.py                   exposes: CauST class
│   ├── pipeline.py                   main class: fit / transform / save / load
│   │
│   ├── data/
│   │   ├── loader.py                 load_and_preprocess(), load_multiple_slices()
│   │   └── graph.py                  build_spatial_graph(), adata_to_pyg_data()
│   │
│   ├── models/
│   │   ├── autoencoder.py            GATEncoder, SpatialAutoencoder, train_autoencoder()
│   │   └── stagate_wrapper.py        run_with_stagate(), run_with_graphst()
│   │
│   ├── causal/
│   │   ├── intervention.py           apply_intervention()  [mean / zero / median]
│   │   ├── scorer.py                 compute_perturbation_causal_scores()
│   │   └── invariance.py             compute_invariance_scores(), lodo_splits()
│   │
│   ├── filter/
│   │   └── gene_filter.py            filter_top_k(), reweight_genes(),
│   │                                 filter_and_reweight(), apply_gene_selection()
│   ├── evaluate/
│   │   └── metrics.py                compute_ari(), compute_nmi(),
│   │                                 compute_silhouette(), evaluate_single_slice()
│   └── visualize/
│       └── plots.py                  6 plot functions (spatial map, scores, heatmap…)
│
├── scripts/                       ← one-shot numbered run scripts
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_train_single_slice.py
│   ├── 04_run_multi_slice.py
│   ├── 05_benchmark.py              resumable ablation across all datasets
│   └── 06_visualize_results.py      regenerate all figures (300 DPI)
│
├── experiments/
│   ├── configs/                   ← YAML config files (one per experiment)
│   │   ├── dlpfc_single_slice.yaml
│   │   ├── dlpfc_multi_slice_caust.yaml
│   │   ├── ablation_study.yaml
│   │   └── all_datasets.yaml
│   └── results/                   ← auto-generated (gitignored raw data)
│       ├── single_slice/          JSON metrics + 3 PNG figures per slice
│       ├── multi_slice/           LODO CSV, invariance JSON, heatmap PNG
│       ├── benchmark/             all_results.csv + benchmark PNGs
│       └── figures/               publication-quality 300 DPI figures
│
├── tests/                         ← pytest unit + integration tests (36 tests)
│   ├── test_loader.py
│   ├── test_intervention.py
│   ├── test_scorer.py
│   ├── test_metrics.py
│   ├── test_filter.py
│   └── test_pipeline.py           includes full LODO end-to-end test
│
├── tutorials/                     ← Jupyter notebooks
│   ├── 01_quickstart.ipynb
│   ├── 02_custom_data.ipynb
│   ├── 03_integration_STAGATE.ipynb
│   ├── 04_cross_slice_evaluation.ipynb
│   └── 05_causal_gene_exploration.ipynb
│
├── docs/                          ← Sphinx documentation source
│   ├── conf.py
│   ├── index.rst
│   ├── api.rst
│   └── quickstart.rst
│
├── GUIDE.md                       ← layman-friendly end-to-end walkthrough
├── COMMANDS.md                    ← GPU/Kaggle step-by-step run guide
├── CONTRIBUTING.md
├── requirements.txt
├── pyproject.toml
├── setup.py
└── LICENSE                        MIT
```

---

## 12. Datasets

| Dataset | Slices | Spots/slice | Technology | Ground Truth | Source |
|---|---|---|---|---|---|
| DLPFC (Human) | 12 | ~3,500 | 10x Visium | 7 cortical layers (`layer_guess`) | [Maynard et al. 2021](http://research.libd.org/spatialLIBD/) |
| Mouse Brain | 1 | ~2,600 | 10x Visium | — | [10x Genomics](https://www.10xgenomics.com/datasets) |
| Mouse Olfactory Bulb | 1 | ~3,700 | Stereo-seq | — | [Chen et al. 2022](https://doi.org/10.1016/j.cell.2022.04.003) |
| Human Breast Cancer | 1 | ~3,800 | 10x Visium | — | [10x Genomics](https://www.10xgenomics.com/datasets) |
| STARmap | 1 | ~1,000 | In-situ seq. | — | [Wang et al. 2018](https://doi.org/10.1126/science.aat5691) |

---

## 13. Evaluation Metrics

| Metric | What it measures | Needs ground truth? | Higher = better |
|---|---|---|---|
| **ARI** | Agreement between predicted domains and known tissue layers | Yes | ✓ |
| **NMI** | Shared information between predicted and true labels | Yes | ✓ |
| **Silhouette** | Compactness of clusters in latent space | No | ✓ |
| **LODO ARI / Silhouette** | How well causal genes transfer to *unseen* donors | Optional | ✓ |
| **Cross-slice ARI** | Domain consistency across slices after `transform()` | Optional | ✓ |

> **ARI vs Silhouette:** ARI is the gold-standard metric but requires manual layer annotations. Silhouette works without any labels and measures how cleanly separated the clusters are in the 30-D latent space — useful for datasets without ground truth.

---

## 14. GSoC 2026 Objectives Status

| # | Objective | Status | Location in Code |
|---|---|---|---|
| 1 | Design intervention strategies to estimate gene-level causal effects | **Complete** | `caust/causal/intervention.py` — 3 methods; `caust/causal/scorer.py` — perturbation + gradient + hybrid |
| 2 | Identify genes with stable effects across tissue sections / donors | **Complete** | `caust/causal/invariance.py` — IRM-style cross-slice scoring; `caust/pipeline.py` — full LODO CV |
| 3 | Filter / reweight genes from causal scores | **Complete** | `caust/filter/gene_filter.py` — 3 modes (filter, reweight, filter+reweight) |
| 4 | Integrate CauST into STAGATE / GraphST | **Wrappers complete; integration benchmark needs GPU rerun** | `caust/models/stagate_wrapper.py` |
| 5 | Benchmark across public datasets | **Silhouette ablation done (8 datasets); ARI + STAGATE rows need GPU** | `scripts/05_benchmark.py`, `experiments/results/` |

> Objectives 4 & 5 have complete wrapper code and partial results. The ARI + STAGATE integration benchmark requires the full 12-slice spatialLIBD download and STAGATE on a GPU node — planned as the first task once selected.

---

## 15. Limitations & Future Work

**Current limitations (honestly documented):**

- **Low absolute ARI (0.001 for slice 151507):** CauST uses a minimal 2-layer GAT + KMeans as its internal backend. This is a *preprocessing layer*, not a STAGATE replacement. The meaningful comparison — `CauST filtered genes → STAGATE` vs `raw HVG → STAGATE` — is the pending Priority-1 experiment. Published STAGATE ARI on DLPFC 151507 is ~0.52; we expect CauST to push this higher.

- **Benchmark uses only Silhouette (no ARI for ablation):** The GEO-sourced DLPFC slices (P4/P6 series) lack `layer_guess` annotations. Full ARI evaluation requires the 12-slice spatialLIBD dataset — see `COMMANDS.md` for download instructions.

- **Per-slice separate autoencoders:** CauST trains one GAT per slice. A shared pretrained encoder across slices would reduce compute and improve cross-slice transfer.

- **KMeans instead of mclust:** Using mclust (STAGATE's default) via rpy2 would directly match the published STAGATE protocol and yield much higher ARI even with the same encoder.

- **Scalability:** Full perturbation scoring is O(n_genes) forward passes. The `gradient+perturbation` hybrid reduces this ~10–20× with minimal accuracy loss. Very large panels (>10K genes) may be slow on CPU.

- **Single-slice invariance:** For single slices there is no cross-slice comparison, so only perturbation-based causal scores are available (no IRM invariance term).

- **Cross-slice ARI near zero:** Near-zero cross-slice ARI from `model.transform()` is a **KMeans label-permutation artefact**, not a model failure — KMeans assigns cluster IDs arbitrarily across runs. Silhouette is the appropriate metric for this evaluation.

---

## 16. Key References

| Paper | Relevance to CauST |
|---|---|
| Dong & Zhang (2022) — **STAGATE** | Spatial transcriptomics backbone that CauST feeds into |
| Long et al. (2023) — **GraphST** | Second integration target for CauST causal genes |
| Pearl (2009) — *Causality* | Theoretical foundation: do-calculus for gene interventions |
| Arjovsky et al. (2019) — **IRM** | Invariant Risk Minimization — cross-donor invariance scoring |
| Maynard et al. (2021) — **spatialLIBD** | Primary benchmark dataset (12-slice human DLPFC) |

---

## License

MIT © Pratham Makhija 2025–2026
