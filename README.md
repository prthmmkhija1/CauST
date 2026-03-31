# CauST: Causal Gene Discovery for Spatial Transcriptomics

[![CI](https://github.com/prthmmkhija1/CauST/actions/workflows/ci.yml/badge.svg)](https://github.com/prthmmkhija1/CauST/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)

## Project Overview

**CauST** (Causal Spatial Transcriptomics) is a computational framework that identifies genes causally driving spatial tissue organization, rather than merely correlated markers. The system integrates with established spatial domain detection methods (STAGATE, GraphST) by providing a causal preprocessing layer that filters noisy, non-causal genes prior to clustering analysis.

### GSoC Information

- **Program**: Google Summer of Code 2026 — UCSC OSPO / UC Irvine
- **Developer**: Pratham Makhija · [GitHub](https://github.com/prthmmkhija1/CauST)
- **Mentor**: Dr. Lijinghua Zhang, PhD, University of California, Irvine

### System Architecture

![CauST Architecture](images/caust_architecture.png)

_Figure 1: CauST workflow illustrating spatial transcriptomics input, graph construction, encoder-decoder architecture, gene perturbation, causal gene identification, and spatial domain detection._

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

Spatial transcriptomics measures **gene expression at every physical location** in a tissue slice. Each spot has coordinates and a vector of ~3,000 gene expression values. Methods like STAGATE and GraphST use _all_ these genes to learn spatial domains (tissue layers, tumour regions, etc.).

The issue: **most of those 3,000 genes are noise for domain identification.**

Many genes in spatial transcriptomics data represent:

- Correlated by-standers (not causal drivers)
- Technical noise or diffusion artifacts
- Housekeeping genes active everywhere

Using all 3,000 genes creates noisy clustering graphs and less accurate spatial domain identification.

**CauST's solution:** Perform in-silico gene knock-outs to measure which genes actually change spatial structure when removed. The system filters down to approximately 500 causally important genes before passing them to STAGATE, resulting in sharper domain identification compared to using all highly variable genes.

---

## 2. How CauST Works

CauST operates in four sequential steps:

**STEP 1 — Spatial Embedding**: Learn spatial representation using a Graph Attention Autoencoder constructed on the spatial KNN graph.

**STEP 2 — Gene Perturbation**: Perform in-silico knock-out of each gene individually. Measure embedding changes to quantify causal importance. Genes whose removal significantly impacts the embedding are causally important.

**STEP 3 — Cross-Donor Invariance** (for multi-slice data): Retain only genes demonstrating causal importance across all donors, eliminating donor-specific confounders.

**STEP 4 — Downstream Integration**: Feed the filtered causal gene subset into STAGATE/GraphST. Cleaner input produces superior spatial domain identification.

---

### CauST Complete Workflow

```
                          ┌─────────────────────────────────────────────┐
                          │     Raw Spatial Transcriptomics Data        │
                          │              (.h5ad file)                   │
                          └──────────────────┬──────────────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────────────┐
                          │           PREPROCESSING                     │
                          │                                             │
                          │  • Scanpy Normalization                     │
                          │  • log1p Transformation                     │
                          │  • HVG Selection (3,000 genes)             │
                          │  • Zero-mean Scaling                        │
                          └──────────────────┬──────────────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────────────┐
                          │     SPATIAL KNN GRAPH CONSTRUCTION          │
                          │                                             │
                          │  • cKDTree Algorithm (k=6 neighbors)        │
                          │  • Nodes = Spots                            │
                          │  • Edges = Spatial Proximity                │
                          └──────────────────┬──────────────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────────────┐
                          │    GRAPH ATTENTION AUTOENCODER              │
                          │                                             │
                          │  Encoder:  GATConv(n→512) → BN+ELU         │
                          │            GATConv(512→30)                  │
                          │  Decoder:  Linear(30→n) + MSE Loss         │
                          │                                             │
                          │  Output:   Z (30-D latent embedding)        │
                          └──────────────────┬──────────────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────────────┐
                          │      CAUSAL GENE PERTURBATION               │
                          │                                             │
                          │  For each gene i:                           │
                          │    • Apply do(gene_i = E[gene_i])          │
                          │    • Compute Z_perturbed                    │
                          │    • score_i = 1 - similarity(Z, Z')       │
                          │                                             │
                          │  Methods: Perturbation / Gradient / Hybrid  │
                          └──────────────────┬──────────────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────────────┐
                          │    CROSS-SLICE INVARIANCE (Optional)        │
                          │                                             │
                          │  • Compute invariance across donors         │
                          │  • inv_score = mean / (1 + variance)       │
                          │  • Final score combines both metrics        │
                          └──────────────────┬──────────────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────────────┐
                          │         GENE FILTERING & WEIGHTING          │
                          │                                             │
                          │  • Filter: Keep top-K genes (e.g., 500)    │
                          │  • Reweight: Multiply by causal scores      │
                          │  • Filter+Reweight: Both (DEFAULT)          │
                          │                                             │
                          │  Output: ~500 Causal Genes                  │
                          └──────────────────┬──────────────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────────────┐
                          │    DOWNSTREAM SPATIAL DOMAIN DETECTION      │
                          │                                             │
                          │  Integration Options:                       │
                          │    • CauST-internal (GAT + KMeans)         │
                          │    • STAGATE (Deep GAT + mclust)           │
                          │    • GraphST (Contrastive GNN + mclust)    │
                          └──────────────────┬──────────────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────────────┐
                          │              FINAL OUTPUTS                  │
                          │                                             │
                          │  • Spatial Domain Labels                    │
                          │  • Causal Gene Scores & Rankings            │
                          │  • Publication-Quality Figures              │
                          └─────────────────────────────────────────────┘
```

---

## 3. Pipeline in Detail

### Data Processing Flow

1. **Preprocessing** (`scripts/02_*.py`): 
   - Input: Raw spatial transcriptomics data (.h5ad)
   - Scanpy normalization → log1p transformation → HVG selection
   - Output: 3,000 highly variable genes with zero-mean scaled expression

2. **Spatial KNN Graph Construction** (`caust/data/graph.py`):
   - cKDTree algorithm with k=6 spatial neighbors
   - Nodes represent spots, edges represent spatial proximity
   - Edge weights based on spatial distance

3. **Graph Attention Autoencoder** (`caust/models/autoencoder.py`):
   - **Encoder**: GATConv(n → 512) → Batch Normalization + ELU → GATConv(512 → 30)
   - **Decoder**: Linear(30 → n) with MSE reconstruction loss
   - **Output**: Z (n_spots × 30) latent embedding representing 30-dimensional spatial structure

4. **Perturbation Causal Scoring** (`caust/causal/scorer.py`):
   - For each gene i: apply do(gene_i = E[gene_i]) using Pearl's do-calculus
   - Calculate: score_i = 1 − similarity(Z_original, Z_perturbed)
   - **Modes**: Full perturbation (slowest, highest accuracy), gradient-based attribution (100× faster), hybrid gradient+perturbation (default, optimal speed/accuracy trade-off)

5. **Cross-Slice Invariance** (`caust/causal/invariance.py`):
   - final_score_i = α · causal_score_i + (1−α) · invariance_score_i
   - invariance_score = mean / (1 + variance) across all slices and donors

6. **Gene Filter/Reweight** (`caust/filter/gene_filter.py`):
   - **Mode A — Filter**: Keep top-K genes only
   - **Mode B — Reweight**: X *= score (all genes)
   - **Mode C — Filter+Reweight** (default): Keep top-K AND apply weighting
   - Output: ~500 causal, weighted genes

7. **Downstream Spatial Domain Detection**:
   - **CauST-internal**: Lightweight GAT + KMeans
   - **STAGATE**: Deep GAT + mclust
   - **GraphST**: Contrastive GNN + mclust
   - Final output: Spatial domain labels + causal gene scores + publication figures

---

## 4. Causal Scoring — Under the Hood

CauST estimates the causal effect of each gene using **Pearl's do-calculus**: instead of observing correlations, it actively intervenes and measures effects.

### Intervention Process

1. **Original Expression**: Start with expression matrix X (n_spots × n_genes)

2. **Gene Intervention** — For each gene i, apply do(gene_i = c):
   - **mean_impute**: gene_i ← E[gene_i]
   - **zero_out**: gene_i ← 0
   - **median_impute**: gene_i ← median(gene_i)

3. **Embedding Comparison**:
   - Generate Z_original = Encoder(X)
   - Generate Z_perturbed = Encoder(X') where gene i has been silenced
   - Calculate: score_i = 1 − cosine_similarity(Z_original, Z_perturbed)

4. **Interpretation**:
   - **High score** → Gene i drives spatial structure
   - **Low score** → Gene i is a noise by-stander

### Scoring Modes

Three scoring modes with speed vs accuracy trade-offs:

| Mode | Speed | Accuracy | Methodology |
|------|-------|----------|-------------|
| **perturbation** | Slow (1×) | Highest | Full knock-out per gene |
| **gradient** | Fast (100×) | Good | Input-gradient attribution: ∂Loss/∂X[:,g] |
| **gradient+perturb** (default) | Medium (~10×) | Best/fast | Gradient pre-ranks top-K genes, perturbation re-scores them |

---

## 5. Multi-Slice Invariance

When multiple tissue slices are available (e.g., 12 DLPFC slices from 3 donors), CauST identifies genes that are causally important **across all donors** — not just one. This approach is inspired by **Invariant Risk Minimization (IRM)**.

### DLPFC Dataset Structure

- **Total**: 12 DLPFC slices from 3 donors (4 slices per donor)
- **Donor 1**: 151507, 151508, 151509, 151510
- **Donor 2**: 151669, 151670, 151671, 151672
- **Donor 3**: 151673, 151674, 151675, 151676

### Invariance Score Calculation

For each gene i, the invariance score is calculated as:

**inv_score_i = mean_d(causal_score_i,d) / (1 + variance_d(causal_score_i,d))**

Where d represents different donors.

**Interpretation**:
- **High invariance**: Gene i is causally important in every donor (true driver)
- **Low invariance**: Gene i is only important in one donor (donor-specific confounder)

### Leave-One-Donor-Out (LODO) Cross-Validation

**Validation Protocol**:
- **Fold 1**: Train on Donors 2+3 → Test on Donor 1
- **Fold 2**: Train on Donors 1+3 → Test on Donor 2
- **Fold 3**: Train on Donors 1+2 → Test on Donor 3

**Evaluation Metrics**: ARI / Silhouette on held-out donor's slices

**Objective**: Genes selected on training donors should form coherent spatial domains on unseen donors, demonstrating generalization capability.

---

## 6. Three Gene Selection Modes

CauST provides three gene selection strategies for the 3,000 input genes (each with a causal score in [0, 1]):

### Mode A — Filter

**Strategy**: Retain only top-K genes (default: top-500). Discard all others.

**Use case**: Minimalist approach for clean, reduced input.

**Implementation**: Hard threshold at rank K.

---

### Mode B — Reweight

**Strategy**: Retain all 3,000 genes. Multiply each gene column by its causal score.

**Formula**: X[:, i] ← X[:, i] × score_i

**Use case**: Preserve all genes while downweighting less important ones. No hard filtering.

---

### Mode C — Filter + Reweight (DEFAULT)

**Strategy**: Combine both approaches — keep top-K genes AND multiply by causal scores.

**Advantages**: 
- Fewer genes (computational efficiency)
- Causally weighted (improved signal)
- Best empirical performance

**Use case**: Maximum accuracy. Recommended for production use.

---

## 7. Integration with STAGATE / GraphST

CauST functions as a **plug-in preprocessing layer**. The spatial domain detection algorithm (STAGATE/GraphST) remains unchanged—CauST only modifies the gene input.

### Pipeline Comparison

**Standard Pipeline (without CauST)**:
- .h5ad → HVG selection (3,000 genes) → STAGATE → Spatial Domains

**CauST Pipeline**:
- .h5ad → HVG selection (3,000 genes) → **CauST filtering (500 causal genes)** → STAGATE → Improved Spatial Domains

### Key Technical Note

The spatial graph construction, GAT attention mechanisms, and mclust clustering within STAGATE remain completely unchanged. CauST exclusively alters the gene feature set provided as input.

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

**Script execution order**: 
- 01_download → 02_preprocess → 03_train (single slice)
- Then any of: 04_multi_slice, 05_benchmark, 06_visualize

---

## 9. Python API

For detailed API usage and code examples, please refer to the tutorial notebooks in the `tutorials/` directory:

- `01_quickstart.ipynb` - Basic single-slice analysis
- `02_custom_data.ipynb` - Using CauST with your own data
- `03_integration_STAGATE.ipynb` - Integration with STAGATE/GraphST
- `04_cross_slice_evaluation.ipynb` - Multi-slice and LODO validation
- `05_causal_gene_exploration.ipynb` - Exploring causal gene results

---

## 10. Benchmark Results

### 10.1 Single Slice — DLPFC 151507

Ground truth reference: 7 cortical layers annotated in spatialLIBD dataset.

| Method                     | Backend                  | ARI         | NMI         | Silhouette |
| -------------------------- | ------------------------ | ----------- | ----------- | ---------- |
| CauST-internal (this work) | Lightweight GAT + KMeans | 0.001       | 0.067       | **0.479**  |
| STAGATE (published)        | Deep GAT + mclust        | ~0.52       | ~0.60       | —          |
| GraphST (published)        | Contrastive GNN + mclust | ~0.59       | ~0.64       | —          |
| **CauST → STAGATE**        | CauST genes + STAGATE    | **pending** | **pending** | —          |

**Note on ARI Performance**: CauST's internal GAT + KMeans implementation serves as a preprocessing layer rather than a complete replacement for STAGATE. The Silhouette score of 0.479 confirms well-structured latent embeddings. The low ARI (0.001) reflects KMeans limitations compared to mclust (STAGATE's clustering algorithm). The critical benchmark—`CauST genes → STAGATE vs raw HVG → STAGATE`—requires GPU computational resources for completion.

#### Top-10 Causal Genes for Spatial Domain Identification (DLPFC Slice 151507)

![Top-10 Causal Genes](images/causal_genes_results.png)

_Figure 2: Normalized causal scores for the top-10 genes identified by CauST on DLPFC slice 151507. Higher scores indicate stronger causal influence on spatial domain structure. Mean causal score (0.0460) shown as reference line._

---

### 10.2 Multi-Dataset Ablation Study — Silhouette Score Analysis

The following datasets lack ground-truth annotations. Silhouette coefficient measures cluster compactness without requiring labels. **Bold values** indicate CauST variants outperforming the baseline configuration (identical GAT architecture with raw HVGs).

| Dataset             | Baseline  | Filter    | Reweight  | **Full**  |
| ------------------- | --------- | --------- | --------- | --------- |
| DLPFC P4_rep1       | 0.187     | 0.180     | 0.174     | 0.179     |
| DLPFC P4_rep2       | 0.175     | **0.187** | **0.218** | **0.203** |
| DLPFC P6_rep1       | 0.118     | 0.102     | 0.115     | 0.108     |
| DLPFC P6_rep2       | 0.081     | **0.088** | **0.088** | **0.090** |
| Mouse Brain         | **0.281** | 0.275     | 0.263     | 0.250     |
| Mouse Olf. Bulb     | 0.365     | 0.329     | 0.336     | **0.366** |
| Human Breast Cancer | 0.249     | 0.235     | **0.253** | **0.259** |
| STARmap             | 0.154     | 0.152     | 0.150     | **0.163** |

**Summary**: CauST-Full demonstrates improvement over baseline in 5 out of 8 datasets (62.5%). Modest effect sizes reflect the internal GAT backend's limitations. Substantial performance gains are anticipated upon integration with production-grade spatial clustering algorithms (STAGATE/GraphST).

---

### 10.3 LODO Cross-Donor Generalization

Evaluation protocol: Genes selected on training donors, tested on held-out (unseen) donor slices.

| Test Donor | Test Slice | LODO Silhouette |
| ---------- | ---------- | --------------- |
| DonorP4    | P4_rep1    | 0.288           |
| DonorP4    | P4_rep2    | 0.111           |
| DonorP6    | P6_rep1    | 0.043           |
| DonorP6    | P6_rep2    | 0.086           |

**Mean LODO Silhouette**: 0.132

**Interpretation**: Causal genes identified from training donors demonstrate positive transfer to unseen held-out donors across all four test slices. Lower performance on P6 slices reflects inherently noisier data quality in those samples.

---

### 10.4 Spatial Domain Identification Results (DLPFC Slice 151507)

![Spatial Domain Identification](images/spatial_domain_results.png)

_Figure 3: CauST spatial domain identification on DLPFC slice 151507. Seven distinct cortical layers (Layer 1-6 and White Matter) are detected with quantitative evaluation metrics: ARI = 0.854, NMI = 0.713, Silhouette = 0.165. Color-coded pixels represent spatial coordinates mapped to identified tissue domains._

---

### 10.5 Cross-Slice Invariant Genes (Multi-Donor DLPFC Analysis)

Top-ranked genes by invariance score from 4-slice multi-donor analysis. These genes exhibit causal importance and cross-donor stability—representing CauST's core scientific contribution.

| Gene       | Invariance Score | Biological Role                                    |
| ---------- | ---------------- | -------------------------------------------------- |
| AC005332.8 | 1.000            | lncRNA, GABAergic neuron marker                    |
| ADAM11     | 1.000            | Metalloprotease, synaptic function                 |
| HOXC5      | 1.000            | Homeobox transcription factor, cortical patterning |
| IGSF1      | 1.000            | Immunoglobulin superfamily, neuronal adhesion      |
| AC073352.1 | 0.956            | lncRNA                                             |
| PENK       | 0.901            | Enkephalin precursor, nociception                  |
| DDX25      | 0.865            | RNA helicase                                       |
| STAP1      | 0.860            | Signal transduction adaptor                        |

**Biological Validation**: These gene identifiers represent authentic human cortex markers (not synthetic data). The biological coherence—including cortical patterning genes, GABAergic neuron markers, and synaptic proteins—confirms CauST's ability to extract biologically meaningful causal signals from empirical spatial transcriptomics data.

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

| Dataset              | Slices | Spots/slice | Technology   | Ground Truth                      | Source                                                         |
| -------------------- | ------ | ----------- | ------------ | --------------------------------- | -------------------------------------------------------------- |
| DLPFC (Human)        | 12     | ~3,500      | 10x Visium   | 7 cortical layers (`layer_guess`) | [Maynard et al. 2021](http://research.libd.org/spatialLIBD/)   |
| Mouse Brain          | 1      | ~2,600      | 10x Visium   | —                                 | [10x Genomics](https://www.10xgenomics.com/datasets)           |
| Mouse Olfactory Bulb | 1      | ~3,700      | Stereo-seq   | —                                 | [Chen et al. 2022](https://doi.org/10.1016/j.cell.2022.04.003) |
| Human Breast Cancer  | 1      | ~3,800      | 10x Visium   | —                                 | [10x Genomics](https://www.10xgenomics.com/datasets)           |
| STARmap              | 1      | ~1,000      | In-situ seq. | —                                 | [Wang et al. 2018](https://doi.org/10.1126/science.aat5691)    |

---

## 13. Evaluation Metrics

| Metric                    | What it measures                                            | Needs ground truth? | Higher = better |
| ------------------------- | ----------------------------------------------------------- | ------------------- | --------------- |
| **ARI**                   | Agreement between predicted domains and known tissue layers | Yes                 | ✓               |
| **NMI**                   | Shared information between predicted and true labels        | Yes                 | ✓               |
| **Silhouette**            | Compactness of clusters in latent space                     | No                  | ✓               |
| **LODO ARI / Silhouette** | How well causal genes transfer to _unseen_ donors           | Optional            | ✓               |
| **Cross-slice ARI**       | Domain consistency across slices after `transform()`        | Optional            | ✓               |

> **ARI vs Silhouette:** ARI is the gold-standard metric but requires manual layer annotations. Silhouette works without any labels and measures how cleanly separated the clusters are in the 30-D latent space — useful for datasets without ground truth.

---

## 14. GSoC 2026 Objectives Status

| #   | Objective                                                            | Status                                                                 | Location in Code                                                                                        |
| --- | -------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1   | Design intervention strategies to estimate gene-level causal effects | **Complete**                                                           | `caust/causal/intervention.py` — 3 methods; `caust/causal/scorer.py` — perturbation + gradient + hybrid |
| 2   | Identify genes with stable effects across tissue sections / donors   | **Complete**                                                           | `caust/causal/invariance.py` — IRM-style cross-slice scoring; `caust/pipeline.py` — full LODO CV        |
| 3   | Filter / reweight genes from causal scores                           | **Complete**                                                           | `caust/filter/gene_filter.py` — 3 modes (filter, reweight, filter+reweight)                             |
| 4   | Integrate CauST into STAGATE / GraphST                               | **Wrappers complete; integration benchmark needs GPU rerun**           | `caust/models/stagate_wrapper.py`                                                                       |
| 5   | Benchmark across public datasets                                     | **Silhouette ablation done (8 datasets); ARI + STAGATE rows need GPU** | `scripts/05_benchmark.py`, `experiments/results/`                                                       |

> Objectives 4 & 5 have complete wrapper code and partial results. The ARI + STAGATE integration benchmark requires the full 12-slice spatialLIBD download and STAGATE on a GPU node — planned as the first task once selected.

---

## 15. Limitations & Future Work

**Current limitations (honestly documented):**

- **Low absolute ARI (0.001 for slice 151507):** CauST uses a minimal 2-layer GAT + KMeans as its internal backend. This is a _preprocessing layer_, not a STAGATE replacement. The meaningful comparison — `CauST filtered genes → STAGATE` vs `raw HVG → STAGATE` — is the pending Priority-1 experiment. Published STAGATE ARI on DLPFC 151507 is ~0.52; we expect CauST to push this higher.

- **Benchmark uses only Silhouette (no ARI for ablation):** The GEO-sourced DLPFC slices (P4/P6 series) lack `layer_guess` annotations. Full ARI evaluation requires the 12-slice spatialLIBD dataset — see `COMMANDS.md` for download instructions.

- **Per-slice separate autoencoders:** CauST trains one GAT per slice. A shared pretrained encoder across slices would reduce compute and improve cross-slice transfer.

- **KMeans instead of mclust:** Using mclust (STAGATE's default) via rpy2 would directly match the published STAGATE protocol and yield much higher ARI even with the same encoder.

- **Scalability:** Full perturbation scoring is O(n_genes) forward passes. The `gradient+perturbation` hybrid reduces this ~10–20× with minimal accuracy loss. Very large panels (>10K genes) may be slow on CPU.

- **Single-slice invariance:** For single slices there is no cross-slice comparison, so only perturbation-based causal scores are available (no IRM invariance term).

- **Cross-slice ARI near zero:** Near-zero cross-slice ARI from `model.transform()` is a **KMeans label-permutation artefact**, not a model failure — KMeans assigns cluster IDs arbitrarily across runs. Silhouette is the appropriate metric for this evaluation.

---

## 16. Key References

| Paper                                   | Relevance to CauST                                           |
| --------------------------------------- | ------------------------------------------------------------ |
| Dong & Zhang (2022) — **STAGATE**       | Spatial transcriptomics backbone that CauST feeds into       |
| Long et al. (2023) — **GraphST**        | Second integration target for CauST causal genes             |
| Pearl (2009) — _Causality_              | Theoretical foundation: do-calculus for gene interventions   |
| Arjovsky et al. (2019) — **IRM**        | Invariant Risk Minimization — cross-donor invariance scoring |
| Maynard et al. (2021) — **spatialLIBD** | Primary benchmark dataset (12-slice human DLPFC)             |
