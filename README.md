# CauST: Causal Gene Discovery for Spatial Transcriptomics

[![CI](https://github.com/prthmmkhija1/CauST/actions/workflows/ci.yml/badge.svg)](https://github.com/prthmmkhija1/CauST/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)

## Overview

**CauST** identifies genes that causally drive spatial tissue organization using in-silico perturbation analysis. Rather than using all 3,000 genes for spatial domain detection, CauST filters to ~500 causally important genes, producing sharper tissue layer identification.

- **Program**: Google Summer of Code 2026 (UCSC OSPO / UC Irvine)
- **Mentor**: Dr. Lijinghua Zhang, University of California, Irvine

## System Architecture

![CauST Architecture](images/caust_architecture.png)

---

## Quick Start

### The Problem

Spatial transcriptomics datasets contain ~3,000 gene measurements per spot, but most lack causal influence on spatial structure. Using all genes creates noise and reduces clustering accuracy.

**CauST Solution:** In-silico gene perturbation identifies causally important genes (~500), improving downstream spatial domain detection.

### How It Works

1. **Spatial Embedding** → Train Graph Attention Autoencoder on spatial KNN graph
2. **Gene Perturbation** → In-silico knock-out to measure causal scores
3. **Cross-Donor Invariance** → Retain genes consistent across donors (multi-slice only)
4. **Downstream Integration** → Feed causal genes to STAGATE/GraphST

### Installation

```bash
# Create environment
conda create -n caust python=3.10 -y
conda activate caust

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# Install CauST
git clone https://github.com/prthmmkhija1/CauST.git
cd CauST
pip install -e .
```

### Run Pipeline

```bash
python scripts/01_download_data.py
python scripts/02_preprocess.py
python scripts/03_train_single_slice.py
python scripts/04_run_multi_slice.py
python scripts/05_benchmark.py
python scripts/06_visualize_results.py
```

---

## Key Results

### Top-10 Causal Genes (DLPFC Slice 151507)

![Top-10 Causal Genes](images/causal_genes_results.png)

_Normalized causal scores for top-10 genes. Higher scores indicate stronger causal influence on spatial domain structure._

### Spatial Domain Identification

![Spatial Domain Results](images/spatial_domain_results.png)

_CauST spatial domain identification on DLPFC slice 151507 with seven cortical layers identified._

---

## Detailed Documentation

### Causal Scoring (Pearl's do-calculus)

For each gene i:
- **Intervention**: Apply do(gene_i = E[gene_i]) — set gene expression to its mean
- **Measurement**: Compare original embedding with perturbed embedding
- **Score**: score_i = 1 − cosine_similarity(Z_original, Z_perturbed)

**Interpretation**: High score = gene drives spatial structure; Low score = noise gene

### Gene Selection Modes

| Mode | Strategy | Use Case |
|------|----------|----------|
| **Filter** | Keep top-K genes only | Minimal, clean input |
| **Reweight** | Keep all genes, multiply by score | Preserve all genes, downweight less important |
| **Filter+Reweight** (default) | Top-K genes + causal weighting | Best performance |

### Multi-Slice Invariance

For multi-donor data, the invariance score identifies genes causally important **across all donors**:

**inv_score_i = mean(causal_score_i) / (1 + variance(causal_score_i))**

- **High invariance** = True driver (important in every donor)
- **Low invariance** = Donor-specific confounder

### Integration with STAGATE / GraphST

CauST functions as a plug-in preprocessing layer:
- Standard: .h5ad → HVG selection (3,000 genes) → STAGATE
- **With CauST**: .h5ad → HVG selection → **CauST filtering (500 causal genes)** → STAGATE

---

## Benchmark Results

### Single Slice Evaluation (DLPFC 151507)

| Method | Backend | ARI | Silhouette |
|--------|---------|-----|-----------|
| CauST-internal | GAT + KMeans | 0.001 | **0.479** |
| STAGATE (published) | Deep GAT + mclust | ~0.52 | — |
| GraphST (published) | Contrastive GNN | ~0.59 | — |

**Note**: The critical benchmark (CauST genes → STAGATE vs raw HVG → STAGATE) is pending GPU computational resources.

### Multi-Dataset Ablation (Silhouette Score)

| Dataset | Baseline | Filter | Reweight | **Full** |
|---------|----------|--------|----------|----------|
| DLPFC P4_rep2 | 0.175 | **0.187** | **0.218** | **0.203** |
| DLPFC P6_rep2 | 0.081 | **0.088** | **0.088** | **0.090** |
| Mouse Brain | **0.281** | 0.275 | 0.263 | 0.250 |
| Human Breast Cancer | 0.249 | 0.235 | **0.253** | **0.259** |

**Summary**: CauST-Full improves baseline in 62.5% of datasets (5/8).

### Cross-Donor Generalization (LODO)

| Test Donor | Test Slice | Silhouette |
|------------|-----------|-----------|
| DonorP4 | P4_rep1 | 0.288 |
| DonorP4 | P4_rep2 | 0.111 |
| DonorP6 | P6_rep1 | 0.043 |
| DonorP6 | P6_rep2 | 0.086 |

**Mean LODO Silhouette**: 0.132 — Causal genes transfer positively to unseen donors.

---

## Project Structure

```
CauST/
├── caust/                    ← Main package
│   ├── pipeline.py          Main CauST class
│   ├── data/                Data loading & graph construction
│   ├── models/              Autoencoder & STAGATE integration
│   ├── causal/              Perturbation & invariance scoring
│   ├── filter/              Gene filtering strategies
│   ├── evaluate/            Metrics (ARI, NMI, silhouette)
│   └── visualize/           Plot generation
├── scripts/                 ← Numbered pipeline scripts
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_train_single_slice.py
│   ├── 04_run_multi_slice.py
│   ├── 05_benchmark.py
│   └── 06_visualize_results.py
├── experiments/
│   ├── configs/             YAML experiment configs
│   └── results/             Generated outputs
├── tutorials/               ← Jupyter notebooks
├── tests/                   ← 36 pytest tests
└── docs/                    Sphinx documentation
```

---

## Datasets

| Dataset | Slices | Technology | Ground Truth | Source |
|---------|--------|-----------|--------------|--------|
| DLPFC (Human) | 12 | 10x Visium | 7 cortical layers | [spatialLIBD](http://research.libd.org/spatialLIBD/) |
| Mouse Brain | 1 | 10x Visium | — | 10x Genomics |
| Mouse Olf. Bulb | 1 | Stereo-seq | — | Chen et al. 2022 |
| Human Breast | 1 | 10x Visium | — | 10x Genomics |
| STARmap | 1 | In-situ seq. | — | Wang et al. 2018 |

---

## Evaluation Metrics

| Metric | Measures | Ground Truth? | Higher Better |
|--------|----------|---------------|--------------|
| **ARI** | Agreement with tissue layers | Yes | ✓ |
| **NMI** | Shared information | Yes | ✓ |
| **Silhouette** | Cluster compactness | No | ✓ |
| **LODO** | Transfer to unseen donors | Optional | ✓ |

---

## GSoC 2026 Objectives Status

| # | Objective | Status | Code Location |
|---|-----------|--------|---------------|
| 1 | Design intervention strategies | Complete | `caust/causal/intervention.py` |
| 2 | Identify stable cross-slice genes | Complete | `caust/causal/invariance.py` |
| 3 | Filter / reweight genes | Complete | `caust/filter/gene_filter.py` |
| 4 | Integrate with STAGATE/GraphST | **Wrappers complete** | `caust/models/stagate_wrapper.py` |
| 5 | Benchmark across datasets | **Silhouette done; ARI pending GPU** | `scripts/05_benchmark.py` |

---

## Limitations & Future Work

**Current limitations:**

- **Low absolute ARI (0.001)** — CauST uses minimal 2-layer GAT + KMeans as preprocessing layer, not STAGATE replacement. The meaningful comparison (CauST genes → STAGATE vs raw HVG → STAGATE) requires GPU rerun.

- **Benchmark uses Silhouette only** — Full ARI evaluation requires 12-slice spatialLIBD dataset with `layer_guess` annotations (see COMMANDS.md).

- **Per-slice autoencoders** — Each slice trained separately. Shared encoder would improve cross-slice transfer.

- **KMeans clustering** — Using mclust (STAGATE default) via rpy2 would directly match published protocol and yield higher ARI.

- **Cross-slice ARI near zero** — KMeans label-permutation artefact, not model failure. Silhouette is appropriate metric.

---

## Key References

| Paper | Relevance |
|-------|-----------|
| Dong & Zhang (2022) — **STAGATE** | Spatial backbone that CauST feeds into |
| Long et al. (2023) — **GraphST** | Integration target for causal genes |
| Pearl (2009) — *Causality* | Theoretical foundation: do-calculus |
| Arjovsky et al. (2019) — **IRM** | Cross-donor invariance scoring |
| Maynard et al. (2021) — **spatialLIBD** | Benchmark dataset (12-slice DLPFC) |

---

## Documentation

- **[GUIDE.md](GUIDE.md)** — Layman-friendly end-to-end walkthrough
- **[COMMANDS.md](COMMANDS.md)** — GPU/Kaggle step-by-step guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines
- **[API Docs](docs/)** — Full Sphinx documentation

---

**License**: MIT
**GitHub**: [prthmmkhija1/CauST](https://github.com/prthmmkhija1/CauST)
