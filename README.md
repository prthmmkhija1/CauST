# CauST — Causal Gene Discovery for Spatial Transcriptomics

> **GSoC 2026 project — UCSC OSPO / UCI**  
> Implementer: **Pratham Makhija** · [GitHub](https://github.com/prthmmkhija1/CauST)

---

## What is CauST?

Modern spatial transcriptomics (ST) captures both _what_ genes are expressed and _where_ — giving each cell a coordinate in tissue. CauST asks a harder question:

> **Which genes are _causally_ responsible for spatial structure, not just correlated with it?**

CauST applies **Pearl's do-calculus** to in-silico gene knock-outs inside a **Graph Attention Autoencoder**. Genes whose removal disrupts the spatial embedding are scored as _causally important_. Genes that are consistently causal across multiple tissue slices receive bonus _invariance_ scores (inspired by IRM).

The resulting causal gene list then feeds into state-of-the-art spatial domain detectors (STAGATE, GraphST) to improve clustering accuracy and biological interpretability.

---

## Architecture

```
Raw spatial transcriptomics (.h5ad)
          │
          ▼
  Preprocessing (scanpy HVG)
          │
          ▼
  Spatial KNN Graph (scipy cKDTree)
          │
          ▼
  ┌─────────────────────────────┐
  │  Graph Attention Autoencoder│  ← GATConv(n→512) → BN+ELU
  │   Encoder: 2-layer GAT      │    → GATConv(512→30)
  │   Decoder: Linear(30→n)     │
  └─────────────────────────────┘
          │  30-D latent space Z
          ▼
  Perturbation Causal Scoring
    do(Gene_i = E[Gene_i])
    score_i = 1 - ARI(Z_orig, Z_perturbed)
          │
          ▼
  Cross-slice Invariance (IRM)
    final_score = α·causal + (1-α)·invariance
          │
          ▼
  Filter + Reweight (top-K genes)
          │
          ▼
  Downstream: STAGATE / GraphST / CauST-internal K-Means
          │
          ▼
  Spatial Domain Labels + Figures
```

---

## Quick Start

### 1. Install

```bash
# Create conda environment (recommended)
conda create -n caust python=3.10 -y
conda activate caust

# Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install CauST (editable)
pip install -e .
```

### 2. Download data

```bash
python scripts/01_download_data.py
```

### 3. Preprocess

```bash
python scripts/02_preprocess.py
```

### 4. Train on a single slice

```bash
python scripts/03_train_single_slice.py
```

### 5. Multi-slice + invariance analysis

```bash
python scripts/04_run_multi_slice.py
```

### 6. Full ablation benchmark

```bash
python scripts/05_benchmark.py
```

### 7. Regenerate figures

```bash
python scripts/06_visualize_results.py
```

---

## Python API

```python
import scanpy as sc
from caust import CauST

# Load preprocessed AnnData
adata = sc.read_h5ad("data/processed/DLPFC/151507.h5ad")

# Train CauST
model = CauST(n_causal_genes=500, n_clusters=7)
adata_out = model.fit_transform(adata)

# Results
print(adata_out.obs["caust_domain"])          # spatial domain labels
print(model.get_top_causal_genes(n=20))       # top causal genes

# Multi-slice with donor mapping
slices = {"151507": adata1, "151508": adata2}
donor_map = {"151507": "Donor1", "151508": "Donor1"}
results = model.fit_multi_slice(slices, donor_map=donor_map)

# Save / load (n_genes auto-detected)
model.save("experiments/models/151507/")
model2 = CauST.load("experiments/models/151507/")
```

---

## Datasets

| Dataset              | Slices | Spots        | Resolution   | Source                                                         |
| -------------------- | ------ | ------------ | ------------ | -------------------------------------------------------------- |
| DLPFC (Human)        | 12     | ~3,500/slice | 10x Visium   | [Maynard et al. 2021](http://research.libd.org/spatialLIBD/)   |
| Mouse Brain          | 1      | ~2,600       | 10x Visium   | [10x Genomics](https://www.10xgenomics.com/datasets)           |
| Mouse Olfactory Bulb | 1      | ~3,700       | Stereo-seq   | [Chen et al. 2022](https://doi.org/10.1016/j.cell.2022.04.003) |
| Human Breast Cancer  | 1      | ~3,800       | 10x Visium   | [10x Genomics](https://www.10xgenomics.com/datasets)           |
| STARmap              | 1      | ~1,000       | In-situ seq. | [Wang et al. 2018](https://doi.org/10.1126/science.aat5691)    |

---

## Project Structure

```
CauST/
├── caust/                      ← installable Python package
│   ├── __init__.py             ← exposes CauST class
│   ├── pipeline.py             ← main CauST class (fit/transform/save/load)
│   ├── data/
│   │   ├── loader.py           ← load_and_preprocess(), load_multiple_slices()
│   │   └── graph.py            ← build_spatial_graph(), adata_to_pyg_data()
│   ├── models/
│   │   ├── autoencoder.py      ← GATEncoder, SpatialAutoencoder, train_autoencoder
│   │   └── stagate_wrapper.py  ← run_with_stagate(), run_with_graphst()
│   ├── causal/
│   │   ├── intervention.py     ← apply_intervention(), apply_batch_interventions()
│   │   ├── scorer.py           ← compute_perturbation_causal_scores()
│   │   └── invariance.py       ← compute_invariance_scores(), lodo_splits()
│   ├── filter/
│   │   └── gene_filter.py      ← filter_top_k(), reweight_genes(), filter_and_reweight()
│   ├── evaluate/
│   │   └── metrics.py          ← compute_ari(), evaluate_single_slice()
│   └── visualize/
│       └── plots.py            ← spatial maps, score bars, heatmaps
│
├── scripts/                    ← one-shot run scripts
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_train_single_slice.py
│   ├── 04_run_multi_slice.py
│   ├── 05_benchmark.py
│   └── 06_visualize_results.py
│
├── experiments/
│   ├── configs/                ← YAML experiment configs
│   └── results/                ← auto-generated (gitignored)
│
├── tests/                      ← pytest unit tests
│   ├── test_loader.py
│   ├── test_intervention.py
│   ├── test_scorer.py
│   ├── test_metrics.py
│   └── test_pipeline.py
│
├── tutorials/                  ← Jupyter notebooks
│   └── 01_quickstart.ipynb
│
├── data/                       ← gitignored; created by scripts
│   ├── raw/
│   └── processed/
│
├── GUIDE.md                    ← layman-friendly end-to-end guide
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## Evaluation Metrics

| Metric          | Description                               | Higher = Better |
| --------------- | ----------------------------------------- | --------------- |
| ARI             | Adjusted Rand Index vs known layer labels | ✓               |
| NMI             | Normalized Mutual Information             | ✓               |
| Silhouette      | Compactness of latent clusters            | ✓               |
| Cross-slice ARI | LODO ARI across donors                    | ✓               |

---

## Key References

- Dong & Zhang (2022) — STAGATE: graph attention autoencoder for ST
- Long et al. (2023) — GraphST: contrastive learning for ST
- Pearl (2009) — _Causality: Models, Reasoning, and Inference_
- Arjovsky et al. (2019) — Invariant Risk Minimization (IRM)
- Maynard et al. (2021) — DLPFC spatial transcriptomics dataset

---

## License

MIT © Pratham Makhija 2025
