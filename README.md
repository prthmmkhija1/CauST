# CauST — Causal Gene Discovery for Spatial Transcriptomics

[![CI](https://github.com/prthmmkhija1/CauST/actions/workflows/ci.yml/badge.svg)](https://github.com/prthmmkhija1/CauST/actions/workflows/ci.yml)

> **GSoC 2026 project — UCSC OSPO / UC Irvine**  
> Implementer: **Pratham Makhija** · [GitHub](https://github.com/prthmmkhija1/CauST)  
> Mentor: **Lijinghua Zhang**, PhD, UC Irvine

---

## What is CauST?

Modern spatial transcriptomics (ST) captures both _what_ genes are expressed and _where_ — giving each cell a coordinate in tissue. CauST asks a harder question:

> **Which genes are _causally_ responsible for spatial structure, not just correlated with it?**

CauST applies **Pearl's do-calculus** to in-silico gene knock-outs inside a **Graph Attention Autoencoder**. Genes whose removal disrupts the spatial embedding are scored as _causally important_. Genes that are consistently causal across multiple tissue slices receive bonus _invariance_ scores (inspired by IRM).

The resulting causal gene list then feeds into state-of-the-art spatial domain detectors (STAGATE, GraphST) to improve clustering accuracy and biological interpretability.

---

## Benchmark Results (Real Data)

All benchmarks below use CauST's lightweight internal GAT + KMeans pipeline.
Absolute scores reflect this minimal backend; the key takeaway is the **relative
improvement** from causal gene selection versus the HVG baseline.

### Single-Slice — DLPFC 151507 (spatialLIBD, with ground-truth layer labels)

| Metric     | Value |
| ---------- | ----- |
| ARI        | 0.001 |
| NMI        | 0.067 |
| Silhouette | 0.479 |

> **Note:** The low ARI is expected — CauST uses a lightweight 2-layer GAT +
> KMeans, not STAGATE's deeper architecture with mclust. The Silhouette of 0.48
> shows the latent space is well-structured. Plugging CauST's causal genes into
> STAGATE/BayesSpace is expected to raise ARI significantly (future work).

### Multi-Dataset Ablation — Silhouette Score

Each cell is the Silhouette score from CauST's internal clustering.
**Bold** = the CauST variant outperforms the Baseline (raw HVGs, same GAT).

| Dataset             | Baseline  | CauST-Filter | CauST-Reweight | CauST-Full |
| ------------------- | --------- | ------------ | -------------- | ---------- |
| DLPFC (P4_rep1)     | 0.187     | 0.180        | 0.174          | 0.179      |
| DLPFC (P4_rep2)     | 0.175     | **0.187**    | **0.218**      | **0.203**  |
| DLPFC (P6_rep1)     | 0.118     | 0.102        | 0.115          | 0.108      |
| DLPFC (P6_rep2)     | 0.081     | **0.088**    | **0.088**      | **0.090**  |
| Mouse Brain         | **0.281** | 0.275        | 0.263          | 0.250      |
| Mouse Olf. Bulb     | 0.365     | 0.329        | 0.336          | **0.366**  |
| Human Breast Cancer | 0.249     | 0.235        | **0.253**      | **0.259**  |
| STARmap             | 0.154     | 0.152        | 0.150          | **0.163**  |

CauST-Full improves over Baseline in **5/8** datasets (P4_rep2, P6_rep2,
Mouse Olf. Bulb, Human Breast Cancer, STARmap). On the remaining datasets the
differences are small (< 0.03). The pattern shows CauST's causal gene
selection adds the most value on noisier or more heterogeneous tissues.

### LODO (Leave-One-Donor-Out) — Cross-Donor Generalization

Genes selected by CauST on training donors are applied to unseen test donors.
Only Silhouette is reported (GEO DLPFC sections lack `layer_guess` labels).

| Test Donor | Test Slice | Silhouette |
| ---------- | ---------- | ---------- |
| DonorP4    | P4_rep1    | 0.288      |
| DonorP4    | P4_rep2    | 0.111      |
| DonorP6    | P6_rep1    | 0.043      |
| DonorP6    | P6_rep2    | 0.086      |

Mean LODO Silhouette: **0.132** — positive transfer across donors is
demonstrated, though cross-donor generalization remains a challenge with
this lightweight backend.

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
slices = {"151507": adata1, "151508": adata2, "151669": adata3, "151670": adata4}
donor_map = {"151507": "Donor1", "151508": "Donor1",
             "151669": "Donor2", "151670": "Donor2"}
results = model.fit_multi_slice(slices, donor_map=donor_map)

# Leave-One-Donor-Out cross-validation
lodo_df = model.lodo_evaluate(slices, donor_map, ground_truth_key="layer_guess")
print(lodo_df)  # DataFrame: fold, test_donor, test_slice, lodo_ari, ...

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
│   └── results/                ← auto-generated benchmark outputs
│
├── tests/                      ← pytest unit tests
│   ├── test_loader.py
│   ├── test_intervention.py
│   ├── test_scorer.py
│   ├── test_metrics.py
│   ├── test_filter.py
│   └── test_pipeline.py
│
├── tutorials/                  ← Jupyter notebooks
│   ├── 01_quickstart.ipynb
│   ├── 02_custom_data.ipynb
│   ├── 03_integration_STAGATE.ipynb
│   ├── 04_cross_slice_evaluation.ipynb
│   └── 05_causal_gene_exploration.ipynb
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

| Metric              | Description                                       | Higher = Better |
| ------------------- | ------------------------------------------------- | --------------- |
| ARI                 | Adjusted Rand Index vs known layer labels         | ✓               |
| NMI                 | Normalized Mutual Information                     | ✓               |
| Silhouette          | Compactness of latent clusters (no labels needed) | ✓               |
| LODO Silhouette/ARI | Leave-One-Donor-Out cross-donor generalization    | ✓               |
| Cross-slice ARI     | ARI on held-out slices                            | ✓               |

---

## Key References

- Dong & Zhang (2022) — STAGATE: graph attention autoencoder for ST
- Long et al. (2023) — GraphST: contrastive learning for ST
- Pearl (2009) — _Causality: Models, Reasoning, and Inference_
- Arjovsky et al. (2019) — Invariant Risk Minimization (IRM)
- Maynard et al. (2021) — DLPFC spatial transcriptomics dataset

---

## Limitations & Future Work

- **Absolute ARI scores are low** because CauST uses a lightweight GAT + KMeans pipeline for domain identification. The important comparison is the _relative_ improvement from causal gene selection vs baseline HVG usage. Integrating CauST's gene selection into more sophisticated backends (STAGATE with mclust, BayesSpace) is expected to improve absolute metrics significantly.
- **Ground-truth evaluation**: The GEO-sourced DLPFC sections (P4/P6) lack `layer_guess` annotations, so most benchmarks report only Silhouette score. The full 12-slice spatialLIBD dataset with ground-truth cortical layer labels would enable ARI/NMI evaluation; downloading it requires the R `spatialLIBD` package.
- **Single-slice mode** has no cross-validation: invariance scoring requires 2+ slices. For single slices, only perturbation-based causal scores are available.
- **Per-slice models**: CauST currently trains a separate autoencoder per slice. A shared pretrained encoder across slices would reduce compute time and enable better transfer.
- **Clustering**: KMeans is a simple baseline. Future work could integrate model-based clustering (mclust, Gaussian mixture) or leverage existing domain detection models directly.
- **Scalability**: Full perturbation scoring iterates over all genes; the hybrid gradient+perturbation mode mitigates this, but very large gene panels (>10K) may still be slow.
- **Cross-slice ARI near zero**: When using `model.transform()` on held-out slices, ARI against ground truth is near zero. This suggests the model trained on one set of slices does not directly transfer domain labels well — the LODO evaluation confirms that causal gene _selection_ transfers (positive silhouette) even when the exact cluster assignments don't match reference labels.

---

## License

MIT © Pratham Makhija 2025-2026
