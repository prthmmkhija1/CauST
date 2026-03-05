# CauST: Complete End-to-End Project Guide

### Causal Gene Intervention for Robust Spatial Domain Identification

**Project Source:** [OSRE 2026 – UC Irvine](https://ucsc-ospo.github.io/project/osre26/uci/caust/)
**Author/Mentor:** Lijinghua Zhang, PhD, UC Irvine
**Effort:** ~350 hours | **Difficulty:** Advanced

---

> **How to read this guide:**
> Every section builds on the previous one. If a word seems confusing, there's a plain-English explanation nearby. You don't need to understand all the biology — focus on the steps.
>
> **Your two machines at a glance:**
> | Task | Use This Machine |
> |---|---|
> | Writing code, reading papers, small experiments | HP 15s (integrated graphics) |
> | Training models, running heavy experiments | HP Victus (GPU) |

---

## TABLE OF CONTENTS

1. [What Are We Actually Building? (Plain English)](#1-what-are-we-actually-building)
2. [Background Knowledge You Need First](#2-background-knowledge-you-need-first)
3. [Setting Up HP 15s (CPU-Only Machine)](#3-setting-up-hp-15s-cpu-only-machine)
4. [Setting Up HP Victus (GPU Machine)](#4-setting-up-hp-victus-gpu-machine)
5. [Understanding the Data](#5-understanding-the-data)
6. [Project Architecture Overview](#6-project-architecture-overview)
7. [Phase 1 – Causal Gene Effect Estimation](#7-phase-1--causal-gene-effect-estimation)
8. [Phase 2 – Invariant Effect Analysis](#8-phase-2--invariant-effect-analysis)
9. [Phase 3 – Causal Gene Filtering & Reweighting](#9-phase-3--causal-gene-filtering--reweighting)
10. [Phase 4 – Integration with Existing Methods](#10-phase-4--integration-with-existing-methods)
11. [Phase 5 – Evaluation & Benchmarking](#11-phase-5--evaluation--benchmarking)
12. [Phase 6 – Visualization Tools](#12-phase-6--visualization-tools)
13. [Phase 7 – Documentation & Tutorials](#13-phase-7--documentation--tutorials)
14. [Folder Structure of the Final Project](#14-folder-structure-of-the-final-project)
15. [Syncing Work Between HP 15s and HP Victus](#15-syncing-work-between-hp-15s-and-hp-victus)
16. [Common Errors and How to Fix Them](#16-common-errors-and-how-to-fix-them)
17. [Glossary (Plain English Definitions)](#17-glossary-plain-english-definitions)
18. [Resources & Papers to Read](#18-resources--papers-to-read)

---

## 1. What Are We Actually Building?

### The Biology (Simplified)

Picture a tiny slice of human brain tissue, thinner than a human hair, placed on a glass slide. A special machine scans this slide and, at every tiny dot/spot on the slide, records **which genes are "switched on" and how strongly**.

- There are roughly **3,000–30,000 spots** on one slide.
- Each spot has readings for **thousands of genes**.
- Scientists want to group these spots into **regions** — like "this group of spots is the outer brain layer", "this group is the inner layer". This grouping is called **Spatial Domain Identification**.

### The Problem with Current Tools

Current tools group spots by asking: _"Which genes tend to be high/low in the same spots?"_ — this is **correlation** (things that go together).

But here's the issue: just because two things go together doesn't mean one causes the other. Example: ice cream sales and drowning rates both go up in summer — they're correlated, but ice cream doesn't cause drowning (summer heat causes both).

Similarly, many genes that are "correlated" with brain layers are NOT the ones actually _forming_ those layers. Using them confuses the model. When you try the model on tissue from a _different person_ or _different slice_, it fails badly.

### What CauST Does Differently

CauST asks: _"If I artificially turned off gene X, does the region classification change?"_

This is **causal thinking** — we simulate "what would happen if…" experiments (called **interventions**) without doing actual lab experiments. If turning off gene X drastically changes how spots are grouped → Gene X is **causal**. If nothing changes → it's just a bystander.

CauST then:

1. Scores every gene by its causal importance
2. Keeps only the truly causal genes
3. Uses those genes to build a **more robust, generalizable** spatial domain model

### End Deliverable

A Python package called `caust` that:

- Takes spatial transcriptomics data as input
- Ranks/filters genes by causal importance
- Outputs better spatial domain groupings
- Works alongside popular existing tools (STAGATE, GraphST)
- Comes with visualizations and documentation

---

## 2. Background Knowledge You Need First

> Do this BEFORE writing any code. Budget: ~2–3 weeks of reading.
> Use HP 15s for this phase (just reading and note-taking).

### 2.1 Biology Concepts (1 week)

You don't need a biology degree, but you need to understand:

| Concept                       | What It Means in Plain English                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| **Gene**                      | A section of DNA that acts like an instruction for making a protein                  |
| **Gene Expression**           | How "active" a gene is — like the volume knob on a radio                             |
| **Spatial Transcriptomics**   | A technology that measures gene expression at specific locations on a tissue         |
| **Tissue Slice**              | A paper-thin cross-section of an organ used for scanning                             |
| **Spatial Domain**            | A biologically meaningful region/cluster in the tissue (e.g., brain layer)           |
| **scRNA-seq**                 | Single-cell RNA sequencing — measures gene expression per cell (older sibling of ST) |
| **ARI (Adjusted Rand Index)** | A score from 0 to 1 measuring how good your clustering is (1 = perfect)              |

**Where to learn:**

- Watch: "Spatial Transcriptomics Explained" on YouTube (10x Genomics channel)
- Read: [Spatially resolved transcriptomics — Nature Methods 2021](https://www.nature.com/articles/s41592-020-01042-x) (just the intro section)
- Read: [STAGATE paper abstract and intro](https://www.nature.com/articles/s41467-022-29439-6)

### 2.2 Machine Learning Concepts (1 week)

| Concept                        | What It Means in Plain English                                                   |
| ------------------------------ | -------------------------------------------------------------------------------- |
| **Clustering**                 | Automatically grouping similar things together (like sorting colored balls)      |
| **Graph Neural Network (GNN)** | A type of AI that works on networks/graphs (nodes connected by edges)            |
| **Autoencoder**                | An AI that learns to compress data and then reconstruct it — learns key features |
| **Latent Space**               | The compressed "summary" representation inside an autoencoder                    |
| **Loss Function**              | A measure of how wrong the AI is — training = minimizing this                    |
| **Batch Normalization**        | A technique to stabilize training                                                |
| **Dropout**                    | Randomly switching off neurons during training to prevent overfitting            |

**Where to learn:**

- [fast.ai Practical Deep Learning Course](https://course.fast.ai/) — Lesson 1–3 (free, very beginner friendly)
- [3Blue1Brown Neural Networks YouTube series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### 2.3 Causal Inference Concepts (1 week)

This is the heart of CauST. The key idea: **correlation ≠ causation**.

| Concept                               | What It Means in Plain English                                                    |
| ------------------------------------- | --------------------------------------------------------------------------------- |
| **Causal Inference**                  | The science of figuring out _cause and effect_ from data                          |
| **Intervention (do-calculus)**        | Simulating "what if I forcibly set X to a certain value?"                         |
| **Counterfactual**                    | "What _would have_ happened if something was different?"                          |
| **Confounder**                        | A hidden third factor that makes two things _look_ related even when they're not  |
| **In-silico Intervention**            | Doing an intervention _in software/simulation_ instead of a real lab              |
| **Invariant Risk Minimization (IRM)** | A technique to find features that work the same way across different environments |

**Where to learn:**

- Book (free PDF): [The Book of Why — Judea Pearl](http://bayes.cs.ucla.edu/WHY/) — Read Chapters 1–3
- Short video: "Causal Inference in Machine Learning" — Bernhard Schölkopf (YouTube)
- Paper: [IRM — Arjovsky et al. 2019](https://arxiv.org/abs/1907.02893) — just read the abstract + intro

### 2.4 Key Existing Tools to Understand

These are the "competitors" or "base tools" that CauST will augment:

| Tool           | What It Does                                       | Where to Find It                                                 |
| -------------- | -------------------------------------------------- | ---------------------------------------------------------------- |
| **STAGATE**    | Graph attention autoencoder for spatial domains    | [GitHub](https://github.com/zhanglabtools/STAGATE)               |
| **GraphST**    | Contrastive GNN for spatial transcriptomics        | [GitHub](https://github.com/JinmiaoChenLab/GraphST)              |
| **BayesSpace** | Statistical clustering using spatial neighborhoods | [Bioconductor](https://www.bioconductor.org/packages/BayesSpace) |
| **Scanpy**     | General single-cell data analysis toolkit          | [GitHub](https://github.com/scverse/scanpy)                      |
| **Squidpy**    | Spatial omics analysis on top of scanpy            | [GitHub](https://github.com/scverse/squidpy)                     |

**Action:** Install each one and run their tutorial notebooks to understand input/output format.

---

## 3. Setting Up HP 15s (CPU-Only Machine)

> Use this machine for: writing code, running small tests, reading, data preprocessing.

### 3.1 Install Python the Right Way

1. **Download Miniconda** (lighter than Anaconda):
   - Go to: https://docs.conda.io/en/latest/miniconda.html
   - Download the **Windows 64-bit installer**
   - Run it, check "Add to PATH" during install

2. **Verify installation** — open PowerShell and type:
   ```powershell
   conda --version
   python --version
   ```
   You should see version numbers.

### 3.2 Create a Project Environment

Think of an "environment" as a clean isolated room for your project — changes here don't affect other projects.

```powershell
# Create a new environment named "caust" with Python 3.10
conda create -n caust python=3.10 -y

# Activate it (you'll do this every time you work on CauST)
conda activate caust
```

You'll know it's active when you see `(caust)` at the start of your terminal line.

### 3.3 Install Core Libraries (CPU Version)

```powershell
# Make sure caust environment is active first
conda activate caust

# Install PyTorch (CPU-only version for HP 15s)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install spatial transcriptomics and data science libraries
pip install scanpy squidpy anndata numpy pandas scipy scikit-learn matplotlib seaborn

# Install graph neural network library
pip install torch-geometric

# Install additional tools
pip install jupyter notebook ipykernel tqdm pyyaml

# Add the environment to Jupyter
python -m ipykernel install --user --name caust --display-name "CauST (Python 3.10)"
```

### 3.4 Install VS Code (Recommended Editor)

1. Download from: https://code.visualstudio.com/
2. Install these extensions inside VS Code:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft)
   - **GitLens** (for Git history)
   - **Pylance** (for code hints)
3. Open your project folder: `File → Open Folder → F:\Projects\CauST`
4. Select the `caust` Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick `caust`

### 3.5 Set Up Git

Git lets you track every change you make (like "Save History" for code) and sync between machines.

```powershell
# Set your identity (one-time setup)
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Navigate to project folder
cd F:\Projects\CauST

# Initialize a git repository
git init

# Set up GitHub (create a free account at github.com first)
# Then create a new repository called "CauST" on GitHub
# Then connect your local folder to it:
git remote add origin https://github.com/YOUR_USERNAME/CauST.git
```

---

## 4. Setting Up HP Victus (GPU Machine)

> Use this machine for: training all neural network models, running benchmarks, heavy data processing.

### 4.1 Check Your GPU

Open PowerShell and run:

```powershell
nvidia-smi
```

This shows your GPU name and VRAM. Note the **CUDA version** shown (top right of output). You'll need this.

### 4.2 Same Miniconda + Environment Setup

Repeat Section 3.1 and 3.2 on the Victus exactly the same way.

### 4.3 Install GPU-Accelerated PyTorch

This is the key difference from HP 15s — you install a GPU version:

```powershell
conda activate caust

# IMPORTANT: Replace "cu121" below with your actual CUDA version
# If nvidia-smi shows CUDA 12.1 → use cu121
# If CUDA 11.8 → use cu118
# If CUDA 12.4 → use cu124
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is detected by PyTorch
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

You should see `True` and your GPU name. If you see `False`, the CUDA version is wrong — retry with a different version.

### 4.4 Install PyTorch Geometric (GPU Version)

```powershell
pip install torch-geometric

# These are extra dependencies needed for graph operations
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

> **Note:** The URL above must match your exact PyTorch + CUDA version. Check https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html for the exact correct URL.

### 4.5 Rest of Libraries (Same as 15s)

```powershell
pip install scanpy squidpy anndata numpy pandas scipy scikit-learn matplotlib seaborn
pip install jupyter notebook ipykernel tqdm pyyaml
python -m ipykernel install --user --name caust --display-name "CauST (Python 3.10)"
```

### 4.6 Clone the Repo on Victus

After you push code from HP 15s to GitHub:

```powershell
cd C:\Projects
git clone https://github.com/YOUR_USERNAME/CauST.git
cd CauST
conda activate caust
```

---

## 5. Understanding the Data

### 5.1 What Does Spatial Transcriptomics Data Look Like?

The data is stored in a format called **AnnData** (`.h5ad` files). Think of it as a giant Excel spreadsheet:

```
Rows    = Spots/Cells (one row per location on the tissue)
Columns = Genes       (one column per gene)
Values  = Expression level of that gene at that spot (a number, often 0)

+ Extra info:
  - X/Y coordinates of each spot on the tissue image
  - Known tissue layer labels (for validation)
```

In Python, it looks like:

```python
import scanpy as sc
adata = sc.read_h5ad("sample.h5ad")
# adata.X       → the big matrix (spots × genes)
# adata.obs     → info about each spot (coordinates, labels)
# adata.var     → info about each gene (gene names)
# adata.obsm    → spatial coordinates stored here
```

### 5.2 Download Public Datasets

These are the datasets you'll use throughout the project. They're free and public.

#### Dataset 1: DLPFC (Human Brain Dorsolateral Prefrontal Cortex)

- **Why:** Most commonly used benchmark for spatial domain identification
- **Contains:** 12 tissue slices from 3 donors (4 slices each), with 6–7 known cortical layers
- **Download from:** http://spatial.libd.org/spatialLIBD/
- Or via R: use the `spatialLIBD` package and export to `.h5ad`
- **Size:** ~500MB total

#### Dataset 2: 10x Visium Mouse Brain

- **Why:** Widely used, well-annotated, good for cross-slice testing
- **Download from:** https://www.10xgenomics.com/resources/datasets (search "Visium Mouse Brain")
- **Size:** ~2GB

#### Dataset 3: Mouse Olfactory Bulb (MOB)

- **Why:** Stereo-seq dataset with clear 6-layer structure, ideal for validating spatial methods
- **Download:** Auto-downloaded via `scripts/01_download_data.py`
- **Size:** ~100MB

#### Dataset 4: Human Breast Cancer (10x Visium)

- **Why:** Invasive ductal carcinoma with complex spatial organization, ~20 distinct regions
- **Download from:** https://www.10xgenomics.com/datasets
- **Size:** ~1GB

#### Dataset 5: STARmap (Mouse Visual Cortex)

- **Why:** In-situ sequencing at subcellular resolution, ~1000 genes; tests CauST on non-Visium data
- **Download:** Auto-downloaded via `scripts/01_download_data.py`
- **Paper:** Wang et al. (Science, 2018)

**Save all datasets to:** `F:\Projects\CauST\data\raw\`

### 5.3 Basic Data Preprocessing Steps

Before training any model, data must be cleaned. Run these steps in a Jupyter notebook:

```python
import scanpy as sc

# Load data
adata = sc.read_h5ad("data/raw/DLPFC_sample1.h5ad")

# Step 1: Filter out spots with too few genes (empty/dead spots)
sc.pp.filter_cells(adata, min_genes=200)

# Step 2: Filter out genes appearing in very few spots
sc.pp.filter_genes(adata, min_cells=3)

# Step 3: Normalize — make every spot comparable (all sum to 10,000)
sc.pp.normalize_total(adata, target_sum=1e4)

# Step 4: Log transform — compress large differences
sc.pp.log1p(adata)

# Step 5: Select Highly Variable Genes (top 3000 most informative)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]

# Step 6: Scale the data
sc.pp.scale(adata, max_value=10)

# Save preprocessed data
adata.write("data/processed/DLPFC_sample1_preprocessed.h5ad")
```

---

## 6. Project Architecture Overview

Think of CauST as a pipeline with 4 stages:

```
Raw Spatial Data (spots × genes)
        ↓
[Stage 1: Spatial Graph Construction]
    → Connect nearby spots as a graph (like connecting neighbors on a map)
        ↓
[Stage 2: Causal Gene Scorer]  ← This is the main innovation of CauST
    → For each gene: simulate "what if this gene was turned off?"
    → Measure change in domain assignments
    → Assign a causal score to each gene
        ↓
[Stage 3: Gene Filtering / Reweighting]
    → Keep only high-causal-score genes
    → OR give high-causal genes more weight in training
        ↓
[Stage 4: Spatial Domain Identification]
    → Feed filtered/reweighted genes into STAGATE or GraphST
    → Output: domain label for each spot
        ↓
Domain assignments + Visualization + Evaluation (ARI score)
```

### File Structure of Your Code

```
CauST/
├── caust/                        ← main Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py             ← load and preprocess AnnData
│   │   └── graph.py              ← build spatial neighbor graphs
│   ├── causal/
│   │   ├── __init__.py
│   │   ├── intervention.py       ← simulate gene knockouts
│   │   ├── scorer.py             ← assign causal scores
│   │   └── invariance.py         ← cross-slice stability analysis
│   ├── filter/
│   │   ├── __init__.py
│   │   └── gene_filter.py        ← filter/reweight genes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── autoencoder.py        ← basic spatial autoencoder
│   │   └── stagate_wrapper.py    ← wrapper around STAGATE
│   ├── evaluate/
│   │   ├── __init__.py
│   │   └── metrics.py            ← ARI, NMI, cross-slice metrics
│   └── visualize/
│       ├── __init__.py
│       └── plots.py              ← all visualization functions
├── data/
│   ├── raw/                      ← original downloaded datasets
│   └── processed/                ← preprocessed .h5ad files
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_causal_scoring.ipynb
│   ├── 04_filtering.ipynb
│   ├── 05_integration.ipynb
│   └── 06_benchmarking.ipynb
├── experiments/
│   ├── configs/                  ← YAML config files for experiments
│   └── results/                  ← saved outputs, metrics, figures
├── tests/                        ← unit tests for each module
├── docs/                         ← documentation
├── requirements.txt
├── setup.py
└── README.md
```

---

## 7. Phase 1 – Causal Gene Effect Estimation

> **Goal:** For each gene, measure how much it causally influences spatial domain assignments.
> **Machine:** HP Victus (GPU needed for training)
> **Time estimate:** ~6–8 weeks

### 7.1 Understand What "In-Silico Intervention" Means

A normal experiment: scientists use CRISPR to turn off a gene in real cells and see what happens. This takes months and costs thousands of dollars.

An **in-silico intervention** (CauST's approach): we do it in software:

1. Train a model on normal data
2. For each gene G, create a "perturbed" copy of the data where gene G is set to 0 (knocked out)
3. Run the model on this perturbed data
4. Compare: "How different are the domain assignments after knocking out gene G?"
5. Big difference = Gene G is causal. Small difference = Gene G is a bystander.

### 7.2 Step-by-Step: Build the Spatial Autoencoder First

Before scoring genes, you need a base model that takes gene expression → domain assignments. We'll use a simplified version of STAGATE (a Graph Attention Autoencoder).

**What is a Graph Attention Autoencoder?**

Imagine each spot on the tissue is a person in a crowd. A "graph" connects each person to their nearest neighbors. The autoencoder learns to:

1. Look at a spot and its neighbors (using "attention" = deciding who to pay attention to more)
2. Compress all that information into a small "summary" (encoder)
3. Reconstruct the original data from that summary (decoder)
4. The summary (latent vector) is then used for clustering

**Notebook: `notebooks/02_graph_construction.ipynb`**

```python
# Build a spatial neighbor graph
# Each spot connects to its K nearest spatial neighbors (K=6 typically)

import squidpy as sq
import scanpy as sc

adata = sc.read_h5ad("data/processed/DLPFC_sample1_preprocessed.h5ad")

# Build spatial neighbor graph (connects spots that are physically close)
sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6)

# This creates:
# adata.obsp["spatial_connectivities"]  → adjacency matrix (who is connected to who)
# adata.obsp["spatial_distances"]       → how far apart connected spots are
```

**Notebook: `notebooks/03_causal_scoring.ipynb` — Train base autoencoder**

The model file will be in `caust/models/autoencoder.py`. Conceptually:

```
Input: expression values of a spot + its neighbors' expressions
  ↓ (Graph Attention Layer)
Hidden representation (768 dimensions → 128 dimensions)
  ↓ (Encoder)
Latent vector (30 dimensions) ← this is the "summary"
  ↓ (Decoder)
Reconstructed expression values
```

Training loop (conceptually):

```
For each epoch:
  1. Pass all spots through the network
  2. Compute "reconstruction loss" = how different is output from input?
  3. Compute gradients (which direction to adjust weights?)
  4. Update weights slightly in that direction
  5. Repeat until loss stops decreasing
```

### 7.3 Step-by-Step: Gene Intervention & Causal Scoring

Once the autoencoder is trained, run interventions:

**Algorithm (in plain English):**

```
For each gene G (out of ~3000 genes):
    1. Take the original data matrix (spots × genes)
    2. Create a copy where column G is set to 0 (gene knockout)
    3. Pass both original and knockout data through the trained model
    4. Get domain assignments (via clustering the latent space) for both
    5. Compute ARI between original assignments and knockout assignments
    6. causal_score[G] = 1 - ARI   (higher = more change = more causal)
```

**Important:** This loop runs 3000 times (once per gene). On GPU, each pass takes ~0.1 seconds. Total: ~5 minutes. On CPU (HP 15s), it would take hours — always run this on HP Victus.

**File: `caust/causal/intervention.py`**

The key function signature you'll implement:

```python
def compute_causal_scores(adata, model, n_genes=3000) -> dict:
    """
    Returns a dictionary: {gene_name: causal_score}
    Higher score = gene is more causally important
    """
```

### 7.4 Controlling for Confounders

A confounding gene is one that _looks_ causal but isn't. For example, a "housekeeping gene" (active in all cell types) might show up as important just because removing it disrupts everything globally, not specifically the spatial domains.

**Solution:** When knocking out gene G, also record how much global gene expression changes. If a gene causes total havoc, normalize its causal score by that global disruption:

```
adjusted_causal_score[G] = change_in_domains[G] / global_disruption[G]
```

This gives a fairer score that highlights genes specifically important for _spatial organization_, not just overall cell health.

---

## 8. Phase 2 – Invariant Effect Analysis

> **Goal:** Find genes whose causal effects are consistent ACROSS multiple slices/donors.
> **Key idea:** A truly causal gene should have the same effect on Slice 1 from Person A as on Slice 2 from Person B.
> **Machine:** HP Victus
> **Time estimate:** ~3–4 weeks

### 8.1 Why Invariance Matters

Imagine a gene that is causal in Person A's brain slice but not in Person B's. That gene might just be a quirk of Person A's biology — it's not a universal driver of spatial organization. We want genes that are _universally_ causal.

### 8.2 The Invariance Score

For each gene G, you have causal scores across multiple slices:

```
Slice 1 (Donor 1): causal_score = 0.81
Slice 2 (Donor 1): causal_score = 0.79
Slice 3 (Donor 2): causal_score = 0.83
Slice 4 (Donor 2): causal_score = 0.15  ← inconsistent!
```

Gene G has an **inconsistent** effect. We can measure this with **variance** (how spread out the scores are). Low variance across slices = high invariance = more likely truly causal.

**Invariance Score formula:**

```
invariance_score[G] = mean(causal_scores across slices) / (1 + variance(causal_scores))
```

High mean + low variance = high invariance score = keep this gene.

### 8.3 Cross-Donor Analysis

The DLPFC dataset has 3 donors with 4 slices each. This is perfect for testing invariance.

**Algorithm:**

```
For each donor pair (e.g., Donor 1 vs Donor 2):
    1. Compute causal scores on all Donor 1 slices
    2. Compute causal scores on all Donor 2 slices
    3. Compute correlation between the two lists of scores
    4. High correlation → genes have similar causal roles across donors
    5. Genes with high cross-donor correlation get higher invariance bonus
```

**File: `caust/causal/invariance.py`**

Key function:

```python
def compute_invariance_scores(causal_scores_per_slice: dict) -> dict:
    """
    Input: {slice_id: {gene: causal_score}}
    Output: {gene: invariance_score}
    """
```

### 8.4 Combined Score

Combine causal score and invariance score:

```
final_score[G] = alpha * mean_causal_score[G] + (1 - alpha) * invariance_score[G]
```

Where `alpha` is a tunable parameter (start with 0.5, i.e., equal weight to both).

---

## 9. Phase 3 – Causal Gene Filtering & Reweighting

> **Goal:** Use the causal scores to select only the best genes for model training.
> **Machine:** HP 15s works for this phase (mostly logic/code, no heavy compute)
> **Time estimate:** ~2–3 weeks

### 9.1 Two Strategies

**Strategy A — Hard Filtering (Easier to implement first)**

Simply keep the top K genes by final_score:

```
Top 500 genes → use these, discard the rest
```

```python
def filter_genes_top_k(adata, gene_scores, k=500):
    top_genes = sorted(gene_scores, key=gene_scores.get, reverse=True)[:k]
    return adata[:, top_genes]
```

Advantages: Simple, interpretable, removes noise clearly.
Disadvantages: Hard threshold is arbitrary — setting K=500 vs K=600 might give different results.

**Strategy B — Soft Reweighting (More principled)**

Instead of removing genes, multiply their expression values by their causal score:

```
gene_matrix_reweighted[spot, gene] = gene_matrix[spot, gene] × final_score[gene]
```

This way, all genes are kept but causal genes are amplified and bystander genes are dimmed.

```python
def reweight_genes(adata, gene_scores):
    import numpy as np
    scores = np.array([gene_scores.get(g, 0.0) for g in adata.var_names])
    adata.X = adata.X * scores[np.newaxis, :]  # multiply each column by its score
    return adata
```

### 9.2 Ablation Study

An ablation study means: _"What happens if I remove this component?"_ — it proves your component is doing something useful.

You'll need to compare:
| Setting | Description |
|---|---|
| Baseline | Use all highly variable genes (standard approach) |
| CauST-Filter | Use only top-K causal genes |
| CauST-Reweight | Use all genes but reweighted by causal scores |
| CauST-Full | Filter + reweight + invariance scoring |

All settings are then evaluated with ARI on DLPFC data. CauST-Full should win.

---

## 10. Phase 4 – Integration with Existing Methods

> **Goal:** Make CauST a "plug-in" that works BEFORE other tools, improving them.
> **Machine:** HP Victus
> **Time estimate:** ~3–4 weeks

### 10.1 The Design Principle

CauST should behave like a preprocessing filter:

```
Your data → [CauST] → filtered/reweighted data → [STAGATE or GraphST] → domain labels
```

Neither STAGATE nor GraphST needs to be modified. CauST just hands them better input.

### 10.2 Integrating with STAGATE

STAGATE is already installable via pip:

```powershell
pip install STAGATE-pyG
```

Usage with CauST preprocessing:

```python
import STAGATE_pyG as STAGATE
from caust import CauST

# Apply CauST
pipeline = CauST(n_causal_genes=500, alpha=0.5)
adata_filtered = pipeline.fit_transform(adata)

# Now pass to STAGATE
STAGATE.run_STAGATE(adata_filtered)
```

### 10.3 Integrating with GraphST

Same idea:

```python
from GraphST import GraphST
from caust import CauST

pipeline = CauST(n_causal_genes=500)
adata_filtered = pipeline.fit_transform(adata)

model = GraphST.GraphST(adata_filtered, device='cuda')
adata_filtered = model.train()
```

### 10.4 Building the CauST API

The main class should be easy to use. Design the API to be sklearn-like (familiar to most Python data scientists):

```python
class CauST:
    def __init__(self, n_causal_genes=500, alpha=0.5, device='cpu'):
        ...

    def fit(self, adata):
        # Learn causal scores from data
        ...

    def transform(self, adata):
        # Apply filtering/reweighting
        ...

    def fit_transform(self, adata):
        return self.fit(adata).transform(adata)

    def get_causal_scores(self):
        # Return the gene scores for inspection
        ...
```

---

## 11. Phase 5 – Evaluation & Benchmarking

> **Goal:** Prove CauST works better than baselines, with numbers.
> **Machine:** HP Victus for running, HP 15s for writing up results
> **Time estimate:** ~3–4 weeks

### 11.1 Evaluation Metrics

**Metric 1: ARI (Adjusted Rand Index)**

- Measures how well your domain labels match the known ground truth labels
- Score: 0 = random, 1 = perfect
- How: compare your predicted labels vs known layer labels in DLPFC

**Metric 2: NMI (Normalized Mutual Information)**

- Similar to ARI but measures shared information between two clusterings
- Also 0 to 1, higher is better

**Metric 3: Cross-Slice Generalization (The Key CauST Metric)**

- Train a model on Slice 1, test it on Slice 4 (different donor)
- Measure ARI on the test slice
- CauST should improve this significantly vs baseline

**Metric 4: Silhouette Score (No Ground Truth Needed)**

- Measures how well-separated the clusters are in latent space
- Useful for datasets with no known labels

### 11.2 Benchmark Experimental Setup

Run ALL combinations:

```
Methods:         [No Filtering, CauST-Filter, CauST-Reweight, CauST-Full]
Base Models:     [STAGATE, GraphST]
Datasets:        [DLPFC, Mouse Brain, Mouse OB, Human Breast Cancer, STARmap]
Evaluations:     [Same-slice ARI, Cross-slice ARI, NMI, Silhouette]
```

Total experiments: 4 × 2 × 5 × 4 = 160 numbers in your results table.

### 11.3 Cross-Slice Generalization Protocol

This is the most important benchmark. For DLPFC (3 donors, 4 slices each):

```
Leave-One-Out setup:
  → Train on slices from Donor 1 + Donor 2
  → Test on Donor 3's slices
  → Repeat 3 times, rotating which donor is held out
  → Average the ARI scores
```

Get help from this experimental design concept: it's called **Leave-One-Donor-Out (LODO)** validation.

### 11.4 Saving and Organizing Results

Create YAML config files for each experiment in `experiments/configs/`:

```yaml
# experiments/configs/dlpfc_caust_stagate.yaml
dataset: DLPFC
slices: [1, 2, 3, 4, 5, 6]
method: CauST-Filter
base_model: STAGATE
n_causal_genes: 500
alpha: 0.5
device: cuda
seed: 42
```

Save results to CSV: `experiments/results/all_results.csv`

---

## 12. Phase 6 – Visualization Tools

> **Goal:** Create clear, informative plots that show what CauST is doing.
> **Machine:** HP 15s fine for this
> **Time estimate:** ~1–2 weeks

### 12.1 Plot 1: Spatial Domain Map

Show the tissue slice colored by domain assignment.

```python
import matplotlib.pyplot as plt
import scanpy as sc

def plot_spatial_domains(adata, label_key='caust_domain', title='CauST Domains'):
    sc.pl.spatial(adata, color=label_key, title=title, spot_size=150)
```

### 12.2 Plot 2: Causal Gene Score Bar Chart

Show top 20 causal genes ranked by score.

```python
def plot_causal_scores(gene_scores, top_k=20):
    top_genes = sorted(gene_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    genes, scores = zip(*top_genes)
    plt.barh(genes[::-1], scores[::-1], color='steelblue')
    plt.xlabel("Causal Score")
    plt.title(f"Top {top_k} Causal Genes")
    plt.tight_layout()
    plt.show()
```

### 12.3 Plot 3: Cross-Slice Heatmap of Causal Scores

For the top 50 genes, show a heatmap of their causal scores across all 12 DLPFC slices. Genes that are consistently high are truly invariant causal genes.

```python
import seaborn as sns

def plot_invariance_heatmap(causal_scores_per_slice, top_k=50):
    # Build DataFrame: rows=genes, columns=slices
    # Use seaborn's clustermap to also cluster similar genes together
    sns.clustermap(df, cmap='viridis', figsize=(15, 10))
```

### 12.4 Plot 4: Intervention Effect Plot

For a specific gene, show the tissue map before and after knocking it out:

```python
def plot_intervention_effect(adata, gene_name, model):
    # Left: normal domain assignments
    # Right: domain assignments after gene knockout
    # Shows visually why that gene is "causal"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ...
```

### 12.5 Plot 5: ARI Comparison Bar Chart

Final benchmark figure showing all methods side by side:

```python
def plot_benchmark_results(results_df):
    # Grouped bar chart: x=method, bars grouped by dataset
    # Separate panels for same-slice vs cross-slice ARI
```

---

## 13. Phase 7 – Documentation & Tutorials

> **Goal:** Make it easy for ANY researcher to use CauST.
> **Machine:** HP 15s
> **Time estimate:** ~2 weeks

### 13.1 README.md

The first thing people see on GitHub. Include:

- What CauST is (2–3 sentences)
- Installation instructions
- Quick start example (5–10 lines of code)
- Links to full documentation
- Citation information

### 13.2 Tutorial Notebooks

Create these notebooks in `notebooks/`:

| Notebook                           | What It Shows                                       |
| ---------------------------------- | --------------------------------------------------- |
| `01_quickstart.ipynb`              | Load data, run CauST, see domain maps in 10 minutes |
| `02_causal_gene_exploration.ipynb` | How to inspect and interpret causal scores          |
| `03_integration_STAGATE.ipynb`     | Using CauST with STAGATE step by step               |
| `04_cross_slice_evaluation.ipynb`  | How to validate on multiple slices                  |
| `05_custom_datasets.ipynb`         | How to use CauST on your own data                   |

### 13.3 API Documentation

Use `sphinx` + `readthedocs` to auto-generate documentation from your code's docstrings:

```powershell
pip install sphinx sphinx-autodoc sphinx-rtd-theme

# In your project root:
sphinx-quickstart docs/
```

### 13.4 requirements.txt

List all dependencies so anyone can install everything at once:

```
torch>=2.0.0
torch-geometric>=2.3.0
scanpy>=1.9.0
squidpy>=1.2.0
anndata>=0.9.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
```

### 13.5 setup.py

So people can install CauST with `pip install -e .`:

```python
from setuptools import setup, find_packages

setup(
    name="caust",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[...],  # list from requirements.txt
    author="Your Name",
    description="Causal Gene Intervention for Robust Spatial Domain Identification",
    python_requires=">=3.9",
)
```

---

## 14. Folder Structure of the Final Project

This is what your complete project should look like when done:

```
F:\Projects\CauST\
│
├── caust/                          ← installable Python package
│   ├── __init__.py                 ← exposes main CauST class
│   ├── data/
│   │   ├── loader.py               ← load .h5ad files, handle preprocessing
│   │   └── graph.py                ← build spatial KNN graphs
│   ├── causal/
│   │   ├── intervention.py         ← gene knockout simulation
│   │   ├── scorer.py               ← compute causal + invariance scores
│   │   └── invariance.py           ← cross-slice stability analysis
│   ├── filter/
│   │   └── gene_filter.py          ← hard filter + soft reweighting
│   ├── models/
│   │   ├── autoencoder.py          ← graph attention autoencoder
│   │   └── stagate_wrapper.py      ← thin wrapper for STAGATE
│   ├── evaluate/
│   │   └── metrics.py              ← ARI, NMI, cross-slice evaluation
│   └── visualize/
│       └── plots.py                ← all 5 visualization functions
│
├── data/
│   ├── raw/                        ← original downloaded files (don't touch)
│   │   ├── DLPFC/
│   │   └── MouseBrain/
│   └── processed/                  ← cleaned .h5ad files
│
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_causal_gene_exploration.ipynb
│   ├── 03_integration_STAGATE.ipynb
│   ├── 04_cross_slice_evaluation.ipynb
│   └── 05_custom_datasets.ipynb
│
├── experiments/
│   ├── configs/                    ← one .yaml per experiment
│   └── results/                    ← CSVs of metrics, saved figures
│
├── tests/
│   ├── test_loader.py
│   ├── test_intervention.py
│   ├── test_scorer.py
│   └── test_metrics.py
│
├── docs/                           ← sphinx documentation
│
├── CauST_ProjectRoadmap.md         ← this file
├── README.md
├── requirements.txt
├── setup.py
└── .gitignore
```

---

## 15. Syncing Work Between HP 15s and HP Victus

### The Golden Rule

> **Never copy files via USB or manually. Always use Git + GitHub.**

### Daily Workflow

**On HP 15s (writing code, notebooks, configs):**

```powershell
cd F:\Projects\CauST
conda activate caust

# ... do your work ...

git add .
git commit -m "Add intervention module scaffold"
git push origin main
```

**On HP Victus (running training/experiments):**

```powershell
cd C:\Projects\CauST
conda activate caust

# Pull latest code from GitHub
git pull origin main

# Run the experiment
python experiments/run_experiment.py --config experiments/configs/dlpfc_caust_stagate.yaml

# Save results and push back
git add experiments/results/
git commit -m "Results: DLPFC STAGATE + CauST ARI=0.62"
git push origin main
```

### What to Put in .gitignore

You don't want to upload large data files to GitHub:

```gitignore
# Data files (too large for GitHub)
data/raw/
data/processed/
*.h5ad
*.h5

# Python artifacts
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/

# Environment files
.env

# Experiment outputs (optional — you may want to keep small CSVs)
experiments/results/figures/
```

### For Large Files (Models, Processed Data)

Use one of these:

- **Google Drive** — simple, free up to 15GB
- **OneDrive** — already on Windows, 5GB free
- **Git LFS** (Git Large File Storage) — for files under 1GB
- **Hugging Face Datasets Hub** — specifically for ML datasets, highly recommended

---

## 16. Common Errors and How to Fix Them

### "CUDA out of memory"

**Meaning:** Your GPU doesn't have enough RAM for the batch size you set.
**Fix:** Reduce batch size. In your config, try `batch_size: 64` instead of `batch_size: 256`.

### "ModuleNotFoundError: No module named 'caust'"

**Meaning:** Python can't find your package.
**Fix:**

```powershell
cd F:\Projects\CauST
pip install -e .
```

### "RuntimeError: Expected all tensors to be on the same device"

**Meaning:** Some tensors are on CPU, some on GPU.
**Fix:** Make sure you call `.to(device)` on every tensor and every model layer consistently.

### "ARI score is very low (below 0.2)"

**Possible causes:**

1. Wrong number of clusters (set n_clusters to actual number of layers, e.g., 7 for DLPFC)
2. Data was not preprocessed properly
3. Graph not built correctly (check spatial coordinates are loaded)

### "conda activate caust" doesn't work in PowerShell

**Fix:**

```powershell
conda init powershell
# Close and reopen PowerShell, then try again
```

### "torch_geometric install fails"

**Fix:** Use the exact install command from https://pytorch-geometric.readthedocs.io matching your Python + CUDA + PyTorch exact versions.

---

## 17. Glossary (Plain English Definitions)

| Term                        | Plain English Meaning                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| **AnnData**                 | The standard data format for single-cell/spatial data. Like a spreadsheet with extra metadata.   |
| **ARI**                     | Adjusted Rand Index — a score measuring how similar two clusterings are. 1 = perfect match.      |
| **Autoencoder**             | Neural net that compresses data then reconstructs it. The compressed part captures key features. |
| **Causal Inference**        | The science of figuring out what _causes_ what, not just what correlates with what.              |
| **Clustering**              | Automatically grouping similar data points. Like sorting photos by location without labels.      |
| **Confounding**             | A hidden third factor that makes two unrelated things look related.                              |
| **Counterfactual**          | "What would have happened IF something was different?" — imaginary but calculated.               |
| **DLPFC**                   | Dorsolateral Prefrontal Cortex — a part of the human brain. A very popular benchmark dataset.    |
| **GNN**                     | Graph Neural Network — AI that works on network/graph data.                                      |
| **Gene Expression**         | How active a gene is — how much "message" it's sending to make proteins.                         |
| **Graph Attention**         | A GNN that learns to pay more attention to some neighbors than others.                           |
| **Highly Variable Genes**   | Genes that differ a lot between cells — these carry the most useful signal.                      |
| **In-silico**               | "In software" — doing experiments computationally instead of in a real lab.                      |
| **Intervention**            | Forcibly changing a variable to see what happens (vs just observing).                            |
| **IRM**                     | Invariant Risk Minimization — a technique to find features that work across environments.        |
| **Latent Space**            | The compressed representation inside a neural network. A "summary" of the data.                  |
| **Normalization**           | Scaling data so different samples are comparable.                                                |
| **Spatial Domain**          | A biologically meaningful region on a tissue slice, identified by gene expression patterns.      |
| **Spatial Transcriptomics** | Technology that measures gene activity at specific physical locations on a tissue.               |
| **STAGATE**                 | A popular existing tool for spatial domain identification using graph attention networks.        |
| **scRNA-seq**               | Single-cell RNA sequencing — measures gene expression per cell (the non-spatial predecessor).    |

---

## 18. Resources & Papers to Read

### Papers (Read in This Order)

1. **Spatial Transcriptomics Overview**
   - "Spatially resolved transcriptomics" — Ståhl et al., _Science_ 2016 (the original paper)
   - "Museum of Spatial Transcriptomics" — Moses & Pachter, _Nature Methods_ 2022

2. **Spatial Domain Identification Methods (the tools you'll improve)**
   - STAGATE: "Deciphering spatial domains from spatially resolved transcriptomics" — Dong & Zhang, _Nature Communications_ 2022
   - GraphST: "Spatially informed clustering, integration, and deconvolution" — Long et al., _Nature Methods_ 2023

3. **Causal Inference**
   - IRM: "Invariant Risk Minimization" — Arjovsky et al. _arXiv_ 2019
   - "Causal Representation Learning" — Schölkopf et al. _PNAS_ 2021
   - Chapter 1–4 of "The Book of Why" — Judea Pearl (book, not a paper)

4. **Gene Perturbation in ML Context**
   - GEARS: "Predicting transcriptional outcomes of novel multi-gene perturbations" — Roohani et al., _Nature Biotechnology_ 2023

### Tools & Documentation

- Scanpy docs: https://scanpy.readthedocs.io
- Squidpy docs: https://squidpy.readthedocs.io
- PyTorch Geometric docs: https://pytorch-geometric.readthedocs.io
- STAGATE GitHub: https://github.com/zhanglabtools/STAGATE
- GraphST GitHub: https://github.com/JinmiaoChenLab/GraphST

### Free Courses

- fast.ai Practical Deep Learning: https://course.fast.ai (do Lessons 1–5)
- Stanford CS224W (Graph Neural Networks): https://web.stanford.edu/class/cs224w/ (YouTube)
- Brady Neal's Causal Inference course: https://www.bradyneal.com/causal-inference-course

---

## Quick-Reference Timeline

| Month   | Focus                                                           | Machine   |
| ------- | --------------------------------------------------------------- | --------- |
| Month 1 | Background reading, environment setup, data download            | HP 15s    |
| Month 2 | Spatial graph construction, base autoencoder implementation     | Both      |
| Month 3 | In-silico intervention + causal scoring                         | HP Victus |
| Month 4 | Invariance scoring across slices                                | HP Victus |
| Month 5 | Filtering/reweighting module + integration with STAGATE/GraphST | Both      |
| Month 6 | Full benchmarking experiments                                   | HP Victus |
| Month 7 | Visualization tools + documentation + tutorials                 | HP 15s    |
| Month 8 | Writing up, cleaning code, final testing, GitHub release        | HP 15s    |

---

_This document will evolve as the project progresses. Update it as you learn new things, hit blockers, or change plans._
