# CauST — Step-by-Step Completion Commands (Linux JupyterLab Terminal)

Run these commands **in order** in your JupyterLab terminal.
Each section is self-contained — if a step succeeds, move to the next.

---

## STEP 0: Setup & Verify Environment

```bash
# Go to project root
cd /CauST

# Make sure CauST is installed in editable mode
pip install -e . --quiet

# Fix NumPy version conflict: PyTorch compiled with NumPy 1.x crashes if NumPy 2.x is installed.
# Must run this BEFORE any import torch check.
pip install "numpy>=1.23,<2.0"

# Verify PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Verify torch-geometric
python -c "import torch_geometric; print(f'PyG {torch_geometric.__version__}')"

# Verify CauST is importable
python -c "from caust import CauST; print('CauST OK')"
```

---

## STEP 1: Download the 12-Slice spatialLIBD DLPFC Data (with ground-truth labels)

Your existing scripts try URLs that may be dead. This method uses a **direct Python
downloader** that pulls from the spatialLIBD data hosted by the Lieber Institute.

### Method A: Direct Python download (try this first)

```bash
python -c "
import scanpy as sc
import os, sys

out_dir = 'data/raw/DLPFC'
os.makedirs(out_dir, exist_ok=True)

samples = [
    '151507', '151508', '151509', '151510',
    '151669', '151670', '151671', '151672',
    '151673', '151674', '151675', '151676',
]

for sid in samples:
    out_path = f'{out_dir}/{sid}.h5ad'
    if os.path.exists(out_path):
        print(f'[skip] {sid}.h5ad already exists')
        continue
    url = f'https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5ad/{sid}_filtered_feature_bc_matrix.h5ad'
    print(f'Downloading {sid} from S3...')
    try:
        import urllib.request
        urllib.request.urlretrieve(url, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f'  OK: {size_mb:.1f} MB')
    except Exception as e:
        print(f'  S3 FAILED: {e}')
        # Try alternative simple name
        url2 = f'https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5ad/{sid}.h5ad'
        print(f'  Trying: {url2}')
        try:
            urllib.request.urlretrieve(url2, out_path)
            size_mb = os.path.getsize(out_path) / 1e6
            print(f'  OK: {size_mb:.1f} MB')
        except Exception as e2:
            print(f'  Also FAILED: {e2}')
            if os.path.exists(out_path):
                os.remove(out_path)
"
```

### Method B: If Method A fails — use the R spatialLIBD package

```bash
# First check if R is already installed in the container (no sudo needed to check)
which R && R --version || echo "R not found — skip to Method B-Python below"
```

**If R is found**, install packages using pre-built binaries from Posit Package Manager (no compilation needed):

```bash
# Export distro so R can read it as an env var (NOT passed as R argument)
export DISTRO=$(. /etc/os-release 2>/dev/null && echo "${VERSION_CODENAME:-noble}")
echo "Distro: $DISTRO"

# Detect distro and show R version (R 4.3.3 requires a dated PPM snapshot)
export DISTRO=$(. /etc/os-release 2>/dev/null && echo "${VERSION_CODENAME:-noble}")
echo "Distro: $DISTRO   R: $(R --version | head -1)"

# STEP 1: Install CRAN deps as pre-built binaries using a DATED PPM snapshot.
# Root cause: PPM noble/latest (2026) serves ggrepel 0.9.7 (needs R>=4.5) and
# Matrix 1.7-4 (needs R>=4.6). R 4.3.3 is too old for those.
# Fix: pin to 2024-06-01 — last snapshot where ggrepel 0.9.5 + Matrix 1.6-5
# were the latest, both compatible with R 4.3.x. Still binary (no compilation).
R --no-save << 'EOF'
lib <- path.expand('~/R/library')
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
.libPaths(lib)
distro <- Sys.getenv('DISTRO', unset='noble')
options(repos = c(
    PPM  = paste0('https://packagemanager.posit.co/cran/__linux__/', distro, '/2024-06-01'),
    CRAN = 'https://cloud.r-project.org'
))
pkgs <- c('ggrepel', 'Matrix', 'BiocManager', 'ggplot2', 'dplyr', 'tibble', 'scales')
install.packages(pkgs, lib=lib, dependencies=TRUE)
cat('ggrepel OK:', 'ggrepel' %in% rownames(installed.packages(lib.loc=lib)), '\n')
cat('Matrix  OK:', 'Matrix'  %in% rownames(installed.packages(lib.loc=lib)), '\n')
EOF

# STEP 2: Now install Bioconductor packages — ggrepel/Matrix already satisfied
R --no-save << 'EOF'
lib <- path.expand('~/R/library')
.libPaths(lib)
BiocManager::install(
    c('scater', 'SpatialExperiment', 'spatialLIBD'),
    ask=FALSE, update=FALSE, force=TRUE, lib=lib
)
EOF
```

Once spatialLIBD is installed, fetch and export the data:

````bash
R -e "
lib <- path.expand('~/R/library')
.libPaths(lib)
library(spatialLIBD)
library(SpatialExperiment)

spe <- fetch_data(type = 'spe')
sample_ids <- unique(colData(spe)\$sample_id)
cat('Samples found:', paste(sample_ids, collapse=', '), '\n')

for (sid in sample_ids) {
    idx <- colData(spe)\$sample_id == sid
    sub <- spe[, idx]
    Matrix::writeMM(counts(sub), paste0('data/raw/DLPFC/', sid, '_counts.mtx'))
    write.csv(as.data.frame(colData(sub)), paste0('data/raw/DLPFC/', sid, '_metadata.csv'))
    write.csv(rownames(sub), paste0('data/raw/DLPFC/', sid, '_genes.csv'))
    write.csv(spatialCoords(sub), paste0('data/raw/DLPFC/', sid, '_coords.csv'))
    cat('Exported:', sid, '- spots:', ncol(sub), '\n')
}
"

```bash
python -c "
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.io
import anndata as ad
import os

raw_dir = 'data/raw/DLPFC'
samples = [
    '151507', '151508', '151509', '151510',
    '151669', '151670', '151671', '151672',
    '151673', '151674', '151675', '151676',
]

for sid in samples:
    out_path = f'{raw_dir}/{sid}.h5ad'
    mtx_path = f'{raw_dir}/{sid}_counts.mtx'
    meta_path = f'{raw_dir}/{sid}_metadata.csv'
    genes_path = f'{raw_dir}/{sid}_genes.csv'
    coords_path = f'{raw_dir}/{sid}_coords.csv'

    if os.path.exists(out_path):
        # Check if it already has layer_guess
        tmp = ad.read_h5ad(out_path, backed='r')
        if 'layer_guess' in tmp.obs.columns or 'layer_guess_reordered' in tmp.obs.columns:
            print(f'[skip] {sid}.h5ad already has ground truth')
            tmp.file.close()
            continue
        tmp.file.close()

    if not os.path.exists(mtx_path):
        print(f'[skip] No R export found for {sid}')
        continue

    print(f'Converting {sid}...')
    X = scipy.io.mmread(mtx_path).T.tocsr()  # genes×cells → cells×genes
    meta = pd.read_csv(meta_path, index_col=0)
    genes = pd.read_csv(genes_path, index_col=0).iloc[:, 0].values
    coords = pd.read_csv(coords_path, index_col=0).values

    adata = ad.AnnData(X=X, obs=meta, var=pd.DataFrame(index=genes))
    adata.obsm['spatial'] = coords.astype(np.float32)
    adata.var_names_make_unique()

    # Ensure layer_guess is available
    gt_cols = [c for c in adata.obs.columns if 'layer' in c.lower()]
    print(f'  Ground truth columns found: {gt_cols}')
    print(f'  Shape: {adata.shape}, spatial coords: {adata.obsm[\"spatial\"].shape}')

    adata.write_h5ad(out_path)
    print(f'  Saved: {out_path}')
"
````

**If R is NOT found** — Method B-Python: pure Python via `spatialdata-io`

```bash
# Install spatialdata ecosystem (pure Python, no R needed)
pip install spatialdata spatialdata-io

python -c "
import os, urllib.request, json
import anndata as ad
import numpy as np

# spatialdata-io can read 10x Visium directly, but we need the layer_guess labels.
# These are bundled in the spatialLIBD ExperimentHub records.
# We download the pre-packaged h5ad files that already include layer_guess
# from the Bioconductor data CDN (no R required — these are static files).

out_dir = 'data/raw/DLPFC'
os.makedirs(out_dir, exist_ok=True)

samples = [
    '151507', '151508', '151509', '151510',
    '151669', '151670', '151671', '151672',
    '151673', '151674', '151675', '151676',
]

# Known mirrors for the annotated h5ad files
MIRRORS = [
    'https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5ad/{sid}_filtered_feature_bc_matrix.h5ad',
    'https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5ad/{sid}.h5ad',
    'https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/812/S-BSST812/Files/processed/{sid}.h5ad',
]

def try_download(sid, out_path):
    for template in MIRRORS:
        url = template.format(sid=sid)
        try:
            print(f'  Trying: {url}')
            urllib.request.urlretrieve(url, out_path)
            size_mb = os.path.getsize(out_path) / 1e6
            # Sanity check: must be > 1 MB (real data)
            if size_mb > 1.0:
                print(f'  OK: {size_mb:.1f} MB')
                return True
            else:
                os.remove(out_path)
        except Exception as e:
            if os.path.exists(out_path):
                os.remove(out_path)
    return False

failed = []
for sid in samples:
    out_path = f'{out_dir}/{sid}.h5ad'
    if os.path.exists(out_path):
        a = ad.read_h5ad(out_path, backed='r')
        n = a.n_obs; a.file.close()
        if n > 500:
            print(f'[skip] {sid}.h5ad ({n} spots)')
            continue
        os.remove(out_path)

    print(f'Downloading {sid}...')
    if not try_download(sid, out_path):
        failed.append(sid)

if failed:
    print(f'FAILED for: {failed}')
    print('These slices could not be auto-downloaded.')
    print('Use the R method or manually upload the files.')
else:
    print('All slices downloaded successfully.')
"
```

### Method C: If A and B both fail — wget with multiple URL fallbacks

```bash
cd /CauST
mkdir -p data/raw/DLPFC
cd data/raw/DLPFC

# These are the spatialLIBD hosted files (try each URL pattern)
for SID in 151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676; do
    if [ -f "${SID}.h5ad" ]; then
        echo "[skip] ${SID}.h5ad exists"
        continue
    fi
    echo "Downloading ${SID}..."
    wget -q "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5ad/${SID}_filtered_feature_bc_matrix.h5ad" -O "${SID}.h5ad" 2>/dev/null \
        || wget -q "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5ad/${SID}.h5ad" -O "${SID}.h5ad" 2>/dev/null \
        || wget -q "http://spatial.libd.org/spatialLIBD/${SID}.h5ad" -O "${SID}.h5ad" 2>/dev/null \
        || echo "  FAILED for ${SID}"

    if [ -f "${SID}.h5ad" ]; then
        ls -lh "${SID}.h5ad"
    fi
done

cd /CauST
```

### Verify downloads have ground truth

```bash
python -c "
import anndata as ad, os
raw_dir = 'data/raw/DLPFC'
for f in sorted(os.listdir(raw_dir)):
    if not f.endswith('.h5ad'): continue
    a = ad.read_h5ad(f'{raw_dir}/{f}', backed='r')
    gt_cols = [c for c in a.obs.columns if 'layer' in c.lower() or 'ground' in c.lower()]
    print(f'{f}: {a.n_obs} spots, {a.n_vars} genes, GT columns: {gt_cols}')
    a.file.close()
"
```

**What you need to see**: Each file should show `GT columns: ['layer_guess']` or
`['layer_guess_reordered']` or similar. If you see `GT columns: []`, the file
doesn't have ground truth and you need Method B (the R approach).

---

## STEP 2: Preprocess All Data

```bash
cd /CauST
python scripts/02_preprocess.py
```

Verify preprocessing worked:

```bash
python -c "
import os
proc = 'data/processed/DLPFC'
if os.path.exists(proc):
    files = sorted(os.listdir(proc))
    print(f'Processed DLPFC files: {len(files)}')
    for f in files:
        print(f'  {f}')
else:
    print('ERROR: No processed DLPFC directory')
"
```

---

## STEP 3: Install STAGATE (Sure-Shot Method)

The problem with STAGATE is version conflicts with your PyTorch/CUDA. Here's the
approach that works:

### Step 3a: Check your PyTorch+CUDA version first

> **If you see `NumPy 1.x / 2.x` crash on `import torch`**, run this first:
>
> ```bash
> pip install "numpy>=1.23,<2.0"
> ```

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
print(f'Python: {__import__(\"sys\").version}')
"
```

### Step 3b: Install STAGATE from source (bypasses PyPI version issues)

```bash
# Clone STAGATE-pyG directly from the source repo
cd /tmp
rm -rf STAGATE_pyG
git clone https://github.com/QIFEIDKN/STAGATE_pyG.git
cd STAGATE_pyG

# Install WITHOUT dependencies flag (avoids it trying to install conflicting torch versions)
pip install --no-deps .

# Go back to project
cd /CauST
```

### Step 3c: Fix STAGATE's torch_sparse dependency

The previous patch attempts left `gat_conv.py` in a broken state. We fix this by:

1. Reinstalling STAGATE from scratch (restores a clean `gat_conv.py`)
2. Creating a thin `torch_sparse` stub module — no file patching needed at all

```bash
# Step 1: Reinstall STAGATE cleanly
cd /tmp
rm -rf STAGATE_pyG
git clone https://github.com/QIFEIDKN/STAGATE_pyG.git
cd STAGATE_pyG
pip install --no-deps --force-reinstall .
cd /CauST

# Step 2: Create a torch_sparse stub in site-packages
# This satisfies STAGATE's import without any compilation or file patching
python -c "
import site, os
for d in site.getsitepackages():
    if os.path.isdir(d):
        stub = os.path.join(d, 'torch_sparse.py')
        with open(stub, 'w') as f:
            f.write('# Stub: torch_sparse not available for this torch build\n')
            f.write('class SparseTensor:\n    pass\n')
            f.write('def set_diag(t, *a, **kw):\n    return t\n')
        print(f'Created stub: {stub}')
        break
"

# Step 3: Pure-Python community detection
pip install python-louvain leidenalg
```

### Step 3d: Verify STAGATE works

```bash
python -c "
import STAGATE_pyG as STAGATE
print(f'STAGATE imported successfully')
print(f'Available functions: {[x for x in dir(STAGATE) if not x.startswith(\"_\")][:10]}')
"
```

### If Step 3b fails — Alternative: install STAGATE_pyG via pip with --no-deps

```bash
pip install STAGATE_pyG --no-deps
pip install louvain python-louvain
python -c "import STAGATE_pyG; print('OK')"
```

### If STAGATE still fails — Nuclear option: install in isolation

```bash
# Find your exact torch+cuda combo
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
# Example output: 2.1.0  11.8

# Then install STAGATE_pyG forcing no dependency resolution
pip install STAGATE_pyG --no-deps --force-reinstall

# Verify
python -c "
import STAGATE_pyG as STAGATE
import torch
print('STAGATE:', dir(STAGATE))
print('Torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
"
```

---

## STEP 4: Run Single-Slice Training (with ground truth)

```bash
cd /CauST

# Check which DLPFC slices have ground truth labels
python -c "
import scanpy as sc, os
proc = 'data/processed/DLPFC'
for f in sorted(os.listdir(proc)):
    if not f.endswith('.h5ad'): continue
    a = sc.read_h5ad(f'{proc}/{f}', backed='r')
    has_gt = 'layer_guess' in a.obs.columns or 'layer_guess_reordered' in a.obs.columns
    gt_key = 'layer_guess' if 'layer_guess' in a.obs.columns else ('layer_guess_reordered' if 'layer_guess_reordered' in a.obs.columns else None)
    print(f'{f}: {a.n_obs} spots, has_gt={has_gt}, key={gt_key}')
    a.file.close()
"
```

Now run single slice training on slice 151507 (or whichever has ground truth):

```bash
python scripts/03_train_single_slice.py
```

**Expected output**: You should see ARI, NMI, Silhouette scores printed. The ARI will
likely be low (0.01-0.20) because this is the lightweight GAT backend — that's OK.

---

## STEP 5: Run Multi-Slice + Invariance Analysis

```bash
python scripts/04_run_multi_slice.py
```

This runs the multi-slice pipeline with LODO cross-validation.

---

## STEP 6: Run Full Benchmark (The Big One)

This runs all 4 CauST variants across all datasets. It's resumable — if it crashes,
just re-run and it picks up where it left off.

```bash
# Delete old benchmark results (they only had Silhouette, no ARI)
rm -f experiments/results/benchmark/all_results.csv

# Run fresh benchmark
python scripts/05_benchmark.py
```

**This will take a while** (hours on GPU, longer on CPU). Each condition is saved
immediately so you can Ctrl+C and resume.

Check progress:

```bash
# See how many conditions are done
wc -l experiments/results/benchmark/all_results.csv
cat experiments/results/benchmark/all_results.csv | head -5
```

---

## STEP 7: Run STAGATE Integration Benchmark

After STAGATE is installed (Step 3) and the benchmark in Step 6 is done,
STAGATE conditions should already be included (the benchmark script tries
STAGATE automatically). If they were skipped, run this standalone:

```bash
python -c "
import sys
sys.path.insert(0, '.')

import scanpy as sc
import pandas as pd
import numpy as np
import torch
from caust import CauST
from caust.models.stagate_wrapper import run_with_stagate
from caust.evaluate.metrics import evaluate_single_slice
from caust.data.loader import load_and_preprocess
import os, json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

results = []
proc_dir = 'data/processed/DLPFC'
gt_key = 'layer_guess'

for f in sorted(os.listdir(proc_dir)):
    if not f.endswith('.h5ad'): continue
    sid = f.replace('.h5ad', '')
    print(f'\n=== {sid} ===')

    adata = sc.read_h5ad(f'{proc_dir}/{f}')
    print(f'  {adata.n_obs} spots × {adata.n_vars} genes')

    # Check ground truth
    has_gt = gt_key in adata.obs.columns
    labels_true = adata.obs[gt_key].values if has_gt else None
    print(f'  Ground truth: {has_gt}')

    # --- Baseline STAGATE (raw HVG, no CauST) ---
    print('  Running STAGATE baseline...')
    try:
        adata_base = run_with_stagate(adata.copy(), device=device)
        Z_base = adata_base.obsm.get('STAGATE', None)
        if Z_base is not None and 'mclust' in adata_base.obs.columns:
            labels_base = adata_base.obs['mclust'].astype(int).values
        else:
            from caust.causal.scorer import cluster_latent
            labels_base = cluster_latent(Z_base, 7)
        m = evaluate_single_slice(labels_base, Z_base, labels_true, prefix='')
        m['slice'] = sid; m['variant'] = 'Baseline'; m['method'] = 'STAGATE'; m['dataset'] = 'DLPFC'
        results.append(m)
        print(f'  STAGATE baseline: ARI={m.get(\"ari\", \"N/A\"):.4f}  Sil={m.get(\"silhouette\", \"N/A\"):.4f}')
    except Exception as e:
        print(f'  STAGATE baseline FAILED: {e}')

    # --- CauST + STAGATE ---
    print('  Running CauST → STAGATE...')
    try:
        model = CauST(n_causal_genes=500, n_clusters=7, epochs=500,
                       filter_mode='filter_and_reweight',
                       scoring_method='gradient+perturbation', verbose=False)
        adata_caust = model.fit_transform(adata.copy())
        print(f'  CauST selected {adata_caust.n_vars} genes')

        adata_stagate = run_with_stagate(adata_caust.copy(), device=device)
        Z_caust = adata_stagate.obsm.get('STAGATE', None)
        if Z_caust is not None and 'mclust' in adata_stagate.obs.columns:
            labels_caust = adata_stagate.obs['mclust'].astype(int).values
        else:
            from caust.causal.scorer import cluster_latent
            labels_caust = cluster_latent(Z_caust, 7)
        m = evaluate_single_slice(labels_caust, Z_caust, labels_true, prefix='')
        m['slice'] = sid; m['variant'] = 'CauST-Full'; m['method'] = 'STAGATE'; m['dataset'] = 'DLPFC'
        results.append(m)
        print(f'  CauST+STAGATE: ARI={m.get(\"ari\", \"N/A\"):.4f}  Sil={m.get(\"silhouette\", \"N/A\"):.4f}')
    except Exception as e:
        print(f'  CauST+STAGATE FAILED: {e}')

    del adata
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Save results
if results:
    df = pd.DataFrame(results)
    os.makedirs('experiments/results/benchmark', exist_ok=True)

    # Append to existing CSV if present
    csv_path = 'experiments/results/benchmark/all_results.csv'
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        # Remove old STAGATE rows to avoid duplicates
        df_old = df_old[df_old['method'] != 'STAGATE']
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f'\nResults saved: {csv_path}')
    print(df.to_string(index=False))
"
```

---

## STEP 8: Regenerate All Figures

```bash
python scripts/06_visualize_results.py
```

Check what figures were generated:

```bash
ls -la experiments/results/figures/
```

---

## STEP 9: Re-run Tutorials with Real Data

```bash
# Open JupyterLab and re-run the quickstart notebook
# Or run from terminal:
pip install jupyter nbconvert  # if not installed

# Run notebook 01 non-interactively
jupyter nbconvert --to notebook --execute tutorials/01_quickstart.ipynb \
    --output 01_quickstart.ipynb --ExecutePreprocessor.timeout=1200

# Run notebook 05 (causal gene exploration)
jupyter nbconvert --to notebook --execute tutorials/05_causal_gene_exploration.ipynb \
    --output 05_causal_gene_exploration.ipynb --ExecutePreprocessor.timeout=1200
```

---

## STEP 10: Run Tests (make sure nothing is broken)

```bash
python -m pytest tests/ -v --tb=short
```

---

## STEP 11: Update README with New Results

After all runs are complete, check what your new numbers look like:

```bash
python -c "
import pandas as pd, json, os

print('=== Benchmark Results ===')
csv = 'experiments/results/benchmark/all_results.csv'
if os.path.exists(csv):
    df = pd.read_csv(csv)
    print(f'Total conditions: {len(df)}')
    print(f'Columns: {list(df.columns)}')
    print()
    # Show ARI if available
    if 'ari' in df.columns:
        print('--- ARI by method×variant ---')
        pivot = df.pivot_table(values='ari', index='slice', columns=['variant','method'], aggfunc='mean')
        print(pivot.to_string())
    print()
    if 'silhouette' in df.columns:
        print('--- Silhouette by method×variant ---')
        pivot = df.pivot_table(values='silhouette', index='slice', columns=['variant','method'], aggfunc='mean')
        print(pivot.to_string())

print()
print('=== Single Slice ===')
for f in ['151507_metrics.json', 'P4_rep1_metrics.json']:
    path = f'experiments/results/single_slice/{f}'
    if os.path.exists(path):
        with open(path) as fh:
            print(f'{f}: {json.load(fh)}')

print()
print('=== Multi-Slice Summary ===')
agg = 'experiments/results/multi_slice/aggregate_summary.json'
if os.path.exists(agg):
    with open(agg) as fh:
        print(json.load(fh))
"
```

Then update the README.md benchmark tables with the new numbers.

---

## STEP 12: Git Commit & Push

```bash
cd /CauST
git add -A
git status

# Review what's changed
git diff --cached --stat

# Commit
git commit -m "feat: complete benchmark with ground-truth ARI + STAGATE integration

- Downloaded 12-slice spatialLIBD DLPFC data with layer_guess labels
- Re-ran all benchmarks with ARI/NMI evaluation
- Added STAGATE integration benchmark (CauST genes → STAGATE)
- Regenerated all figures with real data
- Updated README with honest results
- Re-ran tutorial notebooks with real gene names"

git push origin main
```

---

## Troubleshooting

### "STAGATE import error / version mismatch"

```bash
# Check what's installed
pip show STAGATE-pyG
pip show torch torch-geometric

# Force reinstall from source with no deps
pip install --no-deps --force-reinstall git+https://github.com/QIFEIDKN/STAGATE_pyG.git
```

### "Download failed / URL not found"

```bash
# Check if you can reach the S3 bucket at all
curl -I "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5ad/151507_filtered_feature_bc_matrix.h5ad"

# If 403/404, the files may have moved. Use the R method (Method B in Step 1).
```

### "layer_guess not found in downloaded files"

The GEO-sourced DLPFC files (P4_rep1, etc.) do NOT have layer_guess.
Only the spatialLIBD 151507-151676 files have it.
You MUST use Method B (R download) or find the annotated h5ad files.

### "Out of GPU memory"

```bash
# Reduce batch size or epochs in the scripts
# Or run one slice at a time:
python scripts/05_benchmark.py  # it's resumable — will skip completed conditions
```

### "Preprocess fails on new data"

```bash
# Check the raw file is valid
python -c "import anndata; a = anndata.read_h5ad('data/raw/DLPFC/151507.h5ad'); print(a)"
```
