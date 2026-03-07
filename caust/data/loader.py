"""
caust/data/loader.py
====================
Load and preprocess spatial transcriptomics data from .h5ad files.

AnnData format recap:
    adata.X         — expression matrix  (n_spots × n_genes)
    adata.obs       — per-spot metadata  (coordinates, labels, …)
    adata.var       — per-gene metadata  (gene names, HVG flags, …)
    adata.obsm      — multi-dim per-spot arrays (e.g. spatial coords)
    adata.layers    — copies of the matrix at different processing stages
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Single-slice helpers
# ---------------------------------------------------------------------------

def load_and_preprocess(
    path: Union[str, Path, sc.AnnData],
    n_top_genes: int = 3000,
    min_genes: int = 200,
    min_cells: int = 3,
    normalize: bool = True,
    scale: bool = True,
    spatial_key: str = "spatial",
) -> sc.AnnData:
    """
    Load a spatial transcriptomics .h5ad file (or accept an AnnData directly)
    and run the standard preprocessing pipeline used throughout CauST.

    Pipeline
    --------
    1.  Filter low-quality spots  (< min_genes expressed genes)
    2.  Filter rarely-seen genes  (expressed in < min_cells spots)
    3.  Normalize total counts per spot to 10 000
    4.  Log1p  transform
    5.  Select highly-variable genes (top n_top_genes)
    6.  Scale  (zero-mean, unit-variance, capped at 10)

    Parameters
    ----------
    path          : path to the .h5ad file, or an AnnData object directly
    n_top_genes   : number of highly-variable genes to keep
    min_genes     : spots with fewer expressed genes are dropped
    min_cells     : genes expressed in fewer spots are dropped
    normalize     : apply normalization + log1p?
    scale         : apply per-gene z-score scaling?
    spatial_key   : adata.obsm key holding (x, y) coordinates

    Returns
    -------
    Preprocessed AnnData
    """
    if isinstance(path, sc.AnnData):
        adata = path.copy()
        print(f"[loader] Received AnnData directly")
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        print(f"[loader] Loading: {path}")
        adata = sc.read_h5ad(path)
    print(f"         Raw shape: {adata.n_obs} spots × {adata.n_vars} genes")

    # ── Detect if data is already preprocessed ───────────────────────────
    # Strategy: check for raw/counts first, then fall back to checking
    # whether adata.X has negative values (= already scaled → skip QC).
    already_scaled = False

    if hasattr(adata, "raw") and adata.raw is not None:
        # spatialLIBD files often store raw counts in adata.raw
        print("         Found adata.raw — using raw counts for QC & normalization")
        adata_raw = adata.raw.to_adata()
        # Transfer obsm (spatial coords) and obs (metadata) from original
        adata_raw.obsm = adata.obsm
        adata_raw.obs  = adata.obs.loc[adata_raw.obs_names]
        adata = adata_raw
    elif "counts" in getattr(adata, "layers", {}):
        print("         Found layers['counts'] — using raw counts for QC & normalization")
        import scipy.sparse as _sp
        X_counts = adata.layers["counts"]
        if _sp.issparse(X_counts):
            X_counts = X_counts.toarray()
        adata.X = X_counts
    else:
        # Check if X contains negative values → already scaled; skip QC/norm
        import scipy.sparse as _sp
        X_check = adata.X
        if _sp.issparse(X_check):
            X_check = X_check.toarray()
        if float(X_check.min()) < -0.01:
            already_scaled = True
            print(
                "         Data appears already scaled (min < 0) — "
                "skipping QC filter, normalization, and scaling steps."
            )

    # Preserve raw counts before any modification (for later reference)
    adata.layers["counts"] = adata.X.copy()

    # ── Quality filtering ────────────────────────────────────────────────
    if not already_scaled:
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        print(f"         After QC : {adata.n_obs} spots × {adata.n_vars} genes")

    # ── Normalization ────────────────────────────────────────────────────
    if normalize and not already_scaled:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ── Highly-variable gene selection ───────────────────────────────────
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=False)
    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"         HVGs selected: {n_hvg}")

    # Keep log-normalized expression before scaling
    adata.layers["log_norm"] = adata.X.copy()

    # Subset to HVGs
    adata = adata[:, adata.var["highly_variable"]].copy()

    # ── Scaling ──────────────────────────────────────────────────────────
    if scale and not already_scaled:
        sc.pp.scale(adata, max_value=10)
        adata.layers["scaled"] = adata.X.copy()

    # Verify spatial coordinates
    if not check_spatial_coords(adata, spatial_key):
        warnings.warn(
            f"Spatial coordinates not found at adata.obsm['{spatial_key}']. "
            "Graph construction will fail unless coordinates are added."
        )

    print(f"         Final shape : {adata.n_obs} spots × {adata.n_vars} genes\n")
    return adata


def load_multiple_slices(
    paths: Union[Dict, List[Union[str, Path]]],
    slice_ids: Optional[List] = None,
    **preprocess_kwargs,
) -> Dict:
    """
    Load and preprocess multiple spatial transcriptomics slices.

    Parameters
    ----------
    paths      : dict {slice_id: path_or_AnnData} **or** list of .h5ad file paths
    slice_ids  : identifiers for each slice (defaults to 0, 1, 2, …).
                 Ignored when *paths* is a dict (keys are used instead).
    **preprocess_kwargs : forwarded to load_and_preprocess

    Returns
    -------
    dict  {slice_id: AnnData}
    """
    # Accept a dict mapping slice_id → path/AnnData
    if isinstance(paths, dict):
        items = list(paths.items())
    else:
        if slice_ids is None:
            slice_ids = list(range(len(paths)))
        if len(paths) != len(slice_ids):
            raise ValueError("len(paths) must equal len(slice_ids)")
        items = list(zip(slice_ids, paths))

    slices: Dict = {}
    for sid, path in items:
        print(f"\n{'─'*50}")
        print(f"Loading slice: {sid}")
        slices[sid] = load_and_preprocess(path, **preprocess_kwargs)

    print(f"\n[loader] Loaded {len(slices)} slices successfully.")
    return slices


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def check_spatial_coords(adata: sc.AnnData, spatial_key: str = "spatial") -> bool:
    """Return True if valid 2-D spatial coordinates exist."""
    if spatial_key not in adata.obsm:
        return False
    coords = adata.obsm[spatial_key]
    return coords.ndim == 2 and coords.shape[1] >= 2


def save_processed(adata: sc.AnnData, out_path: Union[str, Path]) -> None:
    """Write a preprocessed AnnData to disk, creating parent dirs as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"[loader] Saved → {out_path}")
