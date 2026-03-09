"""
caust/models/stagate_wrapper.py
================================
Thin wrappers to run STAGATE-pyG and GraphST on CauST-filtered data.

CauST acts as a *preprocessing step* — it selects/reweights genes, then
passes the result straight into existing state-of-the-art pipelines.
Neither STAGATE nor GraphST needs to be modified.

Install
-------
    pip install STAGATE-pyG
    # GraphST: see https://github.com/JinmiaoChenLab/GraphST
"""

from typing import List, Optional

import scanpy as sc
import torch


def run_with_stagate(
    adata: sc.AnnData,
    hidden_dims: Optional[List[int]] = None,
    n_epochs: int = 500,
    rad_cutoff: Optional[float] = None,
    device: str = "auto",
    verbose: bool = True,
) -> sc.AnnData:
    """
    Run STAGATE on CauST-preprocessed data.

    CauST has already selected the causally important genes; STAGATE
    then builds its own graph attention autoencoder on top of them.

    Parameters
    ----------
    adata        : CauST-filtered AnnData (output of CauST.fit_transform)
    hidden_dims  : STAGATE encoder dims, e.g. [512, 30]
    n_epochs     : training epochs
    rad_cutoff   : spatial radius for STAGATE's graph (auto-tuned if None)
    device       : 'auto' → GPU if available, else CPU
    verbose      : print progress

    Returns
    -------
    adata with STAGATE latent space in adata.obsm["STAGATE"]
    """
    try:
        import STAGATE_pyG as STAGATE
    except ImportError:
        raise ImportError(
            "STAGATE is not installed.\n"
            "Run:  pip install STAGATE-pyG\n"
            "Then re-run this function."
        )

    if hidden_dims is None:
        hidden_dims = [512, 30]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(
            f"[stagate_wrapper] Running STAGATE on {adata.n_obs} spots × "
            f"{adata.n_vars} CauST-filtered genes  (device={device})"
        )

    # Build STAGATE's own spatial graph if not already present
    if "spatial_connectivities" not in adata.obsp or rad_cutoff is not None:
        if rad_cutoff is None:
            rad_cutoff = 150  # default for 10x Visium
        STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)

    STAGATE.train_STAGATE(adata, hidden_dims=hidden_dims, n_epochs=n_epochs, device=device)

    if verbose:
        print("  Done. Latent vectors stored in adata.obsm['STAGATE']")

    return adata


def run_with_graphst(
    adata: sc.AnnData,
    n_domains: int = 7,
    device: str = "auto",
    verbose: bool = True,
) -> sc.AnnData:
    """
    Run GraphST on CauST-preprocessed data.

    Parameters
    ----------
    adata      : CauST-filtered AnnData
    n_domains  : expected number of spatial domains
    device     : 'auto' → GPU if available, else CPU

    Returns
    -------
    adata with GraphST representations and domain labels
    """
    try:
        from GraphST import GraphST
    except ImportError:
        raise ImportError(
            "GraphST is not installed.\n"
            "See: https://github.com/JinmiaoChenLab/GraphST\n"
            "or:  pip install GraphST"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(
            f"[graphst_wrapper] Running GraphST on {adata.n_obs} spots × "
            f"{adata.n_vars} CauST-filtered genes  (device={device})"
        )

    model = GraphST.GraphST(adata, device=device)
    adata = model.train()

    if verbose:
        print("  Done.")

    return adata
