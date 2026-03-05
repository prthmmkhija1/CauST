"""
CauST: Causal Gene Intervention for Robust Spatial Domain Identification
=========================================================================

A framework that identifies causally important genes in spatial transcriptomics
by simulating gene knockouts and measuring their effect on spatial domain
assignments. Introduces a causally-grounded perspective to spatial domain
identification by explicitly modeling gene-level interventions.

Project: OSRE 2026 – UC Irvine
Mentor:  Lijinghua Zhang, PhD
Author:  Pratham Makhija (prthmmkhija1)
GitHub:  https://github.com/prthmmkhija1/CauST

Quick Start
-----------
    from caust import CauST
    import scanpy as sc

    adata = sc.read_h5ad("data/processed/DLPFC_sample1_preprocessed.h5ad")

    # Single-slice run
    model = CauST(n_causal_genes=500, alpha=0.5, n_clusters=7)
    adata_filtered = model.fit_transform(adata)

    # Multi-slice run (enables invariance scoring — recommended)
    slices = {"s1": adata_s1, "s2": adata_s2, "s3": adata_s3}
    model.fit_multi_slice(slices, n_clusters=7)
    adata_filtered = model.transform(adata_s1)

    # Inspect causal genes
    print(model.get_top_causal_genes(n=20))
"""

from caust.pipeline import CauST

__version__ = "0.1.0"
__author__ = "Pratham Makhija"
__url__ = "https://github.com/prthmmkhija1/CauST"
__all__ = ["CauST"]
