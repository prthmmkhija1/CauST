"""
caust/filter/gene_filter.py
============================
Gene selection and reweighting strategies based on causal scores.

Three strategies (Option C — all supported):

  A. filter_top_k          — hard selection of top-K causal genes
  B. reweight_genes         — soft reweighting by causal score
  C. filter_and_reweight    — select top-K then reweight (DEFAULT, strongest)

Why is filter_and_reweight the default?
-----------------------------------------
  1. Filtering  : removes the noise of irrelevant genes completely
  2. Reweighting: amplifies strongly causal genes WITHIN the selected set
  Combined, this gives the model the clearest possible signal.
  This is the setting expected to win in the ablation study.
"""

from typing import Dict, List, Literal, Optional

import numpy as np
import scanpy as sc
import scipy.sparse as sp


FilterMode = Literal["filter", "reweight", "filter_and_reweight"]


# ---------------------------------------------------------------------------
# Strategy A: Hard filtering
# ---------------------------------------------------------------------------

def filter_top_k(
    adata: sc.AnnData,
    gene_scores: Dict[str, float],
    k: int = 500,
    score_threshold: Optional[float] = None,
    inplace: bool = False,
) -> sc.AnnData:
    """
    Keep only the top-K genes by causal score.

    Parameters
    ----------
    adata            : preprocessed AnnData
    gene_scores      : {gene_name: causal_score} from scorer
    k                : number of top genes to retain
    score_threshold  : optional minimum score — genes below this are dropped
                       even if they would be in the top-K
    inplace          : modify adata in place, or return a copy

    Returns
    -------
    AnnData with  n_vars == min(k, n_eligible_genes)
    """
    if not inplace:
        adata = adata.copy()

    available_genes = set(adata.var_names)
    scored_genes    = {g: s for g, s in gene_scores.items() if g in available_genes}

    if score_threshold is not None:
        scored_genes = {g: s for g, s in scored_genes.items() if s >= score_threshold}

    if not scored_genes:
        raise ValueError(
            "No genes in gene_scores match any gene in adata.var_names after filtering. "
            "Make sure gene names use the same notation (e.g. 'BRCA1' vs 'brca1')."
        )

    # Select top-K
    top_genes: List[str] = sorted(scored_genes, key=scored_genes.get, reverse=True)[:k]

    # Preserve only those that actually exist in adata
    top_genes = [g for g in top_genes if g in available_genes]

    adata = adata[:, top_genes].copy()

    # Store the scores used for selection in adata.var
    adata.var["causal_score"] = [gene_scores.get(g, 0.0) for g in adata.var_names]

    print(
        f"[filter] Hard filter: {len(top_genes)} / {len(available_genes)} genes kept  "
        f"(requested k={k})"
    )
    return adata


# ---------------------------------------------------------------------------
# Strategy B: Soft reweighting
# ---------------------------------------------------------------------------

def reweight_genes(
    adata: sc.AnnData,
    gene_scores: Dict[str, float],
    inplace: bool = False,
) -> sc.AnnData:
    """
    Multiply each gene's expression by its causal score.

    This keeps all genes but amplifies causally important ones and
    attenuates bystander genes, giving the downstream model a clearer
    signal without discarding any potential information.

        X'[spot, gene] = X[spot, gene] × causal_score[gene]

    Parameters
    ----------
    adata       : preprocessed AnnData
    gene_scores : {gene_name: causal_score}  (values in [0, 1])
    inplace     : modify adata.X in place, or return a copy

    Returns
    -------
    AnnData with reweighted adata.X
    """
    if not inplace:
        adata = adata.copy()

    scores_arr = np.array(
        [gene_scores.get(g, 0.0) for g in adata.var_names], dtype=np.float32
    )

    X = adata.X
    if sp.issparse(X):
        # In-place sparse multiplication: multiply each column
        X = X.toarray()

    X_weighted = X * scores_arr[np.newaxis, :]   # broadcast over rows
    adata.X    = X_weighted

    # Store scores in var
    adata.var["causal_score"] = scores_arr

    n_zero = int((scores_arr == 0.0).sum())
    print(
        f"[filter] Soft reweight: {adata.n_vars} genes reweighted "
        f"({n_zero} genes with zero score effectively silenced)"
    )
    return adata


# ---------------------------------------------------------------------------
# Strategy C: Filter + reweight (DEFAULT)
# ---------------------------------------------------------------------------

def filter_and_reweight(
    adata: sc.AnnData,
    gene_scores: Dict[str, float],
    k: int = 500,
    score_threshold: Optional[float] = None,
    inplace: bool = False,
) -> sc.AnnData:
    """
    First select top-K causal genes, then reweight them by their score.

    This is the DEFAULT CauST mode. It combines the strengths of both:
      - Hard filter: removes noise-only genes entirely
      - Soft reweight: amplifies the strongest causal signals within the kept set

    Parameters
    ----------
    adata            : preprocessed AnnData
    gene_scores      : {gene_name: causal_score}
    k                : genes to keep after hard filtering
    score_threshold  : optional minimum score threshold before reweighting
    inplace          : operate in place or on a copy

    Returns
    -------
    AnnData with n_vars == min(k, eligible) and reweighted expression
    """
    # Step 1: Hard filter
    adata_filtered = filter_top_k(
        adata,
        gene_scores,
        k=k,
        score_threshold=score_threshold,
        inplace=inplace,
    )
    # Step 2: Reweight within the filtered gene set
    adata_filtered = reweight_genes(adata_filtered, gene_scores, inplace=True)

    print(
        f"[filter] CauST filter+reweight: "
        f"{adata_filtered.n_vars} genes, expression reweighted by causal score."
    )
    return adata_filtered


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def apply_gene_selection(
    adata: sc.AnnData,
    gene_scores: Dict[str, float],
    mode: FilterMode = "filter_and_reweight",
    k: int = 500,
    score_threshold: Optional[float] = None,
    inplace: bool = False,
) -> sc.AnnData:
    """
    Apply CauST gene selection in the chosen mode.

    Parameters
    ----------
    adata            : preprocessed AnnData
    gene_scores      : {gene_name: causal_score}
    mode             :
        'filter'            → hard top-K selection only
        'reweight'          → soft reweighting only (all genes kept)
        'filter_and_reweight' → top-K then reweight  [DEFAULT]
    k                : genes to keep (used in 'filter' and 'filter_and_reweight')
    score_threshold  : optional minimum causal score
    inplace          : operate in place or return a fresh copy

    Returns
    -------
    AnnData ready for downstream domain identification
    """
    if mode == "filter":
        return filter_top_k(adata, gene_scores, k=k,
                             score_threshold=score_threshold, inplace=inplace)
    elif mode == "reweight":
        return reweight_genes(adata, gene_scores, inplace=inplace)
    elif mode == "filter_and_reweight":
        return filter_and_reweight(adata, gene_scores, k=k,
                                   score_threshold=score_threshold, inplace=inplace)
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Choose from: 'filter', 'reweight', 'filter_and_reweight'."
        )
