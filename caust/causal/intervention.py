"""
caust/causal/intervention.py
=============================
In-silico gene intervention strategies.

What is an "in-silico intervention"?
--------------------------------------
Instead of doing a real CRISPR knockout in the lab (months of work,
expensive), we simulate it in software in milliseconds:

    Original data      → X  (n_spots × n_genes)
    After knocking out → X' (same shape, but column g is modified)

The default strategy is MEAN IMPUTATION:
    do(Gene_g = E[Gene_g])

This is the most principled approach because:
  1. It removes the *variation* of gene g (which drives domain structure)
  2. It preserves the *mean level* (avoids out-of-distribution artefacts)
  3. It corresponds exactly to Pearl's do-calculus intervention on X_g

Reference: Pearl J. (2009) "Causality: Models, Reasoning and Inference"
"""

from typing import List, Literal

import numpy as np


InterventionMethod = Literal["mean_impute", "zero_out", "median_impute"]


def apply_intervention(
    X: np.ndarray,
    gene_idx: int,
    method: InterventionMethod = "mean_impute",
) -> np.ndarray:
    """
    Apply a single-gene in-silico intervention on expression matrix X.

    Creates a copy of X where the expression of gene `gene_idx` is
    replaced according to the chosen strategy.

    Parameters
    ----------
    X        : expression matrix  (n_spots × n_genes),  float32 preferred
    gene_idx : column index of the gene to intervene on
    method   :
        'mean_impute'   → do(Gene = E[Gene])          — DEFAULT, most principled
        'zero_out'      → do(Gene = 0)                — simulates gene knockout
        'median_impute' → do(Gene = median(Gene))     — robust to outliers

    Returns
    -------
    X_perturbed : copy of X with column gene_idx modified
    """
    X_perturbed = X.copy()

    if method == "mean_impute":
        X_perturbed[:, gene_idx] = float(X[:, gene_idx].mean())
    elif method == "zero_out":
        X_perturbed[:, gene_idx] = 0.0
    elif method == "median_impute":
        X_perturbed[:, gene_idx] = float(np.median(X[:, gene_idx]))
    else:
        raise ValueError(
            f"Unknown intervention method: '{method}'. "
            "Choose from: 'mean_impute', 'zero_out', 'median_impute'."
        )

    return X_perturbed


def apply_batch_interventions(
    X: np.ndarray,
    gene_indices: List[int],
    method: InterventionMethod = "mean_impute",
) -> np.ndarray:
    """
    Apply interventions on *multiple* genes simultaneously.

    Useful for testing the joint effect of a group of genes, or for
    efficiently computing multi-gene knockout experiments.

    Parameters
    ----------
    X            : expression matrix  (n_spots × n_genes)
    gene_indices : list of column indices to intervene on simultaneously
    method       : intervention strategy (same options as apply_intervention)

    Returns
    -------
    X_perturbed : copy with all specified gene columns modified
    """
    X_perturbed = X.copy()
    for idx in gene_indices:
        if method == "mean_impute":
            X_perturbed[:, idx] = float(X[:, idx].mean())
        elif method == "zero_out":
            X_perturbed[:, idx] = 0.0
        elif method == "median_impute":
            X_perturbed[:, idx] = float(np.median(X[:, idx]))
        else:
            raise ValueError(f"Unknown method: '{method}'")
    return X_perturbed


def compute_global_disruption(
    Z_original: np.ndarray,
    Z_perturbed: np.ndarray,
) -> float:
    """
    Measure how much a gene intervention changed the latent space globally.

    Why this is needed (confounder control)
    ----------------------------------------
    A "housekeeping" gene (e.g. a ribosomal gene active everywhere) will
    score high on raw perturbation scoring simply because removing it
    disrupts the model globally.  We normalise the causal score by this
    global disruption to highlight genes specifically important for
    *spatial organisation*, not just general cell function.

        adjusted_score = domain_change / global_disruption

    Parameters
    ----------
    Z_original  : latent matrix before intervention  (n_spots × latent_dim)
    Z_perturbed : latent matrix after  intervention  (n_spots × latent_dim)

    Returns
    -------
    float : mean L2 distance between original and perturbed latent vectors
    """
    diff = Z_original - Z_perturbed
    per_spot_dist = np.linalg.norm(diff, axis=1)   # shape: (n_spots,)
    return float(per_spot_dist.mean())
