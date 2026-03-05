"""
caust/causal/invariance.py
===========================
Cross-slice invariance analysis for causal gene identification.

The core idea (from IRM — Invariant Risk Minimization, Arjovsky 2019)
----------------------------------------------------------------------
A gene is *truly* causal for spatial domain formation if its causal
effect is CONSISTENT across different tissue sections and donors.

  - Gene A: causal scores [0.82, 0.79, 0.83, 0.81] across 4 slices
            → highly invariant  → true causal gene

  - Gene B: causal scores [0.80, 0.78, 0.12, 0.09] across 4 slices
            → inconsistent      → likely confounded by donor-specific effects

We measure invariance with:
    invariance_score[g] = mean(scores) / (1 + variance(scores))

High mean + low variance = high invariance = reliable spatial driver.

We also compute cross-donor Pearson correlation: if the *ranking* of
genes is similar across donors, those top genes are truly universal.

Reference:
    Arjovsky et al. (2019). "Invariant Risk Minimization." arXiv:1907.02893
"""

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Per-gene invariance scoring
# ---------------------------------------------------------------------------

def compute_invariance_scores(
    causal_scores_per_slice: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    Compute an invariance score for every gene across multiple slices.

    Formula
    -------
        invariance_score[g] = mean_score[g] / (1 + variance[g])

    This rewards genes that are:
      1. Consistently *high* in causal score (high mean)
      2. Stable across all slices (low variance)

    Parameters
    ----------
    causal_scores_per_slice : dict  {slice_id: {gene_name: causal_score}}
        Each inner dict maps gene names to their perturbation causal score
        computed on that slice independently.

    Returns
    -------
    dict  {gene_name: invariance_score}  sorted descending
    """
    if not causal_scores_per_slice:
        raise ValueError("causal_scores_per_slice is empty.")

    slice_ids  = list(causal_scores_per_slice.keys())
    all_genes  = set()
    for s in slice_ids:
        all_genes.update(causal_scores_per_slice[s].keys())
    all_genes = sorted(all_genes)

    inv_scores: Dict[str, float] = {}

    for gene in all_genes:
        gene_vals: List[float] = []
        for sid in slice_ids:
            if gene in causal_scores_per_slice[sid]:
                gene_vals.append(causal_scores_per_slice[sid][gene])

        if len(gene_vals) < 2:
            # Only seen in one slice — cannot assess invariance
            inv_scores[gene] = float(np.mean(gene_vals)) if gene_vals else 0.0
            continue

        mu  = float(np.mean(gene_vals))
        var = float(np.var(gene_vals))
        inv_scores[gene] = mu / (1.0 + var)

    # Normalise to [0, 1]
    max_inv = max(inv_scores.values()) if inv_scores else 1.0
    if max_inv > 0:
        inv_scores = {g: v / max_inv for g, v in inv_scores.items()}

    inv_scores = dict(sorted(inv_scores.items(), key=lambda kv: kv[1], reverse=True))

    print(
        f"[invariance] Scored {len(inv_scores)} genes across {len(slice_ids)} slices."
    )
    print(
        "  Top-5 invariant genes: "
        + ", ".join(f"{g}({s:.3f})" for g, s in list(inv_scores.items())[:5])
    )
    return inv_scores


# ---------------------------------------------------------------------------
# Cross-donor correlation analysis
# ---------------------------------------------------------------------------

def compute_cross_donor_correlation(
    causal_scores_per_slice: Dict[str, Dict[str, float]],
    donor_map: Optional[Dict[str, str]] = None,
) -> Tuple[float, float]:
    """
    Measure how similar gene causal rankings are across donors.

    If gene rankings are consistent across donors (e.g. Donor 1 and
    Donor 2 both rank the same genes at the top), this is strong evidence
    that those genes are universal spatial drivers, not donor-specific noise.

    Parameters
    ----------
    causal_scores_per_slice : dict  {slice_id: {gene: score}}
    donor_map               : optional dict  {slice_id: donor_id}
        If provided, computes correlation between donors (not just slices).
        If None, treats each slice as a separate "environment".

    Returns
    -------
    (mean_pearson_r, mean_spearman_r)
        Mean rank correlations across all pairwise combinations.
        Values close to 1.0 mean gene rankings are consistent.
    """
    slice_ids = list(causal_scores_per_slice.keys())
    if len(slice_ids) < 2:
        print("[invariance] Need ≥2 slices for cross-donor correlation.")
        return 1.0, 1.0

    # Collect all common genes
    common_genes = set(causal_scores_per_slice[slice_ids[0]].keys())
    for sid in slice_ids[1:]:
        common_genes &= set(causal_scores_per_slice[sid].keys())
    common_genes = sorted(common_genes)

    if len(common_genes) < 10:
        print(f"[invariance] Warning: only {len(common_genes)} common genes across slices.")
        return 0.0, 0.0

    if donor_map is not None:
        # Group slices by donor and average within each donor
        donor_scores: Dict[str, Dict[str, float]] = {}
        for sid, donor in donor_map.items():
            if sid not in causal_scores_per_slice:
                continue
            if donor not in donor_scores:
                donor_scores[donor] = {g: [] for g in common_genes}
            for g in common_genes:
                v = causal_scores_per_slice[sid].get(g, 0.0)
                donor_scores[donor][g].append(v)
        # Average per-donor
        keys_to_compare = list(donor_scores.keys())
        vecs = {
            d: np.array([
                np.mean(donor_scores[d][g]) for g in common_genes
            ]) for d in keys_to_compare
        }
    else:
        keys_to_compare = slice_ids
        vecs = {
            sid: np.array([causal_scores_per_slice[sid].get(g, 0.0) for g in common_genes])
            for sid in slice_ids
        }

    pearson_vals: List[float] = []
    spearman_vals: List[float] = []

    for key_a, key_b in combinations(keys_to_compare, 2):
        va, vb = vecs[key_a], vecs[key_b]
        pr, _ = pearsonr(va, vb)
        sr, _ = spearmanr(va, vb)
        pearson_vals.append(pr)
        spearman_vals.append(sr)

    mean_pearson  = float(np.mean(pearson_vals))
    mean_spearman = float(np.mean(spearman_vals))

    n_pairs = len(pearson_vals)
    print(
        f"[invariance] Cross-donor correlation  ({n_pairs} pairs):  "
        f"Pearson r = {mean_pearson:.3f},  Spearman ρ = {mean_spearman:.3f}"
    )
    return mean_pearson, mean_spearman


# ---------------------------------------------------------------------------
# Combine causal + invariance into final score
# ---------------------------------------------------------------------------

def combine_causal_and_invariance(
    causal_scores_per_slice: Dict[str, Dict[str, float]],
    invariance_scores: Dict[str, float],
    alpha: float = 0.5,
) -> Dict[str, float]:
    """
    Compute the final CauST gene score by combining:
        (1) Mean causal score across slices
        (2) Invariance score

    Formula
    -------
        final_score[g] = alpha * mean_causal[g] + (1 - alpha) * invariance[g]

    Where:
        alpha = 0.5  →  equal weight to causal strength and consistency
        alpha → 1    →  favour stronger causal effects even if less consistent
        alpha → 0    →  favour cross-slice consistency even if individually weak

    Parameters
    ----------
    causal_scores_per_slice : dict {slice_id: {gene: score}}
    invariance_scores       : dict {gene: invariance_score}
    alpha                   : blending weight in [0, 1]

    Returns
    -------
    dict  {gene_name: final_score}   sorted descending
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    all_genes = set(invariance_scores.keys())
    for sid in causal_scores_per_slice:
        all_genes.update(causal_scores_per_slice[sid].keys())

    final: Dict[str, float] = {}
    for gene in all_genes:
        # Mean causal score across all available slices for this gene
        gene_causal_vals = [
            causal_scores_per_slice[sid].get(gene, 0.0)
            for sid in causal_scores_per_slice
        ]
        mean_causal = float(np.mean(gene_causal_vals))
        inv_score   = invariance_scores.get(gene, 0.0)

        final[gene] = alpha * mean_causal + (1.0 - alpha) * inv_score

    # Normalise to [0, 1]
    max_f = max(final.values()) if final else 1.0
    if max_f > 0:
        final = {g: v / max_f for g, v in final.items()}

    final = dict(sorted(final.items(), key=lambda kv: kv[1], reverse=True))

    print(
        f"[invariance] Final combined scores (alpha={alpha}).\n"
        "  Top-5: "
        + ", ".join(f"{g}({s:.3f})" for g, s in list(final.items())[:5])
    )
    return final


# ---------------------------------------------------------------------------
# Leave-One-Donor-Out (LODO) protocol helper
# ---------------------------------------------------------------------------

def lodo_splits(
    slice_ids: List[str],
    donor_map: Dict[str, str],
) -> List[Tuple[List[str], List[str]]]:
    """
    Generate Leave-One-Donor-Out evaluation splits.

    For DLPFC with 3 donors × 4 slices each, this yields 3 splits:
        Split 0: train on donors {B, C}, test on donor {A}
        Split 1: train on donors {A, C}, test on donor {B}
        Split 2: train on donors {A, B}, test on donor {C}

    Parameters
    ----------
    slice_ids  : all slice identifiers
    donor_map  : {slice_id: donor_id}

    Returns
    -------
    list of (train_slice_ids, test_slice_ids) tuples
    """
    donors       = sorted(set(donor_map.values()))
    splits = []
    for test_donor in donors:
        train_slices = [s for s in slice_ids if donor_map.get(s) != test_donor]
        test_slices  = [s for s in slice_ids if donor_map.get(s) == test_donor]
        splits.append((train_slices, test_slices))
        print(f"  LODO split: test_donor={test_donor}  "
              f"train={train_slices}  test={test_slices}")
    return splits
