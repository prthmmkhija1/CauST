"""
caust/evaluate/metrics.py
==========================
Evaluation metrics for spatial domain identification.

Metrics implemented
--------------------
  ARI  - Adjusted Rand Index          : gold standard for clustering quality
  NMI  - Normalized Mutual Information: complementary to ARI
  Silhouette score                    : no ground truth needed
  Cross-slice ARI                     : the KEY metric for CauST's value

All functions are pure (no side effects) and return plain Python floats.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Adjusted Rand Index (ARI).

    Measures the similarity between two clusterings, adjusting for chance.

    Rules of thumb:
        ARI > 0.65  → excellent
        ARI > 0.40  → good
        ARI < 0.20  → poor
        ARI ≈ 0     → random assignment

    Parameters
    ----------
    labels_true : ground-truth domain labels (e.g. cortical layer annotations)
    labels_pred : predicted domain labels from clustering

    Returns
    -------
    float in [-1, 1]  (negative = worse than random, 1 = perfect)
    """
    return float(adjusted_rand_score(labels_true, labels_pred))


def compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Normalized Mutual Information (NMI).

    Measures how much information two clusterings share, normalized to [0, 1].
    Complementary to ARI — high NMI means the groupings encode similar info.

    Returns
    -------
    float in [0, 1]
    """
    return float(normalized_mutual_info_score(labels_true, labels_pred,
                                               average_method="arithmetic"))


def compute_silhouette(
    Z: np.ndarray,
    labels: np.ndarray,
    subsample: int = 5000,
    random_state: int = 42,
) -> float:
    """
    Silhouette Score.

    Measures how well-separated the clusters are in latent space WITHOUT
    needing ground-truth labels. Useful for datasets with no annotations.

    How it works:
      For each spot, compare:
        a = mean distance to spots in the SAME cluster
        b = mean distance to spots in the NEAREST other cluster
        silhouette = (b - a) / max(a, b)
      Values close to +1 mean the clustering is tight and well-separated.

    Parameters
    ----------
    Z            : latent embedding matrix  (n_spots × latent_dim)
    labels       : cluster assignment per spot
    subsample    : max spots to use (silhouette is expensive on large data)
    random_state : reproducibility seed for subsampling

    Returns
    -------
    float in [-1, 1]
    """
    n = Z.shape[0]
    if n > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=subsample, replace=False)
        Z, labels = Z[idx], labels[idx]

    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return 0.0

    return float(silhouette_score(Z, labels, metric="euclidean"))


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def evaluate_single_slice(
    labels_pred: np.ndarray,
    latent_Z: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute all available metrics for a single slice.

    Parameters
    ----------
    labels_pred  : predicted domain labels
    latent_Z     : latent embedding matrix
    labels_true  : ground-truth labels (if available)
    prefix       : string prefix for metric keys (e.g. 'cross_slice_')

    Returns
    -------
    dict of metric_name → value
    """
    results: Dict[str, float] = {}

    # Always compute silhouette (no ground truth needed)
    results[f"{prefix}silhouette"] = compute_silhouette(latent_Z, labels_pred)

    if labels_true is not None:
        # Drop spots whose ground-truth label is NaN/None (unannotated spots)
        mask = pd.notna(labels_true)
        if mask.all():
            lt_clean = labels_true
            lp_clean = labels_pred
            z_clean  = latent_Z
        else:
            lt_clean = np.asarray(labels_true)[mask]
            lp_clean = np.asarray(labels_pred)[mask]
            z_clean  = np.asarray(latent_Z)[mask]

        if len(np.unique(lt_clean)) >= 2:
            results[f"{prefix}ari"] = compute_ari(lt_clean, lp_clean)
            results[f"{prefix}nmi"] = compute_nmi(lt_clean, lp_clean)
            # Recompute silhouette on annotated spots only when ground truth exists
            results[f"{prefix}silhouette"] = compute_silhouette(z_clean, lp_clean)

    return results


# ---------------------------------------------------------------------------
# Benchmark table builder
# ---------------------------------------------------------------------------

def summarize_results(
    results_list: List[Dict],
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate a list of per-experiment result dicts into a summary DataFrame.

    Each dict should contain (at minimum):
        method, dataset, slice_id, ARI, NMI, silhouette

    Parameters
    ----------
    results_list : list of dicts, one per experiment run
    output_csv   : optional path to save CSV

    Returns
    -------
    pandas DataFrame (pretty-printed automatically)
    """
    df = pd.DataFrame(results_list)

    if not df.empty and output_csv:
        import pathlib
        pathlib.Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[metrics] Results saved → {output_csv}")

    return df


# ---------------------------------------------------------------------------
# Cross-slice generalization helpers
# ---------------------------------------------------------------------------

def compute_cross_slice_ari(
    model_predict_fn,
    test_slices: Dict,
    labels_key: str = "layer",
    n_clusters: int = 7,
) -> Dict[str, float]:
    """
    Evaluate cross-slice ARI on held-out test slices.

    The model was trained on training slices — here we measure how well
    its domain predictions generalize to unseen slices/donors.

    Parameters
    ----------
    model_predict_fn : callable  adata → np.ndarray (predicted labels)
    test_slices      : dict {slice_id: AnnData}  with ground-truth labels
    labels_key       : adata.obs column holding ground-truth domain labels
    n_clusters       : number of clusters for prediction

    Returns
    -------
    dict  {slice_id: cross_slice_ARI}
    """
    cross_ari: Dict[str, float] = {}

    for sid, adata in test_slices.items():
        if labels_key not in adata.obs.columns:
            print(f"  [metrics] Warning: '{labels_key}' not in adata.obs for slice {sid}")
            continue

        labels_true = adata.obs[labels_key].values
        labels_pred = model_predict_fn(adata)

        ari = compute_ari(labels_true, labels_pred)
        cross_ari[sid] = ari
        print(f"  [metrics] Cross-slice ARI  (slice {sid}): {ari:.4f}")

    if cross_ari:
        mean_ari = float(np.mean(list(cross_ari.values())))
        print(f"  [metrics] Mean cross-slice ARI: {mean_ari:.4f}")

    return cross_ari
