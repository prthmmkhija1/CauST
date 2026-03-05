"""
caust/causal/scorer.py
=======================
Compute per-gene causal scores via perturbation-based and gradient-based methods.

Two complementary scoring strategies are implemented:

1. PERTURBATION-BASED (primary, main CauST contribution)
   --------------------------------------------------------
   For every gene g:
     a. Apply do(Gene_g = E[Gene_g]) to get X_perturbed
     b. Run model: get original latent Z and perturbed latent Z'
     c. Cluster both latent spaces → domain labels L, L'
     d. causal_score[g] = 1 - ARI(L, L')
        (higher = more change → more causal influence)
     e. Adjust for global disruption to remove housekeeping-gene bias

2. GRADIENT-BASED (fast approximation, good for initial ranking)
   ---------------------------------------------------------------
   Compute the gradient of the reconstruction loss w.r.t. each gene's
   expression column.  Genes with larger gradients have a bigger
   influence on the model output → likely more causally relevant.
   Much faster (~100× vs perturbation) but less precise.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

from caust.causal.intervention import (
    InterventionMethod,
    apply_intervention,
    compute_global_disruption,
)
from caust.models.autoencoder import SpatialAutoencoder


# ---------------------------------------------------------------------------
# Helper: K-Means clustering of latent space
# ---------------------------------------------------------------------------

def cluster_latent(
    Z: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Cluster the latent space using K-Means.

    K-Means groups the compressed representations into n_clusters spatial
    domains.  Each spot is assigned a cluster label (0 … n_clusters-1).

    Parameters
    ----------
    Z          : latent matrix  (n_spots × latent_dim)
    n_clusters : number of spatial domains expected
    random_state : for reproducibility

    Returns
    -------
    labels  : int array  (n_spots,)
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(Z)


# ---------------------------------------------------------------------------
# Strategy 1: Perturbation-based causal scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perturbation_causal_scores(
    adata,
    model: SpatialAutoencoder,
    edge_index: torch.Tensor,
    n_clusters: int,
    method: InterventionMethod = "mean_impute",
    adjust_for_disruption: bool = True,
    device: str = "cpu",
    gene_indices: Optional[List[int]] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute causal scores for every gene by measuring how much knocking
    each gene out changes the spatial domain assignments.

    Algorithm (for each gene g)
    ----------------------------
    1.  X_orig   → encode → Z_orig   → KMeans → L_orig
    2.  X_perturbed (gene g zeroed/mean-imputed)
                 → encode → Z_perturbed → KMeans → L_perturbed
    3.  raw_score[g]   = 1 − ARI(L_orig, L_perturbed)
    4.  disruption[g]  = mean L2 distance between Z_orig and Z_perturbed
    5.  causal_score[g] = raw_score[g] / (disruption[g] + ε)
        (normalised so housekeeping genes don't dominate)

    Parameters
    ----------
    adata        : preprocessed AnnData (genes are the columns of adata.X)
    model        : trained SpatialAutoencoder
    edge_index   : precomputed edge_index tensor (on device)
    n_clusters   : number of spatial domains for KMeans
    method       : intervention method ('mean_impute' recommended)
    adjust_for_disruption : whether to normalise by latent-space disruption
    device       : 'cpu' or 'cuda'
    gene_indices : subset of gene column indices to score (None = all)
    random_state : reproducibility seed

    Returns
    -------
    dict  {gene_name: causal_score}   sorted descending by score
    """
    model = model.to(device).eval()
    edge_index = edge_index.to(device)

    # Get the expression matrix as a dense float32 array
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    gene_names = list(adata.var_names)
    n_genes = len(gene_names)

    if gene_indices is None:
        gene_indices = list(range(n_genes))

    # ── Compute original latent representation once ───────────────────────
    x_orig_t = torch.FloatTensor(X).to(device)
    Z_orig = model.encode(x_orig_t, edge_index).cpu().numpy()   # (n_spots × latent_dim)
    L_orig = cluster_latent(Z_orig, n_clusters=n_clusters, random_state=random_state)

    raw_scores: Dict[int, float] = {}
    disruptions: Dict[int, float] = {}

    print(f"\n[scorer] Perturbation scoring  —  {len(gene_indices)} genes  "
          f"(method={method}, device={device})")

    for gene_idx in tqdm(gene_indices, desc="Gene interventions", unit="gene"):
        X_perturbed = apply_intervention(X, gene_idx, method=method)
        x_pert_t    = torch.FloatTensor(X_perturbed).to(device)
        Z_pert      = model.encode(x_pert_t, edge_index).cpu().numpy()
        L_pert      = cluster_latent(Z_pert, n_clusters=n_clusters, random_state=random_state)

        ari = adjusted_rand_score(L_orig, L_pert)
        raw_scores[gene_idx]   = max(0.0, 1.0 - ari)   # clip to [0, 1]
        disruptions[gene_idx]  = compute_global_disruption(Z_orig, Z_pert)

    # ── Normalise by global disruption ───────────────────────────────────
    eps = 1e-8
    scores: Dict[str, float] = {}
    for gene_idx in gene_indices:
        gene = gene_names[gene_idx]
        if adjust_for_disruption:
            d = disruptions[gene_idx] + eps
            scores[gene] = raw_scores[gene_idx] / d
        else:
            scores[gene] = raw_scores[gene_idx]

    # Sort descending
    scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    # Normalise to [0, 1]
    max_score = max(scores.values()) if scores else 1.0
    if max_score > 0:
        scores = {g: v / max_score for g, v in scores.items()}

    print(
        f"  Top-5 causal genes: "
        + ", ".join(f"{g}({s:.3f})" for g, s in list(scores.items())[:5])
    )
    return scores


# ---------------------------------------------------------------------------
# Strategy 2: Gradient-based causal scoring (fast approximation)
# ---------------------------------------------------------------------------

def compute_gradient_causal_scores(
    adata,
    model: SpatialAutoencoder,
    edge_index: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Fast causal score approximation using input-gradient attribution.

    For each gene g, compute:
        ∂ Loss / ∂ X[:, g]

    Genes with large mean absolute gradient have a stronger influence on
    the model's reconstruction → likely more causally relevant.

    This is ~100× faster than perturbation scoring and is ideal for:
      - A quick initial ranking to warm-start perturbation scoring
      - Very large gene panels where full perturbation is too slow
      - The HP 15s machine when GPU is not available

    Parameters
    ----------
    adata       : preprocessed AnnData
    model       : trained SpatialAutoencoder
    edge_index  : precomputed edge_index tensor
    device      : 'cpu' or 'cuda'

    Returns
    -------
    dict  {gene_name: gradient_score}  sorted descending
    """
    import torch.nn.functional as F

    model = model.to(device).eval()
    edge_index = edge_index.to(device)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    gene_names = list(adata.var_names)

    x_t = torch.FloatTensor(X).to(device)
    x_t.requires_grad_(True)

    _, x_recon = model(x_t, edge_index)
    loss = F.mse_loss(x_recon, x_t.detach())
    loss.backward()

    # Mean absolute gradient per gene column
    grads = x_t.grad.abs().mean(dim=0).detach().cpu().numpy()  # (n_genes,)

    scores = {gene: float(grads[i]) for i, gene in enumerate(gene_names)}

    # Normalise to [0, 1]
    max_g = max(scores.values()) if scores else 1.0
    if max_g > 0:
        scores = {g: v / max_g for g, v in scores.items()}

    scores = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    print(
        f"[scorer] Gradient scoring done. "
        f"Top-5: " + ", ".join(f"{g}({s:.3f})" for g, s in list(scores.items())[:5])
    )
    return scores
