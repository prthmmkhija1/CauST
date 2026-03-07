"""
caust/pipeline.py
==================
Main CauST pipeline — the single class that ties everything together.

Usage (single slice)
--------------------
    from caust import CauST
    import scanpy as sc

    adata = sc.read_h5ad("data/processed/DLPFC_sample1.h5ad")
    model = CauST(n_causal_genes=500, alpha=0.5, n_clusters=7)
    adata_filtered = model.fit_transform(adata)
    print(model.get_top_causal_genes(20))

Usage (multi-slice — recommended for GSoC evaluation)
------------------------------------------------------
    slices = {
        "s1": adata_s1, "s2": adata_s2,
        "s3": adata_s3, "s4": adata_s4,
    }
    model.fit_multi_slice(slices, n_clusters=7)
    adata_out = model.transform(adata_s1)

sklearn-style API:  fit / transform / fit_transform
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

import scanpy as sc

from caust.causal.invariance import combine_causal_and_invariance, compute_invariance_scores
from caust.causal.scorer import (
    cluster_latent,
    compute_gradient_causal_scores,
    compute_perturbation_causal_scores,
)
from caust.data.graph import build_spatial_graph, adata_to_pyg_data, get_edge_index_from_adata
from caust.filter.gene_filter import apply_gene_selection
from caust.models.autoencoder import SpatialAutoencoder, train_autoencoder


class CauST:
    """
    Causal Gene Intervention framework for Spatial Transcriptomics.

    CauST replaces correlation-driven gene usage with causal gene selection
    by simulating in-silico gene interventions and measuring their impact
    on spatial domain assignments.

    Parameters
    ----------
    n_causal_genes  : number of top-causal genes to keep (k for hard filter)
    alpha           : blend weight  0=invariance only, 1=causal strength only
                      0.5 = equal weight (default)
    n_clusters      : number of spatial domains expected in the tissue
    hidden_dim      : encoder hidden layer size
    latent_dim      : latent space dimensionality
    epochs          : autoencoder training epochs
    lr              : learning rate
    n_neighbors     : spatial graph K nearest neighbours
    filter_mode     : 'filter' | 'reweight' | 'filter_and_reweight' (default)
    scoring_method  : 'perturbation'            — accurate, slow (full gene panel)
                      'gradient'                — fast approximation (~100× faster)
                      'gradient+perturbation'   — recommended default: gradient
                        pre-ranks all genes, then perturbation scores only the
                        top perturbation_top_k candidates (~10-20× faster than
                        full perturbation with minimal accuracy loss)
    intervention    : 'mean_impute' | 'zero_out' | 'median_impute'
    device          : 'auto' | 'cpu' | 'cuda'
    random_state    : reproducibility seed
    verbose         : print progress messages
    """

    def __init__(
        self,
        n_causal_genes: int = 500,
        alpha: float = 0.5,
        n_clusters: int = 7,
        hidden_dim: int = 512,
        latent_dim: int = 30,
        epochs: int = 500,
        lr: float = 1e-3,
        n_neighbors: int = 6,
        filter_mode: str = "filter_and_reweight",
        scoring_method : str = "gradient+perturbation",
        intervention: str = "mean_impute",
        device: str = "auto",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.n_causal_genes  = n_causal_genes
        self.alpha           = alpha
        self.n_clusters      = n_clusters
        self.hidden_dim      = hidden_dim
        self.latent_dim      = latent_dim
        self.epochs          = epochs
        self.lr              = lr
        self.n_neighbors     = n_neighbors
        self.filter_mode     = filter_mode
        self.scoring_method  = scoring_method
        self.intervention    = intervention
        self.random_state    = random_state
        self.verbose         = verbose

        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # State — populated after fit()
        self._model:          Optional[SpatialAutoencoder] = None
        self._causal_scores:  Dict[str, float] = {}     # final combined score
        self._loss_history:   List[float] = []
        self._domain_labels:  Optional[np.ndarray] = None
        self._gene_names:     Optional[List[str]] = None
        self._fitted:         bool = False

        if self.verbose:
            print(
                f"\n{'='*55}\n"
                f"  CauST initialized\n"
                f"  device         : {self.device}\n"
                f"  n_causal_genes : {n_causal_genes}\n"
                f"  filter_mode    : {filter_mode}\n"
                f"  scoring_method : {scoring_method}\n"
                f"  intervention   : {intervention}\n"
                f"  alpha          : {alpha}\n"
                f"{'='*55}\n"
            )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(self, adata: sc.AnnData) -> "CauST":
        """
        Train the autoencoder and compute causal scores on a single slice.

        Parameters
        ----------
        adata : preprocessed AnnData with spatial graph already built,
                OR build_spatial_graph will be called automatically.

        Returns
        -------
        self  (for method chaining: model.fit(adata).transform(adata))
        """
        adata = self._ensure_graph(adata)
        self._gene_names = list(adata.var_names)

        # ── Step 1: Train autoencoder ─────────────────────────────────────
        self._log("Step 1/3 — Training spatial autoencoder …")
        pyg_data = adata_to_pyg_data(adata)
        model    = SpatialAutoencoder(
            in_dim     = adata.n_vars,
            hidden_dim = self.hidden_dim,
            latent_dim = self.latent_dim,
        )
        model, self._loss_history = train_autoencoder(
            model, pyg_data,
            epochs  = self.epochs,
            lr      = self.lr,
            device  = self.device,
            verbose = self.verbose,
        )
        self._model = model

        # ── Step 2: Compute causal scores ─────────────────────────────────
        self._log("Step 2/3 — Computing causal gene scores …")
        edge_index  = get_edge_index_from_adata(adata).to(self.device)
        single_slice_scores = self._compute_scores(adata, edge_index)

        # Single-slice: no invariance possible — use causal scores directly
        self._causal_scores = single_slice_scores
        self._domain_labels = cluster_latent(
            self._model.get_latent(
                torch.FloatTensor(
                    _to_dense(adata.X)
                ).to(self.device),
                edge_index,
            ),
            n_clusters    = self.n_clusters,
            random_state  = self.random_state,
        )
        self._fitted = True
        self._log("Step 3/3 — Fitting complete.")
        return self

    def fit_multi_slice(
        self,
        slices: Dict[str, sc.AnnData],
        n_clusters: Optional[int] = None,
        donor_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, sc.AnnData]:
        """
        Train on multiple slices and compute invariance-corrected scores.

        This is the RECOMMENDED mode — more slices → more reliable invariance
        scoring → better gene selection → better generalization.

        For GSoC: use this with DLPFC (12 slices, 3 donors) for best results.

        Parameters
        ----------
        slices     : dict  {slice_id: AnnData}
        n_clusters : override self.n_clusters if provided
        donor_map  : optional dict {slice_id: donor_id} for cross-donor analysis

        Returns
        -------
        dict {slice_id: filtered AnnData} with caust_domain and caust_latent
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters

        n_slices = len(slices)
        self._log(f"Multi-slice fit: {n_slices} slices")

        causal_scores_per_slice: Dict[str, Dict[str, float]] = {}
        per_slice_models: Dict[str, SpatialAutoencoder] = {}

        for i, (sid, adata) in enumerate(slices.items(), 1):
            self._log(f"\n  ── Slice {i}/{n_slices}: {sid} ──")
            adata = self._ensure_graph(adata)
            slices[sid] = adata  # update with graph

            if self._gene_names is None:
                self._gene_names = list(adata.var_names)

            # Train a fresh model per slice
            pyg_data = adata_to_pyg_data(adata)
            model    = SpatialAutoencoder(
                in_dim     = adata.n_vars,
                hidden_dim = self.hidden_dim,
                latent_dim = self.latent_dim,
            )
            model, loss_hist = train_autoencoder(
                model, pyg_data,
                epochs  = self.epochs,
                lr      = self.lr,
                device  = self.device,
                verbose = self.verbose,
            )
            if not self._loss_history:
                self._loss_history = loss_hist

            per_slice_models[sid] = model

            edge_index = get_edge_index_from_adata(adata).to(self.device)
            slice_scores = _run_scoring(
                adata, model, edge_index,
                method       = self.scoring_method,
                intervention = self.intervention,
                n_clusters   = self.n_clusters,
                device       = self.device,
                random_state = self.random_state,
            )
            causal_scores_per_slice[str(sid)] = slice_scores

        # Store per-slice scores for downstream analysis (heatmaps, LODO, etc.)
        self.per_slice_scores = causal_scores_per_slice
        self._donor_map = donor_map

        # ── Invariance scoring ────────────────────────────────────────────
        self._log("\nComputing invariance scores across all slices …")
        inv_scores = compute_invariance_scores(causal_scores_per_slice)
        self._causal_scores = combine_causal_and_invariance(
            causal_scores_per_slice, inv_scores, alpha=self.alpha
        )

        # Cross-donor correlation if donor_map provided
        if donor_map and len(set(donor_map.values())) >= 2:
            from caust.causal.invariance import compute_cross_donor_correlation
            pearson_r, spearman_r = compute_cross_donor_correlation(
                causal_scores_per_slice, donor_map
            )
            self._cross_donor_corr = (pearson_r, spearman_r)

        # Store the last trained model for inference
        self._model = model
        self._fitted = True
        self._log("Multi-slice fitting complete.")

        # ── Transform every slice and return results dict ─────────────────
        results: Dict[str, sc.AnnData] = {}
        for sid, adata in slices.items():
            results[sid] = self.transform(adata)

        return results

    def transform(
        self,
        adata: sc.AnnData,
        inplace: bool = False,
    ) -> sc.AnnData:
        """
        Apply causal gene filtering/reweighting to a (possibly unseen) slice.

        Stores predicted domain labels in adata.obs["caust_domain"].

        Parameters
        ----------
        adata   : preprocessed AnnData (same gene set as training data)
        inplace : operate in place or return a copy

        Returns
        -------
        AnnData with filtered/reweighted genes and domain labels
        """
        self._check_fitted()

        # Apply gene filtering/reweighting for the returned expression matrix
        adata_out = apply_gene_selection(
            adata,
            self._causal_scores,
            mode    = self.filter_mode,
            k       = self.n_causal_genes,
            inplace = inplace,
        )

        # Encode using the FULL original data — the model was trained on all genes
        adata_full = self._ensure_graph(adata)
        edge_index = get_edge_index_from_adata(adata_full).to(self.device)

        import scipy.sparse as sp
        X = adata_full.X
        if sp.issparse(X):
            X = X.toarray()
        x_t   = torch.FloatTensor(X.astype(np.float32)).to(self.device)
        Z     = self._model.get_latent(x_t, edge_index)
        labels = cluster_latent(Z, n_clusters=self.n_clusters,
                                random_state=self.random_state)

        adata_out.obs["caust_domain"]  = labels.astype(str)
        adata_out.obsm["caust_latent"] = Z

        return adata_out

    def fit_transform(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Fit on adata, then transform it.  Equivalent to fit(adata).transform(adata).
        """
        return self.fit(adata).transform(adata)

    # -----------------------------------------------------------------------
    # Inspection helpers
    # -----------------------------------------------------------------------

    def get_top_causal_genes(self, n: int = 20) -> List[tuple]:
        """
        Return the top-N (gene, score) pairs sorted by causal score.

        Example
        -------
            print(model.get_top_causal_genes(10))
            # [('PVALB', 0.943), ('CALB2', 0.912), …]
        """
        self._check_fitted()
        return list(self._causal_scores.items())[:n]

    def get_causal_scores(self) -> Dict[str, float]:
        """Return the full {gene: score} dictionary."""
        self._check_fitted()
        return self._causal_scores.copy()

    def get_domain_labels(self) -> Optional[np.ndarray]:
        """Return domain labels from the last fit/transform call."""
        return self._domain_labels

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, directory: Union[str, Path]) -> None:
        """
        Save the CauST model and causal scores to disk.

        Files created:
            {directory}/caust_model.pt       — PyTorch model weights
            {directory}/causal_scores.json   — gene causal scores
            {directory}/config.json          — hyperparameters
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), directory / "caust_model.pt")

        with open(directory / "causal_scores.json", "w") as f:
            json.dump(self._causal_scores, f, indent=2)

        config = dict(
            n_causal_genes = self.n_causal_genes,
            alpha          = self.alpha,
            n_clusters     = self.n_clusters,
            hidden_dim     = self.hidden_dim,
            latent_dim     = self.latent_dim,
            n_neighbors    = self.n_neighbors,
            filter_mode    = self.filter_mode,
            scoring_method = self.scoring_method,
            intervention   = self.intervention,
            random_state   = self.random_state,
            in_dim         = self._model.in_dim,   # needed for load()
        )
        with open(directory / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"[CauST] Model saved → {directory}")

    @classmethod
    def load(cls, directory: Union[str, Path], n_genes: int | None = None) -> "CauST":
        """
        Load a previously saved CauST model.

        Parameters
        ----------
        directory : path used in save()
        n_genes   : (optional) number of input genes.  Auto-detected from
                     config if omitted.
        """
        directory = Path(directory)

        with open(directory / "config.json") as f:
            config = json.load(f)

        in_dim = n_genes or config.get("in_dim")
        if in_dim is None:
            # Fallback: infer from saved weights
            state = torch.load(directory / "caust_model.pt", map_location="cpu")
            for key in ("encoder.conv1.lin_src.weight", "encoder.conv1.lin.weight"):
                if key in state:
                    in_dim = state[key].shape[1]
                    break
            if in_dim is None:
                raise ValueError(
                    "Cannot determine n_genes.  Pass it explicitly or "
                    "re-save the model with a newer CauST version."
                )

        # Strip in_dim from config so it doesn't get passed to __init__
        config.pop("in_dim", None)
        obj = cls(**config)

        model = SpatialAutoencoder(
            in_dim     = in_dim,
            hidden_dim = config["hidden_dim"],
            latent_dim = config["latent_dim"],
        )
        model.load_state_dict(
            torch.load(directory / "caust_model.pt", map_location="cpu")
        )
        obj._model = model

        with open(directory / "causal_scores.json") as f:
            obj._causal_scores = json.load(f)

        obj._fitted = True
        print(f"[CauST] Model loaded ← {directory}")
        return obj

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _ensure_graph(self, adata: sc.AnnData) -> sc.AnnData:
        """Build spatial graph if not already present."""
        if "spatial_connectivities" not in adata.obsp:
            adata = build_spatial_graph(adata, n_neighbors=self.n_neighbors)
        return adata

    def _compute_scores(
        self,
        adata: sc.AnnData,
        edge_index: torch.Tensor,
    ) -> Dict[str, float]:
        # For gradient+perturbation, score at least 2× the causal-gene budget
        top_k = max(self.n_causal_genes * 2, 1500)
        return _run_scoring(
            adata, self._model, edge_index,
            method              = self.scoring_method,
            intervention        = self.intervention,
            n_clusters          = self.n_clusters,
            device              = self.device,
            random_state        = self.random_state,
            perturbation_top_k  = top_k,
        )

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "CauST model is not fitted yet. Call fit() or fit_multi_slice() first."
            )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Module-level helper (used by both pipeline and run scripts)
# ---------------------------------------------------------------------------

def _run_scoring(
    adata,
    model: SpatialAutoencoder,
    edge_index: torch.Tensor,
    method: str,
    intervention: str,
    n_clusters: int,
    device: str,
    random_state: int,
    perturbation_top_k: int = 1500,
) -> Dict[str, float]:
    """
    Dispatch to the correct scoring strategy.

    Methods
    -------
    'perturbation'           : full perturbation scoring (accurate, slow)
    'gradient'               : gradient attribution (fast, approximate)
    'gradient+perturbation'  : gradient pre-ranks all genes in seconds, then
                               perturbation scores only the top
                               perturbation_top_k candidates.  ~10-20× faster
                               than full perturbation with minimal accuracy loss.
    """
    if method == "perturbation":
        return compute_perturbation_causal_scores(
            adata, model, edge_index,
            n_clusters   = n_clusters,
            method       = intervention,
            device       = device,
            random_state = random_state,
        )
    elif method == "gradient":
        return compute_gradient_causal_scores(adata, model, edge_index, device=device)
    elif method == "gradient+perturbation":
        # Stage 1: fast gradient ranking
        print(f"[scorer] Stage 1/2 — gradient pre-ranking all {adata.n_vars} genes …")
        grad_scores = compute_gradient_causal_scores(
            adata, model, edge_index, device=device
        )
        gene_names = list(adata.var_names)
        top_genes  = list(grad_scores.keys())[:perturbation_top_k]
        top_indices = [gene_names.index(g) for g in top_genes]
        print(
            f"[scorer] Stage 2/2 — perturbation scoring top {len(top_indices)} "
            f"candidate genes …"
        )
        # Stage 2: accurate perturbation on top-K only
        pert_scores = compute_perturbation_causal_scores(
            adata, model, edge_index,
            n_clusters   = n_clusters,
            method       = intervention,
            device       = device,
            random_state = random_state,
            gene_indices = top_indices,
        )
        # Genes not in top-K receive a score of 0
        all_scores = {g: 0.0 for g in gene_names}
        all_scores.update(pert_scores)
        return dict(sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True))
    else:
        raise ValueError(
            f"Unknown scoring_method '{method}'. "
            f"Choose 'perturbation', 'gradient', or 'gradient+perturbation'."
        )


def _to_dense(X) -> np.ndarray:
    """Convert sparse or dense matrix to np.float32 array."""
    import scipy.sparse as sp
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)
