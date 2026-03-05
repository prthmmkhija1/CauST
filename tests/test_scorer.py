"""
tests/test_scorer.py
=====================
Unit tests for caust.causal.scorer using tiny synthetic data.
Gradient scoring is used here because it is fast (no per-gene forward pass loop).
Run:  pytest tests/test_scorer.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import pytest
import anndata as ad
from scipy.sparse import csr_matrix

from caust.causal.scorer import (
    cluster_latent,
    compute_gradient_causal_scores,
)
from caust.data.graph import build_spatial_graph, adata_to_pyg_data
from caust.models.autoencoder import SpatialAutoencoder, train_autoencoder


def _make_adata(n_obs=40, n_vars=30, seed=42):
    rng = np.random.default_rng(seed)
    X   = rng.integers(1, 20, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=csr_matrix(X))
    adata.obs_names = [f"s{i}" for i in range(n_obs)]
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    adata.obsm["spatial"] = rng.uniform(0, 10, size=(n_obs, 2))
    return adata


@pytest.fixture(scope="module")
def trained_model_and_data():
    adata = _make_adata()
    build_spatial_graph(adata, n_neighbors=4)
    data  = adata_to_pyg_data(adata)
    model = SpatialAutoencoder(in_dim=adata.n_vars, hidden_dim=32, latent_dim=10)
    model, _ = train_autoencoder(model, data, epochs=5)
    return adata, model, data


class TestClusterLatent:
    def test_returns_correct_n_labels(self):
        Z      = np.random.randn(60, 15)
        labels = cluster_latent(Z, n_clusters=3)
        assert labels.shape == (60,)
        assert set(labels.tolist()).issubset({0, 1, 2})

    def test_single_cluster(self):
        Z      = np.random.randn(20, 10)
        labels = cluster_latent(Z, n_clusters=1)
        assert all(l == 0 for l in labels)


class TestGradientCausalScores:
    def test_returns_dict_with_gene_keys(self, trained_model_and_data):
        adata, model, data = trained_model_and_data
        scores = compute_gradient_causal_scores(adata, model, data.edge_index)
        assert isinstance(scores, dict)
        assert len(scores) == adata.n_vars

    def test_all_scores_non_negative(self, trained_model_and_data):
        adata, model, data = trained_model_and_data
        scores = compute_gradient_causal_scores(adata, model, data.edge_index)
        assert all(v >= 0 for v in scores.values())

    def test_scores_normalized(self, trained_model_and_data):
        adata, model, data = trained_model_and_data
        scores = compute_gradient_causal_scores(adata, model, data.edge_index)
        assert max(scores.values()) <= 1.0 + 1e-6

    def test_gene_names_match_adata(self, trained_model_and_data):
        adata, model, data = trained_model_and_data
        scores = compute_gradient_causal_scores(adata, model, data.edge_index)
        assert set(scores.keys()) == set(adata.var_names.tolist())
