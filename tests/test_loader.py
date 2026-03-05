"""
tests/test_loader.py
====================
Unit tests for caust.data.loader using synthetic AnnData objects.
Run:  pytest tests/test_loader.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import anndata as ad
from scipy.sparse import csr_matrix

from caust.data.loader import load_and_preprocess, load_multiple_slices


def _make_adata(n_obs=80, n_vars=200, seed=0):
    """Create a minimal synthetic AnnData with spatial coords."""
    rng = np.random.default_rng(seed)
    X   = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    obs_names = [f"spot_{i}" for i in range(n_obs)]
    var_names = [f"gene_{j}" for j in range(n_vars)]
    adata = ad.AnnData(X=csr_matrix(X),
                       obs={"cell_type": ["A"] * (n_obs // 2) + ["B"] * (n_obs // 2)},
                       var={"gene_ids": var_names})
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(n_obs, 2))
    return adata


class TestLoadAndPreprocess:
    def test_output_shape_preserved(self):
        adata = _make_adata(n_obs=80, n_vars=200)
        out   = load_and_preprocess(adata, n_top_genes=50, min_cells=1, min_genes=1)
        assert out.n_obs > 0
        assert out.n_vars <= 50

    def test_layers_created(self):
        adata = _make_adata()
        out   = load_and_preprocess(adata, n_top_genes=50, min_cells=1, min_genes=1)
        assert "counts"   in out.layers
        assert "log_norm" in out.layers

    def test_highly_variable_flag(self):
        adata = _make_adata(n_vars=200)
        out   = load_and_preprocess(adata, n_top_genes=30, min_cells=1, min_genes=1)
        assert "highly_variable" in out.var.columns


class TestLoadMultipleSlices:
    def test_returns_dict(self):
        a1 = _make_adata(seed=1)
        a2 = _make_adata(seed=2)
        result = load_multiple_slices({"s1": a1, "s2": a2},
                                       n_top_genes=50, min_cells=1, min_genes=1)
        assert set(result.keys()) == {"s1", "s2"}
        assert all(isinstance(v, ad.AnnData) for v in result.values())

    def test_slice_ids_preserved(self):
        a1 = _make_adata(seed=3)
        result = load_multiple_slices({"myslice": a1},
                                       n_top_genes=50, min_cells=1, min_genes=1)
        assert "myslice" in result
