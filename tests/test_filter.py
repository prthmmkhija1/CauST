"""
tests/test_filter.py
=====================
Unit tests for caust.filter.gene_filter.
Run:  pytest tests/test_filter.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import scanpy as sc

from caust.filter.gene_filter import (
    filter_top_k,
    reweight_genes,
    filter_and_reweight,
    apply_gene_selection,
)


def _make_adata(n_obs=50, n_vars=200, seed=42):
    """Create a small synthetic AnnData for testing."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 5, size=(n_obs, n_vars)).astype(np.float32)
    gene_names = [f"gene_{i}" for i in range(n_vars)]
    adata = sc.AnnData(X=X)
    adata.var_names = gene_names
    adata.obs_names = [f"spot_{i}" for i in range(n_obs)]
    return adata


def _make_scores(n_vars=200, seed=42):
    """Create synthetic causal scores for all genes."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0, 1, size=n_vars).astype(float)
    return {f"gene_{i}": float(scores[i]) for i in range(n_vars)}


# ── filter_top_k ──────────────────────────────────────────────────────────

class TestFilterTopK:
    def test_shape(self):
        adata = _make_adata()
        scores = _make_scores()
        out = filter_top_k(adata, scores, k=50)
        assert out.n_vars == 50
        assert out.n_obs == 50

    def test_returns_copy(self):
        adata = _make_adata()
        scores = _make_scores()
        out = filter_top_k(adata, scores, k=50, inplace=False)
        assert out is not adata

    def test_top_k_genes_selected(self):
        adata = _make_adata()
        scores = _make_scores()
        k = 30
        out = filter_top_k(adata, scores, k=k)
        top_genes = sorted(scores, key=scores.get, reverse=True)[:k]
        assert set(out.var_names) == set(top_genes)

    def test_causal_score_stored(self):
        adata = _make_adata()
        scores = _make_scores()
        out = filter_top_k(adata, scores, k=20)
        assert "causal_score" in out.var.columns
        for gene in out.var_names:
            assert np.isclose(out.var.loc[gene, "causal_score"], scores[gene], atol=1e-5)

    def test_score_threshold(self):
        adata = _make_adata()
        scores = _make_scores()
        # Set a high threshold so fewer genes pass
        out = filter_top_k(adata, scores, k=200, score_threshold=0.8)
        for gene in out.var_names:
            assert scores[gene] >= 0.8

    def test_k_larger_than_eligible(self):
        adata = _make_adata(n_vars=10)
        scores = _make_scores(n_vars=10)
        out = filter_top_k(adata, scores, k=100)
        assert out.n_vars == 10  # can't return more genes than exist

    def test_no_matching_genes_raises(self):
        adata = _make_adata()
        bad_scores = {"nonexistent_gene": 1.0}
        with pytest.raises(ValueError, match="No genes"):
            filter_top_k(adata, bad_scores, k=10)


# ── reweight_genes ────────────────────────────────────────────────────────

class TestReweightGenes:
    def test_shape_preserved(self):
        adata = _make_adata()
        scores = _make_scores()
        out = reweight_genes(adata, scores)
        assert out.shape == adata.shape

    def test_returns_copy(self):
        adata = _make_adata()
        scores = _make_scores()
        out = reweight_genes(adata, scores, inplace=False)
        assert out is not adata

    def test_reweighting_values(self):
        adata = _make_adata()
        scores = _make_scores()
        original_X = adata.X.copy()
        out = reweight_genes(adata, scores)
        for i, gene in enumerate(out.var_names):
            expected = original_X[:, i] * scores[gene]
            np.testing.assert_allclose(out.X[:, i], expected, atol=1e-5)

    def test_zero_score_gene_silenced(self):
        adata = _make_adata()
        scores = _make_scores()
        scores["gene_0"] = 0.0
        out = reweight_genes(adata, scores)
        assert np.all(out.X[:, 0] == 0.0)

    def test_causal_score_stored(self):
        adata = _make_adata()
        scores = _make_scores()
        out = reweight_genes(adata, scores)
        assert "causal_score" in out.var.columns


# ── filter_and_reweight ───────────────────────────────────────────────────

class TestFilterAndReweight:
    def test_shape(self):
        adata = _make_adata()
        scores = _make_scores()
        out = filter_and_reweight(adata, scores, k=50)
        assert out.n_vars == 50

    def test_genes_are_top_k(self):
        adata = _make_adata()
        scores = _make_scores()
        k = 40
        out = filter_and_reweight(adata, scores, k=k)
        top_genes = sorted(scores, key=scores.get, reverse=True)[:k]
        assert set(out.var_names) == set(top_genes)

    def test_expression_reweighted(self):
        adata = _make_adata()
        scores = _make_scores()
        k = 30
        out = filter_and_reweight(adata, scores, k=k)
        # After filter+reweight, each gene column = original_value * score
        top_genes = sorted(scores, key=scores.get, reverse=True)[:k]
        original = _make_adata()
        for gene in top_genes:
            orig_idx = list(original.var_names).index(gene)
            out_idx = list(out.var_names).index(gene)
            expected = original.X[:, orig_idx] * scores[gene]
            np.testing.assert_allclose(out.X[:, out_idx], expected, atol=1e-5)


# ── apply_gene_selection (unified entry point) ────────────────────────────

class TestApplyGeneSelection:
    def test_filter_mode(self):
        adata = _make_adata()
        scores = _make_scores()
        out = apply_gene_selection(adata, scores, mode="filter", k=50)
        assert out.n_vars == 50

    def test_reweight_mode(self):
        adata = _make_adata()
        scores = _make_scores()
        out = apply_gene_selection(adata, scores, mode="reweight")
        assert out.n_vars == 200  # all genes kept

    def test_filter_and_reweight_mode(self):
        adata = _make_adata()
        scores = _make_scores()
        out = apply_gene_selection(adata, scores, mode="filter_and_reweight", k=50)
        assert out.n_vars == 50

    def test_invalid_mode_raises(self):
        adata = _make_adata()
        scores = _make_scores()
        with pytest.raises(ValueError, match="Unknown mode"):
            apply_gene_selection(adata, scores, mode="invalid_mode")
