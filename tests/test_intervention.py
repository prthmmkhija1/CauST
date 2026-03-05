"""
tests/test_intervention.py
===========================
Unit tests for caust.causal.intervention.
Run:  pytest tests/test_intervention.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import pytest

from caust.causal.intervention import (
    apply_intervention,
    apply_batch_interventions,
    compute_global_disruption,
)


def _rand_X(n=50, g=100, seed=7):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(0, 5, size=(n, g)), dtype=torch.float32)


class TestApplyIntervention:
    def test_mean_impute_shape(self):
        X   = _rand_X()
        out = apply_intervention(X.clone(), gene_idx=5, method="mean_impute")
        assert out.shape == X.shape

    def test_mean_impute_value(self):
        X   = _rand_X()
        col_mean = X[:, 5].mean().item()
        out = apply_intervention(X.clone(), gene_idx=5, method="mean_impute")
        assert torch.allclose(out[:, 5], torch.full((X.shape[0],), col_mean), atol=1e-4)

    def test_zero_out_value(self):
        X   = _rand_X()
        out = apply_intervention(X.clone(), gene_idx=3, method="zero_out")
        assert torch.all(out[:, 3] == 0.0)

    def test_median_impute_value(self):
        X   = _rand_X()
        col_med = X[:, 10].median().item()
        out = apply_intervention(X.clone(), gene_idx=10, method="median_impute")
        assert torch.allclose(out[:, 10], torch.full((X.shape[0],), col_med), atol=1e-4)

    def test_other_columns_unchanged(self):
        X   = _rand_X()
        out = apply_intervention(X.clone(), gene_idx=0, method="zero_out")
        assert torch.allclose(X[:, 1:], out[:, 1:])

    def test_invalid_method_raises(self):
        X = _rand_X()
        with pytest.raises((ValueError, KeyError)):
            apply_intervention(X.clone(), gene_idx=0, method="magic_method")


class TestApplyBatchInterventions:
    def test_multiple_genes_zeroed(self):
        X    = _rand_X()
        idxs = [0, 5, 10]
        out  = apply_batch_interventions(X.clone(), idxs, method="zero_out")
        for i in idxs:
            assert torch.all(out[:, i] == 0.0)
        # untouched genes
        untouched = list(set(range(X.shape[1])) - set(idxs))
        assert torch.allclose(X[:, untouched], out[:, untouched])

    def test_empty_index_list(self):
        X   = _rand_X()
        out = apply_batch_interventions(X.clone(), [], method="zero_out")
        assert torch.allclose(X, out)


class TestComputeGlobalDisruption:
    def test_same_tensor_zero(self):
        Z = torch.randn(50, 30)
        d = compute_global_disruption(Z, Z)
        assert abs(d) < 1e-5

    def test_different_tensor_positive(self):
        Z1 = torch.randn(50, 30)
        Z2 = torch.randn(50, 30)
        d  = compute_global_disruption(Z1, Z2)
        assert d > 0
