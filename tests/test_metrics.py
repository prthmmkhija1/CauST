"""
tests/test_metrics.py
======================
Unit tests for caust.evaluate.metrics.
Run:  pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from caust.evaluate.metrics import (
    compute_ari,
    compute_nmi,
    compute_silhouette,
    evaluate_single_slice,
)


def _perfect_labels(n=60, k=3):
    """Predictions that perfectly match ground truth."""
    labels = np.repeat(np.arange(k), n // k)
    return labels, labels.copy()


def _random_labels(n=60, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, size=n), rng.integers(0, k, size=n)


class TestComputeARI:
    def test_perfect_match(self):
        true, pred = _perfect_labels()
        assert compute_ari(pred, true) == pytest.approx(1.0, abs=1e-6)

    def test_range(self):
        true, pred = _random_labels()
        score = compute_ari(pred, true)
        assert -1.0 <= score <= 1.0

    def test_single_cluster(self):
        # All spots in one cluster vs K ground truth — should be 0
        true = np.array([0, 1, 2, 0, 1, 2])
        pred = np.zeros(6, dtype=int)
        score = compute_ari(pred, true)
        assert score == pytest.approx(0.0, abs=1e-6)


class TestComputeNMI:
    def test_perfect_match(self):
        true, pred = _perfect_labels()
        score = compute_nmi(pred, true)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_range(self):
        true, pred = _random_labels()
        score = compute_nmi(pred, true)
        assert 0.0 <= score <= 1.0 + 1e-6


class TestComputeSilhouette:
    def test_well_separated_clusters(self):
        # 3 tight, well-separated Gaussian blobs
        rng = np.random.default_rng(99)
        Z   = np.vstack([
            rng.normal([0, 0],   0.1, size=(30, 2)),
            rng.normal([10, 0],  0.1, size=(30, 2)),
            rng.normal([5, 10],  0.1, size=(30, 2)),
        ])
        labels = np.repeat([0, 1, 2], 30)
        score  = compute_silhouette(Z, labels)
        assert score > 0.8

    def test_single_cluster_returns_zero(self):
        Z      = np.random.randn(30, 5)
        labels = np.zeros(30, dtype=int)
        score  = compute_silhouette(Z, labels)
        assert score == 0.0


class TestEvaluateSingleSlice:
    def test_with_ground_truth(self):
        rng    = np.random.default_rng(1)
        Z      = rng.normal(size=(60, 10))
        labels = np.repeat([0, 1, 2], 20)
        true   = np.repeat([0, 1, 2], 20)
        m      = evaluate_single_slice(labels, Z, true)
        assert "ari"        in m
        assert "nmi"        in m
        assert "silhouette" in m
        assert m["ari"]     == pytest.approx(1.0, abs=1e-6)

    def test_without_ground_truth(self):
        rng    = np.random.default_rng(2)
        Z      = rng.normal(size=(60, 10))
        labels = np.repeat([0, 1, 2], 20)
        m      = evaluate_single_slice(labels, Z, labels_true=None)
        assert "silhouette" in m
        assert "ari"        not in m or m.get("ari") is None

    def test_prefix_applied(self):
        rng    = np.random.default_rng(3)
        Z      = rng.normal(size=(30, 5))
        labels = np.repeat([0, 1], 15)
        true   = np.repeat([0, 1], 15)
        m      = evaluate_single_slice(labels, Z, true, prefix="test_")
        assert any(k.startswith("test_") for k in m.keys())
