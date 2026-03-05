"""
tests/test_pipeline.py
=======================
End-to-end integration test for the CauST pipeline on synthetic data.
Uses gradient scoring (fast) to keep CI runtime short.
Run:  pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import anndata as ad
from scipy.sparse import csr_matrix

from caust import CauST


def _make_adata(n_obs=60, n_vars=80, seed=0):
    rng = np.random.default_rng(seed)
    # 3 spatially separated clusters with different expression profiles
    X_parts = [
        rng.integers(0, 10, size=(n_obs // 3, n_vars)).astype(np.float32),
        rng.integers(10, 20, size=(n_obs // 3, n_vars)).astype(np.float32),
        rng.integers(20, 30, size=(n_obs // 3, n_vars)).astype(np.float32),
    ]
    X = np.vstack(X_parts)
    adata = ad.AnnData(X=csr_matrix(X))
    adata.obs_names = [f"s{i}" for i in range(n_obs)]
    adata.var_names = [f"g{j}" for j in range(n_vars)]
    # Spatial coordinates: 3 clusters positioned far apart
    coords = np.vstack([
        rng.normal([0,  0],  1, size=(n_obs // 3, 2)),
        rng.normal([20, 0],  1, size=(n_obs // 3, 2)),
        rng.normal([10, 17], 1, size=(n_obs // 3, 2)),
    ])
    adata.obsm["spatial"] = coords
    return adata


@pytest.fixture(scope="module")
def fitted_caust():
    adata = _make_adata()
    model = CauST(
        n_causal_genes  = 20,
        n_clusters      = 3,
        epochs          = 10,
        scoring_method  = "gradient",   # fast for tests
        filter_mode     = "filter_and_reweight",
        verbose         = False,
    )
    return model, adata


class TestCauSTFitTransform:
    def test_fit_transform_returns_adata(self, fitted_caust):
        model, adata = fitted_caust
        out = model.fit_transform(adata.copy())
        assert isinstance(out, ad.AnnData)

    def test_domain_labels_assigned(self, fitted_caust):
        model, adata = fitted_caust
        out = model.fit_transform(adata.copy())
        assert "caust_domain" in out.obs.columns
        labels = out.obs["caust_domain"].values
        assert len(labels) == out.n_obs

    def test_latent_embedding_shape(self, fitted_caust):
        model, adata = fitted_caust
        out = model.fit_transform(adata.copy())
        assert "caust_latent" in out.obsm
        assert out.obsm["caust_latent"].shape[0] == out.n_obs

    def test_filtered_gene_count(self, fitted_caust):
        model, adata = fitted_caust
        out = model.fit_transform(adata.copy())
        assert out.n_vars <= model.n_causal_genes

    def test_causal_scores_available(self, fitted_caust):
        model, adata = fitted_caust
        model.fit_transform(adata.copy())           # ensure scores computed
        scores = model.get_causal_scores()
        assert isinstance(scores, dict)
        assert len(scores) > 0

    def test_top_causal_genes_n(self, fitted_caust):
        model, adata = fitted_caust
        model.fit_transform(adata.copy())
        top = model.get_top_causal_genes(n=10)
        assert len(top) == 10
        # check descending order
        vals = [v for _, v in top]
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


class TestCauSTSaveLoad:
    def test_save_and_reload(self, tmp_path, fitted_caust):
        model, adata = fitted_caust
        model.fit_transform(adata.copy())
        save_dir = tmp_path / "model"
        model.save(str(save_dir))
        loaded = CauST.load(str(save_dir))
        assert loaded is not None
        # Can call transform on fresh adata without crashing
        out = loaded.transform(adata.copy())
        assert "caust_latent" in out.obsm


class TestMultiSlice:
    def test_fit_multi_slice_returns_dict(self):
        slices = {
            "A": _make_adata(60, 80, seed=0),
            "B": _make_adata(60, 80, seed=1),
        }
        model = CauST(
            n_causal_genes=20, n_clusters=3, epochs=5,
            scoring_method="gradient", verbose=False,
        )
        results = model.fit_multi_slice(slices, n_clusters=3)
        assert isinstance(results, dict)
        assert set(results.keys()) == {"A", "B"}
        for adata_out in results.values():
            assert isinstance(adata_out, ad.AnnData)
            assert "caust_domain" in adata_out.obs.columns
            assert "caust_latent" in adata_out.obsm

    def test_per_slice_scores_stored(self):
        slices = {
            "X": _make_adata(60, 80, seed=2),
            "Y": _make_adata(60, 80, seed=3),
        }
        model = CauST(
            n_causal_genes=20, n_clusters=3, epochs=5,
            scoring_method="gradient", verbose=False,
        )
        model.fit_multi_slice(slices)
        assert hasattr(model, "per_slice_scores")
        assert "X" in model.per_slice_scores
        assert "Y" in model.per_slice_scores

    def test_donor_map_cross_correlation(self):
        slices = {
            "S1": _make_adata(60, 80, seed=4),
            "S2": _make_adata(60, 80, seed=5),
        }
        donor_map = {"S1": "D1", "S2": "D2"}
        model = CauST(
            n_causal_genes=20, n_clusters=3, epochs=5,
            scoring_method="gradient", verbose=False,
        )
        model.fit_multi_slice(slices, donor_map=donor_map)
        assert hasattr(model, "_cross_donor_corr")
