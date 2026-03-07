"""
scripts/03_train_single_slice.py
=================================
ONE-SHOT script: train CauST on a single DLPFC slice and save everything.

Run once. All outputs (model, scores, figures) auto-saved.

    python scripts/03_train_single_slice.py

What it does
------------
  1.  Loads preprocessed DLPFC slice 151507
  2.  Builds spatial KNN graph
  3.  Trains Graph Attention Autoencoder (500 epochs)
  4.  Runs perturbation-based causal scoring on all genes
  5.  Applies filter+reweight (top 500 causal genes)
  6.  Clusters latent space → spatial domains
  7.  Evaluates with ARI / NMI (if ground truth available)
  8.  Saves model, scores, and 3 plots

Edit CONFIG below to change slice, number of genes, etc.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── CONFIG ────────────────────────────────────────────────────────────────
# Auto-detect the first available DLPFC slice if the default is missing
_DLPFC_DIR     = ROOT / "data" / "processed" / "DLPFC"
_default_slice = "151507"
if not (_DLPFC_DIR / f"{_default_slice}.h5ad").exists():
    _avail = sorted(p.stem for p in _DLPFC_DIR.glob("*.h5ad")) if _DLPFC_DIR.exists() else []
    _default_slice = _avail[0] if _avail else _default_slice

SLICE_ID       = _default_slice              # which DLPFC slice to use
N_CAUSAL_GENES = 500                         # genes to keep after filtering
N_CLUSTERS     = 7                           # cortical layers in DLPFC
EPOCHS         = 500
LR             = 1e-3
SCORING_METHOD = "gradient+perturbation"   # gradient pre-rank + perturbation top candidates
FILTER_MODE    = "filter_and_reweight"       # the strongest CauST setting
ALPHA          = 0.5
GROUND_TRUTH_KEY = "layer_guess"             # obs column with known labels
                                             # set to None if unavailable
OUTPUT_DIR     = ROOT / "experiments" / "results" / "single_slice"
MODEL_SAVE_DIR = ROOT / "experiments" / "models" / SLICE_ID
# ─────────────────────────────────────────────────────────────────────────

import scanpy as sc
from caust import CauST
from caust.evaluate.metrics import evaluate_single_slice
from caust.visualize.plots import (
    plot_causal_scores,
    plot_spatial_domains,
    plot_training_loss,
)


def main():
    print("=" * 55)
    print(f"  CauST — Single-Slice Training  (slice {SLICE_ID})")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    data_path = ROOT / "data" / "processed" / "DLPFC" / f"{SLICE_ID}.h5ad"
    if not data_path.exists():
        sys.exit(
            f"\n[ERROR] Preprocessed file not found:\n  {data_path}\n"
            "Run scripts/02_preprocess.py first."
        )

    adata = sc.read_h5ad(data_path)
    print(f"\nLoaded: {adata.n_obs} spots × {adata.n_vars} genes")

    # ── 2. Train CauST ────────────────────────────────────────────────────
    model = CauST(
        n_causal_genes  = N_CAUSAL_GENES,
        alpha           = ALPHA,
        n_clusters      = N_CLUSTERS,
        epochs          = EPOCHS,
        lr              = LR,
        filter_mode     = FILTER_MODE,
        scoring_method  = SCORING_METHOD,
        verbose         = True,
    )

    adata_filtered = model.fit_transform(adata)

    # ── 3. Evaluate ───────────────────────────────────────────────────────
    import numpy as np
    labels_pred = adata_filtered.obs["caust_domain"].astype(int).values
    latent_Z    = adata_filtered.obsm["caust_latent"]

    labels_true = None
    if GROUND_TRUTH_KEY and GROUND_TRUTH_KEY in adata.obs.columns:
        labels_true = adata.obs.loc[adata_filtered.obs_names, GROUND_TRUTH_KEY].values
        print(f"\n[eval] Ground-truth labels found: '{GROUND_TRUTH_KEY}'")

    metrics = evaluate_single_slice(labels_pred, latent_Z, labels_true, prefix="")
    print("\n── Evaluation Metrics ──")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics
    import json
    with open(OUTPUT_DIR / f"{SLICE_ID}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 4. Save model and scores ──────────────────────────────────────────
    model.save(MODEL_SAVE_DIR)

    top_genes = model.get_top_causal_genes(n=20)
    print("\n── Top-20 Causal Genes ──")
    for gene, score in top_genes:
        print(f"  {gene:<12} {score:.4f}")

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("\n── Generating Plots ──")

    plot_training_loss(
        model._loss_history,
        out_path = str(OUTPUT_DIR / f"{SLICE_ID}_training_loss.png"),
    )

    plot_causal_scores(
        model.get_causal_scores(),
        top_k    = 30,
        title    = f"Top-30 Causal Genes — DLPFC {SLICE_ID}",
        out_path = str(OUTPUT_DIR / f"{SLICE_ID}_causal_scores.png"),
    )

    if "spatial" in adata_filtered.obsm:
        plot_spatial_domains(
            adata_filtered,
            label_key = "caust_domain",
            title     = f"CauST Spatial Domains — DLPFC {SLICE_ID}",
            out_path  = str(OUTPUT_DIR / f"{SLICE_ID}_spatial_domains.png"),
        )

    print(f"\n✓ All outputs saved to:  {OUTPUT_DIR}")
    print(f"✓ Model saved to:        {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()
