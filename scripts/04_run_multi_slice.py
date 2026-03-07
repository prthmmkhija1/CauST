"""
scripts/04_run_multi_slice.py
==============================
ONE-SHOT script: train CauST across all 12 DLPFC slices (multi-slice mode).

Run once. All outputs auto-saved.

    python scripts/04_run_multi_slice.py

What it does
------------
  1.  Loads all available preprocessed DLPFC slices
  2.  Computes per-slice perturbation causal scores
  3.  Computes cross-slice IRM-style invariance scores
  4.  Runs LODO (Leave-One-Donor-Out) validation protocol
  5.  Saves per-slice and aggregate metrics to CSV
  6.  Generates heatmap + bar-chart figures

Donor map follows the standard 3-donor DLPFC structure:
  Donor 1: 151507, 151508, 151509, 151510
  Donor 2: 151669, 151670, 151671, 151672
  Donor 3: 151673, 151674, 151675, 151676
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── CONFIG ────────────────────────────────────────────────────────────────
N_CAUSAL_GENES   = 500
N_CLUSTERS       = 7
EPOCHS           = 500
ALPHA            = 0.5
SCORING_METHOD   = "gradient+perturbation"
FILTER_MODE      = "filter_and_reweight"
GROUND_TRUTH_KEY = "layer_guess"

DONOR_MAP = {
    "151507": "Donor1", "151508": "Donor1",
    "151509": "Donor1", "151510": "Donor1",
    "151669": "Donor2", "151670": "Donor2",
    "151671": "Donor2", "151672": "Donor2",
    "151673": "Donor3", "151674": "Donor3",
    "151675": "Donor3", "151676": "Donor3",
}

OUTPUT_DIR = ROOT / "experiments" / "results" / "multi_slice"
# ─────────────────────────────────────────────────────────────────────────

import scanpy as sc
import pandas as pd
from caust import CauST
from caust.causal.invariance import (
    compute_invariance_scores,
    combine_causal_and_invariance,
    lodo_splits,
)
from caust.evaluate.metrics import evaluate_single_slice, compute_cross_slice_ari
from caust.visualize.plots import plot_invariance_heatmap, plot_benchmark_results


def main():
    print("=" * 55)
    print("  CauST — Multi-Slice DLPFC Training")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = ROOT / "data" / "processed" / "DLPFC"
    available = sorted(p.stem for p in data_dir.glob("*.h5ad")) if data_dir.exists() else []

    if not available:
        sys.exit(
            f"\n[ERROR] No processed DLPFC slices found in:\n  {data_dir}\n"
            "Run scripts/02_preprocess.py first."
        )

    print(f"\nFound {len(available)} slices: {available}\n")

    # ── 1. Load all slices ────────────────────────────────────────────────
    slices_dict = {}
    for sid in available:
        path = data_dir / f"{sid}.h5ad"
        adata = sc.read_h5ad(path)
        print(f"  Loaded {sid}: {adata.n_obs} spots × {adata.n_vars} genes")
        slices_dict[sid] = adata

    # ── 2. Train CauST (multi-slice) ──────────────────────────────────────
    model = CauST(
        n_causal_genes  = N_CAUSAL_GENES,
        alpha           = ALPHA,
        n_clusters      = N_CLUSTERS,
        epochs          = EPOCHS,
        scoring_method  = SCORING_METHOD,
        filter_mode     = FILTER_MODE,
        verbose         = True,
    )

    print("\n── Running multi-slice fit ──")
    results = model.fit_multi_slice(slices_dict, donor_map=DONOR_MAP)

    # results is a dict: slice_id -> filtered AnnData
    # model.per_slice_scores stores raw causal dicts

    # ── 3. Per-slice metrics ──────────────────────────────────────────────
    all_metrics = []
    for sid, adata_f in results.items():
        labels_pred = adata_f.obs["caust_domain"].astype(int).values
        latent_Z    = adata_f.obsm["caust_latent"]
        labels_true = None
        adata_raw   = slices_dict[sid]
        if GROUND_TRUTH_KEY and GROUND_TRUTH_KEY in adata_raw.obs.columns:
            labels_true = adata_raw.obs.loc[adata_f.obs_names, GROUND_TRUTH_KEY].values

        m = evaluate_single_slice(labels_pred, latent_Z, labels_true, prefix="")
        m["slice"] = sid
        m["donor"] = DONOR_MAP.get(sid, "Unknown")
        all_metrics.append(m)
        print(f"  {sid}: ARI={m.get('ari', float('nan')):.4f}  "
              f"NMI={m.get('nmi', float('nan')):.4f}  "
              f"Sil={m.get('silhouette', float('nan')):.4f}")

    df = pd.DataFrame(all_metrics)
    df.to_csv(OUTPUT_DIR / "per_slice_metrics.csv", index=False)
    print(f"\n  Per-slice metrics saved.")

    # ── 4. Invariance heatmap ─────────────────────────────────────────────
    if hasattr(model, "per_slice_scores") and len(model.per_slice_scores) > 1:
        inv_scores = compute_invariance_scores(model.per_slice_scores)
        inv_path   = str(OUTPUT_DIR / "invariance_heatmap.png")
        try:
            plot_invariance_heatmap(
                model.per_slice_scores,
                title    = "Cross-Slice Causal Score Invariance (DLPFC)",
                out_path = inv_path,
            )
            print(f"  Invariance heatmap saved.")
        except Exception as e:
            print(f"  [WARN] Heatmap failed: {e}")

        with open(OUTPUT_DIR / "invariance_scores.json", "w") as f:
            json.dump({k: float(v) for k, v in inv_scores.items()}, f, indent=2)

    # ── 5. LODO cross-validation ──────────────────────────────────────────
    lodo_results = []
    slice_ids    = list(slices_dict.keys())
    donor_present = {s: DONOR_MAP[s] for s in slice_ids if s in DONOR_MAP}

    if len(set(donor_present.values())) >= 2 and hasattr(model, "per_slice_scores"):
        print("\n── LODO validation ──")
        for train_ids, test_ids in lodo_splits(slice_ids, donor_present):
            train_scores = {s: model.per_slice_scores[s]
                            for s in train_ids if s in model.per_slice_scores}
            if not train_scores:
                continue

            inv = compute_invariance_scores(train_scores)
            test_donor = donor_present.get(test_ids[0], "Unknown") if test_ids else "Unknown"
            for sid in test_ids:
                if sid not in results:
                    continue
                adata_f = results[sid]
                labels_pred = adata_f.obs["caust_domain"].astype(int).values
                latent_Z    = adata_f.obsm["caust_latent"]
                labels_true = None
                if GROUND_TRUTH_KEY and GROUND_TRUTH_KEY in slices_dict[sid].obs.columns:
                    obs_names   = adata_f.obs_names
                    labels_true = slices_dict[sid].obs.loc[obs_names, GROUND_TRUTH_KEY].values
                m = evaluate_single_slice(labels_pred, latent_Z, labels_true, prefix="lodo_")
                m["test_slice"]  = sid
                m["test_donor"]  = test_donor
                lodo_results.append(m)
                print(f"  LODO test={sid} donor={test_donor}  "
                      f"ARI={m.get('lodo_ari', float('nan')):.4f}")

        if lodo_results:
            pd.DataFrame(lodo_results).to_csv(
                OUTPUT_DIR / "lodo_metrics.csv", index=False
            )
            print("  LODO metrics saved.")

        # Cross-slice ARI via model.transform
        if GROUND_TRUTH_KEY:
            test_all = {sid: slices_dict[sid] for sid in slice_ids
                        if GROUND_TRUTH_KEY in slices_dict[sid].obs.columns}
            if test_all:
                def predict_fn(adata):
                    out = model.transform(adata.copy())
                    return out.obs["caust_domain"].astype(int).values
                cross_ari = compute_cross_slice_ari(
                    predict_fn, test_all, labels_key=GROUND_TRUTH_KEY
                )
                with open(OUTPUT_DIR / "cross_slice_ari.json", "w") as f:
                    json.dump({k: float(v) for k, v in cross_ari.items()}, f, indent=2)
                print("  Cross-slice ARI saved.")

    # ── 6. Aggregate summary ──────────────────────────────────────────────
    summary = {
        "n_slices"        : len(available),
        "mean_ari"        : float(df["ari"].mean())        if "ari"        in df else None,
        "mean_nmi"        : float(df["nmi"].mean())        if "nmi"        in df else None,
        "mean_silhouette" : float(df["silhouette"].mean()) if "silhouette" in df else None,
    }
    with open(OUTPUT_DIR / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Aggregate Summary ──")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
