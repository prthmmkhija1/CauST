"""
scripts/05_benchmark.py
========================
ONE-SHOT script: full ablation + benchmark across datasets and methods.

Run once. All outputs auto-saved.

    python scripts/05_benchmark.py

Conditions tested
-----------------
Variants:
  - Baseline        : raw HVG features, no causal filtering
  - CauST-Filter    : top-K gene filter only
  - CauST-Reweight  : gene reweighting only
  - CauST-Full      : filter + reweight (the default CauST setting)

Downstream methods:
  - CauST-Internal  : CauST's own GAT clustering
  - STAGATE         : if stagate_pyg is installed
  - GraphST         : if GraphST is installed

Datasets:
  - DLPFC 151507  (and more slices if present)
  - MouseBrain    (if processed)

Results saved as:  experiments/results/benchmark/all_results.csv
Figures saved as:  experiments/results/benchmark/benchmark_plot.png
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── CONFIG ────────────────────────────────────────────────────────────────
N_CAUSAL_GENES   = 500
N_CLUSTERS_DLPFC = 7
N_CLUSTERS_MOUSE = 10
EPOCHS           = 500
SCORING_METHOD   = "perturbation"
GROUND_TRUTH_KEY = "layer_guess"   # DLPFC; set to None if unavailable
OUTPUT_DIR       = ROOT / "experiments" / "results" / "benchmark"
# ─────────────────────────────────────────────────────────────────────────

import scanpy as sc
import pandas as pd
from caust import CauST
from caust.causal.scorer import compute_perturbation_causal_scores
from caust.data.graph import build_spatial_graph, adata_to_pyg_data
from caust.filter.gene_filter import filter_and_reweight, filter_top_k, reweight_genes
from caust.models.autoencoder import SpatialAutoencoder, train_autoencoder
from caust.evaluate.metrics import evaluate_single_slice
from caust.visualize.plots import plot_benchmark_results


def run_baseline(adata, n_clusters, device="cpu"):
    """Raw HVG features only — no causal intervention."""
    from caust.data.graph import build_spatial_graph, adata_to_pyg_data
    from caust.models.autoencoder import SpatialAutoencoder, train_autoencoder
    from caust.causal.scorer import cluster_latent
    build_spatial_graph(adata)
    data = adata_to_pyg_data(adata)
    data = data.to(device)
    model = SpatialAutoencoder(data.x.shape[1]).to(device)
    model, _ = train_autoencoder(model, data, epochs=EPOCHS)
    Z = model.get_latent(data).cpu().numpy()
    labels = cluster_latent(Z, n_clusters)
    return Z, labels


def run_caust_variant(adata, n_clusters, filter_mode, device="cpu"):
    """Run CauST with a specific filter_mode."""
    model = CauST(
        n_causal_genes = N_CAUSAL_GENES,
        n_clusters     = n_clusters,
        epochs         = EPOCHS,
        filter_mode    = filter_mode,
        scoring_method = SCORING_METHOD,
        verbose        = False,
    )
    adata_out = model.fit_transform(adata.copy())
    Z      = adata_out.obsm["caust_latent"]
    labels = adata_out.obs["caust_domain"].astype(int).values
    return Z, labels


def try_stagate(adata, device="cpu"):
    """Try STAGATE if installed."""
    try:
        from caust.models.stagate_wrapper import run_with_stagate
        adata_out = run_with_stagate(adata.copy(), device=device)
        Z      = adata_out.obsm.get("STAGATE", None)
        labels_raw = adata_out.obs.get("mclust", None)
        if Z is None or labels_raw is None:
            return None, None
        labels = labels_raw.astype(int).values
        return Z, labels
    except (ImportError, Exception) as e:
        print(f"    [STAGATE skip] {e}")
        return None, None


def try_graphst(adata, n_clusters, device="cpu"):
    """Try GraphST if installed."""
    try:
        from caust.models.stagate_wrapper import run_with_graphst
        adata_out = run_with_graphst(adata.copy(), n_domains=n_clusters, device=device)
        Z      = adata_out.obsm.get("GraphST", None)
        labels_raw = adata_out.obs.get("graphst_domain", None)
        if Z is None or labels_raw is None:
            return None, None
        labels = labels_raw.astype(int).values
        return Z, labels
    except (ImportError, Exception) as e:
        print(f"    [GraphST skip] {e}")
        return None, None


def benchmark_slice(adata, slice_id, n_clusters, gt_key, device):
    records = []

    def record(variant, method, Z, labels):
        if Z is None:
            return
        labels_true = None
        if gt_key and gt_key in adata.obs.columns:
            labels_true = adata.obs[gt_key].values
        m = evaluate_single_slice(labels, Z, labels_true, prefix="")
        m.update({"slice": slice_id, "variant": variant, "method": method})
        records.append(m)
        print(f"    {variant}/{method}: "
              f"ARI={m.get('ari', float('nan')):.4f}  "
              f"Sil={m.get('silhouette', float('nan')):.4f}")

    print(f"\n  === {slice_id} ===")

    # Baseline
    Z, labels = run_baseline(adata, n_clusters, device)
    record("Baseline", "CauST-internal", Z, labels)

    # CauST variants
    for fm, variant_name in [
        ("filter",                "CauST-Filter"),
        ("reweight",              "CauST-Reweight"),
        ("filter_and_reweight",   "CauST-Full"),
    ]:
        try:
            Z, labels = run_caust_variant(adata, n_clusters, fm, device)
            record(variant_name, "CauST-internal", Z, labels)
        except Exception as e:
            print(f"    [{variant_name} failed]: {e}")

    # Optional STAGATE (best CauST filtering + STAGATE downstream)
    Z_stage, labels_stage = try_stagate(adata, device)
    if Z_stage is not None:
        record("CauST-Full", "STAGATE", Z_stage, labels_stage)

    # Optional GraphST
    Z_gst, labels_gst = try_graphst(adata, n_clusters, device)
    if Z_gst is not None:
        record("CauST-Full", "GraphST", Z_gst, labels_gst)

    return records


def main():
    print("=" * 55)
    print("  CauST — Full Benchmark / Ablation Study")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    all_records = []

    # ── DLPFC ─────────────────────────────────────────────────────────────
    dlpfc_dir = ROOT / "data" / "processed" / "DLPFC"
    if dlpfc_dir.exists():
        # Benchmark on first 4 slices to keep runtime manageable
        slices = sorted(p.stem for p in dlpfc_dir.glob("*.h5ad"))[:4]
        for sid in slices:
            adata = sc.read_h5ad(dlpfc_dir / f"{sid}.h5ad")
            records = benchmark_slice(
                adata, sid, N_CLUSTERS_DLPFC, GROUND_TRUTH_KEY, device
            )
            all_records.extend(records)
    else:
        print("\n  [WARN] No DLPFC processed data found — skipping DLPFC benchmark.")

    # ── Mouse Brain ────────────────────────────────────────────────────────
    mb_path = ROOT / "data" / "processed" / "MouseBrain" / "mouse_brain.h5ad"
    if mb_path.exists():
        adata_mb = sc.read_h5ad(mb_path)
        records = benchmark_slice(
            adata_mb, "MouseBrain", N_CLUSTERS_MOUSE, None, device
        )
        all_records.extend(records)
    else:
        print("\n  [WARN] Mouse Brain data not found — skipping MouseBrain benchmark.")

    # ── Mouse Olfactory Bulb ──────────────────────────────────────────────
    mob_path = ROOT / "data" / "processed" / "MouseOB" / "mouse_ob.h5ad"
    if mob_path.exists():
        adata_mob = sc.read_h5ad(mob_path)
        records = benchmark_slice(
            adata_mob, "MouseOB", 7, None, device
        )
        all_records.extend(records)
    else:
        print("\n  [WARN] Mouse OB data not found — skipping MOB benchmark.")

    # ── Human Breast Cancer ───────────────────────────────────────────────
    hbc_path = ROOT / "data" / "processed" / "HumanBreastCancer" / "breast_cancer.h5ad"
    if hbc_path.exists():
        adata_hbc = sc.read_h5ad(hbc_path)
        records = benchmark_slice(
            adata_hbc, "HumanBreastCancer", 20, None, device
        )
        all_records.extend(records)
    else:
        print("\n  [WARN] Human Breast Cancer data not found — skipping HBC benchmark.")

    # ── STARmap ───────────────────────────────────────────────────────────
    star_path = ROOT / "data" / "processed" / "STARmap" / "starmap.h5ad"
    if star_path.exists():
        adata_star = sc.read_h5ad(star_path)
        records = benchmark_slice(
            adata_star, "STARmap", 7, None, device
        )
        all_records.extend(records)
    else:
        print("\n  [WARN] STARmap data not found — skipping STARmap benchmark.")

    if not all_records:
        print("\n  No results to save. Exiting.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    csv_path = OUTPUT_DIR / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved: {csv_path}")
    print(df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        plot_benchmark_results(
            results_df = df,
            group_col  = "variant",
            metric_col = "ari",
            hue_col    = "method",
            title      = "CauST Ablation — ARI across datasets",
            out_path   = str(OUTPUT_DIR / "benchmark_ari.png"),
        )
        plot_benchmark_results(
            results_df = df,
            group_col  = "variant",
            metric_col = "silhouette",
            hue_col    = "method",
            title      = "CauST Ablation — Silhouette across datasets",
            out_path   = str(OUTPUT_DIR / "benchmark_silhouette.png"),
        )
        print("✓ Benchmark plots saved.")
    except Exception as e:
        print(f"  [WARN] Plotting failed: {e}")


if __name__ == "__main__":
    main()
