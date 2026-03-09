"""
scripts/05_benchmark.py
========================
ONE-SHOT script: full ablation + benchmark across datasets and methods.

Resumable: results are saved after each slice+variant. If the script is
interrupted and re-run, completed conditions are skipped automatically.

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
  - DLPFC (first 4 slices)
  - MouseBrain    (if processed)
  - MouseOB       (if processed)
  - HBC           (if processed)
  - STARmap       (if processed)

Results saved as:  experiments/results/benchmark/all_results.csv
Figures saved as:  experiments/results/benchmark/benchmark_*.png
"""

import argparse, sys, json, time, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── CONFIG (defaults — overridable via --config YAML) ─────────────────────
N_CAUSAL_GENES   = 500
N_CLUSTERS_DLPFC = 7
N_CLUSTERS_MOUSE = 10
EPOCHS           = 500
SCORING_METHOD   = "gradient+perturbation"
GROUND_TRUTH_KEY = "layer_guess"
OUTPUT_DIR       = ROOT / "experiments" / "results" / "benchmark"
CSV_PATH         = OUTPUT_DIR / "all_results.csv"
# ─────────────────────────────────────────────────────────────────────────

def _load_config():
    """Parse --config flag and override globals from YAML if provided."""
    parser = argparse.ArgumentParser(description="CauST benchmark / ablation study")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file that overrides defaults.")
    args = parser.parse_args()
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        global N_CAUSAL_GENES, N_CLUSTERS_DLPFC, N_CLUSTERS_MOUSE
        global EPOCHS, SCORING_METHOD, GROUND_TRUTH_KEY
        causal = cfg.get("causal", {})
        model  = cfg.get("model", {})
        dataset = cfg.get("dataset", {})
        N_CAUSAL_GENES   = causal.get("n_causal_genes", N_CAUSAL_GENES)
        N_CLUSTERS_DLPFC = dataset.get("n_clusters_dlpfc", N_CLUSTERS_DLPFC)
        N_CLUSTERS_MOUSE = dataset.get("n_clusters_mouse", N_CLUSTERS_MOUSE)
        EPOCHS           = model.get("epochs", EPOCHS)
        SCORING_METHOD   = causal.get("scoring_method", SCORING_METHOD)
        GROUND_TRUTH_KEY = dataset.get("ground_truth_key", GROUND_TRUTH_KEY)
        print(f"  [config] Loaded overrides from {args.config}")

_load_config()

import scanpy as sc
import pandas as pd
from tqdm import tqdm
from caust import CauST
from caust.causal.scorer import cluster_latent
from caust.data.graph import build_spatial_graph, adata_to_pyg_data
from caust.models.autoencoder import SpatialAutoencoder, train_autoencoder
from caust.evaluate.metrics import evaluate_single_slice
from caust.visualize.plots import plot_benchmark_results


# ---------------------------------------------------------------------------
# Helpers: incremental CSV
# ---------------------------------------------------------------------------

def load_existing_results() -> pd.DataFrame:
    """Load previously completed results, or return empty DataFrame."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        print(f"  Loaded {len(df)} existing results from {CSV_PATH.name}")
        return df
    return pd.DataFrame()


def is_done(df: pd.DataFrame, slice_id: str, variant: str, method: str) -> bool:
    """Check if a (slice, variant, method) combination already has results."""
    if df.empty:
        return False
    mask = (
        (df["slice"] == slice_id) &
        (df["variant"] == variant) &
        (df["method"] == method)
    )
    return mask.any()


def append_and_save(df: pd.DataFrame, record: dict) -> pd.DataFrame:
    """Append one result row and flush to CSV immediately."""
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    return df


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_baseline(adata, n_clusters, device="cpu"):
    """Raw HVG features only — no causal intervention."""
    build_spatial_graph(adata)
    data = adata_to_pyg_data(adata).to(device)
    model = SpatialAutoencoder(data.x.shape[1]).to(device)
    model, _ = train_autoencoder(model, data, epochs=EPOCHS, device=device)
    Z = model.get_latent(data.x, data.edge_index)
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
        verbose        = True,
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
        return Z, labels_raw.astype(int).values
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
        return Z, labels_raw.astype(int).values
    except (ImportError, Exception) as e:
        print(f"    [GraphST skip] {e}")
        return None, None


def evaluate_and_record(df, adata, slice_id, variant, method, Z, labels, gt_key):
    """Evaluate metrics and save result row to CSV immediately."""
    if Z is None:
        return df
    labels_true = None
    if gt_key and gt_key in adata.obs.columns:
        labels_true = adata.obs[gt_key].values
    m = evaluate_single_slice(labels, Z, labels_true, prefix="")
    m.update({"slice": slice_id, "variant": variant, "method": method})
    print(f"    ✓ {variant}/{method}: "
          f"ARI={m.get('ari', float('nan')):.4f}  "
          f"Sil={m.get('silhouette', float('nan')):.4f}")
    return append_and_save(df, m)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  CauST — Full Benchmark / Ablation Study  (resumable)")
    print("=" * 55)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    df = load_existing_results()

    # ── Collect all (dataset_path, slice_id, n_clusters, gt_key) jobs ─────
    jobs = []

    dlpfc_dir = ROOT / "data" / "processed" / "DLPFC"
    if dlpfc_dir.exists():
        slices = sorted(p.stem for p in dlpfc_dir.glob("*.h5ad"))[:4]
        for sid in slices:
            jobs.append((dlpfc_dir / f"{sid}.h5ad", sid, N_CLUSTERS_DLPFC, GROUND_TRUTH_KEY))
    else:
        print("  [WARN] No DLPFC processed data found — skipping.")

    extra_datasets = [
        (ROOT / "data/processed/MouseBrain/mouse_brain.h5ad", "MouseBrain",         N_CLUSTERS_MOUSE, None),
        (ROOT / "data/processed/MouseOB/mouse_ob.h5ad",       "MouseOB",            7,                None),
        (ROOT / "data/processed/HumanBreastCancer/breast_cancer.h5ad", "HumanBreastCancer", 20, None),
        (ROOT / "data/processed/STARmap/starmap.h5ad",         "STARmap",            7,                None),
    ]
    for path, sid, nc, gt in extra_datasets:
        if path.exists():
            jobs.append((path, sid, nc, gt))
        else:
            print(f"  [WARN] {sid} data not found — skipping.")

    if not jobs:
        print("\n  No datasets available. Exiting.")
        return

    # ── Define all (variant, runner) conditions ───────────────────────────
    VARIANTS = [
        ("Baseline",        "CauST-internal"),
        ("CauST-Filter",    "CauST-internal"),
        ("CauST-Reweight",  "CauST-internal"),
        ("CauST-Full",      "CauST-internal"),
        ("CauST-Full",      "STAGATE"),
        ("CauST-Full",      "GraphST"),
    ]

    total_conditions = len(jobs) * len(VARIANTS)
    done_count = sum(
        1 for (_, sid, _, _) in jobs for (v, m) in VARIANTS if is_done(df, sid, v, m)
    )
    remaining = total_conditions - done_count
    print(f"\n  Total conditions: {total_conditions}  "
          f"(already done: {done_count},  remaining: {remaining})")

    pbar = tqdm(total=total_conditions, desc="Benchmark", unit="cond",
                initial=done_count)

    for data_path, slice_id, n_clusters, gt_key in jobs:
        print(f"\n{'='*50}")
        print(f"  === {slice_id} ===")
        print(f"{'='*50}")

        adata = sc.read_h5ad(data_path)
        print(f"  Loaded: {adata.n_obs} spots × {adata.n_vars} genes")

        for variant, method in VARIANTS:

            if is_done(df, slice_id, variant, method):
                pbar.update(1)
                continue

            t0 = time.time()
            print(f"\n  >> {variant} / {method}")

            try:
                if variant == "Baseline" and method == "CauST-internal":
                    Z, labels = run_baseline(adata.copy(), n_clusters, device)
                    df = evaluate_and_record(df, adata, slice_id, variant, method, Z, labels, gt_key)

                elif method == "CauST-internal":
                    fm_map = {
                        "CauST-Filter":   "filter",
                        "CauST-Reweight": "reweight",
                        "CauST-Full":     "filter_and_reweight",
                    }
                    Z, labels = run_caust_variant(adata, n_clusters, fm_map[variant], device)
                    df = evaluate_and_record(df, adata, slice_id, variant, method, Z, labels, gt_key)

                elif method == "STAGATE":
                    Z, labels = try_stagate(adata, device)
                    if Z is not None:
                        df = evaluate_and_record(df, adata, slice_id, variant, method, Z, labels, gt_key)
                    else:
                        print(f"    [skip] STAGATE not available")

                elif method == "GraphST":
                    Z, labels = try_graphst(adata, n_clusters, device)
                    if Z is not None:
                        df = evaluate_and_record(df, adata, slice_id, variant, method, Z, labels, gt_key)
                    else:
                        print(f"    [skip] GraphST not available")

            except Exception as e:
                print(f"    [FAILED] {variant}/{method}: {e}")

            elapsed = time.time() - t0
            print(f"    ({elapsed:.0f}s)")
            pbar.update(1)

        # Release h5ad file handles and model memory before loading next dataset
        del adata
        gc.collect()

    pbar.close()

    if df.empty:
        print("\n  No results to save. Exiting.")
        return

    print(f"\n{'='*55}")
    print(f"  ✓ All results saved: {CSV_PATH}")
    print(f"{'='*55}")
    print(df.to_string(index=False))

    # ── Generate plots ────────────────────────────────────────────────────
    try:
        for metric in ["ari", "silhouette"]:
            if metric in df.columns:
                plot_benchmark_results(
                    results_df = df,
                    group_col  = "variant",
                    metric_col = metric,
                    hue_col    = "method",
                    title      = f"CauST Ablation — {metric.upper()} across datasets",
                    out_path   = str(OUTPUT_DIR / f"benchmark_{metric}.png"),
                )
        print("  ✓ Benchmark plots saved.")
    except Exception as e:
        print(f"  [WARN] Plotting failed: {e}")

    # Final cleanup to prevent segfault during interpreter shutdown
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
