"""
scripts/06_visualize_results.py
================================
ONE-SHOT script: load saved results and regenerate all publication figures.

Run once after 03, 04, and 05 have been executed.

    python scripts/06_visualize_results.py

What it does
------------
  1.  Reads per-slice metrics from multi_slice/per_slice_metrics.csv
  2.  Reads benchmark results from benchmark/all_results.csv
  3.  Reads invariance scores JSON
  4.  Regenerates all figures (paper-quality, 300 DPI)
  5.  Saves everything to experiments/results/figures/
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _bar_with_error(ax, df, metric, title, ylabel):
    """Grouped bar chart with std error bars."""
    if metric not in df.columns:
        ax.set_title(f"{title}\n(column '{metric}' not found)")
        return
    groups  = df.groupby("variant")[metric]
    means   = groups.mean()
    stds    = groups.std().fillna(0)
    x       = np.arange(len(means))
    bars    = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)


def fig_per_slice(df, out_dir):
    """Bar chart of ARI / NMI / Silhouette per slice."""
    metrics = [c for c in ["ari", "nmi", "silhouette"] if c in df.columns]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        vals = df.set_index("slice")[m] if "slice" in df.columns else df[m]
        vals.plot.bar(ax=ax, color="steelblue", alpha=0.8)
        ax.set_title(f"{m.upper()} per DLPFC slice")
        ax.set_ylabel(m)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path = out_dir / "per_slice_metrics.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_benchmark_grouped(df, out_dir):
    """Grouped bar chart: variant × method for ARI."""
    if "ari" not in df.columns or "variant" not in df.columns:
        return
    try:
        import seaborn as sns
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric in zip(axes, ["ari", "silhouette"]):
            if metric not in df.columns:
                continue
            sns.barplot(data=df, x="variant", y=metric,
                        hue="method" if "method" in df.columns else None,
                        ax=ax, capsize=0.1)
            ax.set_title(f"Benchmark — {metric.upper()}")
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        path = out_dir / "benchmark_grouped.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")
    except ImportError:
        # fallback without seaborn
        fig, ax = plt.subplots(figsize=(8, 4))
        _bar_with_error(ax, df, "ari", "Benchmark ARI", "ARI")
        plt.tight_layout()
        path = out_dir / "benchmark_ari_simple.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")


def fig_invariance_bar(scores_dict, out_dir):
    """Horizontal bar chart of cross-slice invariance scores (top 40 genes)."""
    if not scores_dict:
        return
    items  = sorted(scores_dict.items(), key=lambda x: -x[1])[:40]
    genes  = [g for g, _ in items]
    vals   = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6, max(4, len(genes) * 0.25)))
    ax.barh(genes[::-1], vals[::-1], color="darkorange", alpha=0.85)
    ax.set_xlabel("Invariance Score")
    ax.set_title("Top-40 Cross-Slice Invariant Causal Genes")
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    path = out_dir / "invariance_top40.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_lodo(df_lodo, out_dir):
    """Box-plot of LODO ARI across donors."""
    ari_col = "lodo_ari" if "lodo_ari" in df_lodo.columns else (
               "ari"     if "ari"      in df_lodo.columns else None)
    if ari_col is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    if "test_donor" in df_lodo.columns:
        donors = df_lodo["test_donor"].unique()
        data_by_donor = [df_lodo[df_lodo["test_donor"] == d][ari_col].dropna().values
                         for d in donors]
        ax.boxplot(data_by_donor, labels=donors)
        ax.set_xlabel("Held-out Donor")
    else:
        ax.boxplot(df_lodo[ari_col].dropna().values)
    ax.set_ylabel("ARI")
    ax.set_title("LODO Cross-Donor Validation")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = out_dir / "lodo_ari.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  CauST — Figure Regeneration")
    print("=" * 55)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generated = 0

    # Per-slice metrics
    ps_csv = RESULTS_DIR / "multi_slice" / "per_slice_metrics.csv"
    if ps_csv.exists():
        df_ps = pd.read_csv(ps_csv)
        print(f"\nPer-slice data: {len(df_ps)} rows")
        fig_per_slice(df_ps, FIGURES_DIR)
        generated += 1
    else:
        print(f"\n  [SKIP] {ps_csv.name} not found — run 04_run_multi_slice.py first.")

    # Benchmark
    bm_csv = RESULTS_DIR / "benchmark" / "all_results.csv"
    if bm_csv.exists():
        df_bm = pd.read_csv(bm_csv)
        print(f"\nBenchmark data: {len(df_bm)} rows")
        fig_benchmark_grouped(df_bm, FIGURES_DIR)
        generated += 1
    else:
        print(f"  [SKIP] {bm_csv.name} not found — run 05_benchmark.py first.")

    # Invariance scores
    inv_json = RESULTS_DIR / "multi_slice" / "invariance_scores.json"
    if inv_json.exists():
        with open(inv_json) as f:
            inv_scores = json.load(f)
        print(f"\nInvariance scores: {len(inv_scores)} genes")
        fig_invariance_bar(inv_scores, FIGURES_DIR)
        generated += 1
    else:
        print(f"  [SKIP] invariance_scores.json not found.")

    # LODO
    lodo_csv = RESULTS_DIR / "multi_slice" / "lodo_metrics.csv"
    if lodo_csv.exists():
        df_lodo = pd.read_csv(lodo_csv)
        print(f"\nLODO data: {len(df_lodo)} rows")
        fig_lodo(df_lodo, FIGURES_DIR)
        generated += 1
    else:
        print(f"  [SKIP] lodo_metrics.csv not found.")

    if generated:
        print(f"\n✓ {generated} figure group(s) saved to: {FIGURES_DIR}")
    else:
        print("\n  No figures generated. Run scripts 03–05 first.")


if __name__ == "__main__":
    main()
