"""
caust/visualize/plots.py
=========================
All CauST visualization functions.

Every function saves its output to disk instead of just showing it, so
you can run scripts once and keep all plots as files.

Plots implemented
-----------------
  1. plot_spatial_domains        — tissue slice coloured by domain
  2. plot_causal_scores          — bar chart of top-N causal genes
  3. plot_invariance_heatmap     — cross-slice gene-score matrix
  4. plot_intervention_effect    — before/after knockout tissue maps
  5. plot_benchmark_results      — grouped bar chart of ARI comparisons
  6. plot_training_loss          — autoencoder training curve
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "tab20"


# ---------------------------------------------------------------------------
# 1. Spatial domain map
# ---------------------------------------------------------------------------

def plot_spatial_domains(
    adata: sc.AnnData,
    label_key: str = "caust_domain",
    title: str = "CauST Spatial Domains",
    spot_size: int = 120,
    out_path: Optional[str] = None,
) -> None:
    """
    Plot the tissue slice coloured by predicted spatial domain.

    Each dot is one spot on the tissue. Colour = domain assignment.

    Parameters
    ----------
    adata     : AnnData with spatial coordinates and domain labels
    label_key : adata.obs column containing domain labels
    title     : figure title
    spot_size : marker size (pixels)
    out_path  : save path (PNG); if None, saves to 'output/spatial_domains.png'
    """
    out_path = _resolve_path(out_path, "output/spatial_domains.png")

    if label_key not in adata.obs.columns:
        raise KeyError(f"'{label_key}' not in adata.obs. Available: {list(adata.obs.columns)}")

    if "spatial" not in adata.obsm:
        raise KeyError("No spatial coordinates found in adata.obsm['spatial'].")

    coords  = adata.obsm["spatial"][:, :2]
    labels  = adata.obs[label_key].values
    uniq    = sorted(set(labels), key=str)
    cmap    = plt.get_cmap(PALETTE)
    colors  = {lbl: cmap(i / max(len(uniq) - 1, 1)) for i, lbl in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(8, 8))
    for lbl in uniq:
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[lbl]], s=spot_size, label=str(lbl),
            linewidths=0, alpha=0.85,
        )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend(loc="upper right", markerscale=1.4, title="Domain",
               framealpha=0.8, fontsize=9)
    ax.set_aspect("equal")
    ax.invert_yaxis()          # tissue image convention: y increases downward
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Causal gene score bar chart
# ---------------------------------------------------------------------------

def plot_causal_scores(
    gene_scores: Dict[str, float],
    top_k: int = 20,
    title: str = "Top Causal Genes",
    out_path: Optional[str] = None,
) -> None:
    """
    Horizontal bar chart of the top-K causal genes ranked by score.

    Parameters
    ----------
    gene_scores : {gene_name: causal_score}
    top_k       : how many genes to display
    out_path    : save path
    """
    out_path = _resolve_path(out_path, "output/causal_gene_scores.png")

    top = sorted(gene_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    if not top:
        print("[plots] gene_scores is empty — skipping plot.")
        return
    genes, scores = zip(*top)
    genes  = list(genes)[::-1]        # reversed so highest is at top
    scores = list(scores)[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.38)))
    colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.90, top_k))[::-1]
    bars = ax.barh(genes, scores, color=colors, edgecolor="none")

    # Annotate score values
    for bar, score in zip(bars, scores):
        ax.text(
            score + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", ha="left", fontsize=8.5,
        )

    ax.set_xlim(0, max(scores) * 1.15)
    ax.set_xlabel("Causal Score (normalised)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Cross-slice invariance heatmap
# ---------------------------------------------------------------------------

def plot_invariance_heatmap(
    causal_scores_per_slice: Dict[str, Dict[str, float]],
    top_k: int = 50,
    title: str = "Gene Causal Scores Across Slices",
    out_path: Optional[str] = None,
) -> None:
    """
    Heatmap showing causal scores of top-K genes across all slices.

    Genes that are consistently high (hot colour in every column) are the
    truly invariant causal genes CauST prioritises.

    Parameters
    ----------
    causal_scores_per_slice : {slice_id: {gene: score}}
    top_k                   : number of genes to display (rows)
    out_path                : save path
    """
    out_path = _resolve_path(out_path, "output/invariance_heatmap.png")

    # Collect global top-K genes by mean score
    all_genes: Dict[str, List[float]] = {}
    for sid, scores in causal_scores_per_slice.items():
        for g, v in scores.items():
            all_genes.setdefault(g, []).append(v)

    mean_scores = {g: np.mean(vs) for g, vs in all_genes.items()}
    top_genes   = sorted(mean_scores, key=mean_scores.get, reverse=True)[:top_k]

    slice_ids = sorted(causal_scores_per_slice.keys(), key=str)
    data_matrix = np.zeros((len(top_genes), len(slice_ids)), dtype=np.float32)

    for j, sid in enumerate(slice_ids):
        slice_map = causal_scores_per_slice[sid]
        for i, gene in enumerate(top_genes):
            data_matrix[i, j] = slice_map.get(gene, 0.0)

    df = pd.DataFrame(data_matrix, index=top_genes, columns=[str(s) for s in slice_ids])

    fig_h = max(8, top_k * 0.22)
    fig_w = max(6, len(slice_ids) * 0.9)

    cg = sns.clustermap(
        df,
        cmap="YlOrRd",
        figsize=(fig_w, fig_h),
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.15, 0.12),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
    )
    cg.ax_heatmap.set_title(title, fontsize=13, fontweight="bold", pad=12)
    cg.ax_heatmap.tick_params(axis="y", labelsize=7)
    cg.ax_heatmap.tick_params(axis="x", labelsize=9)
    cg.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(cg.figure)
    print(f"[plots] Saved: {out_path}")


# ---------------------------------------------------------------------------
# 4. Intervention effect side-by-side
# ---------------------------------------------------------------------------

def plot_intervention_effect(
    adata: sc.AnnData,
    labels_before: np.ndarray,
    labels_after: np.ndarray,
    gene_name: str,
    out_path: Optional[str] = None,
) -> None:
    """
    Side-by-side tissue maps showing domain assignments before and after
    knocking out a specific gene.

    If the two maps look very different, the gene is causally important.
    If they look identical, the gene is a bystander.

    Parameters
    ----------
    adata         : AnnData with spatial coordinates
    labels_before : domain labels using original data
    labels_after  : domain labels after gene knockout
    gene_name     : name of the knocked-out gene (for the title)
    out_path      : save path
    """
    out_path = _resolve_path(out_path, f"output/intervention_{gene_name}.png")

    coords = adata.obsm["spatial"][:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, labels, subtitle in zip(
        axes,
        [labels_before, labels_after],
        ["Original", f"After do({gene_name} = E[{gene_name}])"],
    ):
        uniq   = sorted(set(labels), key=str)
        cmap   = plt.get_cmap(PALETTE)
        colors = {lbl: cmap(i / max(len(uniq) - 1, 1)) for i, lbl in enumerate(uniq)}
        for lbl in uniq:
            mask = labels == lbl
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[colors[lbl]], s=80, label=str(lbl),
                linewidths=0, alpha=0.85,
            )
        ax.set_title(subtitle, fontsize=12, fontweight="bold")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.legend(markerscale=1.2, title="Domain", fontsize=8, framealpha=0.7)

    fig.suptitle(f"Intervention Effect — Gene: {gene_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Benchmark ARI comparison
# ---------------------------------------------------------------------------

def plot_benchmark_results(
    results_df: pd.DataFrame,
    metric_col: str = "ARI",
    group_col: str = "method",
    hue_col: str = "dataset",
    title: str = "Benchmark: ARI Comparison Across Methods",
    out_path: Optional[str] = None,
) -> None:
    """
    Grouped bar chart comparing all methods across datasets.

    Parameters
    ----------
    results_df  : DataFrame with columns: method, dataset, ARI (and optionally NMI)
    metric_col  : column to plot on y-axis
    group_col   : x-axis grouping (usually 'method')
    hue_col     : colour grouping (usually 'dataset')
    out_path    : save path
    """
    out_path = _resolve_path(out_path, f"output/benchmark_{metric_col}.png")

    if results_df.empty:
        print("[plots] results_df is empty — skipping benchmark plot.")
        return

    required = {metric_col, group_col, hue_col}
    missing  = required - set(results_df.columns)
    if missing:
        raise ValueError(f"Missing columns in results_df: {missing}")

    fig, ax = plt.subplots(figsize=(max(10, len(results_df[group_col].unique()) * 1.5), 6))

    palette = sns.color_palette("Set2", n_colors=results_df[hue_col].nunique())
    sns.barplot(
        data=results_df,
        x=group_col, y=metric_col, hue=hue_col,
        ax=ax, palette=palette,
        order=sorted(results_df[group_col].unique()),
        capsize=0.06, errcolor="0.3", errwidth=1.5,
    )

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel(metric_col, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, min(1.05, results_df[metric_col].max() * 1.15))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title=hue_col.capitalize(), framealpha=0.8)
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Training loss curve
# ---------------------------------------------------------------------------

def plot_training_loss(
    loss_history: List[float],
    title: str = "Autoencoder Training Loss",
    out_path: Optional[str] = None,
) -> None:
    """
    Plot the autoencoder reconstruction loss over training epochs.

    A good training curve decreases smoothly and stabilises.
    If the loss oscillates or doesn't decrease, try lowering the
    learning rate.

    Parameters
    ----------
    loss_history : list of per-epoch MSE loss values
    out_path     : save path
    """
    out_path = _resolve_path(out_path, "output/training_loss.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    epochs  = list(range(1, len(loss_history) + 1))
    ax.plot(epochs, loss_history, color="steelblue", linewidth=1.5, label="Train MSE")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_path(out_path: Optional[str], default: str) -> Path:
    p = Path(out_path if out_path else default)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] Saved: {path}")
