"""
caust/data/graph.py
===================
Build spatial neighborhood graphs and convert AnnData to PyTorch Geometric format.

Think of the tissue as a city map. Each spot is a building. We draw "roads"
(edges) between each building and its K nearest neighbours. The neural network
then uses these roads to share information between nearby spots.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

import scanpy as sc


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_spatial_graph(
    adata: sc.AnnData,
    n_neighbors: int = 6,
    spatial_key: str = "spatial",
) -> sc.AnnData:
    """
    Build a K-nearest-neighbour spatial graph and store it in AnnData.

    Each spot is connected to its n_neighbors spatially closest spots.
    The graph is made *undirected* (symmetric adjacency matrix).

    The result is stored in:
        adata.obsp["spatial_connectivities"]  — binary adjacency matrix
        adata.uns["spatial_neighbors"]        — metadata about the graph

    Parameters
    ----------
    adata        : preprocessed AnnData with spatial coordinates
    n_neighbors  : number of nearest spatial neighbours per spot
    spatial_key  : adata.obsm key holding (x, y) coordinates

    Returns
    -------
    adata  (modified in-place, also returned for chaining)
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"Spatial coordinates key '{spatial_key}' not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}\n"
            "Tip: for 10x Visium data the key is usually 'spatial'."
        )

    coords = np.array(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim == 1 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must have at least 2 columns (x, y).")
    coords = coords[:, :2]  # use only x and y

    n_spots = coords.shape[0]
    k = min(n_neighbors, n_spots - 1)     # guard against tiny test datasets
    print(f"[graph] Building KNN graph: {n_spots} spots, k={k} neighbours")

    # Efficient KD-Tree search
    tree = cKDTree(coords)
    # query k+1 because the first result is always the spot itself
    _, indices = tree.query(coords, k=k + 1)
    indices = indices[:, 1:]              # drop self-loops

    # Build COO edge lists
    row = np.repeat(np.arange(n_spots), k)
    col = indices.flatten()

    # Symmetric (undirected) adjacency matrix
    adj = sp.csr_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)),
        shape=(n_spots, n_spots),
    )
    adj = (adj + adj.T).sign().astype(np.float32)

    adata.obsp["spatial_connectivities"] = adj
    adata.uns["spatial_neighbors"] = {
        "n_neighbors": k,
        "spatial_key": spatial_key,
        "n_edges": int(adj.nnz),
    }

    avg_degree = adj.nnz / n_spots
    print(f"[graph] Edges: {adj.nnz:,}  (avg degree {avg_degree:.1f})\n")
    return adata


# ---------------------------------------------------------------------------
# AnnData  →  PyTorch Geometric Data
# ---------------------------------------------------------------------------

def adata_to_pyg_data(
    adata: sc.AnnData,
    graph_key: str = "spatial_connectivities",
    x_layer: Optional[str] = None,
) -> Data:
    """
    Convert an AnnData object (with a pre-built spatial graph) into a
    PyTorch Geometric ``Data`` object ready for the neural network.

    Parameters
    ----------
    adata      : AnnData with spatial graph in adata.obsp[graph_key]
    graph_key  : which adjacency matrix to use
    x_layer    : layer to use as node features; None → adata.X

    Returns
    -------
    torch_geometric.data.Data  with  .x  and  .edge_index
    """
    if graph_key not in adata.obsp:
        raise KeyError(
            f"Graph key '{graph_key}' not found in adata.obsp. "
            "Run build_spatial_graph() first."
        )

    # ── Node features ────────────────────────────────────────────────────
    if x_layer and x_layer in adata.layers:
        X = adata.layers[x_layer]
    else:
        X = adata.X

    if sp.issparse(X):
        X = X.toarray()
    x_tensor = torch.FloatTensor(X.astype(np.float32))

    # ── Edge index from adjacency matrix ─────────────────────────────────
    adj = adata.obsp[graph_key]
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    adj_coo = adj.tocoo()
    edge_index = torch.LongTensor(
        np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64)
    )

    data = Data(x=x_tensor, edge_index=edge_index)
    data.num_nodes = adata.n_obs
    return data


def get_edge_index_from_adata(
    adata: sc.AnnData,
    graph_key: str = "spatial_connectivities",
) -> torch.Tensor:
    """
    Return only the edge_index tensor from a stored adjacency matrix.

    Useful when you need to reuse the same graph topology for many
    different perturbation forward passes without recreating a full
    Data object each time.
    """
    if graph_key not in adata.obsp:
        raise KeyError(
            f"'{graph_key}' not in adata.obsp. Run build_spatial_graph() first."
        )
    adj = adata.obsp[graph_key]
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)

    adj_coo = adj.tocoo()
    edge_index = torch.LongTensor(
        np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64)
    )
    return edge_index
