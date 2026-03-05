# caust/data/__init__.py
from caust.data.loader import load_and_preprocess, load_multiple_slices, save_processed
from caust.data.graph import build_spatial_graph, adata_to_pyg_data, get_edge_index_from_adata

__all__ = [
    "load_and_preprocess",
    "load_multiple_slices",
    "save_processed",
    "build_spatial_graph",
    "adata_to_pyg_data",
    "get_edge_index_from_adata",
]
