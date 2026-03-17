"""Tool-layer public API."""

from . import core
from . import downstream
from . import graph
from . import train

try:
    from .core import flow_matching
except ModuleNotFoundError as exc:
    if exc.name in {"ot"}:
        flow_matching = None
    else:
        raise

try:
    from .core import methods
except ModuleNotFoundError as exc:
    if exc.name in {"torchdiffeq"}:
        methods = None
    else:
        raise

try:
    from .downstream import analysis
except ModuleNotFoundError as exc:
    if exc.name in {"anndata", "scanpy"}:
        analysis = None
    else:
        raise

from .downstream import (
    LoadedModel,
    MLP,
    adata_to_aligned_dataframe,
    analyze_attention_by_celltype,
    build_time_grid,
    compute_drift,
    compute_drift_from_adata,
    compute_umap_embedding,
    compute_velocity_components,
    compute_velocity_components_from_adata,
    infer_feature_columns,
    infer_time_key,
    load_dynamical_model_from_dir,
    merge_annotation,
    parse_time_value,
    predict_labels_for_trajectories,
    save_interpolated_attention,
    simulate_sde_points,
    simulate_sde_points_split,
    train_mlp_classifier,
    train_mlp_classifier_from_adata,
)
from .graph import train_edge_predictor
from .train import fit, fit_spatial_csv, fit_spatial_h5ad

__all__ = [
    "analysis",
    "core",
    "downstream",
    "fit",
    "fit_spatial_csv",
    "fit_spatial_h5ad",
    "flow_matching",
    "graph",
    "methods",
    "train",
    "train_edge_predictor",
    "load_dynamical_model_from_dir",
    "LoadedModel",
    "simulate_sde_points",
    "simulate_sde_points_split",
    "compute_velocity_components",
    "compute_velocity_components_from_adata",
    "compute_drift",
    "compute_drift_from_adata",
    "compute_umap_embedding",
    "save_interpolated_attention",
    "analyze_attention_by_celltype",
    "train_mlp_classifier",
    "train_mlp_classifier_from_adata",
    "predict_labels_for_trajectories",
    "MLP",
    "parse_time_value",
    "infer_time_key",
    "adata_to_aligned_dataframe",
    "infer_feature_columns",
    "merge_annotation",
    "build_time_grid",
]
