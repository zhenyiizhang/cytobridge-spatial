"""Downstream analysis utilities."""

from .attention import analyze_attention_by_celltype, save_interpolated_attention
from .checkpoint import LoadedModel, load_dynamical_model_from_dir
from .classification import (
    MLP,
    predict_labels_for_trajectories,
    train_mlp_classifier,
    train_mlp_classifier_from_adata,
)
from .downstream_data import (
    adata_to_aligned_dataframe,
    build_time_grid,
    infer_feature_columns,
    infer_time_key,
    merge_annotation,
    parse_time_value,
)
from .simulation import (
    compute_drift,
    compute_drift_from_adata,
    compute_umap_embedding,
    compute_velocity_components,
    compute_velocity_components_from_adata,
    simulate_sde_points,
    simulate_sde_points_split,
)

__all__ = [
    "LoadedModel",
    "MLP",
    "adata_to_aligned_dataframe",
    "analyze_attention_by_celltype",
    "build_time_grid",
    "compute_drift",
    "compute_drift_from_adata",
    "compute_umap_embedding",
    "compute_velocity_components",
    "compute_velocity_components_from_adata",
    "infer_feature_columns",
    "infer_time_key",
    "load_dynamical_model_from_dir",
    "merge_annotation",
    "parse_time_value",
    "predict_labels_for_trajectories",
    "save_interpolated_attention",
    "simulate_sde_points",
    "simulate_sde_points_split",
    "train_mlp_classifier",
    "train_mlp_classifier_from_adata",
]
