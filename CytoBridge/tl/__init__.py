"""Tool-layer public API."""

from . import core
from . import downstream
from . import graph

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
    DynamicalRuntime,
    InterpolationResult,
    LoadedClassifierCache,
    LoadedModel,
    MLP,
    adata_to_aligned_dataframe,
    analyze_attention_by_celltype,
    apply_spatial_warp_to_segments,
    build_dynamical_runtime,
    build_cached_classifier_inputs_from_adata,
    build_time_grid,
    compute_drift,
    compute_drift_from_adata,
    compute_timepoint_communications,
    compute_umap_embedding,
    compute_velocity_components,
    compute_velocity_components_from_adata,
    infer_feature_columns,
    infer_time_key,
    load_cached_mlp_classifier,
    load_dynamical_model_from_dir,
    load_legacy_dynamical_model_from_dir,
    load_label_to_color,
    merge_annotation,
    parse_time_value,
    predict_cached_mlp_classifier_from_adata,
    predict_labels_for_points,
    predict_labels_for_trajectories,
    plot_lineage_sankey,
    plot_spatiotemporal_3d,
    run_interpolation_workflow,
    sample_observed_x0,
    save_interpolated_attention,
    save_timepoint_snapshots,
    set_global_random_seed,
    simulate_sde_points,
    simulate_sde_points_split,
    simulate_sde_points_split_from_x0,
    simulate_piecewise_spatially_warped_split,
    train_mlp_classifier,
    train_mlp_classifier_from_adata,
)
from .graph import train_edge_predictor

__all__ = [
    "analysis",
    "core",
    "downstream",
    "flow_matching",
    "graph",
    "methods",
    "train_edge_predictor",
    "build_dynamical_runtime",
    "load_dynamical_model_from_dir",
    "load_legacy_dynamical_model_from_dir",
    "load_cached_mlp_classifier",
    "load_label_to_color",
    "build_cached_classifier_inputs_from_adata",
    "DynamicalRuntime",
    "InterpolationResult",
    "LoadedModel",
    "LoadedClassifierCache",
    "apply_spatial_warp_to_segments",
    "simulate_sde_points",
    "simulate_sde_points_split",
    "simulate_sde_points_split_from_x0",
    "simulate_piecewise_spatially_warped_split",
    "sample_observed_x0",
    "compute_velocity_components",
    "compute_velocity_components_from_adata",
    "compute_drift",
    "compute_drift_from_adata",
    "compute_timepoint_communications",
    "compute_umap_embedding",
    "save_interpolated_attention",
    "save_timepoint_snapshots",
    "set_global_random_seed",
    "analyze_attention_by_celltype",
    "train_mlp_classifier",
    "train_mlp_classifier_from_adata",
    "predict_cached_mlp_classifier_from_adata",
    "predict_labels_for_points",
    "predict_labels_for_trajectories",
    "plot_lineage_sankey",
    "plot_spatiotemporal_3d",
    "run_interpolation_workflow",
    "MLP",
    "parse_time_value",
    "infer_time_key",
    "adata_to_aligned_dataframe",
    "infer_feature_columns",
    "merge_annotation",
    "build_time_grid",
]
