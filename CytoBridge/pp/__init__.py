from .preprocess import preprocess
from .interaction_graph import (
    estimate_neighborhood_threshold_from_aligned_spatial,
    generate_interaction_graph,
    sanitize_interaction_graph_uns,
)
from .edge_prediction import train_edge_predictor
from ..graph_database import (
    FORMAL_GRAPH_DATABASES,
    bundled_graph_database_path,
    resolve_graph_database,
)
from .spatial_align import (
    AlignConfig,
    align_spatial,
    preprocess_fixed_spatial,
    preprocess_align_to_files,
    preprocess_and_align,
)
from .chicken_heart import (
    apply_chicken_heart_coordinate_validation,
    chicken_heart_anatomical_orientation_qc,
    validate_prepared_chicken_heart_input,
)
from .chicken_heart_input import (
    assemble_chicken_heart_reference_counts,
    prepare_chicken_heart_input,
    prepare_chicken_heart_ot_adata,
    prepare_chicken_heart_ot_input,
    validate_chicken_heart_ot_input,
)
from .legacy_model_input import (
    legacy_model_input_csv_to_adata,
    write_legacy_model_input_h5ad,
)

__all__ = [
    "preprocess",
    "AlignConfig",
    "align_spatial",
    "preprocess_fixed_spatial",
    "preprocess_and_align",
    "preprocess_align_to_files",
    "apply_chicken_heart_coordinate_validation",
    "assemble_chicken_heart_reference_counts",
    "chicken_heart_anatomical_orientation_qc",
    "prepare_chicken_heart_input",
    "prepare_chicken_heart_ot_adata",
    "prepare_chicken_heart_ot_input",
    "validate_chicken_heart_ot_input",
    "validate_prepared_chicken_heart_input",
    "estimate_neighborhood_threshold_from_aligned_spatial",
    "generate_interaction_graph",
    "sanitize_interaction_graph_uns",
    "train_edge_predictor",
    "LREdgePriorConfig",
    "build_lr_edge_prior",
    "estimate_state_space_radius",
    "state_space_fit_params",
    "FORMAL_GRAPH_DATABASES",
    "bundled_graph_database_path",
    "resolve_graph_database",
    "legacy_model_input_csv_to_adata",
    "write_legacy_model_input_h5ad",
]


def __getattr__(name):
    """Load non-spatial graph utilities lazily to avoid a tl/pp import cycle."""

    if name in {"LREdgePriorConfig", "build_lr_edge_prior"}:
        from . import lr_edge_prior

        return getattr(lr_edge_prior, name)
    if name in {"estimate_state_space_radius", "state_space_fit_params"}:
        from . import state_space

        return getattr(state_space, name)
    raise AttributeError(name)
