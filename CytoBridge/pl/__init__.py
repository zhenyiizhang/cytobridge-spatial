from importlib import import_module as _import_module


# ``plot.py`` is a legacy, all-in-one module whose import graph includes
# Scanpy, scVelo, Torch, TorchDiffEq, and the training namespace.  Importing it
# conditionally based on whichever dependency happened to be missing made
# extras non-monotonic: installing preprocessing could make ``CytoBridge.pl``
# fail.  Keep its established function names available, but load the module
# only when a caller actually requests one of them.
_LEGACY_PLOT_EXPORTS = frozenset(
    {
        "analyze_terminal_states",
        "plot_combined_velocity_stream",
        "plot_growth",
        "plot_interaction_potential",
        "plot_interaction_potential_epoch",
        "plot_interaction_stream",
        "plot_landscape",
        "plot_ode",
        "plot_ode_trajectories",
        "plot_score_and_gradient",
        "plot_score_stream",
        "plot_sde_trajectories",
        "plot_velocity_stream",
        "process_sde_classification",
        "sde_plot",
        "visualize_trajectory_classification",
    }
)


def __getattr__(name: str):
    if name not in _LEGACY_PLOT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = _import_module(f"{__name__}.plot")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"CytoBridge.pl.{name} uses the legacy plotting stack. "
            "Install it with: pip install 'CytoBridge[velocity]'"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LEGACY_PLOT_EXPORTS)

# Downstream visualization modules
from .sankey import (
    plot_sankey,
    plot_3d_spatial_sankey,
    plot_3d_spatial_sankey_style,
)
from .trajectory import (
    plot_sde_vs_real,
    plot_sde_vs_real_from_adata,
    plot_trajectory_comparison_grid,
    plot_trajectory_gif,
    plot_trajectory_grid,
)
from .growth import (
    plot_g_values,
    plot_growth_per_time,
    plot_growth_per_time_from_adata,
    gene_velocity_embeddings,
    gene_velocity_embeddings_from_adata,
    plot_gene_expression_trends,
    plot_cell_counts_over_time,
    plot_growth_interaction_bubble,
    plot_growth_timepoint_grid,
)
from .velocity import (
    SpatialDirectionCorrelationResult,
    embed_velocity_to_spatial,
    plot_velocity_component,
    plot_intrinsic_interaction_direction_correlation,
    plot_intrinsic_interaction_direction_correlation_from_adata,
    plot_spatial_component_direction_correlation_roi_from_adata,
)
from .temporal import (
    plot_developmental_wave_heatmap,
    plot_temporal_gene_heatmap,
    plot_temporal_pattern_prototypes,
    plot_temporal_profile_small_multiples,
)
from .celltype import plot_celltype_composition
from .enrichment import plot_enrichment_bar, plot_enrichment_dot
from .training import plot_training_history, summarize_training_history


__all__ = sorted(
    {
        name
        for name, value in globals().items()
        if not name.startswith("_") and callable(value)
    }
    | _LEGACY_PLOT_EXPORTS
)
