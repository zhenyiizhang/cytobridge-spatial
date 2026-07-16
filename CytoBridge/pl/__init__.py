try:
    from .plot import *  # noqa: F401,F403
except ModuleNotFoundError as exc:
    # Keep downstream imports usable when optional plotting deps are missing.
    if exc.name not in {"scanpy", "scvelo"}:
        raise

# Downstream visualization modules
from .sankey import (
    plot_sankey,
    plot_3d_spatial_sankey,
    plot_3d_spatial_sankey_style,
)
from .trajectory import (
    plot_sde_vs_real,
    plot_sde_vs_real_from_adata,
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
    plot_temporal_gene_heatmap,
    plot_temporal_pattern_prototypes,
    plot_temporal_profile_small_multiples,
)
from .celltype import plot_celltype_composition
from .enrichment import plot_enrichment_bar, plot_enrichment_dot
