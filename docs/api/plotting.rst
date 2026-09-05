Plotting API
============

Trajectory and composition
--------------------------

.. automodule:: CytoBridge.pl.trajectory
   :members: plot_sde_vs_real, plot_sde_vs_real_from_adata, plot_trajectory_grid, plot_trajectory_comparison_grid, plot_trajectory_gif

.. automodule:: CytoBridge.pl.celltype
   :members: plot_celltype_composition

.. automodule:: CytoBridge.pl.population
   :members: plot_population_overview

Velocity and growth
-------------------

.. automodule:: CytoBridge.pl.velocity
   :members: embed_velocity_to_spatial, plot_velocity_component, plot_intrinsic_interaction_direction_correlation, plot_intrinsic_interaction_direction_correlation_from_adata, plot_spatial_component_direction_correlation_roi_from_adata

.. automodule:: CytoBridge.pl.growth
   :members: plot_g_values, plot_growth_per_time, plot_growth_per_time_from_adata, plot_growth_interaction_bubble, plot_growth_timepoint_grid

.. automodule:: CytoBridge.pl.growth_summary
   :members: plot_growth_heatmap, plot_growth_size_maps

Temporal, enrichment, and training summaries
--------------------------------------------

.. automodule:: CytoBridge.pl.temporal
   :members: plot_developmental_wave_heatmap, plot_temporal_gene_heatmap, plot_temporal_pattern_prototypes, plot_temporal_profile_small_multiples

.. automodule:: CytoBridge.pl.enrichment
   :members: plot_enrichment_bar, plot_enrichment_dot

.. automodule:: CytoBridge.pl.training
   :members: plot_training_history, summarize_training_history

Lineage and spatial Sankey views
--------------------------------

.. automodule:: CytoBridge.pl.sankey
   :members: plot_sankey, plot_3d_spatial_sankey, plot_3d_spatial_sankey_style
