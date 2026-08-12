Tools API
=========

Model fitting and loading
-------------------------

.. automodule:: CytoBridge.tl.train.fit
   :members: fit, fit_spatial_csv, fit_spatial_h5ad

.. automodule:: CytoBridge.tl.downstream.checkpoint
   :members: LoadedModel, load_dynamical_model_from_dir, load_legacy_dynamical_model_from_dir
   :show-inheritance:

.. automodule:: CytoBridge.tl.downstream.runtime
   :members: DynamicalRuntime, build_dynamical_runtime
   :show-inheritance:

Interpolation, velocity, and growth
-----------------------------------

.. automodule:: CytoBridge.tl.downstream.workflows
   :members: InterpolationResult, run_interpolation_workflow, compute_timepoint_communications, plot_lineage_sankey, plot_spatiotemporal_3d
   :show-inheritance:

.. automodule:: CytoBridge.tl.downstream.simulation
   :members: sample_observed_x0, simulate_sde_points, simulate_sde_points_split, simulate_sde_points_split_from_x0, compute_velocity_components, compute_velocity_components_from_adata, compute_drift, compute_drift_from_adata

.. automodule:: CytoBridge.tl.downstream.celltype
   :members: summarize_label_composition, evaluate_growth_by_timepoint, summarize_growth_interaction_by_celltype

Classification
--------------

.. automodule:: CytoBridge.tl.downstream.classification
   :members: SpatialSmoothingSelection, analyze_spatial_label_sensitivity, select_spatial_smoothing_k, smooth_spatial_labels, train_mlp_classifier_from_adata, train_cached_mlp_classifier_from_adata, predict_labels_for_points, predict_labels_for_trajectories
   :show-inheritance:

Communication and ligand--receptor analysis
--------------------------------------------

.. automodule:: CytoBridge.tl.downstream.attention
   :members: analyze_attention_by_celltype, save_interpolated_attention

.. automodule:: CytoBridge.tl.downstream.lr_projection
   :members: load_ligand_receptor_database, project_communication_to_lr_timecourses, compute_focal_lr_type_hotspots

Gene programs and enrichment
----------------------------

.. automodule:: CytoBridge.tl.downstream.temporal
   :members: inverse_pca_states, evaluate_pca_anchor_reconstruction, analyze_developmental_wave, cluster_temporal_profiles, summarize_temporal_gene_patterns

.. automodule:: CytoBridge.tl.downstream.enrichment
   :members: load_gmt_gene_sets, make_gene_set_library, overrepresentation_analysis

Evaluation and benchmarks
-------------------------

.. automodule:: CytoBridge.tl.downstream.evaluation
   :members: compute_distribution_metrics, compute_local_structure_metrics, evaluate_model_distributions, compare_distribution_metric_tables, save_distribution_evaluation, save_distribution_metric_comparison

.. automodule:: CytoBridge.tl.downstream.benchmark
   :members: FrozenBenchmarkTransform, fit_frozen_benchmark_transform, benchmark_projection_seed, evaluate_spatiotemporal_prediction
   :show-inheritance:

Virtual perturbations
---------------------

.. automodule:: CytoBridge.tl.downstream.ablation
   :members: compute_virtual_ablation_metrics, run_virtual_cell_type_ablation, run_virtual_interaction_ablation
