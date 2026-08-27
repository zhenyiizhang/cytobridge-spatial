# Downstream analysis

The downstream API works from an aligned AnnData object and a fitted model.
The packaged workflow presets call the same public functions described below.

## Interpolated states and labels

`cb.tl.run_interpolation_workflow` generates requested intermediate time
points, assigns cell labels, and returns an `InterpolationResult`. Its
`adata_dict`, `communication_adata_dict`, `ts_points`, and `time_keys` fields
are used by later steps.

The dataset tutorials keep supplied observed slices unchanged and simulate
intermediate points from the preceding observed slice. With this setting, an
intermediate state is a forward simulation within one observed interval; it is
not conditioned on the following endpoint. Enable lineage output only when the
simulation stores persistent particle identifiers.

## Velocity

`cb.tl.compute_velocity_components_from_adata` calculates intrinsic,
interaction, score, and combined model derivatives for observed cells.
Interaction velocity is recomputed within each time slice.

Direct two-dimensional spatial derivatives are plotted in their fitted
coordinates. Higher-dimensional expression-state derivatives can be projected
onto a two-dimensional embedding with scVelo. The two paths should not be
applied to the same vector field.

## Growth

`cb.tl.evaluate_growth_by_timepoint` evaluates the fitted growth head for each
time point. Standard split-SDE workflows use the pre-warp joint state and read
`growth_alpha` from the preset. A value of `1.0` enables the configured growth
effect; `0.0` produces a fixed-population simulation.

Growth summaries are written per cell and can be grouped by time or cell type.

## Sparse communication

`cb.tl.compute_timepoint_communications` constructs spatial-radius candidates
without materializing a dense cell-by-cell matrix. Edges retain sender-to-
receiver direction. The function can write sparse edge arrays and aggregated
cell-type tables.

Each time point records:

- `candidate_count`: pairs within the spatial cutoff;
- `selected_count`: pairs retained by the configured edge gate;
- `selected_fraction`: retained pairs divided by candidates; and
- `status`: the selection state for that time point.

If candidates exist but none pass a learned gate, the sparse result is empty
and the status records that selection outcome. If there are no candidates, the
status records that case separately. Empty sparse arrays use shapes `(2, 0)`
for `edge_index` and `(0,)` for `attn_mean`.

`cb.tl.analyze_attention_by_celltype` aggregates retained edges into cell-type
tables. The scope of those tables is determined by the input cohort, radius
cutoff, and selected edge gate.

## Ligand--receptor projection

`cb.tl.project_communication_to_lr_timecourses` combines communication weights
with expression and a ligand--receptor table. Generated inverse-PCA `log1p`
expression is clipped to non-negative values before scoring.

Complex scoring requires every ligand and receptor subunit:

- `complex_mode="min"` uses the least-expressed required subunit;
- `complex_mode="geometric_mean"` uses a zero-preserving geometric mean; and
- `require_all_subunits=True` keeps incomplete complexes out of both modes.

The returned result contains pair-level time courses and optional cell-type
matrices. Its coverage follows the genes and complexes present in the supplied
database and reference AnnData.

## Gene and module dynamics

Gene reconstruction uses the fitted PCA loadings, gene order, and center.
`cb.tl.inverse_pca_states` reconstructs gene-space values, while the temporal
helpers cluster profiles and summarize gene programs. Genes with inactive PCA
loadings can be reported separately from module means.

## Distribution metrics

`cb.tl.evaluate_model_distributions` calculates model-versus-observed metrics
for requested time points. The output separates joint, spatial, and state
spaces. `cb.tl.save_distribution_evaluation` writes the metric and local-
structure tables.

These fitted-model diagnostics are distinct from the leave-one-timepoint-out
protocol described in {doc}`benchmarks`.

## Figures and files

Downstream calculations write tables before plotting. Plotting helpers under
`cb.pl` read those tables or AnnData objects and write static or interactive
figures. Numerical values remain available in the corresponding CSV, NPZ,
JSON, and H5AD outputs.

See {doc}`data_checkpoints` for the standard output directory and
{doc}`api/tools` for function signatures.
