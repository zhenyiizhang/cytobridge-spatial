# ARISTA salamander brain regeneration

**Notebook:** {download}`arista.ipynb <dataset_workflows/arista.ipynb>`

## Inputs

- `inputs/arista_aligned.h5ad`, with the observation columns and embeddings
  named by the `arista` workflow preset
- `inputs/arista_model/`, containing an existing model checkpoint
- an edge-predictor checkpoint when `RUN_TRAINING=True`
- the ligand--receptor table returned by
  `cb.pp.bundled_graph_database_path("arista")`, or a file set with
  `LR_DATABASE_OVERRIDE`

The notebook writes under `tutorial_outputs/arista/`. Change the input and
output paths in its first section before running it.

## Package calls

The notebook reads settings with `load_workflow_config("arista")`. It can run
`cb.tl.fit` when training is enabled; otherwise it loads the checkpoint with
`cb.tl.load_dynamical_model_from_dir` and prepares the runtime with
`cb.tl.build_dynamical_runtime`.

The analysis then calls:

1. `cb.tl.adata_to_aligned_dataframe`
2. `cb.tl.run_interpolation_workflow`
3. `cb.tl.summarize_label_composition`
4. `cb.tl.compute_velocity_components_from_adata`
5. `cb.tl.evaluate_growth_by_timepoint`
6. `cb.tl.compute_timepoint_communications`
7. `cb.tl.project_communication_to_lr_timecourses`
8. `cb.tl.evaluate_model_distributions`

Set `RUN_FULL_SCOPE=True` to use the time grid and particle setting stored in
the preset. The default uses interval midpoints and a particle cap.

```bash
cytobridge workflow --config arista --dry-run
jupyter lab docs/tutorials/dataset_workflows/arista.ipynb
```

## Figure reproduction

Main Figure 5 and Supplementary Figures S17–S22 have separate notebooks under
`docs/tutorials/paper_figures/`. The full training record, downstream outputs,
figure-building scripts, and figure pages are stored in
`release_artifacts/arista_package_native_spatialqc_z50_retrain_20260824_r1/`.
The compact figure notebooks do not repeat model training.

## Outputs

- interpolated AnnData objects and trajectory metadata under
  `tutorial_outputs/arista/interpolation/`
- the classifier cache at `tutorial_outputs/arista/classifier_resmlp.pt`
- sparse communication files under `tutorial_outputs/arista/communication/`
- composition, velocity, growth, ligand--receptor, and distribution-metric
  tables displayed by the notebook
