# AD mouse disease progression

**Notebook:** {download}`admouse.ipynb <dataset_workflows/admouse.ipynb>`

## Inputs

- `inputs/admouse_aligned.h5ad`, with the observation columns and embeddings
  named by the `admouse` workflow preset
- `inputs/admouse_model/`, containing an existing model checkpoint
- an edge-predictor checkpoint when `RUN_TRAINING=True`
- the ligand--receptor table returned by
  `cb.pp.bundled_graph_database_path("admouse")`, or a file set with
  `LR_DATABASE_OVERRIDE`

The notebook writes under `tutorial_outputs/admouse/`. Change the input and
output paths in its first section before running it.

## Package calls

The notebook reads settings with `load_workflow_config("admouse")`. It can run
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
cytobridge workflow --config admouse --dry-run
jupyter lab docs/tutorials/dataset_workflows/admouse.ipynb
```

## Outputs

- interpolated AnnData objects and trajectory metadata under
  `tutorial_outputs/admouse/interpolation/`
- the classifier cache at `tutorial_outputs/admouse/classifier_resmlp.pt`
- sparse communication files under `tutorial_outputs/admouse/communication/`
- composition, velocity, growth, ligand--receptor, and distribution-metric
  tables displayed by the notebook
