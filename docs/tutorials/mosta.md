# MOSTA mouse organogenesis

**Notebook:** {download}`mosta.ipynb <dataset_workflows/mosta.ipynb>`

## Inputs

- `inputs/mosta_aligned.h5ad`, with the observation columns and embeddings
  named by the `mosta` workflow preset
- `inputs/mosta_model/`, containing an existing model checkpoint
- an edge-predictor checkpoint when `RUN_TRAINING=True`
- the ligand--receptor table returned by
  `cb.pp.bundled_graph_database_path("mosta")`, or a file set with
  `LR_DATABASE_OVERRIDE`

The notebook writes under `tutorial_outputs/mosta/`. Change the input and
output paths in its first section before running it.

## Package calls

The notebook reads settings with `load_workflow_config("mosta")`. It can run
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
cytobridge workflow --config mosta --dry-run
jupyter lab docs/tutorials/dataset_workflows/mosta.ipynb
```

## Figure reproduction

Main Figure 4 is rebuilt by
{download}`main_figure_4.ipynb <paper_figures/main_figure_4.ipynb>` from five
released vector panels. Supplementary Figures S9–S16 are exported under their
current numbering by
{download}`mosta_figures.ipynb <paper_figures/mosta_figures.ipynb>`. The
calculation scripts, renderers, model files, compact numerical inputs, and
vector pages used by both notebooks are stored in
`release_artifacts/mosta_package_native_corrected_20260826_v1/`.

## Outputs

- interpolated AnnData objects and trajectory metadata under
  `tutorial_outputs/mosta/interpolation/`
- the classifier cache at `tutorial_outputs/mosta/classifier_resmlp.pt`
- sparse communication files under `tutorial_outputs/mosta/communication/`
- composition, velocity, growth, ligand--receptor, and distribution-metric
  tables displayed by the notebook
