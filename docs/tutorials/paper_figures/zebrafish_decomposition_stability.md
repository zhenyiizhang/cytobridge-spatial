# Zebrafish decomposition stability

Supplementary Figure S40 compares the fitted decomposition across independent training seeds and after changing selected model settings.

```{image} ../../_static/zebrafish_decomposition_stability_s40.png
:alt: Stability of the interaction and intrinsic-context decomposition in zebrafish
:width: 760px
:align: center
```

## Files used here

The complete figure archive is in `release_artifacts/zebrafish_decomposition_stability_20260903`. It contains the training configurations, analysis scripts, numerical panel tables, figure PDF and PNG, and provenance record.

## Draw the figure

From the repository root, run:

```bash
python release_artifacts/zebrafish_decomposition_stability_20260903/code/plot_zebrafish_decomposition_stability_v2.py \
  --panel-data release_artifacts/zebrafish_decomposition_stability_20260903/panel_data_final \
  --output-dir results/zebrafish_decomposition_stability_figure
```

This command calculates the plotted positions from the included CSV tables and writes a new PDF and PNG. The plotting code does not import a completed figure.

## Repeat the full analysis

The full analysis starts from the aligned Zebrafish H5AD file and the fixed ligand-receptor edge predictor listed in the archive provenance. Choose a new, empty run directory and replace the paths in angle brackets below.

Write the training configurations:

```bash
python release_artifacts/zebrafish_decomposition_stability_20260903/code/prepare_zebrafish_decomposition_stability.py \
  --template release_artifacts/zebrafish_decomposition_stability_20260903/configs/formal_seed42_cutoff1p0.yaml \
  --config-dir <run>/configs \
  --remote-run-root <run>
```

Train the configuration matrix:

```bash
python release_artifacts/zebrafish_decomposition_stability_20260903/code/run_zebrafish_decomposition_stability.py \
  --plan <run>/experiment_plan.json \
  --python python \
  --release . \
  --trainer release_artifacts/zebrafish_decomposition_stability_20260903/code/train_zebrafish_decomposition_stability.py \
  --aligned-h5ad <zebrafish_aligned.h5ad> \
  --edge-predictor <zebrafish_edge_model.pt> \
  --gpu-count 1
```

Evaluate each fitted model:

```bash
python release_artifacts/zebrafish_decomposition_stability_20260903/code/run_zebrafish_decomposition_evaluation_matrix.py \
  --run-root <run> \
  --training-status <run>/status/training_matrix_status.json \
  --python python \
  --release . \
  --evaluator release_artifacts/zebrafish_decomposition_stability_20260903/code/evaluate_zebrafish_decomposition_stability.py \
  --aligned-h5ad <zebrafish_aligned.h5ad> \
  --edge-predictor <zebrafish_edge_model.pt> \
  --gpu-count 1
```

Create the tables and draw the figure:

```bash
python release_artifacts/zebrafish_decomposition_stability_20260903/code/summarize_zebrafish_decomposition_stability.py \
  --evaluation-root <run>/evaluation_edge_centered \
  --output-dir <run>/panel_data

python release_artifacts/zebrafish_decomposition_stability_20260903/code/plot_zebrafish_decomposition_stability_v2.py \
  --panel-data <run>/panel_data \
  --output-dir <run>/figure
```

Training and model evaluation require CUDA. Drawing the archived figure requires only Python, NumPy, pandas, and Matplotlib.
