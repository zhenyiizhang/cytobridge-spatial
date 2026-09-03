# Zebrafish decomposition stability

This directory contains the analysis and plotting code for Supplementary Figure S40. The analysis compares independently trained Zebrafish models and matched changes to the spatial neighborhood, expression-loss weight, and transport-to-mass weight ratio.

## Draw the figure from the included results

Run this command from the repository root:

```bash
python release_artifacts/zebrafish_decomposition_stability_20260903/code/plot_zebrafish_decomposition_stability_v2.py \
  --panel-data release_artifacts/zebrafish_decomposition_stability_20260903/panel_data_final \
  --output-dir results/zebrafish_decomposition_stability_figure
```

The plotting script reads the included numerical tables and writes a vector PDF and a PNG preview. It does not load a completed figure.

## Repeat the full analysis

Start from the aligned Zebrafish H5AD file and the fixed ligand--receptor edge predictor recorded in `provenance/final_manifest_v6.md`. Choose a new, empty run directory and replace the paths in angle brackets below.

First, write the 14 configurations used in the analysis. Thirteen are shown in the figure. The additional 1:10 transport-to-mass setting is retained in the archive but is not plotted.

```bash
python release_artifacts/zebrafish_decomposition_stability_20260903/code/prepare_zebrafish_decomposition_stability.py \
  --template release_artifacts/zebrafish_decomposition_stability_20260903/configs/formal_seed42_cutoff1p0.yaml \
  --config-dir <run>/configs \
  --remote-run-root <run>
```

Train every configuration. `--gpu-count` is the number of available CUDA devices.

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

Evaluate the fitted components on the same observed cells and model times.

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

Create the numerical tables, then draw the figure from those tables.

```bash
python release_artifacts/zebrafish_decomposition_stability_20260903/code/summarize_zebrafish_decomposition_stability.py \
  --evaluation-root <run>/evaluation_edge_centered \
  --output-dir <run>/panel_data

python release_artifacts/zebrafish_decomposition_stability_20260903/code/plot_zebrafish_decomposition_stability_v2.py \
  --panel-data <run>/panel_data \
  --output-dir <run>/figure
```

Training and model evaluation require CUDA. Redrawing the figure from the included tables does not.

## Directory contents

- `code/`: configuration preparation, training, evaluation, summarization, and plotting
- `configs/`: the 14 exact model configurations used in the analysis
- `panel_data_final/`: the numerical tables used to draw every panel
- `figure/`: the vector PDF and PNG preview used in Supplementary Figure S40
- `caption.md`: the corresponding SI caption
- `provenance/`: source paths, model release, rebuild command, and file hashes
- `experiment_plan.json`: the complete condition matrix and training paths used for the reported run
- `archive_manifest.json`: SHA-256 hashes for the code, configurations, tables, figures, and provenance files

The completed model runs remain in the versioned server directory stated in the provenance record. They are not duplicated here because the evaluated cell-level arrays are substantially larger than the compact figure archive.
