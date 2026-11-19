# Zebrafish decomposition stability

Supplementary Figure S10 compares the fitted decomposition across independent training seeds and after changing selected model settings.

```{image} ../../_static/zebrafish_decomposition_stability_s40.png
:alt: Stability of the interaction and intrinsic-context decomposition in zebrafish
:width: 760px
:align: center
```

## Data and code

The model settings, result tables and calculation scripts are in
`paper_resources/zebrafish_stability`.

## Draw the figure

From the repository root, run:

```bash
python -m reproduction.current_figures --figures 10 \
  --output-dir outputs/zebrafish_decomposition_stability
```

The command summarizes the supplied comparison tables and draws S10 as PDF and PNG.

## Repeat the full analysis

To repeat training, use the aligned Zebrafish H5AD file and its ligand–receptor
edge predictor from [Data and models](../../data_checkpoints.md). Replace the
paths in angle brackets below with your files and choose a new run directory.

Write the training configurations:

```bash
python paper_resources/zebrafish_stability/code/prepare_zebrafish_decomposition_stability.py \
  --template paper_resources/zebrafish_stability/configs/formal_seed42_cutoff1p0.yaml \
  --config-dir <run>/configs \
  --remote-run-root <run>
```

Train the configuration matrix:

```bash
python paper_resources/zebrafish_stability/code/run_zebrafish_decomposition_stability.py \
  --plan <run>/experiment_plan.json \
  --python python \
  --release . \
  --trainer paper_resources/zebrafish_stability/code/train_zebrafish_decomposition_stability.py \
  --aligned-h5ad <zebrafish_aligned.h5ad> \
  --edge-predictor <zebrafish_edge_model.pt> \
  --gpu-count 1
```

Evaluate each fitted model:

```bash
python paper_resources/zebrafish_stability/code/run_zebrafish_decomposition_evaluation_matrix.py \
  --run-root <run> \
  --training-status <run>/status/training_matrix_status.json \
  --python python \
  --release . \
  --evaluator paper_resources/zebrafish_stability/code/evaluate_zebrafish_decomposition_stability.py \
  --aligned-h5ad <zebrafish_aligned.h5ad> \
  --edge-predictor <zebrafish_edge_model.pt> \
  --gpu-count 1
```

Create the tables and draw the figure:

```bash
python paper_resources/zebrafish_stability/code/summarize_zebrafish_decomposition_stability.py \
  --evaluation-root <run>/evaluation_edge_centered \
  --output-dir <run>/panel_data

python paper_resources/zebrafish_stability/code/plot_zebrafish_decomposition_stability_v2.py \
  --panel-data <run>/panel_data \
  --output-dir <run>/figure
```

Training and model evaluation require CUDA. Drawing the archived figure requires only Python, NumPy, pandas, and Matplotlib.
