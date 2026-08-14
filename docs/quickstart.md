# Quickstart

Start by inspecting the built-in workflow. A dry run reads parameters and
missing inputs but performs no computation.

```bash
cytobridge workflow --config zebrafish --dry-run
cytobridge workflow --config admouse --dry-run --json
cytobridge workflow --config chicken_heart --dry-run
```

## Analyze an existing model

```bash
cytobridge workflow --config zebrafish --step downstream \
  --aligned-h5ad /data/zebrafish_aligned.h5ad \
  --model-dir /models/zebrafish_alpha0015 \
  --output-dir /results/zebrafish \
  --device cuda
```

The shared downstream step produces interpolated slices, classifier metrics,
time-slice velocity, growth, composition, sparse spatial-attention summaries, readable
tables, and standard figures. It does not silently train a model.

Current predictor-gated checkpoints embed the edge predictor and can be copied
between machines. Older predictor-gated current-format checkpoints without
embedded weights also need `--edge-predictor-path /path/to/edge_model.pt`.

## Optional gene and ligand–receptor analyses

```bash
cytobridge workflow --config zebrafish --step downstream \
  --aligned-h5ad /data/zebrafish_aligned.h5ad \
  --model-dir /models/zebrafish_alpha0015 \
  --output-dir /results/zebrafish \
  --reference-h5ad /data/zebrafish_aligned.h5ad \
  --gene-dynamics \
  --lr-database /data/CellChatDB.ligrec.csv \
  --lr-complex-mode min \
  --device cuda
```

The reference H5AD must retain the fitted PCA loadings in `varm['PCs']`.
Current preprocessing also persists the exact center in `var['pca_center']`;
historical files without it fail closed by default. If and only if the file is
the complete original PCA-fit population, pass
`--allow-complete-reference-pca-center-fallback`; its `X` column mean is then
accepted only after reproducing the saved PCA coordinates.
Ligand–receptor complexes require every subunit;
the minimum subunit is the formal rule. The geometric mean is available only as
an explicit sensitivity analysis.

## Deliberately start a corrected raw-H5AD training run

```bash
cytobridge workflow --config mosta --train \
  --input-h5ad /data/mosta_raw_counts.h5ad \
  --output-dir /results/mosta \
  --device cuda
```

Training requires `--train`; it is never inferred from a missing model. This
command preprocesses and aligns the raw input, constructs interaction graphs
with the bundled mouse CellChatDB, trains a new edge predictor in that same
feature space, fits the six-stage dynamical model, and runs the shared
downstream chain. Supplying an old edge predictor to a raw-H5AD run is rejected.

Chicken heart first uses the anatomy-reviewed adapter shown in its dataset
tutorial. Then the same package workflow validates the fixed H5AD, fits the
graph and predictor, trains, and runs downstream without refitting coordinates:

```bash
cytobridge workflow --config chicken_heart --train \
  --input-h5ad /runs/chicken-input/chicken_heart_aligned_package.h5ad \
  --output-dir /results/chicken-heart \
  --device cuda:0
```

The AD command uses the same graph-label, edge-predictor, six-stage training,
and downstream chain. Its targeted panel contains seven strict complete pairs;
those labels fit the corrected main predictor, whose validation-selected
threshold is `0.9956824779510498`. This is explicitly panel-limited evidence,
not a global CCI screen.

## Run the matched AD no-LR-prior ablation

The corrected predictor-gated artifact is the main. The radius-only condition
is an explicit matched ablation and must select its own packaged contract:

```bash
cytobridge workflow --config admouse --step downstream \
  --training-config admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml \
  --aligned-h5ad /runs/admouse-no-lr/preprocess/admouse_aligned.h5ad \
  --model-dir /runs/admouse-no-lr/training \
  --output-dir /results/admouse-no-lr-reuse \
  --device cuda
```

This all-spatial profile intentionally contains no predictor path or threshold.
