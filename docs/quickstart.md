# Quickstart

Start by inspecting the built-in workflow. A dry run reads parameters and
missing inputs but performs no computation.

```bash
cytobridge workflow --config zebrafish --dry-run
cytobridge workflow --config admouse --dry-run --json
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
time-slice velocity, growth, composition, sparse communication, readable
tables, and standard figures. It does not silently train a model.

Current checkpoints embed the edge predictor and can be copied between
machines. Older current-format checkpoints without embedded predictor weights
also need `--edge-predictor-path /path/to/edge_model.pt`.

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
