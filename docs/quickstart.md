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

The reference H5AD must retain the fitted PCA loadings in `varm['PCs']` and
center in `var['pca_center']`. Ligand–receptor complexes require every subunit;
the minimum subunit is the formal rule. The geometric mean is available only as
an explicit sensitivity analysis.

## Deliberately start training

```bash
cytobridge workflow --config mosta --train --step downstream \
  --aligned-h5ad /data/mosta_aligned.h5ad \
  --edge-predictor-path /models/mosta_edge_predictor.pt \
  --output-dir /results/mosta \
  --device cuda
```

Training requires `--train`; it is never inferred from a missing model.
