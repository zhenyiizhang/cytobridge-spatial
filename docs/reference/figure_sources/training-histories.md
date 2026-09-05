---
orphan: true
---

# Analysis inputs: Supplementary Figure S46: training histories

The [figure notebook](../../tutorials/paper_figures/training_histories.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. record training objectives (S46)

```text
cytobridge workflow --config <dataset> --step preprocess --step train --train --input-h5ad <raw.h5ad> --output-dir <run> --device cuda:0
```

Start with: `raw H5AD, full-model dataset configuration, and matching LR database`

Writes: `<run>/training/training_history.csv; stage checkpoints; <run>/training/training_run_summary.json`

Next: `collect histories`




### 2. collect the five completed histories (S46)

```text
python scripts/collect_training_history_inputs.py \
  --run zebrafish=<zebrafish-run>/training \
  --run mosta=<mosta-run>/training \
  --run arista=<arista-run>/training \
  --run admouse=<admouse-run>/training \
  --run chicken_heart=<heart-run>/training \
  --output-dir <s46-inputs>
```

Start with: `training_history.csv from each of the five completed runs`

Writes: `<s46-inputs>/arista_training_history.csv; panel_metrics.csv; manifest.json`

Next: `draw S46`




### 3. smooth within stage and draw (S46)

```text
cytobridge figure training-histories --results-dir <s46-inputs> --output-dir <figure-dir>
```

Start with: `<s46-inputs>/arista_training_history.csv; panel_metrics.csv; manifest.json`

Writes: `representative_training_curves.pdf/.png and displayed stage metrics`
