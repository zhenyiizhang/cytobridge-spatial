---
orphan: true
---

# Analysis inputs: Supplementary Table 2: full-model compute cost

The [figure notebook](../../tutorials/paper_figures/compute_cost.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. measure each manuscript full-model run (Supplementary Table 2)

```text
cytobridge workflow --config <dataset> --step preprocess --step train --train --input-h5ad <raw.h5ad> --output-dir <run> --device cuda:0
```

Start with: `one raw H5AD and manuscript model configuration per dataset`

Writes: `training_run_summary.json with elapsed seconds, peak host RSS and peak PyTorch allocation`

Next: `collect the five measured runs`




### 2. collect the five training summaries (Supplementary Table 2)

```text
python scripts/collect_full_model_compute_cost.py --run admouse=<admouse-training>/training_run_summary.json --run arista=<arista-training>/training_run_summary.json --run chicken_heart=<heart-training>/training_run_summary.json --run mosta=<mosta-training>/training_run_summary.json --run zebrafish=<zebrafish-training>/training_run_summary.json --output-dir <compute-cost-results>
```

Start with: `five manuscript training_run_summary.json files`

Writes: `full_model_compute_cost.csv and manifest.json`

Next: `check and format the table`




### 3. check and format the collected table (Supplementary Table 2)

```text
python -m scripts.results.build_full_model_compute_cost_table --results-dir <compute-cost-results> --output-dir <formatted-table-run>
```

Start with: `full_model_compute_cost.csv and manifest.json`

Writes: `checked raw table plus formatted CSV and Markdown files`

Next: `format the display values in the notebook`




### 4. format the table (Supplementary Table 2)

```text
python scripts/execute_paper_notebooks.py --notebook compute_cost --output-dir <notebook-run>
```

Start with: `full_model_compute_cost.csv`

Writes: `full_model_compute_cost_formatted.csv/.md`

Next: `copy the displayed values to the TeX-native table`
