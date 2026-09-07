---
orphan: true
---

# Analysis inputs: Supplementary Table 2: full-model compute cost

The [figure notebook](../../tutorials/paper_figures/compute_cost.ipynb) draws the figure from saved numerical results. The steps below calculate those inputs from data and fitted models.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. measure each manuscript full-model run (Supplementary Table 2)

```text
cytobridge workflow --config <dataset> --step preprocess --step train --train --input-h5ad <raw.h5ad> --output-dir <run> --device cuda:0
```

Start with: `one raw H5AD and manuscript model configuration per dataset`

Writes: `<run>/training/training_run_summary.json` with elapsed seconds, peak host RSS and peak PyTorch allocation. The raw H5AD is preprocessed once by this command; preprocessing time is excluded from the reported training time.

Next: `collect the five measured runs`




### 2. collect the five training summaries (Supplementary Table 2)

```text
python scripts/collect_full_model_compute_cost.py --run admouse=<admouse-training>/training_run_summary.json --run arista=<arista-training>/training_run_summary.json --run chicken_heart=<heart-training>/training_run_summary.json --run mosta=<mosta-training>/training_run_summary.json --run zebrafish=<zebrafish-training>/training_run_summary.json --output-dir <compute-cost-results>
```

Start with: `five manuscript training_run_summary.json files`

The paper-format collector requires one NVIDIA GeForce RTX 4090 D for every run, the hardware recorded in Supplementary Table 2. Measurements made on another GPU are separate measurements. To collect those, add `--new-measurements` to this command. That mode writes `new_compute_cost_measurements.csv`, `new_compute_cost_table.csv` (including each row's actual GPU), and `new_measurements_manifest.json`; it does not create paper-table inputs or replace Supplementary Table 2. The numerical units and formatting remain seconds→minutes and MiB→GiB.

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
CYTOBRIDGE_COMPUTE_COST_RESULTS=<compute-cost-results> \
  python scripts/execute_paper_notebooks.py --notebook compute_cost --output-dir <notebook-run>
```

Use the absolute path for `<compute-cost-results>`: the notebook runner changes
the kernel's working directory to its separate run folder.

Start with: `full_model_compute_cost.csv`

The notebook reads this environment variable in its first cell and keeps that selected result object through formatting and export. Omit the variable only when redrawing the included paper measurements.

Writes: `outputs/full_model_compute_cost_notebook/full_model_compute_cost_table.csv` and `.md` within the notebook run directory.

The independent `--new-measurements` output can be inspected directly with `pd.read_csv("<new-measurements>/new_compute_cost_table.csv")`; it is not an input to the paper-only notebook loader.
