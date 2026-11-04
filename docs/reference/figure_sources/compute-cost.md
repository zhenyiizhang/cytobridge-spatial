---
orphan: true
---

# Full-model compute cost for Supplementary Table 2

This route measures training time and memory for the five dataset workflows,
then combines the measurements into a table. Run the commands in the CytoBridge
code folder and replace paths in angle brackets with your file locations.

## 1. Measure each full-model training run

Run this command once for each dataset configuration: `admouse`, `arista`,
`chicken_heart`, `mosta`, and `zebrafish`. Each run reads that dataset's raw H5AD
and writes to a separate new `<run>` directory.

```text
cytobridge workflow --config <dataset> --step preprocess --step train --train --input-h5ad <raw.h5ad> --output-dir <run> --device cuda:0
```

The command preprocesses the H5AD and trains the full model. Its
`<run>/training/training_run_summary.json` records elapsed training seconds,
peak host RSS, and peak PyTorch allocation. Preprocessing time is excluded from
the reported training time.

## 2. Combine the five training summaries

Pass the five `training_run_summary.json` files from step 1 to the collector:

```text
python scripts/collect_full_model_compute_cost.py --run admouse=<admouse-training>/training_run_summary.json --run arista=<arista-training>/training_run_summary.json --run chicken_heart=<heart-training>/training_run_summary.json --run mosta=<mosta-training>/training_run_summary.json --run zebrafish=<zebrafish-training>/training_run_summary.json --output-dir <compute-cost-results>
```

This writes `full_model_compute_cost.csv` and `manifest.json` to
`<compute-cost-results>`. Supplementary Table 2 uses one NVIDIA GeForce RTX
4090 D for each run, so this table format requires that hardware.

For measurements on another GPU, add `--new-measurements` to the collector
command. It saves `new_compute_cost_measurements.csv`,
`new_compute_cost_table.csv`, and `new_measurements_manifest.json` in the chosen
output directory. The formatted table includes the GPU for each run and converts
seconds to minutes and MiB to GiB. Read it directly with
`pd.read_csv("<new-measurements>/new_compute_cost_table.csv")`; steps 3 and 4
use the RTX 4090 D table format.

## 3. Format the collected table

The table builder reads `full_model_compute_cost.csv` and `manifest.json` from
the preceding step:

```text
python -m scripts.results.build_full_model_compute_cost_table --results-dir <compute-cost-results> --output-dir <formatted-table-run>
```

It saves the raw `full_model_compute_cost.csv` and formatted
`full_model_compute_cost_table.csv` and `.md` in `<formatted-table-run>`.

## 4. Display the table in the notebook

The [compute-cost notebook](../../tutorials/paper_figures/compute_cost.ipynb)
provides the same formatting interactively. Point it to the collected raw
results from step 2:

```text
CYTOBRIDGE_COMPUTE_COST_RESULTS=<compute-cost-results> \
  python scripts/execute_paper_notebooks.py --notebook compute_cost --output-dir <notebook-run>
```

Use the absolute path for `<compute-cost-results>`: the notebook runner changes
the kernel's working directory to its separate run folder.

The first cell loads that directory, and the remaining cells format and export
the same results. Omitting the variable loads the included paper measurements.
The notebook writes
`outputs/full_model_compute_cost_notebook/full_model_compute_cost_table.csv`
and `.md` within `<notebook-run>`.
