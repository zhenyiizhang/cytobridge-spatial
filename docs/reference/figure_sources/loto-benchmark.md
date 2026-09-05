---
orphan: true
---

# Analysis inputs: Supplementary Figure S45: five-dataset LOTO benchmark

The [figure notebook](../../tutorials/paper_figures/loto_benchmark.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. prepare held-out benchmark inputs (S45)

```text
python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> prepare
```

Start with: `the same held-out target stages and fixed starting states for all methods`

Writes: `held-out inputs and fixed starting states for every target stage`

Next: `run the compared methods`




### 2. run the compared methods (S45)

```text
python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> --software-root <method-checkouts> run --methods cytobridge stvcr stories mioflow moscot wot paste spateo linear_centroid_shift exact_ot_displacement random_independent_pairs --tracks loto --device cuda
```

Start with: `held-out inputs, fixed starting states, and installed comparison methods`

Writes: `one prediction folder for each method, dataset, and target stage`

Next: `evaluate the predictions`




### 3. evaluate held-out predictions (S45)

```text
python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> evaluate --tracks loto
```

Start with: `held-out truth and method predictions`

Writes: `repeat-level metrics and a table showing which method completed each target`

Next: `merge the updated ARISTA and heart results`




### 4. collect the five completed target summaries (S45)

```text
python scripts/collect_figure_inputs.py s45 \
  --dataset-summary zebrafish=<benchmark-run>/zebrafish/reports/loto/loto_target_summary.csv \
  --dataset-summary mosta=<benchmark-run>/mosta/reports/loto/loto_target_summary.csv \
  --dataset-summary arista=<benchmark-run>/arista/reports/loto/loto_target_summary.csv \
  --dataset-summary admouse=<benchmark-run>/admouse/reports/loto/loto_target_summary.csv \
  --dataset-summary chicken_heart=<benchmark-run>/chicken_heart/reports/loto/loto_target_summary.csv \
  --protocol <s45-protocol.json> \
  --output-dir <s45-inputs>
```

Start with: `the loto_target_summary.csv written for each dataset by the benchmark summarizer`

Writes: `<s45-inputs>/loto_target_stage_means.csv; native_output_support.csv; protocol.json; manifest.json`

Next: `draw S45`


Use the protocol.json included with the S45 paper results unless the benchmark contract itself has changed. Replace the ARISTA or chicken-heart summary path with a retrained run to update that dataset without changing the other three.



### 5. calculate paired ratios and draw (S45)

```text
cytobridge figure loto-benchmark --results-dir <s45-inputs> --output-dir <figure-dir>
```

Start with: `the collected S45 input directory`

Writes: `five_dataset_loto_benchmark.pdf/.png and ratio/summary tables`
