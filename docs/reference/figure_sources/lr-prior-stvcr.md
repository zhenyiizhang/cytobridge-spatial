---
orphan: true
---

# Analysis inputs: Supplementary Figure S42: LR-prior ablation and stVCR comparison

The [figure notebook](https://github.com/zhenyiizhang/cytobridge-spatial/blob/main/docs/tutorials/paper_figures/lr_prior_ablation_stvcr.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. prepare held-out benchmark inputs (S42c-d)

```text
python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> prepare
```

Start with: `the same held-out target stages and a fixed set of 5,000 starting states`

Writes: `held-out inputs and fixed starting states for every target stage`

Next: `run CytoBridge and stVCR`




### 2. run held-out CytoBridge and stVCR (S42c-d)

```text
python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> --software-root <method-checkouts> run --methods cytobridge stvcr --tracks loto --device cuda
```

Start with: `held-out inputs, fixed starting states, and the two installed methods`

Writes: `one prediction folder for each method, dataset, and target stage`

Next: `evaluate the predictions`




### 3. evaluate held-out predictions (S42c-d)

```text
python -m scripts.spatiotemporal_benchmark.run_unified_benchmark --datasets zebrafish mosta arista admouse chicken_heart --model-runs <matched-model-runs> --run-root <benchmark-run> evaluate --tracks loto
```

Start with: `held-out truth and CytoBridge/stVCR predictions`

Writes: `target-stage means and a table showing which method completed each target`

Next: `pair the results for S42`




### 4. collect the five completed LOTO target summaries (S42)

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

Next: `combine the No-LR and stVCR rows`


Use the protocol.json included with the S45 paper results unless the benchmark contract itself has changed.



### 5. combine the matched rows (S42)

```text
python scripts/collect_figure_inputs.py s42 \
  --no-lr-table <matched-ablation-report>/paired_target_deltas.csv \
  --loto-results-dir <s45-inputs> \
  --output-dir <s42-inputs>
```

Start with: `paired_target_deltas.csv from the matched Full/No-LR report and the collected S45 input directory`

Writes: `<s42-inputs>/no_lr_paired_target_deltas.csv; stvcr_paired_target_deltas.csv; manifest.json`

Next: `draw S42`




### 6. summarize and draw (S42)

```text
cytobridge figure lr-prior-stvcr --results-dir <s42-inputs> --output-dir <figure-dir>
```

Start with: `the collected S42 input directory`

Writes: `lr_prior_stvcr_comparison.pdf/.png and panel summaries`




## Archived source files

These entries locate implementation files. They are reference material, not additional commands.


### S42a-b: fit and evaluate Full and No-LR models

`scripts/run_matched_ablation_matrix.py and scripts/run_matched_ablation_benchmark_evaluation.py`

Input: five manuscript aligned H5ADs; matched Full/No-LR configs; seed 42

Output: per-arm full_data_metrics_long.csv and evaluation manifests
