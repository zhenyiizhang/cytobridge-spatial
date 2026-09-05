---
orphan: true
---

# Analysis inputs: Supplementary Figure S42: LR-prior and interaction ablations

The [figure notebook](../../tutorials/paper_figures/interaction_ablation.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. compare inference with and without interaction (S42c-d)

```text
python scripts/paper_figures/interaction_ablation/run_comparison.py --dataset arista --model-dir <run>/training --input-manifest <benchmark-run>/arista/inputs/manifest.json --code-root . --output <inference-results>/arista --seeds 42 43 44 --gpu-metrics --device cuda:0
```

Start with: `the fitted Full model and its matching benchmark input manifest (full_data split)`

Writes: `<inference-results>/arista/metrics.csv and manifest.json, plus paired predictions for each seed`

Next: `repeat for zebrafish, mosta, admouse, and chicken_heart`


No training is repeated. Only the interaction-network output is disabled. Use the corresponding model and input manifest for each dataset.



### 2. collect completed comparisons (S42)

```text
python scripts/collect_figure_inputs.py s42 --no-lr-table <matched-ablation-report>/paired_target_deltas.csv --inference-results-dir <inference-results> --output-dir <s42-inputs>
```

Start with: `the Full/No-LR report and all five completed inference directories`

Writes: `no_lr_paired_target_deltas.csv, inference_metrics.csv, and model records`

Next: `draw all four panels`




### 3. recalculate summaries and draw (S42)

```text
cytobridge figure interaction-ablation --results-dir <s42-inputs> --output-dir <figure-dir>
```

Start with: `the collected numerical results`

Writes: `interaction_ablation.pdf/.png, paired seed/target tables, and caption statistics`




## Archived source files

These entries locate implementation files. They are reference material, not additional commands.


### S42a-b: fit Full and No-LR-prior models

`scripts/run_matched_ablation_matrix.py and scripts/run_matched_ablation_benchmark_evaluation.py`

Input: matched aligned inputs and Full/No-LR training configurations

Output: paired_target_deltas.csv from the completed matched-ablation report
