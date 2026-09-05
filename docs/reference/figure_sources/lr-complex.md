---
orphan: true
---

# Analysis inputs: Supplementary Figure S41: LR-complex aggregation

The [figure notebook](../../tutorials/paper_figures/lr_complex_aggregation.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. calculate both complex rules (S41)

```text
python scripts/run_lr_complex_aggregation_sensitivity.py --workflow-summary <downstream/summary.json> --output-dir <sensitivity>
```

Start with: `each dataset downstream LR trajectories and strict all-subunit coverage`

Writes: `<sensitivity>/comparison/paired_scores.csv and run record`

Next: `merge four datasets`




### 2. collect the four completed sensitivity tables (S41)

```text
python scripts/collect_figure_inputs.py s41 \
  --dataset-result zebrafish=<zebrafish-sensitivity> \
  --dataset-result mosta=<mosta-sensitivity> \
  --dataset-result arista=<arista-sensitivity> \
  --dataset-result chicken_heart=<chicken-heart-sensitivity> \
  --output-dir <s41-inputs>
```

Start with: `comparison/paired_scores.csv from each completed sensitivity run`

Writes: `<s41-inputs>/<dataset>/paired_scores.csv and manifest.json`

Next: `draw S41`




### 3. summarize and draw (S41)

```text
cytobridge figure lr-complex --results-dir <s41-inputs> --output-dir <figure-dir>
```

Start with: `the collected S41 input directory`

Writes: `per-time and dataset summary CSVs; lr_complex_aggregation.pdf/.png`
