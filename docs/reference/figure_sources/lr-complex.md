---
orphan: true
---

# Analysis inputs: Supplementary Figure S41: LR-complex aggregation

The [figure notebook](../../tutorials/paper_figures/lr_complex_aggregation.ipynb) draws the figure from saved numerical results. The steps below calculate those inputs from data and fitted models.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. generate the primary downstream LR result

For each of `zebrafish`, `mosta`, `arista`, and `chicken_heart`, start from the
aligned expression/PCA AnnData and matching fitted model from that dataset's
workflow. For example, after the zebrafish preprocessing and training tutorial:

```bash
cytobridge workflow --config zebrafish --step downstream \
  --aligned-h5ad <zebrafish-run>/preprocess/zebrafish_aligned.h5ad \
  --model-dir <zebrafish-run>/training \
  --reference-h5ad <zebrafish-run>/preprocess/zebrafish_aligned.h5ad \
  --output-dir <zebrafish-primary> --lr-complex-mode min --device cuda:2
```

With downloaded model inputs, use `data/zebrafish/aligned.h5ad` and
`data/zebrafish/model` in those same arguments. The included species-matched LR
database is selected by `--config`; `--lr-database <database.csv>` explicitly
selects another database for a new analysis. Do not set `--skip-lr`.

This command simulates and classifies the configured time slices, calculates
model-derived communication, and projects strict all-subunit LR scores. It
writes `<zebrafish-primary>/downstream/summary.json`, the expression-state
snapshots, `communication/communication_by_celltype.csv`, and
`ligand_receptor/pair_timecourse.csv` under that downstream directory. The
summary binds the exact paths; keep those files together and do not substitute
a summary from another fitted model. Use a new output directory for each run.

Repeat with the other three dataset configurations and their corresponding
aligned H5AD/model directories. This is downstream inference, not model fitting.

### 2. calculate both complex rules (S41)

```text
python scripts/run_lr_complex_aggregation_sensitivity.py \
  --workflow-summary <zebrafish-primary>/downstream/summary.json \
  --output-dir <zebrafish-sensitivity>
```

The program reuses the saved snapshots and communication matrix. It recomputes
the minimum gate and requires agreement with the primary score table before
changing only complex aggregation to the geometric mean. It writes
`<zebrafish-sensitivity>/comparison/paired_scores.csv` and `run_manifest.json`.
Run this once per dataset; no new simulations or classifier fits occur here.

For a legacy ARISTA native summary lacking `simulation.slice_origins_by_time`,
the original native producer declares observed times 0, 1, 2, 3, 4. Pass
`--observed-time-points 0 1 2 3 4` explicitly. No times are guessed. The same
primary-score equality check still applies. The chicken-heart native summary's
`annotation_key: celltype_prediction` is used for both observed and generated
states; it must not be silently replaced by another annotation column.




### 3. collect the four completed sensitivity tables (S41)

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




### 4. summarize and draw (S41)

```text
python -m reproduction.paper_figures --figures 41 --results-dir <s41-inputs> --output-dir <figure-dir>
```

Start with: `the collected S41 input directory`

Writes: `S41.pdf/.png; tables/S41_top100_jaccard.csv; per-time and dataset summary CSVs`

To execute the notebook against the same newly collected directory:

```bash
CYTOBRIDGE_LR_COMPLEX_RESULTS=<s41-inputs> \
  python scripts/execute_paper_notebooks.py --notebook lr_complex_aggregation \
  --output-dir <notebook-run>
```

Use the absolute path for `<s41-inputs>`: the notebook runner changes the
kernel's working directory to its separate run folder.

The notebook passes `results.source_dir` to the S41 plotting function.
