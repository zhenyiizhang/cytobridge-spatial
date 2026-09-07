---
orphan: true
---

# LR-complex aggregation for Supplementary Figure S41

S41 compares minimum and geometric-mean aggregation of ligand-receptor
complex subunits. Both calculations use the same expression states and
model-derived communication matrices. Run these commands in the CytoBridge
code folder, replacing paths in angle brackets with your file locations.

## 1. Calculate the primary LR scores

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
selects another database for a new analysis. Keep LR calculation enabled by
omitting `--skip-lr`.

This command simulates and classifies the configured time slices, calculates
model-derived communication, and projects strict all-subunit LR scores. It
writes `<zebrafish-primary>/downstream/summary.json`, the expression-state
snapshots, `communication/communication_by_celltype.csv`, and
`ligand_receptor/pair_timecourse.csv` under that downstream directory. The
summary lists the files read in step 2, so keep the downstream directory
together. Use a new output directory for each run.

Repeat with the other three dataset configurations and their corresponding
aligned H5AD/model directories. This is downstream inference, not model fitting.

## 2. Compare the two complex rules

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

If an ARISTA summary lacks `simulation.slice_origins_by_time`, add
`--observed-time-points 0 1 2 3 4` for its five observed time points. For chicken
heart, the summary's `annotation_key: celltype_prediction` supplies the labels
for both observed and generated states.

## 3. Collect the four sensitivity tables

After running step 2 for all four datasets, collect their
`comparison/paired_scores.csv` files:

```text
python scripts/collect_figure_inputs.py s41 \
  --dataset-result zebrafish=<zebrafish-sensitivity> \
  --dataset-result mosta=<mosta-sensitivity> \
  --dataset-result arista=<arista-sensitivity> \
  --dataset-result chicken_heart=<chicken-heart-sensitivity> \
  --output-dir <s41-inputs>
```

The collected directory contains `<dataset>/paired_scores.csv` for each
dataset and `manifest.json`. Use this `<s41-inputs>` directory for the plot.

## 4. Summarize and draw S41

```text
python -m reproduction.paper_figures --figures 41 --results-dir <s41-inputs> --output-dir <figure-dir>
```

This reads the collected paired scores and saves `S41.pdf`, `S41.png`,
`tables/S41_top100_jaccard.csv`, and the per-time and dataset summaries in
`<figure-dir>`.

To execute the notebook against the same newly collected directory:

```bash
CYTOBRIDGE_LR_COMPLEX_RESULTS=<s41-inputs> \
  python scripts/execute_paper_notebooks.py --notebook lr_complex_aggregation \
  --output-dir <notebook-run>
```

Use the absolute path for `<s41-inputs>`: the notebook runner changes the
kernel's working directory to its separate run folder.

The [notebook](../../tutorials/paper_figures/lr_complex_aggregation.ipynb) reads
the same paired scores and passes `results.source_dir` to the S41 plotting
function, so its plot uses the directory selected above.
