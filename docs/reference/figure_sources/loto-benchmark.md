---
orphan: true
---

# Five-dataset leave-one-timepoint-out benchmark

This tutorial prepares the data, trains models with one time point held out,
and draws S45 from the resulting comparisons. The downloaded models can be
used directly for inference in the dataset tutorials. Here, each CytoBridge
LOTO model is trained from scratch using the downloaded training configuration.
All calculations write to new directories under `outputs/`.

Follow [Installation](../../installation.md), then run the commands below in
the CytoBridge code folder. They process zebrafish, MOSTA, ARISTA, AD mouse,
and chicken heart.

## 1. Download the data and model configurations

```bash
for dataset in zebrafish mosta arista admouse chicken_heart; do
  python -m CytoBridge.datasets "$dataset" --output-dir paper_run
done
```

Each download creates `paper_run/data/<dataset>/` with `aligned.h5ad`,
`workflow.json`, and `model/config.yaml`. The [data page](../../data_checkpoints.md)
also provides the archives for manual download. Allow about 20 GB for
downloading and extracting MOSTA, plus space for the benchmark splits below.

## 2. Create the benchmark configurations

```bash
python -m scripts.spatiotemporal_benchmark.prepare_reader_configs \
  --data-root paper_run/data --output-dir outputs/benchmark_reader
```

This writes one YAML per dataset under `outputs/benchmark_reader/configs/`.
Each YAML selects that dataset's aligned data, time and annotation fields,
and training configuration. The `formal/<dataset>/training` links point to
the downloaded model directories without copying them.

Use this same configuration directory throughout the remaining steps.
Choose a different output directory when starting another benchmark run.

## 3. Split the data by held-out time point

```bash
python -m scripts.spatiotemporal_benchmark.run_unified_benchmark \
  --config-dir outputs/benchmark_reader/configs \
  --run-root outputs/benchmark_new prepare
```

For each dataset, `outputs/benchmark_new/<dataset>/inputs/` contains the
split files. For example, chicken heart has `loto_t1/` and `loto_t2/`.
Each folder contains a `train.h5ad` without the held-out cells,
`training_reference.npz`, `source_roster.npz`, and separate truth files.

Methods share a support of up to 800 source cells and a bootstrap of 5,000
starting particles. The 50 PCA features and two aligned spatial coordinates
remain fixed across folds. Because these representations were prepared using
all time points, this is a transductive LOTO comparison.

## 4. Train and predict

Set up the comparison methods using the instructions for
[dynamic methods](https://github.com/zhenyiizhang/cytobridge-spatial/blob/main/scripts/spatiotemporal_benchmark/dynamic/README.md)
and [static methods](https://github.com/zhenyiizhang/cytobridge-spatial/blob/main/scripts/spatiotemporal_benchmark/static_baselines/README.md).
The command below expects their code in `software/<method>/`. For a separate
Python environment, add a `--python METHOD=/path/to/env/bin/python` argument
after `run`. Change `cuda:2` to the GPU you want to use.

```bash
python -m scripts.spatiotemporal_benchmark.run_unified_benchmark \
  --config-dir outputs/benchmark_reader/configs \
  --run-root outputs/benchmark_new --software-root software run \
  --methods cytobridge stvcr stories mioflow moscot wot paste spateo \
  linear_centroid_shift exact_ot_displacement random_independent_pairs \
  --tracks loto --device cuda:2
```

For each CytoBridge fold, this calculates interaction graphs from the training
cells, fits an edge predictor, trains the six model stages, and predicts the
held-out time point. The training configuration supplies the architecture,
losses, and training schedule.

Fitted models are saved under `outputs/benchmark_new/<dataset>/fits/` and
predictions under `predictions/loto/<method>/t<target>/`. Check
`status/method_target_status.csv` and the corresponding `logs/` files for
failed or unfinished runs before continuing.

## 5. Evaluate the predictions

```bash
python -m scripts.spatiotemporal_benchmark.run_unified_benchmark \
  --config-dir outputs/benchmark_reader/configs \
  --run-root outputs/benchmark_new evaluate --tracks loto
```

This compares predictions with the held-out cells using five repeats of
1,024 projection directions for Sliced-W2. It writes repeat-level metrics to
`outputs/benchmark_new/<dataset>/evaluation/loto/` and target-level means to
`reports/loto/loto_target_summary.csv`. Methods that did not produce predictions
are recorded with their status instead of a numerical score.

## 6. Collect the results and draw S45

Use the five target-summary files from the preceding step:

```bash
python scripts/collect_figure_inputs.py s45 \
  --dataset-summary zebrafish=outputs/benchmark_new/zebrafish/reports/loto/loto_target_summary.csv \
  --dataset-summary mosta=outputs/benchmark_new/mosta/reports/loto/loto_target_summary.csv \
  --dataset-summary arista=outputs/benchmark_new/arista/reports/loto/loto_target_summary.csv \
  --dataset-summary admouse=outputs/benchmark_new/admouse/reports/loto/loto_target_summary.csv \
  --dataset-summary chicken_heart=outputs/benchmark_new/chicken_heart/reports/loto/loto_target_summary.csv \
  --protocol CytoBridge/results/data/loto_benchmark/protocol.json \
  --output-dir outputs/benchmark_tables
cytobridge figure loto-benchmark --results-dir outputs/benchmark_tables \
  --output-dir outputs/benchmark_figures
```

The collector writes `loto_target_stage_means.csv`,
`native_output_support.csv`, and `protocol.json` to `outputs/benchmark_tables/`.
The plot divides each method's Sliced-W2 by CytoBridge's value for the same
dataset, target, and space, then averages these ratios within each dataset.
The PDF and PNG are saved as `outputs/benchmark_figures/five_dataset_loto_benchmark.*`,
alongside `paired_loto_ratios.csv` and `loto_dataset_summary.csv`.

To add SpaTrack for S44, continue with the
[SpaTrack tutorial](../../tutorials/paper_figures/spatrack_benchmark.md), using
the same `outputs/benchmark_new/<dataset>/inputs/` folders and these new
CytoBridge summaries. To draw S45 from the included paper results, run the
[S45 notebook](../../tutorials/paper_figures/loto_benchmark.ipynb).
