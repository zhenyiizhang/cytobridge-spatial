# Get started

CytoBridge takes a time series of expression counts and spatial coordinates,
fits a dynamical model, and calculates trajectories, growth, and cell–cell
interactions.

## Try a small example

Start with the [preprocessing notebook](tutorials/data_preparation/synthetic_preprocessing.ipynb).
It creates a small count matrix and plots the processed data. It runs on a CPU
without downloading a study dataset. This example covers preprocessing, not
model training.

## Train a model

After [installation](installation.md), put your counts H5AD in a working
directory. For example, with a MOSTA input:

```bash
cytobridge workflow --config mosta --train \
  --input-h5ad data/mosta_raw.h5ad \
  --output-dir outputs/mosta --device cuda
```

This is **one command**. The backslashes continue it across lines.

It runs the following steps in order:

1. Prepare expression features and align the spatial coordinates.
2. Fit the LR edge predictor and train the dynamical model.
3. Calculate the downstream results and draw the standard plots.

**You do not need to run `preprocess` first.** The `--train` command includes
preprocessing. The separate `--step preprocess` option is only for inspecting
the prepared data without training.

The command above expects a prepared counts H5AD, not a downloaded archive.
The [data guide](data_checkpoints.md) lists the required fields and the
availability of the paper's inputs.

## Find the outputs

All results from the command are written under `outputs/mosta/`:

```text
outputs/mosta/
├── preprocess/mosta_aligned.h5ad
├── training/
└── downstream/
    ├── summary.json
    ├── slice_data/
    ├── growth/
    ├── velocity/
    ├── communication/
    └── figures/
```

Training reads the aligned H5AD from `preprocess/`. Downstream analysis reads
that same H5AD and the fitted model in `training/`. The plots in
`downstream/figures/` are drawn from this run's results.

## Choose your next step

- **Your own experiment:** [adapt the input fields and settings](tutorials/your_data.ipynb).
- **A paper dataset:** [follow its training notebook](tutorials/dataset_workflows/index.md).
- **An existing model:** [calculate results without retraining](reuse_model.md).
- **A paper figure:** [draw from the saved numerical results](tutorials/paper_figures/index.md).

The standard analysis plots are not automatically the assembled manuscript
pages. The figure tutorials give the additional calculations and layout steps
where these are available.
