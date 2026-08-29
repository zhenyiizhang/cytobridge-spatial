# Quickstart

Review the planned steps before providing data or starting a fit:

```bash
cytobridge workflow --config zebrafish --check
cytobridge workflow --config admouse --check --json
cytobridge workflow --config chicken_heart --check
```

`--check` prints the selected configuration, steps, options, and intended
paths. It does not open an H5AD, verify its fields, preprocess data, train a
model, or write outputs. AnnData validation begins when preprocessing runs.

`--train` allows model fitting. `--step train` selects the training stage when
you choose stages explicitly; it is not a replacement for `--train`. If the
configuration keeps downstream as its default and preprocessing is enabled,
`--train` with a raw H5AD and no `--step` runs preprocessing, training, and
downstream analysis in that order. This applies to the included configurations
and to copies exported from them.

In short: use `--train` with raw data for the complete run; use `--step train
--train` only when you already have an aligned H5AD and its matching edge
predictor.

## Choose where to start

| What you have | Start with | Main outputs |
| --- | --- | --- |
| A raw H5AD and a new model to fit | `--train --input-h5ad ... --output-dir ...` | `preprocess/`, `training/`, and `downstream/` |
| A raw H5AD whose alignment you want to inspect without training | `--step preprocess --input-h5ad ... --output-dir ...` | An aligned H5AD and preprocessing records only |
| An aligned H5AD and its matching saved edge predictor | `--step train --train --aligned-h5ad ... --edge-predictor-path ... --edge-predictor-threshold ... --output-dir ...` | `training/` |
| An aligned H5AD and its matching trained model | `--step downstream --aligned-h5ad ... --model-dir ... --output-dir ...` | `downstream/` |

Here, a **raw H5AD** is the AnnData file before CytoBridge alignment. An
**aligned H5AD** is the file written by preprocessing and read by training. The
**training directory** contains the fitted model, and **downstream analysis**
uses that model to calculate cell state, velocity, growth, composition, and
cell--cell communication results. A ligand--receptor edge predictor selects
nearby cell pairs that can contribute to the interaction graph.

The preprocessing-only command is optional. It does not fit the edge predictor
or model, and it is not a prerequisite for the raw-data training command. A new
training run performs preprocessing itself so that the aligned data, edge
predictor, and model are made together.

Pass the whole `training/` directory to downstream analysis; you do not choose
a stage checkpoint by hand. `training_run_summary.json` records the exact
aligned H5AD used for fitting, and current-model downstream runs compare its
recorded file identity before loading the data. A newly fitted edge predictor
records the same H5AD identity and its selected threshold in
`<predictor>.meta.json`. A preprocessing-only run creates neither of those
training files. If a matching check fails, return to the aligned H5AD and
`training/` directory from the same run. Do not combine files from different
runs.

## Start from your own AnnData

Export the closest example configuration, edit its data keys and analysis settings,
then inspect the edited config before starting a fit:

```bash
cytobridge workflow --config zebrafish \
  --export-config configs/my_dataset.json

cytobridge workflow --config configs/my_dataset.json --train \
  --input-h5ad inputs/my_dataset.h5ad \
  --output-dir outputs/my_dataset \
  --device cuda --check
```

The [own-data tutorial](tutorials/your_data.ipynb) lists the fields to change
and the output folders passed to downstream analysis. The [small generated-data
example](tutorials/data_preparation/synthetic_preprocessing.ipynb) runs the
public preprocessing API and plots its result.

## Analyze an existing model

```bash
cytobridge workflow --config zebrafish --step downstream \
  --aligned-h5ad inputs/zebrafish_aligned.h5ad \
  --model-dir models/zebrafish \
  --output-dir outputs/zebrafish \
  --device cuda
```

The downstream step loads the aligned AnnData and checkpoint, generates the
requested intermediate slices, and writes classifier, velocity, growth,
composition, sparse communication, and figure outputs. It does not train a
model.

Models trained with a ligand--receptor edge predictor normally store that
predictor in the checkpoint. It selects nearby cell pairs that can contribute
to the interaction graph. If it was saved separately, add:

```text
--edge-predictor-path models/zebrafish/edge_predictor.pt
```

## Add gene and ligand--receptor analyses

```bash
cytobridge workflow --config zebrafish --step downstream \
  --aligned-h5ad inputs/zebrafish_aligned.h5ad \
  --model-dir models/zebrafish \
  --output-dir outputs/zebrafish_gene_lr \
  --reference-h5ad inputs/zebrafish_aligned.h5ad \
  --gene-dynamics \
  --lr-database inputs/CellChatDB.ligrec.csv \
  --lr-complex-mode min \
  --device cuda
```

The reference AnnData must contain the fitted PCA loadings in `.varm['PCs']`,
the matching gene order, and the PCA center in `.var['pca_center']`. Some older
complete PCA objects lack the center. For those files only, add
`--allow-complete-reference-pca-center-fallback`; CytoBridge then checks the
recovered center against the stored latent coordinates.

Ligand--receptor complexes require every subunit. `min` uses the least-
expressed subunit. `geometric_mean` is available as a zero-preserving
alternative.

## Train from raw AnnData

Training starts only when `--train` is present:

```bash
cytobridge workflow --config mosta --train \
  --input-h5ad inputs/mosta_raw_counts.h5ad \
  --output-dir outputs/mosta \
  --device cuda
```

The workflow preprocesses and aligns the raw input, builds interaction graphs,
fits the edge predictor, trains the model components listed in the training
configuration, and runs downstream analysis. A missing model directory does
not enable training automatically.

Chicken heart uses the two preparation scripts described in its dataset
tutorial before the workflow command:

```bash
cytobridge workflow --config chicken_heart --train \
  --input-h5ad inputs/chicken_heart_ot_input.h5ad \
  --output-dir outputs/chicken_heart \
  --device cuda:0
```

## Analyze a model trained with another configuration

Pass `--training-config` when downstream analysis should use settings other
than the dataset default. For example, this AD configuration uses spatial
neighbors instead of a fitted ligand--receptor edge predictor while keeping
the interaction part of the model:

```bash
cytobridge workflow --config admouse --step downstream \
  --training-config admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml \
  --aligned-h5ad inputs/admouse_no_lr/aligned.h5ad \
  --model-dir models/admouse_no_lr \
  --output-dir outputs/admouse_no_lr \
  --device cuda
```

The aligned input and model directory must come from the same training run.
