# Quickstart

Check the inputs and output paths before providing data or starting a fit:

```bash
cytobridge workflow --config zebrafish --check
cytobridge workflow --config admouse --check --json
cytobridge workflow --config chicken_heart --check
```

The check prints the selected configuration, the analysis steps, and any
missing inputs. It does not preprocess data, train a model, or write outputs.

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
and the output folders passed to downstream analysis.

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

Predictor-gated checkpoints normally contain the edge predictor. For a
checkpoint that stores it separately, add:

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
the matching gene order, and the PCA center in `.var['pca_center']`. For a
complete older PCA-fit object without the center, use
`--allow-complete-reference-pca-center-fallback`; the loader checks that the
inferred center reconstructs the stored latent coordinates.

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
fits an edge predictor in the prepared feature space, trains the six-stage
dynamical model, and runs downstream analysis. A missing model directory does
not enable training automatically.

Chicken heart uses the two preparation scripts described in its dataset
tutorial before the workflow command:

```bash
cytobridge workflow --config chicken_heart --train \
  --input-h5ad inputs/chicken_heart_ot_input.h5ad \
  --output-dir outputs/chicken_heart \
  --device cuda:0
```

## Use another training configuration

Pass `--training-config` when a workflow should use settings other than the
dataset default. For example, the AD `all_spatial` configuration removes the learned
LR-informed edge gate while retaining the interaction model:

```bash
cytobridge workflow --config admouse --step downstream \
  --training-config admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml \
  --aligned-h5ad inputs/admouse_no_lr/aligned.h5ad \
  --model-dir models/admouse_no_lr \
  --output-dir outputs/admouse_no_lr \
  --device cuda
```

The aligned input and model directory must come from the same training run.
