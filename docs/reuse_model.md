# Continue from a trained model

Use the [analysis tutorials](tutorials/dataset_workflows/index.md) to calculate
and plot results one step at a time in Python. This page gives the equivalent
command-line option for a dataset's configured analyses.

## Calculate the downstream results

For example, after extracting the MOSTA model and analysis-data downloads:

```bash
cytobridge workflow --config mosta --step downstream \
  --aligned-h5ad data/mosta/aligned.h5ad \
  --model-dir data/mosta/model \
  --edge-predictor-path data/mosta/edge_classifier/mosta_edge_model.pt \
  --output-dir outputs/mosta_analysis --device cuda
```

This reads the downloaded aligned data and model. It writes
new results to `outputs/mosta_analysis/downstream/`. It does not preprocess or
train again.

For a downloaded model, change `--aligned-h5ad` and `--model-dir` to its
locations. Keep the model configuration, score model, and edge predictor with
the dynamical checkpoint. The [download guide](data_checkpoints.md) describes
these files. Use the aligned H5AD fitted with that model: repeating PCA on
another file changes the coordinates that the model expects.

For step-by-step preprocessing and training, follow [Train a model](training.md).
Its `preprocess`, `align_spatial`, and `fit` calls are successive steps. Each
uses the data prepared by the preceding call.

## Inspect a command without running it

Add `--check` to a workflow command to print the selected steps and paths.
This option does not open or validate the input H5AD.
