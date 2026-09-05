# Continue from a trained model

Use this page if training has already finished, or if you have downloaded a
trained model together with its aligned H5AD. You can start here without running
the training tutorial.

## Calculate the downstream results

For example, after the MOSTA training command in [Get started](quickstart.md):

```bash
cytobridge workflow --config mosta --step downstream \
  --aligned-h5ad outputs/mosta/preprocess/mosta_aligned.h5ad \
  --model-dir outputs/mosta/training \
  --output-dir outputs/mosta_analysis --device cuda
```

This reads the aligned data and trained model from the original run. It writes
new results to `outputs/mosta_analysis/downstream/`. It does not preprocess or
train again.

For a downloaded model, change `--aligned-h5ad` and `--model-dir` to its
locations. Keep the model configuration, score model, and edge predictor with
the dynamical checkpoint. The [download guide](data_checkpoints.md) describes
these files. Use the aligned H5AD fitted with that model: repeating PCA on
another file changes the coordinates that the model expects.

## Inspect preprocessing on its own

If you want to examine the alignment before fitting a model:

```bash
cytobridge workflow --config mosta --step preprocess \
  --input-h5ad data/mosta_raw.h5ad \
  --output-dir outputs/mosta_preprocessing --device cuda
```

This writes the prepared H5AD. It does not fit the edge predictor or the
dynamical model. To start the full run, use the raw-count command in
[Get started](quickstart.md), which performs preprocessing again in its own
output directory.

## Inspect a command without running it

Add `--check` to a workflow command to print the selected steps and paths.
This option does not open or validate the input H5AD.
