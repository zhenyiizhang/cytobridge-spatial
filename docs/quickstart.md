# Get started

CytoBridge learns cell-state and spatial dynamics from measurements collected
at several times. The Python API has three parts: `cb.pp` prepares data,
`cb.tl` fits models and calculates results, and `cb.pl` draws plots.

## Try the analysis API

Start with [Velocity and growth](tutorials/model_analysis.ipynb), which uses
the chicken-heart model. Run the cells in order to:

1. Load the aligned data and trained model.
2. Calculate velocity components.
3. Calculate growth across stages.

The returned arrays and tables can also be used in your own analyses.

Continue to the [dataset tutorials](tutorials/dataset_workflows/index.md) for
population simulation and the analyses used in the paper.

## Prepare and train your own data

[Train a model](training.md) starts from a counts AnnData and shows each Python
step separately: expression preprocessing, spatial alignment, LR graph
construction, and model fitting. It ends by loading that model for analysis.

If you first want a small example that runs without study data or a GPU, the
[preprocessing notebook](tutorials/data_preparation/synthetic_preprocessing.ipynb)
creates a count matrix, calculates PCA, and plots the processed features.

## Reproduce a paper figure

Choose a figure in the [paper figure guide](tutorials/paper_figures/index.md)
for its data, calculations and plotting examples.

## Use the command line

The [workflow guide](reuse_model.md) provides commands for running a dataset's
configured sequence from the terminal. Use the Python tutorials when you want
to change or inspect individual analyses.

```{toctree}
:hidden:

tutorials/data_preparation/synthetic_preprocessing
```
