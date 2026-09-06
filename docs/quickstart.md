# Get started

CytoBridge learns cell-state and spatial dynamics from measurements collected
at several times. The Python API has three parts: `cb.pp` prepares data,
`cb.tl` fits models and calculates results, and `cb.pl` draws plots.

## Try the analysis API

Start with [Velocity and growth](tutorials/model_analysis.ipynb), which uses
the chicken-heart model. Run the cells in order to:

1. Load the aligned data and trained model.
2. Calculate and plot velocity components.
3. Calculate and plot growth across stages.

Each calculation returns arrays or tables that can be used in later analyses.
The notebook displays its calculated plots alongside the code.

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

[Paper figures](tutorials/paper_figures/index.md) is a separate collection for
the manuscript's particular comparisons and panel layouts. Those pages name
the numerical files or existing panels that they read.

## Use the command line

The [workflow guide](reuse_model.md) provides commands for running a dataset's
configured sequence from the terminal. Use the Python tutorials when you want
to change or inspect individual analyses.

```{toctree}
:hidden:

tutorials/data_preparation/synthetic_preprocessing
```
