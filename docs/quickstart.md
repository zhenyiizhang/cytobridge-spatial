# Get started

CytoBridge learns cell-state and spatial dynamics from measurements collected
at several times. The Python API has three parts: `cb.pp` prepares data,
`cb.tl` fits models and calculates results, and `cb.pl` draws plots.

## Try the analysis API

Start with the [chicken-heart notebook](tutorials/dataset_workflows/chicken_heart.ipynb).
It is the smallest of the five spatial datasets. After downloading its data
and model, run the cells in order to:

1. Load the aligned data and trained model.
2. Calculate and plot velocity components.
3. Calculate and plot growth across stages.
4. Calculate attention and summarize cell-type interactions.

Each calculation returns arrays or tables that can be used in later analyses.
The notebook displays its calculated plots alongside the code.

The [other dataset tutorials](tutorials/dataset_workflows/index.md) use the same
functions with MOSTA, ARISTA, AD mouse, and Zebrafish data.

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
