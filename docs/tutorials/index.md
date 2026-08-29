# Tutorials

The notebooks are organized in the same order as a CytoBridge analysis. Start
with a small generated example or your own data, follow a paper dataset from
preprocessing through downstream analysis, or open a figure notebook to redraw
one of the paper's saved result files.

## Your data

[Run CytoBridge on your data](your_data.ipynb) shows the AnnData fields,
configuration file, training command, downstream command, and output folders.

## Paper datasets

- [Zebrafish embryogenesis](dataset_workflows/zebrafish.ipynb)
- [MOSTA mouse organogenesis](dataset_workflows/mosta.ipynb)
- [ARISTA salamander brain regeneration](dataset_workflows/arista.ipynb)
- [AD mouse brain](dataset_workflows/admouse.ipynb)
- [Developing chicken heart](dataset_workflows/chicken_heart.ipynb)

Each dataset notebook starts with the raw-data training command, then states
which aligned H5AD and model directory are passed to downstream analysis and
which later commands can analyze that new run. Commands that start from files
retained for the paper are labelled separately because they do not
automatically read the new run. A
separate optional section shows how to inspect preprocessing without starting a
fit.

## Paper figures

[Browse the paper figure notebooks](paper_figures/index.md). Each page states
whether it calculates values, redraws included paper results, or needs an
external paper input. Committed outputs remain visible on this site without
running the long calculations again.

## Examples and reference

- [Small preprocessing example with a plotted result](data_preparation/synthetic_preprocessing.ipynb)
- [Data and checkpoints](../data_checkpoints.md)
- [Benchmarks](../benchmarks.md)
- [Training time and memory](../training_compute.md)

```{toctree}
:hidden:
:maxdepth: 2

your_data
dataset_workflows/index
paper_figures/index
data_preparation/synthetic_preprocessing
```
