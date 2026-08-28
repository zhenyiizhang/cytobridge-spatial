# Tutorials

The notebooks are organized in the same order as a CytoBridge analysis. Start
with your own data, follow a paper dataset from preprocessing through downstream
analysis, or open a figure notebook to reproduce one paper figure.

## Your data

[Run CytoBridge on your data](your_data.ipynb) shows the AnnData fields,
configuration file, training command, downstream command, and output folders.

## Paper datasets

- [Zebrafish embryogenesis](dataset_workflows/zebrafish.ipynb)
- [MOSTA mouse organogenesis](dataset_workflows/mosta.ipynb)
- [ARISTA salamander brain regeneration](dataset_workflows/arista.ipynb)
- [AD mouse brain](dataset_workflows/admouse.ipynb)
- [Developing chicken heart](dataset_workflows/chicken_heart.ipynb)

Each dataset notebook states which file is passed from preprocessing to
training, from training to downstream analysis, and from downstream analysis
to the paper-figure code.

## Paper figures

[Browse the paper figure notebooks](paper_figures/index.md). The calculated
tables and figures are saved in the notebooks, so they are visible on this
site without running the code again.

## Examples and reference

- [Small preprocessing example](data_preparation/synthetic_preprocessing.ipynb)
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
