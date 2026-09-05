---
orphan: true
---

# Tutorials

Start with a small preprocessing example, train a model on your data, or
learn an analysis using one of the paper's trained models.

## Your data

[Train a model](../training.md) explains preprocessing, alignment, graph
construction, and fitting through the Python API. For command-line use,
[configure a workflow](your_data.ipynb).

## Paper datasets

- [Zebrafish embryogenesis](dataset_workflows/zebrafish.ipynb)
- [MOSTA mouse organogenesis](dataset_workflows/mosta.ipynb)
- [ARISTA salamander brain regeneration](dataset_workflows/arista.ipynb)
- [AD mouse brain](dataset_workflows/admouse.ipynb)
- [Developing chicken heart](dataset_workflows/chicken_heart.ipynb)

Each notebook loads a model, calculates results, and plots the returned arrays
and tables. You can change the analysis without fitting the model again.

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
