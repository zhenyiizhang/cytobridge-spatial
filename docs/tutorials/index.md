# Tutorials

The tutorials cover data preparation, complete dataset workflows, and the code
used for the paper figures. The notebooks are stored with their outputs so the
generated tables and plots can be read before running them locally.

## Data preparation

Start with the input checklist for real datasets, or run the compact synthetic
example to inspect the expected AnnData fields.

- [Data and checkpoints](../data_checkpoints.md)
- [Synthetic preprocessing](data_preparation/synthetic_preprocessing.ipynb)

```{toctree}
:hidden:
:maxdepth: 1

../data_checkpoints
data_preparation/synthetic_preprocessing
```

## Dataset tutorials

Each dataset has one notebook. The same page covers setup, preprocessing,
training, downstream analysis, paper figures, and saved files.

[Browse the dataset tutorials](dataset_workflows/index.md).

```{toctree}
:hidden:
:maxdepth: 1

dataset_workflows/index
```

## Paper figures

Each notebook says whether it recalculates plotted values, redraws prepared
summaries, assembles panel files, or exports a reference page.

[Browse the paper figure notebooks](paper_figures/index.md).

```{toctree}
:hidden:
:maxdepth: 1

paper_figures/index
```

## Benchmarks and compute

The benchmark pages describe the package interfaces and the compute table used
in the supplementary material.

- [Benchmarks](../benchmarks.md)
- [Training time and memory](../training_compute.md)

```{toctree}
:hidden:
:maxdepth: 1

../benchmarks
../training_compute
```
