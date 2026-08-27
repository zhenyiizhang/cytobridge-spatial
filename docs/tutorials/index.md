# Tutorials

There are two separate routes: run CytoBridge on a dataset, or reproduce a
paper figure from its released inputs. The paper pages always state which route
they use. Notebook outputs are saved, so figures and tables are visible on the
site before the code is run locally.

## Start with your own data

[Run CytoBridge on your data](your_data.ipynb) covers the AnnData fields,
editable config, dry run, training command, downstream continuation, and output
folders.

## Reuse a paper dataset workflow

The [dataset workflows](dataset_workflows/index.md) contain the preprocessing,
training, and downstream calls for Zebrafish, MOSTA, ARISTA, AD mouse, and
chicken heart.

## Reproduce a paper figure

The [paper figure index](paper_figures/index.md) separates numerical redraws
from result-summary plots, page assembly, and reference export. Run
`cytobridge figure explain <name>` to see the upstream entry and limits of any
figure command.

## Input and analysis references

- [Data and checkpoints](../data_checkpoints.md)
- [Small synthetic preprocessing example](data_preparation/synthetic_preprocessing.ipynb)
- [Benchmarks](../benchmarks.md)
- [Training time and memory](../training_compute.md)

```{toctree}
:hidden:
:maxdepth: 2

your_data
dataset_workflows/index
paper_figures/index
data_preparation/synthetic_preprocessing
../data_checkpoints
../benchmarks
../training_compute
```
