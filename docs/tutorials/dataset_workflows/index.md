# Analysis tutorials

Choose a dataset to work through an analysis from the paper. The notebooks
load a trained model, calculate the quantities shown below, and draw them.
Code and results appear together. You can use the same functions with your
own trained model.

Start with **chicken heart** for a smaller dataset. To fit a new model first,
follow [Train a model](../../training.md). Training is separate from these
notebooks, so you can repeat or change an analysis without fitting again.

| Dataset | What you calculate | Figure |
| --- | --- | --- |
| [Chicken heart](chicken_heart.ipynb) | Daily populations, median growth by cell type, spatial growth maps | S9 |
| [MOSTA](mosta.ipynb) | Cell-state trajectories, brain-cell growth, population composition | S11–S13 |
| [ARISTA](arista.ipynb) | Intermediate populations and spatial growth maps | S20 |
| [AD mouse](admouse.ipynb) | Spatial populations, cell-type proportions and cell numbers | S26 |
| [Zebrafish](zebrafish.ipynb) | Growth at the measured stages | S32 |

Download instructions are collected on the [data page](../../data_checkpoints.md).

```{toctree}
:maxdepth: 1

chicken_heart
mosta
arista
admouse
zebrafish
```

For another experiment, use [Run CytoBridge on your data](../your_data.ipynb).
For the remaining analyses and for the exact saved simulations used in the
paper, continue to [Paper figures](../paper_figures/index.md). Each dataset
notebook also links to its next analyses.
