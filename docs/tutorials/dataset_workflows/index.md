# Analysis tutorials

Choose a dataset to work through an analysis from the paper. The notebooks
load a trained model, calculate the quantities shown below, and draw them.
Code and results appear together. You can use the same functions with your
own trained model.

Start with chicken heart for a smaller dataset. The notebooks can use the
supplied models without training again. AD and the complete zebrafish notebook
also show the training call on the page. [Train a model](../../training.md)
explains preprocessing and fitting for a new input file.

| Dataset | What you calculate | Figure |
| --- | --- | --- |
| [Chicken heart](chicken_heart.ipynb) | Daily populations, median growth by cell type, spatial growth maps | S14 |
| [MOSTA](mosta.ipynb) | Cell-state trajectories, brain-cell growth, population composition | S16–S18 |
| [ARTISTA](arista.ipynb) | Intermediate populations and spatial growth maps | S30 |
| [AD mouse](admouse.ipynb) | Spatial populations, cell-type proportions and cell numbers | S31 |
| [Zebrafish](../paper_figures/zebrafish_si_s31_s38.ipynb) | Populations, growth, gene dynamics and perturbations | S36–S37, S39–S42, S44–S45 |

Download instructions are collected on the [data page](../../data_checkpoints.md).

```{toctree}
:maxdepth: 1

chicken_heart
mosta
arista
admouse
../paper_figures/zebrafish_si_s31_s38
```

For another experiment, use [Run CytoBridge on your data](../your_data.ipynb).
For the remaining analyses and for the exact saved simulations used in the
paper, continue to [Paper figures](../paper_figures/index.md). Each dataset
notebook also links to its next analyses.
