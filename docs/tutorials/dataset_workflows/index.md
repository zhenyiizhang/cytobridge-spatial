# Paper datasets

These notebooks follow the five spatial datasets used in the paper. Each one
contains data preparation, model training, downstream analysis, and the next
commands used to calculate its paper figures.

The time-consuming cells are switched off on the documentation site. Their
commands, paths, and saved output remain visible; set the corresponding
`RUN_...` variable to `True` after adding the data locally.

```{toctree}
:maxdepth: 1

zebrafish
mosta
arista
admouse
chicken_heart
```

For a new experiment, begin with [Run CytoBridge on your
data](../your_data.ipynb). For the final plotting steps, use the [paper figure
notebooks](../paper_figures/index.md).
