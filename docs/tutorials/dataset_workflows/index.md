# Dataset tutorials

Use [Run CytoBridge on your data](../your_data.ipynb) for the shortest route to
a custom config. The notebooks below show the exact presets used for the five
paper datasets. Each page keeps preparation, training, and downstream analysis
in that order and names the files passed between them.

Long runs are disabled in the published notebooks. Their resolved plans,
commands, expected model directory, downstream folder, and paper-figure routes
remain visible in the saved output.

## Spatial datasets

```{toctree}
:maxdepth: 1

zebrafish
mosta
arista
admouse
chicken_heart
```

The [data preparation guide](../../data_checkpoints.md) lists the external
inputs needed for a full run. Paper plotting is kept under
[paper figure notebooks](../paper_figures/index.md), because most paper
commands consume a compact release rather than an arbitrary new downstream
directory.
