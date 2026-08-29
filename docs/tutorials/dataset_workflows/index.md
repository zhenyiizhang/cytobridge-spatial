# Paper datasets

These notebooks follow the five spatial datasets used in the paper. Each one
starts with the command that prepares the raw H5AD and trains a new model, then
shows downstream analysis. Later entries are labelled **Continue from the
model run above** when they read that output, and **Start from the paper's saved
files** when they read the exact tables, arrays, or models retained for the
paper. **Required paper files not included** names an input without pretending
that the new run creates it.
An optional preprocessing-only section is provided for inspecting the aligned
H5AD; it is an alternative check, not a step that must be run before training.

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
