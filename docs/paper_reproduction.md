# Reproduce the paper figures

Find the figure you want in the [paper figure guide](tutorials/paper_figures/index.md).
Download the data listed at the start of its tutorial, then run the cells in
order. The examples show how to evaluate the trained models and draw the
resulting measurements.

Figure 2 covers the AGIST simulation. The remaining main figures and their
supplementary analyses are organized by dataset. Separate tutorials cover
comparisons between methods and changes to model settings. Figure 1, S1 and
the schematic in Figure 2a are supplied as vector artwork.

## Use your own results

Follow [Train a model](training.md), then continue with a
[dataset tutorial](tutorials/dataset_workflows/index.md). The same analysis
functions can be used with a downloaded model. Save their returned arrays and
tables to draw the corresponding panels with your results.

For a shorter plotting example, use one of the tutorials that starts from
included numerical tables, such as the [benchmark summary](tutorials/paper_figures/loto_benchmark_summary.ipynb).

## Find a command or source file

```bash
cytobridge figure list
cytobridge figure explain nonspatial
cytobridge figure explain zebrafish-si
```

{download}`Download the figure index <data/paper_reproduction_registry.csv>`

The index lists the data and code for each main figure, supplementary figure
and video. Some source files retain their original SI numbers. The tutorials
and figure index use the numbers in the manuscript.
