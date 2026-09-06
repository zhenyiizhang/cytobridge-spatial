# Reproduce the paper figures

Choose the path that matches what you want to do.

## Redraw the published result

Open a [paper figure notebook](tutorials/paper_figures/index.md). The numerical
plotting notebooks calculate summaries from CSV, NPZ or H5AD inputs and write
new PDF and PNG files. Figures 4–6, MOSTA S11–S18 and ARISTA S19–S24 have
dedicated plotting programs. Their tutorials identify the required downloads.
The MOSTA supplementary notebook also evaluates the model and recalculates
gene programs, enrichment and LR scores, with separate calculation and
plotting cells.
Figure 2 still reuses its earlier panels a–d while drawing panel e from numbers.

## Continue from a new model run

Follow [Train a model](training.md) to prepare your data and fit CytoBridge.
Then use a [dataset notebook](tutorials/dataset_workflows/index.md) for the
analysis. If you download a trained model, start at the analysis notebook.
A figure notebook can
continue from that run when it gives a command that reads the run's output
directory. Where the conversion to the paper's exact panel files or the final
page assembly is still missing, the figure index says so.

Each notebook lists the command or source file, what it reads, what it writes,
and which step comes next. Saved output shows what a completed run looks like.

The same information is available in the command line:

```bash
cytobridge figure list
cytobridge figure explain nonspatial
cytobridge figure explain zebrafish-si
```

{download}`Download the figure index <data/paper_reproduction_registry.csv>`

The index distinguishes numerical plotting, static artwork and remaining
analysis steps. AD S29–S30 now have their numerical inputs and plotting
commands in the [AD figure guide](tutorials/paper_figures/admouse_figures.md).
The [zebrafish notebook](tutorials/paper_figures/zebrafish_si_s31_s38.ipynb)
puts the model calculations and plotting for S31–S38 on one page. It also
includes the training and evaluation code for the S36 loss-weight comparison.
The [non-spatial workflow](nonspatial_workflows.md) passes newly evaluated
distribution, clone-fate and direction results directly to plotting.
Some remaining S4–S5 panels, S6 and S25 can be redrawn from the
included numbers but still need adapters from a new model run.
S41, S42, S45, and S46 include a command that collects completed analysis
tables before passing them to the figure command.
