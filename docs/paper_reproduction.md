# Reproduce the paper figures

Choose the path that matches what you want to do.

## Redraw the published result

Open a [paper figure notebook](tutorials/paper_figures/index.md). Most notebooks
recalculate the displayed values from numerical CSV or NPZ files included with
CytoBridge, then write new PDF and PNG files. Notebooks described as
"view/export" open or copy a completed page because the original page-layout
files are kept with the paper results rather than in the installed package.

## Continue from a new model run

Use a [dataset notebook](tutorials/dataset_workflows/index.md) to prepare the
data, train CytoBridge, and run its standard analyses. A figure notebook can
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

The index distinguishes figures that are ready to redraw, completed pages that
can be viewed or exported, and analyses whose final page assembly is not yet
available. For AD S29–S30, the exact numerical inputs and plotting program for
the current pages have not yet been matched to the archived analyses. S4–S6,
S25 and S31–S38 can be redrawn from the included numbers, but the conversion
from a newly trained model to those exact notebook inputs is not yet available.
S41, S42, S45, and S46 include a command that collects completed analysis
tables before passing them to the figure command.
