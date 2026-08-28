# Reproduce the paper figures

Use the [dataset notebooks](tutorials/dataset_workflows/index.md) for data
preparation, training, and downstream analysis. Then open the [paper figure
notebooks](tutorials/paper_figures/index.md) for the additional calculation and
plotting code used by each figure.

Every figure notebook begins with:

- the command or Python function that performs each calculation;
- the input files it reads;
- the files it creates; and
- the next calculation that uses those files.

The same information is available in the command line:

```bash
cytobridge figure list
cytobridge figure explain nonspatial
cytobridge figure explain zebrafish-si
```

Most figure notebooks recalculate panel values from CSV or NPZ files and draw
new PDF and PNG files. A few main figures combine existing vector panels; those
notebooks state this directly and require the corresponding result directory.

{download}`Download the figure index <data/paper_reproduction_registry.csv>`

The current AD S29–S30 pages and the assembled chicken-heart S7–S10 pages are
listed, but their exact page-assembly inputs have not yet been identified. The
documentation therefore links the available analyses without presenting a
different script as the source of those pages.
