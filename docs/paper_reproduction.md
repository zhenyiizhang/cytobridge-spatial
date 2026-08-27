# Paper reproduction index

The registry maps every figure, table, and video to a notebook or script. Start
with the mode instead of assuming that every page retrains a model:

- `numeric-redraw` recalculates plotted values from packaged numeric inputs.
- `result-summary-redraw` plots prepared result summaries without rerunning the
  original analysis.
- `reference-export` copies a packaged figure page and makes a preview when
  needed.
- `external-assembly` combines panel files from a repository release or an
  external run.
- `table-only` writes or displays a table and does not draw a figure.

Some rows use two modes because one page contains both redrawn and frozen
panels. `wheel_runnable` is `true` when the entry can run from an installed
wheel after installing the dependency listed in the next column.

{download}`Download the registry <data/paper_reproduction_registry.csv>`

## Check a figure route

```bash
cytobridge figure list
cytobridge figure explain zebrafish-si
cytobridge figure explain nonspatial --json
```

`figure explain` prints the complete handoff: the code for each step, what it
reads, what it writes, and which next step consumes those files. This is also
shown at the top of every paper notebook.

## Entry points

- [Run CytoBridge on your data](tutorials/your_data.ipynb) is the shortest route
  from a custom AnnData file to preprocessing, training, downstream output, and
  standard figures.
- Dataset notebooks cover the five packaged presets and name the aligned H5AD,
  model directory, downstream output, and dataset-specific paper continuation
  passed between steps.
- Paper-figure notebooks state whether they redraw numbers, assemble existing
  panels, or export a reference page. They write only the files supported by
  that mode.
- The zebrafish video page provides the final videos and the commands used to
  render them from trajectory arrays.

The S1 row remains on hold because the earlier result state needed to
recalculate the displayed values is not available. The published image is not
silently replaced by a different simulation run.

The manuscript chicken-heart S7–S10 PNGs and AD S26–S30 PDFs remain visible in
the registry, but the documentation does not present a related script as their
exact generator. Their dataset pages name the available calculations and the
precise missing handoff.
