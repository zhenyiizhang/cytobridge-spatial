# Paper reproduction index

The registry maps every figure, table, and video to its reader-facing notebook
or script. The `reproduction_mode` column says what the entry actually does:

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

## Entry points

- Dataset notebooks cover preprocessing, optional training, model loading, and
  downstream calculations.
- Paper-figure notebooks state whether they redraw numbers, assemble existing
  panels, or export a reference page. They write only the files supported by
  that mode.
- The zebrafish video page provides the final videos and the commands used to
  render them from trajectory arrays.

The S1 row remains on hold because the historical result state needed to
recalculate the displayed values is not available. The published image is not
silently replaced by a different simulation run.
