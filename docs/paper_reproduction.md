# Paper reproduction index

The reproduction registry maps every computational figure, table, and video to
its reader-facing notebook or script. It also separates entries that can run
from compact packaged tables from entries that require external data or model
files.

{download}`Download the registry <data/paper_reproduction_registry.csv>`

## Entry points

- Dataset notebooks cover preprocessing, optional training, model loading, and
  downstream calculations.
- Paper-figure notebooks recalculate plotted summaries from compact result
  tables and write PDF and PNG outputs. Each compact bundle or repository
  release states whether the PDF preserves vector objects or places a compact
  raster on the page.
- The zebrafish video page provides the final videos and the commands used to
  render them from trajectory arrays.

The S1 row remains on hold because the historical result state needed to
recalculate the displayed values is not available. The published image is not
silently replaced by a different simulation run.
