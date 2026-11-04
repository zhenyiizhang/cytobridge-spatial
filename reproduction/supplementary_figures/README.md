# Supplementary figure plotting

Choose a figure in the [paper figure guide](../../docs/tutorials/paper_figures/index.md)
for its calculations and plotting example. To draw from the included numerical
inputs, run from the repository root:

```bash
python -m reproduction.current_figures --figures 5 6 47 --output-dir outputs/supplementary
```

The command writes PDF and PNG figures and their numerical summaries. Use
`--results-dir` with a compatible set of calculated tables to plot your results.

`plot_panels.py` contains the panel layouts. `code/` contains figure-specific
plotting functions and `data/` their numerical inputs. The public command uses
the SI's figure numbers; `reproduction/figure_numbering.json` maps them to the
source filenames.
