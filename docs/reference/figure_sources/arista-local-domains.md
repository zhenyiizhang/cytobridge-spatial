---
orphan: true
---

# Analysis inputs: Supplementary Figure S26: ARTISTA local interaction domains

The [figure notebook](../../tutorials/paper_figures/arista_local_domains.ipynb) draws the figure from saved numerical results. The steps below calculate those inputs from data and fitted models.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. run corrected ARTISTA downstream (S26)

```text
cytobridge workflow --config arista --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training> --output-dir <downstream>
```

Start with: `manuscript ARISTA aligned H5AD and retrained model`

Writes: `5-DPI cell states, velocity components, sparse attention and strict LR tables`

Next: `domain analysis`




### 2. draw from the saved domain and matched-null results (S26)

```text
python scripts/results/plot_arista_local_domains.py --results-dir <domain-result-dir> --output-dir <figure-dir>
```

Start with: `ROI assignments, domain metadata, cell-type edges, and matched-null tables`

Writes: `arista_local_interaction_domains.pdf/.png and displayed tables`

Next: `open the notebook for the same calculation with saved output`




### 3. run the same calculation in the notebook (S26)

```text
python scripts/execute_paper_notebooks.py --notebook arista_local_domains --output-dir <notebook-run>
```

Start with: `ROI, domain, edge and null tables`

Writes: `arista_local_interaction_domains.pdf/.png and displayed tables`
