---
orphan: true
---

# Analysis inputs: Supplementary Figure S39: zebrafish attention and control comparisons

The [figure notebook](../../tutorials/paper_figures/zebrafish_attention.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. compare model scores with external methods (S39)

```text
python -m scripts.run_zebrafish_attention_analysis analyze --spec <analysis-spec.json> --output-dir <attention-analysis> --n-selected-pairs 30
```

Start with: `manuscript zebrafish checkpoint; aligned cells; COMMOT/CellAgentChat outputs; fixed LR universe`

Writes: `directed-pair concordance, expression, display-edge and interaction-sensitivity tables plus analysis_manifest.json`

Next: `combine with JAM controls and draw S39`


Run this command from the root of a cloned CytoBridge GitHub repository. The installed package contains the final figure command, while this manuscript comparison script remains in the repository.



### 2. combine JAM controls and draw S39 (S39)

```text
python -m scripts.run_zebrafish_attention_analysis figure --analysis-dir <attention-analysis> --jam-manifest <trained-jam>/run_manifest.json --jam-manifest <before-interaction-jam>/run_manifest.json --jam-manifest <randomized-jam>/run_manifest.json --output-dir <attention-figure>
```

Start with: `attention-analysis tables and one or more matched JAM control manifests`

Writes: `spatial-null, JAM, summary and panel tables; vector PDF/PNG; report_manifest.json`

Next: `recalculate the displayed statistics`


Run this command from the same repository checkout. Repeat --jam-manifest for the trained, before-interaction, and randomized comparison results.



### 3. recalculate displayed statistics and draw (S39)

```text
python scripts/execute_paper_notebooks.py --notebook zebrafish_attention --output-dir <notebook-run>
```

Start with: `directed_pair_concordance.csv; JAM tables; spatial-null tables; expression and edge tables`

Writes: `zebrafish_attention_controls.pdf/.png and summary tables`
