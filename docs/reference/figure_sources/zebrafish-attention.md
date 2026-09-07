---
orphan: true
---

# Analysis inputs: Supplementary Figure S39: zebrafish attention and control comparisons

The [figure notebook](../../tutorials/paper_figures/zebrafish_attention.ipynb) draws the figure from saved numerical results. The steps below calculate those inputs from data and fitted models.

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
python -m scripts.run_zebrafish_attention_analysis figure \
  --analysis-dir <attention-analysis> \
  --jam-manifest <jam-controls>/manifest.json \
  --jam-manifest <jam-biology>/manifest.json \
  --output-dir <attention-figure>
```

Start with: `attention-analysis tables, the three-condition JAM control manifest, and the JAM spatial/biology manifest`

Writes: `spatial-null, JAM, summary and panel tables; vector PDF/PNG; report_manifest.json`

Next: `collect the numerical inputs for the current figure notebook`


The control manifest is written by `scripts/reviewer_zebrafish_ccc/jam_trained_init_random_control.py`, which processes the trained, pre-interaction and randomized edge tables together. The spatial/biology manifest is written by `scripts/reviewer_zebrafish_ccc/jam_myocyte_case_study.py`.



### 3. collect the calculated panel tables (S39)

```text
python -m scripts.collect_zebrafish_attention_inputs \
  --panel-data-dir <attention-figure>/panel_data \
  --cytobridge-pairs <attribution>/type_pair_summary.csv \
  --commot-pairs <commot>/commot_type_pair_scores.csv.gz \
  --output-dir <s39-inputs>
```

Start with: the ten tables exported in `<attention-figure>/panel_data`, plus the CytoBridge and COMMOT pair-score files bound to the preceding attention analysis. The collector selects their terminal stage using the same rule as `analyze`.

Writes: `<s39-inputs>/manifest.json`, the ten unchanged panel tables, and `commot_comparison/` containing the corresponding pair scores. The collector derives the manifest from these results and checks the existing S39 input contract. It does not run model inference or substitute included paper tables. Use a new output directory.

### 4. draw the collected results (S39)

In the figure notebook, set `results_dir = Path("<s39-inputs>")` and `commot_results_dir = results_dir / "commot_comparison"`, then run all cells. The final call uses those paths:

```python
pdf_path, png_path = draw_supplementary(
    [39], output_dir,
    results_dir=results.source_dir,
    commot_results_dir=commot_results_dir,
)["s39"]
```

Writes: `S39.pdf`, `S39.png`, panel summary tables and the 1,000 within-group COMMOT permutation values. Model inference, external-method execution and the 10,000 JAM label permutations are not repeated by this final step.
