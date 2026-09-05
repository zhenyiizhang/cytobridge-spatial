---
orphan: true
---

# Analysis inputs: Main Figure 2: AGIST benchmark

The [figure notebook](../../tutorials/paper_figures/main_figure_2.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. generate replicate trajectories (Main Figure 2e)

```text
python scripts/run_agist_split_sde_replicates.py --project-root . --config <agist-training.yaml> --checkpoint-dir <checkpoint-dir> --data-csv <agist-cells.csv> --output-dir <agist-replicates> --seeds 1 4 8 32 256 --device cuda
```

Start with: `fixed model checkpoint; AGIST cells; five inference seeds`

Writes: `one trajectory file per inference seed`

Next: `calculate replicate W2`




### 2. calculate replicate W2 (Main Figure 2e)

```text
python scripts/evaluate_and_plot_agist_w2_replicates.py --trajectory-dir <agist-replicates> --truth-csv <observed-agist.csv> --output-dir <agist-w2>
```

Start with: `replicate trajectories and observed AGIST cells`

Writes: `w2_replicates_long.csv; w2_mean_sd_ci.csv; baseline_w2.csv`

Next: `draw panel e`




### 3. draw panel e and assemble (Main Figure 2e)

```text
python scripts/execute_paper_notebooks.py --notebook main_figure_2 --output-dir <notebook-run>
```

Start with: `panel-e tables and the existing panels a–d PDF`

Writes: `Main_Figure_2.pdf/.png and copied panel-e tables`
