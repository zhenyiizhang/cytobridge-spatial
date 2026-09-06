---
orphan: true
---

# Analysis inputs: Main Figure 2: AGIST benchmark

The [figure notebook](../../tutorials/paper_figures/main_figure_2.ipynb) starts from saved numerical results or completed panels. This page records the calculations that precede it.

The [AGIST simulation tutorial](../../tutorials/paper_figures/agist_simulations.md)
contains complete download, simulation, distance-calculation and panel-e plotting commands.

## Calculation programs

Each command lists the input it reads and the output passed to the next calculation. Replace a path in angle brackets with the location of that file on your computer.


### 1. generate replicate trajectories (Main Figure 2e)

```text
python scripts/run_agist_split_sde_replicates.py --config data/agist/config.yaml --checkpoint-dir data/agist/model --edge-predictor data/agist/edge_classifier/mouse.pt --data-csv data/agist/mouse_brain_simulation.csv --output-dir outputs/agist_simulations --simulate-only --device cuda:0
```

Start with: `fixed model checkpoint; AGIST cells; ten inference seeds`

Writes: `one trajectory file per inference seed`

Next: `calculate replicate W2`




### 2. calculate replicate W2 (Main Figure 2e)

```text
python scripts/evaluate_and_plot_agist_w2_replicates.py --trajectory-dir outputs/agist_simulations/trajectories --truth-csv data/agist/mouse_brain_simulation.csv --output-dir outputs/agist_distances
```

Start with: `replicate trajectories and observed AGIST cells`

Writes: `w2_replicates_long.csv; w2_mean_sd_ci.csv; figure2e_agist_w2_mean_sd.pdf/.png`

Next: `draw panel e`




### 3. draw panel e and assemble (Main Figure 2e)

```text
python scripts/execute_paper_notebooks.py --notebook main_figure_2 --output-dir <notebook-run>
```

Start with: `panel-e tables and the existing panels a–d PDF`

Writes: `Main_Figure_2.pdf/.png and copied panel-e tables`
