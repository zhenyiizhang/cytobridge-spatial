# AGIST model simulations: Figure 2e

Start with the trained model and simulated observations. This analysis repeats
the ten inference seeds used in Figure 2e, calculates Wasserstein distances,
and draws the comparison in gene-expression and physical space.

## Download the inputs

From the [CytoBridge code folder](../../installation.md), run:

```bash
python -m CytoBridge.datasets agist --output-dir . --kind analysis
```

This creates `data/agist/`, containing `model/`, `edge_classifier/`,
`config.yaml` and `mouse_brain_simulation.csv`.

## Simulate the populations

```bash
python scripts/run_agist_split_sde_replicates.py \
  --config data/agist/config.yaml \
  --checkpoint-dir data/agist/model \
  --edge-predictor data/agist/edge_classifier/mouse.pt \
  --data-csv data/agist/mouse_brain_simulation.csv \
  --output-dir outputs/agist_simulations \
  --simulate-only \
  --device cuda:0
```

The model remains fixed. The ten seeds change the random interaction groups,
Brownian increments and cell birth or loss draws. Each run starts with all
cells at model time zero and writes its populations and weights to
`outputs/agist_simulations/trajectories/split_sde_seed_<seed>.npz`.
The integration settings are `dt=0.1`, `sigma=0.03`, interaction group size
1,024 and zero daughter-cell displacement.

## Calculate distances and draw panel e

```bash
python scripts/evaluate_and_plot_agist_w2_replicates.py \
  --trajectory-dir outputs/agist_simulations/trajectories \
  --truth-csv data/agist/mouse_brain_simulation.csv \
  --output-dir outputs/agist_distances
```

This reads the trajectories from the preceding command, calculates weighted
Wasserstein-2 distances at times 1, 2 and 3, and writes:

- `w2_replicates_long.csv`: one distance per seed, time and coordinate space.
- `w2_mean_sd_ci.csv`: means and standard deviations across the ten runs.
- `figure2e_agist_w2_mean_sd.pdf` and `.png`: the newly drawn panel.

The STORIES and stVCR values in this panel are the paper's saved benchmark
values. This command recalculates the CytoBridge trajectories and distances.
The [Figure 2 notebook](main_figure_2.ipynb) assembles the complete page from
its saved panel-e results and existing panels a–d.
