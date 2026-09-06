# Virtual YSL and EVL removal: S33–S34

These simulations compare development from time zero with all observed cells,
without YSL cells, and without EVL cells. The three initial populations contain
563, 534 and 291 cells. Each is propagated to time four using the same trained
model, with growth and diffusion active.

## Download the inputs

From the [source checkout](../../installation.md), run:

```bash
python -m CytoBridge.datasets zebrafish --kind analysis --output-dir .
```

This supplies the aligned cells, trained model, learned edge predictor and
cell-type classifier. The analysis uses the learned edge predictor within the
observed expression-state range, as in the paper.

## Run five simulation seeds

Each command calls `cb.tl.run_virtual_cell_type_ablation` for the baseline and
both removals. It resets the random seed for each condition, simulates the
populations, and calculates distribution distances and centroid shifts from
the corresponding baseline.

```bash
python -m reproduction.zebrafish.virtual_removal --seed 42 --output-dir outputs/virtual_removal/seed_42 --device cuda:0
python -m reproduction.zebrafish.virtual_removal --seed 43 --output-dir outputs/virtual_removal/seed_43 --device cuda:0
python -m reproduction.zebrafish.virtual_removal --seed 44 --output-dir outputs/virtual_removal/seed_44 --device cuda:0
python -m reproduction.zebrafish.virtual_removal --seed 45 --output-dir outputs/virtual_removal/seed_45 --device cuda:0
python -m reproduction.zebrafish.virtual_removal --seed 46 --output-dir outputs/virtual_removal/seed_46 --device cuda:0
```

Each output directory contains:

- `experiment/trajectories/`: baseline, YSL-removal and EVL-removal state arrays at 81 time points.
- `experiment/ablation_metrics.csv`: W1, W2, population counts and centroid shifts.
- `run_summary.json`: the model, input data, seed and simulation settings.

The integration step is 0.005, population resampling occurs every 0.05 time
units, and the diffusion coefficient is 0.03. Distribution distances use
uniform empirical weights, with deterministic subsamples of at most 1,024
points per population. Simulated cells are not removed by this subsampling.

## Draw the figures

```bash
python -m reproduction.zebrafish.plot_virtual_removal \
  --data-dir data/zebrafish \
  --run-dir outputs/virtual_removal/seed_42 \
  --run-dir outputs/virtual_removal/seed_43 \
  --run-dir outputs/virtual_removal/seed_44 \
  --run-dir outputs/virtual_removal/seed_45 \
  --run-dir outputs/virtual_removal/seed_46 \
  --output-dir outputs/virtual_removal_figures
```

S33 shows seed 42 at the five integer time points. The plotting command
calculates the classifier's PCA features from the aligned data and assigns
cell types to the simulated states. S34 shows the seed-42 endpoint populations,
mean spatial W1 across five seeds with standard errors, and endpoint centroid
shifts with 95% t intervals.

The command writes new S33/S34 PDF and PNG files, the plotted state arrays,
`spatial_w1_curve.csv`, `centroid_by_seed.csv` and `centroid_summary.csv`.
It does not read the included S33/S34 figure tables.
