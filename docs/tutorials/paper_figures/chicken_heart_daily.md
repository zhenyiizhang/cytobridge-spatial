# Chicken-heart development: trajectories, cell types, and growth

These notebooks calculate lineage transitions, cell-type interactions, and
local velocity fields during chicken-heart development. They read the
populations generated in the [chicken-heart tutorial](../dataset_workflows/chicken_heart.ipynb).

## Prepare the files

Extract the three chicken-heart archives from the [data page](../../data_checkpoints.md)
into your project folder. The model and aligned data are used throughout.
The raw-data archive supplies the original coordinates and section annotations
used by the supplementary plots.

## Continue after the chicken-heart tutorial

From the source checkout, run the analysis step on the populations you
already simulated. Replace `my_project` with the project folder you used in
the tutorial:

```bash
python scripts/run_chicken_heart_daily_notebooks.py --project-dir my_project --step daily
```

This reads the saved `manifest.json` and daily H5AD files, calculates
velocities and cell-type transitions, and draws the corresponding spatial
and interaction-network plots. It does not repeat interpolation.

For the growth and D10 velocity-component calculations:

```bash
python scripts/run_chicken_heart_daily_notebooks.py --project-dir my_project --step supplementary
```

That notebook also retains the collaborator's earlier alignment display.
The current S7–S8 sensitivity analysis is provided on the separate
[alignment page](chicken_heart_alignment.md).

If you set a different `CYTOBRIDGE_HEART_OUTPUT_DIR` in the tutorial, pass
that same directory with `--output-dir` in these commands.

## Start directly from the model

If you have **not** run the chicken-heart tutorial, first calculate its
populations, then run the analysis:

```bash
python scripts/run_chicken_heart_daily_notebooks.py --project-dir my_project --step interpolation
python scripts/run_chicken_heart_daily_notebooks.py --project-dir my_project --step daily
```

The four source notebooks contain these calculations:

1. Calculate daily populations and save their H5AD slices.
2. Read those slices to calculate lineage transitions and draw spatial, velocity,
   and interaction-network plots.
3. Examine velocity within the D10 population.
4. Calculate growth and draw the supplementary growth, alignment, and velocity plots.

Results are saved in `my_project/outputs/chicken_heart_paper/`.
Its `notebooks/` folder contains the executed notebooks. Its
`new_runs_formal_trained/` folder contains each notebook's numerical results and plots.
To save a second calculation separately, add `--output-dir my_second_run`.

The [notebooks and plotting functions on GitHub](https://github.com/zhenyiizhang/cytobridge-spatial/tree/main/reproduction/chicken_heart)
contain the individual calculations. Run the first notebook before the others,
because it creates the daily slices they read.

The daily simulations use observed populations at D4, D7, D10, and D14 as
the starting points of successive intervals. Their cell-type classifier and
sampling settings are recorded alongside the code.
