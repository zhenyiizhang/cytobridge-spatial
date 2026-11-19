---
orphan: true
---

# Supplementary Figure S4: finite-range cell interactions

The [S4 notebook](../../tutorials/paper_figures/agist_figures.ipynb) simulates
400 particles using the radial-interaction model, then calculates the
trajectory, growth and interaction comparisons.

The `simulation` download supplies the observations, reference trajectories
and trained model. Evaluation uses the Score_Refine checkpoint with seeds
1, 4, 8, 32 and 256. The drift and interaction terms drive the simulated
trajectories. The code compares predictions with interaction switched on and
off, calculates population mass, and evaluates the radial interaction curve.

For the AGIST benchmark in Figure 2 and S2–S3, use the
[Figure 2 notebook](../../tutorials/paper_figures/main_figure_2.ipynb).

## Regenerate the S4 observations and references

The downloaded benchmark is sufficient for model evaluation. To repeat its
original data generation, run this command in the CytoBridge code folder:

```bash
python -m scripts.run_spatial_synthetic_benchmark generate \
  --data-dir outputs/attraction_reference_new \
  --version spatial_attraction_2d_gene_2d_space_v8_balanced_joint_interaction \
  --n-particles 400 --interaction-strength 0.5 --gene-interaction-gain 3.0
```

It writes both observed H5AD/CSV files, fixed-particle reference NPZ files and
the complete simulator specification. Select this directory with the
evaluation command's `--data-dir`.

## Train the S4 model

S4 uses a radial-interaction model. Its training implementation and configuration
are supplied together in a small additional download. Download the observations
and model first if you have not run the notebook:

```bash
python -m CytoBridge.datasets simulation --output-dir .
python -m CytoBridge.datasets simulation --output-dir . \
  --kind simulation_training_code.zip
```

This creates `data/simulation/training_code/`. Use the Python environment from
[Installation](../../installation.md). The training command imports the radial
model implementation in that folder and writes a separate model directory:

```bash
python data/simulation/training_code/train.py \
  --input-h5ad data/simulation/data/attractive_observed.h5ad \
  --output-dir outputs/attraction_model_new --device cuda:0
```

The six stages use 100, 100, 50, 2,001, 1,000 and 2,001 epochs, respectively.
Their settings are in `data/simulation/training_code/training.yaml`.
To use regenerated observations, replace the input above with
`outputs/attraction_reference_new/attractive_observed.h5ad`.

Evaluate the final `Score_Refine` stage on the five inference seeds:

```bash
python -m scripts.run_spatial_synthetic_benchmark evaluate \
  --data-dir data/simulation/data --model-dir outputs/attraction_model_new \
  --stage Score_Refine --evaluation-dir outputs/attraction_evaluation_new \
  --seeds 1,4,8,32,256 --no-score --no-plots --device cuda:0
```

For regenerated observations, use `--data-dir outputs/attraction_reference_new`
in this command too. The evaluation writes trajectories, population-mass
comparisons and the radial-response table. To pass them to the S3–S4 notebook,
start it with this directory selected:

```bash
CYTOBRIDGE_ATTRACTION_EVALUATION="$PWD/outputs/attraction_evaluation_new" \
CYTOBRIDGE_ATTRACTION_DATA="$PWD/data/simulation/data" \
  jupyter notebook docs/tutorials/paper_figures/agist_figures.ipynb
```

For regenerated observations, set `CYTOBRIDGE_ATTRACTION_DATA` to
`$PWD/outputs/attraction_reference_new` instead. The same directory supplies
both the evaluation's reference and S4a's measured snapshots.

Alternatively, let the notebook perform the evaluation: omit
`CYTOBRIDGE_ATTRACTION_EVALUATION`, and set `CYTOBRIDGE_ATTRACTION_MODEL` to
`$PWD/outputs/attraction_model_new`, together with the matching data directory.
The notebook calculates S4's panel tables and draws them. `--no-plots` skips
the evaluator's diagnostic figures, leaving the notebook to draw S3–S4.

New random interaction groupings and floating-point arithmetic can change
the numerical summaries slightly. To draw the recorded trajectories, select
their evaluation directory in the notebook.
