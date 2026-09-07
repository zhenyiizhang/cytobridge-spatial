---
orphan: true
---

# AGIST and attraction: inputs for S2–S3

The [S2–S3 notebook](../../tutorials/paper_figures/agist_figures.ipynb) starts
from downloaded observations and trained models. Run its cells in order; it
does not require another analysis notebook or a new training run.

For S2, `prepare_agist_state_clusters.py` fits the state partitions from
`mouse_brain_simulation.csv`. `evaluate_velocity` evaluates the fitted intrinsic
drift, score gradient and grouped interaction on every cell. `velocity_inputs`
calculates cosines against the known generator fields, separately for the two
spatial and fifty gene-state coordinates. The recorded generator fields are
benchmark reference data, not fitted predictions.

For S3, the `simulation` download supplies the observed benchmark, reference
trajectories and the six-stage model. The notebook evaluates the Score_Refine
checkpoint for seeds 1, 4, 8, 32 and 256 with `--no-score`, as in the paper.
It calculates interaction-on/off distances, population mass and radial forces.
The radial model loader uses the original architecture and selected checkpoint.

`collect_agist_inputs` writes all S2/S3 numerical inputs to a new directory.
It exports the observed snapshots, the same 60 selected particle identities,
and the evaluation tables. The notebook passes that directory and its loaded
data object directly to `draw_supplementary`.

## Regenerate the S3 observations and references

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

## Train the S3 model

S3 uses a radial-interaction model. Its training implementation and configuration
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
comparisons and the radial-response table. To pass them to the S2–S3 notebook,
start it with this directory selected:

```bash
CYTOBRIDGE_ATTRACTION_EVALUATION="$PWD/outputs/attraction_evaluation_new" \
CYTOBRIDGE_ATTRACTION_DATA="$PWD/data/simulation/data" \
  jupyter notebook docs/tutorials/paper_figures/agist_figures.ipynb
```

For regenerated observations, set `CYTOBRIDGE_ATTRACTION_DATA` to
`$PWD/outputs/attraction_reference_new` instead. The same directory supplies
both the evaluation's reference and S3a's measured snapshots.

Alternatively, let the notebook perform the evaluation: omit
`CYTOBRIDGE_ATTRACTION_EVALUATION`, and set `CYTOBRIDGE_ATTRACTION_MODEL` to
`$PWD/outputs/attraction_model_new`, together with the matching data directory.
The notebook calculates S3's panel tables and draws them. `--no-plots` skips
the evaluator's diagnostic figures, leaving the notebook to draw S2–S3.

New random interaction groupings and floating-point arithmetic can change
the numerical summaries slightly. To draw the recorded trajectories, select
their evaluation directory in the notebook.
