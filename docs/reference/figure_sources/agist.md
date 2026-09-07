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

New random interaction groupings and floating-point arithmetic can change
the numerical summaries slightly. To draw the recorded trajectories, select
their evaluation directory in the notebook.
