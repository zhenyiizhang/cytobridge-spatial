# Limitations

## Data and model inputs

- Large study datasets and trained checkpoints are not included in the wheel.
  Users must provide aligned inputs or run preprocessing and training.
- Gene reconstruction requires the PCA loading matrix, gene order, and center
  used during fitting. A partial reference AnnData cannot supply a replacement
  center.
- Ligand--receptor outputs are limited to genes and complete complexes in the
  supplied database.

## Simulation and labels

- Intermediate states are model simulations. In the dataset tutorials they
  start from the preceding observed slice and are not conditioned on the next
  observed endpoint.
- Persistent particle identifiers are required for lineage summaries. They are
  unavailable for the current ARISTA trajectory files.
- Classifier smoothing changes the balance between local spatial consistency
  and pointwise labels. The example dataset configurations use `k=10` for Zebrafish, MOSTA,
  and ARISTA, and `k=1` for AD mouse and chicken heart.

## Communication

- Sparse communication depends on the analyzed cohort, spatial cutoff, and
  edge gate. A time point may contain candidates but no retained edges.
- Attention and LR tables are model-derived summaries. They are not direct
  measurements of molecular transport.
- The No-interaction model does not produce communication or LR outputs. The
  No-LR model uses all spatial candidates instead of a learned LR-
  informed gate.

## Evaluation

- In-sample reconstruction diagnostics and leave-one-timepoint-out benchmarks
  answer different questions and should be reported separately.
- Sliced-W2 projection repeats measure numerical projection variation, not
  independent biological or training replicates.
- Methods without a native output in a requested space remain unavailable for
  that method/space combination. The protocol does not synthesize unsupported
  outputs.
- The packaged comparisons do not cover every graph, optimizer, and solver
  setting.

## Compute measurements

- Training time and memory depend on hardware, dependency versions, dataset
  size, and configuration.
- The packaged compute table contains one measured full-model run per dataset,
  not repeated-run averages.
- Neural-ODE and score-matching losses optimize different objectives and
  should not be combined into one continuous loss curve.
