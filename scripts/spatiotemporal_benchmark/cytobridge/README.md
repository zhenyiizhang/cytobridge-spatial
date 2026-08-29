# CytoBridge-0.015 benchmark adapter

This directory is the dataset-agnostic CytoBridge adapter for the immutable
`cytobridge_benchmark_contract` produced by
`scripts/spatiotemporal_benchmark/build_inputs.py`. Dataset paths, time labels,
state/spatial keys, dimensions, targets, and row IDs come from the root input
manifest and train H5AD; the adapter contains no zebrafish-specific labels or
paths.

The shared scientific profile is:

- `alpha_spatial=10`, `alpha_express=0.015`, `sigma=0.03`, seed 42;
- the exact six-stage plan in the selected package YAML. Interaction-enabled
  profiles use `Init_interaction` and `Finetune`; matched no-interaction
  profiles use `Matched_stage_3_no_interaction` and
  `Finetune_no_interaction`, with the same stage count and matched scientific
  settings but `v+g` rather than `v+g+i` in those neural-ODE stages;
- either the full `velocity/growth/score/interaction` component set or the
  explicit no-interaction `velocity/growth/score` component set;
- the exact architecture, epochs, batch sizes, losses and scheduler loaded
  from the selected data-set package YAML. These intentionally differ where
  the package recipe differs (for example, AD uses 3,001 score epochs);
- the interaction prior declared by that YAML: `learned` profiles require a
  validated edge-predictor path and threshold, while `all_spatial` profiles
  retain every within-cutoff candidate inside each stochastic interaction
  group and reject predictor fields. Profiles with no interaction component
  have `interaction_mode=edge_prior_mode=none` and reject cutoffs, predictor
  metadata, and interaction-graph metadata altogether;
- exactly 5,000 source particles from the builder-frozen, method-independent
  canonical source roster, verified before any truth artifact is opened;
- raw two-dimensional frozen spatial coordinates followed by the frozen state
  coordinates; no representation refit and no spatial display warp;
- the official continuous, non-split weighted SDE simulator with native,
  unnormalised growth weights.

All commands refuse incompatible keys or profiles, incomplete stage folders,
target leakage, non-empty output directories, or output overwrite.
They resolve portable artifact `relative_path` values against the root
manifest's `inputs/` directory and never resolve/open `truth*` artifacts.

## Regimes

### LOTO (`loto_t1`, `loto_t2`, `loto_t3`)

For every fold, the held-out target rows must be physically absent. For a
`learned` prior, the adapter regenerates the within-stage ligand-receptor
interaction graphs for the remaining stages and retrains the edge classifier.
For an `all_spatial` no-LR-prior ablation, it prepares only the cutoff declared
by the matched training config: no LR graph or edge-classifier artifact is
created, recorded, or passed to fitting. For a no-interaction ablation,
`prepare-loto` writes only a mode/provenance summary: it does not open the
training H5AD to generate a graph, and fitting receives no cutoff, graph, or
predictor argument. The adapter then runs the profile's exact six stages from
scratch and continuously simulates from the nearest previous observed training
stage to the held-out target. No-interaction inference uses only velocity,
growth, and score. The AD production benchmark remains the corrected learned
predictor trained from its seven strictly retained LR pairs; radius-only AD is
reported only as an ablation or sensitivity analysis.

### Full data / no holdout (`full_data`)

The complete `alpha_express=0.015` checkpoint is reused only after all six stage
files, its resolved package config, interaction-prior mode, and training data
are verified. Learned-mode Finetune checkpoints embed the predictor parameters,
so they remain loadable after being moved. An `all_spatial` checkpoint has no
predictor dependency at all; validation rejects stale predictor metadata. A new
no-interaction checkpoint likewise has no cutoff, graph, predictor, or
interaction-group dependency and is simulated with
`include_interaction=false`. For every mode, both LOTO and full-data reuse
require a field-by-field match to the selected reference package YAML. A new
adapter fit proves linkage via
`benchmark_fit_summary.json`; an existing locked fit without that summary is
accepted only when its saved `adata.h5ad` has the exact frozen state, spatial,
time, and row-order arrays in `training_reference.npz`.

Inference reuses one fixed 5,000-particle t0 bootstrap frozen by the input
builder and makes exactly one non-split SDE call with
`ts_points=[t0,t1,t2,t3,t4]`. Predictions at t1-t4 are
snapshots from that same continuous trajectory: there is no reset at an
intermediate observed stage and no spatial warp. This full-data result is an
in-sample reconstruction reference and must never be pooled into the LOTO
generalization ranking.

## Commands

Run from the repository root. The example paths are placeholders; the adapter
uses the manifest rather than hard-coding a dataset.

```bash
PYTHON=python
ADAPTER=scripts/spatiotemporal_benchmark/cytobridge/run_cytobridge.py
INPUTS=/path/to/benchmark/inputs/manifest.json
CONFIG=CytoBridge/configs/zebrafish_spatial_full_alpha_express_0015.yaml
RUN=/path/to/benchmark/predictions/cytobridge_0015
DB=/path/to/CellNEST_database.csv
DEVICE=cuda:1
```

Read-only input/config preflight:

```bash
$PYTHON $ADAPTER preflight \
  --input-manifest "$INPUTS" \
  --split loto_t1 \
  --training-config "$CONFIG" \
  --device "$DEVICE"
```

For each LOTO target, rebuild training-only graphs/edge classifier, fit six
stages, and infer:

```bash
for TARGET in 1 2 3; do
  SPLIT=loto_t${TARGET}

  $PYTHON $ADAPTER prepare-loto \
    --input-manifest "$INPUTS" \
    --split "$SPLIT" \
    --training-config "$CONFIG" \
    --database "$DB" \
    --output-dir "$RUN/$SPLIT/graph_and_edge" \
    --device "$DEVICE"

  $PYTHON $ADAPTER fit-loto \
    --input-manifest "$INPUTS" \
    --split "$SPLIT" \
    --training-config "$CONFIG" \
    --graph-dir "$RUN/$SPLIT/graph_and_edge" \
    --output-dir "$RUN/$SPLIT/model" \
    --device "$DEVICE"

  $PYTHON $ADAPTER infer-loto \
    --input-manifest "$INPUTS" \
    --split "$SPLIT" \
    --model-dir "$RUN/$SPLIT/model" \
    --training-config "$CONFIG" \
    --output-dir "$RUN/$SPLIT/prediction" \
    --device "$DEVICE"
done
```

In learned mode, omitting `--interaction-cutoff` reads the frozen preprocessing
threshold from
`train.h5ad:uns['interaction_graph']['neighborhood_threshold']`; the adapter
then recomputes graph edges from training rows. When `--edge-threshold` is
omitted, the new fold-specific classifier uses its validation-selected
threshold. In `all_spatial` mode, the cutoff comes from the validated training
YAML; an explicit `--interaction-cutoff` is accepted only when numerically
equal to that value. `--edge-threshold` and `--spot-diameter` are rejected in
that mode, and no learned-graph artifacts are written. In learned mode,
`--database` is required. The AD production benchmark uses learned mode; its
all-spatial profile is a separately labelled no-LR-prior ablation. In
no-interaction mode, `--interaction-cutoff`, `--edge-threshold`,
`--spot-diameter`, and `--database` are all rejected; the `graph-dir` argument
to `fit-loto` names the prepare summary directory, not an interaction graph.

Validate and reuse an existing complete full-data `.015` model, then create all
four snapshots in one continuous call:

```bash
FULL_MODEL=/path/to/complete/alpha_express_0015/model

$PYTHON $ADAPTER validate-model \
  --input-manifest "$INPUTS" \
  --split full_data \
  --model-dir "$FULL_MODEL" \
  --training-config "$CONFIG" \
  --output-json "$RUN/full_data/model_validation.json"

$PYTHON $ADAPTER infer-full \
  --input-manifest "$INPUTS" \
  --split full_data \
  --model-dir "$FULL_MODEL" \
  --training-config "$CONFIG" \
  --output-dir "$RUN/full_data/prediction" \
  --device "$DEVICE"
```

`preflight --model-dir "$FULL_MODEL"` combines config/input/model checks in a
single read-only report.

Inference also binds `--interaction-m` exactly to the loaded checkpoint's
`model.interaction_group_size`; a mismatch is rejected instead of silently
changing the stochastic interaction grouping. That option is not applicable to
no-interaction checkpoints, which have no interaction group and always run
with `include_interaction=false`.

Communication and ligand-receptor interpretation are intentionally outside
this prediction-benchmark adapter. In particular, a no-interaction checkpoint
does not produce communication or LR outputs here; those analyses require a
separate, explicitly scoped downstream contract.

## Outputs

LOTO inference writes:

```text
prediction/
├── source_roster.npz
├── prediction.npz
├── prediction.summary.json
└── summary.json
```

Full-data inference writes one shared roster and one evaluator-compatible
`prediction.npz` per target:

```text
prediction/
├── source_roster.npz
├── run_summary.json
├── t1/{prediction.npz,prediction.summary.json,summary.json}
├── t2/{prediction.npz,prediction.summary.json,summary.json}
├── t3/{prediction.npz,prediction.summary.json,summary.json}
└── t4/{prediction.npz,prediction.summary.json,summary.json}
```

Each prediction NPZ contains `state`, `spatial`, and raw `weights`. Every
summary records `status`, `method='CytoBridge-0.015'`, regime, target,
source time, native joint/mass flags, manifest and training-reference hashes,
config and all six checkpoint hashes, seed/GPU/runtime provenance, source
roster hash, and the no-split/no-warp policy. These are the fields consumed by
`scripts/spatiotemporal_benchmark/evaluate_predictions.py`.

## Contract tests

The tests use tiny synthetic H5AD/NPZ/manifests and mock checkpoint bytes; they
do not require a GPU or run model training:

```bash
python -m unittest -v \
  scripts/spatiotemporal_benchmark/cytobridge/test_run_cytobridge.py
```

They verify train-only artifact resolution even when truth is deliberately
missing, physical LOTO removal, the single full-data t0-to-all-target schedule,
fixed deterministic 5,000-particle bootstrapping, raw-weight export, all four
package profiles, learned-predictor portability, radius-only execution without
predictor artifacts or arguments, no-interaction prepare/fit/infer execution
without graph dependencies, exact real-config reference matching, and
six-stage checkpoint completeness.
