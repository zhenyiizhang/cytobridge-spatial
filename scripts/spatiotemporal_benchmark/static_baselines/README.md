# Static/coupling benchmark adapters

This directory implements the static comparator side of the reusable
spatiotemporal benchmark. It consumes only a builder-produced training H5AD with
`uns['cytobridge_benchmark_contract']`. The primary builder keys are
`obsm['benchmark_state']`, `obsm['benchmark_spatial']`,
`obs['benchmark_time']`, and `obs['row_id']`; key names remain contract-driven
for other datasets. A truth H5AD/NPZ is never an adapter
input.

## Protocols

`loto` requires the requested target stage to be physically absent. The adapter
fits one coupling between the nearest observed left and right stages and
interpolates with the chronological fractional alpha. Prediction count comes
only from the training contract.

`no-holdout` (the legacy CLI alias `full-data` is accepted) fits every adjacent
coupling once: `P01`, `P12`, `P23`, `P34`. Predictions are generated from a
single deterministic t0 bootstrap roster by applying `P01`, `P01 @ P12`, and so
on. It never performs the former previous-stage to observed-target `alpha=1`
shortcut. One run writes t1 through t4 predictions, both separately and in a
combined `trajectory_prediction.npz`.

The bootstrap roster is the builder-frozen canonical source roster shared by
all benchmark families. It is verified by SHA/row ID and saved in
`source_roster.npz`.

## Methods and representation scope

| Method | Matched fit | Output | Note |
|---|---|---|---|
| MOSCOT | signed PCs + aligned space, train-anchor-only block balance | state + space | coupling projection |
| Waddington-OT | signed PCs only | state only | spatial/joint metrics are N/A |
| official PASTE | Euclidean signed-PC feature cost + space | state + space | hybrid coupling adapter |
| Spateo Morpho | Euclidean signed PCs + space | state + space | hybrid coupling adapter |
| spaTrack | N/A | N/A | official matched path is not legal for signed PCs |
| linear centroid shift | shared state + space | state + space | explicit control |
| random independent pairs | shared state + space | state + space | explicit one-hot coupling control |

spaTrack can be run only with `--representation native_gene_sensitivity`. That
non-primary sensitivity validates nonnegative, exactly-once log-normalized `X`,
fits the official expression/spatial transfer matrix, then applies that coupling
to the shared PCA/spatial output. Signed PCs are never shifted or passed as gene
expression.

Every external adapter is official-API-or-fail. Missing dependencies, invalid
coupling orientation, negative/non-finite mass, or zero rows produce a failure
manifest; no surrogate is substituted.

For the matched signed-PC Spateo profile, `nn_init=false` is fixed in the
registry. Exact-API preflight showed that the official nearest-neighbor
initializer is numerically unstable on signed PCs, while the same official
`morpho_align` optimizer with that initializer disabled returns a complete
800-by-800 mapping. The choice is recorded in every run manifest.

## CLI

```bash
python -m scripts.spatiotemporal_benchmark.static_baselines.run registry

python -m scripts.spatiotemporal_benchmark.static_baselines.run run \
  --method paste \
  --evaluation-mode loto \
  --target-time 2 \
  --input-h5ad /path/to/loto_t2/train.h5ad \
  --input-manifest /path/to/inputs/manifest.json \
  --output-dir /path/to/runs/loto_t2/paste \
  --source-root /path/to/official/PASTE

python -m scripts.spatiotemporal_benchmark.static_baselines.run run \
  --method paste \
  --evaluation-mode no-holdout \
  --input-h5ad /path/to/full_data/train.h5ad \
  --input-manifest /path/to/inputs/manifest.json \
  --output-dir /path/to/runs/no_holdout/paste
```

`run_manifest.json` records input and output SHA256 values, dependency version
and checkout provenance, fitted anchor shapes, coupling shapes, composition
formulae, output scope/hybrid status, and proof flags showing that truth and
target population size were not read.
