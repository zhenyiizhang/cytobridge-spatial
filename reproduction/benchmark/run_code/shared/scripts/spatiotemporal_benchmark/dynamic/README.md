# Dynamic benchmark adapters

This directory provides auditable, dataset-configured adapters for the
official stVCR, STORIES, and MIOFlow implementations. The adapters fit external
methods; they do not reimplement or silently substitute any method.

## Input contract

Commands take the root input `manifest.json` emitted by
`scripts/spatiotemporal_benchmark/build_inputs.py` and a split ID. They resolve
only these training artifacts:

- `splits[split].train.h5ad`
- `splits[split].train.training_reference_npz`
- `splits[split].train.source_roster_npz`
- `uns['cytobridge_benchmark_contract']` in the train H5AD

The H5AD and NPZ state, spatial, time, and row-ID arrays must agree exactly up
to float serialization tolerance, and both artifact SHA-256 values must match
the root manifest. Truth paths are never resolved or opened.

The implementation is dataset-agnostic: state/spatial dimensions and keys come
from the H5AD contract. Predictions remain in the original frozen shared state
and spatial coordinates. No adapter refits PCA, aligns/warps space, or applies a
display transform.

## Regimes

`loto_t*` fits one model on a train H5AD from which the requested target rows
are physically absent. It reuses the builder-frozen 5,000-particle source
roster from the nearest earlier observed stage.

`full_data` fits each method exactly once on all observed stages. Fit also
freezes one 5,000-particle roster at the initial stage (t0 for the five-stage
benchmark). Every target prediction reuses that exact roster and runs directly
from t0 to the requested target; inference does not reset from the preceding
observed stage. The stochastic seed is shared across the four full-data targets
to preserve a common trajectory prefix where an official simulator supports
it. The roster is created once at the immutable input boundary from a
deterministic 800-row support, so all benchmark families receive identical
starting row IDs and values for the split.

`prediction_n=5000` is read from and checked against the train contract before
any prediction. It is never inferred from truth or target cell count.

## Method-native outputs

- stVCR fits state plus spatial coordinates and emits native joint output.
  `use_alignment=true` is rejected. With `use_growth=true`, the official
  simulator's native row count is preserved rather than forced back to 5,000.
  If the audited simulator does not expose explicit particle weights, each
  native output particle receives mass `1/5000`, so total output mass is
  `native_output_n/5000`.
- STORIES fits with state and spatial coordinates, but its official public
  transform emits state only. No spatial output is invented.
- MIOFlow is state-only. Its official train-only z transform is saved, hashed,
  verified against the fitted model, and inverted before prediction export.

## Official source and environment provenance

`--source-root` must be the git checkout root (not its `src/` package directory)
and must have no tracked modifications. `method_pins.json` is enforced by
default: stVCR `26aa79a...`, STORIES `7d8269b...`, and MIOFlow `3636540...`.
The expected server checkout roots are typically
`$BENCH/software/stVCR`, `$BENCH/software/stories`, and
`$BENCH/software/MIOFlow`. A deliberately revised audited registry can be
passed with `--pins /path/to/method_pins.json`; the registry hash is recorded.
The source commit/remote, method version, official API signatures, Python
executable, relevant dependency versions, environment fingerprint, root input
manifest hash, train H5AD hash, training-reference hash, fit-manifest hash, and
all output hashes are recorded. A fit and its inference must use the same
source commit and environment fingerprint.

## Commands

Preflight one method/split:

```bash
python scripts/spatiotemporal_benchmark/dynamic/run_dynamic.py preflight \
  --method stvcr \
  --input-manifest /path/to/inputs/manifest.json \
  --split-id loto_t2 \
  --source-root /path/to/official/stVCR
```

Fit a full-data model once:

```bash
python scripts/spatiotemporal_benchmark/dynamic/run_dynamic.py fit \
  --method stories \
  --input-manifest /path/to/inputs/manifest.json \
  --split-id full_data \
  --source-root /path/to/official/STORIES \
  --output-dir /path/to/runs/stories/full_data/fit \
  --seed 20260718
```

Reuse it for every configured target:

```bash
for target in 1 2 3 4; do
  python scripts/spatiotemporal_benchmark/dynamic/run_dynamic.py infer \
    --method stories \
    --input-manifest /path/to/inputs/manifest.json \
    --split-id full_data \
    --source-root /path/to/official/STORIES \
    --fit-dir /path/to/runs/stories/full_data/fit \
    --target-time "$target" \
    --output-dir "/path/to/runs/stories/full_data/t${target}" \
    --seed 20260718
done
```

Method defaults are in `DEFAULT_PARAMS` in `run_dynamic.py`. Audited overrides
can be supplied as inline JSON or a JSON file with `--params-json`.

The matched STORIES profile fixes `max_iter=100`, `batch_size=128`, and
`restore=false`. A real-data calibration on this benchmark measured the
upstream 2,000-iteration, batch-1,000 setting at roughly 135 seconds per
iteration after compilation (about 300 hours for four independent fits), while
10 iterations at batch 128 completed in 45 seconds. The fixed profile therefore
uses a declared, feasible from-scratch budget rather than silently restoring a
checkpoint or leaving a multi-week default run incomplete. These parameters
are recorded in every fit manifest and should be treated as the benchmark
compute profile, not as a claim that the upstream package defaults changed.

## Outputs

Fit directory:

- `fit_manifest.json`
- `summary.json` (fit-summary compatibility alias)
- `source_roster.npz`
- official model/checkpoint artifacts
- MIOFlow only: `state_transform.npz`

Inference directory:

- `prediction.npz` with `state`, optional `spatial`, and optional native
  `weights`
- `run_manifest.json`
- `summary.json` (the same payload for orchestration compatibility)

## Mock tests

The tests do not run an external package:

```bash
PYTHONDONTWRITEBYTECODE=1 \
python -m unittest discover \
  -s scripts/spatiotemporal_benchmark/dynamic/tests -v
```

They cover physical LOTO exclusion, H5AD/NPZ/SHA validation, a single full-data
fit reused from the same t0 roster for t1-t4, variable stVCR growth count and
mass, exact 5,000-row state-only output, and MIOFlow inverse transformation.
