# Reusable spatiotemporal benchmark inputs

`build_inputs.py` creates the immutable input boundary shared by CytoBridge and
external temporal/spatial baselines. Dataset choices live in YAML; the Python
code contains no zebrafish or ARISTA time labels, paths, dimensions, or target
sets.

The checked-in zebrafish configuration is
`configs/zebrafish_clean_benchmark.yaml`. It freezes:

- corrected `obsm['X_latent']` (50-dimensional state) and
  `obsm['spatial_aligned']` (two-dimensional space);
- model times `0..4`;
- LOTO targets `t1,t2,t3`;
- full-data evaluation targets `t1,t2,t3,t4`;
- `prediction_n=5000`, selected before any held-out truth is opened;
- a method-independent 800-row source support and deterministic 5,000-particle
  bootstrap roster for every split;
- the corrected source H5AD SHA-256 and clean-count preprocessing provenance.

## Protocol boundary

Each `loto_tN/train.h5ad` physically excludes every row at `tN`. Its truth
H5AD/CSV/NPZ contains only `tN`. The PCA and spatial coordinates are copied
from the common corrected preprocessing and are not refit per fold. LOTO is
therefore a **transductive frozen-representation** benchmark, not an inductive
raw-gene holdout.

`full_data/train.h5ad` contains all stages. Full-data truth is an identical
hard link (or byte copy when hard links are unavailable), while only configured
evaluation stages `truth_t1.npz` through `truth_t4.npz` are exported. This is
in-sample reconstruction and must be reported separately from LOTO.

## Validate, build, and verify

Run from the repository root on the server:

```bash
CONFIG=configs/zebrafish_clean_benchmark.yaml
OUT=/data/cytobridge/projects/CytoBridge-ST-1104/runs/zebrafish-api/benchmarks/zebrafish-clean-20260718

python scripts/spatiotemporal_benchmark/build_inputs.py \
  --config "$CONFIG" \
  --validate-only

python scripts/spatiotemporal_benchmark/build_inputs.py \
  --config "$CONFIG" \
  --output-dir "$OUT"

python scripts/spatiotemporal_benchmark/verify_inputs.py \
  --output-dir "$OUT" \
  --verify-source
```

`--validate-only` is read-only: it rehashes and inspects the configured source,
checks dimensions/times/annotations/layers, and enforces the preprocessing
contract without creating an output directory. Hash-bound JSON sidecar audits
listed in `preprocess_contract.external_audits` are also verified, including
their configured nested assertions. The normal build refuses a
non-empty `OUT/inputs`. `--overwrite` explicitly replaces only that `inputs`
subdirectory.

The verifier is independent of the source by default and rehashes every output
artifact, both manifest SHA sidecars, row membership/order, CSV aliases, NPZ
shapes, and H5AD contracts. Add `--verify-source` while the source H5AD is
available to rehash it too.

## Output layout

```text
OUT/inputs/
├── manifest.json
├── manifest.json.sha256
├── resolved_config.yaml
├── column_definitions.json
├── full_data/
│   ├── manifest.json[.sha256]
│   ├── train.{h5ad,csv}
│   ├── truth.{h5ad,csv}
│   ├── training_reference.npz
│   ├── source_roster.npz
│   ├── truth.npz
│   └── truth_t{1,2,3,4}.npz
└── loto_t{1,2,3}/
    ├── manifest.json[.sha256]
    ├── train.{h5ad,csv}
    ├── truth.{h5ad,csv}
    ├── training_reference.npz
    ├── source_roster.npz
    ├── truth.npz
    └── truth_tN.npz
```

Every H5AD retains source `X`, all layers (including `layers['counts']`), all
cell/gene metadata, source representations, and adds canonical benchmark fields.
The machine-readable contract is stored in
`uns['cytobridge_benchmark_contract']` (the key is YAML-configurable).

CSV files contain canonical metadata, all source `obs` columns, named spatial
and state columns, and legacy joint aliases `x1..x52`. Compact NPZ files contain
`spatial`, `state`, `time`, `row_id`, and `annotation`. The combined
`truth.npz` mirrors `truth.h5ad`; evaluators should use the single-stage
`truth_tN.npz` files.

`source_roster.npz` is frozen by the input builder, not by an individual
method. It first selects a deterministic source-stage support of
`source_roster_support_n` rows and then bootstraps exactly `prediction_n`
particles from that support. CytoBridge, dynamic methods, and static coupling
adapters verify and reuse the same row IDs and values for a split.

## Evaluation and reporting

Run LOTO and full-data evaluation into separate directories. Always pass the
complete primary method list so an entirely missing method fails during
evaluation. The summarizer verifies the generated evaluation manifest, metrics
CSV hash, complete grid, and reporting registry:

```bash
python scripts/spatiotemporal_benchmark/evaluate_predictions.py \
  --input-manifest "$OUT/inputs/manifest.json" \
  --predictions-root "$OUT/predictions/loto" \
  --track loto \
  --output-dir "$OUT/reports/evaluation/loto" \
  --methods CytoBridge-0.015 stvcr stories mioflow moscot wot paste spateo \
            linear_centroid_shift random_independent_pairs

python scripts/spatiotemporal_benchmark/summarize_results.py \
  --metrics-long "$OUT/reports/evaluation/loto/loto_metrics_long.csv" \
  --evaluation-manifest \
    "$OUT/reports/evaluation/loto/loto_evaluation_manifest.json" \
  --method-registry scripts/spatiotemporal_benchmark/method_registry.json \
  --output-dir "$OUT/reports/summary/loto"
```

Repeat with `--track full_data` and the corresponding paths. Full-data scores
are in-sample reconstruction references; they must not be pooled with or used
to rank LOTO generalization results.

### Optional matched LOTO-versus-full-data diagnostic

`evaluate_matched_tracks.py` is a separate, opt-in diagnostic; it does not
change the primary per-track evaluator above. It restricts evaluation to the
targets shared by both tracks and fits one normalization transform only after
proving that `row_id`, state, spatial, and time arrays at the explicitly named
anchor times are byte-identical in every participating training split. For the
zebrafish comparison, the common anchors are `t0` and `t4`:

```bash
python scripts/spatiotemporal_benchmark/evaluate_matched_tracks.py \
  --input-manifest "$OUT/inputs/manifest.json" \
  --loto-predictions-root "$OUT/predictions/loto" \
  --full-data-predictions-root "$OUT/predictions/full_data" \
  --anchor-times 0 4 \
  --output-dir "$OUT/reports/evaluation/matched" \
  --methods CytoBridge-0.015 stvcr stories mioflow moscot wot paste spateo \
            linear_centroid_shift random_independent_pairs
```

Before evaluating predictions, the script also proves directly from NPZ arrays
that each LOTO training reference is the byte-exact non-target subset of the
full-data reference, that train/truth row IDs are disjoint complements, and
that both track-specific truth files equal the full-data target subset.

The two tracks use the same projection seed for each target/space/repeat. Exact
OT uses separate predicted and observed RNG streams: one observed index set is
fixed, recorded and shared across every method and track for a target/space, so
different prediction counts or weights cannot change the observed subsample.
Outputs comprise a repeat-level long CSV, a semantic split audit, an exact-OT
index audit, the common transform and anchor-byte audit, and a paired
method/space/target summary. Code hashes for this evaluator, the primary
evaluator, `benchmark.py` and `evaluation.py`, plus Python and numerical-library
versions, are embedded in the final manifest.

The output directory is immutable and bound by `run_contract.json`. An
immutable, canonically sorted `prediction_inventory.json` binds the exact NPZ
and summary byte snapshots selected for evaluation, and
`bound_run_contract.json` binds its SHA to the base run contract. Metric
calculation uses only the parsed in-memory snapshots. External prediction files
are rehashed both before metrics and immediately before final publication. An
interrupted run may resume only with byte-identical contracts, inventory and
outputs; changed parameters, predictions or partial outputs are rejected. Once
`matched_evaluation_manifest.json` exists, use a new output directory for any
rerun. Reported LOTO-minus-full gaps are descriptive: full-data is explicitly
marked in-sample, each fitted prediction represents one model-training seed,
and no cross-space score or ranking is produced. TMV is included only for
declared native unnormalised mass; its delta is withheld when the two tracks use
different source-time denominators.

## Adapting another dataset

Copy the YAML and change the source path/hash, time key and explicit mapping,
representation keys/dimensions, annotation key, target lists, fixed population
size, and provenance assertions. All can also be overridden from the CLI:

```bash
python scripts/spatiotemporal_benchmark/build_inputs.py \
  --config configs/my_dataset.yaml \
  --h5ad /path/to/aligned.h5ad \
  --time-key model_time \
  --time-map '[["day0",0],["day2",1],["day5",2]]' \
  --loto-targets 1 \
  --full-data-targets 1,2 \
  --prediction-n 5000 \
  --output-dir /new/run/root
```

Use `--preprocess-contract-json @contract.json` to replace the YAML provenance
contract for a dataset with a different legitimate preprocessing recipe. Do not
weaken that contract merely to make an incompatible or double-transformed input
pass. If a legacy H5AD lacks a provenance field that was verified later, bind
the immutable audit JSON by path, SHA-256, and `required_exact` assertions under
`external_audits` instead of editing the H5AD or pretending the field existed.
