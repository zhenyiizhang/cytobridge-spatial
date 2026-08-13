# Script status

The supported public entry point is `cytobridge workflow`. Its implementation
lives in `CytoBridge.workflow`, so new datasets and scientific fixes should be
added to the package rather than by copying a full pipeline into another
script.

The source distribution also includes maintained helpers for preprocessing,
training, checkpoint conversion, notebook and wheel smoke tests, training-cost
summaries, the matched spatiotemporal benchmark, and the reviewer analyses
documented in this repository. `complete_downstream.py` is a compatibility
alias for `cytobridge workflow`; `run_arista_end_to_end.py` is the same kind of
thin alias with the packaged `arista` preset already selected.

`verify_historical_artifact_compatibility.py` is a read-only maintainer check
for comparing a checkpoint through its original source loader and the current
package loader. Start from
`historical_artifact_compatibility.example.json`; private machine paths are not
stored in this repository.

`run_zebrafish_interval_daughter_noise_sensitivity.py` is the maintained
daughter-noise sensitivity entry point. It runs independent one-sided
midpoint forecasts from every observed left anchor, optionally adds generated
right-end forecasts, and requires exact hashes for the aligned H5AD, learned
model checkpoints, label classifier, and four-dataset acceptance report. Its
lineage keys are interval-local `(anchor_time, source_obs_id)` namespaces and
must never be joined across intervals. Raw states are always retained, and the
paired noise-0 table includes composition/count/lineage deltas plus joint-state
and spatial empirical W1/W2 metrics.

`plot_zebrafish_interval_daughter_noise_sensitivity.py` is the package-owned
publication renderer for that analysis. It requires the producer's exact
`run_manifest.json` SHA-256 and adjacent sidecar plus the exact canonical
acceptance-report artifact and SHA-256. Before rendering, it revalidates the
canonical manifest signature, frozen interval-local settings, all six CSV
artifacts, all 80 retained raw-state hashes, and the acceptance binding. It
plots only paired midpoint changes from daughter noise zero and writes an A4
vector PDF, a 320-dpi PNG, a caption, the plotted mean/SEM table, provenance,
and a signed figure manifest into a new or empty directory:

```text
python scripts/plot_zebrafish_interval_daughter_noise_sensitivity.py \
  --run-manifest <sensitivity-root>/run_manifest.json \
  --expected-manifest-sha256 <run-manifest-sha256> \
  --acceptance-report <canonical-acceptance.json> \
  --expected-acceptance-sha256 <acceptance-sha256> \
  --output-dir <new-publication-bundle-dir>
```

`run_matched_ablation_matrix.py` is the fail-closed server launcher for the
formal four-dataset × three-arm comparison. It accepts exactly one shared
aligned H5AD per dataset, the validation-selected learned predictor plus its
sidecar and input-graph provenance for each full arm, and an explicit CUDA
index for every validator profile. `dry-run` hashes and renders the complete
plan without writing; `prepare` creates a new root once and links the immutable
inputs; `launch-one` starts only one train-only or downstream-only command after
a literal profile confirmation; `status` reports monitor/child PIDs and output
summaries. The no-LR-prior and no-interaction directories never receive graph
or predictor artifacts. The tool rejects a dirty or mismatched release commit,
changed inputs, changed configs, changed package code, existing output roots,
phase reuse, and concurrent reuse of a planned GPU. Use `render` when commands
will be submitted through an external scheduler instead of launched directly.
The commands use the bound server Python as `python -m CytoBridge.cli`, expose
only the assigned physical GPU, address it as logical `cuda:0`, set
`PYTHONHASHSEED=42` before interpreter startup, and use condition-specific
Numba, Matplotlib, and XDG cache roots. The four aligned inputs are accepted
only when their SHA-256 digests exactly match the immutable inputs recorded by
the packaged unified-benchmark configs; a valid H5AD from another dataset is
therefore rejected before launch.

The required assignment keys can be printed with
`python scripts/run_matched_ablation_matrix.py dry-run --help`. A formal call
has this shape (each repeated group must be complete):

```text
python scripts/run_matched_ablation_matrix.py prepare \
  --run-root /data/runs/cytobridge-matched-<release-sha> \
  --release-root /opt/src/CytoBridge \
  --release-commit <40-character-release-commit> \
  --python-executable /data/cytobridge/projects/CytoBridge-ST-1104/envs/arista-api/bin/python \
  --aligned-h5ad zebrafish=/data/accepted/zebrafish_aligned.h5ad [...] \
  --edge-predictor zebrafish=/data/accepted/zebrafish_edge_model.pt [...] \
  --input-graph-dir zebrafish=/data/accepted/zebrafish/input_graph [...] \
  --gpu zebrafish=0 --gpu zebrafish_no_lr_prior=1 \
  --gpu zebrafish_no_interaction=2 [...all 12 profiles...]
```

After preparation, launch one fit with, for example,
`python scripts/run_matched_ablation_matrix.py launch-one --run-root <root>
--profile zebrafish --phase train --confirm-profile zebrafish`. Launch its
downstream phase only after that train-only fit finishes. The manifest renders
the final validator command with all twelve `--datasets` and all four repeated
`--matched-family` arguments.

`run_matched_ablation_benchmark_evaluation.py` is the package-owned quantitative
evaluation and reporting entry point for those twelve accepted profiles. It
binds the matched acceptance report and launcher manifest, the four unified
benchmark input manifests, all twelve resolved training configs and summaries,
all 72 stage checkpoints, and the exact adapter/evaluator implementation. Its
`prepare` command creates a new evaluation root; `render` prints twelve official
`infer-full` commands and twelve frozen `evaluate_predictions` commands for an
external scheduler but never launches them. `validate` fails closed unless all
prediction and score contracts remain paired across the full, radius-only, and
no-interaction arms. `report` produces arm-labelled metric tables, paired
ablation-minus-full deltas by dataset/target/space, an A4 PDF, a 320-dpi PNG,
caption, provenance, and a signed report manifest. This is strictly an
in-sample `full_data` reconstruction comparison: generic downstream directories
are not reconstruction outputs, and the report must not be read as LOTO
generalization evidence.

```text
python scripts/run_matched_ablation_benchmark_evaluation.py prepare \
  --run-root <new-evaluation-root> \
  --launcher-manifest <matched-root>/_matched_launcher/matched_ablation_matrix_manifest.json \
  --expected-launcher-manifest-sha256 <launcher-sha256> \
  --matched-acceptance <matched-acceptance.json> \
  --expected-matched-acceptance-sha256 <acceptance-sha256> \
  --benchmark-input zebrafish=<zebrafish-input-manifest.json> [...] \
  --expected-benchmark-input-sha256 zebrafish=<manifest-sha256> [...]
python scripts/run_matched_ablation_benchmark_evaluation.py render \
  --run-root <new-evaluation-root>
```

Other top-level files under `scripts/` in the Git repository are retained as
historical research records. Some contain workstation-specific paths or calls
from earlier package versions. They are not installed, are not included in the
source distribution, and should not be used as a starting point for new work.
