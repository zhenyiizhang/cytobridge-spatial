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

`prepare_chicken_heart_ot_input.py` converts the count-recovered, fixed-roster
GSE149457 handoff into the coordinate input consumed by the current
`chicken_heart` preset. It uses `spatial_original`, rotates only D7 by 180
degrees around its raw-stage centroid, removes fitted PCA/alignment state, and
writes `spatial_ot_input` plus a hash manifest. Reviewed coordinates are kept
only as `spatial_reviewed_reference` and are not used by OT.

`run_chicken_heart_paper_downstream.py` adds the formal chicken-heart
perturbation bank after the standard `cytobridge workflow --config
chicken_heart` downstream is complete. Every branch starts once from the real
D4 population (processed time 0) and evolves continuously to D14: the runner
does not replace generated states with D7/D10 observations. It runs three
equal-particle, fixed-population cell-type-removal sensitivities and one paired
interaction-on/off sensitivity, then exports spatial comparison grids,
composition/transport tables, captions, and a hash manifest. These are
single-seed model-sensitivity analyses rather than causal knockout or
uncertainty estimates.

```text
python scripts/run_chicken_heart_paper_downstream.py \
  --run-root <formal-training-root> \
  --input-h5ad <formal-training-root>/preprocess/chicken_heart_aligned.h5ad \
  --model-dir <formal-training-root>/training \
  --standard-downstream <formal-downstream-root>/downstream \
  --output-dir <new-chicken-paper-root> \
  --device cuda:0
```

`refresh_velocity_outputs.py` regenerates only the observed-slice velocity
archive and paired spatial/gene vector PDFs from an accepted aligned H5AD and
checkpoint directory. Use it when a completed downstream bank predates the
current velocity contract. Spatial arrows use the first two model dimensions
directly. Gene panels build the scVelo transition graph from the remaining 50
expression dimensions and project it onto the same observed spatial
coordinates. The command writes to a new directory and binds the exact aligned
H5AD and training-summary hashes.

`run_five_dataset_virtual_interaction_ablation.py` evaluates the five accepted
full-model checkpoints with the learned interaction force retained or set to
zero at inference. It never retrains. Each pair starts from the same earliest
observed cells, uses the same stochastic seed, disables growth-dependent
resampling, and evolves continuously without observed-slice re-anchoring. The
`run` subcommand writes one dataset result; `report` aggregates all five into
spatial/expression metrics and A4 figure banks. This is a fixed-model,
single-seed sensitivity analysis rather than a causal knockout or matched
retraining ablation.

`run_five_dataset_weighted_interaction_ablation.py` is the formal comparable
version of that fixed-checkpoint sensitivity. It uses the official continuous
non-split weighted SDE, a shared 5,000-particle source roster and random streams,
native unnormalised growth-mass weights, and the same weighted sliced-W2,
weighted exact W1/W2, and TMV definitions as the matched retraining evaluator.
Positive off-relative error means that disabling interaction worsened
reconstruction; negative values mean it improved reconstruction.

`run_nonspatial_communication_consistency.py` is the maintained Weinreb/scNT
communication-consistency producer and A4 renderer. It runs CellAgentChat in
its official non-spatial mode, prepares the fixed response/candidate tables for
`run_nonspatial_nichenet.R`, and compares CytoBridge, CellChat, CellAgentChat,
and NicheNet on complete directed cell-type-pair grids. The shared-database
mode uses the wheel-bundled mouse CellChatDB for all four methods. Native
CellAgentChat CTPS is primary; threshold-free continuous score is emitted only
as sensitivity. Raw method scores are never pooled across methods.

`run_spatial_communication_consistency.py` is the five-dataset spatial
counterpart for Zebrafish, MOSTA, ARISTA, AdMouse, and Chicken heart. It
freezes one deterministic terminal-stage cell-type-stratified sample per
dataset, preserves the same terminal cells for every external method, and
compares complete directed cell-type-pair ranks. The primary CytoBridge view
is the exact one-layer interaction message contribution; attention-gate
magnitude is retained as a secondary audit. A method enters the main figure
only if it passes the gate recorded in
`configs/spatial_communication_consistency/five_datasets.json`, which was
frozen before the five-dataset outputs were generated. All attempted methods
remain in the complete status and metric tables. Chicken heart is explicitly
labelled as a conserved-symbol human-CellChatDB proxy rather than a native
Gallus gallus screen.

The `prepare-shared-database-proxy` subcommand adapts each accepted
`filtered_lr_database.csv` for spatial CellAgentChat and the NicheNet
candidate gate. Both methods receive the same expression-representable
single-gene LR universe. Multi-subunit ligand or receptor rows are excluded,
not decomposed into artificial singleton pairs. NicheNet retains its official
mouse or human ligand-target matrix because that regulatory prior is part of
the NicheNet method rather than its LR candidate gate. Zebrafish defaults to
the strict Ensembl-116 confidence-1 one-to-one mapping; the broader
symbol-bijective one-to-one mapping is sensitivity-only. ARISTA and chicken
conserved-human-symbol runs are likewise labelled as proxy sensitivity
analyses.

The same orchestrator owns the reviewer-facing model-linked biology addendum.
`select-model-biology` crosses the frozen LR database with every terminal
off-diagonal model cell-type pair and selects by CytoBridge exact-message ×
sender-ligand × receiver-receptor activity only. `score-model-biology` retains
the cell-level audit, while `plot-model-biology` evaluates COMMOT and the frozen
current-database CellAgentChat pair-level CTPS only after selection and renders
the A4 figure. Its compact `panel_data`
directory can reproduce the visual locally without the original large H5ADs.
An unavailable dataset remains explicit rather than being replaced by a
different rule. LR labels are post-hoc molecular compatibility annotations of
learned messages, not native biochemical identities or causal claims.

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

`run_lr_complex_aggregation_sensitivity.py` answers the multi-subunit
aggregation sensitivity without rerunning training or trajectory simulation.
It consumes one completed package-workflow `summary.json`, reconstructs the
saved communication matrices and expression-state snapshots, and requires its
recomputed minimum-gate pair table to match the formal primary output before
writing a zero-preserving geometric-mean result. It then invokes the maintained
comparison renderer to produce paired scores, per-time and per-pair stability,
PDF/PNG, and signed manifests in a new output directory:

```text
python scripts/run_lr_complex_aggregation_sensitivity.py \
  --workflow-summary <formal-downstream>/summary.json \
  --output-dir <new-sensitivity-root>/<dataset>
```

Every subunit remains required in both branches; this is an aggregation-rule
sensitivity, not partial-complex imputation.

`plot_lr_complex_aggregation_reviewer_response.py` combines the four formal
multi-subunit comparison bundles into one concise reviewer-response figure. It
reports evidence coverage, overall rank agreement, and top-ten overlap through
time. AD mouse and pair-specific rank lists are intentionally omitted from the
artwork. The renderer writes vector PDF, 320-dpi PNG, caption, plotting-script
snapshot, provenance note, and signed figure manifest.

`run_zebrafish_classic_s24.py` restores the original unequal-population
zebrafish virtual-removal estimand as a separate, auditable analysis. For each
of simulation seeds 42–46 it starts from the complete 563-cell observed t=0
cohort, deletes the exact YSL (29 cells) or EVL (272 cells) subset without
replacement, and propagates all three branches continuously with learned
growth-driven split/extinction enabled. Five `run-seed` jobs can run in
parallel. `report` requires all five jobs plus a byte-identical seed-42 replay,
selects the latest observed endpoint that passes the predeclared latent-support
gate at every preceding frame for all 15 trajectories, and creates the
manuscript-style morphology, spatial-W1, population-count, and centroid panels.
Spatial W1 uses uniform empirical OT on deterministic supports capped at 1,024
points per cloud; raw trajectories and plotted states retain every particle.
This one-checkpoint virtual removal is not a causal knockout or a biological-
replicate experiment. The
equal-N fixed-population S24 produced by `run_zebrafish_paper_downstream.py`
remains a distinct shape-control analysis.

`plot_zebrafish_s22_article_style.py` is a read-only publication renderer for
the accepted formal S22 stage. It requires the exact SHA-256 values of both the
S22 `stage_manifest.json` and its sibling formal `run_manifest.json`, verifies
every stage-recorded artifact, and renders the original article-style 3 x 3
sequence `Observed t=0`, `Generated t=0.5`, ..., `Observed t=4`. Generated
half-time panels are selected only from the signed continuous global-t0 path;
integer-time panels are independent observed references. The renderer performs
no simulation, adjacent-slice re-anchoring, display warp, or clipping, and does
not replace or modify the retained all-generated fixed-N S22 support audit. It
writes an Arial/Type-42 vector PDF, a 320-dpi PNG, panel-source table, caption,
provenance, and a hashed figure manifest into a new or empty directory:

```text
python scripts/plot_zebrafish_s22_article_style.py \
  --stage-root <formal-run>/s22 \
  --expected-stage-manifest-sha256 <s22-stage-manifest-sha256> \
  --run-manifest <formal-run>/run_manifest.json \
  --expected-run-manifest-sha256 <run-manifest-sha256> \
  --output-dir <new-article-style-s22-bundle>
```

`run_zebrafish_attention_validation.py` is the reviewer-facing, non-circular
validation of the learned zebrafish interaction field. `analyze` consumes
SHA-bound outputs from the accepted model, COMMOT, CellAgentChat, NicheNet,
the frozen 21-axis reference from the 2022 zebrafish atlas, and the
same-checkpoint interaction-on/off sensitivity. It compares every directed
cell-type pair, freezes important pairs using CytoBridge alone before reading
external ranks, retains an expression-only LR baseline, and tests the
interaction modifier against abundance/distance/self-pair-stratified
permutations. Attention remains a signed model gate rather than an LR identity
or communication probability; the exact interaction message is the primary
model contribution view. `report` renders the frozen numerical bundle into an
Arial A4 vector PDF, 320-dpi PNG, caption, reviewer response, panel data, and a
signed manifest. NicheNet is labelled as receiver-transition regulatory
evidence, and interaction-off is an inference sensitivity rather than an LR
knockout or causal perturbation.

```text
python scripts/run_zebrafish_attention_validation.py analyze \
  --spec <sha-bound-zebrafish-spec.json> \
  --output-dir <new-analysis-root>
python scripts/run_zebrafish_attention_validation.py report \
  --analysis-dir <completed-analysis-root> \
  --expected-analysis-manifest-sha256 <analysis-manifest-sha256> \
  --output-dir <new-report-root>
```

Other top-level files under `scripts/` in the Git repository are retained as
historical research records. Some contain workstation-specific paths or calls
from earlier package versions. They are not installed, are not included in the
source distribution, and should not be used as a starting point for new work.
