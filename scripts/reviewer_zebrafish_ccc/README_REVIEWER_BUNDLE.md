# Zebrafish CCC reviewer bundle

`build_reviewer_bundle.py` is a read-only reporting step. It validates and
packages the completed eight-view rank comparison (two CytoBridge readouts and
six external method/database conditions), the reviewer-axis validation, and
all external method manifests.

It does not run, rerun, or alter COMMOT, CellChat, NicheNet, CellAgentChat, or
CytoBridge.

## Required completed inputs

- a formal `compare_multimethod_ccc.py` output with
  `formal_reviewer_ready=true`, all eight score views, and no partial-mode
  issues, including a passing primary-score-artifact hash check;
- a completed `validate_reviewer_axes.py` output;
- the exact COMMOT, CellChat, NicheNet default/custom, and CellAgentChat
  official/custom directories recorded by the comparison manifest.

The method directories are inferred from `comparison/manifest.json`. Explicit
directory arguments are available when the source tree was moved.

## Command

```bash
python scripts/reviewer_zebrafish_ccc/build_reviewer_bundle.py \
  --comparison-dir "$RUN/06_multimethod_comparison_final" \
  --validation-dir "$RUN/04_reviewer_validation_axes" \
  --positive-consistency-dir "$RUN/07_positive_consistency" \
  --biological-consistency-dir "$RUN/10_biological_consistency_visuals" \
  --output-dir "$RUN/reviewer_delivery_20260722"
```

For relocated method results, add:

```bash
  --commot-dir /path/to/commot_current_lr \
  --cellchat-dir /path/to/cellchat_current_lr \
  --nichenet-default-dir /path/to/nichenet/default \
  --nichenet-custom-dir /path/to/nichenet/custom \
  --cellagentchat-dir /path/to/cellagentchat/dual
```

The command refuses a nonempty output directory unless `--overwrite` is
explicitly supplied.

`--positive-consistency-dir` is optional for legacy bundles. When supplied,
the builder verifies its external-only primary design, disclosed self-included
ensemble, CellAgentChat CTPS correction, and every artifact hash before copying
the new figures, tables, Chinese note, and reviewer-response draft.

`--biological-consistency-dir` is also optional. When supplied, the builder
requires a frozen, non-visual ncWNT/CXCL/NOTCH example-selection contract,
rejects raw cross-method score comparisons, verifies every cell-flow/table/
figure hash, and adds the direct spatial LR maps, CCC circles, and temporal LR
bubble plot.

## Direct biological visualization addendum

This addendum is intentionally two-phase so that the final spatial examples
are fixed before the cell-level COMMOT matrices are reconstructed:

```bash
python scripts/reviewer_zebrafish_ccc/biological_consistency_panels.py \
  --validation-dir "$RUN/04_reviewer_validation_axes" \
  --positive-consistency-dir "$RUN/07_positive_consistency" \
  --cytobridge-dir "$RUN/01_cytobridge" \
  --commot-dir "$RUN/03_external_ccc/commot_current_lr" \
  --h5ad /path/to/zebrafish_aligned.h5ad \
  --out-dir "$RUN/08_biological_consistency_preselection" \
  --select-only

/path/to/commot-env/bin/python \
  scripts/reviewer_zebrafish_ccc/run_selected_commot_flows.py \
  --input-dir "$RUN/03_external_ccc/shared_inputs" \
  --examples-csv \
    "$RUN/08_biological_consistency_preselection/biological_example_selection.csv" \
  --out-dir "$RUN/09_selected_commot_cell_flows" \
  --cot-nitermax 2000

python scripts/reviewer_zebrafish_ccc/biological_consistency_panels.py \
  --validation-dir "$RUN/04_reviewer_validation_axes" \
  --positive-consistency-dir "$RUN/07_positive_consistency" \
  --cytobridge-dir "$RUN/01_cytobridge" \
  --commot-dir "$RUN/03_external_ccc/commot_current_lr" \
  --h5ad /path/to/zebrafish_aligned.h5ad \
  --selected-examples-csv \
    "$RUN/08_biological_consistency_preselection/biological_example_selection.csv" \
  --selected-commot-flow-dir "$RUN/09_selected_commot_cell_flows" \
  --out-dir "$RUN/10_biological_consistency_visuals"
```

The selected COMMOT runner uses the same prepared stage matrices, LR database,
spatial cutoff, heteromeric `min` rule, and COT iteration budget as the formal
COMMOT benchmark. It evaluates only the frozen examples and writes every
positive cell-level flow; it never edits the formal benchmark directory.

## Coordinate-level spatial consistency audit

After the reviewer bundle and aligned cell coordinates are available, replace
the density-sensitive arrow/nearest-neighbor view with same-coordinate hotspot
fields, an adaptive fixed-support permutation null, an LR-only component
control, and separate sender/receiver maps:

```bash
python scripts/reviewer_zebrafish_ccc/spatial_coordinate_consistency.py \
  --bundle-dir /path/to/reviewer_delivery \
  --coordinates-csv /path/to/zebrafish_spatial_coordinates.csv.gz \
  --output-dir /path/to/spatial_coordinate_consistency \
  --permutations 1000
```

The default adaptive null uses five distance/LR quantile bins and requires at
least ten edges per realized stratum. It retains exact sender→receiver type
where supported, then explicitly coarsens to covariate-only or distance-only
strata. The run writes `permutation_strata_diagnostics.csv` and fails if more
than 5% of any assignment falls back to one global pool or if fewer than 95%
of edges are movable. For `attention×LR` and `exact-message×LR`, LR activity
and the observed COMMOT field stay fixed while only the CytoBridge modifier is
permuted. The script default of 200 permutations is suitable for development;
use 1,000 for a formal report.

These spatial panels are an audit, not an automatic positive result. Raw
hotspot overlap must exceed the declared null and `attention×LR` must improve
over LR-only before claiming attention-specific spatial consistency.

## Axis-specific gene-input counterfactual

The reusable API is exposed under `CytoBridge.tl`:

- `validate_pca_model_visibility`;
- `apply_projected_gene_knockdowns`;
- `deterministic_fixed_cohort_rollout`;
- `audit_spatial_complete_messages`;
- `compute_fixed_lr_target_message_metrics`;
- `compute_counterfactual_metrics`;
- `compute_interaction_mediation_metrics`;
- `run_gene_counterfactual`.

The zebrafish reviewer runner binds these generic operations to the
`cxcl12a -> cxcr4a` axis:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/path/to/frozen/cytobridge-spatial \
python scripts/reviewer_zebrafish_ccc/lr_gene_counterfactual.py \
  --h5ad /path/to/zebrafish_aligned.h5ad \
  --model-dir /path/to/training \
  --output-dir /path/to/cxcl12a_cxcr4a_gene_counterfactual \
  --anchors 3:4 \
  --screen-anchors 0:1,3:4 \
  --anchor-restriction-post-hoc \
  --technical-smoke-seen-before-formal-run \
  --fractions 0.25,0.5,1 \
  --n-shams 100 \
  --grouping-seeds 101,202,303,404,505 \
  --group-size 1024 \
  --dt 0.05 \
  --device cuda:0
```

For the formal zebrafish analysis, `0:1` remains in the baseline-only
eligibility table but is not encoded as a zero effect: the fixed gated-support
estimand has no eligible edge there. The executed `3:4` analysis is explicitly
recorded as post-hoc exploratory/descriptive because the support audit and a
technical smoke preceded the formal run. It is not a temporal replicate or an
independent confirmation.

The runner fails closed unless the input PCA state reconstructs from the
persisted center and loadings and both axis genes are model-visible. It keeps
cell identities fixed, uses `sigma=0`, disables growth/resampling, and uses
one identity-paired OT support within each anchor across conditions, doses,
and interaction on/off. Matched-HVG shams are selected separately within each
anchor's baseline ligand-positive fixed-sender compartment and are edited only
inside that mask; they provide descriptive reference ranks, not formal
randomization P values.

`D_target` is the generic complete GNN message on a fixed,
LR-expression-conditioned edge support. It is not an LR-specific biochemical
message component. Spatial and state Wasserstein results are primary; joint
Wasserstein mixes coordinate scales and is explicitly descriptive. Only the
first grouping seed drives the trajectory, Wasserstein, and mediation
calculation; all listed seeds repeat the exact-message grouping audit. The
analysis quantifies sensitivity of the trained model and cannot substitute for
an experimental perturbation or establish causal signaling.

## Output contract

The bundle contains:

- `README.md`: reviewer-facing design, exact scores, coverage, rank results,
  controls, validation axes, and interpretation guardrails;
- `README_CN.md`: concise CSV-backed Chinese handoff for lab presentation;
- `figures/`: verified PNG/PDF comparison, validation, and optional direct
  biological/spatial panels;
- `tables/`: verified concordance, coverage, control, LR-axis, and optional
  virtual-removal audit tables;
- `manifests/`: renamed copies of all source manifests, including optional
  biological selection and selected-flow manifests;
- `notes/`: the two upstream interpretation notes;
- `bundle_manifest.json`: source provenance, SHA256 inventory, six-condition
  definitions, and explicit false-claim guardrails.

The report keeps these distinctions explicit:

- CytoBridge attention is not a CCC probability, and exact messages are not
  biochemical flux;
- raw COMMOT, CellChat, NicheNet, CellAgentChat, and CytoBridge values do not
  share units;
- positive-only exports are completed with structural zeros only inside a
  manifest-verified evaluated universe; skipped/ineligible units are not
  filled;
- top-k uses strictly positive support and includes every boundary tie; a
  zero-support comparison is reported as NA;
- NicheNet's type-pair score is derived and non-spatial;
- the project-LR NicheNet condition changes the candidate gate, not the mouse
  ligand-target prior;
- NicheNet and CellAgentChat are cross-species, with strict orthology labelled
  primary and all-confidence orthology labelled sensitivity-only;
- CellChat method-unavailable LR rows are excluded and never interpreted or
  filled as biological zero (this is distinct from evaluated-grid structural
  zero completion);
- virtual removal is a model sensitivity analysis, not a causal perturbation.

## Plain-language reading guide

After building the frozen bundle, generate a separate Chinese reading guide
and direct comparison figures without modifying the bundle:

```bash
python scripts/reviewer_zebrafish_ccc/plain_language_consistency_report.py \
  --bundle-dir /path/to/reviewer_delivery \
  --spatial-consistency-dir /path/to/spatial_coordinate_consistency \
  --output-dir /path/to/zebrafish_ccc_plain_language_guide
```

Start from `START_HERE_CN.md`. The guide first maps every model-native and
post-hoc interaction quantity to the exact analysis that consumes it. It then
adds a 2-by-5 direct CytoBridge-versus-COMMOT scatter, an external-consensus
scatter and a concrete sender-to-receiver checklist. When
`--spatial-consistency-dir` is supplied, it verifies every spatial artifact
hash and incorporates the hotspot, null/sensitivity, LR-only component, and
sender/receiver panels; the old many-to-one location-coverage panel is retained
only as a legacy descriptive audit. It labels every original panel as main
evidence, supporting evidence, an audit, or a limitation. Its top-20%
reconstruction is checked against the formal tie-inclusive table in the source
bundle before any output is written.

Pairwise and positive-consistency top-set selection requires strictly positive
native support before expanding kth-boundary ties. This prevents a method with
an all-zero or sparse zero-tied stage from turning the whole type-pair matrix
into a nominal "top" set.

## Biology-first addendum

The cross-method bundle is a numerical consistency analysis; it does not by
itself establish that attention measures communication. The Jam2a–Jam3b
myocyte-fusion case, trained/initialization/random controls, Delta–Notch
limitation audit, score definitions, and new-dataset adaptation contract are
documented in [README_BIOLOGY_FIRST.md](README_BIOLOGY_FIRST.md).
