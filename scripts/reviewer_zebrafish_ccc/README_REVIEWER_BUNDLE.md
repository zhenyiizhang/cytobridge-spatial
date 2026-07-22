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
  --output-dir /path/to/zebrafish_ccc_plain_language_guide
```

Start from `START_HERE_CN.md`. The guide adds an evidence map, five stage-wise
rank-scatter panels, and a concrete sender-to-receiver arrow checklist. It also
labels every original panel as main evidence, supporting evidence, an audit,
or a limitation. Its top-20% reconstruction is checked against the formal
tie-inclusive table in the source bundle before any output is written.
