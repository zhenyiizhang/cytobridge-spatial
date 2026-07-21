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
  issues;
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

## Output contract

The bundle contains:

- `README.md`: reviewer-facing design, exact scores, coverage, rank results,
  controls, validation axes, and interpretation guardrails;
- `README_CN.md`: concise CSV-backed Chinese handoff for lab presentation;
- `figures/`: verified PNG/PDF comparison and validation panels;
- `tables/`: verified concordance, coverage, control, LR-axis, and optional
  virtual-removal audit tables;
- `manifests/`: renamed copies of all nine source manifests;
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
