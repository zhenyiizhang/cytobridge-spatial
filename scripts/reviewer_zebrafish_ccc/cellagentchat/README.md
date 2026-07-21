# Zebrafish CellAgentChat dual-database workflow

This directory contains the reviewer-specific adapter for CellAgentChat v0.2.0
at commit `310cfc03df91c5ec917f110801e0c2ae4ab57800`. CellAgentChat does not ship a
native zebrafish TF/pathway prior, so the analysis is explicitly a
**mouse-ortholog sensitivity analysis**, not a native zebrafish inference.

## Conditions

Both conditions use the same mapped expression H5AD, physical coordinates,
cell annotations, stages, and frozen sampled-cell IDs.

1. `official_mouse_default_celltalkdb`: the bundled official mouse CellTalkDB.
2. `cytobridge_zebrafish_lr_projected_singletons`: the representable singleton
   subset of the current zebrafish LR database, projected through the same
   zebrafish-to-mouse mapping.

Multi-subunit complexes are exported to an exclusion table. They are not split
into artificial singleton pairs because CellAgentChat v0.2.0 parses LR and
cluster identifiers with underscores.

## Expression contract

The formal primary uses
`strict_one_to_one + strict_log1p_rename + minimum_confidence >= 1`.
Reciprocal one-to-one genes are selected from the existing single-log `X` and
renamed to mouse symbols; the selected expression values are unchanged. The
counts layer is subset in the same order. A strict projection with
`--minimum-confidence 0` is never primary: it is labelled an orthology
coverage sensitivity even though source and target symbols remain bijective.

The optional `many_to_one_sum + counts_sum_then_log1p` adapter sums raw counts
and normalizes to the frozen preprocessing target sum `1105` before `log1p`.
It is a separately labelled sensitivity and never recomputes the target from
the post-filter library median. Running that adapter also requires the explicit
`--allow-nonprimary-preparation` flag; it cannot be silently reported as the
formal primary.

## Prepare inputs

```bash
python scripts/reviewer_zebrafish_ccc/cellagentchat/prepare_inputs.py \
  --expression-h5ad /path/to/zebrafish_aligned.h5ad \
  --orthology-map /path/to/ensembl_compara_drerio_to_mouse_strict_one2one.csv \
  --orthology-manifest /path/to/orthology_manifest.json \
  --orthology-analysis-tier primary \
  --source-gene-column zebrafish_symbol \
  --target-gene-column mouse_symbol \
  --minimum-confidence 1 \
  --custom-lr-database /path/to/CellChatDB.ligrec.zebrafish.csv \
  --cellagentchat-source /path/to/CellAgentChat-v0.2.0 \
  --expected-expression-sha256 433b344b32300c9f58c7de4ac6b8f4ce808934be93b05c939ef24b9ea80fe1cd \
  --expected-custom-lr-sha256 27fd0eb35da035a371ef68783d3e2dcf0729668fd58c2bb59f203173ea1b3f37 \
  --output-dir /path/to/cellagentchat/inputs
```

The default orthology columns are `zebrafish_gene`, `mouse_gene`,
`orthology_type`, and `orthology_confidence`. Override the column flags when a
frozen Ensembl or Alliance export uses different names. When
`--orthology-manifest` is supplied, the adapter verifies the selected mapping
filename and MD5, Ensembl release, exporter policy/tier, filter contract,
selected row count, and (for the all-confidence sensitivity) frozen-raw replay
provenance. The preparation manifest records `orthology_policy`,
`orthology_analysis_tier`, `primary_claim_allowed`, and the hashed mapping
source manifest.

## Ensembl-116 all-confidence sensitivity

The `one2one_bijective_all_confidence` mapping expands coverage while retaining
only symbol-bijective `ortholog_one2one` pairs, but it does not require
Ensembl confidence `1`. Prepare it as a sensitivity with the exact exporter
CSV and manifest:

```bash
python scripts/reviewer_zebrafish_ccc/cellagentchat/prepare_inputs.py \
  --expression-h5ad /path/to/zebrafish_aligned.h5ad \
  --orthology-map /path/to/ensembl_compara_drerio_to_mouse_one2one_bijective_all_confidence.csv \
  --orthology-manifest /path/to/orthology_manifest.json \
  --orthology-analysis-tier sensitivity \
  --source-gene-column zebrafish_symbol \
  --target-gene-column mouse_symbol \
  --orthology-type-column orthology_type \
  --allowed-orthology-types ortholog_one2one \
  --confidence-column orthology_confidence \
  --minimum-confidence 0 \
  --mapping-policy strict_one_to_one \
  --expression-projection-mode strict_log1p_rename \
  --custom-lr-database /path/to/CellChatDB.ligrec.zebrafish.csv \
  --cellagentchat-source /path/to/CellAgentChat-v0.2.0 \
  --expected-expression-sha256 433b344b32300c9f58c7de4ac6b8f4ce808934be93b05c939ef24b9ea80fe1cd \
  --expected-custom-lr-sha256 27fd0eb35da035a371ef68783d3e2dcf0729668fd58c2bb59f203173ea1b3f37 \
  --counts-layer counts \
  --cell-type-key Annotation \
  --time-key time_point_processed \
  --time-label-key time_label \
  --spatial-key spatial_aligned \
  --sampling-seeds 101,202,303 \
  --max-cells-per-type 500 \
  --output-dir /path/to/cellagentchat/all_confidence_inputs
```

Both LR-database conditions must then run through the same preparation. The
explicit allowance is mandatory and is recorded in each condition manifest:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/reviewer_zebrafish_ccc/cellagentchat/run_dual.py \
  --preparation-dir /path/to/cellagentchat/all_confidence_inputs \
  --cellagentchat-source /path/to/CellAgentChat-v0.2.0 \
  --sampling-seeds 101,202,303 \
  --stages 0,1,2,3,4 \
  --epochs 50 \
  --learning-rate 0.1 \
  --batch-size 256 \
  --feature-shuffles 1 \
  --permutation-score-target 10000 \
  --bonferroni-threshold 0.05 \
  --tau 2 \
  --delta 1 \
  --device cuda:0 \
  --allow-nonprimary-preparation \
  --output-dir /path/to/cellagentchat/all_confidence_formal_dual
```

## Smoke test one condition

Use a low permutation target only for smoke testing:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/reviewer_zebrafish_ccc/cellagentchat/run_spatial.py \
  --preparation-dir /path/to/cellagentchat/inputs \
  --cellagentchat-source /path/to/CellAgentChat-v0.2.0 \
  --database-label official_mouse_default_celltalkdb \
  --stages 1 --sampling-seeds 101 \
  --permutation-score-target 100 \
  --output-dir /path/to/cellagentchat/smoke/official
```

Smoke outputs must not be used in the manuscript.

## Formal dual run

The formal default `permutation-score-target` is the official tutorial value
`10000`; spatial distance, distance-scaled permutation background, `tau=2`,
`delta=1`, and Bonferroni `0.05` are enabled.

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/reviewer_zebrafish_ccc/cellagentchat/run_dual.py \
  --preparation-dir /path/to/cellagentchat/inputs \
  --cellagentchat-source /path/to/CellAgentChat-v0.2.0 \
  --output-dir /path/to/cellagentchat/formal
```

The native primary output is the number of Bonferroni-significant LR pairs per
directed sender/receiver cell-type pair. Raw score sums are exported only as
secondary sensitivity views. Absolute significant-pair counts across the two
different LR universes are not directly comparable; use ranks, significant
fractions, or their common mapped-LR universe.

### Assemble two parallel formal runs

When the two `run_spatial.py` conditions are run independently on separate
GPUs, assemble them into the same directory layout as `run_dual.py` with:

```bash
python scripts/reviewer_zebrafish_ccc/cellagentchat/assemble_dual.py \
  --official-run-dir /path/to/parallel/official_mouse_default_celltalkdb \
  --custom-run-dir /path/to/parallel/cytobridge_zebrafish_lr_projected_singletons \
  --output-dir /path/to/cellagentchat/formal_dual
```

The assembler fails closed unless both manifests and all declared artifacts
are intact, use the pinned source, contain exactly stages `0,1,2,3,4` crossed
with seeds `101,202,303`, and use 50 epochs plus 10,000 permutation scores. It
also verifies identical mapped expression, sample plan, preparation claims,
formal design, stage labels, and per-run dimensions, while requiring distinct
LR-database hashes. The fresh output contains condition-directory symlinks,
`dual_condition_run_summary.csv`, and a parent `manifest.json`; an existing
nonempty output is rejected unless `--overwrite` is explicitly supplied.
