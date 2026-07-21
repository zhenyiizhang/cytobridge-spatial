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

The formal primary uses `strict_one_to_one + strict_log1p_rename`. Reciprocal
one-to-one genes are selected from the existing single-log `X` and renamed to
mouse symbols; the selected expression values are unchanged. The counts layer
is subset in the same order.

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
  --orthology-map /path/to/frozen_zebrafish_mouse_orthology.tsv \
  --custom-lr-database /path/to/CellChatDB.ligrec.zebrafish.csv \
  --cellagentchat-source /path/to/CellAgentChat-v0.2.0 \
  --output-dir /path/to/cellagentchat/inputs
```

The default orthology columns are `zebrafish_gene`, `mouse_gene`,
`orthology_type`, and `orthology_confidence`. Override the column flags when a
frozen Ensembl or Alliance export uses different names.

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
