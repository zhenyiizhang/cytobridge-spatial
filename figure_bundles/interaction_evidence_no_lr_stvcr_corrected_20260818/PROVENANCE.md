# Figure provenance

Archived on: `2026-08-18`

Scientific claim: LR-informed graph construction contributes to the matched CytoBridge reconstruction, while comparison with stVCR provides complementary evidence for the value of explicit interaction-aware modeling.

## Files

- Vector figure: `figure/interaction_evidence_no_lr_stvcr_corrected_final.pdf`
- PNG preview: `figure/interaction_evidence_no_lr_stvcr_corrected_final.png`
- Plotting script: `code/build_interaction_evidence_figure.py`
- Caption source: `figure/caption.md`

## Selected experiments

- Matched no-LR evidence: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-matched-ablation-benchmark-20260813-3a380c5-r1/report`
- Chicken-heart matched no-LR evidence: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/chicken-heart-matched-no-lr-20260815-4db603e-r1`
- stVCR evidence: current final unified held-out LOTO target-stage table for Zebrafish, MOSTA, ARISTA, AD mouse, and chicken heart.

## Source paths

- Archived panel inputs: `source_data/`
- Formal no-LR server report: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-matched-ablation-benchmark-20260813-3a380c5-r1/report`
- Local stVCR benchmark table: `source_data/loto_target_stage_means.csv`. This single table replaces the superseded five archived dataset-specific summaries.

## Panel sources

| Panel | Content | Calculation |
|---|---|---|
| a | Full model versus no LR prior | Dataset mean of target- and space-specific relative sliced-W2 differences, displayed with full model = 1. |
| b | LR-prior effect by space | Mean and s.e.m. of `100 * (no-LR - full) / full` across targets. |
| c | CytoBridge versus stVCR | Dataset mean of target- and space-specific relative sliced-W2 differences, displayed with CytoBridge = 1. |
| d | External-baseline effect by space | Mean and s.e.m. of `100 * (stVCR - CytoBridge) / CytoBridge` across targets. |

## Evaluation protocol

- Primary metric: sliced W2. Lower values indicate better reconstruction.
- no-LR scope: full-data in-sample benchmark, 16 target stages across five datasets.
- stVCR scope: held-out LOTO benchmark, 11 target stages across five datasets.
- Final stVCR comparison: CytoBridge lower in 25/33 paired dataset-target-space cells; no ties.
- LOTO projection protocol: 1,024 directions per repeat and five shared repeats.
- Projection uncertainty is not used as biological replication. Error bars summarize variation across target stages.

## Source-file SHA-256

- `source_data/formal_no_lr_paired_target_deltas.csv` — `210d1f5e2fa86630fab665d836c10f96be424e7e6a7041f294c8a720566b137c`
- `source_data/chicken_heart_no_lr/full_data_metrics_long.csv` — `1f11b53c2620bd567d3b6aed149ca66f6e85b5ef1a9eab0ad9fed47a52a303b5`
- `source_data/chicken_heart_no_lr/no_lr_prior_metrics_long.csv` — `6003e2d5e2715bccda20d89484742a35de848b2ee710d2172e2145a357ef23df`
- `source_data/loto_target_stage_means.csv` — `43f3acfa5d508b8d72f0cc02a03c121df58cf2fb2137c54e677f217dbdc4c038`

## Rebuild command

```bash
python code/build_interaction_evidence_figure.py
```

## Interpretation

The no-LR comparison is a matched retraining ablation. stVCR is an external baseline without an explicit cell-cell interaction term, not a component-wise CytoBridge ablation. The two evidence classes are therefore displayed separately and are not pooled into a single effect estimate.

## SHA-256

- Figure PDF: `3fa9bb9dca1fd61b7ce77ba3043cff382bd5a1617407ec56e38faa55de59edf2`
- Figure PNG: `73120dd51d877c41c2646299ec00e3cb403bf485b7eb94c09ee3e7e628b53615`
- Plotting script: `708202833331fd659de45e68f7a8f06fe7ea35a6997e6e8658df4276112d7374`
