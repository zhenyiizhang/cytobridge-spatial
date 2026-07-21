# Corrected zebrafish: COMMOT and CellChat runners

These runners give COMMOT and CellChat the same cells, labels, expression, LR
universe, and observed stages.  The shared expression is reconstructed from
`layers['counts']` using the target frozen by the formal preprocessing audit
(`1105` for the manuscript artifact), followed by exactly one `log1p`.  Every
selected LR-gene value is compared with the existing H5AD `X`; preparation
fails if the matrices differ.

The primary run has three commands:

```bash
python scripts/reviewer_zebrafish_ccc/prepare_inputs.py \
  --h5ad /data/cytobridge/projects/CytoBridge-ST-1104/runs/zebrafish-api/clean-counts-alpha-comparison-20260718/preprocess/zebrafish_aligned.h5ad \
  --lr-database /data/cytobridge/projects/CytoBridge-ST-1104/runs/zebrafish-api/clean-counts-alpha-comparison-20260718/assets/CellChatDB.ligrec.zebrafish.csv \
  --preprocess-audit /path/to/preprocess_audit.json \
  --out-dir /path/to/shared_inputs

python scripts/reviewer_zebrafish_ccc/run_commot.py \
  --input-dir /path/to/shared_inputs \
  --out-dir /path/to/commot_current_db

Rscript scripts/reviewer_zebrafish_ccc/run_cellchat.R \
  --input-dir /path/to/shared_inputs \
  --out-dir /path/to/cellchat_current_db \
  --nboot 100 \
  --seed 20260722 \
  --population-size false \
  --positive-only true \
  --save-rds true
```

Do not add `--max-cells-per-stage` to the formal shared-input command unless a
subsampled sensitivity is explicitly intended.  If it is used, selection is
deterministic and label-stratified, and both methods receive the same cells.

## Method-specific decisions

- COMMOT uses `spatial_aligned`.  By default it reads the exact CytoBridge
  interaction cutoff (`0.09606367405591873`) from the shared manifest, which in
  turn comes from H5AD `uns['interaction_graph']['neighborhood_threshold']`.
  The corresponding recommended spot diameter (`0.024015918513979682`) is
  recorded.  A local k-nearest-neighbor cutoff is available only through the
  explicit `--distance-mode knn` sensitivity; a missing frozen cutoff makes
  the primary mode stop.
- COMMOT receives one `dis_thr` for all LR rows.  Database category is retained
  in output, but Secreted Signaling, ECM-Receptor, and Cell-Cell Contact do not
  silently receive different category-specific distance cutoffs.
- CellChat is the official non-spatial reference.  It reads and order-checks
  `spatial_aligned` but does not use coordinates in its score.  This is stated
  in every manifest rather than presenting CellChat as spatial evidence.
- CellChat resolves each CSV `database_row` against the installed official
  `CellChatDB.zebrafish`.  Expanded ligand, expanded receptor, pathway, and
  annotation are audited row by row.  Ligand, receptor, and annotation must
  match or the run stops; the requested CSV remains authoritative for pathway
  labels and any difference from the pinned CellChat label is recorded.  Thus
  the custom CSV is the executed universe, not just a label attached to a
  default-database run.
- Before inference, CellChat applies a fail-closed executability check matching
  its own `subsetData` rules: a simple token must be an exact `geneInfo` symbol
  present in the shared matrix, while a complex must be declared in the pinned
  complex table and every exact subunit must be present.  The runner never
  invents an undeclared complex by splitting an underscore-delimited token.
  `database_eligibility_audit.csv` records every requested row and
  `excluded_lr_rows.csv` records method-unavailable rows.  Those exclusions
  must be removed from CellChat cross-method comparison universes rather than
  filled as biological zeros.
- COMMOT receives the same structurally available CSV rows.  Exact identical
  flat duplicates are computed once and their source row IDs are retained in
  `database_rows`.
- No stage-specific expression-prevalence filter changes the LR universe.
  Stage-inactive pairs are legitimate zeros.

## Output contract

Each method writes three gzip-compressed long tables:

- `*_lr_scores.csv.gz`: stage × sender type × receiver type × LR interaction;
- `*_pathway_scores.csv.gz`: stage × sender type × receiver type × pathway;
- `*_type_pair_scores.csv.gz`: stage × sender type × receiver type totals.

The common prefix includes `method`, `database_variant`, `stage`,
`stage_time`, `sender_type`, `receiver_type`, `ligand`, `receptor`, `pathway`,
`category`, `interaction_id`, `score`, `p_value`, `significant`, group cell
counts, and a precise score-semantics string.  Native units are deliberately
not treated as numerically interchangeable; comparisons should use within-
stage ranks, top-k overlap, or calibrated/normalized values.

For the abundance-controlled primary comparison, use
`abundance_controlled_score`: COMMOT exports its block mass divided by the
number of possible sender/receiver cell pairs, while CellChat uses its native
score from the `population.size=false` run.  COMMOT's native block-mass `score`
is retained as an abundance-sensitive analysis rather than discarded.

Detailed LR and pathway long files may omit score-zero rows to control file
size. The primary `*_type_pair_scores.csv.gz` files now export every evaluated
stage-specific directed type pair, including native zeros. The comparison
loader remains backward-compatible with historical positive-only type-pair
outputs: it may add zeros only after verifying the input manifest hash, every
stage's `cell_type_counts`, and the runner's explicit positive-only policy.
CellChat-incompatible LR rows are method-unavailable and are never zero-filled.

Every input and result directory has a JSON manifest containing SHA-256 file
records, preprocessing choices, software versions, stage cell counts, spatial
cutoffs, and diagnostics.  Output directories must be new and empty, which
prevents accidental mixing with an earlier benchmark.
