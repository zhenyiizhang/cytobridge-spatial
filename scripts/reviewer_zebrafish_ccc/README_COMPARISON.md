# Zebrafish reviewer CCC comparison

`compare_multimethod_ccc.py` combines the formal directed cell-type-pair
outputs without pretending that their native scores share a unit. It requires
these eight separately labelled score views:

- CytoBridge trained attention-gate magnitude;
- CytoBridge trained exact one-layer message contribution;
- COMMOT with the current project zebrafish LR database;
- CellChat with the current project zebrafish LR database;
- NicheNet-v2 with its official mouse LR prior;
- NicheNet-v2 with the project LR database used as a strict one-to-one
  orthology candidate gate or an explicitly labelled all-confidence
  one-to-one sensitivity gate;
- CellAgentChat with its official mouse CellTalkDB default;
- CellAgentChat with the project LR database projected to supported mouse
  singleton pairs.

The two NicheNet conditions and two CellAgentChat conditions must remain
separate. They are not pooled under a generic method label. NicheNet's
`orthology_policy`, `analysis_tier`, and `primary_claim_allowed` fields are read
from each completed run manifest. The two NicheNet conditions must carry the
same policy/tier. `one2one_bijective_all_confidence` is always displayed as an
orthology sensitivity and can never acquire a primary label in this report.
The same rule is enforced for CellAgentChat through each condition manifest's
`shared_input.preparation_claims`: the official/default and project-LR runs
must share identical orthology policy/tier claims, and an all-confidence
mapping is explicitly labelled `all-confidence orthology sensitivity`.

## Formal command

```bash
python scripts/reviewer_zebrafish_ccc/compare_multimethod_ccc.py \
  --run-root "$RUN" \
  --commot-dir "$RUN/03_external_ccc/commot_current_lr" \
  --cellchat-dir "$RUN/03_external_ccc/cellchat_current_lr" \
  --nichenet-default-dir "$RUN/04_nichenet/02_default_mouse_v2" \
  --nichenet-custom-dir "$RUN/04_nichenet/03_custom_zebrafish_lr" \
  --cellagentchat-dir "$RUN/05_cellagentchat" \
  --output-dir "$RUN/06_multimethod_comparison"
```

The CytoBridge paths default to:

- `$RUN/01_cytobridge`
- `$RUN/02_attention_controls`
- `$RUN/02_attention_controls_init_interaction`
- `$RUN/02_attention_controls_random_seed17`

Every path has an explicit override flag. Formal mode fails immediately if a
required manifest, table, schema field, unique directed key, or finite score is
missing. `--allow-partial` exists only to inspect current coverage while jobs
are still running; its manifest is marked `partial_diagnostic` and
`formal_reviewer_ready=false`.

## Statistical contract

All joins use the exact key
`stage, sender_type, receiver_type`. Native values are preserved for audit but
only within-stage ranks are compared across methods.

- Rank concordance is a stage-wise Spearman correlation on the exact shared
  directed-key universe, summarized across stages.
- Top-edge overlap uses only positive scores. Effective k is
  `min(requested, n_shared, n_positive_left, n_positive_right)`, and all edges
  tied at the kth score boundary are retained. Thus row order or alphabetical
  labels cannot select arbitrary edges from a zero tail. All-zero support is
  NA, and any tie-expanded selection covering the whole shared universe is
  audit-only (`top_k_informative=false`).
- Reciprocal asymmetry is
  `rank_percentile(A→B) - rank_percentile(B→A)`, followed by cross-method
  stage-wise Spearman comparison.
- Stage stability compares the same directed cell-type keys only between
  adjacent global observed stages. Its displayed top-k panel applies the same
  informative-only rule.
- Coverage is reported before pairwise joins on canonical evaluated keys.
  COMMOT and CellChat positive-only aggregation gaps are zero-completed only
  against the hash-verified stage-specific `cell_type_counts` square. NicheNet
  is completed only across verified source-stage sender types inside
  `unit_status==complete` receiver units. Skipped/ineligible units, absent
  transitions, and method-unavailable LR rows remain unavailable, not zero.

The CellChat manifest's executable-database audit is mandatory. Requested LR
rows that the pinned CellChat database cannot represent are copied to
`method_unavailable_lr_rows.csv` and `input_diagnostics.csv`, recorded in the
comparison manifest, and excluded from the CellChat method universe. They are
method-unavailable rows, not biological zeroes, and are never zero-filled.

CytoBridge attention must be described as an internal gate magnitude, not a
CCC probability. Its exact message contribution is a separate model-internal
view. The NicheNet type-pair view is a derived sum of positive
sender-associated `aupr_corrected` activities. NicheNet activity is native to a
transition/receiver/ligand, so this derived view is not sender-specific,
receptor-specific, spatial, or a biochemical communication strength.

## Outputs

The output directory contains audit CSVs, a manifest with SHA256 records, a
standalone interpretation README, and PNG/PDF panels:

- `rank_concordance.*`
- `top_edge_overlap.*`
- `condition_coverage.*`
- `directionality_concordance.*`
- `stage_stability.*`
- `cytobridge_control_panel.*`
- `structural_zero_audit.csv`

The audit tables also include `method_unavailable_lr_rows.csv`; the current
formal zebrafish run is expected to preserve the two CellChat-unexecutable
complex-token rows without treating them as zero communication.

The manifest exposes `score_view_zero_completion` for all eight score views,
`six_condition_execution_complete`, `formal_readiness_checks`, and
`reviewer_reporting_ready`. Reporting readiness is an execution/provenance
contract; it does not override condition-level `primary_claim_allowed=false`
for all-confidence orthology sensitivity analyses.

The control panel reads the trained, `Init_interaction`, and randomized
interaction directories. It shows the strict conditional-permutation residual
association, incremental grouped-CV R², and all-stage reciprocal-direction
association for attention and exact messages. The strict conditional test is
selected by the full
`stage+sender_type+receiver_type+distance_bin+state_bin` stratum label; the
script does not choose rows by position.

## CellAgentChat-Fig.-2-style positive consistency

`paper_style_positive_consistency.py` adds a non-circular external consensus,
the self-included ensemble used for comparison with CellAgentChat Fig. 2A,
paper-style inverse-distance localization, top-20% overlap enrichment,
LR-background pathway enrichment, and NicheNet downstream ligand consistency.

The primary consensus excludes CytoBridge and combines within-stage percentile
ranks from COMMOT, CellAgentChat CTPS, and CellChat triMean. The supporting
all-method ensemble includes CytoBridge and is explicitly labelled as
self-included. A separate threshold-sensitivity consensus uses unthresholded
CellAgentChat scores and CellChat truncatedMean.

CellAgentChat CTPS follows Methods Eq. 8: the sum of significant interaction
scores. Historical result tables that preserved this score-sum but mislabeled
significant-pair count as native primary are consumed without mutation; the
correction is recorded in the new manifest.
