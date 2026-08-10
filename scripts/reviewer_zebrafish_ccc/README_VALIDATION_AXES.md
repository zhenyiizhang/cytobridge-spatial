# Zebrafish reviewer validation axes

`validate_reviewer_axes.py` adds reviewer-facing checks around the corrected
zebrafish CytoBridge interaction output. It does **not** reinterpret model
attention as a cell-cell communication probability.

## Required upstream artifacts

The script consumes the primary grouping-seed edge table from
`analyze_attention_confound_controls.py`, its manifest, and the matching exact
attribution manifest from `run_cytobridge_spatial_attribution.py`. All supplied
H5AD, LR database, and edge-table hashes must match those manifests.

For the corrected full-data run used in the manuscript review, the inputs are:

- H5AD SHA256:
  `433b344b32300c9f58c7de4ac6b8f4ce808934be93b05c939ef24b9ea80fe1cd`
- Finetune weight SHA256:
  `ecadf59ed04f200f3da7c13c1124af7992d381ec5fa59a852f92dd0508c1128f`
- LR database SHA256:
  `27fd0eb35da035a371ef68783d3e2dcf0729668fd58c2bb59f203173ea1b3f37`

## Formal command

```bash
PROJ=/data/cytobridge/projects/CytoBridge-ST-1104
RUN=$PROJ/runs/zebrafish-ccc-reviewer/20260722
CODE=$RUN/source_snapshot/cytobridge-spatial
PY=$PROJ/envs/arista-api/bin/python
H5=$PROJ/runs/zebrafish-api/clean-counts-alpha-comparison-20260718/preprocess/zebrafish_aligned.h5ad
LR=$PROJ/runs/zebrafish-api/clean-counts-alpha-comparison-20260718/assets/CellChatDB.ligrec.zebrafish.csv
ABL=$PROJ/runs/zebrafish-api/clean-counts-alpha-comparison-20260718/conditions/alpha_express_0015/paper_downstream_manuscript_final_v3_20260718

cd "$CODE"
"$PY" scripts/reviewer_zebrafish_ccc/validate_reviewer_axes.py \
  --edge-controls "$RUN/02_attention_controls/edge_controls_seed_101.csv.gz" \
  --attention-controls-manifest "$RUN/02_attention_controls/run_manifest.json" \
  --attribution-manifest "$RUN/01_cytobridge/run_manifest.json" \
  --h5ad "$H5" \
  --lr-database "$LR" \
  --known-axis-provenance scripts/reviewer_zebrafish_ccc/known_zebrafish_axis_provenance.csv \
  --ablation-run-dir "$ABL" \
  --output-dir "$RUN/04_reviewer_validation_axes" \
  --permutations 1000 \
  --top-axes-per-stage 20
```

The optional ablation input above is a completed **full-profile** S24 run, not
a smoke run. Reuse fails closed unless its H5AD and Finetune hashes equal the
attribution inputs, every relevant artifact matches the recorded SHA256, and
the simulation declares continuous pre-warp split-SDE with no re-anchoring or
replacement.

## Statistical units and outputs

`sender_receiver_contexts.csv` uses stage × sender cell type × receiver cell
type as the context unit. `context_enrichment_tests.csv` selects high and low
contexts within each stage and permutes the context target within stage. Both
raw model quantities and out-of-fold confounder residuals are reported. The
eight context-test empirical p-values are Benjamini-Hochberg adjusted as one
declared family.

`degree_matched_conditional_tests.csv` uses only the out-of-fold attention or
exact-message residual from the earlier broad confounder model. The new
conditional permutation is restricted within bins of:

- developmental stage;
- sender and receiver cell types;
- spatial distance;
- non-LR transcriptional-state similarity;
- source out-degree; and
- target in-degree.

Forward LR compatibility and a sender/receiver-swapped reverse control are
always reported together. `degree_matching_strata_audit.csv.gz` exposes every
retained matching stratum. The four degree-matched empirical p-values are
Benjamini-Hochberg adjusted as a separate declared family.

`lr_axis_stage_scores.csv.gz` contains every expression-identifiable LR axis
and stage. Complex activity is the minimum across all underscore-delimited
subunits, scaled by its global positive 95th percentile and clipped to
`[0, 1]`. `top_identifiable_lr_axes.csv` is a descriptive ranking by either
attention × LR activity or exact-message × LR activity. It is not a literature
validation table.

`known_axis_database_provenance.csv` is narrower. Every exact pair must occur
in the supplied LR database, and each row includes primary-source identifiers,
URLs, and an explicit claim guardrail. Those papers support developmental
relevance of pathway/gene-family members; they do not validate the inferred
cell-type direction, exact pair in these cells, stage-specific score, or a CCC
probability. `known_axis_stage_scores.csv` joins that provenance to all five
observed stages.

`virtual_ablation_summary.csv` and
`virtual_ablation_observed_stages.csv` summarize the reused formal S24 model
sensitivity experiment. They separate the immediate time-zero shift caused by
removing cells from later trajectory divergence. This is a one-seed in-silico
sensitivity analysis and must not be described as a causal genetic
perturbation. The figure resolves the ablation grid's internal stage indices
against the one-to-one observed H5AD labels and displays `5.25 hpf`, `10 hpf`,
`12 hpf`, `18 hpf`, and `24 hpf`; it never labels the 0–4 indices as continuous
time. If labels do not match the verified hpf pattern, the plotting code falls
back to `observed stage index`.

Finally, `reviewer_validation_axes.{png,pdf}` provides a compact four-panel
view, `reviewer_validation_summary.md` records the numerical interpretation,
and `run_manifest.json` hashes every input/output and encodes the claim
boundaries above.
