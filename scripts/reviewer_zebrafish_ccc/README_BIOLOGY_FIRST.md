# Biology-first reviewer validation

These scripts turn a cross-method communication benchmark into an auditable
biological case study. They are intentionally separate from training: every
input is a frozen H5AD, edge-attribution export, external-method result, or
literature-provenance table, and every output manifest records content hashes.

The zebrafish example uses an 18 hpf Somite Jam2a–Jam3b myocyte-fusion case as
the primary result and a 24 hpf neural Delta–Notch case as a limitation audit.
The scripts do not assume that attention is a ligand–receptor-specific output,
a communication probability, biochemical flux, or a causal effect.

## Recommended order

1. Run the generic exact-axis screen with `biology_first_case_studies.py`.
2. Run the trained/initialization/random audit with
   `jam_trained_init_random_control.py`.
3. Build the formal Somite case with `jam_myocyte_case_study.py`.
4. Build the Delta–Notch exact and family audits with
   `delta_notch_case_study.py` and `delta_notch_family_audit.py`.
5. Render the reader-facing figures from the formal tables with
   `make_jam_myocyte_biology_figure.py` and
   `make_delta_notch_biology_figure.py`.

Write every run to a new versioned output directory. `--overwrite` is required
to replace a nonempty directory.

## Primary Somite case

```bash
python scripts/reviewer_zebrafish_ccc/jam_myocyte_case_study.py \
  --h5ad /path/to/aligned_counts.h5ad \
  --edge-dir /path/to/cytobridge/stage_3_18hpf \
  --observed-cells /path/to/observed_cells.csv.gz \
  --provenance scripts/reviewer_zebrafish_ccc/jam_myocyte_axis_provenance.csv \
  --external-spec COMMOT \
    /path/to/commot_lr_scores.csv.gz \
    /path/to/commot_lr_axis_stage_availability.csv.gz \
    abundance_controlled_distinct_cell_score distinct_density \
  --trained-init-random-control \
    /path/to/jam_compatibility_percentile_summary.csv \
  --spatial-key spatial_aligned \
  --spatial-cutoff 0.09606367405591873 \
  --spatial-cutoff-source frozen_preprocessing_interaction_threshold \
  --n-permutations 10000 \
  --permutation-seed 20260722 \
  --output-dir /path/to/biology_first/jam_case
```

The spatial cutoff shown above is dataset-specific. For a new dataset, it must
come from the frozen preprocessing/interaction-graph contract; it must not be
silently copied from the zebrafish analysis.

The external specification is repeatable. Each occurrence supplies a method
name, long score table, explicit stage-by-axis availability table, score
column, and declared score semantics. Sparse missing contexts are completed as
zero only when the corresponding stage-by-axis matrix is explicitly available;
unavailable axes remain missing.

## Score definitions

- `raw attention` is a generic edge weight and has no LR identity.
- `LR-only` is q95-scaled ligand activity in the source multiplied by receptor
  activity in the target on the fixed edge scaffold.
- `attention x LR` is a post-hoc compatibility score, not a native LR-specific
  model output.
- `exact message norm x LR` is also post hoc and is not biochemical flux.
- Native CytoBridge values and external-method values do not share units.
  Compare within-method ranks, not raw magnitudes.

The primary native tissue-pair rank uses the edge-mean
`G_AB_attention_mean_mean`. A cell-pair-density score divides edge mass by all
possible distinct sender–receiver cell pairs. These are different estimands;
their ranks must be labelled and must not be interchanged.

## Support, ranks, and denominators

Exact-axis support filtering occurs before ranking. Competition/min rank,
tie count, rank denominator, rank/N, and top-tail fraction are all exported.
Same-cell diagonal terms are excluded from distinct-cell analyses, so the
denominator is

```text
n_sender * n_receiver - n_cells_in_sender_receiver_intersection
```

For same-type contexts this becomes `n * (n - 1)`.

## Spatial null

The Jam-compatible spatial statistic counts each distinct undirected neighbor
pair once when Jam2a is detected at one endpoint and Jam3b at the other, in
either orientation. The null independently permutes the two detection labels
within the selected compartment while preserving their marginal prevalence and
the fixed spatial graph. A plus-one Monte Carlo upper-tail P value is reported.

This tests spatial localization conditional on prevalence and graph topology.
It does not prove direct membrane contact and can still reflect a shared
regional state.

## Controls and interpretation

The trained/initialization/random audit requires identical selected-stage edge
keys. It reports complete directed type-pair ranks, Jam-compatible versus other
edge percentiles, top-versus-bottom-quartile enrichment, and the trained-minus-
initialization edge-percentile delta.

If initialization retains the biological pattern, describe the contribution
of architecture, spatial scaffold, and input state. A random control alone is
not evidence that training learned the pattern.

Published perturbations can validate the selected biological axis and tissue
context. They do not validate a model score unless the perturbed data are
actually passed through the frozen analysis and the prespecified score changes
as predicted.

## New datasets

The computational machinery is reusable, but the biology is not automatic.
For a new dataset, define in data-specific provenance files:

- developmental stage and cell-type labels;
- spatial key and frozen graph cutoff;
- candidate ligand, receptor, downstream markers, and directional guardrails;
- minimum cell and active-edge support;
- external database and explicit matrix-availability table;
- primary literature supporting the axis, context, and any perturbation claim.

Do not select a case solely because it gives the largest attention correlation.
Use a prespecified biological question, report LR-only and initialization
controls, and retain a limitation or negative case when it changes the scope of
the claim.
