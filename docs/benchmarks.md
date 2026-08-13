# Benchmarks and sensitivity analyses

The benchmark and sensitivity tables are evidence, not parameter-tuning
objectives. Unfavourable comparisons and method-native unsupported spaces stay
in the report as measured or `NA` rather than being replaced by surrogate
values.

## Unified sliced-W2 benchmark

The primary four-dataset benchmark is being recomputed against the accepted
matched models. Its result table is pending; no current winner count, rank, or
cross-method superiority conclusion is released yet. The run removes an
internal target from model fitting, fixes the source cohort and output count
before opening target truth, fits transforms on observed training slices, and
evaluates joint/state/spatial sliced W2 with 256 projections for seeds 42–46.
Unsupported method/space combinations remain `NA`, not surrogate values.

The shipped {download}`unified_w2_winners.csv
<data/unified_w2_winners.csv>` is a superseded pre-acceptance snapshot retained
only for provenance. It must not be used as the primary release benchmark or
as evidence for the final matched models. The legacy summary can be inspected
with:

```bash
python scripts/summarize_release_evidence.py
```

`cytobridge workflow --reconstruction-diagnostic` is a fitted-model
reconstruction diagnostic. It is not the cross-method holdout benchmark above.

## Zebrafish formal downstream and daughter-noise sensitivity

The matched Zebrafish paper downstream completed all seven signed stages:
classifier, velocity, observed-anchored S22, growth, S24 sensitivity, S25, and
communication. The canonical reconstruction panels are interval-local and
observed-anchored, not global-t0 rollouts. S24 remains a separately labelled
global-t0 virtual-removal sensitivity and is neither a canonical
reconstruction nor a causal knockout estimate. Older global-t0 reconstruction
and ablation values are not final-model evidence.

The interval-local daughter-noise analysis used four observed intervals,
daughter-noise SD `0`, `0.01`, `0.03`, and `0.06`, and five paired seeds (80
independent interval/noise/seed simulations). Across the 12 nonzero-noise
interval summaries, mean composition TV ranged from 2.47% to 14.18%, relative
particle-count change from -0.25% to 0.33%, joint W2 from 0.596 to 3.499,
spatial W2 from 0.0336 to 0.1564, and mean lineage-fate TV from 0.121 to 0.589.
These are inference-time sensitivities of one frozen learned checkpoint, not a
training-seed hypothesis test or lineage-continuous rollout.

## Classifier spatial smoothing

The common sweep evaluates `k={1,5,10,20,50}` on a fixed observed validation
split. Boundary and rare populations are more sensitive to voting than interior
populations. Formal output uses Z/M/A k10 and AD k1 consistently. The explicit
policy is available as {download}`formal_k_policy.csv
<data/formal_k_policy.csv>`.

## LR complex aggregation

The Zebrafish minimum-versus-geometric-mean comparison has nearly identical
pooled rank correlation but material top-pair changes. ARISTA contains no
strict scored multi-subunit pair and is non-diagnostic for this question.

## Interaction/LR-prior ablations

The package ships matched no-interaction and no-LR-prior training profiles for
all four datasets:

| Dataset | No interaction | No LR prior (`all_spatial`) |
| --- | --- | --- |
| Zebrafish | `zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml` | `zebrafish_spatial_full_alpha_express_0015_no_lr_prior.yaml` |
| MOSTA | `mosta_spatial_full_alpha_express_0015_no_interaction.yaml` | `mosta_spatial_full_alpha_express_0015_no_lr_prior.yaml` |
| ARISTA | `arista_spatial_full_no_interaction.yaml` | `arista_spatial_full_no_lr_prior.yaml` |
| AD mouse | `admouse_spatial_full_alpha_express_0015_no_interaction.yaml` | `admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml` |

Each no-interaction profile preserves the retained velocity, growth, and score
networks plus the full six-stage optimization budget. All three arms explicitly
use the same `velocity_score_cross_term` score-energy objective, so disabling
interaction does not silently substitute a different growth/score regularizer;
the historical interaction-dependent objective remains only as the default for
unrelated legacy configs. Each no-LR-prior profile
preserves the full model and changes only the edge gate from the learned
predictor to `all_spatial`. All 12 profile runs (four datasets × full,
no-LR-prior, and no-interaction) completed training and package downstream.
All 12 profiles and all four matched three-arm families pass the formal
validator under acceptance SHA-256
`c4f8e203e2da73fe78e28525516bbec192d3cbbd35d423dcd64080a0f83a10df`.
No-interaction outputs intentionally omit communication and LR artifacts, so
those analyses are `NA` rather than zero. Comparative matched-ablation metrics
are still pending; old Zebrafish ablation values must not be carried forward as
results for the final model.

## Hyperparameter evidence boundary

The full, no-LR-prior, and no-interaction arms have now been re-fit in one
matched four-dataset grid. This does not make the retained graph-threshold,
score-batch, OT:mass, or scheduler diagnostics exhaustive sweeps around every
formal value, and the primary matched comparison tables remain pending.

The run-resolved values used by the four formal presets are available in
{download}`formal_hyperparameter_settings.csv
<data/formal_hyperparameter_settings.csv>`.

The production selection rules are:

- spatial cutoff: compute the median within-slice nearest-neighbour distance,
  set the recommended spot diameter to `1.2 × median(NN1)`, and use
  `4 × mean(spot diameter)` across observed slices;
- edge threshold: for all four main runs, maximize validation F1 over candidate
  probability thresholds, using accuracy only as a tie-break; corrected AD
  selected `0.9956824779510498` from its strict-seven-pair graph labels;
- score batch: use the dataset recipe and recheck joint, state, and spatial
  metrics if memory constraints require a change;
- scheduler: report the scheduler that actually ran. The historical
  `scheduler_gamma` field was inactive under plateau/cosine schedulers and is
  not treated as sensitivity evidence.

The retained graph and score-batch comparisons are focused legacy diagnostics,
not exhaustive matched sweeps around every current formal value. The current
Zebrafish OT:mass comparison shows an objective-dependent trade-off: the lower
reported ratio improves joint/state fidelity, whereas the formal stagewise
recipe gives the best spatial and transported-mass fidelity. No universal
single-metric optimum is claimed.

## Heart boundary

Heart remains a fifth biological application in the paper but is outside the
four-dataset formal reproducibility chain. The original Figure 3 run has no
recoverable per-epoch or memory provenance. A later Heart-v2 model-fit holdout
is reported separately; it is not substituted for Figure 3. Its mean spatial
W2 across two target slices is 0.0502 (sample SD 0.0072), while state and joint
W2 are not uniformly best. An observed-cell classifier sensitivity check favors
k1 (balanced accuracy 0.7432 versus 0.4385 for k10), but this does not declare a
formal Heart k or recover the historical Figure 3 classifier.
