# Benchmarks and sensitivity analyses

The benchmark and sensitivity tables are evidence, not parameter-tuning
objectives. Unfavourable comparisons and method-native unsupported spaces stay
in the report as measured or `NA` rather than being replaced by surrogate
values.

## Unified sliced-W2 benchmark

The primary four-dataset benchmark is complete against the accepted matched
models. The run removes an internal target from model fitting, fixes the source
cohort and output count before opening target truth, fits transforms on
observed training slices, and evaluates joint/state/spatial sliced W2 with 256
projections for seeds 42–46. Unsupported method/space combinations remain
`NA`, not surrogate values. All 90 LOTO method×target executions completed.
Of the 130 full-data executions, 126 completed; stVCR failed numerically for
the four ARISTA targets because its OT training weights became non-finite, so
those entries remain explicit `NA`.

The authoritative {download}`unified_w2_winners.csv
<data/unified_w2_winners.csv>` records the lowest LOTO sliced-W2 method for
each of the 27 dataset×target×space comparisons. Winner counts are 15 for the
linear-interpolation control, 9 for CytoBridge, 2 for PASTE, and 1 for stVCR.
CytoBridge is lowest in the spatial space for 7 of 9 held-out targets; stVCR
and the linear control each win one. Joint/state results are less favourable:
the linear control wins 15 of 18 such comparisons, CytoBridge wins the two
Zebrafish t1 comparisons, and PASTE wins the two Zebrafish t3 comparisons.
These within-space results do not define a cross-space overall score, and the
five projection repeats measure numerical projection variability rather than
independent biological or training replicates. The compact table can be
inspected with:

```bash
python scripts/summarize_release_evidence.py
```

Full-data results are in-sample reconstruction diagnostics rather than
forecasts. The lowest aggregate sliced-W2 method in each space is:

| Dataset | Joint | Spatial | State |
| --- | --- | --- | --- |
| AD mouse | PASTE | PASTE | PASTE |
| ARISTA | PASTE | CytoBridge | PASTE |
| MOSTA | PASTE | PASTE | PASTE |
| Zebrafish | Random interpolation | CytoBridge | Random interpolation |

The eight signed evaluation/summary manifest hashes are shipped in
{download}`unified_benchmark_manifests.csv
<data/unified_benchmark_manifests.csv>`. Full-data rankings must not be
described as held-out forecasting performance.

`cytobridge workflow --reconstruction-diagnostic` is a fitted-model
reconstruction diagnostic. It is not the cross-method holdout benchmark above.

## Zebrafish formal downstream and daughter-noise sensitivity

The matched Zebrafish paper downstream completed all seven signed stages:
classifier, velocity, global-t0 S22, growth, S24 sensitivity, S25, and
communication. Paper S22 is one continuous generated fixed-population state
transport from `t=0` through `t=4`; learned drift, score, interaction, and
diffusion are retained, while growth-driven birth/extinction is disabled and N
remains fixed. Observed integer slices are separate references and are not
substituted into that path. It is not an abundance forecast or reconstruction
of observed stages. S25 and communication intentionally retain their separate
interval-local, observed-anchored state contract. S22 also exports a full
latent-support audit; because it is an explicit demonstration, rare tail OOD is
reported rather than hidden or replaced with observed slices. S24 uses the
`preterminal_t3_sigma0` protocol for separate YSL- and EVL-exclusion spatial
sensitivities. Each analysis propagates independently sampled, equal-N cohorts
once from `t=0` through observed `t=3`, with `sigma=0`,
`dt=resample_dt=0.005`, and learned growth resampling disabled. Learned velocity
drift, score-gradient correction, and interactions remain active. All four
baseline/exclusion branches must pass the unchanged latent-support gate before
either panel is drawn, and no points are clipped. The preterminal protocol is
defined through observed `t=3`; terminal `t=4` is not evaluated or claimed.
These panels are not terminal or full joint-state terminal evidence, stochastic
forecasts, total-mass deletions, or causal knockout estimates. The older
unequal-N, growth-resampling EVL result is an OOD diagnostic and is not
final-model evidence.

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

The primary LR score is a strict AND gate: all subunits must be present and the
least-expressed subunit determines complex activity. A zero-preserving
geometric mean, still requiring every subunit, was evaluated as a one-factor
sensitivity on the saved formal expression states and communication matrices.
Before comparison, the rerun had to reproduce every primary minimum-gate score;
the maximum absolute reproduction error was at most `4.55e-13`.

| Dataset | Scored pairs | Multi-subunit | Pooled Spearman | Minimum per-time Spearman | Minimum top-10 Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| Zebrafish | 1,832 | 973 | 0.9986 | 0.9546 | 1.000 |
| MOSTA | 1,757 | 887 | 0.9859 | 0.9831 | 1.000 |
| ARISTA | 530 | 293 | 0.9611 | 0.9515 | 0.818 |
| Chicken Heart | 745 | 386 | 0.9771 | 0.9577 | 0.429 |
| AD mouse | 7 | 0 | 1.0000 | 1.0000 | 1.000 |

Thus the broad LR ranking is stable, especially in Zebrafish and MOSTA, but the
choice is not immaterial: ARISTA and especially Chicken Heart show changes
among the strongest pairs, and individual multi-subunit magnitudes can change
substantially. AD mouse is mathematically invariant because none of its seven
strict panel-supported pairs contains a multi-subunit ligand or receptor. The
machine-readable table is {download}`formal_lr_complex_aggregation_sensitivity.csv
<data/formal_lr_complex_aggregation_sensitivity.csv>`. These results support
the minimum gate as the declared primary estimand while requiring top-pair and
pair-specific biological claims to be checked against the geometric-mean
sensitivity rather than presented as aggregation-independent.

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
those analyses are `NA` rather than zero. The formal paired reconstruction
comparison is complete. It evaluates full-data, in-sample reconstruction with
the same target×space pairs in each arm; it is neither a leave-one-timepoint-out
(LOTO) benchmark nor a significance test.

| Dataset | No LR prior vs full | No interaction vs full |
| --- | ---: | ---: |
| AD mouse | +25.46% | -0.04% |
| ARISTA | +59.44% | -6.02% |
| MOSTA | +26.13% | -28.35% |
| Zebrafish | +13.53% | -10.16% |

Values are the mean of paired relative sliced-W2 changes computed separately
for every target×space combination; positive values mean higher (worse) W2
than full. Removing the learned LR prior worsened mean reconstruction W2 in all
four datasets. Removing interaction produced near-zero to lower mean W2, with
substantially different magnitudes by dataset. This establishes a
dataset-dependent reconstruction effect, not uniform full-model superiority or
evidence that interaction is biologically unnecessary. Old Zebrafish ablation
values must not be carried forward as results for the final model.

The signed report manifest has SHA-256
`b96de0c13023b6a4727e76ba8f67b84f3442f9c989b4d7a14dc03f5c1b904fdb`;
its companion PDF, PNG, paired tables, caption, and provenance record are the
authoritative matched-ablation reporting artifacts.

## Hyperparameter evidence boundary

The full, no-LR-prior, and no-interaction arms have now been re-fit in one
matched four-dataset grid. This does not make the retained graph-threshold,
score-batch, OT:mass, or scheduler diagnostics exhaustive sweeps around every
formal value. The separate primary cross-method four-dataset benchmark is
complete; its LOTO findings above remain distinct from the in-sample
three-arm reconstruction comparison.

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
