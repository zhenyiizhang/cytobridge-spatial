# Benchmarks and sensitivity analyses

The benchmark and sensitivity tables are evidence, not parameter-tuning
objectives. Unfavourable comparisons and method-native unsupported spaces stay
in the report as measured or `NA` rather than being replaced by surrogate
values.

## Unified sliced-W2 benchmark

The unified benchmark is complete for the four accepted matched-model datasets
and the separate developing-chicken-heart full model. The run removes an
internal target from model fitting, fixes the source
cohort and output count before opening target truth, fits transforms on
observed training slices, and evaluates joint/state/spatial sliced W2 with 256
projections for seeds 42–46. Unsupported method/space combinations remain
`NA`, not surrogate values. All 110 LOTO method×target executions completed.
Of the 160 full-data executions, 153 completed. stVCR failed numerically for
the four ARISTA targets because its OT training weights became non-finite and
for the three chicken-heart targets after its simulated population became
empty. Those seven entries remain explicit `NA`.

The authoritative {download}`unified_w2_winners.csv
<data/unified_w2_winners.csv>` records the lowest LOTO sliced-W2 method for
each of the 33 dataset×target×space comparisons. Winner counts are 18 for the
linear-interpolation control, 11 for CytoBridge, 2 for PASTE, 1 for stVCR, and
1 for MOSCOT. CytoBridge is lowest in the spatial space for 8 of 11 held-out
targets; stVCR, MOSCOT, and the linear control each win one. Joint/state
results are less favourable: the linear control wins 17 of 22 such
comparisons, CytoBridge wins three, and PASTE wins two.
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
| Chicken heart | Random interpolation | CytoBridge | Random interpolation |
| MOSTA | PASTE | PASTE | PASTE |
| Zebrafish | Random interpolation | CytoBridge | Random interpolation |

The ten signed evaluation/summary manifest hashes are shipped in
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
populations. Formal output uses Z/M/A k10 and AD/chicken-heart k1 consistently.
The explicit policy is available as {download}`formal_k_policy.csv
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
formal value. The separate primary cross-method benchmark is complete for all
five current applications; its LOTO findings above remain distinct from the
in-sample three-arm reconstruction comparison.

The run-resolved values used by the five current full presets are available in
{download}`formal_hyperparameter_settings.csv
<data/formal_hyperparameter_settings.csv>`.

The production selection rules are:

- spatial cutoff: compute the median within-slice nearest-neighbour distance,
  set the recommended spot diameter to `1.2 × median(NN1)`, and use
  `4 × mean(spot diameter)` across observed slices;
- edge threshold: for all five main runs, maximize validation F1 over candidate
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

## Developing chicken heart

Developing chicken heart is now the fifth package-native biological
application. Its current evidence does not reuse the historical Figure 3 fit.
The package rebuilds the aligned input from the four raw GSE149457 count
matrices while preserving the reviewed spot order and coordinates. It applies
only the explicitly recorded D7 horizontal reflection: D7, D10, and D14 place
RV to the right of LV, and all four slices place atrial regions above valve
regions. D4 does not provide a stable LV/RV split and is therefore checked by
the atrial/valve relation rather than by inventing a left-right label.

The current full learned-prior model trains on 3,550 spots with a 50-dimensional
expression state plus the first two fitted spatial dimensions. Its six-stage
training, standard package downstream, corrected velocity outputs, continuous
D4-to-D14 perturbation analysis, and LR/attention figure bank are complete.
The formal classifier policy is k1. Direct spatial velocity uses the two fitted
spatial dimensions. Expression/gene velocity is reconstructed from the full
50-dimensional expression state and projected by scVelo onto the observed
spatial coordinates. The perturbation panels are single-seed model-sensitivity
demonstrations, not causal knockouts.

Chicken heart is a completed single full-model application, not a fifth arm
family in the accepted four-dataset matched ablation matrix. Its separate
10-method LOTO/full-data benchmark uses D4 as the source anchor, D7/D10 as
held-out LOTO targets, and D7/D10/D14 for full-data reconstruction. All 20
LOTO executions completed. CytoBridge is lowest for D7 joint and spatial
sliced-W2, while linear interpolation wins D7 state and D10 joint/state and
MOSCOT wins D10 spatial. In the in-sample full-data diagnostic, CytoBridge has
the lowest aggregate spatial sliced-W2 and random interpolation has the lowest
aggregate joint/state values. stVCR's three full-data outputs are explicit
`NA` after method-native population extinction. The signed LOTO evaluation and
summary hashes are `e38cb01ecd2f65f2d4945ad9e55f883812303bc0812cf8b430b43214def9d537`
and `f025f13e544cd4b93902c9d478dbec8749f4576f686fafc35bd52ecc401cd740`;
the full-data hashes are
`8b3c66053a0773b1b16e34c6f2e9b17ac2a26288f3b89098e9253ed74aa63fea`
and `d0143d4146a34490ee70d290248edf10ddc0a57dcc7beff388b7e908073584fa`.
The older Heart-v2 values are not substituted.
