# Benchmarks and sensitivity analyses

The benchmark and sensitivity tables are evidence, not parameter-tuning
objectives. Unfavourable comparisons and method-native unsupported spaces stay
in the report as measured or `NA` rather than being replaced by surrogate
values.

## Unified sliced-W2 benchmark

The matched benchmark removes an internal target from model fitting, fixes a
source cohort and output count before opening target truth, fits transforms on
observed training slices, and evaluates joint/state/spatial sliced W2 with 256
projections for seeds 42–46. Unsupported method/space combinations are `NA`,
not replaced by a surrogate.

Across 27 dataset-target-space comparisons, Linear ranks first 13 times,
CytoBridge 11, PASTE 2, and MOSCOT 1. CytoBridge ranks first for 7 of 9 spatial
targets. AD shows a small mean advantage over Linear in joint, state, and
spatial W2; the results do not support universal superiority.

The compact winner table used for those counts is shipped as
{download}`unified_w2_winners.csv <data/unified_w2_winners.csv>`. Recreate the
method counts together with the classifier, compute, and hyperparameter table
summaries with:

```bash
python scripts/summarize_release_evidence.py
```

`cytobridge workflow --reconstruction-diagnostic` is a fitted-model
reconstruction diagnostic. It is not the cross-method holdout benchmark above.

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

The matched Zebrafish full/no-interaction/no-LR-prior comparison is
metric-dependent: the full model is not declared universally best. This is the
appropriate interpretation of targeted component ablations.

## Hyperparameter evidence boundary

Current results provide practical guidance for graph thresholds, score batch,
OT:mass, and scheduler settings, but not every current formal run-resolved value
has been exhaustively re-fit in one matched grid. Documentation must preserve
that boundary unless additional matched fits are run.

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
