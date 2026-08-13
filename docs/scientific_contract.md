# Shared scientific contract

## What is shared

- one preprocessing/alignment API;
- one six-stage training implementation;
- `alpha_express=0.015`, `alpha_spatial=10`, seed 42 for the four formal spatial applications;
- one explicit `velocity_score_cross_term` score-energy objective across the
  full, no-LR-prior, and no-interaction arms, independent of the interaction
  gate;
- an independently trained Residual MLP per dataset with hidden width 128,
  Adam at `1e-3`, cosine annealing over 500 selection epochs, a fixed
  stratified 90/10 development split, and balanced-accuracy checkpoint
  selection;
- time-slice velocity, growth, sparse communication, strict LR, gene and
  benchmark APIs;
- quantitative outputs from aligned pre-warp state;
- manuscript-style plotting helpers separated from scientific computation.

## What may differ by dataset

Biological scale and data availability require explicit differences in time
maps, interaction cutoff, accepted matched-full edge threshold,
particle count, annotation key, species database, and whether persistent
particle identities exist. These are stored in small workflow presets rather
than copied implementations.

| Dataset | Interaction cutoff | Main edge prior | Accepted matched-full predictor threshold | Formal k | Particle scope |
| --- | ---: | --- | ---: | ---: | --- |
| Zebrafish | 0.0960636741 | learned predictor | 0.6063615680 | 10 | all available t0 cells |
| MOSTA | 0.0240024405 | learned predictor | 0.1192110926 | 10 | 12,000 |
| ARISTA | 0.0315410515 | learned predictor | 0.5884028673 | 10 | 7,668 |
| AD | 0.0121060429 | learned predictor | 0.9956824780 | 1 | all 53,615 observed t0 cells |

All four corrected de novo workflows keep the preset cutoff, train a new edge
predictor, and use its validation-selected threshold. AD has only seven strict
complete LR pairs in its targeted panel; the main model uses that learned prior
and explicitly limits its biological interpretation. The separately packaged
`all_spatial` profile is the matched no-LR-prior ablation. The 12-run matched
matrix and all four three-arm families pass formal acceptance. The paired
full-data reconstruction comparison is complete, but is neither LOTO nor a
significance test. No-LR prior increases mean paired relative sliced W2 versus
full by 25.46% (AD), 59.44% (ARISTA), 26.13% (MOSTA), and 13.53% (Zebrafish).
No-interaction changes it by -0.04%, -6.02%, -28.35%, and -10.16%,
respectively; the interaction effect is therefore dataset-dependent, not
evidence of uniform full-model superiority.

Downstream sparse-attention export has a different scope. It reconstructs the
full radius graph for each analyzed time-slice cohort, or for the explicit
seeded subsample when a cap is configured, and evaluates its edges in memory
batches. Those reporting edges are not the exact stochastic groups realized
during model fitting or simulation.

For AD main, the mouse CellChatDB supplies the seven strict complete pairs used
both for predictor labels and the downstream projection. The corrected run
selected threshold `0.9956824779510498`. The limited targeted panel means none
of these outputs may be described as global CCI inference. The accepted
radius-only condition is the matched no-LR-prior ablation, not the production
main.

The accuracy sweep selected `k=1` for all four datasets. Z/M/A retain `k=10`
as the manuscript spatial-domain estimand; AD uses `k=1` because larger votes
substantially erase rare and heterogeneous populations.

## Warp policy

Spatial warp is allowed for mosaics, videos, and the display coordinates of 3D
slice figures. Velocity, growth, communication, LR, gene dynamics, benchmarks,
and any valid lineage calculation use pre-warp coordinates and states.

ARISTA split frames do not retain persistent particle identifiers. The formal
workflow therefore omits lineage rather than joining rows across time.
