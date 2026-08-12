# Shared scientific contract

## What is shared

- one preprocessing/alignment API;
- one six-stage training implementation;
- `alpha_express=0.015`, `alpha_spatial=10`, seed 42 for the four formal spatial applications;
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
maps, interaction cutoff, edge threshold, particle count, annotation key,
species database, and whether persistent particle identities exist. These are
stored in small workflow presets rather than copied implementations.

| Dataset | Interaction cutoff | Edge threshold | Formal k | Particle scope |
| --- | ---: | ---: | ---: | --- |
| Zebrafish | 0.0960636741 | 0.4999999702 | 10 | all available t0 cells |
| MOSTA | 0.0240024405 | 0.4499999881 | 10 | 12,000 |
| ARISTA | 0.0315410515 | 0.2399999946 | 10 | 7,668 |
| AD | 0.0121060429 | 0.3299999833 | 1 | all 53,615 observed t0 cells |

The accuracy sweep selected `k=1` for all four datasets. Z/M/A retain `k=10`
as the manuscript spatial-domain estimand; AD uses `k=1` because larger votes
substantially erase rare and heterogeneous populations.

## Warp policy

Spatial warp is allowed for mosaics, videos, and the display coordinates of 3D
slice figures. Velocity, growth, communication, LR, gene dynamics, benchmarks,
and any valid lineage calculation use pre-warp coordinates and states.

ARISTA split frames do not retain persistent particle identifiers. The formal
workflow therefore omits lineage rather than joining rows across time.
