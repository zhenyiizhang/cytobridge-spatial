# Interpretation limits

- The accuracy-optimal classifier smoothing is k1 for all four formal spatial
  datasets. Z/M/A k10 is a deliberate spatial-domain estimand, not a claim of
  better pointwise accuracy.
- ARISTA lineage is unavailable because persistent particle IDs were not saved.
- Historical formal peak RSS/VRAM was not recorded and remains NA.
- Earlier MOSTA/ARISTA histories are sparse, but all 12 current matched runs
  have complete finite six-stage histories.
- The full, no-LR-prior, and no-interaction arms were re-fit in one matched
  grid; exhaustive sweeps around every graph/optimizer value were not run.
- The matched-ablation reconstruction comparison is complete, but is
  full-data/in-sample rather than LOTO and is not a significance test.
  Interaction effects are dataset-dependent; no uniform full-model superiority
  is claimed.
- The primary four-dataset cross-method benchmark remains pending.
- WOT has no native joint or spatial prediction and is NA in those spaces.
- The original Heart Figure 3 application is outside the four-dataset formal
  chain; later Heart-v2 benchmark evidence is a separate context.
- Heart remains a fifth paper application, but no formal Heart classifier-k or
  original Figure 3 training-curve provenance is claimed in this release.
- The package does not bundle large data, trained models, or external LR
  databases. These remain explicit user inputs.
