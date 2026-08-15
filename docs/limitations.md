# Interpretation limits

- The accuracy-optimal classifier smoothing is k1 for all five current full
  workflows. Z/M/A k10 is a deliberate spatial-domain estimand, not a claim of
  better pointwise accuracy; AD and chicken heart retain k1.
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
- The five-application cross-method benchmark is complete, but projection
  repeats are numerical rather than biological replicates and no cross-space
  overall score is defined. Seven stVCR full-data targets are `NA` after
  method-native numerical failures: four ARISTA targets with non-finite OT
  weights and three chicken-heart targets after simulated-population
  extinction.
- WOT has no native joint or spatial prediction and is NA in those spaces.
- The original Heart Figure 3 application remains historical and is not used
  as current evidence. The current package-native chicken-heart full fit has a
  formal k1 policy and complete training telemetry, but no matched
  chicken-heart no-LR/no-interaction family was run.
- The package does not bundle large data, trained models, or external LR
  databases. These remain explicit user inputs.
