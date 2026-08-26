# Reproducible MOSTA manuscript figures

This directory is the reader-facing release of every accepted corrected MOSTA
panel in the main manuscript and Supplementary Information.

## Scope

- Main text: Figure 4a-4e and the complete assembled Figure 4.
- Supplementary Information: Figures S4-S11.
- S12 onward is ARISTA and is deliberately excluded.

All numerical results use the corrected package-native MOSTA training run at
CytoBridge commit `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`.
Generated intermediate stages use global-t0 propagation. The accepted classifier
uses `k=10`. No ARISTA data, labels, palette, checkpoint, or numerical analysis
is used. Historical notebooks are retained only as plotting/style authorities.

## Start here

- `figures/main/Figure_4_complete.pdf`: complete publication Figure 4.
- `figures/main/panels/`: standalone Figure 4 panels.
- `figures/si/`: standalone Figures S4-S11.
- `model/`: corrected Finetune, Score, and generated-cell classifier checkpoints.
- `reproduction/`: exact accepted computation/rendering scripts, compact numerical
  inputs, audit tables, manifests, and provenance.
- `historical_plotting_code/`: historical MOSTA plotting notebooks/scripts with
  notebook outputs removed but all code cells retained.
- `REPRODUCIBILITY.md`: environment, data, recomputation, and rendering order.
- `MANIFEST.json` and `CHECKSUMS.sha256`: release identities.

Dense scatter or spatial layers are intentionally rasterized only where required
for file size. Text, axes, ribbons, arrows, streamlines, legends, borders, and
layout objects remain vector. The complete Figure 4 uses the exact Illustrator
panel coordinates with translation only; no rotation, anisotropic scaling, or warp.
