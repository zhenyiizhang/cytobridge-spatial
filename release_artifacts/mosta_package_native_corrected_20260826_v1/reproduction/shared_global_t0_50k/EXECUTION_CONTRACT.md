# MOSTA SI S4–S10 shared compute execution contract

This bundle computes numerical values only. It does not draw or restyle any panel.

- Server output: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-paper-figures/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`
- Output policy: the target must not exist; the runner refuses reuse and seals a successful result read-only with `COMPLETE` and `SHA256SUMS.txt`.
- Package: release marker `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`.
- Model/input/classifier: latest accepted MOSTA run, pinned by the hashes in `STATIC_REVIEW.json`.
- Simulation: 50,000 t0 particles, seed 42, one global-t0 rollout, 13 quarter-step times, native coordinates, no piecewise restart and no spatial warp.
- S4/S5/S6/S8/S10 use the same fully generated split-SDE state family.
- S7 uses the same workflow's non-split fixed-particle trajectory, preserving all 50,000 identities.
- S8 is Brain-only and uses the original 2,000 statistical HVGs; S10 derives from the identical S8 matrix.
- S9 GO enrichment and S11 ligand–receptor k=3 clustering are deliberately outside this compute and will be chained afterward.

The submitted SI PDF, historical notebooks, and Illustrator artwork remain the only style authority. No ARISTA numerical or styling asset is used.
