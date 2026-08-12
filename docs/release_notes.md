# Release notes

## 1.5.0rc1

- unified the formal Zebrafish, MOSTA, ARISTA, and AD workflow presets;
- added the installed `cytobridge workflow` command;
- corrected time-slice interaction velocity, pre-warp growth, sparse attention,
  generated expression, strict LR, gene/module, and distribution-evaluation
  semantics;
- added the shared classifier protocol and dataset formal-k policy;
- added four package notebooks and ReadTheDocs guides;
- documented honest W2, sensitivity, ablation, training-curve, compute, and
  lineage limitations;
- made the primary workflow and tutorials ordinary package entry points rather
  than release-audit launchers.
- removed checksum-only fields from the public downstream result objects;
  retained only the classifier input fingerprint that prevents stale cache
  reuse and deterministic benchmark seed identities.

This is a release candidate. It should be merged to `main` only after source,
wheel, notebook, documentation, and independent science review pass.
