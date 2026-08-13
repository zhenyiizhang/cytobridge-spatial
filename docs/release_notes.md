# Release notes

## 1.5.0rc1

- unified the formal Zebrafish, MOSTA, ARISTA, and AD workflow presets;
- completed the 12-run matched matrix (full learned prior, no-LR-prior
  `all_spatial`, and no-interaction for all four datasets), including package
  downstream; all 12 profiles and all four three-arm families pass acceptance
  SHA-256
  `c4f8e203e2da73fe78e28525516bbec192d3cbbd35d423dcd64080a0f83a10df`;
- added the installed `cytobridge workflow` command;
- corrected time-slice interaction velocity, pre-warp growth, sparse attention,
  generated expression, strict LR, gene/module, and distribution-evaluation
  semantics;
- added the shared classifier protocol and dataset formal-k policy;
- added four package notebooks and ReadTheDocs guides;
- documented honest W2, sensitivity, ablation, training-curve, compute, and
  lineage limitations;
- completed all seven signed Zebrafish paper-downstream stages and the
  80-run interval-local daughter-noise sensitivity; canonical reconstruction
  remains observed-anchored rather than global-t0;
- made the primary workflow and tutorials ordinary package entry points rather
  than release-audit launchers;
- removed checksum-only fields from the public downstream result objects;
  retained only the classifier input fingerprint that prevents stale cache
  reuse and deterministic benchmark seed identities.

The primary four-dataset benchmark and comparative matched-ablation result
tables are still running/pending. Legacy winner and old ablation summaries are
not evidence for the accepted matched models and are not promoted into release
conclusions.

This is a release candidate. It should be merged to `main` only after source,
wheel, notebook, documentation, and independent science review pass.
