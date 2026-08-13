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

The formal matched-ablation reconstruction comparison is complete. Mean paired
relative sliced-W2 changes for no-LR versus full are +25.46% (AD), +59.44%
(ARISTA), +26.13% (MOSTA), and +13.53% (Zebrafish); no-interaction changes are
-0.04%, -6.02%, -28.35%, and -10.16%, respectively. These are full-data
in-sample comparisons, not LOTO or significance tests, and support a
dataset-dependent interaction effect rather than uniform full-model
superiority. The primary four-dataset cross-method benchmark remains pending;
legacy winner and old ablation summaries are not promoted into release
conclusions. The matched report manifest SHA-256 is
`b96de0c13023b6a4727e76ba8f67b84f3442f9c989b4d7a14dc03f5c1b904fdb`.

This is a release candidate. It should be merged to `main` only after source,
wheel, notebook, documentation, and independent science review pass.
