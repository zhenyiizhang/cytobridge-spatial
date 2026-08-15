# Release notes

## 1.5.0rc1

- unified the formal Zebrafish, MOSTA, ARISTA, and AD workflow presets and
  added the anatomy-reviewed GSE149457 chicken-heart preset;
- completed the 12-run matched matrix (full learned prior, no-LR-prior
  `all_spatial`, and no-interaction for all four datasets), including package
  downstream; all 12 profiles and all four three-arm families pass acceptance
  SHA-256
  `c4f8e203e2da73fe78e28525516bbec192d3cbbd35d423dcd64080a0f83a10df`;
- added the installed `cytobridge workflow` command;
- corrected time-slice interaction velocity, pre-warp growth, sparse attention,
  generated expression, strict LR, gene/module, and distribution-evaluation
  semantics;
- added a formal LR-complex aggregation sensitivity that holds trajectories,
  expression support, communication attention, database, and scored pair
  universe fixed while replacing the primary minimum gate with a
  zero-preserving geometric mean. Zebrafish, MOSTA, ARISTA, and Chicken Heart
  primary tables were reproduced to at most `4.55e-13` absolute error before
  comparison; AD mouse has no scored multi-subunit complex and is invariant;
- coalesced byte-identical exact-OT support points while preserving their
  summed mass, eliminating network-simplex degeneracy without changing the
  empirical measure or the reported point cap;
- added the shared classifier protocol and dataset formal-k policy;
- added five package notebooks and ReadTheDocs guides;
- documented honest W2, sensitivity, ablation, training-curve, compute, and
  lineage limitations;
- completed all seven signed Zebrafish paper-downstream stages and the
  80-run interval-local daughter-noise sensitivity; paper S22 now uses one
  generated global-t0 fixed-population state transport (`growth_alpha=0`) with
  constant N and explicit non-abundance/non-reconstruction labeling, while
  S25/communication remain growth-enabled, interval-local, and
  observed-anchored;
- replaced the unstable unequal-N, growth-resampling S24 EVL panel with the
  `preterminal_t3_sigma0` protocol: target-specific equal-N deterministic
  spatial sensitivities through observed `t=3`, fixed
  `dt=resample_dt=0.005`, and a publication-blocking four-branch latent-support
  audit. The protocol is defined through observed `t=3`; terminal `t=4` is not
  evaluated or claimed, and stochastic-forecast and causal interpretations are
  explicitly out of scope;
- corrected the model-derived velocity renderer so direct 2D fields are not
  projected twice and scVelo's one-component NaN grid mask cannot erase finite
  vector streamlines in PDF output;
- stabilized bounded-memory GNN inference by replacing repeated-target CUDA
  atomic accumulation with target-sorted segment reduction and float64
  accumulation; the message formula and training path are unchanged, while
  five formal-scale replays now reproduce byte-identical trajectories;
- added fail-closed D4/D7/D10/D14 anatomical orientation validation and an
  explicit, distance-preserving repair for the known legacy D7 horizontal
  mirror; chicken-heart raw-count preparation records before/after coordinate
  hashes and never refits the reviewed alignment;
- completed a current-package chicken-heart full learned-prior fit, standard
  downstream, continuous D4-to-D14 perturbation/LR analysis, and paper-style
  figure bank; this is a single full-model application rather than a fifth
  matched three-arm family;
- bounded learned spatial-GNN inference memory by chunking only the no-gradient
  edge-message pass, preserving prediction count, interaction grouping, model
  weights, and the training path;
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
superiority. The five-application cross-method benchmark is complete: all 110
LOTO executions completed, CytoBridge has the lowest spatial sliced-W2 for
8/11 held-out targets, and the linear control wins 17/22 joint/state
comparisons. Seven stVCR full-data targets remain explicit `NA` after
method-native numerical failures (four ARISTA and three chicken heart). Old
benchmark and ablation summaries are not promoted into release conclusions.
The matched report manifest SHA-256 is
`b96de0c13023b6a4727e76ba8f67b84f3442f9c989b4d7a14dc03f5c1b904fdb`.

This is a release candidate. It should be merged to `main` only after source,
wheel, notebook, documentation, and independent science review pass.
