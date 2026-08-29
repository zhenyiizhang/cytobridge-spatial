# ARISTA package-native retraining and legacy-style figure release

Status: PASS

## Scientific run

- Package branch: `audit/arista-pca-root-fix-20260824`
- Package commits: `b0a9faa`, `5e66c03`, `1a2c7de`
- Training cohort: 46,199 observed cells across five time points
- Spatial QC: label-blind per-slice 1-NN robust-z filter, threshold 50; 10 clearly detached observed cells removed before normalization/PCA/training
- Latent feature roster: package-selected 2,000 HVGs plus required LR features, 2,246 genes total; 50 PCs
- Training: all six stages completed; 14,002 optimizer steps; final weights `Finetune`, final score model `Score_Refine`
- Fresh validation-selected edge threshold: 0.6523735523223877
- Downstream velocity, growth, composition, communication, gene dynamics, and ligand-receptor analyses: completed
- LR calculation contract: complete complexes only, all subunits required, persisted PCA center/loadings, no partial-complex fabrication

## Rendering contract

- Scientific values come from the fresh package-native run.
- Figure 5 and S12--S17 reuse the submitted plotting grammar and Illustrator layout.
- Canonical 27-cell-type palette SHA-256: `983b941fc93efe155511994d1d4b16cba5e11982cd81fb298d9a4a78907fbdd7`.
- The package's alphabetically reassigned categorical colors were not used as the paper palette. Generated snapshots were converted semantically as package color -> cell-type label -> submitted color; marker coordinates and counts were unchanged.
- Figure 5a uses the submitted five-slice focus-anchor renderer and only the audited +0.04 foreground-z lift.
- Figure 5b retains all 7,798 fresh generated t=0.5 markers at alpha 0.9; no display filter.
- Figure 5d uses the fresh raw package-native PCA and corrected full gene velocity with the historical scVelo stream style. The historical manual white arrow is removed and no replacement is invented.
- Figure 5d `Other` is the historical display aggregation for the 11 cell types outside the fixed 16-color roster; it is not an unknown model class.

## QA

- Figure 5 aggregate object QA: PASS
- Figure 5 independent pixel-composition QA (MuPDF and Poppler, 1x/2x/4x): PASS
- Figure 5 independent deterministic rebuild: byte-identical PASS
- Figure 5 stale old-marker/Form calls: 0
- Figure 5 stale historical Figure 5d white-arrow objects: 0
- S12--S17 Poppler render and visual inspection: PASS
- S15 strict numerical QA: PASS
- S16 corrected LR-pattern calculation: PASS. All 531 strict complete-complex
  trajectories were normalized row-wise and clustered by deterministic k-means
  (`k=2`, k-means++, 100 initializations, seed 0), producing 217 and 314 pairs.
- S16 model-selection check: PASS. `k=2` has the best silhouette score among
  `k=2` through `k=8`; no singleton cluster remains.
- S17 representative-profile selection: PASS. The 25 profiles nearest each
  corrected pattern mean are displayed (50 pairs, 450 time-course rows).
- The earlier hierarchical S16 (`1` and `530`) and fixed-slot S17 (68 slots,
  51 estimable) remain archived as superseded diagnostics, not current figures.

## Primary PDFs

- `Figure5_fullpage_original_style_v2_final/Figure5_ARISTA_package_native_finalAI_style.pdf`
- `S12_package_native_warpk1_oldstyle_v3_legacy_palette/figures/pdf/FigureS12_ARISTA_package_native_warpk1_oldstyle_FINAL.pdf`
- `S13_S14_package_native_oldstyle_v3_legacy_palette/figures/pdf/FigureS13_ARISTA_package_native_oldstyle_FINAL.pdf`
- `S13_S14_package_native_oldstyle_v3_legacy_palette/figures/pdf/FigureS14_ARISTA_package_native_oldstyle_FINAL.pdf`
- `S15_S17_package_native_strict_oldstyle_v1/figures/FigureS15_ARISTA_strict_corrected_legacy_style.pdf`
- `S16_package_native_kmeans_oldstyle_v1_finalqa/figures/FigureS16_ARISTA_package_native_kmeans_legacy_style.pdf`
- `S17_package_native_balanced25_oldstyle_v1/figures/FigureS17_ARISTA_package_native_balanced_representative_legacy_style.pdf`
