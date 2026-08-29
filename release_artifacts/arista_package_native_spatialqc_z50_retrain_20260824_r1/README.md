# ARISTA package-native retraining release archive

This is the compact GitHub archive of the accepted ARISTA rerun completed on
2026-08-24. It contains the canonical scientific run products and final paper
and reviewer-response artifacts. The complete 3.2 GB server run remains at
`/data/cytobridge/projects/CytoBridge-ST-1104/runs/arista-package-native-spatialqc-z50-20260824-r1`
on `cytobridge-gpu`.

## What changed scientifically

- Expression preprocessing starts from the validated clean-count layer.
- The package selects 2,000 HVGs and adds required LR features, giving 2,246
  latent genes without a predefined paper gene roster.
- Latent normalization and PCA are fitted on the five ARISTA alignment batches.
- A label-blind per-slice 1-nearest-neighbor robust-z spatial QC filter at
  `z > 50` removes 10 detached observed cells before normalization, PCA, and
  training.
- The accepted run contains 46,199 observed cells, 50 PCs, all six training
  stages, 14,002 optimizer steps, and validation-selected edge threshold
  `0.6523735523223877`.
- Velocity, growth, fixed-particle composition, communication, gene dynamics,
  and strict complete-complex LR outputs were recomputed from the new model.

## Archive contents

- `main_run/`: exact transferred run log, configuration, histories, all six
  selected checkpoints, cell identity table, classifier caches, generated
  slices, velocity components, and downstream result tables/figures.
- `Figure5_fullpage_original_style_v2_final/`: accepted Figure 5 PDF/PNG,
  deterministic assembler snapshot, manifest, provenance, and QA records.
- `S12_package_native_warpk1_oldstyle_v3_legacy_palette/`: accepted S12 with
  `spatial_warp_k=1`, including observed and integer-time generated slices.
- `S13_S14_package_native_oldstyle_v3_legacy_palette/`: accepted S13 and S14.
- `S15_S17_package_native_strict_oldstyle_v1/`: retained strict source bundle.
  Its S15 page remains accepted; its hierarchical S16/S17 pages are preserved
  as superseded diagnostics and are not the current paper figures.
- `S16_lr_kmeans_recluster_v1/`: corrected deterministic k-means calculation
  from the 531 strict complete-complex LR trajectories. The two patterns contain
  217 and 314 pairs.
- `S16_package_native_kmeans_oldstyle_v1_finalqa/`: accepted corrected S16
  vector page and the tables used to draw it.
- `S17_package_native_balanced25_oldstyle_v1/`: accepted corrected S17 vector
  page, showing 25 representative pairs from each pattern (50 pairs total).
- `figure5c_two_niche_timecourse_v1/` and
  `figure5c_two_niche_lr_axes_v1/`: fixed-domain, matched-null, time-course,
  pathway, and pair-level LR evidence used for the reviewer response.
- `figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final/`: final
  reviewer-response figure in vector PDF/SVG plus PNG, caption, tables,
  provenance, and summary.
- `figure5c_reviewer_correspondence_v1/`: full biological interpretation and
  reviewer correspondence, including the clearly marked Chinese internal note.

The local parent output directory contains many superseded diagnostic versions.
Those are not part of this archive and must not be treated as accepted results.
Empty classifier lock files are also omitted.

## Code and provenance

The package fixes are part of this release branch. Canonical paper and reviewer
scripts are in `scripts/reviewer_arista_20260824/`. Exact fitted parameters and
runtime information are in `main_run/provenance/`; the primary acceptance report
is `FINAL_RELEASE_QA.md`.

From this directory, verify the seven current paper PDFs with:

```bash
sha256sum -c FINAL_RELEASE_SHA256SUMS.txt
```

The full Figure 5 assembler accepts explicit panel paths. The frozen output was
built with the command recorded in its manifest. Rebuilding the original page
also requires the submitted `Arista.ai` template whose SHA-256 is recorded in
that manifest; the template is not duplicated in this repository.
