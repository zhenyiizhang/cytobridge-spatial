
# ARISTA strict-corrected S15--S17, submitted visual grammar

This immutable bundle replaces S15--S17 numerically without redesigning
the submitted figures. It uses the accepted full-model bank at
`/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/main_run/downstream` and reuses the exact submitted canvas/page geometry, palettes,
labels, axes, line grammar, and historical S17 order.

Scientific guardrails:

- persisted PCA center: `reference var['pca_center']`;
- reconstructed expression: per-cell clip at zero, then arithmetic mean;
- LR complexes: minimum across subunits with every subunit required;
- active-PCA feature universe shared across all nine time points;
- S17 N/E pairs remain visible, with `NaN` numeric values and explicit reasons;
- no output path is overwritten by the builder.

See `QA_REPORT.json`, `MANIFEST.json`, `CAPTION_UPDATE_NOTES.md`, and
`qa/legacy_vs_strict_corrected_contact_sheet.png`.
