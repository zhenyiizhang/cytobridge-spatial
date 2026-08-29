# Figure provenance — Supplementary Figure S6

## Source paths

- Submitted SI style authority: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf`, page 32.
- Submitted plotting notebook snapshot: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s6_composition_20260825_v1/source/mosta-brain-cell-composition-review.ipynb`, cells 6 and 8.
- Saved submitted panel-A SVG style authority: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s6_composition_20260825_v1/source/submitted_s6_panel_a_style_reference.svg`.
- Saved submitted panel-B SVG style authority: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s6_composition_20260825_v1/source/submitted_s6_panel_b_style_reference.svg`.
- Original MOSTA categorical palette: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s6_composition_20260825_v1/input/label_to_color.json`.
- Corrected composition input: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1/s6_composition/celltype_composition_fully_generated.csv`.
- Generated-state inventory: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1/generated_states/state_inventory.csv`.
- Shared server summary: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1/summary.json`.

## Experiment or command

Numerical truth was computed on the server in the immutable run:

`/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-paper-figures/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`

The calculation used package release commit `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`, the accepted MOSTA model, the accepted cached classifier with `k=10`, 50,000 initial particles, split-SDE `dt=0.05`, `sigma=0.03`, growth alpha 1, and seed 42. All 13 quarter-step states were generated from the same global t0 source. No observed-stage restart and no spatial warp were used.

Local audit:

```bash
python output/mosta_si_s6_composition_20260825_v1/source/audit_s6_latest_package_composition.py
```

Local exact-style rendering:

```bash
python output/mosta_si_s6_composition_20260825_v1/source/render_s6_corrected_exact_submitted_style.py
```

## Manuscript protocol

- Panel a is the submitted stacked-area count geometry: continuous time on x, white boundaries, alpha 0.95, line width 0.5, the submitted title/axis strings, y grid, and right-side legend.
- Panel b is the submitted stacked-percentage-bar geometry: 13 bars, width 0.76, white boundaries, line width 0.6, 0–100% y limits, two-decimal time labels, y grid, and right-side legend.
- Font is DejaVu Sans because that is the submitted notebook/SVG font.
- The old palette is retained exactly: Brain orange, Connective tissue cyan, Cavity light gray, Epidermis blue, and the remaining submitted cell-type colors.
- The displayed 15 categories and their stack/legend order are frozen from the submitted notebook output and SI: Brain, Connective tissue, Cavity, Epidermis, Muscle, Jaw and tooth, Meninges, Liver, Cartilage primordium, Spinal cord, Heart, GI tract, Dorsal root ganglion, Cartilage, Adipose tissue; all other corrected labels are collapsed into Other.
- This fixed submitted order is intentional. Re-ranking the corrected values would hide Cartilage in Other and would not be an equivalent replacement of the submitted legend/content selection.
- No smoothing, category cherry-picking, rotation, stretching, or coordinate warp is applied.
- The PDF and SVG are fully vector; the PNG is a preview only.

## Validation status

- Numerical audit: PASS.
- All 13 local state SHA-256 values match the server inventory.
- Composition counts match each H5AD `Annotation` column exactly at every time.
- Counts sum to each declared total and fractions equal count/total; maximum floating error is `9.99e-17`.
- Total population increases continuously from 50,000 to 105,165 without observed-anchor jumps.
- Realized interval population log-growth agrees with the package growth network: correlation `0.999492`, mean absolute error `0.004367`, and every interval rate lies between the two endpoint growth-net means.
- The rejected mixed observed/generated table with SHA-256 `543e07f9775002f7241556d31ceff35c700e2ef91c8d734d95ad569a4886943b` is not used.
- Actual page-1 PDF rendered at 240 dpi and visually inspected: two panels, original order/palette/legend, no clipping, no transform artifact.

## Interpretation

The corrected panel preserves the submitted qualitative message while removing artificial anchor discontinuities. Total abundance grows by about 2.10-fold, with the expansion rate slowing over time in agreement with the learned growth network. Relative composition shows expansion of Connective tissue and late emergence of Spinal cord and Cartilage, while Cartilage primordium and early Muscle fractions decline. These are direct corrected outputs, not presentation-driven edits.

## Rebuild

From `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104`, run the numerical audit first and require `status: PASS`, then run the renderer:

```bash
python output/mosta_si_s6_composition_20260825_v1/source/audit_s6_latest_package_composition.py
python output/mosta_si_s6_composition_20260825_v1/source/render_s6_corrected_exact_submitted_style.py
```

Render the resulting PDF itself for visual QA; do not inspect only the direct PNG export:

```bash
pdftoppm -f 1 -singlefile -png -r 240 \
  output/mosta_si_s6_composition_20260825_v1/figures/Figure_S6_MOSTA_latest_package_global_t0_50k_exact_submitted_style.pdf \
  output/mosta_si_s6_composition_20260825_v1/qa/s6_corrected_fixed_submitted_order_pdf_240dpi
```

## Deliverables

- Vector PDF: `figures/Figure_S6_MOSTA_latest_package_global_t0_50k_exact_submitted_style.pdf`
- Vector SVG: `figures/Figure_S6_MOSTA_latest_package_global_t0_50k_exact_submitted_style.svg`
- Preview PNG: `figures/Figure_S6_MOSTA_latest_package_global_t0_50k_exact_submitted_style.png`
- Numerical audit: `tables/s6_numerical_audit.json`
- Growth-dynamics cross-check: `tables/s6_population_growth_dynamics_crosscheck.csv`
- Figure manifest: `figures/Figure_S6_MOSTA_latest_package_global_t0_50k_exact_submitted_style_manifest.json`

## SHA-256 identities

- Corrected composition table: `35c7118aefbadc2185fbf9ffd7760cdfdcafb93a5d3a7d18172c72525d04a3f3`
- Numerical audit: `2d025e24d308bf2197d74217abfea005460300977443320671793431a5112212`
- Submitted notebook snapshot: `3ae4d01d807d00819b2c93f13822f61eda12a9947d732e749bac517509643efc`
- Panel-A submitted SVG reference: `83417c7018538221f3e80c474778e730db23ce4e55dd7b09e8313d7163980f0b`
- Panel-B submitted SVG reference: `f415a6e8af41b25476d03633228a7f7feb25d4c093e3b25c39501c44d48e376e`
- Palette: `7e95e868e0a6ecd4a2ed13b57e6a8223e77e2302a0f9634ca30f41390c040b71`
- Renderer: `f4f0d5e841d0ba233a990467136ef7943e801f790f4682ffff120e656dea1d8e`
- Audit script: `ce52d5a0271b17ae229f1bdf4a0703e3e3d6a5f3e26d3b1af856d178aa1664e0`
- PDF: `c7d64bcef512c96cfad54d7ada5f3e8bd273067b2729cf454fa9d5a5ed70c9ac`
- SVG: `fe81ea3b70278acc170c52034bc7f1060d6250f2a4af907b24909c4ac0620567`
- PNG: `3836ff59c8a5e88d5219d78172f621eb4372744f54804fd97a8d19a36cef986c`
- 240-dpi rendered PDF QA image: `b8836e297e67f82b321ec7ad004fe6f83bbcb6d2597ef3b354131fdf945c7a51`
