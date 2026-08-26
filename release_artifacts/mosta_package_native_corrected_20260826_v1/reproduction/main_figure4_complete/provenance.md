# Figure provenance

Archived on: `2026-08-26`

Manuscript figure: `Main-text Figure 4 — MOSTA spatiotemporal generation, interaction, lineage transition, and velocity`

Scientific claim: `The accepted corrected/package-native MOSTA results reproduce the original Figure 4 biological narrative while replacing the historical numerical fields with audited outputs.`

## Files

- Vector PDF: `figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout.pdf`
- Vector SVG: `figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout.svg`
- PNG preview: `figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout_300dpi.png`
- Assembly script: `source/assemble_complete_figure4.py`
- Connector extraction audit: `source/extract_ai_strokes.py`
- PDF/SVG render comparison: `source/qa_compare_pdf_svg.py`
- Caption source: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/cytobridge_manuscript_latest_clean.pdf`, Figure 4 caption on PDF page 39
- Historical layout oracle: `output/mosta_main_fig4_completion_20260825_v1/archive_v1/style_authority/Figure_mouse1.ai`

## Selected experiment

- Accepted panel archive: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_main_fig4_completion_20260825_v1/archive_v1`
- Server/latest-model evidence: frozen within each accepted panel's `evidence/` directory
- Package commit: `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`
- Reference MOSTA H5AD SHA-256: `8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25`
- Accepted panel manifest: `output/mosta_main_fig4_completion_20260825_v1/archive_v1/main_fig4_completion_manifest.json`
- Assembly does not retrain or recompute biological fields; it consumes the five independently audited, immutable panels.

## Source paths

- Numerical and panel audit root: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_main_fig4_completion_20260825_v1/archive_v1`
- Layout/style authority: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_main_fig4_completion_20260825_v1/archive_v1/style_authority/Figure_mouse1.ai`
- Assembly source: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_main_figure4_assembled_20260826_v1/source/assemble_complete_figure4.py`
- Manuscript reference: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/cytobridge_manuscript_latest_clean.pdf`

## Panel sources

| Panel | Content | Accepted numerical source | Assembly calculation |
|---|---|---|---|
| a | Generate unseen time points | global-t0, 50,000-particle corrected MOSTA run | Exact 1:1 translation to AI crop `(0, 0, 595.276, 192)` |
| b | Wnt3a–Fzd7/Lrp6 interaction hotspot | corrected package-native interaction result; 50k display density | Exact 1:1 translation to AI crop `(0, 183.5, 326.6, 442)` |
| c | Cartilage lineage transition | latest accepted 52D classifier, `k=10`, E15.0→E15.5 | Exact 1:1 translation to AI crop `(316.086395, 202, 595, 440)` |
| d | Interaction-induced gene velocity | corrected observed E15.5 interaction velocity/communication | Exact 1:1 translation to AI crop `(0, 463.890015, 290, 841.890015)` |
| e | Telencephalon spatial velocity | corrected full/interaction velocity, `m=1024` | Exact 1:1 translation to AI crop `(286, 462, 595.276001, 841.890015)` |

The d→e zoom connectors are restored as two independent vector objects extracted directly from Illustrator-compatible PDF content operations `119949` and `119955`: width `2 pt`, RGB `(0.137, 0.09, 0.082)`, original endpoints unchanged.

## Evaluation protocol

- Panel a starts from the common global `t0` and uses 50,000 particles; generated stages do not restart from the preceding observation.
- Panel b preserves the accepted corrected computation and uses 50k only for display density; unavailable generated cells remain `NaN`/base grey.
- Panel c uses the frozen latest classifier and the requested `k=10`; displayed target percentages are Cartilage `27.1%`, Cartilage primordium `19.7%`, and Connective tissue `19.0%`.
- Panel d uses the same observed E15.5 state for the velocity field and communication graph.
- Panel e uses the accepted `m=1024` decomposition with maximum absolute reconstruction error `1.1915108188986778e-07`.
- No rotation, anisotropic scaling, warping, panel redesign, or ARISTA data/labels/palette/model/analysis was used.

## Raster policy

The PDF intentionally contains 67 embedded raster image objects inherited from the accepted panels. They are the dense embryo/spatial point-cloud and hotspot layers that were also rasterized in the original Illustrator figure or in the audited standalone replacements. Titles, labels, axes, ribbons, nodes, arrows, streamlines, legends, panel borders, and the two d→e connector lines remain vector. No complete panel was flattened.

## Rebuild command

```bash
/opt/anaconda3/bin/python output/mosta_main_figure4_assembled_20260826_v1/source/assemble_complete_figure4.py
/opt/homebrew/bin/rsvg-convert -f png -d 240 -p 240 -o output/mosta_main_figure4_assembled_20260826_v1/candidate_v8/qa/corrected_complete_Figure4_SVG_240dpi.png output/mosta_main_figure4_assembled_20260826_v1/candidate_v8/figures/Figure_4_MOSTA_corrected_complete_exact_AI_layout.svg
/Users/zhenyizhang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 output/mosta_main_figure4_assembled_20260826_v1/source/qa_compare_pdf_svg.py
```

The rebuild script refuses to overwrite `candidate_v8`; a rebuild must change `OUTPUT` to a new immutable candidate directory.

## Interpretation

The assembled figure keeps the original panel geometry, visual hierarchy, cell-type palette, typography, interaction arrows, streamlines, legends, and d→e zoom grammar. The biological narrative remains qualitatively intact, but quantitative statements must use the corrected values shown here: in particular, panel c is E15.0→E15.5 with `k=10`, and panel b's color scale is historical per-panel q1/q99 rather than an absolute cross-time magnitude comparison.

## SHA-256

- Figure PDF: `45beb12c6314052c4e33ce73255dcd8511a2e9e81e0a765ad858b0961cf80b40`
- Figure SVG: `e4c993d0b73456b83ce933196c5e0e70468a78bf59c93dc1b24665b891c5d73e`
- Figure PNG: `d3a33f830ac66f343382b54d1b0fed383ee1a6205bdf02eb437138a2759b559a`
- Assembly script: `3db007ec862b1a3c74c5b0e568c456302761de16197b4e31f05c7922cb8e056b`
- Illustrator style oracle: `340a5ed88dc911d6923bc6b21cf1ceb39fdbef16edf2e325822a5b422045cbc2`
- Submitted manuscript: `94c26a14500b16706ab9647ce26c628b9b7f642a58faf79421dd17577cae4337`
