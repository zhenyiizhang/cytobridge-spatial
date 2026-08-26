# MOSTA main-text and supplementary-figure master delivery

Status: **COMPLETE / ACCEPTED / IMMUTABLE**

This bundle is a convenient, checksum-locked index of every accepted MOSTA figure in the submitted main text and Supplementary Information. It copies the final PDF/SVG artifacts byte-for-byte; complete numerical inputs, audit tables, plotting sources, QA renders, and provenance remain in the referenced source archives listed in `ARCHIVE_CONTRACTS.csv`.

## Scope

- Main text: Fig. 4a-4e.
- Supplementary Information: Fig. S4-S11.
- Fig. S12 onward starts ARISTA and is deliberately out of MOSTA scope.

## Locked interpretation

- Numerical truth: corrected package-native MOSTA model `/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-matched-ablation-20260813-3c87a3e-r1/mosta/training` under release commit `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`.
- Shared dense trajectory truth for S4-S10 where applicable: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`; 50,000 particles, 13 times from t0=0 to t=3, global-t0 propagation, no observed-state restart.
- GO truth for S9/S10: server run `/data/cytobridge/projects/CytoBridge-ST-1104/runs/si-s9-s10-clusterprofiler-mouse-all-pool-2b3c79e-20260826-v1` using clusterProfiler 4.10.0 and org.Mm.eg.db 3.18.0.
- Style truth: submitted manuscript/SI panels, original MOSTA notebooks/scripts, and the original Illustrator layout where applicable.
- No ARISTA data, labels, palette, model result, or analysis logic was used. A historical helper imported for plotting syntax in one archive is style authority only, as declared there.
- No rotation, stretch, shear, projection warp, or other geometry manipulation was used to imitate the submitted appearance.

## Delivered figures

| Panel | PDF | SVG | Numerical/style/QA status |
|---|---|---|---|
| Fig. 4a | `figures/main/Fig4a.pdf` | `figures/main/Fig4a.svg` | ACCEPTED PASS |
| Fig. 4b | `figures/main/Fig4b.pdf` | `figures/main/Fig4b.svg` | ACCEPTED PASS |
| Fig. 4c | `figures/main/Fig4c.pdf` | `figures/main/Fig4c.svg` | ACCEPTED PASS |
| Fig. 4d | `figures/main/Fig4d.pdf` | `figures/main/Fig4d.svg` | ACCEPTED PASS |
| Fig. 4e | `figures/main/Fig4e.pdf` | `figures/main/Fig4e.svg` | ACCEPTED PASS |
| Fig. S4 | `figures/si/Figure_S4.pdf` | `figures/si/Figure_S4.svg` | ACCEPTED PASS |
| Fig. S5 | `figures/si/Figure_S5.pdf` | `figures/si/Figure_S5.svg` | ACCEPTED PASS |
| Fig. S6 | `figures/si/Figure_S6.pdf` | `figures/si/Figure_S6.svg` | ACCEPTED PASS |
| Fig. S7 | `figures/si/Figure_S7.pdf` | `figures/si/Figure_S7.svg` | ACCEPTED PASS |
| Fig. S8 | `figures/si/Figure_S8.pdf` | `figures/si/Figure_S8.svg` | ACCEPTED PASS |
| Fig. S9 | `figures/si/Figure_S9.pdf` | `figures/si/Figure_S9.svg` | ACCEPTED PASS |
| Fig. S10 | `figures/si/Figure_S10.pdf` | `figures/si/Figure_S10.svg` | ACCEPTED PASS |
| Fig. S11 | `figures/si/Figure_S11.pdf` | `figures/si/Figure_S11.svg` | ACCEPTED PASS |

## Important corrected analyses

- Fig. 4c and S7 use the unchanged latest classifier with k=10. The accepted Fig. 4c interval is E15.0 to E15.5; the result was not relabelled to manufacture the old three-category message.
- Fig. S9/S10 use genuine server clusterProfiler enrichment. S9 pattern 2 displays 11 terms because exactly 11 pass the specified significance gate; no filler terms were introduced.
- Fig. S11 uses package `M_sum`, matching the submitted cell-type-aggregated total-attention estimand. The rejected `M_per_source` version created normalization-driven synchronized pulses. Seeds 42/43/44 give median pairwise-profile correlations of 0.9993-0.9996 and adjusted Rand indices of 0.859-0.932; the final 31 representatives are stability/effect-size selected rather than visually cherry-picked.

## Manuscript sources

- Main PDF: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/cytobridge_manuscript_latest_clean.pdf`; SHA-256 `94c26a14500b16706ab9647ce26c628b9b7f642a58faf79421dd17577cae4337`.
- SI PDF: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf`; SHA-256 `150deefb96083732a7aa7ac89bda1556c3ee4900699ed84946ecf7de48f9c93d`.

Every source archive checksum manifest was reverified before copying. `CHECKSUMS.sha256` verifies this master delivery; `SOURCE_ARCHIVE_VERIFICATION.json` records the upstream archive checks.
