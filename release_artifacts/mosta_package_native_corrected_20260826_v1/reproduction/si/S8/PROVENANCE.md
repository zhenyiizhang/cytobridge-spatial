# Figure provenance

Archived on: `2026-08-26`

Manuscript figure: `Supplementary Figure S8a-b — Temporal gene programs in the MOSTA dataset`

Scientific claim: Corrected latest-package MOSTA trajectories separate Brain original-HVG profiles into an early-decreasing program and a late-increasing program without forcing the program sizes reported by the legacy calculation.

## Files

- Vector figure: `figures/Figure_S8_MOSTA_latest_package_brain_gene_programs_exact_submitted_style.pdf`
- Editable vector: `figures/Figure_S8_MOSTA_latest_package_brain_gene_programs_exact_submitted_style.svg`
- PNG preview: `figures/Figure_S8_MOSTA_latest_package_brain_gene_programs_exact_submitted_style.png`
- Actual PDF render used for visual QA: `qa/s8_pdf_200dpi.png`
- Numerical audit: `tables/s8_numerical_audit.json`
- Plotting script: `source/render_s8_corrected_exact_submitted_style.py`
- Audit script: `source/audit_s8_latest_package_gene_programs.py`
- Submitted style reference: `source/submitted_s8_embedded_figure_style_reference.jpeg`
- Historical notebook style oracle: `source/mosta-brain-gene-review.ipynb`
- Caption source: `/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/supp_text/CytoBridge_Supplementary_Figures.tex:89`
- Compiled SI reviewed: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf`, page 34

## Selected experiment

- Server package release: `/data/cytobridge/projects/CytoBridge-ST-1104/software/cytobridge-release-2b3c79e-runtime`
- Package commit: `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`
- Package archive SHA-256: `06852992db0d8ebedd8f0baa19a3f539bcdd271dfc0c6fae9774dd553c8fe55e`
- Server model run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/corrected-matched-ablation-20260813-3c87a3e-r1/mosta/training`
- Finetune checkpoint SHA-256: `d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5`
- Score checkpoint SHA-256: `d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a`
- Classifier: package-native `k=10`, SHA-256 `f938575c145baa9002de695c39a8637f57d6cb3a06ccfaf4d18b707ca962a7e0`
- Aligned H5AD SHA-256: `8b9ca0ad3475040235036548d54b96272bf6c49f057f6c2a643152c11350ce25`
- Server shared analysis run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-paper-figures/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`
- Local read-only download: `output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`
- Render manifest: `figures/render_manifest.json`

## Source paths and panel calculations

| Panel | Content | Source files | Calculation |
|---|---|---|---|
| a | Top 25 variable Brain gene trajectories | `brain_hvg_mean_log1p_by_time.csv`; `brain_hvg_temporal_variance_rank.csv` | Rank the audited 2,000 original statistical HVGs by temporal variance and draw the first 25 corrected clipped mean-log1p trajectories. The display count, line grammar, legend and repeated `tab10` palette match the submitted notebook. |
| b | Two Brain temporal programs | `brain_hvg_gene_wise_zscore.csv`; `brain_hvg_ward_k2_assignments.csv`; `brain_hvg_ward_k2_prototypes.csv`; `brain_program_representative_genes_top5.csv` | Gene-wise population z-score (`ddof=0`), Ward Euclidean linkage, exact `cut_tree(k=2)`, then label programs by prototype peak time. Representative genes are ranked by temporal variance plus prototype correlation. |

## Evaluation protocol

- Initial particles: `50,000` fixed persistent particles sampled at global model time `t=0`
- State times: `0.00, 0.25, ..., 3.00` (13 fully generated states)
- Observed-data restart: `false`
- Spatial warp: `false`
- SDE time step: `0.05`
- Split-SDE diffusion scale: `0.03`
- Growth alpha: `1.0`
- Seed: `42`
- ROI: exact classifier label `Brain`
- Brain cells: `11,397` at model time 0 and `27,595` at model time 3
- Gene universe: `2,000` original statistical HVGs; `747` LR-required latent additions excluded
- Reconstruction: per-cell inverse PCA using persisted `reference_adata.var['pca_center']`, clip reconstructed log1p values at zero, then arithmetic mean
- Corrected program sizes: Pattern 1 `883`; Pattern 2 `1,117`
- Independently recomputed Ward silhouette: `0.6210163711`

## Rebuild

From `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104`:

```bash
python output/mosta_si_s8_gene_programs_20260826_v1/source/audit_s8_latest_package_gene_programs.py \
  --shared-run output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1 \
  --output-dir output/mosta_si_s8_gene_programs_20260826_v1/tables

python output/mosta_si_s8_gene_programs_20260826_v1/source/render_s8_corrected_exact_submitted_style.py \
  --input-dir output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1/s8_gene_programs \
  --output-dir output/mosta_si_s8_gene_programs_20260826_v1/figures/render_v5 \
  --style-reference output/mosta_si_s8_gene_programs_20260826_v1/source/submitted_s8_embedded_figure_style_reference.jpeg
```

The second command requires a new empty output directory because the renderer refuses to overwrite an existing render.

## Visual QA

- PDF page: one page, `606.96 × 792.00 pt`, rotation 0
- PDF is vector-only: no embedded raster images; the continuous colorbar uses a PDF-native Gouraud mesh
- PDF render: `1686 × 2200 px` at 200 dpi, exactly matching the submitted JPEG canvas
- OCR coordinate comparison against the submitted JPEG: `Time` at y=882 vs 882; `Pattern 1` at y=1108 vs 1108; lower `Time` at y=2151 vs 2152; horizontal lower-panel coordinates agree within 0–4 px for the checked labels
- DejaVu Sans is intentional because the submitted S8 was a Photoshop-composed JPEG and its historical notebook/SVG renderer used Matplotlib's DejaVu Sans; Arial substitution would not reproduce this panel's source grammar
- No coordinate rotation, stretch, warp or image-level transformation was used

## Interpretation

The submitted layout and color semantics are preserved, but legacy negative panel-a values are not: the corrected reconstruction is nonnegative after per-cell clipping. Pattern 1 peaks at time 0 and decreases (`r=-0.9932` with time), whereas Pattern 2 peaks at time 3 and increases (`r=0.9909`). The legacy `654/1346` program sizes were not imposed; the corrected exact Ward solution is `883/1117`.

## SHA-256

- Figure PDF: `fe3fbb7d502311f653c6cfcbee42770fdcb2c957c006a2a1e99da318e1e3fc52`
- Figure SVG: `6210997a1769571c843d2532ee04a1f62c1165cbf1e674c7d39d107f328f9011`
- Figure PNG: `f62dc8846ca4f7735f871e7b85c3f89e3d35d54f932760b58d11adde588d4ad0`
- Actual PDF QA render: `43ed87b6fc14d9d5fd739bcb82af1bcd0d5f79abef7feb1023c61a3f573fe676`
- Numerical audit: `5cabe074848a1cb5e67298ad0c2df3f6807e4981467881cc9395f2ef55a2c1b7`
- Plotting script: `9576e71fa6ed9072a2cef8c2cecc8b08bca43697a1ebbbdca25c1b3cad3b1b4b`
- Audit script: `489c3c7dcf1f5aa090e61f20b71410557d9ac0eaba26ce39d74386007d2f75b3`
- Submitted style reference: `49c599ddb076789066a236deecae79dac959199f137a340633b703bdc34bcdf7`
