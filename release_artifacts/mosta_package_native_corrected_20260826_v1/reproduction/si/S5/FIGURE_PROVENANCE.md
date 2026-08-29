# Figure provenance

Archived on: `2026-08-25`

Manuscript figure: `Supplementary Figure S5. Brain growth-rate maps across observed and interpolated time points in the MOSTA dataset`

Scientific claim: Package-predicted local Brain growth decreases smoothly along one fully generated global-t0 MOSTA developmental trajectory.

## Source paths

- Accepted local calculation bundle: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`
- Accepted server calculation bundle: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-paper-figures/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`
- Submitted style notebook: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/source/mosta-brain-cell-composition-review.ipynb`
- Submitted SI PDF: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf`

## Files

- Hybrid-vector PDF: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/figures/Figure_S5_MOSTA_latest_package_global_t0_growth_exact_submitted_style.pdf`
- Hybrid-vector SVG: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/figures/Figure_S5_MOSTA_latest_package_global_t0_growth_exact_submitted_style.svg`
- PNG preview: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/figures/Figure_S5_MOSTA_latest_package_global_t0_growth_exact_submitted_style.png`
- Plotting script: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/source/render_s5_corrected_exact_submitted_style.py`
- Numerical audit: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/tables/s5_numerical_audit.json`
- Spatial-label residual audit: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/tables/s5_spatial_label_residual_audit.json`
- Caption and style authority: `/Users/zhenyizhang/Desktop/202511/nbme预投稿/投稿/投稿修改/si.pdf`, page 31

## Selected experiment

- Local numerical run: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`
- Server numerical run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/mosta-paper-figures/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1`
- Current package: `/data/cytobridge/projects/CytoBridge-ST-1104/software/cytobridge-release-2b3c79e-runtime`, commit `2b3c79eff3face7c4dd33de24d45384b9dbd8a84`
- Calculation script: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/source/server_compute_mosta_si_shared.py`
- Numerical manifest: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1/SHA256SUMS.txt`
- Numerical summary: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1/summary.json`
- Finetune checkpoint SHA-256: `d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5`
- Score_Refine checkpoint SHA-256: `d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a`
- Classifier cache SHA-256: `f938575c145baa9002de695c39a8637f57d6cb3a06ccfaf4d18b707ca962a7e0`; spatial label refinement `k=10`
- Training stages: accepted `Finetune` and `Score_Refine` checkpoints; no figure-specific retraining

## Panel sources

| Panel | Content | Source files | Calculation |
|---|---|---|---|
| S5 | Twelve Brain growth maps at t=0.00-3.00, omitting t=1.50 from display only | `s5_growth/growth_by_cell_fully_generated.csv`, `s5_growth/growth_contract.json`, and thirteen `generated_states/time_*.h5ad` files | `CytoBridge.tl.evaluate_growth_by_timepoint`; filter exact package `Annotation == "Brain"`; use one global 5th-95th percentile scale calculated from all thirteen Brain slices |

## Evaluation protocol

- Initial particles: 50,000 from t=0
- Trajectory: one fully generated global-t0 split-SDE trajectory; no restart from later observed stages
- Time grid: 0.00 to 3.00 in 0.25 increments for calculation; t=1.50 removed from display only
- SDE time step and diffusion: `dt=0.05`, `sigma=0.03`
- Growth handling: split-population dynamics with `growth_alpha=1`; displayed values are direct package `g_net(t, x)` predictions
- Seeds: workflow seed 42; split-population stream 43; split interaction grouping stream 10043
- Spatial handling: native pre-warp coordinates; no rotation, stretch, or spatial warp
- Scale: shared raw-value limits `[-0.025904556736350056, 0.539695218205452]`
- Uncertainty: none; panels show individual classifier-predicted Brain cells

## Rebuild command

```bash
MPLCONFIGDIR=/tmp/mosta_s5_mpl_20260825_v1 python /Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/source/render_s5_corrected_exact_submitted_style.py \
  --numerical-root /Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_shared_compute_20260825_v1/server_download/si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1 \
  --numerical-audit /Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/tables/s5_numerical_audit.json \
  --style-notebook /Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/source/mosta-brain-cell-composition-review.ipynb \
  --style-reference-svg /Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/source/old_formal_growth_style_reference.svg \
  --rejected-mixed-table /Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/input/growth_by_cell.csv \
  --output-dir /Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/mosta_si_s5_growth_20260825_v1/figures
```

## Interpretation

Mean Brain growth decreases from 0.503 at t=0 to 0.038 at t=3, with no temporal reversal in the mean or median. A spatial-label sensitivity audit found that 1.73% of final Brain labels have fewer than five Brain labels among their final ten nearest neighbors and 0.143% are self-only. These are retained because the current package performs one simultaneous k=10 vote and the submitted panel plots every classifier-predicted Brain cell. Excluding low-support labels only as a diagnostic changes the time-point mean by at most 0.00142 and leaves the monotonic result unchanged. The dense scatter layers are intentionally rasterized at 300 dpi and DejaVu Sans is retained because both choices come from the submitted style code; text, axes, titles, spines, and colorbar remain vector elements.

## SHA-256

- Figure PDF: `4fbe2b269ee144e00dda328548875cde823b519b9a7b8a18b108c56281a79ebf`
- Figure SVG: `f564ad5670f99a705295f40ce1ba9a58d09883b5581b969e5851b5c8baf1b866`
- Figure PNG: `8048b47048648fa5d367f396f7ad6e8952cd3e3620c555c2a7a71e3ca476a5b9`
- Plotting script: `782fabc5bb2b0a31f3474ac27e220767a8f2156fec09fc7aa714e39ef32148c1`
