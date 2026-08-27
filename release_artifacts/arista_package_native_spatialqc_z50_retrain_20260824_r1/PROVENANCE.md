# Figure provenance

Archived on: `2026-08-25`

Manuscript figures: `ARISTA Figure 5 and Supplementary Figures S12--S17`

Scientific claim: The corrected package-native ARISTA model reproduces the
paper analyses while retaining the submitted visual grammar, and Figure 5c
identifies two spatially organized, injury-associated interaction niches with
distinct LR programs.

## Files

- Vector Figure 5: `Figure5_fullpage_original_style_v2_final/Figure5_ARISTA_package_native_finalAI_style.pdf`
- Figure 5 PNG preview: `Figure5_fullpage_original_style_v2_final/Figure5_ARISTA_package_native_finalAI_style.png`
- Plotting script: `../../scripts/reviewer_arista_20260824/assemble_figure5_fullpage_from_accepted_panels.py`
- Reviewer figure: `figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final/FigureS_ARISTA_Figure5c_local_interaction_niches_clean.pdf`
- Reviewer caption: `figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final/caption.md`
- Reviewer correspondence: `figure5c_reviewer_correspondence_v1/Figure5c_reviewer_correspondence_and_biological_interpretation.md`

## Selected experiment

- Local run: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1`
- Server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/arista-package-native-spatialqc-z50-20260824-r1`
- Configuration: `main_run/provenance/config.yaml`
- Training manifest: `main_run/provenance/training_run_summary.json`
- Final checkpoint: `main_run/training/Finetune/best_model.pth`, SHA-256 `8e5e085590ade8902111a410ac75aaea37fcd4960e6164e663b48a113b94f043`
- Final score model: `main_run/training/Score_Refine/score_model.pth`, SHA-256 `1c7a8e82045cf4e81209b17959b80a7ac8b07399412bb54f1774e64ba6b23fea`
- Training stages and epoch counts: Pretrain 100, Refine 100,
  Init_interaction 50, Train_Score 2,001, Finetune 1,000, Score_Refine 2,001.

## Source paths

- Release package source: repository root at the release commit containing this
  archive.
- Canonical figure scripts: `../../scripts/reviewer_arista_20260824/`.
- Accepted transferred run: `main_run/`.
- Complete server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/arista-package-native-spatialqc-z50-20260824-r1`.
- The five dense Figure 5 panel payloads are intentionally embedded as
  high-resolution raster layers in the Illustrator-derived A4 page. Text,
  headings, and page geometry remain vector, and pixel-composition QA passed at
  1x, 2x, and 4x rendering scales.

## Panel sources

| Panel | Content | Source files | Calculation |
|---|---|---|---|
| 5a | Spatiotemporal communication | `main_run/downstream/slice_data/`, `main_run/downstream/communication/` | Fresh generated slices and selected sparse-attention edges rendered with the submitted five-slice focus-anchor grammar and audited foreground-z lift |
| 5b | Generated spatial state | `main_run/downstream/slice_data/time_0p5.h5ad` | All 7,798 generated cells retained and recolored through cell-type labels into the submitted palette |
| 5c | Spatial velocity and direction cosine | `main_run/downstream/velocity/velocity_components.npz`, `main_run/downstream/slice_data/time_1.h5ad` | Full and interaction first-two-dimensional velocities are mapped to the spatial basis and compared per cell by cosine, matching the accepted original-style v2 calculation |
| 5d | Gene velocity | `main_run/downstream/velocity/velocity_components.npz`, `main_run/preprocess/aligned_cell_identity.csv` | Corrected full velocity projected through the fresh raw package-native PCA and rendered with the historical scVelo stream style |
| 5e | Growth and interaction | `main_run/downstream/growth/`, `main_run/downstream/communication/` | Fresh package-native cell and cell-type summaries in the submitted layout |
| S12--S14 | Interpolation, growth, lineage and composition | `main_run/downstream/slice_data/`, `main_run/downstream/growth/`, `main_run/downstream/composition/` | Generated and observed slices use `spatial_warp_k=1`; lineage/composition use persistent fixed-particle identities |
| S15 | Gene dynamics | `main_run/downstream/gene_dynamics/` | Fresh reconstruction-derived gene dynamics |
| S16 | LR temporal patterns | `main_run/downstream/ligand_receptor/pair_timecourse.csv`, `S16_lr_kmeans_recluster_v1/` | Row-wise min-max normalization followed by deterministic k-means (`k=2`, k-means++, 100 initializations, seed 0); the two patterns contain 217 and 314 pairs |
| S17 | Representative LR trajectories | `S16_lr_kmeans_recluster_v1/`, `S17_package_native_balanced25_oldstyle_v1/tables/` | The 25 profiles nearest each pattern mean are shown, giving 50 displayed pairs and 450 time-course rows; the balanced display is not a prevalence estimate |

## Evaluation protocol

- Training cells: 46,199 observed cells after removal of 10 label-blind,
  clearly detached spatial outliers.
- Feature space: package-selected 2,000 HVGs plus required LR features,
  2,246 genes total and 50 PCs.
- Interaction graph: cutoff `0.03154105148551745` and validation-selected edge
  threshold `0.6523735523223877`.
- Fixed-particle analyses: persistent cell identities and the package-recorded
  fixed cohort totals; no growth-dependent resampling is used for composition.
- Figure 5c reviewer null: physical domains are fixed before testing;
  communication attention is compared with 9,999 cell-type-matched null
  regions, and LR pathway scores with 1,999 cell-type-matched permutations and
  BH correction.
- Random seed: 42 for the fitted package workflow; analysis-specific seeds are
  recorded in the reviewer manifests.

## Rebuild command

The exact frozen Figure 5 command is recorded in its manifest:

```bash
python scripts/reviewer_arista_20260824/assemble_figure5_fullpage_from_accepted_panels.py --output-dir <NEW_EMPTY_OUTPUT_DIR>
```

The submitted `Arista.ai` template and accepted panel paths must be supplied as
documented by the script's CLI and manifest. The package workflow configuration,
run log, and selected checkpoints are retained for scientific reruns.

## Interpretation

The archive supports replacement of the ARISTA paper figures using corrected
calculations. The hierarchical S16/S17 pages retained in
`S15_S17_package_native_strict_oldstyle_v1/` are superseded diagnostics; the
accepted LR pattern pages are the deterministic k-means S16 and balanced S17
listed above. It does not claim that spatial interaction is uniformly
aligned with the injury contour. The Figure 5c follow-up instead supports two
localized organized interaction niches with distinct repair-associated LR
programs. The analysis is observational and model-derived, so it supports a
spatially structured communication mechanism but not causal proof of injury
repair.

## SHA-256

- Figure 5 PDF: `b8bdf55db1a7bf7801a2ab4dbd33e9a391673cf592a10c55ba2dc98a771cf542`
- Figure 5 PNG: `9a8aa258fe4ca57930c9e15fb9f2058eea01f83577f0f67c5011760451abd35d`
- Figure 5 assembly script: `40c164dd942f7312f5107ee21af95c332808c600c1975c1e5f5792d776899a54`
