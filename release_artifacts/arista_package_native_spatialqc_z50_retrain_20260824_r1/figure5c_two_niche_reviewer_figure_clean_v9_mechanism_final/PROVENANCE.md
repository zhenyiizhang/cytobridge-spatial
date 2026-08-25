# Figure provenance

Archived on: `2026-08-24`

Manuscript figure: `ARISTA Figure 5c reviewer-response supplement`

Scientific claim: `The heterogeneous Figure 5c field contains two localized, model-edge-supported ependymoglial interaction domains with distinct candidate pathway and ligand–receptor axes.`

## Files

- Vector figure: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final/FigureS_ARISTA_Figure5c_local_interaction_niches_clean.pdf`
- PNG preview: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final/FigureS_ARISTA_Figure5c_local_interaction_niches_clean.png`
- Plotting script: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final/plot_figure5c_two_niche_reviewer_figure_clean.py`
- Caption source: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final/caption.md`
- Compiled manuscript or SI: `not yet integrated`

## Source paths

- Figure analysis: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_timecourse_v1`
- Accepted package-native run: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1`
- Server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/arista-package-native-spatialqc-z50-20260824-r1`

## Selected experiment

- Local run: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1`
- Server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/arista-package-native-spatialqc-z50-20260824-r1`
- Configuration: `recorded by the selected run and analysis manifest`
- Manifest: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_timecourse_v1/manifest.json`
- Checkpoint and SHA-256: `inherited from the selected package-native run; exact downstream input hashes are recorded in the analysis manifest`
- Training stages and epoch counts: `not changed by this post hoc figure analysis`

## Panel sources

| Panel | Content | Source files | Calculation |
|---|---|---|---|
| a | Spatial interaction niches | `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_timecourse_v1/tables/roi_two_niche_assignments.csv` | Frozen ROI cosine map plus fixed physical connected-component rule |
| b | Organized cell-state interactions | `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_timecourse_v1/tables/two_niche_attention_matched_null.csv`, `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_timecourse_v1/tables/two_niche_t1_celltype_edges.csv` | Observed selected attention versus 9,999 cell-type-matched null regions; selected-edge attention fractions |
| c | Niche-specific repair programs | `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_timecourse_v1/tables/two_niche_lr_pathway_matched_null.csv` | Package-native LR pathway scores versus 1,999 cell-type-matched permutations with BH correction |
| d | Candidate ligand–receptor axes | `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_lr_axes_v1/two_niche_lr_pair_matched_null.csv.gz` | Exact-component pair-level LR scores versus 1,999 cell-type-composition-matched permutations with BH correction across 531 pairs; one pair per panel-c program selected by minimum q, then score, then fold |

## Evaluation protocol

- Initial cells or particles: `frozen 1,454-cell Figure 5c ROI at 5 DPI`
- Evaluation weights: `package-native selected attention`
- Growth handling: `not applicable`
- Time step and diffusion scale: `5-DPI observed component; no new simulation`
- Seeds: `permutation seeds recorded in the pathway and pair-axis analysis manifests`
- Uncertainty summary: `matched-null mean plus or minus s.d.; empirical permutation P values in the caption; the composition-matched null does not preserve spatial geometry`

## Rebuild command

```bash
python /Users/zhenyizhang/Desktop/CytoBridge-ST-1104/scripts/arista_paper_equivalent/plot_figure5c_two_niche_reviewer_figure_clean.py --output-dir /Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/arista_package_native_spatialqc_z50_retrain_20260824_r1/figure5c_two_niche_reviewer_figure_clean_v9_mechanism_final
```

## Interpretation

`The figure supports spatially organized interaction domains with distinct candidate LR programs and cell-state-resolved axes. The cross-species LR database and model-derived attention make these mechanistic hypotheses rather than causal signaling evidence. No pixel-level wound boundary is available.`

## SHA-256

- Figure PDF: `dce13c9a3379410d70a0f7fb9976c0986c615f9f8f7211c6625b907f453596d8`
- Figure PNG: `aa2b56cf191eb66b180b33b71f9467a6df5beb537e7d88fe11ce95fe1c328cbd`
- Plotting script: `4a6a38636ee74c54f6cec7b016954f0c9e9911c670cff1fa23875137d862046a`
