# Supplementary Figure S44

Updated on 2026-09-04 to include SpaTrack in the five-dataset comparison.

The figure compares held-out reconstruction error in matched datasets, stages, and output spaces. CytoBridge has lower Sliced-W2 than the PCA-based SpaTrack comparison in 30 of 33 settings. All 33 results are included, including the three settings with lower SpaTrack error.

## Source paths

- Final figure: `figure/five_dataset_loto_wins_summary.pdf` and `.png`.
- Plotting code: `code/plot_five_dataset_loto_wins_summary.py`.
- SpaTrack fitting and scoring code: `code/spatrack/run_benchmark.py`.
- Unmodified SpaTrack numerical source: `code/spatrack/official_spatrack_utils.py`, from commit `1cc5edec3699f7d8e29663ce3bb0c02cad5600db` of <https://github.com/yzf072/spaTrack>.
- Numerical results for plotting: `source_data/spatrack_metrics.csv`, plus the existing target-stage means and Linear OT metrics.
- Per-target fitting records: `source_data/spatrack_fit_summaries.json`.
- All method counts: `source_data/loto_win_counts.csv`.
- Actual couplings and predictions: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/experiments/spatrack_benchmark_20260904/server_results`.
- Server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/spatrack-pca-five-dataset-20260904-r1`.
- Server inputs for ARISTA and chicken heart: `runs/rev03-arista-heart-all-method-loto-20260826-r1/<dataset>/inputs` under the same server project.
- Server inputs for Zebrafish, MOSTA, and AD mouse: `runs/corrected-benchmark-20260813-matched-3c87a3e-c4f8e203-ff55c7a-r4/<dataset>/inputs`.

## Calculation

PCA/Euclidean distance replaces the default multi-time-point expression/KL cost. The maximum training-anchor distance sets the PCA cost scale. Spatial distances are normalized by each section's maximum. The official affinity-derived marginals and spatial transport solver are retained. Parameters are identical across all five datasets: spatial weight 0.1, entropic regularization 0.01, balanced transport, and 10 outer iterations.

The coupling is row-normalized and converted to a barycentric target. Linear interpolation in benchmark time supplies the held-out stage. Each comparison uses the existing deterministic 5,000-cell starting roster. The 800-cell anchor selection is the same as for the other static adapters. No growth prediction or target-count resampling is used.

Sliced-W2 uses the original training-fitted transformation, 1,024 directions, and five shared projection seeds. The 165 repeat-level rows cover 11 targets and three output spaces. The retained representation is transductive and frozen across methods. PCA and alignment were not refitted per fold. Fitting never reads the target-stage arrays.

## Panel sources

| Panels | Data | Calculation |
|---|---|---|
| a | `loto_target_stage_means_with_spatrack.csv` | Count the lowest mean Sliced-W2 within each of 33 matched settings |
| b–f | Same table | `100 * (method / CytoBridge - 1)` for each matched target and space |

Open circles show held-out stages. Diamonds show their mean and bars show standard error across stages. Projection repeats quantify numerical variation, not biological or training replicates. Overall averages the three output-space ratios within each target before summarizing across targets.

The first-place counts remain CytoBridge 12, Linear OT 11, PASTE 6, stVCR 2, and MOSCOT 2. SpaTrack has zero first-place counts. As in the previous figure, panel a displays methods with a nonzero first-place count and the random-interpolation control. Every method's count is in the accompanying CSV.

## Rebuild

From this directory:

```bash
python code/plot_five_dataset_loto_wins_summary.py
```

For a fresh SpaTrack calculation from prepared benchmark inputs, see `code/spatrack/README.md`. The notebook `docs/tutorials/paper_figures/loto_benchmark_summary.ipynb` reruns the calculations and plot from these numerical files.

## Documents

- SI entry: `/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/supp_text/overleaf_sync/main.tex`.
- SI figure label: `fig:five-dataset-loto-wins-summary`.
- SI source: `CytoBridge_Supplementary_Figures.tex`, with new text inside `\revision{}`.
- Response: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/manuscript_edits/response_rebuild_20260830/final/response_letter_revised_20260831.tex`.
- All three existing S44 appearances in the response are retained and updated.

## SHA-256

The final figure and plotting-code hashes are recorded in `CHECKSUMS.sha256`. Previous result bundles were not overwritten.
