# Figure provenance

Archived on: `2026-09-01`

Manuscript figure: `Supplementary Fig. S41. Five-dataset leave-one-time-point-out benchmark across methods`

Scientific claim: Across the five leave-one-time-point-out benchmarks, CytoBridge has the lowest Sliced-W2 in the largest number of matched comparisons and has lower mean paired error than the comparison methods in most matched settings.

## Files

- Vector figure: `figure/five_dataset_loto_wins_summary.pdf`
- PNG preview: `figure/five_dataset_loto_wins_summary.png`
- Plotting script: `code/plot_five_dataset_loto_wins_summary.py`
- Figure-style helper: `code/cytobridge_figure_style.py`
- Caption source: `CAPTION.md`
- Final plotted table: `source_data/loto_target_stage_means.csv`
- Final plotted table including Linear OT: `source_data/loto_target_stage_means_with_linear_ot.csv`
- Linear OT formal metrics: `source_data/linear_ot_metrics/<dataset>/loto_metrics_long.csv`
- Derived ranks: `source_data/loto_target_stage_ranks.csv`
- Derived lowest-value counts: `source_data/loto_win_counts.csv`
- Derived paired relative errors: `source_data/loto_model_relative_to_cytobridge.csv`

## Selected experiment

- Local accepted run: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/rev03_five_dataset_loto_refreshed_arista_heart_20260827_v2`
- Refreshed server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/rev03-arista-heart-all-method-loto-20260826-r1`
- Refreshed Linear OT server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/rev03-arista-heart-linear-ot-control-20260901-r1`
- Frozen input contract: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/rev03_five_dataset_loto_refreshed_arista_heart_20260827_v2/manifests/input_contract.json`
- Bundle manifest: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/rev03_five_dataset_loto_refreshed_arista_heart_20260827_v2/manifests/bundle_manifest.json`
- Package release commit used for the benchmark: `838fecea96cd8620837be77eb8ddbd716c956a25`
- Benchmark configuration commit: `61f0b550678ed75e706638ceb7638a0818b7e033`

Zebrafish, MOSTA, and AD mouse use the frozen accepted formal evaluations. ARISTA and chicken heart use the accepted retrained evaluations from the refreshed server run. Random interpolation was included in that refreshed nine-method run. Linear OT displacement was recomputed against the same refreshed ARISTA and chicken-heart inputs on 2026-09-01. This figure does not mix the older ARISTA or chicken-heart model or control metrics with the refreshed results.

## Source paths

- Accepted five-dataset metrics: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/output/rev03_five_dataset_loto_refreshed_arista_heart_20260827_v2/source_data/loto_target_stage_means.csv`
- Refreshed Linear OT results: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/rev03-arista-heart-linear-ot-control-20260901-r1`
- Canonical SI source: `/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/supp_text/CytoBridge_Supplementary_Figures.tex`
- Overleaf SI source: `/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/supp_text/overleaf_sync/CytoBridge_Supplementary_Figures.tex`
- Compiled SI: `/Users/zhenyizhang/Desktop/202511/CytoBRIDGE/supp_text/overleaf_sync/main.pdf`
- Response source: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/manuscript_edits/response_rebuild_20260830/final/response_letter_revised_20260831.tex`
- Compiled response: `/Users/zhenyizhang/Desktop/CytoBridge-ST-1104/manuscript_edits/response_rebuild_20260830/final/response_letter_revised_20260831.pdf`

## Panel sources

| Panel | Content | Source files | Calculation |
|---|---|---|---|
| a | Number of lowest-Sliced-W2 results across 33 benchmark settings | `source_data/loto_target_stage_means_with_linear_ot.csv` | Rank mean Sliced-W2 within each dataset, held-out time point, and space; count the lowest value by method |
| b | Zebrafish paired relative error | `source_data/loto_target_stage_means.csv` | `100 × (method / CytoBridge − 1)` within matched rows |
| c | MOSTA paired relative error | same | same |
| d | ARISTA paired relative error | same | same, using refreshed ARISTA results |
| e | AD mouse paired relative error | same | same |
| f | Chicken-heart paired relative error | same | same, using refreshed chicken-heart results |

## Evaluation protocol

- Initial particles: common 5,000-particle roster for every method.
- stVCR output: evaluated at the native growth-enabled output support.
- Evaluation weights: normalized over all predicted particles.
- Target-size resampling: none.
- Primary metric: Sliced-W2 with 1,024 projections.
- Projection repeats: five shared projection seeds; the five values are averaged within each target-space row.
- Figure uncertainty: mean plus or minus standard error across held-out time points. Projection repeats quantify numerical variation and are not treated as independent biological or training replicates.

## Rebuild command

Run from the bundle root:

```bash
python code/plot_five_dataset_loto_wins_summary.py
```

## Interpretation

The figure summarizes the same accepted five-dataset benchmark as the pairwise relative-error figure, adds the Linear OT control to panel a, and shows all supported method outputs by dataset and space. The number of comparisons with the lowest Sliced-W2 is 12 for CytoBridge, 11 for Linear OT displacement, 6 for PASTE, 2 for stVCR, 2 for MOSCOT, and 0 for random interpolation. State-only methods are not assigned joint or spatial values.

## SHA-256

See `CHECKSUMS.sha256` for the finalized hashes of the figure, plotting code, caption, provenance note, manifest, and source tables.
