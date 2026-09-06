---
orphan: true
---

# MOSTA S11–S18: calculations and figures

The [MOSTA supplementary notebook](../../tutorials/paper_figures/mosta_figures.ipynb)
starts from generated cell states and a trained model. It recalculates growth,
gene programs, GO enrichment and LR scores before plotting them.

| Figure | Calculation in the notebook | Result passed to plotting |
| --- | --- | --- |
| S11 | Read the trajectory's spatial coordinates and tissue labels | Cell-state arrays |
| S12 | `cb.tl.evaluate_growth_by_timepoint` | Per-cell growth rates |
| S13 | `cb.tl.summarize_label_composition` | Tissue counts and fractions |
| S14 | Count consecutive labels of the same particle IDs | Tissue-to-tissue transitions |
| S15 | `cb.tl.summarize_temporal_gene_patterns` | Reconstructed expression and Ward gene groups |
| S16 | `clusterProfiler::enrichGO`, using the S15 groups | Newly calculated GO results |
| S17 | `cb.tl.analyze_developmental_wave`, followed by the same GO test | Ordered expression profiles, phases and enrichment results |
| S18 | `cb.tl.compute_timepoint_communications`, then `cb.tl.project_communication_to_lr_timecourses` | LR scores at each simulated time |

The notebook needs `mosta_figure_data.zip`, `mosta_model.zip` and
`mosta_analysis_data.zip`. Its opening section gives the download cells and R
dependencies. To generate the input populations first, use the
[MOSTA trajectory tutorial](../../tutorials/dataset_workflows/mosta.ipynb),
then set `state_dir` and `lineage_file` in the supplementary notebook to that
tutorial's outputs.

## Redraw saved results without recalculating them

For a quick redraw of the archived numerical results, use:

```bash
python -m CytoBridge.datasets mosta --kind mosta_figure_data.zip --output-dir .
python -m reproduction.mosta.figures --data-dir data/mosta/paper --output-dir outputs/mosta_si
```

The second command uses the previously calculated growth, gene-program, GO
and LR tables. It does not run the model or repeat those analyses.

| Figure | Input under `data/mosta/paper/shared/` | Calculation |
| --- | --- | --- |
| S11 | `s4/observed_t0.h5ad`, `generated_states/` | Draw the spatial populations. |
| S12 | `s5_growth/growth_by_cell_fully_generated.csv` | Select brain cells and draw growth on a shared colour scale. |
| S13 | `s6_composition/celltype_composition_fully_generated.csv` | Group saved counts into displayed classes and calculate proportions. |
| S14 | `s7_lineage/fixed_particle_labels.csv.gz` | Count transitions between the same simulated particles. |
| S15–S17 | Gene profiles, program assignments and GO tables | Draw expression programs and enrichment results. |
| S18 | LR score tables in the source archive | Normalize sampled scores and interpolate the displayed time courses. |

The plotting functions are in `reproduction/mosta/figures.py`. Smaller gene
and LR tables are included in the source archive.
