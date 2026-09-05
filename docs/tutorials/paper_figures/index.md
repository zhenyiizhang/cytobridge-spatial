# Paper figures

Choose a figure below. Each notebook states what its code actually does.

| Starting material | What the notebook does |
| --- | --- |
| Included CSV/NPZ results | Calculates summaries and draws new plots |
| Existing vector panels | Assembles the panels into a page |
| A completed PDF/PNG page | Displays or copies the page |

Figures 4 and 5, MOSTA S11–S18 and ARISTA S19–S24 draw their panels from
cell-state arrays and numerical tables. Figure 2a–d is still being connected
to its numerical inputs.

## Draw from the included results

For example, redraw S4–S5 from their numerical inputs in a source checkout:

```bash
python scripts/execute_paper_notebooks.py \
  --notebook nonspatial_figures \
  --output-dir notebook_runs
```

The plots are written to
`notebook_runs/nonspatial_figures/outputs/nonspatial_figures/`.
Other notebooks use the same
`<output-dir>/<notebook-name>/outputs/<figure-folder>/` structure.

The displayed outputs on this site let you inspect the results before running
the notebook. No training is started by these figure notebooks.

## Recalculate from a model

Use [Train a model](../../training.md) to fit your data, then continue with
the [analysis tutorials](../dataset_workflows/index.md).
Each figure notebook links to a separate guide to its earlier calculations
and input files.

A new model run is not automatically used by a paper redraw command. Several
figures still need code to convert a new run into their exact plotting inputs,
including S4–S5, S31–S36 and S38. S37 has its own simulation and plotting tutorial. The
[figure-by-figure inventory](../../paper_reproduction.md) records those gaps.

## Check the available notebooks

Run all figure notebooks in new output directories and save an execution report:

```bash
python scripts/execute_paper_notebooks.py \
  --output-dir figure_check \
  --report figure_check/report.json
```

The report distinguishes numerical plotting, panel assembly, and page copying.
It tests the available figure notebooks, not training or the complete
raw-data-to-figure analysis.

## Main figures

- [Main Figure 2: AGIST benchmark — redraw panel e and assemble the page](main_figure_2.ipynb)
- [Chicken-heart development: trajectories, cell types, and growth](chicken_heart_daily.md)
- [Main Figure 4: MOSTA](main_figure_4.ipynb)
- [Main Figure 5: ARISTA](main_figure_5.ipynb)
- [Main Figure 6: AD mouse](admouse_figures.md)

## Dataset supplementary figures

- [AGIST, Supplementary Figures S2–S3](agist_figures.ipynb)
- [Weinreb and scNT, Supplementary Figures S4–S5](nonspatial_figures.ipynb)
- [Chicken-heart alignment sensitivity, Supplementary Figures S7–S8](chicken_heart_alignment.md)
- [Chicken-heart growth and velocity analyses](chicken_heart_daily.md)
- [MOSTA, Supplementary Figures S11–S18](mosta_figures.ipynb)
- [ARISTA, Supplementary Figures S19–S24](arista_figures.ipynb)
- [Generate the ARISTA paper populations](arista_populations.md)
- [ARISTA local interaction domains, Supplementary Figure S25](arista_local_domains.ipynb)
- [AD mouse, Supplementary Figures S26–S30](admouse_figures.md)
- [Zebrafish, Supplementary Figures S31–S38](zebrafish_si_s31_s38.ipynb)
- [Recalculate daughter-cell perturbations, Supplementary Figure S37](zebrafish_daughter_noise.md)
- [Zebrafish attention and control comparisons, Supplementary Figure S39](zebrafish_attention.ipynb)
- [Zebrafish decomposition stability, Supplementary Figure S40](zebrafish_decomposition_stability.md)

## Additional analyses

- [Classifier smoothing, Supplementary Figure S6](classifier_smoothing.ipynb)
- [LR-complex aggregation, Supplementary Figure S41](lr_complex_aggregation.ipynb)
- [LR-prior and interaction ablations, Supplementary Figure S42](interaction_ablation.ipynb)
- Supplementary Figure S43 compares spatial communication summaries with COMMOT, CellAgentChat, and NicheNet. Its plotting program and numerical tables are in `release_artifacts/spatial_communication_comparison_s43_20260903`.
- [Cross-dataset benchmark summary, Supplementary Figure S44](loto_benchmark_summary.ipynb)
- [Run the SpaTrack comparison used in S44](spatrack_benchmark.md)
- [Five-dataset benchmark, Supplementary Figure S45](loto_benchmark.ipynb)
- [Training histories, Supplementary Figure S46](training_histories.ipynb)
- Supplementary Figure S47 is the analysis-workflow schematic supplied with the manuscript.
- [Training time and memory](compute_cost.ipynb)

## Videos

- [Zebrafish trajectory videos](../zebrafish_videos.md)

```{toctree}
:hidden:
:maxdepth: 1
arista_populations
zebrafish_daughter_noise

Figure 2: AGIST <main_figure_2>
Chicken-heart daily plots <chicken_heart_daily>
Figure 4: MOSTA <main_figure_4>
Figure 5: ARISTA <main_figure_5>
S2–S3: AGIST <agist_figures>
S4–S5: Non-spatial data <nonspatial_figures>
S6: Classifier smoothing <classifier_smoothing>
S7–S8: Heart alignment <chicken_heart_alignment>
S11–S18: MOSTA <mosta_figures>
S19–S24: ARISTA <arista_figures>
S25: ARISTA local domains <arista_local_domains>
Figure 6 and S26–S30: AD mouse <admouse_figures>
S31–S38: Zebrafish <zebrafish_si_s31_s38>
S39: Attention comparisons <zebrafish_attention>
S40: Decomposition stability <zebrafish_decomposition_stability>
S41: LR complexes <lr_complex_aggregation>
S42: Interaction ablations <interaction_ablation>
S44: Benchmark summary <loto_benchmark_summary>
SpaTrack comparison <spatrack_benchmark>
S45: Benchmark details <loto_benchmark>
S46: Training histories <training_histories>
Training time and memory <compute_cost>
../zebrafish_videos
```
