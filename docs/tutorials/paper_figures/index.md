# Paper figures

Choose a dataset or analysis below. The dataset notebooks calculate dynamics
from a trained model. The figure pages show the panel calculations and plotting
code, including analyses that start from saved intermediate tables.

Start with the [installation instructions](../../installation.md), which set up the
code and data folders for JupyterLab. Small numerical tables are included with
the code. A notebook that needs a larger download names it in its first cell.

## Main figures and dataset analyses

| Dataset | Paper figures | Start here |
| --- | --- | --- |
| AGIST | Figure 2, S2–S3 | [Simulation comparisons](agist_figures.ipynb), [Figure 2](main_figure_2.ipynb) |
| Weinreb and scNT | S4–S5 | [Expression-state dynamics](nonspatial_figures.ipynb) |
| Chicken heart | Figure 3, S7–S10 | [Growth and daily populations](../dataset_workflows/chicken_heart.ipynb), [lineage and velocity](chicken_heart_daily.ipynb), [alignment](chicken_heart_alignment.md) |
| MOSTA | Figure 4, S11–S18 | [Figure 4](main_figure_4.ipynb), [supplementary figures](mosta_figures.ipynb) |
| ARISTA | Figure 5, S19–S25 | [Figure 5](main_figure_5.ipynb), [S19–S24](arista_figures.ipynb), [local interaction domains](arista_local_domains.ipynb) |
| AD mouse | Figure 6, S26–S30 | [Population, interaction and perturbation analyses](admouse_figures.md) |
| Zebrafish | S31–S40 | [S31–S38](zebrafish_si_s31_s38.ipynb), [attention comparisons](zebrafish_attention.ipynb), [stability across seeds and settings](zebrafish_decomposition_stability.md) |

The MOSTA supplementary notebook recalculates its downstream analyses from
the saved cell states and trained model. To simulate a new trajectory, or
calculate the other datasets' results from a model, use these guides:

- [AGIST simulations and Wasserstein distances](agist_simulations.md)
- [Weinreb and scNT preprocessing, training and evaluation](../../nonspatial_workflows.md)
- [MOSTA population simulation and growth](../dataset_workflows/mosta.ipynb)
- [ARISTA population simulation](arista_populations.md) and [velocity/growth calculations](arista_model_fields.md)
- [Zebrafish S31–S38: training, simulations, analyses and figures](zebrafish_si_s31_s38.ipynb)

Figure 2a–d currently uses the assembled original panels. Its panel e is drawn
from numerical results. Figure 3's calculation notebooks generate the individual
panels, but do not yet assemble the manuscript page. Figures 1 and S47 are model
schematics, rather than computational results.

## Comparisons and model settings

| Figure | Analysis |
| --- | --- |
| S6 | [Cell-type classifier smoothing](classifier_smoothing.ipynb) |
| S41 | [LR-complex aggregation](lr_complex_aggregation.ipynb) |
| S42 | [LR-prior and interaction ablations](interaction_ablation.ipynb) |
| S43 | [COMMOT, CellAgentChat and NicheNet comparisons](spatial_communication.ipynb) |
| S44 | [Benchmark summary](loto_benchmark_summary.ipynb), including the [SpaTrack calculation](spatrack_benchmark.md) |
| S45 | [Five-dataset benchmark](loto_benchmark.ipynb) |
| S46 | [Training histories](training_histories.ipynb) |
| Table 2 | [Training time and memory](compute_cost.ipynb) |

The supplementary notebooks use the current SI plotting programs. For example,
this command redraws S4, S5 and S41 from numerical inputs. S4/S5 use
black streamlines, and S41 calculates top-100 overlap from the full LR rankings.

```bash
python -m reproduction.paper_figures --figures 4 5 41 --output-dir outputs/paper_figures
```

See [Zebrafish videos](../zebrafish_videos.md) for trajectory animation.

## Run several notebooks

To run the notebooks that need only the included numerical inputs:

```bash
python scripts/execute_paper_notebooks.py --bundled-only --output-dir notebook_runs --report notebook_runs/report.json
```

To include the larger MOSTA and ARISTA notebooks, set the project folder
containing the downloads. Figure 4 uses a CUDA GPU. MOSTA S11–S18 also needs
its model, aligned data and the R packages listed at the start of that notebook.

```bash
python scripts/execute_paper_notebooks.py --project-dir paper_run --output-dir notebook_runs_full --report notebook_runs_full/report.json
```

With `--project-dir`, figures are saved under `paper_run/outputs/`. Without it,
each notebook writes inside its own folder under `--output-dir`. This command
runs analyses and plotting, not model training.

```{toctree}
:hidden:
:maxdepth: 1

Figure 2: AGIST <main_figure_2>
AGIST attention strength <agist_attention_recovery>
AGIST model simulations <agist_simulations>
Chicken-heart lineage and velocity <chicken_heart_daily>
Figure 4: MOSTA <main_figure_4>
Figure 5: ARISTA <main_figure_5>
S2–S3: Simulations <agist_figures>
S4–S5: Non-spatial data <nonspatial_figures>
S6: Classifier smoothing <classifier_smoothing>
S7–S8: Heart alignment <chicken_heart_alignment>
S11–S18: MOSTA <mosta_figures>
S19–S24: ARISTA <arista_figures>
ARISTA model populations <arista_populations>
ARISTA model fields <arista_model_fields>
S25: ARISTA local domains <arista_local_domains>
Figure 6 and S26–S30: AD mouse <admouse_figures>
S31–S38: Zebrafish <zebrafish_si_s31_s38>
S39: Attention comparisons <zebrafish_attention>
S40: Decomposition stability <zebrafish_decomposition_stability>
S41: LR complexes <lr_complex_aggregation>
S42: Interaction ablations <interaction_ablation>
S43: Spatial communication <spatial_communication>
S44: Benchmark summary <loto_benchmark_summary>
SpaTrack comparison <spatrack_benchmark>
S45: Benchmark details <loto_benchmark>
S46: Training histories <training_histories>
Training time and memory <compute_cost>
../zebrafish_videos
```
