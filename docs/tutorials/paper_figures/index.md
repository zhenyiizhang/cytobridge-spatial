# Paper figures

Choose a main or supplementary figure below. Its tutorial shows which data
to download, how to calculate the results, and how to draw the panels.
Start with [installation](../../installation.md) if you are new to CytoBridge.

## Dataset analyses

| Dataset | Figures | Guide |
| --- | --- | --- |
| AGIST | Figure 2, S2–S3 | [Evaluate the model and draw the simulation results](main_figure_2.ipynb) |
| Finite-range simulation | S4 | [Simulate and evaluate cell interactions](agist_figures.ipynb) |
| LARRY and scNT | S5–S6 | [Compare models with and without interactions](nonspatial_figures.ipynb) |
| Chicken heart | Figure 3, S12–S15 | [Start with the model](../dataset_workflows/chicken_heart.ipynb), then [lineage and velocity](chicken_heart_daily.ipynb) and [alignment](chicken_heart_alignment.md) |
| MOSTA | Figure 4, S16–S23 | [Main figure](main_figure_4.ipynb) and [supplementary analyses](mosta_figures.ipynb) |
| ARTISTA | Figure 5, S24–S30 | [Main figure](main_figure_5.ipynb), [supplementary analyses](arista_figures.ipynb), and [local interaction domains](arista_local_domains.ipynb) |
| AD mouse | Figure 6, S31–S35 | [Population, gene programs and perturbations](admouse_figures.md) |
| Zebrafish | S36–S42 | [Development and virtual tissue removal](zebrafish_si_s31_s38.ipynb), [attention analysis](zebrafish_attention.ipynb) |

Figure 1, Supplementary Figure S1 and the illustration in Figure 2a are supplied
as vector artwork. The chicken-heart guides generate the numerical panels of
Figure 3 separately.

## Comparisons and model settings

| Figure | Guide |
| --- | --- |
| S7–S8 | [Benchmark summary](loto_benchmark_summary.ipynb) and [method comparisons](loto_benchmark.ipynb) |
| S9 | [Ligand–receptor prior and interaction ablations](interaction_ablation.ipynb) |
| S10 | [Training seeds and model settings](zebrafish_decomposition_stability.md) |
| S11 | [COMMOT, CellAgentChat and NicheNet comparisons](spatial_communication.ipynb) |
| S43 | [Training histories](training_histories.ipynb) |
| S44–S45 | [Loss weights and simulation noise](zebrafish_si_s31_s38.ipynb) |
| S46 | [Cell-type classification](classifier_smoothing.ipynb) |
| S47 | [Ligand–receptor complex aggregation](lr_complex_aggregation.ipynb) |
| Table 3 | [Training time and memory](compute_cost.ipynb) |
| Videos 1–5 | [Simulate cell populations and render the videos](../supplementary_videos.md) |

For example, redraw S5, S6 and S47 from the numerical inputs included with the
code:

```bash
python -m reproduction.current_figures --figures 5 6 47 --output-dir outputs/paper_figures
```

This command redraws the saved numerical results. To calculate them from a
trained model, follow the dataset notebooks. The [figure index](../../data/paper_reproduction_registry.csv)
lists the script and inputs for each figure.

## Run the notebooks

Run the notebooks whose numerical inputs are included with the code:

```bash
python scripts/execute_paper_notebooks.py --bundled-only --output-dir notebook_runs --report notebook_runs/report.json
```

For the dataset calculations, select the project directory containing your
downloads. Model evaluation uses a CUDA GPU. Gene enrichment and NicheNet
analyses also require the R packages specified in their guides.

```bash
python scripts/execute_paper_notebooks.py --project-dir paper_run --output-dir notebook_runs_full --report notebook_runs_full/report.json
```

New results go to `paper_run/outputs/`. These commands evaluate the supplied
models and draw figures. Training is described in the corresponding dataset
guides and writes a separate model directory.

```{toctree}
:hidden:
:maxdepth: 1

Figure 2 and S2–S3: AGIST <main_figure_2>
AGIST attention <agist_attention_recovery>
AGIST simulation and training <agist_simulations>
S4: Finite-range simulation <agist_figures>
S5–S6: LARRY and scNT <nonspatial_figures>
S5: Three-model comparison <weinreb_s4>
Chicken-heart lineage and velocity <chicken_heart_daily>
S12–S13: Heart alignment <chicken_heart_alignment>
Figure 4: MOSTA <main_figure_4>
S16–S23: MOSTA <mosta_figures>
Figure 5: ARTISTA <main_figure_5>
S24–S30: ARTISTA <arista_figures>
ARTISTA populations <arista_populations>
ARTISTA model fields <arista_model_fields>
S26: Local interaction domains <arista_local_domains>
Figure 6 and S31–S35: AD mouse <admouse_figures>
S36–S42: Zebrafish <zebrafish_si_s31_s38>
S38: Attention comparisons <zebrafish_attention>
S10: Training seeds and settings <zebrafish_decomposition_stability>
S47: LR complexes <lr_complex_aggregation>
S9: Interaction ablations <interaction_ablation>
S11: Spatial communication <spatial_communication>
S7: Benchmark summary <loto_benchmark_summary>
SpaTrack <spatrack_benchmark>
S8: Benchmark details <loto_benchmark>
S43: Training histories <training_histories>
S46: Classifier smoothing <classifier_smoothing>
Table 3: Training time and memory <compute_cost>
../zebrafish_videos
```
