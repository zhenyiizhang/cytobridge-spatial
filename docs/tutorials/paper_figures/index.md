# Paper figure notebooks

Every notebook begins with the files passed between calculations. Read each
step from top to bottom: code or command, files read, files written, and the
next consumer. The final step names the figure command. The command-line
version is:

```bash
cytobridge figure list
cytobridge figure explain nonspatial
cytobridge figure explain zebrafish-si
```

## What the commands start from

| Paper entry | Mode | Starts from | Figure command |
|---|---|---|---|
| Main Figure 2 | result-summary redraw and assembly | panel-e tables plus frozen panels a–d | `cytobridge figure main-figure-2 --output-dir outputs/main_figure_2` |
| Main Figure 3 | package downstream; final assembly gap | aligned chicken-heart H5AD and retrained model | [chicken-heart dataset page](../dataset_workflows/chicken_heart.ipynb) |
| Main Figure 4 | assembly | external MOSTA vector panels | `cytobridge figure main-figure-4 --results-dir <mosta-release> --output-dir outputs/main_figure_4` |
| Main Figure 5 | reference export | packaged page raster and panel index | `cytobridge figure main-figure-5-reference --output-dir outputs/main_figure_5` |
| Main Figure 6 | related formal builder; exact manuscript-page mapping gap | AD continuous-t0 formal outputs | [AD mouse dataset page](../dataset_workflows/admouse.ipynb) |
| S2–S3 | numerical redraw | packaged cell-level summaries, arrays, and tables | `cytobridge figure agist --output-dir outputs/agist` |
| S4–S5 | numerical redraw | packaged Weinreb and scNT arrays and tables | `cytobridge figure nonspatial --output-dir outputs/nonspatial` |
| S6 | numerical redraw | packaged classifier sensitivity tables | `cytobridge figure classifier-smoothing --output-dir outputs/classifier_smoothing` |
| S7–S10 | standard calculations available; exact manuscript-PNG assembly gap | chicken-heart alignment, growth, and velocity outputs | [chicken-heart dataset page](../dataset_workflows/chicken_heart.ipynb) |
| S11–S18 | reference export | external MOSTA vector pages | `cytobridge figure mosta-reference-pages --results-dir <mosta-release> --output-dir outputs/mosta_si` |
| S23–S24 | numerical redraw | all 531 packaged ARISTA LR profiles | `cytobridge figure arista-lr --output-dir outputs/arista_lr` |
| S25 | numerical redraw | packaged paired LR aggregation scores | `cytobridge figure lr-complex --output-dir outputs/lr_complex` |
| S26–S28 | related formal builder; exact manuscript-PDF mapping gap | AD downstream, gene-program, and GO tables | [AD mouse dataset page](../dataset_workflows/admouse.ipynb) |
| S29–S30 | provenance gap | manuscript PDFs; candidate NicheNet and perturbation code are not claimed as exact generators | [AD mouse dataset page](../dataset_workflows/admouse.ipynb) |
| S31–S38 | numerical redraw | packaged zebrafish panel arrays and tables | `cytobridge figure zebrafish-si --output-dir outputs/zebrafish_si` |
| S39 | numerical redraw | packaged paired error tables | `cytobridge figure interaction-evidence --output-dir outputs/interaction_evidence` |
| S40 | numerical redraw | packaged target-level LOTO results | `cytobridge figure loto-benchmark --output-dir outputs/loto_benchmark` |
| S41 | numerical redraw | packaged per-epoch histories | `cytobridge figure training-histories --output-dir outputs/training_histories` |
| S42 | numerical redraw | packaged ARISTA domain and null tables | `cytobridge figure arista-local-domains --output-dir outputs/arista_local_domains` |
| S43 | numerical redraw | packaged zebrafish attention tables | `cytobridge figure zebrafish-attention --output-dir outputs/zebrafish_attention` |
| Supplementary Table 2 | table formatting | packaged runtime and memory measurements | `cytobridge figure compute-cost --output-dir outputs/compute_cost` |

In a numerical-redraw notebook, a displayed PNG is the output of the plotting
cell immediately above it. Loading a released CSV or NPZ is only the final
redraw step; the step list at the top names the producer of that file. Assembly and
reference-export pages are the only ones that use a PDF or image as a source.
The dataset notebooks contain preprocessing, training, downstream analysis,
and the dataset-specific continuation into the manuscript. A paper command
consumes a new run only when its route says that the output schema matches.

ARISTA S19–S22 remain reference pages. The same ARISTA notebook recalculates
and redraws S23–S24 from the complete LR table.

## Main figures

- [Main Figure 2: AGIST benchmark](main_figure_2.ipynb) (panel-e summary redraw and page assembly)
- [Main Figure 3: chicken heart](../dataset_workflows/chicken_heart.ipynb) (calculation chain and final-assembly gap)
- [Main Figure 4: MOSTA](main_figure_4.ipynb) (vector-panel assembly)
- [Main Figure 5: ARISTA](main_figure_5.ipynb) (reference-page export)
- [Main Figure 6: AD mouse](../dataset_workflows/admouse.ipynb) (formal builder and manuscript-page mapping gap)

## Dataset supplementary figures

- [AGIST, Supplementary Figures S2–S3](agist_figures.ipynb)
- [Grouped non-spatial analyses, Supplementary Figures S4–S5](nonspatial_figures.ipynb)
- [Chicken heart, Supplementary Figures S7–S10](../dataset_workflows/chicken_heart.ipynb)
- [MOSTA, Supplementary Figures S11–S18](mosta_figures.ipynb) (reference-page export)
- [ARISTA, Supplementary Figures S19–S24](arista_figures.ipynb)
- [AD mouse, Supplementary Figures S26–S30](../dataset_workflows/admouse.ipynb)
- [Zebrafish, Supplementary Figures S31–S38](zebrafish_si_s31_s38.ipynb)

## Revision analyses

- [Classifier smoothing, Supplementary Figure S6](classifier_smoothing.ipynb)
- [LR-complex aggregation, Supplementary Figure S25](lr_complex_aggregation.ipynb)
- [Interaction-prior ablation, Supplementary Figure S39](lr_prior_ablation_stvcr.ipynb)
- [Five-dataset benchmark, Supplementary Figure S40](loto_benchmark.ipynb)
- [Training histories, Supplementary Figure S41](training_histories.ipynb)
- [ARISTA local domains, Supplementary Figure S42](arista_local_domains.ipynb)
- [Zebrafish attention validation, Supplementary Figure S43](zebrafish_attention.ipynb)
- [Compute-cost table](compute_cost.ipynb)

## Videos

- [Zebrafish trajectory videos](../zebrafish_videos.md)

```{toctree}
:hidden:
:maxdepth: 1

main_figure_2
main_figure_4
main_figure_5
agist_figures
nonspatial_figures
mosta_figures
arista_figures
zebrafish_si_s31_s38
classifier_smoothing
lr_complex_aggregation
lr_prior_ablation_stvcr
loto_benchmark
arista_local_domains
zebrafish_attention
training_histories
compute_cost
../zebrafish_videos
```
