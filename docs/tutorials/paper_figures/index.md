# Paper figures

Each notebook follows one calculation from its input files to the figure shown
in the paper. Read the steps from top to bottom:

1. Open the dataset notebook for preprocessing, training, and downstream
   analysis.
2. Continue with the figure-specific command listed at the top of the figure
   notebook.
3. Run the calculation and plotting cells. Their tables and figures are also
   saved in the published notebook.

You can print the same steps in a terminal:

```bash
cytobridge figure list
cytobridge figure explain nonspatial
cytobridge figure explain zebrafish-si
```

## Main figures

- [Main Figure 2: AGIST benchmark — panel e calculation and page assembly](main_figure_2.ipynb)
- [Main Figure 3: chicken heart](../dataset_workflows/chicken_heart.ipynb)
- [Main Figure 4: MOSTA — source map and page assembly](main_figure_4.ipynb)
- [Main Figure 5: ARISTA — source map and assembled page](main_figure_5.ipynb)
- [Main Figure 6: AD mouse](../dataset_workflows/admouse.ipynb)

## Dataset supplementary figures

- [AGIST, Supplementary Figures S2–S3](agist_figures.ipynb)
- [Weinreb and scNT, Supplementary Figures S4–S5](nonspatial_figures.ipynb)
- [Chicken heart, Supplementary Figures S7–S10](../dataset_workflows/chicken_heart.ipynb)
- [MOSTA, Supplementary Figures S11–S18 — source map and completed pages](mosta_figures.ipynb)
- [ARISTA, Supplementary Figures S19–S24 — S23–S24 calculations and S19–S22 source map](arista_figures.ipynb)
- [AD mouse, Supplementary Figures S26–S30](../dataset_workflows/admouse.ipynb)
- [Zebrafish, Supplementary Figures S31–S38](zebrafish_si_s31_s38.ipynb)

## Additional analyses

- [Classifier smoothing, Supplementary Figure S6](classifier_smoothing.ipynb)
- [LR-complex aggregation, Supplementary Figure S25](lr_complex_aggregation.ipynb)
- [Interaction-prior ablation, Supplementary Figure S39](lr_prior_ablation_stvcr.ipynb)
- [Five-dataset benchmark, Supplementary Figure S40](loto_benchmark.ipynb)
- [Training histories, Supplementary Figure S41](training_histories.ipynb)
- [ARISTA local domains, Supplementary Figure S42](arista_local_domains.ipynb)
- [Zebrafish attention, Supplementary Figure S43](zebrafish_attention.ipynb)
- [Training time and memory](compute_cost.ipynb)

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
