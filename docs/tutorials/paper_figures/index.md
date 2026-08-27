# Paper figure notebooks

These notebooks use the public result APIs shipped with CytoBridge. Each page
states whether it redraws numeric results, assembles panel files, or exports a
reference page. Saved outputs show the result of the documented code.

The figure commands read released result files; they do not retrain a model.
Run `cytobridge figure list` to see the execution mode for every command.

## Installed commands

| Paper entry | Command |
|---|---|
| Main Figure 2 | `cytobridge figure main-figure-2 --output-dir outputs/main_figure_2` |
| Main Figure 4 | `cytobridge figure main-figure-4 --results-dir /path/to/mosta-release --output-dir outputs/main_figure_4` |
| Main Figure 5 reference page | `cytobridge figure main-figure-5-reference --output-dir outputs/main_figure_5` |
| Supplementary Figures S2–S3 | `cytobridge figure agist --output-dir outputs/agist` |
| Supplementary Figures S4–S5 | `cytobridge figure nonspatial --output-dir outputs/nonspatial` |
| Supplementary Figure S6 | `cytobridge figure classifier-smoothing --output-dir outputs/classifier_smoothing` |
| Supplementary Figures S9–S16 | `cytobridge figure mosta-reference-pages --results-dir /path/to/mosta-release --output-dir outputs/mosta_si` |
| Supplementary Figures S21–S22 | `cytobridge figure arista-lr --output-dir outputs/arista_lr` |
| Supplementary Figure S23 | `cytobridge figure lr-complex --output-dir outputs/lr_complex` |
| Supplementary Figures S27–S34 | `cytobridge figure zebrafish-si --output-dir outputs/zebrafish_si` |
| Supplementary Figure S35 | `cytobridge figure interaction-evidence --output-dir outputs/interaction_evidence` |
| Supplementary Figure S36 | `cytobridge figure loto-benchmark --output-dir outputs/loto_benchmark` |
| Supplementary Figure S37 | `cytobridge figure training-histories --output-dir outputs/training_histories` |
| Supplementary Figure S38 | `cytobridge figure arista-local-domains --output-dir outputs/arista_local_domains` |
| Supplementary Figure S39 | `cytobridge figure zebrafish-attention --output-dir outputs/zebrafish_attention` |
| Supplementary Table 2 | `cytobridge figure compute-cost --output-dir outputs/compute_cost` |

ARISTA S17–S20 are released reference pages rather than numerical redraws; the
ARISTA notebook labels that section separately. Dataset notebooks contain the
preprocessing, training, and downstream commands that precede these compact
figure workflows.

## Main figures

- [Main Figure 2: AGIST benchmark](main_figure_2.ipynb) (panel-e summary redraw and page assembly)
- [Main Figure 4: MOSTA](main_figure_4.ipynb) (vector-panel assembly)
- [Main Figure 5: ARISTA](main_figure_5.ipynb) (reference-page export)

## Dataset supplementary figures

- [AGIST, Supplementary Figures S2–S3](agist_figures.ipynb)
- [Grouped non-spatial analyses, Supplementary Figures S4–S5](nonspatial_figures.ipynb)
- [MOSTA, Supplementary Figures S9–S16](mosta_figures.ipynb) (reference-page export)
- [ARISTA, Supplementary Figures S17–S22](arista_figures.ipynb)
- [Zebrafish, Supplementary Figures S27–S34](zebrafish_si_s27_s34.ipynb)

## Revision analyses

- [Classifier smoothing, Supplementary Figure S6](classifier_smoothing.ipynb)
- [LR-complex aggregation, Supplementary Figure S23](lr_complex_aggregation.ipynb)
- [Interaction-prior ablation, Supplementary Figure S35](lr_prior_ablation_stvcr.ipynb)
- [Five-dataset benchmark, Supplementary Figure S36](loto_benchmark.ipynb)
- [ARISTA local domains, Supplementary Figure S38](arista_local_domains.ipynb)
- [Zebrafish attention validation, Supplementary Figure S39](zebrafish_attention.ipynb)
- [Training histories](training_histories.ipynb)
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
zebrafish_si_s27_s34
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
