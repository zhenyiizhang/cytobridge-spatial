# Paper figures

There are two ways to use these notebooks.

**Redraw a paper figure.** Open a figure notebook and run its calculation and
plotting cells. Most notebooks read numerical CSV or NPZ files included with
CytoBridge and write new PDF and PNG files. A few notebooks instead open or
export a completed page; their link text says so.

**Continue from a new model run.** Start with a
[dataset notebook](../dataset_workflows/index.md), then use a figure notebook
only when it gives a command that accepts the saved files from your run. Some
paper figures still lack the step that converts a new run into the exact panel
inputs or assembles the final page. The
[complete paper index](../../paper_reproduction.md) records this figure by
figure.

The notebooks show the command, its inputs, its outputs, and the next step.
Saved notebook output lets you see the expected tables and figures before
running anything yourself.

The included numerical files can be redrawn for S4–S6, S25, S31–S40, and S42.
S25, S39, S40, and S41 also include a command that collects completed analysis
tables before drawing the figure. S4–S6, S31–S38, and S42 do not yet convert a
new model run into the exact set of paper panel inputs. Their notebooks still
show the available upstream calculations so that their inputs and outputs are
clear.

Commands that start with `cytobridge` work after installation. Commands that
start with `python scripts/...` or `python -m scripts...` use files from the
source repository and should be run from the root of the cloned repository.

When you run a notebook from a cloned repository, for example
`python scripts/execute_paper_notebooks.py --notebook arista_figures
--output-dir notebook_runs`, it creates a notebook-specific working directory.
Files written by that notebook appear under
`notebook_runs/arista_figures/outputs/arista_supplementary_figures`, not
directly under `notebook_runs`. Other notebooks use the same
`<runner output>/<notebook name>/outputs/<notebook-specific folder>` pattern;
their setup cells show the final folder name.

To print the same steps in a terminal:

```bash
cytobridge figure list
cytobridge figure explain nonspatial
cytobridge figure explain zebrafish-si
```

## Main figures

- [Main Figure 2: AGIST benchmark — redraw panel e and assemble the page](main_figure_2.ipynb)
- [Main Figure 3: chicken heart — run the analysis; final page assembly is separate](../dataset_workflows/chicken_heart.ipynb)
- [Main Figure 4: MOSTA — assemble the page](main_figure_4.ipynb)
- [Main Figure 5: ARISTA — view the assembled page](main_figure_5.ipynb)
- [Main Figure 6: AD mouse — run the analysis; final page assembly is separate](../dataset_workflows/admouse.ipynb)

## Dataset supplementary figures

- [AGIST, Supplementary Figures S2–S3](agist_figures.ipynb)
- [Weinreb and scNT, Supplementary Figures S4–S5](nonspatial_figures.ipynb)
- [Chicken heart, Supplementary Figures S7–S10 — run the analyses; final page assembly is separate](../dataset_workflows/chicken_heart.ipynb)
- [MOSTA, Supplementary Figures S11–S18 — view completed pages and find their code](mosta_figures.ipynb)
- [ARISTA, Supplementary Figures S19–S24](arista_figures.ipynb)
- [AD mouse, Supplementary Figures S26–S30 — available analyses and saved pages](../dataset_workflows/admouse.ipynb)
- [Zebrafish, Supplementary Figures S31–S38](zebrafish_si_s31_s38.ipynb)

## Additional analyses

- [Classifier smoothing, Supplementary Figure S6](classifier_smoothing.ipynb)
- [LR-complex aggregation, Supplementary Figure S25](lr_complex_aggregation.ipynb)
- [LR-prior ablation and stVCR comparison, Supplementary Figure S39](lr_prior_ablation_stvcr.ipynb)
- [Five-dataset benchmark, Supplementary Figure S40](loto_benchmark.ipynb)
- [Training histories, Supplementary Figure S41](training_histories.ipynb)
- [ARISTA local domains, Supplementary Figure S42](arista_local_domains.ipynb)
- [Zebrafish attention and control comparisons, Supplementary Figure S43](zebrafish_attention.ipynb)
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
