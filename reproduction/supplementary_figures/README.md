# Supplementary figure plotting

Recreate the September 6 versions of S2–S7, S25, S34, S36, and S39–S46 from the
numerical results included with this repository. The scripts calculate the plotted
summaries and draw the panels. Existing PDF or PNG figures are not used as inputs.

From the repository root, after installing CytoBridge and the plotting dependencies:

```bash
python -m pip install matplotlib scipy pandas numpy pymupdf
python reproduction/supplementary_figures/plot_figures.py --output-dir outputs/supplementary_figures
```

To draw selected figures:

```bash
python reproduction/supplementary_figures/plot_figures.py --figures S39 S41 --output-dir outputs/selected_figures
```

Each figure is saved as a vector PDF and a 320-dpi PNG. The output directory also
contains a combined PDF and the calculated plot tables. Use Arial to match the
manuscript typography. Training is not repeated by this plotting command. Dataset
preprocessing and training are documented in their respective tutorials.

## Inputs and calculations

| Figures | Input tables or arrays | Plotting code |
| --- | --- | --- |
| S2–S3 | `CytoBridge/results/data/agist_figures` | `plot_summaries.s2`, `plot_panels.agist` |
| S4–S5 | `CytoBridge/results/data/nonspatial_figures` | `plot_panels.nonspatial` |
| S6 | Classifier-smoothing result tables loaded by `CytoBridge.results.classifier_smoothing` | `plot_panels.classifier` |
| S7 | `data/heart_alignment` | `plot_panels.heart` |
| S25 | ARISTA local-domain tables loaded by `CytoBridge.results.arista_local_domains` | `plot_domains.s25` |
| S34, S36 | Zebrafish numerical results loaded by `CytoBridge.results.zebrafish_si` | `plot_panels.zebrafish` |
| S39 | `data/commot_comparison` and `CytoBridge/results/data/zebrafish_attention` | `commot_permutations`, `plot_panels.attention` |
| S40 | `data/stability` | `plot_summaries.s40` |
| S41 | Complete paired LR scores loaded by `CytoBridge.results.lr_complex_aggregation` | `plot_panels.lr` |
| S42 | Package No-LR result tables and `data/interaction_ablation` | `plot_panels.ablation` |
| S43 | `data/communication` | `plot_panels.communication` |
| S44 | `data/benchmark` | `plot_panels.wins` |
| S45 | Leave-one-time-point-out tables loaded by `CytoBridge.results.loto_benchmark` | `plot_panels.benchmark` |
| S46 | Training histories loaded by `CytoBridge.results.training_histories` | `plot_panels.training` |

S2 uses medians. S4 and S5 use black streamlines. S39a repeats the original 1,000
within-group permutations from the pair scores and checks their summaries against
the recorded analysis. S41c calculates top-100 Jaccard overlap from the full score
tables, with the same cutoff in all four datasets. S42c/d compare the same trained
model with and without its interaction output during inference. S44 retains
Linear OT, random interpolation and SpaTrack and uses black markers for all methods.

The submitted-style PDFs and PNGs are retained in
`release_artifacts/supplementary_figures_20260906/figures`. That archive also includes
the top-10, top-50 and top-100 LR comparisons. Paths beginning with `data/` in the
table are relative to this directory. Package paths are relative to the repository
root. S34 and S36 share a plotting routine, so requesting either produces both.
