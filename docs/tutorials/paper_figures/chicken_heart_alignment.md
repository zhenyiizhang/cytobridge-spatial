# Chicken-heart alignment sensitivity: Figures S7–S8

These figures use the four chicken-heart stages D4, D7, D10, and D14. Input
coordinates were translated, rotated, or both. Each condition was then aligned,
trained, and analyzed again using the same settings.

## Draw the figures

From the root of the source repository, run:

```bash
python release_artifacts/chicken_heart_alignment_sensitivity_20260831/figure_code/plot_heart_alignment_sensitivity.py \
  --output-dir outputs/heart_alignment
```

This step needs NumPy, pandas, and Matplotlib. It reads the archived coordinate
arrays and numerical tables, then draws both figures as PDF and PNG. Arial is
used when installed.

The inputs are in `release_artifacts/chicken_heart_alignment_sensitivity_20260831`:

| Panels | Numerical inputs |
|---|---|
| S7a and S8 | `data/plot_inputs.npz`: original, perturbed, and aligned coordinates |
| S7b | `data/coordinate_metrics.csv`, `data/velocity_metrics_pooled.csv`, and `data/interaction_metrics.csv` |
| S7c | `manifests/input_manifest.json` and `manifests/lower_input_manifest.json`: applied translations and rotations |

The [analysis code and command sequence](https://github.com/zhenyiizhang/cytobridge-spatial/tree/main/release_artifacts/chicken_heart_alignment_sensitivity_20260831#full-calculation-order)
record preparation, alignment, training, comparison, and export of these inputs.
The plotting command above starts from those completed calculations.

## Supplementary Figure S7

Panel a shows the input and aligned sections. Panel b summarizes alignment,
velocity, and interaction-weight agreement. Panel c shows the applied
perturbation sizes. Heatmap values are rounded to two decimal places, and
`1.00*` marks values below one before rounding.

```{image} ../../_static/figures/heart_alignment_sensitivity_S7_final.png
:alt: Chicken-heart sections, alignment-sensitivity heatmaps, and perturbation sizes.
:width: 100%
```

## Supplementary Figure S8

The first four columns compare the original and perturbed input coordinates.
The last column shows the aligned sections for each condition. Later sections
are drawn first, leaving D4 and D7 visible in front.

```{image} ../../_static/figures/heart_alignment_sensitivity_S8_final.png
:alt: Original and perturbed coordinates for four chicken-heart stages, with their aligned overlays.
:width: 100%
```
