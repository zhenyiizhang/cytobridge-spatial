# Chicken-heart alignment sensitivity: Figures S7–S8

These figures use the four chicken-heart stages D4, D7, D10, and D14. Input
coordinates were translated by 1 or 2 times the median nearest-neighbor distance,
rotated by 1° or 3°, or both. Each condition was then aligned, trained, and
analyzed again using seed 42 and the same model settings.

## Draw the figures

From the root of the source repository, run:

```bash
python reproduction/chicken_heart/alignment_sensitivity_20260906/plot_sensitivity.py \
  --output-dir outputs/heart_alignment
```

This step needs NumPy, pandas, SciPy, Matplotlib, and PyMuPDF. It reads the coordinate
arrays and numerical tables, then draws both figures as PDF and PNG. Arial is
used when installed.

The inputs are included in `reproduction/chicken_heart/alignment_sensitivity_20260906`:

| Panels | Numerical inputs |
|---|---|
| S7a and S8 | `summary/plot_inputs.npz`: original, perturbed, and aligned coordinates |
| S7b | `summary/coordinate_metrics.csv`, `summary/velocity_metrics_pooled.csv`, and `summary/interaction_metrics.csv` |
| S7c | `input_manifest.json`: applied translations and rotations in every stage |

The [analysis code and calculation record](https://github.com/zhenyiizhang/cytobridge-spatial/tree/main/reproduction/chicken_heart/alignment_sensitivity_20260906)
record preparation, alignment, training, comparison, and export of these inputs.
The plotting command above starts from those completed calculations.

## Supplementary Figure S7

Panel a shows the input and aligned sections. Panel b summarizes alignment,
velocity, and interaction-weight agreement. Panel c shows the applied
perturbation sizes. Each perturbed model is compared with the new unperturbed
model. The first row compares that unperturbed repeat with the original model.
Values are rounded for display, and
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
