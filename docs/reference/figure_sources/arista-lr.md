---
orphan: true
---

# ARTISTA S24–S25, S27–S30: inputs and plotting code

The [ARTISTA supplementary notebook](../../tutorials/paper_figures/arista_figures.ipynb)
draws all six figures from numerical data.

## Download and draw

```bash
python -m CytoBridge.datasets arista --kind arista_figure_data.zip --output-dir .
python -m CytoBridge.datasets arista --kind arista_spatial_display_data.zip --output-dir .
python -m reproduction.arista.supplementary --data-dir data/arista/paper --output-dir outputs/arista_si
```

| Figure | Input | Calculation |
| --- | --- | --- |
| S24 | Observed and generated H5AD populations | Draw the nine-time spatial population comparison. |
| S30 | Per-cell growth table | Select the seed-42 display sample and calculate each panel's colour range. |
| S25 | Fixed-particle cell-type labels | Count transitions, counts and proportions. |
| S27 | Gene trajectories, program summaries and GO results | Draw expression profiles, mean and standard-deviation curves, and enrichment plots. |
| S28 | All 531 LR profiles | Normalize, cluster and calculate program means. |
| S29 | Selected LR profiles | Draw 25 time courses from each of the two programs. |

The plotting functions are in `reproduction/arista/supplementary.py`.
Gene and LR tables are included in `CytoBridge/results/data/arista_supplementary_figures/`.

To reproduce the population simulation first, follow
[Generate the ARTISTA paper populations](../../tutorials/paper_figures/arista_populations.md).
