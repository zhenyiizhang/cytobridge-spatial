---
orphan: true
---

# Figure 5: inputs and plotting code

The [Figure 5 notebook](../../tutorials/paper_figures/main_figure_5.ipynb)
draws the five panels from cell states, communication scores, and velocity
and growth arrays.

## Download and draw

```bash
python -m CytoBridge.datasets arista --kind arista_figure_data.zip --output-dir .
python -m CytoBridge.datasets arista --kind arista_spatial_display_data.zip --output-dir .
python -m reproduction.arista.main_figure --data-dir data/arista/paper --output-dir outputs/figure5
```

Each line is a separate command. The first two download the inputs. The third
draws the panels and writes their numerical summaries.

| Panel | Input | Calculation |
| --- | --- | --- |
| a | Display populations, communication scores and fixed-particle labels | Calculate spatial anchors and draw communication and lineage connections. |
| b | Generated population at time 0.5 | Draw the cell coordinates and labels. |
| c | Per-cell full and interaction spatial velocity | Calculate cosine similarities and the spatial velocity grid. |
| d | Full gene-velocity vectors and PCA coordinates | Calculate the stream grid. |
| e | Per-cell growth and interaction values | Calculate means by time and cell type. |

Panel a–b populations are in `data/arista/paper/display_states/`.
Panel c–e arrays and their plotting functions are in `reproduction/arista/`.

To reproduce the population simulation first, follow
[Generate the ARISTA paper populations](../../tutorials/paper_figures/arista_populations.md).
