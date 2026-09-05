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
| a | Original unwarped populations in `slice_data`, communication scores and fixed-particle labels | Calculate spatial anchors and draw communication and lineage connections. |
| b | Generated population at time 0.5 | Draw the cell coordinates and labels. |
| c | Full spatial velocity and full/interaction 52D fields projected to spatial coordinates | Calculate the streamline grid and per-cell projected-vector cosine similarities. |
| d | Intrinsic gene-drift vectors and PCA coordinates | Calculate the stream grid from intrinsic drift, not full velocity. |
| e | Per-cell growth and interaction values | Calculate means by time and cell type. |

Panel a uses `data/arista/paper/slice_data/`. Panel b uses
`data/arista/paper/display_states/time_0p5.h5ad`.
Panel c–e arrays and their plotting functions are in `reproduction/arista/`.

The [model calculation page](../../tutorials/paper_figures/arista_model_fields.md)
gives runnable velocity, growth and projection steps and shows how to pass their
outputs to these plotting functions. Figure 5e's original random cell grouping
was not saved, so the notebook uses the archived per-cell values for exact
placement of its published circles.

```{toctree}
:hidden:

../../tutorials/paper_figures/arista_model_fields
```
