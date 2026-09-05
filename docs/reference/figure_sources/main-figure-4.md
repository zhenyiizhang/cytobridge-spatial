---
orphan: true
---

# Figure 4: inputs and plotting code

The [Figure 4 notebook](../../tutorials/paper_figures/main_figure_4.ipynb)
draws the panels from numerical inputs. It does not copy the scientific
panels from a completed page.

## Download and draw

From the source checkout:

```bash
python -m CytoBridge.datasets mosta --kind mosta_figure_data.zip --output-dir .
python -m reproduction.mosta.main_figure --data-dir data/mosta/paper --output-dir outputs/figure4
```

The first command downloads the cell-state inputs. The second calculates the
plotting summaries and writes PDF and PNG files.

| Panel | Numerical input | Calculation |
| --- | --- | --- |
| a | `data/mosta/paper/figure4a/slice_data/*.h5ad` | Plot observed and generated coordinates with their cell-type labels. |
| b | `data/mosta/paper/figure4b/cell_mapping.csv.gz` | Normalize the interaction scores using each time's 1st and 99th percentiles. |
| c | Cartilage particle states in the source archive | Count lineage destinations and calculate the three largest transition fractions. |
| d | Per-cell gene velocity and cell-type communication tables | Calculate the velocity grid, cell-type anchors and communication arrow widths. |
| e | Brain states and full/interaction velocity arrays | Calculate gene-space and spatial velocity grids in the selected region. |

The smaller panel c–e arrays are included under
`release_artifacts/mosta_package_native_corrected_20260826_v1/reproduction/main_fig4_panels/`.
The Python plotting functions are in `reproduction/mosta/main_figure.py`.
Panels a–b retain the frame and label layout from the paper, but replace every
scientific point layer with a newly drawn layer.

## Continue from the trained model

The [MOSTA analysis tutorial](../../tutorials/dataset_workflows/mosta.ipynb)
loads the matching model and aligned data, generates populations, and calculates
growth and cell-type composition. The figure notebook uses the saved paper
states so stochastic simulation does not change which cells are displayed.
