---
orphan: true
---

# Figure 4: inputs and plotting code

The [Figure 4 notebook](../../tutorials/paper_figures/main_figure_4.ipynb)
draws the panels from numerical inputs. It does not copy the scientific
panels from a completed page.

## Download and draw

From the CytoBridge code folder:

```bash
python -m CytoBridge.datasets mosta --kind mosta_figure_data.zip --output-dir .
python -m reproduction.mosta.main_figure --data-dir data/mosta/paper --output-dir outputs/figure4
```

The first command downloads the cell-state inputs. The second calculates the
plotting summaries and writes PDF and PNG files.
These commands redraw saved results. Execute the notebook to recalculate the
panel-b scores and the panel-d/e model fields before drawing.

| Panel | Numerical input | Calculation |
| --- | --- | --- |
| a | `data/mosta/paper/figure4a/slice_data/*.h5ad` | Plot observed and generated coordinates with their cell-type labels. |
| b | Cell-type expression and communication table, plus spatial cell mapping | Multiply ligand and receptor-complex means by communication weight. Sum incoming and outgoing scores and map them to cells. |
| c | The 1,282 E15 cartilage-primordium particles and their E15.5 labels | Count destinations of the same particles and divide by 1,282. |
| d | Trained model and the 8,000 observed E15.5 cell states used in the paper | Evaluate interaction gene velocity and communication on the same cells, then project the 50D gene derivative onto spatial coordinates. |
| e | Trained model and all 17,071 E15.5 Brain cell states | Calculate full/interaction fields and project gene derivatives before selecting the spatial region. |

The smaller panel c–e arrays are included under
`release_artifacts/mosta_package_native_corrected_20260826_v1/reproduction/main_fig4_panels/`.
The Python plotting functions are in `reproduction/mosta/main_figure.py`.
The LR calculations and model-evaluation functions are in
`reproduction/mosta/calculations.py`.
Panels a–b retain the frame and label layout from the paper, but replace every
scientific point layer with a newly drawn layer.

## Continue from the trained model

The [MOSTA analysis tutorial](../../tutorials/dataset_workflows/mosta.ipynb)
loads the matching model and aligned data, generates populations, and calculates
growth and cell-type composition. The figure notebook uses saved population
states for a–c, then evaluates the downloaded model for d–e. It passes the newly
calculated velocity and communication files directly to the plotting functions.
