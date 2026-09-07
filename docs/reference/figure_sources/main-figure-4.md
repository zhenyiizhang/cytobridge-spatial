---
orphan: true
---

# Figure 4: inputs and plotting code

Run the [Figure 4 notebook](../../tutorials/paper_figures/main_figure_4.ipynb)
from top to bottom to calculate the panels from measured MOSTA data and the
trained model. You do not need to run another notebook first. New results go
to `outputs/figure4_from_model`.

## Calculate from the model

| Panel | Calculation | Result passed to the plot |
| --- | --- | --- |
| a | `run_interpolation_workflow` simulates growing populations and classifies generated cells. Observed stages retain measured cells. | New `populations/figure4a/slice_data/time_*.h5ad` files. |
| b | A 12,000-particle simulation supplies the calculation cohort. `compute_timepoint_communications` calculates attention. `compute_focal_lr_type_hotspots` combines communication with Wnt3a and Fzd7/Lrp6 expression, then maps type scores to panel-a cells. | `mapping`, also saved as `cell_mapping.csv.gz`. |
| c | `simulate_sde_points` retains particle identities. `predict_labels_for_trajectories` labels the cells. Select cartilage-primordium particles at E15 and follow them to E15.5. | The `lineage` dictionary, also saved as `cartilage_lineage.npz`. |
| d | Evaluate interaction gene velocity and communication on 8,000 observed E15.5 cells. | New `calculated_d/numeric_inputs.npz` and `communication_all_type_edges.csv.gz`. |
| e | Evaluate full and interaction velocity on 17,071 observed E15.5 Brain cells. | New `calculated_e/numeric_inputs.npz`. |

The small d/e input arrays specify measured cell states and anatomical labels.
The notebook recalculates their velocities and communication. The lineage
simulation is separate from the growing populations because it retains one
row per particle throughout the trajectory.

## Redraw saved paper results

From the CytoBridge code folder:

```bash
python -m CytoBridge.datasets mosta --kind mosta_figure_data.zip --output-dir .
python -m reproduction.mosta.main_figure --data-dir data/mosta/paper --output-dir outputs/figure4
```

The first command downloads the cell-state inputs. The second calculates the
plotting summaries and writes PDF and PNG files.
These commands redraw saved results. Execute the notebook above to calculate
new populations, LR scores, lineage labels and model fields before drawing.

| Panel | Numerical input | Calculation |
| --- | --- | --- |
| a | `data/mosta/paper/figure4a/slice_data/*.h5ad` | Plot observed and generated coordinates with their cell-type labels. |
| b | Cell-type expression and communication table, plus spatial cell mapping | Multiply ligand and receptor-complex means by communication weight. Sum incoming and outgoing scores and map them to cells. |
| c | The 1,282 E15 cartilage-primordium particles and their E15.5 labels | Count destinations of the same particles and divide by 1,282. |
| d | Saved velocity arrays and communication table for the 8,000 observed E15.5 cells | Draw the projected interaction gene velocity and cell-type communication. |
| e | Saved full/interaction velocity arrays for the 17,071 observed E15.5 Brain cells | Select the spatial region and draw the four velocity fields. |

The smaller panel c–e arrays are included under
`release_artifacts/mosta_package_native_corrected_20260826_v1/reproduction/main_fig4_panels/`.
The Python plotting functions are in `reproduction/mosta/main_figure.py`.
The LR calculations and model-evaluation functions are in
`reproduction/mosta/calculations.py`.
Panels a–b retain the frame and label layout from the paper, but replace every
scientific point layer with a newly drawn layer.

The MOSTA dataset tutorial concerns S11–S13. It is not a prerequisite for
Figure 4, whose required calculations are included directly in its notebook.
