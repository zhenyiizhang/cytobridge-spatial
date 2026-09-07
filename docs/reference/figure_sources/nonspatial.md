---
orphan: true
---

# Non-spatial analyses: inputs for S4–S5

The [S4–S5 notebook](../../tutorials/paper_figures/nonspatial_figures.ipynb)
calculates the S4b field from its three fitted models and converts explicit
model-analysis outputs into every other plotted input. The resulting data
object and directory are passed directly to the existing renderer.

The `weinreb` and `scnt_cortex` downloads provide observations, fitted models
and the Full versus No-interaction evaluations. The
`nonspatial_paper_inputs.zip` supplement contains the three LR checkpoints
for S4b and the numerical trajectory, message and pathway outputs used before
the display selections.

## Which calculation supplies each panel?

| Panels | Numerical source and calculation |
| --- | --- |
| S4a, S5a | Read measured coordinates, times and labels from the original prepared H5AD. |
| S4b | Evaluate all 49,302 measured 50-PC states with LR training seeds 42–44; average five groupings per model and then the three models; smooth PC1–PC2 on the original data-supported 50×50 grid. |
| S4c/d | Read the Full/No-interaction distribution and clone-fate evaluations; calculate the displayed changes. The Full model here is the radius-graph seed-42 model, whereas S4b uses the LR ensemble. |
| S5b | Calculate finite-difference fields from the Full model's dense trajectory; calculate the interaction-only map from exact sender-specific GNN messages summed by receiver type. |
| S5c/d | Read the selected models' distribution and new-RNA direction evaluations, then calculate the displayed comparisons. |
| S4e, S5e | Select network edges from directed numerical scores, calculate observed node counts, and recalculate CellChat concordance from all eligible directed type pairs. |
| S4f, S5f | Select and rank the displayed pathways from the full LR-attribution tables, retaining each dataset's original scoring rule. |

`reproduction.nonspatial.inputs.downloaded_sources` spells out all published
filenames. `collect_nonspatial_inputs(weinreb, scnt, output_dir)` accepts those
paths, including paths to a new evaluation, and writes the figure-input directory.

This route reevaluates S4b and recalculates display inputs for all panels. It
reuses the other declared model-analysis outputs and the independent CellChat
reference. Use each model with the prepared H5AD containing the PCA coordinates
on which it was trained.
