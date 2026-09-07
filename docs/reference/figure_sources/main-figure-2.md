---
orphan: true
---

# Main Figure 2: simulation and numerical inputs

The [Figure 2 notebook](../../tutorials/paper_figures/main_figure_2.ipynb)
downloads the AGIST observations, checkpoint and edge predictor, generates the
four observed snapshot plots, evaluates the model fields and reconstructs the
scVelo velocity comparisons. It also generates ten split-SDE trajectories and
calculates weighted W2 at times 1–3. The resulting tables are passed directly
to the panel-e renderer. Previously generated trajectories or distances can be
selected explicitly.

Only CytoBridge is reevaluated in panel e. STORIES and stVCR remain the paper's
recorded external benchmark values. No model training is performed.

The numerical methods follow the original AGIST plotting notebook: the same
30% seed-0 rows for both velocity comparisons, 30-neighbor graphs, and the
first two gene-state coordinates for the gene display. The model's growth and
time-zero attention graph are also calculated from the checkpoint.

Two generator reference files are still absent: `g_values.npy` and
`attn_matrix_time0.npy`, originally written under
`results/mosta_interaction_1017_tiaocan/`. The velocity archives contain neither
file. Supply these observation-matched arrays to complete c/d using the
notebook's growth normalization, regression and attention-flow calculations.
Until those inputs are available, the notebook draws a's snapshots, b and e.
Only the upper illustration in a is a non-numerical schematic.
