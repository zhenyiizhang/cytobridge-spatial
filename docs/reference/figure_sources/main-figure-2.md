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

## Models and generator references

The Figure 2 notebook downloads `agist_figure2_inputs.zip` in addition to the
standard AGIST data. It contains the original attention model (`1214`), its
edge predictor, and the generator's `g_values.npy` and `attn_matrix_time0.npy`.
These arrays match the 31,816-row CSV, with 5,023 cells at T0. Panel c evaluates
the original attention model. Panels b, d and e use the released six-stage
model. In particular, panel d's normalized growth Pearson correlation is
0.9553, which rounds to 0.96. The older model instead gives 0.9534.

The comparison cells infer new predictions, calculate the attention-flow grid
or growth regression, then draw the panels. Only the upper illustration in a
is a non-numerical schematic.

To download just the additional inputs:

```bash
python -m CytoBridge.datasets agist --output-dir . --kind agist_figure2_inputs.zip
```

The files are extracted to `data/agist/figure2/`. The plotting and model
evaluation functions are in `reproduction/agist/main_figure.py`. The download
also retains the supplied model implementation and saved prediction matrices
for comparison. The notebooks do not use the saved predictions for drawing.

## Attention strength across time

The [attention-strength notebook](../../tutorials/paper_figures/agist_attention_recovery.ipynb)
uses a separate 6,707-cell simulation, model `0210`, and generator `0207`.
It evaluates all cells at each time, averages the absolute first-layer
attention over eight heads, and compares each cell's mean outgoing attention
with the generator. Both arrays have the same nonzero-cell mask at each time.

The four Spearman correlations are 0.8672, 0.7661, 0.7314 and 0.7451.
Their equal-weight mean is 0.77746, or 0.78 to two decimal places.
The original notebook printed the four values separately. The public notebook
adds the explicit averaging step. This summary is separate from the T0 maps
in Figure 2c, whose nonzero-cell Spearman correlation is approximately 0.9753.
