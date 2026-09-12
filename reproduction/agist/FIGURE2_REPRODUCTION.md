# Figure 2 calculations and models

The Figure 2 notebook evaluates models and passes the calculated arrays to
the plotting functions in `main_figure.py`. It does not read finished panels.

| Panel | Inputs | Calculation |
| --- | --- | --- |
| a | AGIST CSV, 31,816 cells | Four spatial snapshots |
| b | CSV, six-stage model, score model, generator fields | Full velocity and scVelo projection |
| c | Same CSV, model 1214, mouse.pt, generator T0 attention | All-cell T0 graph, outgoing strengths and top-5,000-edge flow |
| d | Same CSV, six-stage model, generator growth | Cell-level growth, percentile scaling and regression |
| e | Same CSV, six-stage model and score model | Ten split-SDE simulations, weighted W2, mean and sample SD |

The illustration above panel a is hand-drawn. The STORIES and stVCR values in
panel e are the recorded external benchmark values, not new baseline runs.

## September 12 model check

The supplied model implementation and the package implementation produced
identical attention outputs on the same GPU. For model 1214 at T0, the maximum
absolute difference from the saved prediction matrix was 6.68e-6. For model
0210 at each of four times it was below 8.35e-7. All matrices agreed within
`rtol=1e-5, atol=1e-6`. No model was retrained or selected by its correlation.

The original attention model is used only for c. It gives normalized growth
Pearson r=0.95343, whereas the released six-stage model used for d gives
0.95533, or 0.96 when rounded. These model versions are named explicitly in
the notebook and are not substituted for one another.

## Separate four-time-point attention analysis

`agist_attention_recovery.ipynb` uses `mouse_brain_simulation_new.csv`, with
6,707 cells, model 0210 and generator 0207. It evaluates the graph anew at
each time and compares mean outgoing attention among the same nonzero cells.
The four Spearman correlations are:

| Time | Cells | Compared nonzero cells | Spearman |
| --- | ---: | ---: | ---: |
| 0 | 1,147 | 659 | 0.867203 |
| 1 | 1,619 | 694 | 0.766144 |
| 2 | 1,986 | 1,329 | 0.731388 |
| 3 | 1,955 | 1,419 | 0.745090 |

Their equal-weight mean is 0.777456, which rounds to 0.78. The original
notebook recorded the four separate values but did not contain an averaging
step. The public tutorial calculates the mean explicitly. It is not the
correlation of the Figure 2c maps from the original larger simulation.

The input download `agist_figure2_inputs.zip` includes the generator arrays,
models, edge predictors, original code and saved predictions. The two
notebooks calculate new predictions, rather than drawing the saved ones.

## Notebook execution

Both notebooks were executed top-to-bottom on 13 September: eight code cells
and six displayed plots for Figure 2, and three code cells for attention
strength. Models were freshly evaluated. Panel e used the retained ten
simulated populations and recalculated all 60 distances, matching the earlier
distance table exactly. The default notebook also provides the simulation
step for readers without those trajectories. This test did not retrain models
or repeat the population simulations. Executed outputs are saved in both
public notebooks.
