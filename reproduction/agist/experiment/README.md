# AGIST simulation and model evaluation

These scripts generate AGIST observations, fit CytoBridge and the comparison
methods, evaluate predictions, and draw Figure 2 and Supplementary Figures S2–S3.

Follow [AGIST simulation and training](../../../docs/tutorials/paper_figures/agist_simulations.md)
for the download and training commands. The
[Figure 2 notebook](../../../docs/tutorials/paper_figures/main_figure_2.ipynb)
evaluates the supplied checkpoints and draws the panels.

| Script | Calculation |
| --- | --- |
| `run_cytobridge.py` | Generate observations, train CytoBridge and simulate predictions |
| `run_baseline.py` | Fit stVCR or STORIES and predict cell populations |
| `evaluate.py` | Evaluate velocity and growth fields and calculate distribution errors |
| `supplementary.py` | Calculate attention agreement and supplementary comparisons |
| `layout/layout.py` | Draw Figure 2 from the calculated arrays and tables |

The four observations contain 5,000, 7,637, 9,733 and 10,667 cells.
CytoBridge and stVCR each have ten simulation repeats from one trained model.
Panel e shows their mean Wasserstein distance and sample standard deviation.
STORIES provides one gene-expression prediction per target time.
