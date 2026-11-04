---
orphan: true
---

# Figure 2: AGIST simulations

The [Figure 2 notebook](../../tutorials/paper_figures/main_figure_2.ipynb)
evaluates CytoBridge against the AGIST generator and draws Figure 2 and
Supplementary Figures S2–S3.

## Data and models

```bash
python -m CytoBridge.datasets agist --output-dir .
```

The download creates `data/agist_final/`. It contains the simulated
observations, generator and CytoBridge checkpoints, together with stVCR and
STORIES predictions. The accompanying score model and model settings are
included with each checkpoint.

## Calculate the panels

The notebook first evaluates velocity, growth and attention on the observed
cells, using the generator as the reference. It then calculates velocity
agreement, growth correlation and attention correlation for the plots.
For panel e, it calculates Wasserstein distances for the saved predictions.
The ten CytoBridge and stVCR predictions use different simulation seeds with
one trained model per method.

The plotting cells draw these results using the paper's panel arrangement.
The illustration above panel a is supplied separately as vector artwork.

The [attention notebook](../../tutorials/paper_figures/agist_attention_recovery.ipynb)
shows the attention correlations at each of the four observed times and their
mean. For simulation, training and new predictions, continue to
[AGIST simulation and training](../../tutorials/paper_figures/agist_simulations.md).
