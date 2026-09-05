# Generate the ARISTA paper populations

This page generates the populations used in Figure 5a–b and Supplementary
Figure S19 from the trained ARISTA model. For a shorter introduction to the
analysis APIs, start with the [ARISTA tutorial](../dataset_workflows/arista.ipynb).

## Download the model and aligned data

Run this from the [source checkout](../../installation.md):

```python
import CytoBridge as cb

cb.datasets.download("arista", destination=".", kind="analysis")
```

The downloaded `data/arista/` directory contains the aligned H5AD, dynamical
model, edge classifier and cell-type classifier. No additional training is
needed to continue from this model.

## Simulate the populations

```bash
python -m reproduction.arista.simulate_paper_populations \
  --data-dir data/arista \
  --classifier-cache data/arista/classifier_cache/classifier_resmlp_dedb1d6442f4d3d3.pt \
  --output-dir outputs/arista_populations \
  --device cuda
```

This uses the paper's time grid, integration step, noise level, classifier and
random seed. It also uses the original shared random stream for interaction
grouping and diffusion. New analyses use separate streams by default.

The command writes three sets of H5AD files:

| Directory | Contents |
| --- | --- |
| `display_states/` | Observed populations at measured times and generated populations at intermediate times. |
| `generated_display_states/` | Generated populations at all nine times. |
| `model_states/` | Unwarped states for quantitative analysis. |

In the two display directories, `obsm["spatial"]` stores the spatially anchored
coordinates used for plotting. The model states and cell-type labels are
calculated before that display transform. The fixed-particle label file tracks
the same cells across time for lineage analysis.

## Draw the populations

```python
from reproduction.arista.supplementary import draw_supplementary

draw_supplementary(
    "outputs/arista_populations",
    "outputs/arista_population_figures",
    figures=[19],
)
```

This reads the newly simulated H5AD files and writes the S19 population plot
as PDF and PNG. The [Figure 5 tutorial](main_figure_5.ipynb) and
[S19–S24 tutorial](arista_figures.ipynb) also provide the saved numerical paper
inputs for drawing the remaining panels.
