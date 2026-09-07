# Generate the ARISTA paper populations

This page generates the populations used in Figure 5a–b and Supplementary
Figure S19 from the trained ARISTA model. For a shorter introduction to the
analysis APIs, start with the [ARISTA tutorial](../dataset_workflows/arista.ipynb).

## Download the model and aligned data

Run this from the [CytoBridge code folder](../../installation.md):

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

The command writes four sets of H5AD files:

| Directory | Contents |
| --- | --- |
| `display_states/` | Observed populations at measured times and generated populations at intermediate times. |
| `generated_display_states/` | Generated populations at all nine times. |
| `model_states/` | Populations used for quantitative analysis. |
| `slice_data/` | The same observed and intermediate populations as `display_states/`, in the file layout read by Figure 5a–b. |

For generated cells, `X` contains the two simulated spatial coordinates followed
by the 50 gene-state features. `obsm["spatial"]` is a copy of the first two columns
of `X`. Plotting uses these simulated coordinates directly. The fixed-particle
label file `fixed_particle_lineage_labels.npz` tracks the same cells across time
for lineage analysis. The command also evaluates attention on the separate
`model_states` populations with `compute_timepoint_communications` and saves
`all_time_communications.pkl`. Its per-time records contain cell-type labels and
the attention matrices used by Figure 5a. The settings come from
`CytoBridge/configs/arista_downstream.yaml`: self loops are retained and the
winsorization quantile is 0.995. `communication_settings.json` records those settings.

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
as PDF and PNG. Use the same run's populations, particle labels and calculated
communication matrices for Figure 5a–b:

```python
from reproduction.arista.main_figure import draw_main_figure

draw_main_figure(
    "outputs/arista_populations",
    "outputs/arista_main_figure",
    panels="ab",
)
```

This call reads `slice_data/time_0.h5ad` through `time_2.h5ad` at half-time
intervals, plus `all_time_communications.pkl` and
`fixed_particle_lineage_labels.npz`, from the preceding simulation output.
For Figure 5c–e, continue with the
[velocity and growth calculations](arista_model_fields.md).

The [Figure 5 tutorial](main_figure_5.ipynb) and
[S19–S24 tutorial](arista_figures.ipynb) also provide the saved numerical paper
inputs for drawing the remaining panels.
