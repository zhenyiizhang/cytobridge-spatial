# Generate the ARISTA paper populations

The [ARISTA tutorial](../dataset_workflows/arista.ipynb) is the primary route
from model selection through simulation and analysis. This reference gives
the equivalent population-generation command for Figure 5a–b and S19.

## Download the model and aligned data

Run this from the [CytoBridge code folder](../../installation.md):

```python
import CytoBridge as cb

cb.datasets.download("arista", destination=".", kind="analysis")
```

The downloaded `data/arista/` directory contains the aligned H5AD, dynamical
model, edge classifier and cell-type classifier used by the following command.

## Simulate the populations

```bash
python -m reproduction.arista.simulate_paper_populations \
  --data-dir data/arista \
  --classifier-cache data/arista/classifier_cache/classifier_resmlp_dedb1d6442f4d3d3.pt \
  --output-dir outputs/arista/populations \
  --device cuda
```

This uses the paper's time grid, integration step, noise level, classifier and
random seed, with a shared random stream for interaction grouping and diffusion.

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
`post_simulation_rng.npz` records the random-state continuation used by the
observed-only field calculation in Figure 5c. It is generated during simulation,
before attention and expression analysis. `model_selection.json` records the
model directory used by the simulation. Supply `--model-dir` to continue from
a newly trained model instead of the downloaded model.

## Draw the populations

```python
from reproduction.arista.supplementary import draw_supplementary

draw_supplementary(
    "outputs/arista/populations",
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
    "outputs/arista/populations",
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
[S19–S24 tutorial](arista_figures.ipynb) continue from these same outputs and
calculate the model fields, gene programs and LR time courses before drawing.
