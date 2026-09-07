# Calculate the Figure 5 model fields

The [Figure 5 notebook](main_figure_5.ipynb) contains the model calculation
and plots. First run the [ARISTA dataset notebook](../dataset_workflows/arista.ipynb),
which generates the states, communication and post-simulation random state.
The calls below continue from those outputs on a CUDA GPU.

## Model and population states

```python
from pathlib import Path
import json
data = Path("data/arista")
analysis = Path("outputs/arista")
populations = analysis / "populations"
states = populations / "model_states"
model_dir = Path(json.loads((populations / "model_selection.json").read_text())["model_dir"])
```

These are the nine unwarped measured/intermediate populations exported by the
dataset notebook. Use a new output directory for each calculation.

## Growth and interaction

At each time, `cb.tl.compute_velocity_components` returns `drift`, `interaction`,
`score` and their sum, `full`. Growth is evaluated by `model.predict_growth`.
The interaction magnitude is the Euclidean norm over all 52 model dimensions.

```python
import anndata as ad
import numpy as np
import torch
import CytoBridge as cb

loaded = cb.tl.load_dynamical_model_from_dir(model_dir, dim=52, device="cuda")
cells = ad.read_h5ad(states / "time_0.h5ad")
x = np.asarray(cells.X, dtype=np.float32)
cb.tl.set_global_random_seed(42)
fields = cb.tl.compute_velocity_components(
    x, 0., loaded.model, interaction_m=1024,
    interaction_threshold=loaded.model.interaction_net.cutoff, device="cuda",
)
interaction_magnitude = np.linalg.norm(fields["interaction"], axis=1)
with torch.no_grad():
    growth = loaded.model.predict_growth(
        t=torch.zeros((len(x), 1), device="cuda"),
        x=torch.as_tensor(x, device="cuda"),
    ).cpu().numpy().ravel()
```

Run this calculation at all nine times and save the per-cell and grouped tables:

```python
from reproduction.arista.model_fields import calculate_fields

field_output = analysis / "fields"
grouped = calculate_fields(model_dir, states, field_output, device="cuda", seed=42)
grouped.head()
```

The output contains `velocity_time_*.npz`,
`figure5e_growth_interaction_by_cell.csv` and
`figure5e_growth_interaction_by_celltype.csv`. Continue directly to plotting:

```python
from reproduction.arista.main_figure import draw_growth_interaction

figures = Path("outputs/arista_calculated_figures")
figures.mkdir(parents=True, exist_ok=True)
draw_growth_interaction(figures, table_path=field_output / "figure5e_growth_interaction_by_cell.csv")
```

The Figure 5 notebook redraws the paper panel from its original per-cell table.
The calls above plot a new model evaluation, whose interaction values can differ
because interaction is evaluated in random 1,024-cell groups.

## Spatial velocity

Restore the post-simulation random state and evaluate the five observed times
in order, using random groups of 1,024 cells for interaction. Model time 1 is
5 DPI. The left panel projects the two spatial components
of full velocity. The enlarged panel compares the full and interaction 52D
directions on a 30-neighbour graph built in spatial coordinates (`X_spatial`).
The scVelo transition directions use all 52 state dimensions on this spatial graph.

```python
import json
from reproduction.arista.spatial_velocity import calculate_spatial_velocity
from reproduction.arista.main_figure import SOURCE, draw_spatial_velocity
from reproduction.arista.model_fields import calculate_observed_fields

observed = calculate_observed_fields(data, populations, analysis / "observed_fields",
                                     device="cuda", model_dir=model_dir)
spatial_output = calculate_spatial_velocity(
    observed / "velocity_time_1.npz", analysis / "spatial_velocity",
)
palette = json.loads((SOURCE / "label_to_color.json").read_text())
draw_spatial_velocity(figures, palette, state_dir=spatial_output)
```

The returned directory contains the projected vectors and their per-cell
cosine similarities, which are passed to the plotting function.

## Intrinsic-context gene velocity

Figure 5d uses intrinsic drift. `model.predict_velocity` evaluates this term
on the 46,199 observed cells. The gene calculation takes its last 50 dimensions,
fits a two-component PCA to the gene states and projects the drift through a
30-neighbour scVelo graph.

```python
from reproduction.arista.gene_velocity import calculate_gene_velocity
from reproduction.arista.main_figure import draw_gene_velocity

gene_file = calculate_gene_velocity(data, analysis / "gene_velocity", device="cuda", model_dir=model_dir)
draw_gene_velocity(figures, palette, state_path=gene_file)
```

`gene_file` is `figure5d_intrinsic_gene_velocity_state.npz`. It stores the PCA
coordinates, projected vectors, cell labels and the `drift` component tag.
