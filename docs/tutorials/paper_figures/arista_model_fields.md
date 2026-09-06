# Calculate the Figure 5 model fields

The [Figure 5 notebook](main_figure_5.ipynb) draws the paper's numerical results.
This page shows how the velocity and growth values are calculated from the
trained model. Run the examples from the CytoBridge code folder on a CUDA GPU.

## Model and population states

```python
from pathlib import Path
import CytoBridge as cb

cb.datasets.download("arista", destination=".", kind="analysis")
cb.datasets.download("arista", destination=".", kind="arista_growth_model_states.zip")

data = Path("data/arista")
states = data / "paper/growth_model_states"
```

The second download contains the nine unwarped populations used for Figure 5e.
They contain 82,306 cells in total. Use these files for the quantitative
calculation, rather than the separately generated display populations.

## Growth and interaction

At each time, `cb.tl.compute_velocity_components` returns `drift`, `interaction`,
`score` and their sum, `full`. Growth is evaluated by `model.predict_growth`.
The interaction magnitude is the Euclidean norm over all 52 model dimensions.

```python
import anndata as ad
import numpy as np
import torch

loaded = cb.tl.load_dynamical_model_from_dir(data / "model", dim=52, device="cuda")
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

field_output = Path("outputs/arista_model_fields")
grouped = calculate_fields(data / "model", states, field_output, device="cuda", seed=42)
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

Interaction evaluation uses randomly grouped cells. The original grouping was
not stored in the Figure 5e run, so a new evaluation does not reproduce every
interaction value exactly. The Figure 5 notebook uses the saved per-cell values
to reproduce the published circles. Growth values are deterministic for these
states and the supplied checkpoint.

## Spatial velocity

Continue from the `velocity_time_1.npz` file just calculated. Model time 1 is
5 DPI. The left panel projects the two spatial components of full velocity.
The enlarged panel compares the full and interaction 52D fields, each projected
onto the observed spatial coordinates using a 30-neighbour graph.

```python
import json
from reproduction.arista.spatial_velocity import calculate_spatial_velocity
from reproduction.arista.main_figure import SOURCE, draw_spatial_velocity

spatial_output = calculate_spatial_velocity(
    field_output / "velocity_time_1.npz", "outputs/arista_spatial_velocity",
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

gene_file = calculate_gene_velocity(data, "outputs/arista_intrinsic_velocity", device="cuda")
draw_gene_velocity(figures, palette, state_path=gene_file)
```

`gene_file` is `figure5d_intrinsic_gene_velocity_state.npz`. It stores the PCA
coordinates, projected vectors, cell labels and an explicit `drift` component
tag. The plotting function checks this tag before drawing.
