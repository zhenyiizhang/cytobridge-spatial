# Simulate trajectories

Continue from a [dataset analysis notebook](tutorials/dataset_workflows/index.md).
It creates `states`, `model`, and `OUTPUT_DIR`. Here we use that model
to follow cells between two observed stages.

## Choose a time interval

Start with the cells at the first observed stage. The example predicts their
states halfway to the next measurement and at the next measured time.

```python
import numpy as np

interval_index = 0
t0, t1 = observed_times[interval_index:interval_index + 2]
prediction_times = [t0, (t0 + t1) / 2, t1]
n_initial = int((states.obs[time_key] == t0).sum())
simulation_sigma = 0.03
```

The examples use a diffusion amplitude of 0.03. Set `simulation_sigma` to the
value used in your training run if you changed it.

## Follow cells and their weights

`simulate_sde_points` integrates the fitted dynamics. It returns a cell-state
array at every requested time and a corresponding array of population weights.
Growth changes those weights while the number of trajectories stays fixed.

```python
cb.tl.set_global_random_seed(42)
trajectories, weights = cb.tl.simulate_sde_points(
    states, model, time_index=interval_index, n_samples=n_initial,
    ts_points=prediction_times, dt=0.05,
    sigma=simulation_sigma, include_score=True,
    interaction_m=1024, interaction_seed=42,
    time_key=time_key, device=DEVICE,
)
trajectories[1].shape, weights[1].shape
```

The first two state columns are spatial coordinates. Plot the intermediate
population, using the weights to show relative cell mass:

```python
import matplotlib.pyplot as plt

points = np.asarray(trajectories[1])
mass = np.asarray(weights[1]).reshape(-1)
fig, ax = plt.subplots(figsize=(5, 5))
plot = ax.scatter(points[:, 0], points[:, 1], c=mass, s=4, cmap="viridis")
ax.set_aspect("equal")
ax.set_title(f"Predicted population, t = {prediction_times[1]:g}")
fig.colorbar(plot, ax=ax, label="Cell weight")
fig.savefig(OUTPUT_DIR / "intermediate_population.pdf", bbox_inches="tight")
plt.show()
```

## Generate a changing-size population

For a population whose size changes with growth, use
`simulate_sde_points_split_from_x0`. Cells are copied or removed during the
simulation according to their weights. The returned arrays can therefore have
different numbers of rows at different times.

```python
runtime = cb.tl.build_dynamical_runtime(model)
initial = states[states.obs[time_key] == t0].X
populations, lineage_ids = cb.tl.simulate_sde_points_split_from_x0(
    x0=initial, f_net=runtime.f_net, score_net=runtime.score_net,
    ts_points=prediction_times, dt=0.05,
    sigma=simulation_sigma, sigma_by_dim=None,
    growth_alpha=1.0, interaction_m=1024, device=DEVICE,
    resample_dt=0.05, max_particles=100000,
    return_lineage_ids=True, interaction_seed=42,
)
[len(population) for population in populations]
```

`lineage_ids` connects each generated cell to its initial cell in this simulation.
To examine another interval, start from its observed population and change
`prediction_times`. The paper's dataset-specific simulations use the time grids
and sampling settings given in their figure tutorials.

For cell-type assignment, LR summaries, and multistage population plots, see
the [interpolation API](api/tools.rst).
