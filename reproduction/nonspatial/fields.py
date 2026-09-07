"""Calculate S4b model fields using its original three-model ensemble.

Numerics follow analyze_weinreb_growth_model_fields.py (18 July 2026):
50 PCs, full drift = intrinsic + score gradient + interaction, five fixed
16-cell groupings per model, and the data-derived 50 × 50 support grid.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree


def support_grid(coordinates, indices):
    lower, upper = np.quantile(coordinates, [0.005, 0.995], axis=0)
    span = upper - lower
    lower -= .025 * span
    upper += .025 * span
    x, y = np.linspace(lower[0], upper[0], 50), np.linspace(lower[1], upper[1], 50)
    gx, gy = np.meshgrid(x, y)
    tree = cKDTree(coordinates)
    distances, neighbors = tree.query(np.column_stack((gx.ravel(), gy.ravel())), k=min(100, len(coordinates)))
    cell_distance, _ = tree.query(coordinates, k=min(11, len(coordinates)))
    cutoff = np.quantile(cell_distance[:, -1], .95) * 1.5
    mask = (distances[:, 0] <= cutoff).reshape(50, 50)
    return x, y, mask, indices[neighbors], distances


def smooth_grid(grid, values):
    x, y, mask, indices, distances = grid
    bandwidth = np.maximum(distances[:, -1:], np.finfo(float).eps)
    weights = np.exp(-.5 * np.square(distances / bandwidth))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), np.finfo(float).eps)
    smoothed = np.sum(np.asarray(values, float)[indices] * weights[:, :, None], axis=1)
    u = np.ma.masked_where(~mask, smoothed[:, 0].reshape(mask.shape))
    v = np.ma.masked_where(~mask, smoothed[:, 1].reshape(mask.shape))
    return u.filled(np.nan), v.filled(np.nan), np.ma.sqrt(u*u + v*v).filled(np.nan)


def calculate_weinreb_fields(prepared_h5ad, model_dirs, output_dir, *, device="cuda", grouping_seeds=(0, 1, 2, 3, 4)):
    """Evaluate every supplied LR model; return the newly written stream-grid NPZ.

    For S4b, model_dirs must be the LR-informed training seeds 42,43,44;
    the separate frozen radius model used in S4c/d is not this ensemble.
    """
    import anndata as ad
    from reproduction.nonspatial.model import load_state_model as load_dynamical_model_from_dir
    from CytoBridge.nonspatial.grouping import runtime_style_random_groups
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = ad.read_h5ad(prepared_h5ad)
    latent = np.asarray(data.obsm["X_latent"], np.float32)
    times = data.obs["time_point_processed"].to_numpy(float)
    if latent.shape != (49302, 50) or set(times) != {0., 1., 2.}:
        raise ValueError("S4b requires the 49,302 measured cells in the original 50-PC space")
    if len(model_dirs) != 3 or len(grouping_seeds) != 5:
        raise ValueError("S4b uses three training models and five interaction groupings per model")
    full_fields, interaction_fields, records = [], [], []
    for model_dir in model_dirs:
        loaded = load_dynamical_model_from_dir(model_dir, dim=50, device=device, stage="Finetune")
        model = loaded.model.eval()
        if not bool(getattr(model.interaction_net, "state_space", False)):
            raise ValueError("S4b requires a non-spatial state-space interaction model")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        velocity = np.empty((len(latent), 2), np.float32)
        score = np.empty_like(velocity)
        for start in range(0, len(latent), 4096):
            stop = min(start + 4096, len(latent))
            x = torch.tensor(latent[start:stop], device=device)
            t = torch.tensor(times[start:stop, None], dtype=x.dtype, device=device)
            with torch.no_grad():
                velocity[start:stop] = model.predict_velocity(t=t, x=x)[:, :2].cpu().numpy()
            with torch.enable_grad():
                _, gradient = model.compute_score(t=t, x=x.requires_grad_(True), create_graph=False)
                score[start:stop] = gradient[:, :2].detach().cpu().numpy()
        samples = []
        for seed in grouping_seeds:
            interaction = np.zeros_like(velocity)
            for time in sorted(set(times)):
                indices = np.flatnonzero(np.isclose(times, time, rtol=0., atol=1e-8))
                groups = runtime_style_random_groups(indices, group_size=16, seed=int(seed))
                t = torch.tensor([float(time)], dtype=torch.float32, device=device)
                with torch.no_grad():
                    for group in groups:
                        x = torch.tensor(latent[group], device=device)
                        log_weights = torch.full((len(group), 1), -math.log(len(group)), device=device)
                        interaction[group] = model.interaction_net(x, log_weights, t)[:, :2].cpu().numpy()
            samples.append(interaction)
        interaction = np.stack(samples).mean(axis=0)
        full_fields.append(velocity + score + interaction)
        interaction_fields.append(interaction)
        records.append({"weight_path": str(loaded.weight_path), "score_path": str(loaded.score_path)})
        del model, loaded
    # As in the original, average the per-training-seed fields in float64 first,
    # then smooth that ensemble mean on a shared data-only support grid.
    fields = {"lr_full_drift": np.asarray(full_fields, dtype=np.float64).mean(axis=0), "lr_interaction": np.asarray(interaction_fields, dtype=np.float64).mean(axis=0)}
    arrays = {}
    for time, day in ((0., 2), (1., 4), (2., 6)):
        rows = np.flatnonzero(np.isclose(times, time))
        grid = support_grid(latent[rows, :2].astype(float), rows)
        arrays.update({f"day{day}_x_axis": grid[0], f"day{day}_y_axis": grid[1], f"day{day}_support_mask": grid[2]})
        for name, values in fields.items():
            arrays.update({f"day{day}_{name}_{suffix}": value for suffix, value in zip(("u", "v", "speed"), smooth_grid(grid, values))})
    destination = output / "support_masked_stream_grids.npz"
    np.savez_compressed(destination, **arrays)
    (output / "manifest.json").write_text(json.dumps({"prepared_h5ad": str(prepared_h5ad), "models": records, "grouping_seeds": list(grouping_seeds), "group_size": 16, "output": str(destination)}, indent=2) + "\n")
    return destination
