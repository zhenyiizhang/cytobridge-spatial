"""Evaluate ARISTA velocity, growth and interaction at the paper cell states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import torch
import CytoBridge as cb


def calculate_observed_fields(data_dir, population_dir, output_dir, device='cuda', *, model_dir=None):
    """Continue the original post-simulation observed-only velocity calculation.

    Interaction evaluation uses random groups of 1024. The RNG snapshot is
    produced by the same simulation, not a seed selected to fit a figure.
    """
    data, output = Path(data_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    reference = ad.read_h5ad(data / 'aligned.h5ad')
    selected_model = data / 'model' if model_dir is None else Path(model_dir)
    loaded = cb.tl.load_dynamical_model_from_dir(selected_model, dim=52, device=device)
    with np.load(Path(population_dir) / 'post_simulation_rng.npz', allow_pickle=False) as rng:
        torch.set_rng_state(torch.from_numpy(rng['cpu']))
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state(torch.from_numpy(rng['cuda']), device=device)
    components = cb.tl.compute_velocity_components_from_adata(
        reference, loaded.model, interaction_m=1024,
        interaction_threshold=loaded.model.interaction_net.cutoff,
        device=device, time_key='time_point_processed', obsm_key='X_latent',
        spatial_key='spatial_aligned', concat_spatial=True,
        write_to_adata=False, reuse_if_present=False)
    output.mkdir(parents=True)
    for time in np.unique(components['times']):
        selected = np.isclose(components['times'], time)
        np.savez_compressed(output / f'velocity_time_{time:g}.npz',
                            **{k: v[selected] for k, v in components.items() if k != 'times'})
    return output


def calculate_fields(model_dir, state_dir, output_dir, device='cuda', seed=42, *, save_components=True):
    """Write per-cell values and the time-by-cell-type means in Figure 5e.

    ``state_dir`` contains the nine unwarped population files used for this
    analysis. For the paper, use the archived ``growth_model_states`` download.
    A new simulation can also be supplied, but will give different particles.
    The first two state columns are spatial.
    The remaining fifty columns are the gene representation used in training.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    loaded = cb.tl.load_dynamical_model_from_dir(model_dir, dim=52, device=device)
    output.mkdir(parents=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rows = []
    for time in np.arange(0., 4.01, .5):
        token = f'{time:g}'.replace('.', 'p')
        cells = ad.read_h5ad(Path(state_dir) / f'time_{token}.h5ad')
        values = np.asarray(cells.X, dtype=np.float32)
        components = cb.tl.compute_velocity_components(
            values, float(time), loaded.model, interaction_m=1024,
            interaction_threshold=loaded.model.interaction_net.cutoff,
            device=device, spatial_dim=2)
        with torch.no_grad():
            x = torch.as_tensor(values, device=device)
            t = torch.full((len(values), 1), float(time), device=device)
            growth = loaded.model.predict_growth(t=t, x=x).cpu().numpy().ravel()
        rows.append(pd.DataFrame({
            'time': time, 'celltype': cells.obs.Annotation.astype(str).to_numpy(),
            'growth': growth, 'interaction': np.linalg.norm(components['interaction'], axis=1)}))
        if save_components:
            np.savez_compressed(output / f'velocity_time_{token}.npz',
                                features=values, **components)
    per_cell = pd.concat(rows, ignore_index=True)
    per_cell.to_csv(output / 'figure5e_growth_interaction_by_cell.csv', index=False)
    grouped = per_cell.groupby(['time', 'celltype'], as_index=False).agg(
        growth_mean=('growth', 'mean'), interaction_mean=('interaction', 'mean'),
        n=('growth', 'size'))
    grouped.to_csv(output / 'figure5e_growth_interaction_by_celltype.csv', index=False)
    (output / 'field_settings.json').write_text(json.dumps({
        'seed': seed, 'time_order': list(np.arange(0., 4.01, .5)),
        'interaction_group_size': 1024, 'device': str(device),
        'state_dir': str(Path(state_dir).resolve()),
        'model_dir': str(Path(model_dir).resolve()),
        'grouping': 'torch randperm per time, base 1024; nonzero remainder merged into final group',
        'save_components': bool(save_components)}, indent=2) + '\n')
    return grouped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--state-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(calculate_fields(args.model_dir, args.state_dir, args.output_dir, args.device, args.seed))
