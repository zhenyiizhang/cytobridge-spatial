"""Calculations behind the Figure 4 interaction and velocity panels."""
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import CytoBridge as cb

from .main_figure import PANELS


def interaction_scores(edges):
    """Wnt3a × Fzd7/Lrp6 expression × directed communication weight.

    Receptor means in this table were calculated using the minimum expression
    of the two subunits. Rows are directed cell-type pairs at each time.
    """
    pairs = edges.copy()
    pairs['lr_score'] = (pairs.ligand_mean * pairs.receptor_mean *
                         pairs.communication_weight)
    incoming = pairs.groupby(['time', 'receiver_type']).lr_score.sum(min_count=1)
    outgoing = pairs.groupby(['time', 'sender_type']).lr_score.sum(min_count=1)
    incoming.index.names = outgoing.index.names = ['time', 'cell_type']
    scores = pd.concat([incoming.rename('incoming'), outgoing.rename('outgoing')], axis=1)
    scores['total'] = scores.incoming + scores.outgoing
    return scores.reset_index()


def map_interaction_scores(cells, scores):
    """Assign the type-level result to its cells on the spatial map."""
    result = cells.drop(columns=['incoming', 'outgoing', 'total_raw',
                                 'cell_type_score_available'], errors='ignore')
    result = result.merge(scores, on=['time', 'cell_type'], how='left',
                          validate='many_to_one', sort=False)
    result = result.rename(columns={'total': 'total_raw'})
    result['cell_type_score_available'] = result.total_raw.notna()
    return result


def save_population_states(populations, output_dir):
    """Save simulated/observed AnnData states for the Figure 4a renderer."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for time, cells in populations.items():
        name = f'{float(time):g}'.replace('.', 'p')
        cells.write_h5ad(output / f'time_{name}.h5ad')
    return output


def cartilage_lineage_inputs(states, labels, times, reference, *,
                            source_time=2.5, target_time=3.):
    """Select the same particles at E15 and E15.5, with observed background.

    ``states`` and ``labels`` are returned by ``simulate_sde_points`` and
    ``predict_labels_for_trajectories``. Rows must retain particle identities.
    """
    times = np.asarray(times, dtype=float)
    hits = [np.flatnonzero(np.isclose(times, time, rtol=0, atol=1e-10))
            for time in (source_time, target_time)]
    if any(len(hit) != 1 for hit in hits):
        raise ValueError('Source and target times must each occur once.')
    start, end = (int(hit[0]) for hit in hits)
    if start >= end:
        raise ValueError('Target time must follow source time.')
    source, target = np.asarray(states[start]), np.asarray(states[end])
    source_labels, target_labels = np.asarray(labels[start]).astype(str), np.asarray(labels[end]).astype(str)
    if len(source) != len(target) or len(source_labels) != len(source) or len(target_labels) != len(target):
        raise ValueError('Lineage analysis requires the same particles and one label per particle.')
    selected = source_labels == 'Cartilage primordium'
    if not selected.any():
        raise ValueError('No cartilage-primordium particles at the source time.')
    observed = np.isclose(reference.obs['time_point_processed'].to_numpy(dtype=float), target_time)
    if not observed.any():
        raise ValueError('No observed population at the target time.')
    return {
        'source_background_spatial': source[:, :2],
        'selected_source_spatial': source[selected, :2],
        'target_spatial': target[selected, :2],
        'target_labels': target_labels[selected],
        'observed_target_spatial': np.asarray(reference.obsm['spatial_aligned'])[observed, :2],
        'selected_lineage_id': np.flatnonzero(selected),
    }


def calculate_velocity_panel(model_dir, output_dir, panel, device='cuda', *, overwrite=False):
    """Evaluate the model on the exact observed cells used for panel d or e.

    The archived input supplies cell identities, 52D states and annotations.
    All drift, interaction, score and projected fields are recalculated here.
    """
    if panel not in ('d', 'e'):
        raise ValueError('Choose panel d or e.')
    output = Path(output_dir).resolve()
    for protected in (PANELS.resolve(), Path(model_dir).resolve()):
        if output == protected or protected in output.parents:
            raise ValueError('Save calculations outside the paper inputs and model directory.')
    if output.exists() and not overwrite:
        raise FileExistsError(f'Choose a new output directory: {output}')
    with np.load(PANELS / f'fig4{panel}/evidence/numeric_inputs.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    features = arrays['features']
    coordinates = arrays['compute_spatial']
    labels = (arrays['compute_labels'] if panel == 'd'
              else arrays['telencephalon_notebook_labels'])
    loaded = cb.tl.load_dynamical_model_from_dir(model_dir, dim=52, device=device)
    cb.tl.set_global_random_seed(42)
    components = cb.tl.compute_velocity_components(
        features, 3., loaded.model, interaction_m=8000 if panel == 'd' else 1024,
        interaction_threshold=loaded.model.interaction_net.cutoff, device=device)
    arrays.update(components)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('full', 'interaction'):
        cb.tl.set_global_random_seed(42)
        projected = cb.pl.plot_velocity_component(
            coords=coordinates, velocity=components[name][:, 2:],
            feature_matrix=features[:, 2:], labels=labels, n_neighbors=30,
            density=2. if panel == 'd' else 1., basis='spatial', show=False)
        arrays[f'gene_{name}_projected_spatial'] = np.asarray(
            projected.obsm['velocity_spatial'], dtype=np.float32)
        arrays[f'physical_{name}'] = components[name][:, :2]
    numeric_path = output / 'numeric_inputs.npz'
    np.savez_compressed(numeric_path, **arrays)
    if panel == 'd':
        cells = ad.AnnData(X=features.copy())
        cells.obs['Annotation'] = labels
        cells.obsm['spatial'] = coordinates
        cb.tl.set_global_random_seed(42)
        runtime = cb.tl.build_dynamical_runtime(loaded)
        communication = cb.tl.compute_timepoint_communications(
            adata_dict={'3.0': cells}, time_points=(3.,), annotation_key='Annotation',
            f_net=runtime.f_net, device=device, out_dir=str(output / 'attention'),
            remove_self_loop=True, winsor_quantile=.995, max_cells_per_timepoint=None)
        result = communication['3.0']
        types = list(result['types'])
        weights = np.asarray(result['M_per_source'])
        table = pd.DataFrame([
            {'source': s, 'target': t, 'is_same_type': s == t,
             'weight_per_source': weights[i, j]}
            for i, s in enumerate(types) for j, t in enumerate(types)])
        table.to_csv(output / 'communication_all_type_edges.csv.gz', index=False)
    return numeric_path
