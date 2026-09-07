"""Simulate the ARISTA populations from the trained model and classify cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import yaml
import CytoBridge as cb


def generate(data_dir, output_dir, classifier_cache, device='cuda', *, model_dir=None):
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    population = ad.read_h5ad(data / 'aligned.h5ad')
    frame, time_key = cb.tl.adata_to_aligned_dataframe(
        population, time_key='time_point_processed', obsm_key='X_latent',
        spatial_key='spatial_aligned', concat_spatial=True, annotation_key='Annotation')
    selected_model = (data / 'model' if model_dir is None else Path(model_dir)).resolve()
    loaded = cb.tl.load_dynamical_model_from_dir(selected_model, dim=52, device=device)
    runtime = cb.tl.build_dynamical_runtime(loaded)
    output.mkdir(parents=True)
    result = cb.tl.run_interpolation_workflow(
        df=frame, dim=52, annotation_key='Annotation', runtime=runtime,
        device=device, output_dir=str(output),
        requested_plot_points=list(np.arange(0., 4.01, .5)),
        interp_time_points=[.5, 1.5, 2.5, 3.5], use_real_for_observed=True,
        classifier_cache_path=str(Path(classifier_cache).resolve()),
        classifier_adata=population, classifier_time_key=time_key,
        classifier_obsm_key='X_latent', classifier_spatial_key='spatial_aligned',
        classifier_concat_spatial=True, classifier_knn_neighbors=10,
        sde_n_samples=7668, sde_dt=.05, split_sde_dt=.01,
        split_sigma_scalar=.03, split_growth_alpha=1.,
        spatial_warp_to_observed=False,
        spatial_warp_to_observed_piecewise=False,
        random_seed=42,
        separate_interaction_random_stream=False)
    # The original downstream evaluates observed vector fields immediately
    # after simulation. Preserve that random-group continuation explicitly;
    # resetting to seed 42 would evaluate different interaction neighborhoods.
    rng = {'cpu': torch.get_rng_state().cpu().numpy()}
    if str(device).startswith('cuda'):
        rng['cuda'] = torch.cuda.get_rng_state(device).cpu().numpy()
    np.savez(output / 'post_simulation_rng.npz', **rng)
    for folder in ('display_states', 'model_states', 'generated_display_states', 'slice_data'):
        (output / folder).mkdir()
    records = []
    for index, time in enumerate(result.ts_points):
        token = f'{time:g}'.replace('.', 'p')
        result.adata_dict[str(time)].write_h5ad(
            output / f'display_states/time_{token}.h5ad', compression='gzip')
        # Figure 5 uses these same observed/intermediate populations. Export
        # their states directly, without another simulation or display warp.
        result.adata_dict[str(time)].write_h5ad(
            output / f'slice_data/time_{token}.h5ad', compression='gzip')
        result.communication_adata_dict[str(time)].write_h5ad(
            output / f'model_states/time_{token}.h5ad', compression='gzip')
        generated = ad.AnnData(X=np.asarray(result.sde_points_split[index], dtype=np.float32))
        generated.obsm['spatial'] = np.asarray(generated.X)[:, :2].copy()
        generated.obs['Annotation'] = np.asarray(result.slice_labels_split[index]).astype(str)
        generated.write_h5ad(output / f'generated_display_states/time_{token}.h5ad', compression='gzip')
        records.append({'time': float(time), 'observed_or_interpolated': result.adata_dict[str(time)].n_obs,
                        'generated': generated.n_obs})
    np.savez_compressed(output / 'fixed_particle_lineage_labels.npz',
                        time_points=np.asarray(result.ts_points),
                        **{f'labels_{i}': np.asarray(labels).astype(str)
                           for i, labels in enumerate(result.predicted_labels_list)})
    config_path = Path(cb.__file__).resolve().parent / 'configs/arista_downstream.yaml'
    communication_settings = yaml.safe_load(config_path.read_text())['communication']
    cb.tl.compute_timepoint_communications(
        adata_dict=result.communication_adata_dict,
        time_points=result.ts_points, annotation_key='Annotation',
        f_net=runtime.f_net, device=device, out_dir=str(output / 'attention'),
        remove_self_loop=communication_settings['remove_self_loop'],
        winsor_quantile=communication_settings['winsor_quantile'],
        save_pickle_path=str(output / 'all_time_communications.pkl'))
    (output / 'communication_settings.json').write_text(
        json.dumps(communication_settings, indent=2) + '\n')
    (output / 'population_sizes.json').write_text(json.dumps(records, indent=2) + '\n')
    (output / 'model_selection.json').write_text(json.dumps({
        'model_dir': str(selected_model),
        'classifier_cache': str(Path(classifier_cache).resolve()),
        'aligned_h5ad': str(data / 'aligned.h5ad')}, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/arista'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--classifier-cache', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    print(generate(args.data_dir, args.output_dir, args.classifier_cache, args.device,
                   model_dir=args.model_dir))
