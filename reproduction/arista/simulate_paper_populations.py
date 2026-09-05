"""Generate the ARISTA populations used for the paper's spatial displays.

Spatial anchoring changes display coordinates only. Cell labels, lineage and
communication use the model states before that transform.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import CytoBridge as cb


def generate(data_dir, output_dir, classifier_cache, device='cuda'):
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    population = ad.read_h5ad(data / 'aligned.h5ad')
    frame, time_key = cb.tl.adata_to_aligned_dataframe(
        population, time_key='time_point_processed', obsm_key='X_latent',
        spatial_key='spatial_aligned', concat_spatial=True, annotation_key='Annotation')
    loaded = cb.tl.load_dynamical_model_from_dir(data / 'model', dim=52, device=device)
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
        spatial_warp_to_observed_piecewise=True,
        spatial_warp_visualization_only=True, spatial_warp_k=1,
        spatial_warp_eps=1e-6, random_seed=42,
        separate_interaction_random_stream=False)
    for folder in ('display_states', 'model_states', 'generated_display_states'):
        (output / folder).mkdir()
    records = []
    for index, time in enumerate(result.ts_points):
        token = f'{time:g}'.replace('.', 'p')
        result.adata_dict[str(time)].write_h5ad(
            output / f'display_states/time_{token}.h5ad', compression='gzip')
        result.communication_adata_dict[str(time)].write_h5ad(
            output / f'model_states/time_{token}.h5ad', compression='gzip')
        generated = ad.AnnData(X=np.asarray(result.sde_points_split_prewarp[index], dtype=np.float32))
        generated.obsm['spatial'] = np.asarray(result.sde_points_split[index], dtype=np.float32)[:, :2]
        generated.obs['Annotation'] = np.asarray(result.slice_labels_split[index]).astype(str)
        generated.write_h5ad(output / f'generated_display_states/time_{token}.h5ad', compression='gzip')
        records.append({'time': float(time), 'observed_or_interpolated': result.adata_dict[str(time)].n_obs,
                        'generated': generated.n_obs})
    np.savez_compressed(output / 'fixed_particle_lineage_labels.npz',
                        time_points=np.asarray(result.ts_points),
                        **{f'labels_{i}': np.asarray(labels).astype(str)
                           for i, labels in enumerate(result.predicted_labels_list)})
    (output / 'population_sizes.json').write_text(json.dumps(records, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/arista'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--classifier-cache', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(generate(args.data_dir, args.output_dir, args.classifier_cache, args.device))
