"""Simulate cell populations, assign cell types, and calculate interactions for the supplementary videos."""
from pathlib import Path
import argparse
import json

import anndata as ad
import numpy as np

SETTINGS = {
    'mosta': dict(end=3., step=.05, n=50000, dt=.05, k=10,
                  annotation='Annotation', classifier='classifier_resmlp_6d2d7acf7d0ed92d.pt'),
    'arista': dict(end=2., step=.01, n=7668, dt=.01, k=10,
                   annotation='Annotation', classifier='classifier_resmlp_dedb1d6442f4d3d3.pt'),
    'admouse': dict(end=2., step=.02, n=None, dt=.01, k=1,
                    annotation='major_annotation', classifier='classifier_resmlp_46ee959d0b1f14db.pt'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=SETTINGS)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='Dataset directory containing aligned.h5ad, model/ and classifier_cache/.')
    parser.add_argument('--model-dir', type=Path, help='Override data-dir/model.')
    parser.add_argument('--classifier', type=Path, help='Override the dataset classifier checkpoint.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    import CytoBridge as cb
    cfg = SETTINGS[args.dataset]
    source = args.data_dir.resolve()
    model = (args.model_dir or source / 'model').resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # X contains the full count matrix, which is not an inference input.
    # Read only metadata and the saved, correctly preprocessed model features.
    population = ad.read_h5ad(source / 'aligned.h5ad', backed='r')
    frame, time_key = cb.tl.adata_to_aligned_dataframe(
        population, time_key='time_point_processed', obsm_key='X_latent',
        spatial_key='spatial_aligned', concat_spatial=True,
        annotation_key=cfg['annotation'])
    compact = ad.AnnData(obs=population.obs.copy())
    compact.obsm['X_latent'] = np.asarray(population.obsm['X_latent']).copy()
    compact.obsm['spatial_aligned'] = np.asarray(population.obsm['spatial_aligned']).copy()
    population.file.close()
    # The submitted ARISTA video shows the first three stages, not all five.
    frame = frame.loc[frame.samples <= cfg['end']].copy()
    times = np.round(np.arange(0, cfg['end'] + cfg['step']/2, cfg['step']), 8)
    classifier = (args.classifier or source / 'classifier_cache' / cfg['classifier']).resolve()
    if not classifier.is_file():
        raise FileNotFoundError(classifier)
    loaded = cb.tl.load_dynamical_model_from_dir(model, dim=52, device=args.device)
    runtime = cb.tl.build_dynamical_runtime(loaded)
    result = cb.tl.run_interpolation_workflow(
        df=frame, dim=52, annotation_key=cfg['annotation'], runtime=runtime,
        device=args.device, output_dir=str(args.output / 'inference'),
        requested_plot_points=times.tolist(),
        interp_time_points=[float(t) for t in times if not float(t).is_integer()],
        use_real_for_observed=False,
        classifier_cache_path=str(classifier),
        classifier_adata=compact, classifier_time_key=time_key,
        classifier_obsm_key='X_latent', classifier_spatial_key='spatial_aligned',
        classifier_concat_spatial=True, classifier_knn_neighbors=cfg['k'],
        sde_n_samples=cfg['n'], sde_dt=.05, split_sde_dt=cfg['dt'],
        skip_nonsplit_sde=args.dataset != 'arista',
        split_sigma_scalar=.03, split_growth_alpha=1.,
        split_daughter_noise_std=0., split_interaction_m=1024,
        split_sde_piecewise=False,
        spatial_warp_to_observed=False,
        spatial_warp_to_observed_piecewise=False,
        random_seed=42,
        separate_interaction_random_stream=args.dataset != 'arista')
    state_dir = args.output / 'states'
    state_dir.mkdir()
    records = []
    communication = {}
    for index, time in enumerate(result.ts_points):
        points = np.asarray(result.sde_points_split[index], dtype=np.float32)
        labels = np.asarray(result.slice_labels_split[index]).astype(str)
        if len(labels) != len(points) or not np.isfinite(points).all():
            raise ValueError(f'Invalid states at {time}')
        np.savez_compressed(state_dir / f'frame_{index:03d}.npz',
                            time=float(time), points=points, labels=labels)
        records.append({'time': float(time), 'n': len(points)})
        if args.dataset != 'admouse':
            # Evaluate learned directed attention on the generated population.
            # MOSTA uses the 12k sampling size of its formal communication run.
            cells = ad.AnnData(X=points)
            cells.obsm['spatial'] = points[:, :2].copy()
            cells.obs[cfg['annotation']] = labels
            comm = cb.tl.compute_timepoint_communications(
                adata_dict={str(time): cells}, time_points=[time],
                annotation_key=cfg['annotation'], f_net=runtime.f_net,
                device=args.device, out_dir=str(args.output / 'attention'),
                max_cells_per_timepoint=12000 if args.dataset == 'mosta' else None,
                random_seed=42, remove_self_loop=False, winsor_quantile=.995)
            # Keep the aggregation needed for arrows, not a pickled plotting object.
            c = comm[str(time)]
            communication[str(index)] = {
                'types': list(map(str, c['types'])),
                'M_per_source': np.asarray(c['M_per_source']).tolist()}
        print(f'Exported {args.dataset} frame {index+1}/{len(times)}: {len(points)} cells', flush=True)
    (args.output / 'communication.json').write_text(json.dumps(communication))
    manifest = {
        'dataset': args.dataset, 'source': str(source),
        'model': str(model), 'classifier': str(classifier),
        'settings': cfg, 'time_points': list(map(float, result.ts_points)),
        'populations': records, 'package': cb.__file__,
        'continuous_from_initial_stage': True, 'spatial_warp': False,
        'observed_snapshots_substituted': False,
        'random_seed': 42, 'diffusion': .03, 'growth_alpha': 1.,
        'separate_interaction_random_stream': args.dataset != 'arista',
        'note': 'Dense animation evaluation of the accepted checkpoint. No model fitting.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
