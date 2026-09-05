"""Calculate the intrinsic-context gene-velocity field in Figure 5d."""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scvelo as scv
import torch

import CytoBridge as cb


def project_gene_velocity(gene_state, intrinsic_velocity):
    """Apply the Figure 5d PCA, neighbor graph and velocity projection."""
    gene_state = np.asarray(gene_state, dtype=np.float32)
    intrinsic_velocity = np.asarray(intrinsic_velocity, dtype=np.float32)
    if gene_state.shape != intrinsic_velocity.shape or gene_state.ndim != 2:
        raise ValueError('Gene states and intrinsic velocities must have matching cell-by-feature shapes.')
    if not np.isfinite(gene_state).all() or not np.isfinite(intrinsic_velocity).all():
        raise ValueError('Gene states and intrinsic velocities must be finite.')
    population = ad.AnnData(X=gene_state.copy())
    sc.tl.pca(population, n_comps=2, svd_solver='arpack', random_state=0)
    population.layers['spliced'] = gene_state.copy()
    population.layers['Ms'] = gene_state.copy()
    population.layers['velocity'] = intrinsic_velocity.copy()
    sc.pp.neighbors(population, n_neighbors=30, use_rep='X', random_state=0)
    scv.tl.velocity_graph(population, vkey='velocity', xkey='Ms',
                          n_jobs=1, show_progress_bar=False)
    scv.tl.velocity_embedding(population, basis='pca', vkey='velocity')
    return (np.asarray(population.obsm['X_pca'], dtype=np.float32),
            np.asarray(population.obsm['velocity_pca'], dtype=np.float32))


def calculate_gene_velocity(data_dir, output_dir, device='cuda'):
    """Evaluate the downloaded model and write the numerical Figure 5d input.

    This is the selected ``--velocity-component drift`` calculation from
    ``build_figure5d_package_native_gene_velocity_state.py``. It uses all
    observed cells, the 50 gene features and no spatial display transform.
    """
    data, output = Path(data_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    population = ad.read_h5ad(data / 'aligned.h5ad')
    times = population.obs['time_point_processed'].to_numpy(dtype=float)
    order = np.argsort(times, kind='stable')
    population = population[order].copy()
    times = times[order]
    genes = np.asarray(population.obsm['X_latent'], dtype=np.float32)
    spatial = np.asarray(population.obsm['spatial_aligned'], dtype=np.float32)
    state = np.concatenate([spatial, genes], axis=1)
    loaded = cb.tl.load_dynamical_model_from_dir(data / 'model', dim=state.shape[1], device=device)
    intrinsic = np.empty_like(genes)
    for time in np.unique(times):
        selected = np.isclose(times, time)
        x = torch.as_tensor(state[selected], dtype=torch.float32, device=device)
        t = torch.full((len(x), 1), float(time), dtype=torch.float32, device=device)
        with torch.no_grad():
            intrinsic[selected] = loaded.model.predict_velocity(t=t, x=x)[:, 2:].cpu().numpy()
    coordinates, embedded = project_gene_velocity(genes, intrinsic)
    output.mkdir(parents=True)
    path = output / 'figure5d_intrinsic_gene_velocity_state.npz'
    np.savez_compressed(path, corrected_raw_pca=coordinates,
                        embedded_gene_velocity_pca=embedded,
                        labels=population.obs['Annotation'].astype(str).to_numpy(dtype=str),
                        times=times, display_mask=np.ones(len(times), dtype=bool),
                        velocity_component=np.array('drift'))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/arista'))
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(calculate_gene_velocity(args.data_dir, args.output_dir, args.device))
