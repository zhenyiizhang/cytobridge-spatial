"""Project model velocities for the selected Figure 5c analysis."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvelo as scv


def project_velocity(features, coordinates, velocity):
    population = ad.AnnData(X=np.asarray(features, dtype=np.float32))
    population.layers['Ms'] = np.asarray(features, dtype=np.float32)
    population.layers['velocity'] = np.asarray(velocity, dtype=np.float32)
    population.obsm['X_spatial'] = np.asarray(coordinates, dtype=np.float32)
    sc.pp.neighbors(population, n_neighbors=30, use_rep='X', random_state=0)
    scv.tl.velocity_graph(population, vkey='velocity', xkey='Ms',
                          n_jobs=1, show_progress_bar=False)
    scv.tl.velocity_embedding(population, basis='spatial', vkey='velocity')
    return np.asarray(population.obsm['velocity_spatial'], dtype=np.float32)


def calculate_spatial_velocity(velocity_path, output_dir):
    """Continue from ``model_fields`` at model time 1 (5 DPI).

    The left field uses the first two spatial components. The selected paper
    inset compares the full and interaction 52D fields after independently
    projecting them onto the spatial coordinates. The coordinate reference
    fixes the displayed tissue orientation and the grey ROI only.
    """
    from .main_figure import SOURCE, ROI
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    reference = pd.read_csv(SOURCE / 'figure5c_all_cells_velocity.csv')
    with np.load(velocity_path, allow_pickle=False) as state:
        features, full, interaction = [state[key] for key in ('features', 'full', 'interaction')]
    coordinates = features[:, :2]
    if len(coordinates) != len(reference) or not np.allclose(
            coordinates, reference[['current_x', 'current_y']], atol=1e-6):
        raise ValueError('Figure 5c needs the observed 5 DPI cells in their original row order.')
    direct = project_velocity(coordinates, coordinates, full[:, :2])
    full_embedded = project_velocity(features, coordinates, full)
    interaction_embedded = project_velocity(features, coordinates, interaction)
    denominator = np.linalg.norm(full_embedded, axis=1) * np.linalg.norm(interaction_embedded, axis=1)
    cosine = np.divide(np.sum(full_embedded * interaction_embedded, axis=1), denominator,
                       out=np.zeros(len(coordinates)), where=denominator > 0)
    cosine = np.clip(cosine, -1., 1.)
    paper = reference[['paper_x', 'paper_y']].to_numpy()
    p = paper - paper.mean(axis=0)
    c = coordinates - coordinates.mean(axis=0)
    u, singular, vt = np.linalg.svd(p.T @ c)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    rotation = u @ vt
    scale = singular.sum() / np.square(p).sum()
    displayed_velocity = direct @ rotation.T / scale
    output.mkdir(parents=True)
    for prefix, values in (
        ('direct_spatial', full[:, :2]),
        ('direct_embedded', direct),
        ('full_embedded', full_embedded),
        ('interaction_embedded', interaction_embedded),
    ):
        reference[f'{prefix}_vx'] = values[:, 0]
        reference[f'{prefix}_vy'] = values[:, 1]
    reference['display_direct_embedded_vx_paper_basis'] = displayed_velocity[:, 0]
    reference['display_direct_embedded_vy_paper_basis'] = displayed_velocity[:, 1]
    reference = reference.drop(columns=['in_nested_red_display_annotation'], errors='ignore')
    reference['cosine_full_vs_interaction'] = cosine
    reference['in_roi'] = ((paper[:, 0] >= ROI[0]) & (paper[:, 0] <= ROI[1]) &
                           (paper[:, 1] >= ROI[2]) & (paper[:, 1] <= ROI[3]))
    reference.to_csv(output / 'figure5c_all_cells_velocity.csv', index=False)
    np.savez_compressed(output / 'figure5c_embedded_velocity.npz',
                        manuscript_display_coordinates=paper,
                        manuscript_display_direct_embedded_velocity=displayed_velocity,
                        full_embedded_velocity=full_embedded,
                        interaction_embedded_velocity=interaction_embedded,
                        cosine=cosine)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--velocity-path', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(calculate_spatial_velocity(args.velocity_path, args.output_dir))
