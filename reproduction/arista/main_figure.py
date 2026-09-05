"""Draw ARISTA Figure 5 from the released numerical inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting

SOURCE = Path(__file__).parent / 'data'
TIMES = (0., .5, 1., 1.5, 2.)
ROI = (-.0959183471277356, .7083190018311143, -.9299020797014237, -.239640411734581)
FOCUS = (.356063043, .629503742, -.856734343, -.683478664)
GENE_TYPES = ('cckIN', 'dpEX', 'mpEX', 'mpIN', 'nptxEX', 'npyIN', 'ntng1IN',
              'rIPC1', 'rIPC2', 'rIPC4', 'reaEGC', 'ribEGC', 'scgnIN', 'sfrpEGC',
              'sstIN', 'wntEGC')


def load_populations(data_dir):
    return {str(time): ad.read_h5ad(Path(data_dir) / 'display_states' /
                                  f"time_{f'{time:g}'.replace('.', 'p')}.h5ad")
            for time in TIMES}


def display_coordinates(coordinates, time):
    """Map spatial coordinates to the paper's per-slice display canvas.

    This is the affine axes transform of the 4.2-inch spatial panels, calculated
    directly from their coordinates. It does not read a finished figure.
    """
    import matplotlib as mpl
    with mpl.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10}):
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=72)
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=2.5, linewidths=0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        title = f't = {time:.1f}' + (' (Observed)' if time in (0., 1., 2., 3., 4.) else '')
        ax.set_title(title, fontsize=12, pad=6)
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer()).padded(.1)
        result = ax.transData.transform(coordinates).copy()
        result[:, 0] -= bounds.x0 * 72
        result[:, 1] -= bounds.y1 * 72
        plt.close(fig)
    return result


def draw_stack(data_dir, output, palette):
    """Recalculate spatial anchors and lineage/communication drawing paths."""
    from scripts.reviewer_arista_20260824.arista_helpers_focus_anchor import (
        plot_3d_spatial_sankey_style_focus_anchor,
    )
    populations = load_populations(data_dir)
    for time, population in populations.items():
        population.obsm['spatial'] = display_coordinates(
            np.asarray(population.obsm['spatial']), float(time))
    with (data_dir / 'all_time_communications.pkl').open('rb') as handle:
        communication = pickle.load(handle)
    with np.load(data_dir / 'fixed_particle_lineage_labels.npz', allow_pickle=False) as state:
        labels = []
        for time in TIMES:
            index = np.flatnonzero(np.isclose(state['time_points'], time))
            if len(index) != 1:
                raise ValueError(f'The lineage input has no unique time {time}.')
            labels.append(state[f'labels_{index[0]}'].astype(str))
    xy = np.concatenate([np.asarray(a.obsm['spatial']) for a in populations.values()])
    settings = json.loads((Path(__file__).parent / 'stack_style.json').read_text())
    settings.update(bidirectional_offset=.06 * np.linalg.norm(np.ptp(xy, axis=0)),
                    observed_time_points=[0., 1., 2.], generated_time_points=[.5, 1.5],
                    width=1800, height=1350, show_time_axis=True, show_legend=True,
                    font_color='black')
    fig = plot_3d_spatial_sankey_style_focus_anchor(
        populations, communication, [str(t) for t in TIMES], palette, labels, **settings)
    fig.update_layout(scene_camera={'eye': {'x': 1.7, 'y': 1., 'z': .9},
                                    'projection': {'type': 'orthographic'}},
                      scene={'aspectratio': {'x': 1.2, 'y': 1., 'z': 1.6}},
                      font={'family': 'Arial', 'size': 16, 'color': 'black'})
    paths = [output / 'Figure5a_spatiotemporal_map.pdf', output / 'Figure5a_spatiotemporal_map.png']
    fig.write_html(output / 'Figure5a_spatiotemporal_map.html')
    fig.write_image(str(paths[0]))
    fig.write_image(str(paths[1]), scale=2)
    pd.DataFrame({'time': TIMES, 'cells': [a.n_obs for a in populations.values()]}).to_csv(
        output / 'Figure5a_population_counts.csv', index=False)
    return paths


def draw_generated_population(data_dir, output, palette):
    population = ad.read_h5ad(data_dir / 'display_states/time_0p5.h5ad')
    xy = np.asarray(population.obsm['spatial'])
    table = pd.DataFrame({'x': xy[:, 0], 'y': xy[:, 1],
                          'celltype': population.obs['Annotation'].astype(str).to_numpy(),
                          'displayed_point_glyph': True})
    return plotting.plot_figure5b(table, palette, output / 'Figure5b_generated_population')


def draw_spatial_velocity(output, palette):
    """Recalculate the display grid and full-versus-interaction cosine values."""
    table = pd.read_csv(SOURCE / 'figure5c_all_cells_velocity.csv')
    with np.load(SOURCE / 'figure5c_embedded_velocity.npz', allow_pickle=False) as state:
        coordinates = state['manuscript_display_coordinates']
        velocity = state['manuscript_display_direct_embedded_velocity']
        full = state['full_embedded_velocity']
        interaction = state['interaction_embedded_velocity']
    norm = np.linalg.norm(full, axis=1) * np.linalg.norm(interaction, axis=1)
    cosine = np.divide(np.einsum('ij,ij->i', full, interaction), norm,
                       out=np.zeros(len(norm)), where=norm > 1e-12)
    np.testing.assert_allclose(cosine, table.cosine_full_vs_interaction, atol=2e-6)
    table['cosine_full_vs_interaction'] = cosine
    table.to_csv(output / 'Figure5c_spatial_velocity.csv', index=False)
    population = ad.AnnData(X=np.zeros((len(table), 1), dtype=np.float32))
    population.obs['Annotation'] = table.celltype.astype(str).to_numpy()
    population.obsm['X_spatial'] = coordinates
    population.obsm['velocity_spatial'] = velocity
    result = plotting.plot_figure5c(population, table, ROI, FOCUS, palette, output)
    return result['Figure5c_spatial_and_roi']


def draw_gene_velocity(output, palette):
    """Calculate a stream grid from all full gene-velocity vectors."""
    import scvelo as scv
    from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid
    with np.load(SOURCE / 'figure5d_corrected_gene_velocity_state.npz', allow_pickle=False) as state:
        coordinates = state['corrected_raw_pca'].copy()
        velocity = state['embedded_gene_velocity_pca'].copy()
        labels = state['labels'].astype(str)
        display = state['display_mask'].copy()
    x_grid, v_grid = compute_velocity_on_grid(
        X_emb=coordinates, V_emb=velocity, density=1, smooth=None, min_mass=None,
        n_neighbors=None, autoscale=False, adjust_for_stream=True, cutoff_perc=None)
    speed = np.sqrt(np.sum(v_grid**2, axis=0))
    # scVelo masks grid locations without enough nearby cells with NaN.
    # Keep that mask in v_grid, but give the undrawn segments a finite width
    # so Matplotlib can export the stream collection as a vector PDF.
    linewidth = np.nan_to_num(2 * speed / np.nanmax(speed), nan=0.)
    np.savez_compressed(output / 'Figure5d_stream_grid.npz',
                        coordinates=x_grid, velocity=v_grid, linewidth=linewidth)
    categories = list(GENE_TYPES) + ['Other']
    palette = dict(palette, Other='#D0D0D0')
    celltypes = np.where(np.isin(labels, GENE_TYPES), labels, 'Other')
    population = ad.AnnData(X=np.zeros((display.sum(), 1), dtype=np.float32))
    population.obsm['X_pca'] = coordinates[display]
    population.obsm['velocity_pca'] = velocity[display]
    population.obs['celltype'] = pd.Categorical(celltypes[display], categories=categories)
    population.uns['celltype_colors'] = [palette[label] for label in categories]
    ax = scv.pl.velocity_embedding_stream(
        population, basis='pca', vkey='velocity', color='celltype', density=2,
        smooth=None, min_mass=None, cutoff_perc=None, arrow_color='black',
        arrow_size=1, arrow_style='-|>', max_length=4, integration_direction='both',
        linewidth=linewidth, n_neighbors=None, recompute=False,
        palette=[palette[label] for label in categories], size=None, alpha=.3,
        X_grid=x_grid, V_grid=v_grid, sort_order=True, legend_loc='right',
        title='Gene velocity', figsize=(6, 6), frameon=None, marker='.', show=False)
    paths = plotting.save_figure(ax.figure, output / 'Figure5d_gene_velocity')
    plt.close(ax.figure)
    return paths


def draw_growth_interaction(output):
    """Calculate one mean growth/interaction point per time and cell type."""
    table = pd.read_csv(SOURCE / 'figure5e_growth_interaction_by_cell.csv')
    grouped = table.groupby(['time', 'celltype'], as_index=False).agg(
        growth_mean=('growth', 'mean'), interaction_mean=('interaction', 'mean'), n=('growth', 'size'))
    grouped['time_idx'] = grouped.time.map({t: i for i, t in enumerate(sorted(grouped.time.unique()))})
    grouped.to_csv(output / 'Figure5e_growth_interaction.csv', index=False)
    return plotting.plot_figure5e(grouped, output / 'Figure5e_growth_interaction')


def draw_main_figure(data_dir: str | Path, output_dir: str | Path, panels='abcde'):
    """Draw Figure 5 panels from numerical files, without reading a finished image."""
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output == data or data in output.parents or SOURCE.resolve() in output.parents:
        raise ValueError('Choose an output directory outside the input data.')
    output.mkdir(parents=True, exist_ok=True)
    palette = json.loads((SOURCE / 'label_to_color.json').read_text())
    result = {}
    for panel in panels:
        plotting.configure_style()
        if panel == 'a':
            result[panel] = draw_stack(data, output, palette)
        elif panel == 'b':
            result[panel] = draw_generated_population(data, output, palette)
        elif panel == 'c':
            result[panel] = draw_spatial_velocity(output, palette)
        elif panel == 'd':
            result[panel] = draw_gene_velocity(output, palette)
        elif panel == 'e':
            result[panel] = draw_growth_interaction(output)
        else:
            raise ValueError('Choose Figure 5 panels a, b, c, d, or e.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/arista/paper'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--panels', default='abcde')
    args = parser.parse_args()
    for panel, paths in draw_main_figure(args.data_dir, args.output_dir, args.panels).items():
        print(panel, [str(path) for path in paths])
