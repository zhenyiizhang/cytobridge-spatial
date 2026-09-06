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
    # Use the coordinates produced by the simulation for every generated map.
    populations = {str(time): ad.read_h5ad(Path(data_dir) / 'slice_data' /
                                         f"time_{f'{time:g}'.replace('.', 'p')}.h5ad")
                   for time in TIMES}
    for population in populations.values():
        population.obsm['spatial'] = np.asarray(population.X)[:, :2].copy()
    return populations


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
                    width=int(11.69 * 300), height=int(8.27 * 300),
                    show_time_axis=False, show_legend=False,
                    font_color='black')
    fig = plot_3d_spatial_sankey_style_focus_anchor(
        populations, communication, [str(t) for t in TIMES], palette, labels, **settings)
    fig.update_layout(scene_camera={'eye': {'x': 1.7, 'y': 1., 'z': .9},
                                    'projection': {'type': 'orthographic'}},
                      margin=dict(l=10, r=10, t=10, b=10),
                      scene={'domain': {'x': [0., 1.], 'y': [0., 1.]},
                             'aspectratio': {'x': 1.2, 'y': 1., 'z': 1.6}},
                      font={'family': 'Arial', 'size': 16, 'color': 'black'})
    for trace in fig.data:
        if trace.type != 'scatter3d':
            continue
        if trace.mode == 'lines' and trace.hoverinfo == 'skip' and len(trace.x) == 5:
            continue  # slice borders stay on their planes
        trace.z = [None if z is None else float(z) + .04 for z in trace.z]
    paths = [output / 'Figure5a_spatiotemporal_map.pdf', output / 'Figure5a_spatiotemporal_map.png']
    fig.write_html(output / 'Figure5a_spatiotemporal_map.html')
    core = output / 'Figure5a_calculated_core.png'
    fig.write_image(str(core), scale=2)
    _place_stack_in_paper_layout(core, paths)
    pd.DataFrame({'time': TIMES, 'cells': [a.n_obs for a in populations.values()]}).to_csv(
        output / 'Figure5a_population_counts.csv', index=False)
    return paths


def _place_stack_in_paper_layout(core_path, paths):
    """Apply the archived Figure 5a canvas placement to a freshly drawn core."""
    import fitz
    from PIL import Image
    from io import BytesIO
    transform = np.array([[1.16519507, -.00272857794, -1326.42311],
                          [-.00257634855, 1.17106427, -26.494962], [0., 0., 1.]])
    with Image.open(core_path) as image:
        image = image.convert('RGB').resize((7014, 4962), Image.Resampling.LANCZOS)
        image = image.transform((5723, 5761), Image.Transform.AFFINE,
                                 tuple(np.linalg.inv(transform)[:2].ravel()),
                                 resample=Image.Resampling.BICUBIC, fillcolor='white')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
    # This PDF contains labels and arrows only. Its former population image
    # and all lower panels were removed when this layout asset was archived.
    document = fitz.open(SOURCE / 'figure5a_labels.pdf')
    page = document[0]
    page.insert_image(fitz.Rect(23.7819900513, 14.293762207, 441.495361328, 434.780700684),
                      stream=buffer.getvalue(), overlay=False)
    document.save(paths[0], garbage=4, deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(paths[1])
    document.close()


def draw_generated_population(data_dir, output, palette):
    population = ad.read_h5ad(data_dir / 'slice_data/time_0p5.h5ad')
    xy = np.asarray(population.X)[:, :2]
    table = pd.DataFrame({'x': xy[:, 0], 'y': xy[:, 1],
                          'celltype': population.obs['Annotation'].astype(str).to_numpy(),
                          'displayed_point_glyph': True})
    table.to_csv(output / 'Figure5b_cells.csv', index=False)
    return plotting.plot_figure5b(table, palette, output / 'Figure5b_generated_population')


def draw_spatial_velocity(output, palette, state_dir=None):
    """Recalculate the display grid and full-versus-interaction cosine values."""
    source = SOURCE if state_dir is None else Path(state_dir)
    table = pd.read_csv(source / 'figure5c_all_cells_velocity.csv')
    with np.load(source / 'figure5c_embedded_velocity.npz', allow_pickle=False) as state:
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


def draw_gene_velocity(output, palette, state_path=None):
    """Draw the intrinsic-context gene velocity selected for Figure 5d."""
    import scvelo as scv
    from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid
    path = SOURCE / 'figure5d_intrinsic_gene_velocity_state.npz' if state_path is None else Path(state_path)
    with np.load(path, allow_pickle=False) as state:
        if str(state['velocity_component']) != 'drift':
            raise ValueError('Figure 5d requires intrinsic drift, not full velocity.')
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
        X_grid=x_grid, V_grid=v_grid, sort_order=True, legend_loc='none',
        title='Gene velocity', figsize=(6, 6), frameon=None, marker='.', show=False)
    from matplotlib import patheffects
    placements = pd.read_csv(SOURCE / 'figure5d_label_layout.csv')
    for row in placements.itertuples(index=False):
        subset = coordinates[display & (celltypes == row.label)]
        if not len(subset):
            continue
        anchor = np.median(subset, axis=0)
        ax.annotate(row.label, xy=anchor,
                    xytext=(row.offset_x_points, row.offset_y_points),
                    textcoords='offset points', ha='center', va='center',
                    fontsize=11, weight='bold', color='black',
                    path_effects=[patheffects.withStroke(linewidth=1.8, foreground='white')])
    for end in ((.12, .08), (.03, .18)):
        ax.annotate('', xy=end, xytext=(.03, .08), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.))
    ax.text(.048, .175, 'PC1', transform=ax.transAxes, fontsize=9, color='black')
    ax.text(.067, .09, 'PC2', transform=ax.transAxes, fontsize=9, color='black')
    paths = plotting.save_figure(ax.figure, output / 'Figure5d_gene_velocity')
    plt.close(ax.figure)
    return paths


def draw_growth_interaction(output, table_path=None):
    """Calculate one mean growth/interaction point per time and cell type."""
    path = SOURCE / 'figure5e_growth_interaction_by_cell.csv' if table_path is None else Path(table_path)
    table = pd.read_csv(path)
    grouped = table.groupby(['time', 'celltype'], as_index=False).agg(
        growth_mean=('growth', 'mean'), interaction_mean=('interaction', 'mean'), n=('growth', 'size'))
    grouped['time_idx'] = grouped.time.map({t: i for i, t in enumerate(sorted(grouped.time.unique()))})
    grouped.to_csv(output / 'Figure5e_growth_interaction.csv', index=False)
    from .growth_plot import plot_growth_interaction
    return plot_growth_interaction(grouped, output / 'Figure5e_growth_interaction')


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
