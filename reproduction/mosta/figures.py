"""Draw MOSTA supplementary figures from cell states and numerical tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'release_artifacts/mosta_package_native_corrected_20260826_v1/reproduction'
PALETTE_FILE = SOURCE / 'main_fig4_panels/style_authority/label_to_color.json'
TIMES = tuple(float(t) for t in np.arange(0, 3.0001, .25))
CELL_TYPES = (
    'Brain', 'Connective tissue', 'Cavity', 'Epidermis', 'Muscle', 'Jaw and tooth',
    'Meninges', 'Liver', 'Cartilage primordium', 'Spinal cord', 'Heart', 'GI tract',
    'Dorsal root ganglion', 'Cartilage', 'Adipose tissue',
)


def style():
    mpl.rcParams.update({
        'font.family': 'Arial', 'font.size': 9, 'text.color': 'black',
        'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
    })


def save(fig, output: Path, stem: str):
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f'{stem}.pdf', output / f'{stem}.png']
    fig.savefig(paths[0], bbox_inches='tight')
    fig.savefig(paths[1], bbox_inches='tight', dpi=300)
    plt.close(fig)
    return paths


def token(time):
    return f'{time:g}'.replace('.', 'p')


def draw_spatial_states(shared: Path, output: Path):
    """S11: draw eight observed/generated spatial populations."""
    palette = json.loads(PALETTE_FILE.read_text())
    panels = [('0.0 (observed)', shared / 's4/observed_t0.h5ad')]
    panels += [(f'{t:.1f}' + (' (generated)' if t in (0, 1, 2, 3) else ''),
                shared / 'generated_states' / f'time_{token(t)}.h5ad')
               for t in np.arange(0, 3.001, .5)]
    fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.4))
    rows = []
    for ax, (title, path) in zip(axes.flat, panels):
        population = ad.read_h5ad(path, backed='r')
        xy = np.asarray(population.obsm['spatial'])
        labels = population.obs['Annotation'].astype(str)
        ax.scatter(xy[:, 0], xy[:, 1], c=[palette[label] for label in labels],
                   s=2.5, alpha=.9, linewidths=0)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(title, fontsize=9, pad=3)
        rows.append({'panel': title, 'cells': len(xy)})
        population.file.close()
    fig.tight_layout()
    pd.DataFrame(rows).to_csv(output / 'S11_population_counts.csv', index=False)
    return save(fig, output, 'Figure_S11_MOSTA_spatial_states')


def draw_growth(shared: Path, output: Path):
    """S12: select brain cells and draw the shared-scale growth maps."""
    growth = pd.read_csv(shared / 's5_growth/growth_by_cell_fully_generated.csv')
    brain = growth.loc[growth.celltype.eq('Brain')]
    settings = json.loads((shared / 's5_growth/growth_contract.json').read_text())
    vmin, vmax = settings['vmin'], settings['vmax']
    if not np.isfinite(brain[['x', 'y', 'growth']]).all().all():
        raise ValueError('Growth inputs contain non-finite values.')
    low, high = brain[['x', 'y']].min(), brain[['x', 'y']].max()
    padding = .04 * (high - low)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'growth', ['#17324d', '#245b78', '#1f8a8a', '#7bc8a4', '#e8f6ef'])
    display_times = [t for t in TIMES if t != 1.5]
    fig = plt.figure(figsize=(12.63, 7.59))
    grid = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, .07],
                            wspace=.10, hspace=.16)
    axes = [fig.add_subplot(grid[row, column]) for row in range(3) for column in range(4)]
    for ax, time in zip(axes, display_times):
        cells = brain.loc[np.isclose(brain.time, time)]
        ax.scatter(cells.x, cells.y, c=np.clip(cells.growth, vmin, vmax),
                   cmap=cmap, norm=mpl.colors.Normalize(vmin, vmax),
                   s=2.2, alpha=.92, linewidths=0)
        ax.set(xlim=(low.x-padding.x, high.x+padding.x),
               ylim=(low.y-padding.y, high.y+padding.y), xticks=[], yticks=[])
        ax.set_aspect('equal')
        ax.set_title(f't={time:.2f}', loc='left', fontsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(.7)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax),
                                                cmap=cmap), cax=fig.add_subplot(grid[:, -1]))
    colorbar.set_label('Growth rate')
    brain.groupby('time').growth.agg(['count', 'median']).to_csv(output / 'S12_growth_summary.csv')
    return save(fig, output, 'Figure_S12_MOSTA_growth')


def draw_composition(shared: Path, output: Path):
    """S13: calculate counts and proportions, including the Other group."""
    table = pd.read_csv(shared / 's6_composition/celltype_composition_fully_generated.csv')
    counts = table.pivot(index='time', columns='celltype', values='count').fillna(0)
    selected = counts.reindex(columns=CELL_TYPES, fill_value=0).copy()
    selected['Other'] = counts.drop(columns=list(CELL_TYPES), errors='ignore').sum(axis=1)
    fractions = selected.div(selected.sum(axis=1), axis=0)
    palette = json.loads(PALETTE_FILE.read_text())
    colors = [palette.get(label, '#c9c3b8') for label in selected]
    selected.to_csv(output / 'S13_cell_counts.csv')
    fractions.to_csv(output / 'S13_cell_fractions.csv')
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8))
    axes[0].stackplot(selected.index, selected.to_numpy().T, labels=selected.columns,
                      colors=colors, alpha=.95, linewidth=.5, edgecolor='white')
    axes[0].set(xlim=(0, 3), xlabel='Time', ylabel='Number of cells')
    axes[0].legend(loc='center left', bbox_to_anchor=(1.01, .5), frameon=False)
    bottom = np.zeros(len(fractions))
    for label, color in zip(fractions, colors):
        y = 100 * fractions[label].to_numpy()
        axes[1].bar(np.arange(len(y)), y, bottom=bottom, color=color, width=.76,
                    edgecolor='white', linewidth=.6)
        bottom += y
    axes[1].set_xticks(np.arange(len(fractions)), [f'{t:.2f}' for t in fractions.index])
    axes[1].set(xlabel='Time', ylabel='Cell proportion (%)', ylim=(0, 100))
    for letter, ax in zip('ab', axes):
        ax.text(-.08, 1.04, letter, transform=ax.transAxes, fontsize=14, weight='bold')
    fig.tight_layout()
    return save(fig, output, 'Figure_S13_MOSTA_composition')


def draw_lineage(shared: Path, output: Path):
    """S14: count transitions between labels of the same simulated particles."""
    from CytoBridge.pl import plot_sankey
    table = pd.read_csv(shared / 's7_lineage/fixed_particle_labels.csv.gz')
    times = np.arange(0, 3.001, .5)
    labels, particle_ids = [], None
    transitions = []
    for time in times:
        cells = table.loc[np.isclose(table.time, time)].sort_values('particle_id')
        current_ids = cells.particle_id.to_numpy()
        if particle_ids is not None and not np.array_equal(current_ids, particle_ids):
            raise ValueError('Lineage labels must follow the same particles at every time.')
        particle_ids = current_ids
        labels.append(cells.celltype.astype(str).to_list())
    for i, (before, after) in enumerate(zip(labels[:-1], labels[1:])):
        edges = pd.DataFrame({'source': before, 'target': after}).value_counts().rename('count').reset_index()
        edges['start_time'], edges['end_time'] = times[i], times[i+1]
        transitions.append(edges)
    pd.concat(transitions, ignore_index=True).to_csv(output / 'S14_lineage_transitions.csv', index=False)
    fig = plot_sankey(predicted_labels_list=labels,
                      out_html=str(output / 'Figure_S14_MOSTA_lineage.html'),
                      start_index=0, time_keys=[f'{t:.1f}' for t in times],
                      show_time_axis=True, min_flow=None, keep_source_cumfrac=.8,
                      normalize_mode=None, label_to_color=json.loads(PALETTE_FILE.read_text()),
                      lineage_anchor_mode=False, style='nature-methods',
                      title='Cell fate transitions', width=None, height=None)
    fig.update_layout(font={'family': 'Arial', 'color': 'black'})
    paths = [output / 'Figure_S14_MOSTA_lineage.pdf', output / 'Figure_S14_MOSTA_lineage.png']
    fig.write_image(str(paths[0]))
    fig.write_image(str(paths[1]), scale=2)
    return paths


def draw_lr_profiles(output: Path):
    """S18: normalize and draw the 31 LR profiles shown in the paper."""
    source = SOURCE / 'si/S11'
    table = pd.read_csv(source / 'numerical_truth/seed42_M_sum/lr_pair_timecourse.csv')
    selected = pd.read_csv(source / 'tables/s11_msum_stable_representative31.csv').sort_values('display_order')
    colors = {1: '#D97757', 2: '#2A7F9E', 3: '#6A994E'}
    fig, axes = plt.subplots(8, 4, figsize=(13.6, 23.2))
    plotted = []
    for ax, row in zip(axes.flat, selected.itertuples()):
        values = table.loc[table.pair_id.astype(str).eq(str(row.pair_id))].sort_values('time')
        x, y = values.time.to_numpy(float), values.score.to_numpy(float)
        normalized = (y-y.min()) / max(float(np.ptp(y)), 1e-12)
        dense_x = np.linspace(x.min(), x.max(), 200)
        color = colors[int(row.cluster)]
        ax.plot(dense_x, PchipInterpolator(x, normalized)(dense_x), color=color, lw=2.2)
        ax.scatter(x, normalized, s=18, color=color, edgecolor='white', linewidth=.6, zorder=3)
        ax.set_title(f'Pattern {row.cluster}  {row.pair}', loc='left', fontsize=9)
        ax.set(xlim=(x.min(), x.max()), ylim=(-.03, 1.03), xlabel='Time', ylabel='Normalized score')
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        for time, raw, value in zip(x, y, normalized):
            plotted.append({'pair_id': row.pair_id, 'pair': row.pair, 'cluster': row.cluster,
                            'time': time, 'score': raw, 'normalized_score': value})
    for ax in list(axes.flat)[len(selected):]:
        ax.set_axis_off()
    fig.tight_layout()
    pd.DataFrame(plotted).to_csv(output / 'S18_LR_normalized_profiles.csv', index=False)
    return save(fig, output, 'Figure_S18_MOSTA_LR_profiles')


def draw_supplementary(data_dir: str | Path, output_dir: str | Path, figures=range(11, 19)):
    """Draw selected S11–S18 panels, returning their newly generated PDF/PNG paths.

    ``data_dir`` is ``data/mosta/paper`` from ``mosta_figure_data.zip``.
    The source checkout supplies the small GO and LR result tables.
    """
    from . import gene_enrichment, gene_programs
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output == data or data in output.parents or SOURCE in output.parents:
        raise ValueError('Choose an output directory outside the input files.')
    output.mkdir(parents=True, exist_ok=True)
    shared = data / 'shared'
    paths = {}
    functions = {11: draw_spatial_states, 12: draw_growth, 13: draw_composition, 14: draw_lineage}
    go_dir = SOURCE / 'si/S9_S10/numerical_inputs/clusterprofiler_server_run'
    for number in figures:
        style()
        if number in functions:
            paths[number] = functions[number](shared, output)
        elif number == 15:
            result = gene_programs.render(gene_programs.load_inputs(shared / 's8_gene_programs'), output)
            paths[number] = [result['pdf'], result['png']]
        elif number in (16, 17):
            result, _ = (gene_enrichment.render_s9(go_dir, output) if number == 16
                         else gene_enrichment.render_s10(shared, go_dir, output))
            paths[number] = [result['pdf'], result['png']]
        elif number == 18:
            paths[number] = draw_lr_profiles(output)
        else:
            raise ValueError('Choose supplementary figure numbers from 11 to 18.')
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/mosta/paper'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figures', nargs='+', type=int, default=list(range(11, 19)))
    args = parser.parse_args()
    for number, paths in draw_supplementary(args.data_dir, args.output_dir, args.figures).items():
        print(f'S{number}: {paths[0]}')
