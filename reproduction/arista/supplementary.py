"""Draw ARISTA Supplementary Figures S19–S24 from numerical inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import fitz
import matplotlib as mpl
import numpy as np
import pandas as pd

from CytoBridge.pl import plot_sankey
from . import spatial_plots as spatial
from . import gene_programs as genes

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'CytoBridge/results/data/arista_supplementary_figures'
PALETTE = Path(__file__).parent / 'data/label_to_color.json'


def directories(output):
    paths = {key: output / key for key in ('vector', 'pdf', 'png', 'jpeg', 'tables')}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def draw_populations(data, output, palette):
    panels = {}
    for time in np.arange(0, 4.01, .5):
        token = f'{time:g}'.replace('.', 'p')
        sources = [('Generated', 'generated_display_states')]
        if time.is_integer():
            sources.append(('Observed', 'display_states'))
        for kind, folder in sources:
            path = data / folder / f'time_{token}.h5ad'
            population = ad.read_h5ad(path)
            xy = np.asarray(population.obsm['spatial'])
            panels[(float(time), kind)] = spatial.SpatialPanel(
                float(time), kind, xy[:, 0], xy[:, 1],
                population.obs.Annotation.astype(str).to_numpy(),
                np.arange(population.n_obs).astype(str), path, 'spatial')
    paths, counts = spatial.plot_s12(
        panels, palette, set(), False, 'Figure_S19_ARISTA_populations',
        directories(output), font_family='Arial')
    counts.drop(columns='input_path').to_csv(output / 'S19_population_counts.csv', index=False)
    return [paths['pdf'], paths['png']]


def draw_growth(data, output):
    table = pd.read_csv(data / 'growth_by_cell.csv.gz')
    rng = np.random.default_rng(42)
    selected = []
    for time, cells in table.groupby('time', sort=True):
        cells = cells.sort_values('cell_index').reset_index(drop=True)
        indices = np.sort(rng.choice(len(cells), min(2500, len(cells)), replace=False))
        subset = cells.iloc[indices].copy()
        subset['objective_isolation_flag'] = False
        subset['n_compute_panel'] = len(cells)
        selected.append(subset)
    sample = pd.concat(selected, ignore_index=True)
    sample.to_csv(output / 'S20_display_cells.csv', index=False)
    dirs = directories(output)
    paths, _ = spatial.plot_s13(
        sample, False, 'Figure_S20_ARISTA_growth', dirs, dirs['tables'],
        fit_package_native_canvas=True, annotate_injury_reference=True,
        annotate_injury_all_panels=True, font_family='Arial')
    return [paths['pdf'], paths['png']]


def draw_lineage(data, output, palette):
    with np.load(data / 'fixed_particle_lineage_labels.npz', allow_pickle=False) as archive:
        times = archive['time_points']
        labels = [archive[f'labels_{i}'].astype(str) for i in range(len(times))]
    if len({len(values) for values in labels}) != 1:
        raise ValueError('Lineage input must track the same particles at every time.')
    counts = pd.DataFrame([pd.Series(values).value_counts() for values in labels], index=times).fillna(0)
    counts.to_csv(output / 'S21_cell_counts.csv', index_label='time')
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fractions.to_csv(output / 'S21_cell_fractions.csv', index_label='time')
    transitions = []
    for i in range(len(times)-1):
        edges = pd.DataFrame({'source': labels[i], 'target': labels[i+1]}).value_counts().rename('count').reset_index()
        edges['start_time'], edges['end_time'] = times[i], times[i+1]
        transitions.append(edges)
    pd.concat(transitions).to_csv(output / 'S21_lineage_transitions.csv', index=False)
    fig = plot_sankey(
        predicted_labels_list=labels, out_html=str(output / 'S21a_lineage.html'),
        start_index=0, time_keys=[f'{time:.1f}' for time in times], show_time_axis=True,
        min_flow=None, keep_source_cumfrac=.8, normalize_mode=None,
        label_to_color=palette, lineage_anchor_mode=False, style='nature-methods',
        title='Cell Fate Transitions', width=1600, height=1000)
    fig.update_layout(font={'family': 'Arial', 'color': 'black'})
    first = output / 'S21a_lineage.pdf'
    fig.write_image(str(first))
    dirs = directories(output)
    second, _ = spatial.plot_s14b(fractions, palette, 'S21b_composition', dirs, dirs['tables'])
    document = fitz.open()
    page = document.new_page(width=560, height=576)
    with fitz.open(first) as source:
        # Plotly's layout uses pixels; PDF pages use points (72/96 inch).
        scale = source[0].rect.width / 1600
        page.show_pdf_page(fitz.Rect(0, 0, 560, 316.75), source, 0,
                           clip=fitz.Rect(0, 95 * scale, 1600 * scale, 1000 * scale))
    with fitz.open(second['pdf']) as source:
        page.show_pdf_page(fitz.Rect(24, 337, 554, 565), source, 0)
    for letter, point in [('a', (0, 18)), ('b', (0, 349))]:
        page.insert_text(point, letter, fontsize=20, fontname='hebo')
    paths = [output / 'Figure_S21_ARISTA_lineage_composition.pdf',
             output / 'Figure_S21_ARISTA_lineage_composition.png']
    document.save(paths[0], deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False).save(paths[1])
    document.close()
    return paths


def draw_gene_programs(output):
    genes._configure_legacy_style()
    mpl.rcParams.update({'font.family': 'Arial', 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})
    expression = pd.read_csv(TABLES / 'gene_trajectories.csv', index_col=0)
    expression.columns = expression.columns.astype(float)
    roster = pd.read_csv(TABLES / 'gene_display_roster.csv')
    paths = {letter: output / f'S22{letter}.svg' for letter in 'abcd'}
    genes._plot_s15a(expression, roster.head(18), paths['a'])
    genes._plot_s15b(pd.read_csv(TABLES / 'gene_program_prototypes.csv'), paths['b'])
    genes._plot_s15c(pd.read_csv(TABLES / 'gene_program_1_GO_terms.csv'), paths['c'], output / 'S22c.png')
    genes._plot_s15d(pd.read_csv(TABLES / 'gene_program_2_GO_terms.csv'), paths['d'], output / 'S22d.png')
    document = fitz.open()
    page = document.new_page(width=576, height=372.96)
    for letter, placement in genes.S15_LEGACY_PLACEMENTS.items():
        svg = paths[letter]
        with fitz.open(svg.with_suffix('.pdf')) as source:
            x, y, width = (placement[key] * .24 for key in ('x', 'y', 'width'))
            height = width * source[0].rect.height / source[0].rect.width
            page.show_pdf_page(fitz.Rect(x, y, x+width, y+height), source, 0)
        crop = genes.S15_LABEL_CROPS[letter]
        page.insert_text((crop[0]*.24+2, crop[1]*.24+16), letter, fontsize=16, fontname='hebo')
    result = [output / 'Figure_S22_ARISTA_gene_programs.pdf', output / 'Figure_S22_ARISTA_gene_programs.png']
    document.save(result[0], deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False).save(result[1])
    document.close()
    return result


def draw_supplementary(data_dir, output_dir, figures=(19, 20, 21, 22, 23, 24)):
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output == data or data in output.parents:
        raise ValueError('Choose an output directory outside the inputs.')
    output.mkdir(parents=True, exist_ok=True)
    palette = json.loads(PALETTE.read_text())
    result = {}
    for number in figures:
        if number == 19:
            result[number] = draw_populations(data, output, palette)
        elif number == 20:
            result[number] = draw_growth(data, output)
        elif number == 21:
            result[number] = draw_lineage(data, output, palette)
        elif number == 22:
            result[number] = draw_gene_programs(output)
        elif number in (23, 24):
            from CytoBridge.results import load_arista_supplementary_figures, plot_arista_ligand_receptor_figures
            plots = plot_arista_ligand_receptor_figures(load_arista_supplementary_figures(),
                                                       output_dir=output / f'S{number}')
            result[number] = list(plots[f'S{number}'])
        else:
            raise ValueError('Choose Supplementary Figures 19 through 24.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/arista/paper'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--figures', nargs='+', type=int, default=[19, 20, 21, 22, 23, 24])
    args = parser.parse_args()
    for number, paths in draw_supplementary(args.data_dir, args.output_dir, args.figures).items():
        print(number, [str(path) for path in paths])
