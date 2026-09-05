"""Draw Figure 4 from cell states, lineage labels and velocity arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from .figures import SOURCE, style

PANELS = SOURCE / 'main_fig4_panels'


def save_page(document, rectangle, output, stem):
    """Save the completed panel without the rest of the layout template."""
    panel = fitz.open()
    page = panel.new_page(width=rectangle.width, height=rectangle.height)
    page.show_pdf_page(page.rect, document, 0, clip=rectangle)
    paths = [output / f'{stem}.pdf', output / f'{stem}.png']
    panel.save(paths[0], garbage=4, deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False).save(paths[1])
    panel.close()
    return paths


def draw_populations(data, output, palette):
    from . import population_a as plot
    source = data / 'figure4a/slice_data'
    panels = plot.prepare_render_panels(source)
    labels = set()
    for item in panels:
        population = ad.read_h5ad(source / item['file'], backed='r')
        labels.update(population.obs.Annotation.astype(str).unique())
        population.file.close()
    colors = dict(palette, **plot.LEGACY_AI_PALETTE)
    order = [label for label in plot.LEGACY_AI_ORDER if label in labels]
    order += [label for label in palette if label in labels and label not in order]
    if set(order) != labels:
        raise ValueError('The palette does not cover all cell types.')
    # The template supplies frames and labels. All seven scientific point
    # layers are drawn below from H5AD arrays before this panel is exported.
    with fitz.open(PANELS / 'style_authority/Figure_mouse1.ai') as document:
        page = document[0]
        images = {item[0]: item for item in page.get_images(full=True)}
        placements = {int(item['xref']): item for item in page.get_image_info(xrefs=True)}
        for item in panels:
            path = output / f"Figure4a_time_{item['time']:g}.png"
            plot.render_transparent_points(source, item, colors, order, path)
            xref = int(item['xref'])
            if placements[xref]['transform'][3] < 0:
                with Image.open(path) as image:
                    image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(path)
            page.replace_image(xref, filename=str(path))
            if item['placement_mode'] != 'exact_original_ai_bbox':
                plot.update_original_form_placement(document, page, images[xref][9], item['placement_bbox'])
        plot.replace_legend(page, colors, order)
        return save_page(document, plot.CROP, output, 'Figure4a_spatial_populations')


def draw_interaction_maps(data, output):
    from . import population_b as plot
    table = pd.read_csv(data / 'figure4b/cell_mapping.csv.gz', low_memory=False)
    table[plot.NORM_COLUMN] = np.nan
    bounds = []
    with fitz.open(PANELS / 'style_authority/Figure_mouse1.ai') as document:
        page = document[0]
        for item in plot.PANELS:
            selected = np.isclose(table.time, item['time']) & table.cell_type_score_available.astype(bool)
            values = table.loc[selected, 'total_raw'].to_numpy()
            low, high = np.percentile(values, [1, 99])
            if not np.isfinite(values).all() or high <= low:
                raise ValueError(f"Cannot normalize the interaction scores at time {item['time']}.")
            table.loc[selected, plot.NORM_COLUMN] = np.clip((values-low)/(high-low), 0, 1)
            path = output / f"Figure4b_time_{item['time']:g}.png"
            plot.render_panel(table.loc[np.isclose(table.time, item['time'])], item, path)
            page.replace_image(item['xref'], filename=str(path))
            bounds.append({'time': item['time'], 'q01': low, 'q99': high})
        plot.add_generated_panel_annotations(page)
        paths = save_page(document, plot.CROP, output, 'Figure4b_interaction_maps')
    pd.DataFrame(bounds).to_csv(output / 'Figure4b_colour_bounds.csv', index=False)
    return paths


def draw_cartilage(output, palette):
    from . import cartilage as plot
    with np.load(PANELS / 'fig4c/evidence/numeric_render_state.npz', allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    labels = arrays['target_labels'].astype(str)
    if len(labels) != len(arrays['selected_lineage_id']):
        raise ValueError('The target labels and lineage identifiers must have the same length.')
    unique, counts = np.unique(labels, return_counts=True)
    transitions = []
    for label, count in sorted(zip(unique, counts), key=lambda pair: pair[1], reverse=True):
        centroid = arrays['target_spatial'][labels == label, :2].mean(axis=0)
        transitions.append(plot.Transition(str(label), int(count), float(count/len(labels)), tuple(centroid)))
    scatter = output / 'Figure4c_cell_coordinates.png'
    centroids, _ = plot.create_scatter_layer(arrays, transitions, palette, scatter)
    paths = {extension: output / f'Figure4c_cartilage_lineage.{extension}' for extension in ('pdf', 'svg', 'png')}
    plot.assemble_panel(scatter, centroids, transitions, palette, paths)
    pd.DataFrame([{'cell_type': t.target_label, 'cells': t.count, 'fraction': t.probability}
                  for t in transitions]).to_csv(output / 'Figure4c_lineage_fractions.csv', index=False)
    return list(paths.values())


def draw_brain_velocity(output):
    from .velocity import plot_single_velocity_field
    from .figures import save
    categories = ('Apical Progenitors (RG)', 'Basal Progenitors (IP)', 'Choroid Plexus',
                  'Excitatory Neurons', 'Glioblasts', 'Inhibitory Neurons', 'Other')
    palette = dict(zip(categories, ('#1f77b4', '#aec7e8', '#7f7f7f', '#ffbb78', '#8c564b', '#9467bd', '#d9d9d9')))
    with np.load(PANELS / 'fig4e/evidence/numeric_inputs.npz', allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    labels = values['telencephalon_notebook_labels'].astype(str)
    labels[~np.isin(labels, categories)] = 'Other'
    population = ad.AnnData(X=values['features'].astype(np.float32))
    population.obsm['X_spatial'] = values['compute_spatial'].astype(np.float32)
    population.obs['telencephalon'] = pd.Categorical(labels, categories=categories, ordered=True)
    paths = []
    for name, title, field in (
        ('gene_full', 'Gene space: full velocity', 'gene_full_projected_spatial'),
        ('gene_interaction', 'Gene space: interaction velocity', 'gene_interaction_projected_spatial'),
        ('physical_full', 'Spatial velocity: full', 'physical_full'),
        ('physical_interaction', 'Spatial velocity: interaction', 'physical_interaction'),
    ):
        population.obsm['velocity_spatial'] = values[field].astype(np.float32)
        fig, ax = plot_single_velocity_field(
            population, velocity_key='velocity', density=1., figsize=(7, 5),
            flip_y=False, flip_x=False, title=title, color_key='telencephalon',
            mode='default', remove_outliers=True, timepoint_str='E15.5',
            plot_region=(-1.3, -.5, 3.3, 4.2), palette=palette)
        for text in fig.findobj(plt.Text):
            text.set_fontfamily('Arial')
        paths.extend(save(fig, output, f'Figure4e_{name}'))
    return paths


def draw_main_figure(data_dir, output_dir, panels='abcde'):
    data, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    if output == data or data in output.parents or SOURCE in output.parents:
        raise ValueError('Choose an output directory outside the input files.')
    output.mkdir(parents=True, exist_ok=True)
    palette = json.loads((PANELS / 'style_authority/label_to_color.json').read_text())
    result = {}
    for panel in panels:
        style()
        if panel == 'a': result[panel] = draw_populations(data, output, palette)
        elif panel == 'b': result[panel] = draw_interaction_maps(data, output)
        elif panel == 'c': result[panel] = draw_cartilage(output, palette)
        elif panel == 'd':
            from .interaction_velocity import draw_interaction_velocity
            source = PANELS / 'fig4d/evidence'
            result[panel] = draw_interaction_velocity(source / 'numeric_inputs.npz', source / 'communication_all_type_edges.csv.gz', palette, output)
        elif panel == 'e': result[panel] = draw_brain_velocity(output)
        else: raise ValueError('Choose Figure 4 panels a, b, c, d, or e.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/mosta/paper'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--panels', default='abcde')
    args = parser.parse_args()
    for panel, paths in draw_main_figure(args.data_dir, args.output_dir, args.panels).items():
        print(panel, [str(path) for path in paths])
