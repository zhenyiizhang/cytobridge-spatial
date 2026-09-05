"""Draw AD paper panels from the published numerical inputs.

Run from the source checkout, for example:
python reproduction/admouse/draw_figures.py --panels cd e --output-dir outputs/admouse
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHIVE = Path(__file__).parent / 'final_figures'


def source_module(name):
    path = ARCHIVE / 'main/scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_gene_programs(output: Path, input_dir: Path | None = None):
    """Cluster gene profiles and draw the Figure 6c–d heatmap and curves."""
    module = source_module('ad_main_figurecd')
    input_dir = input_dir or ARCHIVE / 'main/data/ad_main_figurecd'
    module.PROFILE_FILE = input_dir / 'gene_zscore_profiles.csv'
    module.METADATA_FILE = input_dir / 'gene_temporal_metadata.csv'
    module.DATA_DIR = output / 'tables'
    module.FIGURE_DIR = output
    module.main()
    assignments = pd.read_csv(module.DATA_DIR / 'gene_cluster_assignments_weighted.csv')
    for cluster, genes in assignments.groupby('cluster', sort=True):
        genes.to_csv(module.DATA_DIR / f'pattern_{int(cluster)}_genes.csv', index=False)


def draw_lr_profiles(output: Path, input_file: Path | None = None):
    """Calculate within-pair z-scores and draw Figure 6e."""
    module = source_module('ad_main_figuree')
    module.DATA_FILE = input_file or ARCHIVE / 'main/data/ad_main_figuree/lr_pair_timecourse.csv'
    module.FIGURE_DIR = output
    wide, z = module.lr_matrix()
    (output / 'tables').mkdir(parents=True, exist_ok=True)
    wide.to_csv(output / 'tables/lr_scores.csv')
    z.to_csv(output / 'tables/lr_zscores.csv')
    module.main()


def draw_spatial_populations(data: Path, output: Path):
    """Draw Figure 6b from saved states and labels at the three shown ages."""
    module = source_module('ad_main_figureb')
    module.STATE_DIR = data / 'populations/compat_base/01_interpolation'
    module.LABEL_DIR = data / 'populations/whole_tissue/baseline_labels_k1'
    module.OUTPUT_DIR = output
    module.main()


def draw_trem2_spatial(data: Path, output: Path):
    """Recalculate endpoint composition and draw Figure 6f from particle states."""
    source = source_module('admouse_trem2_plot')
    source.SOURCE_DIR = data / 'trem2/figure_source'
    source.PANEL_DATA_DIR = output / 'tables'
    source.apply_style()
    module = source_module('ad_main_figuref')
    module.FIGURES = output
    cases = {condition: source.load_case('whole_tissue', condition)
             for condition in source.SPATIAL_CONDITIONS}
    tables = source.prepare_scope_tables('whole_tissue')
    spatial_window = source.spatial_limits({'whole_tissue': cases})
    limits = source.shared_plot_limits({'whole_tissue': cases}, {'whole_tissue': tables})
    module.draw_panel_a(source, cases, tables['composition'], spatial_window,
                        limits['spatial_attention_vmax'])
    module.draw_panel_b(source, tables['composition'])


def draw_module_response(input_file: Path, output: Path, gene: str):
    """Select the paper's module contrasts and draw Figure 6g or S30."""
    module = source_module('ad_main_figureg')
    module.configure_style()
    if gene == 'Trem2':
        modules, labels, time, panel = module.MODULES, module.LABELS, 2.4, 'g'
    elif gene == 'Spp1':
        modules = ['Myelination_Oligo', 'Endothelial_BBB', 'SPP1_CD44_axis',
                   'DAM_microglia', 'Antigen_Presentation_MHCII']
        labels = ['Myelination/\noligo.', 'Endothelial/\nBBB', 'SPP1-CD44',
                  'DAM', 'Antigen\npresentation']
        time, panel = 2.5, ''
    else:
        raise ValueError('Choose Trem2 or Spp1')
    table = pd.read_csv(input_file)
    selected = table[table.perturbed_gene.eq(gene) & np.isclose(table.time, time)
                     & table.population.eq('all_particles') & table.module.isin(modules)]
    values = selected.pivot(index='module', columns='direction', values='delta').reindex(modules)
    if values[['high', 'low']].isna().any().any():
        raise ValueError('The requested module contrasts are incomplete')
    (output / 'tables').mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / 'tables' / f'{gene}_module_contrasts.csv', index=False)
    fig, ax = plt.subplots(figsize=(11.08, 5.48), dpi=180)
    x = np.arange(len(modules))
    for direction, offset, color, label in [('high', -.18, '#B5667A', 'Activation'),
                                           ('low', .18, '#9EBED0', 'Knockdown')]:
        ax.bar(x + offset, values[direction], .36, color=color, edgecolor='white',
               linewidth=.7, label=f'In silico {gene} {label}', zorder=3)
    ax.axhline(0, color='black', ls='--', lw=1.5, zorder=4)
    for value in (-.5, .5):
        ax.axhline(value, color='#D9D9D9', lw=2, zorder=1)
    limit = max(.82, np.ceil((np.max(np.abs(values[['high', 'low']].values)) + .08) * 10) / 10)
    ax.set(xlim=(-.5, len(modules)-.5), ylim=(-limit, limit), ylabel='Module Score Change')
    ax.set_yticks([-.5, .5], ['−0.5', '0.5'])
    ax.set_xticks(x, labels, rotation=48, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', direction='out', width=1.3, length=5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.005), ncol=2, frameon=False,
              handlelength=1.5, handletextpad=.35, columnspacing=3, borderaxespad=0)
    ax.text(-.095, 1.17, panel, transform=ax.transAxes, fontsize=22, weight='bold', va='top')
    fig.subplots_adjust(left=.105, right=.985, bottom=.34, top=.79)
    for extension in ['pdf', 'png']:
        fig.savefig(output / f'{gene.lower()}_module_response.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data/admouse'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--panels', nargs='+', choices=['b', 'cd', 'e', 'f', 'g', 's30'], required=True)
    args = parser.parse_args()
    data, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output == data or data in output.parents or ARCHIVE.resolve() in output.parents:
        parser.error('Choose an output directory outside the input data and source archive.')
    output.mkdir(parents=True, exist_ok=True)
    for panel in args.panels:
        if panel == 'b':
            draw_spatial_populations(data, output)
        elif panel == 'cd':
            draw_gene_programs(output)
        elif panel == 'e':
            draw_lr_profiles(output)
        elif panel == 'f':
            draw_trem2_spatial(data, output)
        elif panel == 'g':
            draw_module_response(data / 'trem2/trem2_module_scores.csv', output, 'Trem2')
        else:
            draw_module_response(ARCHIVE / 'supplementary/downstream/data/ad_supplementary2/whole_tissue_spp1_modules_endpoint.csv', output, 'Spp1')
        print(f'Drew {panel} in {output}')


if __name__ == '__main__':
    main()
