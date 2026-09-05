#!/usr/bin/env python3
"""Build the MOSTA and ARISTA tutorials that draw from numerical inputs."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'docs/tutorials/paper_figures'


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def setup(dataset, required):
    cell = code(f'''
import os
from pathlib import Path

import CytoBridge as cb
from IPython.display import Image, display

project = Path(os.environ.get("CYTOBRIDGE_PROJECT_DIR", ".")).resolve()
data = project / "data/{dataset}/paper"
if not (data / "{required}").is_file():
    cb.datasets.download("{dataset}", destination=project,
                         kind="{dataset}_figure_data.zip")
output = Path("outputs/{dataset}_paper")
output.mkdir(parents=True, exist_ok=True)

def show(paths):
    for path in paths:
        if Path(path).suffix == ".png":
            display(Image(filename=str(path), width=850))
''')
    if dataset == 'arista':
        cell.source += '''\n\nif not (data / "display_states/time_0.h5ad").is_file():
    cb.datasets.download("arista", destination=project,
                         kind="arista_spatial_display_data.zip")'''
    return cell


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    notebook.metadata.language_info = {'name': 'python'}
    nbf.write(notebook, DESTINATION / name)


def mosta():
    cells = [markdown('''
# Supplementary Figures S11–S18: MOSTA

Draw the spatial populations, growth maps, cell-type proportions, lineage
diagram, and gene and ligand–receptor time courses used in the paper.
The input files contain cell states and numerical analysis results, not images.

Run this notebook from the [source checkout](../../installation.md).
The first cell downloads the paper inputs if they are not already present.
It needs about 550 MB of disk space for the extracted data. Small enrichment
and LR tables are included with the code.

To generate cell states with the trained model first, follow the
[MOSTA analysis tutorial](../dataset_workflows/mosta.ipynb).
The plotting functions below use the saved paper states so that stochastic
simulation does not change the populations shown in the manuscript.
'''), setup('mosta', 'shared/s4/observed_t0.h5ad'),
             code('from reproduction.mosta.figures import draw_supplementary')]
    descriptions = {
        11: ('Spatial populations', 'Plot the observed starting population and the generated populations from the H5AD coordinate and cell-type arrays.'),
        12: ('Brain growth', 'Select brain cells from the per-cell growth table and draw the time points with one shared colour scale.'),
        13: ('Cell-type composition', 'Calculate cell-type counts and proportions from the population table. The output also includes the counts and fractions as CSV files.'),
        14: ('Lineage transitions', 'Count the transitions between the labels of the same simulated particles and draw the resulting lineage diagram.'),
        15: ('Brain gene programs', 'Draw the expression profiles and gene-program assignments from the saved numerical gene tables.'),
        16: ('Gene-program enrichment', 'Draw the enriched biological processes from the gene-program enrichment tables.'),
        17: ('Developmental expression', 'Plot the developmental gene profiles and their enrichment results.'),
        18: ('Ligand–receptor time courses', 'Normalize the LR time courses and interpolate the displayed curves from their sampled values.'),
    }
    for number, (title, description) in descriptions.items():
        cells += [markdown(f'## S{number}. {title}\n\n{description}'),
                  code(f'figures = draw_supplementary(data, output, figures=[{number}])\nfor paths in figures.values():\n    show(paths)')]
    write('mosta_figures.ipynb', cells)


def mosta_main():
    cells = [markdown('''
# Figure 4: MOSTA

Draw the mouse-embryo populations, interaction maps, cartilage-lineage
transitions and velocity fields shown in Figure 4.

Run this notebook from the [source checkout](../../installation.md).
The first cell downloads the cell-state inputs if needed. The smaller lineage,
communication and velocity arrays are included with the code.
The [MOSTA analysis tutorial](../dataset_workflows/mosta.ipynb) starts with the
trained model and shows how to generate populations and evaluate growth.

Panels a and b retain the paper's frame and label layout. Their cell layers
are regenerated from coordinates and interaction scores. The remaining panels
are drawn from lineage and velocity arrays.
'''), setup('mosta', 'figure4a/slice_data/time_0.h5ad'),
             code('from reproduction.mosta.main_figure import draw_main_figure')]
    for panel, title, description in [
        ('a', 'Spatial populations', 'Plot the four observed populations and the three generated intermediate populations with the paper’s cell-type colours.'),
        ('b', 'Interaction maps', 'Calculate the colour range separately at each time from the 1st and 99th percentiles of the interaction scores, then draw the three spatial maps.'),
        ('c', 'Cartilage lineage', 'Count the destination labels of the simulated cartilage-primordium cells. Draw the three largest transitions and calculate their percentages.'),
        ('d', 'Interaction-induced gene velocity', 'Draw the projected gene-velocity streamlines and communication arrows. Arrow widths use the saved cell-type communication scores.'),
        ('e', 'Brain velocity fields', 'Draw full and interaction velocity in gene and spatial coordinates for the selected brain region.'),
    ]:
        cells += [markdown(f'## {panel}. {title}\n\n{description}'),
                  code(f'figures = draw_main_figure(data, output, panels="{panel}")\nshow(figures["{panel}"])')]
    write('main_figure_4.ipynb', cells)


def arista_main():
    cells = [markdown('''
# Figure 5: ARISTA

Draw Figure 5 from generated cell states, communication scores, and per-cell
velocity and growth arrays. No completed figure is used as an input.

Run this notebook from the [source checkout](../../installation.md).
The first cell downloads the spatial populations, communication scores and
lineage labels. The velocity and growth arrays are included with the code.

The [ARISTA analysis tutorial](../dataset_workflows/arista.ipynb) shows how to
load the trained model, simulate populations and evaluate growth. Here we use
the saved paper populations and velocity arrays to reproduce the displayed panels.
The spatial display uses the same coordinate anchoring as the paper. This
changes the plotted coordinates, not the model's state or cell-type labels.
To regenerate these populations, follow [Generate the ARISTA paper populations](arista_populations.md).
'''), setup('arista', 'all_time_communications.pkl'),
             code('from reproduction.arista.main_figure import draw_main_figure')]
    panels = [
        ('a', 'Spatial dynamics', 'Calculate the spatial anchors and draw lineage transitions and cell-type communication between the five populations.'),
        ('b', 'Generated population', 'Plot the generated population at the intermediate time point using the paper’s cell-type colours.'),
        ('c', 'Spatial velocity', 'Calculate the full-versus-interaction velocity cosine for each cell and the spatial velocity grid. The right panel enlarges the selected region.'),
        ('d', 'Gene velocity', 'Interpolate the full gene-velocity vectors in PCA space and draw the streamlines over the cells.'),
        ('e', 'Growth and interaction', 'Group the per-cell values by time and cell type, calculate the means, and draw the growth–interaction comparison.'),
    ]
    for panel, title, description in panels:
        cells += [markdown(f'## {panel}. {title}\n\n{description}'),
                  code(f'figures = draw_main_figure(data, output, panels="{panel}")\nshow(figures["{panel}"])')]
    write('main_figure_5.ipynb', cells)


def arista_supplementary():
    cells = [markdown('''
# Supplementary Figures S19–S24: ARISTA

Draw the spatial populations, growth maps, lineage transitions, gene programs
and ligand–receptor profiles used in the supplementary figures.

Run this notebook from the [source checkout](../../installation.md).
The first cell downloads numerical cell states and per-cell growth values.
Gene and LR tables are included with the code. Each section below recalculates
its plot from these numerical inputs. It does not load a completed figure.

The [ARISTA analysis tutorial](../dataset_workflows/arista.ipynb) introduces
the model-loading and analysis APIs. [Generate the paper populations](arista_populations.md)
gives the simulation command used for the spatial displays below.
'''), setup('arista', 'all_time_communications.pkl'),
             code('from reproduction.arista.supplementary import draw_supplementary')]
    for number, title, description in [
        (19, 'Spatial populations', 'Plot observed and generated populations at the nine sampled times.'),
        (20, 'Growth maps', 'Select up to 2,500 cells at each time using seed 42 and calculate the 5th–95th percentile colour range for each panel.'),
        (21, 'Lineage and composition', 'Count transitions between the same simulated particles, then calculate cell-type counts and proportions at each time.'),
        (22, 'Gene programs', 'Plot the gene trajectories, program means and standard deviations, and GO enrichment results.'),
        (23, 'Ligand–receptor programs', 'Normalize and cluster all 531 LR profiles and draw the program means.'),
        (24, 'Ligand–receptor time courses', 'Draw the 50 selected LR profiles, with 25 from each program.'),
    ]:
        cells += [markdown(f'## S{number}. {title}\n\n{description}'),
                  code(f'figures = draw_supplementary(data, output, figures=[{number}])\nshow(figures[{number}])')]
    write('arista_figures.ipynb', cells)


if __name__ == '__main__':
    mosta()
    mosta_main()
    arista_main()
    arista_supplementary()
