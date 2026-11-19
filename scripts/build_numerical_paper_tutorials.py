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
output = project / Path("outputs/{dataset}_paper")
output.mkdir(parents=True, exist_ok=True)

def show(paths):
    for path in paths:
        if Path(path).suffix == ".png":
            display(Image(filename=str(path), width=850))
''')
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
    from build_mosta_main_tutorial import main
    main()


def arista_main():
    from build_arista_reader_tutorials import main_figure
    main_figure()


def arista_supplementary():
    from build_arista_reader_tutorials import supplementary
    supplementary()

if __name__ == '__main__':
    mosta()
    mosta_main()
    arista_main()
    arista_supplementary()
