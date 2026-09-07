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
Panels a and b use the same simulated populations in `slice_data`.
Their spatial coordinates are the first two columns of each H5AD's `X` matrix.
[Generate the Figure 5a–b inputs](arista_populations.md) gives the simulation
and communication calculation that writes `slice_data/time_*.h5ad`,
`fixed_particle_lineage_labels.npz` and `all_time_communications.pkl`,
then passes that output directory directly to the same plotting function.

For model evaluation before plotting, see [Calculate Figure 5 from the trained
model](arista_model_fields.md). That page gives the model-loading, velocity
projection and growth calculations, including their output filenames.
'''), setup('arista', 'all_time_communications.pkl'),
             code('''import json
import anndata as ad
import numpy as np
import pandas as pd
from reproduction.arista import plotting
from reproduction.arista.main_figure import SOURCE, ROI, FOCUS, draw_main_figure
from reproduction.arista.growth_plot import plot_growth_interaction

palette = json.loads((SOURCE / "label_to_color.json").read_text())''')]
    panels = [
        ('a', 'Spatial dynamics', '''Use the five unwarped populations at model times 0, 0.5, 1, 1.5 and 2 (2, 3.5, 5, 7.5 and 10 DPI). Their cell counts are 7,668, 7,780, 8,106, 8,608 and 9,436.

Count transitions between labels of the same simulated particles to draw lineage connections. Communication arrows use the saved model-derived cell-type matrices. Spatial anchors are calculated from the coordinates of each cell type. The plotting code applies the original camera, colours and annotations.'''),
        ('b', 'Generated population', 'Plot the 7,780 generated cells at 3.5 DPI from `slice_data/time_0p5.h5ad`, using the same population and simulated coordinates as panel a.'),
        ('c', 'Spatial velocity', '''The left plot uses the full spatial velocity at 5 DPI. The right plot compares the full and interaction fields after each 52-dimensional field has been projected onto spatial coordinates with a 30-neighbour graph.

For each cell, divide the dot product of these two projected vectors by their lengths to calculate cosine similarity. The per-cell table contains the cell type, displayed coordinates, ROI membership and this calculated value. The next cell saves and passes that table directly to the enlarged-region plot, together with the spatial coordinates and full spatial vectors for the left plot.'''),
        ('d', 'Intrinsic-context gene velocity', '''This panel uses **intrinsic drift**, not full velocity. Evaluate `model.predict_velocity` on all 46,199 observed cells and take the last 50 (gene-state) dimensions. Fit a two-component PCA to the gene states, construct a 30-neighbour graph in the 50-dimensional space and project the drift into PCA coordinates with scVelo.

The supplied numerical file stores the resulting coordinates and intrinsic vectors. The plotting function interpolates those vectors onto the scVelo streamline grid and uses the original cell-type label positions. The [model calculation](arista_model_fields.md#intrinsic-context-gene-velocity) regenerates this file from the checkpoint.'''),
        ('e', 'Growth and interaction', '''Evaluate `model.predict_growth` and the Euclidean norm of the 52-dimensional interaction vector at each of the nine population states. Group the per-cell values by time and cell type, then calculate their arithmetic means. Each circle represents one group and its size encodes the number of cells.

The paper calculation contains 82,306 cells and 177 groups. The nine original input populations are available separately from the display populations. The [model calculation](arista_model_fields.md#growth-and-interaction) shows how to evaluate them. The `means` table below has one row per time and cell type, with mean growth, mean interaction magnitude and cell count `n`. The following cell saves and plots this same table.'''),
    ]
    for panel, title, description in panels:
        cells += [markdown(f'## {panel}. {title}\n\n{description}')]
        if panel == 'c':
            cells += [code('''with np.load(SOURCE / "figure5c_embedded_velocity.npz", allow_pickle=False) as state:
    full = state["full_embedded_velocity"]
    interaction = state["interaction_embedded_velocity"]
    coordinates = state["manuscript_display_coordinates"]
    spatial_vectors = state["manuscript_display_direct_embedded_velocity"]
denominator = np.linalg.norm(full, axis=1) * np.linalg.norm(interaction, axis=1)
cosine = np.divide((full * interaction).sum(axis=1), denominator,
                   out=np.zeros(len(full)), where=denominator > 1e-12)
velocity_table = pd.read_csv(SOURCE / "figure5c_all_cells_velocity.csv")
velocity_table["cosine_full_vs_interaction"] = cosine
velocity_population = ad.AnnData(X=np.zeros((len(velocity_table), 1), dtype=np.float32))
velocity_population.obs["Annotation"] = velocity_table["celltype"].astype(str).to_numpy()
velocity_population.obsm["X_spatial"] = coordinates
velocity_population.obsm["velocity_spatial"] = spatial_vectors
display(velocity_table["cosine_full_vs_interaction"].describe())'''), code('''velocity_table.to_csv(output / "Figure5c_spatial_velocity.csv", index=False)
plotting.configure_style()
figures_c = plotting.plot_figure5c(
    velocity_population, velocity_table, ROI, FOCUS, palette, output,
)
show(figures_c["Figure5c_spatial_and_roi"])''')]
        elif panel == 'e':
            cells += [code('''values = pd.read_csv(SOURCE / "figure5e_growth_interaction_by_cell.csv")
means = values.groupby(["time", "celltype"], as_index=False).agg(
    growth_mean=("growth", "mean"), interaction_mean=("interaction", "mean"),
    n=("growth", "size"),
)
means["time_idx"] = means["time"].map({t: i for i, t in enumerate(sorted(means["time"].unique()))})
display(means.head())
print(f"{len(values):,} cells, {len(means)} time-by-cell-type groups")'''), code('''means.to_csv(output / "Figure5e_growth_interaction.csv", index=False)
figures_e = plot_growth_interaction(means, output / "Figure5e_growth_interaction")
show(figures_e)''')]
        else:
            cells += [code(f'figures = draw_main_figure(data, output, panels="{panel}")\nshow(figures["{panel}"])')]
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
    cells[1].source += '''\n\npopulations = data / "populations"
if not (populations / "generated_display_states/time_0p5.h5ad").is_file():
    cb.datasets.download("arista", destination=project,
                         kind="arista_spatial_populations.zip")'''
    for number, title, description in [
        (19, 'Spatial populations', '''Plot observed and generated populations at the nine sampled times. Generated
coordinates are the first two columns of each H5AD's `X` matrix. To use a new
simulation from the population-generation command above, set `populations`
to that command's output directory.'''),
        (20, 'Growth maps', 'Select up to 2,500 cells at each time using seed 42 and calculate the 5th–95th percentile colour range for each panel.'),
        (21, 'Lineage and composition', 'Count transitions between the same simulated particles, then calculate cell-type counts and proportions at each time.'),
        (22, 'Gene programs', 'Plot the gene trajectories, program means and standard deviations, and GO enrichment results.'),
        (23, 'Ligand–receptor programs', 'Normalize and cluster all 531 LR profiles and draw the program means.'),
        (24, 'Ligand–receptor time courses', 'Draw the 50 selected LR profiles, with 25 from each program.'),
    ]:
        source = 'populations' if number == 19 else 'data'
        cells += [markdown(f'## S{number}. {title}\n\n{description}'),
                  code(f'figures = draw_supplementary({source}, output, figures=[{number}])\nshow(figures[{number}])')]
    write('arista_figures.ipynb', cells)


if __name__ == '__main__':
    mosta()
    mosta_main()
    arista_main()
    arista_supplementary()
