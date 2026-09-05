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
Work through the panels in order. Panel b calculates ligand–receptor scores,
panel c counts lineage destinations, and panels d–e evaluate the trained model.
The saved population states keep the cells in a–c identical to those in the
article. The [MOSTA analysis tutorial](../dataset_workflows/mosta.ipynb)
introduces population simulation.

A CUDA GPU is used for the model calculations in d–e below. Panels a–c only
need the numerical figure data. Each plot is generated from cell coordinates,
labels or calculated values. Frames and annotations retain the paper's layout.
'''), setup('mosta', 'figure4a/slice_data/time_0.h5ad'),
             code('''import json
import numpy as np
import pandas as pd
from reproduction.mosta.main_figure import (
    PANELS, draw_main_figure, draw_interaction_maps, draw_brain_velocity,
)
from reproduction.mosta.calculations import interaction_scores, map_interaction_scores
from reproduction.mosta.figures import style

style()
palette = json.loads((PANELS / "style_authority/label_to_color.json").read_text())''')]
    for panel, title, description in [
        ('a', 'Spatial populations', 'Plot the observed E12.5, E13.5, E14.5 and E15.5 cells and the generated E13, E14 and E15 populations. The generated states come from one trajectory starting at E12.5 with 50,000 particles, time step 0.05, diffusion 0.03 and seeds 42/43. The simulation includes population growth. Coordinates are not warped. The plotted populations contain 51,365, 63,533, 77,369, 86,610, 102,519, 100,871 and 113,350 cells in time order.'),
        ('b', 'Wnt3a–Fzd7/Lrp6 interaction maps', '''For each directed cell-type pair, multiply mean ligand expression, mean receptor-complex expression and its communication weight:

`LR score = ligand_mean * receptor_mean * communication_weight`

The receptor complex uses the minimum expression of Fzd7 and Lrp6. The communication weight is `M_per_source`, calculated from the model's attention. Sum incoming and outgoing pair scores for each cell type, then map that value to its cells. These are cell-type scores, not separate LR measurements for each cell.

E13 uses 15,144 generated cells for the calculation and 63,533 cells for the spatial display. The table preserves that distinction. Cells without a corresponding score stay grey. The final plot scales colours to the 1st–99th percentiles separately at each time.'''),
        ('c', 'Cartilage lineage', '''Follow the same simulated particles from E15 to E15.5 (model times 2.5 to 3). Select the 1,282 particles labelled cartilage primordium at E15 and count their destination labels. Divide by 1,282 to obtain transition fractions. This analysis uses a fixed-particle trajectory, so particle identities remain available across time.

The three largest destinations are cartilage, cartilage primordium and connective tissue. The plot shows their simulated destinations over the observed E15.5 background.'''),
        ('d', 'Interaction-induced gene velocity', '''Evaluate the downloaded model on the same 8,000 observed E15.5 cells used in the paper. `compute_velocity_components` returns intrinsic drift, interaction and score terms. Full velocity is their sum.

Take the last 50 dimensions of the interaction term and project this gene-state derivative onto spatial coordinates using a 30-neighbour graph. Calculate attention and cell-type communication on the same 8,000 cells. The displayed arrows connect Choroid plexus, Meninges and Brain. The Brain→Brain loop aggregates communication between brain cells.

`calculate_velocity_panel` runs these public APIs and writes the numerical fields and communication table. The next cell passes those newly calculated files to the paper's plotting function.'''),
        ('e', 'Brain velocity fields', '''Evaluate all 17,071 observed E15.5 Brain cells with interaction group size 1,024. Project the gene derivatives before selecting the displayed brain region. All four plots use spatial coordinates: two show projected gene derivatives and two show the first two (spatial) model dimensions.

The comparison is full velocity (`drift + interaction + score`) versus interaction velocity. The displayed region is −1.3 < x < −0.5 and 3.3 < y < 4.2.'''),
    ]:
        cells += [markdown(f'## {panel}. {title}\n\n{description}')]
        if panel == 'b':
            cells += [code('''edges = pd.read_csv(PANELS / "fig4b/evidence/type_matrix.csv")
scores = interaction_scores(edges)
display(scores[["time", "cell_type", "incoming", "outgoing", "total"]].head())

cells_to_plot = pd.read_csv(data / "figure4b/cell_mapping.csv.gz", low_memory=False)
mapping = map_interaction_scores(cells_to_plot, scores)
show(draw_interaction_maps(data, output, mapping=mapping))''')]
        elif panel == 'c':
            cells += [code('''with np.load(PANELS / "fig4c/evidence/numeric_render_state.npz", allow_pickle=False) as state:
    destination = pd.Series(state["target_labels"].astype(str))
counts = destination.value_counts()
fractions = counts / len(destination)
display(pd.DataFrame({"cells": counts, "fraction": fractions}).head(3))
figures = draw_main_figure(data, output, panels="c")
show(figures["c"])''')]
        elif panel == 'd':
            cells += [code('''from reproduction.mosta.calculations import calculate_velocity_panel
from reproduction.mosta.interaction_velocity import draw_interaction_velocity

model_dir = project / "data/mosta/model"
if not (model_dir / "config.yaml").is_file():
    cb.datasets.download("mosta", destination=project, kind="mosta_model.zip")

numeric_d = calculate_velocity_panel(model_dir, output / "calculated_d", "d", device="cuda", overwrite=True)
show(draw_interaction_velocity(
    numeric_d, numeric_d.parent / "communication_all_type_edges.csv.gz", palette, output,
))''')]
        elif panel == 'e':
            cells += [code('''numeric_e = calculate_velocity_panel(model_dir, output / "calculated_e", "e", device="cuda", overwrite=True)
show(draw_brain_velocity(output, numeric_path=numeric_e))''')]
        else:
            cells += [code(f'figures = draw_main_figure(data, output, panels="{panel}")\nshow(figures["{panel}"])')]
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
Panel a uses the original unwarped populations in `slice_data`. Panel b uses
the separately anchored display population in `display_states`. These are
different files and are selected explicitly below.

For model evaluation before plotting, see [Calculate Figure 5 from the trained
model](arista_model_fields.md). That page gives the model-loading, velocity
projection and growth calculations, including their output filenames.
'''), setup('arista', 'all_time_communications.pkl'),
             code('''import numpy as np
import pandas as pd
from reproduction.arista.main_figure import SOURCE, draw_main_figure''')]
    panels = [
        ('a', 'Spatial dynamics', '''Use the five unwarped populations at model times 0, 0.5, 1, 1.5 and 2 (2, 3.5, 5, 7.5 and 10 DPI). Their cell counts are 7,668, 7,780, 8,106, 8,608 and 9,436.

Count transitions between labels of the same simulated particles to draw lineage connections. Communication arrows use the saved model-derived cell-type matrices. Spatial anchors are calculated from the coordinates of each cell type. The plotting code applies the original camera, colours and annotations.'''),
        ('b', 'Generated population', 'Plot the 7,798 generated cells at 3.5 DPI from `display_states/time_0p5.h5ad`. This panel uses the spatially anchored display coordinates. Panel a instead uses the unwarped states from the original simulation.'),
        ('c', 'Spatial velocity', '''The left plot uses the full spatial velocity at 5 DPI. The right plot compares the full and interaction fields after each 52-dimensional field has been projected onto spatial coordinates with a 30-neighbour graph.

For each cell, divide the dot product of these two projected vectors by their lengths to calculate cosine similarity. Draw the enlarged region using that value. The code below performs this calculation before plotting.'''),
        ('d', 'Intrinsic-context gene velocity', '''This panel uses **intrinsic drift**, not full velocity. Evaluate `model.predict_velocity` on all 46,199 observed cells and take the last 50 (gene-state) dimensions. Fit a two-component PCA to the gene states, construct a 30-neighbour graph in the 50-dimensional space and project the drift into PCA coordinates with scVelo.

The supplied numerical file stores the resulting coordinates and intrinsic vectors. The plotting function interpolates those vectors onto the scVelo streamline grid and uses the original cell-type label positions. The [model calculation](arista_model_fields.md#intrinsic-context-gene-velocity) regenerates this file from the checkpoint.'''),
        ('e', 'Growth and interaction', '''Evaluate `model.predict_growth` and the Euclidean norm of the 52-dimensional interaction vector at each of the nine population states. Group the per-cell values by time and cell type, then calculate their arithmetic means. Each circle represents one group and its size encodes the number of cells.

The paper calculation contains 82,306 cells and 177 groups. The nine original input populations are available separately from the display populations. The [model calculation](arista_model_fields.md#growth-and-interaction) shows how to evaluate them.'''),
    ]
    for panel, title, description in panels:
        cells += [markdown(f'## {panel}. {title}\n\n{description}')]
        if panel == 'c':
            cells += [code('''with np.load(SOURCE / "figure5c_embedded_velocity.npz", allow_pickle=False) as state:
    full = state["full_embedded_velocity"]
    interaction = state["interaction_embedded_velocity"]
denominator = np.linalg.norm(full, axis=1) * np.linalg.norm(interaction, axis=1)
cosine = np.divide((full * interaction).sum(axis=1), denominator,
                   out=np.zeros(len(full)), where=denominator > 0)
display(pd.Series(cosine, name="cosine similarity").describe())''')]
        elif panel == 'e':
            cells += [code('''values = pd.read_csv(SOURCE / "figure5e_growth_interaction_by_cell.csv")
means = values.groupby(["time", "celltype"], as_index=False).agg(
    growth_mean=("growth", "mean"), interaction_mean=("interaction", "mean"),
    cells=("growth", "size"),
)
display(means.head())
print(f"{len(values):,} cells, {len(means)} time-by-cell-type groups")''')]
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
