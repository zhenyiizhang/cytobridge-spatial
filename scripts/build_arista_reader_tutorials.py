"""Authoritative ARISTA reader notebooks: calculate once, continue with explicit outputs."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def setup():
    return code('''
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import CytoBridge as cb
from IPython.display import Image, display

project = Path(os.environ.get("CYTOBRIDGE_PROJECT_DIR", ".")).resolve()
data = project / "data/arista"
analysis = Path(os.environ.get("CYTOBRIDGE_ARISTA_OUTPUT_DIR", project / "outputs/arista")).resolve()
device = os.environ.get("CYTOBRIDGE_DEVICE", "cuda")
populations = analysis / "populations"

def show(paths):
    for path in paths:
        if Path(path).suffix == ".png":
            display(Image(filename=str(path), width=850))
''')


def write(relative, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    notebook.metadata.language_info = dict(name='python')
    if relative == 'dataset_workflows/arista.ipynb':
        notebook.metadata.cytobridge = dict(dataset='arista', runs_training=False,
                                           optional_training_flag='TRAIN_MODEL', paper_panels=['S20'])
    nbf.write(notebook, ROOT / 'docs/tutorials' / relative)
    return notebook


def dataset():
    return write('dataset_workflows/arista.ipynb', [md('''
# ARISTA: model populations, communication and growth

Start here to calculate the inputs for Figure 5 and S19–S25 from the measured
ARISTA states and selected model. The default route loads the downloaded model;
the optional training cell can fit and select a new model instead.
Follow [installation](../../installation.md) and extract `arista_model.zip` and
`arista_analysis_data.zip` from the [data page](../../data_checkpoints.md).

Set `CYTOBRIDGE_PROJECT_DIR` to the folder containing `data/`. All four ARISTA
notebooks use `outputs/arista` (or `CYTOBRIDGE_ARISTA_OUTPUT_DIR`) and the device
in `CYTOBRIDGE_DEVICE`. Each calculation requires a new output directory.
The simulation, exported states and later numerical analyses need
about 0.5 GB; high-resolution figures may add several hundred MB.
'''), setup(), md('''
## Select the model, or train a new one

To train a new model, set `TRAIN_MODEL` to `True` below. The aligned input
contains the two spatial coordinates
and 50 PCA features. `fit` uses those features directly with the ARISTA full
training configuration and the supplied learned edge prior. Training writes
to a new directory and then selects that model
for the simulation and all following field calculations.
'''), code('''
TRAIN_MODEL = False
model_dir = data / "model"
if TRAIN_MODEL:
    model_dir = analysis / "trained_model"
    if model_dir.exists():
        raise FileExistsError(f"Choose a new training directory: {model_dir}")
    training_config = Path(cb.__file__).resolve().parent / "configs/arista_spatial_full.yaml"
    trained = cb.tl.fit(
        data / "aligned.h5ad", config=str(training_config), device=device,
        time_key="time_point_processed", obsm_key="X_latent", is_spatial=True,
        spatial_key="spatial_aligned",
        edge_predictor_path=str(data / "edge_classifier/arista_edge_model.pt"),
        ckpt_dir=model_dir, evaluate_after_training=False)
print("Model:", model_dir)
'''), md('''
## Check the selected inputs

The joint feature order is two `spatial_aligned` coordinates followed by the
50 `X_latent` features. Model times 0–4 correspond to 2, 5, 10, 15 and 20 DPI.
The selected model directory supplies its Finetune and Score_Refine checkpoints.
The pretrained label classifier below has 53 inputs (time plus these 52 state
features) and uses 10-neighbour spatial smoothing.
'''), code('''
classifier = data / "classifier_cache/classifier_resmlp_dedb1d6442f4d3d3.pt"
for path in [data / "aligned.h5ad", model_dir / "Finetune/best_model.pth",
             model_dir / "Score_Refine/score_model.pth", classifier]:
    if not path.is_file():
        raise FileNotFoundError(f"Extract the ARISTA model and analysis downloads first: {path}")
reference = ad.read_h5ad(data / "aligned.h5ad", backed="r")
display(reference.obs.groupby("time_point_processed", observed=True).size().rename("observed cells"))
reference.file.close()
'''), md('''
## Generate populations and model communication

`generate` starts with 7,668 cells, seed 42, split-SDE step 0.01, diffusion 0.03,
growth multiplier 1 and interaction groups of 1,024. It also follows a fixed
particle trajectory at step 0.05 for lineage. The measured populations are
retained at observed times; simulated populations supply the four intermediate
times. Coordinates remain unwarped.

The call exports the populations to `slice_data/`, `model_states/`,
`display_states/` and `generated_display_states/`.
`fixed_particle_lineage_labels_unsmoothed.npz` contains direct classifier
predictions for Figure 5a and the lineage/composition analysis in S21.
The spatial maps keep their existing annotations. The separate
`fixed_particle_lineage_labels.npz` retains spatially smoothed labels.
`all_time_communications.pkl` and `attention/*.npy` are computed from these model
states with self-loops retained and winsor quantile 0.995.
`post_simulation_rng.npz` stores the random state for Figure 5c's observed-field
calculation.
'''), code('''
from reproduction.arista.simulate_paper_populations import generate
generate(data, populations, classifier, device=device, model_dir=model_dir)
display(pd.read_json(populations / "population_sizes.json"))
'''), md('''
## Calculate growth and draw S20

Evaluate growth on every measured/intermediate model population. The table
`growth/growth_by_cell.csv.gz` contains all cells. The S20 renderer selects up
to 2,500 cells per time with seed 42 and computes the displayed sample's 5th and
95th percentile colour limits.
'''), code('''
from reproduction.arista.analysis import calculate_growth
from reproduction.arista.supplementary import draw_supplementary
growth = calculate_growth(data, populations, analysis / "growth", device=device, model_dir=model_dir)
display(growth.groupby("time")["growth"].agg(["count", "median"]))
figures = draw_supplementary(analysis / "growth", analysis / "figures/S20", figures=[20])
show(figures[20])
'''), md('''
## Continue with the same calculated run

[Figure 5](../paper_figures/main_figure_5.ipynb) reads `populations/` and calculates
its velocity fields. [S19–S24](../paper_figures/arista_figures.ipynb) reads those
populations and this growth table, then reconstructs gene and LR time courses.
[S25](../paper_figures/arista_local_domains.ipynb) follows their calculated
Figure 5c field, sparse attention and LR pair universe through domain segmentation
and cell-type-matched permutation tests. Keep the same project and analysis-directory settings.
`populations/model_selection.json` records the selected model directory for
those subsequent notebooks.

The observed-or-interpolated states retain measurements at observed times.
S19 separately shows the fully generated population at all nine times.
''')])


def main_figure():
    return write('paper_figures/main_figure_5.ipynb', [md('''
# Figure 5: ARISTA

Continue after the [ARISTA model calculation](../dataset_workflows/arista.ipynb).
That notebook generates `outputs/arista/populations/` from `aligned.h5ad`, the
Finetune/Score_Refine model and the selected classifier. Here we use those
populations for a/b and evaluate the selected model for c–e.
'''), setup(), code('''
from reproduction.arista.main_figure import (SOURCE, ROI, FOCUS, draw_main_figure, draw_gene_velocity)
from reproduction.arista import plotting
from reproduction.arista.growth_plot import plot_growth_interaction
palette = json.loads((SOURCE / "label_to_color.json").read_text())
output = analysis / "figures/main5"
output.mkdir(parents=True, exist_ok=True)
for relative in ["slice_data/time_0p5.h5ad", "all_time_communications.pkl",
                 "fixed_particle_lineage_labels_unsmoothed.npz", "post_simulation_rng.npz", "model_selection.json"]:
    if not (populations / relative).is_file():
        raise FileNotFoundError(f"Run the ARISTA dataset notebook first: {populations / relative}")
model_dir = Path(json.loads((populations / "model_selection.json").read_text())["model_dir"])
'''), md('''
## a–b. Spatial dynamics and the generated 3.5-DPI population

Count transitions between direct classifier predictions, without spatial
label smoothing, for the lineage links. Use the calculated
cell-type communication matrices, and derive spatial anchors from each cell
type's coordinates. Panels a and b both read `slice_data/time_0p5.h5ad`.
It uses unwarped simulated coordinates; the renderer adjusts the canvas and
camera for display.

The panel is exported directly to PDF before adding its labels. For the
renderer versions and the standalone export command, see
[Figure 5a export](../../reference/figure5a_export.md).
'''), code('''
figures = draw_main_figure(populations, output, panels="ab")
show(figures["a"])
show(figures["b"])
'''), md('''
## c. Evaluate and project the observed spatial field

Restore `post_simulation_rng.npz` and evaluate the five observed times in order,
using random groups of 1,024 cells for interaction. At 5 DPI, project the two
spatial components of full velocity for the left plot. Independently project
the full and interaction 52D fields onto spatial coordinates using a 30-neighbour
graph built in spatial coordinates (`X_spatial`), then calculate their cellwise
cosine for the ROI. The graph is spatial; scVelo transition directions remain 52D.

The coordinate reference supplies the cell identities, display orientation and
ROI. The calculation writes the projected vectors and cosine values to
`spatial_velocity/`; the following cells load them into the plot inputs.
'''), code('''
from reproduction.arista.model_fields import calculate_observed_fields
from reproduction.arista.spatial_velocity import calculate_spatial_velocity
observed_fields = calculate_observed_fields(data, populations, analysis / "observed_fields",
                                           device=device, model_dir=model_dir)
spatial_fields = calculate_spatial_velocity(
    observed_fields / "velocity_time_1.npz", analysis / "spatial_velocity")
velocity_table = pd.read_csv(spatial_fields / "figure5c_all_cells_velocity.csv")
with np.load(spatial_fields / "figure5c_embedded_velocity.npz", allow_pickle=False) as state:
    coordinates = state["manuscript_display_coordinates"]
    spatial_vectors = state["manuscript_display_direct_embedded_velocity"]
velocity_population = ad.AnnData(X=np.zeros((len(velocity_table), 1), dtype=np.float32))
velocity_population.obs["Annotation"] = velocity_table.celltype.astype(str).to_numpy()
velocity_population.obsm["X_spatial"] = coordinates
velocity_population.obsm["velocity_spatial"] = spatial_vectors
display(velocity_table.cosine_full_vs_interaction.describe())
'''), code('''
velocity_table.to_csv(output / "Figure5c_spatial_velocity.csv", index=False)
plotting.configure_style()
figures_c = plotting.plot_figure5c(velocity_population, velocity_table, ROI, FOCUS, palette, output)
show(figures_c["Figure5c_spatial_and_roi"])
'''), md('''
## d. Intrinsic-context gene velocity

Evaluate intrinsic drift on all 46,199 observed cells and take the last 50
gene-state dimensions. Fit a two-component PCA, build the 30-neighbour graph
in 50D and project drift with scVelo. The calculation returns the NPZ path
read by `draw_gene_velocity`.
'''), code('''
from reproduction.arista.gene_velocity import calculate_gene_velocity
gene_state = calculate_gene_velocity(data, analysis / "gene_velocity", device=device, model_dir=model_dir)
show(draw_gene_velocity(output, palette, state_path=gene_state))
'''), md('''
## e. Growth and interaction

Evaluate growth and the 52D interaction-vector norm on the nine measured/
intermediate populations, then group the values by time and cell type.
This evaluation uses seed 42 and writes `fields/figure5e_growth_interaction_by_cell.csv`
and `fields/figure5e_growth_interaction_by_celltype.csv`.
'''), code('''
from reproduction.arista.model_fields import calculate_fields
means = calculate_fields(model_dir, populations / "model_states",
                         analysis / "fields", device=device, seed=42)
means["time_idx"] = means.time.map({t: i for i, t in enumerate(sorted(means.time.unique()))})
display(means.head())
'''), md('''
The paper panel below is redrawn from the original per-cell table.
Interaction evaluation uses random 1,024-cell groups, so new evaluations can
give different interaction values.
'''), code('''
paper_values = pd.read_csv(SOURCE / "figure5e_growth_interaction_by_cell.csv")
paper_means = paper_values.groupby(["time", "celltype"], as_index=False).agg(
    growth_mean=("growth", "mean"), interaction_mean=("interaction", "mean"),
    n=("growth", "size"))
paper_means.to_csv(output / "Figure5e_paper_group_means.csv", index=False)
show(plot_growth_interaction(paper_means, output / "Figure5e_growth_interaction"))
'''), md('''
[S25](arista_local_domains.ipynb) continues from the cosine values in
`spatial_velocity/figure5c_all_cells_velocity.csv`, the population attention
files and the LR time courses calculated in [S23](arista_figures.ipynb).
To repeat calculations, select a new analysis directory and begin with the
dataset notebook.
The [model-fields reference](arista_model_fields.md) also lists the individual calls.

To plot the model-derived `means` object, use the same plotting function:

```python
new_figures = analysis / "figures/new_field_evaluation"
new_figures.mkdir(parents=True, exist_ok=True)
plot_growth_interaction(means, new_figures / "seed42_growth_interaction")
```
''')])


def supplementary():
    return write('paper_figures/arista_figures.ipynb', [md('''
# Supplementary Figures S19–S24: ARISTA

First run the [ARISTA dataset notebook](../dataset_workflows/arista.ipynb) to
create `populations/` and `growth/growth_by_cell.csv.gz` from measured states and
the selected model. Keep the same analysis directory below. S19–S21 use those
outputs; S22–S24 calculate their gene/LR inputs here before plotting.
The GO BP 2023 library and CellChatDB input are included. The calculations below
use them to compute enrichment and LR activity from the model populations.
'''), setup(), code('''
from reproduction.arista.supplementary import draw_supplementary
from reproduction.arista.analysis import (calculate_gene_programs,
    calculate_lr_timecourses, calculate_lr_panels)
output = analysis / "figures/supplementary"
for path in [populations / "model_states/time_0p5.h5ad",
             analysis / "growth/growth_by_cell.csv.gz"]:
    if not path.is_file():
        raise FileNotFoundError(f"Run the ARISTA dataset notebook first: {path}")
'''), md('''
## S19–S21. Populations, growth, lineage and composition

S19 reads observed and generated states at all nine times. S20 samples the
per-cell growth table. S21 counts transitions and cell-type fractions from
`fixed_particle_lineage_labels_unsmoothed.npz`, using direct classifier predictions
without spatial label smoothing. These particles are a separate cohort from
the growth-dependent split populations. Coordinates remain unwarped.
'''), code('''
for number in [19, 20, 21]:
    selected = analysis / "growth" if number == 20 else populations
    figures = draw_supplementary(selected, output, figures=[number])
    show(figures[number])
'''), md('''
## S22. Gene trajectories, clustering and GO enrichment

Reconstruct expression from each model state's 50 PCA features using the
measured reference's persisted loadings and centre, clipping each cell at zero
before averaging. Rank temporal variance; average-linkage clustering of the
top 2,000 row-z-scored genes defines two programs ordered by peak time. The top
18 genes form the trajectory panel. GO over-representation uses the represented
expression background, set sizes 5–5,000, minimum overlap 2 and BH correction.

The GO BP 2023 GMT and its [source/license notice](../../reference/figure_sources/arista-go.md)
are included. `calculate_gene_programs` writes the five numerical plot tables to
`genes/`; `tables_dir` selects those files for S22.
'''), code('''
gene_programs = calculate_gene_programs(data, populations, analysis / "genes")
display(gene_programs)
figures = draw_supplementary(populations, output, figures=[22], tables_dir=analysis / "genes")
show(figures[22])
'''), md('''
## S23–S24. Strict LR projection and temporal programs

Use real log1p expression at observed times and inverse-PCA expression at
intermediates. Every subunit must be supported; the minimum defines complex
activity. Multiply sender-ligand and receiver-receptor means by model-derived
communication per source cell. Save all pair time courses and coverage records
to `lr/`. Min-max normalize each trajectory, apply the paper's k=2 k-means,
report the k=2–8 silhouette diagnostic, and select the 25 profiles nearest each
program mean. The plotting call takes the resulting `lr_panels` object.
'''), code('''
pair_timecourse = calculate_lr_timecourses(data, populations, analysis / "lr")
lr_panels = calculate_lr_panels(pair_timecourse)
display(lr_panels.assignments.groupby("cluster").size().rename("LR pairs"))
display(lr_panels.k_selection)
for number in [23, 24]:
    figures = draw_supplementary(populations, output, figures=[number], lr_panels=lr_panels)
    show(figures[number])
'''), md('''
Continue to [S25](arista_local_domains.ipynb) after running Figure 5c. It uses
`lr/pair_timecourse.csv` from this notebook together with the same run's sparse
attention and calculated spatial field. `lr/coverage.csv` records supported and
excluded complexes; unsupported complexes do not enter the LR time courses.

To draw these calculated tables from a terminal, select the same inputs explicitly:

```bash
python -m reproduction.arista.supplementary --data-dir outputs/arista/populations \\
  --output-dir outputs/arista/figures/supplementary --figures 22 23 24 \\
  --tables-dir outputs/arista/genes --lr-input outputs/arista/lr/pair_timecourse.csv
```
''')])


def local_domains():
    return write('paper_figures/arista_local_domains.ipynb', [md('''
# Supplementary Figure S25: ARISTA local interaction domains

Run the [ARISTA dataset notebook](../dataset_workflows/arista.ipynb),
[Figure 5c](main_figure_5.ipynb) and [S23](arista_figures.ipynb) first, using the
same analysis directory. This notebook uses their cosine field, sparse attention
and LR time courses to segment domains and run cell-type-matched permutations.
It exports six numerical tables and uses them to calculate the S25 panels.
'''), setup(), md('''
## Segment and test the model-derived field

Select the upper quartile within the fixed 5-DPI ROI, connect cells on the
physical-radius graph (cutoff 0.03154105148551745), retain components with at least
20 cells and internal selected-attention edges, then annotate their cell-state
networks. Attention tests use 9,999 cell-type-matched permutations; pathway and
LR-pair tests use 1,999. Each permutation samples cells without replacement
within cell type. Seeds are 42 for domain/pathway tests and 260824 for pair
tests, with per-domain offsets. BH correction is within each domain.
'''), code('''
from reproduction.arista.local_domains import generate
velocity_path = analysis / "spatial_velocity/figure5c_all_cells_velocity.csv"
pair_path = analysis / "lr/pair_timecourse.csv"
for path in [velocity_path, pair_path, populations / "attention/edge_index_interp_t1.0.npy"]:
    if not path.is_file():
        raise FileNotFoundError(f"Complete the linked calculation notebooks first: {path}")
domain_data = generate(data, populations, velocity_path, pair_path, analysis / "local_domains")
display(domain_data.domain_metadata)
'''), md('''
## Calculate panels and draw S25

The preceding call exports `roi_assignments.csv`, `domain_metadata.csv`,
`celltype_edges.csv`, `attention_null.csv`, `pathway_null.csv` and
`lr_pair_null.csv.gz` to `local_domains/inputs/`. `domain_data` contains these
tables. `calculate_arista_local_domain_panels` uses them to prepare the map,
attention ratios, pathway contrasts and significant LR pairs for plotting.
'''), code('''
from CytoBridge.results import calculate_arista_local_domain_panels
from reproduction.paper_figures import draw_supplementary
panels = calculate_arista_local_domain_panels(domain_data)
display(panels.attention)
figures = draw_supplementary([25], output_dir=analysis / "figures/S25",
                             data=domain_data, panels=panels)
for paths in figures.values():
    show(paths)
'''), md('''
To repeat all production steps, select a new analysis directory and begin with
the dataset notebook.
''')])


if __name__ == '__main__':
    dataset()
    main_figure()
    supplementary()
    local_domains()
