#!/usr/bin/env python3
"""Build reader-followable AGIST and non-spatial numerical figure tutorials."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/tutorials/paper_figures"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python"}
    path = DEST / name
    if path.is_file():
        previous = nbf.read(path, as_version=4)
        old_code = [cell for cell in previous.cells if cell.cell_type == "code"]
        new_code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if [cell.source for cell in old_code] == [cell.source for cell in new_code]:
            for old, new in zip(old_code, new_code):
                new.outputs = old.outputs
                new.execution_count = old.execution_count
    nbf.write(notebook, DEST / name)


SETUP = """
import os
import sys
import subprocess
from pathlib import Path

import CytoBridge as cb
from IPython.display import Image, display

project = Path(os.environ.get("CYTOBRIDGE_PROJECT_DIR", ".")).resolve()
device = os.environ.get("CYTOBRIDGE_DEVICE", "cuda")
run_name = os.environ.get("CYTOBRIDGE_RUN_LABEL", "reader_run")
repo = Path(cb.__file__).resolve().parents[1]

def run(module, *arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                   cwd=repo, check=True)
"""


def agist():
    cells = [md("""
# Supplementary Figures S2–S3: AGIST and finite-range attraction

Starting with the distributed observations and trained models, calculate the
velocity agreement in S2 and the five-seed attraction evaluation in S3.
Then convert those calculations to the plotted arrays and draw both pages.

Run these cells in order with the CytoBridge code folder available.
The downloads total about 80 MB.
Set `CYTOBRIDGE_DEVICE` to your GPU (or `cpu`, which is much slower).
Choose a new `CYTOBRIDGE_RUN_LABEL` when repeating the calculations.

S2 uses the recorded generator velocity as its known reference, and evaluates
the fitted velocity, score-gradient and grouped interaction on the same rows.
The groups are broad unsupervised state partitions, not annotated cell types.
S3 holds the model fixed and changes the five inference seeds.
"""), code(SETUP + """
agist_root = Path(os.environ.get("CYTOBRIDGE_AGIST_DATA", project / "data/agist"))
simulation_root = Path(os.environ.get("CYTOBRIDGE_SIMULATION_DATA", project / "data/simulation"))
if not (agist_root / "model/model_final").is_file():
    cb.datasets.download("agist", destination=project)
if not (simulation_root / "training/model/Score_Refine/best_model.pth").is_file():
    cb.datasets.download("simulation", destination=project)
output = project / "outputs" / ("agist_" + run_name)
output.mkdir(parents=True, exist_ok=False)
"""), md("""
## S2. Define state partitions from the 50 gene-state coordinates

Fit the original MiniBatchKMeans candidates (3–8 groups), select by silhouette
score, and number the selected groups by their mean first principal component.
This writes both assignments and selection diagnostics from the observations.
"""), code("""
clusters = output / "state_partitions"
run("scripts.prepare_agist_state_clusters",
    "--input", agist_root / "mouse_brain_simulation.csv",
    "--output-dir", clusters)
"""), md("""
## S2. Evaluate the fitted model and compare its full velocity with the generator

The full velocity is intrinsic drift + interaction + score gradient. Cosines
are calculated separately in the two spatial coordinates and all 50 gene-state
coordinates; the displayed gene score is not a two-dimensional projection.
"""), code("""
import numpy as np
from reproduction.agist.inputs import evaluate_velocity, velocity_inputs

observed, predicted = evaluate_velocity(
    agist_root / "mouse_brain_simulation.csv", agist_root / "config.yaml",
    agist_root / "model", agist_root / "edge_classifier/mouse.pt",
    device=device, seed=42)
np.savez_compressed(output / "s2_model_velocity.npz", **predicted)
velocity = velocity_inputs(
    observed, predicted, agist_root / "figure_fields/figure_fields.npz",
    clusters / "agist_state_cluster_assignments.csv",
    clusters / "cluster_selection_diagnostics.csv")
velocity["source_velocity_overall"]
"""), md("""
## S3. Simulate and evaluate the fitted attraction model

Use the original 400-particle reference, five seeds, and Score_Refine checkpoint.
The paper's S3 evaluation does **not** include the score term. It compares
interaction-on and interaction-off trajectories from the same checkpoint,
evaluates growth with learned masses, and probes the radial interaction curve.

For exactly the recorded stochastic trajectories, set
`CYTOBRIDGE_ATTRACTION_EVALUATION` to an existing evaluation directory.
That explicit choice reuses model results; otherwise this cell performs a new
evaluation. Small stochastic/numerical differences from the paper are possible.
"""), code("""
evaluation_override = os.environ.get("CYTOBRIDGE_ATTRACTION_EVALUATION")
if evaluation_override:
    evaluation = Path(evaluation_override)
else:
    evaluation = output / "attraction_evaluation"
    run("scripts.run_spatial_synthetic_benchmark", "evaluate",
        "--data-dir", simulation_root / "data",
        "--model-dir", simulation_root / "training/model",
        "--stage", "Score_Refine", "--evaluation-dir", evaluation,
        "--seeds", "1,4,8,32,256", "--no-score", "--device", device)
"""), md("""
## Convert the calculated results to the displayed arrays

S3a reads the measured snapshots. S3b selects the original 60 particle
identities from the dense seed-1 rollout (selection seed 11).
The growth, radial-force and interaction-ablation tables come from the
evaluation directory above.
"""), code("""
from reproduction.agist.inputs import attraction_inputs, collect_agist_inputs
from CytoBridge.results import calculate_agist_figure_panels, write_agist_figure_tables

attraction = attraction_inputs(simulation_root / "data/attractive_observed.h5ad", evaluation)
data = collect_agist_inputs(output / "plot_inputs", velocity=velocity, attraction=attraction)
panels = calculate_agist_figure_panels(data)
tables = write_agist_figure_tables(panels, output / "figures")
tables
"""), md("## Draw S2–S3 from those results"), code("""
from reproduction.paper_figures import draw_supplementary
figures = draw_supplementary(
    [2, 3], output / "figures", results_dir=data.source_dir,
    data=data, panels=panels)
for name, (_, png) in figures.items():
    print(name.upper())
    display(Image(filename=str(png), width=720))
""")]
    write("agist_figures.ipynb", cells)


def main2():
    cells = [md("""
# Main Figure 2: AGIST benchmark

This notebook reproduces a's snapshots, b and e. The c/d comparisons require
two missing generator files: `attn_matrix_time0.npy` and `g_values.npy`.

Evaluate the model, reconstruct its velocity display and calculate ten-replicate
distances. The predicted attention and growth are calculated too; the c/d cells
complete their comparisons when the two reference files are supplied.

Run these cells with the CytoBridge code folder available. The AGIST download
provides observations, model and edge predictor.
Choose a new `CYTOBRIDGE_RUN_LABEL` for a new simulation.
"""), code(SETUP + """
agist_root = Path(os.environ.get("CYTOBRIDGE_AGIST_DATA", project / "data/agist"))
if not (agist_root / "model/model_final").is_file():
    cb.datasets.download("agist", destination=project)
output = project / "outputs" / ("main_figure_2_" + run_name)
output.mkdir(parents=True, exist_ok=False)
figures = output / "figures"

def show_calculated(path, width=720):
    if Path(path).suffix == ".pdf":
        import pymupdf
        with pymupdf.open(path) as document:
            display(Image(data=document[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).tobytes("png"), width=width))
    else:
        display(Image(filename=str(path), width=width))
"""), md("""
## a. Draw the four simulated snapshots

The original CSV already contains the generator's four sampled populations.
Plot the spatial coordinates, with every second cell displayed as in the
original plotting code. The upper brain/embryo illustration is a hand-drawn
schematic and does not require model output.
"""), code("""
import pandas as pd
from reproduction.agist.main_figure import draw_observed_snapshots

observed = pd.read_csv(agist_root / "mouse_brain_simulation.csv")
snapshot_pdf = draw_observed_snapshots(observed, figures)
show_calculated(snapshot_pdf)
"""), md("""
## b. Evaluate the velocity and calculate its displayed projection

Evaluate intrinsic drift, interaction and score gradient on the observed
cells. Sum them and compare with the saved generator fields. The original
display uses the same seed-0 30% sample for prediction and reference,
30-neighbor graphs, then scVelo velocity projection. The physical plot uses
T=0; the gene plot uses all times and the first two gene-state coordinates.
The `X_umap` name in scVelo is only a display key here, not a fitted UMAP.
"""), code("""
import numpy as np
from reproduction.agist.inputs import evaluate_velocity
from reproduction.agist.main_figure import calculate_velocity_display, draw_velocity_display

observed, predicted = evaluate_velocity(
    agist_root / "mouse_brain_simulation.csv", agist_root / "config.yaml",
    agist_root / "model", agist_root / "edge_classifier/mouse.pt",
    device=device, seed=42)
np.savez_compressed(output / "model_velocity.npz", **predicted)
velocity_display = calculate_velocity_display(
    observed, predicted, agist_root / "figure_fields/figure_fields.npz",
    output / "velocity_display", n_jobs=4)
velocity_figures = draw_velocity_display(velocity_display, figures)
for space, path in velocity_figures.items():
    print(space)
    show_calculated(path)
"""), md("""
## c/d. Calculate attention and growth, then compare with generator truth

The c calculation uses all time-zero cells in one graph, not the random
groups used for simulation. It averages the absolute first-layer attention
over eight heads, sums outgoing edges, and forms the original 50×50 flow grid
from the top 5,000 positive edges. Sparse edges avoid storing a dense matrix.
The d calculation evaluates growth on every observed cell, independently
clips each distribution at its 1st/99th percentiles, and fits the regression.

Put the original generator files
in `data/agist/generator_reference/`, or set `CYTOBRIDGE_AGIST_GENERATOR_REFERENCE`
to their directory. They were produced by the generator checkpoint used for
these simulated observations.
"""), code("""
from reproduction.agist.main_figure import (
    evaluate_growth_attention, calculate_attention_display,
    draw_attention_display, draw_growth_correlation,
)

growth, attention_edges = evaluate_growth_attention(
    agist_root / "mouse_brain_simulation.csv", agist_root / "config.yaml",
    agist_root / "model", agist_root / "edge_classifier/mouse.pt",
    output / "growth_attention", device=device)
coordinates_t0 = observed.loc[observed["samples"].eq(0), ["x1", "x2"]].to_numpy()
predicted_attention = calculate_attention_display(attention_edges, coordinates_t0)
np.savez_compressed(output / "predicted_attention_display.npz", **predicted_attention)
reference = Path(os.environ.get("CYTOBRIDGE_AGIST_GENERATOR_REFERENCE",
                               agist_root / "generator_reference"))
required = [reference / "attn_matrix_time0.npy", reference / "g_values.npy"]
missing = [path for path in required if not path.is_file()]
if missing:
    display({
        "c/d comparison": "Not drawn: matching generator inputs are unavailable",
        "missing files": [str(path) for path in missing],
        "calculated growth values": len(growth),
        "calculated attention edges": len(attention_edges["attention"]),
    })
else:
    truth_attention = calculate_attention_display(
        np.load(required[0], allow_pickle=False), coordinates_t0)
    attention_pdf = draw_attention_display(
        coordinates_t0, predicted_attention, truth_attention, figures)
    growth_pdf, growth_metrics = draw_growth_correlation(
        growth, np.load(required[1], allow_pickle=False), figures)
    show_calculated(attention_pdf)
    show_calculated(growth_pdf)
    display(growth_metrics)
"""), md("""
## e. Simulate ten independently seeded populations

The checkpoint and time-zero observations remain fixed. The seeds vary the
interaction grouping, Brownian increments and cell birth/loss draws.
The original settings are dt=0.1, sigma=0.03, group size 1,024 and zero
daughter displacement.

To evaluate previously generated populations, set
`CYTOBRIDGE_AGIST_TRAJECTORIES` to their `trajectories` directory.
This reuses trajectories, not distances or a finished figure.
"""), code("""
trajectory_override = os.environ.get("CYTOBRIDGE_AGIST_TRAJECTORIES")
distance_override = os.environ.get("CYTOBRIDGE_AGIST_DISTANCES")
if distance_override:
    trajectories = None  # Reuse explicitly selected complete W2 evaluation.
elif trajectory_override:
    trajectories = Path(trajectory_override)
else:
    simulation = output / "simulation"
    run("scripts.run_agist_split_sde_replicates",
        "--config", agist_root / "config.yaml",
        "--checkpoint-dir", agist_root / "model",
        "--edge-predictor", agist_root / "edge_classifier/mouse.pt",
        "--data-csv", agist_root / "mouse_brain_simulation.csv",
        "--output-dir", simulation, "--simulate-only", "--device", device)
    trajectories = simulation / "trajectories"
"""), md("""
## e. Calculate the weighted distances in the original coordinate spaces

Read the populations above, calculate W2 at times 1, 2 and 3 in the two
spatial coordinates and all 50 gene coordinates, then summarize the ten
replicates as means and sample standard deviations. This is exact transport
on the full populations and can take tens of minutes on CPU.

For a previously completed calculation, `CYTOBRIDGE_AGIST_DISTANCES` may point
to its distance directory. That explicit choice reuses numeric evaluation
results; leaving it unset runs the calculation below.
"""), code("""
if distance_override:
    distances = Path(distance_override)
else:
    distances = output / "distances"
    run("scripts.evaluate_and_plot_agist_w2_replicates",
        "--trajectory-dir", trajectories,
        "--truth-csv", agist_root / "mouse_brain_simulation.csv",
        "--output-dir", distances)
"""), md("""
## e. Draw the newly selected distance calculation

The CytoBridge values are read from `distances`. STORIES and stVCR use the
paper's recorded external benchmark values.
"""), code("""
from reproduction.agist.inputs import main_figure_2_from_evaluation
from reproduction.agist.main_figure import draw_distance_panels

figure_data = main_figure_2_from_evaluation(distances)
display(figure_data.summary)
pdf, png = draw_distance_panels(figure_data, figures)
display(Image(filename=str(png), width=720))
pdf, png
""")]
    write("main_figure_2.ipynb", cells)


def nonspatial():
    cells = [md("""
# Supplementary Figures S4–S5: non-spatial model analyses

Calculate the S4b velocity field directly from the three fitted LR models.
Convert the measured cells, explicit trajectory outputs, directed interaction
tables and evaluation results into the numerical inputs for S4–S5, then draw
the figures from that new directory.

S4b uses the LR-informed training seeds 42, 43 and 44, each averaged over five
16-cell groupings; S4c/d compare the radius-graph Full and No-interaction models.
S5 uses the corresponding cortical Full model.

Run these cells with the CytoBridge code folder available. The downloads
provide the prepared data, trained models and numerical analysis outputs.
"""), code(SETUP + """
import json
from reproduction.nonspatial.inputs import downloaded_sources

source_override = os.environ.get("CYTOBRIDGE_NONSPATIAL_SOURCES")
if source_override:
    selected = json.loads(Path(source_override).read_text())
    weinreb, scnt = selected["weinreb"], selected["scnt"]
    field_models = selected["weinreb_field_models"]
else:
    for dataset in ("weinreb", "scnt_cortex"):
        if not (project / "data" / dataset / "full/model/config.yaml").is_file():
            cb.datasets.download(dataset, destination=project)
    paper = project / "data/nonspatial/paper"
    if not (paper / "weinreb/lr_seed44/model/config.yaml").is_file():
        cb.datasets.download("nonspatial", destination=project,
                             kind="nonspatial_paper_inputs.zip")
    weinreb, scnt, field_models = downloaded_sources(project)

output = project / "outputs" / ("nonspatial_" + run_name)
output.mkdir(parents=True, exist_ok=False)
"""), md("""
## S4b. Evaluate and smooth the fitted velocity field

Evaluate intrinsic drift and the score gradient on every measured 50-PC
state. Evaluate interactions for grouping seeds 0–4, average those fields
within each model, then average the three models. Smooth PC1–PC2 values on
the original 50×50 grid, retaining only the data-supported region.

"""), code("""
from reproduction.nonspatial.fields import calculate_weinreb_fields
weinreb["model_grids"] = calculate_weinreb_fields(
    weinreb["prepared_h5ad"], field_models, output / "weinreb_model_fields",
    device=device, grouping_seeds=(0, 1, 2, 3, 4))
weinreb["model_grids"]
"""), md("""
## Select the numerical results for the other panels

The source dictionaries below identify each input file. Distribution and
clone-fate tables are recorded evaluations of the selected fitted models.
The S5 dense trajectory records the Full model's simulated 50-PC states;
its streamlines are calculated here by finite differences, not read as a plot.
The directed GNN-message and LR-attribution tables are model-analysis outputs.
CellChat scores are an independent reference. This cell does not rerun those
upstream evaluations.

To use newly calculated results, replace the corresponding dictionary path
with your evaluation's output before continuing. Source filenames and producer
descriptions are listed in `reproduction/nonspatial/source_keys.json`.
"""), code("""
display(weinreb)
display(scnt)
"""), md("""
## Calculate all displayed arrays and tables

Read measured labels and coordinates from H5AD. Calculate cell counts,
distribution changes, clone/direction summaries, trajectory streamlines,
receiver interaction vectors, the displayed network edges and pathway ranks.
Recalculate the CellChat correlations from all eligible directed type pairs
(100/121/121 for Weinreb days 2/4/6; 81 for the final cortical time point).
The existing paper plotting functions keep their original axes and style.
"""), code("""
from reproduction.nonspatial.inputs import collect_nonspatial_inputs
from CytoBridge.results import calculate_nonspatial_panels, write_nonspatial_tables

data = collect_nonspatial_inputs(weinreb, scnt, output / "plot_inputs")
panels = calculate_nonspatial_panels(data)
tables = write_nonspatial_tables(panels, output / "figures")
tables
"""), md("## Draw S4–S5 from the new inputs"), code("""
from reproduction.paper_figures import draw_supplementary
figures = draw_supplementary(
    [4, 5], output / "figures", results_dir=data.source_dir,
    data=data, panels=panels)
for name, (_, png) in figures.items():
    print(name.upper())
    display(Image(filename=str(png), width=720))
""")]
    write("nonspatial_figures.ipynb", cells)


if __name__ == "__main__":
    agist()
    main2()
    nonspatial()
