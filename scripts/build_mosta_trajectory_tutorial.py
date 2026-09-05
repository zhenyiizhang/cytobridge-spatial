"""Build the MOSTA tutorial using the paper's continuous-population analysis."""
import ast

import nbformat

from build_dataset_tutorials import ROOT, write_notebook


ARCHIVE = "release_artifacts/mosta_package_native_corrected_20260826_v1"


def interpolation_call():
    path = ROOT / ARCHIVE / "reproduction/shared_global_t0_50k/source/server_compute_mosta_si_shared.py"
    source = path.read_text()
    candidates = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
                  and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute)
                  and node.value.func.attr == "run_interpolation_workflow"]
    if len(candidates) != 1:
        raise ValueError("Expected the paper's single interpolation call")
    result = ast.get_source_segment(source, candidates[0])
    for before, after in {"args.device": "DEVICE", "args.n_samples": "N_SAMPLES", "args.seed": "SEED"}.items():
        result = result.replace(before, after)
    return result


def build_notebook():
    md = lambda source: nbformat.v4.new_markdown_cell(source.strip())
    code = lambda source: nbformat.v4.new_code_cell(source.strip())
    cells = [md("""
# MOSTA: mouse organogenesis

Generate a continuous population from the earliest MOSTA stage, then plot
its spatial organization, Brain-cell growth, and cell-type composition.
These are the analyses in Supplementary Figs. S11–S13.

The simulation starts once at t=0 and continues to t=3. In contrast to the
chicken-heart interpolation, it does not restart at each observed stage.
All subsequent generated states come from that same starting population.

## Prepare the data and model

Follow [installation](../../installation.md), then extract `mosta_model.zip`
and `mosta_analysis_data.zip` from the [data page](../../data_checkpoints.md).
Set `PROJECT_DIR` to that directory. Run the notebook from the source checkout.
The aligned file is about 15.5 GB when extracted. This tutorial uses the
paper's 50,000-particle calculation and is intended to run on a GPU.

To train the model first, see [Train a model](../../training.md).
"""), code(f"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import CytoBridge as cb

PROJECT_DIR = Path(os.environ.get("CYTOBRIDGE_PROJECT_DIR", ".")).resolve()
REPO_ROOT = Path(os.environ.get("CYTOBRIDGE_SOURCE_DIR", ".")).resolve()
DATA_DIR = PROJECT_DIR / "data" / "mosta"
output = PROJECT_DIR / "outputs" / "mosta_trajectory"
output.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
N_SAMPLES = 50000
SEED = 42
TIMES = tuple(float(t) for t in np.arange(0, 3.0001, 0.25))
INTERMEDIATE_TIMES = tuple(t for t in TIMES if t not in (0., 1., 2., 3.))
classifier_cache = DATA_DIR / "classifier_cache" / "classifier_resmlp_6d2d7acf7d0ed92d.pt"
palette_path = REPO_ROOT / "{ARCHIVE}" / "reproduction/main_fig4_panels/style_authority/label_to_color.json"
palette = json.loads(palette_path.read_text())
plt.rcParams.update({{
    "font.family": "Arial", "font.size": 9, "text.color": "black",
    "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "svg.fonttype": "none",
}})
"""), md("""
## Load the observed states and model

The aligned coordinates and PCA scores are the model features. The
classifier uses the same features and assigns tissue labels to generated
cells. `frame` is passed to the simulation in the next section.
"""), code("""
reference = ad.read_h5ad(DATA_DIR / "aligned.h5ad")
frame, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
    reference, time_key="time_point_processed", obsm_key="X_latent",
    spatial_key="spatial_aligned", concat_spatial=True, annotation_key="Annotation",
)
feature_columns = cb.tl.infer_feature_columns(frame, annotation_column="Annotation")
loaded = cb.tl.load_dynamical_model_from_dir(
    DATA_DIR / "model", dim=len(feature_columns), device=DEVICE,
    edge_predictor_path=DATA_DIR / "edge_classifier" / "mosta_edge_model.pt",
)
runtime = cb.tl.build_dynamical_runtime(loaded)
frame.groupby("samples").size().rename("Observed cells")
"""), md("""
## Generate the trajectory

The paper evaluates 13 quarter-step states. The split-population simulation
includes growth, while the fixed-population simulation retains particle
identities for lineage analysis. Both use the same initial sampling size.
The figures below use the split population. No spatial warp is applied.
"""), code(interpolation_call()), md("""
Keep the generated populations separate from the measured populations.
The returned split-state arrays contain a generated population at every
time, including the times where an observation is available.
"""), code("""
points = result.sde_points_split_prewarp
if points is None:
    points = result.sde_points_split
labels = result.predicted_labels_split_prewarp
if labels is None:
    labels = result.slice_labels_split
generated = {}
state_dir = output / "generated_states"
state_dir.mkdir(exist_ok=True)
for time, values, celltypes in zip(TIMES, points, labels):
    population = ad.AnnData(X=np.asarray(values, dtype=np.float32))
    population.obs["Annotation"] = np.asarray(celltypes).astype(str)
    population.obsm["spatial"] = population.X[:, :2].copy()
    population.uns["time"] = time
    generated[str(time)] = population
    population.write_h5ad(state_dir / f"time_{time:g}.h5ad")
pd.Series({time: population.n_obs for time, population in generated.items()}, name="Generated cells")
"""), md("""
## Spatial organization — S11

The first panel is the observed t=0 population. The remaining panels show
the generated trajectory at half-step intervals. Tissue colours follow the
paper. All points come from the populations above, not from a saved image.
"""), code("""
initial = frame.loc[np.isclose(frame["samples"], 0)]
display_times = list(np.arange(0, 3.001, 0.5))
fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.4))
panels = [("t=0 (observed)", initial[feature_columns].to_numpy()[:, :2], initial["Annotation"].astype(str))]
panels += [(f"t={t:g} (generated)", generated[str(float(t))].obsm["spatial"],
            generated[str(float(t))].obs["Annotation"]) for t in display_times]
for ax, (title, coords, celltypes) in zip(axes.flat, panels):
    ax.scatter(coords[:, 0], coords[:, 1], c=[palette[label] for label in celltypes],
               s=2.5, alpha=0.9, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(output / "spatial_states_S11.pdf", bbox_inches="tight")
plt.show()
"""), md("""
## Brain-cell growth — S12

Evaluate growth on each generated cell and select cells labelled Brain.
One colour scale and one coordinate frame are calculated from all 13
Brain populations. As in S12, t=1.5 is omitted from the display only.
"""), code("""
growth = cb.tl.evaluate_growth_by_timepoint(
    generated, loaded.model, time_points=TIMES, annotation_key="Annotation", device=DEVICE,
)
growth.to_csv(output / "growth_by_cell.csv", index=False)
brain = growth.loc[growth["celltype"].eq("Brain")].copy()
vmin, vmax = np.quantile(brain["growth"], [0.05, 0.95])
brain.groupby("time")["growth"].agg(["count", "median"])
"""), code("""
cmap = LinearSegmentedColormap.from_list("growth", ["#17324d", "#245b78", "#1f8a8a", "#7bc8a4", "#e8f6ef"])
limits = brain[["x", "y"]].agg(["min", "max"])
padding = 0.04 * (limits.loc["max"] - limits.loc["min"])
display_times = [t for t in TIMES if t != 1.5]
fig, axes = plt.subplots(3, 4, figsize=(12, 7.5))
for ax, time in zip(axes.flat, display_times):
    cells = brain.loc[np.isclose(brain["time"], time)]
    scatter = ax.scatter(cells["x"], cells["y"], c=np.clip(cells["growth"], vmin, vmax),
                         norm=Normalize(vmin, vmax), cmap=cmap, s=2.2, alpha=0.92, linewidths=0)
    ax.set_xlim(limits.loc["min", "x"] - padding["x"], limits.loc["max", "x"] + padding["x"])
    ax.set_ylim(limits.loc["min", "y"] - padding["y"], limits.loc["max", "y"] + padding["y"])
    ax.set_title(f"t={time:.2f}", loc="left", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.subplots_adjust(right=0.9, wspace=0.1, hspace=0.2)
fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.02, label="Growth rate")
fig.savefig(output / "brain_growth_S12.pdf", bbox_inches="tight")
plt.show()
"""), md("""
## Cell counts and composition — S13

Count the classifier-assigned tissue labels in each generated population.
Fractions use the total population at that time as the denominator. The
15 displayed tissue labels are the same as in S13. All remaining labels
are added together as Other, preserving the total cell count.
"""), code("""
composition = cb.tl.summarize_label_composition(labels, TIMES)
composition.to_csv(output / "celltype_composition.csv", index=False)
counts = composition.pivot(index="time", columns="celltype", values="count").fillna(0)
selected = ["Brain", "Connective tissue", "Cavity", "Epidermis", "Muscle", "Jaw and tooth",
            "Meninges", "Liver", "Cartilage primordium", "Spinal cord", "Heart", "GI tract",
            "Dorsal root ganglion", "Cartilage", "Adipose tissue"]
display_counts = counts.reindex(columns=selected, fill_value=0).copy()
display_counts["Other"] = counts.drop(columns=selected, errors="ignore").sum(axis=1)
fractions = display_counts.div(display_counts.sum(axis=1), axis=0)
display_counts
"""), code("""
colours = [palette.get(label, "#c9c3b8") for label in display_counts]
fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
axes[0].stackplot(display_counts.index, display_counts.to_numpy().T,
                  labels=display_counts.columns, colors=colours, alpha=0.95,
                  linewidth=0.5, edgecolor="white")
axes[0].set(xlabel="Time", ylabel="Number of cells", xlim=(0, 3))
bottom = np.zeros(len(fractions))
for label, colour in zip(fractions, colours):
    values = 100 * fractions[label].to_numpy()
    axes[1].bar(np.arange(len(values)), values, bottom=bottom, width=0.76,
                color=colour, edgecolor="white", linewidth=0.6)
    bottom += values
axes[1].set_xticks(np.arange(len(TIMES)), [f"{t:.2f}" for t in TIMES])
axes[1].set(xlabel="Time", ylabel="Cell proportion (%)", ylim=(0, 100))
axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(output / "celltype_composition_S13.pdf", bbox_inches="tight")
plt.show()
"""), md("""
## Saved results and other MOSTA figures

`generated_states/` contains the populations used in these three analyses.
The growth and composition CSVs contain the values passed to the plotting
cells. You can change a plot without repeating the simulation.

The [Figure 4 page](../paper_figures/main_figure_4.ipynb) assembles the paper's
saved panels. The [S14–S18 page](../paper_figures/mosta_figures.ipynb) displays
the saved supplementary pages and links to their analysis scripts. Those
pages do not read the populations generated in this notebook.
""")]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "cytobridge": {"dataset": "mosta", "runs_training": False,
                        "paper_panels": ["S11", "S12", "S13a", "S13b"],
                        "analysis_source": f"{ARCHIVE}/reproduction/shared_global_t0_50k/source/server_compute_mosta_si_shared.py"},
    })
    for i, cell in enumerate(notebook.cells):
        cell.id = f"mosta-trajectory-{i:02d}"
    return notebook


if __name__ == "__main__":
    write_notebook(build_notebook(), ROOT / "docs/tutorials/dataset_workflows/mosta.ipynb")
