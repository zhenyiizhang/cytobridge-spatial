"""Create dataset notebooks that calculate and plot one analysis at a time."""

from __future__ import annotations

import nbformat


def build_analysis_notebook(tutorial):
    dataset = tutorial.dataset

    def md(source):
        return nbformat.v4.new_markdown_cell(source.strip())

    def code(source):
        return nbformat.v4.new_code_cell(source.strip())

    figures = "\n".join(
        f"- [{label}](../paper_figures/{target})"
        for label, target, _ in tutorial.figure_links
    )
    cells = [
        md(f"""
# {tutorial.title}

Use a trained CytoBridge model to calculate cell-state velocities, growth rates,
and cell-type interactions. Each calculation below returns arrays or a table
that you can inspect before plotting. These functions also work with a model
trained on your own experiment.

This notebook starts with aligned data and a trained model. For training, use
[Train a model](../../training.md). You do not need to preprocess or train again
to run the analysis here.
"""),
        md(f"""
## Load the data

After [installation](../../installation.md), download `{dataset}_model.zip` and
`{dataset}_analysis_data.zip` from the [data page](../../data_checkpoints.md).
Extract both into the same project folder. Set `PROJECT_DIR` below to that
folder. Its `data/{dataset}/` directory should contain `aligned.h5ad`,
`workflow.json`, `model/`, and `edge_classifier/`.

All paths below are relative to `PROJECT_DIR`, regardless of where you open
the notebook. New results are written under `outputs/{dataset}_analysis/`.
"""),
        code(f"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import CytoBridge as cb
from CytoBridge.workflow import load_workflow_config

PROJECT_DIR = Path(os.environ.get("CYTOBRIDGE_PROJECT_DIR", ".")).resolve()
DATA_DIR = PROJECT_DIR / "data" / "{dataset}"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "{dataset}_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"
DATASET_CONFIG = {dataset!r}

config, _ = load_workflow_config(DATA_DIR / "workflow.json")
fields = config["dataset"]
cb.tl.set_global_random_seed(42)
plt.rcParams.update({{
    "font.family": "Arial", "font.size": 10, "text.color": "black",
    "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "svg.fonttype": "none",
}})
"""),
        md("""
The aligned file contains the coordinates and PCA scores used for training.
`model_state_adata` puts them into a smaller AnnData object for model evaluation.
It copies the existing features without calculating PCA again. Its `X` contains
the spatial coordinates first, followed by the expression features.

Opening the file in backed mode avoids loading the full gene-expression matrix.
"""),
        code("""
aligned = sc.read_h5ad(DATA_DIR / "aligned.h5ad", backed="r")
try:
    states = cb.tl.model_state_adata(
        aligned,
        time_key=fields["time_key"],
        latent_key=fields["obsm_key"],
        spatial_key=fields["spatial_key"],
    )
finally:
    aligned.file.close()

time_key = "time_point_processed"
annotation_key = fields["annotation_key"]
observed_times = sorted(states.obs[time_key].unique())
states.obs.groupby(time_key, observed=True).size().rename("Cells / spots").to_frame()
"""),
        md("""
## Load the model

`load_dynamical_model_from_dir` reads the model configuration and the trained
velocity, growth, interaction, and score networks. The feature dimension is
taken from the data. `loaded.model` is passed to each analysis below.
"""),
        code(f"""
loaded = cb.tl.load_dynamical_model_from_dir(
    DATA_DIR / "model", dim=states.n_vars, device=DEVICE,
    edge_predictor_path=DATA_DIR / "edge_classifier" / "{dataset}_edge_model.pt",
)
model = loaded.model
pd.Series({{"Cells / spots": states.n_obs, "Model features": states.n_vars}})
"""),
        md("""
## Calculate velocity components

Choose one observed stage for this calculation. Change `analysis_time` to
another value in the table above to repeat it at a different stage.
The complete cell population at that stage is retained when calculating interactions.

`compute_velocity_components_from_adata` returns one vector per cell for the
intrinsic component (`drift`), interaction component (`interaction`), score term
(`score`), and their sum (`full`). It also saves these arrays in `stage.obsm`.
The first two columns are spatial components. Later columns correspond to
the expression features.
"""),
        code("""
analysis_time = observed_times[len(observed_times) // 2]
stage = states[states.obs[time_key] == analysis_time].copy()
velocity = cb.tl.compute_velocity_components_from_adata(
    stage, model, time_key=time_key, device=DEVICE,
    interaction_m=1024,
)
pd.DataFrame({
    name: np.linalg.norm(velocity[name], axis=1)
    for name in ("drift", "interaction", "score", "full")
}, index=stage.obs_names).head()
"""),
        md("""
### Plot the interaction component

Pass the coordinates and the calculated spatial vectors to
`plot_velocity_component`. This draws a streamline plot from those vectors.
Use `velocity["drift"]` instead to plot the intrinsic component.
"""),
        code("""
velocity_plot = cb.pl.plot_velocity_component(
    coords=stage.obsm["spatial"],
    velocity=velocity["interaction"][:, :2],
    title=f"Interaction component, t = {analysis_time:g}",
    out_path=str(OUTPUT_DIR / "interaction_velocity.pdf"),
    show=True,
)
"""),
        md("""
## Calculate growth rates

`evaluate_growth_by_timepoint` evaluates the growth network on each observed
population. It returns a table with the model time, cell type, coordinates,
and growth rate. It also adds `growth_rate` to the observations in each slice.
Here, `slices` contains the observed model states, not simulated cells.
"""),
        code("""
slices = {
    str(t): states[states.obs[time_key] == t].copy()
    for t in observed_times
}
growth = cb.tl.evaluate_growth_by_timepoint(
    slices, model, time_points=observed_times,
    annotation_key=annotation_key, device=DEVICE,
)
growth.groupby(["time", "celltype"], observed=True)["growth"].mean().head(10)
"""),
        md("""
### Plot growth across stages

This plot reads the growth values calculated above. A shared colour scale
allows comparison across stages. Values outside the 5th–95th percentile range
share the end colours. The saved table retains the original growth rates.
"""),
        code("""
growth_plot = cb.pl.plot_growth_timepoint_grid(
    slices, time_points=observed_times,
    out_path=str(OUTPUT_DIR / "growth.pdf"),
    n_cols=min(4, len(observed_times)),
    scale_mode="global_limits", shared_colorbar=True,
    colorbar_label="Growth rate", show=True,
)
"""),
        md("""
## Calculate cell-type interactions

First calculate attention on the spatial graph for the selected stage.
`edge_index[0]` identifies sender cells and `edge_index[1]` identifies receivers.
`attn_mean` contains their first-layer attention weights, averaged across heads.
The calculation uses the model's learned edge predictor and neighbourhood radius.
"""),
        code("""
attention = cb.tl.save_interpolated_attention(
    stage, time_value=float(analysis_time), model=model, device=DEVICE,
    out_dir=str(OUTPUT_DIR / "attention"),
)
pd.Series({
    "Cells / spots": stage.n_obs,
    "Directed edges": attention["edge_index"].shape[1],
})
"""),
        md("""
Next aggregate the edge weights by sender and receiver cell type.
`M_per_source` divides the summed attention by the number of sender cells of
each type. Rows are senders and columns are receivers.
Before summing, the function clips attention weights above the 99.5th percentile
to that percentile. This reduces the influence of a few large edge weights.
"""),
        code("""
communication = cb.tl.analyze_attention_by_celltype(
    edge_index=attention["edge_index"], attn=attention["attn_mean"],
    labels=stage.obs[annotation_key].astype(str).to_numpy(),
    remove_self_loop=True, distance_bins=None, plot=False,
)
pair_scores = pd.DataFrame(
    communication["M_per_source"],
    index=communication["types"], columns=communication["types"],
)
pair_scores.index.name = "Sender"
pair_scores.columns.name = "Receiver"
pair_scores.iloc[:5, :5]
"""),
        md("""
### Plot the interaction matrix

The returned table can be plotted with Matplotlib or seaborn. This heatmap uses
the per-sender scores directly. Changing the display does not require another
model evaluation.
"""),
        code("""
import seaborn as sns

size = max(5, 0.3 * len(pair_scores))
fig, ax = plt.subplots(figsize=(size + 1, size))
sns.heatmap(pair_scores, cmap="viridis", ax=ax,
            cbar_kws={"label": "Attention per sender cell"})
ax.set_title(f"Cell-type interactions, t = {analysis_time:g}")
fig.savefig(OUTPUT_DIR / "cell_type_interactions.pdf", bbox_inches="tight")
plt.show()
"""),
        md("""
## Save the calculated results

Save the growth table, interaction matrix, and per-cell velocity arrays for
further analysis. The H5AD stores the selected stage and its computed velocity
components, so the coordinates, cell identities, and vectors stay together.
"""),
        code("""
growth.to_csv(OUTPUT_DIR / "growth.csv", index=False)
pair_scores.to_csv(OUTPUT_DIR / "cell_type_interactions.csv")
stage.write_h5ad(OUTPUT_DIR / "velocity.h5ad", compression="gzip")
"""),
        md(f"""
## Continue the analysis

Use [trajectory simulation](../../trajectory_analysis.md) to predict cells at
intermediate times with this model. The [Python reference](../../api/tools.rst)
also covers LR analysis, gene programmes, and virtual perturbations.

The examples above introduce the analysis functions. For the paper's particular
time grids, comparisons, and panel layouts, continue to these figure notebooks:

{figures}
"""),
    ]
    # Stable cell IDs make rebuilding preserve outputs for unchanged code.
    for i, cell in enumerate(cells):
        cell.id = f"{dataset}-analysis-{i:02d}"
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "cytobridge": {"requires_study_data": True, "runs_training": False},
    })
