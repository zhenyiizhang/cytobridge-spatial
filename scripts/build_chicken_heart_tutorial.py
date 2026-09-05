"""Build the chicken-heart tutorial from the study's simulation and growth code."""
from pathlib import Path

import nbformat

from build_dataset_tutorials import ROOT, write_notebook


def build_notebook():
    original = nbformat.read(ROOT / "reproduction/chicken_heart/notebooks/formal_daily_piecewise_interpolation_celltypecorrected.ipynb", 4)
    md = lambda text: nbformat.v4.new_markdown_cell(text.strip())
    code = lambda text: nbformat.v4.new_code_cell(text.strip())
    cells = [md("""
# Chicken-heart development

Calculate populations between D4, D7, D10, and D14, then evaluate their growth
rates. The two plots follow the analyses in Supplementary Fig. S9: median
growth by cell type and growth mapped onto the tissue coordinates.

The calculation starts from the trained model, not from a growth table or an
image. The simulation is stochastic. Newly generated populations need not be
identical to the population used to prepare the printed figure.

## Prepare the data and model

Follow [installation](../../installation.md), then extract
`chicken_heart_model.zip` and `chicken_heart_analysis_data.zip` from the
[data page](../../data_checkpoints.md) into your project directory.
The model archive contains the cell-type classifier as well as the dynamical
model. No raw-data archive is needed for this analysis.

Set `PROJECT_DIR` to that directory. Run the notebook from the source checkout.
For model fitting, see [Train a model](../../training.md).
"""), code(original.cells[1].source), code(original.cells[2].source),
        md("""
## Load the observed states and trained model

`adata_to_aligned_dataframe` reads the existing aligned coordinates and PCA
features, in the order used for training. `load_dynamical_model_from_dir`
loads the trained model. Neither function fits a new PCA or model.
"""), code(original.cells[3].source),
        md("""
## Simulate the intervening days

`run_interpolation_workflow` propagates each measured population until the
next measured stage. It generates D5–D6 from D4, D8–D9 from D7, and D11–D13
from D10. The measured populations themselves are retained at D4, D7, D10,
and D14. The classifier assigns cell types to generated cells.

`result.adata_dict` contains these populations, with one AnnData per time.
Its `X` contains the two spatial coordinates followed by the 50 PCA features.
Growth-dependent division and removal are included in the simulation.
"""), code(original.cells[4].source),
        md("""
## Evaluate growth on each population

The growth network is evaluated on every cell, using its model state and
time. `growth` contains those per-cell values, their labels, and coordinates.
"""), code("""
import matplotlib.pyplot as plt
from downstream_helpers.heart import HEART_LABEL_TO_COLOR

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "text.color": "black",
    "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
time_order = [time_label_map[round(float(t), 9)] for t in result.ts_points]
slices = {label: result.adata_dict[str(float(t))]
          for label, t in zip(time_order, result.ts_points)}
growth = cb.tl.evaluate_growth_by_timepoint(
    slices, loaded.model, time_points=result.ts_points, time_keys=time_order,
    annotation_key="celltype_prediction", device=DEVICE,
)
growth.to_csv(OUTPUT_DIR / "growth_by_cell.csv", index=False)
growth.head()
"""), md("""
## Median growth by cell type — S9a

Group cells by type and day, take the median growth value, and arrange the
result into a cell-type-by-day table. Rows are ordered by their mean across
the available days. A blank entry means that cell type was absent at that
time, not that its growth was zero. Dashed boxes mark generated stages.
"""), code("""
median_growth = growth.groupby(["celltype", "time_key"], observed=True)["growth"].median().unstack()
median_growth = median_growth.reindex(columns=time_order)
median_growth = median_growth.loc[median_growth.mean(axis=1).sort_values(ascending=False).index]
median_growth.to_csv(OUTPUT_DIR / "median_growth_by_celltype.csv")
median_growth
"""), code("""
ax = cb.pl.plot_growth_heatmap(
    median_growth, observed_times=["D4", "D7", "D10", "D14"],
)
ax.figure.set_size_inches(10.5, 6.5)
ax.figure.tight_layout()
ax.figure.savefig(OUTPUT_DIR / "growth_heatmap.pdf", bbox_inches="tight")
ax.figure.savefig(OUTPUT_DIR / "growth_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
"""), md("""
## Spatial growth maps — S9b

Plot the same per-cell growth values on the tissue coordinates. Colour
denotes cell type. Point area and opacity increase linearly with growth,
using one scale across all days. These maps do not involve another model
evaluation.
"""), code("""
fig, axes = cb.pl.plot_growth_size_maps(
    growth, palette=HEART_LABEL_TO_COLOR, time_order=time_order,
    observed_times=["D4", "D7", "D10", "D14"],
)
fig.savefig(OUTPUT_DIR / "growth_spatial_maps.pdf", bbox_inches="tight")
fig.savefig(OUTPUT_DIR / "growth_spatial_maps.png", dpi=300, bbox_inches="tight")
plt.show()
"""), md("""
## Save the populations

Save one H5AD per day for the lineage and velocity notebooks. The spatial
and communication populations are saved separately, as returned by the
simulation. `manifest.json` records which files belong to each day.
"""), code("""
slice_records = []
for time, label in zip(result.ts_points, time_order):
    key = str(float(time))
    slug = f"{float(time):.9f}".rstrip("0").rstrip(".").replace(".", "p")
    record = {
        "time_float": float(time), "time_key": key, "time_label": label,
        "is_observed": label in {"D4", "D7", "D10", "D14"},
    }
    for populations, directory, field in (
        (result.adata_dict, SLICE_DIR, "slice_h5ad"),
        (result.communication_adata_dict, COMM_DIR, "communication_h5ad"),
    ):
        population = populations[key].copy()
        for name in ("time_float", "time_label", "is_observed"):
            population.obs[name] = record[name]
            population.uns[name] = record[name]
        path = directory / f"time_{slug}.h5ad"
        population.write_h5ad(path)
        record[field] = str(path)
    population = result.adata_dict[key]
    record.update(
        n_cells=population.n_obs,
        slice_origin=population.uns["slice_origin"],
        source_anchor_time=float(population.uns["source_anchor_time"]),
    )
    slice_records.append(record)
manifest = {"observed_times": observed_times, "slices": slice_records,
            "simulation_seeds": result.simulation_seeds}
(OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
pd.DataFrame(slice_records)[["time_label", "time_float", "is_observed", "n_cells"]]
"""), md("""
## Continue with the same simulation

The simulation, growth table, and plots are saved in the `OUTPUT_DIR` set
above. The saved `manifest.json` names every population and its time.

The [chicken-heart analysis notebooks](../paper_figures/chicken_heart_daily.md)
use these populations for lineage transitions, cell-type interactions, and
velocity analysis. Run their analysis steps rather than repeating interpolation.
The separate [S7–S8 alignment analysis](../paper_figures/chicken_heart_alignment.md)
compares perturbed input coordinates and independently fitted models.
""")]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "cytobridge": {"dataset": "chicken_heart", "runs_training": False,
                        "paper_panels": ["S9a", "S9b"],
                        "analysis_source": "reproduction/chicken_heart/notebooks/formal_daily_piecewise_interpolation_celltypecorrected.ipynb",
                        "model_commit": "c72e592"},
    })
    for i, cell in enumerate(notebook.cells):
        cell.id = f"heart-growth-{i:02d}"
    return notebook


if __name__ == "__main__":
    write_notebook(build_notebook(), ROOT / "docs/tutorials/dataset_workflows/chicken_heart.ipynb")
