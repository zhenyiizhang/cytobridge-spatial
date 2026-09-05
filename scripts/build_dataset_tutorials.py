#!/usr/bin/env python3
"""Build the five public dataset tutorial notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "tutorials" / "dataset_workflows"
OWN_DATA_NOTEBOOK = ROOT / "docs" / "tutorials" / "your_data.ipynb"
SYNTHETIC_NOTEBOOK = (
    ROOT / "docs" / "tutorials" / "data_preparation" / "synthetic_preprocessing.ipynb"
)


@dataclass(frozen=True)
class Tutorial:
    dataset: str
    title: str
    raw_filename: str
    figure_links: tuple[tuple[str, str, str], ...]


TUTORIALS = (
    Tutorial(
        "zebrafish",
        "Zebrafish embryogenesis",
        "zebrafish_raw.h5ad",
        (
            (
                "Supplementary Figures S31–S38",
                "zebrafish_si_s31_s38.ipynb",
                "zebrafish-si",
            ),
            (
                "Supplementary Figure S39",
                "zebrafish_attention.ipynb",
                "zebrafish-attention",
            ),
            (
                "Supplementary Figure S40",
                "zebrafish_decomposition_stability.md",
                "zebrafish-decomposition-stability",
            ),
        ),
    ),
    Tutorial(
        "mosta",
        "MOSTA mouse organogenesis",
        "mosta_raw.h5ad",
        (
            ("Main Figure 4", "main_figure_4.ipynb", "main-figure-4"),
            (
                "Supplementary Figures S11–S18",
                "mosta_figures.ipynb",
                "mosta-reference-pages",
            ),
        ),
    ),
    Tutorial(
        "arista",
        "ARISTA salamander brain regeneration",
        "arista_raw.h5ad",
        (
            (
                "Main Figure 5",
                "main_figure_5.ipynb",
                "main-figure-5-reference",
            ),
            (
                "Supplementary Figures S19–S24",
                "arista_figures.ipynb",
                "arista-lr",
            ),
            (
                "Supplementary Figure S25",
                "arista_local_domains.ipynb",
                "arista-local-domains",
            ),
        ),
    ),
    Tutorial(
        "admouse",
        "AD mouse brain",
        "admouse_raw.h5ad",
        (
            (
                "LR-prior and interaction ablations",
                "interaction_ablation.ipynb",
                "lr-prior-stvcr",
            ),
            ("Five-dataset benchmark", "loto_benchmark.ipynb", "loto-benchmark"),
            (
                "Training histories",
                "training_histories.ipynb",
                "training-histories",
            ),
        ),
    ),
    Tutorial(
        "chicken_heart",
        "Developing chicken heart",
        "chicken_heart_raw.h5ad",
        (
            (
                "Cell-state trajectories, lineage transitions, and growth",
                "chicken_heart_daily.md",
                "chicken-heart-daily",
            ),
            (
                "Supplementary Figures S7–S8: alignment sensitivity",
                "chicken_heart_alignment.md",
                "chicken-heart-alignment",
            ),
            (
                "LR-prior and interaction ablations",
                "interaction_ablation.ipynb",
                "lr-prior-stvcr",
            ),
            ("Five-dataset benchmark", "loto_benchmark.ipynb", "loto-benchmark"),
            (
                "Training histories",
                "training_histories.ipynb",
                "training-histories",
            ),
        ),
    ),
)


def markdown(text: str, *, cell_id: str | None = None):
    cell = nbformat.v4.new_markdown_cell(text.strip())
    if cell_id is not None:
        cell["id"] = cell_id
    return cell


def code(text: str, *, cell_id: str | None = None):
    cell = nbformat.v4.new_code_cell(text.strip())
    if cell_id is not None:
        cell["id"] = cell_id
    return cell




def build_notebook(tutorial: Tutorial):
    if tutorial.dataset == "admouse":
        from build_admouse_population_tutorial import build_notebook as build_admouse

        return build_admouse()
    if tutorial.dataset == "arista":
        from build_arista_growth_tutorial import build_notebook as build_arista

        return build_arista()
    if tutorial.dataset == "mosta":
        from build_mosta_trajectory_tutorial import build_notebook as build_mosta

        return build_mosta()
    if tutorial.dataset == "zebrafish":
        from build_zebrafish_growth_tutorial import build_notebook as build_zebrafish

        return build_zebrafish()
    if tutorial.dataset == "chicken_heart":
        from build_chicken_heart_tutorial import build_notebook as build_heart

        return build_heart()
    from build_analysis_tutorials import build_analysis_notebook

    return build_analysis_notebook(tutorial)


def build_own_data_notebook():
    """Show the input fields and one complete training command."""
    cells = [
        markdown("""
# Configure a command-line workflow

Start with an AnnData file containing raw counts, a time label, cell-type labels,
and spatial coordinates. This guide shows how to tell CytoBridge where those
fields are stored, then run preprocessing, training, and analysis with one
command. To work through the Python functions individually, use
[Train a model](../training.md).

The [small preprocessing example](data_preparation/synthetic_preprocessing.ipynb)
shows how to create an AnnData object. To reuse an existing model, see
[Continue from a trained model](../reuse_model.md).
"""),
        markdown("""
## Describe your input

In the example below, counts are in `layers['counts']`, time is in
`obs['stage']`, cell type is in `obs['cell_type']`, and two-dimensional
coordinates are in `obsm['spatial']`.

Use raw, non-negative counts, unique observation names, and gene symbols
matching your LR table. Do not normalize or run PCA first: preprocessing does
that.
"""),
        markdown("""
## Create a configuration

The following cells load an example configuration, change its input fields for
observations at stages D0, D2, and D4, and show the settings before you save it.
Edit the values in these cells for your experiment. They create a new
configuration rather than editing an existing JSON file.
"""),
        code("""
from pathlib import Path
import json
import pandas as pd

from CytoBridge.workflow import load_workflow_config

CONFIG_PATH = Path("configs/my_dataset.json")
config, _ = load_workflow_config("zebrafish")

config["dataset"]["name"] = "my_dataset"
config["preprocess"]["time_key"] = "stage"
config["preprocess"]["annotation_source"] = "cell_type"
config["preprocess"]["batch_indices"] = None  # Use all observed stages.
align = config["preprocess"]["align"]
align["expression_layer"] = "counts"
align["input_spatial_key"] = "spatial"
align.pop("spatial_obs_keys", None)
align["time_mapping"] = {"D0": 0, "D2": 1, "D4": 2}
config["downstream"]["observed"] = [0, 1, 2]
config["downstream"]["interpolated"] = [0.5, 1.5]
"""),
        markdown("""
These are example times, not values to copy unchanged. Replace them with your
measured stages and the times you want to predict. Keep the ordering and
relative spacing of the observed times consistent with the experiment.
The example removes the Zebrafish-specific stage selection so that all three
stages are used.

The table below lists the main settings to review before saving. In particular,
choose a suitable LR database and species, spatial scale, and neighborhood
size. The dataset configurations contain the paper's settings. The
[dataset tutorials](dataset_workflows/index.md) cover analysis after training.
"""),
        code("""
pd.DataFrame({
    "Setting": [
        "Count layer", "Time column", "Cell-type column", "Coordinates",
        "Time mapping", "LR species", "Interaction neighborhood",
    ],
    "Value": [
        align["expression_layer"],
        config["preprocess"]["time_key"],
        config["preprocess"]["annotation_source"],
        align["input_spatial_key"],
        str(align["time_mapping"]),
        config["downstream"].get("preferred_species_tag"),
        config["train"].get("interaction_cutoff"),
    ],
})
"""),
        markdown("""
After editing the values, save the configuration. Run this cell locally:

```python
CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\\n")
```

The configuration above starts from Zebrafish settings. For another species,
supply your LR CSV in the command below and set
`downstream.preferred_species_tag` to its species tag. For graph construction,
the CSV needs `Ligand`, `Receptor`, `Pathway`, and `Annotation` columns, as in
the included CellChatDB tables. Keep the interaction-type annotations from
the database.
"""),
        markdown("""
## Train and calculate results

Put your input at `data/my_dataset_raw.h5ad`, then run:

```bash
cytobridge workflow --config configs/my_dataset.json --train \
  --input-h5ad data/my_dataset_raw.h5ad \
  --output-dir outputs/my_dataset --device cuda
```

This **one command** preprocesses the input, fits the LR edge predictor, trains
the model, and calculates the downstream results. Do not run `preprocess`
first.

If you are using a custom LR table, use this version of the same command:

```bash
cytobridge workflow --config configs/my_dataset.json --train \
  --input-h5ad data/my_dataset_raw.h5ad \
  --graph-database data/my_ligand_receptor_table.csv \
  --lr-database data/my_ligand_receptor_table.csv \
  --output-dir outputs/my_dataset --device cuda
```

Choose one of these commands. The second supplies the same LR table for
training and downstream analysis.
"""),
        markdown("""
## Open the results

The trained model is in `outputs/my_dataset/training/`. Growth, velocities,
trajectories, and interaction tables are in `outputs/my_dataset/downstream/`.
Open its `summary.json` for the list of analyses and `figures/` for the plots.

To change the downstream analysis without training again, follow
[Continue from a trained model](../reuse_model.md) with your own configuration,
aligned H5AD, and training directory.
"""),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def build_synthetic_preprocessing_notebook():
    """Build a small, fully executable preprocessing example."""

    cells = [
        markdown(
            """
# CytoBridge synthetic preprocessing

This notebook creates a small spatial count matrix and runs
`CytoBridge.pp.preprocess`. It requires Python 3.10 or later, AnnData, and
`CytoBridge[preprocess]` installed in the current Jupyter kernel.

The example uses generated data. Replace the data-construction cell with a
dataset loader that provides raw counts, a time column, cell-type labels, and
spatial coordinates.
""",
            cell_id="intro",
        ),
        code(
            """
from __future__ import annotations

from io import BytesIO
import platform
from importlib.metadata import version

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from IPython.display import Image, display

import CytoBridge

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
    }
)

SEED = 42
rng = np.random.default_rng(SEED)
environment = {
    "cytobridge": version("CytoBridge"),
    "python": platform.python_version(),
    "seed": SEED,
}
environment
""",
            cell_id="setup",
        ),
        markdown(
            """
## 1. Create the input AnnData object

The input contains non-negative integer counts in `layers['counts']`, stage and
annotation columns in `obs`, and two spatial coordinates in
`obsm['spatial']`. The assertions check the array type, range, and dimensions
before preprocessing.
""",
            cell_id="raw-input",
        ),
        code(
            """
n_cells, n_genes = 72, 40
stages = np.repeat(np.array(["E0", "E1", "E2"]), n_cells // 3)
counts = rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)
counts[stages == "E1", :5] += 2
counts[stages == "E2", 5:10] += 3
counts[:, 39] = 0  # deterministic low-information marker used in the PCA feature check
spatial = np.column_stack(
    [
        np.linspace(0.0, 1.0, n_cells),
        rng.normal(0.0, 0.08, size=n_cells),
    ]
).astype(np.float32)

obs = pd.DataFrame(
    {
        "stage": stages,
        "Annotation": np.where(np.arange(n_cells) % 2 == 0, "TypeA", "TypeB"),
    },
    index=[f"Cell{i:03d}" for i in range(n_cells)],
)
var = pd.DataFrame(index=[f"Gene{i:03d}" for i in range(n_genes)])
adata = AnnData(X=counts.copy(), obs=obs, var=var)
adata.layers["counts"] = counts.copy()
adata.obsm["spatial"] = spatial

assert np.isfinite(counts).all() and (counts >= 0).all()
assert np.allclose(counts, np.rint(counts), rtol=0.0, atol=0.0)
assert adata.obsm["spatial"].shape == (n_cells, 2)
adata.obs.groupby(["stage", "Annotation"], observed=True).size().rename("cells")
""",
            cell_id="make-data",
        ),
        markdown(
            """
## 2. Run preprocessing

`expression_layer='counts'` selects the raw source. Strict validation checks
the selected layer before `normalize_total(target_sum=10000)` and `log1p`.
Highly variable genes define the PCA fit without removing genes from the
expression matrix, and requested features are added to that fit when needed.
""",
            cell_id="preprocessing",
        ),
        code(
            """
processed = CytoBridge.pp.preprocess(
    adata.copy(),
    time_key="stage",
    time_mapping={"E0": 0.0, "E1": 1.0, "E2": 2.0},
    n_top_genes=20,
    n_pcs=8,
    expression_layer="counts",
    raw_count_validation="strict",
    required_latent_features=["Gene000", "Gene039"],
)

{
    "X_shape": processed.X.shape,
    "X_latent_shape": processed.obsm["X_latent"].shape,
    "mapped_times": sorted(
        processed.obs["time_point_processed"].unique().tolist()
    ),
}
""",
            cell_id="run-preprocess",
        ),
        markdown(
            """
## 3. Check preprocessing metadata and arrays

The processed AnnData records the expression source, validation mode,
transformation order, time mapping, PCA feature count, and PCA center in
`uns['preprocess_info']`. The checks below also verify array shapes, finite
values, mapped times, and requested PCA features.
""",
            cell_id="metadata",
        ),
        code(
            """
info = processed.uns["preprocess_info"]
assert info["expression_source"] == "layers['counts']"
assert info["raw_count_validation_effective"] == "strict"
assert info["transformation_sequence"] == ["normalize_total", "log1p"]
assert processed.obsm["X_latent"].shape == (n_cells, 8)
assert processed.var["pca_center"].shape == (n_genes,)
assert np.isfinite(processed.obsm["X_latent"]).all()
assert np.isfinite(processed.var["pca_center"].to_numpy()).all()
assert np.allclose(processed.obsm["X_latent"].mean(axis=0), 0.0, atol=1e-5)
assert sorted(processed.obs["time_point_processed"].unique().tolist()) == [
    0.0,
    1.0,
    2.0,
]
assert all(
    bool(processed.var.loc[name, "highly_variable"])
    for name in ["Gene000", "Gene039"]
)

preprocess_summary = {
    "expression_source": info["expression_source"],
    "raw_count_validation": info["raw_count_validation_effective"],
    "transformations": info["transformation_sequence"],
    "n_latent_fit_features": info["n_latent_fit_features"],
    "latent_shape": processed.obsm["X_latent"].shape,
    "latent_all_finite": bool(np.isfinite(processed.obsm["X_latent"]).all()),
    "mapped_times": sorted(
        processed.obs["time_point_processed"].unique().tolist()
    ),
    "required_features_in_pca": ["Gene000", "Gene039"],
}
preprocess_summary
""",
            cell_id="inspect-metadata",
        ),
        markdown(
            """
## 4. Plot the processed coordinates

The left panel shows the spatial coordinates supplied in the input AnnData.
The right panel uses the first two columns of the `X_latent` matrix produced by
the preprocessing call above.
""",
            cell_id="plot-result",
        ),
        code(
            """
stage_colors = {"E0": "#59616A", "E1": "#07838B", "E2": "#D28C3C"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

for stage in stage_colors:
    mask = processed.obs["stage"].astype(str).to_numpy() == stage
    axes[0].scatter(
        processed.obsm["spatial"][mask, 0],
        processed.obsm["spatial"][mask, 1],
        s=22,
        color=stage_colors[stage],
        linewidth=0,
        label=stage,
    )
    axes[1].scatter(
        processed.obsm["X_latent"][mask, 0],
        processed.obsm["X_latent"][mask, 1],
        s=22,
        color=stage_colors[stage],
        linewidth=0,
        label=stage,
    )

for letter, ax in zip(("a", "b"), axes):
    ax.text(
        -0.15,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set(
    title="Input spatial coordinates",
    xlabel="Spatial coordinate 1",
    ylabel="Spatial coordinate 2",
)
axes[1].set(
    title="Processed latent coordinates",
    xlabel="PC 1",
    ylabel="PC 2",
)
axes[1].legend(frameon=False, title="Stage", markerscale=0.9)
fig.tight_layout()
image_buffer = BytesIO()
fig.savefig(image_buffer, format="png", dpi=144, bbox_inches="tight")
plt.close(fig)
display(Image(data=image_buffer.getvalue()))
""",
            cell_id="plot-processed",
        ),
        markdown(
            """
## 5. Validate reuse of a processed object

Passing the transformed matrix through the default preprocessing path a second
time raises a `ValueError`. Start a new preprocessing run from the raw count
layer.
""",
            cell_id="input-validation",
        ),
        code(
            """
try:
    CytoBridge.pp.preprocess(
        processed.copy(),
        time_key="stage",
        n_top_genes=20,
        n_pcs=8,
    )
except ValueError as exc:
    message = str(exc)
    assert "double-transform" in message
    print(message.splitlines()[0])
else:
    raise AssertionError("Expected preprocessing to reject transformed X")
""",
            cell_id="double-transform-check",
        ),
        markdown(
            """
## Outputs

`processed` contains the normalized expression matrix, `obsm['X_latent']`,
PCA metadata, mapped numeric times, and the original spatial coordinates.
`preprocess_summary` collects the fields checked in this example, and the plot
above is drawn directly from `processed`.
""",
            cell_id="outputs",
        ),
    ]
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def write_notebook(notebook, path):
    """Preserve outputs only if the entire code sequence has not changed."""
    if path.exists():
        previous = nbformat.read(path, as_version=4)
        old = [c for c in previous.cells if c.cell_type == "code"]
        new = [c for c in notebook.cells if c.cell_type == "code"]
        if [c.source for c in old] == [c.source for c in new]:
            for before, after in zip(old, new):
                after.outputs = before.outputs
                after.execution_count = before.execution_count
    nbformat.write(notebook, path)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for tutorial in TUTORIALS:
        path = NOTEBOOK_DIR / f"{tutorial.dataset}.ipynb"
        write_notebook(build_notebook(tutorial), path)
        print(path.relative_to(ROOT))
    write_notebook(build_own_data_notebook(), OWN_DATA_NOTEBOOK)
    print(OWN_DATA_NOTEBOOK.relative_to(ROOT))
    SYNTHETIC_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    write_notebook(build_synthetic_preprocessing_notebook(), SYNTHETIC_NOTEBOOK)
    print(SYNTHETIC_NOTEBOOK.relative_to(ROOT))


if __name__ == "__main__":
    main()
