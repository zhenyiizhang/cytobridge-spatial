#!/usr/bin/env python3
"""Keep paper notebooks explicit about inputs, upstream work, and outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "docs" / "tutorials" / "paper_figures"

WORKFLOWS = {
    "agist_figures.ipynb": "agist",
    "arista_figures.ipynb": "arista-lr",
    "arista_local_domains.ipynb": "arista-local-domains",
    "classifier_smoothing.ipynb": "classifier-smoothing",
    "compute_cost.ipynb": "compute-cost",
    "loto_benchmark.ipynb": "loto-benchmark",
    "lr_complex_aggregation.ipynb": "lr-complex",
    "lr_prior_ablation_stvcr.ipynb": "interaction-evidence",
    "main_figure_2.ipynb": "main-figure-2",
    "main_figure_4.ipynb": "main-figure-4",
    "main_figure_5.ipynb": "main-figure-5-reference",
    "mosta_figures.ipynb": "mosta-reference-pages",
    "nonspatial_figures.ipynb": "nonspatial",
    "training_histories.ipynb": "training-histories",
    "zebrafish_attention.ipynb": "zebrafish-attention",
    "zebrafish_si_s31_s38.ipynb": "zebrafish-si",
}

ROUTE_CELL_TAG = "cytobridge-reproduction-route"
DETAIL_CELL_TAG = "cytobridge-upstream-detail"
PREVIEW_CELL_TAG = "cytobridge-generated-preview"


def _markdown(source: str, tag: str):
    cell = nbformat.v4.new_markdown_cell(source.strip())
    cell.metadata["tags"] = [tag]
    return cell


def _code(source: str, tag: str):
    cell = nbformat.v4.new_code_cell(source.strip())
    cell.metadata["tags"] = [tag]
    return cell


def _remove_generated_cells(notebook) -> None:
    generated_tags = {ROUTE_CELL_TAG, DETAIL_CELL_TAG, PREVIEW_CELL_TAG}
    notebook.cells = [
        cell
        for cell in notebook.cells
        if generated_tags.isdisjoint(cell.metadata.get("tags", ()))
    ]


def _insert_route(notebook, workflow: str) -> None:
    route_markdown = _markdown(
        """
## Reproduction route

This route states where the plotted values begin, how the upstream analysis is
run when it is available, and what this notebook does not recalculate.
""",
        ROUTE_CELL_TAG,
    )
    route_code = _code(
        f"""
from CytoBridge.results import describe_figure_workflow

route = describe_figure_workflow({workflow!r})
{{
    key: route[key]
    for key in (
        "paper_location",
        "mode",
        "starts_from",
        "upstream_entry",
        "upstream_command",
        "figure_command",
        "scope",
    )
}}
""",
        ROUTE_CELL_TAG,
    )
    notebook.cells[1:1] = [route_markdown, route_code]


def _insert_after_source(notebook, marker: str, cells: list) -> None:
    for index, cell in enumerate(notebook.cells):
        if marker in "".join(cell.source):
            notebook.cells[index + 1 : index + 1] = cells
            return
    raise ValueError(f"Notebook cell marker not found: {marker}")


def _add_agist_input_registry(notebook) -> None:
    _insert_after_source(
        notebook,
        "data = load_agist_figures()",
        [
            _markdown(
                """
### Inputs needed before the S2–S3 redraw

The compact package contains the numerical values used by the plotting cells.
The table below lists the simulation data, model configuration, checkpoint,
and rollout files needed to rebuild those compact values. Those external files
are not generated in this notebook.
""",
                DETAIL_CELL_TAG,
            ),
            _code("display(data.full_recompute_inputs)", DETAIL_CELL_TAG),
        ],
    )


def _add_nonspatial_route(notebook) -> None:
    for cell in notebook.cells:
        if cell.cell_type == "code" and "from pathlib import Path" in "".join(
            cell.source
        ):
            source = "".join(cell.source)
            if "from IPython.display import Image, display" not in source:
                source = source.replace(
                    "from pathlib import Path\n",
                    "from pathlib import Path\n\nfrom IPython.display import Image, display\n",
                )
                cell.source = source
            break

    _insert_after_source(
        notebook,
        "results = load_nonspatial_figures()",
        [
            _markdown(
                """
### Full-analysis continuation

The complete Weinreb and scNT workflows are separate from the compact figure
bundle. They prepare the H5AD, train the full and no-interaction models, run
distribution and dataset-specific evaluations, and calculate interaction
attribution. Use the commands in
[Non-spatial workflows](../../nonspatial_workflows.md). The cell below shows
which archived analysis files were condensed into the released S4–S5 inputs.
""",
                DETAIL_CELL_TAG,
            ),
            _code(
                """
{
    dataset: {
        "upstream_files": len(details["full_rerun_inputs"]),
        "figure_builder": details["renderer"],
    }
    for dataset, details in results.external_inputs["datasets"].items()
}
""",
                DETAIL_CELL_TAG,
            ),
        ],
    )
    notebook.cells.extend(
        [
            _markdown(
                """
## Preview the generated figures

These PNG files were written by `plot_nonspatial_figures` in the preceding
cell; they are not input files.
""",
                PREVIEW_CELL_TAG,
            ),
            _code(
                """
for figure_id, (_, png_path) in figures.items():
    print(figure_id.upper())
    display(Image(filename=str(png_path), width=720))
""",
                PREVIEW_CELL_TAG,
            ),
        ]
    )


def _add_zebrafish_route(notebook) -> None:
    for cell in notebook.cells:
        if cell.cell_type == "code" and "from pathlib import Path" in "".join(
            cell.source
        ):
            source = "".join(cell.source)
            if "from IPython.display import Image, display" not in source:
                source = source.replace(
                    "from pathlib import Path\n",
                    "from pathlib import Path\n\nfrom IPython.display import Image, display\n",
                )
                cell.source = source
            break

    _insert_after_source(
        notebook,
        "results = load_zebrafish_si_results()",
        [
            _markdown(
                """
### Continue from a trained zebrafish model

For a new run, first use the
[zebrafish dataset workflow](../dataset_workflows/zebrafish.ipynb) to create
the aligned H5AD, training directory, and standard downstream outputs. The
paper-specific continuation is `scripts/run_zebrafish_paper_downstream.py`;
its live `--help` lists the required model, database, and run-record inputs.
The compact arrays loaded here are the released plotting form of those larger
outputs.
""",
                DETAIL_CELL_TAG,
            ),
            _code(
                """
{
    "source_directory": str(results.source_dir),
    "figure_mapping": {
        current: details["source_figure"]
        for current, details in results.manifest["figures"].items()
    },
}
""",
                DETAIL_CELL_TAG,
            ),
        ],
    )
    notebook.cells.extend(
        [
            _markdown(
                """
## Preview the generated figures

The preceding plotting call created these PNG files during this notebook run.
""",
                PREVIEW_CELL_TAG,
            ),
            _code(
                """
for figure_id, (_, png_path) in figures.items():
    print(figure_id.upper())
    display(Image(filename=str(png_path), width=720))
""",
                PREVIEW_CELL_TAG,
            ),
        ]
    )


def _rename_short_headings(notebook) -> None:
    replacements = {
        "## Load": "## Load figure inputs",
        "## Calculate": "## Recalculate panel values",
        "## Plot and save": "## Draw and save the figure",
        "## Preview": "## Preview the generated figure",
    }
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        source = "".join(cell.source)
        for old, new in replacements.items():
            if source.strip() == old:
                cell.source = new
                break


def update(path: Path, workflow: str) -> None:
    notebook = nbformat.read(path, as_version=4)
    _remove_generated_cells(notebook)
    _insert_route(notebook, workflow)
    if path.name == "agist_figures.ipynb":
        _add_agist_input_registry(notebook)
    elif path.name == "nonspatial_figures.ipynb":
        _add_nonspatial_route(notebook)
    elif path.name == "zebrafish_si_s31_s38.ipynb":
        _add_zebrafish_route(notebook)
    _rename_short_headings(notebook)
    nbformat.write(notebook, path)


def main() -> None:
    for name, workflow in WORKFLOWS.items():
        path = NOTEBOOK_DIR / name
        update(path, workflow)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
