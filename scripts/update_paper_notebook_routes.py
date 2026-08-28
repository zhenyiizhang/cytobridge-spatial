#!/usr/bin/env python3
"""Keep paper notebooks explicit about inputs, upstream work, and outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat

from CytoBridge.results.reproduction_chains import (
    describe_figure_reproduction_chain,
)


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


def _configure_dataframe_display(notebook) -> None:
    settings = """import pandas as pd

pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)

"""
    for cell in notebook.cells:
        if cell.cell_type == "code":
            source = "".join(cell.source)
            if 'pd.set_option("display.max_colwidth", None)' not in source:
                cell.source = settings + source
            return


def _step_markdown(row: dict[str, str], number: int) -> str:
    return f"""
### Step {number}: {row['step']}

**Paper:** {row['paper_part']}

```text
{row['code_or_command']}
```

**Reads:** `{row['reads']}`

**Writes:** `{row['writes']}`

**Next:** `{row['next_step']}`

**Availability:** {row['availability']}
"""


def _insert_route(notebook, workflow: str) -> None:
    route_markdown = _markdown(
        """
## Reproduction route: files passed between analysis steps

Read the steps from top to bottom. Each one names the code that runs, the files
it reads, the files it writes, and the next command that consumes those files.
Steps marked `manuscript result bundle` record the files used for the paper;
public steps can be run from this checkout.

Replace text inside angle brackets with your own path or value; do not type the
brackets. A choice such as `<weinreb|scnt_cortex>` means run the command with
one of those two values, not the text containing the vertical bar.
""",
        ROUTE_CELL_TAG,
    )
    route_steps = [
        _markdown(_step_markdown(row, number), ROUTE_CELL_TAG)
        for number, row in enumerate(
            describe_figure_reproduction_chain(workflow), start=1
        )
    ]
    notebook.cells[1:1] = [route_markdown, *route_steps]


def _insert_after_source(notebook, marker: str, cells: list) -> None:
    for index, cell in enumerate(notebook.cells):
        if marker in "".join(cell.source):
            notebook.cells[index + 1 : index + 1] = cells
            return
    raise ValueError(f"Notebook cell marker not found: {marker}")


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
    _configure_dataframe_display(notebook)
    _insert_route(notebook, workflow)
    if path.name == "nonspatial_figures.ipynb":
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
