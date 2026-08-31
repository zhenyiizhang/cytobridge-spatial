#!/usr/bin/env python3
"""Add the calculation steps and saved previews to every paper notebook."""

from __future__ import annotations

from pathlib import Path
import re

import nbformat

from CytoBridge.results.reproduction_chains import describe_figure_steps


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
    "lr_prior_ablation_stvcr.ipynb": "lr-prior-stvcr",
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
    generated_tags = {ROUTE_CELL_TAG, DETAIL_CELL_TAG}
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
    note_text = re.sub(
        r"(?<!`)(<[^<>\n]+>)(?!`)", r"`\1`", row.get("note", "")
    )
    note = f"\n{note_text}\n" if note_text else ""
    next_line = ""
    if row["next_step"] not in {
        "finished figure",
        "finished figures",
        "finished page copy",
    }:
        next_line = f"\nNext: `{row['next_step']}`\n"
    if row.get("entry_type") == "source":
        entry = f"Source files: `{row['code_or_command']}`"
    else:
        language = "python" if row["code_or_command"].startswith("from ") else "text"
        entry = f"""```{language}
{row['code_or_command']}
```"""
    return f"""
### {number}. {row['step']} ({row['paper_part']})

{entry}

Start with: `{row['reads']}`

Writes: `{row['writes']}`
{next_line}
{note}
"""


def _insert_route(notebook, workflow: str) -> None:
    route_markdown = _markdown(
        """
## Run the notebook

The plotting cells use numerical files included with CytoBridge and write new
PDF and PNG files. The steps below list the commands and source files that
produced those inputs. Each step names what it reads, what it writes, and what
comes next.

A command can be repeated when you have the inputs named under **Start with**.
**Source files** points to calculation code kept with the paper results. The
plotting cells use a new run only when the final plotting command explicitly
accepts that run's result directory. The paper figure index marks notebooks
where this conversion step is not yet available.

Replace text inside angle brackets with your path or value. For example,
`<output-dir>` means the directory where you want the files to be written.
""",
        ROUTE_CELL_TAG,
    )
    route_steps = [
        _markdown(_step_markdown(row, number), ROUTE_CELL_TAG)
        for number, row in enumerate(
            describe_figure_steps(workflow), start=1
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

    if any(
        PREVIEW_CELL_TAG in cell.metadata.get("tags", ())
        for cell in notebook.cells
    ):
        return

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

    if any(
        PREVIEW_CELL_TAG in cell.metadata.get("tags", ())
        for cell in notebook.cells
    ):
        return

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


def _humanize_existing_markdown(notebook) -> None:
    replacements = {
        "Recalculate the plotted summaries and rebuild both figures from the packaged processed inputs. This compact workflow does not run model training or regenerate the processed inputs.": "The calculation cells read the included per-cell velocity and synthetic-benchmark files, recalculate the displayed summaries, and draw S2 and S3. The steps below show how to generate the simulation, train the model, and evaluate the rollout.",
        "Supplementary Figures S23 and S24 are table-driven scientific redraws: the notebook clusters all 531 released ligand–receptor time courses, selects 25 representative pairs per cluster, and draws both figures from the tables calculated in this run. The complete PDF pages remain available as separate references. Figures S19–S22 are included only as released reference pages because their complete layout inputs are not all distributed with the wheel.": "This notebook recalculates S23 and S24 from all 531 ligand–receptor time courses, selects 25 representative pairs per cluster, and draws both figures. For S19–S22, it lists the original calculation and rendering files and shows the completed pages because some layout inputs are stored with the paper results rather than the installed package.",
        "Load the packaged tables, recalculate the panel values, and save the figure.": "Read the included domain, null-model, pathway, and ligand–receptor tables; recalculate the displayed values; and draw the figure.",
        "Load the compact processed results and reproduce the three figure panels.": "Read the per-dataset neighbor-sweep and generated-state sensitivity tables, recalculate the panel summaries, and draw all three panels.",
        "Load the packaged measurements, apply manuscript units, and save the table.": "Read the timing and memory measurements from the five training runs, convert the units used in the paper, and write the table.",
        "Load the compact target-level results, recalculate the matched ratios, and reproduce the manuscript figure.": "Read the target-level benchmark results, recalculate each matched ratio, and draw the figure.",
        "Load paired LR scores, recalculate the panel tables, and reproduce the manuscript figure.": "Read the paired ligand–receptor scores, recalculate the panel tables, and draw the figure.",
        "Load the two compact paired-error tables and reproduce the four figure panels.": "Read the paired No-LR and stVCR error tables, recalculate the displayed comparisons, and draw all four panels.",
        "This page assembles two kinds of input. Panels a–d come from the existing vector page. Panel e is redrawn from included W2 summary tables. Loading also recalculates the CytoBridge mean and sample SD from the replicate table and requires an exact match. The notebook does not rerun model training, simulations, or baseline methods.": "This notebook recalculates panel e from the included replicate-level W2 table, checks the mean and sample SD, and combines it with the existing vector panels a–d. The steps below identify the simulation, model evaluation, and baseline results used for the complete figure.",
        "This notebook assembles the complete page from five released vector panel PDFs and two stored page-space connectors. It does not recalculate the five panels. The figure index records the calculation and rendering scripts in the repository release.": "This notebook assembles Main Figure 4 from its five completed vector panels. The steps below show the MOSTA downstream command and the calculation and rendering files for each panel; those panel-building files are stored with the paper results.",
        "This notebook validates and exports the packaged scientific-label reference page. It copies the PNG without changing its pixels and places the same raster on an A4 PDF page. It does not recalculate panel values or rebuild vector objects.": "This page checks the assembled Main Figure 5 image and writes viewable PDF and PNG copies. The steps below show the ARISTA downstream run and the panel-building files used before page assembly.",
        "Export the released vector PDF and SVG pages under the current supplementary numbering and render PNG previews. This notebook does not rerun the numerical analyses or redraw the pages. The figure index points to the calculation and rendering scripts in the repository release.": "This notebook indexes the calculation and rendering files for S11–S18, then writes viewable copies of the completed vector pages under the current supplementary numbering. The numerical panel-building files are stored with the paper results.",
        "The final cells redraw S4 and S5 from the released numerical bundle. The route below shows the earlier data-preparation, Full-model training, No-interaction training, and downstream commands that produced that bundle. Those long-running training commands are documented here but are not executed during the documentation build.": "The calculation cells read the included cell arrays and result tables, recalculate every displayed summary, and draw S4 and S5. The steps below show how raw data proceeds through Full and No-interaction training, evaluation, attribution, and dataset-specific panel preparation.",
        "Load the compact history, calculate stage-specific centered moving means, and save the figure and panel tables.": "Read the per-epoch training histories, calculate stage-specific centered moving means, and draw the figure and panel tables.",
        "This notebook loads the packaged panel data, recalculates the displayed summaries, and saves the PDF and PNG. It does not run model training or download external data.": "The calculation cells read the directed-pair, expression, JAM-control, and spatial-null tables, recalculate the displayed summaries, and draw S44. The steps below show how those tables are produced from model and comparison-method outputs.",
        "This notebook loads the compact packaged inputs, recalculates the plotted quantities, and saves all eight PDF and PNG figure pairs. It does not train a model or download external data.": "The calculation cells read the included state arrays and analysis tables, recalculate every displayed value, and draw S31–S38. The steps below show how training, downstream analysis, and the two sensitivity analyses produce those inputs.",
        "The formal PDFs remain separate references.": "The complete PDF pages remain available as separate references.",
        "## Formal sources and full-recalculation inputs": "## Source files for the complete pages",
        "packaged frozen vector page": "existing vector page",
        "packaged W2 summaries": "included W2 summary tables",
        "The formal page archive is available": "The complete page files are available",
        "# Supplementary Figure S39: interaction evidence": "# Supplementary Figure S39: LR-prior ablation and stVCR comparison",
        "# Supplementary Figure S43: zebrafish attention validation": "# Supplementary Figure S44: zebrafish attention and control comparisons",
        "evaluate the rollout.": "evaluate the generated trajectories.",
        "## Validate the reference page": "## Check the reference page",
        "## Check the reference page": "## Check the assembled figure",
        "## Export the reference page": "## Save PDF and PNG copies",
        "## Released reference pages for S19–S22": "## Released pages for S19–S22",
        "## Preview the table-driven scientific redraws": "## Preview S23 and S24",
        "## Load the current-number mapping": "## Match release files to S11–S18",
        "## Upstream analysis and API": "## Use results from another run",
        "The command-line entry is `scripts/results/plot_interaction_evidence.py`. The reader API is in `CytoBridge/results/interaction_evidence.py`. Model fitting and projection generation are upstream of the packaged tables.": "Pass another directory of paired No-LR and stVCR result tables to the installed command:\n\n```bash\ncytobridge figure lr-prior-stvcr --results-dir <paired-results> --output-dir outputs/lr_prior_stvcr\n```",
        "stVCR's native output support": "predictions produced directly by stVCR",
        "The steps above": "The steps below",
    }
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        source = "".join(cell.source)
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell.source = source


def _simplify_arista_source_table(notebook) -> None:
    """Keep the useful S19-S22 source map without exposing archive bookkeeping."""

    for cell in notebook.cells:
        if cell.cell_type == "code" and "formal_release =" in "".join(cell.source):
            cell.source = """display(
    data.full_recompute_inputs[["input_id", "stage", "figures"]]
)"""
            cell.outputs = []
            cell.execution_count = None
        elif cell.cell_type == "code" and "load_arista_figure_release," in "".join(
            cell.source
        ):
            cell.source = "".join(cell.source).replace(
                "    load_arista_figure_release,\n", ""
            ).replace(
                "    write_arista_source_index,\n", ""
            )

    for cell in notebook.cells:
        source = "".join(cell.source)
        if (
            cell.cell_type == "code"
            and "reference_pages = export_arista_reference_pages" in source
        ):
            if "for figure, (_, png_path) in reference_pages.items():" not in source:
                cell.source = source.rstrip() + """

for figure, (_, png_path) in reference_pages.items():
    print(figure)
    display(Image(filename=str(png_path), width=720))
"""
            break

    calculation_start = next(
        index
        for index, cell in enumerate(notebook.cells)
        if "## Calculate S23 and S24" in "".join(cell.source)
    )
    reference_start = next(
        index
        for index, cell in enumerate(notebook.cells)
        if "## Released pages for S19–S22" in "".join(cell.source)
    )
    if reference_start > calculation_start:
        reference_cells = notebook.cells[reference_start:]
        del notebook.cells[reference_start:]
        notebook.cells[calculation_start:calculation_start] = reference_cells


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
    _humanize_existing_markdown(notebook)
    if path.name == "lr_prior_ablation_stvcr.ipynb":
        for cell in notebook.cells:
            if cell.cell_type == "code":
                source = "".join(cell.source)
                source = source.replace(
                    """from CytoBridge.results import (
    load_interaction_evidence_results,
    plot_interaction_evidence,
)
from CytoBridge.results.interaction_evidence import (
    interaction_evidence_statistics,
    write_interaction_evidence_tables,
)""",
                    """from CytoBridge.results import (
    load_lr_prior_stvcr_results,
    lr_prior_stvcr_statistics,
    plot_lr_prior_stvcr,
    write_lr_prior_stvcr_tables,
)""",
                )
                replacements = {
                    "load_interaction_evidence_results": "load_lr_prior_stvcr_results",
                    "interaction_evidence_statistics": "lr_prior_stvcr_statistics",
                    "write_interaction_evidence_tables": "write_lr_prior_stvcr_tables",
                    "plot_interaction_evidence": "plot_lr_prior_stvcr",
                    'Path("outputs") / "interaction_evidence"': 'Path("outputs") / "lr_prior_stvcr"',
                }
                for old, new in replacements.items():
                    source = source.replace(old, new)
                cell.source = source
    if path.name == "arista_figures.ipynb":
        _simplify_arista_source_table(notebook)
    nbformat.write(notebook, path)


def main() -> None:
    for name, workflow in WORKFLOWS.items():
        path = NOTEBOOK_DIR / name
        update(path, workflow)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
