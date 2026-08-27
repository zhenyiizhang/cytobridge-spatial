"""Installed entry points for the paper-figure notebooks.

The workflow label says what the command actually does.  A numerical redraw
starts from released arrays or tables.  A reference export copies a released
page, and an external assembly places already-rendered panels on a page.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from ._cli import new_output_dir, write_run_summary


@dataclass(frozen=True)
class FigureWorkflow:
    """Description of one public figure command."""

    name: str
    paper_location: str
    mode: str
    wheel_runnable: bool
    description: str


FIGURE_WORKFLOWS = (
    FigureWorkflow(
        "agist",
        "Supplementary Figures S2-S3",
        "numeric-redraw",
        True,
        "Recalculate AGIST panel values and draw both figures.",
    ),
    FigureWorkflow(
        "nonspatial",
        "Supplementary Figures S4-S5",
        "numeric-redraw",
        True,
        "Recalculate the grouped Weinreb and scNT panels and draw both figures.",
    ),
    FigureWorkflow(
        "classifier-smoothing",
        "Supplementary Figure S6",
        "numeric-redraw",
        True,
        "Summarize the released sensitivity results and draw the figure.",
    ),
    FigureWorkflow(
        "arista-lr",
        "Supplementary Figures S21-S22",
        "numeric-redraw",
        True,
        "Recluster all 531 ARISTA LR profiles and draw the corrected figures.",
    ),
    FigureWorkflow(
        "lr-complex",
        "Supplementary Figure S23",
        "numeric-redraw",
        True,
        "Recalculate LR-complex sensitivity summaries and draw the figure.",
    ),
    FigureWorkflow(
        "zebrafish-si",
        "Supplementary Figures S27-S34",
        "numeric-redraw",
        True,
        "Recalculate the released zebrafish panel values and draw eight figures.",
    ),
    FigureWorkflow(
        "interaction-evidence",
        "Supplementary Figure S35",
        "numeric-redraw",
        True,
        "Summarize the matched No-LR and stVCR results and draw the figure.",
    ),
    FigureWorkflow(
        "loto-benchmark",
        "Supplementary Figure S36",
        "numeric-redraw",
        True,
        "Recalculate matched benchmark ratios and draw the figure.",
    ),
    FigureWorkflow(
        "training-histories",
        "Supplementary Figure S37",
        "numeric-redraw",
        True,
        "Smooth the released per-epoch losses and draw the training histories.",
    ),
    FigureWorkflow(
        "arista-local-domains",
        "Supplementary Figure S38",
        "numeric-redraw",
        True,
        "Recalculate the displayed domain summaries and draw the figure.",
    ),
    FigureWorkflow(
        "zebrafish-attention",
        "Supplementary Figure S39",
        "numeric-redraw",
        True,
        "Recalculate the released attention summaries and draw the figure.",
    ),
    FigureWorkflow(
        "compute-cost",
        "Supplementary Table 2",
        "table-only",
        True,
        "Format the released runtime and memory measurements.",
    ),
    FigureWorkflow(
        "main-figure-2",
        "Main Figure 2",
        "result-summary-redraw + external-assembly",
        True,
        "Draw panel e and place it over the packaged frozen panels a-d.",
    ),
    FigureWorkflow(
        "main-figure-5-reference",
        "Main Figure 5",
        "reference-export",
        True,
        "Validate and export the released compact reference page.",
    ),
    FigureWorkflow(
        "main-figure-4",
        "Main Figure 4",
        "external-assembly",
        False,
        "Assemble five vector panels from a separately downloaded MOSTA release.",
    ),
    FigureWorkflow(
        "mosta-reference-pages",
        "Supplementary Figures S9-S16",
        "reference-export",
        False,
        "Export released vector pages from a separately downloaded MOSTA release.",
    ),
)

_WORKFLOW_BY_NAME = {workflow.name: workflow for workflow in FIGURE_WORKFLOWS}


def list_figure_workflows() -> list[dict[str, object]]:
    """Return the public figure commands and their execution modes."""

    return [asdict(workflow) for workflow in FIGURE_WORKFLOWS]


def _paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_paths(item) for item in value]
    if isinstance(value, list):
        return [_paths(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _paths(item) for key, item in value.items()}
    return value


def _standard_run(
    results_dir: Path | None,
    output: Path,
    *,
    loader: Callable[[Path | None], Any],
    table_writer: Callable[[Any, Path], Any],
    plotter: Callable[[Any, Path], Any],
) -> dict[str, object]:
    data = loader(results_dir)
    tables = table_writer(data, output)
    figures = plotter(data, output)
    return {
        "input_directory": str(data.source_dir),
        "figures": _paths(figures),
        "tables": _paths(tables),
    }


def _run_agist(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .agist_figures import (
        calculate_agist_figure_panels,
        load_agist_figures,
        plot_agist_figures,
        write_agist_figure_tables,
    )

    data = load_agist_figures(results_dir)
    panels = calculate_agist_figure_panels(data)
    return {
        "input_directory": str(data.source_dir),
        "figures": _paths(plot_agist_figures(data, panels, output)),
        "tables": _paths(write_agist_figure_tables(panels, output)),
    }


def _run_nonspatial(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .nonspatial_figures import (
        calculate_nonspatial_panels,
        load_nonspatial_figures,
        plot_nonspatial_figures,
        write_nonspatial_tables,
    )

    data = load_nonspatial_figures(results_dir)
    panels = calculate_nonspatial_panels(data)
    return {
        "input_directory": str(data.source_dir),
        "figures": _paths(plot_nonspatial_figures(data, output, panels)),
        "tables": _paths(write_nonspatial_tables(panels, output)),
    }


def _run_arista_lr(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .arista_supplementary_figures import (
        calculate_arista_ligand_receptor_panels,
        load_arista_supplementary_figures,
        plot_arista_ligand_receptor_figures,
        write_arista_ligand_receptor_tables,
    )

    data = load_arista_supplementary_figures(results_dir)
    panels = calculate_arista_ligand_receptor_panels(data)
    selected_k = int(
        panels.k_selection.sort_values(
            ["silhouette", "k"], ascending=[False, True]
        ).iloc[0]["k"]
    )
    return {
        "input_directory": str(data.source_dir),
        "figures": _paths(plot_arista_ligand_receptor_figures(data, output, panels)),
        "tables": _paths(write_arista_ligand_receptor_tables(data, output, panels)),
        "input_profiles": int(len(panels.assignments)),
        "selected_k": selected_k,
        "cluster_counts": _paths(
            panels.assignments.groupby("cluster").size().astype(int).to_dict()
        ),
        "displayed_per_cluster": _paths(
            panels.display_roster.groupby("cluster").size().astype(int).to_dict()
        ),
        "displayed_pairs": int(len(panels.display_roster)),
        "displayed_timecourse_rows": int(len(panels.display_timecourse)),
    }


def _run_zebrafish_si(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .zebrafish_si import (
        calculate_zebrafish_si_panels,
        load_zebrafish_si_results,
        plot_zebrafish_si,
        write_zebrafish_si_tables,
    )

    data = load_zebrafish_si_results(results_dir)
    panels = calculate_zebrafish_si_panels(data)
    return {
        "input_directory": str(data.source_dir),
        "figures": _paths(plot_zebrafish_si(data, output, panels)),
        "tables": _paths(write_zebrafish_si_tables(panels, output)),
    }


def _run_arista_local_domains(
    results_dir: Path | None, output: Path
) -> dict[str, object]:
    from .arista_local_domains import (
        calculate_arista_local_domain_panels,
        load_arista_local_domains,
        plot_arista_local_domains,
        write_arista_local_domain_tables,
    )

    data = load_arista_local_domains(results_dir)
    panels = calculate_arista_local_domain_panels(data)
    return {
        "input_directory": str(data.source_dir),
        "figures": _paths(plot_arista_local_domains(data, output, panels)),
        "tables": _paths(write_arista_local_domain_tables(panels, output)),
    }


def _run_compute_cost(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .compute_cost import (
        load_full_model_compute_cost,
        write_full_model_compute_cost_tables,
    )

    data = load_full_model_compute_cost(results_dir)
    return {
        "input_directory": str(data.source_dir),
        "tables": _paths(write_full_model_compute_cost_tables(data, output)),
    }


def _run_main_figure_2(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .main_figure_2 import (
        assemble_main_figure_2,
        load_main_figure_2,
        write_main_figure_2_tables,
    )

    data = load_main_figure_2(results_dir)
    return {
        "input_directory": str(data.source_dir),
        "figures": _paths(assemble_main_figure_2(data, output)),
        "tables": _paths(write_main_figure_2_tables(data, output)),
    }


def _run_main_figure_5_reference(
    results_dir: Path | None, output: Path
) -> dict[str, object]:
    from .main_figure_5 import (
        export_main_figure_5_reference_page,
        load_main_figure_5,
        validate_main_figure_5_reference_page,
        write_main_figure_5_tables,
    )

    data = load_main_figure_5(results_dir)
    page = validate_main_figure_5_reference_page(data)
    return {
        "input_directory": str(data.source_dir),
        "figures": _paths(export_main_figure_5_reference_page(data, output, page)),
        "tables": _paths(write_main_figure_5_tables(data, page, output)),
    }


def _run_main_figure_4(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .mosta_figures import (
        assemble_main_figure_4,
        load_mosta_figure_release,
        write_mosta_figure_index,
    )

    release = load_mosta_figure_release(results_dir)
    return {
        "input_directory": str(release.root),
        "figures": _paths(assemble_main_figure_4(release, output)),
        "figure_index": str(write_mosta_figure_index(release, output)),
    }


def _run_mosta_reference_pages(
    results_dir: Path | None, output: Path
) -> dict[str, object]:
    from .mosta_figures import (
        export_mosta_supplementary_figures,
        load_mosta_figure_release,
        write_mosta_figure_index,
    )

    release = load_mosta_figure_release(results_dir)
    return {
        "input_directory": str(release.root),
        "figures": _paths(export_mosta_supplementary_figures(release, output)),
        "figure_index": str(write_mosta_figure_index(release, output)),
    }


def _run_classifier(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .classifier_smoothing import (
        load_classifier_smoothing_results,
        plot_classifier_smoothing,
        write_classifier_smoothing_tables,
    )

    return _standard_run(
        results_dir,
        output,
        loader=load_classifier_smoothing_results,
        table_writer=write_classifier_smoothing_tables,
        plotter=plot_classifier_smoothing,
    )


def _run_lr_complex(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .lr_complex_aggregation import (
        load_lr_complex_aggregation_results,
        plot_lr_complex_aggregation,
        write_lr_complex_aggregation_tables,
    )

    return _standard_run(
        results_dir,
        output,
        loader=load_lr_complex_aggregation_results,
        table_writer=write_lr_complex_aggregation_tables,
        plotter=plot_lr_complex_aggregation,
    )


def _run_interaction(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .interaction_evidence import (
        load_interaction_evidence_results,
        plot_interaction_evidence,
        write_interaction_evidence_tables,
    )

    return _standard_run(
        results_dir,
        output,
        loader=load_interaction_evidence_results,
        table_writer=write_interaction_evidence_tables,
        plotter=plot_interaction_evidence,
    )


def _run_loto(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .loto_benchmark import (
        load_loto_benchmark,
        plot_loto_benchmark,
        write_loto_benchmark_tables,
    )

    return _standard_run(
        results_dir,
        output,
        loader=load_loto_benchmark,
        table_writer=write_loto_benchmark_tables,
        plotter=plot_loto_benchmark,
    )


def _run_training(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .training_histories import (
        load_training_history_results,
        plot_training_histories,
        write_training_history_tables,
    )

    return _standard_run(
        results_dir,
        output,
        loader=load_training_history_results,
        table_writer=write_training_history_tables,
        plotter=plot_training_histories,
    )


def _run_zebrafish_attention(
    results_dir: Path | None, output: Path
) -> dict[str, object]:
    from .zebrafish_attention import (
        load_zebrafish_attention_results,
        plot_zebrafish_attention,
        write_zebrafish_attention_tables,
    )

    return _standard_run(
        results_dir,
        output,
        loader=load_zebrafish_attention_results,
        table_writer=write_zebrafish_attention_tables,
        plotter=plot_zebrafish_attention,
    )


_RUNNERS = {
    "agist": _run_agist,
    "arista-local-domains": _run_arista_local_domains,
    "arista-lr": _run_arista_lr,
    "classifier-smoothing": _run_classifier,
    "compute-cost": _run_compute_cost,
    "interaction-evidence": _run_interaction,
    "loto-benchmark": _run_loto,
    "lr-complex": _run_lr_complex,
    "main-figure-2": _run_main_figure_2,
    "main-figure-4": _run_main_figure_4,
    "main-figure-5-reference": _run_main_figure_5_reference,
    "mosta-reference-pages": _run_mosta_reference_pages,
    "nonspatial": _run_nonspatial,
    "training-histories": _run_training,
    "zebrafish-attention": _run_zebrafish_attention,
    "zebrafish-si": _run_zebrafish_si,
}


def run_figure_workflow(
    name: str,
    output_dir: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> dict[str, object]:
    """Run one figure command and write ``run_summary.json``."""

    if name not in _WORKFLOW_BY_NAME:
        raise ValueError(f"Unknown figure workflow: {name}")
    output = new_output_dir(output_dir)
    source = None if results_dir is None else Path(results_dir).expanduser().resolve()
    details = _RUNNERS[name](source, output)
    workflow = _WORKFLOW_BY_NAME[name]
    summary = {
        "workflow": workflow.name,
        "paper_location": workflow.paper_location,
        "mode": workflow.mode,
        **details,
    }
    write_run_summary(output, summary)
    return summary


__all__ = [
    "FIGURE_WORKFLOWS",
    "FigureWorkflow",
    "list_figure_workflows",
    "run_figure_workflow",
]
