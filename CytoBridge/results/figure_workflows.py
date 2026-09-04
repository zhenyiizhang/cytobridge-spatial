"""Commands used by the paper-figure notebooks."""

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
    starts_from: str
    upstream_entry: str
    upstream_command: str | None
    figure_command: str
    scope: str


FIGURE_WORKFLOWS = (
    FigureWorkflow(
        "agist",
        "Supplementary Figures S2-S3",
        "numeric-redraw",
        True,
        "Recalculate AGIST panel values and draw both figures.",
        "Included cell-level summaries, processed simulation arrays, and tables.",
        "package resource: agist_figures/full_recompute_inputs.csv",
        None,
        "cytobridge figure agist --output-dir outputs/agist",
        "Redraws S2-S3. The preceding steps show how to generate the simulation, fit the model, and evaluate the rollout.",
    ),
    FigureWorkflow(
        "nonspatial",
        "Supplementary Figures S4-S5",
        "numeric-redraw",
        True,
        "Recalculate the grouped Weinreb and scNT panels and draw both figures.",
        "Included Weinreb and scNT panel arrays and tables.",
        "docs/nonspatial_workflows.md",
        "cytobridge nonspatial plan --dataset weinreb",
        "cytobridge figure nonspatial --output-dir outputs/nonspatial",
        "Redraws S4-S5. Use the preceding commands to create new model results and panel tables.",
    ),
    FigureWorkflow(
        "classifier-smoothing",
        "Supplementary Figure S6",
        "numeric-redraw",
        True,
        "Summarize the included sensitivity results and draw the figure.",
        "Included classifier metrics and generated-frame sensitivity tables.",
        "package resource: classifier_smoothing/manifest.json",
        None,
        "cytobridge figure classifier-smoothing --output-dir outputs/classifier_smoothing",
        "Redraws S6. Classifier fits and generated-frame labels are not rerun by this command.",
    ),
    FigureWorkflow(
        "arista-lr",
        "Supplementary Figures S23-S24",
        "numeric-redraw",
        True,
        "Recluster all 531 ARISTA LR profiles and draw the corrected figures.",
        "Included all-pair ARISTA ligand-receptor time courses.",
        "docs/tutorials/dataset_workflows/arista.ipynb",
        "cytobridge workflow --config arista --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training-dir> --output-dir <downstream-dir>",
        "cytobridge figure arista-lr --output-dir outputs/arista_lr",
        "Recalculates clustering and redraws S23-S24; it does not rerun ARISTA training.",
    ),
    FigureWorkflow(
        "lr-complex",
        "Supplementary Figure S41",
        "numeric-redraw",
        True,
        "Recalculate LR-complex sensitivity summaries and draw the figure.",
        "Included paired minimum-subunit and geometric-mean LR scores.",
        "scripts/collect_figure_inputs.py",
        "python scripts/collect_figure_inputs.py s41 --dataset-result zebrafish=<zebrafish-sensitivity> --dataset-result mosta=<mosta-sensitivity> --dataset-result arista=<arista-sensitivity> --dataset-result chicken_heart=<chicken-heart-sensitivity> --output-dir <s41-inputs>",
        "cytobridge figure lr-complex --results-dir <s41-inputs> --output-dir outputs/lr_complex",
        "Redraws S41. The collector accepts the paired score table written by each completed sensitivity run.",
    ),
    FigureWorkflow(
        "zebrafish-si",
        "Supplementary Figures S31-S38",
        "numeric-redraw",
        True,
        "Recalculate the included zebrafish panel values and draw eight figures.",
        "Included zebrafish panel arrays and result tables.",
        "scripts/run_zebrafish_paper_downstream.py",
        "python -m scripts.run_zebrafish_paper_downstream --aligned-h5ad <run>/preprocess/zebrafish_aligned.h5ad --model-dir <run>/training --acceptance-report <run>/matched_ablation_acceptance.json --lr-database <zebrafish-lr.csv> --output-dir <zebrafish-paper-output> --stage all --device cuda",
        "cytobridge figure zebrafish-si --output-dir outputs/zebrafish_si",
        "Redraws S31-S38 from the included result tables. The preceding command creates the full zebrafish analysis used to prepare those tables.",
    ),
    FigureWorkflow(
        "interaction-ablation",
        "Supplementary Figure S42",
        "numeric-redraw",
        True,
        "Draw the LR-prior and inference-time interaction ablations.",
        "Matched Full/No-LR errors and projection-level interaction-on/off errors.",
        "scripts/paper_figures/interaction_ablation/run_comparison.py",
        "python scripts/paper_figures/interaction_ablation/run_comparison.py --dataset arista --model-dir <run>/training --input-manifest <benchmark-run>/arista/inputs/manifest.json --code-root . --output <inference-results>/arista --seeds 42 43 44 --device cuda:0",
        "cytobridge figure interaction-ablation --output-dir outputs/interaction_ablation",
        "Recalculates S42 from included numerical results. Use the notebook commands to evaluate another set of fitted models.",
    ),
    FigureWorkflow(
        "lr-prior-stvcr",
        "Earlier LR-prior and stVCR comparison",
        "numeric-redraw",
        True,
        "Summarize the matched No-LR and stVCR results and draw the figure.",
        "Included paired target-level error tables.",
        "scripts/collect_figure_inputs.py",
        "python scripts/collect_figure_inputs.py lr-prior-stvcr --no-lr-table <matched-ablation-report>/paired_target_deltas.csv --loto-results-dir <s45-inputs> --output-dir <comparison-inputs>",
        "cytobridge figure lr-prior-stvcr --results-dir <s42-inputs> --output-dir outputs/lr_prior_stvcr",
        "Reproduces the earlier comparison. Current S42 uses the interaction-ablation command.",
    ),
    FigureWorkflow(
        "loto-benchmark",
        "Supplementary Figure S45",
        "numeric-redraw",
        True,
        "Recalculate matched benchmark ratios and draw the figure.",
        "Included target-level LOTO means and completion table for each method.",
        "scripts/collect_figure_inputs.py",
        "python scripts/collect_figure_inputs.py s45 --dataset-summary zebrafish=<zebrafish-target-summary.csv> --dataset-summary mosta=<mosta-target-summary.csv> --dataset-summary arista=<arista-target-summary.csv> --dataset-summary admouse=<admouse-target-summary.csv> --dataset-summary chicken_heart=<heart-target-summary.csv> --protocol <s45-protocol.json> --output-dir <s45-inputs>",
        "cytobridge figure loto-benchmark --results-dir <s45-inputs> --output-dir outputs/loto_benchmark",
        "Redraws S45 from five completed per-dataset LOTO target summaries.",
    ),
    FigureWorkflow(
        "training-histories",
        "Supplementary Figure S46",
        "numeric-redraw",
        True,
        "Smooth the included per-epoch losses and draw the training histories.",
        "Included per-epoch histories and checkpoint summaries.",
        "scripts/collect_training_history_inputs.py",
        "python scripts/collect_training_history_inputs.py --run zebrafish=<zebrafish-run>/training --run mosta=<mosta-run>/training --run arista=<arista-run>/training --run admouse=<admouse-run>/training --run chicken_heart=<heart-run>/training --output-dir <s46-inputs>",
        "cytobridge figure training-histories --output-dir outputs/training_histories",
        "Redraws S46 from the histories written by completed model runs.",
    ),
    FigureWorkflow(
        "arista-local-domains",
        "Supplementary Figure S25",
        "numeric-redraw",
        True,
        "Recalculate the displayed domain summaries and draw the figure.",
        "Included ARISTA ROI, domain, pathway, and null tables.",
        "docs/tutorials/dataset_workflows/arista.ipynb",
        "cytobridge workflow --config arista --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training-dir> --output-dir <downstream-dir>",
        "cytobridge figure arista-local-domains --output-dir outputs/arista_local_domains",
        "Redraws S25 after ROI selection and permutation-table calculation.",
    ),
    FigureWorkflow(
        "zebrafish-attention",
        "Supplementary Figure S39",
        "numeric-redraw",
        True,
        "Recalculate the included attention summaries and draw the figure.",
        "Included directed-pair, expression, and spatial-null tables.",
        "scripts/run_zebrafish_attention_analysis.py",
        "python -m scripts.run_zebrafish_attention_analysis analyze --spec <analysis-spec.json> --output-dir <attention-analysis> --n-selected-pairs 30",
        "cytobridge figure zebrafish-attention --output-dir outputs/zebrafish_attention",
        "Redraws S39 after calculating model and comparison-method outputs.",
    ),
    FigureWorkflow(
        "compute-cost",
        "Supplementary Table 2",
        "table-only",
        True,
        "Format the included runtime and memory measurements.",
        "Included one-run-per-dataset timing and memory table.",
        "docs/training_compute.md",
        None,
        "cytobridge figure compute-cost --output-dir outputs/compute_cost",
        "Formats Supplementary Table 2; it does not benchmark hardware during the command.",
    ),
    FigureWorkflow(
        "main-figure-2",
        "Main Figure 2",
        "result-summary-redraw + external-assembly",
        True,
        "Draw panel e and combine it with the existing panels a-d.",
        "Included panel-e replicate tables and the existing vector panels a-d.",
        "package resource: main_figure_2/manifest.json",
        None,
        "cytobridge figure main-figure-2 --output-dir outputs/main_figure_2",
        "Redraws panel e and assembles the page. Panels a-d are reused from the existing vector page.",
    ),
    FigureWorkflow(
        "main-figure-5-reference",
        "Main Figure 5",
        "reference-export",
        True,
        "Check the assembled Main Figure 5 page and write a viewable copy.",
        "Included Main Figure 5 page image and panel index.",
        "docs/tutorials/dataset_workflows/arista.ipynb",
        "cytobridge workflow --config arista --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training-dir> --output-dir <downstream-dir>",
        "cytobridge figure main-figure-5-reference --output-dir outputs/main_figure_5",
        "This command checks and copies the assembled page. Follow the preceding ARISTA steps for the panel calculations.",
    ),
    FigureWorkflow(
        "main-figure-4",
        "Main Figure 4",
        "external-assembly",
        False,
        "Assemble five vector panels from a separately downloaded MOSTA release.",
        "Five vector panel PDFs in the downloaded MOSTA figure files.",
        "docs/tutorials/dataset_workflows/mosta.ipynb",
        "cytobridge workflow --config mosta --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training-dir> --output-dir <downstream-dir>",
        "cytobridge figure main-figure-4 --results-dir <mosta-release> --output-dir outputs/main_figure_4",
        "Assembles Main Figure 4; it does not redraw the five source panels.",
    ),
    FigureWorkflow(
        "mosta-reference-pages",
        "Supplementary Figures S11-S18",
        "reference-export",
        False,
        "Write viewable copies of the completed MOSTA vector pages.",
        "Vector PDF and SVG pages in the downloaded MOSTA figure files.",
        "docs/tutorials/dataset_workflows/mosta.ipynb",
        "cytobridge workflow --config mosta --step downstream --aligned-h5ad <aligned.h5ad> --model-dir <training-dir> --output-dir <downstream-dir>",
        "cytobridge figure mosta-reference-pages --results-dir <mosta-release> --output-dir outputs/mosta_si",
        "This command copies the completed S11-S18 pages. Their calculation and rendering scripts are listed in the figure index.",
    ),
)

_WORKFLOW_BY_NAME = {workflow.name: workflow for workflow in FIGURE_WORKFLOWS}


def list_figure_workflows() -> list[dict[str, object]]:
    """Return the public figure commands and their execution modes."""

    return [asdict(workflow) for workflow in FIGURE_WORKFLOWS]


def describe_figure_workflow(name: str) -> dict[str, object]:
    """Return the complete input-to-figure route for one public command."""

    try:
        workflow = _WORKFLOW_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"Unknown figure workflow: {name}") from error
    return asdict(workflow)


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


def _run_interaction_ablation(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .interaction_ablation import (
        load_interaction_ablation_results, plot_interaction_ablation,
        write_interaction_ablation_tables,
    )
    return _standard_run(results_dir, output, loader=load_interaction_ablation_results,
                         table_writer=write_interaction_ablation_tables, plotter=plot_interaction_ablation)


def _run_lr_prior_stvcr(results_dir: Path | None, output: Path) -> dict[str, object]:
    from .interaction_evidence import (
        load_lr_prior_stvcr_results,
        plot_lr_prior_stvcr,
        write_lr_prior_stvcr_tables,
    )

    return _standard_run(
        results_dir,
        output,
        loader=load_lr_prior_stvcr_results,
        table_writer=write_lr_prior_stvcr_tables,
        plotter=plot_lr_prior_stvcr,
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
    "interaction-ablation": _run_interaction_ablation,
    "lr-prior-stvcr": _run_lr_prior_stvcr,
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
        **describe_figure_workflow(name),
        "workflow": workflow.name,
        **details,
    }
    write_run_summary(output, summary)
    return summary


__all__ = [
    "FIGURE_WORKFLOWS",
    "FigureWorkflow",
    "describe_figure_workflow",
    "list_figure_workflows",
    "run_figure_workflow",
]
