from __future__ import annotations

from pathlib import Path

import pandas as pd

from CytoBridge.cli import main as cli_main
from CytoBridge.results.figure_workflows import (
    FIGURE_WORKFLOWS,
    list_figure_workflows,
    run_figure_workflow,
)


def test_figure_workflow_registry_is_complete_and_explicit() -> None:
    names = [workflow.name for workflow in FIGURE_WORKFLOWS]
    assert len(names) == len(set(names))
    modes = {
        part.strip()
        for workflow in FIGURE_WORKFLOWS
        for part in workflow.mode.split("+")
    }
    assert {
        "numeric-redraw",
        "result-summary-redraw",
        "reference-export",
        "external-assembly",
        "table-only",
    }.issubset(modes)
    assert {row["name"] for row in list_figure_workflows()} == set(names)
    assert next(
        workflow for workflow in FIGURE_WORKFLOWS if workflow.name == "main-figure-5-reference"
    ).mode == "reference-export"


def test_compute_cost_installed_workflow(tmp_path: Path) -> None:
    output = tmp_path / "compute"
    summary = run_figure_workflow("compute-cost", output)
    assert summary["mode"] == "table-only"
    assert "figures" not in summary
    table = pd.read_csv(output / "full_model_compute_cost_table.csv")
    assert "Time points used for training" in table.columns
    assert (output / "run_summary.json").is_file()


def test_figure_workflow_rejects_unknown_name(tmp_path: Path) -> None:
    try:
        run_figure_workflow("not-a-figure", tmp_path / "out")
    except ValueError as error:
        assert "Unknown figure workflow" in str(error)
    else:
        raise AssertionError("Unknown workflow was accepted")


def test_installed_figure_cli_lists_execution_modes(capsys) -> None:
    assert cli_main(["figure", "list"]) == 0
    output = capsys.readouterr().out
    assert "arista-lr\tnumeric-redraw\tyes" in output
    assert "main-figure-5-reference\treference-export\tyes" in output
    assert "main-figure-4\texternal-assembly\texternal input" in output
