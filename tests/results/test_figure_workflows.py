from __future__ import annotations

from pathlib import Path

import pandas as pd

from CytoBridge.cli import main as cli_main
from CytoBridge.results.figure_workflows import (
    FIGURE_WORKFLOWS,
    describe_figure_workflow,
    list_figure_workflows,
    run_figure_workflow,
)
from CytoBridge.results.reproduction_chains import (
    describe_dataset_paper_steps,
    describe_dataset_run_steps,
    describe_figure_steps,
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
    assert (
        next(
            workflow
            for workflow in FIGURE_WORKFLOWS
            if workflow.name == "main-figure-5-reference"
        ).mode
        == "reference-export"
    )
    for workflow in FIGURE_WORKFLOWS:
        assert workflow.starts_from
        assert workflow.upstream_entry
        assert workflow.figure_command.startswith("cytobridge figure ")
        assert workflow.scope


def test_figure_workflow_explanation_has_complete_route() -> None:
    route = describe_figure_workflow("zebrafish-si")
    assert route["starts_from"].startswith("Included zebrafish")
    assert route["upstream_entry"] == "scripts/run_zebrafish_paper_downstream.py"
    assert "--aligned-h5ad" in route["upstream_command"]
    assert route["figure_command"].startswith("cytobridge figure zebrafish-si")


def test_every_figure_workflow_has_ordered_calculation_steps() -> None:
    for workflow in FIGURE_WORKFLOWS:
        chain = describe_figure_steps(workflow.name)
        assert len(chain) >= 2
        for row in chain:
            assert set(row) == {
                "paper_part",
                "step",
                "code_or_command",
                "reads",
                "writes",
                "next_step",
                "note",
                "entry_type",
            }
            assert all(str(row[key]).strip() for key in row if key != "note")
            assert row["entry_type"] in {"command", "source"}
            assert "..." not in row["code_or_command"]
            assert "; " not in row["code_or_command"]


def test_dataset_run_steps_name_each_handoff() -> None:
    chain = describe_dataset_run_steps("chicken_heart")
    assert [row["step"] for row in chain] == [
        "Run a new dataset from raw counts",
        "Inspect the aligned data without training (optional)",
        "Run downstream analysis again (optional)",
    ]
    assert "chicken_heart_aligned.h5ad" in chain[0]["writes"]
    assert "training_run_summary.json" in chain[0]["writes"]
    assert "<run>/downstream/summary.json" in chain[0]["writes"]
    assert "no edge predictor or CytoBridge model" in chain[1]["writes"]
    assert "velocity_components.npz" in chain[2]["writes"]
    assert "<downstream-rerun>/downstream" in chain[2]["writes"]


def test_every_dataset_names_its_paper_continuation_and_known_breaks() -> None:
    for preset in ("zebrafish", "mosta", "arista", "admouse", "chicken_heart"):
        rows = describe_dataset_paper_steps(preset)
        assert rows
        assert all(row["paper_part"] and row["code_or_command"] for row in rows)
        assert all("..." not in row["code_or_command"] for row in rows)
    ad_rows = describe_dataset_paper_steps("admouse")
    heart_rows = describe_dataset_paper_steps("chicken_heart")
    assert any("not currently contain" in row["note"] for row in ad_rows)
    assert any("not included" in row["note"] for row in heart_rows)
    assert all("pca_artifacts.npz" not in row["code_or_command"] for row in ad_rows)
    assert all("alignment_manifest.json" not in row["code_or_command"] for row in heart_rows)


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


def test_installed_figure_cli_lists_paper_location_and_input(capsys) -> None:
    assert cli_main(["figure", "list"]) == 0
    output = capsys.readouterr().out
    assert "arista-lr\tSupplementary Figures S23-S24\tincluded paper results" in output
    assert "main-figure-5-reference\tMain Figure 5\tincluded paper results" in output
    assert "main-figure-4\tMain Figure 4\tseparate result directory" in output


def test_installed_figure_cli_explains_calculation_steps(capsys) -> None:
    assert cli_main(["figure", "explain", "nonspatial"]) == 0
    output = capsys.readouterr().out
    assert "Start with: Included Weinreb and scNT" in output
    assert "Figure command: cytobridge figure nonspatial" in output
    assert "Steps that produce the input:" in output
    assert "command:" in output
    assert "source:" not in output
    assert "start with:" in output
    assert "writes:" in output


def test_s39_uses_a_plain_language_command_name(capsys) -> None:
    assert cli_main(["figure", "explain", "lr-prior-stvcr"]) == 0
    output = capsys.readouterr().out
    assert "Figure command: cytobridge figure lr-prior-stvcr" in output
    assert "interaction-evidence" not in output


def test_old_s39_command_name_still_explains_the_canonical_command(capsys) -> None:
    assert cli_main(["figure", "explain", "interaction-evidence"]) == 0
    output = capsys.readouterr().out
    assert "Figure command: cytobridge figure lr-prior-stvcr" in output
