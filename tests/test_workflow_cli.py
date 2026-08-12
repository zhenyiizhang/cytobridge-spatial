from __future__ import annotations

import json

import pytest

from CytoBridge.cli import main
from CytoBridge.workflow import (
    WorkflowOptions,
    build_workflow_plan,
    load_workflow_config,
)


@pytest.mark.parametrize(
    ("name", "classifier_k"),
    (("zebrafish", 10), ("mosta", 10), ("arista", 10), ("admouse", 1)),
)
def test_packaged_presets_share_formal_scientific_defaults(name, classifier_k):
    config, source = load_workflow_config(name)

    assert source == f"packaged preset: {name}"
    assert config["scientific"] == {
        "seed": 42,
        "alpha_express": 0.015,
        "classifier_k": classifier_k,
    }


def test_training_is_skipped_until_explicitly_enabled():
    config, source = load_workflow_config("zebrafish")
    base = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(),
    )
    enabled = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(train=True),
    )

    base_training = next(step for step in base["steps"] if step["name"] == "train")
    enabled_training = next(step for step in enabled["steps"] if step["name"] == "train")
    assert base_training["status"] == "skipped; add --train to run"
    assert enabled_training["status"] == "missing input"
    assert "--edge-predictor-path" in enabled_training["missing"]


def test_dry_run_is_read_only_and_reports_scientific_parameters(capsys):
    assert main(["workflow", "--config", "admouse", "--dry-run"]) == 0
    output = capsys.readouterr().out

    assert "alpha_express=0.015" in output
    assert "seed=42" in output
    assert "classifier_k=1" in output
    assert "train: skipped; add --train to run" in output
    assert "dry-run: no work executed" in output


def test_dry_run_json_is_machine_readable(capsys):
    assert main(
        ["workflow", "--config", "mosta", "--dry-run", "--json"]
    ) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["dataset"]["name"] == "mosta"
    assert plan["scientific"]["classifier_k"] == 10
    assert plan["scientific"]["alpha_express"] == 0.015

