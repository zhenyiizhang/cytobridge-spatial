from __future__ import annotations

import json

import pytest
import CytoBridge as cb

from CytoBridge.cli import main
from CytoBridge.workflow import (
    WorkflowOptions,
    build_workflow_plan,
    load_workflow_config,
    _run_train,
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
        "alpha_spatial": 10.0,
        "alpha_express": 0.015,
        "classifier_k": classifier_k,
    }


@pytest.mark.parametrize(
    ("name", "cutoff", "edge_threshold"),
    (
        ("mosta", 0.02400244047956264, 0.44999998807907104),
        ("arista", 0.03154105148551745, 0.23999999463558197),
        ("zebrafish", 0.09606367405591873, 0.4999999701976776),
        ("admouse", 0.012106042891492197, 0.32999998331069946),
    ),
)
def test_packaged_presets_plan_the_formal_graph_contract(
    name,
    cutoff,
    edge_threshold,
):
    config, source = load_workflow_config(name)
    plan = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(train=True),
    )
    training = next(step for step in plan["steps"] if step["name"] == "train")

    assert training["training_config"] == config["train"]["config"]
    assert training["interaction_cutoff"] == cutoff
    assert training["edge_predictor_threshold"] == edge_threshold


def test_admouse_preset_reuses_formal_aligned_input_and_current_training(tmp_path):
    config, source = load_workflow_config("admouse")
    plan = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(
            train=True,
            aligned_h5ad=tmp_path / "admouse_aligned.h5ad",
            edge_predictor_path=tmp_path / "admouse_edge_model.pt",
            output_dir=tmp_path / "run",
        ),
    )
    preprocessing = next(step for step in plan["steps"] if step["name"] == "preprocess")
    training = next(step for step in plan["steps"] if step["name"] == "train")
    downstream = next(step for step in plan["steps"] if step["name"] == "downstream")

    assert preprocessing["status"] == "skipped"
    assert training["training_config"] == "admouse_spatial_full_alpha_express_0015.yaml"
    assert training["status"] == "ready"
    assert downstream["model_format"] == "current"

    preprocess_only = build_workflow_plan(
        config,
        source=source,
        options=WorkflowOptions(steps=("preprocess",)),
    )
    explicit_preprocess = next(
        step for step in preprocess_only["steps"] if step["name"] == "preprocess"
    )
    assert explicit_preprocess["status"] == "unavailable in this preset"
    assert "released aligned H5AD" in explicit_preprocess["note"]


def test_training_passes_formal_graph_values_as_explicit_fit_arguments(
    monkeypatch,
    tmp_path,
):
    config, _ = load_workflow_config("admouse")
    edge_model = tmp_path / "admouse_edge_model.pt"
    captured = {}

    def fake_fit(aligned_h5ad, **kwargs):
        captured["aligned_h5ad"] = aligned_h5ad
        captured.update(kwargs)

    monkeypatch.setattr(cb.tl, "fit", fake_fit)
    _run_train(
        config,
        WorkflowOptions(edge_predictor_path=edge_model, device="cpu"),
        aligned_h5ad=tmp_path / "admouse_aligned.h5ad",
        model_dir=tmp_path / "training",
    )

    assert captured["interaction_cutoff"] == 0.012106042891492197
    assert captured["edge_predictor_threshold"] == 0.32999998331069946
    assert captured["edge_predictor_path"] == str(edge_model.resolve())
    resolved = captured["config"]
    assert resolved["training"]["defaults"]["alpha_spatial"] == 10.0
    assert resolved["training"]["defaults"]["alpha_express"] == 0.015
    assert resolved["model"]["interaction_net"]["cutoff"] == 0.012106042891492197
    assert captured["time_key"] == "time_point_processed"
    assert captured["obsm_key"] == "X_latent"
    assert captured["spatial_key"] == "spatial_aligned"
    assert captured["is_spatial"] is True


def test_training_uses_custom_dataset_schema(monkeypatch, tmp_path):
    config, _ = load_workflow_config("zebrafish")
    config["dataset"] = {
        **config["dataset"],
        "time_key": "stage",
        "obsm_key": "latent_custom",
        "spatial_key": "xy_custom",
        "concat_spatial": False,
    }
    captured = {}

    def fake_fit(_aligned_h5ad, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cb.tl, "fit", fake_fit)
    _run_train(
        config,
        WorkflowOptions(edge_predictor_path=tmp_path / "edge.pt", device="cpu"),
        aligned_h5ad=tmp_path / "aligned.h5ad",
        model_dir=tmp_path / "training",
    )

    assert captured["time_key"] == "stage"
    assert captured["obsm_key"] == "latent_custom"
    assert captured["spatial_key"] == "xy_custom"
    assert captured["is_spatial"] is False


def test_formal_downstream_simulation_profiles_are_packaged():
    expected = {
        "zebrafish": (None, 9, 0.05),
        "mosta": (12000, 13, 0.05),
        "arista": (7668, 9, 0.01),
        "admouse": (None, 26, 0.01),
    }
    for dataset, (particles, n_times, split_dt) in expected.items():
        config, _ = load_workflow_config(dataset)
        downstream = config["downstream"]
        time_grid = set(downstream["observed"]) | set(downstream["interpolated"])
        assert downstream["sde_n_samples"] == particles
        assert len(time_grid) == n_times
        assert downstream["split_sde_dt"] == split_dt


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
    enabled_training = next(
        step for step in enabled["steps"] if step["name"] == "train"
    )
    assert base_training["status"] == "skipped; add --train to run"
    assert enabled_training["status"] == "missing input"
    assert "--edge-predictor-path" in enabled_training["missing"]


def test_dry_run_is_read_only_and_reports_scientific_parameters(capsys):
    assert main(["workflow", "--config", "admouse", "--dry-run"]) == 0
    output = capsys.readouterr().out

    assert "alpha_express=0.015" in output
    assert "alpha_spatial=10" in output
    assert "seed=42" in output
    assert "classifier_k=1" in output
    assert "train: skipped; add --train to run" in output
    assert "dry-run: no work executed" in output


def test_dry_run_json_is_machine_readable(capsys):
    assert main(["workflow", "--config", "mosta", "--dry-run", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)

    assert plan["dataset"]["name"] == "mosta"
    assert plan["scientific"]["classifier_k"] == 10
    assert plan["scientific"]["alpha_express"] == 0.015
    assert plan["scientific"]["alpha_spatial"] == 10.0
