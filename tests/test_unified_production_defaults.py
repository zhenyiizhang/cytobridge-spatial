from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

import CytoBridge.tl.downstream.classification as classification
from CytoBridge.tl.core.losses import calc_ot_loss
from CytoBridge.tl.downstream.analysis import (
    train_mlp_classifier as train_legacy_mlp_classifier,
)
from CytoBridge.tl.downstream.workflows import run_interpolation_workflow
from CytoBridge.tl.train.trainer import TrainingPipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "CytoBridge" / "configs"


def _default(callable_object, parameter: str):
    return inspect.signature(callable_object).parameters[parameter].default


def _load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))


def test_public_scientific_api_defaults_match_the_production_protocol() -> None:
    for trainer in (
        classification.train_mlp_classifier,
        classification.train_mlp_classifier_from_adata,
        classification.train_cached_mlp_classifier_from_adata,
    ):
        assert _default(trainer, "hidden_size") == 128
        assert _default(trainer, "epochs") == 500
        assert _default(trainer, "lr") == 1e-3
        assert _default(trainer, "seed") == 42

    assert (
        _default(
            classification.train_cached_mlp_classifier_from_adata,
            "best_epoch_metric",
        )
        == "bacc"
    )
    assert (
        _default(
            classification._train_mlp_classifier_arrays_detailed,
            "best_epoch_metric",
        )
        == "bacc"
    )
    assert _default(classification.predict_labels_for_points, "knn_neighbors") == 10
    assert (
        _default(classification.predict_labels_for_trajectories, "knn_neighbors")
        == 10
    )
    assert _default(classification.smooth_spatial_labels, "k") == 10

    assert _default(train_legacy_mlp_classifier, "hidden_size") == 128
    assert (
        _default(train_legacy_mlp_classifier, "train_mlp_classifier_epoches")
        == 500
    )
    assert _default(train_legacy_mlp_classifier, "lr") == 1e-3
    assert _default(train_legacy_mlp_classifier, "seed") == 42

    assert _default(run_interpolation_workflow, "classifier_hidden_size") == 128
    assert _default(run_interpolation_workflow, "classifier_epochs") == 500
    assert _default(run_interpolation_workflow, "classifier_lr") == 1e-3
    assert _default(run_interpolation_workflow, "classifier_best_metric") == "bacc"
    assert _default(run_interpolation_workflow, "classifier_knn_neighbors") == 10
    assert _default(run_interpolation_workflow, "random_seed") == 42

    assert _default(calc_ot_loss, "alpha_spatial") == 10.0
    assert _default(calc_ot_loss, "alpha_express") == 0.015


def test_non_cached_classifier_selects_by_balanced_accuracy(monkeypatch) -> None:
    captured = {}

    def fake_detailed(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(), object(), 0.75, 0.5, {"selection": {}}

    monkeypatch.setattr(
        classification,
        "_train_mlp_classifier_arrays_detailed",
        fake_detailed,
    )
    model, _, accuracy = classification._train_mlp_classifier_arrays(
        np.zeros((4, 2), dtype=np.float32),
        np.asarray(["A", "A", "B", "B"]),
        device="cpu",
    )

    assert captured["hidden_size"] == 128
    assert captured["epochs"] == 500
    assert captured["lr"] == 1e-3
    assert captured["seed"] == 42
    assert captured["best_epoch_metric"] == "bacc"
    assert model.classifier_evaluation_ == {"selection": {}}
    assert accuracy == 0.75


def test_training_pipeline_materializes_missing_seed_and_ot_weights(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(TrainingPipeline, "_setup_logger", lambda self: None)
    config = {
        "ckpt_dir": str(tmp_path),
        "model": {"components": []},
        "training": {"defaults": {}, "plan": []},
    }

    TrainingPipeline(
        torch.nn.Linear(1, 1),
        config,
        batch_size=1,
        device="cpu",
        data=[torch.zeros((1, 1))],
    )

    assert config["seed"] == 42
    assert config["training"]["defaults"]["alpha_spatial"] == 10.0
    assert config["training"]["defaults"]["alpha_express"] == 0.015


def test_released_spatial_configs_share_weights_but_keep_graph_thresholds() -> None:
    production_configs = (
        "admouse_spatial_full_alpha_express_0015.yaml",
        "arista_spatial_full.yaml",
        "arista_spatial_smoke.yaml",
        "mosta_spatial_full_alpha_express_0015.yaml",
        "simulation_config.yaml",
        "st_simulation_mosta.yaml",
        "st_spatial.yaml",
        "st_spatial_smoke.yaml",
        "st_spatial_smoke_mosta.yaml",
        "zebrafish_spatial_full.yaml",
        "zebrafish_spatial_full_alpha_express_0015.yaml",
        "zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml",
        "zebrafish_spatial_full_alpha_express_0015_no_lr_prior.yaml",
        "zebrafish_training.yaml",
    )
    for name in production_configs:
        config = _load_config(name)
        assert config["seed"] == 42, name
        defaults = config["training"]["defaults"]
        assert defaults["alpha_spatial"] == 10.0, name
        assert defaults["alpha_express"] == 0.015, name

    expected_graph_contracts = {
        "admouse_spatial_full_alpha_express_0015.yaml": (
            0.012106042891492197,
            0.32999998331069946,
        ),
        "arista_spatial_full.yaml": (
            0.03154105148551745,
            0.23999999463558197,
        ),
        "mosta_spatial_full_alpha_express_0015.yaml": (
            0.02400244047956264,
            0.44999998807907104,
        ),
        "zebrafish_spatial_full.yaml": (
            0.09606367405591873,
            0.4999999701976776,
        ),
    }
    for name, expected in expected_graph_contracts.items():
        interaction = _load_config(name)["model"]["interaction_net"]
        assert (interaction["cutoff"], interaction["edge_predictor_thre"]) == expected


def test_admouse_profile_preserves_the_resolved_six_stage_schedule() -> None:
    config = _load_config("admouse_spatial_full_alpha_express_0015.yaml")

    assert config["model"]["interaction_group_size"] == 1024
    assert config["model"]["score_net"] == {
        "hidden_dim": 256,
        "n_layers": 5,
        "activation": "leaky_relu",
    }
    assert [stage["name"] for stage in config["training"]["plan"]] == [
        "Pretrain",
        "Refine",
        "Init_interaction",
        "Train_Score",
        "Finetune",
        "Score_Refine",
    ]
    assert [stage["epochs"] for stage in config["training"]["plan"]] == [
        100,
        100,
        50,
        3001,
        1000,
        3001,
    ]
    assert config["training"]["plan"][3]["batch_size"] == 256
    assert config["training"]["plan"][5]["batch_size"] == 256


def test_shared_downstream_config_uses_the_classifier_production_defaults() -> None:
    classifier_config = _load_config("arista_downstream.yaml")["classifier"]
    assert classifier_config["epochs"] == 500
    assert classifier_config["hidden_size"] == 128
    assert classifier_config["lr"] == 1e-3
    assert classifier_config["best_metric"] == "bacc"
    assert classifier_config["knn_neighbors"] == 10
