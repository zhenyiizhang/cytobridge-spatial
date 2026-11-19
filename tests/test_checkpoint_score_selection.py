from __future__ import annotations

import pytest
import torch
import yaml

import CytoBridge.tl.downstream.checkpoint as checkpoint_module
from CytoBridge.tl.downstream.checkpoint import (
    load_dynamical_model_from_dir,
    _resolve_stage_checkpoint,
    _score_stage_candidates,
)


def test_score_stage_candidates_prefer_last_configured_score_stage() -> None:
    config = {
        "training": {
            "plan": [
                {"name": "Pretrain", "mode": "neural_ode"},
                {"name": "Train_Score", "mode": "score_matching"},
                {"name": "Finetune", "mode": "neural_ode"},
                {"name": "Score_Refine", "mode": "score_matching"},
            ]
        }
    }

    candidates = _score_stage_candidates(config, None)

    assert candidates[:2] == ["Score_Refine", "Train_Score"]


def test_explicit_score_stage_preference_is_respected() -> None:
    config = {"training": {"plan": []}}

    candidates = _score_stage_candidates(
        config,
        ["custom_score", "Train_Score", "custom_score"],
    )

    assert candidates == ["custom_score", "Train_Score"]


def test_weight_checkpoint_respects_configured_save_strategy(tmp_path) -> None:
    stage_dir = tmp_path / "Finetune"
    stage_dir.mkdir()
    best = stage_dir / "best_model.pth"
    last = stage_dir / "last_model.pth"
    best.touch()
    last.touch()

    best_config = {
        "training": {"plan": [{"name": "Finetune", "save_strategy": "best"}]}
    }
    last_config = {
        "training": {"plan": [{"name": "Finetune", "save_strategy": "last"}]}
    }

    assert _resolve_stage_checkpoint(tmp_path, "Finetune", best_config) == best
    assert _resolve_stage_checkpoint(tmp_path, "Finetune", last_config) == last


def test_current_checkpoint_embeds_predictor_and_ignores_stale_external_path(
    tmp_path, monkeypatch
) -> None:
    stage = tmp_path / "Finetune"
    stage.mkdir()
    checkpoint_path = stage / "last_model.pth"
    checkpoint_path.touch()
    config = {
        "model": {
            "components": ["interaction"],
            "interaction_net": {
                "edge_prior_mode": "learned",
                "edge_predictor_path": "/old-machine/missing-edge-model.pt",
            },
        },
        "training": {"plan": [{"name": "Finetune", "save_strategy": "last"}]},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    state = {"interaction_net.link_predictor.layer.weight": torch.ones(1)}
    captured = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, dim, model_config):
            super().__init__()
            captured["dim"] = dim
            captured["config"] = model_config
            self.components = []

        def load_state_dict(self, loaded_state, strict=True):
            captured["state"] = loaded_state
            captured["strict"] = strict

    monkeypatch.setattr(checkpoint_module, "DynamicalModel", FakeModel)
    monkeypatch.setattr(
        checkpoint_module, "_load_state_dict", lambda *args, **kwargs: state
    )

    loaded = load_dynamical_model_from_dir(tmp_path, dim=4)

    assert loaded.weight_path == checkpoint_path
    assert (
        captured["config"]["interaction_net"]["load_edge_predictor_from_path"] is False
    )
    assert captured["state"] is state
    assert captured["strict"] is True


def test_current_checkpoint_accepts_predictor_path_override_when_not_embedded(
    tmp_path, monkeypatch
) -> None:
    stage = tmp_path / "Finetune"
    stage.mkdir()
    (stage / "last_model.pth").touch()
    config = {
        "model": {
            "components": ["velocity", "interaction"],
            "interaction_type": "gnn",
            "interaction_net": {"edge_prior_mode": "learned"},
        },
        "training": {"plan": [{"name": "Finetune", "save_strategy": "last"}]},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    captured = {}

    class FakeModel(torch.nn.Module):
        def __init__(self, _dim, model_config):
            super().__init__()
            captured["config"] = model_config
            self.components = ["velocity", "interaction"]
            self.interaction_net = torch.nn.Module()
            self.interaction_net.link_predictor = torch.nn.Linear(2, 1)

        def load_state_dict(self, state, strict=True):
            captured["state"] = state
            captured["strict"] = strict

    monkeypatch.setattr(checkpoint_module, "DynamicalModel", FakeModel)
    monkeypatch.setattr(
        checkpoint_module, "_load_state_dict", lambda *args, **kwargs: {}
    )
    replacement = tmp_path / "portable-edge-model.pt"

    load_dynamical_model_from_dir(
        tmp_path,
        dim=4,
        edge_predictor_path=replacement,
    )

    assert captured["config"]["interaction_net"]["edge_predictor_path"] == str(
        replacement.resolve()
    )
    assert captured["strict"] is True
    assert set(captured["state"]) == {
        "interaction_net.link_predictor.weight",
        "interaction_net.link_predictor.bias",
    }


def test_no_interaction_checkpoint_rejects_even_null_interaction_section(
    tmp_path, monkeypatch
) -> None:
    stage = tmp_path / "Finetune_no_interaction"
    stage.mkdir()
    (stage / "best_model.pth").touch()
    config = {
        "model": {
            "components": ["velocity", "growth", "score"],
            "interaction_net": None,
        },
        "training": {
            "plan": [
                {
                    "name": "Finetune_no_interaction",
                    "save_strategy": "best",
                }
            ]
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        checkpoint_module, "_load_state_dict", lambda *args, **kwargs: {}
    )

    with pytest.raises(ValueError, match="interaction_net is inert"):
        load_dynamical_model_from_dir(tmp_path, dim=4)
