from __future__ import annotations

from CytoBridge.tl.downstream.checkpoint import (
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
        "training": {
            "plan": [{"name": "Finetune", "save_strategy": "best"}]
        }
    }
    last_config = {
        "training": {
            "plan": [{"name": "Finetune", "save_strategy": "last"}]
        }
    }

    assert _resolve_stage_checkpoint(tmp_path, "Finetune", best_config) == best
    assert _resolve_stage_checkpoint(tmp_path, "Finetune", last_config) == last
