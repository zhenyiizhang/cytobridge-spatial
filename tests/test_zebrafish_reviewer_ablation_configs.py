from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


CONFIG_DIR = (
    Path(__file__).resolve().parents[1] / "CytoBridge" / "configs"
)


def _load(name: str) -> dict:
    return yaml.safe_load(
        (CONFIG_DIR / name).read_text(encoding="utf-8")
    )


def _plan_contract(config: dict) -> list[dict]:
    ignored = {"name", "train_strategy", "interaction_use"}
    return [
        {
            key: value
            for key, value in stage.items()
            if key not in ignored
        }
        for stage in config["training"]["plan"]
    ]


def test_no_lr_prior_changes_only_edge_gate_and_output_directory() -> None:
    full = _load("zebrafish_spatial_full_alpha_express_0015.yaml")
    no_lr = _load(
        "zebrafish_spatial_full_alpha_express_0015_no_lr_prior.yaml"
    )

    assert no_lr["training"] == full["training"]
    assert no_lr["seed"] == full["seed"] == 42
    assert no_lr["reverse"] == full["reverse"] is True

    full_model = deepcopy(full["model"])
    no_lr_model = deepcopy(no_lr["model"])
    full_interaction = full_model.pop("interaction_net")
    no_lr_interaction = no_lr_model.pop("interaction_net")
    assert no_lr_model == full_model

    assert no_lr_interaction.pop("edge_prior_mode") == "all_spatial"
    full_interaction.pop("edge_predictor_path")
    full_interaction.pop("edge_predictor_thre")
    assert no_lr_interaction == full_interaction
    assert no_lr["ckpt_dir"] != full["ckpt_dir"]


def test_no_interaction_preserves_retained_nets_and_training_budget() -> None:
    full = _load("zebrafish_spatial_full_alpha_express_0015.yaml")
    no_interaction = _load(
        "zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml"
    )

    assert no_interaction["model"]["components"] == [
        "velocity",
        "growth",
        "score",
    ]
    for component in ("velocity_net", "growth_net", "score_net"):
        assert no_interaction["model"][component] == full["model"][component]
    assert no_interaction["training"]["defaults"] == full["training"]["defaults"]
    assert _plan_contract(no_interaction) == _plan_contract(full)
    assert [
        stage["epochs"] for stage in no_interaction["training"]["plan"]
    ] == [100, 100, 50, 2001, 1000, 2001]
    assert no_interaction["seed"] == full["seed"] == 42
    assert no_interaction["reverse"] == full["reverse"] is True

    interaction_sensitive_stages = [
        no_interaction["training"]["plan"][2],
        no_interaction["training"]["plan"][4],
    ]
    assert all(
        "i" not in stage["train_strategy"]
        and stage["interaction_use"] is False
        for stage in interaction_sensitive_stages
    )
