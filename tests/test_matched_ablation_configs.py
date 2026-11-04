from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "CytoBridge" / "configs"

PROFILES = {
    "zebrafish": {
        "full": "zebrafish_spatial_full_alpha_express_0015.yaml",
        "no_interaction": (
            "zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml"
        ),
        "no_lr_prior": ("zebrafish_spatial_full_alpha_express_0015_no_lr_prior.yaml"),
        "epochs": [100, 100, 50, 2001, 1000, 2001],
    },
    "mosta": {
        "full": "mosta_spatial_full_alpha_express_0015.yaml",
        "no_interaction": ("mosta_spatial_full_alpha_express_0015_no_interaction.yaml"),
        "no_lr_prior": ("mosta_spatial_full_alpha_express_0015_no_lr_prior.yaml"),
        "epochs": [100, 100, 50, 2001, 1000, 2001],
    },
    "arista": {
        "full": "arista_spatial_full.yaml",
        "no_interaction": "arista_spatial_full_no_interaction.yaml",
        "no_lr_prior": "arista_spatial_full_no_lr_prior.yaml",
        "epochs": [100, 100, 50, 2001, 1000, 2001],
    },
    "admouse": {
        "full": "admouse_spatial_full_alpha_express_0015.yaml",
        "no_interaction": (
            "admouse_spatial_full_alpha_express_0015_no_interaction.yaml"
        ),
        "no_lr_prior": ("admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml"),
        "epochs": [100, 100, 50, 3001, 1000, 3001],
    },
}

EXPECTED_ABLATIONS = {
    profile[condition]
    for profile in PROFILES.values()
    for condition in ("no_interaction", "no_lr_prior")
}


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _without(mapping: dict, *keys: str) -> dict:
    result = deepcopy(mapping)
    for key in keys:
        result.pop(key, None)
    return result


def _stage_contract(stage: dict) -> dict:
    return _without(stage, "name", "train_strategy", "interaction_use")


def _expected_matched_contract(dataset: str, arm: str) -> dict:
    return {
        "schema_version": 1,
        "family": f"{dataset}-full-no-lr-no-interaction-v1",
        "dataset": dataset,
        "arm": arm,
        "protocol": "isolated-interaction-crn-v1",
        "shared_seed": 42,
        "interaction_grouping_seed_offset": 10000,
        "input_contract": "exact-shared-aligned-h5ad",
        "implementation_contract": "exact-shared-training-code-sha256",
    }


def test_complete_spatial_ablation_matrix_is_packaged_in_wheel_and_sdist() -> None:
    tracked_ablations = {
        path.name
        for path in CONFIG_DIR.glob("*.yaml")
        if path.stem.endswith(("_no_interaction", "_no_lr_prior"))
        and "nonspatial" not in path.stem
    }
    assert tracked_ablations == EXPECTED_ABLATIONS

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for name in EXPECTED_ABLATIONS:
        assert f'"configs/{name}"' in pyproject

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include CytoBridge/configs *.yaml" in manifest


@pytest.mark.parametrize("dataset", PROFILES)
def test_every_matched_family_declares_the_same_fail_closed_protocol(
    dataset: str,
) -> None:
    profile = PROFILES[dataset]
    for arm in ("full", "no_lr_prior", "no_interaction"):
        config = _load(profile[arm])
        assert config["matched_ablation"] == _expected_matched_contract(dataset, arm)
        assert config["seed"] == config["matched_ablation"]["shared_seed"]
        assert (
            config["training"]["defaults"]["score_energy_objective"]
            == "velocity_score_cross_term"
        )


@pytest.mark.parametrize("dataset", PROFILES)
def test_no_lr_prior_changes_only_the_interaction_gate(dataset: str) -> None:
    profile = PROFILES[dataset]
    full = _load(profile["full"])
    no_lr = _load(profile["no_lr_prior"])

    assert _without(no_lr, "model", "ckpt_dir", "matched_ablation") == _without(
        full, "model", "ckpt_dir", "matched_ablation"
    )
    assert no_lr["training"] == full["training"]
    assert no_lr["seed"] == full["seed"] == 42
    assert no_lr["reverse"] == full["reverse"] is True
    assert no_lr["ckpt_dir"] == f'{full["ckpt_dir"]}_no_lr_prior'

    full_model = deepcopy(full["model"])
    no_lr_model = deepcopy(no_lr["model"])
    full_interaction = full_model.pop("interaction_net")
    no_lr_interaction = no_lr_model.pop("interaction_net")
    assert no_lr_model == full_model

    assert full_interaction.pop("edge_prior_mode") == "learned"
    assert no_lr_interaction.pop("edge_prior_mode") == "all_spatial"
    full_interaction.pop("edge_predictor_path")
    full_interaction.pop("edge_predictor_thre")
    assert "edge_predictor_path" not in no_lr_interaction
    assert "edge_predictor_thre" not in no_lr_interaction
    assert no_lr_interaction == full_interaction


@pytest.mark.parametrize("dataset", PROFILES)
def test_no_interaction_preserves_matched_model_and_six_stage_budget(
    dataset: str,
) -> None:
    profile = PROFILES[dataset]
    full = _load(profile["full"])
    no_interaction = _load(profile["no_interaction"])

    assert set(no_interaction) == set(full)
    assert _without(
        no_interaction, "model", "training", "ckpt_dir", "matched_ablation"
    ) == _without(full, "model", "training", "ckpt_dir", "matched_ablation")
    assert no_interaction["seed"] == full["seed"] == 42
    assert no_interaction["reverse"] == full["reverse"] is True
    assert no_interaction["ckpt_dir"] == f'{full["ckpt_dir"]}_no_interaction'
    assert no_interaction["model"]["components"] == [
        component
        for component in full["model"]["components"]
        if component != "interaction"
    ]
    assert set(no_interaction["model"]) == {
        "components",
        "velocity_net",
        "growth_net",
        "score_net",
    }
    for network in ("velocity_net", "growth_net", "score_net"):
        assert no_interaction["model"][network] == full["model"][network]

    assert no_interaction["training"]["defaults"] == full["training"]["defaults"]
    full_plan = full["training"]["plan"]
    ablated_plan = no_interaction["training"]["plan"]
    assert len(full_plan) == len(ablated_plan) == 6
    assert [stage["epochs"] for stage in ablated_plan] == profile["epochs"]
    assert [_stage_contract(stage) for stage in ablated_plan] == [
        _stage_contract(stage) for stage in full_plan
    ]

    for full_stage, ablated_stage in zip(full_plan, ablated_plan, strict=True):
        if full_stage["mode"] == "neural_ode":
            assert ablated_stage["interaction_use"] is False
        else:
            assert "interaction_use" not in ablated_stage
        assert ablated_stage["train_strategy"] == full_stage["train_strategy"].replace(
            "+i", ""
        )

    assert [stage["name"] for stage in ablated_plan] == [
        "Pretrain",
        "Refine",
        "Matched_stage_3_no_interaction",
        "Train_Score",
        "Finetune_no_interaction",
        "Score_Refine",
    ]
