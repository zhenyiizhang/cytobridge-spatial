from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "reviewer_zebrafish_response"
    / "audit_matched_training_configs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reviewer_audit_matched_training_configs", SCRIPT
)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _full_config() -> dict:
    return {
        "model": {
            "components": ["velocity", "growth", "score", "interaction"],
            "interaction_type": "gnn",
            "interaction_group_size": 1024,
            "velocity_net": {"hidden_dim": 10},
            "interaction_net": {
                "cutoff": 0.1,
                "edge_predictor_path": "/tmp/edge.pt",
                "edge_predictor_thre": 0.5,
            },
        },
        "ckpt_dir": "/tmp/full",
        "reverse": True,
        "seed": 42,
        "training": {
            "defaults": {
                "batch_size": 32,
                "alpha_spatial": 10.0,
                "alpha_express": 0.015,
            },
            "plan": [
                {"name": "a", "mode": "neural_ode", "epochs": 1},
                {"name": "b", "mode": "neural_ode", "epochs": 2},
                {
                    "name": "Init_interaction",
                    "mode": "neural_ode",
                    "epochs": 3,
                    "train_strategy": "v+g+i",
                },
                {"name": "score", "mode": "score_matching", "epochs": 4},
                {
                    "name": "Finetune",
                    "mode": "neural_ode",
                    "epochs": 5,
                    "train_strategy": "v+g+i",
                },
                {"name": "score2", "mode": "score_matching", "epochs": 6},
            ],
        },
    }


def _matched_configs() -> tuple[dict, dict, dict]:
    full = _full_config()
    no_interaction = copy.deepcopy(full)
    no_interaction["ckpt_dir"] = "/tmp/no_interaction"
    no_interaction["model"]["components"] = ["velocity", "growth", "score"]
    del no_interaction["model"]["interaction_type"]
    del no_interaction["model"]["interaction_group_size"]
    del no_interaction["model"]["interaction_net"]
    for index in (0, 1, 2, 4):
        no_interaction["training"]["plan"][index]["interaction_use"] = False
    no_interaction["training"]["plan"][2]["name"] = "Matched_stage_3_no_interaction"
    no_interaction["training"]["plan"][2]["train_strategy"] = "v+g"
    no_interaction["training"]["plan"][4]["name"] = "Finetune_no_interaction"
    no_interaction["training"]["plan"][4]["train_strategy"] = "v+g"

    no_lr = copy.deepcopy(full)
    no_lr["ckpt_dir"] = "/tmp/no_lr"
    no_lr["model"]["interaction_net"]["edge_prior_mode"] = "all_spatial"
    del no_lr["model"]["interaction_net"]["edge_predictor_path"]
    del no_lr["model"]["interaction_net"]["edge_predictor_thre"]
    return full, no_interaction, no_lr


def test_audit_accepts_only_the_predeclared_component_differences() -> None:
    full, no_interaction, no_lr = _matched_configs()
    differences, contract = analysis.audit_configs(
        full, no_interaction, no_lr
    )
    assert not differences.empty
    assert differences["allowed_intended_difference"].all()
    assert contract["total_epochs"].nunique() == 1
    assert contract["total_epochs"].iloc[0] == 21


def test_audit_rejects_unexpected_loss_weight_change() -> None:
    full, no_interaction, no_lr = _matched_configs()
    no_lr["training"]["defaults"]["alpha_express"] = 0.05
    with pytest.raises(ValueError, match="Training-budget contract differs"):
        analysis.audit_configs(full, no_interaction, no_lr)


def test_cli_writes_hash_complete_audit(tmp_path: Path) -> None:
    full, no_interaction, no_lr = _matched_configs()
    paths = {}
    for name, config in (
        ("full", full),
        ("no_interaction", no_interaction),
        ("no_lr", no_lr),
    ):
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths[name] = path
    output = tmp_path / "audit"
    assert (
        analysis.main(
            [
                "--full-config",
                str(paths["full"]),
                "--no-interaction-config",
                str(paths["no_interaction"]),
                "--no-lr-prior-config",
                str(paths["no_lr"]),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    contract = pd.read_csv(output / "matched_training_contract.csv")
    assert contract["total_epochs"].nunique() == 1
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["result"]["matched_training_budget"] is True
    assert manifest["result"]["unexpected_difference_count"] == 0
