from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "CytoBridge" / "configs"
CANONICAL_CONFIGS = (
    "zebrafish_spatial_full_alpha_express_0015.yaml",
    "zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml",
    "zebrafish_spatial_full_alpha_express_0015_no_lr_prior.yaml",
)
EXPECTED_EPOCHS = [100, 100, 50, 2001, 1000, 2001]
EXPECTED_MODES = [
    "neural_ode",
    "neural_ode",
    "neural_ode",
    "score_matching",
    "neural_ode",
    "score_matching",
]


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_canonical_zebrafish_training_contract_is_locked() -> None:
    for name in CANONICAL_CONFIGS:
        config = _load(name)
        defaults = config["training"]["defaults"]
        plan = config["training"]["plan"]
        epochs = [stage["epochs"] for stage in plan]

        assert config["seed"] == 42, name
        assert config["reverse"] is True, name
        assert defaults["alpha_spatial"] == 10.0, name
        assert defaults["alpha_express"] == 0.015, name
        assert defaults["sigma"] == 0.03, name
        assert epochs == EXPECTED_EPOCHS, name
        assert [stage["mode"] for stage in plan] == EXPECTED_MODES, name
        assert sum(epochs) == 5252, name


def test_shortened_zebrafish_config_is_explicitly_legacy_and_portable() -> None:
    config = _load("zebrafish_training.yaml")
    release = config["release"]
    interaction = config["model"]["interaction_net"]

    assert release["status"] == "legacy_verification_only"
    assert release["canonical_config"] == CANONICAL_CONFIGS[0]
    assert release["canonical_runner"] == "scripts/run_zebrafish_end_to_end.py"
    assert config["training"]["defaults"]["alpha_express"] == 0.05
    assert [stage["epochs"] for stage in config["training"]["plan"]] == [
        100,
        100,
        50,
        500,
        200,
        500,
    ]
    assert not Path(interaction["edge_predictor_path"]).is_absolute()
    assert not Path(config["ckpt_dir"]).is_absolute()


def test_alpha_005_sensitivity_differs_only_in_alpha_and_output_directory() -> None:
    canonical = _load("zebrafish_spatial_full_alpha_express_0015.yaml")
    comparator = deepcopy(_load("zebrafish_spatial_full.yaml"))

    assert comparator["training"]["defaults"]["alpha_express"] == 0.05
    assert comparator["ckpt_dir"] != canonical["ckpt_dir"]
    comparator["training"]["defaults"]["alpha_express"] = 0.015
    comparator["ckpt_dir"] = canonical["ckpt_dir"]
    assert comparator == canonical
