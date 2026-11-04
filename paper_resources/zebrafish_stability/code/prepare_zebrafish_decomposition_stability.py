#!/usr/bin/env python3
"""Prepare the independent Zebrafish training runs used in the stability analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


REFERENCE_CUTOFF = 0.09606367405591873
EDGE_THRESHOLD = 0.6063615679740906
DEFAULT_ALPHA_EXPRESS = 0.015
DEFAULT_LAMBDA_OT = 10.0
DEFAULT_LAMBDA_MASS = 10.0


def analysis_conditions() -> list[dict]:
    """Return every independently fitted condition used in the archived analysis."""
    conditions = [
        {
            "condition": f"formal_seed{seed}_cutoff1p0",
            "training_seed": seed,
            "cutoff_factor": 1.0,
            "alpha_express": DEFAULT_ALPHA_EXPRESS,
            "lambda_ot": DEFAULT_LAMBDA_OT,
            "lambda_mass": DEFAULT_LAMBDA_MASS,
            "analysis_group": "training_seed",
            "displayed_in_figure": True,
        }
        for seed in (42, 43, 44, 46, 47)
    ]
    conditions.extend(
        {
            "condition": f"formal_seed{seed}_cutoff{factor_text}",
            "training_seed": seed,
            "cutoff_factor": float(factor_text.replace("p", ".")),
            "alpha_express": DEFAULT_ALPHA_EXPRESS,
            "lambda_ot": DEFAULT_LAMBDA_OT,
            "lambda_mass": DEFAULT_LAMBDA_MASS,
            "analysis_group": "interaction_cutoff",
            "displayed_in_figure": True,
        }
        for factor_text in ("0p8", "1p2")
        for seed in (42, 43, 44)
    )
    conditions.extend(
        [
            {
                "condition": "alpha_expr_005_seed42_cutoff1p0",
                "training_seed": 42,
                "cutoff_factor": 1.0,
                "alpha_express": 0.05,
                "lambda_ot": DEFAULT_LAMBDA_OT,
                "lambda_mass": DEFAULT_LAMBDA_MASS,
                "analysis_group": "expression_weight",
                "displayed_in_figure": True,
            },
            {
                "condition": "ot_mass_10_to_1_seed42_cutoff1p0",
                "training_seed": 42,
                "cutoff_factor": 1.0,
                "alpha_express": DEFAULT_ALPHA_EXPRESS,
                "lambda_ot": 18.181818181818183,
                "lambda_mass": 1.8181818181818181,
                "analysis_group": "transport_mass_weight",
                "displayed_in_figure": True,
            },
            {
                "condition": "ot_mass_1_to_10_seed42_cutoff1p0",
                "training_seed": 42,
                "cutoff_factor": 1.0,
                "alpha_express": DEFAULT_ALPHA_EXPRESS,
                "lambda_ot": 1.8181818181818181,
                "lambda_mass": 18.181818181818183,
                "analysis_group": "transport_mass_weight",
                "displayed_in_figure": False,
            },
        ]
    )
    return conditions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--remote-run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    template_path = args.template.expanduser().resolve(strict=True)
    config_dir = args.config_dir.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    remote_root = args.remote_run_root
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    jobs = []
    for specification in analysis_conditions():
        condition = specification["condition"]
        seed = int(specification["training_seed"])
        cutoff_factor = float(specification["cutoff_factor"])
        cutoff = REFERENCE_CUTOFF * cutoff_factor
        config = json.loads(json.dumps(template))
        config["seed"] = seed
        config["model"]["interaction_net"]["cutoff"] = cutoff
        defaults = config["training"]["defaults"]
        defaults["alpha_express"] = float(specification["alpha_express"])
        defaults["lambda_ot"] = float(specification["lambda_ot"])
        defaults["lambda_mass"] = float(specification["lambda_mass"])
        for stage in config["training"]["plan"]:
            if stage.get("name") in {"Init_interaction", "Finetune"}:
                stage["lambda_ot"] = float(specification["lambda_ot"])
                stage["lambda_mass"] = float(specification["lambda_mass"])
        training_dir = remote_root / condition / "training"
        config["ckpt_dir"] = str(training_dir)
        output_path = config_dir / f"{condition}.yaml"
        output_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        jobs.append(
            {
                "condition": condition,
                "training_seed": seed,
                "cutoff_factor": cutoff_factor,
                "interaction_cutoff": cutoff,
                "expected_alpha_express": float(specification["alpha_express"]),
                "analysis_group": specification["analysis_group"],
                "displayed_in_figure": bool(specification["displayed_in_figure"]),
                "config": str(remote_root / "configs" / output_path.name),
                "training_dir": str(training_dir),
                "config_sha256": sha256(output_path),
            }
        )
    plan = {
        "schema_version": 1,
        "status": "prepared",
        "purpose": (
            "Independent training-seed and spatial-neighborhood stability analysis "
            "for the Zebrafish interaction/intrinsic decomposition."
        ),
        "reference_condition": "formal_seed42_cutoff1p0",
        "reference_training_dir": str(
            remote_root / "formal_seed42_cutoff1p0" / "training"
        ),
        "reference_cutoff": REFERENCE_CUTOFF,
        "edge_predictor_threshold": EDGE_THRESHOLD,
        "training_seed_definition": (
            "The seed passed in the training configuration. These are not simulation "
            "or rollout seeds."
        ),
        "jobs": jobs,
    }
    plan_path = config_dir.parent / "experiment_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(plan_path)
    print(f"prepared {len(jobs)} independent fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
