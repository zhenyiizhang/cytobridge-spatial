#!/usr/bin/env python3
"""Write the four training configurations used for zebrafish loss sensitivity."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml


REFERENCE_PAIR = (10.0, 10.0)
PAIR_SUM = sum(REFERENCE_PAIR)
SETTINGS = {
    "reference": (0.015, REFERENCE_PAIR),
    "alpha_expr_005": (0.05, REFERENCE_PAIR),
    "ot_mass_10_to_1": (0.015, (PAIR_SUM * 10.0 / 11.0, PAIR_SUM / 11.0)),
    "ot_mass_1_to_10": (0.015, (PAIR_SUM / 11.0, PAIR_SUM * 10.0 / 11.0)),
}
RATIO_STAGES = {"Init_interaction", "Finetune"}
BACKGROUND_STAGES = {"Pretrain", "Refine"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def make_config(base: dict, name: str, output_dir: Path) -> dict:
    alpha_expr, (lambda_ot, lambda_mass) = SETTINGS[name]
    config = deepcopy(base)
    config.pop("matched_ablation", None)
    config["ckpt_dir"] = str((output_dir / name / "training").resolve())
    defaults = config["training"]["defaults"]
    defaults["alpha_express"] = float(alpha_expr)
    defaults["lambda_ot"] = float(lambda_ot)
    defaults["lambda_mass"] = float(lambda_mass)

    ratio_stages = set()
    background_stages = set()
    for stage in config["training"]["plan"]:
        stage_name = str(stage["name"])
        if stage_name in RATIO_STAGES:
            stage["lambda_ot"] = float(lambda_ot)
            stage["lambda_mass"] = float(lambda_mass)
            ratio_stages.add(stage_name)
        elif stage_name in BACKGROUND_STAGES:
            if float(stage["lambda_ot"]) != 1.0 or float(stage["lambda_mass"]) != 0.01:
                raise ValueError(f"Unexpected background coefficients in {stage_name}")
            background_stages.add(stage_name)
    if ratio_stages != RATIO_STAGES or background_stages != BACKGROUND_STAGES:
        raise ValueError("The base configuration does not contain the expected training stages")
    return config


def main() -> int:
    args = parse_args()
    base_path = args.base_config.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("The base training configuration must be a mapping")

    rows = []
    for name, (alpha_expr, (lambda_ot, lambda_mass)) in SETTINGS.items():
        path = output_dir / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(make_config(base, name, output_dir), sort_keys=False),
            encoding="utf-8",
        )
        rows.append(
            {
                "name": name,
                "alpha_express": float(alpha_expr),
                "lambda_ot": float(lambda_ot),
                "lambda_mass": float(lambda_mass),
                "config": str(path),
            }
        )
    summary = output_dir / "settings.json"
    summary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
