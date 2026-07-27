#!/usr/bin/env python3
"""Fail-closed audit of full/no-interaction/no-LR-prior training configs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"commit": None, "dirty": None, "error": str(error)}


def _load_yaml(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config must contain a mapping: {resolved}")
    return dict(payload)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], path))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            result.update(_flatten(item, path))
        if not value:
            result[prefix] = []
        return result
    return {prefix: value}


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _difference_rows(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    comparison: str,
    allowed_exact: set[str],
    allowed_prefixes: Sequence[str],
) -> pd.DataFrame:
    left = _flatten(reference)
    right = _flatten(candidate)
    rows = []
    for path in sorted(set(left) | set(right)):
        left_present = path in left
        right_present = path in right
        equal = (
            left_present
            and right_present
            and _json_value(left[path]) == _json_value(right[path])
        )
        if equal:
            continue
        allowed = path in allowed_exact or any(
            path == prefix or path.startswith(prefix + ".")
            for prefix in allowed_prefixes
        )
        rows.append(
            {
                "comparison": comparison,
                "path": path,
                "reference_present": left_present,
                "candidate_present": right_present,
                "reference_value": (
                    _json_value(left[path]) if left_present else "<MISSING>"
                ),
                "candidate_value": (
                    _json_value(right[path]) if right_present else "<MISSING>"
                ),
                "allowed_intended_difference": bool(allowed),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "comparison",
            "path",
            "reference_present",
            "candidate_present",
            "reference_value",
            "candidate_value",
            "allowed_intended_difference",
        ],
    )


def _training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    defaults = config.get("training", {}).get("defaults", {})
    plan = config.get("training", {}).get("plan", [])
    if not isinstance(plan, list) or not plan:
        raise ValueError("Every config must contain a non-empty training.plan.")
    return {
        "seed": int(config["seed"]),
        "reverse": bool(config["reverse"]),
        "alpha_spatial": float(defaults["alpha_spatial"]),
        "alpha_express": float(defaults["alpha_express"]),
        "n_stages": int(len(plan)),
        "total_epochs": int(sum(int(stage["epochs"]) for stage in plan)),
        "epochs_by_stage_index": [int(stage["epochs"]) for stage in plan],
        "modes_by_stage_index": [str(stage["mode"]) for stage in plan],
        "batch_sizes_by_stage_index": [
            int(stage.get("batch_size", defaults["batch_size"])) for stage in plan
        ],
    }


def audit_configs(
    full: Mapping[str, Any],
    no_interaction: Mapping[str, Any],
    no_lr_prior: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_allowed = {"ckpt_dir"}
    no_interaction_differences = _difference_rows(
        full,
        no_interaction,
        comparison="full_vs_no_interaction",
        allowed_exact={
            *common_allowed,
            "model.interaction_type",
            "model.interaction_group_size",
            "training.plan.2.name",
            "training.plan.2.train_strategy",
            "training.plan.4.name",
            "training.plan.4.train_strategy",
        },
        allowed_prefixes=(
            "model.components",
            "model.interaction_net",
            "training.plan.0.interaction_use",
            "training.plan.1.interaction_use",
            "training.plan.2.interaction_use",
            "training.plan.4.interaction_use",
        ),
    )
    no_lr_differences = _difference_rows(
        full,
        no_lr_prior,
        comparison="full_vs_no_lr_prior",
        allowed_exact={
            *common_allowed,
            "model.interaction_net.edge_prior_mode",
            "model.interaction_net.edge_predictor_path",
            "model.interaction_net.edge_predictor_thre",
        },
        allowed_prefixes=(),
    )
    differences = pd.concat(
        [no_interaction_differences, no_lr_differences], ignore_index=True
    )

    contracts = []
    for name, config in (
        ("full", full),
        ("no_interaction", no_interaction),
        ("no_lr_prior", no_lr_prior),
    ):
        contracts.append({"condition": name, **_training_contract(config)})
    contract_frame = pd.DataFrame(contracts)
    reference = contract_frame.iloc[0].drop(labels="condition")
    for row in contract_frame.iloc[1:].itertuples(index=False):
        candidate = pd.Series(row._asdict()).drop(labels="condition")
        if not reference.equals(candidate):
            raise ValueError(
                f"Training-budget contract differs for {row.condition}: "
                f"{candidate.to_dict()} versus {reference.to_dict()}."
            )
    unexpected = differences.loc[~differences["allowed_intended_difference"]]
    if not unexpected.empty:
        raise ValueError(
            "Unexpected config differences violate the matched design: "
            f"{unexpected[['comparison', 'path']].to_dict(orient='records')}."
        )
    return differences, contract_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-config", required=True, type=Path)
    parser.add_argument("--no-interaction-config", required=True, type=Path)
    parser.add_argument("--no-lr-prior-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    paths = {
        "full": args.full_config.expanduser().resolve(),
        "no_interaction": args.no_interaction_config.expanduser().resolve(),
        "no_lr_prior": args.no_lr_prior_config.expanduser().resolve(),
    }
    output = args.output_dir.expanduser().resolve()
    known = (
        "intended_config_differences.csv",
        "matched_training_contract.csv",
        "run_manifest.json",
    )
    existing = [output / name for name in known if (output / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    differences, contract = audit_configs(
        _load_yaml(paths["full"]),
        _load_yaml(paths["no_interaction"]),
        _load_yaml(paths["no_lr_prior"]),
    )
    differences_path = output / "intended_config_differences.csv"
    contract_path = output / "matched_training_contract.csv"
    differences.to_csv(differences_path, index=False)
    contract.to_csv(contract_path, index=False)
    manifest_path = output / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "analysis": "matched_training_config_audit",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(SCRIPT_PATH), *arguments],
        "git": _git_state(),
        "inputs": {name: _file_record(path) for name, path in paths.items()},
        "result": {
            "matched_training_budget": True,
            "unexpected_difference_count": 0,
            "intended_difference_count": int(len(differences)),
        },
        "outputs": {
            "differences": _file_record(differences_path),
            "training_contract": _file_record(contract_path),
        },
        "manifest": {"path": str(manifest_path), "self_hash_omitted": True},
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
