#!/usr/bin/env python3
"""Run the two formal CellAgentChat LR-database conditions sequentially."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

try:
    from .common import (
        CONDITION_LABELS,
        artifact,
        csv_ints,
        json_value,
        prepare_output,
        utc_now,
        write_json,
    )
    from . import run_spatial
except ImportError:  # pragma: no cover - direct CLI execution
    from common import (  # type: ignore
        CONDITION_LABELS,
        artifact,
        csv_ints,
        json_value,
        prepare_output,
        utc_now,
        write_json,
    )
    import run_spatial  # type: ignore


def _csv_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError("Expected unique comma-separated numbers.")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-dir", required=True, type=Path)
    parser.add_argument("--cellagentchat-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sampling-seeds", type=csv_ints)
    parser.add_argument("--stages", type=_csv_floats)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-shuffles", type=int, default=1)
    parser.add_argument("--permutation-score-target", type=int, default=10_000)
    parser.add_argument("--bonferroni-threshold", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-unpinned-source", action="store_true")
    parser.add_argument(
        "--allow-nonprimary-preparation",
        action="store_true",
        help="Explicitly allow a preparation labelled as a sensitivity rather than formal primary.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def condition_specs(output: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "database_label": label,
            "output_dir": output / label,
        }
        for label in CONDITION_LABELS
    )


def validate_paired_manifests(manifests: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    if len(manifests) != 2:
        raise ValueError("Exactly two CellAgentChat condition manifests are required.")
    labels = [str(manifest["database_condition"]) for manifest in manifests]
    if tuple(labels) != tuple(CONDITION_LABELS):
        raise RuntimeError(
            f"Unexpected CellAgentChat condition order: {labels}; expected {CONDITION_LABELS}."
        )
    fields = ("mapped_expression", "sample_plan", "preparation_manifest")
    shared: dict[str, str] = {}
    for field in fields:
        hashes = {
            str(manifest["shared_input"][field]["sha256"]) for manifest in manifests
        }
        if len(hashes) != 1:
            raise RuntimeError(
                f"CellAgentChat conditions do not share the same {field} SHA256: {hashes}."
            )
        shared[field] = hashes.pop()
    database_hashes = {
        str(manifest["shared_input"]["database"]["sha256"]) for manifest in manifests
    }
    if len(database_hashes) != 2:
        raise RuntimeError(
            "The two CellAgentChat conditions unexpectedly use the same LR database."
        )
    claims = [
        manifest["shared_input"].get("preparation_claims") for manifest in manifests
    ]
    if claims[0] != claims[1]:
        raise RuntimeError(
            "The two CellAgentChat conditions do not share identical preparation claims."
        )
    return shared


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = prepare_output(args.output_dir, bool(args.overwrite))
    manifests: list[Mapping[str, Any]] = []
    for spec in condition_specs(output):
        child_args = argparse.Namespace(
            preparation_dir=args.preparation_dir,
            cellagentchat_source=args.cellagentchat_source,
            database_label=spec["database_label"],
            output_dir=spec["output_dir"],
            sampling_seeds=args.sampling_seeds,
            stages=args.stages,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            feature_shuffles=args.feature_shuffles,
            permutation_score_target=args.permutation_score_target,
            bonferroni_threshold=args.bonferroni_threshold,
            tau=args.tau,
            delta=args.delta,
            device=args.device,
            allow_unpinned_source=args.allow_unpinned_source,
            allow_nonprimary_preparation=args.allow_nonprimary_preparation,
            overwrite=False,
        )
        manifests.append(run_spatial.run(child_args))
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - runtime dependency
            pass

    shared_hashes = validate_paired_manifests(manifests)
    rows = []
    for manifest in manifests:
        counts = manifest["counts"]
        rows.append(
            {
                "database_condition": manifest["database_condition"],
                "n_runs": counts["n_runs"],
                "type_pair_rows_by_seed": counts["type_pair_rows_by_seed"],
                "raw_lr_rows": counts["raw_lr_rows"],
                "significant_lr_rows": counts["significant_lr_rows"],
                "sample_plan_sha256": manifest["shared_input"]["sample_plan"]["sha256"],
                "mapped_expression_sha256": manifest["shared_input"][
                    "mapped_expression"
                ]["sha256"],
                "database_sha256": manifest["shared_input"]["database"]["sha256"],
                "orthology_policy": manifest["shared_input"]["preparation_claims"][
                    "orthology_policy"
                ],
                "orthology_analysis_tier": manifest["shared_input"][
                    "preparation_claims"
                ]["orthology_analysis_tier"],
                "primary_claim_allowed": manifest["shared_input"]["preparation_claims"][
                    "primary_claim_allowed"
                ],
            }
        )
    readiness_path = output / "dual_condition_run_summary.csv"
    pd.DataFrame(rows).to_csv(readiness_path, index=False)
    manifest_paths = [output / label / "manifest.json" for label in CONDITION_LABELS]
    dual_manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workflow": "official_cellagentchat_spatial_dual_lr_database",
        "conditions": list(CONDITION_LABELS),
        "same_mapped_expression_and_sample_plan_verified": True,
        "same_preparation_manifest_and_orthology_claims_verified": True,
        "shared_sha256": shared_hashes,
        "preparation_claims": manifests[0]["shared_input"]["preparation_claims"],
        "database_sha256_are_distinct": True,
        "condition_manifests": {
            label: artifact(path)
            for label, path in zip(CONDITION_LABELS, manifest_paths)
        },
        "artifacts": {readiness_path.name: artifact(readiness_path)},
    }
    write_json(output / "manifest.json", dual_manifest)
    return dual_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(json.dumps(json_value({"status": "ok", **manifest}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
