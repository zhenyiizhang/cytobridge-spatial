from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scanpy as sc

import prepare_and_run as base


LOWER_TRANSLATION = {
    timepoint: base.RigidPerturbation(
        translate_x_nn=spec.translate_x_nn * 0.5,
        translate_y_nn=spec.translate_y_nn * 0.5,
    )
    for timepoint, spec in base._translation("moderate").items()
}
LOWER_ROTATION = {
    timepoint: base.RigidPerturbation(rotation_deg=spec.rotation_deg * 0.5)
    for timepoint, spec in base._rotation("moderate").items()
}
LOWER_COMBINED = {
    timepoint: base.RigidPerturbation(
        rotation_deg=LOWER_ROTATION[timepoint].rotation_deg,
        translate_x_nn=LOWER_TRANSLATION[timepoint].translate_x_nn,
        translate_y_nn=LOWER_TRANSLATION[timepoint].translate_y_nn,
    )
    for timepoint in base.TIME_ORDER
}
LOWER_SPECS = {
    "translate_low": LOWER_TRANSLATION,
    "rotate_low": LOWER_ROTATION,
    "translate_rotate_low": LOWER_COMBINED,
}


def prepare() -> None:
    source = sc.read_h5ad(base.SOURCE_INPUT)
    source_hash = base._sha256(base.SOURCE_INPUT)
    records = []
    for variant, specs in LOWER_SPECS.items():
        output = base.INPUT_DIR / f"{variant}.h5ad"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing input: {output}")
        adata = source.copy()
        accepted = np.asarray(adata.obsm[base.SPATIAL_KEY], dtype=np.float64).copy()
        perturbed = accepted.copy()
        stages = []
        for timepoint in base.TIME_ORDER:
            mask = np.asarray(adata.obs["timepoint"].astype(str) == timepoint)
            transformed, nn_distance = base._apply_rigid(accepted[mask], specs[timepoint])
            perturbed[mask] = transformed
            stages.append(
                {
                    "timepoint": timepoint,
                    "median_nn_distance": nn_distance,
                    **asdict(specs[timepoint]),
                    "max_input_shift": float(
                        np.linalg.norm(transformed - accepted[mask], axis=1).max()
                    ),
                }
            )
        adata.obsm[f"{base.SPATIAL_KEY}_accepted_baseline"] = accepted
        adata.obsm[base.SPATIAL_KEY] = perturbed
        adata.uns["alignment_sensitivity_audit"] = {
            "source_input": str(base.SOURCE_INPUT),
            "source_sha256": source_hash,
            "source_coordinate_key": base.SPATIAL_KEY,
            "variant": variant,
            "design": "0.5 times the preregistered moderate rigid-perturbation vector",
            "stage_records_json": json.dumps(stages, sort_keys=True),
            "important": "D7 accepted 180-degree pre-orientation is retained before perturbation.",
        }
        adata.write_h5ad(output)
        records.append(
            {
                "variant": variant,
                "path": str(output),
                "sha256": base._sha256(output),
                "source_sha256": source_hash,
                "stages": stages,
            }
        )
    (base.AUDIT_ROOT / "mild_input_manifest.json").write_text(
        json.dumps(
            {
                "design": "0.5 times the preregistered moderate perturbation vector",
                "accepted_source": str(base.SOURCE_INPUT),
                "accepted_source_sha256": source_hash,
                "coordinate_key": base.SPATIAL_KEY,
                "variants": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def run(variant: str, device: str) -> None:
    input_path = base.INPUT_DIR / f"{variant}.h5ad"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    run_dir = base.RUNS_DIR / variant
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(base.CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    base._validate_template(config)
    config = deepcopy(config)
    config["steps"]["default"] = ["preprocess", "downstream"]
    config["preprocess"]["note"] = (
        "Low-level alignment-sensitivity audit from the accepted chicken-heart input. "
        "The perturbation is 0.5 times the preregistered moderate vector and is applied "
        "to obsm['spatial_ot_input'], retaining the accepted D7 pre-orientation."
    )
    config_path = run_dir / "workflow_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "CytoBridge.cli",
        "workflow",
        "--config",
        str(config_path),
        "--train",
        "--input-h5ad",
        str(input_path),
        "--output-dir",
        str(run_dir),
        "--device",
        device,
    ]
    environment = os.environ.copy()
    if environment.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Launch with PYTHONHASHSEED=0")
    manifest = {
        "variant": variant,
        "status": "started",
        "paired_reference": str(base.RUNS_DIR / "baseline_repeat"),
        "accepted_package_commit": "c72e592d0dea70941bc4971a79c3c903d7454b08",
        "input_h5ad": str(input_path),
        "input_sha256": base._sha256(input_path),
        "config": str(config_path),
        "command": command,
        "pythonhashseed": environment["PYTHONHASHSEED"],
    }
    manifest_path = run_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    subprocess.run(command, check=True, cwd=base.PACKAGE_ROOT, env=environment)
    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("variant", choices=tuple(LOWER_SPECS))
    run_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run(args.variant, args.device)


if __name__ == "__main__":
    main()
