from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scanpy as sc


PROJECT_ROOT = Path("/data/cytobridge/projects/CytoBridge-ST-1104")
SOURCE_INPUT = (
    PROJECT_ROOT
    / "runs/chicken-heart-ot-alignment-20260822-f5550e1-r1/result/chicken_heart_ot_aligned.h5ad"
)
ACCEPTED_RUN = PROJECT_ROOT / "runs/chicken-heart-full-ot-20260823-r2"
PACKAGE_ROOT = ACCEPTED_RUN / "software"
CONFIG_TEMPLATE = PACKAGE_ROOT / "CytoBridge/workflow_configs/chicken_heart.json"
AUDIT_ROOT = PROJECT_ROOT / "runs/chicken-heart-alignment-sensitivity-audit-20260831-r1"
INPUT_DIR = AUDIT_ROOT / "inputs"
RUNS_DIR = AUDIT_ROOT / "runs"

TIME_ORDER = ("D4", "D7", "D10", "D14")
SPATIAL_KEY = "spatial_ot_input"


@dataclass(frozen=True)
class RigidPerturbation:
    rotation_deg: float = 0.0
    translate_x_nn: float = 0.0
    translate_y_nn: float = 0.0


def _translation(level: str) -> dict[str, RigidPerturbation]:
    values = {
        "moderate": {
            "D4": (-0.56, 0.44),
            "D7": (0.60, -0.18),
            "D10": (-0.34, -0.48),
            "D14": (0.44, 0.40),
        },
        "strong": {
            "D4": (-1.05, 0.82),
            "D7": (1.08, -0.34),
            "D10": (-0.62, -0.92),
            "D14": (0.82, 0.78),
        },
    }[level]
    return {
        tp: RigidPerturbation(translate_x_nn=xy[0], translate_y_nn=xy[1])
        for tp, xy in values.items()
    }


def _rotation(level: str) -> dict[str, RigidPerturbation]:
    values = {
        "moderate": {"D4": 5.5, "D7": -5.8, "D10": 6.2, "D14": -6.0},
        "strong": {"D4": 10.5, "D7": -11.0, "D10": 12.0, "D14": -11.5},
    }[level]
    return {tp: RigidPerturbation(rotation_deg=value) for tp, value in values.items()}


def _combined(level: str) -> dict[str, RigidPerturbation]:
    translations = _translation(level)
    rotations = _rotation(level)
    return {
        tp: RigidPerturbation(
            rotation_deg=rotations[tp].rotation_deg,
            translate_x_nn=translations[tp].translate_x_nn,
            translate_y_nn=translations[tp].translate_y_nn,
        )
        for tp in TIME_ORDER
    }


VARIANT_SPECS = {
    "translate_moderate": _translation("moderate"),
    "translate_strong": _translation("strong"),
    "rotate_moderate": _rotation("moderate"),
    "rotate_strong": _rotation("strong"),
    "translate_rotate_moderate": _combined("moderate"),
    "translate_rotate_strong": _combined("strong"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _median_nn_distance(xy: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(distances[:, 1]))


def _apply_rigid(xy: np.ndarray, spec: RigidPerturbation) -> tuple[np.ndarray, float]:
    xy = np.asarray(xy, dtype=np.float64)
    center = xy.mean(axis=0, keepdims=True)
    theta = np.deg2rad(spec.rotation_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )
    nn_distance = _median_nn_distance(xy)
    transformed = (xy - center) @ rotation.T + center
    transformed[:, 0] += spec.translate_x_nn * nn_distance
    transformed[:, 1] += spec.translate_y_nn * nn_distance
    return transformed, nn_distance


def _validate_template(config: dict) -> None:
    checks = {
        "dataset": config["dataset"]["name"],
        "scientific_seed": config["scientific"]["seed"],
        "preprocess_mode": config["preprocess"]["mode"],
        "input_spatial_key": config["preprocess"]["align"]["input_spatial_key"],
        "alignment_seed": config["preprocess"]["align"]["random_seed"],
    }
    expected = {
        "dataset": "chicken_heart",
        "scientific_seed": 42,
        "preprocess_mode": "fit_spatial_alignment",
        "input_spatial_key": SPATIAL_KEY,
        "alignment_seed": 42,
    }
    if checks != expected:
        raise RuntimeError(f"Unexpected accepted workflow config: {checks}; expected {expected}")


def prepare_inputs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = sc.read_h5ad(SOURCE_INPUT)
    if SPATIAL_KEY not in source.obsm:
        raise KeyError(f"Accepted input lacks obsm[{SPATIAL_KEY!r}]")
    if "counts" not in source.layers:
        raise KeyError("Accepted input lacks layers['counts']")

    source_hash = _sha256(SOURCE_INPUT)
    records: list[dict] = []
    for variant, specs in VARIANT_SPECS.items():
        output_path = INPUT_DIR / f"{variant}.h5ad"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing input: {output_path}")
        adata = source.copy()
        base = np.asarray(adata.obsm[SPATIAL_KEY], dtype=np.float64).copy()
        perturbed = base.copy()
        stage_records = []
        for timepoint in TIME_ORDER:
            mask = np.asarray(adata.obs["timepoint"].astype(str) == timepoint)
            transformed, nn_distance = _apply_rigid(base[mask], specs[timepoint])
            perturbed[mask] = transformed
            stage_records.append(
                {
                    "timepoint": timepoint,
                    "median_nn_distance": nn_distance,
                    **asdict(specs[timepoint]),
                    "max_input_shift": float(np.linalg.norm(transformed - base[mask], axis=1).max()),
                }
            )
        adata.obsm[f"{SPATIAL_KEY}_accepted_baseline"] = base
        adata.obsm[SPATIAL_KEY] = perturbed
        adata.uns["alignment_sensitivity_audit"] = {
            "source_input": str(SOURCE_INPUT),
            "source_sha256": source_hash,
            "source_coordinate_key": SPATIAL_KEY,
            "variant": variant,
            "stage_records_json": json.dumps(stage_records, sort_keys=True),
            "important": "D7 accepted 180-degree pre-orientation is retained before perturbation.",
        }
        adata.write_h5ad(output_path)
        records.append(
            {
                "variant": variant,
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "source_sha256": source_hash,
                "stage_records": stage_records,
            }
        )

    manifest = {
        "accepted_source": str(SOURCE_INPUT),
        "accepted_source_sha256": source_hash,
        "coordinate_key": SPATIAL_KEY,
        "baseline_repeat_uses_accepted_source_directly": True,
        "variants": records,
    }
    (AUDIT_ROOT / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _input_for_variant(variant: str) -> Path:
    if variant == "baseline_repeat":
        return SOURCE_INPUT
    if variant not in VARIANT_SPECS:
        raise KeyError(f"Unknown variant: {variant}")
    return INPUT_DIR / f"{variant}.h5ad"


def run_variant(variant: str, device: str) -> None:
    input_path = _input_for_variant(variant)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    run_dir = RUNS_DIR / variant
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG_TEMPLATE.read_text(encoding="utf-8"))
    _validate_template(config)
    config = deepcopy(config)
    config["steps"]["default"] = ["preprocess", "downstream"]
    config["preprocess"]["note"] = (
        "Alignment-sensitivity audit from the accepted chicken-heart input. "
        "Perturbations are applied to obsm['spatial_ot_input'], preserving the "
        "accepted D7 180-degree pre-orientation."
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
        raise RuntimeError("Launch with PYTHONHASHSEED=0 for a paired deterministic audit")
    started_manifest = {
        "variant": variant,
        "status": "started",
        "accepted_package_commit": "c72e592d0dea70941bc4971a79c3c903d7454b08",
        "package_root": str(PACKAGE_ROOT),
        "input_h5ad": str(input_path),
        "input_sha256": _sha256(input_path),
        "config": str(config_path),
        "command": command,
        "pythonhashseed": environment["PYTHONHASHSEED"],
    }
    manifest_path = run_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(started_manifest, indent=2), encoding="utf-8")
    subprocess.run(command, check=True, cwd=PACKAGE_ROOT, env=environment)
    started_manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(started_manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("variant", choices=["baseline_repeat", *VARIANT_SPECS])
    run_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_inputs()
    else:
        run_variant(args.variant, args.device)


if __name__ == "__main__":
    main()
