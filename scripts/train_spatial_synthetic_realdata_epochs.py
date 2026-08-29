#!/usr/bin/env python3
"""Train the v11b benchmark with the standard real-data epoch allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml


EXPECTED_STAGES = (
    "Pretrain",
    "Refine",
    "Init_interaction",
    "Train_Score",
    "Finetune",
    "Score_Refine",
)
EXPECTED_TOTAL_EPOCHS = 5252


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from CytoBridge.tl.train import fit

    data_dir = args.data_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    input_h5ad = data_dir / "attractive_observed.h5ad"
    data_manifest = data_dir / "manifest.json"
    for path in (input_h5ad, data_manifest, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output-root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    provenance = output_root / "provenance"
    provenance.mkdir()
    shutil.copy2(config_path, provenance / config_path.name)
    shutil.copy2(Path(__file__).resolve(), provenance / Path(__file__).name)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stages = config["training"]["plan"]
    stage_names = tuple(str(stage["name"]) for stage in stages)
    if stage_names != EXPECTED_STAGES:
        raise ValueError(
            f"Expected controlled six-stage sequence {EXPECTED_STAGES}; got {stage_names}."
        )
    total_epochs = sum(int(stage["epochs"]) for stage in stages)
    if total_epochs != EXPECTED_TOTAL_EPOCHS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_EPOCHS} total epochs; got {total_epochs}."
        )
    config["ckpt_dir"] = str(output_root / "model")

    fit(
        str(input_h5ad),
        config=config,
        device=args.device,
        time_key="time_point_processed",
        obsm_key="X_latent",
        is_spatial=True,
        spatial_key="spatial_aligned",
        evaluate_after_training=False,
    )

    final_model_dir = output_root / "model"
    final_checkpoint = final_model_dir / "Score_Refine/best_model.pth"
    if not final_checkpoint.is_file():
        raise FileNotFoundError(final_checkpoint)
    manifest = {
        "schema_version": "cytobridge_spatial_attraction_realdata_epochs/1",
        "input_h5ad": str(input_h5ad),
        "input_h5ad_sha256": _sha256(input_h5ad),
        "data_manifest": str(data_manifest),
        "data_manifest_sha256": _sha256(data_manifest),
        "config_source": str(config_path),
        "config_source_sha256": _sha256(config_path),
        "resolved_config": str(final_model_dir / "config.yaml"),
        "device": str(args.device),
        "training_seed": int(config["seed"]),
        "stage_names": list(stage_names),
        "total_epochs": total_epochs,
        "final_model_dir": str(final_model_dir),
        "final_stage": "Score_Refine",
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": _sha256(final_checkpoint),
        "force_supervision_used": False,
        "analytic_potential_used_for_fitting": False,
        "controlled_comparison": {
            "reference": "v11b_six_stage_control_same_arch_loss_budget",
            "held_constant": [
                "input data and GT",
                "model architecture",
                "radius pairwise_radial nonlinear interaction head",
                "fixed Gamma = 3 I_2 gene-force map",
                "cutoff 0.30",
                "sigma 0.015",
                "alpha_spatial 1.0 and alpha_express 1.0",
                "group and batch size 192",
                "OT and flow-matching loss families and coefficients",
                "training seed 42",
            ],
            "intended_difference": (
                "benchmark epoch allocation 100/100/100/3001/200/100 versus "
                "standard real-data allocation 100/100/50/2001/1000/2001"
            ),
        },
    }
    (output_root / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
