#!/usr/bin/env python3
"""Train one Zebrafish loss-sensitivity model through the public package API."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import CytoBridge as cb


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--edge-predictor-path", type=Path, required=True)
    parser.add_argument("--edge-predictor-threshold", type=float, required=True)
    parser.add_argument("--interaction-cutoff", type=float, required=True)
    parser.add_argument("--expected-alpha-express", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _args()
    aligned_h5ad = args.aligned_h5ad.expanduser().resolve(strict=True)
    config_path = args.training_config.expanduser().resolve(strict=True)
    edge_predictor = args.edge_predictor_path.expanduser().resolve(strict=True)
    training_dir = args.training_dir.expanduser().resolve()
    if training_dir.exists() and any(training_dir.iterdir()):
        raise FileExistsError(f"Training directory is not empty: {training_dir}")
    training_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    actual_alpha = float(config["training"]["defaults"]["alpha_express"])
    if actual_alpha != float(args.expected_alpha_express):
        raise ValueError(
            f"Expected alpha_express={args.expected_alpha_express}, got {actual_alpha}"
        )
    cb.tl.fit(
        str(aligned_h5ad),
        config=config,
        device=str(args.device),
        time_key="time_point_processed",
        obsm_key="X_latent",
        is_spatial=True,
        spatial_key="spatial_aligned",
        ckpt_dir=str(training_dir),
        interaction_cutoff=float(args.interaction_cutoff),
        edge_predictor_path=str(edge_predictor),
        edge_predictor_threshold=float(args.edge_predictor_threshold),
        evaluate_after_training=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
