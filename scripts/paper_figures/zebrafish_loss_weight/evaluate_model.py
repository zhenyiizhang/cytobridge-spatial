#!/usr/bin/env python3
"""Evaluate one trained zebrafish model for the loss-weight comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-h5ad", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--max-ot-points", type=int, default=1024)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import scanpy as sc

    from CytoBridge.tl import (
        evaluate_model_distributions,
        load_dynamical_model_from_dir,
        save_distribution_evaluation,
    )

    aligned_h5ad = args.aligned_h5ad.expanduser().resolve(strict=True)
    training_dir = args.training_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    adata = sc.read_h5ad(aligned_h5ad)
    loaded = load_dynamical_model_from_dir(
        training_dir,
        dim=int(adata.obsm["spatial_aligned"].shape[1] + adata.obsm["X_latent"].shape[1]),
        device=str(args.device),
    )
    result = evaluate_model_distributions(
        adata,
        loaded.model,
        n_samples=int(args.n_samples),
        dt=float(args.dt),
        sigma=float(loaded.config["training"]["defaults"].get("sigma", 0.03)),
        include_score=True,
        interaction_m=int(args.interaction_m),
        max_ot_points=int(args.max_ot_points),
        device=str(args.device),
        random_seed=int(args.random_seed),
        include_initial_time=False,
    )
    paths = save_distribution_evaluation(result, output_dir)
    record = {
        "condition": str(args.condition),
        "aligned_h5ad": str(aligned_h5ad),
        "training_dir": str(training_dir),
        "checkpoint": str(loaded.weight_path),
        "score_checkpoint": str(loaded.score_path),
        "settings": dict(result.settings),
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
