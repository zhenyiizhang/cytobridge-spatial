#!/usr/bin/env python3
"""Score one corrected no-interaction model beside a frozen Full baseline.

This analysis deliberately reuses the accepted Full metrics and checkpoint.
Only the no-interaction arm is newly simulated, under the historical runtime
and exact accepted inference settings for the corresponding dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import pandas as pd

from CytoBridge.tl.downstream import (
    evaluate_model_distributions,
    load_dynamical_model_from_dir,
    save_distribution_evaluation,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("weinreb", "scnt_cortex"), required=True)
    parser.add_argument("--prepared-h5ad", required=True, type=Path)
    parser.add_argument("--full-metrics", required=True, type=Path)
    parser.add_argument("--full-weight", required=True, type=Path)
    parser.add_argument("--full-score", required=True, type=Path)
    parser.add_argument("--no-interaction-model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    prepared_path = args.prepared_h5ad.expanduser().resolve()
    no_model_dir = args.no_interaction_model_dir.expanduser().resolve()
    adata = ad.read_h5ad(prepared_path)
    if tuple(adata.obsm["X_latent"].shape)[1] != 50:
        raise ValueError("Expected a 50-dimensional X_latent model input")
    expected_times = {
        "weinreb": [0.0, 1.0, 2.0],
        "scnt_cortex": [0.0, 0.25, 0.5, 1.0, 2.0],
    }[args.dataset]
    actual_times = sorted(
        float(value) for value in pd.unique(adata.obs["time_point_processed"])
    )
    if actual_times != expected_times:
        raise ValueError(f"Unexpected time grid: {actual_times}")

    loaded = load_dynamical_model_from_dir(
        no_model_dir,
        dim=50,
        device=args.device,
        stage="Finetune_no_interaction",
    )
    result = evaluate_model_distributions(
        adata,
        loaded.model,
        time_points=expected_times,
        n_samples=2048,
        dt=0.05,
        sigma=0.1,
        include_score=True,
        interaction_m=16,
        max_ot_points=1024,
        structure_max_points=2048,
        device=args.device,
        time_key="time_point_processed",
        obsm_key="X_latent",
        concat_spatial=False,
        random_seed=42,
        include_initial_time=False,
        verbose=True,
        noise_seed=42,
    )
    no_dir = output / "no_interaction_noise"
    artifacts = save_distribution_evaluation(result, no_dir)
    no_metrics = Path(artifacts["metrics"]).resolve()

    full_metrics = args.full_metrics.expanduser().resolve()
    full = pd.read_csv(full_metrics)
    no_interaction = pd.read_csv(no_metrics)
    keys = ["time", "space"]
    if full.duplicated(keys).any() or no_interaction.duplicated(keys).any():
        raise ValueError("Distribution metrics have duplicate time/space rows")
    paired = full.merge(
        no_interaction,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_full", "_no_interaction"),
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Full and no-interaction metrics do not share exact rows")
    paired = paired.drop(columns="_merge")
    for metric in ("w1", "w2", "tmv"):
        paired[f"{metric}_delta_no_interaction_minus_full"] = (
            paired[f"{metric}_no_interaction"] - paired[f"{metric}_full"]
        )
        paired[f"{metric}_relative_change"] = (
            paired[f"{metric}_delta_no_interaction_minus_full"]
            / paired[f"{metric}_full"]
        )
    paired_path = output / "paired_weighted_distribution_metrics.csv"
    paired.to_csv(paired_path, index=False)

    script = Path(__file__).resolve()
    weight_path = Path(loaded.weight_path).resolve()
    score_path = Path(loaded.score_path).resolve()
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "frozen existing Full checkpoint versus corrected retrained no-interaction",
        "claim_scope": (
            "descriptive single-seed frozen-baseline ablation; not strict CRN matched retraining"
        ),
        "dataset": args.dataset,
        "training_seed": 42,
        "simulation": {
            "continuous_from_t0": True,
            "n_samples": 2048,
            "dt": 0.05,
            "sigma": 0.1,
            "include_score": True,
            "include_interaction": False,
            "interaction_m": 16,
            "max_ot_points": 1024,
            "structure_max_points": 2048,
            "random_seed": 42,
            "noise_seed": 42,
            "weighted_ot": True,
            "native_unnormalized_growth_mass_for_tmv": True,
        },
        "inputs": {
            "prepared_h5ad": file_record(prepared_path),
            "frozen_full_metrics": file_record(full_metrics),
            "frozen_full_weight": file_record(args.full_weight),
            "frozen_full_score": file_record(args.full_score),
            "corrected_no_interaction_config": file_record(no_model_dir / "config.yaml"),
            "corrected_no_interaction_weight": file_record(weight_path),
            "corrected_no_interaction_score": file_record(score_path),
        },
        "model_load": {
            "weight_stage": loaded.weight_stage,
            "score_stage": loaded.score_stage,
        },
        "outputs": {
            "no_interaction_metrics": file_record(no_metrics),
            "paired_metrics": file_record(paired_path),
        },
        "implementation": file_record(script),
    }
    manifest_path = output / "frozen_baseline_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(paired[["time", "space", "w1_relative_change", "w2_relative_change", "tmv_relative_change"]].to_string(index=False))
    print(manifest_path)


if __name__ == "__main__":
    main()
