#!/usr/bin/env python3
"""Run the canonical clean-counts zebrafish preprocessing/training/evaluation workflow.

The manuscript-selected condition uses ``alpha_express=0.015``. The optional
``alpha_express=0.05`` sensitivity comparator shares the same preprocessing and
differs only in ``training.defaults.alpha_express``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402
from CytoBridge.pp import AlignConfig  # noqa: E402
from CytoBridge.tl import (  # noqa: E402
    adata_to_aligned_dataframe,
    compute_velocity_components_from_adata,
    evaluate_model_distributions,
    load_dynamical_model_from_dir,
    save_distribution_evaluation,
)
from CytoBridge.utils.config import load_config  # noqa: E402
from preprocess_pipeline import run_preprocessing_pipeline  # noqa: E402


CONDITIONS = {
    "alpha_express_0015": {
        "alpha_express": 0.015,
        "config": "zebrafish_spatial_full_alpha_express_0015.yaml",
    },
    "alpha_express_005": {
        "alpha_express": 0.05,
        "config": "zebrafish_spatial_full.yaml",
    },
}

ZEBRAFISH_TIME_MAPPING = {
    "3.3hpf": -1.0,
    "5.25hpf": 0.0,
    "10hpf": 1.0,
    "12hpf": 2.0,
    "18hpf": 3.0,
    "24hpf": 4.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _git_revision() -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": revision, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def _paths(output_dir: Path, condition: str | None) -> dict[str, Path]:
    preprocess_dir = output_dir / "preprocess"
    result = {
        "preprocess_dir": preprocess_dir,
        "aligned_csv": preprocess_dir / "zebrafish_aligned.csv",
        "aligned_h5ad": preprocess_dir / "zebrafish_aligned.h5ad",
        "annotated_csv": preprocess_dir / "zebrafish_aligned_with_annotation.csv",
        "edge_path": preprocess_dir / "edge_classifier" / "zebrafish_edge_model.pt",
    }
    if condition is not None:
        condition_dir = output_dir / "conditions" / condition
        result.update(
            {
                "condition_dir": condition_dir,
                "training_dir": condition_dir / "training",
                "downstream_dir": condition_dir / "downstream",
            }
        )
    return result


def _profile_defaults(profile: str) -> dict[str, int | None]:
    if profile == "smoke":
        return {
            "max_cells": 64,
            "phase1_epochs": 2,
            "phase2_epochs": 1,
            "edge_epochs": 1,
            "evaluation_n_samples": 64,
        }
    return {
        "max_cells": None,
        "phase1_epochs": 10000,
        "phase2_epochs": 500,
        "edge_epochs": 100,
        "evaluation_n_samples": 5000,
    }


def _write_annotated_table(aligned_h5ad: Path, annotated_csv: Path) -> None:
    adata = sc.read_h5ad(aligned_h5ad)
    required = ["bin_annotation", "colors", "time"]
    missing = [key for key in required if key not in adata.obs]
    if missing:
        raise KeyError(f"Aligned zebrafish H5AD is missing obs columns: {missing}")

    adata.obs["Annotation"] = adata.obs["bin_annotation"].astype(str).values
    adata.obs["Color"] = adata.obs["colors"].astype(str).values
    adata.obs["time_label"] = adata.obs["time"].astype(str).values
    adata.write_h5ad(aligned_h5ad)

    frame, _ = adata_to_aligned_dataframe(
        adata,
        time_key="time_point_processed",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        annotation_key="Annotation",
    )
    by_cell = adata.obs[["Color", "time_label"]].copy()
    by_cell["cell_id"] = adata.obs_names.astype(str)
    frame = frame.merge(by_cell, on="cell_id", how="left", validate="1:1")
    annotated_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(annotated_csv, index=False)


def _run_preprocess(args, paths: dict[str, Path], defaults: dict) -> dict:
    cfg = AlignConfig(
        n_top_genes=2000,
        n_pcs=50,
        normalization_target_sum=None,
        spatial_dim=2,
        shared_scale=500.0,
        center_x=True,
        center_y=False,
        center_z=False,
        flip_y=False,
        phase1_epochs=int(defaults["phase1_epochs"]),
        phase2_epochs=int(defaults["phase2_epochs"]),
        alpha=5.0,
        beta=0.01,
        lambda_local=100.0,
        lambda_ot=1.0,
        batch_size=1024,
        distance_pairs=10000,
        learning_rate=1e-3,
        random_seed=int(args.random_seed),
        expression_layer="counts",
        allow_retransform_preprocessed_x=False,
        time_mapping=ZEBRAFISH_TIME_MAPPING,
    )
    result = run_preprocessing_pipeline(
        data_name="zebrafish",
        h5ad_path=str(args.h5ad_path),
        time_key="time",
        output_dir=str(paths["preprocess_dir"]),
        align_config=cfg,
        database_path=str(args.database_path),
        edge_epochs=int(defaults["edge_epochs"]),
        batch_indices=[1, 2, 3, 4, 5],
        max_input_cells_per_timepoint=defaults["max_cells"],
        strip_uns_arrays_larger_than_mb=10.0,
        edge_predictor_threshold=None,
        neighborhood_threshold=None,
        spatial_key="spatial_aligned",
        device=str(args.device),
    )
    _write_annotated_table(paths["aligned_h5ad"], paths["annotated_csv"])
    adata = sc.read_h5ad(paths["aligned_h5ad"], backed="r")
    actual_times = sorted(
        pd.to_numeric(adata.obs["time_point_processed"], errors="raise").unique().tolist()
    )
    if actual_times != [0.0, 1.0, 2.0, 3.0, 4.0]:
        raise RuntimeError(f"Expected canonical zebrafish model times 0..4, got {actual_times}")
    return result


def _resolved_training_config(args, condition: str, training_dir: Path) -> dict:
    condition_spec = CONDITIONS[condition]
    config_path = (
        Path(args.training_config).expanduser().resolve()
        if args.training_config is not None
        else REPO_ROOT / "CytoBridge" / "configs" / condition_spec["config"]
    )
    config = load_config(str(_require_file(config_path, "zebrafish training config")))
    config["seed"] = int(args.random_seed)
    config["ckpt_dir"] = str(training_dir)
    config["model"]["spatial_dim"] = 2
    config["training"]["defaults"]["alpha_spatial"] = 10.0
    config["training"]["defaults"]["alpha_express"] = float(
        condition_spec["alpha_express"]
    )
    if args.profile == "smoke":
        for stage in config["training"]["plan"]:
            stage["epochs"] = 1
            stage["batch_size"] = min(int(stage.get("batch_size", 32)), 32)
    return config


def _run_train(args, paths: dict[str, Path], condition: str) -> None:
    aligned_h5ad = _require_file(paths["aligned_h5ad"], "aligned zebrafish H5AD")
    config = _resolved_training_config(args, condition, paths["training_dir"])
    paths["training_dir"].mkdir(parents=True, exist_ok=True)
    launch_manifest = {
        "condition": condition,
        "alpha_spatial": float(config["training"]["defaults"]["alpha_spatial"]),
        "alpha_express": float(config["training"]["defaults"]["alpha_express"]),
        "profile": args.profile,
        "requested_device": str(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "assigned_gpu": os.environ.get("CYTOBRIDGE_ASSIGNED_GPU"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "git": _git_revision(),
        "aligned_h5ad": str(aligned_h5ad),
        "aligned_h5ad_sha256": _sha256(aligned_h5ad),
        "training_config": (
            str(Path(args.training_config).expanduser().resolve())
            if args.training_config is not None
            else None
        ),
    }
    (paths["training_dir"] / "launch_manifest.json").write_text(
        json.dumps(_json_ready(launch_manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cb.tl.fit(
        str(aligned_h5ad),
        config=config,
        device=str(args.device),
        ckpt_dir=str(paths["training_dir"]),
        evaluate_after_training=False,
    )


def _run_downstream(args, paths: dict[str, Path], condition: str, defaults: dict) -> dict:
    aligned_h5ad = _require_file(paths["aligned_h5ad"], "aligned zebrafish H5AD")
    training_dir = paths["training_dir"].resolve()
    output_dir = paths["downstream_dir"].resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(aligned_h5ad)
    spatial_dim = int(adata.obsm["spatial_aligned"].shape[1])
    latent_dim = int(adata.obsm["X_latent"].shape[1])
    dim = spatial_dim + latent_dim
    loaded = load_dynamical_model_from_dir(
        training_dir,
        dim=dim,
        device=str(args.device),
    )
    configured_score_stages = [
        str(stage["name"])
        for stage in loaded.config.get("training", {}).get("plan", [])
        if stage.get("name")
        and (
            str(stage.get("mode", "")).lower() == "score_matching"
            or str(stage.get("train_strategy", "")).lower() == "s"
        )
    ]
    expected_score_stage = configured_score_stages[-1]
    if loaded.score_stage != expected_score_stage:
        raise RuntimeError(
            f"Expected final score stage {expected_score_stage}, loaded {loaded.score_stage}"
        )

    interaction_cfg = loaded.config["model"].get("interaction_net", {})
    interaction_cutoff = float(interaction_cfg.get("cutoff", 0.0))
    components = compute_velocity_components_from_adata(
        adata,
        loaded.model,
        dim=dim,
        interaction_m=int(args.interaction_m),
        interaction_threshold=interaction_cutoff,
        device=str(args.device),
        time_key="time_point_processed",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        write_to_adata=False,
        reuse_if_present=False,
    )
    identity_error = float(
        np.max(
            np.abs(
                components["full"]
                - components["drift"]
                - components["interaction"]
                - components["score"]
            )
        )
    )
    np.savez_compressed(
        output_dir / "velocity_components.npz",
        times=components["times"],
        features=components["features"],
        drift=components["drift"],
        interaction=components["interaction"],
        score=components["score"],
        full=components["full"],
    )

    evaluation = evaluate_model_distributions(
        adata,
        loaded.model,
        n_samples=int(defaults["evaluation_n_samples"]),
        dt=float(args.evaluation_dt),
        sigma=float(loaded.config["training"]["defaults"].get("sigma", 0.03)),
        include_score=True,
        interaction_m=int(args.interaction_m),
        max_ot_points=int(args.evaluation_max_ot_points),
        device=str(args.device),
        random_seed=int(args.random_seed),
    )
    evaluation_paths = save_distribution_evaluation(
        evaluation,
        output_dir / "distribution_evaluation",
    )

    manifest = {
        "workflow": "zebrafish_clean_counts_end_to_end",
        "condition": condition,
        "alpha_spatial": float(loaded.config["training"]["defaults"]["alpha_spatial"]),
        "alpha_express": float(loaded.config["training"]["defaults"]["alpha_express"]),
        "runtime": {
            "requested_device": str(args.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "assigned_gpu": os.environ.get("CYTOBRIDGE_ASSIGNED_GPU"),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
        },
        "git": _git_revision(),
        "input_h5ad": (
            str(args.h5ad_path) if args.h5ad_path is not None else None
        ),
        "input_h5ad_sha256": (
            _sha256(args.h5ad_path) if args.h5ad_path is not None else None
        ),
        "aligned_h5ad": str(aligned_h5ad),
        "aligned_h5ad_sha256": _sha256(aligned_h5ad),
        "preprocess_expression_source": adata.uns.get("preprocess_info", {}).get(
            "expression_source"
        ),
        "pca_center_info": adata.uns.get("pca_center_info"),
        "time_mapping": ZEBRAFISH_TIME_MAPPING,
        "time_counts": {
            str(key): int(value)
            for key, value in adata.obs["time_point_processed"].value_counts().sort_index().items()
        },
        "training_dir": str(training_dir),
        "resolved_training_config": str(training_dir / "config.yaml"),
        "weight_stage": loaded.weight_stage,
        "weight_checkpoint": str(loaded.weight_path),
        "weight_checkpoint_sha256": _sha256(loaded.weight_path),
        "score_stage": loaded.score_stage,
        "score_checkpoint": str(loaded.score_path),
        "score_checkpoint_sha256": _sha256(loaded.score_path),
        "interaction_cutoff": (
            interaction_cutoff if interaction_cfg else None
        ),
        "edge_prior_mode": (
            str(interaction_cfg.get("edge_prior_mode", "learned"))
            if interaction_cfg
            else None
        ),
        "edge_predictor_path": (
            str(interaction_cfg["edge_predictor_path"])
            if interaction_cfg.get("edge_predictor_path") is not None
            else None
        ),
        "edge_predictor_threshold": (
            float(interaction_cfg["edge_predictor_thre"])
            if interaction_cfg.get("edge_predictor_thre") is not None
            else None
        ),
        "full_identity_max_error": identity_error,
        "all_finite": bool(
            all(np.isfinite(components[key]).all() for key in ("drift", "interaction", "score", "full"))
        ),
        "distribution_evaluation": {
            "paths": evaluation_paths,
            "settings": dict(evaluation.settings),
            "mean_by_space": (
                evaluation.metrics.groupby("space", sort=True)
                .mean(numeric_only=True)
                .reset_index()
                .to_dict(orient="records")
            ),
        },
    }
    manifest = _json_ready(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5ad-path",
        type=Path,
        default=None,
        help="Raw source H5AD; required only when preprocessing is requested.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help="LR database; required only when preprocessing is requested.",
    )
    parser.add_argument(
        "--preprocessed-h5ad",
        type=Path,
        default=None,
        help=(
            "Reuse an existing canonical aligned H5AD for isolated train or "
            "downstream runs. The source is never copied or modified."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "train", "downstream", "all"],
        default="all",
    )
    parser.add_argument("--condition", choices=sorted(CONDITIONS), default=None)
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--evaluation-max-ot-points", type=int, default=1024)
    parser.add_argument("--evaluation-dt", type=float, default=0.01)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.stage in {"preprocess", "all"}:
        if args.h5ad_path is None or args.database_path is None:
            raise ValueError(
                "--h5ad-path and --database-path are required for preprocessing."
            )
        args.h5ad_path = _require_file(args.h5ad_path, "zebrafish source H5AD")
        args.database_path = _require_file(
            args.database_path, "ligand-receptor database"
        )
    else:
        if args.h5ad_path is not None:
            args.h5ad_path = _require_file(
                args.h5ad_path, "zebrafish source H5AD"
            )
        if args.database_path is not None:
            args.database_path = _require_file(
                args.database_path, "ligand-receptor database"
            )
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"train", "downstream", "all"} and args.condition is None:
        raise ValueError("--condition is required for train/downstream/all stages")

    defaults = _profile_defaults(args.profile)
    paths = _paths(args.output_dir, args.condition)
    if args.preprocessed_h5ad is not None:
        if args.stage in {"preprocess", "all"}:
            raise ValueError(
                "--preprocessed-h5ad is only valid for train/downstream stages."
            )
        paths["aligned_h5ad"] = _require_file(
            args.preprocessed_h5ad, "preprocessed zebrafish H5AD"
        )
    preprocess_result = None
    downstream_manifest = None
    if args.stage in {"preprocess", "all"}:
        preprocess_result = _run_preprocess(args, paths, defaults)
    if args.stage in {"train", "all"}:
        _run_train(args, paths, args.condition)
    if args.stage in {"downstream", "all"}:
        downstream_manifest = _run_downstream(
            args, paths, args.condition, defaults
        )

    print(
        json.dumps(
            _json_ready(
                {
                    "workflow": "zebrafish_clean_counts_end_to_end",
                    "profile": args.profile,
                    "stage": args.stage,
                    "condition": args.condition,
                    "paths": paths,
                    "preprocess": preprocess_result,
                    "downstream": downstream_manifest,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
