#!/usr/bin/env python3
"""Run the canonical ARISTA H5AD -> preprocessing -> training -> downstream workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from CytoBridge.pl import plot_velocity_component  # noqa: E402
from CytoBridge.pp import AlignConfig  # noqa: E402
from CytoBridge.tl import (  # noqa: E402
    compute_velocity_components_from_adata,
    evaluate_model_distributions,
    load_dynamical_model_from_dir,
    load_label_to_color,
    save_distribution_evaluation,
)
from CytoBridge.utils.config import load_config  # noqa: E402
from preprocess_pipeline import run_preprocessing_pipeline  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _profile_defaults(profile: str) -> dict:
    if profile == "smoke":
        return {
            "max_cells": 128,
            "phase1_epochs": 20,
            "phase2_epochs": 5,
            "edge_epochs": 2,
            "training_config": REPO_ROOT
            / "CytoBridge"
            / "configs"
            / "arista_spatial_smoke.yaml",
        }
    return {
        "max_cells": None,
        "phase1_epochs": 10000,
        "phase2_epochs": 500,
        "edge_epochs": 100,
        "training_config": REPO_ROOT
        / "CytoBridge"
        / "configs"
        / "arista_spatial_full.yaml",
    }


def _resolved_paths(output_dir: Path) -> dict[str, Path]:
    preprocess_dir = output_dir / "preprocess"
    edge_path = preprocess_dir / "edge_classifier" / "arista_edge_model.pt"
    return {
        "preprocess_dir": preprocess_dir,
        "aligned_csv": preprocess_dir / "arista_aligned.csv",
        "aligned_h5ad": preprocess_dir / "arista_aligned.h5ad",
        "edge_path": edge_path,
        "edge_meta": Path(f"{edge_path}.meta.json"),
        "training_dir": output_dir / "training",
        "downstream_dir": output_dir / "downstream",
    }


def _run_preprocess(args, paths: dict[str, Path], defaults: dict) -> dict:
    max_cells = (
        defaults["max_cells"]
        if args.max_cells_per_timepoint is None
        else args.max_cells_per_timepoint
    )
    phase1_epochs = (
        defaults["phase1_epochs"] if args.phase1_epochs is None else args.phase1_epochs
    )
    phase2_epochs = (
        defaults["phase2_epochs"] if args.phase2_epochs is None else args.phase2_epochs
    )
    edge_epochs = (
        defaults["edge_epochs"] if args.edge_epochs is None else args.edge_epochs
    )
    published_thresholds = args.threshold_policy == "published"
    neighborhood_threshold = (
        args.neighborhood_threshold
        if args.neighborhood_threshold is not None
        else 0.05
        if published_thresholds
        else None
    )
    edge_predictor_threshold = (
        args.edge_predictor_threshold
        if args.edge_predictor_threshold is not None
        else 0.45
        if published_thresholds
        else None
    )
    expression_layer_text = str(args.expression_layer).strip()
    expression_layer = (
        None
        if expression_layer_text.lower() in {"x", "none", "null"}
        else expression_layer_text
    )
    config = AlignConfig(
        n_top_genes=2000,
        n_pcs=50,
        normalization_target_sum=None,
        spatial_dim=2,
        shared_scale=3000.0,
        center_x=True,
        center_y=False,
        phase1_epochs=int(phase1_epochs),
        phase2_epochs=int(phase2_epochs),
        alpha=5.0,
        beta=0.01,
        lambda_local=30.0,
        lambda_ot=1.0,
        batch_size=1024,
        distance_pairs=10000,
        learning_rate=1e-3,
        random_seed=int(args.random_seed),
        expression_layer=expression_layer,
        allow_retransform_preprocessed_x=bool(args.allow_retransform_preprocessed_x),
    )
    return run_preprocessing_pipeline(
        data_name="arista",
        h5ad_path=str(args.h5ad_path),
        time_key="time",
        output_dir=str(paths["preprocess_dir"]),
        align_config=config,
        database_path=str(args.database_path),
        edge_epochs=int(edge_epochs),
        edge_predictor_threshold=edge_predictor_threshold,
        max_input_cells_per_timepoint=max_cells,
        strip_uns_arrays_larger_than_mb=float(args.strip_uns_arrays_larger_than_mb),
        neighborhood_threshold=neighborhood_threshold,
        device=str(args.device),
    )


def _load_edge_meta(paths: dict[str, Path]) -> dict:
    edge_meta = _require_file(paths["edge_meta"], "edge-predictor metadata")
    return json.loads(edge_meta.read_text(encoding="utf-8"))


def _run_train(args, paths: dict[str, Path], defaults: dict) -> None:
    aligned_h5ad = _require_file(paths["aligned_h5ad"], "aligned ARISTA H5AD")
    _require_file(paths["edge_path"], "ARISTA edge predictor")
    _load_edge_meta(paths)
    training_config = (
        Path(args.training_config).expanduser().resolve()
        if args.training_config is not None
        else Path(defaults["training_config"]).resolve()
    )
    config = load_config(str(_require_file(training_config, "training config")))
    config["seed"] = int(args.random_seed)
    config["ckpt_dir"] = str(paths["training_dir"])
    config["model"]["spatial_dim"] = 2
    cb.tl.fit(
        str(aligned_h5ad),
        config=config,
        device=str(args.device),
        ckpt_dir=str(paths["training_dir"]),
        interaction_cutoff=args.training_interaction_cutoff,
        edge_predictor_path=(
            str(args.training_edge_predictor_path.expanduser().resolve())
            if args.training_edge_predictor_path is not None
            else None
        ),
        edge_predictor_threshold=args.training_edge_predictor_threshold,
        evaluate_after_training=False,
    )


def _component_summary(components: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in ("drift", "interaction", "score", "full"):
        values = np.asarray(components[name], dtype=np.float32)
        norms = np.linalg.norm(values, axis=1)
        rows.append(
            {
                "component": name,
                "n_cells": int(values.shape[0]),
                "dim": int(values.shape[1]),
                "all_finite": bool(np.isfinite(values).all()),
                "mean_norm": float(norms.mean()),
                "median_norm": float(np.median(norms)),
                "max_norm": float(norms.max()),
            }
        )
    return pd.DataFrame(rows)


def _run_downstream(args, paths: dict[str, Path]) -> dict:
    aligned_h5ad = _require_file(paths["aligned_h5ad"], "aligned ARISTA H5AD")
    edge_path = _require_file(paths["edge_path"], "ARISTA edge predictor")
    training_dir = paths["training_dir"].resolve()
    training_config_path = _require_file(
        training_dir / "config.yaml",
        "resolved training config",
    )
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
    expected_score_stage = (
        configured_score_stages[-1] if configured_score_stages else None
    )
    if expected_score_stage is not None and loaded.score_stage != expected_score_stage:
        raise RuntimeError(
            "Downstream score checkpoint mismatch: "
            f"expected final configured stage {expected_score_stage!r}, "
            f"loaded {loaded.score_stage!r}."
        )
    interaction_config = loaded.config["model"]["interaction_net"]
    cutoff = float(interaction_config["cutoff"])
    edge_meta = _load_edge_meta(paths)
    components = compute_velocity_components_from_adata(
        adata,
        loaded.model,
        dim=dim,
        interaction_m=int(args.interaction_m),
        interaction_threshold=cutoff,
        device=str(args.device),
        time_key="time_point_processed",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        write_to_adata=True,
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
    labels_all = adata.obs["Annotation"].astype(str).to_numpy()
    label_to_color = load_label_to_color(
        labels_all,
        color_h5ad=str(aligned_h5ad),
        annotation_key="Annotation",
    )
    mask = np.isclose(components["times"], float(args.target_timepoint))
    if not np.any(mask):
        raise ValueError(f"No cells at target timepoint {args.target_timepoint}.")
    coords = np.asarray(adata.obsm["spatial_aligned"])[mask]
    labels = labels_all[mask]
    panels = {
        "spatial_intrinsic": components["drift"][mask, :2],
        "spatial_interaction": components["interaction"][mask, :2],
        "spatial_full": components["full"][mask, :2],
        "gene_intrinsic": components["drift"][mask, 2:4],
        "gene_interaction": components["interaction"][mask, 2:4],
        "gene_full": components["full"][mask, 2:4],
    }
    figures = {}
    fallbacks = {}
    time_tag = f"{float(args.target_timepoint):g}".replace(".", "p")
    for name, velocity in panels.items():
        figure_path = output_dir / f"velocity_{name}_t{time_tag}.svg"
        panel_adata = plot_velocity_component(
            coords=coords,
            velocity=velocity,
            labels=labels,
            label_to_color=label_to_color,
            title=f"ARISTA {name.replace('_', ' ')} (t={args.target_timepoint:g})",
            out_path=str(figure_path),
            density=float(args.density),
            show_legend=False,
        )
        figures[name] = str(figure_path)
        fallbacks[name] = panel_adata.uns.get("velocity_plot_fallback")

    evaluation_n_samples = (
        int(args.evaluation_n_samples)
        if args.evaluation_n_samples is not None
        else 128
        if args.profile == "smoke"
        else 5000
    )
    evaluation = evaluate_model_distributions(
        adata,
        loaded.model,
        n_samples=evaluation_n_samples,
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
    summary_columns = [
        name
        for name in (
            "w1",
            "w2",
            "tmv",
            "nn_dispersion_ratio",
            "support_recall_at_observed_q95",
            "support_precision_at_observed_q95",
            "clump_fraction_at_0_1_observed_nn",
        )
        if name in evaluation.metrics.columns
    ]
    evaluation_summary = (
        evaluation.metrics.groupby("space", sort=True)[summary_columns]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )

    _component_summary(components).to_csv(
        output_dir / "velocity_component_summary.csv",
        index=False,
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
    manifest = {
        "git": _git_revision(),
        "workflow": "arista_end_to_end",
        "input_h5ad": str(args.h5ad_path),
        "input_h5ad_sha256": _sha256(Path(args.h5ad_path)),
        "aligned_h5ad": str(aligned_h5ad),
        "aligned_h5ad_sha256": _sha256(aligned_h5ad),
        "edge_predictor": str(edge_path),
        "edge_predictor_sha256": _sha256(edge_path),
        "training_dir": str(training_dir),
        "resolved_training_config": str(training_config_path),
        "resolved_training_config_sha256": _sha256(training_config_path),
        "training_seed": loaded.config.get("seed"),
        "training_plan": [
            {
                "name": stage.get("name"),
                "mode": stage.get("mode"),
                "epochs": stage.get("epochs"),
                "save_strategy": stage.get("save_strategy", "best"),
                "checkpoint_metric": stage.get("checkpoint_metric"),
                "scheduler_type": stage.get("scheduler_type"),
                "scheduler_metric": stage.get("scheduler_metric"),
                "scheduler_step_before_reverse": stage.get(
                    "scheduler_step_before_reverse"
                ),
                "max_grad_norm": stage.get("max_grad_norm"),
            }
            for stage in loaded.config.get("training", {}).get("plan", [])
        ],
        "cytobridge_module": str(Path(cb.__file__).resolve()),
        "model_class": type(loaded.model).__name__,
        "weight_stage": loaded.weight_stage,
        "weight_checkpoint": str(loaded.weight_path),
        "weight_checkpoint_sha256": _sha256(loaded.weight_path),
        "score_stage": loaded.score_stage,
        "score_checkpoint": (
            None if loaded.score_path is None else str(loaded.score_path)
        ),
        "score_checkpoint_sha256": (
            None if loaded.score_path is None else _sha256(loaded.score_path)
        ),
        "expected_score_stage": expected_score_stage,
        "components": list(loaded.model.components),
        "n_cells": int(adata.n_obs),
        "dim": int(dim),
        "preprocess_expression_source": adata.uns.get("preprocess_info", {}).get(
            "expression_source"
        ),
        "preprocess_input_x_state_detected": adata.uns.get("preprocess_info", {}).get(
            "input_x_state_detected"
        ),
        "time_counts": {
            str(key): int(value)
            for key, value in pd.Series(components["times"])
            .value_counts()
            .sort_index()
            .items()
        },
        "interaction_cutoff": cutoff,
        "rbf_trainable": bool(interaction_config.get("rbf_trainable", False)),
        "edge_predictor_threshold": float(
            interaction_config["edge_predictor_thre"]
        ),
        "edge_predictor_threshold_selected": float(
            edge_meta["edge_predictor_threshold_selected"]
        ),
        "edge_predictor_threshold_source": str(edge_meta["selection_source"]),
        "full_identity_max_error": identity_error,
        "all_finite": bool(
            all(
                np.isfinite(components[name]).all()
                for name in ("drift", "interaction", "score", "full")
            )
        ),
        "plot_fallbacks": fallbacks,
        "figures": figures,
        "distribution_evaluation": {
            "paths": evaluation_paths,
            "settings": dict(evaluation.settings),
            "mean_by_space": evaluation_summary,
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad-path", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "train", "downstream", "all"],
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-cells-per-timepoint", type=int, default=None)
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--phase2-epochs", type=int, default=None)
    parser.add_argument("--edge-epochs", type=int, default=None)
    parser.add_argument(
        "--expression-layer",
        default="counts",
        help=(
            "AnnData layer copied into X before normalize/log1p. ARISTA defaults "
            "to layers['counts'] because the source H5AD X is already log1p. "
            "Pass 'X' together with --allow-retransform-preprocessed-x only for "
            "the historical double-transform replay."
        ),
    )
    parser.add_argument(
        "--allow-retransform-preprocessed-x",
        action="store_true",
        help="Permit a labelled legacy replay that normalizes/log1p-transforms an already transformed X.",
    )
    parser.add_argument(
        "--threshold-policy",
        choices=["preprocess", "published"],
        default="preprocess",
        help=(
            "Use thresholds calibrated by preprocessing (default), or the "
            "published ARISTA 0.05/0.45 thresholds."
        ),
    )
    parser.add_argument("--neighborhood-threshold", type=float, default=None)
    parser.add_argument("--edge-predictor-threshold", type=float, default=None)
    parser.add_argument(
        "--training-interaction-cutoff",
        type=float,
        default=None,
        help=(
            "Explicit fit-time spatial cutoff. This has higher priority than "
            "the value stored by preprocessing and is intended for paired controls."
        ),
    )
    parser.add_argument(
        "--training-edge-predictor-threshold",
        type=float,
        default=None,
        help=(
            "Explicit fit-time edge decision threshold for paired controls; "
            "does not retrain the edge classifier."
        ),
    )
    parser.add_argument(
        "--training-edge-predictor-path",
        type=Path,
        default=None,
        help="Explicit fit-time edge-classifier weights for a frozen-edge control.",
    )
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument("--strip-uns-arrays-larger-than-mb", type=float, default=10.0)
    parser.add_argument("--target-timepoint", type=float, default=1.0)
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--density", type=float, default=2.0)
    parser.add_argument("--evaluation-n-samples", type=int, default=None)
    parser.add_argument("--evaluation-max-ot-points", type=int, default=1024)
    parser.add_argument("--evaluation-dt", type=float, default=0.01)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    args.h5ad_path = _require_file(args.h5ad_path, "ARISTA source H5AD")
    args.database_path = _require_file(args.database_path, "ligand-receptor database")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    defaults = _profile_defaults(args.profile)
    paths = _resolved_paths(args.output_dir)

    if args.stage in {"preprocess", "all"}:
        _run_preprocess(args, paths, defaults)
    if args.stage in {"train", "all"}:
        _run_train(args, paths, defaults)
    manifest = None
    if args.stage in {"downstream", "all"}:
        manifest = _run_downstream(args, paths)

    payload = {
        "git": _git_revision(),
        "profile": args.profile,
        "stage": args.stage,
        "threshold_policy": args.threshold_policy,
        "expression_layer": args.expression_layer,
        "allow_retransform_preprocessed_x": bool(args.allow_retransform_preprocessed_x),
        "paths": {key: str(value) for key, value in paths.items()},
        "downstream_manifest": manifest,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
