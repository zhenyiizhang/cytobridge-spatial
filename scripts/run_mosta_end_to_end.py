#!/usr/bin/env python3
"""Run corrected-count MOSTA preprocessing, training, and core evaluation.

The dataset adapter owns only MOSTA-specific schema and feature declarations:
raw counts live in ``layers['count']``, PCA is fitted on all eight stages,
E12.5--E15.5 are retained for alignment/model training, and mouse CellChat
ligand/receptor subunits present in the dataset are forced into the PCA feature
mask.  All numerical work is delegated to the public CytoBridge APIs.
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
from scipy import sparse


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

try:  # noqa: E402
    from scripts.preprocess_pipeline import run_preprocessing_pipeline
except ImportError:  # direct ``python scripts/run_mosta_end_to_end.py`` execution
    from preprocess_pipeline import run_preprocessing_pipeline


MOSTA_TIME_MAPPING = {
    "E9.5": -3.0,
    "E10.5": -2.0,
    "E11.5": -1.0,
    "E12.5": 0.0,
    "E13.5": 1.0,
    "E14.5": 2.0,
    "E15.5": 3.0,
    "E16.5": 4.0,
}
TRAINING_TIMES = ("E12.5", "E13.5", "E14.5", "E15.5")
DEFAULT_CONFIG = (
    REPO_ROOT / "CytoBridge" / "configs" / "mosta_spatial_full_alpha_express_0015.yaml"
)


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    return resolved


def _git_revision() -> dict[str, object]:
    try:
        commit = subprocess.run(
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
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def _paths(root: Path) -> dict[str, Path]:
    preprocess_dir = root / "preprocess"
    return {
        "root": root,
        "input_contract": root / "input_contract",
        "preprocess_dir": preprocess_dir,
        "aligned_csv": preprocess_dir / "mosta_aligned.csv",
        "aligned_h5ad": preprocess_dir / "mosta_aligned.h5ad",
        "annotated_csv": preprocess_dir / "mosta_aligned_with_annotation.csv",
        "edge_path": preprocess_dir / "edge_classifier" / "mosta_edge_model.pt",
        "pca_contract": preprocess_dir / "pca_contract",
        "training_dir": root / "training",
        "evaluation_dir": root / "evaluation",
        "downstream_dir": root / "downstream",
        "logs_dir": root / "logs",
        "status_dir": root / "status",
    }


def _profile_defaults(profile: str) -> dict[str, object]:
    if profile == "smoke":
        return {
            "max_input_cells": 64,
            "alignment_max_cells": None,
            "phase1_epochs": 2,
            "phase2_epochs": 1,
            "edge_epochs": 1,
            "edge_max_train_edges": 10000,
            "evaluation_n_samples": 64,
        }
    return {
        "max_input_cells": None,
        "alignment_max_cells": None,
        "phase1_epochs": 10000,
        "phase2_epochs": 500,
        "edge_epochs": 100,
        "edge_max_train_edges": None,
        "evaluation_n_samples": 5000,
    }


def _resolve_lr_columns(table: pd.DataFrame) -> tuple[str, str]:
    lower = {str(column).lower(): str(column) for column in table.columns}
    if "ligand" in lower and "receptor" in lower:
        return lower["ligand"], lower["receptor"]
    if "0" in table.columns and "1" in table.columns:
        return "0", "1"
    raise ValueError(
        "Could not identify ligand/receptor columns. Expected ligand/receptor or 0/1."
    )


def _lr_feature_contract(
    source_h5ad: Path, database_path: Path, output_dir: Path
) -> tuple[tuple[str, ...], dict[str, object]]:
    database = pd.read_csv(database_path)
    ligand_column, receptor_column = _resolve_lr_columns(database)
    requested = set()
    for column in (ligand_column, receptor_column):
        for value in database[column].dropna().astype(str):
            requested.update(part for part in value.split("_") if part)

    source = sc.read_h5ad(source_h5ad, backed="r")
    source_genes = set(map(str, source.var_names))
    present = tuple(sorted(requested.intersection(source_genes)))
    missing = sorted(requested.difference(source_genes))
    required_panel = {"Wnt3a", "Fzd7", "Lrp6"}
    absent_panel = sorted(required_panel.difference(present))
    if absent_panel:
        raise RuntimeError(
            "MOSTA source is missing required Wnt panel subunits: " f"{absent_panel}."
        )
    contract = {
        "database_path": str(database_path),
        "database_sha256": _sha256(database_path),
        "ligand_column": ligand_column,
        "receptor_column": receptor_column,
        "n_database_rows": int(len(database)),
        "n_unique_subunits": int(len(requested)),
        "n_subunits_present_in_source": int(len(present)),
        "n_subunits_missing_from_source": int(len(missing)),
        "missing_subunits": missing,
        "required_focal_panel_subunits": sorted(required_panel),
        "policy": "PCA mask = top HVGs union all database subunits present in source",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lr_feature_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "lr_features_present.txt").write_text(
        "\n".join(present) + "\n", encoding="utf-8"
    )
    return present, contract


def _write_annotation_outputs(aligned_h5ad: Path, annotated_csv: Path) -> None:
    adata = sc.read_h5ad(aligned_h5ad)
    if "annotation" not in adata.obs:
        raise KeyError("MOSTA aligned H5AD is missing obs['annotation'].")
    labels = adata.obs["annotation"].astype(str)
    adata.obs["Annotation"] = labels.to_numpy()
    adata.obs["time_label"] = adata.obs["timepoint"].astype(str).to_numpy()

    categories = (
        list(adata.obs["annotation"].cat.categories.astype(str))
        if isinstance(adata.obs["annotation"].dtype, pd.CategoricalDtype)
        else sorted(labels.unique())
    )
    colors = list(map(str, adata.uns.get("annotation_colors", [])))
    if len(colors) != len(categories):
        import matplotlib.pyplot as plt

        palette = plt.get_cmap("tab20")
        colors = [
            "#{:02x}{:02x}{:02x}".format(
                *tuple((np.asarray(palette(i % 20)[:3]) * 255).astype(int))
            )
            for i in range(len(categories))
        ]
    label_to_color = dict(zip(categories, colors))
    adata.obs["Color"] = labels.map(label_to_color).fillna("#808080").to_numpy()
    adata.uns["cytobridge_label_to_color"] = label_to_color
    adata.write_h5ad(aligned_h5ad)

    frame, _ = adata_to_aligned_dataframe(
        adata,
        time_key="time_point_processed",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        annotation_key="Annotation",
    )
    extra = adata.obs[["Color", "time_label"]].copy()
    extra["cell_id"] = adata.obs_names.astype(str)
    frame = frame.merge(extra, on="cell_id", how="left", validate="1:1")
    annotated_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(annotated_csv, index=False)


def _dense_rows(matrix, rows: np.ndarray) -> np.ndarray:
    values = matrix[rows]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def _write_pca_contract(aligned_h5ad: Path, output_dir: Path) -> dict[str, object]:
    adata = sc.read_h5ad(aligned_h5ad)
    if "PCs" not in adata.varm or "pca_center" not in adata.var:
        raise KeyError("Aligned H5AD lacks PCA loadings or persisted fit center.")
    loadings = np.asarray(adata.varm["PCs"], dtype=np.float32)
    center = adata.var["pca_center"].to_numpy(dtype=np.float32)
    loading_norm = np.linalg.norm(loadings, axis=1)
    active = loading_norm > 1e-10
    required = set(
        map(
            str,
            adata.uns.get("preprocess_info", {}).get(
                "required_latent_features_requested", []
            ),
        )
    )
    inactive_required = sorted(
        required.difference(set(adata.var_names[active].astype(str)))
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    component_columns = [f"PC{i + 1}" for i in range(loadings.shape[1])]
    loadings_table = pd.DataFrame(loadings, columns=component_columns)
    loadings_table.insert(0, "gene", adata.var_names.astype(str))
    loadings_table["loading_norm"] = loading_norm
    loadings_table["active_pca_feature"] = active
    loadings_table["required_latent_feature"] = loadings_table["gene"].isin(required)
    loadings_table.to_csv(output_dir / "pca_loadings.csv", index=False)
    pd.DataFrame(
        {
            "gene": adata.var_names.astype(str),
            "mean": center,
            "active_pca_feature": active,
        }
    ).to_csv(output_dir / "pca_center.csv", index=False)

    rows = np.unique(
        np.linspace(0, adata.n_obs - 1, num=min(256, adata.n_obs), dtype=int)
    )
    expression = _dense_rows(adata.X, rows)[:, active]
    active_loadings = loadings[active]
    active_center = center[active]
    expected_latent = (expression - active_center[None, :]) @ active_loadings
    stored_latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)[rows]
    latent_error = expected_latent - stored_latent
    reconstructed = stored_latent @ active_loadings.T + active_center[None, :]
    inverse_forward = (reconstructed - active_center[None, :]) @ active_loadings
    gene_error = reconstructed - expression
    report = {
        "n_vars": int(adata.n_vars),
        "n_components": int(loadings.shape[1]),
        "n_active_pca_features": int(active.sum()),
        "n_inactive_features": int((~active).sum()),
        "n_required_latent_features": int(len(required)),
        "n_inactive_required_latent_features": int(len(inactive_required)),
        "inactive_required_latent_features": inactive_required,
        "sample_n_cells": int(len(rows)),
        "observed_expression_to_latent_max_abs_error": float(
            np.max(np.abs(latent_error))
        ),
        "observed_expression_to_latent_mae": float(np.mean(np.abs(latent_error))),
        "inverse_forward_latent_max_abs_error": float(
            np.max(np.abs(inverse_forward - stored_latent))
        ),
        "rank50_gene_log1p_rmse": float(np.sqrt(np.mean(np.square(gene_error)))),
        "rank50_gene_log1p_negative_fraction": float((reconstructed < 0).mean()),
        "loading_order_sha256": hashlib.sha256(
            "\n".join(map(str, adata.var_names)).encode("utf-8")
        ).hexdigest(),
    }
    (output_dir / "pca_roundtrip_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _run_preprocess(args, paths: dict[str, Path], defaults: dict[str, object]):
    required_features, lr_contract = _lr_feature_contract(
        args.h5ad_path, args.database_path, paths["input_contract"]
    )
    alignment_cap = (
        args.alignment_max_cells_per_timepoint
        if args.alignment_max_cells_per_timepoint is not None
        else defaults["alignment_max_cells"]
    )
    cfg = AlignConfig(
        n_top_genes=2000,
        n_pcs=50,
        normalization_target_sum=1e4,
        expression_layer="count",
        counts_layer="count",
        raw_count_validation="strict",
        allow_retransform_preprocessed_x=False,
        required_latent_features=required_features,
        time_mapping=MOSTA_TIME_MAPPING,
        spatial_dim=2,
        auto_scale_from_centered_x_max=False,
        center_x=True,
        center_y=False,
        flip_y=True,
        scale_x=0.01,
        scale_y=0.01,
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
        max_cells_per_timepoint=(None if alignment_cap is None else int(alignment_cap)),
    )
    result = run_preprocessing_pipeline(
        data_name="mosta",
        h5ad_path=str(args.h5ad_path),
        time_key="timepoint",
        output_dir=str(paths["preprocess_dir"]),
        align_config=cfg,
        database_path=str(args.database_path),
        batch_indices=[3, 4, 5, 6],
        max_input_cells_per_timepoint=defaults["max_input_cells"],
        edge_epochs=int(defaults["edge_epochs"]),
        edge_max_train_edges_per_epoch=defaults["edge_max_train_edges"],
        edge_predictor_threshold=None,
        neighborhood_threshold=None,
        spatial_key="spatial_aligned",
        device=str(args.device),
    )
    _write_annotation_outputs(paths["aligned_h5ad"], paths["annotated_csv"])
    pca_report = _write_pca_contract(paths["aligned_h5ad"], paths["pca_contract"])
    adata = sc.read_h5ad(paths["aligned_h5ad"], backed="r")
    times = sorted(
        pd.to_numeric(adata.obs["time_point_processed"], errors="raise")
        .unique()
        .tolist()
    )
    if times != [0.0, 1.0, 2.0, 3.0]:
        raise RuntimeError(f"Expected canonical MOSTA model times 0..3, got {times}")
    return {"pipeline": result, "lr_feature_contract": lr_contract, "pca": pca_report}


def _resolved_training_config(args, paths: dict[str, Path]) -> dict:
    config_path = args.training_config or DEFAULT_CONFIG
    config = load_config(str(_require_file(config_path, "MOSTA training config")))
    config["seed"] = int(args.random_seed)
    config["ckpt_dir"] = str(paths["training_dir"])
    config["model"]["spatial_dim"] = 2
    defaults = config["training"]["defaults"]
    defaults["alpha_spatial"] = 10.0
    defaults["alpha_express"] = 0.015
    if args.profile == "smoke":
        for stage in config["training"]["plan"]:
            stage["epochs"] = 1
            stage["batch_size"] = min(int(stage.get("batch_size", 32)), 32)
    return config


def _run_train(args, paths: dict[str, Path]) -> None:
    _require_file(paths["aligned_h5ad"], "corrected MOSTA aligned H5AD")
    _require_file(paths["edge_path"], "corrected MOSTA edge predictor")
    config = _resolved_training_config(args, paths)
    paths["training_dir"].mkdir(parents=True, exist_ok=True)
    launch = {
        "workflow": "mosta_corrected_counts_six_stage",
        "profile": args.profile,
        "alpha_spatial": 10.0,
        "alpha_express": 0.015,
        "requested_device": str(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "git": _git_revision(),
    }
    (paths["training_dir"] / "launch_manifest.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True), encoding="utf-8"
    )
    cb.tl.fit(
        str(paths["aligned_h5ad"]),
        config=config,
        device=str(args.device),
        ckpt_dir=str(paths["training_dir"]),
        evaluate_after_training=False,
    )


def _run_evaluate(args, paths: dict[str, Path], defaults: dict[str, object]):
    adata = sc.read_h5ad(paths["aligned_h5ad"])
    spatial_dim = int(adata.obsm["spatial_aligned"].shape[1])
    latent_dim = int(adata.obsm["X_latent"].shape[1])
    dim = spatial_dim + latent_dim
    loaded = load_dynamical_model_from_dir(
        paths["training_dir"], dim=dim, device=str(args.device)
    )
    interaction = loaded.config["model"]["interaction_net"]
    paths["evaluation_dir"].mkdir(parents=True, exist_ok=True)
    components = compute_velocity_components_from_adata(
        adata,
        loaded.model,
        dim=dim,
        interaction_m=int(args.interaction_m),
        interaction_threshold=float(interaction["cutoff"]),
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
        paths["evaluation_dir"] / "velocity_components.npz", **components
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
        evaluation, paths["evaluation_dir"] / "distribution_evaluation"
    )
    manifest = {
        "workflow": "mosta_corrected_counts_core_evaluation",
        "alpha_spatial": 10.0,
        "alpha_express": 0.015,
        "weight_stage": loaded.weight_stage,
        "score_stage": loaded.score_stage,
        "interaction_cutoff": float(interaction["cutoff"]),
        "edge_predictor_path": str(interaction["edge_predictor_path"]),
        "edge_predictor_threshold": float(interaction["edge_predictor_thre"]),
        "velocity_component_identity_max_error": identity_error,
        "distribution_evaluation_paths": evaluation_paths,
        "distribution_mean_by_space": (
            evaluation.metrics.groupby("space", sort=True)
            .mean(numeric_only=True)
            .reset_index()
            .to_dict(orient="records")
        ),
        "settings": dict(evaluation.settings),
        "git": _git_revision(),
    }
    manifest = _json_ready(manifest)
    (paths["evaluation_dir"] / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
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
        choices=["preprocess", "train", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--alignment-max-cells-per-timepoint", type=int, default=None)
    parser.add_argument("--interaction-m", type=int, default=1024)
    parser.add_argument("--evaluation-max-ot-points", type=int, default=1024)
    parser.add_argument("--evaluation-dt", type=float, default=0.01)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    args.h5ad_path = _require_file(args.h5ad_path, "MOSTA source H5AD")
    args.database_path = _require_file(args.database_path, "mouse LR database")
    if args.training_config is not None:
        args.training_config = _require_file(
            args.training_config, "MOSTA training config"
        )
    args.output_dir = args.output_dir.expanduser().resolve()
    paths = _paths(args.output_dir)
    for key in ("root", "logs_dir", "status_dir", "downstream_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    defaults = _profile_defaults(args.profile)

    result: dict[str, object] = {
        "workflow": "mosta_corrected_counts_end_to_end",
        "profile": args.profile,
        "stage": args.stage,
        "paths": paths,
    }
    if args.stage in {"preprocess", "all"}:
        result["preprocess"] = _run_preprocess(args, paths, defaults)
    if args.stage in {"train", "all"}:
        _run_train(args, paths)
    if args.stage in {"evaluate", "all"}:
        result["evaluation"] = _run_evaluate(args, paths, defaults)
    result["git"] = _git_revision()
    manifest = _json_ready(result)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
