#!/usr/bin/env python3
"""Evaluate current full-cortex models against sealed scNT new-RNA direction.

This command is post-training only. It validates the completed Full and
No-interaction runs before opening the raw ``new`` RNA layer, projects a
one-shot labeling direction through the exact HVG2000 PCA loadings, and
compares that direction with each fitted model's instantaneous inference drift
at the observed states. The metabolic direction never enters optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from CytoBridge.tl.downstream import (
    load_dynamical_model_from_dir,
)
from .grouping import runtime_style_random_groups
from .labeling_velocity import (
    estimate_one_shot_labeling_velocity,
    project_log_velocity_to_pca,
    row_cosine_similarity,
)


EXPECTED_SOURCE_SHAPE = (20_547, 24_078)
EXPECTED_PREPARED_SHAPE = (20_547, 50)
EXPECTED_TIMES = (0.0, 0.25, 0.5, 1.0, 2.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5ad", required=True, type=Path)
    parser.add_argument("--prepared-h5ad", required=True, type=Path)
    parser.add_argument("--pca-artifacts-npz", required=True, type=Path)
    parser.add_argument("--full-run-dir", required=True, type=Path)
    parser.add_argument("--no-interaction-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument(
        "--grouping-seeds", nargs="+", type=int, default=[101, 202, 303, 404, 505]
    )
    parser.add_argument("--labeling-time-hours", type=float, default=2.0)
    parser.add_argument("--target-sum", type=float, default=1.0e4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-batch-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(require_file(path).read_text())
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON mapping")
    return value


def validate_runs(prepared_path: Path, full_run: Path, no_run: Path, seed: int) -> dict:
    prepared_hash = sha256(prepared_path)
    records = {}
    for condition, run_dir in (("full", full_run), ("no_interaction", no_run)):
        manifest_path = require_file(run_dir / "run_manifest.json")
        manifest = read_json(manifest_path)
        if manifest.get("condition") != condition:
            raise ValueError(f"{manifest_path} condition is not {condition}")
        if int(manifest.get("seed", -1)) != int(seed):
            raise ValueError(f"{manifest_path} is not training seed {seed}")
        if bool(manifest.get("smoke", True)):
            raise ValueError(f"{manifest_path} is a smoke run")
        if str(manifest.get("prepared_sha256")) != prepared_hash:
            raise ValueError(f"{manifest_path} was not trained from this prepared H5AD")
        if bool(manifest.get("uses_metabolic_velocity_for_training", True)):
            raise ValueError(f"{manifest_path} reports metabolic leakage into training")
        model_dir = run_dir / "model"
        require_file(model_dir / "adata.h5ad")
        require_file(model_dir / "config.yaml")
        records[condition] = {
            "run_manifest": str(manifest_path),
            "run_manifest_sha256": sha256(manifest_path),
            "model_dir": str(model_dir.resolve()),
        }
    return {"prepared_sha256": prepared_hash, "runs": records}


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def build_direction_reference(
    source_path: Path,
    prepared_path: Path,
    artifacts_path: Path,
    *,
    target_sum: float,
    labeling_time: float,
) -> tuple[ad.AnnData, np.ndarray, np.ndarray, dict]:
    source = ad.read_h5ad(source_path)
    prepared = ad.read_h5ad(prepared_path)
    if tuple(source.shape) != EXPECTED_SOURCE_SHAPE:
        raise ValueError(f"Unexpected source shape {tuple(source.shape)}")
    if tuple(prepared.shape) != EXPECTED_PREPARED_SHAPE:
        raise ValueError(f"Unexpected prepared shape {tuple(prepared.shape)}")
    if not np.array_equal(source.obs_names.to_numpy(), prepared.obs_names.to_numpy()):
        raise ValueError("Source and prepared cell order differ")
    if prepared.layers:
        raise ValueError("Prepared training H5AD contains sealed count layers")
    if set(("new", "total")).difference(source.layers):
        raise ValueError("Source H5AD lacks new or total layers")
    model_times = np.asarray(prepared.obs["time_point_processed"], dtype=float)
    if not np.allclose(np.unique(model_times), EXPECTED_TIMES, rtol=0, atol=1e-8):
        raise ValueError("Unexpected model times")

    gene_symbols = (
        source.var["gene_short_name"].astype(str).to_numpy()
        if "gene_short_name" in source.var.columns
        else source.var_names.astype(str).to_numpy()
    )
    if len(set(gene_symbols.tolist())) != len(gene_symbols):
        raise ValueError("Source gene symbols are not unique")
    with np.load(artifacts_path, allow_pickle=False) as payload:
        required = {"feature_gene_names", "pca_loadings", "pca_mean"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"PCA artifacts lack {sorted(missing)}")
        feature_genes = np.asarray(payload["feature_gene_names"]).astype(str)
        loadings = np.asarray(payload["pca_loadings"], dtype=np.float64)
        pca_mean = np.asarray(payload["pca_mean"], dtype=np.float64)
    lookup = {name: index for index, name in enumerate(gene_symbols)}
    absent = [name for name in feature_genes if name not in lookup]
    if absent:
        raise ValueError(f"PCA genes absent from source: {absent[:5]}")
    indices = np.asarray([lookup[name] for name in feature_genes], dtype=int)
    if loadings.shape != (len(indices), prepared.shape[1]) or pca_mean.shape != (
        len(indices),
    ):
        raise ValueError("PCA artifact dimensions differ from prepared state")

    library_sum = (
        np.asarray(source.layers["total"].sum(axis=1)).reshape(-1).astype(np.float64)
    )
    if np.any(library_sum <= 0):
        raise ValueError("Non-positive total-RNA library")
    scale = float(target_sum) / library_sum
    total = (
        np.asarray(dense(source.layers["total"][:, indices]), dtype=np.float64)
        * scale[:, None]
    )
    new = (
        np.asarray(dense(source.layers["new"][:, indices]), dtype=np.float64)
        * scale[:, None]
    )
    if np.any(new > total + 1.0e-6):
        raise ValueError("Normalized new RNA exceeds normalized total RNA")
    reconstructed = (np.log1p(total) - pca_mean) @ loadings
    latent = np.asarray(prepared.obsm["X_latent"], dtype=np.float64)
    reconstruction_max_abs = float(np.max(np.abs(reconstructed - latent)))
    if reconstruction_max_abs > 1.0e-4:
        raise ValueError(
            f"PCA reconstruction mismatch: max_abs={reconstruction_max_abs:.6g}"
        )

    estimate = estimate_one_shot_labeling_velocity(
        total,
        new,
        baseline_mask=np.isclose(model_times, 0.0),
        labeling_time=float(labeling_time),
    )
    direction = project_log_velocity_to_pca(estimate.velocity_log1p, loadings).astype(
        np.float32
    )
    audit = {
        "source_shape": list(source.shape),
        "prepared_shape": list(prepared.shape),
        "feature_gene_count": len(indices),
        "target_sum": float(target_sum),
        "labeling_time_hours": float(labeling_time),
        "pca_reconstruction_max_abs": reconstruction_max_abs,
        "direction_norm_median": float(np.median(np.linalg.norm(direction, axis=1))),
        "new_rna_used_for_training": False,
    }
    return prepared, model_times, direction, audit


def model_field(
    model,
    latent: np.ndarray,
    model_times: np.ndarray,
    *,
    include_interaction: bool,
    grouping_seeds: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    values = np.asarray(latent, dtype=np.float32)
    times = np.asarray(model_times, dtype=np.float32)
    velocity = np.zeros_like(values)
    score = np.zeros_like(values)
    model.eval()
    for start in range(0, len(values), int(batch_size)):
        stop = min(start + int(batch_size), len(values))
        z = torch.as_tensor(values[start:stop], device=device)
        t = torch.as_tensor(times[start:stop, None], device=device)
        with torch.no_grad():
            velocity[start:stop] = model.predict_velocity(t=t, x=z).cpu().numpy()
        with torch.enable_grad():
            z_score = z.detach().requires_grad_(True)
            _, gradient = model.compute_score(
                t=t.detach(), x=z_score, create_graph=False
            )
            score[start:stop] = gradient.detach().cpu().numpy()

    interaction = np.zeros_like(values)
    grouping_sd_norm = np.zeros(len(values), dtype=np.float32)
    if include_interaction:
        repeats = np.zeros((len(grouping_seeds),) + values.shape, dtype=np.float32)
        group_size = int(getattr(model, "interaction_group_size", 16))
        for repeat_index, grouping_seed in enumerate(grouping_seeds):
            for time_value in EXPECTED_TIMES:
                indices = np.flatnonzero(
                    np.isclose(times, time_value, rtol=0, atol=1e-8)
                )
                groups = runtime_style_random_groups(
                    indices, group_size=group_size, seed=int(grouping_seed)
                )
                for group in groups:
                    z = torch.as_tensor(values[group], device=device)
                    lnw = torch.full(
                        (len(group), 1),
                        -float(np.log(len(group))),
                        dtype=z.dtype,
                        device=device,
                    )
                    t = torch.as_tensor([time_value], dtype=z.dtype, device=device)
                    with torch.no_grad():
                        force = model.interaction_net(z, lnw, t)
                    repeats[repeat_index, group] = force.detach().cpu().numpy()
        interaction = repeats.mean(axis=0)
        grouping_sd_norm = np.linalg.norm(repeats.std(axis=0), axis=1).astype(
            np.float32
        )
    return {
        "velocity": velocity,
        "score": score,
        "interaction": interaction,
        "inference_drift": velocity + score + interaction,
        "grouping_sd_norm": grouping_sd_norm,
    }


def alignment_tables(
    *,
    seed: int,
    condition: str,
    obs_names,
    times: np.ndarray,
    reference: np.ndarray,
    field: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = ~np.isclose(times, 0.0)
    drift = np.asarray(field["inference_drift"])
    cosine = row_cosine_similarity(drift[selected], reference[selected])
    cellwise = pd.DataFrame(
        {
            "training_seed": seed,
            "condition": condition,
            "cell_id": np.asarray(obs_names, dtype=object)[selected],
            "time_hours": times[selected],
            "cosine_inference_drift_vs_scnt": cosine,
            "scnt_direction_norm": np.linalg.norm(reference[selected], axis=1),
            "inference_drift_norm": np.linalg.norm(drift[selected], axis=1),
            "velocity_norm": np.linalg.norm(
                np.asarray(field["velocity"])[selected], axis=1
            ),
            "score_norm": np.linalg.norm(np.asarray(field["score"])[selected], axis=1),
            "interaction_norm": np.linalg.norm(
                np.asarray(field["interaction"])[selected], axis=1
            ),
            "interaction_grouping_sd_norm": np.asarray(field["grouping_sd_norm"])[
                selected
            ],
        }
    )
    rows = []
    for time_value in EXPECTED_TIMES[1:]:
        mask = np.isclose(times, time_value, rtol=0, atol=1e-8)
        values = (
            cellwise.loc[
                np.isclose(cellwise["time_hours"], time_value),
                "cosine_inference_drift_vs_scnt",
            ]
            .dropna()
            .to_numpy()
        )
        centroid = row_cosine_similarity(
            drift[mask].mean(axis=0, keepdims=True),
            reference[mask].mean(axis=0, keepdims=True),
        )[0]
        rows.append(
            {
                "training_seed": seed,
                "condition": condition,
                "time_hours": float(time_value),
                "n_cells": int(mask.sum()),
                "centroid_cosine_inference_drift_vs_scnt": float(centroid),
                "cell_cosine_mean": float(np.mean(values)),
                "cell_cosine_median": float(np.median(values)),
                "cell_cosine_sd": float(np.std(values, ddof=1)),
            }
        )
    return cellwise, pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = require_file(args.source_h5ad)
    prepared = require_file(args.prepared_h5ad)
    artifacts = require_file(args.pca_artifacts_npz)
    full_run = args.full_run_dir.expanduser().resolve()
    no_run = args.no_interaction_run_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing non-empty output directory {output}")
    output.mkdir(parents=True, exist_ok=True)

    run_audit = validate_runs(prepared, full_run, no_run, int(args.training_seed))
    prepared_adata, times, reference, reference_audit = build_direction_reference(
        source,
        prepared,
        artifacts,
        target_sum=float(args.target_sum),
        labeling_time=float(args.labeling_time_hours),
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    latent = np.asarray(prepared_adata.obsm["X_latent"], dtype=np.float32)

    cell_frames = []
    time_frames = []
    model_audit = []
    for condition, label, run_dir in (
        ("full", "full_interaction_noise", full_run),
        ("no_interaction", "no_interaction_noise", no_run),
    ):
        loaded = load_dynamical_model_from_dir(
            run_dir / "model", dim=latent.shape[1], device=device, stage="Finetune"
        )
        field = model_field(
            loaded.model,
            latent,
            times,
            include_interaction=condition == "full",
            grouping_seeds=tuple(args.grouping_seeds),
            batch_size=int(args.model_batch_size),
            device=device,
        )
        cells, timewise = alignment_tables(
            seed=int(args.training_seed),
            condition=label,
            obs_names=prepared_adata.obs_names.astype(str),
            times=times,
            reference=reference,
            field=field,
        )
        cell_frames.append(cells)
        time_frames.append(timewise)
        model_audit.append(
            {
                "condition": condition,
                "weight_stage": loaded.weight_stage,
                "weight_path": str(loaded.weight_path.resolve()),
                "weight_sha256": sha256(loaded.weight_path),
                "score_stage": loaded.score_stage,
                "score_path": str(loaded.score_path.resolve())
                if loaded.score_path
                else None,
                "score_sha256": sha256(loaded.score_path)
                if loaded.score_path
                else None,
            }
        )
        del loaded, field
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cellwise = pd.concat(cell_frames, ignore_index=True)
    timewise = pd.concat(time_frames, ignore_index=True)
    pivot = timewise.pivot(
        index="time_hours",
        columns="condition",
        values=["cell_cosine_mean", "cell_cosine_median"],
    ).reset_index()
    paired = pd.DataFrame(
        {
            "time_hours": pivot["time_hours"],
            "mean_full_minus_no_interaction": pivot[
                ("cell_cosine_mean", "full_interaction_noise")
            ]
            - pivot[("cell_cosine_mean", "no_interaction_noise")],
            "median_full_minus_no_interaction": pivot[
                ("cell_cosine_median", "full_interaction_noise")
            ]
            - pivot[("cell_cosine_median", "no_interaction_noise")],
        }
    )
    cell_path = output / "cellwise_scnt_direction_alignment.csv.gz"
    time_path = output / "timewise_scnt_direction_alignment.csv"
    paired_path = output / "paired_scnt_direction_deltas.csv"
    cellwise.to_csv(cell_path, index=False, compression="gzip")
    timewise.to_csv(time_path, index=False)
    paired.to_csv(paired_path, index=False)
    conclusion = {
        "mean_cellwise_cosine_full": float(
            timewise.loc[
                timewise.condition == "full_interaction_noise", "cell_cosine_mean"
            ].mean()
        ),
        "mean_cellwise_cosine_no_interaction": float(
            timewise.loc[
                timewise.condition == "no_interaction_noise", "cell_cosine_mean"
            ].mean()
        ),
        "median_cellwise_cosine_full": float(
            timewise.loc[
                timewise.condition == "full_interaction_noise", "cell_cosine_median"
            ].mean()
        ),
        "median_cellwise_cosine_no_interaction": float(
            timewise.loc[
                timewise.condition == "no_interaction_noise", "cell_cosine_median"
            ].mean()
        ),
        "mean_endpoint_win_count": int(
            (paired["mean_full_minus_no_interaction"] > 0).sum()
        ),
        "median_endpoint_win_count": int(
            (paired["median_full_minus_no_interaction"] > 0).sum()
        ),
        "endpoint_count": int(len(paired)),
        "inferential_scope": "one paired computational training seed; descriptive, not a significance test",
    }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "sealed post-training one-shot scNT new-RNA direction alignment",
        "reference_model": "new = alpha/gamma * (1-exp(-gamma*tau)); baseline cells estimate gene-specific gamma",
        "inference_drift": {
            "full_interaction_noise": "velocity + score_gradient + LR-GNN interaction",
            "no_interaction_noise": "velocity + score_gradient",
        },
        "training_seed": int(args.training_seed),
        "grouping_seeds": list(map(int, args.grouping_seeds)),
        "run_audit": run_audit,
        "reference_audit": reference_audit,
        "model_load_audit": model_audit,
        "conclusion": conclusion,
        "inputs": {
            "source_h5ad": {"path": str(source), "sha256": sha256(source)},
            "prepared_h5ad": {"path": str(prepared), "sha256": sha256(prepared)},
            "pca_artifacts_npz": {"path": str(artifacts), "sha256": sha256(artifacts)},
        },
        "artifacts": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (cell_path, time_path, paired_path)
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
        },
        "new_rna_used_for_training": False,
    }
    manifest_path = output / "scnt_direction_evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(conclusion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
