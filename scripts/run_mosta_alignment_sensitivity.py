#!/usr/bin/env python3
"""Compare capped-fit and uncapped-fit MOSTA spatial alignment.

The formal MOSTA workflow may fit the coordinate transformer on a deterministic
per-timepoint subset while applying the learned transform to every cell.  This
script reruns *alignment only* on all cells using the already-fitted PCA latent
coordinates, then quantifies whether the capped fit changes spatial geometry.

No expression preprocessing, PCA fitting, interaction graph construction, or
CytoBridge dynamics training is repeated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree, distance

from CytoBridge.pp import AlignConfig, align_spatial

try:
    import ot
except ImportError:  # pragma: no cover - formal runtime includes POT
    ot = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-h5ad", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-key", default="timepoint")
    parser.add_argument("--processed-time-key", default="time_point_processed")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--ot-sample-per-timepoint", type=int, default=1024)
    parser.add_argument("--plot-sample-per-timepoint", type=int, default=2500)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _restore_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _restore_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_none(item) for item in value]
    if value == "none":
        return None
    return value


def _load_lightweight_alignment_input(
    path: Path,
    *,
    time_key: str,
) -> tuple[ad.AnnData, np.ndarray, dict[str, Any], dict[str, Any], str]:
    source = sc.read_h5ad(path, backed="r")
    try:
        required_obsm = ("X_latent", "spatial_original", "spatial_aligned")
        missing = [key for key in required_obsm if key not in source.obsm]
        if missing:
            raise KeyError(f"Missing required obsm entries: {missing}")
        if time_key not in source.obs:
            raise KeyError(f"Missing time column {time_key!r}")

        obs = source.obs.copy()
        if isinstance(obs[time_key].dtype, pd.CategoricalDtype):
            obs[time_key] = obs[time_key].cat.remove_unused_categories()
        latent = np.asarray(source.obsm["X_latent"], dtype=np.float32)
        spatial_original = np.asarray(
            source.obsm["spatial_original"], dtype=np.float32
        )
        spatial_capped = np.asarray(source.obsm["spatial_aligned"], dtype=np.float32)
        info = _json_safe(source.uns.get("spatial_alignment_info", {}))
        interaction_graph_info = _json_safe(source.uns.get("interaction_graph", {}))
        obs_name_sha = hashlib.sha256(
            "\n".join(map(str, source.obs_names)).encode("utf-8")
        ).hexdigest()
    finally:
        source.file.close()

    n_obs = obs.shape[0]
    light = ad.AnnData(X=sp.csr_matrix((n_obs, 0), dtype=np.float32), obs=obs)
    light.obsm["X_latent"] = latent
    light.obsm["spatial_original"] = spatial_original
    light.obsm["spatial"] = spatial_original.copy()
    return light, spatial_capped, info, interaction_graph_info, obs_name_sha


def _uncapped_config(
    stored_info: dict[str, Any], *, random_seed: int | None
) -> AlignConfig:
    raw = stored_info.get("config")
    if not isinstance(raw, dict):
        raise ValueError(
            "aligned H5AD lacks spatial_alignment_info.config; refusing to guess "
            "the formal alignment configuration"
        )
    raw = _restore_none(raw)
    accepted = {field.name for field in fields(AlignConfig)}
    unknown = sorted(set(raw) - accepted)
    if unknown:
        raise ValueError(f"Stored alignment config has unknown fields: {unknown}")
    config = dict(raw)
    config["input_spatial_key"] = "spatial_original"
    config["spatial_obs_keys"] = None
    config["max_cells_per_timepoint"] = None
    if random_seed is not None:
        config["random_seed"] = int(random_seed)
    return AlignConfig(**config)


def _orthogonal_procrustes(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map candidate onto reference by weighted translation and orthogonal map."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("reference and candidate must have the same 2D shape")
    if reference.shape[0] == 0 or reference.shape[1] != 2:
        raise ValueError("reference and candidate must have shape (n_cells, 2)")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("reference and candidate coordinates must be finite")
    if weights is None:
        normalized_weights = np.full(
            reference.shape[0], 1.0 / reference.shape[0], dtype=np.float64
        )
        weighting = "uniform_cell"
    else:
        normalized_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if normalized_weights.shape[0] != reference.shape[0]:
            raise ValueError("weights must contain one value per coordinate row")
        if (
            not np.isfinite(normalized_weights).all()
            or np.any(normalized_weights < 0)
            or float(normalized_weights.sum()) <= 0
        ):
            raise ValueError("weights must be finite, non-negative, and sum positive")
        normalized_weights = normalized_weights / normalized_weights.sum()
        weighting = "caller_supplied"
    ref_center = np.sum(reference * normalized_weights[:, None], axis=0)
    cand_center = np.sum(candidate * normalized_weights[:, None], axis=0)
    ref0 = reference - ref_center
    cand0 = candidate - cand_center
    cross_covariance = (cand0 * normalized_weights[:, None]).T @ ref0
    u, singular, vt = np.linalg.svd(cross_covariance, full_matrices=False)
    rotation = u @ vt
    aligned = cand0 @ rotation + ref_center
    return aligned, {
        "weighting": weighting,
        "effective_sample_size": float(
            1.0 / np.square(normalized_weights).sum()
        ),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "cross_covariance_nuclear_norm": float(singular.sum()),
        "reference_center_x": float(ref_center[0]),
        "reference_center_y": float(ref_center[1]),
        "candidate_center_x": float(cand_center[0]),
        "candidate_center_y": float(cand_center[1]),
    }


def _equal_group_weights(groups: np.ndarray) -> np.ndarray:
    """Give every group equal total mass and every cell equal mass within group."""
    values = np.asarray(groups)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional array")
    if pd.isna(values).any():
        raise ValueError("groups must not contain missing values")
    codes, uniques = pd.factorize(values, sort=True)
    if len(uniques) == 0 or np.any(codes < 0):
        raise ValueError("groups must contain at least one valid group")
    counts = np.bincount(codes, minlength=len(uniques)).astype(np.float64)
    weights = 1.0 / counts[codes]
    return weights / weights.sum()


def _nn1(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] < 2:
        return np.zeros(coords.shape[0], dtype=np.float64)
    distances, _ = cKDTree(coords).query(coords, k=2, workers=-1)
    return np.asarray(distances[:, 1], dtype=np.float64)


def _knn_indices(coords: np.ndarray, *, k: int) -> np.ndarray:
    """Return exactly k non-self neighbors, without assuming self ranks first."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("coords must be a finite array with shape (n_cells, 2)")
    n_obs = int(coords.shape[0])
    if int(k) <= 0 or n_obs <= int(k):
        raise ValueError(f"Need more than k={k} cells, got {n_obs}")
    _, queried = cKDTree(coords).query(coords, k=int(k) + 1, workers=-1)
    queried = np.asarray(queried, dtype=np.int64)
    rows = np.arange(n_obs, dtype=np.int64)[:, None]
    nonself = queried != rows
    if np.any(nonself.sum(axis=1) < int(k)):
        raise RuntimeError("kNN query did not return enough non-self neighbors")
    ranks = np.cumsum(nonself, axis=1) - 1
    keep = nonself & (ranks < int(k))
    result = np.empty((n_obs, int(k)), dtype=np.int64)
    result[rows.repeat(queried.shape[1], axis=1)[keep], ranks[keep]] = queried[keep]
    if np.any(result == rows) or np.any(np.diff(np.sort(result, axis=1), axis=1) == 0):
        raise RuntimeError("kNN construction produced self or duplicate neighbors")
    return result


def _knn_jaccard(
    reference: np.ndarray, candidate: np.ndarray, *, k: int, chunk_size: int = 20000
) -> np.ndarray:
    n_obs = reference.shape[0]
    if n_obs <= k:
        raise ValueError(f"Need more than k={k} cells, got {n_obs}")
    ref_neighbors = _knn_indices(reference, k=k)
    cand_neighbors = _knn_indices(candidate, k=k)
    scores = np.empty(n_obs, dtype=np.float64)
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        a = ref_neighbors[start:end]
        b = cand_neighbors[start:end]
        intersection = (a[:, :, None] == b[:, None, :]).any(axis=2).sum(axis=1)
        scores[start:end] = intersection / (2 * k - intersection)
    return scores


def _uniform_ot(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    n = min(int(sample_size), reference.shape[0])
    indices = rng.choice(reference.shape[0], size=n, replace=False)
    a = reference[indices].astype(np.float64, copy=False)
    b = candidate[indices].astype(np.float64, copy=False)
    if ot is None:
        return float("nan"), float("nan"), n
    weights = np.full(n, 1.0 / n, dtype=np.float64)
    cost = distance.cdist(a, b, metric="euclidean")
    w1 = float(ot.emd2(weights, weights, cost, numItermax=1_000_000))
    w2 = float(
        np.sqrt(
            max(
                0.0,
                ot.emd2(weights, weights, np.square(cost), numItermax=1_000_000),
            )
        )
    )
    return w1, w2, n


def _summarize_stage(
    stage: str,
    mask: np.ndarray,
    capped: np.ndarray,
    full: np.ndarray,
    *,
    k: int,
    ot_sample: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    cap = capped[mask]
    uncapped = full[mask]
    cap_nn1 = _nn1(cap)
    full_nn1 = _nn1(uncapped)
    displacement = np.linalg.norm(uncapped - cap, axis=1)
    positive_nn = cap_nn1[cap_nn1 > 0]
    floor = np.finfo(np.float64).eps
    if positive_nn.size:
        floor = max(floor, float(np.percentile(positive_nn, 1)) * 1e-3)
    ratio = displacement / np.maximum(cap_nn1, floor)
    jaccard = _knn_jaccard(cap, uncapped, k=k)
    w1, w2, n_ot = _uniform_ot(
        cap, uncapped, sample_size=ot_sample, rng=rng
    )
    row = {
        "timepoint": str(stage),
        "n_cells": int(mask.sum()),
        "capped_zero_nn1_count": int(np.count_nonzero(cap_nn1 <= 0.0)),
        "uncapped_zero_nn1_count": int(np.count_nonzero(full_nn1 <= 0.0)),
        "capped_nn1_median": float(np.median(cap_nn1)),
        "uncapped_nn1_median": float(np.median(full_nn1)),
        "nn1_median_relative_difference": float(
            abs(np.median(full_nn1) - np.median(cap_nn1))
            / max(np.median(cap_nn1), floor)
        ),
        "matched_displacement_median": float(np.median(displacement)),
        "matched_displacement_p95": float(np.percentile(displacement, 95)),
        "matched_displacement_median_over_capped_nn1_median": float(
            np.median(displacement) / max(np.median(cap_nn1), floor)
        ),
        "displacement_over_capped_nn1_median": float(np.median(ratio)),
        "displacement_over_capped_nn1_p95": float(np.percentile(ratio, 95)),
        "knn_jaccard_k": int(k),
        "knn_jaccard_mean": float(np.mean(jaccard)),
        "knn_jaccard_median": float(np.median(jaccard)),
        "knn_jaccard_p05": float(np.percentile(jaccard, 5)),
        "spatial_w1": w1,
        "spatial_w2": w2,
        "spatial_ot_n": int(n_ot),
    }
    return row, ratio, jaccard


def _fresh_threshold(
    stage_rows: pd.DataFrame,
    prefix: str,
    *,
    recommended_spot_scale: float = 1.2,
    neighborhood_factor: float = 4.0,
) -> float:
    recommended_spot_scale = float(recommended_spot_scale)
    neighborhood_factor = float(neighborhood_factor)
    if (
        not np.isfinite(recommended_spot_scale)
        or recommended_spot_scale <= 0
        or not np.isfinite(neighborhood_factor)
        or neighborhood_factor <= 0
    ):
        raise ValueError("threshold scale and factor must be positive and finite")
    medians = stage_rows[f"{prefix}_nn1_median"].to_numpy(dtype=float)
    if medians.size == 0 or not np.isfinite(medians).all():
        raise ValueError("stage NN1 medians must be non-empty and finite")
    return float(neighborhood_factor * np.mean(recommended_spot_scale * medians))


def _stored_threshold_contract(
    interaction_graph_info: dict[str, Any],
) -> tuple[float, float, float]:
    required = (
        "recommended_spot_scale",
        "neighborhood_factor",
        "neighborhood_threshold",
    )
    missing = [key for key in required if key not in interaction_graph_info]
    if missing:
        raise ValueError(
            "aligned H5AD interaction_graph lacks cutoff provenance fields: "
            f"{missing}"
        )
    values = tuple(float(interaction_graph_info[key]) for key in required)
    if not np.isfinite(values).all() or any(value <= 0 for value in values):
        raise ValueError("stored interaction-graph cutoff values must be positive")
    return values


def _acceptance_gates(
    stage_metrics: pd.DataFrame,
    *,
    pooled_jaccard: np.ndarray,
    pooled_displacement_ratio: np.ndarray,
    cutoff_relative_difference: float,
    stored_cutoff_relative_error: float,
) -> dict[str, bool]:
    """Evaluate pooled and per-stage gates so large stages cannot hide failures."""
    return {
        "pooled_median_knn_jaccard_at_least_0.90": bool(
            np.median(pooled_jaccard) >= 0.90
        ),
        "every_timepoint_median_knn_jaccard_at_least_0.90": bool(
            (stage_metrics["knn_jaccard_median"] >= 0.90).all()
        ),
        "cutoff_relative_difference_at_most_0.05": bool(
            float(cutoff_relative_difference) <= 0.05
        ),
        "every_timepoint_nn1_median_relative_difference_at_most_0.05": bool(
            (stage_metrics["nn1_median_relative_difference"] <= 0.05).all()
        ),
        "pooled_median_displacement_over_nn1_at_most_0.5": bool(
            np.median(pooled_displacement_ratio) <= 0.5
        ),
        "every_timepoint_median_displacement_over_nn1_at_most_0.5": bool(
            (stage_metrics["displacement_over_capped_nn1_median"] <= 0.5).all()
        ),
        "recomputed_capped_cutoff_matches_stored_at_most_1e-5": bool(
            float(stored_cutoff_relative_error) <= 1e-5
        ),
        "no_exact_coordinate_duplicates": bool(
            int(stage_metrics["capped_zero_nn1_count"].sum()) == 0
            and int(stage_metrics["uncapped_zero_nn1_count"].sum()) == 0
        ),
    }


def _plot_summary(
    output_dir: Path,
    obs: pd.DataFrame,
    processed_time_key: str,
    capped: np.ndarray,
    uncapped: np.ndarray,
    ratios: dict[str, np.ndarray],
    *,
    plot_sample: int,
    rng: np.random.Generator,
) -> None:
    numeric_time = pd.to_numeric(obs[processed_time_key], errors="raise").to_numpy(
        dtype=float
    )
    times = sorted(np.unique(numeric_time))
    fig, axes = plt.subplots(2, len(times), figsize=(4.0 * len(times), 7.2))
    if len(times) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, time in enumerate(times):
        stage = str(float(time))
        mask = numeric_time == float(time)
        indices = np.flatnonzero(mask)
        n = min(int(plot_sample), indices.size)
        chosen = rng.choice(indices, size=n, replace=False)
        cap = capped[chosen]
        full = uncapped[chosen]
        ax = axes[0, col]
        for p, q in zip(cap[:500], full[:500]):
            ax.plot([p[0], q[0]], [p[1], q[1]], color="#9e9e9e", lw=0.25, alpha=0.25)
        ax.scatter(cap[:, 0], cap[:, 1], s=1.5, c="#2166ac", alpha=0.35, label="20k fit")
        ax.scatter(full[:, 0], full[:, 1], s=1.5, c="#b2182b", alpha=0.35, label="full fit")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"t={stage}: aligned overlay")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.legend(frameon=False, markerscale=5, fontsize=8)

        ax = axes[1, col]
        clipped = np.clip(ratios[stage], 0, np.percentile(ratios[stage], 99))
        ax.hist(clipped, bins=60, color="#4d4d4d", alpha=0.85)
        ax.axvline(np.median(ratios[stage]), color="#b2182b", lw=1.5)
        ax.set_title(
            "displacement / NN1\nmedian=" f"{np.median(ratios[stage]):.3f}"
        )
        ax.set_xlabel("ratio (clipped at p99)")
        ax.set_ylabel("cells")
    fig.suptitle("MOSTA 20k-per-stage fit vs uncapped full-cell alignment", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "alignment_sensitivity_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "alignment_sensitivity_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if args.knn_k <= 0:
        raise ValueError("--knn-k must be positive")
    if args.ot_sample_per_timepoint <= 0:
        raise ValueError("--ot-sample-per-timepoint must be positive")
    if args.plot_sample_per_timepoint <= 0:
        raise ValueError("--plot-sample-per-timepoint must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; pass --overwrite to reuse it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))

    (
        light,
        capped,
        stored_info,
        interaction_graph_info,
        obs_name_sha,
    ) = _load_lightweight_alignment_input(
        args.aligned_h5ad.resolve(), time_key=args.time_key
    )
    if args.processed_time_key not in light.obs:
        raise KeyError(f"Missing processed time column {args.processed_time_key!r}")
    config = _uncapped_config(stored_info, random_seed=args.random_seed)
    original_cap = (
        stored_info.get("config", {}).get("max_cells_per_timepoint")
        if isinstance(stored_info.get("config"), dict)
        else None
    )
    if original_cap in (None, "none"):
        raise ValueError(
            "Input artifact was not produced with capped alignment; sensitivity comparison is inapplicable"
        )

    aligned = align_spatial(
        light,
        time_key=args.time_key,
        cfg=config,
        batch_indices=None,
        device=args.device,
        verbose=True,
        copy_adata=False,
    )
    full_raw = np.asarray(aligned.obsm["spatial_aligned"], dtype=np.float64)
    processed = pd.to_numeric(
        aligned.obs[args.processed_time_key], errors="raise"
    ).to_numpy(dtype=float)
    procrustes_weights = _equal_group_weights(processed)
    full, procrustes = _orthogonal_procrustes(
        capped,
        full_raw,
        weights=procrustes_weights,
    )
    procrustes["weighting"] = "equal_total_mass_per_timepoint"
    procrustes["n_timepoints"] = int(np.unique(processed).size)

    seed = int(config.random_seed)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    ratios: dict[str, np.ndarray] = {}
    jaccards: list[np.ndarray] = []
    displacements: list[np.ndarray] = []
    capped_nn1_all: list[np.ndarray] = []
    for time in sorted(np.unique(processed)):
        mask = processed == time
        stage = str(float(time))
        row, ratio, jaccard = _summarize_stage(
            stage,
            mask,
            capped,
            full,
            k=args.knn_k,
            ot_sample=args.ot_sample_per_timepoint,
            rng=rng,
        )
        rows.append(row)
        ratios[stage] = ratio
        jaccards.append(jaccard)
        displacements.append(np.linalg.norm(full[mask] - capped[mask], axis=1))
        capped_nn1_all.append(_nn1(capped[mask]))

    stage_metrics = pd.DataFrame(rows)
    stage_metrics.to_csv(output_dir / "alignment_sensitivity_by_timepoint.csv", index=False)
    (
        recommended_spot_scale,
        neighborhood_factor,
        stored_capped_cutoff,
    ) = _stored_threshold_contract(interaction_graph_info)
    capped_cutoff = _fresh_threshold(
        stage_metrics,
        "capped",
        recommended_spot_scale=recommended_spot_scale,
        neighborhood_factor=neighborhood_factor,
    )
    uncapped_cutoff = _fresh_threshold(
        stage_metrics,
        "uncapped",
        recommended_spot_scale=recommended_spot_scale,
        neighborhood_factor=neighborhood_factor,
    )
    cutoff_relative_difference = abs(uncapped_cutoff - capped_cutoff) / max(
        capped_cutoff, np.finfo(float).eps
    )
    stored_cutoff_relative_error = abs(stored_capped_cutoff - capped_cutoff) / max(
        stored_capped_cutoff, np.finfo(float).eps
    )
    all_jaccard = np.concatenate(jaccards)
    all_displacement = np.concatenate(displacements)
    all_capped_nn1 = np.concatenate(capped_nn1_all)
    positive = all_capped_nn1[all_capped_nn1 > 0]
    floor = np.finfo(float).eps
    if positive.size:
        floor = max(floor, float(np.percentile(positive, 1)) * 1e-3)
    all_ratio = all_displacement / np.maximum(all_capped_nn1, floor)

    gates = _acceptance_gates(
        stage_metrics,
        pooled_jaccard=all_jaccard,
        pooled_displacement_ratio=all_ratio,
        cutoff_relative_difference=cutoff_relative_difference,
        stored_cutoff_relative_error=stored_cutoff_relative_error,
    )
    summary = {
        "schema_version": 2,
        "input_aligned_h5ad": str(args.aligned_h5ad.resolve()),
        "input_obs_names_sha256": obs_name_sha,
        "n_cells": int(aligned.n_obs),
        "original_fit_cap_per_timepoint": int(original_cap),
        "uncapped_fit_cap_per_timepoint": None,
        "random_seed": seed,
        "procrustes": procrustes,
        "cutoff_contract": {
            "recommended_spot_scale": recommended_spot_scale,
            "neighborhood_factor": neighborhood_factor,
            "formula": "neighborhood_factor * mean(recommended_spot_scale * per_timepoint_median_nn1)",
        },
        "stored_capped_neighborhood_cutoff": stored_capped_cutoff,
        "capped_fresh_neighborhood_cutoff": capped_cutoff,
        "uncapped_fresh_neighborhood_cutoff": uncapped_cutoff,
        "stored_capped_cutoff_relative_error": float(
            stored_cutoff_relative_error
        ),
        "cutoff_relative_difference": float(cutoff_relative_difference),
        "pooled_knn_jaccard_median": float(np.median(all_jaccard)),
        "pooled_knn_jaccard_mean": float(np.mean(all_jaccard)),
        "pooled_displacement_over_nn1_median": float(np.median(all_ratio)),
        "pooled_displacement_over_nn1_p95": float(np.percentile(all_ratio, 95)),
        "worst_timepoint_knn_jaccard_median": float(
            stage_metrics["knn_jaccard_median"].min()
        ),
        "worst_timepoint_nn1_median_relative_difference": float(
            stage_metrics["nn1_median_relative_difference"].max()
        ),
        "worst_timepoint_displacement_over_nn1_median": float(
            stage_metrics["displacement_over_capped_nn1_median"].max()
        ),
        "acceptance_gates": gates,
        "failed_acceptance_gates": sorted(
            name for name, passed in gates.items() if not passed
        ),
        "scientifically_acceptable": bool(all(gates.values())),
        "alignment_config_uncapped": _json_safe(config.__dict__),
        "notes": [
            "PCA and expression preprocessing were not repeated.",
            "Full-fit coordinates were mapped onto capped-fit coordinates by one global orthogonal Procrustes without rescaling; each timepoint had equal total fit weight.",
            "Acceptance requires both pooled and every-timepoint gates so a large stage cannot hide a smaller-stage failure.",
            "Spatial W1/W2 use exact uniform discrete OT on a matched seeded sample per timepoint.",
        ],
    }
    (output_dir / "alignment_sensitivity_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    coordinate_table = pd.DataFrame(
        {
            "obs_name": aligned.obs_names.astype(str),
            args.processed_time_key: processed,
            "capped_fit_x": capped[:, 0],
            "capped_fit_y": capped[:, 1],
            "uncapped_fit_raw_x": full_raw[:, 0],
            "uncapped_fit_raw_y": full_raw[:, 1],
            "uncapped_fit_procrustes_x": full[:, 0],
            "uncapped_fit_procrustes_y": full[:, 1],
        }
    )
    coordinate_table.to_csv(
        output_dir / "full_fit_spatial_coordinates.csv.gz",
        index=False,
        compression="gzip",
    )
    _plot_summary(
        output_dir,
        aligned.obs,
        args.processed_time_key,
        capped,
        full,
        ratios,
        plot_sample=args.plot_sample_per_timepoint,
        rng=rng,
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
