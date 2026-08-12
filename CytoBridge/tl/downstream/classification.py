"""Cell-type classification for downstream trajectory labeling (AnnData-first)."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha1
import json
import os
from pathlib import Path
import re
import time
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "build_cached_classifier_inputs_from_adata",
    "LoadedClassifierCache",
    "train_mlp_classifier",
    "train_mlp_classifier_from_adata",
    "train_cached_mlp_classifier_from_adata",
    "load_cached_mlp_classifier",
    "predict_cached_mlp_classifier_from_adata",
    "smooth_spatial_labels",
    "SpatialSmoothingSelection",
    "select_spatial_smoothing_k",
    "analyze_spatial_label_sensitivity",
    "predict_labels_for_points",
    "predict_labels_for_trajectories",
    "MLP",
]


def _validate_spatial_smoothing_inputs(
    labels: Sequence[str],
    spatial_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return defensive arrays for the pure spatial-smoothing utilities."""

    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("labels must be a one-dimensional sequence.")

    coords = np.asarray(spatial_coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("spatial_coords must be a two-dimensional array.")
    if coords.shape[0] != label_array.shape[0]:
        raise ValueError(
            "labels and spatial_coords must contain the same number of rows "
            f"({label_array.shape[0]} != {coords.shape[0]})."
        )
    if coords.shape[1] == 0:
        raise ValueError("spatial_coords must contain at least one coordinate column.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("spatial_coords must contain only finite values.")
    return label_array.copy(), coords.copy()


def _normalize_requested_k(k: int) -> int:
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer, not a boolean.")
    requested = int(k)
    if float(requested) != float(k):
        raise ValueError("k must be an integer.")
    return requested


def _stable_spatial_neighbors(
    spatial_coords: np.ndarray,
    *,
    k: int,
    include_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact neighbors with explicit self and index-stable boundary ties."""
    from scipy.spatial import cKDTree

    coords = np.asarray(spatial_coords, dtype=np.float64)
    n_samples = int(coords.shape[0])
    requested_other = int(k) - (1 if include_self else 0)
    requested_other = max(0, requested_other)
    other_indices = np.empty((n_samples, requested_other), dtype=np.int64)
    other_distances = np.empty((n_samples, requested_other), dtype=np.float64)
    if requested_other:
        tree = cKDTree(coords)
        query_k = min(n_samples, requested_other + 1)
        initial_distances, initial_indices = tree.query(
            coords,
            k=query_k,
            workers=1,
        )
        if query_k == 1:
            initial_distances = np.asarray(initial_distances)[:, None]
            initial_indices = np.asarray(initial_indices)[:, None]
        for row in range(n_samples):
            keep = initial_indices[row] != row
            candidates = initial_indices[row, keep].astype(np.int64, copy=False)
            candidate_distances = initial_distances[row, keep]
            if candidates.size < requested_other:  # pragma: no cover - defensive
                raise RuntimeError("Could not resolve enough non-self spatial neighbors.")
            order = np.lexsort((candidates, candidate_distances))
            cutoff = float(candidate_distances[order[requested_other - 1]])
            candidates = np.asarray(
                [
                    index
                    for index in tree.query_ball_point(
                        coords[row], np.nextafter(cutoff, np.inf)
                    )
                    if index != row
                ],
                dtype=np.int64,
            )
            candidate_distances = np.linalg.norm(
                coords[candidates] - coords[row],
                axis=1,
            )
            order = np.lexsort((candidates, candidate_distances))[:requested_other]
            other_indices[row] = candidates[order]
            other_distances[row] = candidate_distances[order]

    if include_self:
        indices = np.column_stack(
            (np.arange(n_samples, dtype=np.int64), other_indices)
        )
        distances = np.column_stack(
            (np.zeros(n_samples, dtype=np.float64), other_distances)
        )
        return distances, indices
    return other_distances, other_indices


def _smooth_spatial_labels_detailed(
    labels: Sequence[str],
    spatial_coords: np.ndarray,
    *,
    k: int = 10,
    include_self: bool = True,
    weights: str = "uniform",
    tie_policy: str = "sklearn_legacy",
) -> tuple[np.ndarray, dict]:
    """Implementation shared by smoothing and reviewer sensitivity reporting."""

    label_array, coords = _validate_spatial_smoothing_inputs(labels, spatial_coords)
    requested_k = _normalize_requested_k(k)
    weights_normalized = str(weights).strip().lower()
    tie_policy_normalized = str(tie_policy).strip().lower()
    if weights_normalized not in {"uniform", "distance"}:
        raise ValueError("weights must be one of {'uniform', 'distance'}.")
    if tie_policy_normalized != "sklearn_legacy":
        raise ValueError("tie_policy currently supports only 'sklearn_legacy'.")

    n_samples = int(label_array.shape[0])
    maximum_k = n_samples if include_self else max(0, n_samples - 1)
    effective_k = min(maximum_k, max(0, requested_k))
    metadata = {
        "requested_k": int(requested_k),
        "effective_k": int(effective_k),
        "include_self": bool(include_self),
        "weights": weights_normalized,
        "tie_policy": tie_policy_normalized,
        "neighbor_algorithm": "scipy.spatial.cKDTree_exact_boundary_ties",
        "self_inclusion_contract": (
            "forced_query_row" if include_self else "excluded_query_row"
        ),
        "even_effective_k": bool(effective_k > 0 and effective_k % 2 == 0),
        "n_vote_ties": 0,
    }

    # Historical classifier code bypassed k-NN when k <= 1. Keeping that
    # behavior also makes k=1 a transparent raw-prediction reference point.
    if requested_k <= 1 or effective_k <= 1 or n_samples <= 1:
        return label_array.copy(), metadata

    # np.unique supplies the same ordered-class tie resolution used by the
    # historical sklearn KNeighborsClassifier: the lowest sorted class wins.
    try:
        classes, encoded = np.unique(label_array, return_inverse=True)
    except TypeError as exc:
        raise TypeError(
            "labels must have one mutually comparable dtype for "
            "tie_policy='sklearn_legacy'."
        ) from exc

    distances, neighbor_indices = _stable_spatial_neighbors(
        coords,
        k=effective_k,
        include_self=include_self,
    )

    refined_encoded = np.empty(n_samples, dtype=np.int64)
    n_ties = 0
    for row in range(n_samples):
        row_indices = neighbor_indices[row]
        row_distances = distances[row]
        if row_indices.shape[0] < effective_k:
            # This is only reachable for pathological neighbor backends. Fail
            # closed instead of silently changing the requested vote size.
            raise RuntimeError(
                f"Could only resolve {row_indices.shape[0]} of {effective_k} neighbors."
            )

        if weights_normalized == "uniform":
            vote_weights = np.ones(effective_k, dtype=np.float64)
        else:
            zero_distance = row_distances == 0.0
            if np.any(zero_distance):
                vote_weights = zero_distance.astype(np.float64)
            else:
                vote_weights = 1.0 / row_distances

        scores = np.bincount(
            encoded[row_indices],
            weights=vote_weights,
            minlength=len(classes),
        )
        winners = np.flatnonzero(
            np.isclose(scores, scores.max(), rtol=1e-12, atol=1e-15)
        )
        if winners.shape[0] > 1:
            n_ties += 1
        refined_encoded[row] = int(winners[0])

    metadata["n_vote_ties"] = int(n_ties)
    return np.asarray(classes[refined_encoded]).copy(), metadata


def smooth_spatial_labels(
    labels: Sequence[str],
    spatial_coords: np.ndarray,
    *,
    k: int = 10,
    include_self: bool = True,
    weights: str = "uniform",
    tie_policy: str = "sklearn_legacy",
) -> np.ndarray:
    """Smooth precomputed labels by an explicit spatial k-nearest-neighbor vote.

    This function is deliberately independent of classifier feature order. It
    never mutates ``labels`` or ``spatial_coords`` and returns raw labels
    unchanged for ``k <= 1``. Requests larger than the available neighborhood
    are deterministically clamped.
    """

    refined, _ = _smooth_spatial_labels_detailed(
        labels,
        spatial_coords,
        k=k,
        include_self=include_self,
        weights=weights,
        tie_policy=tie_policy,
    )
    return refined


@dataclass(frozen=True)
class SpatialSmoothingSelection:
    """Held-out result for choosing the spatial label-vote neighborhood."""

    selected_k: int
    selected_labels: np.ndarray
    scores: tuple[dict, ...]
    selection_rule: str


def select_spatial_smoothing_k(
    predicted_labels: Sequence[str],
    true_labels: Sequence[str],
    spatial_coords: np.ndarray,
    *,
    k_values: Sequence[int] = (1, 5, 10, 20, 50),
    score_mask: Optional[Sequence[bool]] = None,
    groups: Optional[Sequence[object]] = None,
    include_self: bool = True,
    weights: str = "uniform",
    tie_policy: str = "sklearn_legacy",
) -> SpatialSmoothingSelection:
    """Select spatial smoothing on a fixed classifier holdout.

    ``predicted_labels`` and ``spatial_coords`` should contain each complete
    observed slice so held-out rows retain their real spatial context.
    ``score_mask`` limits metrics to the fixed held-out rows, while ``groups``
    keeps voting within real time slices. Ground truth is never used as a
    neighbor vote.

    The highest balanced accuracy wins. An exact balanced-accuracy tie is
    resolved by macro-F1; an exact tie in both metrics keeps the smaller k.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    predicted, coords = _validate_spatial_smoothing_inputs(
        predicted_labels, spatial_coords
    )
    observed = np.asarray(true_labels)
    if observed.ndim != 1 or observed.shape[0] != predicted.shape[0]:
        raise ValueError(
            "true_labels must be one-dimensional and match predicted_labels."
        )

    if score_mask is None:
        evaluated = np.ones(predicted.shape[0], dtype=bool)
    else:
        evaluated = np.asarray(score_mask, dtype=bool)
        if evaluated.ndim != 1 or evaluated.shape[0] != predicted.shape[0]:
            raise ValueError(
                "score_mask must be one-dimensional and match predicted_labels."
            )
        if not np.any(evaluated):
            raise ValueError("score_mask must select at least one row.")

    if groups is None:
        group_values = None
    else:
        group_values = np.asarray(groups)
        if group_values.ndim != 1 or group_values.shape[0] != predicted.shape[0]:
            raise ValueError(
                "groups must be one-dimensional and match predicted_labels."
            )

    candidates = tuple(sorted({_normalize_requested_k(value) for value in k_values}))
    if not candidates:
        raise ValueError("k_values must contain at least one candidate.")

    records: list[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}

    for candidate in candidates:
        if group_values is None:
            smoothed, _ = _smooth_spatial_labels_detailed(
                predicted,
                coords,
                k=candidate,
                include_self=include_self,
                weights=weights,
                tie_policy=tie_policy,
            )
        else:
            smoothed = predicted.copy()
            for group in np.unique(group_values):
                in_group = group_values == group
                smoothed[in_group], _ = _smooth_spatial_labels_detailed(
                    predicted[in_group],
                    coords[in_group],
                    k=candidate,
                    include_self=include_self,
                    weights=weights,
                    tie_policy=tie_policy,
                )

        scored_truth = observed[evaluated]
        scored_prediction = smoothed[evaluated]
        balanced_accuracy = float(
            balanced_accuracy_score(scored_truth, scored_prediction)
        )
        macro_f1 = float(
            f1_score(scored_truth, scored_prediction, average="macro", zero_division=0)
        )
        records.append(
            {
                "k": int(candidate),
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
                "n_scored": int(np.sum(evaluated)),
            }
        )
        labels_by_k[candidate] = smoothed

    best_record = min(
        records,
        key=lambda record: (
            -record["balanced_accuracy"],
            -record["macro_f1"],
            record["k"],
        ),
    )
    best_k = int(best_record["k"])

    return SpatialSmoothingSelection(
        selected_k=int(best_k),
        selected_labels=labels_by_k[best_k].copy(),
        scores=tuple(records),
        selection_rule=(
            "maximum balanced accuracy; exact tie: maximum macro-F1; "
            "exact tie again: smaller k"
        ),
    )


def _infer_nearest_neighbor_boundary(
    labels: np.ndarray,
    spatial_coords: np.ndarray,
) -> np.ndarray:
    """Mark a point as boundary when its nearest *other* point has another label."""

    n_samples = int(labels.shape[0])
    if n_samples <= 1:
        return np.zeros(n_samples, dtype=bool)
    _, indices = _stable_spatial_neighbors(
        spatial_coords,
        k=1,
        include_self=False,
    )
    return np.asarray(labels[indices[:, 0]] != labels, dtype=bool)


def analyze_spatial_label_sensitivity(
    labels: Sequence[str],
    spatial_coords: np.ndarray,
    *,
    k_values: Sequence[int] = (1, 5, 10, 20, 50),
    include_self: bool = True,
    weights: str = "uniform",
    tie_policy: str = "sklearn_legacy",
    boundary_mask: Optional[Sequence[bool]] = None,
) -> dict:
    """Evaluate spatial-label smoothing across k without rerunning the MLP.

    The returned records compare each smoothed result with the same raw label
    vector. Composition changes are fractions (TV) and percentage points
    (``max_absolute_change_pp`` / per-type ``abundance_change_pp``).
    """

    raw_labels, coords = _validate_spatial_smoothing_inputs(labels, spatial_coords)
    resolved_k_values = tuple(_normalize_requested_k(value) for value in k_values)
    if boundary_mask is None:
        resolved_boundary = _infer_nearest_neighbor_boundary(raw_labels, coords)
        boundary_source = "nearest_other_label_disagreement"
    else:
        resolved_boundary = np.asarray(boundary_mask, dtype=bool)
        if resolved_boundary.ndim != 1 or resolved_boundary.shape[0] != raw_labels.shape[0]:
            raise ValueError("boundary_mask must be one-dimensional and match labels.")
        resolved_boundary = resolved_boundary.copy()
        boundary_source = "provided"

    classes = np.unique(raw_labels)
    n_samples = int(raw_labels.shape[0])
    raw_counts = {value: int(np.sum(raw_labels == value)) for value in classes}
    raw_fraction = {
        value: (float(count) / n_samples if n_samples else 0.0)
        for value, count in raw_counts.items()
    }

    results = []
    for requested_k in resolved_k_values:
        refined, smoothing_metadata = _smooth_spatial_labels_detailed(
            raw_labels,
            coords,
            k=requested_k,
            include_self=include_self,
            weights=weights,
            tie_policy=tie_policy,
        )
        changed = np.asarray(refined != raw_labels, dtype=bool)
        refined_counts = {value: int(np.sum(refined == value)) for value in classes}
        refined_fraction = {
            value: (float(count) / n_samples if n_samples else 0.0)
            for value, count in refined_counts.items()
        }
        absolute_changes = np.asarray(
            [abs(refined_fraction[value] - raw_fraction[value]) for value in classes],
            dtype=np.float64,
        )
        per_type = {}
        for value in classes:
            original_mask = raw_labels == value
            retained = int(np.sum(refined[original_mask] == value))
            raw_count = raw_counts[value]
            per_type[str(value)] = {
                "raw_count": int(raw_count),
                "smoothed_count": int(refined_counts[value]),
                "raw_fraction": float(raw_fraction[value]),
                "smoothed_fraction": float(refined_fraction[value]),
                "abundance_change_pp": float(
                    100.0 * (refined_fraction[value] - raw_fraction[value])
                ),
                "retained_count": int(retained),
                "retention_fraction": (
                    float(retained) / raw_count if raw_count else None
                ),
            }

        boundary_count = int(np.sum(resolved_boundary))
        interior_mask = ~resolved_boundary
        interior_count = int(np.sum(interior_mask))
        results.append(
            {
                **smoothing_metadata,
                "changed_count": int(np.sum(changed)),
                "changed_fraction": float(np.mean(changed)) if n_samples else 0.0,
                "composition_total_variation": (
                    float(0.5 * np.sum(absolute_changes)) if absolute_changes.size else 0.0
                ),
                "max_absolute_change_pp": (
                    float(100.0 * np.max(absolute_changes)) if absolute_changes.size else 0.0
                ),
                "per_type": per_type,
                "boundary_flip_rate": (
                    float(np.mean(changed[resolved_boundary])) if boundary_count else None
                ),
                "interior_flip_rate": (
                    float(np.mean(changed[interior_mask])) if interior_count else None
                ),
            }
        )

    return {
        "n_samples": n_samples,
        "k_values_requested": list(resolved_k_values),
        "include_self": bool(include_self),
        "weights": str(weights).strip().lower(),
        "tie_policy": str(tie_policy).strip().lower(),
        "boundary_definition": boundary_source,
        "n_boundary": int(np.sum(resolved_boundary)),
        "n_interior": int(np.sum(~resolved_boundary)),
        "results": results,
    }


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        hidden_size = int(hidden_size)
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0.")
        wide_size = 4 * hidden_size
        middle_size = 2 * hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, wide_size),
            nn.LeakyReLU(0.2),
        )
        self.res1 = ResidualBlock(wide_size, wide_size)
        self.res2 = ResidualBlock(wide_size, middle_size)
        self.res3 = ResidualBlock(middle_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_proj(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        return self.fc_out(out)


MLP = ResidualMLP


@contextmanager
def _classifier_cache_lock(path: Path):
    """Serialize writers of one classifier cache on POSIX filesystems."""

    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX fallback
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        torch.save(payload, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class LoadedClassifierCache:
    model: MLP
    label_encoder: object
    feature_cols: tuple[str, ...]
    label_col: str
    accuracy: Optional[float]
    balanced_accuracy: Optional[float]
    metadata: dict
    evaluation: dict

    @property
    def include_time_feature(self) -> bool:
        return bool(self.feature_cols) and str(self.feature_cols[0]) == "samples"

    @property
    def feature_dim(self) -> int:
        if self.include_time_feature:
            return max(0, len(self.feature_cols) - 1)
        return len(self.feature_cols)


def load_cached_mlp_classifier(
    cache_path: str,
    *,
    device: str = "cpu",
) -> LoadedClassifierCache:
    """Load an MLP classifier checkpoint saved by the legacy downstream cache logic.

    The old MOSTA/ARISTA review scripts store classifier checkpoints as a dict with:
    - ``state_dict``
    - ``meta`` containing ``feature_cols`` / ``classes`` / ``input_size``
    - optional ``acc`` / ``bacc``
    """
    from sklearn.preprocessing import LabelEncoder

    try:
        payload = torch.load(str(cache_path), map_location="cpu")
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported classifier cache payload type: {type(payload)}")

    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise TypeError("Classifier cache is missing a valid `meta` dictionary.")

    feature_cols = tuple(str(c) for c in meta.get("feature_cols", []))
    classes = tuple(str(c) for c in meta.get("classes", payload.get("classes", [])))
    if not feature_cols:
        raise KeyError("Classifier cache meta is missing `feature_cols`.")
    if not classes:
        raise KeyError("Classifier cache meta is missing `classes`.")
    if "state_dict" not in payload:
        raise KeyError("Classifier cache payload is missing `state_dict`.")

    input_size = int(meta.get("input_size", payload.get("input_size", len(feature_cols))))
    hidden_size = int(meta.get("hidden_size", payload.get("hidden_size", 128)))
    num_classes = int(payload.get("num_classes", len(classes)))
    model = ResidualMLP(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    model.load_state_dict(payload["state_dict"], strict=True)

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model = model.to(dev)
    model.eval()

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.asarray(classes)

    return LoadedClassifierCache(
        model=model,
        label_encoder=label_encoder,
        feature_cols=feature_cols,
        label_col=str(meta.get("label_col", "Annotation")),
        accuracy=float(payload["acc"]) if payload.get("acc") is not None else None,
        balanced_accuracy=float(payload["bacc"]) if payload.get("bacc") is not None else None,
        metadata=dict(meta),
        evaluation=dict(payload.get("evaluation", {})),
    )


def _resolve_cached_latent_indices(feature_cols: Sequence[str], latent_dim: int) -> np.ndarray:
    if not feature_cols:
        return np.zeros((0,), dtype=int)

    indices = []
    for col in feature_cols:
        match = re.fullmatch(r"x(\d+)", str(col))
        if match is None:
            raise ValueError(
                "Unsupported cached classifier feature columns. "
                "Expected latent feature names like x1, x2, ..., xN; "
                f"got {tuple(feature_cols)}."
            )
        idx = int(match.group(1)) - 1
        if idx < 0 or idx >= int(latent_dim):
            raise IndexError(
                f"Cached classifier requests latent dimension {idx + 1}, "
                f"but adata.obsm latent matrix only has {latent_dim} columns."
            )
        indices.append(idx)
    return np.asarray(indices, dtype=int)


def build_cached_classifier_inputs_from_adata(
    adata,
    cached: LoadedClassifierCache,
    *,
    latent_key: str = "latent_x",
    samples_column: str = "samples",
) -> np.ndarray:
    """Build an input matrix compatible with a legacy cached classifier.

    Legacy downstream caches store feature names such as ``samples`` and ``x1``..``x10``.
    This helper maps those feature names onto an AnnData object with latent coordinates in
    ``adata.obsm[latent_key]``.
    """
    if not (hasattr(adata, "obs") and hasattr(adata, "obsm")):
        raise TypeError(
            "Cached-classifier utilities require AnnData input. "
            f"Got: {type(adata)}"
        )
    if latent_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{latent_key}'] is required to rebuild cached classifier inputs.")

    latent = np.asarray(adata.obsm[latent_key], dtype=np.float32)
    blocks = []
    latent_feature_cols = cached.feature_cols

    if cached.include_time_feature:
        if samples_column not in adata.obs.columns:
            raise KeyError(f"adata.obs['{samples_column}'] is required when the cached classifier uses `samples`.")
        from CytoBridge.tl.downstream.downstream_data import parse_time_value

        samples = np.asarray(
            [parse_time_value(v) for v in adata.obs[samples_column].values],
            dtype=np.float32,
        ).reshape(-1, 1)
        blocks.append(samples)
        latent_feature_cols = cached.feature_cols[1:]

    latent_indices = _resolve_cached_latent_indices(latent_feature_cols, latent.shape[1])
    blocks.append(latent[:, latent_indices].astype(np.float32))

    X = np.hstack(blocks).astype(np.float32)
    expected_dim = int(cached.metadata.get("input_size", X.shape[1]))
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Rebuilt cached classifier input has shape {X.shape[1]}, "
            f"but checkpoint metadata expects {expected_dim}."
        )
    return X


def predict_cached_mlp_classifier_from_adata(
    adata,
    cached: LoadedClassifierCache,
    *,
    latent_key: str = "latent_x",
    samples_column: str = "samples",
    device: Optional[str] = None,
) -> np.ndarray:
    """Predict labels for an AnnData object using a loaded legacy classifier cache."""
    X = build_cached_classifier_inputs_from_adata(
        adata,
        cached,
        latent_key=latent_key,
        samples_column=samples_column,
    )

    if device is None:
        dev = next(cached.model.parameters()).device
        model = cached.model
    else:
        dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        model = cached.model.to(dev)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=dev))
        pred_idx = torch.argmax(logits, dim=1).detach().cpu().numpy()
    return cached.label_encoder.inverse_transform(pred_idx)


def _normalize_column_indices(
    indices: Sequence[int],
    *,
    n_columns: int,
    name: str,
) -> np.ndarray:
    resolved = np.asarray(tuple(indices), dtype=np.int64)
    if resolved.ndim != 1 or resolved.size == 0:
        raise ValueError(f"{name} must contain at least one column index.")
    if np.any(resolved < 0) or np.any(resolved >= int(n_columns)):
        raise IndexError(
            f"{name}={tuple(int(value) for value in resolved)} is outside "
            f"the available point columns [0, {int(n_columns) - 1}]."
        )
    if np.unique(resolved).size != resolved.size:
        raise ValueError(f"{name} must not contain duplicate column indices.")
    return resolved


def _resolve_prediction_arrays(
    points: np.ndarray,
    *,
    feature_dim: int,
    feature_indices: Optional[Sequence[int]],
    spatial_coords: Optional[np.ndarray],
    spatial_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2:
        raise ValueError("points must be a two-dimensional array.")
    if int(feature_dim) <= 0:
        raise ValueError("feature_dim must be > 0.")

    if feature_indices is None:
        if int(feature_dim) > pts.shape[1]:
            raise ValueError(
                f"feature_dim={feature_dim} exceeds the {pts.shape[1]} available point columns."
            )
        resolved_features = np.arange(int(feature_dim), dtype=np.int64)
    else:
        resolved_features = _normalize_column_indices(
            feature_indices,
            n_columns=pts.shape[1],
            name="feature_indices",
        )
        if resolved_features.size != int(feature_dim):
            raise ValueError(
                f"feature_indices contains {resolved_features.size} columns, "
                f"but feature_dim={feature_dim}."
            )

    if spatial_coords is None:
        resolved_spatial = _normalize_column_indices(
            spatial_indices,
            n_columns=pts.shape[1],
            name="spatial_indices",
        )
        coords = pts[:, resolved_spatial].astype(np.float64, copy=True)
    else:
        coords = np.asarray(spatial_coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] != pts.shape[0]:
            raise ValueError(
                "spatial_coords must be two-dimensional and have one row per point."
            )
        if coords.shape[1] == 0 or not np.all(np.isfinite(coords)):
            raise ValueError(
                "spatial_coords must have at least one column and contain only finite values."
            )
        coords = coords.copy()

    return pts[:, resolved_features].astype(np.float32, copy=True), coords


def predict_labels_for_points(
    *,
    points: np.ndarray,
    time_value: float,
    model: MLP,
    label_encoder,
    feature_dim: int,
    device: str = "cuda",
    knn_neighbors: int = 10,
    include_time_feature: bool = True,
    feature_indices: Optional[Sequence[int]] = None,
    spatial_coords: Optional[np.ndarray] = None,
    spatial_indices: Sequence[int] = (0, 1),
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2:
        raise ValueError("points must be a two-dimensional array.")
    n = int(pts.shape[0])
    if n == 0:
        return np.asarray([], dtype=str)

    classifier_features, resolved_spatial_coords = _resolve_prediction_arrays(
        pts,
        feature_dim=feature_dim,
        feature_indices=feature_indices,
        spatial_coords=spatial_coords,
        spatial_indices=spatial_indices,
    )

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model.eval()
    model.to(dev)

    feature_t = torch.tensor(classifier_features, dtype=torch.float32)
    if include_time_feature:
        samples_t = torch.full((n, 1), fill_value=float(time_value), dtype=torch.float32)
        input_t = torch.cat((samples_t, feature_t), dim=1)
    else:
        input_t = feature_t

    with torch.no_grad():
        outputs = model(input_t.float().to(dev))
        _, predicted = torch.max(outputs, 1)
        predicted_labels = label_encoder.inverse_transform(predicted.detach().cpu().numpy())

    refined_labels = smooth_spatial_labels(
        predicted_labels,
        resolved_spatial_coords,
        k=knn_neighbors,
        include_self=True,
        weights="uniform",
        tie_policy="sklearn_legacy",
    )
    return np.asarray(refined_labels).astype(str)


def _prepare_classifier_arrays(
    adata,
    *,
    label_col: str,
    time_key: Optional[str],
    obsm_key: str,
    spatial_key: str,
    concat_spatial: Optional[bool],
    samples_column: str,
    include_time_feature: bool,
    n_features: Optional[int] = None,
):
    if not (hasattr(adata, "obs") and hasattr(adata, "obsm")):
        raise TypeError(
            "Downstream classification APIs require AnnData input. "
            f"Got: {type(adata)}"
        )

    from CytoBridge.tl.downstream.downstream_data import infer_time_key, parse_time_value

    if label_col not in adata.obs.columns:
        raise KeyError(
            f"Label column '{label_col}' not found in adata.obs."
        )

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    raw_times = adata.obs[resolved_time_key].values
    samples = np.asarray([parse_time_value(v) for v in raw_times], dtype=np.float32).reshape(-1, 1)

    if obsm_key in adata.obsm:
        latent = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
    else:
        latent = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        latent = np.asarray(latent, dtype=np.float32)

    use_spatial = bool(concat_spatial) if concat_spatial is not None else (spatial_key in adata.obsm)
    if use_spatial:
        if spatial_key not in adata.obsm:
            raise KeyError(f"concat_spatial=True but adata.obsm['{spatial_key}'] is missing.")
        spatial = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        if spatial.shape[0] != latent.shape[0]:
            raise ValueError(
                f"Row mismatch between '{spatial_key}' ({spatial.shape[0]}) and "
                f"'{obsm_key}' ({latent.shape[0]})."
            )
        features = np.hstack((spatial, latent)).astype(np.float32)
    else:
        features = latent.astype(np.float32)

    if n_features is not None:
        n_features = int(n_features)
        if n_features <= 0:
            raise ValueError("n_features must be positive when supplied.")
        if n_features > features.shape[1]:
            raise ValueError(
                f"n_features={n_features} exceeds the available joint feature "
                f"dimension {features.shape[1]}."
            )
        features = features[:, :n_features]

    X = np.hstack((samples, features)).astype(np.float32) if include_time_feature else features.astype(np.float32)
    y = adata.obs[label_col].astype(str).values
    return X, y


def _train_mlp_classifier_arrays(
    X: np.ndarray,
    y: Sequence[str],
    hidden_size: int = 128,
    epochs: int = 500,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    train_on_full_data: bool = False,
    refit_on_full_data_after_selection: bool = False,
    stratify_split: bool = True,
    strict_stratification: bool = False,
) -> Tuple[MLP, object, float]:
    model, label_encoder, accuracy, _, evaluation = _train_mlp_classifier_arrays_detailed(
        X=X,
        y=y,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        test_size=test_size,
        seed=seed,
        device=device,
        best_epoch_metric="bacc",
        train_on_full_data=train_on_full_data,
        refit_on_full_data_after_selection=refit_on_full_data_after_selection,
        stratify_split=stratify_split,
        strict_stratification=strict_stratification,
    )
    # Preserve the historical three-value return while exposing the detailed
    # Phase-A/refit provenance to callers that do not use the cache wrapper.
    model.classifier_evaluation_ = evaluation
    return model, label_encoder, accuracy


def _split_classifier_indices(
    y_encoded: np.ndarray,
    *,
    test_size: float,
    seed: int,
    stratify_split: bool,
    strict_stratification: bool,
    train_on_full_data: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resolve the historical full-data scope or a reproducible Phase-A split."""

    from sklearn.model_selection import train_test_split

    encoded = np.asarray(y_encoded, dtype=np.int64)
    if encoded.ndim != 1:
        raise ValueError("y_encoded must be one-dimensional.")
    all_indices = np.arange(encoded.shape[0], dtype=np.int64)
    if train_on_full_data:
        return all_indices.copy(), all_indices.copy(), {
            "strategy": "legacy_full_data_training_scope",
            "stratify_requested": bool(stratify_split),
            "strict_stratification": bool(strict_stratification),
            "stratify_used": False,
            "stratification_fallback_reason": "not_applicable_train_on_full_data",
        }

    if strict_stratification and not stratify_split:
        raise ValueError(
            "strict_stratification=True requires stratify_split=True."
        )

    n_classes = int(np.max(encoded) + 1) if encoded.size else 0
    class_counts = np.bincount(encoded, minlength=n_classes)
    n_validation = int(np.ceil(float(test_size) * encoded.shape[0]))
    n_train = int(encoded.shape[0] - n_validation)
    fallback_reasons = []
    if stratify_split:
        if np.any(class_counts < 2):
            fallback_reasons.append("class_with_fewer_than_two_rows")
        if n_validation < n_classes:
            fallback_reasons.append("validation_smaller_than_number_of_classes")
        if n_train < n_classes:
            fallback_reasons.append("training_smaller_than_number_of_classes")
    can_stratify = bool(stratify_split and not fallback_reasons)
    if strict_stratification and not can_stratify:
        reason = ", ".join(fallback_reasons) or "unknown_stratification_constraint"
        raise ValueError(
            "Strict stratification could not be honored: "
            f"{reason}; class_counts={class_counts.tolist()}, "
            f"n_train={n_train}, n_validation={n_validation}."
        )

    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=seed,
        stratify=encoded if can_stratify else None,
    )
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
        {
            "strategy": "held_out_train_validation",
            "stratify_requested": bool(stratify_split),
            "strict_stratification": bool(strict_stratification),
            "stratify_used": bool(can_stratify),
            "stratification_fallback_reason": (
                None
                if can_stratify
                else (
                    "disabled_by_request"
                    if not stratify_split
                    else ", ".join(fallback_reasons)
                )
            ),
        },
    )


def _seed_classifier_training(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _train_mlp_classifier_arrays_detailed(
    X: np.ndarray,
    y: Sequence[str],
    hidden_size: int = 128,
    epochs: int = 500,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    best_epoch_metric: str = "bacc",
    train_on_full_data: bool = False,
    refit_on_full_data_after_selection: bool = False,
    stratify_split: bool = True,
    strict_stratification: bool = False,
) -> Tuple[MLP, object, float, float, dict]:
    import copy

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.preprocessing import LabelEncoder

    metric = str(best_epoch_metric).strip().lower()
    if metric not in {"accuracy", "bacc"}:
        raise ValueError("best_epoch_metric must be one of {'accuracy', 'bacc'}.")
    if int(epochs) <= 0:
        raise ValueError("epochs must be > 0.")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be in (0, 1).")
    if train_on_full_data and refit_on_full_data_after_selection:
        raise ValueError(
            "train_on_full_data=True is the legacy training-scope evaluation mode "
            "and cannot be combined with refit_on_full_data_after_selection=True."
        )

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    if X.shape[0] != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    train_indices, test_indices, split_metadata = _split_classifier_indices(
        y_encoded,
        test_size=test_size,
        seed=seed,
        stratify_split=stratify_split,
        strict_stratification=strict_stratification,
        train_on_full_data=train_on_full_data,
    )
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=dev)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=dev)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=dev)

    input_size = X_train_t.shape[1]
    num_classes = int(len(label_encoder.classes_))
    _seed_classifier_training(seed)
    model = ResidualMLP(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs), eta_min=1e-5)

    best_score = float("-inf")
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = accuracy_score(y_test, val_preds.detach().cpu().numpy())
            val_bacc = balanced_accuracy_score(y_test, val_preds.detach().cpu().numpy())
            score = float(val_acc if metric == "accuracy" else val_bacc)
            if score > best_score:
                best_score = score
                best_epoch = int(epoch + 1)
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        test_preds = torch.argmax(model(X_test_t), dim=1).detach().cpu().numpy()
        train_preds = torch.argmax(model(X_train_t), dim=1).detach().cpu().numpy()
    accuracy = accuracy_score(y_test, test_preds)
    balanced_accuracy = balanced_accuracy_score(y_test, test_preds)
    train_accuracy = accuracy_score(y_train, train_preds)
    train_balanced_accuracy = balanced_accuracy_score(y_train, train_preds)
    labels = np.arange(len(label_encoder.classes_), dtype=int)
    per_class = classification_report(
        y_test,
        test_preds,
        labels=labels,
        target_names=label_encoder.classes_.astype(str),
        output_dict=True,
        zero_division=0,
    )
    overlap = np.intersect1d(train_indices, test_indices, assume_unique=False)
    union = np.union1d(train_indices, test_indices)

    returned_model = model
    refit_evaluation = {
        "requested": bool(refit_on_full_data_after_selection),
        "performed": False,
        "fresh_model_instantiated": False,
        "scope": None,
        "seed": None,
        "n_train": 0,
        "epochs": 0,
        "optimizer_steps": 0,
        "scheduler_t_max": None,
        "scheduler_last_epoch": None,
        "train_accuracy": None,
        "train_balanced_accuracy": None,
        "initial_loss": None,
        "final_loss": None,
    }
    if refit_on_full_data_after_selection:
        all_indices = np.arange(len(y_encoded), dtype=np.int64)
        X_full_t = torch.tensor(X, dtype=torch.float32, device=dev)
        y_full_t = torch.tensor(y_encoded, dtype=torch.long, device=dev)

        # Deliberately recreate the model after reseeding. Phase A selected only
        # the epoch count; no fitted weights or optimizer state cross into refit.
        _seed_classifier_training(seed)
        refit_model = ResidualMLP(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
        ).to(dev)
        refit_optimizer = torch.optim.Adam(refit_model.parameters(), lr=lr)
        # Preserve the Phase-A scheduler horizon rather than compressing the
        # cosine schedule to the selected best_epoch.
        refit_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            refit_optimizer,
            T_max=int(epochs),
            eta_min=1e-5,
        )
        refit_losses = []
        for _ in range(int(best_epoch)):
            refit_model.train()
            refit_optimizer.zero_grad()
            refit_outputs = refit_model(X_full_t)
            refit_loss = criterion(refit_outputs, y_full_t)
            refit_loss.backward()
            refit_optimizer.step()
            refit_scheduler.step()
            refit_losses.append(float(refit_loss.detach().cpu().item()))

        refit_model.eval()
        with torch.no_grad():
            refit_preds = torch.argmax(refit_model(X_full_t), dim=1).detach().cpu().numpy()
        refit_evaluation = {
            "requested": True,
            "performed": True,
            "fresh_model_instantiated": True,
            "scope": "all rows after held-out Phase-A model selection",
            "seed": int(seed),
            "n_train": int(len(all_indices)),
            "epochs": int(best_epoch),
            "optimizer_steps": int(best_epoch),
            "scheduler_t_max": int(epochs),
            "scheduler_last_epoch": int(refit_scheduler.last_epoch),
            "train_accuracy": float(accuracy_score(y_encoded, refit_preds)),
            "train_balanced_accuracy": float(
                balanced_accuracy_score(y_encoded, refit_preds)
            ),
            "initial_loss": refit_losses[0] if refit_losses else None,
            "final_loss": refit_losses[-1] if refit_losses else None,
        }
        returned_model = refit_model

    split_contract = {
        "strategy": split_metadata["strategy"],
        "n_total": int(len(y_encoded)),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(test_indices)),
        "n_overlap": int(len(overlap)),
        "n_union": int(len(union)),
        "disjoint": bool(len(overlap) == 0),
        "covers_all_rows": bool(len(union) == len(y_encoded)),
    }
    selection_scope = (
        "training data used for model selection (legacy full-data mode)"
        if train_on_full_data
        else "held-out validation split used for Phase-A model selection"
    )
    evaluation = {
        "metric_scope": "training data" if train_on_full_data else "held-out validation split",
        "validation_is_independent_test": False,
        "best_epoch": int(best_epoch),
        "best_epoch_metric": metric,
        "best_epoch_score": float(best_score),
        "train_accuracy": float(train_accuracy),
        "train_balanced_accuracy": float(train_balanced_accuracy),
        "validation_accuracy": float(accuracy),
        "validation_balanced_accuracy": float(balanced_accuracy),
        "selection_scope": selection_scope,
        "returned_model_scope": (
            refit_evaluation["scope"]
            if refit_evaluation["performed"]
            else "Phase-A selected model"
        ),
        "stratify_requested": split_metadata["stratify_requested"],
        "stratify_used": split_metadata["stratify_used"],
        "strict_stratification": split_metadata["strict_stratification"],
        "stratification_fallback_reason": split_metadata[
            "stratification_fallback_reason"
        ],
        "n_train": int(len(train_indices)),
        "n_validation": int(len(test_indices)),
        "split_contract": split_contract,
        "selection": {
            "scope": selection_scope,
            "seed": int(seed),
            "uses_validation_for_epoch_selection": bool(not train_on_full_data),
            "epochs_run": int(epochs),
            "scheduler_t_max": int(epochs),
            "best_epoch": int(best_epoch),
            "metric": metric,
            "best_score": float(best_score),
            "train_accuracy": float(train_accuracy),
            "train_balanced_accuracy": float(train_balanced_accuracy),
            "validation_accuracy": float(accuracy),
            "validation_balanced_accuracy": float(balanced_accuracy),
        },
        "refit": refit_evaluation,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_test, test_preds, labels=labels).tolist(),
    }
    return returned_model, label_encoder, float(accuracy), float(balanced_accuracy), evaluation


def _classifier_cache_fingerprint(
    adata,
    *,
    label_col: str,
    time_key: Optional[str],
    classifier_inputs: Optional[np.ndarray] = None,
) -> str:
    """Build a stable fingerprint over identities, labels, times, and classifier inputs."""
    from CytoBridge.tl.downstream.downstream_data import infer_time_key

    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    digest = sha1()
    digest.update(f"{adata.n_obs}|{adata.n_vars}|{label_col}|{resolved_time_key}".encode("utf-8"))
    for values in (
        adata.obs_names.astype(str),
        adata.obs[label_col].astype(str).values,
        adata.obs[resolved_time_key].astype(str).values,
    ):
        digest.update("\x1f".join(map(str, values)).encode("utf-8"))
    if classifier_inputs is not None:
        inputs = np.ascontiguousarray(classifier_inputs, dtype=np.float32)
        digest.update(str(inputs.shape).encode("utf-8"))
        digest.update(inputs.tobytes())
    return digest.hexdigest()


def train_cached_mlp_classifier_from_adata(
    adata,
    *,
    cache_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    cache_tag: Optional[str] = None,
    reuse_if_compatible: bool = True,
    label_col: str = "Annotation",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    samples_column: str = "samples",
    hidden_size: int = 128,
    epochs: int = 500,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    include_time_feature: bool = True,
    n_features: Optional[int] = None,
    best_epoch_metric: str = "bacc",
    train_on_full_data: bool = False,
    refit_on_full_data_after_selection: bool = False,
    stratify_split: bool = True,
    strict_stratification: bool = False,
) -> tuple[LoadedClassifierCache, Path]:
    """Train, persist, and reload a trajectory-label classifier from AnnData.

    The cache format is intentionally compatible with historical ARISTA/MOSTA
    ``classifier_resmlp_*.pt`` files, so current workflows can either reuse an old
    cache or create one through this public API. Feature names describe the joint
    aligned state (``samples``, then ``x1..xD``), independent of dataset-specific
    AnnData key names. ``n_features`` selects the leading joint dimensions after
    optional spatial concatenation; it exists for compatibility with historical
    classifiers whose caches used ``x1..xN``.
    """
    if cache_path is None and cache_dir is None:
        raise ValueError("Provide cache_path or cache_dir.")
    if train_on_full_data and refit_on_full_data_after_selection:
        raise ValueError(
            "train_on_full_data=True cannot be combined with "
            "refit_on_full_data_after_selection=True."
        )

    X, y = _prepare_classifier_arrays(
        adata,
        label_col=label_col,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        samples_column=samples_column,
        include_time_feature=include_time_feature,
        n_features=n_features,
    )
    feature_dim = int(X.shape[1] - (1 if include_time_feature else 0))
    feature_cols = ([samples_column] if include_time_feature else []) + [
        f"x{i + 1}" for i in range(feature_dim)
    ]
    classes = sorted({str(v) for v in y})
    metadata = {
        "version": 6,
        "cache_tag": str(cache_tag or ""),
        "feature_cols": feature_cols,
        "label_col": str(label_col),
        "hidden_size": int(hidden_size),
        "epochs": int(epochs),
        "lr": float(lr),
        "test_size": float(test_size),
        "seed": int(seed),
        "input_size": int(X.shape[1]),
        "classes": classes,
        "best_epoch_metric": str(best_epoch_metric).strip().lower(),
        "train_on_full_data": bool(train_on_full_data),
        "refit_on_full_data_after_selection": bool(
            refit_on_full_data_after_selection
        ),
        "stratify_split": bool(stratify_split),
        "strict_stratification": bool(strict_stratification),
        "selection_scope": (
            "legacy_full_data_training_scope"
            if train_on_full_data
            else "held_out_validation_phase_a"
        ),
        "include_time_feature": bool(include_time_feature),
        "feature_selection": {
            "kind": "leading_joint_dimensions",
            "n_features": int(feature_dim),
            "requested_n_features": (
                None if n_features is None else int(n_features)
            ),
        },
        "source": {
            "kind": "AnnData",
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "fingerprint": _classifier_cache_fingerprint(
                adata,
                label_col=label_col,
                time_key=time_key,
                classifier_inputs=X,
            ),
            "obsm_key": str(obsm_key),
            "spatial_key": str(spatial_key),
            "concat_spatial": concat_spatial,
        },
    }

    if cache_path is None:
        cache_root = Path(cache_dir).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        key = sha1(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        resolved_path = cache_root / f"classifier_resmlp_{key}.pt"
    else:
        resolved_path = Path(cache_path).expanduser().resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with _classifier_cache_lock(resolved_path):
        # Recheck after acquiring the lock: another process may have completed
        # the same deterministic cache while this process was waiting.
        if reuse_if_compatible and resolved_path.exists():
            try:
                cached = load_cached_mlp_classifier(
                    str(resolved_path), device=device
                )
            except Exception:
                cached = None
            if cached is not None and cached.metadata == metadata:
                return cached, resolved_path

        model, label_encoder, accuracy, balanced_accuracy, evaluation = (
            _train_mlp_classifier_arrays_detailed(
                X=X,
                y=y,
                hidden_size=hidden_size,
                epochs=epochs,
                lr=lr,
                test_size=test_size,
                seed=seed,
                device=device,
                best_epoch_metric=best_epoch_metric,
                train_on_full_data=train_on_full_data,
                refit_on_full_data_after_selection=refit_on_full_data_after_selection,
                stratify_split=stratify_split,
                strict_stratification=strict_stratification,
            )
        )
        payload = {
            "meta": metadata,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "acc": float(accuracy),
            "bacc": float(balanced_accuracy),
            "evaluation": evaluation,
            "num_classes": int(len(label_encoder.classes_)),
            "saved_at": float(time.time()),
        }
        _atomic_torch_save(payload, resolved_path)
        return (
            load_cached_mlp_classifier(str(resolved_path), device=device),
            resolved_path,
        )


def train_mlp_classifier(
    adata,
    *,
    label_col: str = "Annotation",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    samples_column: str = "samples",
    hidden_size: int = 128,
    epochs: int = 500,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    include_time_feature: bool = True,
    n_features: Optional[int] = None,
    train_on_full_data: bool = False,
    refit_on_full_data_after_selection: bool = False,
    stratify_split: bool = True,
    strict_stratification: bool = False,
) -> Tuple[MLP, object, float]:
    """Train a downstream MLP classifier from AnnData.

    The production protocol uses a 128-unit residual MLP, 500 selection
    epochs, Adam at ``1e-3``, seed 42, and held-out balanced accuracy for
    checkpoint selection. Choose spatial label smoothing separately on the
    same fixed holdout with :func:`select_spatial_smoothing_k`.
    """
    if train_on_full_data and refit_on_full_data_after_selection:
        raise ValueError(
            "train_on_full_data=True cannot be combined with "
            "refit_on_full_data_after_selection=True."
        )
    X, y = _prepare_classifier_arrays(
        adata,
        label_col=label_col,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        samples_column=samples_column,
        include_time_feature=include_time_feature,
        n_features=n_features,
    )
    return _train_mlp_classifier_arrays(
        X=X,
        y=y,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        test_size=test_size,
        seed=seed,
        device=device,
        train_on_full_data=train_on_full_data,
        refit_on_full_data_after_selection=refit_on_full_data_after_selection,
        stratify_split=stratify_split,
        strict_stratification=strict_stratification,
    )


def train_mlp_classifier_from_adata(
    adata,
    *,
    label_col: str = "Annotation",
    time_key: Optional[str] = None,
    obsm_key: str = "X_latent",
    spatial_key: str = "spatial_aligned",
    concat_spatial: Optional[bool] = None,
    samples_column: str = "samples",
    hidden_size: int = 128,
    epochs: int = 500,
    lr: float = 1e-3,
    test_size: float = 0.1,
    seed: int = 42,
    device: str = "cuda",
    include_time_feature: bool = True,
    n_features: Optional[int] = None,
    train_on_full_data: bool = False,
    refit_on_full_data_after_selection: bool = False,
    stratify_split: bool = True,
    strict_stratification: bool = False,
) -> Tuple[MLP, object, float]:
    return train_mlp_classifier(
        adata,
        label_col=label_col,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        samples_column=samples_column,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        test_size=test_size,
        seed=seed,
        device=device,
        include_time_feature=include_time_feature,
        n_features=n_features,
        train_on_full_data=train_on_full_data,
        refit_on_full_data_after_selection=refit_on_full_data_after_selection,
        stratify_split=stratify_split,
        strict_stratification=strict_stratification,
    )


def predict_labels_for_trajectories(
    sde_points: np.ndarray,
    ts_points: Sequence[float],
    model: MLP,
    label_encoder,
    feature_dim: int,
    device: str = "cuda",
    knn_neighbors: int = 10,
    include_time_feature: bool = True,
    feature_indices: Optional[Sequence[int]] = None,
    spatial_coords: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    spatial_indices: Sequence[int] = (0, 1),
) -> Sequence[np.ndarray]:
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    model.eval()
    model.to(dev)

    if len(sde_points) != len(ts_points):
        raise ValueError(
            "sde_points and ts_points must contain the same number of time slices."
        )

    spatial_coords_by_time: Optional[list[np.ndarray]] = None
    if spatial_coords is not None:
        if isinstance(spatial_coords, np.ndarray) and spatial_coords.ndim == 2:
            if len(ts_points) != 1:
                raise ValueError(
                    "A two-dimensional spatial_coords array is only valid for one time slice."
                )
            spatial_coords_by_time = [spatial_coords]
        else:
            spatial_coords_by_time = [np.asarray(value) for value in spatial_coords]
            if len(spatial_coords_by_time) != len(ts_points):
                raise ValueError(
                    "spatial_coords must contain one coordinate array per time slice."
                )

    predicted_labels_list = []
    for i, t in enumerate(ts_points):
        traj_t = np.asarray(sde_points[i], dtype=float)
        if traj_t.ndim == 1:
            traj_t = traj_t.reshape(1, -1)
        explicit_spatial_coords = (
            None if spatial_coords_by_time is None else spatial_coords_by_time[i]
        )
        classifier_features, resolved_spatial_coords = _resolve_prediction_arrays(
            traj_t,
            feature_dim=feature_dim,
            feature_indices=feature_indices,
            spatial_coords=explicit_spatial_coords,
            spatial_indices=spatial_indices,
        )
        feature_t = torch.tensor(classifier_features, dtype=torch.float32)
        n_samples = int(feature_t.shape[0])

        if include_time_feature:
            samples_t = torch.full((n_samples, 1), fill_value=float(t), dtype=torch.float32)
            input_t = torch.cat((samples_t, feature_t), dim=1)
        else:
            input_t = feature_t

        with torch.no_grad():
            outputs = model(input_t.float().to(dev))
            _, predicted = torch.max(outputs, 1)
            predicted_labels = label_encoder.inverse_transform(predicted.detach().cpu().numpy())

        refined_labels = smooth_spatial_labels(
            predicted_labels,
            resolved_spatial_coords,
            k=knn_neighbors,
            include_self=True,
            weights="uniform",
            tie_policy="sklearn_legacy",
        )

        predicted_labels_list.append(refined_labels)

    return predicted_labels_list
