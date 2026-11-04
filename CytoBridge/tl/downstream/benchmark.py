"""Comparable distribution metrics for spatiotemporal benchmarks.

The helpers in this module deliberately separate two concerns:

* :class:`FrozenBenchmarkTransform` is fitted once on training rows and then
  reused for every method and evaluation split.  It gives the state and
  spatial blocks unit expected squared norm before they are concatenated, so
  neither the number nor the native units of their coordinates dominate the
  joint distance.
* :func:`evaluate_spatiotemporal_prediction` compares empirical measures
  without matching their numbers of particles.  Its primary metric is POT's
  weighted sliced Wasserstein-2 distance.  Exact EMD W1/W2 are retained as
  bounded-size secondary diagnostics for continuity with the existing
  CytoBridge evaluation API.

State-only methods may omit both spatial arrays.  Their result then contains
only ``space == "state"`` rows; joint and spatial scores are intentionally not
invented or filled with misleading numerical values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from .evaluation import compute_distribution_metrics

__all__ = [
    "FrozenBenchmarkTransform",
    "benchmark_projection_seed",
    "evaluate_spatiotemporal_prediction",
    "fit_frozen_benchmark_transform",
]


_TRANSFORM_SCHEMA_VERSION = 1
_SEED_NAMESPACE = "cytobridge-spatiotemporal-benchmark-v1"


def _as_finite_matrix(values: object, *, name: str) -> np.ndarray:
    """Return a finite, non-empty float64 matrix or raise a useful error."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    try:
        is_exact = float(value) == float(parsed)
    except (TypeError, ValueError, OverflowError):
        is_exact = False
    if parsed <= 0 or not is_exact:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _as_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    try:
        is_exact = float(value) == float(parsed)
    except (TypeError, ValueError, OverflowError):
        is_exact = False
    if parsed < 0 or not is_exact:
        raise ValueError(f"{name} must be a non-negative integer.")
    return parsed


def _as_label(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    label = value.strip()
    if not label:
        raise ValueError(f"{name} must be a non-empty string.")
    return label


def _normalized_predicted_weights(
    weights: Optional[np.ndarray],
    *,
    n_predicted: int,
) -> tuple[np.ndarray, float]:
    if weights is None:
        return (
            np.full(n_predicted, 1.0 / n_predicted, dtype=np.float64),
            float(n_predicted),
        )
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("predicted_weights must be a one-dimensional array.")
    if values.shape[0] != n_predicted:
        raise ValueError(
            "predicted_weights has "
            f"{values.shape[0]} entries, expected {n_predicted}."
        )
    if not np.isfinite(values).all():
        raise ValueError("predicted_weights must contain only finite values.")
    if np.any(values < 0):
        raise ValueError("predicted_weights must be non-negative.")
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("predicted_weights must have a finite positive sum.")
    return values / total, total


@dataclass(frozen=True)
class FrozenBenchmarkTransform:
    """Train-only normalization shared by all benchmark methods.

    State dimensions are centered and standardized separately, then divided
    by ``sqrt(state_dim)``.  Spatial coordinates are centered per coordinate
    but divided by one shared root-mean-square scale, preserving their aspect
    ratio and geometry, and then by ``sqrt(spatial_dim)``.  Consequently each
    block has mean squared Euclidean norm one on the training rows.  A joint
    vector is simply ``[state_block, spatial_block]``, giving both blocks equal
    contribution in expectation without fitting on a held-out time point.
    """

    state_center: tuple[float, ...]
    state_scale: tuple[float, ...]
    spatial_center: Optional[tuple[float, ...]] = None
    spatial_rms_scale: Optional[float] = None
    schema_version: int = _TRANSFORM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _as_positive_int(self.schema_version, name="schema_version")
        object.__setattr__(self, "schema_version", schema_version)
        state_center = tuple(float(value) for value in self.state_center)
        state_scale = tuple(float(value) for value in self.state_scale)
        object.__setattr__(self, "state_center", state_center)
        object.__setattr__(self, "state_scale", state_scale)

        if schema_version != _TRANSFORM_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported benchmark-transform schema_version "
                f"{self.schema_version!r}; expected {_TRANSFORM_SCHEMA_VERSION}."
            )
        if not state_center:
            raise ValueError("state_center must contain at least one dimension.")
        if len(state_center) != len(state_scale):
            raise ValueError("state_center and state_scale dimensions must match.")
        if not np.isfinite(np.asarray(state_center)).all():
            raise ValueError("state_center must contain only finite values.")
        if not np.isfinite(np.asarray(state_scale)).all() or any(
            value <= 0 for value in state_scale
        ):
            raise ValueError("state_scale must contain finite positive values.")

        has_spatial_center = self.spatial_center is not None
        has_spatial_scale = self.spatial_rms_scale is not None
        if has_spatial_center != has_spatial_scale:
            raise ValueError(
                "spatial_center and spatial_rms_scale must either both be set "
                "or both be None."
            )
        if has_spatial_center:
            assert self.spatial_center is not None
            assert self.spatial_rms_scale is not None
            spatial_center = tuple(float(value) for value in self.spatial_center)
            spatial_rms_scale = float(self.spatial_rms_scale)
            object.__setattr__(self, "spatial_center", spatial_center)
            object.__setattr__(self, "spatial_rms_scale", spatial_rms_scale)
            if not spatial_center:
                raise ValueError(
                    "spatial_center must contain at least one dimension when set."
                )
            if not np.isfinite(np.asarray(spatial_center)).all():
                raise ValueError("spatial_center must contain only finite values.")
            if not np.isfinite(spatial_rms_scale) or spatial_rms_scale <= 0:
                raise ValueError("spatial_rms_scale must be finite and positive.")

    @property
    def state_dim(self) -> int:
        return len(self.state_center)

    @property
    def spatial_dim(self) -> int:
        return 0 if self.spatial_center is None else len(self.spatial_center)

    @property
    def has_spatial(self) -> bool:
        return self.spatial_center is not None

    @classmethod
    def fit(
        cls,
        train_state: np.ndarray,
        train_spatial: Optional[np.ndarray] = None,
    ) -> "FrozenBenchmarkTransform":
        """Fit normalization parameters using training rows only."""

        state = _as_finite_matrix(train_state, name="train_state")
        if state.shape[0] < 2:
            raise ValueError("train_state must contain at least two rows.")
        state_center = state.mean(axis=0)
        state_scale = state.std(axis=0, ddof=0)
        if not np.isfinite(state_scale).all() or np.any(state_scale <= 0):
            bad = np.flatnonzero(~np.isfinite(state_scale) | (state_scale <= 0))
            raise ValueError(
                "Every train_state dimension must have positive finite variance; "
                f"invalid dimensions={bad.tolist()}."
            )

        if train_spatial is None:
            return cls(
                state_center=tuple(state_center),
                state_scale=tuple(state_scale),
            )

        spatial = _as_finite_matrix(train_spatial, name="train_spatial")
        if spatial.shape[0] != state.shape[0]:
            raise ValueError(
                "train_state and train_spatial must have the same number of rows; "
                f"got {state.shape[0]} and {spatial.shape[0]}."
            )
        spatial_center = spatial.mean(axis=0)
        centered_spatial = spatial - spatial_center
        # One scalar across all entries is the isotropic analogue of the
        # per-coordinate state standard deviations.  Dividing the resulting
        # coordinates by sqrt(d) makes the spatial block's expected squared
        # norm exactly one on the training rows while preserving aspect ratio.
        spatial_rms_scale = float(np.sqrt(np.mean(centered_spatial**2)))
        if not np.isfinite(spatial_rms_scale) or spatial_rms_scale <= 0:
            raise ValueError(
                "train_spatial must have positive finite isotropic RMS spread."
            )
        return cls(
            state_center=tuple(state_center),
            state_scale=tuple(state_scale),
            spatial_center=tuple(spatial_center),
            spatial_rms_scale=spatial_rms_scale,
        )

    def transform_state(self, state: np.ndarray) -> np.ndarray:
        values = _as_finite_matrix(state, name="state")
        if values.shape[1] != self.state_dim:
            raise ValueError(
                f"state has {values.shape[1]} dimensions, expected {self.state_dim}."
            )
        return (
            (values - np.asarray(self.state_center, dtype=np.float64))
            / np.asarray(self.state_scale, dtype=np.float64)
            / math.sqrt(self.state_dim)
        )

    def transform_spatial(self, spatial: np.ndarray) -> np.ndarray:
        if not self.has_spatial:
            raise ValueError("This frozen transform was fitted without spatial data.")
        values = _as_finite_matrix(spatial, name="spatial")
        if values.shape[1] != self.spatial_dim:
            raise ValueError(
                "spatial has "
                f"{values.shape[1]} dimensions, expected {self.spatial_dim}."
            )
        assert self.spatial_center is not None
        assert self.spatial_rms_scale is not None
        return (
            (values - np.asarray(self.spatial_center, dtype=np.float64))
            / float(self.spatial_rms_scale)
            / math.sqrt(self.spatial_dim)
        )

    def transform_joint(
        self,
        state: np.ndarray,
        spatial: np.ndarray,
    ) -> np.ndarray:
        state_values = self.transform_state(state)
        spatial_values = self.transform_spatial(spatial)
        if state_values.shape[0] != spatial_values.shape[0]:
            raise ValueError(
                "state and spatial must have the same number of rows; got "
                f"{state_values.shape[0]} and {spatial_values.shape[0]}."
            )
        return np.concatenate((state_values, spatial_values), axis=1)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible, dimension-explicit representation."""

        return {
            "schema_version": _TRANSFORM_SCHEMA_VERSION,
            "state_dim": self.state_dim,
            "spatial_dim": self.spatial_dim,
            "state_center": list(self.state_center),
            "state_scale": list(self.state_scale),
            "spatial_center": (
                None if self.spatial_center is None else list(self.spatial_center)
            ),
            "spatial_rms_scale": self.spatial_rms_scale,
        }

    def to_json(self) -> str:
        """Serialize the transform to deterministic strict JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "FrozenBenchmarkTransform":
        """Restore a transform and validate serialized dimensions."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        required = {
            "schema_version",
            "state_dim",
            "spatial_dim",
            "state_center",
            "state_scale",
            "spatial_center",
            "spatial_rms_scale",
        }
        missing = sorted(required.difference(payload))
        unknown = sorted(set(payload).difference(required))
        if missing or unknown:
            raise ValueError(
                "Invalid benchmark-transform fields: "
                f"missing={missing}, unknown={unknown}."
            )
        state_dim = _as_positive_int(payload["state_dim"], name="state_dim")
        spatial_dim = _as_nonnegative_int(payload["spatial_dim"], name="spatial_dim")
        if not isinstance(payload["state_center"], (list, tuple)) or not isinstance(
            payload["state_scale"], (list, tuple)
        ):
            raise ValueError("state_center and state_scale must be arrays.")
        state_center = tuple(payload["state_center"])
        state_scale = tuple(payload["state_scale"])
        raw_spatial_center = payload["spatial_center"]
        if raw_spatial_center is None:
            spatial_center = None
        else:
            if not isinstance(raw_spatial_center, (list, tuple)):
                raise ValueError("spatial_center must be an array or None.")
            spatial_center = tuple(raw_spatial_center)
        if len(state_center) != state_dim or len(state_scale) != state_dim:
            raise ValueError(
                "Serialized state_dim does not match state_center/state_scale."
            )
        if spatial_dim == 0:
            if spatial_center is not None or payload["spatial_rms_scale"] is not None:
                raise ValueError(
                    "spatial_dim=0 requires null spatial_center and "
                    "spatial_rms_scale."
                )
        elif spatial_center is None or len(spatial_center) != spatial_dim:
            raise ValueError("Serialized spatial_dim does not match spatial_center.")
        return cls(
            schema_version=_as_positive_int(
                payload["schema_version"], name="schema_version"
            ),
            state_center=state_center,
            state_scale=state_scale,
            spatial_center=spatial_center,
            spatial_rms_scale=payload["spatial_rms_scale"],  # type: ignore[arg-type]
        )

    @classmethod
    def from_json(cls, payload: str) -> "FrozenBenchmarkTransform":
        """Restore a transform from :meth:`to_json` output."""

        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("payload must be valid benchmark-transform JSON.") from exc
        if not isinstance(decoded, Mapping):
            raise ValueError("Benchmark-transform JSON must encode an object.")
        return cls.from_dict(decoded)


def fit_frozen_benchmark_transform(
    train_state: np.ndarray,
    train_spatial: Optional[np.ndarray] = None,
) -> FrozenBenchmarkTransform:
    """Fit a :class:`FrozenBenchmarkTransform` on training rows."""

    return FrozenBenchmarkTransform.fit(train_state, train_spatial)


def benchmark_projection_seed(
    benchmark: str,
    split: str,
    space: str,
    repeat: int,
) -> int:
    """Return the shared sliced-W2 seed for one benchmark comparison row.

    The method name is intentionally absent.  Every method evaluated for the
    same benchmark, split, feature space, and repeat therefore receives the
    same random projection basis.
    """

    fields = {
        "namespace": _SEED_NAMESPACE,
        "benchmark": _as_label(benchmark, name="benchmark"),
        "split": _as_label(split, name="split"),
        "space": _as_label(space, name="space"),
        "repeat": _as_nonnegative_int(repeat, name="repeat"),
    }
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    # RandomState, used by POT 0.9.x, requires a seed in [0, 2**32 - 1].
    return int.from_bytes(
        hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "little"
    )


def _exact_ot_seed(benchmark: str, split: str, space: str) -> int:
    canonical = json.dumps(
        {
            "namespace": f"{_SEED_NAMESPACE}-exact-ot",
            "benchmark": benchmark,
            "split": split,
            "space": space,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int.from_bytes(
        hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "little"
    )


def _sliced_w2(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    predicted_weights: np.ndarray,
    n_projections: int,
    seed: int,
) -> tuple[float, str]:
    import ot

    observed_weights = np.full(
        observed.shape[0], 1.0 / observed.shape[0], dtype=np.float64
    )
    value, details = ot.sliced_wasserstein_distance(
        predicted,
        observed,
        a=predicted_weights,
        b=observed_weights,
        n_projections=n_projections,
        p=2,
        seed=seed,
        log=True,
    )
    projections = np.ascontiguousarray(np.asarray(details["projections"], dtype="<f8"))
    projection_hash = hashlib.sha256(projections.tobytes()).hexdigest()
    sliced_w2 = float(value)
    if not np.isfinite(sliced_w2):
        raise ValueError("POT returned a non-finite sliced Wasserstein distance.")
    return sliced_w2, projection_hash


def evaluate_spatiotemporal_prediction(
    *,
    transform: FrozenBenchmarkTransform,
    benchmark: str,
    split: str,
    method: str,
    predicted_state: np.ndarray,
    observed_state: np.ndarray,
    predicted_spatial: Optional[np.ndarray] = None,
    observed_spatial: Optional[np.ndarray] = None,
    predicted_weights: Optional[np.ndarray] = None,
    n_projections: int = 256,
    projection_repeats: int = 5,
    max_ot_points: Optional[int] = 1024,
) -> pd.DataFrame:
    """Evaluate one method/split and return repeat-level long-form metrics.

    Sliced-W2 uses every predicted and observed point, including their unequal
    sample counts; no target-size resampling is performed.  ``max_ot_points``
    applies only to the secondary exact EMD W1/W2 calculation inherited from
    :func:`compute_distribution_metrics`.
    """

    if not isinstance(transform, FrozenBenchmarkTransform):
        raise TypeError("transform must be a FrozenBenchmarkTransform.")
    benchmark_label = _as_label(benchmark, name="benchmark")
    split_label = _as_label(split, name="split")
    method_label = _as_label(method, name="method")
    n_projections = _as_positive_int(n_projections, name="n_projections")
    projection_repeats = _as_positive_int(projection_repeats, name="projection_repeats")
    if max_ot_points is not None:
        max_ot_points = _as_positive_int(max_ot_points, name="max_ot_points")

    predicted_state_raw = _as_finite_matrix(predicted_state, name="predicted_state")
    observed_state_raw = _as_finite_matrix(observed_state, name="observed_state")
    predicted_state_values = transform.transform_state(predicted_state_raw)
    observed_state_values = transform.transform_state(observed_state_raw)
    normalized_weights, raw_weight_sum = _normalized_predicted_weights(
        predicted_weights,
        n_predicted=predicted_state_values.shape[0],
    )

    has_predicted_spatial = predicted_spatial is not None
    has_observed_spatial = observed_spatial is not None
    if has_predicted_spatial != has_observed_spatial:
        raise ValueError(
            "predicted_spatial and observed_spatial must either both be set "
            "or both be None."
        )

    spaces: dict[str, tuple[np.ndarray, np.ndarray]]
    if not has_predicted_spatial:
        spaces = {"state": (predicted_state_values, observed_state_values)}
    else:
        assert predicted_spatial is not None
        assert observed_spatial is not None
        predicted_spatial_raw = _as_finite_matrix(
            predicted_spatial, name="predicted_spatial"
        )
        observed_spatial_raw = _as_finite_matrix(
            observed_spatial, name="observed_spatial"
        )
        if predicted_spatial_raw.shape[0] != predicted_state_raw.shape[0]:
            raise ValueError(
                "predicted_state and predicted_spatial must have the same "
                "number of rows."
            )
        if observed_spatial_raw.shape[0] != observed_state_raw.shape[0]:
            raise ValueError(
                "observed_state and observed_spatial must have the same "
                "number of rows."
            )
        predicted_spatial_values = transform.transform_spatial(predicted_spatial_raw)
        observed_spatial_values = transform.transform_spatial(observed_spatial_raw)
        spaces = {
            "joint": (
                np.concatenate(
                    (predicted_state_values, predicted_spatial_values), axis=1
                ),
                np.concatenate(
                    (observed_state_values, observed_spatial_values), axis=1
                ),
            ),
            "state": (predicted_state_values, observed_state_values),
            "spatial": (predicted_spatial_values, observed_spatial_values),
        }

    rows: list[dict[str, object]] = []
    for space, (predicted_values, observed_values) in spaces.items():
        exact = compute_distribution_metrics(
            predicted_values,
            observed_values,
            predicted_weights=normalized_weights,
            max_ot_points=max_ot_points,
            random_seed=_exact_ot_seed(benchmark_label, split_label, space),
        )
        for repeat in range(projection_repeats):
            projection_seed = benchmark_projection_seed(
                benchmark_label, split_label, space, repeat
            )
            sliced_w2, projection_hash = _sliced_w2(
                predicted_values,
                observed_values,
                predicted_weights=normalized_weights,
                n_projections=n_projections,
                seed=projection_seed,
            )
            rows.append(
                {
                    "benchmark": benchmark_label,
                    "split": split_label,
                    "method": method_label,
                    "space": space,
                    "projection_repeat": repeat,
                    "projection_seed": projection_seed,
                    "projection_sha256": projection_hash,
                    "n_projections": n_projections,
                    "primary_metric": "sliced_w2",
                    "primary_value": sliced_w2,
                    "sliced_w2": sliced_w2,
                    "exact_w1": float(exact["w1"]),
                    "exact_w2": float(exact["w2"]),
                    "n_predicted": int(predicted_values.shape[0]),
                    "n_observed": int(observed_values.shape[0]),
                    "predicted_weight_sum": raw_weight_sum,
                    "exact_ot_predicted_points": int(exact["ot_predicted_points"]),
                    "exact_ot_observed_points": int(exact["ot_observed_points"]),
                }
            )

    return pd.DataFrame.from_records(rows)
