#!/usr/bin/env python3
"""Evaluate fixed-checkpoint interaction on/off with the formal weighted metric.

The experiment does not retrain.  Both branches load the accepted full-model
checkpoint, start from one shared 5,000-particle source roster, reset the same
global and interaction-grouping random streams, and make one continuous
non-split weighted-SDE call from source time to every observed target.  The off
branch only replaces the learned interaction force with an exact-zero adapter.

Predictions retain native unnormalised growth mass.  Evaluation mirrors the
formal matched-ablation benchmark: weighted sliced-W2 (1,024 projections, five
repeats), weighted exact W1/W2 on at most 800 support points, and growth-mass
TMV, in frozen joint/state/spatial coordinates.  Positive relative error means
that turning interaction off worsened reconstruction; negative means it
improved reconstruction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402
import CytoBridge.workflow as workflow  # noqa: E402


SCHEMA_VERSION = 1
DATASETS = ("zebrafish", "mosta", "arista", "admouse", "chicken_heart")
DISPLAY_NAMES = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "admouse": "AdMouse",
    "chicken_heart": "Chicken heart",
}
SEED = 42
INTERACTION_SEED = 10_042
INTERACTION_M = 1_024
PREDICTION_N = 5_000
N_PROJECTIONS = 1_024
PROJECTION_REPEATS = 5
MAX_OT_POINTS = 800
SIGMA = 0.03
DT = 0.01
SPATIAL_DIM = 2
STATE_DIM = 50
SPACES = ("joint", "state", "spatial")
ARMS = ("interaction_on", "interaction_off")
PROJECTION_NAMESPACE = "cytobridge-spatiotemporal-benchmark-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size": int(resolved.stat().st_size),
    }


def _require_hash(path: Path, expected: str, label: str) -> None:
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid expected SHA-256 for {label}: {expected!r}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )


def _new_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _record_path(record: Mapping[str, Any], root: Path) -> Path:
    raw = record.get("path")
    if isinstance(raw, str) and raw.strip():
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = root / path
    else:
        relative = record.get("relative_path")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError("Artifact record has no path")
        path = root / relative
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    expected = record.get("sha256")
    if not isinstance(expected, str) or _sha256(resolved) != expected:
        raise ValueError(f"Artifact SHA-256 mismatch: {resolved}")
    return resolved


def _projection_seed(dataset: str, space: str, repeat: int) -> int:
    canonical = json.dumps(
        {
            "namespace": PROJECTION_NAMESPACE,
            "benchmark": dataset,
            "split": "full_data",
            "space": space,
            "repeat": int(repeat),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int.from_bytes(
        hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "little"
    )


def _projection_sha256(dimension: int, seed: int) -> str:
    projections = np.random.RandomState(seed).randn(dimension, N_PROJECTIONS)
    projections /= np.sqrt(np.sum(projections**2, axis=0, keepdims=True))
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(projections, dtype="<f8")).tobytes()
    ).hexdigest()


def _exact_ot_seed(dataset: str, space: str) -> int:
    canonical = json.dumps(
        {
            "namespace": f"{PROJECTION_NAMESPACE}-exact-ot",
            "benchmark": dataset,
            "split": "full_data",
            "space": space,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int.from_bytes(
        hashlib.sha256(canonical.encode("utf-8")).digest()[:4], "little"
    )


def _load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} lacks keys {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def _benchmark_input(path: Path, expected_sha: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    _require_hash(path, expected_sha, "benchmark input manifest")
    root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split = manifest.get("splits", {}).get("full_data")
    if (
        manifest.get("status") != "complete"
        or not isinstance(split, dict)
        or int(split.get("prediction_n", -1)) != PREDICTION_N
    ):
        raise ValueError("Benchmark input is not a complete full-data contract")
    train = split.get("train", {})
    training_path = _record_path(train["training_reference_npz"], root)
    roster_path = _record_path(train["source_roster_npz"], root)
    training = _load_npz(training_path, {"state", "spatial", "time", "row_id"})
    roster = _load_npz(
        roster_path, {"indices", "row_id", "source_time", "state", "spatial"}
    )
    targets = tuple(float(value) for value in split["evaluation_targets"])
    truth: dict[float, dict[str, np.ndarray]] = {}
    truth_paths: dict[float, Path] = {}
    by_time = split.get("truth_by_time_npz", {})
    for target in targets:
        key = str(int(target)) if target.is_integer() else str(target)
        truth_path = _record_path(by_time[key], root)
        truth[target] = _load_npz(truth_path, {"state", "spatial"})
        truth_paths[target] = truth_path
    if (
        np.asarray(training["state"]).shape[1] != STATE_DIM
        or np.asarray(training["spatial"]).shape[1] != SPATIAL_DIM
        or np.asarray(roster["state"]).shape != (PREDICTION_N, STATE_DIM)
        or np.asarray(roster["spatial"]).shape != (PREDICTION_N, SPATIAL_DIM)
    ):
        raise ValueError("Benchmark state/spatial dimensions are not 50+2")
    source_time = float(np.asarray(roster["source_time"]).reshape(-1)[0])
    return {
        "dataset": str(manifest["dataset_id"]),
        "source_time": source_time,
        "targets": targets,
        "training": {
            "state": np.asarray(training["state"], dtype=np.float32),
            "spatial": np.asarray(training["spatial"], dtype=np.float32),
            "time": np.asarray(training["time"], dtype=np.float64),
            "row_id": np.asarray(training["row_id"]).astype(str),
        },
        "roster": {
            "state": np.asarray(roster["state"], dtype=np.float32),
            "spatial": np.asarray(roster["spatial"], dtype=np.float32),
            "row_id": np.asarray(roster["row_id"]).astype(str),
        },
        "truth": truth,
        "provenance": {
            "mode": "canonical_unified_benchmark_full_data",
            "input_manifest": _artifact(path),
            "training_reference": _artifact(training_path),
            "source_roster": _artifact(roster_path),
            "truth": {
                str(target): _artifact(truth_paths[target]) for target in targets
            },
        },
    }


def _aligned_input(path: Path, expected_sha: str, dataset: str) -> dict[str, Any]:
    import anndata as ad

    path = path.expanduser().resolve()
    _require_hash(path, expected_sha, "aligned H5AD")
    config, source = workflow.load_workflow_config(dataset)
    dataset_config = config["dataset"]
    adata = ad.read_h5ad(path)
    dataframe, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=dataset_config.get("time_key"),
        obsm_key=str(dataset_config.get("obsm_key", "X_latent")),
        spatial_key=str(dataset_config.get("spatial_key", "spatial_aligned")),
        concat_spatial=True,
        annotation_key=str(dataset_config.get("annotation_key", "Annotation")),
    )
    features = cb.tl.infer_feature_columns(
        dataframe,
        annotation_column=str(dataset_config.get("annotation_key", "Annotation")),
    )
    if len(features) != STATE_DIM + SPATIAL_DIM:
        raise ValueError(f"Expected 52 aligned dimensions, found {len(features)}")
    joint = dataframe[features].to_numpy(dtype=np.float32)
    times = dataframe["samples"].to_numpy(dtype=np.float64)
    observed = tuple(float(value) for value in config["downstream"]["observed"])
    source_time = min(observed)
    targets = tuple(value for value in observed if value > source_time)
    source_indices = np.flatnonzero(np.isclose(times, source_time, rtol=0.0, atol=1e-8))
    rng = np.random.default_rng(SEED)
    roster_indices = rng.choice(
        source_indices,
        size=PREDICTION_N,
        replace=len(source_indices) < PREDICTION_N,
    )
    truth = {
        target: {
            "spatial": joint[
                np.isclose(times, target, rtol=0.0, atol=1e-8), :SPATIAL_DIM
            ],
            "state": joint[
                np.isclose(times, target, rtol=0.0, atol=1e-8), SPATIAL_DIM:
            ],
        }
        for target in targets
    }
    config_path = (
        Path(cb.__file__).resolve().parent / "workflow_configs" / f"{dataset}.json"
        if str(source).startswith("packaged preset:")
        else Path(source).resolve()
    )
    return {
        "dataset": dataset,
        "source_time": source_time,
        "targets": targets,
        "training": {
            "spatial": joint[:, :SPATIAL_DIM],
            "state": joint[:, SPATIAL_DIM:],
            "time": times,
            "row_id": np.asarray(adata.obs_names.astype(str)),
        },
        "roster": {
            "spatial": joint[roster_indices, :SPATIAL_DIM],
            "state": joint[roster_indices, SPATIAL_DIM:],
            "row_id": np.asarray(adata.obs_names.astype(str))[roster_indices],
        },
        "truth": truth,
        "provenance": {
            "mode": "aligned_h5ad_derived_full_data",
            "aligned_h5ad": _artifact(path),
            "workflow_config": _artifact(config_path),
            "resolved_time_key": str(resolved_time_key),
            "source_available_n": int(len(source_indices)),
            "source_sampling": "numpy.Generator(PCG64), seed=42",
            "sampled_with_replacement": bool(len(source_indices) < PREDICTION_N),
        },
    }


def _seed_runtime() -> None:
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@contextmanager
def _working_directory(path: Path):
    prior = Path.cwd()
    try:
        import os

        os.chdir(path)
        yield
    finally:
        import os

        os.chdir(prior)


def _bootstrap_adata(input_data: Mapping[str, Any]):
    import anndata as ad

    roster = input_data["roster"]
    source = float(input_data["source_time"])
    names = np.asarray(
        [f"bootstrap_{index:05d}" for index in range(PREDICTION_N)], dtype=str
    )
    obs = pd.DataFrame(
        {"time": np.full(PREDICTION_N, source, dtype=np.float32)},
        index=pd.Index(names, name="row_id"),
    )
    result = ad.AnnData(X=np.asarray(roster["state"], dtype=np.float32), obs=obs)
    result.obsm["X_latent"] = np.asarray(roster["state"], dtype=np.float32)
    result.obsm["spatial_aligned"] = np.asarray(roster["spatial"], dtype=np.float32)
    return result


def _interaction_off_model(base_model):
    import torch

    wrapped = getattr(base_model, "interaction_net", None)
    if wrapped is None:
        raise TypeError("Accepted full model has no interaction_net")

    class ZeroInteraction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.requires_time = bool(getattr(wrapped, "requires_time", False))

        def forward(self, x, *args, **kwargs):
            if self.requires_time:
                return x * 0.0
            return (x * 0.0).sum(dim=-1, keepdim=True)

    class InteractionOff(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base_model
            self.interaction_net = ZeroInteraction()
            self.components = list(getattr(base_model, "components", []))
            self.interaction_group_size = int(
                getattr(base_model, "interaction_group_size", 1024)
            )
            self.use_growth_in_ode_inter = bool(
                getattr(base_model, "use_growth_in_ode_inter", True)
            )

        def predict_velocity(self, *args, **kwargs):
            return self.base_model.predict_velocity(*args, **kwargs)

        def predict_growth(self, *args, **kwargs):
            return self.base_model.predict_growth(*args, **kwargs)

        def compute_score(self, *args, **kwargs):
            return self.base_model.compute_score(*args, **kwargs)

    return InteractionOff()


def _simulate_pair(
    input_data: Mapping[str, Any], model_dir: Path, device: str
) -> dict[str, dict[str, list[np.ndarray]]]:
    from CytoBridge.tl.downstream.simulation import simulate_sde_points

    bootstrap = _bootstrap_adata(input_data)
    times = [float(input_data["source_time"]), *map(float, input_data["targets"])]
    _seed_runtime()
    with _working_directory(REPO_ROOT):
        loaded = cb.tl.load_dynamical_model_from_dir(
            model_dir, dim=STATE_DIM + SPATIAL_DIM, device=device
        )
    if "interaction" not in {
        str(value).strip().lower() for value in loaded.model.components
    }:
        raise ValueError("Accepted model is not the full interaction arm")
    off_model = _interaction_off_model(loaded.model).to(device).eval()
    result: dict[str, dict[str, list[np.ndarray]]] = {}
    for arm, model in (
        ("interaction_on", loaded.model),
        ("interaction_off", off_model),
    ):
        _seed_runtime()
        points, weights = simulate_sde_points(
            adata=bootstrap,
            model=model,
            dim=STATE_DIM + SPATIAL_DIM,
            time_index=0,
            n_samples=PREDICTION_N,
            ts_points=times,
            dt=DT,
            sigma=SIGMA,
            include_score=True,
            interaction_m=INTERACTION_M,
            device=device,
            time_key="time",
            obsm_key="X_latent",
            spatial_key="spatial_aligned",
            concat_spatial=True,
            interaction_seed=INTERACTION_SEED,
            verbose=True,
        )
        result[arm] = {
            "points": [np.asarray(value, dtype=np.float32) for value in points],
            "weights": [
                np.asarray(value, dtype=np.float64).reshape(-1) for value in weights
            ],
        }
    return result


def _transform(training: Mapping[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    state = np.asarray(training["state"], dtype=np.float64)
    spatial = np.asarray(training["spatial"], dtype=np.float64)
    state_center = state.mean(axis=0)
    state_scale = state.std(axis=0, ddof=0)
    spatial_center = spatial.mean(axis=0)
    spatial_rms = float(np.sqrt(np.mean((spatial - spatial_center) ** 2)))
    if np.any(state_scale <= 0) or not math.isfinite(spatial_rms) or spatial_rms <= 0:
        raise ValueError("Frozen evaluation transform is degenerate")
    return {
        "state_center": state_center,
        "state_scale": state_scale,
        "spatial_center": spatial_center,
        "spatial_rms": spatial_rms,
    }


def _metric_rows(
    *,
    dataset: str,
    arm: str,
    target: float,
    points: np.ndarray,
    weights: np.ndarray,
    truth: Mapping[str, np.ndarray],
    training: Mapping[str, np.ndarray],
    transform: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import ot
    from scipy.spatial.distance import cdist

    predicted_state = (
        (points[:, SPATIAL_DIM:].astype(np.float64) - transform["state_center"])
        / transform["state_scale"]
        / math.sqrt(STATE_DIM)
    )
    observed_state = (
        (np.asarray(truth["state"], dtype=np.float64) - transform["state_center"])
        / transform["state_scale"]
        / math.sqrt(STATE_DIM)
    )
    predicted_spatial = (
        (points[:, :SPATIAL_DIM].astype(np.float64) - transform["spatial_center"])
        / float(transform["spatial_rms"])
        / math.sqrt(SPATIAL_DIM)
    )
    observed_spatial = (
        (np.asarray(truth["spatial"], dtype=np.float64) - transform["spatial_center"])
        / float(transform["spatial_rms"])
        / math.sqrt(SPATIAL_DIM)
    )
    spaces = {
        "joint": (
            np.concatenate((predicted_state, predicted_spatial), axis=1),
            np.concatenate((observed_state, observed_spatial), axis=1),
        ),
        "state": (predicted_state, observed_state),
        "spatial": (predicted_spatial, observed_spatial),
    }
    raw_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    normalized = raw_weights / raw_weights.sum()
    source_n = int(
        np.count_nonzero(
            np.isclose(
                np.asarray(training["time"], dtype=np.float64),
                np.min(np.asarray(training["time"], dtype=np.float64)),
                rtol=0.0,
                atol=1e-8,
            )
        )
    )
    observed_mass = float(len(observed_state) / source_n)
    predicted_mass = float(raw_weights.sum())
    tmv_absolute = abs(predicted_mass - observed_mass)
    tmv = tmv_absolute / observed_mass
    rows: list[dict[str, Any]] = []
    for space, (predicted, observed) in spaces.items():
        rng = np.random.default_rng(_exact_ot_seed(dataset, space))
        if len(predicted) > MAX_OT_POINTS:
            indices = rng.choice(
                len(predicted),
                size=MAX_OT_POINTS,
                replace=True,
                p=normalized,
            )
            exact_predicted = predicted[indices]
            exact_weights = np.full(MAX_OT_POINTS, 1.0 / MAX_OT_POINTS)
        else:
            exact_predicted = predicted
            exact_weights = normalized
        if len(observed) > MAX_OT_POINTS:
            indices = rng.choice(len(observed), size=MAX_OT_POINTS, replace=False)
            exact_observed = observed[indices]
        else:
            exact_observed = observed
        observed_exact_weights = np.full(len(exact_observed), 1.0 / len(exact_observed))
        distances = cdist(exact_predicted, exact_observed, metric="euclidean")
        exact_w1 = float(
            ot.emd2(
                exact_weights, observed_exact_weights, distances, numItermax=int(1e7)
            )
        )
        exact_w2 = math.sqrt(
            max(
                float(
                    ot.emd2(
                        exact_weights,
                        observed_exact_weights,
                        distances**2,
                        numItermax=int(1e7),
                    )
                ),
                0.0,
            )
        )
        observed_weights = np.full(len(observed), 1.0 / len(observed))
        dimension = int(predicted.shape[1])
        for repeat in range(PROJECTION_REPEATS):
            seed = _projection_seed(dataset, space, repeat)
            sliced_w2 = float(
                ot.sliced_wasserstein_distance(
                    predicted,
                    observed,
                    a=normalized,
                    b=observed_weights,
                    n_projections=N_PROJECTIONS,
                    p=2,
                    seed=seed,
                    log=False,
                )
            )
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "target": float(target),
                    "space": space,
                    "projection_repeat": repeat,
                    "projection_seed": seed,
                    "projection_sha256": _projection_sha256(dimension, seed),
                    "n_projections": N_PROJECTIONS,
                    "sliced_w2": sliced_w2,
                    "exact_w1": exact_w1,
                    "exact_w2": exact_w2,
                    "n_predicted": len(predicted),
                    "n_observed": len(observed),
                    "predicted_mass": predicted_mass,
                    "observed_mass_relative": observed_mass,
                    "tmv_absolute": tmv_absolute,
                    "tmv": tmv,
                    "weights_semantics": "native_unnormalised_growth_mass",
                }
            )
    return rows


def relative_tables(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_rows: list[dict[str, Any]] = []
    for (dataset, target, space), group in metrics.groupby(
        ["dataset", "target", "space"], sort=True
    ):
        values = group.groupby("arm", as_index=True)["sliced_w2"].mean().to_dict()
        on = float(values["interaction_on"])
        off = float(values["interaction_off"])
        target_rows.append(
            {
                "dataset": dataset,
                "target": target,
                "space": space,
                "interaction_on": on,
                "interaction_off": off,
                "off_minus_on": off - on,
                "off_relative_to_on": (off - on) / on,
            }
        )
    targets = pd.DataFrame(target_rows)
    summary = (
        targets.groupby(["dataset", "space"], as_index=False)
        .agg(
            n_targets=("target", "size"),
            mean_relative_change=("off_relative_to_on", "mean"),
            sem_relative_change=(
                "off_relative_to_on",
                lambda x: float(x.std(ddof=1) / math.sqrt(len(x)))
                if len(x) > 1
                else 0.0,
            ),
        )
        .sort_values(["dataset", "space"], ignore_index=True)
    )
    overall = (
        targets.groupby("dataset", as_index=False)
        .agg(
            mean_relative_change=("off_relative_to_on", "mean"),
            sem_relative_change=(
                "off_relative_to_on",
                lambda x: float(x.std(ddof=1) / math.sqrt(len(x)))
                if len(x) > 1
                else 0.0,
            ),
        )
        .sort_values("dataset", ignore_index=True)
    )
    tmv = metrics.groupby(["dataset", "target", "arm"], as_index=False)["tmv"].first()
    pivot = tmv.pivot(
        index=["dataset", "target"], columns="arm", values="tmv"
    ).reset_index()
    pivot.columns.name = None
    pivot["off_minus_on"] = pivot["interaction_off"] - pivot["interaction_on"]
    pivot["off_relative_to_on"] = np.where(
        pivot["interaction_on"] > 0,
        pivot["off_minus_on"] / pivot["interaction_on"],
        np.nan,
    )
    return (
        targets,
        summary.merge(overall, on="dataset", suffixes=("", "_overall")),
        pivot,
    )


def run_dataset(args: argparse.Namespace) -> Path:
    dataset = str(args.dataset)
    output = _new_directory(args.output_dir)
    model_dir = args.model_dir.expanduser().resolve()
    training_summary = model_dir / "training_run_summary.json"
    _require_hash(
        training_summary,
        args.expected_training_summary_sha256,
        "training run summary",
    )
    if args.benchmark_input_manifest is not None:
        input_data = _benchmark_input(
            args.benchmark_input_manifest, args.expected_benchmark_input_sha256
        )
    else:
        if args.aligned_h5ad is None or args.expected_aligned_sha256 is None:
            raise ValueError(
                "Aligned H5AD and its SHA-256 are required without a benchmark input"
            )
        input_data = _aligned_input(
            args.aligned_h5ad, args.expected_aligned_sha256, dataset
        )
    if input_data["dataset"] != dataset:
        raise ValueError("Dataset identity differs from input contract")
    simulated = _simulate_pair(input_data, model_dir, args.device)
    transform = _transform(input_data["training"])
    rows: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for arm in ARMS:
        arm_dir = output / arm
        arm_dir.mkdir()
        for index, target in enumerate(input_data["targets"], start=1):
            points = simulated[arm]["points"][index]
            weights = simulated[arm]["weights"][index]
            prediction = arm_dir / f"t{target:g}.npz"
            np.savez_compressed(
                prediction,
                spatial=points[:, :SPATIAL_DIM],
                state=points[:, SPATIAL_DIM:],
                weights=weights,
                source_time=np.asarray([input_data["source_time"]]),
                target_time=np.asarray([target]),
            )
            outputs.append(prediction)
            rows.extend(
                _metric_rows(
                    dataset=dataset,
                    arm=arm,
                    target=target,
                    points=points,
                    weights=weights,
                    truth=input_data["truth"][target],
                    training=input_data["training"],
                    transform=transform,
                )
            )
    metrics = pd.DataFrame(rows)
    metrics_path = output / "weighted_metrics_long.csv"
    metrics.to_csv(metrics_path, index=False, float_format="%.12g")
    outputs.append(metrics_path)
    target, summary, tmv = relative_tables(metrics)
    for path, frame in (
        (output / "target_relative_sliced_w2.csv", target),
        (output / "dataset_space_summary.csv", summary),
        (output / "tmv_relative.csv", tmv),
    ):
        frame.to_csv(path, index=False, float_format="%.12g")
        outputs.append(path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _utc_now(),
        "dataset": dataset,
        "claim_scope": "fixed-checkpoint inference-only interaction sensitivity",
        "training_performed": False,
        "model_dir": str(model_dir),
        "training_summary": _artifact(training_summary),
        "input": input_data["provenance"],
        "protocol": {
            "arms": list(ARMS),
            "same_full_model_checkpoint": True,
            "only_interaction_force_disabled": True,
            "source_roster_n": PREDICTION_N,
            "source_time": input_data["source_time"],
            "targets": list(input_data["targets"]),
            "single_continuous_non_split_call": True,
            "intermediate_reset": False,
            "same_global_seed": SEED,
            "same_interaction_grouping_seed": INTERACTION_SEED,
            "simulation_mode": "continuous_non_split_weighted_sde",
            "weights_semantics": "native_unnormalised_growth_mass",
            "dt": DT,
            "sigma": SIGMA,
            "primary_metric": "weighted_sliced_w2",
            "n_projections": N_PROJECTIONS,
            "projection_repeats": PROJECTION_REPEATS,
            "exact_ot_max_points": MAX_OT_POINTS,
            "interpretation": (
                "positive off-relative error means interaction helps reconstruction; "
                "negative means inference-only removal improves reconstruction"
            ),
        },
        "runner": _artifact(Path(__file__)),
        "outputs": [_artifact(path) for path in outputs],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json.sha256").write_text(
        f"{_sha256(manifest_path)}  manifest.json\n", encoding="utf-8"
    )
    print(manifest_path)
    return manifest_path


def _style() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
    }


def _report_figure(summary: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    order = list(DATASETS)
    x = np.arange(len(order))
    overall = summary.drop_duplicates("dataset").set_index("dataset").loc[order]
    colors = [
        "#C85C5C" if value > 0 else "#07838B"
        for value in overall["mean_relative_change_overall"]
    ]
    with mpl.rc_context(_style()):
        figure, axes = plt.subplots(2, 1, figsize=(8.27, 8.2))
        values = 100.0 * overall["mean_relative_change_overall"].to_numpy()
        errors = 100.0 * overall["sem_relative_change_overall"].to_numpy()
        axes[0].bar(x, values, yerr=errors, color=colors, capsize=3)
        axes[0].axhline(0, color="#24313A", linewidth=0.8)
        axes[0].set_xticks(x, [DISPLAY_NAMES[value] for value in order])
        axes[0].set_ylabel("Off vs on weighted sliced-W2 (%)")
        axes[0].set_title(
            "Overall change pooled across target times and spaces", fontweight="bold"
        )
        width = 0.23
        palette = {"joint": "#3B6FB6", "state": "#8E6BBE", "spatial": "#52A675"}
        for index, space in enumerate(SPACES):
            space_frame = (
                summary.loc[summary["space"].eq(space)].set_index("dataset").loc[order]
            )
            axes[1].bar(
                x + (index - 1) * width,
                100.0 * space_frame["mean_relative_change"].to_numpy(),
                width=width,
                color=palette[space],
                label=space.capitalize(),
            )
        axes[1].axhline(0, color="#24313A", linewidth=0.8)
        axes[1].set_xticks(x, [DISPLAY_NAMES[value] for value in order])
        axes[1].set_ylabel("Off vs on weighted sliced-W2 (%)")
        axes[1].set_title("Change by evaluation space", fontweight="bold")
        axes[1].legend(frameon=False, ncol=3)
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#D8DEE3", linewidth=0.6)
        figure.text(0.025, 0.985, "a", fontsize=14, fontweight="bold", va="top")
        figure.text(0.025, 0.49, "b", fontsize=14, fontweight="bold", va="top")
        figure.tight_layout(rect=(0.04, 0.03, 0.99, 0.98), h_pad=3.0)
        pdf = output / "weighted_interaction_off_relative_effect.pdf"
        png = output / "weighted_interaction_off_relative_effect.png"
        figure.savefig(pdf, bbox_inches="tight")
        figure.savefig(png, dpi=320, bbox_inches="tight")
        plt.close(figure)
    return pdf, png


def report(args: argparse.Namespace) -> Path:
    root = args.run_root.expanduser().resolve()
    output = _new_directory(args.output_dir)
    runner_sha = _sha256(Path(__file__))
    metrics: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for dataset in DATASETS:
        manifest_path = root / dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("dataset") != dataset
            or manifest.get("runner", {}).get("sha256") != runner_sha
            or manifest.get("training_performed") is not False
            or manifest.get("protocol", {}).get("weights_semantics")
            != "native_unnormalised_growth_mass"
        ):
            raise ValueError(f"Dataset result violates weighted protocol: {dataset}")
        manifests.append(_artifact(manifest_path))
        metrics.append(pd.read_csv(root / dataset / "weighted_metrics_long.csv"))
    combined = pd.concat(metrics, ignore_index=True)
    target, summary, tmv = relative_tables(combined)
    overall = summary.drop_duplicates("dataset")[
        [
            "dataset",
            "mean_relative_change_overall",
            "sem_relative_change_overall",
        ]
    ].copy()
    overall = overall.rename(
        columns={
            "mean_relative_change_overall": "mean_relative_change",
            "sem_relative_change_overall": "sem_relative_change",
        }
    )
    overall["direction"] = np.where(
        overall["mean_relative_change"] > 0,
        "interaction_off_worse",
        "interaction_off_better",
    )
    files: list[Path] = []
    for path, frame in (
        (output / "weighted_metrics_long.csv", combined),
        (output / "target_relative_sliced_w2.csv", target),
        (output / "dataset_space_summary.csv", summary),
        (output / "overall_direction_summary.csv", overall),
        (output / "tmv_relative.csv", tmv),
    ):
        frame.to_csv(path, index=False, float_format="%.12g")
        files.append(path)
    pdf, png = _report_figure(summary, output)
    files.extend((pdf, png))
    caption = output / "CAPTION.md"
    lines = [
        "# Fixed-checkpoint weighted interaction on/off sensitivity",
        "",
        "Positive values mean that turning the learned interaction force off increased weighted sliced-W2 reconstruction error; negative values mean it decreased error. Both branches use the same accepted full-model checkpoint, 5,000-particle source roster, global seed, interaction-grouping seed, continuous non-split weighted SDE, frozen transform, observed truth, and projection bases. Native unnormalised growth mass weights the predicted distribution. This is an inference-only sensitivity, not retraining or a causal knockout.",
        "",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"- {DISPLAY_NAMES[row.dataset]}: {100.0 * row.mean_relative_change:+.2f}% ({row.direction})."
        )
    caption.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.append(caption)
    provenance = output / "PROVENANCE.md"
    provenance.write_text(
        "\n".join(
            (
                "# Provenance",
                "",
                "## Source paths",
                f"- Run root: `{root}`",
                f"- Runner: `{Path(__file__).resolve()}`",
                "",
                "## Rebuild",
                "Run the report subcommand against the same immutable run root.",
                "",
            )
        ),
        encoding="utf-8",
    )
    files.append(provenance)
    report_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": _utc_now(),
        "dataset_manifests": manifests,
        "runner": _artifact(Path(__file__)),
        "outputs": [_artifact(path) for path in files],
    }
    manifest_path = output / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(report_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "report_manifest.json.sha256").write_text(
        f"{_sha256(manifest_path)}  report_manifest.json\n", encoding="utf-8"
    )
    print(manifest_path)
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run one weighted interaction on/off pair")
    run.add_argument("--dataset", choices=DATASETS, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--expected-training-summary-sha256", required=True)
    run.add_argument("--benchmark-input-manifest", type=Path)
    run.add_argument("--expected-benchmark-input-sha256")
    run.add_argument("--aligned-h5ad", type=Path)
    run.add_argument("--expected-aligned-sha256")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    report_parser = sub.add_parser("report", help="Aggregate all five datasets")
    report_parser.add_argument("--run-root", type=Path, required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        run_dataset(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
