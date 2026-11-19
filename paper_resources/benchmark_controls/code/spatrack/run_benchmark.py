"""Test SpaTrack's multi-time transport solver with the shared benchmark PCA features.

The distance change is explicit: Euclidean PCA distances replace expression KL.
The official USOT solver, spatial term, and expression-derived marginals are used
unchanged. A barycentric interpolation supplies the held-out-stage prediction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import ot
import pandas as pd
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
DEFAULT_BENCHMARK_CODE = next(
    (parent for parent in ROOT.parents if (parent / "CytoBridge/tl/downstream/benchmark.py").is_file()),
    WORKSPACE / "worktrees/package-final-audit-20260829",
)
BENCHMARK_CODE = Path(os.environ.get("CYTOBRIDGE_BENCHMARK_CODE", DEFAULT_BENCHMARK_CODE))
sys.path.insert(0, str(BENCHMARK_CODE))

from CytoBridge.tl.downstream.benchmark import (  # noqa: E402
    evaluate_spatiotemporal_prediction,
    fit_frozen_benchmark_transform,
)
from scripts.spatiotemporal_benchmark.static_baselines.data import (  # noqa: E402
    _ranked_indices,
)
from scripts.spatiotemporal_benchmark.static_baselines.coupling import (  # noqa: E402
    validate_and_row_normalize,
)


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def official_solver(source: Path):
    spec = importlib.util.spec_from_file_location("spatrack_official_utils", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_and_predict(target: int, output: Path, solver, args) -> dict:
    """Only training arrays and the pre-existing source roster are read here."""
    split = args.input_root / f"loto_t{target}"
    train = read_arrays(split / "training_reference.npz")
    roster = read_arrays(split / "source_roster.npz")
    manifest = json.loads((split / "manifest.json").read_text())
    times = sorted(np.unique(train["time"]).tolist())
    assert target not in times
    left_time = max(value for value in times if value < target)
    right_time = min(value for value in times if value > target)
    fraction = (target - left_time) / (right_time - left_time)
    stages = []
    for stage_time in (left_time, right_time):
        candidates = np.flatnonzero(train["time"] == stage_time)
        selected = _ranked_indices(
            train["row_id"][candidates], 800,
            int(manifest["source_roster_seed"]), stage_time,
        )
        stages.append(candidates[selected])
    left, right = stages
    left_ids = train["row_id"][left]
    np.testing.assert_array_equal(left_ids, roster["support_row_id"])
    # Stored indices refer to the full training table, not the source-stage slice.
    lookup = {row_id: index for index, row_id in enumerate(left_ids)}
    roster_positions = np.array([lookup[row_id] for row_id in roster["row_id"]])
    np.testing.assert_array_equal(left_ids[roster_positions], roster["row_id"])
    np.testing.assert_array_equal(
        train["state"][left][roster_positions], roster["state"]
    )
    assert len(roster["indices"]) == manifest["prediction_n"] == 5000

    start = time.perf_counter()
    cost = solver.gene_dist(
        train["state"][left].astype(float),
        train["state"][right].astype(float), gene_method="euclidean",
    )
    cost_scale = float(cost.max())
    if not np.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("PCA cost has no finite positive scale")
    cost /= cost_scale
    spatial_costs = []
    for selected in (left, right):
        geometry = ot.dist(
            train["spatial"][selected].astype(float),
            train["spatial"][selected].astype(float), metric="euclidean",
        )
        spatial_costs.append(geometry / geometry.max())

    # Same marginals as official transfer_matrix, using the adapted PCA cost.
    affinities = np.exp(1 - cost)
    p = affinities.sum(axis=1)
    q = affinities.sum(axis=0)
    p /= p.sum()
    q /= q.sum()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        plan = solver.usot(
            p, q, cost, spatial_costs[0], spatial_costs[1], args.alpha,
            epsilon=args.epsilon, niter=10, rho=np.inf,
        )
    row_plan, checks = validate_and_row_normalize(plan, (len(left), len(right)))
    joint = np.concatenate([train["spatial"], train["state"]], axis=1)
    predicted_support = (
        (1 - fraction) * joint[left].astype(float)
        + fraction * (row_plan @ joint[right].astype(float))
    ).astype(np.float32)
    predicted = predicted_support[roster_positions]
    elapsed = time.perf_counter() - start
    spatial_dim = train["spatial"].shape[1]
    np.savez_compressed(
        output / "prediction.npz", spatial=predicted[:, :spatial_dim], state=predicted[:, spatial_dim:],
        row_id=roster["row_id"],
    )
    np.savez_compressed(
        output / "coupling.npz", plan=plan, left_row_id=left_ids,
        right_row_id=train["row_id"][right], left_marginal=p, right_marginal=q,
    )
    details = {
        "dataset": args.dataset, "target": target,
        "anchor_times": [left_time, right_time], "anchor_counts": [len(left), len(right)],
        "prediction_n": len(predicted), "interpolation_fraction": fraction,
        "fit_seconds": elapsed, "pca_distance_max": cost_scale,
        "marginal_l1_error_left": float(np.abs(plan.sum(axis=1) - p).sum()),
        "marginal_l1_error_right": float(np.abs(plan.sum(axis=0) - q).sum()),
        "warnings": list(dict.fromkeys(str(item.message) for item in recorded)),
        "coupling_checks": asdict(checks),
        "training_input": str(split / "training_reference.npz"),
        "source_roster": str(split / "source_roster.npz"),
        "native_growth_prediction": False,
    }
    save_json(output / "fit_summary.json", details)
    print(f"t{target}: prediction saved, {elapsed:.1f} s, warnings={details['warnings']}", flush=True)
    return details


def evaluate(target: int, output: Path, method: str, args) -> pd.DataFrame:
    """Held-out observations enter only after all predictions have been saved."""
    split = args.input_root / f"loto_t{target}"
    train = read_arrays(split / "training_reference.npz")
    truth = read_arrays(split / f"truth_t{target}.npz")
    prediction = read_arrays(output / "prediction.npz")
    transform = fit_frozen_benchmark_transform(train["state"], train["spatial"])
    (output / "evaluation_transform.json").write_text(transform.to_json() + "\n")
    metrics = evaluate_spatiotemporal_prediction(
        transform=transform, benchmark=args.dataset, split=f"loto_t{target}",
        method=method, predicted_state=prediction["state"], observed_state=truth["state"],
        predicted_spatial=prediction["spatial"], observed_spatial=truth["spatial"],
        n_projections=1024, projection_repeats=5, max_ot_points=800,
    )
    metrics["target"] = target
    metrics["dataset"] = args.dataset
    metrics.to_csv(output / "metrics.csv", index=False)
    print(f"t{target}: evaluation finished", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="chicken_heart")
    parser.add_argument("--input-root", type=Path, default=ROOT / "inputs")
    parser.add_argument("--targets", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--official-source", type=Path, default=ROOT / "vendor/spaTrack/spaTrack/multiple_time/utils.py")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1 or args.epsilon <= 0:
        parser.error("alpha must be in [0,1] and epsilon must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    method = "SpaTrack (PCA/Euclidean)"
    settings = {
        "method": method, "alpha": args.alpha, "epsilon": args.epsilon,
        "niter": 10, "rho": "infinity", "distance": "Euclidean PCA / maximum anchor distance",
        "interpolation": "row-normalized barycentric mapping, then linear in benchmark time",
        "representation": "existing transductive frozen PCA and aligned spatial coordinates",
        "official_repository": "https://github.com/yzf072/spaTrack",
        "official_commit": "1cc5edec3699f7d8e29663ce3bb0c02cad5600db",
        "official_solver_source": str(args.official_source.resolve()),
        "dataset": args.dataset, "targets": args.targets,
        "input_root": str(args.input_root.resolve()),
        "evaluator_code": str(BENCHMARK_CODE / "CytoBridge/tl/downstream/benchmark.py"),
        "python": platform.python_version(), "numpy": np.__version__, "pot": ot.__version__,
        "status": "running",
    }
    save_json(output / "run.json", settings)
    with threadpool_limits(limits=args.threads):
        solver = official_solver(args.official_source)
        fits = []
        for target in args.targets:
            folder = output / f"t{target}"
            folder.mkdir()
            fits.append(fit_and_predict(target, folder, solver, args))
        metrics = pd.concat(
            [evaluate(t, output / f"t{t}", method, args) for t in args.targets], ignore_index=True
        )
    metrics.to_csv(output / "metrics_long.csv", index=False)
    metrics.groupby(["method", "target", "space"], sort=False)[
        ["sliced_w2", "exact_w1", "exact_w2"]
    ].mean().reset_index().to_csv(output / "metrics_means.csv", index=False)
    settings.update(status="complete", fits=fits)
    save_json(output / "run.json", settings)
    print(f"Saved results in {output}", flush=True)


if __name__ == "__main__":
    main()
