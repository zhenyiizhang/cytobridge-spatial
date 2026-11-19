"""Compare one fitted CytoBridge model with and without interaction at inference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-metrics", action="store_true")
    parser.add_argument("--reuse-predictions", type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Inference seeds must be distinct")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.code_root.resolve()))
    source = Path(__file__).with_name("weighted_simulation.py")
    spec = importlib.util.spec_from_file_location("paired_interaction", source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.REPO_ROOT = args.code_root.resolve()
    import torch
    torch.set_num_threads(4)

    input_record = fingerprint(args.input_manifest)
    inputs = core._benchmark_input(args.input_manifest, input_record["sha256"])
    if inputs["dataset"] != args.dataset:
        raise ValueError("The requested dataset differs from the input manifest")
    training_summary = args.model_dir / "training_run_summary.json"
    training = json.loads(training_summary.read_text())
    input_manifest = json.loads(args.input_manifest.read_text())
    trained_input = training["data"].get("input_h5ad", {})
    accepted_input = input_manifest["source"].get("h5ad_sha256")
    if trained_input.get("sha256") != accepted_input:
        raise ValueError("Training and evaluation refer to different aligned inputs")

    transform = core._transform(inputs["training"])
    np.savez_compressed(args.output / "initial_cells.npz", **inputs["roster"])
    all_rows = []
    checks = []
    for seed in args.seeds:
        print(f"{args.dataset}: paired inference seed {seed}", flush=True)
        core.SEED = seed
        core.INTERACTION_SEED = 10000 + seed
        pair = None
        if args.reuse_predictions:
            previous = args.reuse_predictions
            paths = [previous / f"seed_{seed}" / arm / f"t{target:g}.npz"
                     for arm in core.ARMS for target in inputs["targets"]]
            if all(path.is_file() for path in paths):
                try:
                    with np.load(previous / "initial_cells.npz", allow_pickle=False) as original:
                        for key in inputs["roster"]:
                            np.testing.assert_array_equal(original[key], inputs["roster"][key])
                    pair = {}
                    for arm in core.ARMS:
                        point_frames, weight_frames = [None], [None]
                        for target in inputs["targets"]:
                            with np.load(previous / f"seed_{seed}" / arm / f"t{target:g}.npz") as frame:
                                assert float(frame["target_time"]) == target
                                point_frames.append(np.concatenate((frame["spatial"], frame["state"]), 1))
                                weight_frames.append(frame["weights"].copy())
                        pair[arm] = {"points": point_frames, "weights": weight_frames}
                    checks.append({"seed": seed, "initial_roster_matches": True,
                                   "inference_source": str(previous / f"seed_{seed}"),
                                   "paired_initial_state_checks": "passed in the original run before predictions were saved"})
                except (OSError, ValueError, EOFError):
                    pair = None
        if pair is None:
            pair = core._simulate_pair(inputs, args.model_dir, args.device)
            np.testing.assert_array_equal(pair["interaction_on"]["points"][0],
                                          pair["interaction_off"]["points"][0])
            np.testing.assert_array_equal(pair["interaction_on"]["weights"][0],
                                          pair["interaction_off"]["weights"][0])
            checks.append({"seed": seed, "initial_states_identical": True,
                           "initial_weights_identical": True})
        seed_rows = []
        for arm in core.ARMS:
            destination = args.output / f"seed_{seed}" / arm
            destination.mkdir(parents=True)
            for index, target in enumerate(inputs["targets"], start=1):
                points = pair[arm]["points"][index]
                weights = pair[arm]["weights"][index]
                if not np.isfinite(points).all() or not np.isfinite(weights).all():
                    raise ValueError(f"Nonfinite prediction: {arm}, seed {seed}, t={target}")
                if (weights < 0).any() or weights.sum() <= 0:
                    raise ValueError("Prediction weights are invalid")
                np.savez_compressed(destination / f"t{target:g}.npz",
                                    spatial=points[:, :2], state=points[:, 2:],
                                    weights=weights, source_time=inputs["source_time"],
                                    target_time=target)
                kwargs = dict(dataset=args.dataset, arm=arm, target=target,
                              points=points, weights=weights, truth=inputs["truth"][target],
                              training=inputs["training"], transform=transform)
                if args.gpu_metrics:
                    from accelerated_metrics import metric_rows
                    rows = metric_rows(core, **kwargs, device=args.device)
                else:
                    rows = core._metric_rows(**kwargs)
                for row in rows:
                    row["inference_seed"] = seed
                seed_rows.extend(rows)
        pd.DataFrame(seed_rows).to_csv(args.output / f"metrics_seed_{seed}.csv", index=False)
        all_rows.extend(seed_rows)
        del pair
        torch.cuda.empty_cache()
        print(f"{args.dataset}: seed {seed} complete", flush=True)

    metrics_path = args.output / "metrics.csv"
    pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
    checkpoints = [fingerprint(path) for path in sorted(args.model_dir.rglob("*.pth"))]
    manifest = {
        "status": "complete", "dataset": args.dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(args.model_dir), "training_performed": False,
        "training_summary": fingerprint(training_summary),
        "checkpoints": checkpoints, "input": inputs["provenance"],
        "aligned_input_matches_training": True,
        "runner": fingerprint(Path(__file__)), "upstream_runner": fingerprint(source),
        "metric_backend": "POT float64 GPU, 64 projection directions per batch" if args.gpu_metrics else "POT NumPy",
        "accelerated_metrics_source": fingerprint(Path(__file__).parent / "accelerated_metrics.py") if args.gpu_metrics else None,
        "package_root": str(args.code_root), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "protocol": {"intervention": "interaction network output set to zero at inference",
                     "unchanged": ["checkpoint", "initial cells", "intrinsic drift network",
                                   "growth network", "score network", "diffusion", "evaluation transform"],
                     "inference_seeds": args.seeds, "paired_random_streams": True,
                     "growth": "continuous mass weights retained; no resampling",
                     "evaluation": "full-data reconstruction against observed snapshots",
                     "n_initial": core.PREDICTION_N, "dt": core.DT, "sigma": core.SIGMA,
                     "targets": list(inputs["targets"]), "n_projections": core.N_PROJECTIONS,
                     "projection_repeats": core.PROJECTION_REPEATS,
                     "same_projection_directions_across_arms_and_seeds": True},
        "paired_checks": checks, "metrics": fingerprint(metrics_path),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

