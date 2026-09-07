#!/usr/bin/env python3
"""Generate and evaluate the two distinct zebrafish S6 trajectory populations.

The growing split population supplies composition frames; an independent,
non-split rollout supplies fixed-row transitions. Neither uses held-out labels
to choose a trajectory classifier or changes a fitted dynamics model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.results.classifier_smoothing import (
    DATASET_ORDER, FORMAL_K, K_VALUES, load_classifier_smoothing_results,
)


def artifact(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def write_record(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def raw_labels(classifier, points, time):
    from CytoBridge.tl.downstream.classification import predict_labels_for_points
    return predict_labels_for_points(
        points=points, time_value=float(time), model=classifier.model,
        label_encoder=classifier.label_encoder, feature_dim=12, device="cpu",
        knn_neighbors=1, include_time_feature=True, feature_indices=list(range(12)),
        spatial_coords=points[:, :2], spatial_indices=(0, 1),
    ).astype(str)


def smooth(raw, points, k):
    from CytoBridge.tl.downstream.classification import smooth_spatial_labels
    return smooth_spatial_labels(raw, points[:, :2], k=k, include_self=True,
                                 weights="uniform", tie_policy="sklearn_legacy").astype(str)


def trajectory_classifier(h5ad, output_dir, device):
    import anndata as ad
    from CytoBridge.tl.downstream.classification import train_cached_mlp_classifier_from_adata
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _, checkpoint = train_cached_mlp_classifier_from_adata(
        ad.read_h5ad(h5ad), cache_path=output / "classifier.pt", reuse_if_compatible=False,
        cache_tag="s6-trajectory-spatial2-latent10", label_col="Annotation",
        time_key="time_point_processed", obsm_key="X_latent", spatial_key="spatial_aligned",
        concat_spatial=True, n_features=12, include_time_feature=True, hidden_size=128,
        epochs=500, lr=.001, test_size=.1, seed=42, device=device,
        best_epoch_metric="accuracy", train_on_full_data=False, stratify_split=True,
    )
    write_record(output / "training_manifest.json", {"input_h5ad": artifact(h5ad),
                 "classifier": artifact(checkpoint), "purpose": "S6b/c trajectory labels; not the S6a k-selection classifier"})
    return checkpoint


def _validate_classifier(classifier):
    expected = ("samples", *(f"x{i}" for i in range(1, 13)))
    if tuple(classifier.feature_cols) != expected or not classifier.include_time_feature:
        raise ValueError("S6 trajectories require time + spatial2 + latent10 classifier features")
    if classifier.metadata.get("best_epoch_metric") != "accuracy" or classifier.metadata.get("train_on_full_data") is not False:
        raise ValueError("S6 trajectory classifier must use accuracy-selected held-out training, not a full-data/ablation classifier")


def generate(h5ad, model_dir, classifier_cache, output_dir, device):
    import anndata as ad
    import CytoBridge as cb
    from CytoBridge.tl.downstream.classification import load_cached_mlp_classifier

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    adata = ad.read_h5ad(h5ad)
    times = adata.obs["time_point_processed"].to_numpy(float)
    points = np.column_stack((adata.obsm["spatial_aligned"], adata.obsm["X_latent"])).astype(np.float32)
    if points.shape[1] != 52 or np.count_nonzero(np.isclose(times, 0)) != 563:
        raise ValueError("S6 requires the zebrafish 52-feature reference and its ordered 563-cell t0 cohort")
    loaded = cb.tl.load_dynamical_model_from_dir(model_dir, dim=52, device=device)
    loaded.model.eval()
    runtime = cb.tl.build_dynamical_runtime(loaded)
    classifier = load_cached_mlp_classifier(str(classifier_cache), device="cpu")
    _validate_classifier(classifier)
    df = pd.DataFrame(points, columns=[f"x{i}" for i in range(1, 53)])
    df.insert(0, "samples", times)
    common = dict(df=df, dim=52, f_net=runtime.f_net, score_net=runtime.score_net,
                  time_index=0, n_samples=5000, sigma=.03, interaction_m=1024,
                  device=device, verbose=False)
    cb.tl.set_global_random_seed(42)
    # The original S6 composition experiment used the unwarped, global-t0
    # growing population, not today's fixed-population S22 display trajectory.
    grid = np.round(np.arange(0, 4.0001, .1), 8).tolist()
    split = cb.tl.simulate_sde_points_split(
        **common, ts_points=grid, dt=.05, growth_alpha=1., resample_dt=.05,
        max_particles=100000, daughter_noise_std=0.,
    )
    frame_dir = output / "generated_frames"
    frame_dir.mkdir()
    frames = []
    for i, time in enumerate(np.arange(0, 4.01, .5)):
        frame = np.asarray(split[int(round(time * 10))], dtype=np.float32)
        labels = smooth(raw_labels(classifier, frame, time), frame, 10)
        path = frame_dir / f"frame_{i:03d}.npz"
        np.savez_compressed(path, points=frame, labels=labels)
        frames.append(dict(time=float(time), file=path.name, sha256=artifact(path)["sha256"]))
    write_record(frame_dir / "index.json", {"frames": frames})
    del split
    cb.tl.set_global_random_seed(42)
    cohort, weights = cb.tl.simulate_sde_points(
        **common, ts_points=[0, 1, 2, 3, 4], dt=.01, include_score=True,
    )
    fixed = np.stack([np.asarray(frame, dtype=np.float32) for frame in cohort])
    np.savez_compressed(output / "fixed_cohort.npz", times=np.arange(5.), points=fixed,
                        weights=weights, particle_ids=np.arange(563))
    write_record(output / "generation_manifest.json", {
        "operation": "new full-model simulations; no model or classifier fitting",
        "h5ad": artifact(h5ad), "model": artifact(loaded.weight_path),
        "score": artifact(loaded.score_path), "classifier": artifact(classifier_cache),
        "seed": 42, "sigma": .03, "interaction_m": 1024,
        "frames": {"population": "global-t0 split-growth", "dt": .05, "resample_dt": .05,
                   "growth_alpha": 1., "daughter_noise_std": 0., "max_particles": 100000,
                   "simulation_grid": grid, "index": str(frame_dir / "index.json")},
        "cohort": {"population": "global-t0 fixed-row non-split", "dt": .01,
                   "n_particles": 563, "trajectory": str(output / "fixed_cohort.npz")},
    })
    return output


def evaluate(state_index, trajectory, classifier_cache, output_dir):
    from CytoBridge.tl.downstream.classification import (
        analyze_spatial_label_sensitivity, load_cached_mlp_classifier,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    index_path = Path(state_index).resolve()
    index = json.loads(index_path.read_text())
    if sorted(float(frame["time"]) for frame in index["frames"]) != np.arange(0, 4.01, .5).tolist():
        raise ValueError("S6 composition requires nine half-time frames")
    classifier = load_cached_mlp_classifier(str(classifier_cache), device="cpu")
    _validate_classifier(classifier)
    rows, frame_records = [], []
    t0_points = t0_labels = None
    for record in index["frames"]:
        time = float(record["time"])
        path = index_path.parent / record["file"]
        source = artifact(path)
        if record.get("sha256") and record["sha256"] != source["sha256"]:
            raise ValueError(f"Frame hash mismatch: {path}")
        frame_records.append(source)
        with np.load(path, allow_pickle=False) as data:
            points, formal = data["points"], data["labels"].astype(str)
        raw = raw_labels(classifier, points, time)
        if not np.array_equal(smooth(raw, points, 10), formal):
            raise ValueError(f"The selected classifier does not reproduce k=10 at t={time}")
        if time == 0:
            t0_points, t0_labels = points, formal
        analysis = analyze_spatial_label_sensitivity(raw, points[:, :2], k_values=K_VALUES,
            include_self=True, weights="uniform", tie_policy="sklearn_legacy")
        for result in analysis["results"]:
            rows.append(dict(dataset="zebrafish", time=time, n_samples=len(points),
                             requested_k=result["requested_k"],
                             composition_tv=result["composition_total_variation"],
                             changed_fraction=result["changed_fraction"]))
    with np.load(trajectory, allow_pickle=False) as data:
        times, points = np.asarray(data["times"], float), np.asarray(data["points"], np.float32)
    if points.shape != (5, 563, 52) or not np.array_equal(times, np.arange(5.)):
        raise ValueError("S6 transition input must be 5 times × 563 fixed particles × 52 features")
    if not np.array_equal(points[0], t0_points):
        raise ValueError("Fixed-cohort and generated-frame ordered t0 cells differ")
    labels = {k: [] for k in K_VALUES}
    for time, frame in zip(times, points):
        raw = raw_labels(classifier, frame, time)
        for k in K_VALUES:
            labels[k].append(smooth(raw, frame, k))
    if not np.array_equal(labels[10][0], t0_labels):
        raise ValueError("Fixed-cohort initial labels disagree with generated-frame labels")
    intervals = [dict(time_from=float(times[i]), time_to=float(times[i+1]), k=k,
                      particle_count=563, transition_fraction=float(np.mean(labels[k][i] != labels[k][i+1])))
                 for i in range(4) for k in K_VALUES]
    pd.DataFrame(rows).to_csv(output / "frame_sensitivity.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "transition_by_interval.csv", index=False)
    write_record(output / "evaluation_manifest.json", {
        "operation": "classifier inference and spatial voting on caller-selected states",
        "state_index": artifact(index_path), "frames": frame_records,
        "trajectory": artifact(trajectory), "classifier": artifact(classifier_cache),
        "formal_k10_matches": 9, "fixed_cohort_t0_matches": True,
    })
    return output


def collect(heldout_root, generated_results, output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    heldout_root, generated_results = Path(heldout_root), Path(generated_results)
    metrics, selections, sources = [], {}, []
    for dataset in DATASET_ORDER:
        path = heldout_root / dataset / "k_metrics.csv"
        metrics.append(pd.read_csv(path))
        sources.append(artifact(path))
        selection_path = path.parent / "selection.json"
        selections[dataset] = json.loads(selection_path.read_text())
        sources.append(artifact(selection_path))
    pd.concat(metrics, ignore_index=True).to_csv(output / "five_dataset_k_metrics.csv", index=False)
    pd.DataFrame([dict(dataset=d, accuracy_best_k=selections[d]["selected_k"],
                       formal_analysis_k=FORMAL_K[d]) for d in DATASET_ORDER]).to_csv(
                           output / "formal_k_policy.csv", index=False)
    write_record(output / "arista_selection.json", selections["arista"])
    write_record(output / "heart_selection.json", selections["chicken_heart"])
    for name in ("frame_sensitivity.csv", "transition_by_interval.csv"):
        sources.append(artifact(generated_results / name))
        shutil.copyfile(generated_results / name, output / name)
    write_record(output / "manifest.json", {"analysis": "classifier_smoothing", "sources": sources,
        "operation": "collect newly evaluated tables; preserve predeclared formal k policy"})
    # A result whose best-k choices differ remains a valid new experiment, but
    # cannot silently be drawn with the paper's fixed policy annotations.
    load_classifier_smoothing_results(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("trajectory-classifier")
    training.add_argument("--h5ad", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--device", default="cuda")
    gen = sub.add_parser("generate")
    gen.add_argument("--h5ad", type=Path, required=True)
    gen.add_argument("--model-dir", type=Path, required=True)
    gen.add_argument("--classifier-cache", type=Path, required=True)
    gen.add_argument("--device", default="cuda")
    gen.add_argument("--output-dir", type=Path, required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--state-index", type=Path, required=True)
    ev.add_argument("--trajectory", type=Path, required=True)
    ev.add_argument("--classifier-cache", type=Path, required=True)
    ev.add_argument("--output-dir", type=Path, required=True)
    coll = sub.add_parser("collect")
    coll.add_argument("--heldout-root", type=Path, required=True)
    coll.add_argument("--generated-results", type=Path, required=True)
    coll.add_argument("--output-dir", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    print({"generate": generate, "evaluate": evaluate, "collect": collect,
           "trajectory-classifier": trajectory_classifier}[command](**args))


if __name__ == "__main__":
    main()
