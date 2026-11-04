#!/usr/bin/env python3
"""Recluster corrected ARISTA LR trajectories without singleton artifacts.

The script deliberately imports ``cluster_temporal_profiles`` from the active
CytoBridge package worktree.  It preserves the package-native strict LR scores
and changes only the temporal clustering method from average linkage to a
deterministic k-means implementation exposed by the package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_counts(assignments: pd.DataFrame) -> str:
    counts = assignments["cluster"].value_counts().sort_index()
    return ";".join(f"{int(cluster)}:{int(count)}" for cluster, count in counts.items())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-timecourse", required=True, type=Path)
    parser.add_argument("--old-pattern-summary", required=True, type=Path)
    parser.add_argument("--package-worktree", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--noise-replicates", type=int, default=25)
    args = parser.parse_args()

    pair_path = args.pair_timecourse.expanduser().resolve()
    old_summary_path = args.old_pattern_summary.expanduser().resolve()
    package_worktree = args.package_worktree.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")
    if args.noise_replicates <= 0:
        raise ValueError("noise_replicates must be positive")

    sys.path.insert(0, str(package_worktree))
    from CytoBridge.tl.downstream.temporal import cluster_temporal_profiles

    pair_timecourse = pd.read_csv(pair_path)
    old_summary = pd.read_csv(old_summary_path)
    required = {
        "time", "pair_id", "ligand", "receptor", "pair", "score", "max_edge",
        "nonzero_edges", "n_cell_types", "peak_sender", "peak_receiver",
    }
    missing = required - set(pair_timecourse.columns)
    if missing:
        raise KeyError(f"pair_timecourse is missing columns: {sorted(missing)}")
    if pair_timecourse.duplicated(["pair", "time"]).any():
        raise ValueError("pair_timecourse contains duplicate pair/time rows")
    time_values = np.sort(pair_timecourse["time"].unique().astype(float))
    expected_times = np.arange(0.0, 4.0 + 0.5, 0.5)
    if not np.array_equal(time_values, expected_times):
        raise ValueError(f"Unexpected ARISTA time grid: {time_values.tolist()}")
    per_pair_counts = pair_timecourse.groupby("pair", sort=False)["time"].nunique()
    if not bool((per_pair_counts == len(expected_times)).all()):
        raise ValueError("At least one LR pair lacks the complete nine-time-point grid")

    profiles = pair_timecourse.pivot(index="pair", columns="time", values="score")
    profiles = profiles.loc[:, expected_times].sort_index()
    if not np.isfinite(profiles.to_numpy(dtype=float)).all():
        raise ValueError("LR profile matrix contains non-finite values")
    if len(profiles) != 531:
        raise ValueError(f"Expected 531 strict LR pairs, found {len(profiles)}")

    primary = cluster_temporal_profiles(
        profiles,
        n_clusters=2,
        normalization="minmax",
        method="kmeans",
        cluster_order="peak_time",
    )
    primary_assignments = primary.assignments.set_index("profile").loc[profiles.index].reset_index()
    if primary_assignments["cluster"].value_counts().min() < 20:
        raise AssertionError("Corrected k-means unexpectedly produced a very small LR cluster")

    identity = pair_timecourse.drop_duplicates("pair").set_index("pair")
    summary = old_summary.drop(columns=["cluster"], errors="ignore").set_index("pair")
    if set(profiles.index) != set(identity.index) or set(profiles.index) != set(summary.index):
        raise ValueError("LR pair sets differ between timecourse and summary inputs")
    assignments = primary_assignments.rename(columns={"profile": "pair"}).set_index("pair")
    corrected_summary = summary.join(identity[["pair_id", "ligand", "receptor"]], rsuffix="_identity")
    corrected_summary = corrected_summary.join(assignments).reset_index()
    corrected_summary = corrected_summary.sort_values(
        ["cluster", "peak_time", "auc", "pair"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )

    normalized = primary.normalized_profiles.copy()
    normalized.index.name = "pair"
    normalized.columns = [f"time_{float(value):.1f}" for value in normalized.columns]
    prototypes = primary.prototypes.rename(
        columns={"mean": "mean_normalized_score", "std": "std_normalized_score", "n_profiles": "n_pairs"}
    )

    k_rows = []
    for k in range(2, 9):
        result = cluster_temporal_profiles(
            profiles,
            n_clusters=k,
            normalization="minmax",
            method="kmeans",
            cluster_order="peak_time",
        )
        diagnostic = result.diagnostics.iloc[0]
        k_rows.append({
            "k": k,
            "silhouette": float(diagnostic["silhouette"]),
            "minimum_cluster_size": int(diagnostic["minimum_cluster_size"]),
            "maximum_cluster_size": int(diagnostic["maximum_cluster_size"]),
            "cluster_counts": cluster_counts(result.assignments),
        })
    k_selection = pd.DataFrame(k_rows)
    if int(k_selection.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"]) != 2:
        raise AssertionError("k=2 is not the best silhouette solution among k=2..8")

    reference_labels = primary_assignments["cluster"].to_numpy(dtype=int)
    normalized_values = primary.normalized_profiles.loc[profiles.index].to_numpy(dtype=float)
    rng = np.random.default_rng(20260825)
    noise_rows = []
    for sigma in (0.01, 0.02, 0.05, 0.10):
        for replicate in range(args.noise_replicates):
            perturbed = np.clip(
                normalized_values + rng.normal(0.0, sigma, size=normalized_values.shape),
                0.0,
                1.0,
            )
            result = cluster_temporal_profiles(
                pd.DataFrame(perturbed, index=profiles.index, columns=profiles.columns),
                n_clusters=2,
                normalization="none",
                method="kmeans",
                cluster_order="peak_time",
            )
            labels = result.assignments.set_index("profile").loc[profiles.index, "cluster"].to_numpy(dtype=int)
            noise_rows.append({
                "noise_sigma": sigma,
                "replicate": replicate,
                "adjusted_rand_index": float(adjusted_rand_score(reference_labels, labels)),
                "cluster_counts": cluster_counts(result.assignments),
            })
    noise_stability = pd.DataFrame(noise_rows)
    noise_summary = noise_stability.groupby("noise_sigma", as_index=False).agg(
        median_adjusted_rand_index=("adjusted_rand_index", "median"),
        minimum_adjusted_rand_index=("adjusted_rand_index", "min"),
        maximum_adjusted_rand_index=("adjusted_rand_index", "max"),
    )

    leave_rows = []
    for time_value in profiles.columns:
        reduced = profiles.drop(columns=time_value)
        result = cluster_temporal_profiles(
            reduced,
            n_clusters=2,
            normalization="minmax",
            method="kmeans",
            cluster_order="peak_time",
        )
        labels = result.assignments.set_index("profile").loc[profiles.index, "cluster"].to_numpy(dtype=int)
        leave_rows.append({
            "omitted_time": float(time_value),
            "adjusted_rand_index": float(adjusted_rand_score(reference_labels, labels)),
            "cluster_counts": cluster_counts(result.assignments),
        })
    leave_time = pd.DataFrame(leave_rows)

    sensitivity_rows = []
    for method, normalization in [
        ("kmeans", "zscore"),
        ("ward", "minmax"),
        ("complete", "minmax"),
        ("average", "minmax"),
        ("single", "minmax"),
    ]:
        result = cluster_temporal_profiles(
            profiles,
            n_clusters=2,
            normalization=normalization,
            method=method,
            cluster_order="peak_time",
        )
        labels = result.assignments.set_index("profile").loc[profiles.index, "cluster"].to_numpy(dtype=int)
        diagnostic = result.diagnostics.iloc[0]
        sensitivity_rows.append({
            "method": method,
            "normalization": normalization,
            "adjusted_rand_index_vs_primary": float(adjusted_rand_score(reference_labels, labels)),
            "silhouette": float(diagnostic["silhouette"]),
            "minimum_cluster_size": int(diagnostic["minimum_cluster_size"]),
            "maximum_cluster_size": int(diagnostic["maximum_cluster_size"]),
            "cluster_counts": cluster_counts(result.assignments),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        corrected_summary.to_csv(stage / "S16_lr_kmeans_pattern_summary.csv", index=False)
        assignments.reset_index().to_csv(stage / "S16_lr_kmeans_assignments.csv", index=False)
        normalized.reset_index().to_csv(stage / "S16_lr_kmeans_normalized_profiles.csv", index=False)
        prototypes.to_csv(stage / "S16_lr_kmeans_prototypes.csv", index=False)
        primary.diagnostics.to_csv(stage / "S16_lr_kmeans_diagnostics.csv", index=False)
        k_selection.to_csv(stage / "S16_lr_k_selection.csv", index=False)
        noise_stability.to_csv(stage / "S16_lr_noise_stability_replicates.csv", index=False)
        noise_summary.to_csv(stage / "S16_lr_noise_stability_summary.csv", index=False)
        leave_time.to_csv(stage / "S16_lr_leave_one_time_stability.csv", index=False)
        sensitivity.to_csv(stage / "S16_lr_method_sensitivity.csv", index=False)
        shutil.copy2(Path(__file__).resolve(), stage / Path(__file__).name)
        manifest = {
            "schema": "cytobridge.arista.S16.strict-lr-kmeans.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_contract": {
                "scores": "unchanged package-native strict all-subunit LR pair_timecourse",
                "n_pairs": int(len(profiles)),
                "time_points": expected_times.tolist(),
                "clustering": "CytoBridge.tl.downstream.temporal.cluster_temporal_profiles",
                "normalization": "row-wise minmax",
                "method": "deterministic kmeans++ n_init=100 seed=0",
                "n_clusters": 2,
                "cluster_order": "prototype peak time",
                "reason": "avoid average-linkage singleton artifact while preserving corrected scores",
            },
            "inputs": {
                str(pair_path): sha256(pair_path),
                str(old_summary_path): sha256(old_summary_path),
                str(package_worktree / "CytoBridge/tl/downstream/temporal.py"): sha256(
                    package_worktree / "CytoBridge/tl/downstream/temporal.py"
                ),
            },
            "qa": {
                "passed": True,
                "cluster_counts": cluster_counts(primary.assignments),
                "silhouette": float(primary.diagnostics.iloc[0]["silhouette"]),
                "k2_best_silhouette_among_k2_to_k8": True,
                "minimum_cluster_size": int(primary.diagnostics.iloc[0]["minimum_cluster_size"]),
                "noise_median_ari_at_sigma_0p05": float(
                    noise_summary.loc[noise_summary["noise_sigma"].eq(0.05), "median_adjusted_rand_index"].iloc[0]
                ),
                "minimum_leave_one_time_ari": float(leave_time["adjusted_rand_index"].min()),
            },
        }
        (stage / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"output_dir": str(output_dir), "qa": manifest["qa"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
