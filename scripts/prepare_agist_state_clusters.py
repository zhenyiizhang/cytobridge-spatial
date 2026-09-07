#!/usr/bin/env python3
"""Prepare reproducible unsupervised AGIST state-space clusters.

The AGIST archive does not retain trustworthy cell-type annotations.  This
script defines broad data-driven populations from the 50-dimensional gene-state
coordinates.  It does not calculate velocity agreement; row-level cosine data
must be joined separately before any cluster-stratified performance figure is
made.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


DEFAULT_INPUT = Path("data/mouse_brain_simulation.csv")
DEFAULT_OUTPUT_DIR = Path("results/agist_velocity_cluster_breakdown_20260811")


def stratified_sample_indices(times: np.ndarray, per_time: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for time in sorted(np.unique(times)):
        candidates = np.flatnonzero(times == time)
        selected.append(
            rng.choice(candidates, size=min(per_time, len(candidates)), replace=False)
        )
    return np.concatenate(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--sample-per-time", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gene_columns = [f"x{i}" for i in range(3, 53)]
    table = pd.read_csv(args.input, usecols=["samples", *gene_columns])
    gene_state = table[gene_columns].to_numpy(dtype=np.float32)
    times = table["samples"].to_numpy(dtype=float)
    sample_index = stratified_sample_indices(times, args.sample_per_time, args.seed)

    diagnostics = []
    fitted = {}
    for n_clusters in range(args.min_k, args.max_k + 1):
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=args.seed,
            n_init=20,
            batch_size=2048,
        ).fit(gene_state)
        score = float(
            silhouette_score(gene_state[sample_index], model.labels_[sample_index])
        )
        counts = np.bincount(model.labels_, minlength=n_clusters)
        diagnostics.append(
            {
                "n_clusters": n_clusters,
                "silhouette": score,
                "minimum_cluster_size": int(counts.min()),
                "maximum_cluster_size": int(counts.max()),
                "selection_sample_size": int(len(sample_index)),
            }
        )
        fitted[n_clusters] = model

    diagnostics_table = pd.DataFrame(diagnostics)
    selected_k = int(
        diagnostics_table.sort_values(
            ["silhouette", "n_clusters"], ascending=[False, True]
        ).iloc[0]["n_clusters"]
    )
    selected_model = fitted[selected_k]

    # Give cluster identifiers a stable left-to-right order in a two-dimensional
    # PCA view rather than exposing arbitrary estimator label numbers.
    pca = PCA(n_components=2, random_state=args.seed)
    embedding = pca.fit_transform(gene_state)
    raw_labels = selected_model.labels_
    cluster_order = sorted(
        range(selected_k), key=lambda label: float(embedding[raw_labels == label, 0].mean())
    )
    label_map = {raw: f"C{rank + 1}" for rank, raw in enumerate(cluster_order)}
    labels = np.array([label_map[label] for label in raw_labels], dtype=object)

    assignment = pd.DataFrame(
        {
            "row_index": np.arange(len(table), dtype=int),
            "time": times,
            "state_cluster": labels,
            "state_pc1": embedding[:, 0],
            "state_pc2": embedding[:, 1],
        }
    )
    cluster_summary = (
        assignment.groupby("state_cluster", observed=True)
        .agg(
            n=("row_index", "size"),
            state_pc1_mean=("state_pc1", "mean"),
            state_pc2_mean=("state_pc2", "mean"),
        )
        .reset_index()
        .sort_values("state_cluster")
    )
    time_counts = (
        assignment.groupby(["state_cluster", "time"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["state_cluster", "time"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_table.to_csv(args.output_dir / "cluster_selection_diagnostics.csv", index=False)
    assignment.to_csv(args.output_dir / "agist_state_cluster_assignments.csv", index=False)
    cluster_summary.to_csv(args.output_dir / "agist_state_cluster_summary.csv", index=False)
    time_counts.to_csv(args.output_dir / "agist_state_cluster_time_counts.csv", index=False)
    (args.output_dir / "cluster_selection.txt").write_text(
        "\n".join(
            [
                f"selected_k={selected_k}",
                f"selection_space=gene_state_x3_x52",
                f"selection_method=maximum_silhouette_k_{args.min_k}_to_{args.max_k}",
                f"selection_sample_size={len(sample_index)}",
                f"random_seed={args.seed}",
                "interpretation=broad_unsupervised_state_partition_not_cell_type",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Selected k={selected_k}")
    print(args.output_dir / "agist_state_cluster_assignments.csv")


if __name__ == "__main__":
    main()

