#!/usr/bin/env python3
"""Select 31 robust, effect-size-gated representatives for corrected MOSTA S11."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUOTAS = {1: 12, 2: 11, 3: 8}
TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-component-root", required=True)
    parser.add_argument("--seed-stability-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path) -> dict[str, Any]:
    if not (root / "COMPLETE").is_file() or not (root / "SHA256SUMS.txt").is_file():
        raise RuntimeError(f"Input is not sealed: {root}")
    checked = 0
    for raw in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        path = root / relative.lstrip("*")
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Checksum mismatch: {path}")
        checked += 1
    return {
        "root": str(root),
        "manifest_sha256": sha256(root / "SHA256SUMS.txt"),
        "files_verified": checked,
    }


def normalized_by_pair(path: Path, assignments: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0)
    table.columns = [float(value) for value in table.columns]
    table.index = table.index.astype(str)
    mapping = assignments.set_index("profile")["pair_id"].astype(str)
    if set(table.index) == set(mapping.index):
        table.index = [mapping.loc[value] for value in table.index]
    elif set(table.index) != set(assignments["pair_id"].astype(str)):
        raise RuntimeError(f"Normalized profile identity mismatch: {path}")
    return table.loc[:, TIMES].sort_index()


def seed_tables(root: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if seed == 42:
        base = root / "tables" / "M_sum"
    else:
        base = root / "tables" / f"seed{seed}"
    assignments = pd.read_csv(base / "lr_pattern_assignments.csv")
    pair = pd.read_csv(base / "lr_pair_timecourse.csv")
    prototypes = pd.read_csv(base / "lr_pattern_prototypes.csv")
    normalized = normalized_by_pair(base / "lr_normalized_profiles.csv", assignments)
    return assignments, pair, prototypes, normalized


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_component_root).resolve()
    stability_root = Path(args.seed_stability_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    baseline_contract = verify(baseline_root)
    stability_contract = verify(stability_root)

    data: dict[int, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {
        42: seed_tables(baseline_root, 42),
        43: seed_tables(stability_root, 43),
        44: seed_tables(stability_root, 44),
    }
    assignment_frames = []
    raw_totals = []
    distances = []
    for seed, (assignments, pair, prototypes, normalized) in data.items():
        if len(assignments) != 1757 or assignments["pair_id"].duplicated().any():
            raise RuntimeError(f"Unexpected assignment universe for seed {seed}")
        assignment_frames.append(
            assignments[["pair_id", "cluster"]].rename(columns={"cluster": f"cluster_seed{seed}"})
        )
        totals = pair.groupby("pair_id", sort=False)["score"].sum().rename(f"total_score_seed{seed}")
        raw_totals.append(totals)
        prototype_matrix = prototypes.pivot(index="cluster", columns="time", values="mean").loc[:, TIMES]
        cluster_map = assignments.set_index("pair_id")["cluster"].astype(int)
        distance_rows = {}
        for pair_id in normalized.index:
            cluster = int(cluster_map.loc[pair_id])
            delta = normalized.loc[pair_id].to_numpy(dtype=float) - prototype_matrix.loc[cluster].to_numpy(dtype=float)
            distance_rows[pair_id] = float(np.square(delta).sum())
        distances.append(pd.Series(distance_rows, name=f"prototype_distance_seed{seed}"))

    audit = assignment_frames[0]
    for frame in assignment_frames[1:]:
        audit = audit.merge(frame, on="pair_id", validate="one_to_one")
    for series in raw_totals + distances:
        audit = audit.merge(series, left_on="pair_id", right_index=True, validate="one_to_one")
    audit["stable_cluster"] = (
        (audit["cluster_seed42"] == audit["cluster_seed43"])
        & (audit["cluster_seed42"] == audit["cluster_seed44"])
    )
    audit["cluster"] = audit["cluster_seed42"].astype(int)
    score_columns = [f"total_score_seed{seed}" for seed in (42, 43, 44)]
    audit["geometric_mean_total_score"] = np.exp(
        np.log(audit[score_columns].clip(lower=np.finfo(float).tiny)).mean(axis=1)
    )
    distance_columns = [f"prototype_distance_seed{seed}" for seed in (42, 43, 44)]
    audit["mean_prototype_squared_distance"] = audit[distance_columns].mean(axis=1)

    seed42_assignments = data[42][0].copy()
    metadata = seed42_assignments.set_index("pair_id")[["profile", "ligand", "receptor"]]
    audit = audit.merge(metadata, left_on="pair_id", right_index=True, validate="one_to_one")
    selected_rows = []
    gate_rows = []
    for cluster, quota in QUOTAS.items():
        stable = audit.loc[audit["stable_cluster"] & (audit["cluster"] == cluster)].copy()
        if len(stable) < quota:
            raise RuntimeError(f"Cluster {cluster} has only {len(stable)} stable profiles")
        strength_gate = float(stable["geometric_mean_total_score"].median())
        eligible = stable.loc[stable["geometric_mean_total_score"] >= strength_gate].copy()
        if len(eligible) < quota:
            raise RuntimeError(f"Cluster {cluster} effect-size gate leaves only {len(eligible)} profiles")
        eligible = eligible.sort_values(
            ["mean_prototype_squared_distance", "geometric_mean_total_score", "pair_id"],
            ascending=[True, False, True],
            kind="mergesort",
        ).head(quota)
        gate_rows.append(
            {
                "cluster": cluster,
                "quota": quota,
                "n_cluster_seed42": int((audit["cluster"] == cluster).sum()),
                "n_stable_all_three_seeds": len(stable),
                "geometric_mean_total_score_median_gate": strength_gate,
                "n_eligible_above_gate": len(stable.loc[stable["geometric_mean_total_score"] >= strength_gate]),
            }
        )
        for within_order, row in enumerate(eligible.itertuples(index=False), start=1):
            selected_rows.append(
                {
                    "display_order": len(selected_rows) + 1,
                    "cluster": cluster,
                    "within_cluster_order": within_order,
                    "pair_id": str(row.pair_id),
                    "pair": str(row.profile),
                    "ligand": str(row.ligand),
                    "receptor": str(row.receptor),
                    "cluster_seed42": int(row.cluster_seed42),
                    "cluster_seed43": int(row.cluster_seed43),
                    "cluster_seed44": int(row.cluster_seed44),
                    "geometric_mean_total_score": float(row.geometric_mean_total_score),
                    "mean_prototype_squared_distance": float(row.mean_prototype_squared_distance),
                    "selection_policy": "stable_all_seeds_then_top_half_strength_then_prototype_distance",
                }
            )
    selection = pd.DataFrame(selected_rows)
    if len(selection) != 31 or selection["cluster"].value_counts().sort_index().to_dict() != QUOTAS:
        raise RuntimeError("Corrected S11 31-profile geometry failed")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_csv, index=False)

    manifest = {
        "schema_version": 1,
        "status": "complete_selection_not_figure",
        "dataset": "MOSTA",
        "panel": "Supplementary Figure S11",
        "purpose": "Robust representative content selection within corrected package-native M_sum clusters",
        "inputs": {
            "baseline_seed42_component": baseline_contract,
            "seed43_seed44_stability": stability_contract,
        },
        "selection": {
            "n_profiles": 31,
            "cluster_quotas": {str(key): value for key, value in QUOTAS.items()},
            "geometry_reason": "preserve submitted 31-profile 4x8 small-multiple layout; first 12 orange slots retained, remaining slots split 11 blue and 8 green to resolve the submitted caption/figure contradiction",
            "stability_gate": "same peak-ordered k=3 assignment at seeds 42, 43, and 44",
            "effect_size_gate": "geometric mean of seven-time total M_sum LR score at or above the stable-cluster median",
            "ranking": "ascending mean squared distance to the matching seed-specific prototype; descending strength and pair_id tie breaks",
            "gate_audit": gate_rows,
            "does_not_modify_clusters": True,
            "does_not_modify_curves": True,
            "manual_biological_cherry_picking": False,
        },
        "output": {"path": str(output_csv), "sha256": sha256(output_csv)},
        "arista_assets_used": False,
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
