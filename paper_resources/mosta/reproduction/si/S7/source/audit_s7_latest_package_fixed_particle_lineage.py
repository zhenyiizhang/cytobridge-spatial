#!/usr/bin/env python3
"""Audit the corrected fixed-particle lineage used for MOSTA SI Figure S7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
BUNDLE = Path(__file__).resolve().parents[1]
SHARED = (
    REPO
    / "output/mosta_si_shared_compute_20260825_v1/server_download"
    / "si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1"
)
LINEAGE = SHARED / "s7_lineage"
LABELS_PATH = LINEAGE / "fixed_particle_labels.csv.gz"
NODES_PATH = LINEAGE / "lineage_nodes.csv"
EDGES_PATH = LINEAGE / "lineage_edges.csv"
CONTRACT_PATH = LINEAGE / "lineage_contract.json"
SUMMARY_PATH = SHARED / "summary.json"
EXPECTED_TIMES = np.arange(0.0, 3.0001, 0.5, dtype=float)
EXPECTED_PARTICLES = 50000
KEEP_SOURCE_CUMFRAC = 0.8
EXPECTED_PACKAGE_COMMIT = "2b3c79eff3face7c4dd33de24d45384b9dbd8a84"
EXPECTED_FINETUNE_SHA256 = "d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5"
EXPECTED_SCORE_SHA256 = "d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a"
EXPECTED_CLASSIFIER_SHA256 = "f938575c145baa9002de695c39a8637f57d6cb3a06ccfaf4d18b707ca962a7e0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_table_equal(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    left = left[columns].sort_values(columns, kind="stable").reset_index(drop=True)
    right = right[columns].sort_values(columns, kind="stable").reset_index(drop=True)
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) or pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                atol=1e-15,
                rtol=0,
            ):
                return False
        elif not left[column].astype(str).equals(right[column].astype(str)):
            return False
    return len(left) == len(right)


def filter_keep_source_cumfrac(edges: pd.DataFrame, cumfrac: float) -> pd.DataFrame:
    """Exact old helper rule: minimal descending prefix per interval/source."""
    keep: list[int] = []
    for _, group in edges.groupby(["source_time", "target_time", "source"], sort=False):
        group = group.sort_values("count", ascending=False, kind="stable")
        total = float(group["count"].sum())
        cumulative = (group["count"].cumsum() / total).to_numpy(dtype=float)
        keep_n = int(np.searchsorted(cumulative, cumfrac, side="left")) + 1
        keep.extend(group.head(max(1, min(len(group), keep_n))).index.tolist())
    return edges.loc[keep].copy().reset_index(drop=True)


def main() -> None:
    labels = pd.read_csv(LABELS_PATH)
    nodes = pd.read_csv(NODES_PATH)
    edges = pd.read_csv(EDGES_PATH)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []

    if summary.get("package_commit") != EXPECTED_PACKAGE_COMMIT:
        errors.append("package commit mismatch")
    if summary["model"]["finetune"]["sha256"] != EXPECTED_FINETUNE_SHA256:
        errors.append("Finetune hash mismatch")
    if summary["model"]["score"]["sha256"] != EXPECTED_SCORE_SHA256:
        errors.append("Score hash mismatch")
    if summary["classifier"]["sha256"] != EXPECTED_CLASSIFIER_SHA256:
        errors.append("classifier hash mismatch")
    if int(summary["classifier"]["k"]) != 10:
        errors.append("classifier k is not 10")
    if summary["panels"]["S7"]["state_source"] != "global_t0_non_split_fixed_particle":
        errors.append("summary S7 state source is wrong")
    if summary["trajectory"]["lineage_state"] != "non_split_fixed_particle":
        errors.append("summary lineage state is not non-split fixed-particle")

    expected_contract = {
        "trajectory_mode": "global_t0_non_split_fixed_particle",
        "n_particles": EXPECTED_PARTICLES,
        "particle_id_persistent": True,
        "restart_from_observed_anchor": False,
        "classifier_k": 10,
        "spatial_warp": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            errors.append(f"lineage contract mismatch for {key}")
    if not np.array_equal(np.asarray(contract["time_points"], dtype=float), EXPECTED_TIMES):
        errors.append("lineage contract time grid mismatch")

    if len(labels) != EXPECTED_PARTICLES * len(EXPECTED_TIMES):
        errors.append("fixed-particle label table has wrong row count")
    found_times = np.sort(labels["time"].unique().astype(float))
    if not np.array_equal(found_times, EXPECTED_TIMES):
        errors.append("fixed-particle label table has wrong times")

    expected_ids = np.arange(EXPECTED_PARTICLES, dtype=int)
    per_time_checks: list[dict[str, object]] = []
    labels_wide: dict[float, pd.Series] = {}
    for time in EXPECTED_TIMES:
        subset = labels[np.isclose(labels["time"], time)].copy()
        duplicate_ids = int(subset["particle_id"].duplicated().sum())
        ids = np.sort(subset["particle_id"].to_numpy(dtype=int))
        ids_exact = bool(np.array_equal(ids, expected_ids))
        if duplicate_ids or not ids_exact or len(subset) != EXPECTED_PARTICLES:
            errors.append(f"particle identity contract failed at t={time:g}")
        ordered = subset.sort_values("particle_id", kind="stable")
        labels_wide[float(time)] = ordered.set_index("particle_id")["celltype"].astype(str)
        per_time_checks.append(
            {
                "time": float(time),
                "n_rows": int(len(subset)),
                "n_unique_particle_ids": int(subset["particle_id"].nunique()),
                "duplicate_particle_ids": duplicate_ids,
                "ids_exact_0_to_49999": ids_exact,
                "n_labels": int(subset["celltype"].nunique()),
            }
        )

    recomputed_nodes = (
        labels.groupby(["time", "celltype"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    nodes_exact = exact_table_equal(recomputed_nodes, nodes, ["time", "celltype", "count"])
    if not nodes_exact:
        errors.append("provided lineage_nodes.csv differs from fixed-particle labels")

    recomputed_edge_parts: list[pd.DataFrame] = []
    interval_metrics: list[dict[str, object]] = []
    key_source_rows: list[dict[str, object]] = []
    for source_time, target_time in zip(EXPECTED_TIMES[:-1], EXPECTED_TIMES[1:]):
        source = labels_wide[float(source_time)]
        target = labels_wide[float(target_time)]
        pair = pd.DataFrame({"source": source, "target": target})
        counts = (
            pair.groupby(["source", "target"], sort=True)
            .size()
            .rename("count")
            .reset_index()
        )
        source_total = pair.groupby("source", sort=True).size()
        counts["source_fraction"] = counts["count"] / counts["source"].map(source_total)
        counts.insert(0, "target_time", float(target_time))
        counts.insert(0, "source_time", float(source_time))
        recomputed_edge_parts.append(counts)

        same = source.to_numpy(dtype=str) == target.to_numpy(dtype=str)
        dominant = counts.loc[counts.groupby("source")["count"].idxmax()]
        interval_metrics.append(
            {
                "source_time": float(source_time),
                "target_time": float(target_time),
                "n_particles": int(len(pair)),
                "n_edges": int(len(counts)),
                "same_label_fraction": float(same.mean()),
                "mean_source_dominant_target_fraction": float(dominant["source_fraction"].mean()),
                "weighted_source_dominant_target_fraction": float(dominant["count"].sum() / len(pair)),
            }
        )
        for focal in ("Brain", "Cartilage primordium", "Connective tissue", "Muscle", "Cartilage"):
            focal_counts = counts[counts["source"].eq(focal)].sort_values(
                ["count", "target"], ascending=[False, True], kind="stable"
            )
            if focal_counts.empty:
                continue
            for rank, row in enumerate(focal_counts.head(5).itertuples(index=False), start=1):
                key_source_rows.append(
                    {
                        "source_time": float(source_time),
                        "target_time": float(target_time),
                        "source": focal,
                        "target_rank": rank,
                        "target": str(row.target),
                        "count": int(row.count),
                        "source_fraction": float(row.source_fraction),
                    }
                )

    recomputed_edges = pd.concat(recomputed_edge_parts, ignore_index=True)
    edges_exact = exact_table_equal(
        recomputed_edges,
        edges,
        ["source_time", "target_time", "source", "target", "count", "source_fraction"],
    )
    if not edges_exact:
        errors.append("provided lineage_edges.csv differs from fixed-particle labels")

    source_balance = (
        recomputed_edges.groupby(["source_time", "target_time", "source"], sort=True)["count"]
        .sum()
        .rename("edge_out_count")
        .reset_index()
        .merge(
            recomputed_nodes.rename(columns={"time": "source_time", "celltype": "source", "count": "node_count"}),
            on=["source_time", "source"],
            how="left",
        )
    )
    target_balance = (
        recomputed_edges.groupby(["source_time", "target_time", "target"], sort=True)["count"]
        .sum()
        .rename("edge_in_count")
        .reset_index()
        .merge(
            recomputed_nodes.rename(columns={"time": "target_time", "celltype": "target", "count": "node_count"}),
            on=["target_time", "target"],
            how="left",
        )
    )
    source_balanced = bool((source_balance["edge_out_count"] == source_balance["node_count"]).all())
    target_balanced = bool((target_balance["edge_in_count"] == target_balance["node_count"]).all())
    if not source_balanced:
        errors.append("outgoing edge counts do not conserve source nodes")
    if not target_balanced:
        errors.append("incoming edge counts do not conserve target nodes")

    filtered = filter_keep_source_cumfrac(recomputed_edges, KEEP_SOURCE_CUMFRAC)
    coverage = (
        filtered.groupby(["source_time", "target_time", "source"], sort=True)["count"]
        .sum()
        .rename("kept_count")
        .reset_index()
        .merge(source_balance, on=["source_time", "target_time", "source"], how="left")
    )
    coverage["kept_fraction"] = coverage["kept_count"] / coverage["edge_out_count"]
    filter_coverage_ok = bool((coverage["kept_fraction"] >= KEEP_SOURCE_CUMFRAC - 1e-15).all())
    if not filter_coverage_ok:
        errors.append("old 80% source-cumulative filter failed coverage")

    tables = BUNDLE / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_time_checks).to_csv(tables / "s7_particle_identity_by_time.csv", index=False)
    pd.DataFrame(interval_metrics).to_csv(tables / "s7_adjacent_interval_metrics.csv", index=False)
    pd.DataFrame(key_source_rows).to_csv(tables / "s7_key_source_top_targets.csv", index=False)
    filtered.to_csv(tables / "s7_edges_after_exact_old_80pct_filter.csv", index=False)
    coverage.to_csv(tables / "s7_old_filter_source_coverage.csv", index=False)

    audit = {
        "schema_version": 1,
        "panel": "Supplementary Figure S7",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "inputs": {
            "fixed_particle_labels": {"path": str(LABELS_PATH), "sha256": sha256(LABELS_PATH)},
            "lineage_nodes": {"path": str(NODES_PATH), "sha256": sha256(NODES_PATH)},
            "lineage_edges": {"path": str(EDGES_PATH), "sha256": sha256(EDGES_PATH)},
            "lineage_contract": {"path": str(CONTRACT_PATH), "sha256": sha256(CONTRACT_PATH)},
            "shared_summary": {"path": str(SUMMARY_PATH), "sha256": sha256(SUMMARY_PATH)},
        },
        "release": {
            "package_commit": summary["package_commit"],
            "package_archive_sha256": summary["package_release"]["archive_sha256"],
            "aligned_h5ad_sha256": summary["aligned_h5ad"]["sha256"],
            "finetune_sha256": summary["model"]["finetune"]["sha256"],
            "score_sha256": summary["model"]["score"]["sha256"],
            "classifier_sha256": summary["classifier"]["sha256"],
            "classifier_k": int(summary["classifier"]["k"]),
            "classifier_accuracy": float(summary["classifier"]["accuracy"]),
            "classifier_balanced_accuracy": float(summary["classifier"]["balanced_accuracy"]),
        },
        "lineage_contract": contract,
        "checks": {
            "rows": int(len(labels)),
            "times": EXPECTED_TIMES.tolist(),
            "particles_per_time": EXPECTED_PARTICLES,
            "persistent_ids_exact_at_all_times": bool(all(row["ids_exact_0_to_49999"] for row in per_time_checks)),
            "provided_nodes_exactly_recomputed": nodes_exact,
            "provided_edges_exactly_recomputed": edges_exact,
            "source_node_mass_conserved": source_balanced,
            "target_node_mass_conserved": target_balanced,
            "interval_totals_all_50000": bool(
                (recomputed_edges.groupby(["source_time", "target_time"])["count"].sum() == EXPECTED_PARTICLES).all()
            ),
            "spatial_warp": False,
            "observed_anchor_restart": False,
        },
        "submitted_plot_filter": {
            "keep_source_cumfrac": KEEP_SOURCE_CUMFRAC,
            "algorithm": "minimal descending outgoing prefix per interval/source whose cumulative fraction reaches or exceeds 0.8",
            "unfiltered_edges": int(len(recomputed_edges)),
            "filtered_edges": int(len(filtered)),
            "all_sources_meet_coverage": filter_coverage_ok,
            "minimum_kept_fraction": float(coverage["kept_fraction"].min()),
            "maximum_kept_fraction": float(coverage["kept_fraction"].max()),
        },
        "interpretability_diagnostics": {
            "same_label_fraction_by_interval": {
                f"{row['source_time']:g}->{row['target_time']:g}": row["same_label_fraction"]
                for row in interval_metrics
            },
            "key_source_top_targets_table": str((tables / "s7_key_source_top_targets.csv").resolve()),
        },
    }
    audit_path = tables / "s7_numerical_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
