#!/usr/bin/env python3
"""Audit the corrected numerical table used for Supplementary Figure S6.

This script is calculation-only.  It verifies that S6 is derived from the
single, fully generated, global-t0, 50k split-SDE trajectory produced with the
accepted package/model/classifier, and that the summarized count/fraction
table agrees exactly with the labels stored in every generated H5AD state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
BUNDLE = Path(__file__).resolve().parents[1]
SHARED = (
    REPO
    / "output/mosta_si_shared_compute_20260825_v1/server_download"
    / "si-s4-s10-global-t0-n50000-dense13-2b3c79e-seed42-20260825-v1"
)
COMPOSITION = SHARED / "s6_composition/celltype_composition_fully_generated.csv"
GROWTH = SHARED / "s5_growth/growth_by_cell_fully_generated.csv"
SUMMARY = SHARED / "summary.json"
STATE_INVENTORY = SHARED / "generated_states/state_inventory.csv"
EXPECTED_TIMES = np.arange(0.0, 3.0001, 0.25, dtype=float)
EXPECTED_PACKAGE_COMMIT = "2b3c79eff3face7c4dd33de24d45384b9dbd8a84"
EXPECTED_FINETUNE_SHA256 = "d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5"
EXPECTED_SCORE_SHA256 = "d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a"
EXPECTED_CLASSIFIER_SHA256 = "f938575c145baa9002de695c39a8637f57d6cb3a06ccfaf4d18b707ca962a7e0"
SUBMITTED_DISPLAY_CELLTYPES = (
    "Brain",
    "Connective tissue",
    "Cavity",
    "Epidermis",
    "Muscle",
    "Jaw and tooth",
    "Meninges",
    "Liver",
    "Cartilage primordium",
    "Spinal cord",
    "Heart",
    "GI tract",
    "Dorsal root ganglion",
    "Cartilage",
    "Adipose tissue",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_time(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    composition = pd.read_csv(COMPOSITION)
    inventory = pd.read_csv(STATE_INVENTORY)

    errors: list[str] = []
    if summary.get("status") != "complete":
        errors.append("shared server computation is not complete")
    if summary.get("dataset") != "mosta":
        errors.append("dataset is not MOSTA")
    if summary.get("package_commit") != EXPECTED_PACKAGE_COMMIT:
        errors.append("package commit mismatch")
    if summary["model"]["finetune"]["sha256"] != EXPECTED_FINETUNE_SHA256:
        errors.append("Finetune weight hash mismatch")
    if summary["model"]["score"]["sha256"] != EXPECTED_SCORE_SHA256:
        errors.append("Score weight hash mismatch")
    if summary["classifier"]["sha256"] != EXPECTED_CLASSIFIER_SHA256:
        errors.append("classifier hash mismatch")
    if int(summary["classifier"]["k"]) != 10:
        errors.append("classifier k is not 10")

    trajectory = summary["trajectory"]
    if trajectory.get("mode") != "global_t0_extrapolation":
        errors.append("trajectory mode is not global_t0_extrapolation")
    if bool(trajectory.get("restart_from_preceding_observed_stage")):
        errors.append("trajectory restarts from a preceding observed stage")
    if bool(trajectory.get("spatial_warp")):
        errors.append("trajectory contains a spatial warp")
    if int(trajectory.get("n_initial")) != 50000:
        errors.append("trajectory did not start from 50,000 particles")

    found_times = np.sort(composition["time"].unique().astype(float))
    if not np.array_equal(found_times, EXPECTED_TIMES):
        errors.append(f"wrong composition time grid: {found_times.tolist()}")
    inventory_times = inventory["time"].to_numpy(dtype=float)
    if not np.array_equal(inventory_times, EXPECTED_TIMES):
        errors.append("state inventory time grid mismatch")
    if not (inventory["origin"].astype(str) == "generated_global_t0").all():
        errors.append("not all states are generated_global_t0")
    if not np.allclose(inventory["source_anchor_time"].to_numpy(dtype=float), 0.0):
        errors.append("not all states use source anchor t0")
    if inventory["spatial_warp"].astype(bool).any():
        errors.append("state inventory contains a warped state")

    table_rows: list[dict[str, object]] = []
    max_fraction_error = 0.0
    state_hashes_verified = 0
    for time_index, time in enumerate(EXPECTED_TIMES):
        subset = composition[np.isclose(composition["time"], time)].copy()
        if subset.empty:
            errors.append(f"missing composition time {time:g}")
            continue
        if subset["time_index"].nunique() != 1 or int(subset["time_index"].iloc[0]) != time_index:
            errors.append(f"wrong time_index at t={time:g}")
        if subset["celltype"].duplicated().any():
            errors.append(f"duplicate cell type at t={time:g}")
        declared_totals = subset["total"].unique()
        if len(declared_totals) != 1:
            errors.append(f"inconsistent total column at t={time:g}")
            continue
        declared_total = int(declared_totals[0])
        count_total = int(subset["count"].sum())
        if declared_total != count_total:
            errors.append(f"count sum != total at t={time:g}")
        fraction_sum = float(subset["fraction"].sum())
        fraction_error = float(
            np.max(
                np.abs(
                    subset["fraction"].to_numpy(dtype=float)
                    - subset["count"].to_numpy(dtype=float) / declared_total
                )
            )
        )
        max_fraction_error = max(max_fraction_error, fraction_error)
        if abs(fraction_sum - 1.0) > 1e-12:
            errors.append(f"fractions do not sum to one at t={time:g}")
        if fraction_error > 1e-15:
            errors.append(f"fraction != count/total at t={time:g}")

        state_path = SHARED / f"generated_states/time_{safe_time(time)}.h5ad"
        state_row = inventory[np.isclose(inventory["time"], time)].iloc[0]
        state_hash = sha256(state_path)
        if state_hash != str(state_row["sha256"]):
            errors.append(f"state hash mismatch at t={time:g}")
        else:
            state_hashes_verified += 1
        state = ad.read_h5ad(state_path, backed="r")
        try:
            if "Annotation" not in state.obs:
                errors.append(f"Annotation missing at t={time:g}")
                continue
            state_counts = state.obs["Annotation"].astype(str).value_counts().sort_index()
            table_counts = subset.set_index("celltype")["count"].astype(int).sort_index()
            all_labels = state_counts.index.union(table_counts.index)
            state_counts = state_counts.reindex(all_labels, fill_value=0).astype(int)
            table_counts = table_counts.reindex(all_labels, fill_value=0).astype(int)
            exact_label_count_match = bool(state_counts.equals(table_counts))
            if not exact_label_count_match:
                errors.append(f"label counts disagree with H5AD at t={time:g}")
            if int(state.n_obs) != declared_total:
                errors.append(f"H5AD n_obs != total at t={time:g}")
            if int(state_row["n_cells"]) != declared_total:
                errors.append(f"state inventory n_cells != total at t={time:g}")
            if int(state_row["n_labels"]) != int((table_counts > 0).sum()):
                errors.append(f"state inventory n_labels mismatch at t={time:g}")
            table_rows.append(
                {
                    "time_index": time_index,
                    "time": float(time),
                    "total": declared_total,
                    "n_labels": int((table_counts > 0).sum()),
                    "fraction_sum": fraction_sum,
                    "max_fraction_error": fraction_error,
                    "h5ad_label_counts_exact": exact_label_count_match,
                    "state_sha256": state_hash,
                }
            )
        finally:
            state.file.close()

    per_time = pd.DataFrame(table_rows)
    totals = per_time["total"].to_numpy(dtype=int)
    if not bool(np.all(np.diff(totals) > 0)):
        errors.append("split-population totals are not strictly increasing")

    # Cross-panel dynamical gate: the realized population log-growth over each
    # quarter step should agree with the interval-average package growth net.
    growth = pd.read_csv(GROWTH, usecols=["time", "growth"])
    growth_mean = (
        growth.groupby("time", sort=True)["growth"].mean().reindex(EXPECTED_TIMES)
    )
    if growth_mean.isna().any():
        errors.append("growth table does not cover the exact 13-time grid")
    realized_log_rate = np.log(totals[1:] / totals[:-1]) / 0.25
    interval_growth_mean = 0.5 * (
        growth_mean.to_numpy(dtype=float)[:-1]
        + growth_mean.to_numpy(dtype=float)[1:]
    )
    growth_rate_mae = float(np.mean(np.abs(realized_log_rate - interval_growth_mean)))
    growth_rate_correlation = float(
        np.corrcoef(realized_log_rate, interval_growth_mean)[0, 1]
    )
    realized_within_endpoint_range = bool(
        np.all(
            (realized_log_rate >= np.minimum(
                growth_mean.to_numpy(dtype=float)[:-1],
                growth_mean.to_numpy(dtype=float)[1:],
            ))
            & (realized_log_rate <= np.maximum(
                growth_mean.to_numpy(dtype=float)[:-1],
                growth_mean.to_numpy(dtype=float)[1:],
            ))
        )
    )
    if growth_rate_correlation < 0.99:
        errors.append("realized population growth is not correlated with package growth net")
    if growth_rate_mae > 0.02:
        errors.append("realized population growth differs excessively from package growth net")
    if not realized_within_endpoint_range:
        errors.append("realized interval growth falls outside endpoint growth-net means")

    growth_consistency = pd.DataFrame(
        {
            "time_start": EXPECTED_TIMES[:-1],
            "time_end": EXPECTED_TIMES[1:],
            "total_start": totals[:-1],
            "total_end": totals[1:],
            "realized_log_growth_rate": realized_log_rate,
            "growth_net_mean_start": growth_mean.to_numpy(dtype=float)[:-1],
            "growth_net_mean_end": growth_mean.to_numpy(dtype=float)[1:],
            "growth_net_interval_endpoint_mean": interval_growth_mean,
            "absolute_error": np.abs(realized_log_rate - interval_growth_mean),
        }
    )

    pivot_fraction = composition.pivot(
        index="time", columns="celltype", values="fraction"
    ).fillna(0.0)
    global_order = pivot_fraction.mean(axis=0).sort_values(ascending=False, kind="stable")
    selected = list(SUBMITTED_DISPLAY_CELLTYPES)
    missing_submitted_labels = [label for label in selected if label not in pivot_fraction]
    if missing_submitted_labels:
        errors.append(
            "submitted S6 display labels missing from corrected table: "
            + ", ".join(missing_submitted_labels)
        )
    other_labels = [label for label in map(str, global_order.index) if label not in selected]
    collapsed_fraction = pivot_fraction[selected].copy()
    collapsed_fraction["Other"] = pivot_fraction[other_labels].sum(axis=1)
    if not np.allclose(collapsed_fraction.sum(axis=1), 1.0, atol=1e-12, rtol=0):
        errors.append("top15+Other collapsed fractions do not sum to one")

    audit = {
        "schema_version": 1,
        "panel": "Supplementary Figure S6a-b",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "input": {
            "composition_path": str(COMPOSITION),
            "composition_sha256": sha256(COMPOSITION),
            "growth_path": str(GROWTH),
            "growth_sha256": sha256(GROWTH),
            "summary_path": str(SUMMARY),
            "summary_sha256": sha256(SUMMARY),
            "state_inventory_path": str(STATE_INVENTORY),
            "state_inventory_sha256": sha256(STATE_INVENTORY),
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
        "trajectory_contract": {
            "mode": trajectory["mode"],
            "origin_all_times": "generated_global_t0",
            "source_anchor_time_all_times": 0.0,
            "restart_from_preceding_observed_stage": False,
            "spatial_warp": False,
            "n_initial": 50000,
            "n_final": int(totals[-1]),
            "time_points": EXPECTED_TIMES.tolist(),
            "split_sde": trajectory["split_sde"],
        },
        "composition_checks": {
            "rows": int(len(composition)),
            "times": int(len(found_times)),
            "all_13_state_hashes_verified": state_hashes_verified == 13,
            "all_h5ad_label_counts_exact": bool(per_time["h5ad_label_counts_exact"].all()),
            "all_fraction_sums_one": bool(np.allclose(per_time["fraction_sum"], 1.0, atol=1e-12, rtol=0)),
            "max_fraction_error": max_fraction_error,
            "totals_strictly_increasing": bool(np.all(np.diff(totals) > 0)),
            "submitted_display_rule": "freeze the 15-category legend/stack order visible in the submitted notebook output and SI; replace values only; collapse every remaining corrected label to Other",
            "submitted_display_celltypes": selected,
            "other_labels": other_labels,
            "top15_plus_other_fraction_sums_one": bool(
                np.allclose(collapsed_fraction.sum(axis=1), 1.0, atol=1e-12, rtol=0)
            ),
            "growth_dynamics_crosscheck": {
                "definition": "realized log(total[t+0.25]/total[t])/0.25 versus mean of package growth-net means at interval endpoints",
                "correlation": growth_rate_correlation,
                "mae": growth_rate_mae,
                "all_realized_rates_within_endpoint_means": realized_within_endpoint_range,
                "status": "PASS" if (
                    growth_rate_correlation >= 0.99
                    and growth_rate_mae <= 0.02
                    and realized_within_endpoint_range
                ) else "FAIL",
            },
        },
        "rejected_old_contract": {
            "sha256": "543e07f9775002f7241556d31ceff35c700e2ef91c8d734d95ad569a4886943b",
            "reason": "mixed full observed anchor totals with capped generated intermediate totals",
            "used_for_rendering": False,
        },
    }

    tables = BUNDLE / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    per_time.to_csv(tables / "s6_composition_temporal_audit.csv", index=False)
    growth_consistency.to_csv(
        tables / "s6_population_growth_dynamics_crosscheck.csv", index=False
    )
    pd.DataFrame(
        {
            "celltype": list(map(str, global_order.index)),
            "mean_fraction": global_order.to_numpy(dtype=float),
            "in_submitted_display": [str(label) in selected for label in global_order.index],
        }
    ).to_csv(tables / "s6_celltype_mean_fraction_order.csv", index=False)
    (tables / "s6_numerical_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
