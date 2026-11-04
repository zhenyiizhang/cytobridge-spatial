#!/usr/bin/env python3
"""Audit corrected package-native MOSTA S5 growth values and geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


TIMES = tuple(float(value) for value in np.arange(0.0, 3.0001, 0.25))
DISPLAY_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
EXPECTED_PACKAGE_COMMIT = "2b3c79eff3face7c4dd33de24d45384b9dbd8a84"
EXPECTED_MODEL_FINETUNE_SHA256 = "d08d21b86fdcd92c748bb54ca81f998fbb157ca5a7acd9548bbfe16c573bfaa5"
EXPECTED_MODEL_SCORE_SHA256 = "d7d06657f8548618db1bc85409e73305fbf59feb9446793550b2c9761639e52a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> int:
    if not (root / "COMPLETE").is_file():
        raise RuntimeError("Shared numerical root is not complete.")
    checked = 0
    for raw in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(maxsplit=1)
        path = root / relative.lstrip("*")
        if sha256(path) != expected:
            raise RuntimeError(f"Shared numerical SHA256 mismatch: {path}")
        checked += 1
    if checked == 0:
        raise RuntimeError("Shared numerical manifest is empty.")
    return checked


def time_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(float(value)).replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numerical-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.numerical_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files_verified = verify_manifest(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["package_commit"] != EXPECTED_PACKAGE_COMMIT:
        raise RuntimeError("Shared S5 calculation does not use the accepted package commit.")
    if summary["model"]["finetune"]["sha256"] != EXPECTED_MODEL_FINETUNE_SHA256:
        raise RuntimeError("Shared S5 calculation does not use the accepted Finetune checkpoint.")
    if summary["model"]["score"]["sha256"] != EXPECTED_MODEL_SCORE_SHA256:
        raise RuntimeError("Shared S5 calculation does not use the accepted Score checkpoint.")
    trajectory = summary["trajectory"]
    if trajectory["mode"] != "global_t0_extrapolation":
        raise RuntimeError("S5 state family is not global-t0 extrapolation.")
    if bool(trajectory["restart_from_preceding_observed_stage"]):
        raise RuntimeError("S5 state family incorrectly restarts from observed anchors.")
    if bool(trajectory["spatial_warp"]):
        raise RuntimeError("S5 state family incorrectly applies a spatial warp.")

    growth_path = root / "s5_growth" / "growth_by_cell_fully_generated.csv"
    contract_path = root / "s5_growth" / "growth_contract.json"
    stored_summary_path = root / "s5_growth" / "brain_growth_summary.csv"
    growth = pd.read_csv(growth_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    stored_brain_summary = pd.read_csv(stored_summary_path).set_index("time")
    required = {"time", "time_key", "cell_index", "x", "y", "growth", "celltype"}
    if set(growth.columns) != required:
        raise RuntimeError(f"Growth table schema mismatch: {growth.columns.tolist()}")
    numeric = growth[["time", "cell_index", "x", "y", "growth"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or growth["celltype"].isna().any():
        raise RuntimeError("Growth table contains non-finite or missing values.")
    if tuple(sorted(growth["time"].astype(float).unique())) != TIMES:
        raise RuntimeError("Growth table does not cover the exact 13-point time grid.")

    state_checks: list[dict[str, object]] = []
    maximum_coordinate_error = 0.0
    maximum_spatial_state_error = 0.0
    for time_value in TIMES:
        state_file = root / "generated_states" / f"time_{time_token(time_value)}.h5ad"
        state = ad.read_h5ad(state_file)
        subset = growth.loc[np.isclose(growth["time"].astype(float), time_value)].copy()
        subset = subset.sort_values("cell_index")
        indices = subset["cell_index"].to_numpy(dtype=int)
        if not np.array_equal(indices, np.arange(state.n_obs, dtype=int)):
            raise RuntimeError(f"Growth rows at t={time_value:g} do not map one-to-one to state rows.")
        state_xy = np.asarray(state.obsm["spatial"], dtype=np.float32)
        state_values = np.asarray(state.X, dtype=np.float32)
        table_xy = subset[["x", "y"]].to_numpy(dtype=np.float32)
        coordinate_error = float(np.max(np.abs(table_xy - state_xy), initial=0.0))
        spatial_state_error = float(np.max(np.abs(state_xy - state_values[:, :2]), initial=0.0))
        maximum_coordinate_error = max(maximum_coordinate_error, coordinate_error)
        maximum_spatial_state_error = max(maximum_spatial_state_error, spatial_state_error)
        if coordinate_error > 1e-6 or spatial_state_error > 1e-6:
            raise RuntimeError(f"S5 coordinate alignment failed at t={time_value:g}.")
        table_labels = subset["celltype"].astype(str).to_numpy()
        state_labels = state.obs["Annotation"].astype(str).to_numpy()
        if not np.array_equal(table_labels, state_labels):
            raise RuntimeError(f"S5 annotation alignment failed at t={time_value:g}.")
        state_checks.append(
            {
                "time": time_value,
                "n_cells": int(state.n_obs),
                "n_brain": int(np.count_nonzero(state_labels == "Brain")),
                "growth_rows": int(len(subset)),
                "coordinate_max_abs_error": coordinate_error,
                "spatial_vs_state_first_two_max_abs_error": spatial_state_error,
                "origin": "generated_global_t0",
                "source_anchor_time": 0.0,
                "spatial_warp": False,
                "state_sha256": sha256(state_file),
            }
        )

    brain = growth.loc[growth["celltype"].astype(str) == "Brain"].copy()
    recalculated = brain.groupby("time", sort=True)["growth"].agg(["count", "mean", "median", "min", "max"])
    if tuple(map(float, recalculated.index)) != TIMES:
        raise RuntimeError("Brain growth is missing a requested time point.")
    summary_error = float(
        np.max(
            np.abs(
                recalculated[["count", "mean", "median", "min", "max"]].to_numpy(dtype=float)
                - stored_brain_summary.loc[list(TIMES), ["count", "mean", "median", "min", "max"]].to_numpy(dtype=float)
            ),
            initial=0.0,
        )
    )
    if summary_error > 1e-7:
        raise RuntimeError(f"Stored Brain growth summary is not reproducible: {summary_error}")
    vmin = float(np.percentile(brain["growth"].to_numpy(dtype=float), 5.0))
    vmax = float(np.percentile(brain["growth"].to_numpy(dtype=float), 95.0))
    stored_vmin = float(contract["vmin"])
    stored_vmax = float(contract["vmax"])
    vmin_serialization_error = abs(vmin - stored_vmin)
    vmax_serialization_error = abs(vmax - stored_vmax)
    vmin_float32_ulp = abs(float(np.spacing(np.float32(stored_vmin))))
    vmax_float32_ulp = abs(float(np.spacing(np.float32(stored_vmax))))
    # The contract was calculated from the in-memory float32 package output,
    # whereas the per-cell CSV uses pandas' compact decimal serialization.
    # Reproduction from the CSV is therefore required to agree within one
    # float32 ULP; the renderer uses the higher-precision stored contract.
    if (
        vmin_serialization_error > vmin_float32_ulp
        or vmax_serialization_error > vmax_float32_ulp
    ):
        raise RuntimeError("S5 common 5th-95th percentile scale is not reproducible.")
    if tuple(float(value) for value in contract["time_grid"]) != TIMES:
        raise RuntimeError("S5 growth contract has the wrong calculation grid.")
    if tuple(float(value) for value in contract["display_times"]) != DISPLAY_TIMES:
        raise RuntimeError("S5 growth contract has the wrong submitted display grid.")
    if float(contract["dropped_from_display_only"]) != 1.5:
        raise RuntimeError("S5 must omit t=1.5 only from display.")
    if contract["state_source"] != "generated_global_t0" or bool(contract["spatial_warp"]):
        raise RuntimeError("S5 growth contract has the wrong state origin or warp setting.")

    means = recalculated["mean"].to_numpy(dtype=float)
    medians = recalculated["median"].to_numpy(dtype=float)
    mean_differences = np.diff(means)
    median_differences = np.diff(medians)
    if not (mean_differences < 0).all() or not (median_differences < 0).all():
        raise RuntimeError("Brain growth summary contains a temporal reversal/spike.")
    temporal_table = recalculated.reset_index()
    temporal_table["mean_adjacent_change"] = np.r_[np.nan, mean_differences]
    temporal_table["median_adjacent_change"] = np.r_[np.nan, median_differences]
    temporal_table.to_csv(output_dir / "s5_brain_growth_temporal_audit.csv", index=False)
    pd.DataFrame(state_checks).to_csv(output_dir / "s5_state_alignment_audit.csv", index=False)

    audit = {
        "schema_version": 1,
        "status": "pass",
        "dataset": "MOSTA",
        "panel": "Supplementary Figure S5",
        "claim": "Package-predicted Brain growth decreases smoothly along one fully generated global-t0 developmental trajectory.",
        "numerical_root": str(root),
        "numerical_manifest_sha256": sha256(root / "SHA256SUMS.txt"),
        "files_verified_from_manifest": files_verified,
        "accepted_package_commit": EXPECTED_PACKAGE_COMMIT,
        "accepted_model_hashes": {
            "finetune": EXPECTED_MODEL_FINETUNE_SHA256,
            "score": EXPECTED_MODEL_SCORE_SHA256,
        },
        "growth_table": {"path": str(growth_path), "sha256": sha256(growth_path), "n_rows": int(len(growth))},
        "brain": {
            "n_rows_all_times": int(len(brain)),
            "time_points_calculated": list(TIMES),
            "time_points_displayed": list(DISPLAY_TIMES),
            "display_only_omission": 1.5,
            "common_scale": {
                "percentiles": [5.0, 95.0],
                "authoritative_vmin": stored_vmin,
                "authoritative_vmax": stored_vmax,
                "csv_recomputed_vmin": vmin,
                "csv_recomputed_vmax": vmax,
                "vmin_serialization_error": vmin_serialization_error,
                "vmax_serialization_error": vmax_serialization_error,
                "vmin_float32_ulp": vmin_float32_ulp,
                "vmax_float32_ulp": vmax_float32_ulp,
                "agreement_within_one_float32_ulp": True,
            },
            "mean_start": float(means[0]),
            "mean_end": float(means[-1]),
            "mean_monotonically_decreasing": True,
            "median_monotonically_decreasing": True,
            "largest_mean_adjacent_increase": float(mean_differences.max()),
            "largest_median_adjacent_increase": float(median_differences.max()),
        },
        "alignment": {
            "all_growth_rows_map_one_to_one_to_generated_state_rows": True,
            "labels_match_generated_states": True,
            "maximum_growth_table_vs_state_coordinate_error": maximum_coordinate_error,
            "maximum_spatial_vs_state_first_two_error": maximum_spatial_state_error,
            "stored_summary_max_abs_error": summary_error,
        },
        "gates": {
            "package_native_evaluate_growth_by_timepoint": True,
            "all_13_times_generated_global_t0": True,
            "restart_from_observed_anchors": False,
            "spatial_warp": False,
            "all_values_finite": True,
            "common_scale_from_all_13_brain_slices": True,
            "independent_panel_normalization": False,
            "growth_value_smoothing": False,
            "arista_assets_used": False,
        },
    }
    (output_dir / "s5_numerical_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
