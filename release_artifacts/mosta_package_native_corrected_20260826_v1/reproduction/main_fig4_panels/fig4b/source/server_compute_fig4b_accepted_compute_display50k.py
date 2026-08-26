#!/usr/bin/env python3
"""Compute Fig. 4b on the accepted cohort and map it to the 50k display state."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
from pathlib import Path
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

import CytoBridge as cb

from server_compute_fig4b_focal_hotspot import (
    EXPECTED_PACKAGE_COMMIT,
    EXPECTED_REFERENCE_SHA256,
    PANEL_TIMES,
    communications_from_table,
    empty_contract_field,
    fresh_output,
    input_record,
    sha256,
    state_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-commit", required=True)
    parser.add_argument("--workflow-summary", type=Path, required=True)
    parser.add_argument("--display-summary", type=Path, required=True)
    parser.add_argument("--reference-h5ad", type=Path, required=True)
    parser.add_argument("--full50k-sensitivity-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    output_dir = fresh_output(args.output_dir)
    package_root = args.package_root.expanduser().resolve()
    package_commit = str(args.package_commit).strip()
    if package_commit != EXPECTED_PACKAGE_COMMIT:
        raise RuntimeError("Package commit mismatch.")

    workflow_path = args.workflow_summary.expanduser().resolve()
    display_summary_path = args.display_summary.expanduser().resolve()
    reference_path = args.reference_h5ad.expanduser().resolve()
    sensitivity_dir = args.full50k_sensitivity_run.expanduser().resolve()
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    display_summary = json.loads(display_summary_path.read_text(encoding="utf-8"))
    sensitivity_gate = json.loads(
        (sensitivity_dir / "calculation_gate.json").read_text(encoding="utf-8")
    )
    sensitivity_reproduction = pd.read_csv(
        sensitivity_dir / "formal_primary_reproduction.csv"
    )
    simulation = workflow["simulation"]
    lr_summary = workflow["analyses"]["ligand_receptor"]
    communication_summary = workflow["analyses"]["communication"]
    if workflow.get("dataset") != "mosta" or display_summary.get("dataset") != "mosta":
        raise RuntimeError("This runner accepts MOSTA only.")
    if simulation.get("trajectory_mode") != "global_t0_extrapolation":
        raise RuntimeError("Accepted workflow is not global-t0.")
    if simulation["slice_origins_by_time"].get("0.5") != "generated_global_t0":
        raise RuntimeError("Accepted t=0.5 state is not global-t0.")
    if float(simulation["source_anchor_times_by_time"].get("0.5")) != 0.0:
        raise RuntimeError("Accepted t=0.5 state is not anchored at t0.")
    if display_summary.get("n_samples") != 50000:
        raise RuntimeError("Display summary is not the selected 50k run.")
    if display_summary.get("restart_from_preceding_observed_stage") is not False:
        raise RuntimeError("Display run restarts from an observed intermediate stage.")
    if display_summary.get("spatial_warp") is not False:
        raise RuntimeError("Display run applies a spatial warp.")
    if lr_summary.get("complex_mode") != "min":
        raise RuntimeError("Accepted LR primary is not minimum-complex mode.")
    if lr_summary.get("require_all_subunits") is not True:
        raise RuntimeError("Accepted LR primary does not require every subunit.")

    compute_paths = {
        0.0: Path(workflow["snapshots"][0]),
        0.5: Path(workflow["snapshots"][2]),
        1.0: Path(workflow["snapshots"][4]),
    }
    display_paths = {
        float(row["time"]): Path(row["path"])
        for row in display_summary["snapshots"]
        if float(row["time"]) in PANEL_TIMES
    }
    if set(display_paths) != set(PANEL_TIMES):
        raise RuntimeError("Display summary lacks a required panel time.")
    compute_states = {
        str(t): ad.read_h5ad(compute_paths[t]) for t in PANEL_TIMES
    }
    display_states = {
        str(t): ad.read_h5ad(display_paths[t]) for t in PANEL_TIMES
    }
    compute_state_gates = [
        state_gate(
            compute_states[str(t)],
            time_value=t,
            expected_origin="generated_global_t0" if t == 0.5 else "observed_real",
        )
        for t in PANEL_TIMES
    ]
    display_state_gates = [
        state_gate(
            display_states[str(t)],
            time_value=t,
            expected_origin="generated_global_t0" if t == 0.5 else "observed_real",
        )
        for t in PANEL_TIMES
    ]
    if not all(row["pass"] for row in compute_state_gates + display_state_gates):
        raise RuntimeError("State integrity gate failed.")

    communication_path = Path(communication_summary["table"])
    communications = communications_from_table(communication_path, PANEL_TIMES)
    reference = ad.read_h5ad(reference_path)
    result = cb.tl.compute_focal_lr_type_hotspots(
        compute_states,
        reference,
        communications,
        ligand="Wnt3a",
        receptor="Fzd7_Lrp6",
        time_points=PANEL_TIMES,
        annotation_key="Annotation",
        matrix_key="M_per_source",
        spatial_key="spatial",
        spatial_dim=2,
        loadings_key="PCs",
        expression_space="log1p",
        complex_mode="min",
        require_all_subunits=True,
        preferred_species_tag=lr_summary.get("preferred_species_tag", "mouse"),
        observed_adata=reference,
        observed_time_points=(0.0, 1.0, 2.0, 3.0),
        observed_annotation_key="Annotation",
        observed_expression_space="log1p",
    )

    formal = pd.read_csv(lr_summary["tables"]["pair_timecourse"])
    formal = formal.loc[
        (formal["ligand"].astype(str) == "Wnt3a")
        & (formal["receptor"].astype(str) == "Fzd7_Lrp6")
        & formal["time"].isin(PANEL_TIMES)
    ].copy()
    recomputed = (
        result.type_matrix.groupby("time", as_index=False)["lr_score"]
        .sum(min_count=1)
        .rename(columns={"lr_score": "score_recomputed"})
    )
    reproduction = formal[["time", "score"]].merge(recomputed, on="time", how="outer")
    reproduction["absolute_error"] = np.abs(
        reproduction["score"] - reproduction["score_recomputed"]
    )
    reproduction["relative_error"] = reproduction["absolute_error"] / np.maximum(
        np.maximum(np.abs(reproduction["score"]), np.abs(reproduction["score_recomputed"])),
        1e-12,
    )
    reproduction_max_error = float(reproduction["absolute_error"].max())
    reproduction_max_relative_error = float(reproduction["relative_error"].max())

    mapped_tables: list[pd.DataFrame] = []
    availability_rows: list[dict[str, object]] = []
    for time_value in PANEL_TIMES:
        state = display_states[str(time_value)]
        labels = state.obs["Annotation"].astype(str).to_numpy()
        xy = np.asarray(state.obsm["spatial"], dtype=np.float64)
        scores = result.type_scores.loc[
            np.isclose(result.type_scores["time"], time_value, rtol=0.0, atol=1e-12)
        ].set_index("cell_type")
        available = np.asarray([label in scores.index for label in labels], dtype=bool)
        missing_counts = pd.Series(labels[~available]).value_counts().sort_index()
        incoming = np.asarray(
            [float(scores.loc[label, "incoming"]) if label in scores.index else np.nan for label in labels]
        )
        outgoing = np.asarray(
            [float(scores.loc[label, "outgoing"]) if label in scores.index else np.nan for label in labels]
        )
        total = incoming + outgoing
        mapped_tables.append(
            pd.DataFrame(
                {
                    "time": time_value,
                    "pair_id": result.type_scores["pair_id"].iloc[0],
                    "ligand": "Wnt3a",
                    "receptor": "Fzd7_Lrp6",
                    "pair": "Wnt3a_Fzd7_Lrp6",
                    "cell_index": np.arange(len(labels), dtype=int),
                    "cell_id": state.obs_names.astype(str),
                    "cell_type": labels,
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "incoming": incoming,
                    "outgoing": outgoing,
                    "total_raw": total,
                    "cell_type_score_available": available,
                    "expression_source": (
                        "inverse_pca" if time_value == 0.5 else "observed"
                    ),
                }
            )
        )
        availability_rows.append(
            {
                "time": time_value,
                "n_display_cells": int(len(labels)),
                "n_available_cells": int(available.sum()),
                "n_unavailable_cells": int((~available).sum()),
                "unavailable_fraction": float((~available).mean()),
                "unavailable_types": ";".join(missing_counts.index.astype(str)),
                "unavailable_type_counts": ";".join(
                    f"{label}:{int(count)}" for label, count in missing_counts.items()
                ),
                "policy": "retain coordinates; unavailable NaN; grey base only; never zero-fill",
            }
        )
    cell_mapping = pd.concat(mapped_tables, ignore_index=True)
    availability = pd.DataFrame(availability_rows)
    finite = cell_mapping.loc[
        cell_mapping["cell_type_score_available"], "total_raw"
    ].to_numpy(dtype=np.float64)
    if not len(finite) or not np.isfinite(finite).all() or (finite < 0).any():
        raise RuntimeError("Available display scores are invalid.")
    shared_low = float(np.percentile(finite, 1.0))
    shared_high = float(np.percentile(finite, 99.0))
    if not shared_high > shared_low:
        raise RuntimeError("Shared robust normalization is degenerate.")
    cell_mapping["total_norm_shared_q1_q99"] = np.nan
    available_mask = cell_mapping["cell_type_score_available"].to_numpy(dtype=bool)
    cell_mapping.loc[available_mask, "total_norm_shared_q1_q99"] = np.clip(
        (cell_mapping.loc[available_mask, "total_raw"] - shared_low)
        / (shared_high - shared_low),
        0.0,
        1.0,
    )

    accepted_n = int(compute_states["0.5"].n_obs)
    full_n = int(display_states["0.5"].n_obs)
    accepted_edges = communication_summary["edge_selection_by_time"]["0.5"]
    full_edges = sensitivity_gate["generated_t0p5_communication"]["edge_selection"]
    accepted_score = float(
        reproduction.loc[np.isclose(reproduction["time"], 0.5), "score"].iloc[0]
    )
    full_score = float(
        sensitivity_reproduction.loc[
            np.isclose(sensitivity_reproduction["time"], 0.5), "score_recomputed"
        ].iloc[0]
    )
    density_sensitivity = {
        "status": "REJECT_FULL50K_AS_PRIMARY_COMPUTE",
        "reason": "Full-radius M_per_source hotspot scales with numerical particle density; 50k is display-only.",
        "accepted_compute_cells": accepted_n,
        "full50k_display_cells": full_n,
        "cell_count_ratio_full_over_accepted": full_n / accepted_n,
        "accepted_candidate_edges": int(accepted_edges["candidate_count"]),
        "full50k_candidate_edges": int(full_edges["candidate_count"]),
        "candidate_edges_per_cell_accepted": int(accepted_edges["candidate_count"]) / accepted_n,
        "candidate_edges_per_cell_full50k": int(full_edges["candidate_count"]) / full_n,
        "candidate_edges_per_cell_ratio": (
            int(full_edges["candidate_count"]) / full_n
        ) / (int(accepted_edges["candidate_count"]) / accepted_n),
        "accepted_pair_score": accepted_score,
        "full50k_pair_score": full_score,
        "pair_score_ratio_full_over_accepted": full_score / accepted_score,
    }

    audit = result.audit.copy()
    within_type_counts = (
        result.cell_mapping.groupby(["time", "cell_type"])["total_raw"].nunique(dropna=False)
    )
    gate = {
        "status": "PASS",
        "dataset": "mosta",
        "panel": "Fig4b",
        "pair": "Wnt3a_Fzd7_Lrp6",
        "hotspot": "total",
        "estimand": "mean_sender(Wnt3a) * mean_receiver(min(Fzd7,Lrp6)) * M_per_source; incoming+outgoing",
        "complex_mode": "min",
        "require_all_subunits": True,
        "compute_cohort": "accepted corrected workflow snapshots (configured initial cap 12000; t0.5 realized 15144)",
        "display_cohort": "50k initial-particle global-t0 manuscript run",
        "display_mapping_reestimates_scores": False,
        "trajectory_mode": "global_t0_extrapolation",
        "restart_from_preceding_observed_stage": False,
        "spatial_warp": False,
        "package_commit": package_commit,
        "max_formula_abs_error": float(audit["max_formula_abs_error"].max()),
        "formal_primary_reproduction_max_abs_error": reproduction_max_error,
        "formal_primary_reproduction_max_relative_error": (
            reproduction_max_relative_error
        ),
        "formal_primary_reproduction_tolerance": {
            "absolute": 1e-8,
            "relative": 1e-8,
            "rationale": "float32 saved-state reconstruction and summation-order tolerance",
        },
        "missing_subunits_empty": bool(
            audit["missing_subunits"].map(empty_contract_field).all()
        ),
        "unreconstructable_subunits_empty": bool(
            audit["unreconstructable_subunits"].map(empty_contract_field).all()
        ),
        "core_compute_n_unmapped_cells": int(audit["n_unmapped_cells"].sum()),
        "within_time_type_score_max_nunique": int(within_type_counts.max()),
        "display_mapping": {
            "n_display_cells": int(len(cell_mapping)),
            "n_available_cells": int(available_mask.sum()),
            "n_unavailable_cells": int((~available_mask).sum()),
            "n_dropped_cells": 0,
            "unavailable_types": sorted(
                cell_mapping.loc[~available_mask, "cell_type"].astype(str).unique()
            ),
            "unavailable_policy": "retain and render base grey only; NaN unavailable; no zero imputation",
        },
        "shared_normalization": {
            "scope": "all finite display scores across the three panels",
            "q_low_percent": 1.0,
            "q_high_percent": 99.0,
            "low": shared_low,
            "high": shared_high,
            "legacy_per_panel_normalization_used": False,
            "unavailable_cells_excluded": True,
        },
        "particle_density_sensitivity": density_sensitivity,
        "compute_state_gates": compute_state_gates,
        "display_state_gates": display_state_gates,
    }
    pass_checks = [
        gate["max_formula_abs_error"] <= 1e-12,
        gate["formal_primary_reproduction_max_abs_error"] <= 1e-8,
        gate["formal_primary_reproduction_max_relative_error"] <= 1e-8,
        gate["missing_subunits_empty"],
        gate["unreconstructable_subunits_empty"],
        gate["core_compute_n_unmapped_cells"] == 0,
        gate["within_time_type_score_max_nunique"] == 1,
        gate["display_mapping"]["n_dropped_cells"] == 0,
        cell_mapping.loc[~available_mask, "total_raw"].isna().all(),
    ]
    if not all(pass_checks):
        gate["status"] = "FAIL"

    result.type_matrix.to_csv(output_dir / "type_matrix.csv", index=False)
    result.type_scores.to_csv(output_dir / "type_scores.csv", index=False)
    audit.to_csv(output_dir / "package_audit.csv", index=False)
    availability.to_csv(output_dir / "display_mapping_availability.csv", index=False)
    reproduction.to_csv(output_dir / "formal_primary_reproduction.csv", index=False)
    with gzip.open(output_dir / "cell_mapping.csv.gz", "wt", encoding="utf-8") as handle:
        cell_mapping.to_csv(handle, index=False)
    (output_dir / "settings.json").write_text(
        json.dumps(result.settings, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "calculation_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "complete" if gate["status"] == "PASS" else "failed_gate",
        "dataset": "mosta",
        "panel": "Fig4b",
        "scientific_contract": gate,
        "inputs": {
            "workflow_summary": input_record(workflow_path),
            "display_summary": input_record(display_summary_path),
            "reference_h5ad": input_record(
                reference_path, known_sha256=EXPECTED_REFERENCE_SHA256
            ),
            "communication_by_celltype": input_record(communication_path),
            "formal_pair_timecourse": input_record(
                Path(lr_summary["tables"]["pair_timecourse"])
            ),
            "compute_snapshots": [input_record(compute_paths[t]) for t in PANEL_TIMES],
            "display_snapshots": [input_record(display_paths[t]) for t in PANEL_TIMES],
            "full50k_sensitivity_gate": input_record(
                sensitivity_dir / "calculation_gate.json"
            ),
            "full50k_sensitivity_reproduction": input_record(
                sensitivity_dir / "formal_primary_reproduction.csv"
            ),
        },
        "package": {
            "root": str(package_root),
            "commit": package_commit,
            "module": str(Path(cb.__file__).resolve()),
            "source_files": {
                "lr_projection.py": input_record(
                    package_root / "CytoBridge" / "tl" / "downstream" / "lr_projection.py"
                )
            },
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "wall_seconds": float(time.time() - started),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    checksum_path = output_dir / "SHA256SUMS.txt"
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output_dir)}\n" for path in files),
        encoding="utf-8",
    )
    if gate["status"] != "PASS":
        raise RuntimeError("Fig. 4b accepted-compute/display-50k gate failed.")
    (output_dir / "COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
