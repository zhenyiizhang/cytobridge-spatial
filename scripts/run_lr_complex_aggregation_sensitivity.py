#!/usr/bin/env python3
"""Recompute a formal LR result under min and geometric-mean complex gates.

This command deliberately reuses the saved expression-state snapshots and
cell-type communication matrix from a completed package workflow.  It first
recomputes the primary minimum-gate table and requires agreement with the
published primary table; only then does it write the geometric-mean
sensitivity result.  Simulation, classification, communication attention,
the LR database, expression support, and the scored time/pair universe are
therefore held fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.tl.downstream.lr_projection import (  # noqa: E402
    project_communication_to_lr_timecourses,
)


TABLE_NAMES = (
    "pair_timecourse",
    "celltype_timecourse",
    "pattern_summary",
    "coverage",
    "trajectory_coverage",
    "dropped_trajectories",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _fresh_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _communications_from_table(
    table_path: Path, time_points: list[float]
) -> dict[str, dict[str, object]]:
    table = pd.read_csv(table_path, keep_default_na=False)
    required = {"time", "source", "target", "attention_per_source"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Communication table is missing columns: {missing}")
    table["time"] = pd.to_numeric(table["time"], errors="raise")
    table["attention_per_source"] = pd.to_numeric(
        table["attention_per_source"], errors="raise"
    )
    if not np.isfinite(table["attention_per_source"]).all():
        raise ValueError("Communication table contains non-finite values.")
    if (table["attention_per_source"] < 0).any():
        raise ValueError("Communication table contains negative attention values.")
    if table.duplicated(["time", "source", "target"]).any():
        raise ValueError(
            "Communication table contains duplicate time/source/target rows."
        )

    records: dict[str, dict[str, object]] = {}
    observed_times = sorted(table["time"].unique().astype(float).tolist())
    if observed_times != sorted(map(float, time_points)):
        raise ValueError(
            "Communication time grid differs from workflow summary: "
            f"{observed_times} != {sorted(map(float, time_points))}"
        )
    for time_value in time_points:
        subset = table.loc[np.isclose(table["time"], float(time_value))].copy()
        types = subset["source"].astype(str).drop_duplicates().tolist()
        target_types = subset["target"].astype(str).drop_duplicates().tolist()
        if types != target_types:
            raise ValueError(
                f"Communication type order differs for senders and receivers at "
                f"time {time_value}."
            )
        expected_rows = len(types) ** 2
        if len(subset) != expected_rows:
            raise ValueError(
                f"Communication table at time {time_value} has {len(subset)} rows, "
                f"expected {expected_rows}."
            )
        matrix = (
            subset.pivot(
                index="source", columns="target", values="attention_per_source"
            )
            .reindex(index=types, columns=types)
            .to_numpy(dtype=np.float64)
        )
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Communication matrix at time {time_value} is incomplete."
            )
        records[str(float(time_value))] = {
            "types": np.asarray(types, dtype=object),
            "M_per_source": matrix,
        }
    return records


def _observed_times(summary: dict[str, Any]) -> list[float]:
    origins = summary.get("simulation", {}).get("slice_origins_by_time", {})
    if not isinstance(origins, dict) or not origins:
        raise ValueError("Workflow summary lacks simulation.slice_origins_by_time.")
    observed = sorted(
        float(time_value)
        for time_value, origin in origins.items()
        if str(origin) == "observed_real"
    )
    if not observed:
        raise ValueError("Workflow summary identifies no observed_real time points.")
    return observed


def _snapshot_dict(
    summary: dict[str, Any], time_points: list[float]
) -> tuple[dict[str, ad.AnnData], list[Path]]:
    raw_paths = summary.get("snapshots")
    if not isinstance(raw_paths, list) or len(raw_paths) != len(time_points):
        raise ValueError("Workflow summary snapshot list does not match time_points.")
    paths = [_required_file(value, "snapshot") for value in raw_paths]
    snapshots: dict[str, ad.AnnData] = {}
    for time_value, path in zip(time_points, paths):
        snapshot = ad.read_h5ad(path)
        if "Annotation" not in snapshot.obs:
            raise KeyError(f"Snapshot lacks obs['Annotation']: {path}")
        snapshots[str(float(time_value))] = snapshot
    return snapshots, paths


def _tables(result: object) -> dict[str, pd.DataFrame]:
    return {name: getattr(result, name) for name in TABLE_NAMES}


def _write_tables(result: object, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths: dict[str, str] = {}
    for name, table in _tables(result).items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)
    return paths


def _verify_primary_reproduction(
    primary_path: Path, recomputed: pd.DataFrame
) -> dict[str, float | int]:
    primary = pd.read_csv(primary_path, keep_default_na=False)
    required = {"time", "pair", "score"}
    for name, table in (("primary", primary), ("recomputed", recomputed)):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} pair table is missing columns: {missing}")
        if table.duplicated(["time", "pair"]).any():
            raise ValueError(f"{name} pair table contains duplicate time/pair rows.")
    left = primary[["time", "pair", "score"]].rename(columns={"score": "score_primary"})
    right = recomputed[["time", "pair", "score"]].rename(
        columns={"score": "score_recomputed"}
    )
    merged = left.merge(
        right, on=["time", "pair"], how="outer", validate="1:1", indicator=True
    )
    if not merged["_merge"].eq("both").all():
        examples = merged.loc[
            ~merged["_merge"].eq("both"), ["time", "pair", "_merge"]
        ].head()
        raise ValueError(
            "Recomputed minimum gate does not reproduce the formal time/pair "
            f"universe: {examples.to_dict(orient='records')}"
        )
    differences = np.abs(
        merged["score_primary"].to_numpy(dtype=float)
        - merged["score_recomputed"].to_numpy(dtype=float)
    )
    scale = np.maximum(np.abs(merged["score_primary"].to_numpy(dtype=float)), 1.0)
    relative = differences / scale
    max_absolute = float(differences.max(initial=0.0))
    max_relative = float(relative.max(initial=0.0))
    if not np.allclose(
        merged["score_primary"],
        merged["score_recomputed"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Saved snapshots/communication do not reproduce the formal minimum-gate "
            f"scores (max_abs={max_absolute}, max_rel={max_relative})."
        )
    return {
        "n_rows": int(len(merged)),
        "max_absolute_score_difference": max_absolute,
        "max_scaled_score_difference": max_relative,
    }


def _input_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "sha256": _sha256(path)}


def _run_comparator(
    min_table: Path, geometric_table: Path, database: Path, output_dir: Path
) -> None:
    comparator = (
        REPO_ROOT
        / "scripts/reviewer_zebrafish_response/compare_lr_complex_aggregation.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(comparator),
            "--min-table",
            str(min_table),
            "--geometric-table",
            str(geometric_table),
            "--lr-database",
            str(database),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def run(summary_path: Path, output_dir: Path) -> Path:
    summary_path = _required_file(summary_path, "workflow summary")
    output_dir = _fresh_output_dir(output_dir)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lr_summary = summary.get("analyses", {}).get("ligand_receptor", {})
    communication_summary = summary.get("analyses", {}).get("communication", {})
    if lr_summary.get("status") != "completed":
        raise ValueError("Workflow ligand_receptor analysis is not completed.")
    if lr_summary.get("complex_mode") != "min":
        raise ValueError("Formal primary LR analysis must use complex_mode='min'.")
    if lr_summary.get("require_all_subunits") is not True:
        raise ValueError("Formal primary LR analysis must require all subunits.")
    if communication_summary.get("status") != "completed":
        raise ValueError("Workflow communication analysis is not completed.")

    time_points = [float(value) for value in summary.get("time_points", [])]
    if not time_points or time_points != sorted(set(time_points)):
        raise ValueError("Workflow summary has an invalid time_points grid.")
    reference_path = _required_file(summary["reference_h5ad"], "reference H5AD")
    database_path = _required_file(lr_summary["database"], "LR database")
    primary_path = _required_file(
        lr_summary["tables"]["pair_timecourse"], "formal min pair table"
    )
    communication_path = _required_file(
        communication_summary["table"], "communication table"
    )
    snapshots, snapshot_paths = _snapshot_dict(summary, time_points)
    communications = _communications_from_table(communication_path, time_points)
    reference = ad.read_h5ad(reference_path)

    shared = dict(
        adata_dict=snapshots,
        reference_adata=reference,
        communications=communications,
        lr_database=database_path,
        time_points=time_points,
        annotation_key="Annotation",
        spatial_dim=2,
        expression_space="log1p",
        require_all_subunits=True,
        preferred_species_tag=lr_summary.get("preferred_species_tag"),
        observed_adata=reference,
        observed_time_points=_observed_times(summary),
        observed_annotation_key="Annotation",
        observed_expression_space="log1p",
    )
    minimum = project_communication_to_lr_timecourses(**shared, complex_mode="min")
    reproduction = _verify_primary_reproduction(primary_path, minimum.pair_timecourse)
    geometric = project_communication_to_lr_timecourses(
        **shared, complex_mode="geometric_mean"
    )

    min_paths = _write_tables(minimum, output_dir / "min_recomputed")
    geometric_paths = _write_tables(geometric, output_dir / "geometric_mean")
    comparison_dir = output_dir / "comparison"
    _run_comparator(
        Path(min_paths["pair_timecourse"]),
        Path(geometric_paths["pair_timecourse"]),
        database_path,
        comparison_dir,
    )

    inputs = {
        "workflow_summary": _input_record(summary_path),
        "reference_h5ad": _input_record(reference_path),
        "lr_database": _input_record(database_path),
        "formal_min_pair_timecourse": _input_record(primary_path),
        "communication_by_celltype": _input_record(communication_path),
        "snapshots": [_input_record(path) for path in snapshot_paths],
    }
    comparison_summary = json.loads(
        (comparison_dir / "summary.json").read_text(encoding="utf-8")
    )
    output_files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": summary.get("dataset"),
        "scientific_contract": {
            "primary_rule": "minimum expression across every required subunit",
            "sensitivity_rule": "zero-preserving geometric mean across every required subunit",
            "require_all_subunits": True,
            "held_fixed": [
                "saved expression-state snapshots",
                "observed expression rows",
                "cell-type communication attention",
                "LR database and gene mapping",
                "time grid and scored pair universe",
            ],
        },
        "formal_primary_reproduction": reproduction,
        "comparison_summary": comparison_summary,
        "inputs": inputs,
        "outputs": [
            _input_record(path)
            for path in output_files
            if path.name != "run_manifest.json"
        ],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run(args.workflow_summary, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
