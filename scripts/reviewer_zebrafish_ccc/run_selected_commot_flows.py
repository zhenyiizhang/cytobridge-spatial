#!/usr/bin/env python3
"""Reconstruct cell-level COMMOT flows for preselected biological examples.

The formal all-LR COMMOT run exports type-pair summaries and deliberately does
not retain its large cell-by-cell matrices.  This narrow runner uses the exact
same prepared stage inputs, spatial cutoff, heteromeric rule, and COT iteration
budget, but evaluates only the LR pairs listed in an auditable examples table.
It is a visualization addendum and never mutates the formal benchmark output.
"""

from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from .common import file_record, json_dump, software_versions, utc_now
    from .run_commot import DATABASE_NAME, load_stage
except ImportError:  # direct script execution
    from common import file_record, json_dump, software_versions, utc_now
    from run_commot import DATABASE_NAME, load_stage


REQUIRED_EXAMPLE_COLUMNS = (
    "example_id",
    "stage",
    "stage_label",
    "ligand",
    "receptor",
    "pathways",
    "categories",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--examples-csv", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cot-nitermax", type=int, default=2000)
    parser.add_argument("--distance-threshold", type=float, default=None)
    return parser


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks columns: {missing}")


def _stage_key(value: object) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Invalid stage value: {value!r}")
    return result


def extract_selected_cell_flows(
    snapshot: Any,
    examples: pd.DataFrame,
    *,
    database_name: str = DATABASE_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Extract every positive sender-row/receiver-column value for each LR."""

    _require_columns(examples, REQUIRED_EXAMPLE_COLUMNS, "examples table")
    cell_ids = snapshot.obs["cell_id"].astype(str).to_numpy()
    labels = snapshot.obs["commot_label"].astype(str).to_numpy()
    prefix = f"commot-{database_name}-"
    flow_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for example in examples.itertuples(index=False):
        key = f"{prefix}{example.ligand}-{example.receptor}"
        if key not in snapshot.obsp:
            raise KeyError(f"COMMOT did not create selected LR matrix {key!r}")
        matrix = sparse.coo_matrix(snapshot.obsp[key])
        positive = np.asarray(matrix.data, dtype=float) > 0
        row = np.asarray(matrix.row, dtype=int)[positive]
        col = np.asarray(matrix.col, dtype=int)[positive]
        value = np.asarray(matrix.data, dtype=float)[positive]
        flows = pd.DataFrame(
            {
                "example_id": str(example.example_id),
                "stage": _stage_key(example.stage),
                "stage_label": str(example.stage_label),
                "ligand": str(example.ligand),
                "receptor": str(example.receptor),
                "pathways": str(example.pathways),
                "categories": str(example.categories),
                "source_index_stage": row,
                "target_index_stage": col,
                "source_cell_id": cell_ids[row],
                "target_cell_id": cell_ids[col],
                "sender_type": labels[row],
                "receiver_type": labels[col],
                "commot_flow": value,
                "matrix_key": key,
            }
        ).sort_values(
            ["commot_flow", "source_index_stage", "target_index_stage"],
            ascending=[False, True, True],
        )
        flow_frames.append(flows)
        outgoing = np.asarray(snapshot.obsp[key].sum(axis=1)).ravel().astype(float)
        incoming = np.asarray(snapshot.obsp[key].sum(axis=0)).ravel().astype(float)
        summary_frames.append(
            pd.DataFrame(
                {
                    "example_id": str(example.example_id),
                    "stage": _stage_key(example.stage),
                    "stage_label": str(example.stage_label),
                    "ligand": str(example.ligand),
                    "receptor": str(example.receptor),
                    "cell_index_stage": np.arange(snapshot.n_obs, dtype=int),
                    "cell_id": cell_ids,
                    "cell_type": labels,
                    "commot_outgoing": outgoing,
                    "commot_incoming": incoming,
                }
            )
        )
        diagnostics.append(
            {
                "example_id": str(example.example_id),
                "matrix_key": key,
                "n_positive_cell_flows": int(positive.sum()),
                "total_cell_flow": float(value.sum()),
                "max_cell_flow": float(value.max()) if value.size else 0.0,
            }
        )
    flows = pd.concat(flow_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    return flows, summaries, diagnostics


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cot_nitermax < 1:
        raise ValueError("--cot-nitermax must be positive")
    if args.distance_threshold is not None and args.distance_threshold <= 0:
        raise ValueError("--distance-threshold must be positive")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")

    input_dir = args.input_dir.expanduser().resolve()
    examples_path = args.examples_csv.expanduser().resolve()
    examples = pd.read_csv(examples_path)
    _require_columns(examples, REQUIRED_EXAMPLE_COLUMNS, "examples table")
    if examples.empty or examples["example_id"].duplicated().any():
        raise ValueError("Examples must be non-empty with unique example_id values")
    input_manifest_path = input_dir / "input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    database_path = input_dir / "filtered_lr_database.csv"
    database = pd.read_csv(database_path)
    graph_cutoff = (
        input_manifest.get("preprocessing", {})
        .get("interaction_graph", {})
        .get("neighborhood_threshold")
    )
    cutoff = (
        float(args.distance_threshold)
        if args.distance_threshold is not None
        else float(graph_cutoff)
    )
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("A positive frozen or explicit spatial cutoff is required")

    stage_records = {
        _stage_key(record["stage"]): record for record in input_manifest["stages"]
    }
    selected_stages = sorted({_stage_key(value) for value in examples["stage"]})
    missing_stages = sorted(set(selected_stages) - set(stage_records))
    if missing_stages:
        raise ValueError(f"Examples use unavailable stages: {missing_stages}")

    # COMMOT 0.0.3 references a NumPy alias removed in newer NumPy.
    np.Inf = np.inf
    import commot as ct

    flow_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    stage_diagnostics: list[dict[str, Any]] = []
    for stage in selected_stages:
        stage_examples = examples.loc[
            examples["stage"].map(_stage_key).eq(stage)
        ].copy()
        selected_database = database.merge(
            stage_examples[["ligand", "receptor"]].drop_duplicates(),
            on=["ligand", "receptor"],
            how="inner",
        )
        absent = (
            stage_examples[["ligand", "receptor"]]
            .drop_duplicates()
            .merge(
                selected_database[["ligand", "receptor"]].drop_duplicates(),
                on=["ligand", "receptor"],
                how="left",
                indicator=True,
            )
        )
        absent = absent.loc[absent["_merge"].eq("left_only")]
        if not absent.empty:
            raise ValueError(
                "Selected examples are absent from prepared LR database: "
                + absent[["ligand", "receptor"]].to_dict("records").__repr__()
            )
        commot_database = selected_database[
            ["ligand", "receptor", "pathway", "category"]
        ].drop_duplicates()
        commot_database.columns = ["0", "1", "2", "3"]
        snapshot = load_stage(input_dir, stage_records[stage])
        ct.tl.spatial_communication(
            snapshot,
            database_name=DATABASE_NAME,
            df_ligrec=commot_database,
            pathway_sum=True,
            heteromeric=True,
            heteromeric_rule="min",
            heteromeric_delimiter="_",
            dis_thr=cutoff,
            cot_nitermax=args.cot_nitermax,
        )
        flows, summaries, diagnostics = extract_selected_cell_flows(
            snapshot, stage_examples
        )
        flow_frames.append(flows)
        summary_frames.append(summaries)
        stage_diagnostics.append(
            {
                "stage": stage,
                "stage_label": str(stage_examples["stage_label"].iloc[0]),
                "n_cells": int(snapshot.n_obs),
                "n_lr_examples": int(len(stage_examples)),
                "distance_threshold": cutoff,
                "examples": diagnostics,
            }
        )

    flows = pd.concat(flow_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    flow_path = out_dir / "selected_commot_cell_flows.csv.gz"
    summary_path = out_dir / "selected_commot_cell_summary.csv.gz"
    flows.to_csv(flow_path, index=False, compression="gzip")
    summaries.to_csv(summary_path, index=False, compression="gzip")
    versions = software_versions()
    try:
        versions["commot"] = importlib_metadata.version("commot")
    except importlib_metadata.PackageNotFoundError:
        versions["commot"] = str(getattr(ct, "__version__", "unknown"))
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workflow": "zebrafish_selected_lr_commot_cell_flow_reconstruction",
        "status": "complete",
        "scope": "visualization addendum; formal all-LR benchmark is unchanged",
        "design": {
            "same_prepared_stage_inputs_as_formal_commot": True,
            "database_name": DATABASE_NAME,
            "spatial_matrix_orientation": "rows sender cells; columns receiver cells",
            "distance_threshold": cutoff,
            "distance_threshold_source": (
                "explicit_cli_override"
                if args.distance_threshold is not None
                else "input_manifest.preprocessing.interaction_graph.neighborhood_threshold"
            ),
            "cot_nitermax": args.cot_nitermax,
            "heteromeric": True,
            "heteromeric_rule": "min",
            "positive_flow_rows_only": True,
            "all_positive_selected_lr_cell_flows_retained": True,
        },
        "inputs": {
            "input_manifest": file_record(input_manifest_path),
            "filtered_lr_database": file_record(database_path),
            "examples": file_record(examples_path),
        },
        "stage_diagnostics": stage_diagnostics,
        "software": versions,
        "artifacts": {
            "selected_commot_cell_flows": file_record(flow_path),
            "selected_commot_cell_summary": file_record(summary_path),
        },
    }
    json_dump(manifest, out_dir / "manifest.json")
    print(f"Selected COMMOT cell flows completed in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
