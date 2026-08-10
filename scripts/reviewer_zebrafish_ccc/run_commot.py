#!/usr/bin/env python3
"""Run COMMOT once across all prepared observed zebrafish stages.

The runner consumes the method-neutral bundle created by ``prepare_inputs.py``.
It does not re-normalize expression, re-filter LR rows by stage, or silently
subsample cells. Detailed LR/pathway tables remain positive-only, while the
primary type-pair table exports the complete directed cell-type square.
"""

from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from scipy.spatial import cKDTree

try:
    from .common import (
        COMMON_SCORE_COLUMNS,
        ensure_common_score_schema,
        file_record,
        json_dump,
        software_versions,
        utc_now,
    )
except ImportError:  # direct script execution
    from common import (
        COMMON_SCORE_COLUMNS,
        ensure_common_score_schema,
        file_record,
        json_dump,
        software_versions,
        utc_now,
    )


DATABASE_NAME = "zebrafish_current_cellchatdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--distance-mode",
        choices=["cytobridge_graph", "knn"],
        default="cytobridge_graph",
        help="Primary reads the frozen CytoBridge graph cutoff; 'knn' is an explicit sensitivity.",
    )
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--distance-k", type=int, default=10)
    parser.add_argument("--distance-quantile", type=float, default=0.95)
    parser.add_argument("--cot-nitermax", type=int, default=2000)
    parser.add_argument("--save-result-h5ad", action="store_true")
    return parser.parse_args()


def load_stage(input_dir: Path, stage_record: dict[str, object]) -> ad.AnnData:
    stage_dir = input_dir / "stages" / str(stage_record["token"])
    expression = mmread(stage_dir / "expression_genes_by_cells.mtx")
    expression = sparse.csr_matrix(expression).T.tocsr().astype(np.float32)
    genes = [
        line.strip()
        for line in (stage_dir / "genes.txt").read_text().splitlines()
        if line.strip()
    ]
    metadata = pd.read_csv(
        stage_dir / "metadata.csv", dtype={"cell_id": str, "label": str, "stage": str}
    )
    spatial = pd.read_csv(stage_dir / "spatial_aligned.csv", dtype={"cell_id": str})
    if expression.shape != (len(metadata), len(genes)):
        raise ValueError(
            f"Stage {stage_record['stage']!r} matrix shape {expression.shape} does not "
            f"match metadata/genes {(len(metadata), len(genes))}"
        )
    if metadata["cell_id"].tolist() != spatial["cell_id"].tolist():
        raise ValueError(
            f"Stage {stage_record['stage']!r} metadata/spatial cell order differs"
        )
    if metadata["cell_id"].duplicated().any() or len(set(genes)) != len(genes):
        raise ValueError(
            f"Stage {stage_record['stage']!r} has duplicate cell or gene IDs"
        )
    coordinate_columns = [column for column in spatial if column.startswith("coord_")]
    if len(coordinate_columns) < 2:
        raise ValueError(
            "Prepared spatial input must have at least two coordinate columns"
        )
    result = ad.AnnData(
        X=expression,
        obs=metadata.set_index("cell_id", drop=False),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    result.obs["commot_label"] = result.obs["label"].astype(str)
    result.obsm["spatial"] = spatial[coordinate_columns].to_numpy(dtype=float)
    return result


def infer_distance_threshold(
    coordinates: np.ndarray, *, k: int = 10, quantile: float = 0.95
) -> float:
    """Infer a local cutoff as a quantile of each cell's k-th-neighbor distance."""

    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("At least two spatial points are required")
    if k < 1:
        raise ValueError("distance k must be positive")
    if not 0 < quantile <= 1:
        raise ValueError("distance quantile must be in (0, 1]")
    effective_k = min(int(k), coordinates.shape[0] - 1)
    distances, _ = cKDTree(coordinates).query(coordinates, k=effective_k + 1)
    kth = np.asarray(distances)[:, effective_k]
    cutoff = float(np.quantile(kth[np.isfinite(kth)], quantile))
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Could not infer a positive finite distance threshold")
    return cutoff


def aggregate_matrix_by_labels(
    matrix: object, labels: Iterable[str], *, include_zeros: bool = False
) -> pd.DataFrame:
    """Aggregate a sender-row/receiver-column COMMOT matrix by cell labels."""

    label_array = np.asarray(list(labels), dtype=str)
    groups = np.sort(pd.unique(label_array))
    group_lookup = {group: idx for idx, group in enumerate(groups)}
    group_index = np.asarray([group_lookup[label] for label in label_array], dtype=int)
    counts = np.bincount(group_index, minlength=len(groups))
    coo = sparse.coo_matrix(matrix)
    flat_index = group_index[coo.row] * len(groups) + group_index[coo.col]
    sums = np.bincount(
        flat_index, weights=coo.data, minlength=len(groups) ** 2
    ).reshape(len(groups), len(groups))
    diagonal_mask = coo.row == coo.col
    diagonal_sums = np.bincount(
        group_index[coo.row[diagonal_mask]],
        weights=coo.data[diagonal_mask],
        minlength=len(groups),
    )
    rows: list[dict[str, object]] = []
    for sender_idx, sender in enumerate(groups):
        for receiver_idx, receiver in enumerate(groups):
            value = float(sums[sender_idx, receiver_idx])
            if not include_zeros and value == 0:
                continue
            denominator = int(counts[sender_idx] * counts[receiver_idx])
            shared_cells = int(counts[sender_idx]) if sender_idx == receiver_idx else 0
            distinct_denominator = denominator - shared_cells
            distinct_value = (
                value - float(diagonal_sums[sender_idx])
                if sender_idx == receiver_idx
                else value
            )
            tolerance = 1e-12 * max(1.0, abs(value))
            if distinct_value < -tolerance:
                raise ValueError("Cell-diagonal COMMOT mass exceeds its type block")
            distinct_value = max(0.0, distinct_value)
            if distinct_denominator <= 0:
                distinct_mean = np.nan
            else:
                distinct_mean = distinct_value / distinct_denominator
            rows.append(
                {
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "score": value,
                    "score_mean_possible_cell_pairs": value / denominator,
                    "score_distinct_cell_pairs": distinct_value,
                    "score_mean_possible_distinct_cell_pairs": distinct_mean,
                    "n_shared_sender_receiver_cells": shared_cells,
                    "n_possible_distinct_cell_pairs": distinct_denominator,
                    "n_sender_cells": int(counts[sender_idx]),
                    "n_receiver_cells": int(counts[receiver_idx]),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "sender_type",
            "receiver_type",
            "score",
            "score_mean_possible_cell_pairs",
            "score_distinct_cell_pairs",
            "score_mean_possible_distinct_cell_pairs",
            "n_shared_sender_receiver_cells",
            "n_possible_distinct_cell_pairs",
            "n_sender_cells",
            "n_receiver_cells",
        ],
    )


def _common_context(
    frame: pd.DataFrame,
    *,
    stage: str,
    stage_time: float | None,
    ligand: str,
    receptor: str,
    pathway: str,
    category: str,
    interaction_id: str,
    matrix_key: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=COMMON_SCORE_COLUMNS
            + [
                "score_mean_possible_cell_pairs",
                "abundance_controlled_score",
                "score_distinct_cell_pairs",
                "score_mean_possible_distinct_cell_pairs",
                "abundance_controlled_distinct_cell_score",
                "n_shared_sender_receiver_cells",
                "n_possible_distinct_cell_pairs",
                "matrix_key",
            ]
        )
    result = frame.copy()
    result.insert(0, "method", "COMMOT")
    result.insert(1, "database_variant", "current_zebrafish_lr_database")
    result.insert(2, "stage", stage)
    result.insert(3, "stage_time", stage_time)
    result["ligand"] = ligand
    result["receptor"] = receptor
    result["pathway"] = pathway
    result["category"] = category
    result["interaction_id"] = interaction_id
    result["p_value"] = np.nan
    result["significant"] = pd.NA
    result[
        "score_semantics"
    ] = "sum of COMMOT sender-row/receiver-column cell-cell OT communication mass"
    result["abundance_controlled_score"] = result["score_mean_possible_cell_pairs"]
    result["abundance_controlled_distinct_cell_score"] = result[
        "score_mean_possible_distinct_cell_pairs"
    ]
    result["matrix_key"] = matrix_key
    return ensure_common_score_schema(result)


def extract_commot_tables(
    adata: ad.AnnData,
    database: pd.DataFrame,
    *,
    stage: str,
    stage_time: float | None,
    database_name: str = DATABASE_NAME,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    labels = adata.obs["commot_label"].astype(str).to_numpy()
    prefix = f"commot-{database_name}-"
    lr_frames: list[pd.DataFrame] = []
    missing_lr_keys: list[str] = []
    availability_rows: list[dict[str, object]] = []
    # COMMOT identifies a cell-cell matrix by expanded ligand/receptor.  Exact
    # flat duplicates are emitted once, with every source database_row retained
    # in the audit column rather than counted repeatedly as independent results.
    unique_lr = database.drop_duplicates(["ligand", "receptor", "pathway", "category"])
    for row in unique_lr.itertuples(index=False):
        key = f"{prefix}{row.ligand}-{row.receptor}"
        method_available = key in adata.obsp
        availability_rows.append(
            {
                "stage": stage,
                "stage_time": stage_time,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "pathway": row.pathway,
                "category": row.category,
                "matrix_key": key,
                "method_available": bool(method_available),
            }
        )
        if not method_available:
            missing_lr_keys.append(key)
            continue
        aggregated = aggregate_matrix_by_labels(adata.obsp[key], labels)
        if aggregated.empty:
            continue
        source_rows = database.loc[
            (database["ligand"] == row.ligand)
            & (database["receptor"] == row.receptor)
            & (database["pathway"] == row.pathway)
            & (database["category"] == row.category),
            "database_row",
        ]
        context = _common_context(
            aggregated,
            stage=stage,
            stage_time=stage_time,
            ligand=row.ligand,
            receptor=row.receptor,
            pathway=row.pathway,
            category=row.category,
            interaction_id=f"{row.ligand}->{row.receptor}|{row.pathway}|{row.category}",
            matrix_key=key,
        )
        context["database_rows"] = ";".join(str(int(value)) for value in source_rows)
        lr_frames.append(context)

    pathway_frames: list[pd.DataFrame] = []
    for pathway in sorted(database["pathway"].unique()):
        key = f"{prefix}{pathway}"
        if key not in adata.obsp:
            continue
        aggregated = aggregate_matrix_by_labels(adata.obsp[key], labels)
        categories = ";".join(
            sorted(database.loc[database["pathway"] == pathway, "category"].unique())
        )
        pathway_frames.append(
            _common_context(
                aggregated,
                stage=stage,
                stage_time=stage_time,
                ligand="",
                receptor="",
                pathway=pathway,
                category=categories,
                interaction_id=f"pathway:{pathway}",
                matrix_key=key,
            )
        )

    total_key = f"{prefix}total-total"
    if total_key not in adata.obsp:
        raise KeyError(f"COMMOT did not create required total matrix {total_key!r}")
    total = _common_context(
        aggregate_matrix_by_labels(adata.obsp[total_key], labels, include_zeros=True),
        stage=stage,
        stage_time=stage_time,
        ligand="",
        receptor="",
        pathway="__all__",
        category="__all__",
        interaction_id="total",
        matrix_key=total_key,
    )
    lr = (
        pd.concat(lr_frames, ignore_index=True)
        if lr_frames
        else pd.DataFrame(columns=COMMON_SCORE_COLUMNS)
    )
    pathway = (
        pd.concat(pathway_frames, ignore_index=True)
        if pathway_frames
        else pd.DataFrame(columns=COMMON_SCORE_COLUMNS)
    )
    diagnostics = {
        "n_unique_flat_lr_rows": int(len(unique_lr)),
        "n_lr_matrix_keys_missing": len(missing_lr_keys),
        "missing_lr_matrix_keys_first_20": missing_lr_keys[:20],
        "n_lr_positive_context_rows": int(len(lr)),
        "n_pathway_positive_context_rows": int(len(pathway)),
        "n_total_context_rows": int(len(total)),
        "n_total_positive_context_rows": int((total["score"] > 0).sum()),
        "n_total_structural_zero_rows": int((total["score"] == 0).sum()),
    }
    return lr, pathway, total, pd.DataFrame(availability_rows), diagnostics


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression="gzip")


def main() -> None:
    args = parse_args()
    if args.distance_threshold is not None and args.distance_threshold <= 0:
        raise ValueError("--distance-threshold must be positive")
    if args.cot_nitermax < 1:
        raise ValueError("--cot-nitermax must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.out_dir}")

    input_manifest_path = args.input_dir / "input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    database_path = args.input_dir / "filtered_lr_database.csv"
    database = pd.read_csv(database_path)
    commot_database = database[
        ["ligand", "receptor", "pathway", "category"]
    ].drop_duplicates()
    commot_database.columns = ["0", "1", "2", "3"]

    # COMMOT 0.0.3 references the removed NumPy alias.  This compatibility
    # assignment does not alter numerical behavior.
    np.Inf = np.inf
    import commot as ct

    lr_frames: list[pd.DataFrame] = []
    pathway_frames: list[pd.DataFrame] = []
    total_frames: list[pd.DataFrame] = []
    availability_frames: list[pd.DataFrame] = []
    stage_diagnostics: list[dict[str, object]] = []
    interaction_graph = input_manifest.get("preprocessing", {}).get(
        "interaction_graph", {}
    )
    graph_cutoff = interaction_graph.get("neighborhood_threshold")
    spot_diameter = interaction_graph.get("recommended_spot_diameter")
    if args.distance_threshold is not None:
        distance_rule = "explicit_cli_override"
        shared_cutoff = float(args.distance_threshold)
    elif args.distance_mode == "cytobridge_graph":
        if (
            graph_cutoff is None
            or not np.isfinite(float(graph_cutoff))
            or float(graph_cutoff) <= 0
        ):
            raise ValueError(
                "Primary COMMOT requires input_manifest preprocessing.interaction_graph."
                "neighborhood_threshold; use --distance-mode knn only for an explicit sensitivity"
            )
        distance_rule = "frozen_h5ad_interaction_graph_neighborhood_threshold"
        shared_cutoff = float(graph_cutoff)
    else:
        distance_rule = "stage_specific_knn_sensitivity"
        shared_cutoff = None
    for stage_record in input_manifest["stages"]:
        stage = str(stage_record["stage"])
        stage_time = stage_record.get("stage_time")
        snapshot = load_stage(args.input_dir, stage_record)
        cutoff = (
            shared_cutoff
            if shared_cutoff is not None
            else infer_distance_threshold(
                snapshot.obsm["spatial"],
                k=args.distance_k,
                quantile=args.distance_quantile,
            )
        )
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
        lr, pathway, total, availability, diagnostics = extract_commot_tables(
            snapshot, database, stage=stage, stage_time=stage_time
        )
        lr_frames.append(lr)
        pathway_frames.append(pathway)
        total_frames.append(total)
        availability_frames.append(availability)
        diagnostics.update(
            {
                "stage": stage,
                "stage_time": stage_time,
                "n_cells": int(snapshot.n_obs),
                "n_cell_types": int(snapshot.obs["commot_label"].nunique()),
                "distance_threshold": cutoff,
            }
        )
        stage_diagnostics.append(diagnostics)
        if args.save_result_h5ad:
            snapshot.write_h5ad(
                args.out_dir / f"commot_{stage_record['token']}.h5ad",
                compression="gzip",
            )

    lr_all = pd.concat(lr_frames, ignore_index=True)
    pathway_all = pd.concat(pathway_frames, ignore_index=True)
    total_all = pd.concat(total_frames, ignore_index=True)
    availability_all = pd.concat(availability_frames, ignore_index=True)
    output_paths = {
        "lr_scores": args.out_dir / "commot_lr_scores.csv.gz",
        "pathway_scores": args.out_dir / "commot_pathway_scores.csv.gz",
        "type_pair_scores": args.out_dir / "commot_type_pair_scores.csv.gz",
        "lr_axis_stage_availability": (
            args.out_dir / "commot_lr_axis_stage_availability.csv.gz"
        ),
    }
    _write_table(lr_all, output_paths["lr_scores"])
    _write_table(pathway_all, output_paths["pathway_scores"])
    _write_table(total_all, output_paths["type_pair_scores"])
    _write_table(availability_all, output_paths["lr_axis_stage_availability"])

    versions = software_versions()
    commot_version = str(getattr(ct, "__version__", "")).strip()
    if not commot_version or commot_version.casefold() == "unknown":
        try:
            commot_version = importlib_metadata.version("commot")
        except importlib_metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "COMMOT version is unavailable from both the module and package metadata"
            ) from error
    versions["commot"] = commot_version
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "method": "COMMOT",
        "database_variant": "current_zebrafish_lr_database",
        "input_manifest": file_record(input_manifest_path),
        "database": file_record(database_path),
        "design": {
            "all_prepared_observed_stages": True,
            "expression_retransformed_in_runner": False,
            "spatial_coordinates": "prepared spatial_aligned coordinates",
            "spatial_matrix_orientation": "rows are sender cells; columns are receiver cells",
            "heteromeric": True,
            "heteromeric_rule": "min",
            "distance_mode": args.distance_mode,
            "distance_rule_used": distance_rule,
            "explicit_distance_threshold": args.distance_threshold,
            "frozen_interaction_graph_neighborhood_threshold": graph_cutoff,
            "recommended_spot_diameter": spot_diameter,
            "threshold_over_spot_diameter": (
                None
                if graph_cutoff is None or spot_diameter is None
                else float(graph_cutoff) / float(spot_diameter)
            ),
            "knn_sensitivity_rule": {
                "rule": "quantile of per-cell k-th-nearest-neighbor distance",
                "k": args.distance_k,
                "quantile": args.distance_quantile,
            },
            "category_distance_handling": (
                "one dis_thr is passed to COMMOT for every LR row; Secreted Signaling, "
                "ECM-Receptor, and Cell-Cell Contact categories are recorded but do not "
                "receive category-specific cutoffs in this run"
            ),
            "cot_nitermax": args.cot_nitermax,
            "stage_prevalence_filter": None,
            "flat_exact_duplicates": "deduplicated for COMMOT execution and linked back through database_rows",
            "long_table_zero_policy": (
                "LR and pathway structural zeros omitted; primary type-pair table exports "
                "the complete directed stage-specific cell-type square"
            ),
            "lr_axis_stage_availability": (
                "explicit matrix-key availability for every requested stage x LR row; "
                "downstream loaders may zero-complete context rows only when this table "
                "marks the stage x axis available"
            ),
            "type_pair_grid_export": {
                "complete_directed_stage_type_square": True,
                "zero_score_semantics": (
                    "evaluated COMMOT total communication mass is exactly zero for this "
                    "sender/receiver type block"
                ),
                "universe_source": "input_manifest.stages[].cell_type_counts",
                "loader_zero_completion_required": False,
            },
        },
        "score_semantics": {
            "score": "sum of native COMMOT cell-cell OT communication mass within a sender/receiver type block",
            "score_mean_possible_cell_pairs": "score divided by n_sender_cells*n_receiver_cells; not a COMMOT-native probability",
            "score_distinct_cell_pairs": (
                "COMMOT block mass after removing cell-diagonal entries for "
                "homotypic sender/receiver labels"
            ),
            "score_mean_possible_distinct_cell_pairs": (
                "distinct-cell score divided by n_sender*n_receiver minus the "
                "sender/receiver cell-set overlap; homotypic denominator is n*(n-1)"
            ),
            "abundance_controlled_distinct_cell_score": (
                "score_mean_possible_distinct_cell_pairs; required for comparisons "
                "to methods that exclude self edges"
            ),
            "abundance_controlled_score": (
                "score_mean_possible_cell_pairs; use this rank for the primary comparison "
                "to CellChat population.size=false, with native score retained as sensitivity"
            ),
            "p_value": "not provided by COMMOT and exported missing",
            "raw_cross_method_units_comparable": False,
        },
        "stage_diagnostics": stage_diagnostics,
        "software": versions,
        "artifacts": {name: file_record(path) for name, path in output_paths.items()},
    }
    json_dump(manifest, args.out_dir / "manifest.json")
    print(f"COMMOT completed {len(stage_diagnostics)} stages in {args.out_dir}")


if __name__ == "__main__":
    main()
