#!/usr/bin/env python3
"""Build biology-first zebrafish communication case studies.

The first stage of this workflow screens *exact* directed circuits rather than
selecting a ligand-receptor axis after collapsing over cell identity.  A row is
therefore one observed developmental stage, one ligand-receptor axis, and one
sender-cell-type -> receiver-cell-type circuit.  CytoBridge and COMMOT are
ranked only after they have been put on this identical biological unit.

This script deliberately distinguishes model-native quantities from post-hoc
LR-compatible scores.  In particular, ``attention * LR`` is never called a
communication probability and literature support for a pathway is never used
as proof of an inferred cell-type direction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


EDGE_COLUMNS = [
    "stage",
    "stage_label",
    "grouping_seed",
    "source_index",
    "target_index",
    "sender_type",
    "receiver_type",
    "attention_abs_mean",
    "edge_message_norm_joint",
    "spatial_distance",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--h5ad", required=True, type=Path)
    result.add_argument("--attribution-dir", required=True, type=Path)
    result.add_argument(
        "--observed-cells",
        type=Path,
        help=(
            "Attribution observed_cells.csv.gz. Defaults to the file directly "
            "under --attribution-dir and is required for endpoint mapping."
        ),
    )
    result.add_argument("--commot-lr-scores", required=True, type=Path)
    result.add_argument(
        "--commot-axis-availability",
        type=Path,
        help=(
            "COMMOT stage x LR matrix-key availability table. Defaults beside "
            "--commot-lr-scores and is required to distinguish sparse structural "
            "zeros from unavailable axes."
        ),
    )
    result.add_argument("--known-axis-scores", required=True, type=Path)
    result.add_argument("--nichenet-detail", type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--min-active-edges", type=int, default=5)
    result.add_argument("--min-cells-per-side", type=int, default=10)
    result.add_argument("--top-circuits-per-axis-stage", type=int, default=5)
    result.add_argument("--overwrite", action="store_true")
    return result


def require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_values(series: pd.Series, label: str) -> np.ndarray:
    values = pd.to_numeric(series, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} must contain finite integer values")
    return values.astype(int)


def resolve_observed_cells_path(
    attribution_dir: Path, supplied: Path | None
) -> Path:
    path = supplied if supplied is not None else attribution_dir / "observed_cells.csv.gz"
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "A valid observed_cells.csv.gz is required to map attribution-global "
            f"edge indices into the supplied H5AD: {path}"
        )
    return path


def observed_global_to_h5ad(
    path: Path, data: ad.AnnData
) -> tuple[dict[int, int], dict[str, object]]:
    """Map attribution-global IDs by obs_name and verify the complete cell contract."""
    observed = pd.read_csv(path)
    require(
        observed,
        ["global_index", "obs_name", "stage", "stage_label", "cell_type"],
        "observed-cells table",
    )
    global_index = integer_values(observed["global_index"], "global_index")
    if len(np.unique(global_index)) != len(global_index):
        raise ValueError("observed-cells global_index is not unique")
    if set(global_index) != set(range(len(observed))):
        raise ValueError("observed-cells global_index must be a complete 0..N-1 grid")
    obs_names = observed["obs_name"].astype(str)
    if obs_names.duplicated().any():
        raise ValueError("observed-cells obs_name is not unique")
    if not data.obs_names.is_unique:
        raise ValueError("H5AD obs_names must be unique")
    h5_names = data.obs_names.astype(str)
    if set(obs_names) != set(h5_names) or len(observed) != data.n_obs:
        raise ValueError(
            "observed-cells and H5AD must contain exactly the same obs_name universe"
        )
    h5_lookup = {name: index for index, name in enumerate(h5_names)}
    h5_index = obs_names.map(h5_lookup).to_numpy(int)
    observed_stage = pd.to_numeric(observed["stage"], errors="raise").to_numpy(float)
    h5_stage = pd.to_numeric(
        data.obs.iloc[h5_index]["time_point_processed"], errors="raise"
    ).to_numpy(float)
    if not np.isclose(observed_stage, h5_stage, rtol=0.0, atol=1e-12).all():
        raise ValueError("observed-cells stage disagrees with H5AD")
    observed_type = observed["cell_type"].astype(str).to_numpy()
    h5_type = data.obs.iloc[h5_index]["Annotation"].astype(str).to_numpy()
    if not np.array_equal(observed_type, h5_type):
        raise ValueError("observed-cells cell_type disagrees with H5AD Annotation")
    if "time" in data.obs:
        h5_label = data.obs.iloc[h5_index]["time"].astype(str).to_numpy()
        if not np.array_equal(observed["stage_label"].astype(str).to_numpy(), h5_label):
            raise ValueError("observed-cells stage_label disagrees with H5AD time")
    return dict(zip(global_index, h5_index)), {
        "mode": "observed_cells_obs_name_to_h5ad",
        "observed_cells": str(path),
        "observed_cells_sha256": sha256(path),
        "n_cells": int(len(observed)),
        "global_index_order_assumed_without_validation": False,
    }


def complex_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(value).split("_") if token.strip())


def casefold_gene_index(var_names: Iterable[str]) -> dict[str, int]:
    grouped: dict[str, list[int]] = {}
    for index, gene in enumerate(var_names):
        grouped.setdefault(str(gene).casefold(), []).append(index)
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}


def dense_column_min(matrix, indices: Sequence[int]) -> np.ndarray:
    values = matrix[:, list(indices)]
    values = values.toarray() if sparse.issparse(values) else np.asarray(values)
    return np.min(np.asarray(values, dtype=np.float32), axis=1)


def scaled_gene_activities(
    data: ad.AnnData, tokens: Iterable[str]
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Return q95-scaled complex activity and an auditable availability table."""
    lookup = casefold_gene_index(data.var_names.astype(str))
    result: dict[str, np.ndarray] = {}
    audit: list[dict[str, object]] = []
    for token in sorted(set(str(value) for value in tokens)):
        genes = complex_tokens(token)
        missing = [gene for gene in genes if gene.casefold() not in lookup]
        if missing:
            audit.append(
                {
                    "token": token,
                    "available": False,
                    "missing_subunits": ";".join(missing),
                    "positive_cells": 0,
                    "positive_q95_scale": np.nan,
                }
            )
            continue
        raw = dense_column_min(data.X, [lookup[gene.casefold()] for gene in genes])
        positive = raw[raw > 0]
        scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        result[token] = np.clip(raw / scale, 0.0, 1.0).astype(np.float32)
        audit.append(
            {
                "token": token,
                "available": True,
                "missing_subunits": "",
                "positive_cells": int(positive.size),
                "positive_q95_scale": scale,
            }
        )
    return result, pd.DataFrame(audit)


def load_known_axes(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require(
        frame,
        ["ligand", "receptor", "evidence_scope", "claim_guardrail"],
        "known-axis table",
    )
    optional = [
        column
        for column in ["source_ids", "source_urls", "pathways", "categories"]
        if column in frame
    ]
    columns = ["ligand", "receptor", "evidence_scope", "claim_guardrail", *optional]
    axes = frame[columns].drop_duplicates(["ligand", "receptor"]).copy()
    axes["axis_id"] = axes["ligand"].astype(str) + "->" + axes["receptor"].astype(str)
    return axes.reset_index(drop=True)


def load_edges(
    directory: Path, data: ad.AnnData, observed_cells: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    paths = sorted(directory.glob("stage_*/edges_seed_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No stage_*/edges_seed_*.csv.gz under {directory}")
    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []
    mapping, resolution = observed_global_to_h5ad(observed_cells, data)
    h5_stage = pd.to_numeric(
        data.obs["time_point_processed"], errors="raise"
    ).to_numpy(float)
    h5_type = data.obs["Annotation"].astype(str).to_numpy()
    for path in paths:
        frame = pd.read_csv(path, usecols=EDGE_COLUMNS)
        require(frame, EDGE_COLUMNS, str(path))
        if frame.empty:
            raise ValueError(f"Empty edge table: {path}")
        stage = pd.to_numeric(frame["stage"], errors="raise").to_numpy(float)
        if not np.isclose(stage, stage[0], rtol=0.0, atol=1e-12).all():
            raise ValueError(f"Mixed stages in edge table: {path}")
        if frame["stage_label"].astype(str).nunique() != 1:
            raise ValueError(f"Mixed stage labels in edge table: {path}")
        seeds = integer_values(frame["grouping_seed"], "grouping_seed")
        if np.unique(seeds).size != 1:
            raise ValueError(f"Mixed grouping seeds in edge table: {path}")
        source_attribution = integer_values(frame["source_index"], "source_index")
        target_attribution = integer_values(frame["target_index"], "target_index")
        missing = sorted(
            (set(source_attribution) | set(target_attribution)).difference(mapping)
        )
        if missing:
            raise ValueError(
                f"Edge table contains attribution indices absent from observed-cells: {missing[:5]}"
            )
        source = np.asarray([mapping[value] for value in source_attribution], dtype=int)
        target = np.asarray([mapping[value] for value in target_attribution], dtype=int)
        if not np.isclose(h5_stage[source], stage, rtol=0.0, atol=1e-12).all() or not np.isclose(
            h5_stage[target], stage, rtol=0.0, atol=1e-12
        ).all():
            raise ValueError(f"Edge endpoints disagree with the recorded stage: {path}")
        if not np.array_equal(frame["sender_type"].astype(str).to_numpy(), h5_type[source]):
            raise ValueError(f"sender_type disagrees with mapped H5AD cells: {path}")
        if not np.array_equal(frame["receiver_type"].astype(str).to_numpy(), h5_type[target]):
            raise ValueError(f"receiver_type disagrees with mapped H5AD cells: {path}")
        if np.any(source == target):
            raise ValueError(f"Self edge found after observed-cell mapping: {path}")
        frame = frame.copy()
        frame["source_index_attribution"] = source_attribution
        frame["target_index_attribution"] = target_attribution
        frame["source_index"] = source
        frame["target_index"] = target
        frames.append(frame)
        inventory.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "stage": float(stage[0]),
                "stage_label": str(frame["stage_label"].iloc[0]),
                "grouping_seed": int(seeds[0]),
                "n_rows": int(len(frame)),
                "index_mapping": resolution["mode"],
            }
        )
    edges = pd.concat(frames, ignore_index=True)
    edges["stage"] = pd.to_numeric(edges["stage"], errors="raise").astype(float)
    for column in ["source_index", "target_index", "grouping_seed"]:
        edges[column] = pd.to_numeric(edges[column], errors="raise").astype(int)
    return edges, pd.DataFrame(inventory), resolution


def stage_type_counts(data: ad.AnnData) -> pd.DataFrame:
    require(data.obs, ["time_point_processed", "Annotation"], "H5AD obs")
    frame = data.obs[["time_point_processed", "Annotation"]].copy()
    frame.columns = ["stage", "cell_type"]
    frame["stage"] = pd.to_numeric(frame["stage"], errors="raise").astype(float)
    return (
        frame.groupby(["stage", "cell_type"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )


def expression_context_summary(
    data: ad.AnnData,
    axes: pd.DataFrame,
    activities: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    obs = data.obs[["time_point_processed", "Annotation"]].copy()
    stages = pd.to_numeric(obs["time_point_processed"], errors="raise").to_numpy(float)
    labels = obs["Annotation"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    contexts = pd.DataFrame({"stage": stages, "cell_type": labels})
    for axis in axes.itertuples(index=False):
        if axis.ligand not in activities or axis.receptor not in activities:
            continue
        for role, token in [("ligand", axis.ligand), ("receptor", axis.receptor)]:
            values = activities[token]
            local = contexts.copy()
            local["value"] = values
            local["positive"] = values > 0
            summary = (
                local.groupby(["stage", "cell_type"], observed=True)
                .agg(
                    mean_scaled_expression=("value", "mean"),
                    positive_fraction=("positive", "mean"),
                    n_cells=("value", "size"),
                )
                .reset_index()
            )
            summary["axis_id"] = axis.axis_id
            summary["role"] = role
            rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def descending_competition_rank(series: pd.Series) -> pd.DataFrame:
    """Return tie-auditable descending min ranks on a finite evaluated set."""
    values = pd.to_numeric(series, errors="raise").astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Evaluated ranking values must be finite")
    n = int(len(values))
    if n == 0:
        raise ValueError("Cannot rank an empty evaluated context universe")
    rank = values.rank(method="min", ascending=False).astype(int)
    tie_count = values.groupby(values, dropna=False).transform("size").astype(int)
    return pd.DataFrame(
        {
            "rank_from_top": rank,
            "rank_fraction": rank / n,
            "tie_count": tie_count,
            "context_percentile": 1.0 - (rank - 1) / n,
            "top_percent": 100.0 * (rank - 1) / n,
        },
        index=series.index,
    )


def score_exact_circuits(
    edges: pd.DataFrame,
    axes: pd.DataFrame,
    activities: Mapping[str, np.ndarray],
    counts: pd.DataFrame,
    expression: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate LR-compatible CytoBridge scores on exact directed circuits."""
    count_lookup = counts.set_index(["stage", "cell_type"])["n_cells"]
    n_seeds = edges.groupby("stage")["grouping_seed"].nunique().to_dict()
    rows: list[pd.DataFrame] = []
    for axis in axes.itertuples(index=False):
        if axis.ligand not in activities or axis.receptor not in activities:
            continue
        source = edges["source_index"].to_numpy(int)
        target = edges["target_index"].to_numpy(int)
        lr = activities[axis.ligand][source] * activities[axis.receptor][target]
        local = edges[
            [
                "stage",
                "stage_label",
                "grouping_seed",
                "source_index",
                "target_index",
                "sender_type",
                "receiver_type",
                "attention_abs_mean",
                "edge_message_norm_joint",
                "spatial_distance",
            ]
        ].copy()
        local["lr_activity"] = lr
        local["attention_lr"] = lr * local["attention_abs_mean"].to_numpy(float)
        local["exact_message_lr"] = (
            lr * local["edge_message_norm_joint"].to_numpy(float)
        )
        local["active"] = lr > 0
        grouped = (
            local.groupby(
                ["stage", "stage_label", "sender_type", "receiver_type"],
                observed=True,
                as_index=False,
            )
            .agg(
                attention_lr_sum_occurrences=("attention_lr", "sum"),
                exact_message_lr_sum_occurrences=("exact_message_lr", "sum"),
                lr_activity_sum_occurrences=("lr_activity", "sum"),
                mean_attention_on_occurrences=("attention_abs_mean", "mean"),
                mean_exact_message_on_occurrences=("edge_message_norm_joint", "mean"),
                mean_spatial_distance=("spatial_distance", "mean"),
                n_edge_occurrences=("source_index", "size"),
                n_active_occurrences=("active", "sum"),
            )
        )
        unique_counts = (
            local.loc[local["active"]]
            .drop_duplicates(
                ["stage", "sender_type", "receiver_type", "source_index", "target_index"]
            )
            .groupby(["stage", "sender_type", "receiver_type"], observed=True)
            .size()
            .rename("n_active_unique_edges")
            .reset_index()
        )
        grouped = grouped.merge(
            unique_counts,
            on=["stage", "sender_type", "receiver_type"],
            how="left",
        )
        grouped["n_active_unique_edges"] = grouped["n_active_unique_edges"].fillna(0).astype(int)
        grouped["n_grouping_seeds"] = grouped["stage"].map(n_seeds).astype(int)
        grouped["n_sender_cells"] = [
            int(count_lookup.get((stage, sender), 0))
            for stage, sender in zip(grouped["stage"], grouped["sender_type"])
        ]
        grouped["n_receiver_cells"] = [
            int(count_lookup.get((stage, receiver), 0))
            for stage, receiver in zip(grouped["stage"], grouped["receiver_type"])
        ]
        grouped["n_shared_sender_receiver_cells"] = np.where(
            grouped["sender_type"].eq(grouped["receiver_type"]),
            grouped["n_sender_cells"],
            0,
        )
        grouped["n_possible_distinct_cell_pairs"] = (
            grouped["n_sender_cells"] * grouped["n_receiver_cells"]
            - grouped["n_shared_sender_receiver_cells"]
        )
        if grouped["n_possible_distinct_cell_pairs"].le(0).any():
            raise ValueError("Exact circuit has no possible distinct-cell pairs")
        seeds = grouped["n_grouping_seeds"].clip(lower=1)
        possible = grouped["n_possible_distinct_cell_pairs"]
        grouped["cytobridge_attention_lr_density"] = (
            grouped["attention_lr_sum_occurrences"] / seeds / possible
        )
        grouped["cytobridge_exact_message_lr_density"] = (
            grouped["exact_message_lr_sum_occurrences"] / seeds / possible
        )
        grouped["cytobridge_lr_only_density"] = (
            grouped["lr_activity_sum_occurrences"] / seeds / possible
        )
        grouped["cytobridge_attention_lr_per_active_edge"] = (
            grouped["attention_lr_sum_occurrences"]
            / grouped["n_active_occurrences"].clip(lower=1)
        )
        grouped["cytobridge_exact_message_lr_per_active_edge"] = (
            grouped["exact_message_lr_sum_occurrences"]
            / grouped["n_active_occurrences"].clip(lower=1)
        )
        grouped["axis_id"] = axis.axis_id
        grouped["ligand"] = axis.ligand
        grouped["receptor"] = axis.receptor
        grouped["evidence_scope"] = axis.evidence_scope
        grouped["claim_guardrail"] = axis.claim_guardrail
        for column in ["source_ids", "source_urls", "pathways", "categories"]:
            if hasattr(axis, column):
                grouped[column] = getattr(axis, column)
        rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)

    ligand_expression = expression.loc[expression["role"].eq("ligand")].rename(
        columns={
            "cell_type": "sender_type",
            "mean_scaled_expression": "sender_ligand_mean_scaled_expression",
            "positive_fraction": "sender_ligand_positive_fraction",
            "n_cells": "sender_expression_n_cells",
        }
    )
    receptor_expression = expression.loc[expression["role"].eq("receptor")].rename(
        columns={
            "cell_type": "receiver_type",
            "mean_scaled_expression": "receiver_receptor_mean_scaled_expression",
            "positive_fraction": "receiver_receptor_positive_fraction",
            "n_cells": "receiver_expression_n_cells",
        }
    )
    result = result.merge(
        ligand_expression[
            [
                "stage",
                "axis_id",
                "sender_type",
                "sender_ligand_mean_scaled_expression",
                "sender_ligand_positive_fraction",
                "sender_expression_n_cells",
            ]
        ],
        on=["stage", "axis_id", "sender_type"],
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        receptor_expression[
            [
                "stage",
                "axis_id",
                "receiver_type",
                "receiver_receptor_mean_scaled_expression",
                "receiver_receptor_positive_fraction",
                "receiver_expression_n_cells",
            ]
        ],
        on=["stage", "axis_id", "receiver_type"],
        how="left",
        validate="many_to_one",
    )
    return result


def collapse_commot(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require(
        frame,
        [
            "stage",
            "sender_type",
            "receiver_type",
            "ligand",
            "receptor",
            "score",
            "abundance_controlled_score",
            "score_distinct_cell_pairs",
            "abundance_controlled_distinct_cell_score",
            "n_possible_distinct_cell_pairs",
            "database_rows",
        ],
        "COMMOT LR scores",
    )
    frame["stage"] = pd.to_numeric(frame["stage"], errors="raise").astype(float)
    frame["axis_id"] = frame["ligand"].astype(str) + "->" + frame["receptor"].astype(str)
    key = ["stage", "axis_id", "ligand", "receptor", "sender_type", "receiver_type"]
    # A collapsed ligand-receptor axis can retain multiple database provenance
    # rows.  COMMOT's cell-flow matrix is identical across those rows, so max
    # avoids double-counting the same inferred flow.
    grouped = (
        frame.groupby(key, observed=True, as_index=False)
        .agg(
            commot_total_flow=("score_distinct_cell_pairs", "max"),
            commot_total_flow_min=("score_distinct_cell_pairs", "min"),
            commot_abundance_controlled_score=(
                "abundance_controlled_distinct_cell_score", "max"
            ),
            commot_abundance_controlled_score_min=(
                "abundance_controlled_distinct_cell_score", "min"
            ),
            commot_n_possible_distinct_cell_pairs=(
                "n_possible_distinct_cell_pairs", "max"
            ),
            commot_n_possible_distinct_cell_pairs_min=(
                "n_possible_distinct_cell_pairs", "min"
            ),
            commot_database_rows=("database_rows", lambda x: ";".join(sorted(set(map(str, x))))),
        )
    )
    if not np.isclose(
        grouped["commot_total_flow"],
        grouped["commot_total_flow_min"],
        rtol=1e-10,
        atol=1e-12,
    ).all() or not np.isclose(
        grouped["commot_abundance_controlled_score"],
        grouped["commot_abundance_controlled_score_min"],
        rtol=1e-10,
        atol=1e-12,
    ).all():
        raise ValueError(
            "COMMOT duplicate provenance rows disagree despite sharing one matrix key"
        )
    return grouped


def load_commot_availability(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    require(
        frame,
        ["stage", "ligand", "receptor", "method_available", "matrix_key"],
        "COMMOT LR axis-stage availability",
    )
    frame = frame.copy()
    frame["stage"] = pd.to_numeric(frame["stage"], errors="raise").astype(float)
    frame["axis_id"] = (
        frame["ligand"].astype(str) + "->" + frame["receptor"].astype(str)
    )
    available_text = frame["method_available"].astype(str).str.casefold()
    values = available_text.map({"true": True, "false": False})
    if values.isna().any():
        raise ValueError("COMMOT method_available must contain only true/false")
    frame["commot_axis_stage_available"] = values.astype(bool)
    key = ["stage", "axis_id", "ligand", "receptor"]
    consistency = frame.groupby(key, observed=True)[
        "commot_axis_stage_available"
    ].nunique()
    if (consistency > 1).any():
        raise ValueError("COMMOT availability disagrees across provenance rows")
    return (
        frame.groupby(key, observed=True, as_index=False)
        .agg(
            commot_axis_stage_available=(
                "commot_axis_stage_available", "first"
            ),
            commot_matrix_keys=(
                "matrix_key",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
        )
    )


def join_and_rank(
    cytobridge: pd.DataFrame,
    commot: pd.DataFrame,
    commot_availability: pd.DataFrame,
    *,
    min_active_edges: int,
    min_cells_per_side: int,
) -> pd.DataFrame:
    """Join methods and rank only a shared, support-qualified context universe.

    COMMOT extraction tables are sparse.  A missing context row is completed
    as a structural zero only when the explicit stage x axis availability
    table confirms that the corresponding COMMOT matrix existed; otherwise
    the entire stage x axis remains unavailable.  All displayed method ranks
    use the same support-qualified context rows.
    """
    key = ["stage", "axis_id", "ligand", "receptor", "sender_type", "receiver_type"]
    result = cytobridge.merge(commot, on=key, how="left", validate="one_to_one")
    availability_key = ["stage", "axis_id", "ligand", "receptor"]
    result = result.merge(
        commot_availability,
        on=availability_key,
        how="left",
        validate="many_to_one",
    )
    result["commot_axis_stage_available"] = result[
        "commot_axis_stage_available"
    ].fillna(False).astype(bool)
    commot_payload_present = result["commot_database_rows"].notna()
    if (commot_payload_present & ~result["commot_axis_stage_available"]).any():
        raise ValueError(
            "COMMOT score rows exist for a stage x axis that the explicit "
            "availability table does not mark available"
        )
    inconsistent_commot_denominator = (
        result["commot_n_possible_distinct_cell_pairs"].notna()
        & result["commot_n_possible_distinct_cell_pairs_min"].ne(
            result["commot_n_possible_distinct_cell_pairs"]
        )
    )
    if inconsistent_commot_denominator.any():
        raise ValueError("COMMOT duplicate rows disagree on distinct-cell denominator")
    denominator_mismatch = (
        result["commot_n_possible_distinct_cell_pairs"].notna()
        & result["commot_n_possible_distinct_cell_pairs"].ne(
            result["n_possible_distinct_cell_pairs"]
        )
    )
    if denominator_mismatch.any():
        raise ValueError(
            "CytoBridge and COMMOT disagree on the exact distinct-cell denominator"
        )
    structural_zero = (
        result["commot_axis_stage_available"]
        & result["commot_abundance_controlled_score"].isna()
    )
    result["commot_context_was_sparse_structural_zero"] = structural_zero
    result.loc[structural_zero, "commot_total_flow"] = 0.0
    result.loc[structural_zero, "commot_abundance_controlled_score"] = 0.0
    result.loc[
        structural_zero, "commot_n_possible_distinct_cell_pairs"
    ] = result.loc[structural_zero, "n_possible_distinct_cell_pairs"].to_numpy()
    for column in ["commot_total_flow", "commot_abundance_controlled_score"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["commot_context_available"] = result["commot_axis_stage_available"]
    result["passes_context_support_filter"] = (
        result["n_active_unique_edges"].ge(int(min_active_edges))
        & result["n_sender_cells"].ge(int(min_cells_per_side))
        & result["n_receiver_cells"].ge(int(min_cells_per_side))
    )
    result["is_evaluated_context"] = (
        result["passes_context_support_filter"]
        & result["commot_context_available"]
    )
    result["n_evaluated_contexts"] = (
        result["is_evaluated_context"]
        .astype(int)
        .groupby([result["stage"], result["axis_id"]], observed=True)
        .transform("sum")
        .astype(int)
    )
    rank_columns = {
        "cytobridge_attention_lr_density": "attention",
        "cytobridge_exact_message_lr_density": "exact_message",
        "cytobridge_lr_only_density": "lr_only",
        "commot_abundance_controlled_score": "commot",
    }
    evaluated = result["is_evaluated_context"]
    evaluated_groups = result.loc[evaluated].groupby(
        ["stage", "axis_id"], observed=True
    ).groups
    for score, prefix in rank_columns.items():
        columns = {
            "rank_from_top": f"{prefix}_context_rank_from_top",
            "rank_fraction": f"{prefix}_context_rank_fraction",
            "tie_count": f"{prefix}_context_tie_count",
            "context_percentile": f"{prefix}_context_percentile",
            "top_percent": f"{prefix}_context_top_percent",
        }
        for column in columns.values():
            result[column] = np.nan
        for indices in evaluated_groups.values():
            ranked = descending_competition_rank(result.loc[indices, score])
            for source_column, output_column in columns.items():
                result.loc[indices, output_column] = ranked[source_column]
    result["attention_commot_joint_percentile"] = result[
        ["attention_context_percentile", "commot_context_percentile"]
    ].mean(axis=1)
    result["attention_commot_min_percentile"] = result[
        ["attention_context_percentile", "commot_context_percentile"]
    ].min(axis=1)
    result["exact_commot_joint_percentile"] = result[
        ["exact_message_context_percentile", "commot_context_percentile"]
    ].mean(axis=1)
    result["exact_commot_min_percentile"] = result[
        ["exact_message_context_percentile", "commot_context_percentile"]
    ].min(axis=1)
    result["attention_increment_over_lr_only"] = (
        result["attention_context_percentile"] - result["lr_only_context_percentile"]
    )
    return result


def attach_nichenet(frame: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    result = frame.copy()
    result["nichenet_receiver_transition_support"] = False
    result["nichenet_aupr_corrected"] = np.nan
    result["nichenet_reported_targets"] = ""
    if path is None:
        return result
    detail = pd.read_csv(path)
    require(
        detail,
        [
            "source_stage_id",
            "receiver",
            "ligand_key",
            "aupr_corrected",
            "reported_nichenet_targets",
        ],
        "NicheNet detail",
    )
    detail = detail.copy()
    detail["stage"] = pd.to_numeric(detail["source_stage_id"], errors="raise").astype(float)
    detail["ligand_key"] = detail["ligand_key"].astype(str).str.casefold()
    detail["receiver_type"] = detail["receiver"].astype(str)
    reduced = (
        detail.sort_values("aupr_corrected", ascending=False)
        .drop_duplicates(["stage", "receiver_type", "ligand_key"])
        [[
            "stage",
            "receiver_type",
            "ligand_key",
            "target_stage_id",
            "target_stage_label",
            "aupr_corrected",
            "reported_nichenet_targets",
        ]]
    )
    result["ligand_key"] = result["ligand"].astype(str).str.casefold()
    result = result.drop(
        columns=[
            "nichenet_receiver_transition_support",
            "nichenet_aupr_corrected",
            "nichenet_reported_targets",
        ]
    ).merge(
        reduced,
        on=["stage", "receiver_type", "ligand_key"],
        how="left",
        validate="many_to_one",
    )
    result["nichenet_receiver_transition_support"] = result["aupr_corrected"].notna()
    result = result.rename(
        columns={
            "aupr_corrected": "nichenet_aupr_corrected",
            "reported_nichenet_targets": "nichenet_reported_targets",
        }
    )
    result["nichenet_reported_targets"] = result[
        "nichenet_reported_targets"
    ].fillna("")
    return result


def select_candidates(
    frame: pd.DataFrame,
    *,
    min_active_edges: int,
    min_cells_per_side: int,
    top_per_axis_stage: int,
) -> pd.DataFrame:
    eligible = frame.loc[
        frame["is_evaluated_context"]
        & frame["cytobridge_attention_lr_density"].gt(0)
        & frame["commot_abundance_controlled_score"].gt(0)
        & frame["sender_ligand_positive_fraction"].gt(0)
        & frame["receiver_receptor_positive_fraction"].gt(0)
    ].copy()
    eligible["passes_attention_increment_control"] = eligible[
        "attention_increment_over_lr_only"
    ].gt(0)
    eligible = eligible.sort_values(
        [
            "stage",
            "axis_id",
            "exact_commot_min_percentile",
            "attention_commot_min_percentile",
            "n_active_unique_edges",
            "sender_type",
            "receiver_type",
        ],
        ascending=[True, True, False, False, False, True, True],
    )
    eligible["candidate_rank_within_axis_stage"] = (
        eligible.groupby(["stage", "axis_id"], observed=True).cumcount() + 1
    )
    return eligible.loc[
        eligible["candidate_rank_within_axis_stage"].le(int(top_per_axis_stage))
    ].reset_index(drop=True)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def main() -> None:
    args = parser().parse_args()
    prepare_output(args.output_dir, args.overwrite)
    data = ad.read_h5ad(args.h5ad)
    require(data.obs, ["time_point_processed", "Annotation"], "H5AD obs")
    observed_cells = resolve_observed_cells_path(
        args.attribution_dir, args.observed_cells
    )
    axes = load_known_axes(args.known_axis_scores)
    tokens = set(axes["ligand"]) | set(axes["receptor"])
    activities, gene_audit = scaled_gene_activities(data, tokens)
    axes = axes.loc[
        axes["ligand"].isin(activities) & axes["receptor"].isin(activities)
    ].reset_index(drop=True)
    edges, edge_inventory, index_resolution = load_edges(
        args.attribution_dir, data, observed_cells
    )
    counts = stage_type_counts(data)
    expression = expression_context_summary(data, axes, activities)
    cytobridge = score_exact_circuits(
        edges, axes, activities, counts, expression
    )
    commot = collapse_commot(args.commot_lr_scores)
    commot_availability_path = (
        args.commot_axis_availability
        if args.commot_axis_availability is not None
        else args.commot_lr_scores.parent / "commot_lr_axis_stage_availability.csv.gz"
    ).expanduser().resolve()
    if not commot_availability_path.is_file():
        raise FileNotFoundError(
            "COMMOT stage x axis availability is required to distinguish "
            f"structural zeros from unavailable axes: {commot_availability_path}"
        )
    commot_availability = load_commot_availability(commot_availability_path)
    screen = join_and_rank(
        cytobridge,
        commot,
        commot_availability,
        min_active_edges=args.min_active_edges,
        min_cells_per_side=args.min_cells_per_side,
    )
    screen = attach_nichenet(screen, args.nichenet_detail)
    candidates = select_candidates(
        screen,
        min_active_edges=args.min_active_edges,
        min_cells_per_side=args.min_cells_per_side,
        top_per_axis_stage=args.top_circuits_per_axis_stage,
    )

    tables = args.output_dir / "tables"
    tables.mkdir()
    gene_audit.to_csv(tables / "gene_activity_availability.csv", index=False)
    edge_inventory.to_csv(tables / "edge_input_inventory.csv", index=False)
    counts.to_csv(tables / "stage_cell_type_counts.csv", index=False)
    expression.to_csv(tables / "known_axis_expression_by_context.csv.gz", index=False)
    screen.to_csv(tables / "known_axis_exact_circuit_screen.csv.gz", index=False)
    candidates.to_csv(tables / "top_exact_circuit_candidates.csv", index=False)

    manifest = {
        "schema_version": 2,
        "workflow": "zebrafish_biology_first_case_studies",
        "status": "screen_complete",
        "selection_unit": (
            "observed stage + exact ligand-receptor axis + exact directed "
            "sender-cell-type -> receiver-cell-type circuit"
        ),
        "ranking_universe": (
            "within each stage x axis: contexts meeting min cell/active-edge "
            "support for an explicitly available COMMOT stage x axis; sparse "
            "missing context rows on an available matrix are structural zero, "
            "whereas unavailable stage x axes remain NA"
        ),
        "ranking_definition": (
            "descending competition min-rank; rank_fraction=r_min/N, tie_count "
            "reported, top_percent=100*(r_min-1)/N"
        ),
        "score_semantics": {
            "cytobridge_attention_lr_density": (
                "post-hoc |attention| x q95-scaled sender-ligand x receiver-"
                "receptor activity, summed across observed model edges, averaged "
                "across grouping seeds, divided by possible distinct-cell pairs "
                "n_sender*n_receiver-|sender_cells intersect receiver_cells|"
            ),
            "cytobridge_exact_message_lr_density": (
                "post-hoc exact-message norm x LR activity on the same denominator"
            ),
            "commot_abundance_controlled_score": (
                "COMMOT cell-flow mass with cell-diagonal entries removed, divided "
                "by n_sender*n_receiver minus sender/receiver cell-set overlap"
            ),
        },
        "guardrails": {
            "attention_is_lr_specific": False,
            "attention_times_lr_is_model_native_probability": False,
            "literature_support_validates_inferred_direction": False,
            "nichenet_is_independent_experimental_validation": False,
            "candidate_screen_is_final_biological_claim": False,
            "global_h5ad_row_order_was_silently_assumed": False,
        },
        "index_resolution": index_resolution,
        "parameters": {
            "min_active_edges": int(args.min_active_edges),
            "min_cells_per_side": int(args.min_cells_per_side),
            "top_circuits_per_axis_stage": int(args.top_circuits_per_axis_stage),
        },
        "inputs": {
            "h5ad": str(args.h5ad.resolve()),
            "attribution_dir": str(args.attribution_dir.resolve()),
            "observed_cells": str(observed_cells),
            "commot_lr_scores": str(args.commot_lr_scores.resolve()),
            "commot_axis_availability": str(commot_availability_path),
            "known_axis_scores": str(args.known_axis_scores.resolve()),
            "nichenet_detail": (
                None if args.nichenet_detail is None else str(args.nichenet_detail.resolve())
            ),
        },
        "counts": {
            "known_axes_available": int(len(axes)),
            "screen_rows": int(len(screen)),
            "candidate_rows": int(len(candidates)),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
