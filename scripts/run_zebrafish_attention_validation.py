#!/usr/bin/env python3
"""Run reviewer-facing zebrafish interaction validation without circular selection.

The workflow consumes manifest-bound outputs from the accepted CytoBridge
model, COMMOT, CellAgentChat, NicheNet, and the fixed-checkpoint interaction
on/off sensitivity.  ``analyze`` writes numerical tables only.  ``report``
renders those frozen tables, and ``validate`` re-hashes every input and output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse, stats

from CytoBridge.zebrafish_attention_validation import (
    CYTOBRIDGE_VIEWS,
    PRIMARY_PERMUTATIONS,
    PRIMARY_RANDOM_SEED,
    PRIMARY_TOP_FRACTION,
    adaptive_pair_strata,
    build_pair_lr_activity_matrix,
    collapse_commot_lr_scores,
    collapse_lr_database,
    collapse_nichenet_lr_scores,
    complete_directed_pair_table,
    controlled_pair_concordance,
    external_ranks_for_selected_pairs,
    jointly_supported_lr_targets,
    lr_scores_from_pair_modifiers,
    modifier_permutation_test,
    pair_method_concordance,
    paper_reference_enrichment,
    positive_rank_weights,
    rank_metrics,
    scaled_expression_by_type,
    select_pairs_by_cytobridge_only,
    shared_lr_rank_metrics,
)


SCHEMA_VERSION = 2
WORKFLOW = "zebrafish_attention_lr_independent_validation"
REPO_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    "CytoBridge/zebrafish_attention_validation.py",
    "scripts/run_zebrafish_attention_validation.py",
    "scripts/reviewer_zebrafish_ccc/original_paper_21_lr.csv",
)
REQUIRED_INPUTS = (
    "matched_acceptance",
    "sample_h5ad",
    "sample_manifest",
    "cytobridge_type_pair",
    "cytobridge_manifest",
    "commot_type_pair",
    "commot_lr",
    "commot_manifest",
    "cellagentchat_type_pair",
    "cellagentchat_manifest",
    "nichenet_lr",
    "nichenet_targets",
    "nichenet_manifest",
    "lr_database",
    "interaction_target_metrics",
    "interaction_manifest",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_sha_sidecar(path: Path) -> Path:
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{_sha256(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def _implementation_identity() -> dict[str, object]:
    files = {
        relative: _artifact(REPO_ROOT / relative) for relative in IMPLEMENTATION_FILES
    }
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(f"{relative}:{files[relative]['sha256']}\n".encode())
    return {"files": files, "aggregate_sha256": digest.hexdigest()}


def _load_spec(path: Path) -> tuple[Mapping[str, Any], dict[str, dict[str, object]]]:
    payload = _load_json(path, label="analysis spec")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"analysis spec requires schema_version={SCHEMA_VERSION}")
    if payload.get("workflow") != WORKFLOW or payload.get("dataset") != "zebrafish":
        raise ValueError(
            "analysis spec workflow/dataset is not the frozen zebrafish contract"
        )
    records = payload.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("analysis spec lacks artifacts")
    missing = sorted(set(REQUIRED_INPUTS).difference(records))
    if missing:
        raise ValueError(f"analysis spec lacks required artifacts: {missing}")
    verified: dict[str, dict[str, object]] = {}
    for label in REQUIRED_INPUTS:
        record = records[label]
        if not isinstance(record, dict):
            raise ValueError(f"artifact {label!r} is not an object")
        path_value = record.get("path")
        expected = str(record.get("sha256", "")).casefold()
        if not isinstance(path_value, str) or len(expected) != 64:
            raise ValueError(f"artifact {label!r} lacks exact path/SHA-256")
        observed = _artifact(Path(path_value))
        if observed["sha256"] != expected:
            raise ValueError(
                f"artifact {label!r} SHA mismatch: expected={expected}, "
                f"actual={observed['sha256']}"
            )
        verified[label] = observed
    acceptance = _load_json(
        Path(str(verified["matched_acceptance"]["path"])),
        label="matched acceptance",
    )
    accepted = acceptance.get(
        "overall_status", acceptance.get("status", acceptance.get("overall"))
    )
    if str(accepted).upper() != "PASS":
        raise ValueError(f"matched acceptance is not PASS: {accepted!r}")
    return payload, verified


def _prepare_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _read_csv(path: str | Path, *, label: str) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    if frame.empty:
        raise ValueError(f"{label} table is empty")
    return frame


def _terminal_stage(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if "stage" not in frame:
        return frame.copy()
    values = pd.to_numeric(frame["stage"], errors="raise")
    terminal = float(values.max())
    local = frame.loc[np.isclose(values, terminal)].copy()
    if local.empty:
        raise ValueError(f"{label} lacks a terminal-stage table")
    return local


def _cell_type_grid(cytobridge: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    types = sorted(
        set(cytobridge["sender_type"].astype(str))
        | set(cytobridge["receiver_type"].astype(str))
    )
    if len(types) < 2:
        raise ValueError("CytoBridge table contains fewer than two cell types")
    score_tables = []
    for view, column in CYTOBRIDGE_VIEWS.items():
        table = complete_directed_pair_table(
            cytobridge, score_column=column, cell_types=types
        ).rename(columns={column: f"cytobridge_{view}"})
        score_tables.append(table)
    pair_grid = score_tables[0].merge(
        score_tables[1], on=["sender_type", "receiver_type"], validate="one_to_one"
    )
    metadata = cytobridge[
        [
            "sender_type",
            "receiver_type",
            "n_sender_cells_mean",
            "n_receiver_cells_mean",
            "spatial_distance_mean_mean",
        ]
    ].copy()
    if metadata.duplicated(["sender_type", "receiver_type"]).any():
        raise ValueError("CytoBridge metadata contains duplicate pairs")
    pair_grid = pair_grid.merge(
        metadata, on=["sender_type", "receiver_type"], validate="one_to_one"
    )
    return types, pair_grid


def _load_selected_expression(
    h5ad_path: Path,
    lr_candidates: pd.DataFrame,
    *,
    cell_type_key: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    import anndata as ad

    data = ad.read_h5ad(h5ad_path, backed="r")
    if cell_type_key not in data.obs:
        raise KeyError(f"sample H5AD lacks obs[{cell_type_key!r}]")
    gene_candidates: dict[str, list[tuple[int, str]]] = {}
    for index, name in enumerate(data.var_names.astype(str)):
        key = name.casefold()
        gene_candidates.setdefault(key, []).append((index, name))
    requested = sorted(
        set(lr_candidates["ligand_key"].astype(str))
        | set(lr_candidates["receptor_key"].astype(str))
    )
    gene_lookup: dict[str, tuple[int, str]] = {}
    ambiguous: list[str] = []
    for key in requested:
        candidates = gene_candidates.get(key, [])
        exact_lowercase = [item for item in candidates if item[1] == key]
        if len(exact_lowercase) == 1:
            gene_lookup[key] = exact_lowercase[0]
        elif len(candidates) == 1:
            gene_lookup[key] = candidates[0]
        elif candidates:
            ambiguous.append(key)
    present = [key for key in requested if key in gene_lookup]
    indices = [gene_lookup[key][0] for key in present]
    subset = data[:, indices].X
    values = subset.toarray() if sparse.issparse(subset) else np.asarray(subset)
    expression = pd.DataFrame(values, columns=[gene_lookup[key][1] for key in present])
    labels = data.obs[cell_type_key].astype(str).tolist()
    means = scaled_expression_by_type(expression, labels)
    return means, {
        "n_cells": int(data.n_obs),
        "n_genes": int(data.n_vars),
        "cell_type_key": cell_type_key,
        "n_cell_types": int(pd.Series(labels).nunique()),
        "expression_matrix": "H5AD X; nonnegative single-log expression",
        "gene_scaling": "per-gene positive-cell 95th percentile, clipped to [0,1]",
        "n_lr_gene_symbols_requested": int(len(requested)),
        "n_lr_gene_symbols_present": int(len(present)),
        "missing_lr_gene_symbols": sorted(set(requested).difference(present)),
        "ambiguous_casefold_gene_symbols_excluded": ambiguous,
        "casefold_resolution": (
            "prefer one exact lowercase zebrafish symbol; otherwise require one "
            "case-insensitive candidate"
        ),
    }


def _verified_cytobridge_edges(
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    manifest = _load_json(manifest_path, label="CytoBridge exact-message manifest")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("CytoBridge manifest has no exact-message edge runs")
    frames: list[pd.DataFrame] = []
    records: dict[str, dict[str, object]] = {}
    required = {
        "source_index_stage",
        "target_index_stage",
        "attention_abs_mean",
        "edge_message_norm_joint",
        "sender_type",
        "receiver_type",
    }
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("CytoBridge edge run is not an object")
        seed = int(run.get("grouping_seed"))
        artifact = run.get("artifacts", {}).get("edges")
        if not isinstance(artifact, dict):
            raise ValueError(f"CytoBridge edge run {seed} lacks an edge artifact")
        path = Path(str(artifact.get("path", "")))
        observed = _artifact(path)
        if observed["sha256"] != str(artifact.get("sha256", "")).casefold():
            raise ValueError(f"CytoBridge edge artifact changed for seed {seed}")
        expected_size = artifact.get("size", artifact.get("bytes"))
        if expected_size is not None and int(expected_size) != observed["size_bytes"]:
            raise ValueError(f"CytoBridge edge artifact size changed for seed {seed}")
        frame = _read_csv(path, label=f"CytoBridge edges seed {seed}")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"CytoBridge edges seed {seed} lack columns: {missing}")
        frame = frame.copy()
        frame["grouping_seed"] = seed
        frames.append(frame)
        records[f"cytobridge_edges_seed_{seed}"] = observed
    return pd.concat(frames, ignore_index=True), records


def _resolve_h5ad_gene(data: Any, key: str) -> tuple[int, str]:
    candidates = [
        (index, str(name))
        for index, name in enumerate(data.var_names.astype(str))
        if str(name).casefold() == key.casefold()
    ]
    exact = [item for item in candidates if item[1] == key.casefold()]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"spatial LR axis gene {key!r} is absent or case-insensitively ambiguous"
    )


def _weighted_radius(points: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    points = np.asarray(points, dtype=float)
    positive = np.isfinite(weights) & (weights > 0)
    if not positive.any():
        return float("nan")
    local_weights = weights[positive]
    local_points = points[positive]
    center = np.average(local_points, axis=0, weights=local_weights)
    squared = np.sum((local_points - center) ** 2, axis=1)
    return float(np.sqrt(np.average(squared, weights=local_weights)))


def _spatial_axis_tables(
    h5ad_path: Path,
    edges: pd.DataFrame,
    *,
    ligand: str,
    receptor: str,
    cell_type_key: str,
    spatial_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import anndata as ad

    data = ad.read_h5ad(h5ad_path, backed="r")
    if cell_type_key not in data.obs:
        raise KeyError(f"sample H5AD lacks obs[{cell_type_key!r}]")
    if spatial_key not in data.obsm:
        raise KeyError(f"sample H5AD lacks obsm[{spatial_key!r}]")
    coordinates = np.asarray(data.obsm[spatial_key], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("sample spatial coordinates are not at least two-dimensional")
    coordinates = coordinates[:, :2]
    if not np.isfinite(coordinates).all():
        raise ValueError("sample spatial coordinates contain non-finite values")
    ligand_index, ligand_name = _resolve_h5ad_gene(data, ligand)
    receptor_index, receptor_name = _resolve_h5ad_gene(data, receptor)
    matrix = data[:, [ligand_index, receptor_index]].X
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("spatial LR expression is invalid")
    scaled = np.zeros_like(values)
    for column in range(2):
        positive = values[:, column][values[:, column] > 0]
        scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
        scaled[:, column] = np.clip(values[:, column] / max(scale, 1e-12), 0, 1)
    cells = pd.DataFrame(
        {
            "cell_index": np.arange(data.n_obs, dtype=int),
            "cell_type": data.obs[cell_type_key].astype(str).to_numpy(),
            "spatial_x": coordinates[:, 0],
            "spatial_y": coordinates[:, 1],
            "ligand": ligand_name,
            "receptor": receptor_name,
            "ligand_scaled_expression": scaled[:, 0],
            "receptor_scaled_expression": scaled[:, 1],
        }
    )
    local = edges.copy()
    for column in ("source_index_stage", "target_index_stage"):
        local[column] = pd.to_numeric(local[column], errors="raise").astype(int)
        if (local[column] < 0).any() or (local[column] >= data.n_obs).any():
            raise ValueError(f"edge {column} lies outside the sample H5AD")
    aggregation = {
        "sender_type": "first",
        "receiver_type": "first",
        "attention_abs_mean": "mean",
        "edge_message_norm_joint": "mean",
        "grouping_seed": "nunique",
    }
    local = (
        local.groupby(["source_index_stage", "target_index_stage"], as_index=False)
        .agg(aggregation)
        .rename(columns={"grouping_seed": "grouping_seed_n"})
    )
    source = local["source_index_stage"].to_numpy(dtype=int)
    target = local["target_index_stage"].to_numpy(dtype=int)
    local["source_x"] = coordinates[source, 0]
    local["source_y"] = coordinates[source, 1]
    local["target_x"] = coordinates[target, 0]
    local["target_y"] = coordinates[target, 1]
    local["midpoint_x"] = (local["source_x"] + local["target_x"]) / 2
    local["midpoint_y"] = (local["source_y"] + local["target_y"]) / 2
    local["lr_activity"] = scaled[source, 0] * scaled[target, 1]
    local["attention_lr_score"] = local["lr_activity"] * positive_rank_weights(
        local["attention_abs_mean"]
    )
    local["exact_message_lr_score"] = local["lr_activity"] * positive_rank_weights(
        local["edge_message_norm_joint"]
    )
    positive = local["exact_message_lr_score"].to_numpy(dtype=float) > 0
    top_n = max(1, int(np.ceil(0.02 * positive.sum()))) if positive.any() else 0
    local["top_exact_message_lr_edge"] = False
    if top_n:
        top_index = (
            local.loc[positive, "exact_message_lr_score"]
            .sort_values(ascending=False, kind="mergesort")
            .head(top_n)
            .index
        )
        local.loc[top_index, "top_exact_message_lr_edge"] = True
    points = local[["midpoint_x", "midpoint_y"]].to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "lr_id": f"{ligand.casefold()}->{receptor.casefold()}",
                "ligand_h5ad_symbol": ligand_name,
                "receptor_h5ad_symbol": receptor_name,
                "selection_rule": (
                    "highest CytoBridge exact-message score among the frozen "
                    "original-paper 21 LR reference axes"
                ),
                "n_unique_edges": int(len(local)),
                "n_positive_lr_edges": int((local["lr_activity"] > 0).sum()),
                "n_top_exact_message_lr_edges": int(top_n),
                "lr_only_weighted_spatial_radius": _weighted_radius(
                    points, local["lr_activity"].to_numpy(dtype=float)
                ),
                "attention_weighted_spatial_radius": _weighted_radius(
                    points, local["attention_lr_score"].to_numpy(dtype=float)
                ),
                "exact_message_weighted_spatial_radius": _weighted_radius(
                    points, local["exact_message_lr_score"].to_numpy(dtype=float)
                ),
            }
        ]
    )
    return cells, local, summary


def _selected_axis_context_ranks(
    cell_types: list[str],
    spatial_edges: pd.DataFrame,
    commot_lr_contexts: pd.DataFrame,
    *,
    ligand: str,
    receptor: str,
    n_selected: int = 8,
) -> tuple[pd.DataFrame, float]:
    """Attach COMMOT ranks after selecting cell-type circuits by CytoBridge only."""

    required = {
        "sender_type",
        "receiver_type",
        "ligand",
        "receptor",
        "abundance_controlled_distinct_cell_score",
    }
    missing = sorted(required.difference(commot_lr_contexts.columns))
    if missing:
        raise ValueError(f"COMMOT LR context table lacks columns: {missing}")
    grid = pd.MultiIndex.from_product(
        [cell_types, cell_types], names=["sender_type", "receiver_type"]
    ).to_frame(index=False)
    model = spatial_edges.groupby(["sender_type", "receiver_type"], as_index=False).agg(
        cytobridge_exact_message_lr_score=("exact_message_lr_score", "mean")
    )
    local = commot_lr_contexts.copy()
    local["ligand_key"] = local["ligand"].astype(str).str.casefold()
    local["receptor_key"] = local["receptor"].astype(str).str.casefold()
    local = local.loc[
        local["ligand_key"].eq(ligand.casefold())
        & local["receptor_key"].eq(receptor.casefold())
    ].copy()
    if local.empty:
        raise ValueError(f"COMMOT lacks cell-type contexts for {ligand}->{receptor}")
    local["abundance_controlled_distinct_cell_score"] = pd.to_numeric(
        local["abundance_controlled_distinct_cell_score"], errors="raise"
    )
    if (~np.isfinite(local["abundance_controlled_distinct_cell_score"])).any() or (
        local["abundance_controlled_distinct_cell_score"] < 0
    ).any():
        raise ValueError("COMMOT LR context scores are invalid")
    external = local.groupby(["sender_type", "receiver_type"], as_index=False).agg(
        commot_abundance_controlled_distinct_cell_score=(
            "abundance_controlled_distinct_cell_score",
            "sum",
        )
    )
    result = grid.merge(
        model, on=["sender_type", "receiver_type"], how="left", validate="one_to_one"
    ).merge(
        external,
        on=["sender_type", "receiver_type"],
        how="left",
        validate="one_to_one",
    )
    score_columns = (
        "cytobridge_exact_message_lr_score",
        "commot_abundance_controlled_distinct_cell_score",
    )
    result[list(score_columns)] = result[list(score_columns)].fillna(0.0)
    result["cytobridge_rank_percentile"] = result[
        "cytobridge_exact_message_lr_score"
    ].rank(method="average", pct=True)
    result["commot_rank_percentile"] = result[
        "commot_abundance_controlled_distinct_cell_score"
    ].rank(method="average", pct=True)
    off_diagonal = result.loc[
        ~result["sender_type"].eq(result["receiver_type"])
    ].sort_values(
        ["cytobridge_exact_message_lr_score", "sender_type", "receiver_type"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    selected_indices = off_diagonal.head(n_selected).index
    result["selected_by_cytobridge"] = False
    result["cytobridge_selection_rank"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )
    result.loc[selected_indices, "selected_by_cytobridge"] = True
    result.loc[selected_indices, "cytobridge_selection_rank"] = np.arange(
        1, len(selected_indices) + 1, dtype=int
    )
    correlation = stats.spearmanr(
        result["cytobridge_exact_message_lr_score"],
        result["commot_abundance_controlled_distinct_cell_score"],
    )
    return result, float(correlation.statistic)


def _pair_metrics_table(
    cytobridge: pd.DataFrame,
    commot: pd.DataFrame,
    cellagentchat: pd.DataFrame,
    *,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    external = (
        ("COMMOT", commot, "abundance_controlled_distinct_cell_score"),
        ("CellAgentChat", cellagentchat, "cellagentchat_native_primary_mean"),
    )
    for view_index, (view, score) in enumerate(CYTOBRIDGE_VIEWS.items()):
        for method_index, (method, table, external_score) in enumerate(external):
            metric = pair_method_concordance(
                cytobridge,
                table,
                cytobridge_score=score,
                external_score=external_score,
                top_fraction=PRIMARY_TOP_FRACTION,
            )
            controlled = controlled_pair_concordance(
                cytobridge,
                table,
                cytobridge_score=score,
                external_score=external_score,
                permutations=permutations,
                seed=seed + 10000 * view_index + 1000 * method_index,
            )
            rows.append(
                {
                    "cytobridge_view": view,
                    "external_method": method,
                    "cytobridge_score_column": score,
                    "external_score_column": external_score,
                    **metric.__dict__,
                    **controlled,
                }
            )
    return pd.DataFrame(rows)


def _paper_rank_increment_table(
    cytobridge_lr: pd.DataFrame, paper_axes: pd.DataFrame
) -> pd.DataFrame:
    reference = {
        f"{ligand.casefold()}->{receptor.casefold()}"
        for ligand, receptor in paper_axes[["ligand", "receptor"]]
        .astype(str)
        .itertuples(index=False, name=None)
    }
    local = cytobridge_lr.copy()
    paper = local["lr_id"].isin(reference).to_numpy(bool)
    rows = []
    for view in ("attention", "exact_message"):
        shift = (
            local[f"{view}_rank_percentile"] - local["lr_only_rank_percentile"]
        ).to_numpy(dtype=float)
        result = stats.mannwhitneyu(shift[paper], shift[~paper], alternative="greater")
        rows.append(
            {
                "cytobridge_view": view,
                "n_paper_reference_present": int(paper.sum()),
                "n_background_nonreference": int((~paper).sum()),
                "paper_median_rank_shift_vs_lr_only": float(np.median(shift[paper])),
                "background_median_rank_shift_vs_lr_only": float(
                    np.median(shift[~paper])
                ),
                "paper_axes_with_positive_rank_shift": int((shift[paper] > 0).sum()),
                "mannwhitney_p_greater": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def _selected_pair_table(
    cytobridge: pd.DataFrame,
    commot: pd.DataFrame,
    cellagentchat: pd.DataFrame,
    *,
    n_pairs: int,
) -> pd.DataFrame:
    selected = select_pairs_by_cytobridge_only(
        cytobridge,
        score_column=CYTOBRIDGE_VIEWS["exact_message"],
        n_pairs=n_pairs,
    )
    tables = [
        external_ranks_for_selected_pairs(
            selected,
            commot,
            external_score="abundance_controlled_distinct_cell_score",
            external_method="COMMOT",
        ),
        external_ranks_for_selected_pairs(
            selected,
            cellagentchat,
            external_score="cellagentchat_native_primary_mean",
            external_method="CellAgentChat",
        ),
    ]
    return pd.concat(tables, ignore_index=True)


def _lr_comparison(
    cytobridge_lr: pd.DataFrame,
    commot_lr: pd.DataFrame,
    nichenet_lr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    merged_tables: dict[str, pd.DataFrame] = {}
    for method, table, score in (
        ("COMMOT", commot_lr, "commot_score"),
        ("NicheNet", nichenet_lr, "nichenet_score"),
    ):
        merged, metrics = shared_lr_rank_metrics(
            cytobridge_lr,
            table,
            external_score=score,
            top_fraction=PRIMARY_TOP_FRACTION,
        )
        metrics.insert(1, "external_method", method)
        metrics["external_score_column"] = score
        rows.append(metrics)
        merged_tables[method] = merged
    external_shared = commot_lr.merge(
        nichenet_lr, on="lr_id", how="inner", validate="one_to_one"
    )
    external_metric = rank_metrics(
        external_shared["commot_score"],
        external_shared["nichenet_score"],
        top_fraction=PRIMARY_TOP_FRACTION,
    )
    external_row = pd.DataFrame(
        [
            {
                "cytobridge_view": "external_only",
                "external_method": "COMMOT vs NicheNet",
                **external_metric.__dict__,
                "external_score_column": "commot_score vs nichenet_score",
            }
        ]
    )
    return (
        pd.concat([*rows, external_row], ignore_index=True),
        external_shared,
        merged_tables,
    )


def analyze(spec_path: Path, output_dir: Path, *, n_selected_pairs: int) -> None:
    spec, inputs = _load_spec(spec_path)
    output = _prepare_output(output_dir)
    copied_spec = output / "input_spec.json"
    shutil.copy2(spec_path.expanduser().resolve(), copied_spec)

    cytobridge = _terminal_stage(
        _read_csv(inputs["cytobridge_type_pair"]["path"], label="CytoBridge pairs"),
        label="CytoBridge pairs",
    )
    commot_pairs = _terminal_stage(
        _read_csv(inputs["commot_type_pair"]["path"], label="COMMOT pairs"),
        label="COMMOT pairs",
    )
    cellagentchat = _terminal_stage(
        _read_csv(
            inputs["cellagentchat_type_pair"]["path"], label="CellAgentChat pairs"
        ),
        label="CellAgentChat pairs",
    )
    types, pair_grid = _cell_type_grid(cytobridge)

    permutations = int(spec.get("permutations", PRIMARY_PERMUTATIONS))
    random_seed = int(spec.get("random_seed", PRIMARY_RANDOM_SEED))
    pair_metrics = _pair_metrics_table(
        cytobridge,
        commot_pairs,
        cellagentchat,
        permutations=permutations,
        seed=random_seed,
    )
    selected_pairs = _selected_pair_table(
        cytobridge,
        commot_pairs,
        cellagentchat,
        n_pairs=n_selected_pairs,
    )

    database_raw = _read_csv(inputs["lr_database"]["path"], label="LR database")
    paper_path = REPO_ROOT / "scripts/reviewer_zebrafish_ccc/original_paper_21_lr.csv"
    paper_axes = pd.read_csv(paper_path)
    paper_for_union = paper_axes[["ligand", "receptor"]].copy()
    paper_for_union["pathway"] = "Original-paper-2022-reference"
    paper_for_union["category"] = "Independent observational reference"
    database = collapse_lr_database(
        pd.concat([database_raw, paper_for_union], ignore_index=True)
    )
    expression, expression_audit = _load_selected_expression(
        Path(str(inputs["sample_h5ad"]["path"])),
        database,
        cell_type_key=str(spec.get("cell_type_key", "Annotation")),
    )
    if set(types) != set(expression.index.astype(str)):
        raise ValueError(
            "CytoBridge pair cell types differ from sample-H5AD cell types: "
            f"pair_only={sorted(set(types).difference(expression.index.astype(str)))}, "
            f"expression_only={sorted(set(expression.index.astype(str)).difference(types))}"
        )
    activity, represented = build_pair_lr_activity_matrix(
        pair_grid, expression, database
    )
    represented = represented.copy()
    represented["lr_matrix_index"] = np.arange(len(represented), dtype=int)
    modifiers = {
        "attention": positive_rank_weights(pair_grid["cytobridge_attention"]),
        "exact_message": positive_rank_weights(pair_grid["cytobridge_exact_message"]),
    }
    cytobridge_lr = lr_scores_from_pair_modifiers(activity, represented, modifiers)
    commot_lr_contexts = _terminal_stage(
        _read_csv(inputs["commot_lr"]["path"], label="COMMOT LR"),
        label="COMMOT LR",
    )
    commot_lr = collapse_commot_lr_scores(commot_lr_contexts)
    nichenet_raw = _read_csv(inputs["nichenet_lr"]["path"], label="NicheNet LR")
    nichenet_lr = collapse_nichenet_lr_scores(nichenet_raw)
    lr_metrics, external_shared, merged_by_method = _lr_comparison(
        cytobridge_lr, commot_lr, nichenet_lr
    )

    strata = adaptive_pair_strata(pair_grid)
    permutation_rows = []
    for method, merged in merged_by_method.items():
        external_column = "commot_score" if method == "COMMOT" else "nichenet_score"
        indices = merged["lr_matrix_index"].to_numpy(dtype=int)
        external_values = merged[external_column].to_numpy(dtype=float)
        lr_only_metric = rank_metrics(
            merged["lr_only_score"],
            external_values,
            top_fraction=PRIMARY_TOP_FRACTION,
        )
        for view in ("attention", "exact_message"):
            result = modifier_permutation_test(
                activity,
                modifiers[view],
                external_values,
                indices,
                strata,
                permutations=int(spec.get("permutations", PRIMARY_PERMUTATIONS)),
                seed=int(spec.get("random_seed", PRIMARY_RANDOM_SEED))
                + (0 if method == "COMMOT" else 1000)
                + (0 if view == "attention" else 100),
                top_fraction=PRIMARY_TOP_FRACTION,
            )
            result.update(
                {
                    "cytobridge_view": view,
                    "external_method": method,
                    "lr_only_spearman_rho": lr_only_metric.spearman_rho,
                    "spearman_delta_vs_lr_only": result["observed_spearman_rho"]
                    - lr_only_metric.spearman_rho,
                    "lr_only_top_jaccard": lr_only_metric.top_jaccard,
                    "top_jaccard_delta_vs_lr_only": result["observed_top_jaccard"]
                    - lr_only_metric.top_jaccard,
                }
            )
            permutation_rows.append(result)
    permutation_table = pd.DataFrame(permutation_rows)

    paper_annotated, paper_enrichment = paper_reference_enrichment(
        cytobridge_lr,
        paper_axes,
        score_columns=("lr_only_score", "attention_score", "exact_message_score"),
        top_fraction=0.10,
    )
    paper_scores = paper_annotated.loc[paper_annotated["paper_2022_reference"]].copy()
    paper_scores = paper_axes.merge(
        paper_scores,
        on=["ligand", "receptor"],
        how="left",
        validate="one_to_one",
        suffixes=("_paper", ""),
    )
    paper_scores["represented_in_current_expression"] = paper_scores["lr_id"].notna()
    paper_rank_increment = _paper_rank_increment_table(cytobridge_lr, paper_axes)

    represented_paper = paper_scores.loc[
        paper_scores["represented_in_current_expression"].astype(bool)
    ].copy()
    spatial_axis = represented_paper.sort_values(
        ["exact_message_score", "paper_display_order_paper"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    edge_rows, discovered_inputs = _verified_cytobridge_edges(
        Path(str(inputs["cytobridge_manifest"]["path"]))
    )
    spatial_cells, spatial_edges, spatial_summary = _spatial_axis_tables(
        Path(str(inputs["sample_h5ad"]["path"])),
        edge_rows,
        ligand=str(spatial_axis["ligand"]),
        receptor=str(spatial_axis["receptor"]),
        cell_type_key=str(spec.get("cell_type_key", "Annotation")),
        spatial_key=str(spec.get("spatial_key", "spatial_aligned")),
    )
    spatial_contexts, spatial_context_rho = _selected_axis_context_ranks(
        types,
        spatial_edges,
        commot_lr_contexts,
        ligand=str(spatial_axis["ligand"]),
        receptor=str(spatial_axis["receptor"]),
    )
    spatial_summary["cell_type_context_spearman_rho"] = spatial_context_rho
    spatial_summary["context_selection_rule"] = (
        "top eight off-diagonal contexts selected by CytoBridge exact-message LR "
        "score only; COMMOT ranks attached after selection"
    )

    targets = _read_csv(inputs["nichenet_targets"]["path"], label="NicheNet targets")
    shared_support, target_links = jointly_supported_lr_targets(
        cytobridge_lr,
        commot_lr,
        nichenet_lr,
        targets,
        cytobridge_view="exact_message",
        top_fraction=PRIMARY_TOP_FRACTION,
    )
    interaction = _read_csv(
        inputs["interaction_target_metrics"]["path"],
        label="fixed-checkpoint interaction metrics",
    )
    if "dataset" in interaction:
        interaction = interaction.loc[
            interaction["dataset"].astype(str).str.casefold().eq("zebrafish")
        ].copy()
    if interaction.empty:
        raise ValueError("interaction sensitivity table lacks zebrafish rows")

    tables = {
        "directed_pair_concordance.csv": pair_metrics,
        "cytobridge_selected_pair_external_ranks.csv": selected_pairs,
        "pair_grid_and_modifiers.csv.gz": pair_grid.assign(
            attention_rank_weight=modifiers["attention"],
            exact_message_rank_weight=modifiers["exact_message"],
            permutation_stratum=strata,
        ),
        "cytobridge_lr_scores.csv.gz": cytobridge_lr,
        "commot_lr_scores_collapsed.csv.gz": commot_lr,
        "nichenet_lr_scores_collapsed.csv.gz": nichenet_lr,
        "lr_rank_concordance.csv": lr_metrics,
        "commot_nichenet_shared_lr.csv.gz": external_shared,
        "lr_modifier_permutation_tests.csv": permutation_table,
        "original_paper_21_lr_scores.csv": paper_scores,
        "original_paper_21_lr_enrichment.csv": paper_enrichment,
        "original_paper_21_rank_increment.csv": paper_rank_increment,
        "selected_paper_axis_spatial_cells.csv.gz": spatial_cells,
        "selected_paper_axis_spatial_edges.csv.gz": spatial_edges,
        "selected_paper_axis_spatial_summary.csv": spatial_summary,
        "selected_paper_axis_context_external_ranks.csv": spatial_contexts,
        "jointly_supported_lr.csv": shared_support,
        "jointly_supported_lr_targets.csv": target_links,
        "fixed_checkpoint_interaction_on_off.csv": interaction,
    }
    artifacts: dict[str, dict[str, object]] = {}
    for filename, table in tables.items():
        path = output / filename
        table.to_csv(
            path, index=False, compression="gzip" if filename.endswith(".gz") else None
        )
        artifacts[filename] = _artifact(path)

    manifest_path = output / "analysis_manifest.json"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "claim_contract": {
            "attention_is_probability": False,
            "attention_is_lr_identity": False,
            "primary_model_contribution_view": "exact_message",
            "secondary_gate_view": "attention",
            "pair_selection": "CytoBridge exact message only; external scores inspected after selection",
            "lr_mapping": "complete directed type-pair LR activity reweighted by positive CytoBridge within-view ranks",
            "lr_only_baseline_required": True,
            "modifier_permutation_required": True,
            "nichenet_scope": "receiver-transition regulatory evidence; not native cell-pair strength",
            "original_paper_21_scope": "independent observational reference; not experimental ground truth",
            "interaction_on_off_scope": "fixed-checkpoint inference sensitivity; not a ligand knockout",
        },
        "settings": {
            "top_fraction": PRIMARY_TOP_FRACTION,
            "n_selected_pairs": int(n_selected_pairs),
            "permutations": permutations,
            "random_seed": random_seed,
            "expression_audit": expression_audit,
            "n_directed_pairs": int(len(pair_grid)),
            "n_represented_lr": int(len(cytobridge_lr)),
            "n_jointly_supported_lr": int(shared_support["jointly_supported"].sum()),
            "selected_axis_context_spearman_rho": spatial_context_rho,
        },
        "input_spec": _artifact(copied_spec),
        "inputs": inputs,
        "discovered_inputs": discovered_inputs,
        "implementation": _implementation_identity(),
        "artifacts": artifacts,
    }
    _write_json(manifest_path, manifest)
    _write_sha_sidecar(manifest_path)
    print(manifest_path)


def _verified_frozen_analysis(
    analysis_dir: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], dict[str, Path]]:
    analysis = analysis_dir.expanduser().resolve()
    manifest_path = analysis / "analysis_manifest.json"
    expected = expected_sha256.casefold()
    if len(expected) != 64 or _sha256(manifest_path) != expected:
        raise ValueError("analysis manifest does not match the required SHA-256")
    manifest = _load_json(manifest_path, label="analysis manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("workflow") != WORKFLOW
        or manifest.get("status") != "complete"
    ):
        raise ValueError("analysis manifest violates the frozen report contract")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("analysis manifest has no artifacts")
    paths: dict[str, Path] = {}
    for filename, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError(f"analysis artifact record is invalid: {filename}")
        path = analysis / str(filename)
        observed = _artifact(path)
        if observed["sha256"] != record.get("sha256") or observed[
            "size_bytes"
        ] != record.get("size_bytes"):
            raise ValueError(f"frozen analysis artifact changed: {filename}")
        paths[str(filename)] = path
    return manifest, paths


def _short_cell_type(value: str, width: int = 24) -> str:
    replacements = {
        "Musculature System, Yolk Syncytial Layer": "Musculature/YSL",
        "Posterior Erythroid Lineage Cell": "Posterior erythroid",
        "Anterior Erythroid Lineage Cell": "Anterior erythroid",
        "Spinal Cord Anterior Region": "Anterior spinal cord",
        "Spinal Cord Ventral Region": "Ventral spinal cord",
        "Spinal Cord Dorsal Region": "Dorsal spinal cord",
        "Nervous System": "Nervous system",
        "Fast Muscle Cell": "Fast muscle",
        "Slow Muscle Cell": "Slow muscle",
    }
    text = replacements.get(str(value), str(value))
    return text if len(text) <= width else text[: width - 1] + "…"


def _read_required_table(paths: Mapping[str, Path], filename: str) -> pd.DataFrame:
    if filename not in paths:
        raise ValueError(f"analysis lacks report table {filename}")
    return _read_csv(paths[filename], label=filename)


def report(
    analysis_dir: Path,
    output_dir: Path,
    *,
    expected_analysis_manifest_sha256: str,
) -> None:
    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    analysis_manifest, paths = _verified_frozen_analysis(
        analysis_dir, expected_analysis_manifest_sha256
    )
    output = _prepare_output(output_dir)
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError("Arial is required for the publication report") from error
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    pair = _read_required_table(paths, "directed_pair_concordance.csv")
    selected = _read_required_table(
        paths, "cytobridge_selected_pair_external_ranks.csv"
    )
    lr = _read_required_table(paths, "lr_rank_concordance.csv")
    permutation = _read_required_table(paths, "lr_modifier_permutation_tests.csv")
    cytobridge_lr = _read_required_table(paths, "cytobridge_lr_scores.csv.gz")
    commot_lr = _read_required_table(paths, "commot_lr_scores_collapsed.csv.gz")
    paper = _read_required_table(paths, "original_paper_21_lr_scores.csv")
    paper_enrichment = _read_required_table(
        paths, "original_paper_21_lr_enrichment.csv"
    )
    if "jointly_supported_lr_targets.csv" not in paths:
        raise ValueError("analysis lacks report table jointly_supported_lr_targets.csv")
    targets = pd.read_csv(paths["jointly_supported_lr_targets.csv"])
    cells = _read_required_table(paths, "selected_paper_axis_spatial_cells.csv.gz")
    edges = _read_required_table(paths, "selected_paper_axis_spatial_edges.csv.gz")
    spatial_summary = _read_required_table(
        paths, "selected_paper_axis_spatial_summary.csv"
    )
    spatial_contexts = _read_required_table(
        paths, "selected_paper_axis_context_external_ranks.csv"
    )

    navy = "#214E78"
    teal = "#07838B"
    pale_teal = "#A6D7D8"
    gold = "#D9A441"
    coral = "#CC6677"
    paper_red = "#B33A3A"
    pale_grey = "#D7DDE2"
    background_grey = "#E4E7E9"

    fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.95, 0.95, 1.35),
        left=0.09,
        right=0.985,
        top=0.985,
        bottom=0.065,
        hspace=0.25,
    )

    def _panel(parent, label: str, title: str):
        nested = parent.subgridspec(2, 1, height_ratios=(0.14, 1.0), hspace=0.04)
        heading = fig.add_subplot(nested[0])
        heading.axis("off")
        heading.text(
            0.0,
            0.52,
            label,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="black",
        )
        heading.text(
            0.09,
            0.52,
            title,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
        )
        return fig.add_subplot(nested[1])

    top = outer[0].subgridspec(1, 2, width_ratios=(0.40, 0.60), wspace=0.33)
    ax_a = _panel(top[0], "a", "Complete cell-pair agreement")
    pair_order = [
        ("attention", "COMMOT"),
        ("exact_message", "COMMOT"),
        ("attention", "CellAgentChat"),
        ("exact_message", "CellAgentChat"),
    ]
    rows = pair.set_index(["cytobridge_view", "external_method"]).loc[pair_order]
    y = np.arange(len(rows))[::-1]
    labels = [
        "COMMOT\nAttention",
        "COMMOT\nExact message",
        "CellAgentChat\nAttention",
        "CellAgentChat\nExact message",
    ]
    colors = [pale_teal, teal, pale_teal, teal]
    ax_a.hlines(y, 0, rows["spearman_rho"], color=pale_grey, linewidth=1.7)
    ax_a.scatter(
        rows["spearman_rho"],
        y,
        s=48,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        label="Observed",
        zorder=3,
    )
    ax_a.scatter(
        rows["adjusted_spearman_rho"],
        y,
        s=42,
        marker="D",
        facecolor="white",
        edgecolor=colors,
        linewidth=1.2,
        label="Adjusted",
        zorder=4,
    )
    ax_a.set_yticks(y, labels)
    ax_a.set_xlim(0, 1.0)
    ax_a.set_xlabel("Spearman rank correlation across 361 directed pairs")
    ax_a.axvline(0, color="black", linewidth=0.6)
    ax_a.grid(axis="x", color=pale_grey, linewidth=0.5, alpha=0.8)
    ax_a.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)

    ax_b = _panel(top[1], "b", "External ranks of CytoBridge-selected interactions")
    pivot = selected.pivot(
        index="cytobridge_selection_rank",
        columns="external_method",
        values="external_rank_percentile",
    ).reindex(columns=["COMMOT", "CellAgentChat"])
    selected_labels = (
        selected.drop_duplicates("cytobridge_selection_rank")
        .sort_values("cytobridge_selection_rank")
        .apply(
            lambda row: (
                f"{int(row['cytobridge_selection_rank'])}. "
                f"{_short_cell_type(row['sender_type'], 15)}\n"
                f"→ {_short_cell_type(row['receiver_type'], 15)}"
            ),
            axis=1,
        )
        .tolist()
    )
    y_b = np.arange(len(pivot))
    for method, color, marker in (
        ("COMMOT", teal, "s"),
        ("CellAgentChat", gold, "D"),
    ):
        ax_b.scatter(
            100 * pivot[method].to_numpy(dtype=float),
            y_b,
            s=45,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.6,
            label=method,
            zorder=3,
        )
    ax_b.axvspan(80, 100, color="#EEF6F3", zorder=0)
    ax_b.axvline(80, color="#9DA9A5", linewidth=0.7, linestyle="--")
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(len(pivot) - 0.5, -0.5)
    ax_b.set_yticks(y_b, selected_labels, fontsize=6.2, linespacing=0.92)
    ax_b.set_xlabel("External within-method rank percentile")
    ax_b.grid(axis="x", color=pale_grey, linewidth=0.5, alpha=0.8)
    ax_b.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    middle = outer[1].subgridspec(1, 1)
    c_spec = middle[0].subgridspec(2, 1, height_ratios=(0.20, 1.0), hspace=0.12)
    c_heading = fig.add_subplot(c_spec[0])
    c_heading.axis("off")
    c_heading.text(0.0, 0.52, "c", fontsize=14, fontweight="bold", va="center")
    c_heading.text(
        0.035,
        0.52,
        "Independent LR evidence and formal COMMOT overlap",
        fontsize=12,
        fontweight="bold",
        va="center",
    )
    c_content = c_spec[1].subgridspec(1, 2, width_ratios=(0.58, 0.42), wspace=0.24)
    ax_c1 = fig.add_subplot(c_content[0])
    paper_shared = (
        paper.loc[paper["represented_in_current_expression"].astype(bool)]
        .merge(
            commot_lr[["lr_id", "commot_rank_percentile"]],
            on="lr_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("paper_display_order_paper", kind="mergesort")
    )
    represented_paper = paper.loc[
        paper["represented_in_current_expression"].astype(bool)
    ].copy()
    source_axis_count = int(len(paper))
    represented_axis_count = int(len(represented_paper))
    if source_axis_count < 2 or represented_axis_count < 2:
        raise ValueError("source-atlas LR coverage requires at least two axes")
    reference_ids = set(represented_paper["lr_id"].astype(str))
    background = cytobridge_lr.loc[
        ~cytobridge_lr["lr_id"].astype(str).isin(reference_ids),
        "exact_message_rank_percentile",
    ].to_numpy(dtype=float)
    reference_values = represented_paper["exact_message_rank_percentile"].to_numpy(
        dtype=float
    )
    rng = np.random.default_rng(20260817)
    ax_c1.scatter(
        background,
        rng.normal(0.0, 0.055, len(background)),
        s=7,
        color="#B8C0C6",
        alpha=0.28,
        linewidth=0,
        rasterized=False,
        label=f"Other representable LR axes (n={len(background)})",
    )
    ax_c1.scatter(
        reference_values,
        1.0 + rng.normal(0.0, 0.055, len(reference_values)),
        s=28,
        color=paper_red,
        edgecolor="white",
        linewidth=0.5,
        label=(
            "Source-atlas Figure 5B axes "
            f"({represented_axis_count}/{source_axis_count} represented)"
        ),
        zorder=3,
    )
    enrichment = paper_enrichment.loc[
        paper_enrichment["score_column"].eq("exact_message_score")
    ].iloc[0]
    ax_c1.axvline(0.9, color="#8A949C", linewidth=0.7, linestyle="--")
    ax_c1.set_xlim(0, 1.01)
    ax_c1.set_ylim(-0.32, 1.32)
    ax_c1.set_yticks([0, 1], ["Other LR axes", "Figure 5B axes"])
    ax_c1.set_xlabel("CytoBridge exact-message LR rank percentile")
    ax_c1.grid(axis="x", color=pale_grey, linewidth=0.45, alpha=0.7)
    ax_c1.legend(frameon=False, fontsize=6.8, loc="lower left")
    ax_c1.set_title(
        f"{represented_axis_count}/{source_axis_count} represented; "
        f"{int(enrichment.paper_reference_in_top_n)}/{represented_axis_count} in top decile",
        fontsize=9,
        pad=4,
    )

    ax_c2 = fig.add_subplot(c_content[1])
    if paper_shared.empty:
        raise ValueError("no Figure 5B reference axes overlap the shared LR universe")
    paper_shared = paper_shared.sort_values(
        "exact_message_rank_percentile", ascending=True, kind="mergesort"
    )
    y_c2 = np.arange(len(paper_shared))
    for y_value, row in zip(y_c2, paper_shared.itertuples(), strict=True):
        ax_c2.plot(
            [row.exact_message_rank_percentile, row.commot_rank_percentile],
            [y_value, y_value],
            color="#AEB7BD",
            linewidth=1.2,
            zorder=1,
        )
    ax_c2.scatter(
        paper_shared["exact_message_rank_percentile"],
        y_c2,
        s=43,
        color=navy,
        edgecolor="white",
        linewidth=0.6,
        label="CytoBridge exact message",
        zorder=3,
    )
    ax_c2.scatter(
        paper_shared["commot_rank_percentile"],
        y_c2,
        s=43,
        marker="s",
        color=teal,
        edgecolor="white",
        linewidth=0.6,
        label="COMMOT",
        zorder=3,
    )
    c2_min = min(
        0.80,
        float(
            paper_shared[["exact_message_rank_percentile", "commot_rank_percentile"]]
            .min()
            .min()
        )
        - 0.03,
    )
    ax_c2.set_xlim(max(0.0, c2_min), 1.01)
    ax_c2.set_yticks(
        y_c2,
        [
            f"{row.ligand}–{row.receptor}"
            for row in paper_shared.itertuples(index=False)
        ],
        fontsize=7.2,
    )
    ax_c2.set_xlabel("Within-method LR rank percentile")
    ax_c2.set_title(
        f"{len(paper_shared)}/{represented_axis_count} in formal COMMOT CellChatDB",
        fontsize=9,
        pad=4,
    )
    ax_c2.grid(axis="x", color=pale_grey, linewidth=0.5, alpha=0.8)
    ax_c2.legend(frameon=False, fontsize=7.0, loc="lower right")
    commot_null = permutation.loc[
        permutation["external_method"].eq("COMMOT")
        & permutation["cytobridge_view"].eq("exact_message")
    ].iloc[0]

    bottom = outer[2].subgridspec(1, 2, width_ratios=(0.28, 0.72), wspace=0.24)
    ax_d = _panel(bottom[0], "d", "Receiver-response support")
    if targets.empty:
        ax_d.text(0.5, 0.5, "No jointly supported NicheNet target links", ha="center")
        ax_d.set_axis_off()
    else:
        target_summary = (
            targets.groupby(["lr_id", "target"], as_index=False)
            .agg(ligand_target_evidence=("ligand_target_evidence", "max"))
            .sort_values(
                ["ligand_target_evidence", "target"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(6)
            .sort_values("ligand_target_evidence")
        )
        target_summary = target_summary.reset_index(drop=True)
        target_y = np.arange(len(target_summary))
        ax_d.hlines(
            target_y,
            0,
            target_summary["ligand_target_evidence"],
            color=pale_grey,
            linewidth=1.4,
        )
        ax_d.scatter(
            target_summary["ligand_target_evidence"],
            target_y,
            s=45,
            color=coral,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax_d.set_yticks(target_y, target_summary["target"])
        lr_label = str(target_summary["lr_id"].iloc[0])
        ax_d.set_xlabel("NicheNet ligand–target evidence")
        ax_d.set_title(lr_label.replace("->", "–"), fontsize=9, pad=4)
    ax_d.grid(axis="x", color=pale_grey, linewidth=0.5, alpha=0.8)

    e_spec = bottom[1].subgridspec(2, 1, height_ratios=(0.14, 1.0), hspace=0.04)
    e_heading = fig.add_subplot(e_spec[0])
    e_heading.axis("off")
    e_heading.text(0.0, 0.52, "e", fontsize=14, fontweight="bold", va="center")
    e_heading.text(
        0.04,
        0.52,
        "Spatial model edges and external circuit ranks",
        fontsize=12,
        fontweight="bold",
        va="center",
    )
    e_content = e_spec[1].subgridspec(1, 2, width_ratios=(0.55, 0.45), wspace=0.27)
    ax_e1 = fig.add_subplot(e_content[0])
    ax_e2 = fig.add_subplot(e_content[1])
    x_min, x_max = cells["spatial_x"].min(), cells["spatial_x"].max()
    y_min, y_max = cells["spatial_y"].min(), cells["spatial_y"].max()
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    ax_e1.scatter(
        cells["spatial_x"],
        cells["spatial_y"],
        s=2.2,
        color=background_grey,
        alpha=0.68,
        linewidth=0,
        zorder=0,
    )
    all_top_edges = edges.loc[
        edges["top_exact_message_lr_edge"].astype(bool)
    ].sort_values("exact_message_lr_score", ascending=False, kind="mergesort")
    display_edges = all_top_edges.head(16).copy()
    if display_edges.empty:
        raise ValueError("selected spatial LR axis has no positive model edges")
    for row in display_edges.itertuples(index=False):
        ax_e1.annotate(
            "",
            xy=(row.target_x, row.target_y),
            xytext=(row.source_x, row.source_y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": gold,
                "linewidth": 1.7,
                "alpha": 0.95,
                "mutation_scale": 10.0,
                "shrinkA": 1.0,
                "shrinkB": 1.0,
                "connectionstyle": "arc3,rad=0.16",
            },
            zorder=2,
        )
    sender_cells = cells.loc[
        cells["cell_index"].isin(display_edges["source_index_stage"])
    ]
    receiver_cells = cells.loc[
        cells["cell_index"].isin(display_edges["target_index_stage"])
    ]
    ax_e1.scatter(
        sender_cells["spatial_x"],
        sender_cells["spatial_y"],
        s=6 + 15 * sender_cells["ligand_scaled_expression"],
        color=paper_red,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.88,
        zorder=3,
    )
    ax_e1.scatter(
        receiver_cells["spatial_x"],
        receiver_cells["spatial_y"],
        s=6 + 15 * receiver_cells["receptor_scaled_expression"],
        color=navy,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.88,
        zorder=3,
    )
    ax_e1.set_xlim(x_min, x_max)
    ax_e1.set_ylim(y_min, y_max)
    ax_e1.set_aspect("equal", adjustable="box")
    ax_e1.set_xticks([])
    ax_e1.set_yticks([])
    for spine in ax_e1.spines.values():
        spine.set_visible(False)
    lr_display = str(spatial_summary.lr_id.iloc[0]).replace("->", "–")
    ax_e1.set_title(
        f"{lr_display}: 16 highest model edges shown",
        fontsize=9,
        pad=3,
    )
    ax_e1.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=paper_red,
                markeredgecolor="white",
                markersize=6,
                label="Ligand-high sender",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=navy,
                markeredgecolor="white",
                markersize=6,
                label="Receptor-high receiver",
            ),
            Line2D([0], [0], color=gold, linewidth=2, label="Exact-message LR edge"),
        ],
        frameon=False,
        fontsize=6.8,
        loc="lower left",
    )

    selected_contexts = (
        spatial_contexts.loc[spatial_contexts["selected_by_cytobridge"].astype(bool)]
        .sort_values("cytobridge_selection_rank", kind="mergesort")
        .head(6)
    )
    selected_contexts = selected_contexts.iloc[::-1].reset_index(drop=True)
    y_e2 = np.arange(len(selected_contexts))
    for y_value, row in zip(y_e2, selected_contexts.itertuples(), strict=True):
        ax_e2.plot(
            [row.cytobridge_rank_percentile, row.commot_rank_percentile],
            [y_value, y_value],
            color="#AEB7BD",
            linewidth=1.15,
            zorder=1,
        )
    ax_e2.scatter(
        selected_contexts["cytobridge_rank_percentile"],
        y_e2,
        s=39,
        color=navy,
        edgecolor="white",
        linewidth=0.55,
        label="CytoBridge exact message",
        zorder=3,
    )
    ax_e2.scatter(
        selected_contexts["commot_rank_percentile"],
        y_e2,
        s=39,
        marker="s",
        color=teal,
        edgecolor="white",
        linewidth=0.55,
        label="COMMOT",
        zorder=3,
    )
    context_labels = [
        f"{_short_cell_type(row.sender_type, 18)} →\n"
        f"{_short_cell_type(row.receiver_type, 18)}"
        for row in selected_contexts.itertuples(index=False)
    ]
    ax_e2.set_yticks(y_e2, context_labels, fontsize=6.3, linespacing=0.9)
    context_min = min(
        0.60,
        float(
            selected_contexts[["cytobridge_rank_percentile", "commot_rank_percentile"]]
            .min()
            .min()
        )
        - 0.03,
    )
    ax_e2.set_xlim(max(0.0, context_min), 1.01)
    ax_e2.set_xlabel("Within-axis cell-type-circuit rank percentile")
    ax_e2.set_title(
        f"Top circuits selected by CytoBridge only\n"
        "Full 19 × 19 context ρ = "
        f"{float(spatial_summary.cell_type_context_spearman_rho.iloc[0]):.3f}",
        fontsize=8.6,
        pad=3,
    )
    ax_e2.grid(axis="x", color=pale_grey, linewidth=0.5, alpha=0.8)
    ax_e2.legend(frameon=False, fontsize=6.7, loc="lower left")

    pdf_path = output / "zebrafish_attention_validation_a4.pdf"
    png_path = output / "zebrafish_attention_validation_a4.png"
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=320, facecolor="white")
    plt.close(fig)

    commot_pair = pair.set_index(["cytobridge_view", "external_method"])
    attn_commot = commot_pair.loc[("attention", "COMMOT")]
    exact_commot = commot_pair.loc[("exact_message", "COMMOT")]
    lr_commot = lr.set_index(["cytobridge_view", "external_method"])
    exact_lr = lr_commot.loc[("exact_message", "COMMOT")]
    context_rho = float(spatial_summary.cell_type_context_spearman_rho.iloc[0])
    n_displayed_edges = min(16, len(all_top_edges))
    n_top_edges = int(spatial_summary.n_top_exact_message_lr_edges.iloc[0])
    caption = (
        "Independent validation of the learned zebrafish interaction field. "
        "(a) Across the complete 19 x 19 directed cell-type-pair universe, "
        f"CytoBridge attention and exact message agreed with COMMOT (Spearman rho="
        f"{attn_commot.spearman_rho:.3f} and {exact_commot.spearman_rho:.3f}). Open "
        "diamonds show rank agreement after linear adjustment for sender/receiver "
        "abundance, mean spatial distance, and self-pair status. (b) External rank "
        "percentiles were read only after the eight highest off-diagonal pairs had "
        "been selected by CytoBridge exact message. (c) The independent zebrafish "
        f"expression-atlas paper reported {source_axis_count} developmental LR axes. "
        f"{represented_axis_count} were representable in the current H5AD and were "
        f"compared with all {len(background)} other representable LR axes; "
        f"{int(enrichment.paper_reference_in_top_n)}/{represented_axis_count} fell in "
        "the CytoBridge exact-message top decile "
        f"(rank AUC={float(enrichment.paper_reference_auc):.3f}, one-sided "
        f"Mann-Whitney P={float(enrichment.mannwhitney_p_greater):.2g}). "
        f"{len(paper_shared)} of the {represented_axis_count} were also present in the "
        "stricter CytoBridge-COMMOT shared LR universe and "
        "are shown in the paired rank plot. The equivalent expression-only enrichment "
        "means this source-atlas result establishes molecular compatibility rather "
        "than an incremental attention effect. Across all 538 shared LR candidates, "
        f"the exact-message ranking agreed with COMMOT (rho={exact_lr.spearman_rho:.3f}) "
        "and exceeded the geometry- and abundance-stratified modifier-permutation null "
        f"(P={commot_null.spearman_empirical_p_upper:.3g}). (d) The "
        "jointly supported "
        "col1a2-sdc4 axis is linked to NicheNet receiver-response targets. NicheNet is "
        "used as regulatory evidence rather than as a cell-pair strength. (e) The "
        "highest exact-message-scoring reference axis, mdka-sdc4, is localized on the "
        f"observed terminal sample. Gold arrows show the {n_displayed_edges} strongest "
        f"edges used for display from the predeclared top-2% set (n={n_top_edges}); "
        "sender and receiver nodes are scaled by ligand and receptor expression. The six "
        "highest off-diagonal cell-type circuits were selected by CytoBridge only and "
        f"then assigned COMMOT ranks (full 19 x 19 context rho={context_rho:.3f}). "
        "Exact-message LR scores are post-hoc model-contribution-by-LR-activity scores, "
        "not biochemical probabilities or native ligand identities."
    )
    (output / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    reviewer_response = f"""# Response to reviewer concern on attention interpretability

We agree that attention weights alone cannot be read as biochemical communication probabilities. We therefore separated the validation into three predeclared levels and revised the interpretation accordingly.

1. **Complete cell-pair field.** All 361 directed cell-type pairs were retained, including structural zeros. Attention agreed with COMMOT across the full field (Spearman rho={attn_commot.spearman_rho:.3f}), while the exact interaction message, which contains the learned direction and magnitude rather than only the gate, showed stronger agreement (rho={exact_commot.spearman_rho:.3f}). The analysis also reports correlations after adjustment for abundance, spatial distance, and self-pair status and an empirical within-stratum permutation test.
2. **No circular pair selection.** Eight off-diagonal pairs were selected using CytoBridge exact message only. COMMOT and CellAgentChat ranks were attached after the selection had been frozen.
3. **Independent molecular interpretation.** Every representable LR candidate was scored over the complete pair field. The independent source list contains {source_axis_count} developmental axes from Figure 5B of *Spatiotemporal mapping of gene expression landscapes and developmental trajectories during zebrafish embryogenesis*. {represented_axis_count} of those {source_axis_count} axes were representable in the current H5AD; {int(enrichment.paper_reference_in_top_n)}/{represented_axis_count} fell in the CytoBridge exact-message top decile (rank AUC={float(enrichment.paper_reference_auc):.3f}, one-sided Mann-Whitney P={float(enrichment.mannwhitney_p_greater):.2g}). {len(paper_shared)} of the {represented_axis_count} were also present in the stricter CytoBridge-COMMOT shared LR universe, and both methods ranked those axes highly. The equivalent expression-only enrichment is reported explicitly, so the source-atlas result is interpreted as molecular compatibility rather than an incremental attention contribution. Separately, exact-message reweighting over all 538 shared LR candidates showed COMMOT rank agreement of rho={exact_lr.spearman_rho:.3f} and exceeded the stratified modifier-permutation null (P={commot_null.spearman_empirical_p_upper:.3g}). NicheNet connects a jointly supported LR axis to receiver-response targets and is labelled as regulatory evidence rather than pair strength.
4. **Spatial and circuit-level localization.** The mdka-sdc4 axis was chosen by the highest CytoBridge exact-message score within the frozen source-atlas list. The map now shows directed model edges rather than an expression-only proxy: gold arrows are exact-message-by-LR-activity edges, red nodes are ligand-high senders, and blue nodes are receptor-high receivers. The highest off-diagonal cell-type circuits were selected by CytoBridge alone and received COMMOT ranks only afterward; agreement over the full 19 x 19 context grid was rho={context_rho:.3f}.

Together, the complete-pair, non-circular selection, molecular-ranking, independent developmental-axis, receiver-target, and spatial-circuit analyses show that the learned interaction operator is associated with independently derived communication structure and known zebrafish signaling axes. They do not establish that an attention coefficient is itself a biochemical ligand-receptor strength. The non-monotonic fixed-checkpoint interaction-off sensitivity remains in the analysis archive but is not used as validation evidence in the reviewer-facing figure.
"""
    (output / "reviewer_response.md").write_text(reviewer_response, encoding="utf-8")
    provenance = (
        "# Provenance\n\n"
        "## Source paths\n\n"
        f"- Frozen analysis directory: `{analysis_dir.expanduser().resolve()}`\n"
        f"- Upstream analysis manifest SHA-256: `{expected_analysis_manifest_sha256}`\n"
        f"- Workflow: `{WORKFLOW}` schema {SCHEMA_VERSION}\n"
        "- Pair selection: CytoBridge exact-message only; external methods inspected afterward.\n"
        "- LR baseline: expression-only activity is retained in every comparison.\n"
        "- NicheNet scope: all-confidence zebrafish-to-mouse mapping sensitivity; regulatory evidence only.\n"
        "- Original-paper reference: frozen 21-axis list from Figure 5B of the independent zebrafish expression-atlas paper.\n"
        "- Fixed-checkpoint sensitivity: retained in the analysis archive but omitted from the reviewer-facing figure and claim.\n"
        "\n## Rebuild\n\n"
        "```text\n"
        "python scripts/run_zebrafish_attention_validation.py report \\\n"
        f"  --analysis-dir {analysis_dir.expanduser().resolve()} \\\n"
        f"  --expected-analysis-manifest-sha256 {expected_analysis_manifest_sha256} \\\n"
        f"  --output-dir {output_dir.expanduser().resolve()}\n"
        "```\n"
    )
    (output / "provenance.md").write_text(provenance, encoding="utf-8")
    shutil.copy2(
        analysis_dir.expanduser().resolve() / "analysis_manifest.json",
        output / "analysis_manifest.json",
    )
    panel_dir = output / "panel_data"
    panel_dir.mkdir()
    panel_filenames = (
        "directed_pair_concordance.csv",
        "cytobridge_selected_pair_external_ranks.csv",
        "lr_rank_concordance.csv",
        "lr_modifier_permutation_tests.csv",
        "cytobridge_lr_scores.csv.gz",
        "commot_lr_scores_collapsed.csv.gz",
        "original_paper_21_lr_scores.csv",
        "original_paper_21_lr_enrichment.csv",
        "jointly_supported_lr_targets.csv",
        "selected_paper_axis_spatial_cells.csv.gz",
        "selected_paper_axis_spatial_edges.csv.gz",
        "selected_paper_axis_spatial_summary.csv",
        "selected_paper_axis_context_external_ranks.csv",
    )
    for filename in panel_filenames:
        shutil.copy2(paths[filename], panel_dir / filename)
    report_manifest_path = output / "report_manifest.json"
    report_artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != report_manifest_path:
            report_artifacts[str(path.relative_to(output))] = _artifact(path)
    report_manifest = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "analysis_manifest_sha256": expected_analysis_manifest_sha256.casefold(),
        "implementation": _implementation_identity(),
        "claim_contract": analysis_manifest.get("claim_contract"),
        "artifacts": report_artifacts,
    }
    _write_json(report_manifest_path, report_manifest)
    _write_sha_sidecar(report_manifest_path)
    print(report_manifest_path)


def validate_report(output_dir: Path) -> None:
    output = output_dir.expanduser().resolve()
    manifest_path = output / "report_manifest.json"
    manifest = _load_json(manifest_path, label="report manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("workflow") != WORKFLOW
        or manifest.get("status") != "complete"
    ):
        raise ValueError("report manifest violates the frozen contract")
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[
        0
    ] != _sha256(manifest_path):
        raise ValueError("report manifest SHA sidecar is missing or stale")
    if _implementation_identity() != manifest.get("implementation"):
        raise ValueError("report implementation changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("report manifest has no artifacts")
    for relative, expected in artifacts.items():
        path = output / relative
        observed = _artifact(path)
        if observed["sha256"] != expected.get("sha256") or observed[
            "size_bytes"
        ] != expected.get("size_bytes"):
            raise ValueError(f"report artifact changed: {relative}")
    print("PASS")


def validate(output_dir: Path) -> None:
    output = output_dir.expanduser().resolve()
    manifest_path = output / "analysis_manifest.json"
    manifest = _load_json(manifest_path, label="analysis manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("workflow") != WORKFLOW
        or manifest.get("status") != "complete"
    ):
        raise ValueError("analysis manifest violates the frozen contract")
    sidecar = manifest_path.with_name(manifest_path.name + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[
        0
    ] != _sha256(manifest_path):
        raise ValueError("analysis manifest SHA sidecar is missing or stale")
    spec_path = Path(str(manifest["input_spec"]["path"]))
    _, current_inputs = _load_spec(spec_path)
    if current_inputs != manifest.get("inputs"):
        raise ValueError("analysis inputs changed after analysis")
    if _implementation_identity() != manifest.get("implementation"):
        raise ValueError("analysis implementation changed after analysis")
    discovered = manifest.get("discovered_inputs")
    if not isinstance(discovered, dict) or not discovered:
        raise ValueError("analysis manifest has no discovered edge inputs")
    for label, expected in discovered.items():
        if _artifact(Path(str(expected["path"]))) != expected:
            raise ValueError(f"analysis discovered input changed: {label}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("analysis manifest has no artifacts")
    for label, expected in artifacts.items():
        if _artifact(Path(str(expected["path"]))) != expected:
            raise ValueError(f"analysis artifact changed: {label}")
    print("PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze", help="Create frozen numerical evidence")
    analyze_parser.add_argument("--spec", required=True, type=Path)
    analyze_parser.add_argument("--output-dir", required=True, type=Path)
    analyze_parser.add_argument("--n-selected-pairs", type=int, default=8)
    report_parser = sub.add_parser("report", help="Render one frozen analysis")
    report_parser.add_argument("--analysis-dir", required=True, type=Path)
    report_parser.add_argument("--output-dir", required=True, type=Path)
    report_parser.add_argument("--expected-analysis-manifest-sha256", required=True)
    validate_parser = sub.add_parser("validate", help="Re-hash one completed analysis")
    validate_parser.add_argument("--output-dir", required=True, type=Path)
    validate_report_parser = sub.add_parser(
        "validate-report", help="Re-hash one completed figure report"
    )
    validate_report_parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "analyze":
        analyze(args.spec, args.output_dir, n_selected_pairs=int(args.n_selected_pairs))
    elif args.command == "report":
        report(
            args.analysis_dir,
            args.output_dir,
            expected_analysis_manifest_sha256=str(
                args.expected_analysis_manifest_sha256
            ),
        )
    elif args.command == "validate":
        validate(args.output_dir)
    else:
        validate_report(args.output_dir)


if __name__ == "__main__":
    main()
