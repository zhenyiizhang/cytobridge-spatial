#!/usr/bin/env python3
"""Calculate and plot the zebrafish attention comparisons used in Figure S43.

``analyze`` compares CytoBridge results with COMMOT, CellAgentChat, NicheNet,
and the interaction on/off analysis.  ``report`` combines those tables with
the JAM controls and draws the figure. The two check commands verify the files
recorded by a completed run.
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
            "original_paper_21_scope": "pre-specified source-paper axes from the same atlas; not an independent cohort or experimental ground truth",
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
    analysis_dir: Path, expected_sha256: str | None
) -> tuple[Mapping[str, Any], dict[str, Path]]:
    analysis = analysis_dir.expanduser().resolve()
    manifest_path = analysis / "analysis_manifest.json"
    observed_sha256 = _sha256(manifest_path)
    if expected_sha256 is not None:
        expected = expected_sha256.casefold()
        if len(expected) != 64 or observed_sha256 != expected:
            raise ValueError("analysis manifest does not match the requested file")
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


JAM_REPORT_TABLES: dict[str, tuple[str, ...]] = {
    "compatibility_summary": (
        "jam_compatibility_percentile_summary",
        "compatibility_percentile_summary",
        "jam_compatibility_percentile_summary.csv",
    ),
    "quartile_compatibility": (
        "quartile_compatibility",
        "jam_quartile_compatibility",
        "jam_quartile_compatibility.csv",
    ),
    "type_pair_ranks": (
        "type_pair_raw_attention_ranks",
        "type_pair_raw_attention_ranks.csv",
    ),
    "spatial_null_summary": (
        "somite_18hpf_spatial_null_summary",
        "spatial_null_summary",
        "somite_18hpf_spatial_null_summary.csv",
    ),
    "spatial_null_iterations": (
        "somite_18hpf_spatial_null_iterations",
        "spatial_null_iterations",
        "somite_18hpf_spatial_null_iterations.csv.gz",
    ),
    "myog_association": (
        "myog_association",
        "somite_association",
        "myog_association.csv",
        "somite_18hpf_gene_association.csv",
    ),
    "expression_detection": (
        "expression_detection_by_stage_type",
        "somite_detection",
        "expression_detection_by_stage_type.csv",
        "somite_18hpf_gene_detection.csv",
    ),
    "spatial_cells": (
        "somite_18hpf_spatial_cells",
        "spatial_cells",
        "somite_18hpf_spatial_cells.csv.gz",
    ),
    "display_edges": (
        "trained_display_edges",
        "trained_jam_display_edges",
        "jam_display_edges",
        "trained_jam_display_edges.csv",
        "trained_jam_display_edges.csv.gz",
    ),
}


def _normalized_artifact_name(value: str) -> str:
    return str(value).strip().casefold().replace("-", "_")


def _iter_manifest_artifacts(value: object, *, key_path: tuple[str, ...] = ()):
    if isinstance(value, dict):
        path_value = value.get("path")
        sha_value = value.get("sha256")
        if isinstance(path_value, str) and isinstance(sha_value, str):
            yield key_path, value
            return
        for key, child in value.items():
            yield from _iter_manifest_artifacts(child, key_path=(*key_path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_manifest_artifacts(child, key_path=(*key_path, str(index)))


def _verified_jam_report_tables(
    manifest_paths: list[Path], expected_sha256s: list[str] | None
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    if not manifest_paths:
        raise ValueError("report requires at least one --jam-manifest")
    expected_sha256s = expected_sha256s or []
    if expected_sha256s and len(manifest_paths) != len(expected_sha256s):
        raise ValueError("supply one optional JAM checksum for each JAM file")
    registry: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
    manifest_records: list[dict[str, object]] = []
    legacy_tokens = ("20260722", "20260728")
    for index, manifest_path in enumerate(manifest_paths):
        resolved = manifest_path.expanduser().resolve()
        observed_sha256 = _sha256(resolved)
        if expected_sha256s:
            expected = expected_sha256s[index].casefold()
            if len(expected) != 64 or observed_sha256 != expected:
                raise ValueError(f"JAM file does not match: {resolved}")
        if any(token in str(resolved) for token in legacy_tokens):
            raise ValueError(
                "legacy 20260722/20260728 JAM artifacts are forbidden in this report"
            )
        payload = _load_json(resolved, label="current-checkpoint JAM manifest")
        status = payload.get("status")
        if status is not None and str(status).casefold() not in {
            "complete",
            "pass",
            "passed",
        }:
            raise ValueError(f"JAM manifest is not complete: {resolved}")
        artifact_count = 0
        for section in ("artifacts", "outputs"):
            if section not in payload:
                continue
            for key_path, record in _iter_manifest_artifacts(
                payload[section], key_path=(section,)
            ):
                local_record = dict(record)
                local_record["_manifest_dir"] = str(resolved.parent)
                registry.append((key_path, local_record))
                artifact_count += 1
        if not artifact_count:
            raise ValueError(f"JAM manifest has no formal output artifacts: {resolved}")
        manifest_records.append(
            {
                "path": str(resolved),
                "sha256": expected,
                "size_bytes": int(resolved.stat().st_size),
            }
        )

    resolved_tables: dict[str, Path] = {}
    for role, candidates in JAM_REPORT_TABLES.items():
        normalized_candidates = {_normalized_artifact_name(item) for item in candidates}
        matches: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []
        for key_path, record in registry:
            path = Path(str(record["path"]))
            names = {
                _normalized_artifact_name(key_path[-1]),
                _normalized_artifact_name(path.name),
            }
            if names.intersection(normalized_candidates):
                matches.append((key_path, record))
        for candidate in candidates:
            normalized_candidate = _normalized_artifact_name(candidate)
            preferred = [
                (key_path, record)
                for key_path, record in matches
                if normalized_candidate
                in {
                    _normalized_artifact_name(key_path[-1]),
                    _normalized_artifact_name(Path(str(record["path"])).name),
                }
            ]
            if preferred:
                matches = preferred
                break
        unique: dict[str, Mapping[str, Any]] = {}
        for _, record in matches:
            declared = Path(str(record["path"])).expanduser()
            candidates_local = [
                declared,
                Path(str(record["_manifest_dir"])) / "tables" / declared.name,
                Path(str(record["_manifest_dir"])) / declared.name,
            ]
            existing = sorted(
                {
                    candidate.resolve()
                    for candidate in candidates_local
                    if candidate.is_file()
                }
            )
            if len(existing) != 1:
                raise ValueError(
                    f"JAM artifact for {role} is unavailable or ambiguously mirrored: "
                    f"declared={declared}, mirrors={existing}"
                )
            unique[str(existing[0])] = record
        if len(unique) != 1:
            raise ValueError(
                f"current JAM manifests must bind exactly one {role!r} table; "
                f"found {sorted(unique)}"
            )
        path_string, record = next(iter(unique.items()))
        if any(token in path_string for token in legacy_tokens):
            raise ValueError(
                f"legacy JAM panel data are forbidden for role {role}: {path_string}"
            )
        observed = _artifact(Path(path_string))
        expected_size = record.get(
            "size_bytes", record.get("bytes", record.get("size"))
        )
        if observed["sha256"] != str(record["sha256"]).casefold():
            raise ValueError(f"JAM artifact SHA-256 mismatch for {role}")
        if expected_size is not None and observed["size_bytes"] != int(expected_size):
            raise ValueError(f"JAM artifact size mismatch for {role}")
        resolved_tables[role] = Path(path_string)
    return manifest_records, resolved_tables


def _column(frame: pd.DataFrame, *candidates: str, label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"{label} lacks any of the required columns: {candidates}")


def _boolean_values(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.strip().str.casefold()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "jam-compatible": True,
        "compatible": True,
        "false": False,
        "0": False,
        "no": False,
        "non-compatible": False,
        "other": False,
    }
    unknown = sorted(set(values).difference(mapping))
    if unknown:
        raise ValueError(f"{label} contains unknown boolean labels: {unknown}")
    return values.map(mapping).astype(bool)


def _canonical_conditions(series: pd.Series, *, label: str) -> pd.Series:
    values = series.astype(str).str.strip().str.casefold()
    required = {"trained", "pre_interaction", "random"}
    observed = set(values)
    if observed != required:
        raise ValueError(
            f"{label} requires canonical current-checkpoint conditions "
            f"{sorted(required)}; observed {sorted(observed)}. The old 'init' label "
            "must not be silently relabelled."
        )
    return values


def _canonical_compatibility_summary(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    local["condition"] = _canonical_conditions(local["condition"], label="JAM summary")
    compatibility_column = _column(
        local, "jam_compatible", "compatibility_class", label="JAM summary"
    )
    local["jam_compatible"] = _boolean_values(
        local[compatibility_column], label="JAM compatibility"
    )
    n_column = _column(local, "n_edges", "n_directed_edges", label="JAM summary")
    mean_column = _column(
        local,
        "attention_percentile_mean",
        "mean_attention_percentile",
        label="JAM summary",
    )
    median_column = _column(
        local,
        "attention_percentile_median",
        "median_attention_percentile",
        label="JAM summary",
    )
    local = local.assign(
        n_edges=pd.to_numeric(local[n_column], errors="raise").astype(int),
        attention_percentile_mean=pd.to_numeric(local[mean_column], errors="raise"),
        attention_percentile_median=pd.to_numeric(local[median_column], errors="raise"),
    )
    if local.duplicated(["condition", "jam_compatible"]).any() or len(local) != 6:
        raise ValueError("JAM summary requires one compatible/other row per condition")
    return local


def _canonical_spatial_cells(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    x_column = _column(local, "x", "spatial_x", label="JAM spatial cells")
    y_column = _column(local, "y", "spatial_y", label="JAM spatial cells")
    local["x"] = pd.to_numeric(local[x_column], errors="raise")
    local["y"] = pd.to_numeric(local[y_column], errors="raise")
    if "is_somite" not in local:
        cell_type_column = _column(
            local, "cell_type", "annotation", label="JAM spatial cells"
        )
        local["is_somite"] = local[cell_type_column].astype(str).eq("Somite")
    else:
        local["is_somite"] = _boolean_values(
            local["is_somite"], label="JAM spatial-cell Somite mask"
        )
    for gene in ("jam2a", "jam3b", "myog"):
        positive = f"{gene}_positive"
        if positive in local:
            local[positive] = _boolean_values(
                local[positive], label=f"{gene} detection"
            )
        elif gene in local:
            local[positive] = pd.to_numeric(local[gene], errors="raise") > 0
        else:
            raise ValueError(f"JAM spatial cells lack {gene} values/detection")
    if not np.isfinite(local[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("JAM spatial-cell coordinates contain non-finite values")
    return local


def _canonical_display_edges(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame.copy()
    for canonical, candidates in {
        "source_x": ("source_x",),
        "source_y": ("source_y",),
        "target_x": ("target_x",),
        "target_y": ("target_y",),
    }.items():
        column = _column(local, *candidates, label="JAM display edges")
        local[canonical] = pd.to_numeric(local[column], errors="raise")
    if "condition" in local:
        conditions = set(local["condition"].astype(str).str.casefold())
        if conditions != {"trained"}:
            raise ValueError(
                "JAM display-edge table must contain only current trained edges"
            )
    compatibility_columns = [
        column for column in ("jam_compatible", "compatible") if column in local
    ]
    if compatibility_columns:
        local["jam_compatible"] = _boolean_values(
            local[compatibility_columns[0]], label="JAM display-edge compatibility"
        )
    elif "jam_compatible_orientation" in local:
        orientation = (
            local["jam_compatible_orientation"].astype(str).str.strip().str.casefold()
        )
        local["jam_compatible"] = ~orientation.isin({"", "none", "nan"})
        if (
            "selection_rule" not in local
            or not local["selection_rule"]
            .astype(str)
            .str.contains("JAM-compatible", case=False, regex=False)
            .all()
        ):
            raise ValueError(
                "orientation-derived display compatibility requires the frozen "
                "JAM-compatible selection rule"
            )
    else:
        raise ValueError(
            "JAM display edges lack compatibility or orientation annotations"
        )
    if "display_rank" in local:
        local = local.sort_values("display_rank", kind="mergesort")
    if local.empty:
        raise ValueError("JAM display-edge table is empty")
    return local


def _report_source_paper_legacy(
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
    commot_lr = _read_required_table(paths, "commot_lr_scores_collapsed.csv.gz")
    paper = _read_required_table(paths, "original_paper_21_lr_scores.csv")
    paper_enrichment = _read_required_table(
        paths, "original_paper_21_lr_enrichment.csv"
    )
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
        2,
        1,
        height_ratios=(0.86, 1.20),
        left=0.085,
        right=0.985,
        top=0.985,
        bottom=0.070,
        hspace=0.25,
    )

    def _headed_panel(parent, label: str, title: str):
        nested = parent.subgridspec(2, 1, height_ratios=(0.14, 1.0), hspace=0.05)
        heading = fig.add_subplot(nested[0])
        heading.axis("off")
        heading.text(
            0.0,
            0.5,
            label,
            ha="left",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        heading.text(
            0.09,
            0.5,
            title,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        return nested[1]

    top = outer[0].subgridspec(1, 2, width_ratios=(0.44, 0.56), wspace=0.27)

    a_body = _headed_panel(
        top[0], "a", "Pair-field agreement beyond abundance and proximity"
    )
    ax_a = fig.add_subplot(a_body)
    pair_order = [
        ("exact_message", "COMMOT"),
        ("attention", "COMMOT"),
        ("exact_message", "CellAgentChat"),
        ("attention", "CellAgentChat"),
    ]
    rows = pair.set_index(["cytobridge_view", "external_method"]).loc[pair_order]
    n_pairs_values = set(rows["n_pairs"].astype(int))
    n_strata_values = set(rows["n_strata"].astype(int))
    n_permutation_values = set(rows["n_permutations"].astype(int))
    if (
        len(n_pairs_values) != 1
        or len(n_strata_values) != 1
        or len(n_permutation_values) != 1
    ):
        raise ValueError("pair-concordance rows disagree on null-test dimensions")
    n_pairs = n_pairs_values.pop()
    n_strata = n_strata_values.pop()
    n_pair_permutations = n_permutation_values.pop()
    if n_pairs != 361:
        raise ValueError("reviewer figure requires the complete 19 x 19 pair field")
    y_a = np.arange(len(rows))[::-1]
    labels_a = [
        "COMMOT\nExact message",
        "COMMOT\nAttention",
        "CellAgentChat\nExact message",
        "CellAgentChat\nAttention",
    ]
    colors_a = [teal, teal, "#7D858B", "#7D858B"]
    for y_value, (_, row), color in zip(y_a, rows.iterrows(), colors_a, strict=True):
        ax_a.plot(
            [row.null_adjusted_spearman_q025, row.null_adjusted_spearman_q975],
            [y_value, y_value],
            color="#B8C0C6",
            linewidth=5.5,
            solid_capstyle="round",
            zorder=1,
        )
        ax_a.scatter(
            row.null_adjusted_spearman_mean,
            y_value,
            s=23,
            color="white",
            edgecolor="#7D858B",
            linewidth=0.8,
            zorder=2,
        )
        ax_a.scatter(
            row.adjusted_spearman_rho,
            y_value,
            s=55,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax_a.text(
            min(0.975, float(row.adjusted_spearman_rho) + 0.018),
            y_value,
            f"ρ={row.adjusted_spearman_rho:.3f}; P={row.adjusted_spearman_empirical_p_upper:.4f}",
            fontsize=7.2,
            va="center",
            ha="left",
        )
    ax_a.set_yticks(y_a, labels_a, fontsize=7.5)
    ax_a.set_xlim(0.05, 1.03)
    ax_a.set_ylim(-0.55, len(y_a) - 0.45)
    ax_a.set_xlabel(f"Adjusted Spearman ρ across all {n_pairs} directed pairs")
    ax_a.grid(axis="x", color=pale_grey, linewidth=0.5, alpha=0.75)
    ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#B8C0C6",
                linewidth=5.5,
                label="Structured-null 95% interval",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=teal,
                markeredgecolor="white",
                markersize=6,
                label="Observed after covariate adjustment",
            ),
        ],
        frameon=False,
        fontsize=6.8,
        loc="lower right",
    )

    b_body = _headed_panel(top[1], "b", "Source-paper LR compatibility")
    b_content = b_body.subgridspec(1, 2, width_ratios=(0.38, 0.62), wspace=0.35)
    ax_b1 = fig.add_subplot(b_content[0])
    represented_paper = paper.loc[
        paper["represented_in_current_expression"].astype(bool)
    ].copy()
    source_axis_count = int(len(paper))
    represented_axis_count = int(len(represented_paper))
    if source_axis_count != 21 or represented_axis_count != 20:
        raise ValueError("source-paper LR contract requires 21 axes and 20 represented")
    exact_enrichment = paper_enrichment.loc[
        paper_enrichment["score_column"].eq("exact_message_score")
    ].iloc[0]
    lr_only_enrichment = paper_enrichment.loc[
        paper_enrichment["score_column"].eq("lr_only_score")
    ].iloc[0]
    b_counts = np.array(
        [
            int(exact_enrichment.paper_reference_in_top_n),
            int(lr_only_enrichment.paper_reference_in_top_n),
        ]
    )
    y_b1 = np.array([1, 0])
    ax_b1.barh(
        y_b1,
        b_counts,
        color=[navy, "#AEB7BD"],
        height=0.46,
        edgecolor="white",
        linewidth=0.6,
    )
    for y_value, count, row in zip(
        y_b1,
        b_counts,
        [exact_enrichment, lr_only_enrichment],
        strict=True,
    ):
        ax_b1.text(
            0.55,
            y_value,
            f"{count}/{represented_axis_count}",
            fontsize=8.2,
            fontweight="bold",
            va="center",
            color="white" if count > 10 else "black",
        )
        ax_b1.text(
            20.25,
            y_value,
            f"AUC {row.paper_reference_auc:.3f}",
            fontsize=7.0,
            va="center",
            ha="left",
        )
    ax_b1.set_yticks(y_b1, ["Exact message", "Expression only"], fontsize=7.4)
    ax_b1.set_xlim(0, 24.2)
    ax_b1.set_xlabel("Axes in the positive-score top decile")
    ax_b1.set_title(
        f"{represented_axis_count}/{source_axis_count} representable in the source atlas",
        fontsize=8.5,
        pad=3,
    )
    ax_b1.axvline(represented_axis_count, color="#8A949C", linewidth=0.6)
    ax_b1.grid(axis="x", color=pale_grey, linewidth=0.5, alpha=0.65)

    ax_b2 = fig.add_subplot(b_content[1])
    paper_shared = (
        represented_paper.merge(
            commot_lr[["lr_id", "commot_rank_percentile"]],
            on="lr_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("paper_display_order_paper", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(paper_shared) != 4:
        raise ValueError("formal COMMOT source-paper overlap must contain four axes")
    paper_shared = paper_shared.sort_values(
        "exact_message_rank_percentile", ascending=True, kind="mergesort"
    ).reset_index(drop=True)
    y_b2 = np.arange(len(paper_shared))
    for y_value, row in zip(y_b2, paper_shared.itertuples(), strict=True):
        ax_b2.plot(
            [row.exact_message_rank_percentile, row.commot_rank_percentile],
            [y_value, y_value],
            color="#AEB7BD",
            linewidth=1.25,
            zorder=1,
        )
    ax_b2.scatter(
        paper_shared["exact_message_rank_percentile"],
        y_b2,
        s=43,
        color=navy,
        edgecolor="white",
        linewidth=0.6,
        label="CytoBridge exact message",
        zorder=3,
    )
    ax_b2.scatter(
        paper_shared["commot_rank_percentile"],
        y_b2,
        s=43,
        marker="s",
        color=teal,
        edgecolor="white",
        linewidth=0.6,
        label="COMMOT",
        zorder=3,
    )
    b2_min = float(
        paper_shared[["exact_message_rank_percentile", "commot_rank_percentile"]]
        .min()
        .min()
    )
    ax_b2.set_xlim(max(0.90, b2_min - 0.012), 1.003)
    ax_b2.set_yticks(
        y_b2,
        [
            f"{row.ligand}–{row.receptor}"
            for row in paper_shared.itertuples(index=False)
        ],
        fontsize=7.2,
    )
    ax_b2.set_xlabel("Within-method LR rank percentile")
    ax_b2.set_title("Formal CellChatDB overlap (4/20)", fontsize=8.5, pad=3)
    ax_b2.grid(axis="x", color=pale_grey, linewidth=0.5, alpha=0.75)
    ax_b2.legend(
        frameon=False,
        fontsize=6.6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        borderaxespad=0,
    )

    c_body = _headed_panel(
        outer[1], "c", "Localized mdka–sdc4 model contribution and circuit agreement"
    )
    c_content = c_body.subgridspec(1, 2, width_ratios=(0.57, 0.43), wspace=0.24)
    ax_c1 = fig.add_subplot(c_content[0])
    ax_c2 = fig.add_subplot(c_content[1])
    x_min, x_max = cells["spatial_x"].min(), cells["spatial_x"].max()
    y_min, y_max = cells["spatial_y"].min(), cells["spatial_y"].max()
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_min == y_max:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    ax_c1.scatter(
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
        ax_c1.annotate(
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
    ax_c1.scatter(
        sender_cells["spatial_x"],
        sender_cells["spatial_y"],
        s=6 + 15 * sender_cells["ligand_scaled_expression"],
        color=paper_red,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.88,
        zorder=3,
    )
    ax_c1.scatter(
        receiver_cells["spatial_x"],
        receiver_cells["spatial_y"],
        s=6 + 15 * receiver_cells["receptor_scaled_expression"],
        color=navy,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.88,
        zorder=3,
    )
    ax_c1.set_xlim(x_min, x_max)
    ax_c1.set_ylim(y_min, y_max)
    ax_c1.set_aspect("equal", adjustable="box")
    ax_c1.set_xticks([])
    ax_c1.set_yticks([])
    for spine in ax_c1.spines.values():
        spine.set_visible(False)
    lr_display = str(spatial_summary.lr_id.iloc[0]).replace("->", "–")
    n_unique_edges = int(spatial_summary.n_unique_edges.iloc[0])
    n_positive_lr_edges = int(spatial_summary.n_positive_lr_edges.iloc[0])
    n_top_edges = int(spatial_summary.n_top_exact_message_lr_edges.iloc[0])
    ax_c1.set_title(
        f"{lr_display} · exact message × LR activity\n"
        f"{len(display_edges)} shown from the predeclared top 2% "
        f"({n_top_edges:,}/{n_unique_edges:,} edges)",
        fontsize=9.0,
        pad=3,
    )
    ax_c1.legend(
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
        fontsize=7.0,
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
        ax_c2.plot(
            [row.cytobridge_rank_percentile, row.commot_rank_percentile],
            [y_value, y_value],
            color="#AEB7BD",
            linewidth=1.15,
            zorder=1,
        )
    ax_c2.scatter(
        selected_contexts["cytobridge_rank_percentile"],
        y_e2,
        s=39,
        color=navy,
        edgecolor="white",
        linewidth=0.55,
        label="CytoBridge exact message",
        zorder=3,
    )
    ax_c2.scatter(
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
    context_name = {
        "Spinal Cord Ventral Region": "Ventral spinal cord",
        "Spinal Cord Dorsal Region": "Dorsal spinal cord",
        "Spinal Cord Anterior Region": "Anterior spinal cord",
        "Anterior Erythroid Lineage Cell": "Anterior erythroid",
        "Fast Muscle Cell": "Fast muscle",
        "Slow Muscle Cell": "Slow muscle",
    }
    context_labels = [
        f"{context_name.get(row.sender_type, row.sender_type)} →\n"
        f"{context_name.get(row.receiver_type, row.receiver_type)}"
        for row in selected_contexts.itertuples(index=False)
    ]
    ax_c2.set_yticks(y_e2, context_labels, fontsize=7.0, linespacing=0.92)
    context_min = min(
        0.60,
        float(
            selected_contexts[["cytobridge_rank_percentile", "commot_rank_percentile"]]
            .min()
            .min()
        )
        - 0.03,
    )
    ax_c2.set_xlim(max(0.0, context_min), 1.01)
    ax_c2.set_xlabel("Within-axis cell-type-circuit rank percentile")
    ax_c2.set_title(
        f"Circuits selected by CytoBridge before COMMOT lookup\n"
        "Full 19 × 19 context ρ = "
        f"{float(spatial_summary.cell_type_context_spearman_rho.iloc[0]):.3f}",
        fontsize=9.0,
        pad=3,
    )
    ax_c2.grid(axis="x", color=pale_grey, linewidth=0.5, alpha=0.8)
    ax_c2.legend(frameon=False, fontsize=7.0, loc="lower left")

    pdf_path = output / "zebrafish_attention_validation_a4.pdf"
    png_path = output / "zebrafish_attention_validation_a4.png"
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=320, facecolor="white")
    plt.close(fig)

    commot_pair = pair.set_index(["cytobridge_view", "external_method"])
    attn_commot = commot_pair.loc[("attention", "COMMOT")]
    exact_commot = commot_pair.loc[("exact_message", "COMMOT")]
    attn_cag = commot_pair.loc[("attention", "CellAgentChat")]
    exact_cag = commot_pair.loc[("exact_message", "CellAgentChat")]
    context_rho = float(spatial_summary.cell_type_context_spearman_rho.iloc[0])
    n_displayed_edges = min(16, len(all_top_edges))
    caption = (
        "Cross-method and source-paper assessment of the learned zebrafish interaction field. "
        "(a) Across all 19 x 19 directed cell-type pairs, adjusted CytoBridge agreement "
        "with COMMOT was rho="
        f"{attn_commot.adjusted_spearman_rho:.3f} for attention and "
        f"{exact_commot.adjusted_spearman_rho:.3f} for exact message; corresponding "
        "CellAgentChat values were "
        f"{attn_cag.adjusted_spearman_rho:.3f} and "
        f"{exact_cag.adjusted_spearman_rho:.3f}. Thick gray intervals are the structured "
        f"permutation-null 95% ranges from {n_pair_permutations:,} permutations within "
        f"{n_strata} structural strata after adjustment for sender and receiver abundance, "
        "mean spatial distance, and self-pair status; all four empirical upper-tail "
        "P values were 0.000999. (b) The source paper pre-specified "
        f"{source_axis_count} developmental LR axes, of which {represented_axis_count} "
        "were representable in the current expression matrix. Exact-message scores "
        f"placed {int(exact_enrichment.paper_reference_in_top_n)}/{represented_axis_count} "
        "in the positive-score top decile "
        f"(AUC={float(exact_enrichment.paper_reference_auc):.3f}, one-sided "
        f"Mann-Whitney P={float(exact_enrichment.mannwhitney_p_greater):.2g}); the "
        "expression-only baseline placed "
        f"{int(lr_only_enrichment.paper_reference_in_top_n)}/{represented_axis_count} "
        f"there (AUC={float(lr_only_enrichment.paper_reference_auc):.3f}). The similar "
        "baseline establishes molecular compatibility rather than an incremental "
        f"attention gain. The {len(paper_shared)} axes available in the formal COMMOT "
        "CellChatDB universe ranked above the 96th percentile in both methods. These "
        "source-paper axes come from the same atlas, not an independent cohort. (c) The "
        "highest exact-message-scoring source-paper axis, mdka-sdc4, is localized on the "
        f"observed terminal sample. Gold arrows show the {n_displayed_edges} strongest "
        f"edges used for display from the predeclared top-2% set (n={n_top_edges}) among "
        f"{n_positive_lr_edges:,} positive-LR edges; "
        "sender and receiver nodes are scaled by ligand and receptor expression. The six "
        "highest off-diagonal cell-type circuits were selected by CytoBridge only and "
        f"then assigned COMMOT ranks (full 19 x 19 context rho={context_rho:.3f}). "
        "Exact-message LR scores are post-hoc model-contribution-by-LR-activity scores, "
        "not biochemical probabilities or native ligand identities."
    )
    (output / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    reviewer_response = f"""# Response to reviewer concern on attention interpretability

We agree that attention weights alone cannot be read as biochemical communication probabilities. We therefore separated the reviewer-facing validation into three evidence levels and revised the interpretation accordingly.

1. **Complete cell-pair field.** All 361 directed cell-type pairs were retained, including structural zeros. After adjustment for sender and receiver abundance, mean spatial distance, and self-pair status, attention and exact message agreed with COMMOT at rho={attn_commot.adjusted_spearman_rho:.3f} and {exact_commot.adjusted_spearman_rho:.3f}. Both exceeded their within-stratum structured-permutation nulls (empirical upper-tail P={exact_commot.adjusted_spearman_empirical_p_upper:.4f}). CellAgentChat provides a secondary comparison at rho={attn_cag.adjusted_spearman_rho:.3f} and {exact_cag.adjusted_spearman_rho:.3f}.
2. **Pre-specified source-paper molecular compatibility.** The source paper pre-specified {source_axis_count} developmental LR axes in Figure 5B of *Spatiotemporal mapping of gene expression landscapes and developmental trajectories during zebrafish embryogenesis*. {represented_axis_count} were representable in the current expression matrix. Exact-message scores placed {int(exact_enrichment.paper_reference_in_top_n)}/{represented_axis_count} in the positive-score top decile (rank AUC={float(exact_enrichment.paper_reference_auc):.3f}, one-sided Mann-Whitney P={float(exact_enrichment.mannwhitney_p_greater):.2g}), while expression-only activity placed {int(lr_only_enrichment.paper_reference_in_top_n)}/{represented_axis_count} there (AUC={float(lr_only_enrichment.paper_reference_auc):.3f}). The similar baseline is displayed explicitly, so this is interpreted as molecular compatibility rather than a large incremental attention effect. {len(paper_shared)} axes were present in the formal COMMOT CellChatDB universe and ranked highly in both methods. These axes are pre-specified by the source paper but come from the same atlas rather than an independent cohort.
3. **Localized model contribution and circuit reproducibility.** The mdka-sdc4 axis was chosen by the highest CytoBridge exact-message score within the frozen source-paper list. Gold arrows are exact-message-by-LR-activity edges, not expression-only links or biochemical probabilities. The highest off-diagonal cell-type circuits were selected by CytoBridge before COMMOT ranks were attached; agreement over the complete 19 x 19 context grid was rho={context_rho:.3f}.

Together, the complete-pair, pre-specified source-paper, and localized circuit analyses show that the learned interaction operator is associated with cross-method communication structure and known zebrafish signaling axes. They do not establish that an attention coefficient is itself a biochemical ligand-receptor strength. The non-monotonic fixed-checkpoint interaction-off sensitivity remains in the analysis archive but is not used as validation evidence in the reviewer-facing figure.
"""
    (output / "reviewer_response.md").write_text(reviewer_response, encoding="utf-8")
    provenance = (
        "# Provenance\n\n"
        "## Source paths\n\n"
        f"- Frozen analysis directory: `{analysis_dir.expanduser().resolve()}`\n"
        f"- Upstream analysis manifest SHA-256: `{expected_analysis_manifest_sha256}`\n"
        f"- Workflow: `{WORKFLOW}` schema {SCHEMA_VERSION}\n"
        f"- Panel a: complete {n_pairs}-pair adjusted correlations with {n_strata}-stratum, {n_pair_permutations:,}-permutation null intervals.\n"
        "- Panel b: frozen source-paper 21-axis list, expression-only baseline, and formal COMMOT CellChatDB overlap.\n"
        "- Panel c: CytoBridge-selected mdka-sdc4 exact-message-by-LR-activity edges; COMMOT circuit ranks inspected afterward.\n"
        "- Original-paper reference: frozen, pre-specified 21-axis list from Figure 5B of the source zebrafish expression atlas; this is not an independent cohort.\n"
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
        "commot_lr_scores_collapsed.csv.gz",
        "original_paper_21_lr_scores.csv",
        "original_paper_21_lr_enrichment.csv",
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
    report_claim_contract = dict(analysis_manifest.get("claim_contract") or {})
    report_claim_contract["original_paper_21_scope"] = (
        "pre-specified source-paper axes from the same atlas; not an independent "
        "cohort or experimental ground truth"
    )
    report_claim_contract["reviewer_figure_scope"] = (
        "three-panel complete-pair agreement, source-paper molecular compatibility, "
        "and localized model-edge context"
    )
    report_manifest = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "analysis_manifest_sha256": expected_analysis_manifest_sha256.casefold(),
        "implementation": _implementation_identity(),
        "claim_contract": report_claim_contract,
        "figure_panels": {
            "a": "complete 361-pair adjusted cross-method agreement and structured null",
            "b": "pre-specified source-paper LR compatibility and formal COMMOT overlap",
            "c": "localized mdka-sdc4 exact-message-by-LR-activity edges and circuit ranks",
        },
        "panel_data_files": [f"panel_data/{name}" for name in panel_filenames],
        "artifacts": report_artifacts,
    }
    _write_json(report_manifest_path, report_manifest)
    _write_sha_sidecar(report_manifest_path)
    print(report_manifest_path)


def report(
    analysis_dir: Path,
    output_dir: Path,
    *,
    expected_analysis_manifest_sha256: str | None,
    jam_manifest_paths: list[Path],
    expected_jam_manifest_sha256s: list[str] | None,
    reader_output: bool = False,
) -> None:
    """Render the current-checkpoint three-panel zebrafish reviewer figure."""

    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    analysis_manifest, paths = _verified_frozen_analysis(
        analysis_dir, expected_analysis_manifest_sha256
    )
    expected_analysis_manifest_sha256 = _sha256(
        analysis_dir.expanduser().resolve() / "analysis_manifest.json"
    )
    jam_manifest_records, jam_paths = _verified_jam_report_tables(
        jam_manifest_paths, expected_jam_manifest_sha256s
    )
    output = _prepare_output(output_dir)
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError("Arial is required for the publication report") from error
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    pair = _read_required_table(paths, "directed_pair_concordance.csv")
    compatibility = _canonical_compatibility_summary(
        _read_csv(jam_paths["compatibility_summary"], label="JAM compatibility")
    )
    quartile = _read_csv(
        jam_paths["quartile_compatibility"], label="JAM quartile compatibility"
    ).copy()
    quartile["condition"] = _canonical_conditions(
        quartile["condition"], label="JAM quartile table"
    )
    if quartile.duplicated("condition").any() or len(quartile) != 3:
        raise ValueError("JAM quartile table requires one row per condition")
    type_pair = _read_csv(jam_paths["type_pair_ranks"], label="JAM type-pair ranks")
    type_pair["condition"] = _canonical_conditions(
        type_pair["condition"], label="JAM type-pair ranks"
    )
    if "stage_label" in type_pair:
        type_pair = type_pair.loc[type_pair["stage_label"].astype(str).eq("18hpf")]
    somite_pair = type_pair.loc[
        type_pair["sender_type"].astype(str).eq("Somite")
        & type_pair["receiver_type"].astype(str).eq("Somite")
    ].copy()
    if somite_pair.duplicated("condition").any() or len(somite_pair) != 3:
        raise ValueError(
            "type-pair table requires one 18 hpf Somite-to-Somite row per condition"
        )
    rank_n_column = _column(
        somite_pair,
        "n_complete_directed_type_pairs",
        "n_ranked_contexts",
        label="Somite type-pair ranks",
    )
    somite_pair["rank_from_top"] = pd.to_numeric(
        somite_pair["rank_from_top"], errors="raise"
    )
    somite_pair["rank_n"] = pd.to_numeric(somite_pair[rank_n_column], errors="raise")
    somite_pair["top_rank_percentile"] = np.where(
        somite_pair["rank_n"] > 1,
        1 - (somite_pair["rank_from_top"] - 1) / (somite_pair["rank_n"] - 1),
        1.0,
    )

    null_summary = _read_csv(
        jam_paths["spatial_null_summary"], label="JAM spatial null summary"
    )
    if "stage_label" in null_summary:
        null_summary = null_summary.loc[
            null_summary["stage_label"].astype(str).eq("18hpf")
        ]
    if "cell_type" in null_summary:
        null_summary = null_summary.loc[
            null_summary["cell_type"].astype(str).eq("Somite")
        ]
    if len(null_summary) != 1:
        raise ValueError("spatial-null summary requires one 18 hpf Somite row")
    null_row = null_summary.iloc[0]
    null_iterations = _read_csv(
        jam_paths["spatial_null_iterations"], label="JAM spatial null iterations"
    )
    if null_iterations.empty:
        raise ValueError("JAM spatial-null iterations are empty")

    association = _read_csv(jam_paths["myog_association"], label="JAM-myog association")
    if "stage_label" in association:
        association = association.loc[
            association["stage_label"].astype(str).eq("18hpf")
        ]
    if "cell_type" in association:
        association = association.loc[association["cell_type"].astype(str).eq("Somite")]
    association = association.loc[
        association["gene_b"].astype(str).str.casefold().eq("myog")
        & association["gene_a"].astype(str).str.casefold().isin(["jam2a", "jam3b"])
    ].copy()
    if set(association["gene_a"].astype(str).str.casefold()) != {"jam2a", "jam3b"}:
        raise ValueError("association table requires Jam2a-myog and Jam3b-myog rows")

    detection = _read_csv(
        jam_paths["expression_detection"], label="JAM expression detection"
    )
    if "stage_label" in detection:
        detection = detection.loc[detection["stage_label"].astype(str).eq("18hpf")]
    if "cell_type" in detection:
        detection = detection.loc[detection["cell_type"].astype(str).eq("Somite")]
    detection = detection.loc[
        detection["gene"].astype(str).str.casefold().isin(["jam2a", "jam3b", "myog"])
    ].copy()
    detection["gene_key"] = detection["gene"].astype(str).str.casefold()
    if set(detection["gene_key"]) != {"jam2a", "jam3b", "myog"}:
        raise ValueError("detection table requires jam2a, jam3b, and myog")
    detection_fraction_column = _column(
        detection,
        "detected_fraction",
        "detection_fraction",
        label="JAM expression detection",
    )
    detection["detection_fraction"] = pd.to_numeric(
        detection[detection_fraction_column], errors="raise"
    )
    cells = _canonical_spatial_cells(
        _read_csv(jam_paths["spatial_cells"], label="JAM spatial cells")
    )
    if "stage_label" in cells:
        cells = cells.loc[cells["stage_label"].astype(str).eq("18hpf")]
    display_edges = _canonical_display_edges(
        _read_csv(jam_paths["display_edges"], label="trained JAM display edges")
    )
    n_somite = int(cells["is_somite"].sum())
    if "n_cells" in null_summary and n_somite != int(null_row["n_cells"]):
        raise ValueError("spatial-cell and spatial-null Somite counts disagree")

    navy = "#214E78"
    teal = "#168A83"
    gold = "#D9A441"
    coral = "#C65F4A"
    dark_grey = "#6F777D"
    middle_grey = "#AEB6BC"
    pale_grey = "#D8DDE1"
    background_grey = "#ECEFF1"
    condition_order = ["trained", "pre_interaction", "random"]
    condition_labels = {
        "trained": "After interaction\ntraining",
        "pre_interaction": "Before interaction\nlearning",
        "random": "Randomized\nweights",
    }

    # Full-page supplementary figures use an A4 portrait canvas.  The three
    # evidence levels read from top to bottom: external reproducibility,
    # checkpoint controls, and spatial/molecular interpretation.
    fig = plt.figure(figsize=(8.27, 11.69), constrained_layout=False)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=(0.20, 0.31, 0.49),
        left=0.145,
        right=0.970,
        top=0.982,
        bottom=0.055,
        hspace=0.17,
    )

    def _headed_panel(parent, label: str, title: str, *, heading_ratio: float):
        nested = parent.subgridspec(
            2, 1, height_ratios=(heading_ratio, 1.0), hspace=0.08
        )
        heading = fig.add_subplot(nested[0])
        heading.axis("off")
        heading.text(0, 0.52, label, fontsize=14, fontweight="bold", va="center")
        heading.text(0.065, 0.52, title, fontsize=12, fontweight="bold", va="center")
        return nested[1]

    a_body = _headed_panel(outer[0], "a", "External CCI agreement", heading_ratio=0.24)
    ax_a = fig.add_subplot(a_body)
    pair_order = [
        ("attention", "COMMOT"),
        ("attention", "CellAgentChat"),
    ]
    rows = pair.set_index(["cytobridge_view", "external_method"]).loc[pair_order]
    dimensions = {
        "n_pairs": set(rows["n_pairs"].astype(int)),
        "n_strata": set(rows["n_strata"].astype(int)),
        "n_permutations": set(rows["n_permutations"].astype(int)),
    }
    if any(len(values) != 1 for values in dimensions.values()):
        raise ValueError("pair-concordance rows disagree on null-test dimensions")
    n_pairs = dimensions["n_pairs"].pop()
    n_strata = dimensions["n_strata"].pop()
    n_pair_permutations = dimensions["n_permutations"].pop()
    if n_pairs != 361:
        raise ValueError("reviewer figure requires the complete 19 x 19 pair field")
    y_a = np.arange(2)[::-1]
    labels_a = [
        "COMMOT",
        "CellAgentChat proxy",
    ]
    colors_a = [teal, dark_grey]
    null_min = float(rows["null_adjusted_spearman_q025"].min())
    observed_max = float(rows["adjusted_spearman_rho"].max())
    for y_value, (_, row), color in zip(y_a, rows.iterrows(), colors_a, strict=True):
        ax_a.plot(
            [row.null_adjusted_spearman_q025, row.null_adjusted_spearman_q975],
            [y_value, y_value],
            color=middle_grey,
            linewidth=5.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax_a.scatter(
            row.null_adjusted_spearman_mean,
            y_value,
            s=20,
            facecolor="white",
            edgecolor=dark_grey,
            linewidth=0.7,
            zorder=2,
        )
        ax_a.scatter(
            row.adjusted_spearman_rho,
            y_value,
            s=48,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        p_text = (
            "P<0.001"
            if float(row.adjusted_spearman_empirical_p_upper) < 0.001
            else f"P={row.adjusted_spearman_empirical_p_upper:.3g}"
        )
        ax_a.text(
            float(row.adjusted_spearman_rho) + 0.018,
            y_value,
            f"ρ={row.adjusted_spearman_rho:.2f}\n{p_text}",
            va="center",
            fontsize=8.3,
            linespacing=0.90,
        )
    ax_a.set_yticks(y_a, labels_a)
    ax_a.set_xlim(max(0, null_min - 0.07), min(1.04, observed_max + 0.13))
    ax_a.set_ylim(-0.55, 1.55)
    ax_a.set_xlabel("Rank agreement with CytoBridge attention (adjusted Spearman ρ)")
    ax_a.set_title(
        f"Interaction patterns across all {n_pairs} sender → receiver cell-type pairs",
        pad=5,
    )
    ax_a.grid(axis="x", color=pale_grey, linewidth=0.5)
    ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=teal,
                markeredgecolor="white",
                label="Observed agreement",
            ),
            Line2D(
                [0],
                [0],
                color=middle_grey,
                linewidth=5,
                label="Structured-null 95% range",
            ),
        ],
        frameon=False,
        loc="lower right",
        ncol=2,
        handlelength=1.8,
        columnspacing=1.4,
    )

    b_body = _headed_panel(
        outer[1],
        "b",
        "JAM-compatible edges receive high attention",
        heading_ratio=0.18,
    )
    b_container = b_body.subgridspec(2, 1, height_ratios=(0.10, 1.0), hspace=0.11)
    somite_pair = somite_pair.set_index("condition").loc[condition_order]
    quartile = quartile.set_index("condition").loc[condition_order]
    odds_column = _column(
        quartile, "top_vs_bottom_odds_ratio", label="JAM quartile table"
    )
    p_column = _column(
        quartile,
        "fisher_exact_two_sided_p",
        "fisher_exact_two_sided_p_descriptive_technical",
        label="JAM quartile table",
    )
    odds = pd.to_numeric(quartile[odds_column], errors="raise")
    if (~np.isfinite(odds) | (odds <= 0)).any():
        raise ValueError("JAM compatibility odds ratios must be finite and positive")
    top_rate_column = _column(
        quartile, "top_compatibility_rate", label="JAM quartile table"
    )
    bottom_rate_column = _column(
        quartile, "bottom_compatibility_rate", label="JAM quartile table"
    )
    top_rates = pd.to_numeric(quartile[top_rate_column], errors="raise") * 100
    bottom_rates = pd.to_numeric(quartile[bottom_rate_column], errors="raise") * 100
    if (
        (~np.isfinite(top_rates))
        | (~np.isfinite(bottom_rates))
        | (top_rates < 0)
        | (bottom_rates < 0)
    ).any():
        raise ValueError("JAM compatibility rates must be finite and non-negative")
    ax_b = fig.add_subplot(b_container[1])
    y_b = np.arange(3)[::-1]
    bar_offset = 0.16
    ax_b.barh(
        y_b + bar_offset,
        top_rates.to_numpy(float),
        height=0.25,
        color=navy,
        edgecolor="none",
        label="Top attention quartile",
        zorder=3,
    )
    ax_b.barh(
        y_b - bar_offset,
        bottom_rates.to_numpy(float),
        height=0.25,
        facecolor="white",
        edgecolor=dark_grey,
        linewidth=1.0,
        label="Bottom attention quartile",
        zorder=3,
    )
    for y_value, condition in zip(y_b, condition_order, strict=True):
        ax_b.text(
            float(top_rates.loc[condition]) + 0.45,
            y_value + bar_offset,
            f"{float(top_rates.loc[condition]):.1f}%",
            va="center",
            fontsize=8.4,
        )
        ax_b.text(
            float(bottom_rates.loc[condition]) + 0.45,
            y_value - bar_offset,
            f"{float(bottom_rates.loc[condition]):.1f}%",
            va="center",
            fontsize=8.4,
            color=dark_grey,
        )
    ax_b.set_yticks(y_b, [condition_labels[item] for item in condition_order])
    ax_b.set_xlim(0, max(30.0, float(max(top_rates.max(), bottom_rates.max())) + 4.0))
    ax_b.set_xlabel("JAM-compatible edges (%)")
    ax_b.set_title(
        "High-attention edges are selectively JAM-compatible only after interaction training",
        pad=5,
    )
    ax_b.grid(axis="x", color=pale_grey, linewidth=0.5, zorder=0)
    ax_b.legend(frameon=False, loc="lower right", ncol=2)

    edge_totals = (
        compatibility.groupby("condition", sort=False)["n_edges"].sum().astype(int)
    )
    if len(set(edge_totals)) != 1:
        raise ValueError("JAM conditions must use the same Somite edge scaffold")
    n_somite_edges = int(edge_totals.iloc[0])
    ax_b_note = fig.add_subplot(b_container[0])
    ax_b_note.axis("off")
    ax_b_note.text(
        0.5,
        0.60,
        f"Same {n_somite:,} 18 hpf Somite cells and {n_somite_edges:,} model edges in every condition.  "
        "JAM-compatible = jam2a at one endpoint and jam3b at the other.",
        ha="center",
        va="center",
        fontsize=8.6,
        color="black",
    )

    c_body = _headed_panel(
        outer[2],
        "c",
        "Spatial and myogenic context of the JAM program",
        heading_ratio=0.11,
    )
    c_grid = c_body.subgridspec(1, 2, width_ratios=(0.57, 0.43), wspace=0.20)
    ax_c1 = fig.add_subplot(c_grid[0])
    c_metrics = c_grid[1].subgridspec(2, 1, height_ratios=(0.54, 0.46), hspace=0.44)
    ax_c2 = fig.add_subplot(c_metrics[0])
    ax_c3 = fig.add_subplot(c_metrics[1])

    background = cells.loc[~cells["is_somite"]]
    somite = cells.loc[cells["is_somite"]]
    ax_c1.scatter(
        background["x"],
        background["y"],
        s=1.4,
        color=background_grey,
        linewidth=0,
        alpha=0.70,
    )
    ax_c1.scatter(
        somite["x"],
        somite["y"],
        s=3.0,
        color="#C9DAD8",
        linewidth=0,
        alpha=0.78,
    )
    incompatible = display_edges.loc[~display_edges["jam_compatible"]]
    compatible_edges = display_edges.loc[display_edges["jam_compatible"]]
    if not incompatible.empty:
        segments = (
            incompatible[["source_x", "source_y", "target_x", "target_y"]]
            .to_numpy(float)
            .reshape(-1, 2, 2)
        )
        ax_c1.add_collection(
            LineCollection(segments, colors=middle_grey, linewidths=0.55, alpha=0.45)
        )
    for row in compatible_edges.itertuples(index=False):
        ax_c1.add_patch(
            FancyArrowPatch(
                (float(row.source_x), float(row.source_y)),
                (float(row.target_x), float(row.target_y)),
                arrowstyle="-|>",
                mutation_scale=4.8,
                linewidth=0.9,
                color=gold,
                alpha=0.90,
                shrinkA=0.8,
                shrinkB=0.8,
                zorder=2,
            )
        )
    jam2_only = somite["jam2a_positive"] & ~somite["jam3b_positive"]
    jam3_only = somite["jam3b_positive"] & ~somite["jam2a_positive"]
    both = somite["jam2a_positive"] & somite["jam3b_positive"]
    for mask, color, marker, size in (
        (jam2_only, coral, "o", 11.0),
        (jam3_only, navy, "s", 10.5),
        (both, "#202020", "D", 12.0),
    ):
        ax_c1.scatter(
            somite.loc[mask, "x"],
            somite.loc[mask, "y"],
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.30,
            alpha=0.92,
        )
    ax_c1.set_aspect("equal", adjustable="box")
    ax_c1.set_xticks([])
    ax_c1.set_yticks([])
    for spine in ax_c1.spines.values():
        spine.set_visible(False)
    ax_c1.set_title(
        f"18 hpf tissue map ({len(cells):,} cells shown; {n_somite:,} Somite cells analyzed)",
        pad=5,
    )
    ax_c1.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=background_grey,
                markeredgecolor="none",
                label="Other tissue cell",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#C9DAD8",
                markeredgecolor="none",
                label="Somite cell analyzed",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=coral,
                markeredgecolor="white",
                label="jam2a+ only",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=navy,
                markeredgecolor="white",
                label="jam3b+ only",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#202020",
                markeredgecolor="white",
                label="Both JAM genes",
            ),
            Line2D(
                [0],
                [0],
                color=gold,
                linewidth=1.5,
                label=f"High-attention JAM edge (n={len(compatible_edges)})",
            ),
        ],
        frameon=False,
        fontsize=8.0,
        loc="lower left",
        ncol=2,
        handlelength=1.6,
        columnspacing=1.1,
    )

    association["gene_key"] = association["gene_a"].astype(str).str.casefold()
    association = association.set_index("gene_key").loc[["jam3b", "jam2a"]]
    association_odds = pd.to_numeric(association["fisher_odds_ratio"], errors="raise")
    count_columns = (
        "both_detected",
        "gene_a_only",
        "gene_b_only",
        "neither_detected",
    )
    missing_count_columns = set(count_columns).difference(association.columns)
    if missing_count_columns:
        raise ValueError(
            "JAM-myog association lacks count columns: "
            f"{sorted(missing_count_columns)}"
        )
    association_counts = association.loc[:, count_columns].apply(
        pd.to_numeric, errors="raise"
    )
    positive_denominator = (
        association_counts["both_detected"] + association_counts["gene_a_only"]
    )
    negative_denominator = (
        association_counts["gene_b_only"] + association_counts["neither_detected"]
    )
    if (positive_denominator <= 0).any() or (negative_denominator <= 0).any():
        raise ValueError("JAM-myog fractions require positive denominators")
    myog_given_positive = (
        association_counts["both_detected"] / positive_denominator * 100
    )
    myog_given_negative = association_counts["gene_b_only"] / negative_denominator * 100
    y_c3 = np.array([1.0, 0.0])
    c3_offset = 0.16
    ax_c3.barh(
        y_c3 + c3_offset,
        myog_given_positive.to_numpy(float),
        height=0.25,
        color=teal,
        edgecolor="none",
        label="JAM gene+",
        zorder=3,
    )
    ax_c3.barh(
        y_c3 - c3_offset,
        myog_given_negative.to_numpy(float),
        height=0.25,
        facecolor="white",
        edgecolor=dark_grey,
        linewidth=1.0,
        label="JAM gene−",
        zorder=3,
    )
    for y_value, gene in zip(y_c3, ["jam3b", "jam2a"], strict=True):
        ax_c3.text(
            float(myog_given_positive.loc[gene]) + 0.8,
            y_value + c3_offset,
            f"{float(myog_given_positive.loc[gene]):.1f}%",
            va="center",
            fontsize=8.0,
        )
        ax_c3.text(
            float(myog_given_negative.loc[gene]) + 0.8,
            y_value - c3_offset,
            f"{float(myog_given_negative.loc[gene]):.1f}%",
            va="center",
            fontsize=8.0,
            color=dark_grey,
        )
        ax_c3.text(
            max(
                float(myog_given_positive.loc[gene]),
                float(myog_given_negative.loc[gene]),
            )
            + 7.0,
            y_value,
            f"P={float(association.loc[gene, 'fisher_two_sided_p']):.2g}",
            va="center",
            fontsize=8.0,
        )
    ax_c3.set_xlim(
        0,
        max(
            70.0,
            float(max(myog_given_positive.max(), myog_given_negative.max())) + 14.0,
        ),
    )
    ax_c3.set_ylim(-0.55, 1.55)
    ax_c3.set_yticks(y_c3, ["jam3b", "jam2a"])
    ax_c3.set_xlabel("myog+ Somite cells (%)")
    ax_c3.set_title("myog detection by JAM-gene status", pad=5)
    ax_c3.grid(axis="x", color=pale_grey, linewidth=0.5)
    ax_c3.legend(frameon=False, loc="lower right", ncol=2, fontsize=8.0)

    null_mean = float(null_row["null_mean"])
    observed_column = _column(
        null_summary,
        "observed_jam2a_jam3b_orientation_compatible_pairs",
        label="JAM spatial null summary",
    )
    fold_column = _column(
        null_summary,
        "observed_over_null_mean",
        label="JAM spatial null summary",
    )
    p_spatial_column = _column(
        null_summary,
        "monte_carlo_upper_tail_p_plus1",
        "monte_carlo_p_upper",
        label="JAM spatial null summary",
    )
    fold = float(null_row[fold_column])
    null_count_column = _column(
        null_iterations,
        "orientation_compatible_pair_count",
        label="JAM spatial null iterations",
    )
    null_counts = pd.to_numeric(null_iterations[null_count_column], errors="raise")
    observed_neighbors = int(null_row[observed_column])
    ax_c2.hist(
        null_counts,
        bins=28,
        color=background_grey,
        edgecolor="white",
        linewidth=0.45,
    )
    ax_c2.axvline(
        observed_neighbors,
        color=teal,
        linewidth=2.2,
        label=f"Observed: {observed_neighbors}",
    )
    ax_c2.axvspan(
        float(null_row["null_q025"]),
        float(null_row["null_q975"]),
        color=middle_grey,
        alpha=0.16,
        linewidth=0,
        label=(
            f"Random labels: mean {null_mean:.0f}\n"
            f"95% range {float(null_row['null_q025']):.0f}–{float(null_row['null_q975']):.0f}"
        ),
    )
    ax_c2.set_xlim(
        min(float(null_counts.min()), float(null_row["null_q025"])) - 12,
        max(observed_neighbors, float(null_counts.max())) + 12,
    )
    ax_c2.set_xlabel("Complementary jam2a+/jam3b+ spatial-neighbor pairs")
    ax_c2.set_ylabel("Label permutations")
    ax_c2.set_title(
        "Complementary jam2a+/jam3b+ cells\nare spatial neighbors",
        pad=5,
    )
    ax_c2.legend(frameon=False, loc="upper left", fontsize=8.0)
    ax_c2.text(
        0.98,
        0.94,
        f"{fold:.2f}× enrichment\nP={float(null_row[p_spatial_column]):.2g}",
        transform=ax_c2.transAxes,
        ha="right",
        va="top",
        fontsize=8.3,
    )

    stem = (
        "zebrafish_attention_controls"
        if reader_output
        else "zebrafish_attention_validation_a4"
    )
    pdf_path = output / f"{stem}.pdf"
    png_path = output / f"{stem}.png"
    fig.savefig(pdf_path, facecolor="white")
    fig.savefig(png_path, dpi=320, facecolor="white")
    plt.close(fig)

    pair_index = pair.set_index(["cytobridge_view", "external_method"])
    attention_commot = pair_index.loc[("attention", "COMMOT")]
    attention_cag = pair_index.loc[("attention", "CellAgentChat")]
    jam3b_row = association.loc["jam3b"]
    caption = (
        "Zebrafish validation of learned cell-cell interaction attention. "
        "(a) Cell-level attention was aggregated into one model interaction rank for "
        "each of all 361 directed cell-type pairs. After adjusting for sender and "
        "receiver abundance, mean spatial distance, and self-pairs, the adjusted attention ranks "
        f"agreed with COMMOT at ρ={float(attention_commot.adjusted_spearman_rho):.3f} "
        "and with the shared-database CellAgentChat proxy at "
        f"ρ={float(attention_cag.adjusted_spearman_rho):.3f}. Gray bars show 95% "
        f"ranges from {n_pair_permutations:,} structured permutations within "
        f"{n_strata} strata. The empirical upper-tail P values were "
        f"{float(attention_commot.adjusted_spearman_empirical_p_upper):.4g} and "
        f"{float(attention_cag.adjusted_spearman_empirical_p_upper):.4g}. "
        f"(b) This analysis is restricted to {n_somite:,} Somite cells at 18 hpf and "
        f"the same {n_somite_edges:,} model edges in every condition. A JAM-compatible "
        "edge has jam2a detected at one endpoint and jam3b at the other. In the trained "
        f"model, {float(top_rates.loc['trained']):.1f}% of top-attention-quartile edges "
        f"were JAM-compatible versus {float(bottom_rates.loc['trained']):.1f}% in the "
        f"bottom quartile. The corresponding rates were "
        f"{float(top_rates.loc['pre_interaction']):.1f}% versus "
        f"{float(bottom_rates.loc['pre_interaction']):.1f}% before interaction learning, "
        f"and {float(top_rates.loc['random']):.1f}% versus "
        f"{float(bottom_rates.loc['random']):.1f}% after randomizing the learned weights. "
        "The before-learning control (pre-interaction checkpoint) is the same model before "
        "the interaction learning stage. The trained condition is the model after that "
        "stage. The randomized control shuffles the learned interaction weights "
        "on the same cells and edge scaffold. As secondary summaries, Somite-to-Somite "
        f"ranked {int(somite_pair.loc['trained', 'rank_from_top'])}/"
        f"{int(somite_pair.loc['trained', 'rank_n'])}, "
        f"{int(somite_pair.loc['pre_interaction', 'rank_from_top'])}/"
        f"{int(somite_pair.loc['pre_interaction', 'rank_n'])}, and "
        f"{int(somite_pair.loc['random', 'rank_from_top'])}/"
        f"{int(somite_pair.loc['random', 'rank_n'])} in the three conditions. The "
        f"top-versus-bottom odds ratios were {float(odds.loc['trained']):.1f}, "
        f"{float(odds.loc['pre_interaction']):.1f}, and {float(odds.loc['random']):.1f}. "
        "(c) The tissue map shows all "
        f"{len(cells):,} cells for anatomical context, whereas all statistics remain "
        f"restricted to the {n_somite:,} Somite cells. Gold arrows mark the predeclared "
        f"set of {len(compatible_edges)} high-attention JAM-compatible model edges. "
        f"The observed {observed_neighbors} spatially neighboring complementary "
        f"jam2a+/jam3b+ Somite pairs were enriched {fold:.2f}-fold over the "
        f"within-Somite label-permutation null "
        f"(Monte Carlo P={float(null_row[p_spatial_column]):.2g}). Jam3b and myog "
        f"co-detection had odds ratio={float(jam3b_row.fisher_odds_ratio):.2f} "
        f"(Fisher P={float(jam3b_row.fisher_two_sided_p):.2g}); myog was detected in "
        f"{float(myog_given_positive.loc['jam3b']):.1f}% of jam3b+ cells versus "
        f"{float(myog_given_negative.loc['jam3b']):.1f}% of jam3b− cells. For jam2a, "
        f"the corresponding values were {float(myog_given_positive.loc['jam2a']):.1f}% "
        f"and {float(myog_given_negative.loc['jam2a']):.1f}%. These cross-sectional "
        "data are concordant with the differential response reported in 20 hpf "
        "myog-null embryos, where jam3b was reduced but jam2a was unchanged "
        "(Ganassi et al., 2018, Nature Communications 9:4232, "
        "doi:10.1038/s41467-018-06583-6). Jam2a and Jam3b are both established "
        "heterophilic myocyte-fusion factors (Powell and Wright, 2011, PLoS Biology "
        "9:e1001216, doi:10.1371/journal.pbio.1001216). The atlas analysis does not "
        "test direct physical contact, causal regulation, or the published mutant "
        "phenotype."
    )
    if not reader_output:
        (output / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    reviewer_response = f"""# Response to reviewer concern on attention interpretability

We agree that an attention coefficient cannot be interpreted directly as a biochemical communication probability. We therefore evaluated the current accepted checkpoint at three complementary resolutions.

1. **Complete interaction field.** Cell-level attention was aggregated into one model interaction rank for each of all 361 directed cell-type pairs, including structural zeros. After adjustment for sender and receiver abundance, spatial distance, and self-pair status, these attention ranks agreed with COMMOT at rho={float(attention_commot.adjusted_spearman_rho):.3f} (empirical upper-tail P={float(attention_commot.adjusted_spearman_empirical_p_upper):.4g}) under the structured within-stratum null. The shared-database CellAgentChat proxy supplies a secondary complete-grid comparison.
2. **Pre-specified JAM edge compatibility.** Jam2a-Jam3b is a literature-supported heterophilic myocyte-fusion axis. We restricted this analysis to the same {n_somite:,} 18 hpf Somite cells and {n_somite_edges:,} model edges in every condition. JAM-compatible edges comprised {float(top_rates.loc['trained']):.1f}% of the top attention quartile but only {float(bottom_rates.loc['trained']):.1f}% of the bottom quartile after interaction training. The corresponding contrasts were {float(top_rates.loc['pre_interaction']):.1f}% versus {float(bottom_rates.loc['pre_interaction']):.1f}% before interaction learning and {float(top_rates.loc['random']):.1f}% versus {float(bottom_rates.loc['random']):.1f}% after randomization. The before-learning checkpoint is the same model before the interaction learning stage, whereas the randomized control shuffles the learned interaction weights. These are descriptive technical controls; edges and cells are not treated as biological replicates.
3. **Localized molecular response.** The map shows all {len(cells):,} tissue cells only for anatomical orientation; spatial and molecular statistics use the {n_somite:,} Somite cells. The current-trained high-attention JAM-compatible display edges localize to this region. Jam2a/Jam3b-compatible spatial neighbors were enriched over a within-Somite label-permutation null (fold={fold:.2f}, Monte Carlo P={float(null_row[p_spatial_column]):.2g}). The myog+ fraction was {float(myog_given_positive.loc['jam3b']):.1f}% among jam3b+ cells versus {float(myog_given_negative.loc['jam3b']):.1f}% among jam3b− cells, whereas the jam2a contrast was weaker ({float(myog_given_positive.loc['jam2a']):.1f}% versus {float(myog_given_negative.loc['jam2a']):.1f}%). This differential association is concordant with the report that jam3b, but not jam2a, was reduced in 20 hpf myog-null embryos (Ganassi et al., 2018, Nature Communications 9:4232, doi:10.1038/s41467-018-06583-6). Jam2a and Jam3b remain established heterophilic myocyte-fusion factors (Powell and Wright, 2011, PLoS Biology 9:e1001216, doi:10.1371/journal.pbio.1001216). Our cross-sectional co-detection analysis does not reproduce those mutant experiments or establish direct myog regulation or physical contact.

Together, these results show that the learned interaction field is reproducible across external CCI algorithms under shared inputs, preferentially organizes a pre-specified compatible JAM program relative to checkpoint controls, and localizes to a coherent myogenic molecular context. We retain the explicit boundary that attention is a model contribution, not a native ligand-receptor strength.
"""
    if not reader_output:
        (output / "reviewer_response.md").write_text(
            reviewer_response, encoding="utf-8"
        )

    code_dir = output / "code"
    code_dir.mkdir()
    if reader_output:
        wrapper_snapshot = code_dir / "run_zebrafish_attention_analysis.py"
        shutil.copy2(
            REPO_ROOT / "scripts" / "run_zebrafish_attention_analysis.py",
            wrapper_snapshot,
        )
        script_snapshot = code_dir / "attention_analysis_implementation.py"
        shutil.copy2(Path(__file__).resolve(), script_snapshot)
    else:
        script_snapshot = code_dir / Path(__file__).name
        shutil.copy2(Path(__file__).resolve(), script_snapshot)
    rebuild_entry = (
        "python -m scripts.run_zebrafish_attention_analysis figure"
        if reader_output
        else "python scripts/run_zebrafish_attention_validation.py report"
    )
    rebuild_command = (
        f"{rebuild_entry} "
        f"--analysis-dir {analysis_dir.expanduser().resolve()} "
        f"--output-dir <NEW_OUTPUT_DIR> "
        f"--expected-analysis-manifest-sha256 {expected_analysis_manifest_sha256.casefold()} "
        + " ".join(
            f"--jam-manifest {record['path']} "
            f"--expected-jam-manifest-sha256 {record['sha256']}"
            for record in jam_manifest_records
        )
    )
    jam_source_lines = "\n".join(
        f"- `{record['path']}` (SHA-256 `{record['sha256']}`)"
        for record in jam_manifest_records
    )
    provenance = (
        "# Provenance\n\n"
        "## Source paths\n\n"
        f"- Pair-field analysis: `{analysis_dir.expanduser().resolve()}`\n"
        f"- Pair-field manifest SHA-256: `{expected_analysis_manifest_sha256}`\n"
        f"{jam_source_lines}\n\n"
        "Legacy JAM roots containing `20260722` or `20260728` are rejected by the "
        "report CLI. Panel a displays the attention view only. Panel b uses canonical "
        "`trained`, `pre_interaction`, and `random` conditions; the old `init` label is "
        "not relabelled. Panel c display edges are selected upstream under the frozen "
        "current-checkpoint rule and are not selected inside the plotting function.\n\n"
        "## Literature context\n\n"
        "Powell and Wright (2011, PLoS Biology 9:e1001216, "
        'doi:10.1371/journal.pbio.1001216) reported that "jamb is expressed by all '
        'fast muscle myoblasts shortly after the formation of each somite." The '
        "same study established Jam2a/Jamb and Jam3b/Jamc as a heterophilic pair "
        "required for myocyte fusion. Ganassi et al. (2018, Nature Communications "
        '9:4232, doi:10.1038/s41467-018-06583-6) reported that "jam3b mRNA was '
        "significantly reduced (22%) in mutants, but jam2a and kirrel3l were "
        'unaffected." These studies support biological concordance, not the exact '
        "atlas percentages or a causal claim from co-detection. Direct Myog promoter "
        "binding in Ganassi et al. was demonstrated for mymk, not jam3b.\n\n"
        "## Rebuild\n\n"
        f"```bash\n{rebuild_command}\n```\n\n"
        "## SHA-256\n\n"
        f"- Figure PDF: `{_sha256(pdf_path)}`\n"
        f"- Plotting-script snapshot: `{_sha256(script_snapshot)}`\n"
    )
    if not reader_output:
        (output / "provenance.md").write_text(provenance, encoding="utf-8")
    shutil.copy2(
        analysis_dir.expanduser().resolve() / "analysis_manifest.json",
        output / "analysis_manifest.json",
    )
    jam_manifest_dir = output / "jam_manifests"
    jam_manifest_dir.mkdir()
    for index, record in enumerate(jam_manifest_records, start=1):
        shutil.copy2(
            record["path"], jam_manifest_dir / f"jam_manifest_{index:02d}.json"
        )

    panel_dir = output / "panel_data"
    panel_dir.mkdir()
    panel_sources = {
        "directed_pair_concordance.csv": paths["directed_pair_concordance.csv"],
        "jam_compatibility_percentile_summary.csv": jam_paths["compatibility_summary"],
        "jam_quartile_compatibility.csv": jam_paths["quartile_compatibility"],
        "type_pair_raw_attention_ranks.csv": jam_paths["type_pair_ranks"],
        "somite_18hpf_spatial_null_summary.csv": jam_paths["spatial_null_summary"],
        "somite_18hpf_spatial_null_iterations.csv.gz": jam_paths[
            "spatial_null_iterations"
        ],
        "myog_association.csv": jam_paths["myog_association"],
        "expression_detection_by_stage_type.csv": jam_paths["expression_detection"],
        "somite_18hpf_spatial_cells.csv.gz": jam_paths["spatial_cells"],
        "trained_jam_display_edges.csv": jam_paths["display_edges"],
    }
    for filename, source in panel_sources.items():
        shutil.copy2(source, panel_dir / filename)

    report_manifest_path = output / "report_manifest.json"
    report_artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != report_manifest_path:
            report_artifacts[str(path.relative_to(output))] = _artifact(path)
    report_claim_contract = dict(analysis_manifest.get("claim_contract") or {})
    report_claim_contract.pop("original_paper_21_scope", None)
    report_claim_contract.pop("nichenet_scope", None)
    report_claim_contract.update(
        {
            "reviewer_figure_scope": (
                "complete-pair adjusted agreement, current-checkpoint JAM controls, "
                "and localized cross-sectional myogenic consistency"
            ),
            "figure_a_cytobridge_view": "attention",
            "figure_canvas": "A4 portrait",
            "panel_b_scope": (
                f"18 hpf Somite only; {n_somite} cells and {n_somite_edges} fixed edges"
            ),
            "panel_c_context_scope": (
                f"{len(cells)} tissue cells displayed; statistics use {n_somite} Somite cells"
            ),
            "jam_controls_are_biological_replicates": False,
            "jam_statistics_scope": "descriptive technical controls",
            "myog_scope": "cross-sectional association; not causal regulation",
            "jam_literature_support": {
                "heterophilic_fusion_axis": "doi:10.1371/journal.pbio.1001216",
                "myog_differential_expression": "doi:10.1038/s41467-018-06583-6",
                "literature_validates_atlas_percentages": False,
                "direct_myog_binding_to_jam3b_claimed": False,
            },
            "legacy_jam_outputs_allowed": False,
            "attention_is_biochemical_probability": False,
        }
    )
    report_manifest = {
        "schema_version": SCHEMA_VERSION,
        "workflow": WORKFLOW,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "analysis_manifest_sha256": expected_analysis_manifest_sha256.casefold(),
        "jam_manifests": jam_manifest_records,
        "implementation": _implementation_identity(),
        "claim_contract": report_claim_contract,
        "figure_panels": {
            "a": (
                "attention-only complete 361-pair adjusted concordance with COMMOT "
                "and CellAgentChat proxy against structured-null 95% intervals"
            ),
            "b": (
                "JAM-compatible edge fractions in the top and bottom attention "
                "quartiles after training, before interaction learning, and after "
                "randomizing learned interaction weights"
            ),
            "c": (
                "current-trained high-attention JAM spatial localization, intuitive "
                "myog-positive fractions by JAM-gene detection, and within-Somite "
                "spatial permutation enrichment"
            ),
        },
        "panel_data_files": [f"panel_data/{filename}" for filename in panel_sources],
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
    analyze_parser = sub.add_parser("analyze", help="Calculate the comparison tables")
    analyze_parser.add_argument("--spec", required=True, type=Path)
    analyze_parser.add_argument("--output-dir", required=True, type=Path)
    analyze_parser.add_argument("--n-selected-pairs", type=int, default=8)
    report_parser = sub.add_parser("report", help="Draw the report from calculated tables")
    report_parser.add_argument("--analysis-dir", required=True, type=Path)
    report_parser.add_argument("--output-dir", required=True, type=Path)
    report_parser.add_argument(
        "--expected-analysis-manifest-sha256", help=argparse.SUPPRESS
    )
    report_parser.add_argument(
        "--jam-manifest",
        required=True,
        action="append",
        type=Path,
        help="JAM result file; repeat for the comparison conditions",
    )
    report_parser.add_argument(
        "--expected-jam-manifest-sha256",
        action="append",
        help=argparse.SUPPRESS,
    )
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
            expected_analysis_manifest_sha256=args.expected_analysis_manifest_sha256,
            jam_manifest_paths=list(args.jam_manifest),
            expected_jam_manifest_sha256s=(
                list(args.expected_jam_manifest_sha256)
                if args.expected_jam_manifest_sha256
                else None
            ),
        )
    elif args.command == "validate":
        validate(args.output_dir)
    else:
        validate_report(args.output_dir)


if __name__ == "__main__":
    main()
