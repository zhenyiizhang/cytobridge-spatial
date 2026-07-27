#!/usr/bin/env python3
"""Audit observed-anchor LR scores under min versus geometric-mean complexes.

Both score views are projected from the same observed normalized-expression
matrix and the same persisted cell-type communication matrices.  The analysis
uses strict all-subunit eligibility and does not construct an external-method
consensus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
from typing import Mapping, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CytoBridge.tl.downstream.downstream_data import (  # noqa: E402
    infer_time_key,
    parse_time_value,
)
from CytoBridge.tl.downstream.lr_projection import (  # noqa: E402
    load_ligand_receptor_database,
    project_communication_to_lr_timecourses,
)
from CytoBridge.tl.downstream.temporal import simplify_gene_names  # noqa: E402


TIME_ATOL = 1e-8
SCORE_TOL = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _jsonable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, float):
        return value
    return str(value)


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        return {
            "commit": commit,
            "dirty": bool(status.strip()),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except Exception:
        return {
            "commit": None,
            "dirty": None,
            "tracked_diff_sha256": None,
        }


def _split_complex(token: str) -> list[str]:
    return [part.strip() for part in str(token).split("_") if part.strip()]


def _parse_float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    parsed = [parse_time_value(item) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--stages must contain at least one value.")
    if len(set(parsed)) != len(parsed):
        raise ValueError("--stages contains duplicate values.")
    return sorted(map(float, parsed))


def _parse_top_k(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("--top-k must contain positive comma-separated integers.")
    return sorted(set(parsed))


def _matrix_audit(matrix) -> dict[str, object]:
    values = matrix
    if sparse.issparse(values):
        data = np.asarray(values.data, dtype=np.float64)
        minimum = float(min(0.0, data.min())) if data.size else 0.0
        maximum = float(max(0.0, data.max())) if data.size else 0.0
        finite = bool(np.isfinite(data).all())
        nonzero = int(values.nnz)
        total = int(values.shape[0] * values.shape[1])
    else:
        values = np.asarray(values, dtype=np.float64)
        minimum = float(values.min()) if values.size else 0.0
        maximum = float(values.max()) if values.size else 0.0
        finite = bool(np.isfinite(values).all())
        nonzero = int(np.count_nonzero(values))
        total = int(values.size)
    return {
        "minimum": minimum,
        "maximum": maximum,
        "finite": finite,
        "n_nonzero": nonzero,
        "nonzero_fraction": float(nonzero / total) if total else 0.0,
    }


def _load_communications(path: Path) -> Mapping[str, Mapping[str, object]]:
    with path.open("rb") as handle:
        communications = pickle.load(handle)
    if not isinstance(communications, Mapping) or not communications:
        raise ValueError(
            "The communication pickle must contain a non-empty time-keyed mapping."
        )
    malformed = [
        str(key)
        for key, record in communications.items()
        if not isinstance(record, Mapping)
    ]
    if malformed:
        raise ValueError(
            "Communication records must be mappings; malformed keys="
            f"{malformed[:10]}."
        )
    return communications


def _match_communication_record(
    communications: Mapping[str, Mapping[str, object]],
    stage: float,
) -> tuple[str, Mapping[str, object]]:
    matches = []
    for key, record in communications.items():
        try:
            value = parse_time_value(key)
        except ValueError:
            continue
        if np.isclose(value, stage, rtol=0.0, atol=TIME_ATOL):
            matches.append((str(key), record))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one communication record for observed stage {stage}; "
            f"found keys={[key for key, _ in matches]}."
        )
    return matches[0]


def _select_observed_stages(
    observed_times: np.ndarray,
    requested: Sequence[float] | None,
) -> list[float]:
    available = sorted(np.unique(observed_times).astype(float).tolist())
    if requested is None:
        selected = available
    else:
        selected = []
        for value in requested:
            matches = [
                candidate
                for candidate in available
                if np.isclose(candidate, value, rtol=0.0, atol=TIME_ATOL)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Requested stage {value} did not uniquely match observed "
                    f"stages {available}."
                )
            selected.append(float(matches[0]))
    if len(selected) < 2:
        raise ValueError(
            "Observed-anchor sensitivity requires at least two stages so that "
            "per-pair temporal stability is defined."
        )
    return selected


def _eligibility_table(
    adata,
    lr_database: Path,
    *,
    preferred_species_tag: str | None,
) -> pd.DataFrame:
    database = load_ligand_receptor_database(lr_database)
    aliases = simplify_gene_names(
        tuple(map(str, adata.var_names)),
        preferred_species_tag=preferred_species_tag,
    )
    available = set(aliases["gene_symbol"].astype(str))
    rows = []
    for record in database.itertuples(index=False):
        ligand = str(record.ligand)
        receptor = str(record.receptor)
        ligand_subunits = _split_complex(ligand)
        receptor_subunits = _split_complex(receptor)
        requested = ligand_subunits + receptor_subunits
        missing = sorted({subunit for subunit in requested if subunit not in available})
        rows.append(
            {
                "pair": f"{ligand}_{receptor}",
                "ligand": ligand,
                "receptor": receptor,
                "ligand_n_subunits": len(ligand_subunits),
                "receptor_n_subunits": len(receptor_subunits),
                "n_subunits_total": len(requested),
                "is_multisubunit": bool(
                    len(ligand_subunits) > 1 or len(receptor_subunits) > 1
                ),
                "strict_all_subunits_eligible": not missing,
                "n_missing_subunits": len(missing),
                "missing_subunits": ";".join(missing),
            }
        )
    result = pd.DataFrame(rows)
    collisions = result.groupby("pair")[["ligand", "receptor"]].nunique()
    collisions = collisions.loc[
        (collisions["ligand"] > 1) | (collisions["receptor"] > 1)
    ]
    if not collisions.empty:
        raise ValueError(
            "Underscore-concatenated LR pair identifiers are ambiguous for "
            f"{collisions.index.tolist()[:10]}."
        )
    return result.drop_duplicates("pair").sort_values("pair").reset_index(drop=True)


def _communication_scaffold(
    communications: Mapping[str, Mapping[str, object]],
    stages: Sequence[float],
    *,
    matrix_key: str,
    observed_labels: np.ndarray,
    observed_times: np.ndarray,
) -> tuple[dict[str, Mapping[str, object]], pd.DataFrame]:
    selected: dict[str, Mapping[str, object]] = {}
    rows = []
    for stage in stages:
        original_key, record = _match_communication_record(communications, stage)
        if "types" not in record or matrix_key not in record:
            raise KeyError(
                f"Communication record {original_key!r} must contain 'types' and "
                f"{matrix_key!r}."
            )
        cell_types = np.asarray(record["types"]).astype(str)
        if cell_types.ndim != 1 or len(set(cell_types.tolist())) != cell_types.size:
            raise ValueError(
                f"Communication types at stage {stage} must be a unique vector."
            )
        matrix = np.asarray(record[matrix_key], dtype=np.float64)
        expected = (cell_types.size, cell_types.size)
        if matrix.shape != expected:
            raise ValueError(
                f"Communication matrix at stage {stage} has shape {matrix.shape}; "
                f"expected {expected}."
            )
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError(
                f"Communication matrix at stage {stage} must be finite and "
                "non-negative."
            )
        stage_mask = np.isclose(observed_times, stage, rtol=0.0, atol=TIME_ATOL)
        present_types = set(observed_labels[stage_mask])
        missing_types = sorted(set(cell_types) - present_types)
        canonical_key = str(float(stage))
        selected[canonical_key] = record
        support = np.asarray(matrix > 0, dtype=np.uint8)
        rows.append(
            {
                "stage": float(stage),
                "source_pickle_key": original_key,
                "matrix_key": matrix_key,
                "n_cell_types": int(cell_types.size),
                "n_nonzero_edges": int(support.sum()),
                "edge_density": float(support.mean()) if support.size else 0.0,
                "communication_sum": float(matrix.sum()),
                "communication_max": float(matrix.max()) if matrix.size else 0.0,
                "types_sha256": _array_sha256(cell_types.astype("U")),
                "matrix_sha256": _array_sha256(matrix),
                "support_sha256": _array_sha256(support),
                "n_types_without_observed_cells": len(missing_types),
                "types_without_observed_cells": ";".join(missing_types),
                "identical_input_for_both_aggregation_modes": True,
            }
        )
    return selected, pd.DataFrame(rows)


def _safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=np.float64)
    y = np.asarray(y[finite], dtype=np.float64)
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    raise ValueError(f"Unknown correlation method {method!r}.")


def _finite_mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _merge_score_views(
    minimum,
    geometric,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    left = minimum.pair_timecourse[
        ["time", "pair", "score", "max_edge", "nonzero_edges"]
    ].rename(
        columns={
            "score": "score_min",
            "max_edge": "max_edge_min",
            "nonzero_edges": "nonzero_edges_min",
        }
    )
    right = geometric.pair_timecourse[
        ["time", "pair", "score", "max_edge", "nonzero_edges"]
    ].rename(
        columns={
            "score": "score_geometric_mean",
            "max_edge": "max_edge_geometric_mean",
            "nonzero_edges": "nonzero_edges_geometric_mean",
        }
    )
    merged = left.merge(
        right,
        on=["time", "pair"],
        how="outer",
        validate="1:1",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        mismatch = merged.loc[~merged["_merge"].eq("both"), ["time", "pair", "_merge"]]
        raise RuntimeError(
            "Min and zero-preserving geometric mean produced different "
            "time/pair support despite identical strict eligibility; examples="
            f"{mismatch.head().to_dict(orient='records')}."
        )
    merged = merged.drop(columns="_merge").merge(
        eligibility,
        on="pair",
        how="left",
        validate="many_to_one",
    )
    if merged["strict_all_subunits_eligible"].isna().any():
        missing = sorted(
            merged.loc[merged["strict_all_subunits_eligible"].isna(), "pair"].unique()
        )
        raise RuntimeError(
            f"Scored LR pairs are absent from the eligibility audit: {missing[:10]}."
        )
    if not merged["strict_all_subunits_eligible"].all():
        raise RuntimeError("An LR pair missing at least one subunit was scored.")
    for column in ("score_min", "score_geometric_mean"):
        values = merged[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise RuntimeError(f"{column} must be finite and non-negative.")
    merged["score_difference_geometric_minus_min"] = (
        merged["score_geometric_mean"] - merged["score_min"]
    )
    merged["absolute_score_difference"] = merged[
        "score_difference_geometric_minus_min"
    ].abs()
    denominator = np.maximum.reduce(
        [
            merged["score_min"].abs().to_numpy(dtype=np.float64),
            merged["score_geometric_mean"].abs().to_numpy(dtype=np.float64),
            np.full(len(merged), np.finfo(np.float64).eps),
        ]
    )
    merged["symmetric_relative_difference"] = (
        merged["absolute_score_difference"].to_numpy(dtype=np.float64) / denominator
    )
    merged["both_zero"] = merged["score_min"].abs().le(SCORE_TOL) & merged[
        "score_geometric_mean"
    ].abs().le(SCORE_TOL)
    merged["aggregation_sensitive"] = merged["absolute_score_difference"].gt(SCORE_TOL)
    for mode in ("min", "geometric_mean"):
        score_column = f"score_{mode}"
        merged[f"rank_{mode}"] = merged.groupby("time")[score_column].rank(
            ascending=False, method="average"
        )
        group_size = merged.groupby("time")[score_column].transform("size")
        merged[f"rank_percentile_{mode}"] = 1.0 - (
            merged[f"rank_{mode}"] - 1.0
        ) / np.maximum(group_size - 1, 1)
    return merged.sort_values(["time", "pair"]).reset_index(drop=True)


def _stability_metrics(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_rows = []
    for stage, current in merged.groupby("time", sort=True):
        for scope, subset in (
            ("all_scored_pairs", current),
            ("multisubunit_pairs", current.loc[current["is_multisubunit"]]),
        ):
            x = subset["score_min"].to_numpy(dtype=np.float64)
            y = subset["score_geometric_mean"].to_numpy(dtype=np.float64)
            stage_rows.append(
                {
                    "stage": float(stage),
                    "scope": scope,
                    "n_pairs": int(len(subset)),
                    "score_spearman": _safe_correlation(x, y, "spearman"),
                    "score_pearson": _safe_correlation(x, y, "pearson"),
                    "rank_percentile_pearson": _safe_correlation(
                        subset["rank_percentile_min"].to_numpy(dtype=np.float64),
                        subset["rank_percentile_geometric_mean"].to_numpy(
                            dtype=np.float64
                        ),
                        "pearson",
                    ),
                    "n_aggregation_sensitive": int(
                        subset["aggregation_sensitive"].sum()
                    ),
                    "fraction_aggregation_sensitive": (
                        float(subset["aggregation_sensitive"].mean())
                        if len(subset)
                        else np.nan
                    ),
                    "mean_absolute_score_difference": (
                        float(subset["absolute_score_difference"].mean())
                        if len(subset)
                        else np.nan
                    ),
                    "max_symmetric_relative_difference": (
                        float(subset["symmetric_relative_difference"].max())
                        if len(subset)
                        else np.nan
                    ),
                }
            )
    by_stage = pd.DataFrame(stage_rows)

    overall_rows = []
    for scope, subset in (
        ("all_scored_pairs", merged),
        ("multisubunit_pairs", merged.loc[merged["is_multisubunit"]]),
    ):
        stage_subset = by_stage.loc[by_stage["scope"].eq(scope)]
        x = subset["score_min"].to_numpy(dtype=np.float64)
        y = subset["score_geometric_mean"].to_numpy(dtype=np.float64)
        overall_rows.append(
            {
                "scope": scope,
                "n_stage_pair_observations": int(len(subset)),
                "n_stages": int(subset["time"].nunique()),
                "n_pairs": int(subset["pair"].nunique()),
                "pooled_score_spearman": _safe_correlation(x, y, "spearman"),
                "pooled_score_pearson": _safe_correlation(x, y, "pearson"),
                "pooled_within_stage_rank_spearman": _safe_correlation(
                    subset["rank_percentile_min"].to_numpy(dtype=np.float64),
                    subset["rank_percentile_geometric_mean"].to_numpy(dtype=np.float64),
                    "spearman",
                ),
                "macro_mean_stage_score_spearman": _finite_mean(
                    stage_subset["score_spearman"]
                ),
                "minimum_stage_score_spearman": (
                    float(stage_subset["score_spearman"].dropna().min())
                    if stage_subset["score_spearman"].notna().any()
                    else np.nan
                ),
                "n_aggregation_sensitive_observations": int(
                    subset["aggregation_sensitive"].sum()
                ),
                "fraction_aggregation_sensitive_observations": (
                    float(subset["aggregation_sensitive"].mean())
                    if len(subset)
                    else np.nan
                ),
                "max_symmetric_relative_difference": (
                    float(subset["symmetric_relative_difference"].max())
                    if len(subset)
                    else np.nan
                ),
            }
        )
    return by_stage, pd.DataFrame(overall_rows)


def _top_ids(
    subset: pd.DataFrame,
    *,
    score_column: str,
    top_k: int,
) -> tuple[set[str], int]:
    if subset.empty:
        return set(), 0
    table = subset.assign(
        _unit_id=(
            subset["time"].map(lambda value: f"{float(value):.12g}")
            + "|"
            + subset["pair"].astype(str)
        )
    ).sort_values(
        [score_column, "time", "pair"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    selected_n = min(int(top_k), len(table))
    return set(table.head(selected_n)["_unit_id"]), selected_n


def _top_k_overlap(
    merged: pd.DataFrame,
    top_k_values: Sequence[int],
) -> pd.DataFrame:
    rows = []
    stage_groups: list[tuple[str, pd.DataFrame]] = [
        (f"{float(stage):.12g}", subset)
        for stage, subset in merged.groupby("time", sort=True)
    ]
    stage_groups.append(("ALL_STAGES_POOLED", merged))
    for stage_label, current in stage_groups:
        for scope, subset in (
            ("all_scored_pairs", current),
            ("multisubunit_pairs", current.loc[current["is_multisubunit"]]),
        ):
            for requested_k in top_k_values:
                top_min, n_min = _top_ids(
                    subset, score_column="score_min", top_k=requested_k
                )
                top_geometric, n_geometric = _top_ids(
                    subset,
                    score_column="score_geometric_mean",
                    top_k=requested_k,
                )
                intersection = top_min & top_geometric
                union = top_min | top_geometric
                rows.append(
                    {
                        "stage": stage_label,
                        "scope": scope,
                        "requested_top_k": int(requested_k),
                        "effective_top_k_min": int(n_min),
                        "effective_top_k_geometric_mean": int(n_geometric),
                        "overlap_n": int(len(intersection)),
                        "overlap_fraction_of_k": (
                            float(len(intersection) / min(n_min, n_geometric))
                            if min(n_min, n_geometric)
                            else np.nan
                        ),
                        "jaccard": (
                            float(len(intersection) / len(union)) if union else np.nan
                        ),
                        "tie_policy": (
                            "deterministic exact-k: score descending, then "
                            "stage and pair ascending"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _multisubunit_pair_stability(
    merged: pd.DataFrame,
    *,
    primary_top_k: int,
) -> pd.DataFrame:
    top_membership: dict[tuple[float, str, str], bool] = {}
    for stage, subset in merged.groupby("time", sort=True):
        for mode in ("min", "geometric_mean"):
            selected, _ = _top_ids(
                subset,
                score_column=f"score_{mode}",
                top_k=primary_top_k,
            )
            for pair in subset["pair"].astype(str):
                unit = f"{float(stage):.12g}|{pair}"
                top_membership[(float(stage), pair, mode)] = unit in selected

    rows = []
    multi = merged.loc[merged["is_multisubunit"]]
    for pair, subset in multi.groupby("pair", sort=True):
        subset = subset.sort_values("time")
        x = subset["score_min"].to_numpy(dtype=np.float64)
        y = subset["score_geometric_mean"].to_numpy(dtype=np.float64)
        memberships = [
            (
                top_membership[(float(row.time), str(pair), "min")],
                top_membership[(float(row.time), str(pair), "geometric_mean")],
            )
            for row in subset.itertuples(index=False)
        ]
        first = subset.iloc[0]
        rows.append(
            {
                "pair": str(pair),
                "ligand": str(first["ligand"]),
                "receptor": str(first["receptor"]),
                "ligand_n_subunits": int(first["ligand_n_subunits"]),
                "receptor_n_subunits": int(first["receptor_n_subunits"]),
                "n_stages": int(len(subset)),
                "trajectory_score_spearman": _safe_correlation(x, y, "spearman"),
                "trajectory_score_pearson": _safe_correlation(x, y, "pearson"),
                "trajectory_rank_percentile_spearman": _safe_correlation(
                    subset["rank_percentile_min"].to_numpy(dtype=np.float64),
                    subset["rank_percentile_geometric_mean"].to_numpy(dtype=np.float64),
                    "spearman",
                ),
                "mean_absolute_score_difference": float(
                    subset["absolute_score_difference"].mean()
                ),
                "max_absolute_score_difference": float(
                    subset["absolute_score_difference"].max()
                ),
                "mean_symmetric_relative_difference": float(
                    subset["symmetric_relative_difference"].mean()
                ),
                "max_symmetric_relative_difference": float(
                    subset["symmetric_relative_difference"].max()
                ),
                "maximum_absolute_rank_shift": float(
                    (subset["rank_min"] - subset["rank_geometric_mean"]).abs().max()
                ),
                "n_aggregation_sensitive_stages": int(
                    subset["aggregation_sensitive"].sum()
                ),
                "primary_top_k": int(primary_top_k),
                "n_top_min": int(sum(left for left, _ in memberships)),
                "n_top_geometric_mean": int(sum(right for _, right in memberships)),
                "n_top_both": int(sum(left and right for left, right in memberships)),
                "top_membership_agreement_fraction": float(
                    np.mean([left == right for left, right in memberships])
                ),
            }
        )
    return pd.DataFrame(rows)


def _coverage_table(
    minimum,
    geometric,
    merged: pd.DataFrame,
    eligibility: pd.DataFrame,
    scaffold: pd.DataFrame,
) -> pd.DataFrame:
    min_coverage = minimum.coverage.rename(
        columns=lambda column: (f"{column}_min" if column != "time" else "time")
    )
    geometric_coverage = geometric.coverage.rename(
        columns=lambda column: (
            f"{column}_geometric_mean" if column != "time" else "time"
        )
    )
    api_coverage = min_coverage.merge(
        geometric_coverage,
        on="time",
        how="outer",
        validate="1:1",
    )
    n_database_pairs = int(len(eligibility))
    eligible = eligibility.loc[eligibility["strict_all_subunits_eligible"]]
    n_eligible = int(len(eligible))
    n_eligible_multi = int(eligible["is_multisubunit"].sum())
    scored_pairs = set(merged["pair"].astype(str))
    scored_multi_pairs = set(merged.loc[merged["is_multisubunit"], "pair"].astype(str))
    rows = []
    for stage, subset in merged.groupby("time", sort=True):
        api = api_coverage.loc[
            np.isclose(
                api_coverage["time"].astype(float),
                float(stage),
                rtol=0.0,
                atol=TIME_ATOL,
            )
        ]
        if len(api) != 1:
            raise RuntimeError(f"Missing API coverage row for stage {stage}.")
        scaffold_row = scaffold.loc[
            np.isclose(
                scaffold["stage"].astype(float),
                float(stage),
                rtol=0.0,
                atol=TIME_ATOL,
            )
        ]
        if len(scaffold_row) != 1:
            raise RuntimeError(f"Missing communication audit row for stage {stage}.")
        api_row = api.iloc[0]
        scaffold_values = scaffold_row.iloc[0]
        rows.append(
            {
                "stage": float(stage),
                "n_database_pairs": n_database_pairs,
                "n_strict_all_subunits_eligible_pairs": n_eligible,
                "n_strict_eligible_multisubunit_pairs": n_eligible_multi,
                "n_pairs_missing_at_least_one_subunit": int(
                    (~eligibility["strict_all_subunits_eligible"]).sum()
                ),
                "n_nonzero_pairs_scored_both_modes": int(len(subset)),
                "n_nonzero_multisubunit_pairs_scored_both_modes": int(
                    subset["is_multisubunit"].sum()
                ),
                "n_strict_eligible_pairs_all_zero_or_unscored": int(
                    n_eligible - len(scored_pairs)
                ),
                "n_strict_eligible_multisubunit_pairs_all_zero_or_unscored": int(
                    n_eligible_multi - len(scored_multi_pairs)
                ),
                "n_zero_score_pairs_min": int(
                    subset["score_min"].abs().le(SCORE_TOL).sum()
                ),
                "n_zero_score_pairs_geometric_mean": int(
                    subset["score_geometric_mean"].abs().le(SCORE_TOL).sum()
                ),
                "strict_eligible_nonzero_coverage_fraction": (
                    float(len(subset) / n_eligible) if n_eligible else np.nan
                ),
                "api_n_pairs_scored_min": int(api_row["n_lr_pairs_scored_min"]),
                "api_n_pairs_scored_geometric_mean": int(
                    api_row["n_lr_pairs_scored_geometric_mean"]
                ),
                "api_expression_source_min": str(api_row["expression_source_min"]),
                "api_expression_source_geometric_mean": str(
                    api_row["expression_source_geometric_mean"]
                ),
                "api_n_expression_cells_min": int(api_row["n_expression_cells_min"]),
                "api_n_expression_cells_geometric_mean": int(
                    api_row["n_expression_cells_geometric_mean"]
                ),
                "communication_n_cell_types": int(scaffold_values["n_cell_types"]),
                "communication_n_nonzero_edges": int(
                    scaffold_values["n_nonzero_edges"]
                ),
                "communication_support_sha256": str(scaffold_values["support_sha256"]),
                "n_communication_types_without_observed_cells": int(
                    scaffold_values["n_types_without_observed_cells"]
                ),
                "strict_all_subunit_policy": True,
                "identical_communication_scaffold": True,
            }
        )
    return pd.DataFrame(rows)


def _plot_bundle(
    merged: pd.DataFrame,
    by_stage: pd.DataFrame,
    top_overlap: pd.DataFrame,
    multisubunit: pd.DataFrame,
    *,
    primary_top_k: int,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
    colors = np.where(merged["is_multisubunit"], "#D95F02", "#3569A8")
    x = np.log1p(merged["score_min"].to_numpy(dtype=np.float64))
    y = np.log1p(merged["score_geometric_mean"].to_numpy(dtype=np.float64))
    axes[0, 0].scatter(x, y, c=colors, s=18, alpha=0.68, linewidths=0)
    lower = float(min(x.min(), y.min()))
    upper = float(max(x.max(), y.max()))
    axes[0, 0].plot([lower, upper], [lower, upper], "--", color="black", lw=1)
    axes[0, 0].set(
        xlabel="log1p score: minimum",
        ylabel="log1p score: geometric mean",
        title="Same observed anchors and communication scaffold",
    )

    for scope, color, label in (
        ("all_scored_pairs", "#3569A8", "all pairs"),
        ("multisubunit_pairs", "#D95F02", "multi-subunit"),
    ):
        subset = by_stage.loc[by_stage["scope"].eq(scope)]
        axes[0, 1].plot(
            subset["stage"],
            subset["score_spearman"],
            marker="o",
            color=color,
            label=label,
        )
    axes[0, 1].axhline(1.0, color="black", ls="--", lw=1)
    axes[0, 1].set(
        ylim=(-0.05, 1.05),
        xlabel="Observed stage",
        ylabel="Spearman rho",
        title="Stage-wise score stability",
    )
    axes[0, 1].legend(frameon=False)

    overlap = top_overlap.loc[
        top_overlap["requested_top_k"].eq(primary_top_k)
        & ~top_overlap["stage"].eq("ALL_STAGES_POOLED")
    ]
    for scope, color, label in (
        ("all_scored_pairs", "#2A9D8F", "all pairs"),
        ("multisubunit_pairs", "#E9C46A", "multi-subunit"),
    ):
        subset = overlap.loc[overlap["scope"].eq(scope)].copy()
        subset["_stage_float"] = pd.to_numeric(subset["stage"])
        subset = subset.sort_values("_stage_float")
        axes[1, 0].plot(
            subset["_stage_float"],
            subset["overlap_fraction_of_k"],
            marker="o",
            color=color,
            label=label,
        )
    axes[1, 0].axhline(1.0, color="black", ls="--", lw=1)
    axes[1, 0].set(
        ylim=(-0.05, 1.05),
        xlabel="Observed stage",
        ylabel=f"Top-{primary_top_k} overlap / k",
        title="Top-signal stability",
    )
    axes[1, 0].legend(frameon=False)

    if multisubunit.empty:
        axes[1, 1].text(
            0.5,
            0.5,
            "No strictly eligible\nmulti-subunit LR pairs",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_axis_off()
    else:
        sensitive = multisubunit.nlargest(
            min(12, len(multisubunit)),
            "max_symmetric_relative_difference",
        ).sort_values("max_symmetric_relative_difference")
        axes[1, 1].barh(
            sensitive["pair"],
            sensitive["max_symmetric_relative_difference"],
            color="#D95F02",
        )
        axes[1, 1].set(
            xlabel="Maximum symmetric relative difference",
            title="Most sensitive multi-subunit pairs",
        )
    for axis in axes.ravel():
        axis.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle(
        "Observed-anchor LR complex sensitivity: min vs zero-preserving geometric mean"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _write_table(table: pd.DataFrame, path: Path) -> Path:
    table.to_csv(path, index=False)
    return path


def run_analysis(
    *,
    h5ad_path: Path,
    communications_path: Path,
    lr_database_path: Path,
    output_dir: Path,
    time_key: str | None = None,
    annotation_key: str = "Annotation",
    matrix_key: str = "M_per_source",
    observed_layer: str | None = None,
    observed_expression_space: str = "log1p",
    score_expression_space: str = "count",
    preferred_species_tag: str | None = None,
    stages: Sequence[float] | None = None,
    top_k_values: Sequence[int] = (10, 20, 50),
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    input_paths = [
        Path(h5ad_path).expanduser().resolve(),
        Path(communications_path).expanduser().resolve(),
        Path(lr_database_path).expanduser().resolve(),
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if observed_expression_space not in {"log1p", "count"}:
        raise ValueError("observed_expression_space must be log1p or count.")
    if score_expression_space not in {"log1p", "count"}:
        raise ValueError("score_expression_space must be log1p or count.")
    top_k_values = sorted(set(map(int, top_k_values)))
    if not top_k_values or any(value <= 0 for value in top_k_values):
        raise ValueError("top_k_values must contain positive integers.")

    adata = ad.read_h5ad(input_paths[0])
    resolved_time_key = infer_time_key(adata.obs, preferred=time_key)
    if annotation_key not in adata.obs:
        raise KeyError(f"adata.obs is missing annotation key {annotation_key!r}.")
    if adata.obs[annotation_key].isna().any():
        raise ValueError(f"adata.obs[{annotation_key!r}] contains missing labels.")
    if observed_layer == "X":
        observed_layer = None
    if observed_layer is not None and observed_layer not in adata.layers:
        raise KeyError(f"adata.layers is missing {observed_layer!r}.")
    expression_matrix = (
        adata.X if observed_layer is None else adata.layers[observed_layer]
    )
    expression_audit = _matrix_audit(expression_matrix)
    if not expression_audit["finite"] or expression_audit["minimum"] < 0:
        raise ValueError(
            "Observed normalized expression must be finite and non-negative."
        )

    observed_times = np.asarray(
        [parse_time_value(value) for value in adata.obs[resolved_time_key]],
        dtype=np.float64,
    )
    selected_stages = _select_observed_stages(observed_times, stages)
    observed_labels = adata.obs[annotation_key].astype(str).to_numpy()
    communications = _load_communications(input_paths[1])
    selected_communications, scaffold = _communication_scaffold(
        communications,
        selected_stages,
        matrix_key=matrix_key,
        observed_labels=observed_labels,
        observed_times=observed_times,
    )
    eligibility = _eligibility_table(
        adata,
        input_paths[2],
        preferred_species_tag=preferred_species_tag,
    )
    if not eligibility["strict_all_subunits_eligible"].any():
        raise ValueError(
            "No LR pair passes strict all-subunit eligibility in the H5AD."
        )

    observed_slices = {}
    for stage in selected_stages:
        mask = np.isclose(observed_times, stage, rtol=0.0, atol=TIME_ATOL)
        if not mask.any():
            raise RuntimeError(f"No observed cells remain for stage {stage}.")
        observed_slices[str(float(stage))] = adata[mask]

    projection_kwargs = {
        "time_points": selected_stages,
        "annotation_key": annotation_key,
        "matrix_key": matrix_key,
        "reference_layer": observed_layer,
        "expression_space": score_expression_space,
        "require_all_subunits": True,
        "duplicate_policy": "first",
        "preferred_species_tag": preferred_species_tag,
        "n_clusters": 2,
        "profile_linkage_method": "average",
        "profile_cluster_order": "dendrogram",
        "observed_adata": adata,
        "observed_time_key": resolved_time_key,
        "observed_time_points": selected_stages,
        "observed_annotation_key": annotation_key,
        "observed_layer": observed_layer,
        "observed_expression_space": observed_expression_space,
        "observed_missing_time_policy": "error",
        "observed_time_atol": TIME_ATOL,
    }
    minimum = project_communication_to_lr_timecourses(
        observed_slices,
        adata,
        selected_communications,
        input_paths[2],
        complex_mode="min",
        **projection_kwargs,
    )
    geometric = project_communication_to_lr_timecourses(
        observed_slices,
        adata,
        selected_communications,
        input_paths[2],
        complex_mode="geometric_mean",
        **projection_kwargs,
    )
    if minimum.settings["uses_inverse_pca"] or geometric.settings["uses_inverse_pca"]:
        raise RuntimeError(
            "Observed-anchor sensitivity unexpectedly used inverse-PCA expression."
        )
    if (
        not minimum.coverage["expression_source"].eq("observed").all()
        or not geometric.coverage["expression_source"].eq("observed").all()
    ):
        raise RuntimeError("Every selected stage must use observed expression.")

    paired = _merge_score_views(minimum, geometric, eligibility)
    by_stage, overall = _stability_metrics(paired)
    top_overlap = _top_k_overlap(paired, top_k_values)
    multisubunit = _multisubunit_pair_stability(paired, primary_top_k=top_k_values[0])
    coverage = _coverage_table(
        minimum,
        geometric,
        paired,
        eligibility,
        scaffold,
    )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        _write_table(paired, output_dir / "observed_anchor_paired_scores.csv"),
        _write_table(by_stage, output_dir / "stage_score_stability.csv"),
        _write_table(overall, output_dir / "overall_score_stability.csv"),
        _write_table(top_overlap, output_dir / "top_k_overlap.csv"),
        _write_table(
            multisubunit,
            output_dir / "multisubunit_pair_stability.csv",
        ),
        _write_table(coverage, output_dir / "coverage_by_stage.csv"),
        _write_table(eligibility, output_dir / "strict_subunit_eligibility.csv"),
        _write_table(scaffold, output_dir / "communication_scaffold_audit.csv"),
    ]

    contracts_path = output_dir / "projection_contracts.json"
    contracts_path.write_text(
        json.dumps(
            {
                "minimum": minimum.settings,
                "zero_preserving_geometric_mean": geometric.settings,
            },
            indent=2,
            sort_keys=True,
            default=_jsonable,
        ),
        encoding="utf-8",
    )
    outputs.append(contracts_path)
    for suffix in ("png", "pdf"):
        outputs.append(
            _plot_bundle(
                paired,
                by_stage,
                top_overlap,
                multisubunit,
                primary_top_k=top_k_values[0],
                output_path=output_dir
                / f"observed_anchor_lr_complex_sensitivity.{suffix}",
            )
        )

    all_row = overall.loc[overall["scope"].eq("all_scored_pairs")].iloc[0]
    multi_row = overall.loc[overall["scope"].eq("multisubunit_pairs")].iloc[0]
    pooled_overlap = top_overlap.loc[
        top_overlap["stage"].eq("ALL_STAGES_POOLED")
        & top_overlap["scope"].eq("all_scored_pairs")
        & top_overlap["requested_top_k"].eq(top_k_values[0])
    ].iloc[0]
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Observed-anchor LR complex sensitivity",
                "",
                "This run recomputes both views from the same real observed "
                "normalized-expression cells and the same persisted communication "
                "matrix at every selected stage.",
                "",
                "- Eligibility is strict: every subunit of both ligand and receptor "
                "must be present.",
                "- `geometric_mean` is zero-preserving: a zero subunit makes the "
                "complex activity zero.",
                "- No external-method consensus is constructed, so CytoBridge is "
                "not included in an external consensus.",
                f"- Selected stages: {', '.join(map(str, selected_stages))}.",
                f"- Strictly eligible LR pairs: "
                f"{int(eligibility['strict_all_subunits_eligible'].sum())}.",
                f"- Strictly eligible multi-subunit pairs: "
                f"{int(eligibility.loc[eligibility['strict_all_subunits_eligible'], 'is_multisubunit'].sum())}.",
                f"- Pooled all-pair score Spearman: "
                f"{float(all_row['pooled_score_spearman']):.6g}.",
                f"- Pooled multi-subunit score Spearman: "
                f"{float(multi_row['pooled_score_spearman']):.6g}.",
                f"- Pooled top-{top_k_values[0]} overlap fraction: "
                f"{float(pooled_overlap['overlap_fraction_of_k']):.6g}.",
                "",
                "Use `stage_score_stability.csv` for per-stage correlations, "
                "`top_k_overlap.csv` for exact deterministic top-k agreement, and "
                "`multisubunit_pair_stability.csv` to inspect each complex "
                "separately. `communication_scaffold_audit.csv` records the exact "
                "matrix/support hashes shared by both projections.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs.append(readme_path)

    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "analysis": "observed_anchor_lr_complex_aggregation_sensitivity",
        "command": list(command) if command is not None else None,
        "script": {
            "path": str(script_path),
            "sha256": _sha256(script_path),
        },
        "git": _git_state(),
        "runtime": {
            "python": platform.python_version(),
            "anndata": ad.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": [
            {
                "role": role,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for role, path in zip(
                ("observed_h5ad", "communication_pickle", "lr_database"),
                input_paths,
            )
        ],
        "parameters": {
            "time_key": resolved_time_key,
            "annotation_key": annotation_key,
            "matrix_key": matrix_key,
            "observed_layer": ("X" if observed_layer is None else observed_layer),
            "observed_expression_space": observed_expression_space,
            "score_expression_space": score_expression_space,
            "preferred_species_tag": preferred_species_tag,
            "stages": selected_stages,
            "top_k_values": top_k_values,
            "complex_modes": ["min", "geometric_mean"],
            "geometric_mean_zero_preserving": True,
            "require_all_subunits": True,
            "same_communication_object_for_both_modes": True,
            "time_atol": TIME_ATOL,
        },
        "expression_audit": {
            **expression_audit,
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "preprocess_info": _jsonable(adata.uns.get("preprocess_info")),
        },
        "consensus_contract": {
            "external_consensus_constructed": False,
            "cytobridge_in_external_consensus": False,
            "reason": (
                "This is a within-CytoBridge aggregation sensitivity analysis, "
                "not an external-method consensus."
            ),
        },
        "result_summary": {
            "pooled_all_pair_score_spearman": all_row["pooled_score_spearman"],
            "macro_mean_stage_all_pair_score_spearman": all_row[
                "macro_mean_stage_score_spearman"
            ],
            "pooled_multisubunit_score_spearman": multi_row["pooled_score_spearman"],
            "pooled_primary_top_k_overlap_fraction": pooled_overlap[
                "overlap_fraction_of_k"
            ],
            "n_strict_eligible_pairs": int(
                eligibility["strict_all_subunits_eligible"].sum()
            ),
            "n_strict_eligible_multisubunit_pairs": int(
                eligibility.loc[
                    eligibility["strict_all_subunits_eligible"],
                    "is_multisubunit",
                ].sum()
            ),
        },
        "outputs": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            default=_jsonable,
        ),
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "paired_scores": paired,
        "stage_stability": by_stage,
        "overall_stability": overall,
        "top_k_overlap": top_overlap,
        "multisubunit_stability": multisubunit,
        "coverage": coverage,
        "eligibility": eligibility,
        "scaffold": scaffold,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--communications-pickle", type=Path, required=True)
    parser.add_argument("--lr-database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-key")
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument("--matrix-key", default="M_per_source")
    parser.add_argument(
        "--observed-layer",
        default="X",
        help="Observed expression layer; use X for adata.X (default).",
    )
    parser.add_argument(
        "--observed-expression-space",
        choices=("log1p", "count"),
        default="log1p",
    )
    parser.add_argument(
        "--score-expression-space",
        choices=("log1p", "count"),
        default="count",
        help=(
            "Space used for cell-type means. Default count converts each "
            "normalized log1p cell with expm1 before averaging."
        ),
    )
    parser.add_argument("--preferred-species-tag")
    parser.add_argument(
        "--stages",
        help="Optional comma-separated observed stages; default is every H5AD stage.",
    )
    parser.add_argument(
        "--top-k",
        default="10,20,50",
        help="Comma-separated exact top-k values.",
    )
    args = parser.parse_args(argv)
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    run_analysis(
        h5ad_path=args.h5ad,
        communications_path=args.communications_pickle,
        lr_database_path=args.lr_database,
        output_dir=args.output_dir,
        time_key=args.time_key,
        annotation_key=args.annotation_key,
        matrix_key=args.matrix_key,
        observed_layer=args.observed_layer,
        observed_expression_space=args.observed_expression_space,
        score_expression_space=args.score_expression_space,
        preferred_species_tag=args.preferred_species_tag,
        stages=_parse_float_list(args.stages),
        top_k_values=_parse_top_k(args.top_k),
        command=[str(Path(__file__).resolve()), *effective_argv],
    )
    print(Path(args.output_dir).expanduser().resolve() / "run_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
