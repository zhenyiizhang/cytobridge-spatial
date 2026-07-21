#!/usr/bin/env python3
"""Validate reviewer-facing CCC consistency axes without treating attention as CCC.

This script consumes the edge-level table produced by
``analyze_attention_confound_controls.py``.  It adds three deliberately
separate analyses:

1. stage-by-sender-by-receiver context enrichment for database-supported LR
   co-activity among high-attention or high-exact-message contexts;
2. a stricter conditional randomization that additionally matches binned
   source out-degree and target in-degree, on top of time, cell identity,
   distance, and non-LR transcriptional state; and
3. an optional, hash-checked summary of an existing continuous pre-warp
   virtual cell-type-removal experiment.

The reported attention gate is a signed, non-softmax model gate and is not a
communication probability.  An exact GNN message is a model contribution and
is not a biochemical flux.  LR agreement and virtual removal are internal
consistency/sensitivity analyses, not independent experimental validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ATTENTION_RESIDUAL = "oof_log1p_attention_confounder_residual"
MESSAGE_RESIDUAL = "oof_log1p_edge_message_joint_confounder_residual"
ABLATION_FIGURE_NOTE = "full S24; one-seed model sensitivity; not a causal perturbation"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-controls", required=True, type=Path)
    parser.add_argument("--attention-controls-manifest", required=True, type=Path)
    parser.add_argument("--attribution-manifest", required=True, type=Path)
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--lr-database", required=True, type=Path)
    parser.add_argument(
        "--known-axis-provenance",
        type=Path,
        default=Path(__file__).with_name("known_zebrafish_axis_provenance.csv"),
        help=(
            "Exact database axes with primary-source provenance and deliberately "
            "narrow evidence-scope statements."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--ablation-run-dir",
        type=Path,
        help=(
            "Optional completed zebrafish paper-downstream run containing "
            "run_manifest.json and ablation/. It is reused only after hash and "
            "simulation-contract validation."
        ),
    )
    parser.add_argument("--context-min-edges", type=int, default=10)
    parser.add_argument("--high-quantile", type=float, default=0.80)
    parser.add_argument("--matching-bins", type=int, default=5)
    parser.add_argument("--degree-bins", type=int, default=3)
    parser.add_argument("--matching-min-stratum-size", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--top-axes-per-stage", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return payload


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}.")


def _resolve_column(frame: pd.DataFrame, names: Sequence[str]) -> str:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    for name in names:
        found = lookup.get(str(name).casefold())
        if found is not None:
            return found
    raise KeyError(
        f"Could not resolve any of {list(names)} from {list(frame.columns)}."
    )


def _validate_primary_inputs(
    *,
    edge_path: Path,
    controls_manifest_path: Path,
    attribution_manifest_path: Path,
    h5ad_path: Path,
    lr_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (
        edge_path,
        controls_manifest_path,
        attribution_manifest_path,
        h5ad_path,
        lr_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    controls = _load_json(controls_manifest_path)
    attribution = _load_json(attribution_manifest_path)
    controls_h5ad = controls.get("input", {}).get("h5ad", {})
    attribution_h5ad = attribution.get("input", {}).get("h5ad", {})
    actual_h5ad_sha = _sha256(h5ad_path)
    for label, record in (
        ("attention-controls", controls_h5ad),
        ("attribution", attribution_h5ad),
    ):
        expected = str(record.get("sha256", ""))
        if not expected or expected != actual_h5ad_sha:
            raise ValueError(
                f"{label} H5AD hash does not match the supplied H5AD: "
                f"expected={expected!r}, actual={actual_h5ad_sha!r}."
            )
    controls_lr = controls.get("input", {}).get("lr_database", {})
    expected_lr_sha = str(controls_lr.get("sha256", ""))
    actual_lr_sha = _sha256(lr_path)
    if not expected_lr_sha or expected_lr_sha != actual_lr_sha:
        raise ValueError(
            "Attention-controls LR database hash does not match the supplied "
            f"database: expected={expected_lr_sha!r}, actual={actual_lr_sha!r}."
        )
    edge_record = controls.get("artifacts", {}).get("edge_controls", {})
    expected_edge_sha = str(edge_record.get("sha256", ""))
    actual_edge_sha = _sha256(edge_path)
    if not expected_edge_sha or expected_edge_sha != actual_edge_sha:
        raise ValueError(
            "Edge-controls hash does not match its manifest: "
            f"expected={expected_edge_sha!r}, actual={actual_edge_sha!r}."
        )
    if attribution.get("interpretation", {}).get("probability_claim") is not False:
        raise ValueError(
            "Attribution manifest must explicitly reject a probability claim."
        )
    return controls, attribution


def summarize_sender_receiver_contexts(
    edges: pd.DataFrame, *, min_edges: int
) -> pd.DataFrame:
    """Aggregate edge evidence without changing the unit labels."""
    required = [
        "stage",
        "stage_label",
        "sender_type",
        "receiver_type",
        "source_index",
        "target_index",
        "attention_abs_mean",
        "edge_message_norm_joint",
        ATTENTION_RESIDUAL,
        MESSAGE_RESIDUAL,
        "lr_compatibility_forward",
        "lr_compatibility_reverse",
        "active_lr_count",
        "spatial_distance",
        "source_outdegree",
        "target_indegree",
    ]
    _require_columns(edges, required, "edge-controls table")
    keys = ["stage", "stage_label", "sender_type", "receiver_type"]
    grouped = edges.groupby(keys, sort=True, dropna=False)
    summary = grouped.agg(
        n_edges=("target_index", "size"),
        n_sender_cells=("source_index", "nunique"),
        n_receiver_cells=("target_index", "nunique"),
        attention_mean=("attention_abs_mean", "mean"),
        attention_median=("attention_abs_mean", "median"),
        exact_message_mean=("edge_message_norm_joint", "mean"),
        exact_message_median=("edge_message_norm_joint", "median"),
        attention_confounder_residual_mean=(ATTENTION_RESIDUAL, "mean"),
        exact_message_confounder_residual_mean=(MESSAGE_RESIDUAL, "mean"),
        lr_forward_mean=("lr_compatibility_forward", "mean"),
        lr_reverse_mean=("lr_compatibility_reverse", "mean"),
        active_lr_count_mean=("active_lr_count", "mean"),
        lr_supported_edge_fraction=(
            "lr_compatibility_forward",
            lambda values: float(np.mean(np.asarray(values, dtype=float) > 0)),
        ),
        spatial_distance_mean=("spatial_distance", "mean"),
        source_outdegree_mean=("source_outdegree", "mean"),
        target_indegree_mean=("target_indegree", "mean"),
    ).reset_index()
    summary["passes_min_edges"] = summary["n_edges"] >= int(min_edges)
    rank_columns = {
        "attention_mean": "attention_within_stage_percentile",
        "exact_message_mean": "exact_message_within_stage_percentile",
        "attention_confounder_residual_mean": (
            "attention_residual_within_stage_percentile"
        ),
        "exact_message_confounder_residual_mean": (
            "exact_message_residual_within_stage_percentile"
        ),
        "lr_forward_mean": "lr_forward_within_stage_percentile",
    }
    for source, target in rank_columns.items():
        summary[target] = summary.groupby("stage", sort=False)[source].rank(
            method="average", pct=True
        )
    return summary


def _within_stratum_centered_ranks(
    values: np.ndarray, groups: Sequence[np.ndarray]
) -> np.ndarray:
    from scipy.stats import rankdata

    result = np.zeros(len(values), dtype=float)
    for indices in groups:
        ranked = rankdata(np.asarray(values)[indices], method="average").astype(float)
        result[indices] = ranked - ranked.mean()
    return result


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg q-values for one declared family."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(
            "Benjamini-Hochberg input must be a finite one-dimensional array."
        )
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Benjamini-Hochberg p-values must lie in [0, 1].")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _conditional_rank_permutation(
    frame: pd.DataFrame,
    *,
    target: str,
    score: str,
    keys: Sequence[str],
    min_stratum_size: int,
    n_permutations: int,
    random_state: int,
) -> dict[str, Any]:
    """Within-stratum rank association with the score permuted in each stratum."""
    _require_columns(frame, [target, score, *keys], "conditional-test table")
    grouped = frame.groupby(list(keys), sort=False, dropna=False)
    indices = [
        np.asarray(value, dtype=int)
        for value in grouped.indices.values()
        if len(value) >= int(min_stratum_size)
        and frame.iloc[value][target].nunique(dropna=True) > 1
        and frame.iloc[value][score].nunique(dropna=True) > 1
    ]
    retained = np.zeros(len(frame), dtype=bool)
    for value in indices:
        retained[value] = True
    subset = frame.loc[retained].reset_index(drop=True)
    if subset.empty:
        raise ValueError("No non-degenerate strata survived the conditional test.")
    # Recreate indices after resetting the row index.
    groups = [
        np.asarray(value, dtype=int)
        for value in subset.groupby(
            list(keys), sort=False, dropna=False
        ).indices.values()
        if len(value) >= int(min_stratum_size)
    ]
    x = _within_stratum_centered_ranks(subset[target].to_numpy(float), groups)
    y = _within_stratum_centered_ranks(subset[score].to_numpy(float), groups)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 0:
        raise ValueError("Conditional rank statistic has a zero denominator.")
    observed = float(np.dot(x, y) / denominator)
    rng = np.random.default_rng(int(random_state))
    null = np.empty(int(n_permutations), dtype=float)
    permuted = np.empty_like(y)
    for iteration in range(int(n_permutations)):
        for value in groups:
            permuted[value] = y[rng.permutation(value)]
        null[iteration] = float(np.dot(x, permuted) / denominator)
    return {
        "target": target,
        "score": score,
        "strata": "+".join(keys),
        "min_stratum_size": int(min_stratum_size),
        "n_edges_total": int(len(frame)),
        "n_edges_retained": int(len(subset)),
        "retained_fraction": float(len(subset) / len(frame)),
        "n_strata": int(len(groups)),
        "conditional_rank_correlation": observed,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)),
        "observed_minus_null_mean": float(observed - np.mean(null)),
        "empirical_p_greater": float(
            (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
        ),
        "n_permutations": int(n_permutations),
    }


def _quantile_bin_by_stage(frame: pd.DataFrame, source: str, bins: int) -> pd.Series:
    def transform(values: pd.Series) -> pd.Series:
        q = min(int(bins), len(values))
        if q < 2 or values.nunique(dropna=True) < 2:
            return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
        return pd.qcut(
            values.rank(method="first"), q=q, labels=False, duplicates="drop"
        ).astype(int)

    return frame.groupby("stage", sort=False)[source].transform(transform)


def run_degree_matched_tests(
    edges: pd.DataFrame,
    *,
    matching_bins: int,
    degree_bins: int,
    min_stratum_size: int,
    n_permutations: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Condition on proximity, degree, time, identity, and non-LR state."""
    result = edges.copy().reset_index(drop=True)
    sources = (
        ("spatial_distance", "distance_match_bin", int(matching_bins)),
        ("residual_non_lr_pca_cosine", "state_match_bin", int(matching_bins)),
        ("source_outdegree", "source_degree_match_bin", int(degree_bins)),
        ("target_indegree", "target_degree_match_bin", int(degree_bins)),
    )
    _require_columns(result, [item[0] for item in sources], "edge-controls table")
    for source, target, bins in sources:
        result[target] = _quantile_bin_by_stage(result, source, bins)
    keys = [
        "stage",
        "sender_type",
        "receiver_type",
        "distance_match_bin",
        "state_match_bin",
        "source_degree_match_bin",
        "target_degree_match_bin",
    ]
    rows: list[dict[str, Any]] = []
    for target in (ATTENTION_RESIDUAL, MESSAGE_RESIDUAL):
        for score in ("lr_compatibility_forward", "lr_compatibility_reverse"):
            rows.append(
                _conditional_rank_permutation(
                    result,
                    target=target,
                    score=score,
                    keys=keys,
                    min_stratum_size=int(min_stratum_size),
                    n_permutations=int(n_permutations),
                    random_state=int(random_state),
                )
            )
    bin_audit = (
        result.groupby(keys, sort=True, dropna=False)
        .agg(
            n_edges=("target_index", "size"),
            distance_min=("spatial_distance", "min"),
            distance_max=("spatial_distance", "max"),
            source_degree_min=("source_outdegree", "min"),
            source_degree_max=("source_outdegree", "max"),
            target_degree_min=("target_indegree", "min"),
            target_degree_max=("target_indegree", "max"),
        )
        .reset_index()
    )
    tests = pd.DataFrame(rows)
    tests["bh_q_within_degree_matched_family"] = _benjamini_hochberg(
        tests["empirical_p_greater"]
    )
    return tests, bin_audit


def _context_enrichment_test(
    contexts: pd.DataFrame,
    *,
    target: str,
    score: str,
    quantile: float,
    n_permutations: int,
    random_state: int,
) -> dict[str, Any]:
    """Stage-stratified context-level top-versus-low enrichment."""
    from scipy.stats import rankdata

    frame = contexts.loc[contexts["passes_min_edges"]].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No sender-receiver contexts pass the edge-count threshold.")
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("high quantile must be strictly between 0.5 and 1.")
    groups = [
        np.asarray(value, dtype=int)
        for value in frame.groupby("stage", sort=False).indices.values()
    ]

    def statistic(
        target_values: np.ndarray,
    ) -> tuple[float, float, float, int, int, float]:
        high_mask = np.zeros(len(frame), dtype=bool)
        low_mask = np.zeros(len(frame), dtype=bool)
        centered_target = np.zeros(len(frame), dtype=float)
        centered_score = np.zeros(len(frame), dtype=float)
        for indices in groups:
            target_rank = rankdata(target_values[indices], method="average")
            target_pct = target_rank / len(indices)
            high_mask[indices] = target_pct >= float(quantile)
            low_mask[indices] = target_pct <= (1.0 - float(quantile) + 1e-12)
            score_rank = rankdata(
                frame.iloc[indices][score].to_numpy(float), method="average"
            )
            centered_target[indices] = target_rank - target_rank.mean()
            centered_score[indices] = score_rank - score_rank.mean()
        if not high_mask.any() or not low_mask.any():
            raise ValueError("Context quantiles produced an empty high or low set.")
        high = float(frame.loc[high_mask, score].mean())
        low = float(frame.loc[low_mask, score].mean())
        denominator = np.linalg.norm(centered_target) * np.linalg.norm(centered_score)
        correlation = (
            float(np.dot(centered_target, centered_score) / denominator)
            if denominator
            else float("nan")
        )
        return (
            high - low,
            high,
            low,
            int(high_mask.sum()),
            int(low_mask.sum()),
            correlation,
        )

    target_values = frame[target].to_numpy(float)
    observed, high, low, n_high, n_low, correlation = statistic(target_values)
    rng = np.random.default_rng(int(random_state))
    null = np.empty(int(n_permutations), dtype=float)
    permuted = target_values.copy()
    for iteration in range(int(n_permutations)):
        for indices in groups:
            permuted[indices] = target_values[rng.permutation(indices)]
        null[iteration] = statistic(permuted)[0]
    return {
        "target": target,
        "score": score,
        "stratification": "stage",
        "context_definition": "stage+sender_type+receiver_type",
        "minimum_edges_observed_in_retained_contexts": int(
            contexts.loc[contexts["passes_min_edges"], "n_edges"].min()
        ),
        "high_quantile": float(quantile),
        "n_contexts": int(len(frame)),
        "n_high_contexts": n_high,
        "n_low_contexts": n_low,
        "mean_score_high_contexts": high,
        "mean_score_low_contexts": low,
        "high_minus_low_score": observed,
        "within_stage_rank_correlation": correlation,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)),
        "empirical_p_greater": float(
            (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
        ),
        "n_permutations": int(n_permutations),
    }


def run_context_enrichment_tests(
    contexts: pd.DataFrame,
    *,
    quantile: float,
    n_permutations: int,
    random_state: int,
) -> pd.DataFrame:
    targets = (
        "attention_mean",
        "exact_message_mean",
        "attention_confounder_residual_mean",
        "exact_message_confounder_residual_mean",
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        for score in ("lr_forward_mean", "lr_reverse_mean"):
            rows.append(
                _context_enrichment_test(
                    contexts,
                    target=target,
                    score=score,
                    quantile=float(quantile),
                    n_permutations=int(n_permutations),
                    random_state=int(random_state),
                )
            )
    tests = pd.DataFrame(rows)
    tests["bh_q_within_context_family"] = _benjamini_hochberg(
        tests["empirical_p_greater"]
    )
    return tests


def _subunits(token: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in str(token).split("_") if part.strip())
    if not values:
        raise ValueError(f"Empty complex token: {token!r}.")
    return values


def _load_unique_lr_axes(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    ligand = _resolve_column(raw, ("ligand", "0"))
    receptor = _resolve_column(raw, ("receptor", "1"))
    pathway = _resolve_column(raw, ("pathway", "2"))
    category = _resolve_column(raw, ("category", "annotation", "3"))
    database_rows = pd.to_numeric(
        raw.get("Unnamed: 0", pd.Series(np.arange(len(raw)))), errors="raise"
    ).astype(int)
    table = pd.DataFrame(
        {
            "database_row": database_rows,
            "ligand": raw[ligand].astype(str).str.strip(),
            "receptor": raw[receptor].astype(str).str.strip(),
            "pathway": raw[pathway].astype(str).str.strip(),
            "category": raw[category].astype(str).str.strip(),
        }
    )
    if (table[["ligand", "receptor", "pathway", "category"]] == "").any().any():
        raise ValueError("LR database contains empty identifiers.")

    def joined(values: pd.Series) -> str:
        return ";".join(sorted(set(values.astype(str))))

    return (
        table.groupby(["ligand", "receptor"], sort=True, as_index=False)
        .agg(
            database_rows=(
                "database_row",
                lambda values: ";".join(map(str, sorted(set(values)))),
            ),
            pathways=("pathway", joined),
            categories=("category", joined),
        )
        .assign(axis_id=lambda value: value["ligand"] + "->" + value["receptor"])
    )


def _casefold_gene_index(
    var_names: Iterable[object],
) -> tuple[dict[str, int], set[str]]:
    candidates: dict[str, list[int]] = {}
    for index, gene in enumerate(var_names):
        candidates.setdefault(str(gene).casefold(), []).append(index)
    ambiguous = {key for key, values in candidates.items() if len(values) != 1}
    unique = {key: values[0] for key, values in candidates.items() if len(values) == 1}
    return unique, ambiguous


def _complex_activity(matrix, indices: Sequence[int]) -> np.ndarray:
    values = matrix[:, list(indices)]
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    values = np.asarray(values, dtype=np.float32)
    return np.min(values, axis=1)


def _available_axes_and_activities(
    data, axes: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    gene_index, ambiguous = _casefold_gene_index(data.var_names)
    keep: list[bool] = []
    audit_rows: list[dict[str, Any]] = []
    tokens: set[str] = set()
    for row in axes.itertuples(index=False):
        required = (*_subunits(row.ligand), *_subunits(row.receptor))
        missing = [gene for gene in required if gene.casefold() not in gene_index]
        ambiguous_genes = [gene for gene in required if gene.casefold() in ambiguous]
        available = not missing and not ambiguous_genes
        keep.append(available)
        if available:
            tokens.update((row.ligand, row.receptor))
        audit_rows.append(
            {
                "axis_id": row.axis_id,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "available": bool(available),
                "missing_subunits": ";".join(missing),
                "ambiguous_subunits": ";".join(ambiguous_genes),
            }
        )
    available_axes = axes.loc[keep].reset_index(drop=True)
    if available_axes.empty:
        raise ValueError("No unambiguous LR axes are available in the H5AD.")
    activities: dict[str, np.ndarray] = {}
    for token in sorted(tokens):
        indices = [gene_index[gene.casefold()] for gene in _subunits(token)]
        raw = _complex_activity(data.X, indices)
        positive = raw[raw > 0]
        scale = float(np.quantile(positive, 0.95)) if positive.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        activities[token] = np.clip(raw / scale, 0.0, 1.0).astype(np.float32)
    return available_axes, activities, pd.DataFrame(audit_rows)


def score_identifiable_lr_axes(
    edges: pd.DataFrame,
    axes: pd.DataFrame,
    activities: Mapping[str, np.ndarray],
    *,
    top_per_stage: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank database-identifiable axes descriptively, without literature claims."""
    _require_columns(
        edges,
        [
            "stage",
            "stage_label",
            "sender_type",
            "receiver_type",
            "source_index",
            "target_index",
            "attention_abs_mean",
            "edge_message_norm_joint",
        ],
        "edge-controls table",
    )
    source = edges["source_index"].to_numpy(int)
    target = edges["target_index"].to_numpy(int)
    attention = edges["attention_abs_mean"].to_numpy(float)
    message = edges["edge_message_norm_joint"].to_numpy(float)
    context_keys = ["stage", "stage_label", "sender_type", "receiver_type"]
    context_code, context_values = pd.factorize(
        pd.MultiIndex.from_frame(edges[context_keys]), sort=True
    )
    context_table = context_values.to_frame(index=False)
    context_table.columns = context_keys
    rows: list[dict[str, Any]] = []
    stage_values = sorted(edges["stage"].unique())
    stage_masks = {stage: edges["stage"].to_numpy() == stage for stage in stage_values}
    for axis in axes.itertuples(index=False):
        product = activities[axis.ligand][source] * activities[axis.receptor][target]
        attention_product = product * attention
        message_product = product * message
        attention_context = np.bincount(
            context_code, weights=attention_product, minlength=len(context_table)
        )
        message_context = np.bincount(
            context_code, weights=message_product, minlength=len(context_table)
        )
        for stage in stage_values:
            mask = stage_masks[stage]
            n_edges = int(mask.sum())
            stage_context = context_table["stage"].to_numpy() == stage
            attention_candidates = np.flatnonzero(stage_context)
            message_candidates = attention_candidates
            best_attention_code = int(
                attention_candidates[np.argmax(attention_context[attention_candidates])]
            )
            best_message_code = int(
                message_candidates[np.argmax(message_context[message_candidates])]
            )
            best_attention = context_table.iloc[best_attention_code]
            best_message = context_table.iloc[best_message_code]
            has_attention_context = bool(attention_context[best_attention_code] > 0)
            has_message_context = bool(message_context[best_message_code] > 0)
            values = product[mask]
            attention_values = attention_product[mask]
            message_values = message_product[mask]
            mean_product = float(np.mean(values))
            attention_denominator = float(np.sum(attention[mask]))
            message_denominator = float(np.sum(message[mask]))
            rows.append(
                {
                    "stage": stage,
                    "stage_label": str(edges.loc[mask, "stage_label"].iloc[0]),
                    "axis_id": axis.axis_id,
                    "ligand": axis.ligand,
                    "receptor": axis.receptor,
                    "database_rows": axis.database_rows,
                    "pathways": axis.pathways,
                    "categories": axis.categories,
                    "n_model_edges": n_edges,
                    "n_active_edges": int(np.count_nonzero(values > 0)),
                    "active_edge_fraction": float(np.mean(values > 0)),
                    "mean_scaled_lr_activity": mean_product,
                    "mean_attention_times_lr_activity": float(
                        np.mean(attention_values)
                    ),
                    "mean_exact_message_times_lr_activity": float(
                        np.mean(message_values)
                    ),
                    "attention_weighted_mean_lr_activity": (
                        float(np.sum(attention_values) / attention_denominator)
                        if attention_denominator > 0
                        else np.nan
                    ),
                    "exact_message_weighted_mean_lr_activity": (
                        float(np.sum(message_values) / message_denominator)
                        if message_denominator > 0
                        else np.nan
                    ),
                    "top_attention_sender_type": (
                        best_attention["sender_type"] if has_attention_context else ""
                    ),
                    "top_attention_receiver_type": (
                        best_attention["receiver_type"] if has_attention_context else ""
                    ),
                    "top_exact_message_sender_type": (
                        best_message["sender_type"] if has_message_context else ""
                    ),
                    "top_exact_message_receiver_type": (
                        best_message["receiver_type"] if has_message_context else ""
                    ),
                }
            )
    scores = pd.DataFrame(rows)
    rank_specs = {
        "attention": "mean_attention_times_lr_activity",
        "exact_message": "mean_exact_message_times_lr_activity",
    }
    selected: list[pd.DataFrame] = []
    for target_name, score in rank_specs.items():
        ranked = scores.sort_values(
            ["stage", score, "axis_id"], ascending=[True, False, True]
        ).copy()
        ranked["rank"] = ranked.groupby("stage").cumcount() + 1
        ranked = ranked.loc[ranked["rank"] <= int(top_per_stage)].copy()
        ranked.insert(2, "ranking_target", target_name)
        ranked.insert(4, "ranking_score_column", score)
        selected.append(ranked)
    top = pd.concat(selected, ignore_index=True)
    top[
        "interpretation"
    ] = "database-identifiable descriptive axis; no literature-validation claim"
    return scores, top


def attach_known_axis_provenance(
    provenance_path: Path,
    axes: pd.DataFrame,
    axis_scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tie narrowly scoped literature provenance to exact database rows."""
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    provenance = pd.read_csv(provenance_path, dtype=str).fillna("")
    required = [
        "ligand",
        "receptor",
        "evidence_scope",
        "source_ids",
        "source_urls",
        "claim_guardrail",
    ]
    _require_columns(provenance, required, "known-axis provenance")
    if provenance[required].apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError("Known-axis provenance contains an empty required value.")
    if provenance.duplicated(["ligand", "receptor"]).any():
        raise ValueError("Known-axis provenance contains duplicate exact LR pairs.")
    if (
        not provenance["source_urls"]
        .str.split(";")
        .map(
            lambda values: all(
                value.startswith(("https://", "http://")) for value in values
            )
        )
        .all()
    ):
        raise ValueError("Every known-axis source URL must be an explicit HTTP(S) URL.")
    database = axes[
        [
            "ligand",
            "receptor",
            "axis_id",
            "database_rows",
            "pathways",
            "categories",
        ]
    ]
    audit = provenance.merge(
        database,
        on=["ligand", "receptor"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    audit["database_present"] = audit["_merge"].eq("both")
    audit = audit.drop(columns="_merge")
    if not audit["database_present"].all():
        missing = audit.loc[~audit["database_present"], ["ligand", "receptor"]].to_dict(
            "records"
        )
        raise ValueError(
            "Known-axis provenance is not tied to the supplied LR database: "
            f"{missing}."
        )
    stage_scores = audit.merge(
        axis_scores,
        on=[
            "axis_id",
            "ligand",
            "receptor",
            "database_rows",
            "pathways",
            "categories",
        ],
        how="left",
        validate="one_to_many",
    )
    if stage_scores["stage"].isna().any():
        raise ValueError("A known database axis is unavailable in the corrected H5AD.")
    return audit, stage_scores


def _find_artifact_hash(stage_manifest: Mapping[str, Any], suffix: str) -> str:
    matches = [
        record
        for record in stage_manifest.get("output_artifacts", [])
        if str(record.get("path", "")).endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one stage-manifest artifact ending {suffix!r}; "
            f"found {len(matches)}."
        )
    return str(matches[0].get("sha256", ""))


def validate_ablation_bundle(
    run_dir: Path,
    *,
    expected_h5ad_sha256: str,
    expected_model_weight_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load an existing virtual-removal run only after fail-closed validation."""
    run_dir = run_dir.expanduser().resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    stage_manifest_path = run_dir / "ablation" / "stage_manifest.json"
    experiment_dir = run_dir / "ablation" / "experiment"
    experiment_manifest_path = experiment_dir / "manifest.json"
    metrics_path = experiment_dir / "ablation_metrics.csv"
    composition_path = experiment_dir / "label_composition.csv"
    for path in (
        run_manifest_path,
        stage_manifest_path,
        experiment_manifest_path,
        metrics_path,
        composition_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_manifest = _load_json(run_manifest_path)
    stage_manifest = _load_json(stage_manifest_path)
    experiment_manifest = _load_json(experiment_manifest_path)
    actual_h5ad = str(run_manifest.get("common", {}).get("aligned_h5ad_sha256", ""))
    actual_weight = str(run_manifest.get("common", {}).get("weight_sha256", ""))
    if actual_h5ad != str(expected_h5ad_sha256):
        raise ValueError(
            "Ablation H5AD hash does not match the reviewer analysis: "
            f"expected={expected_h5ad_sha256!r}, found={actual_h5ad!r}."
        )
    if actual_weight != str(expected_model_weight_sha256):
        raise ValueError(
            "Ablation model-weight hash does not match the attribution model: "
            f"expected={expected_model_weight_sha256!r}, found={actual_weight!r}."
        )
    if run_manifest.get("profile") != "full" or "ablation" not in run_manifest.get(
        "completed_stages", []
    ):
        raise ValueError("Ablation reuse requires a completed full-profile run.")
    settings = experiment_manifest.get("settings", {})
    simulation = str(settings.get("simulation", "")).casefold()
    required_phrases = ("no re-anchoring", "no spatial warp", "no replacement")
    if any(phrase not in simulation for phrase in required_phrases):
        raise ValueError(
            "Ablation simulation is not the required continuous pre-warp contract: "
            f"{settings.get('simulation')!r}."
        )
    if settings.get("common_random_seed") is not True:
        raise ValueError("Ablation branches must declare common_random_seed=true.")
    if (
        settings.get("concat_spatial") is not True
        or int(settings.get("spatial_dim", -1)) != 2
    ):
        raise ValueError("Ablation must use joint spatial2+latent state.")
    if not np.isclose(float(settings.get("start_time", np.nan)), 0.0):
        raise ValueError("Ablation must start from observed stage 0.")
    for path, suffix in (
        (metrics_path, "/ablation_metrics.csv"),
        (composition_path, "/label_composition.csv"),
        (experiment_manifest_path, "/manifest.json"),
    ):
        expected = _find_artifact_hash(stage_manifest, suffix)
        actual = _sha256(path)
        if expected != actual:
            raise ValueError(
                f"Ablation artifact hash mismatch for {path}: "
                f"expected={expected!r}, actual={actual!r}."
            )
    metadata = {
        "run_manifest": _artifact(run_manifest_path),
        "stage_manifest": _artifact(stage_manifest_path),
        "experiment_manifest": _artifact(experiment_manifest_path),
        "metrics": _artifact(metrics_path),
        "label_composition": _artifact(composition_path),
        "simulation": settings.get("simulation"),
        "common_random_seed": settings.get("common_random_seed"),
        "random_stream_coupling": settings.get("random_stream_coupling"),
        "simulation_seeds": settings.get("simulation_seeds"),
        "ablations": settings.get("ablations"),
        "n_initial": settings.get("n_initial"),
        "variant_initial_counts": settings.get("variant_initial_counts"),
        "single_seed_no_uncertainty": True,
    }
    return pd.read_csv(metrics_path), pd.read_csv(composition_path), metadata


def summarize_virtual_ablation(
    metrics: pd.DataFrame, composition: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_metrics = [
        "variant",
        "time",
        "space",
        "n_baseline",
        "n_ablation",
        "count_ratio",
        "centroid_shift",
        "baseline_rms_radius",
    ]
    required_composition = [
        "variant",
        "time",
        "label",
        "baseline_fraction",
        "ablation_fraction",
    ]
    _require_columns(metrics, required_metrics, "ablation metrics")
    _require_columns(composition, required_composition, "ablation composition")
    table = metrics.copy()
    table["normalized_centroid_shift"] = np.divide(
        table["centroid_shift"].to_numpy(float),
        table["baseline_rms_radius"].to_numpy(float),
        out=np.full(len(table), np.nan),
        where=table["baseline_rms_radius"].to_numpy(float) > 0,
    )
    initial = (
        table.sort_values("time")
        .groupby(["variant", "space"], sort=False)["normalized_centroid_shift"]
        .transform("first")
    )
    table["normalized_shift_minus_t0"] = table["normalized_centroid_shift"] - initial
    composition_tv = (
        composition.assign(
            absolute_fraction_delta=lambda value: np.abs(
                value["ablation_fraction"] - value["baseline_fraction"]
            )
        )
        .groupby(["variant", "time"], sort=True, as_index=False)[
            "absolute_fraction_delta"
        ]
        .sum()
        .rename(columns={"absolute_fraction_delta": "twice_total_variation"})
    )
    composition_tv["composition_total_variation"] = (
        0.5 * composition_tv["twice_total_variation"]
    )
    table = table.merge(
        composition_tv[["variant", "time", "composition_total_variation"]],
        on=["variant", "time"],
        how="left",
        validate="many_to_one",
    )
    observed = table.loc[
        np.isclose(table["time"], np.round(table["time"]), atol=1e-10)
    ].copy()
    rows: list[dict[str, Any]] = []
    for (variant, space), frame in table.groupby(["variant", "space"], sort=True):
        frame = frame.sort_values("time")
        time = frame["time"].to_numpy(float)
        normalized = frame["normalized_centroid_shift"].to_numpy(float)
        dynamic = frame["normalized_shift_minus_t0"].to_numpy(float)
        rows.append(
            {
                "variant": variant,
                "space": space,
                "n_timepoints": int(len(frame)),
                "start_time": float(time[0]),
                "end_time": float(time[-1]),
                "initial_count_ratio": float(frame["count_ratio"].iloc[0]),
                "endpoint_count_ratio": float(frame["count_ratio"].iloc[-1]),
                "initial_normalized_centroid_shift": float(normalized[0]),
                "endpoint_normalized_centroid_shift": float(normalized[-1]),
                "endpoint_normalized_shift_minus_t0": float(dynamic[-1]),
                "max_normalized_centroid_shift": float(np.nanmax(normalized)),
                "auc_normalized_centroid_shift": float(np.trapz(normalized, time)),
                "auc_normalized_shift_minus_t0": float(np.trapz(dynamic, time)),
                "endpoint_composition_total_variation": float(
                    frame["composition_total_variation"].iloc[-1]
                ),
            }
        )
    return pd.DataFrame(rows), observed


def _verified_stage_label_map(frame: pd.DataFrame) -> dict[float, str]:
    """Require one observed label per numeric model stage."""
    _require_columns(frame, ["stage", "stage_label"], "stage-label table")
    table = frame[["stage", "stage_label"]].copy()
    table["stage"] = pd.to_numeric(table["stage"], errors="raise")
    table["stage_label"] = table["stage_label"].astype(str).str.strip()
    if table["stage_label"].eq("").any():
        raise ValueError("Observed stage labels must be non-empty.")
    counts = table.groupby("stage", sort=True)["stage_label"].nunique()
    if (counts != 1).any():
        conflicts = counts.loc[counts != 1].index.astype(float).tolist()
        raise ValueError(f"Numeric stages map to multiple labels: {conflicts}.")
    unique = table.drop_duplicates(["stage", "stage_label"]).sort_values("stage")
    if unique["stage_label"].duplicated().any():
        duplicates = unique.loc[
            unique["stage_label"].duplicated(keep=False), "stage_label"
        ].tolist()
        raise ValueError(f"Observed stage labels map to multiple stages: {duplicates}.")
    return {
        float(row.stage): str(row.stage_label) for row in unique.itertuples(index=False)
    }


def _observed_stage_axis_spec(
    observed_times: Sequence[float], stage_label_by_time: Mapping[float, str]
) -> dict[str, Any]:
    """Use verified hpf labels, otherwise explicitly label stage indices."""
    ticks = sorted(set(float(value) for value in observed_times))
    labels: list[str] = []
    for tick in ticks:
        matches = [
            str(label)
            for stage, label in stage_label_by_time.items()
            if np.isclose(float(stage), tick, rtol=0.0, atol=1e-10)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Observed ablation time {tick:g} does not map to exactly one stage label."
            )
        labels.append(matches[0])
    hpf_pattern = re.compile(r"^(?:\d+(?:\.\d+)?|\.\d+)hpf$", re.IGNORECASE)
    verified_hpf = bool(labels) and all(
        hpf_pattern.fullmatch(label) for label in labels
    )
    if verified_hpf:
        display_labels = [
            re.sub(r"hpf$", " hpf", label, flags=re.IGNORECASE) for label in labels
        ]
        xlabel = "observed developmental stage"
    else:
        display_labels = [f"stage {tick:g}" for tick in ticks]
        xlabel = "observed stage index"
    return {
        "ticks": ticks,
        "labels": display_labels,
        "source_labels": labels,
        "verified_hpf_labels": verified_hpf,
        "xlabel": xlabel,
    }


def _plot_validation(
    context_tests: pd.DataFrame,
    matched_tests: pd.DataFrame,
    observed_ablation: pd.DataFrame | None,
    *,
    stage_label_by_time: Mapping[float, str],
    output_png: Path,
    output_pdf: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.8), constrained_layout=True)
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.055, 1.0, 0.96))
    target_order = [
        "attention_mean",
        "exact_message_mean",
        "attention_confounder_residual_mean",
        "exact_message_confounder_residual_mean",
    ]
    labels = ["attention", "exact message", "attention residual", "message residual"]
    forward = (
        context_tests.loc[context_tests["score"] == "lr_forward_mean"]
        .set_index("target")
        .reindex(target_order)
    )
    axes[0, 0].bar(
        np.arange(len(target_order)),
        forward["high_minus_low_score"],
        color=["#4C78A8", "#F58518", "#72B7B2", "#E45756"],
    )
    axes[0, 0].axhline(0, color="black", lw=0.8)
    axes[0, 0].set_xticks(np.arange(len(target_order)), labels, rotation=25, ha="right")
    axes[0, 0].set_ylabel("high − low mean forward LR score")
    axes[0, 0].set_title("Sender–receiver context enrichment")

    targets = [ATTENTION_RESIDUAL, MESSAGE_RESIDUAL]
    x = np.arange(len(targets))
    width = 0.35
    for offset, score, color, label in (
        (-width / 2, "lr_compatibility_forward", "#4C78A8", "forward LR"),
        (width / 2, "lr_compatibility_reverse", "#B279A2", "reverse control"),
    ):
        values = (
            matched_tests.loc[matched_tests["score"] == score]
            .set_index("target")
            .reindex(targets)
        )
        axes[0, 1].bar(
            x + offset,
            values["observed_minus_null_mean"],
            width,
            yerr=values["null_sd"],
            color=color,
            label=label,
        )
    axes[0, 1].axhline(0, color="black", lw=0.8)
    axes[0, 1].set_xticks(x, ["attention residual", "message residual"])
    axes[0, 1].set_ylabel("conditional rank effect − null mean")
    axes[0, 1].set_title("Matched beyond proximity and degree")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].text(
        0.01,
        0.99,
        "error bars: 1 null SD",
        transform=axes[0, 1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )

    if observed_ablation is None or observed_ablation.empty:
        for axis in axes[1]:
            axis.text(0.5, 0.5, "No ablation bundle supplied", ha="center", va="center")
            axis.set_axis_off()
    else:
        stage_axis = _observed_stage_axis_spec(
            observed_ablation["time"].unique(), stage_label_by_time
        )
        spatial = observed_ablation.loc[observed_ablation["space"] == "spatial"]
        for variant, frame in spatial.groupby("variant", sort=True):
            axes[1, 0].plot(
                frame["time"],
                frame["normalized_centroid_shift"],
                marker="o",
                label=str(variant),
            )
        axes[1, 0].set_xticks(stage_axis["ticks"], stage_axis["labels"])
        axes[1, 0].set_xlabel(stage_axis["xlabel"])
        axes[1, 0].set_ylabel("centroid shift / baseline RMS radius")
        axes[1, 0].set_title("Virtual-removal spatial sensitivity")
        axes[1, 0].legend(frameon=False)
        tv = observed_ablation.loc[observed_ablation["space"] == "joint"]
        for variant, frame in tv.groupby("variant", sort=True):
            axes[1, 1].plot(
                frame["time"],
                frame["composition_total_variation"],
                marker="o",
                label=str(variant),
            )
        axes[1, 1].set_xticks(stage_axis["ticks"], stage_axis["labels"])
        axes[1, 1].set_xlabel(stage_axis["xlabel"])
        axes[1, 1].set_ylabel("label-composition total variation")
        axes[1, 1].set_title("Virtual-removal label sensitivity")
        axes[1, 1].legend(frameon=False)
    figure.suptitle(
        "CytoBridge reviewer checks: internal consistency, not CCC probabilities",
        fontsize=13,
    )
    if observed_ablation is not None and not observed_ablation.empty:
        figure.text(
            0.5,
            0.014,
            ABLATION_FIGURE_NOTE,
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
        )
    figure.savefig(output_png, dpi=220)
    figure.savefig(output_pdf)
    plt.close(figure)


def _format_float(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "NA" if not np.isfinite(parsed) else f"{parsed:.4g}"


def _write_markdown_summary(
    path: Path,
    *,
    context_tests: pd.DataFrame,
    matched_tests: pd.DataFrame,
    ablation_summary: pd.DataFrame | None,
    n_axes: int,
) -> None:
    lines = [
        "# Zebrafish reviewer validation axes",
        "",
        "## Scope and claim boundary",
        "",
        (
            "CytoBridge attention is a signed, non-softmax model gate, not a cell–cell "
            "communication probability. The exact spatial-GNN message is an exactly "
            "reconstructed model contribution, not biochemical flux. LR agreement and "
            "virtual cell-type removal are internal consistency/sensitivity checks; they "
            "are not independent experimental or causal validation."
        ),
        "",
        "## Sender–receiver context enrichment",
        "",
        (
            "Contexts are stage × sender cell type × receiver cell type. High and low "
            "contexts are selected within stage; residual targets use the prior "
            "out-of-fold confounder model."
        ),
        "",
        "| target | LR orientation | high−low LR score | within-stage rho | empirical p | BH q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in context_tests.itertuples(index=False):
        lines.append(
            "| {target} | {score} | {delta} | {rho} | {p} | {q} |".format(
                target=row.target,
                score=row.score,
                delta=_format_float(row.high_minus_low_score),
                rho=_format_float(row.within_stage_rank_correlation),
                p=_format_float(row.empirical_p_greater),
                q=_format_float(row.bh_q_within_context_family),
            )
        )
    lines.extend(
        [
            "",
            "## Spatial localization beyond proximity",
            "",
            (
                "The conditional randomization matches time, sender type, receiver type, "
                "spatial-distance bin, non-LR state-similarity bin, source out-degree bin, "
                "and target in-degree bin. Targets are out-of-fold residuals from the "
                "broader confounder model."
            ),
            "",
            "| target residual | LR orientation | conditional rho | rho−null | empirical p | BH q | retained edges |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matched_tests.itertuples(index=False):
        lines.append(
            "| {target} | {score} | {rho} | {delta} | {p} | {q} | {n} |".format(
                target=row.target,
                score=row.score,
                rho=_format_float(row.conditional_rank_correlation),
                delta=_format_float(row.observed_minus_null_mean),
                p=_format_float(row.empirical_p_greater),
                q=_format_float(row.bh_q_within_degree_matched_family),
                n=int(row.n_edges_retained),
            )
        )
    lines.extend(
        [
            "",
            "## Database-identifiable axes",
            "",
            (
                f"The full table contains {n_axes} available LR-axis × stage rows. "
                "`top_identifiable_lr_axes.csv` ranks expression-supported axes using "
                "attention×LR activity or exact-message×LR activity. These labels come "
                "from the supplied LR database; no literature-validation claim is made."
            ),
            "",
            "## Virtual-removal sensitivity",
            "",
        ]
    )
    if ablation_summary is None:
        lines.append("No compatible virtual-ablation bundle was supplied.")
    else:
        lines.extend(
            [
                (
                    "The reused run is full-data, continuous split-SDE from observed t0, "
                    "pre-warp, without re-anchoring or replacement. It uses one common "
                    "branch seed, so no replicate uncertainty is available. The t0 shift "
                    "is reported separately because simply removing a cell population "
                    "changes the cohort immediately."
                ),
                "",
                "| variant | space | endpoint normalized shift | endpoint minus t0 | endpoint composition TV |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in ablation_summary.itertuples(index=False):
            lines.append(
                "| {variant} | {space} | {endpoint} | {dynamic} | {tv} |".format(
                    variant=row.variant,
                    space=row.space,
                    endpoint=_format_float(row.endpoint_normalized_centroid_shift),
                    dynamic=_format_float(row.endpoint_normalized_shift_minus_t0),
                    tv=_format_float(row.endpoint_composition_total_variation),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- The frozen edge classifier is LR-informed, so LR enrichment is partly circular and is not independent validation.",
            "- Forward-versus-reverse controls must be reported together; a stronger reverse result does not support biochemical directionality.",
            "- Edge rows share cells and are not independent biological replicates; empirical p-values describe the specified conditional randomization only.",
            "- Virtual removal changes initial cell composition and particle count. It is a model sensitivity analysis, not a genetic perturbation estimate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    import anndata as ad

    edge_path = args.edge_controls.expanduser().resolve()
    controls_manifest_path = args.attention_controls_manifest.expanduser().resolve()
    attribution_manifest_path = args.attribution_manifest.expanduser().resolve()
    h5ad_path = args.h5ad.expanduser().resolve()
    lr_path = args.lr_database.expanduser().resolve()
    controls_manifest, attribution_manifest = _validate_primary_inputs(
        edge_path=edge_path,
        controls_manifest_path=controls_manifest_path,
        attribution_manifest_path=attribution_manifest_path,
        h5ad_path=h5ad_path,
        lr_path=lr_path,
    )
    output = _prepare_output(args.output_dir, bool(args.overwrite))
    edges = pd.read_csv(edge_path)
    contexts = summarize_sender_receiver_contexts(
        edges, min_edges=int(args.context_min_edges)
    )
    stage_label_by_time = _verified_stage_label_map(contexts)
    contexts_path = output / "sender_receiver_contexts.csv"
    contexts.to_csv(contexts_path, index=False)
    context_tests = run_context_enrichment_tests(
        contexts,
        quantile=float(args.high_quantile),
        n_permutations=int(args.permutations),
        random_state=int(args.random_state),
    )
    context_tests_path = output / "context_enrichment_tests.csv"
    context_tests.to_csv(context_tests_path, index=False)
    matched_tests, matching_audit = run_degree_matched_tests(
        edges,
        matching_bins=int(args.matching_bins),
        degree_bins=int(args.degree_bins),
        min_stratum_size=int(args.matching_min_stratum_size),
        n_permutations=int(args.permutations),
        random_state=int(args.random_state),
    )
    matched_tests_path = output / "degree_matched_conditional_tests.csv"
    matched_tests.to_csv(matched_tests_path, index=False)
    matching_audit_path = output / "degree_matching_strata_audit.csv.gz"
    matching_audit.to_csv(matching_audit_path, index=False, compression="gzip")

    data = ad.read_h5ad(h5ad_path)
    axes = _load_unique_lr_axes(lr_path)
    available_axes, activities, axis_audit = _available_axes_and_activities(data, axes)
    axis_scores, top_axes = score_identifiable_lr_axes(
        edges,
        available_axes,
        activities,
        top_per_stage=int(args.top_axes_per_stage),
    )
    axis_audit_path = output / "lr_axis_availability_audit.csv"
    axis_audit.to_csv(axis_audit_path, index=False)
    axis_scores_path = output / "lr_axis_stage_scores.csv.gz"
    axis_scores.to_csv(axis_scores_path, index=False, compression="gzip")
    top_axes_path = output / "top_identifiable_lr_axes.csv"
    top_axes.to_csv(top_axes_path, index=False)
    known_axis_path = args.known_axis_provenance.expanduser().resolve()
    known_axis_audit, known_axis_scores = attach_known_axis_provenance(
        known_axis_path, available_axes, axis_scores
    )
    known_axis_audit_path = output / "known_axis_database_provenance.csv"
    known_axis_scores_path = output / "known_axis_stage_scores.csv"
    known_axis_audit.to_csv(known_axis_audit_path, index=False)
    known_axis_scores.to_csv(known_axis_scores_path, index=False)

    ablation_summary: pd.DataFrame | None = None
    observed_ablation: pd.DataFrame | None = None
    ablation_metadata: dict[str, Any] | None = None
    ablation_artifacts: dict[str, Any] = {}
    if args.ablation_run_dir is not None:
        weight_sha = str(
            attribution_manifest.get("checkpoint", {})
            .get("weight", {})
            .get("sha256", "")
        )
        if not weight_sha:
            raise ValueError(
                "Attribution manifest does not provide the model weight hash."
            )
        metrics, composition, ablation_metadata = validate_ablation_bundle(
            args.ablation_run_dir,
            expected_h5ad_sha256=_sha256(h5ad_path),
            expected_model_weight_sha256=weight_sha,
        )
        ablation_summary, observed_ablation = summarize_virtual_ablation(
            metrics, composition
        )
        ablation_summary_path = output / "virtual_ablation_summary.csv"
        observed_ablation_path = output / "virtual_ablation_observed_stages.csv"
        ablation_summary.to_csv(ablation_summary_path, index=False)
        observed_ablation.to_csv(observed_ablation_path, index=False)
        ablation_artifacts = {
            "virtual_ablation_summary": _artifact(ablation_summary_path),
            "virtual_ablation_observed_stages": _artifact(observed_ablation_path),
        }

    figure_png = output / "reviewer_validation_axes.png"
    figure_pdf = output / "reviewer_validation_axes.pdf"
    _plot_validation(
        context_tests,
        matched_tests,
        observed_ablation,
        stage_label_by_time=stage_label_by_time,
        output_png=figure_png,
        output_pdf=figure_pdf,
    )
    summary_path = output / "reviewer_validation_summary.md"
    _write_markdown_summary(
        summary_path,
        context_tests=context_tests,
        matched_tests=matched_tests,
        ablation_summary=ablation_summary,
        n_axes=int(len(axis_scores)),
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "zebrafish_reviewer_ccc_internal_consistency_axes",
        "claims": {
            "attention_is_ccc_probability": False,
            "exact_message_is_biochemical_flux": False,
            "lr_database_agreement_is_independent_validation": False,
            "virtual_ablation_is_causal_perturbation": False,
            "known_axis_literature_claim": False,
        },
        "input": {
            "edge_controls": _artifact(edge_path),
            "attention_controls_manifest": _artifact(controls_manifest_path),
            "attribution_manifest": _artifact(attribution_manifest_path),
            "h5ad": _artifact(h5ad_path),
            "lr_database": _artifact(lr_path),
            "known_axis_provenance": _artifact(known_axis_path),
        },
        "context_enrichment": {
            "unit": "stage+sender_type+receiver_type",
            "minimum_edges": int(args.context_min_edges),
            "high_quantile": float(args.high_quantile),
            "permutation": "permute target context scores within stage",
            "permutations": int(args.permutations),
            "multiple_testing": "Benjamini-Hochberg across the 8 context tests",
        },
        "spatial_localization_control": {
            "target": "out-of-fold residual from the broader confounders-only model",
            "matched_factors": [
                "stage",
                "sender_type",
                "receiver_type",
                "spatial_distance_bin",
                "non_lr_state_similarity_bin",
                "source_outdegree_bin",
                "target_indegree_bin",
            ],
            "matching_bins": int(args.matching_bins),
            "degree_bins": int(args.degree_bins),
            "minimum_stratum_size": int(args.matching_min_stratum_size),
            "permutations": int(args.permutations),
            "forward_and_reverse_reported_together": True,
            "multiple_testing": (
                "Benjamini-Hochberg across the 4 degree-matched tests"
            ),
        },
        "lr_axis_ranking": {
            "complex_rule": "minimum across underscore-delimited subunits",
            "activity_scale": "global positive q95 then clip to [0,1]",
            "database_rows_are_collapsed_by_unique_ligand_receptor": True,
            "top_per_stage_per_target": int(args.top_axes_per_stage),
            "literature_claim": False,
            "known_axis_scope": (
                "primary sources support developmental relevance of the listed "
                "gene/pathway members only; exact pair, cell-type direction, and "
                "CytoBridge ranking are not literature-validated"
            ),
        },
        "virtual_ablation_reuse": ablation_metadata,
        "figure_stage_axis": (
            _observed_stage_axis_spec(
                observed_ablation["time"].unique(), stage_label_by_time
            )
            if observed_ablation is not None and not observed_ablation.empty
            else {
                "stage_label_by_time": {
                    str(stage): label for stage, label in stage_label_by_time.items()
                },
                "verified_hpf_labels": False,
                "xlabel": "observed stage index",
            }
        ),
        "limitations": [
            "the frozen edge classifier is LR-informed, creating partial circularity",
            "edge rows share cells and are not independent biological replicates",
            "single-model attention and exact messages do not estimate CCC probabilities",
            "virtual removal changes the starting cohort and is a one-seed sensitivity analysis",
            "database-identifiable top axes were not asserted as literature-validated zebrafish axes",
        ],
        "artifacts": {
            "sender_receiver_contexts": _artifact(contexts_path),
            "context_enrichment_tests": _artifact(context_tests_path),
            "degree_matched_conditional_tests": _artifact(matched_tests_path),
            "degree_matching_strata_audit": _artifact(matching_audit_path),
            "lr_axis_availability_audit": _artifact(axis_audit_path),
            "lr_axis_stage_scores": _artifact(axis_scores_path),
            "top_identifiable_lr_axes": _artifact(top_axes_path),
            "known_axis_database_provenance": _artifact(known_axis_audit_path),
            "known_axis_stage_scores": _artifact(known_axis_scores_path),
            "figure_png": _artifact(figure_png),
            "figure_pdf": _artifact(figure_pdf),
            "markdown_summary": _artifact(summary_path),
            **ablation_artifacts,
        },
        "upstream_claims": controls_manifest.get("claims"),
    }
    manifest_path = output / "run_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(json.dumps({"status": "ok", "artifacts": manifest["artifacts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
