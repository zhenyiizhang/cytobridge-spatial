#!/usr/bin/env python3
"""Prepare expression tables for temporal NicheNet.

The script reads linear expression from an AnnData object, normalizes and
log-transforms it once, and calculates receiver-specific differential response
from library-level pseudobulks.  It writes the four CSV files used by the R
analysis plus a JSON record of the inputs and settings.  The default intervals
are Weinreb Day 2 to Day 4 and Day 4 to Day 6.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student_t


def _parse_transitions(value: str) -> tuple[tuple[float, float], ...]:
    transitions: list[tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        pieces = item.split(":")
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError(
                "--transitions must look like '2:4,4:6'."
            )
        try:
            early, late = (float(piece.strip()) for piece in pieces)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "Transition endpoints must be numeric."
            ) from error
        if not (math.isfinite(early) and math.isfinite(late) and late > early):
            raise argparse.ArgumentTypeError(
                "Each transition must have finite endpoints with late > early."
            )
        transitions.append((early, late))
    if not transitions or len(set(transitions)) != len(transitions):
        raise argparse.ArgumentTypeError(
            "--transitions must contain unique numeric transitions."
        )
    return tuple(transitions)


def _time_token(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def _transition_token(early: float, late: float) -> str:
    return f"{_time_token(early)}_to_{_time_token(late)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-h5ad", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gene-symbol-key", default="gene")
    parser.add_argument("--time-key", default="Time point")
    parser.add_argument("--cell-type-key", default="Cell type annotation")
    parser.add_argument("--library-key", default="Library")
    parser.add_argument("--block-key", default="Starting population")
    parser.add_argument(
        "--transitions", type=_parse_transitions, default=((2.0, 4.0), (4.0, 6.0))
    )
    parser.add_argument("--target-sum", type=float, default=10_000.0)
    parser.add_argument("--minimum-cells-per-pseudobulk", type=int, default=5)
    parser.add_argument("--minimum-pseudobulks-per-time", type=int, default=2)
    parser.add_argument(
        "--expression-minimum-fraction",
        type=float,
        default=0.05,
        help="Detection fraction used for sender/receiver expressed-gene exports.",
    )
    parser.add_argument(
        "--background-minimum-fraction",
        type=float,
        default=0.05,
        help="Minimum receiver detection in either endpoint for DE background.",
    )
    parser.add_argument("--target-fdr", type=float, default=0.05)
    parser.add_argument("--target-minimum-effect", type=float, default=0.0)
    parser.add_argument("--minimum-targets", type=int, default=20)
    parser.add_argument("--fallback-target-count", type=int, default=200)
    parser.add_argument("--maximum-targets", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    digest = sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
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


def _matrix_minimum_and_finite(matrix: Any) -> tuple[float, bool]:
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    values = np.asarray(values)
    return (
        float(values.min()) if values.size else 0.0,
        bool(np.isfinite(values).all()),
    )


def _normalize_linear(matrix: Any, target_sum: float) -> sparse.csr_matrix:
    """Apply exactly one full-library normalize_total and log1p transform."""

    result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    result.eliminate_zeros()
    minimum, finite = _matrix_minimum_and_finite(result)
    if not finite or minimum < 0:
        raise ValueError("Expression X must be finite, non-negative linear expression.")
    library_sums = np.asarray(result.sum(axis=1), dtype=np.float64).reshape(-1)
    if not np.isfinite(library_sums).all() or np.any(library_sums <= 0):
        raise ValueError("Every cell must have a positive finite library sum.")
    result = sparse.diags(float(target_sum) / library_sums).dot(result).tocsr()
    np.log1p(result.data, out=result.data)
    return result


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg correction with deterministic finite-value handling."""

    p_values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.ones(p_values.shape, dtype=np.float64)
    finite = np.isfinite(p_values)
    if not finite.any():
        return adjusted
    observed = np.clip(p_values[finite], 0.0, 1.0)
    order = np.argsort(observed, kind="mergesort")
    ranked = observed[order]
    n = len(ranked)
    corrected = ranked * n / np.arange(1, n + 1, dtype=np.float64)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)
    inverse = np.empty(n, dtype=int)
    inverse[order] = np.arange(n)
    adjusted[finite] = corrected[inverse]
    return adjusted


def _fit_vectorized_ols(
    design: np.ndarray,
    response: np.ndarray,
    *,
    coefficient_index: int = 1,
) -> dict[str, np.ndarray | int]:
    """Fit all genes at once and test one coefficient with a t statistic."""

    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    if design.ndim != 2 or response.ndim != 2 or design.shape[0] != response.shape[0]:
        raise ValueError("Design and response must be aligned two-dimensional arrays.")
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError("Pseudobulk design is rank deficient.")
    residual_df = int(design.shape[0] - rank)
    if residual_df <= 0:
        raise ValueError("Pseudobulk design has no residual degrees of freedom.")
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ response
    residuals = response - design @ coefficients
    residual_variance = np.sum(residuals * residuals, axis=0) / residual_df
    variance = np.maximum(
        residual_variance * float(xtx_inverse[coefficient_index, coefficient_index]),
        0.0,
    )
    standard_error = np.sqrt(variance)
    effect = coefficients[coefficient_index]
    t_statistic = np.zeros_like(effect)
    nonzero_se = standard_error > 0
    t_statistic[nonzero_se] = effect[nonzero_se] / standard_error[nonzero_se]
    exact_nonzero = (~nonzero_se) & (effect != 0)
    t_statistic[exact_nonzero] = np.sign(effect[exact_nonzero]) * np.inf
    p_value = 2.0 * student_t.sf(np.abs(t_statistic), residual_df)
    p_value[(~nonzero_se) & (effect == 0)] = 1.0
    p_value = np.clip(p_value, 0.0, 1.0)
    return {
        "effect": effect,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "q_value": _benjamini_hochberg(p_value),
        "residual_df": residual_df,
        "rank": rank,
    }


def _group_expression(
    normalized: sparse.csr_matrix,
    linear: sparse.csr_matrix,
    indices: np.ndarray,
    labels: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Return mean log expression and linear nonzero detection by group."""

    result: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    selected_labels = labels[indices].astype(str)
    for label in sorted(set(selected_labels)):
        group_indices = indices[selected_labels == label]
        mean_expression = np.asarray(
            normalized[group_indices].mean(axis=0), dtype=np.float64
        ).reshape(-1)
        detected = linear[group_indices].copy()
        detected.data = np.ones_like(detected.data, dtype=np.float64)
        fraction = np.asarray(detected.mean(axis=0), dtype=np.float64).reshape(-1)
        result[label] = (mean_expression, fraction, int(len(group_indices)))
    return result


def _pseudobulk_means(
    normalized: sparse.csr_matrix,
    metadata: pd.DataFrame,
    *,
    minimum_cells: int,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Aggregate cells to equal-weight Library pseudobulks."""

    columns = ["receiver", "library", "time", "block"]
    groups: list[tuple[tuple[str, str, float, str], np.ndarray]] = []
    grouped = metadata.groupby(columns, observed=True, sort=True, dropna=False)
    for key, positions in grouped.indices.items():
        positions = np.asarray(positions, dtype=int)
        if len(positions) >= minimum_cells:
            groups.append((key, np.sort(positions)))
    if not groups:
        return sparse.csr_matrix((0, normalized.shape[1])), pd.DataFrame(
            columns=columns + ["n_cells"]
        )
    row_indices = np.concatenate(
        [np.full(len(positions), row, dtype=int) for row, (_, positions) in enumerate(groups)]
    )
    cell_indices = np.concatenate([positions for _, positions in groups])
    weights = np.concatenate(
        [np.full(len(positions), 1.0 / len(positions)) for _, positions in groups]
    )
    aggregator = sparse.csr_matrix(
        (weights, (row_indices, cell_indices)),
        shape=(len(groups), normalized.shape[0]),
    )
    values = aggregator.dot(normalized).tocsr()
    rows = [
        {
            "receiver": str(key[0]),
            "library": str(key[1]),
            "time": float(key[2]),
            "block": str(key[3]),
            "n_cells": int(len(positions)),
        }
        for key, positions in groups
    ]
    return values, pd.DataFrame(rows)


def _design_matrix(meta: pd.DataFrame, late: float) -> tuple[np.ndarray, list[str]]:
    late_indicator = np.isclose(
        meta["time"].to_numpy(float), float(late), rtol=0.0, atol=1e-12
    ).astype(float)
    pieces = [np.ones((len(meta), 1)), late_indicator[:, None]]
    names = ["intercept", "late_time"]
    block = pd.get_dummies(meta["block"].astype(str), drop_first=True, dtype=float)
    for column in block.columns:
        values = block[column].to_numpy(float)
        if np.any(values != values[0]):
            pieces.append(values[:, None])
            names.append(f"block[{column}]")
    return np.hstack(pieces), names


def _validate_args(args: argparse.Namespace) -> None:
    finite_positive = {
        "target_sum": args.target_sum,
    }
    for name, value in finite_positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive and finite.")
    for name in (
        "expression_minimum_fraction",
        "background_minimum_fraction",
        "target_fdr",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0, 1].")
    for name in (
        "minimum_cells_per_pseudobulk",
        "minimum_pseudobulks_per_time",
        "minimum_targets",
        "fallback_target_count",
        "maximum_targets",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")
    if args.minimum_targets > args.maximum_targets:
        raise ValueError("--minimum-targets cannot exceed --maximum-targets.")
    if args.fallback_target_count > args.maximum_targets:
        raise ValueError("--fallback-target-count cannot exceed --maximum-targets.")


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} is non-empty; pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare all temporal NicheNet inputs and return the written manifest."""

    _validate_args(args)
    expression_path = args.expression_h5ad.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not expression_path.is_file():
        raise FileNotFoundError(expression_path)
    _prepare_output(output_dir, bool(args.overwrite))

    adata = ad.read_h5ad(expression_path)
    required_obs = (
        args.time_key,
        args.cell_type_key,
        args.library_key,
        args.block_key,
    )
    missing_obs = [key for key in required_obs if key not in adata.obs]
    if missing_obs:
        raise KeyError(f"Missing adata.obs columns: {missing_obs}")
    if args.gene_symbol_key not in adata.var:
        raise KeyError(f"Missing adata.var[{args.gene_symbol_key!r}].")
    if adata.obs[list(required_obs)].isna().any().any():
        raise ValueError("Required observation metadata must not contain missing values.")

    genes = adata.var[args.gene_symbol_key].astype(str).to_numpy()
    duplicated = pd.Series(genes).duplicated(keep=False).to_numpy()
    if duplicated.any():
        examples = sorted(set(genes[duplicated]))[:5]
        raise ValueError(f"Gene symbols must be unique; duplicates include {examples}.")
    times = pd.to_numeric(adata.obs[args.time_key], errors="raise").to_numpy(float)
    if not np.isfinite(times).all():
        raise ValueError("Time values must be finite numbers.")
    labels = adata.obs[args.cell_type_key].astype(str).to_numpy()
    libraries = adata.obs[args.library_key].astype(str).to_numpy()
    blocks = adata.obs[args.block_key].astype(str).to_numpy()

    linear = sparse.csr_matrix(adata.X, dtype=np.float64, copy=True)
    linear.eliminate_zeros()
    minimum, finite = _matrix_minimum_and_finite(linear)
    if not finite or minimum < 0:
        raise ValueError("Expression X must be finite, non-negative linear expression.")
    library_sums = np.asarray(linear.sum(axis=1), dtype=np.float64).reshape(-1)
    normalized = _normalize_linear(linear, float(args.target_sum))

    de_frames: list[pd.DataFrame] = []
    background_frames: list[pd.DataFrame] = []
    sender_frames: list[pd.DataFrame] = []
    receptor_frames: list[pd.DataFrame] = []
    transition_summaries: dict[str, Any] = {}

    for early, late in args.transitions:
        transition = _transition_token(early, late)
        early_indices = np.flatnonzero(np.isclose(times, early, rtol=0.0, atol=1e-12))
        late_indices = np.flatnonzero(np.isclose(times, late, rtol=0.0, atol=1e-12))
        if not len(early_indices) or not len(late_indices):
            raise ValueError(
                f"Transition {transition} has {len(early_indices)} early and "
                f"{len(late_indices)} late cells."
            )

        early_expression = _group_expression(
            normalized, linear, early_indices, labels
        )
        late_expression = _group_expression(normalized, linear, late_indices, labels)
        for sender, (mean, fraction, n_cells) in early_expression.items():
            keep = fraction >= float(args.expression_minimum_fraction)
            sender_frames.append(
                pd.DataFrame(
                    {
                        "transition": transition,
                        "sender": sender,
                        "gene": genes[keep],
                        "mean_expression": mean[keep],
                        "expression_fraction": fraction[keep],
                        "n_cells": n_cells,
                    }
                )
            )

        transition_indices = np.concatenate([early_indices, late_indices])
        metadata = pd.DataFrame(
            {
                "receiver": labels[transition_indices].astype(str),
                "library": libraries[transition_indices].astype(str),
                "time": times[transition_indices].astype(float),
                "block": blocks[transition_indices].astype(str),
            }
        )
        transition_matrix = normalized[transition_indices]
        pseudobulk, pseudobulk_meta = _pseudobulk_means(
            transition_matrix,
            metadata,
            minimum_cells=int(args.minimum_cells_per_pseudobulk),
        )
        receiver_status: dict[str, Any] = {}
        common_receivers = sorted(set(early_expression).intersection(late_expression))
        for receiver in common_receivers:
            rows = np.flatnonzero(
                pseudobulk_meta["receiver"].astype(str).to_numpy() == receiver
            )
            receiver_meta = pseudobulk_meta.iloc[rows].reset_index(drop=True)
            n_early = int(
                np.isclose(
                    receiver_meta["time"].to_numpy(float), early, rtol=0.0, atol=1e-12
                ).sum()
            )
            n_late = int(
                np.isclose(
                    receiver_meta["time"].to_numpy(float), late, rtol=0.0, atol=1e-12
                ).sum()
            )
            status: dict[str, Any] = {
                "n_cells_early": int(early_expression[receiver][2]),
                "n_cells_late": int(late_expression[receiver][2]),
                "n_pseudobulks_early": n_early,
                "n_pseudobulks_late": n_late,
            }
            if min(n_early, n_late) < int(args.minimum_pseudobulks_per_time):
                status.update(
                    status="skipped",
                    reason="insufficient_pseudobulks_per_time",
                )
                receiver_status[receiver] = status
                continue
            design, design_columns = _design_matrix(receiver_meta, late)
            response = pseudobulk[rows].toarray()
            try:
                fit = _fit_vectorized_ols(design, response)
            except ValueError as error:
                status.update(status="skipped", reason=str(error))
                receiver_status[receiver] = status
                continue

            early_rows = np.isclose(
                receiver_meta["time"].to_numpy(float), early, rtol=0.0, atol=1e-12
            )
            mean_early = response[early_rows].mean(axis=0)
            mean_late = response[~early_rows].mean(axis=0)
            early_mean, early_fraction, early_cells = early_expression[receiver]
            _, late_fraction, _ = late_expression[receiver]
            in_background = (
                np.maximum(early_fraction, late_fraction)
                >= float(args.background_minimum_fraction)
            )
            significant = (
                in_background
                & (np.asarray(fit["q_value"]) <= float(args.target_fdr))
                & (np.asarray(fit["effect"]) > float(args.target_minimum_effect))
            )
            candidates = np.flatnonzero(significant)
            selection_mode = "fdr_positive"
            if len(candidates) < int(args.minimum_targets):
                candidates = np.flatnonzero(
                    in_background
                    & (
                        np.asarray(fit["effect"])
                        > float(args.target_minimum_effect)
                    )
                )
                selection_mode = "fallback_ranked_positive"
                limit = min(
                    int(args.fallback_target_count), int(args.maximum_targets)
                )
            else:
                limit = int(args.maximum_targets)
            candidate_order = sorted(
                candidates.tolist(),
                key=lambda index: (
                    float(np.asarray(fit["q_value"])[index]),
                    float(np.asarray(fit["p_value"])[index]),
                    -float(np.asarray(fit["effect"])[index]),
                    str(genes[index]),
                ),
            )[:limit]
            selected_target = np.zeros(len(genes), dtype=bool)
            selected_target[candidate_order] = True
            target_rank = np.full(len(genes), np.nan)
            if candidate_order:
                target_rank[candidate_order] = np.arange(1, len(candidate_order) + 1)
            gene_selection_mode = np.full(len(genes), "not_selected", dtype=object)
            gene_selection_mode[candidate_order] = selection_mode

            de_frames.append(
                pd.DataFrame(
                    {
                        "transition": transition,
                        "receiver": receiver,
                        "gene": genes,
                        "effect": np.asarray(fit["effect"]),
                        "standard_error": np.asarray(fit["standard_error"]),
                        "t_statistic": np.asarray(fit["t_statistic"]),
                        "p_value": np.asarray(fit["p_value"]),
                        "q_value": np.asarray(fit["q_value"]),
                        "mean_early": mean_early,
                        "mean_late": mean_late,
                        "n_pseudobulks_early": n_early,
                        "n_pseudobulks_late": n_late,
                        "in_background": in_background,
                        "selected_target": selected_target,
                        "target_selection_mode": gene_selection_mode,
                        "target_rank": target_rank,
                    }
                )
            )
            background_frames.append(
                pd.DataFrame(
                    {
                        "transition": transition,
                        "receiver": receiver,
                        "gene": genes[in_background],
                    }
                )
            )
            receptor_keep = early_fraction >= float(args.expression_minimum_fraction)
            receptor_frames.append(
                pd.DataFrame(
                    {
                        "transition": transition,
                        "receiver": receiver,
                        "gene": genes[receptor_keep],
                        "mean_expression": early_mean[receptor_keep],
                        "expression_fraction": early_fraction[receptor_keep],
                        "n_cells": early_cells,
                    }
                )
            )
            status.update(
                status="included",
                design_columns=design_columns,
                design_rank=int(fit["rank"]),
                residual_df=int(fit["residual_df"]),
                n_background_genes=int(in_background.sum()),
                n_fdr_positive_genes=int(significant.sum()),
                n_selected_targets=int(selected_target.sum()),
                target_selection_mode=selection_mode,
            )
            receiver_status[receiver] = status

        transition_summaries[transition] = {
            "early_time": float(early),
            "late_time": float(late),
            "n_cells_early": int(len(early_indices)),
            "n_cells_late": int(len(late_indices)),
            "n_pseudobulks_after_minimum_cell_filter": int(len(pseudobulk_meta)),
            "receivers": receiver_status,
        }

    if not de_frames:
        raise RuntimeError("No receiver passed the pseudobulk design requirements.")

    de = pd.concat(de_frames, ignore_index=True).sort_values(
        ["transition", "receiver", "gene"], kind="mergesort"
    )
    background = pd.concat(background_frames, ignore_index=True).drop_duplicates()
    background = background.sort_values(
        ["transition", "receiver", "gene"], kind="mergesort"
    )
    sender = pd.concat(sender_frames, ignore_index=True).sort_values(
        ["transition", "sender", "gene"], kind="mergesort"
    )
    receptors = pd.concat(receptor_frames, ignore_index=True).sort_values(
        ["transition", "receiver", "gene"], kind="mergesort"
    )
    outputs = {
        "receiver_de_genes.csv": de,
        "receiver_expressed_genes.csv": background,
        "sender_expressed_genes_long.csv": sender,
        "receiver_receptor_expression.csv": receptors,
    }
    artifacts: dict[str, Any] = {}
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False, float_format="%.17g")
        artifacts[filename] = {
            "path": str(path.resolve()),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "temporal_receiver_pseudobulk_nichenet_input",
        "input": {
            "expression_h5ad": {
                "path": str(expression_path),
                "sha256": _sha256_file(expression_path),
                "shape": [int(adata.n_obs), int(adata.n_vars)],
                "gene_order_sha256": _sha256_strings(genes),
            }
        },
        "keys": {
            "gene_symbol_key": args.gene_symbol_key,
            "time_key": args.time_key,
            "cell_type_key": args.cell_type_key,
            "library_key": args.library_key,
            "block_key": args.block_key,
        },
        "normalization": {
            "input_semantics": "finite non-negative linear expression",
            "library_size_scope": "all input genes before any gene filtering",
            "target_sum": float(args.target_sum),
            "normalize_total_applications": 1,
            "log1p_applications": 1,
            "pseudobulk_statistic": "mean of cell-level log1p normalized expression",
            "library_sum_before": {
                "min": float(library_sums.min()),
                "median": float(np.median(library_sums)),
                "max": float(library_sums.max()),
            },
        },
        "differential_expression": {
            "unit": "Library pseudobulk within receiver cell type",
            "model": "expression ~ intercept + late_time + categorical block",
            "effect": "late_time coefficient on mean log1p normalized expression",
            "multiple_testing": "Benjamini-Hochberg within transition and receiver",
            "minimum_cells_per_pseudobulk": int(args.minimum_cells_per_pseudobulk),
            "minimum_pseudobulks_per_time": int(args.minimum_pseudobulks_per_time),
        },
        "gene_selection": {
            "expression_minimum_fraction": float(args.expression_minimum_fraction),
            "background_definition": (
                "receiver gene detected in at least background_minimum_fraction "
                "of cells at either transition endpoint"
            ),
            "background_minimum_fraction": float(args.background_minimum_fraction),
            "target_fdr": float(args.target_fdr),
            "target_minimum_effect": float(args.target_minimum_effect),
            "minimum_targets_before_fallback": int(args.minimum_targets),
            "fallback": (
                "rank background genes with positive effect by q, p, decreasing "
                "effect, and gene name"
            ),
            "fallback_target_count": int(args.fallback_target_count),
            "maximum_targets": int(args.maximum_targets),
        },
        "transitions": transition_summaries,
        "data_usage": {
            "uses_spatial_coordinates": False,
            "uses_spring_coordinates": False,
            "uses_clone_or_lineage": False,
            "uses_fate_labels": False,
            "cell_type_annotations_used_only_for_sender_receiver_grouping": True,
        },
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_value(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_inputs(args)
    summary = {
        transition: {
            "included_receivers": sum(
                status.get("status") == "included"
                for status in details["receivers"].values()
            ),
            "skipped_receivers": sum(
                status.get("status") == "skipped"
                for status in details["receivers"].values()
            ),
        }
        for transition, details in manifest["transitions"].items()
    }
    print(json.dumps({"status": "ok", "transitions": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
