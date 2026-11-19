#!/usr/bin/env python3
"""Resolve the Figure 5c reviewer question with model-native local networks.

The accepted Figure 5c-right cosine map is treated as immutable.  This script
links its supported spatial domains to the *fresh model's* selected attention
edges and to the package-native ligand-receptor score

    mean_source(ligand) * mean_target(receptor) * attention_per_source_cell.

Only H1-H3 and L1-L2 are retained because the other extreme-cosine domains
have no internal selected model edges.  The supported domains form three
network modules: H1, H2/L1, and H3/L2.  A cell-type-stratified permutation
tests whether each module concentrates more selected attention than an equally
sized random subset of the same 5-DPI injured dorsal ROI.

This is an observed-time, single-section mechanistic characterization.  It is
not a replicate-level differential-expression test and it does not turn model
attention into a biophysical signaling rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


SUPPORTED_DOMAINS = ("H1", "H2", "H3", "L1", "L2")
MODULE_DOMAINS = {
    "M1_H1": ("H1",),
    "M2_H2_L1": ("H2", "L1"),
    "M3_H3_L2": ("H3", "L2"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-h5ad", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--edge-index", type=Path, required=True)
    parser.add_argument("--attention", type=Path, required=True)
    parser.add_argument("--lr-database", type=Path, required=True)
    parser.add_argument("--pair-timecourse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--n-lr-permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_human_symbol(value: object) -> str | None:
    """Mirror selected_feature_symbol(..., preferred_species_tag='hs')."""
    candidates: list[tuple[str, str | None]] = []
    for raw in str(value).split("|"):
        token = raw.strip()
        if not token or token.casefold() == "nan" or token.upper().startswith("AMEX"):
            continue
        match = re.match(r"^(.*?)(?:\[([^\]]+)\])?$", token)
        if match is None:
            continue
        symbol = match.group(1).strip()
        species = match.group(2)
        if symbol:
            candidates.append((symbol, species.casefold() if species else None))
    selected = next((symbol for symbol, species in candidates if species == "hs"), None)
    if selected is not None:
        return selected
    if any(species is not None for _, species in candidates):
        return None
    if candidates:
        return candidates[0][0]
    fallback = str(value).strip()
    return fallback or None


def _complex_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in str(value).split("_") if token.strip())


def _combine_min(
    token: str,
    symbol_expression: dict[str, np.ndarray],
) -> np.ndarray | None:
    subunits = _complex_tokens(token)
    if not subunits or any(subunit not in symbol_expression for subunit in subunits):
        return None
    return np.min(np.stack([symbol_expression[subunit] for subunit in subunits]), axis=0)


def _load_lr_table(database_path: Path, pair_timecourse_path: Path) -> pd.DataFrame:
    database = pd.read_csv(database_path)
    usable = [column for column in database.columns if not str(column).lower().startswith("unnamed")]
    if len(usable) < 2:
        raise ValueError(f"Could not identify LR columns: {database.columns.tolist()}")
    ligand_col, receptor_col = usable[:2]
    pathway_col = usable[2] if len(usable) >= 3 else None
    class_col = usable[3] if len(usable) >= 4 else None
    rename = {ligand_col: "ligand", receptor_col: "receptor"}
    if pathway_col is not None:
        rename[pathway_col] = "pathway"
    if class_col is not None:
        rename[class_col] = "interaction_class"
    result = database[list(rename)].rename(columns=rename).copy()
    result["ligand"] = result["ligand"].astype(str).str.strip()
    result["receptor"] = result["receptor"].astype(str).str.strip()
    if "pathway" not in result:
        result["pathway"] = "Unannotated"
    if "interaction_class" not in result:
        result["interaction_class"] = "Unannotated"
    result["pathway"] = result["pathway"].fillna("Unannotated").astype(str)
    result["interaction_class"] = result["interaction_class"].fillna("Unannotated").astype(str)
    result = result.drop_duplicates(["ligand", "receptor"], keep="first")

    scored = pd.read_csv(pair_timecourse_path, usecols=["ligand", "receptor"])
    scored = scored.drop_duplicates().copy()
    result = result.merge(scored, on=["ligand", "receptor"], how="inner", validate="1:1")
    if len(result) != len(scored):
        missing = scored.merge(result, on=["ligand", "receptor"], how="left", indicator=True)
        missing = missing.loc[missing["_merge"].eq("left_only")]
        raise AssertionError(f"Scored LR pairs missing database annotations: {missing.head().to_dict('records')}")
    result["pair"] = result["ligand"] + "_" + result["receptor"]
    return result.reset_index(drop=True)


def _mean_columns(matrix: sparse.csr_matrix, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(matrix.shape[1], dtype=np.float64)
    subset = matrix[mask]
    return np.asarray(subset.mean(axis=0)).reshape(-1).astype(np.float64)


def _bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def _symbol_expression_by_cell(
    adata,
    global_cell_indices: np.ndarray,
    requested_symbols: set[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    mapped = pd.DataFrame(
        {
            "feature_index": np.arange(adata.n_vars, dtype=int),
            "var_name": adata.var_names.astype(str).to_numpy(),
        }
    )
    mapped["gene_symbol"] = [
        _selected_human_symbol(value) for value in mapped["var_name"].astype(str)
    ]
    mapped = mapped.loc[mapped["gene_symbol"].isin(requested_symbols)].copy()
    if mapped.empty:
        raise ValueError("No aligned-h5ad features overlap the scored LR subunits.")
    selected_indices = mapped["feature_index"].to_numpy(dtype=int)
    expression = adata.X[global_cell_indices, selected_indices]
    if not sparse.issparse(expression):
        expression = sparse.csr_matrix(expression)
    expression = expression.tocsr().astype(np.float64)
    # Package-native observed LR means convert each log1p cell to count space
    # before aggregation; zeros remain zeros.
    expression.data = np.clip(np.expm1(expression.data), 0.0, None)
    expression.eliminate_zeros()
    if not np.isfinite(expression.data).all():
        raise ValueError("Observed LR expression conversion produced non-finite values.")

    symbol_expression: dict[str, np.ndarray] = {}
    coverage_rows: list[dict[str, Any]] = []
    for symbol, rows in mapped.groupby("gene_symbol", sort=True):
        local_columns = mapped.index.get_indexer(rows.index)
        if np.any(local_columns < 0):
            raise AssertionError("Internal feature indexing failure.")
        values = expression[:, local_columns]
        mean_across_duplicates = np.asarray(values.mean(axis=1)).reshape(-1)
        symbol_expression[str(symbol)] = mean_across_duplicates.astype(np.float64)
        coverage_rows.append(
            {
                "gene_symbol": str(symbol),
                "n_feature_rows": int(len(rows)),
                "var_names": ";".join(rows["var_name"].astype(str)),
                "n_detected_cells_t1": int(np.count_nonzero(mean_across_duplicates > 0)),
            }
        )
    return symbol_expression, pd.DataFrame(coverage_rows)


def _type_pair_attention(
    mask_source: np.ndarray,
    mask_target: np.ndarray,
    labels: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    attention: np.ndarray,
) -> pd.DataFrame:
    selected = mask_source[edge_source] & mask_target[edge_target]
    if not np.any(selected):
        return pd.DataFrame(
            columns=[
                "sender", "receiver", "n_edges", "attention_sum",
                "n_source_cells", "attention_per_source_cell",
            ]
        )
    table = pd.DataFrame(
        {
            "sender": labels[edge_source[selected]],
            "receiver": labels[edge_target[selected]],
            "attention": attention[selected],
        }
    )
    result = (
        table.groupby(["sender", "receiver"], sort=True)["attention"]
        .agg(n_edges="size", attention_sum="sum")
        .reset_index()
    )
    source_counts = pd.Series(labels[mask_source]).value_counts()
    result["n_source_cells"] = result["sender"].map(source_counts).astype(int)
    result["attention_per_source_cell"] = (
        result["attention_sum"] / result["n_source_cells"].clip(lower=1)
    )
    return result


def _expression_by_type(
    mask: np.ndarray,
    labels: np.ndarray,
    symbol_expression: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for cell_type in sorted(np.unique(labels[mask])):
        selected = mask & (labels == str(cell_type))
        output[str(cell_type)] = {
            symbol: float(np.mean(values[selected]))
            for symbol, values in symbol_expression.items()
        }
    return output


def _score_lr_for_domain_pair(
    *,
    domain_source: str,
    domain_target: str,
    module: str,
    mask_source: np.ndarray,
    mask_target: np.ndarray,
    labels: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    attention: np.ndarray,
    lr_table: pd.DataFrame,
    symbol_expression: dict[str, np.ndarray],
    roi_type_pair: pd.DataFrame,
    roi_expression_by_type: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    type_edges = _type_pair_attention(
        mask_source, mask_target, labels, edge_source, edge_target, attention
    )
    if type_edges.empty:
        return pd.DataFrame(), type_edges
    source_expression = _expression_by_type(mask_source, labels, symbol_expression)
    target_expression = _expression_by_type(mask_target, labels, symbol_expression)
    roi_edge_lookup = roi_type_pair.set_index(["sender", "receiver"])[
        "attention_per_source_cell"
    ].to_dict()

    rows: list[dict[str, Any]] = []
    for edge in type_edges.itertuples(index=False):
        sender = str(edge.sender)
        receiver = str(edge.receiver)
        sender_values = source_expression[sender]
        receiver_values = target_expression[receiver]
        roi_sender_values = roi_expression_by_type.get(sender, {})
        roi_receiver_values = roi_expression_by_type.get(receiver, {})
        roi_comm = float(roi_edge_lookup.get((sender, receiver), 0.0))
        for pair in lr_table.itertuples(index=False):
            ligand_subunits = _complex_tokens(str(pair.ligand))
            receptor_subunits = _complex_tokens(str(pair.receptor))
            if any(token not in sender_values for token in ligand_subunits):
                continue
            if any(token not in receiver_values for token in receptor_subunits):
                continue
            ligand_mean = float(min(sender_values[token] for token in ligand_subunits))
            receptor_mean = float(min(receiver_values[token] for token in receptor_subunits))
            local_score = ligand_mean * receptor_mean * float(edge.attention_per_source_cell)
            roi_ligand = (
                float(min(roi_sender_values.get(token, 0.0) for token in ligand_subunits))
                if ligand_subunits else 0.0
            )
            roi_receptor = (
                float(min(roi_receiver_values.get(token, 0.0) for token in receptor_subunits))
                if receptor_subunits else 0.0
            )
            roi_score = roi_ligand * roi_receptor * roi_comm
            relative = local_score / roi_score if roi_score > 0 else np.nan
            rows.append(
                {
                    "module": module,
                    "source_domain": domain_source,
                    "target_domain": domain_target,
                    "sender": sender,
                    "receiver": receiver,
                    "n_edges": int(edge.n_edges),
                    "attention_sum": float(edge.attention_sum),
                    "n_source_cells": int(edge.n_source_cells),
                    "attention_per_source_cell": float(edge.attention_per_source_cell),
                    "ligand": str(pair.ligand),
                    "receptor": str(pair.receptor),
                    "pair": str(pair.pair),
                    "pathway": str(pair.pathway),
                    "interaction_class": str(pair.interaction_class),
                    "ligand_mean_count": ligand_mean,
                    "receptor_mean_count": receptor_mean,
                    "local_lr_score": local_score,
                    "roi_attention_per_source_cell": roi_comm,
                    "roi_ligand_mean_count": roi_ligand,
                    "roi_receptor_mean_count": roi_receptor,
                    "roi_lr_score": roi_score,
                    "local_to_roi_score_ratio": relative,
                    "log2_local_to_roi_score_ratio": (
                        float(np.log2(relative)) if np.isfinite(relative) and relative > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows), type_edges


def _permutation_module_attention(
    *,
    module: str,
    actual_mask: np.ndarray,
    roi_mask: np.ndarray,
    labels: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    attention: np.ndarray,
    n_permutations: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    actual_edge_mask = actual_mask[edge_source] & actual_mask[edge_target]
    actual_n = int(actual_mask.sum())
    actual_edges = int(actual_edge_mask.sum())
    actual_attention = float(attention[actual_edge_mask].sum())
    actual_per_cell = actual_attention / max(actual_n, 1)
    type_counts = pd.Series(labels[actual_mask]).value_counts().sort_index()
    pools = {
        str(cell_type): np.flatnonzero(roi_mask & (labels == str(cell_type)))
        for cell_type in type_counts.index
    }
    rng = np.random.default_rng(seed)
    null_edges = np.empty(n_permutations, dtype=np.int32)
    null_attention = np.empty(n_permutations, dtype=np.float64)
    null_per_cell = np.empty(n_permutations, dtype=np.float64)
    sampled_mask = np.zeros(len(labels), dtype=bool)
    for iteration in range(n_permutations):
        sampled_mask[:] = False
        for cell_type, count in type_counts.items():
            pool = pools[str(cell_type)]
            if int(count) > len(pool):
                raise AssertionError(f"Permutation pool too small for {module}/{cell_type}")
            sampled_mask[rng.choice(pool, size=int(count), replace=False)] = True
        selected = sampled_mask[edge_source] & sampled_mask[edge_target]
        null_edges[iteration] = int(selected.sum())
        null_attention[iteration] = float(attention[selected].sum())
        null_per_cell[iteration] = null_attention[iteration] / max(actual_n, 1)
    summary = {
        "module": module,
        "n_cells": actual_n,
        "n_celltypes": int(len(type_counts)),
        "celltype_counts": ";".join(f"{key}:{int(value)}" for key, value in type_counts.items()),
        "observed_internal_edges": actual_edges,
        "null_internal_edges_mean": float(null_edges.mean()),
        "null_internal_edges_sd": float(null_edges.std(ddof=1)),
        "internal_edges_empirical_p_greater": float(
            (1 + np.count_nonzero(null_edges >= actual_edges)) / (n_permutations + 1)
        ),
        "observed_attention_sum": actual_attention,
        "null_attention_sum_mean": float(null_attention.mean()),
        "null_attention_sum_sd": float(null_attention.std(ddof=1)),
        "attention_sum_empirical_p_greater": float(
            (1 + np.count_nonzero(null_attention >= actual_attention)) / (n_permutations + 1)
        ),
        "observed_attention_per_cell": actual_per_cell,
        "null_attention_per_cell_mean": float(null_per_cell.mean()),
        "null_attention_per_cell_sd": float(null_per_cell.std(ddof=1)),
        "attention_per_cell_empirical_p_greater": float(
            (1 + np.count_nonzero(null_per_cell >= actual_per_cell)) / (n_permutations + 1)
        ),
        "n_permutations": int(n_permutations),
        "null_sampling": "ROI cells sampled without replacement within each cell type",
    }
    null = pd.DataFrame(
        {
            "module": module,
            "permutation": np.arange(1, n_permutations + 1, dtype=int),
            "internal_edges": null_edges,
            "attention_sum": null_attention,
            "attention_per_cell": null_per_cell,
        }
    )
    return summary, null


def _module_lr_pathway_scores(
    *,
    mask: np.ndarray,
    labels: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    attention: np.ndarray,
    symbol_names: list[str],
    symbol_matrix: np.ndarray,
    lr_table: pd.DataFrame,
    ligand_indices: list[np.ndarray],
    receptor_indices: list[np.ndarray],
) -> np.ndarray:
    """Return package-formula LR score summed within each pathway."""
    cell_types = sorted(np.unique(labels[mask]))
    type_index = {cell_type: index for index, cell_type in enumerate(cell_types)}
    type_codes = np.full(len(labels), -1, dtype=int)
    for cell_type, index in type_index.items():
        type_codes[labels == cell_type] = index
    type_counts = np.asarray(
        [np.count_nonzero(mask & (labels == cell_type)) for cell_type in cell_types],
        dtype=float,
    )
    expression = np.zeros((len(cell_types), len(symbol_names)), dtype=float)
    for index, cell_type in enumerate(cell_types):
        selected = mask & (labels == cell_type)
        expression[index] = symbol_matrix[selected].mean(axis=0)

    selected_edges = mask[edge_source] & mask[edge_target]
    communication = np.zeros((len(cell_types), len(cell_types)), dtype=float)
    if np.any(selected_edges):
        np.add.at(
            communication,
            (type_codes[edge_source[selected_edges]], type_codes[edge_target[selected_edges]]),
            attention[selected_edges],
        )
    communication = np.divide(
        communication,
        type_counts[:, None],
        out=np.zeros_like(communication),
        where=type_counts[:, None] > 0,
    )

    ligand = np.column_stack(
        [np.min(expression[:, indices], axis=1) for indices in ligand_indices]
    )
    receptor = np.column_stack(
        [np.min(expression[:, indices], axis=1) for indices in receptor_indices]
    )
    pair_scores = np.einsum("ap,bp,ab->p", ligand, receptor, communication, optimize=True)
    pathway_codes, pathway_names = pd.factorize(lr_table["pathway"], sort=True)
    pathway_scores = np.bincount(
        pathway_codes, weights=pair_scores, minlength=len(pathway_names)
    ).astype(float)
    return pathway_scores


def _permutation_module_lr_pathways(
    *,
    module: str,
    actual_mask: np.ndarray,
    roi_mask: np.ndarray,
    labels: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    attention: np.ndarray,
    symbol_expression: dict[str, np.ndarray],
    lr_table: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol_names = sorted(symbol_expression)
    symbol_index = {symbol: index for index, symbol in enumerate(symbol_names)}
    symbol_matrix = np.column_stack([symbol_expression[symbol] for symbol in symbol_names])
    ligand_indices = [
        np.asarray([symbol_index[token] for token in _complex_tokens(value)], dtype=int)
        for value in lr_table["ligand"].astype(str)
    ]
    receptor_indices = [
        np.asarray([symbol_index[token] for token in _complex_tokens(value)], dtype=int)
        for value in lr_table["receptor"].astype(str)
    ]
    if any(len(indices) == 0 for indices in ligand_indices + receptor_indices):
        raise AssertionError("Empty ligand/receptor complex in retained LR universe.")
    pathway_names = pd.Index(sorted(lr_table["pathway"].astype(str).unique()))
    actual = _module_lr_pathway_scores(
        mask=actual_mask,
        labels=labels,
        edge_source=edge_source,
        edge_target=edge_target,
        attention=attention,
        symbol_names=symbol_names,
        symbol_matrix=symbol_matrix,
        lr_table=lr_table,
        ligand_indices=ligand_indices,
        receptor_indices=receptor_indices,
    )
    if len(actual) != len(pathway_names):
        raise AssertionError("LR pathway factorization changed unexpectedly.")

    type_counts = pd.Series(labels[actual_mask]).value_counts().sort_index()
    pools = {
        str(cell_type): np.flatnonzero(roi_mask & (labels == str(cell_type)))
        for cell_type in type_counts.index
    }
    rng = np.random.default_rng(seed)
    sampled_mask = np.zeros(len(labels), dtype=bool)
    null = np.empty((n_permutations, len(pathway_names)), dtype=np.float64)
    for iteration in range(n_permutations):
        sampled_mask[:] = False
        for cell_type, count in type_counts.items():
            sampled_mask[
                rng.choice(pools[str(cell_type)], size=int(count), replace=False)
            ] = True
        null[iteration] = _module_lr_pathway_scores(
            mask=sampled_mask,
            labels=labels,
            edge_source=edge_source,
            edge_target=edge_target,
            attention=attention,
            symbol_names=symbol_names,
            symbol_matrix=symbol_matrix,
            lr_table=lr_table,
            ligand_indices=ligand_indices,
            receptor_indices=receptor_indices,
        )

    p_values = (1 + np.count_nonzero(null >= actual[None, :], axis=0)) / (
        n_permutations + 1
    )
    summary = pd.DataFrame(
        {
            "module": module,
            "pathway": pathway_names.to_numpy(dtype=str),
            "observed_lr_score": actual,
            "null_mean": null.mean(axis=0),
            "null_sd": null.std(axis=0, ddof=1),
            "empirical_p_greater": p_values,
            "n_permutations": int(n_permutations),
        }
    )
    summary["fold_over_null_mean"] = np.divide(
        summary["observed_lr_score"],
        summary["null_mean"],
        out=np.full(len(summary), np.nan),
        where=summary["null_mean"].to_numpy(dtype=float) > 0,
    )
    summary["adjusted_p_value"] = _bh(summary["empirical_p_greater"].to_numpy())
    summary = summary.sort_values(
        ["adjusted_p_value", "empirical_p_greater", "fold_over_null_mean", "observed_lr_score"],
        ascending=[True, True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    null_long = pd.DataFrame(
        null,
        columns=pathway_names,
    )
    null_long.insert(0, "permutation", np.arange(1, n_permutations + 1, dtype=int))
    null_long.insert(0, "module", module)
    return summary, null_long


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def main() -> None:
    args = _parser().parse_args()
    inputs = {
        "aligned_h5ad": args.aligned_h5ad.expanduser().resolve(),
        "assignments": args.assignments.expanduser().resolve(),
        "edge_index": args.edge_index.expanduser().resolve(),
        "attention": args.attention.expanduser().resolve(),
        "lr_database": args.lr_database.expanduser().resolve(),
        "pair_timecourse": args.pair_timecourse.expanduser().resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=str(output_dir.parent))
    )
    try:
        assignments = pd.read_csv(inputs["assignments"])
        if len(assignments) != 1454:
            raise AssertionError(f"Expected 1,454 ROI cells, got {len(assignments)}")
        assignments["hotspot"] = assignments["hotspot"].fillna("").astype(str)
        edge_index = np.load(inputs["edge_index"]).astype(np.int64)
        attention = np.load(inputs["attention"]).astype(np.float64).reshape(-1)
        if edge_index.shape != (2, len(attention)):
            raise AssertionError(
                f"Attention archive mismatch: edge_index={edge_index.shape}, attention={attention.shape}"
            )
        edge_source, edge_target = edge_index

        adata = ad.read_h5ad(inputs["aligned_h5ad"], backed="r")
        if "time_point_processed" not in adata.obs:
            raise KeyError("aligned h5ad is missing obs['time_point_processed']")
        time_indices = np.flatnonzero(
            np.isclose(
                adata.obs["time_point_processed"].to_numpy(dtype=float),
                float(args.time),
                rtol=0.0,
                atol=1e-9,
            )
        )
        if len(time_indices) != 8106:
            raise AssertionError(f"Expected 8,106 t=1 cells, got {len(time_indices)}")
        labels = adata.obs.iloc[time_indices]["Annotation"].astype(str).to_numpy()
        if int(edge_index.max()) >= len(labels):
            raise AssertionError("Attention edge indices exceed t=1 cells.")
        roi_indices = assignments["cell_index"].to_numpy(dtype=int)
        if not np.array_equal(labels[roi_indices], assignments["celltype"].astype(str)):
            raise AssertionError("ROI assignments do not align to t=1 labels.")
        coordinates = np.asarray(adata.obsm["spatial_aligned"])[time_indices]
        if not np.allclose(
            coordinates[roi_indices],
            assignments[["current_x", "current_y"]].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise AssertionError("ROI assignments do not align to spatial coordinates.")

        roi_mask = np.zeros(len(labels), dtype=bool)
        roi_mask[roi_indices] = True
        domain = np.full(len(labels), "", dtype=object)
        domain[roi_indices] = assignments["hotspot"].to_numpy(dtype=object)
        module_by_domain = {
            domain_name: module
            for module, domain_names in MODULE_DOMAINS.items()
            for domain_name in domain_names
        }
        module_labels = np.asarray([module_by_domain.get(str(value), "") for value in domain])

        lr_table = _load_lr_table(inputs["lr_database"], inputs["pair_timecourse"])
        requested_symbols = {
            token
            for row in lr_table.itertuples(index=False)
            for token in _complex_tokens(str(row.ligand)) + _complex_tokens(str(row.receptor))
        }
        symbol_expression, feature_coverage = _symbol_expression_by_cell(
            adata, time_indices, requested_symbols
        )

        # Add the biological section labels without using them to define hotspots.
        for column in ("inj_uninj", "D_V", "Batch"):
            if column in adata.obs:
                assignments[column] = (
                    adata.obs.iloc[time_indices[roi_indices]][column].astype(str).to_numpy()
                )
        adata.file.close()

        roi_type_pair = _type_pair_attention(
            roi_mask, roi_mask, labels, edge_source, edge_target, attention
        )
        roi_expression_by_type = _expression_by_type(
            roi_mask, labels, symbol_expression
        )

        # Model-selected domain-pair network and its LR projection.
        network_rows: list[pd.DataFrame] = []
        lr_rows: list[pd.DataFrame] = []
        supported_domain_pairs: list[dict[str, Any]] = []
        for source_domain in SUPPORTED_DOMAINS:
            source_mask = domain == source_domain
            for target_domain in SUPPORTED_DOMAINS:
                target_mask = domain == target_domain
                selected = source_mask[edge_source] & target_mask[edge_target]
                if not np.any(selected):
                    continue
                source_module = module_by_domain[source_domain]
                target_module = module_by_domain[target_domain]
                module = source_module if source_module == target_module else f"{source_module}__to__{target_module}"
                scored, type_edges = _score_lr_for_domain_pair(
                    domain_source=source_domain,
                    domain_target=target_domain,
                    module=module,
                    mask_source=source_mask,
                    mask_target=target_mask,
                    labels=labels,
                    edge_source=edge_source,
                    edge_target=edge_target,
                    attention=attention,
                    lr_table=lr_table,
                    symbol_expression=symbol_expression,
                    roi_type_pair=roi_type_pair,
                    roi_expression_by_type=roi_expression_by_type,
                )
                type_edges.insert(0, "module", module)
                type_edges.insert(1, "source_domain", source_domain)
                type_edges.insert(2, "target_domain", target_domain)
                network_rows.append(type_edges)
                if not scored.empty:
                    lr_rows.append(scored)
                supported_domain_pairs.append(
                    {
                        "module": module,
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "n_source_cells": int(source_mask.sum()),
                        "n_target_cells": int(target_mask.sum()),
                        "n_edges": int(selected.sum()),
                        "attention_sum": float(attention[selected].sum()),
                        "attention_per_source_cell": float(attention[selected].sum() / max(source_mask.sum(), 1)),
                    }
                )

        network = pd.concat(network_rows, ignore_index=True)
        lr_scores = pd.concat(lr_rows, ignore_index=True)
        lr_scores = lr_scores.sort_values(
            ["local_lr_score", "local_to_roi_score_ratio", "module", "sender", "receiver", "pair"],
            ascending=[False, False, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        pathway = (
            lr_scores.groupby(
                ["module", "source_domain", "target_domain", "pathway", "interaction_class"],
                sort=True,
            )
            .agg(
                local_lr_score=("local_lr_score", "sum"),
                roi_lr_score=("roi_lr_score", "sum"),
                n_nonzero_pairs=("local_lr_score", lambda values: int(np.count_nonzero(np.asarray(values) > 0))),
                n_sender_receiver_pairs=("sender", "size"),
            )
            .reset_index()
        )
        pathway["local_to_roi_score_ratio"] = np.divide(
            pathway["local_lr_score"],
            pathway["roi_lr_score"],
            out=np.full(len(pathway), np.nan),
            where=pathway["roi_lr_score"].to_numpy(dtype=float) > 0,
        )
        pathway["log2_local_to_roi_score_ratio"] = np.where(
            pathway["local_to_roi_score_ratio"] > 0,
            np.log2(pathway["local_to_roi_score_ratio"]),
            np.nan,
        )
        pathway = pathway.sort_values(
            ["local_lr_score", "local_to_roi_score_ratio"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)

        # Per-cell participation in supported within-module selected edges.
        incoming_n = np.zeros(len(labels), dtype=int)
        outgoing_n = np.zeros(len(labels), dtype=int)
        incoming_attention = np.zeros(len(labels), dtype=float)
        outgoing_attention = np.zeros(len(labels), dtype=float)
        same_supported_module = (
            (module_labels[edge_source] != "")
            & (module_labels[edge_source] == module_labels[edge_target])
        )
        np.add.at(outgoing_n, edge_source[same_supported_module], 1)
        np.add.at(incoming_n, edge_target[same_supported_module], 1)
        np.add.at(outgoing_attention, edge_source[same_supported_module], attention[same_supported_module])
        np.add.at(incoming_attention, edge_target[same_supported_module], attention[same_supported_module])
        assignments["supported_module"] = [module_by_domain.get(value, "") for value in assignments["hotspot"]]
        assignments["network_outgoing_edges"] = outgoing_n[roi_indices]
        assignments["network_incoming_edges"] = incoming_n[roi_indices]
        assignments["network_outgoing_attention"] = outgoing_attention[roi_indices]
        assignments["network_incoming_attention"] = incoming_attention[roi_indices]
        assignments["network_participant"] = (
            assignments["network_outgoing_edges"] + assignments["network_incoming_edges"] > 0
        )

        permutation_summaries: list[dict[str, Any]] = []
        permutation_nulls: list[pd.DataFrame] = []
        lr_permutation_summaries: list[pd.DataFrame] = []
        lr_permutation_nulls: list[pd.DataFrame] = []
        for module_index, (module, domains) in enumerate(MODULE_DOMAINS.items()):
            actual_mask = np.isin(domain, domains)
            summary, null = _permutation_module_attention(
                module=module,
                actual_mask=actual_mask,
                roi_mask=roi_mask,
                labels=labels,
                edge_source=edge_source,
                edge_target=edge_target,
                attention=attention,
                n_permutations=int(args.n_permutations),
                seed=int(args.seed) + module_index * 1009,
            )
            permutation_summaries.append(summary)
            permutation_nulls.append(null)
            lr_summary, lr_null = _permutation_module_lr_pathways(
                module=module,
                actual_mask=actual_mask,
                roi_mask=roi_mask,
                labels=labels,
                edge_source=edge_source,
                edge_target=edge_target,
                attention=attention,
                symbol_expression=symbol_expression,
                lr_table=lr_table,
                n_permutations=int(args.n_lr_permutations),
                seed=int(args.seed) + 7919 + module_index * 1009,
            )
            lr_permutation_summaries.append(lr_summary)
            lr_permutation_nulls.append(lr_null)

        tables = temporary / "tables"
        tables.mkdir(parents=True)
        outputs = {
            "roi_cells_with_network_participation.csv": assignments,
            "supported_domain_pair_summary.csv": pd.DataFrame(supported_domain_pairs),
            "supported_domain_celltype_edges.csv": network,
            "hotspot_local_lr_scores.csv.gz": lr_scores,
            "hotspot_local_lr_pathways.csv": pathway,
            "roi_celltype_attention_baseline.csv": roi_type_pair,
            "lr_feature_coverage.csv": feature_coverage,
            "module_attention_permutation_summary.csv": pd.DataFrame(permutation_summaries),
            "module_attention_permutation_null.csv.gz": pd.concat(permutation_nulls, ignore_index=True),
            "module_lr_pathway_permutation_summary.csv": pd.concat(
                lr_permutation_summaries, ignore_index=True
            ),
            "module_lr_pathway_permutation_null.csv.gz": pd.concat(
                lr_permutation_nulls, ignore_index=True
            ),
        }
        for name, frame in outputs.items():
            _write_csv(frame, tables / name)

        injury_contract: dict[str, Any] = {}
        for column in ("inj_uninj", "D_V", "Batch"):
            if column in assignments:
                injury_contract[column] = assignments[column].value_counts(dropna=False).to_dict()
        manifest = {
            "schema": "cytobridge.arista.figure5c-interaction-mechanism.v1",
            "time": float(args.time),
            "roi_n": int(len(assignments)),
            "supported_domains": list(SUPPORTED_DOMAINS),
            "excluded_domains": {
                "H4": "no internal selected model edges",
                "L3": "no internal selected model edges",
                "L4": "no internal selected model edges",
            },
            "network_modules": {key: list(value) for key, value in MODULE_DOMAINS.items()},
            "injury_section_contract": injury_contract,
            "attention_contract": {
                "edge_direction": "edge_index row 0 sender/source -> row 1 target/receiver",
                "value": "absolute multi-head mean from fresh model selected sparse edges",
                "normalization": "attention sum divided by number of source cells in the source domain/type",
                "interpretation": "relative model interaction influence, not a biophysical signaling rate",
            },
            "lr_contract": {
                "formula": "mean_source(ligand/complex) * mean_target(receptor/complex) * attention_per_source_cell",
                "expression": "observed t=1 adata.X; per-cell expm1 from persisted log1p space before arithmetic means",
                "complex_mode": "strict minimum; require all subunits",
                "pair_universe": "fresh package pair_timecourse retained universe",
                "database": str(inputs["lr_database"]),
                "reference": "full 1,454-cell ROI, matched sender/receiver cell type",
            },
            "permutation_contract": {
                "n_permutations": int(args.n_permutations),
                "n_lr_pathway_permutations": int(args.n_lr_permutations),
                "seed": int(args.seed),
                "null": "same module size and exact cell-type counts, sampled within the frozen ROI",
                "purpose": "test selected-attention concentration beyond cell-type composition",
            },
            "limitations": [
                "Single 5-DPI injured dorsal section; cells are not biological replicates.",
                "Hotspots are defined from model-derived velocity cosine and characterized by the same fitted model.",
                "The analysis supports organized local interaction hypotheses but not causal signaling claims.",
                "The ROI is injury-associated by section metadata; no pixel-level wound-contour annotation is available here.",
            ],
            "inputs": {
                key: {"path": str(path), "sha256": _sha256(path)} for key, path in inputs.items()
            },
            "outputs": {},
        }
        for path in sorted(tables.iterdir()):
            manifest["outputs"][path.name] = {
                "sha256": _sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
