#!/usr/bin/env python3
"""Compute exact pair-level LR axes inside the final two Figure 5c niches."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from server_analyze_figure5c_interaction_mechanism import (
    _bh,
    _complex_tokens,
    _load_lr_table,
    _symbol_expression_by_cell,
)


NICHE_ORDER = ("N1_sfrpEGC_VLMC", "N2_reaEGC_wntEGC")
DISPLAY_PATHWAYS = {
    "N1_sfrpEGC_VLMC": ("AGRN", "LAMININ", "TENASCIN", "NRG", "THBS"),
    "N2_reaEGC_wntEGC": ("GRN", "L1CAM", "NRXN", "SEMA3", "FN1"),
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
    parser.add_argument("--n-permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=260824)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_scores(
    *,
    mask: np.ndarray,
    labels: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    attention: np.ndarray,
    symbol_matrix: np.ndarray,
    ligand_indices: list[np.ndarray],
    receptor_indices: list[np.ndarray],
    return_dominant: bool,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray, np.ndarray] | np.ndarray:
    cell_types = sorted(np.unique(labels[mask]))
    type_codes = np.full(len(labels), -1, dtype=int)
    type_counts = np.empty(len(cell_types), dtype=float)
    expression = np.empty((len(cell_types), symbol_matrix.shape[1]), dtype=float)
    for index, cell_type in enumerate(cell_types):
        selected = mask & (labels == cell_type)
        type_codes[selected] = index
        type_counts[index] = float(selected.sum())
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
    contributions = (
        ligand[:, None, :] * receptor[None, :, :] * communication[:, :, None]
    )
    totals = contributions.sum(axis=(0, 1))
    if not return_dominant:
        return totals

    flat = contributions.reshape(len(cell_types) * len(cell_types), contributions.shape[2])
    dominant_flat = np.argmax(flat, axis=0)
    source_codes, target_codes = np.unravel_index(
        dominant_flat, (len(cell_types), len(cell_types))
    )
    dominant_scores = flat[dominant_flat, np.arange(flat.shape[1])]
    dominant_fraction = np.divide(
        dominant_scores,
        totals,
        out=np.zeros_like(dominant_scores),
        where=totals > 0,
    )
    return (
        totals,
        [cell_types[index] for index in source_codes],
        [cell_types[index] for index in target_codes],
        dominant_scores,
        dominant_fraction,
    )


def _analyze_niche(
    *,
    niche: str,
    actual_mask: np.ndarray,
    roi_mask: np.ndarray,
    labels: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    attention: np.ndarray,
    symbol_names: list[str],
    symbol_matrix: np.ndarray,
    lr_table: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol_index = {symbol: index for index, symbol in enumerate(symbol_names)}
    ligand_indices = [
        np.asarray([symbol_index[token] for token in _complex_tokens(value)], dtype=int)
        for value in lr_table["ligand"].astype(str)
    ]
    receptor_indices = [
        np.asarray([symbol_index[token] for token in _complex_tokens(value)], dtype=int)
        for value in lr_table["receptor"].astype(str)
    ]
    actual_result = _pair_scores(
        mask=actual_mask,
        labels=labels,
        edge_source=edge_source,
        edge_target=edge_target,
        attention=attention,
        symbol_matrix=symbol_matrix,
        ligand_indices=ligand_indices,
        receptor_indices=receptor_indices,
        return_dominant=True,
    )
    actual, dominant_sender, dominant_receiver, dominant_score, dominant_fraction = actual_result

    type_counts = pd.Series(labels[actual_mask]).value_counts().sort_index()
    pools = {
        str(cell_type): np.flatnonzero(roi_mask & (labels == str(cell_type)))
        for cell_type in type_counts.index
    }
    rng = np.random.default_rng(seed)
    sampled_mask = np.zeros(len(labels), dtype=bool)
    null = np.empty((n_permutations, len(lr_table)), dtype=np.float64)
    for iteration in range(n_permutations):
        sampled_mask[:] = False
        for cell_type, count in type_counts.items():
            sampled_mask[
                rng.choice(pools[str(cell_type)], size=int(count), replace=False)
            ] = True
        null[iteration] = _pair_scores(
            mask=sampled_mask,
            labels=labels,
            edge_source=edge_source,
            edge_target=edge_target,
            attention=attention,
            symbol_matrix=symbol_matrix,
            ligand_indices=ligand_indices,
            receptor_indices=receptor_indices,
            return_dominant=False,
        )

    empirical_p = (1 + np.count_nonzero(null >= actual[None, :], axis=0)) / (
        n_permutations + 1
    )
    summary = lr_table[
        ["ligand", "receptor", "pair", "pathway", "interaction_class"]
    ].copy()
    summary.insert(0, "niche", niche)
    summary["observed_pair_score"] = actual
    summary["null_mean"] = null.mean(axis=0)
    summary["null_sd"] = null.std(axis=0, ddof=1)
    summary["fold_over_null_mean"] = np.divide(
        actual,
        summary["null_mean"].to_numpy(float),
        out=np.full(len(summary), np.nan),
        where=summary["null_mean"].to_numpy(float) > 0,
    )
    summary["log2_fold_over_null"] = np.log2(summary["fold_over_null_mean"])
    summary["empirical_p_greater"] = empirical_p
    summary["adjusted_p_value"] = _bh(empirical_p)
    summary["dominant_sender"] = dominant_sender
    summary["dominant_receiver"] = dominant_receiver
    summary["dominant_pair_score"] = dominant_score
    summary["dominant_contribution_fraction"] = dominant_fraction
    summary["n_permutations"] = int(n_permutations)

    candidate_rows = []
    for pathway in DISPLAY_PATHWAYS[niche]:
        candidates = summary.loc[
            summary["pathway"].eq(pathway)
            & summary["observed_pair_score"].gt(0)
            & np.isfinite(summary["fold_over_null_mean"])
        ].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(
            ["adjusted_p_value", "observed_pair_score", "fold_over_null_mean"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        selected = candidates.iloc[0].copy()
        selected["selection_rule"] = (
            "Within each displayed FDR-significant pathway, minimum pair-level BH q, "
            "then maximum observed pair score, then maximum fold over null"
        )
        candidate_rows.append(selected)
    selected_axes = pd.DataFrame(candidate_rows)
    return summary, selected_axes


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
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        assignments = pd.read_csv(inputs["assignments"])
        aligned = ad.read_h5ad(inputs["aligned_h5ad"], backed="r")
        t_indices = np.flatnonzero(
            np.isclose(
                aligned.obs["time_point_processed"].to_numpy(float),
                float(args.time),
                rtol=0.0,
                atol=1e-9,
            )
        )
        labels = aligned.obs.iloc[t_indices]["Annotation"].astype(str).to_numpy()
        roi_indices = assignments["cell_index"].to_numpy(int)
        if not np.array_equal(labels[roi_indices], assignments["celltype"].astype(str)):
            raise AssertionError("ROI assignments do not align to the selected time-point cells.")

        edge_index = np.load(inputs["edge_index"]).astype(np.int64)
        attention = np.load(inputs["attention"]).astype(np.float64).reshape(-1)
        if edge_index.shape != (2, len(attention)) or int(edge_index.max()) >= len(labels):
            raise AssertionError("Attention archive does not align to the selected time point.")

        lr_table = _load_lr_table(inputs["lr_database"], inputs["pair_timecourse"])
        requested_symbols = {
            token
            for row in lr_table.itertuples(index=False)
            for token in _complex_tokens(str(row.ligand)) + _complex_tokens(str(row.receptor))
        }
        symbol_expression, feature_coverage = _symbol_expression_by_cell(
            aligned, t_indices, requested_symbols
        )
        aligned.file.close()
        symbol_names = sorted(symbol_expression)
        symbol_matrix = np.column_stack([symbol_expression[symbol] for symbol in symbol_names])

        roi_mask = np.zeros(len(labels), dtype=bool)
        roi_mask[roi_indices] = True
        summaries = []
        selected = []
        for niche_index, niche in enumerate(NICHE_ORDER):
            mask = np.zeros(len(labels), dtype=bool)
            selected_roi = assignments["two_niche_region"].fillna("").eq(niche).to_numpy()
            mask[roi_indices[selected_roi]] = True
            summary, axes = _analyze_niche(
                niche=niche,
                actual_mask=mask,
                roi_mask=roi_mask,
                labels=labels,
                edge_source=edge_index[0],
                edge_target=edge_index[1],
                attention=attention,
                symbol_names=symbol_names,
                symbol_matrix=symbol_matrix,
                lr_table=lr_table,
                n_permutations=int(args.n_permutations),
                seed=int(args.seed) + niche_index * 1009,
            )
            summaries.append(summary)
            selected.append(axes)

        all_pairs = pd.concat(summaries, ignore_index=True)
        selected_axes = pd.concat(selected, ignore_index=True)
        all_pairs.to_csv(stage / "two_niche_lr_pair_matched_null.csv.gz", index=False)
        selected_axes.to_csv(stage / "candidate_display_lr_axes.csv", index=False)
        feature_coverage.to_csv(stage / "lr_feature_coverage.csv", index=False)
        shutil.copy2(Path(__file__), stage / Path(__file__).name)

        manifest: dict[str, Any] = {
            "schema": "cytobridge.arista.figure5c-two-niche-pair-axes.v1",
            "calculation": (
                "mean source ligand/complex expression * mean target receptor/complex "
                "expression * selected attention per source cell, computed inside each exact spatial niche"
            ),
            "pair_null": (
                "ROI cells sampled without replacement within each cell type; exact niche "
                "cell-type counts preserved"
            ),
            "pair_multiple_testing": "Benjamini-Hochberg within each niche across all retained LR pairs",
            "display_selection": {
                niche: list(pathways) for niche, pathways in DISPLAY_PATHWAYS.items()
            },
            "inputs": {
                key: {"path": str(path), "sha256": _sha256(path)} for key, path in inputs.items()
            },
            "outputs": {},
        }
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                manifest["outputs"][path.name] = {
                    "sha256": _sha256(path),
                    "size_bytes": int(path.stat().st_size),
                }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        stage.rename(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
