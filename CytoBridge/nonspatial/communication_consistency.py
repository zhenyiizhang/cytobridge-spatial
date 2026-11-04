"""Utilities for cross-method non-spatial communication consistency analyses.

The comparison deliberately operates on directed sender/receiver cell-type
pairs.  Raw scores from CytoBridge, CellChat, CellAgentChat, and NicheNet are
not commensurate, so only within-method ranks and top-set membership are
compared across methods.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests


METHODS = ("CytoBridge", "CellChat", "CellAgentChat", "NicheNet")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_sample_indices(
    labels: Iterable[object], *, total: int, seed: int
) -> np.ndarray:
    """Return a deterministic, near-proportional sample retaining every group."""

    labels = np.asarray([str(value) for value in labels], dtype=object)
    if total <= 0:
        raise ValueError("total must be positive")
    if len(labels) <= total:
        return np.arange(len(labels), dtype=np.int64)
    groups = sorted(set(labels))
    counts = {group: int(np.sum(labels == group)) for group in groups}
    raw = {group: total * counts[group] / len(labels) for group in groups}
    allocation = {
        group: min(counts[group], max(1, int(math.floor(raw[group]))))
        for group in groups
    }
    while sum(allocation.values()) < total:
        eligible = [group for group in groups if allocation[group] < counts[group]]
        group = max(
            eligible, key=lambda key: (raw[key] - allocation[key], counts[key], key)
        )
        allocation[group] += 1
    while sum(allocation.values()) > total:
        eligible = [group for group in groups if allocation[group] > 1]
        group = min(
            eligible, key=lambda key: (raw[key] - allocation[key], -counts[key], key)
        )
        allocation[group] -= 1
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for group in groups:
        candidates = np.flatnonzero(labels == group)
        selected.append(
            np.sort(rng.choice(candidates, allocation[group], replace=False))
        )
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def encode_cellagentchat_labels(
    labels: Iterable[object],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Encode cell types for CellAgentChat's underscore-delimited pair keys.

    CellAgentChat internally serializes a directed pair as
    ``"sender_receiver"`` and later splits that string on underscores.  Raw
    biological labels containing underscores therefore cannot be passed to
    the official implementation without ambiguity.  This reversible mapping
    uses underscore-free identifiers while retaining the original labels for
    every reported result.
    """

    original = np.asarray([str(value) for value in labels], dtype=object)
    levels = sorted(set(original))
    mapping = pd.DataFrame(
        {
            "cellagentchat_label": [f"ct{index:03d}" for index in range(len(levels))],
            "cell_type": levels,
        }
    )
    encode = dict(zip(mapping["cell_type"], mapping["cellagentchat_label"]))
    encoded = np.asarray([encode[value] for value in original], dtype=object)
    if any("_" in value for value in encoded):
        raise AssertionError("CellAgentChat labels must be underscore-free")
    return encoded, mapping


def summarize_cellagentchat_pair_matrices(
    raw_scores: pd.DataFrame,
    significant_scores: pd.DataFrame,
    label_map: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize official CellAgentChat LR matrices on directed type pairs.

    CellAgentChat's continuous output contains one LR score per matrix row and
    directed ``sender_receiver`` pair per column.  Its native CTPS statistic is
    the sum of the *significant* LR scores for a directed pair, rather than the
    number of significant LR pairs.  Both quantities and the threshold-free
    continuous sum are retained so the primary and sensitivity analyses cannot
    be confused.
    """

    required = {"cellagentchat_label", "cell_type"}
    if not required.issubset(label_map.columns):
        raise ValueError(
            "label_map is missing " f"{sorted(required.difference(label_map.columns))}"
        )
    if label_map["cellagentchat_label"].duplicated().any():
        raise ValueError("CellAgentChat label map contains duplicate encoded labels")
    decode = dict(
        zip(
            label_map["cellagentchat_label"].astype(str),
            label_map["cell_type"].astype(str),
            strict=True,
        )
    )
    tokens = sorted(decode)
    expected_columns = {
        f"{sender}_{receiver}" for sender in tokens for receiver in tokens
    }

    def summarize(frame: pd.DataFrame, name: str) -> dict[str, float]:
        if frame.columns.duplicated().any():
            raise ValueError(f"CellAgentChat {name} matrix has duplicate columns")
        # The official threshold-free CSV appends a per-LR ``total`` column;
        # it is not a directed cell-type pair and must not enter pair scores.
        frame = frame.drop(columns=["total"], errors="ignore")
        unknown = set(frame.columns.astype(str)).difference(expected_columns)
        if unknown:
            raise ValueError(
                f"CellAgentChat {name} matrix contains unknown directed pairs: "
                f"{sorted(unknown)[:5]}"
            )
        numeric = frame.apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=float)
        if values.size and not np.isfinite(values).all():
            raise ValueError(f"CellAgentChat {name} matrix contains non-finite scores")
        return {str(column): float(numeric[column].sum()) for column in numeric}

    raw_sum = summarize(raw_scores, "continuous")
    ctps_sum = summarize(significant_scores, "significant")
    significant_count = {
        str(column): int(
            np.count_nonzero(
                pd.to_numeric(significant_scores[column], errors="raise").to_numpy(
                    dtype=float
                )
            )
        )
        for column in significant_scores
    }
    rows: list[dict[str, object]] = []
    for sender in tokens:
        for receiver in tokens:
            key = f"{sender}_{receiver}"
            rows.append(
                {
                    "sender_type": decode[sender],
                    "receiver_type": decode[receiver],
                    "cellagentchat_native_ctps": ctps_sum.get(key, 0.0),
                    "cellagentchat_continuous_score": raw_sum.get(key, 0.0),
                    "cellagentchat_significant_lr_count": significant_count.get(key, 0),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["sender_type", "receiver_type"], kind="mergesort", ignore_index=True
    )


def prepare_shared_lr_database(
    database_path: str | Path, output_dir: str | Path
) -> dict[str, object]:
    """Convert one CellChatDB file into method-specific transport formats.

    The full database is retained.  Downstream methods intersect it with their
    expressed gene universe, so underscore-delimited complexes that a method
    cannot represent are excluded by that method rather than replaced by
    biologically different Cartesian subunit pairs.
    """

    database_path = Path(database_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    frame = pd.read_csv(database_path)
    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    ligand_column = next(
        (
            normalized[name]
            for name in ("ligand", "source", "from")
            if name in normalized
        ),
        None,
    )
    receptor_column = next(
        (
            normalized[name]
            for name in ("receptor", "target", "to")
            if name in normalized
        ),
        None,
    )
    if ligand_column is None or receptor_column is None:
        usable = [
            column
            for column in frame.columns
            if not str(column).strip().casefold().startswith("unnamed")
        ]
        if len(usable) < 2:
            raise ValueError("could not identify ligand/receptor columns")
        ligand_column, receptor_column = usable[:2]
    pairs = (
        frame[[ligand_column, receptor_column]]
        .rename(columns={ligand_column: "ligand", receptor_column: "receptor"})
        .dropna()
        .astype(str)
    )
    pairs = pairs[
        (pairs.ligand.str.strip() != "") & (pairs.receptor.str.strip() != "")
    ].drop_duplicates(ignore_index=True)
    pairs["database_row"] = np.arange(len(pairs), dtype=int)
    pairs["monomeric_for_gene_level_methods"] = ~(
        pairs.ligand.str.contains("_", regex=False)
        | pairs.receptor.str.contains("_", regex=False)
    )
    output_dir.mkdir(parents=True)
    pairs.to_csv(output_dir / "shared_lr_pairs.csv", index=False)
    pairs[["database_row", "ligand", "receptor"]].to_csv(
        output_dir / "cellagentchat_lr_pairs.tsv", sep="\t", index=False
    )
    pairs[["ligand", "receptor"]].rename(
        columns={"ligand": "from", "receptor": "to"}
    ).to_csv(output_dir / "nichenet_lr_network.csv", index=False)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_database": {
            "path": str(database_path),
            "sha256": sha256_file(database_path),
        },
        "n_unique_lr_pairs": int(len(pairs)),
        "n_monomeric_lr_pairs": int(pairs["monomeric_for_gene_level_methods"].sum()),
        "complex_policy": "retain exact CellChatDB strings; each method intersects with its representable expressed-gene universe",
        "outputs": {},
    }
    for path in sorted(output_dir.glob("*.csv")) + sorted(output_dir.glob("*.tsv")):
        manifest["outputs"][path.name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return manifest


def _mean_var_fraction(matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        square_mean = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        fraction = np.asarray((matrix > 0).mean(axis=0)).ravel()
    else:
        array = np.asarray(matrix, dtype=np.float64)
        mean = array.mean(axis=0)
        square_mean = np.square(array).mean(axis=0)
        fraction = np.mean(array > 0, axis=0)
    variance = np.maximum(square_mean - np.square(mean), 0.0)
    return mean, variance, fraction


def prepare_nichenet_tables(
    adata,
    *,
    dataset: str,
    cell_type_key: str,
    time_key: str,
    terminal_time: float,
    previous_time: float,
    lr_network: pd.DataFrame,
    output_dir: str | Path,
    expression_fraction: float = 0.05,
    response_fraction: float = 0.10,
    response_top_n: int = 100,
) -> dict[str, object]:
    """Build receiver response sets and sender/receiver LR candidates.

    Expression values are assumed to be log-normalized.  Receiver response
    genes are the fixed top-N positive terminal-versus-previous changes among
    genes expressed in at least ``response_fraction`` of terminal cells.  A
    fixed effect-size ranking avoids letting receiver cell count (and hence
    p-value power) change the response-set size.  Welch and Benjamini-Hochberg
    values are retained for audit but do not define membership.
    """

    required = {cell_type_key, time_key}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"AnnData is missing required obs columns: {sorted(missing)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cell_types = adata.obs[cell_type_key].astype(str).to_numpy()
    times = pd.to_numeric(adata.obs[time_key], errors="raise").to_numpy(float)
    genes = np.asarray(adata.var_names.astype(str), dtype=object)
    if len(set(genes)) != len(genes):
        raise ValueError("NicheNet input requires unique gene symbols")

    network = lr_network.rename(columns={"from": "ligand", "to": "receptor"}).copy()
    if not {"ligand", "receptor"}.issubset(network.columns):
        raise ValueError("lr_network must contain from/to or ligand/receptor columns")
    network = network[["ligand", "receptor"]].dropna().astype(str).drop_duplicates()
    gene_set = set(genes)
    network = network[
        network["ligand"].isin(gene_set) & network["receptor"].isin(gene_set)
    ].reset_index(drop=True)
    gene_index = {gene: index for index, gene in enumerate(genes)}
    ligand_indices = {gene: gene_index[gene] for gene in network["ligand"].unique()}
    receptor_indices = {gene: gene_index[gene] for gene in network["receptor"].unique()}

    terminal = np.isclose(times, float(terminal_time), rtol=0, atol=1e-8)
    previous = np.isclose(times, float(previous_time), rtol=0, atol=1e-8)
    if not terminal.any() or not previous.any():
        raise ValueError("terminal and previous timepoints must both contain cells")

    receiver_rows: list[dict[str, object]] = []
    expression_rows: list[dict[str, object]] = []
    response_unavailable: list[dict[str, object]] = []
    cell_type_values = sorted(set(cell_types[terminal]))
    for cell_type in cell_type_values:
        terminal_mask = terminal & (cell_types == cell_type)
        previous_mask = previous & (cell_types == cell_type)
        terminal_matrix = adata.X[terminal_mask]
        _, _, fraction = _mean_var_fraction(terminal_matrix)
        for gene, index in ligand_indices.items():
            expression_rows.append(
                {
                    "dataset": dataset,
                    "cell_type": cell_type,
                    "role": "ligand",
                    "gene": gene,
                    "fraction": float(fraction[index]),
                }
            )
        for gene, index in receptor_indices.items():
            expression_rows.append(
                {
                    "dataset": dataset,
                    "cell_type": cell_type,
                    "role": "receptor",
                    "gene": gene,
                    "fraction": float(fraction[index]),
                }
            )
        if terminal_mask.sum() < 3 or previous_mask.sum() < 3:
            response_unavailable.append(
                {
                    "cell_type": cell_type,
                    "terminal_cells": int(terminal_mask.sum()),
                    "previous_cells": int(previous_mask.sum()),
                    "reason": "fewer_than_three_cells_at_terminal_or_previous_time",
                }
            )
            continue
        terminal_mean, terminal_var, terminal_fraction = _mean_var_fraction(
            adata.X[terminal_mask]
        )
        previous_mean, previous_var, _ = _mean_var_fraction(adata.X[previous_mask])
        _, pvalues = stats.ttest_ind_from_stats(
            terminal_mean,
            np.sqrt(terminal_var),
            int(terminal_mask.sum()),
            previous_mean,
            np.sqrt(previous_var),
            int(previous_mask.sum()),
            equal_var=False,
        )
        pvalues = np.nan_to_num(pvalues, nan=1.0, posinf=1.0, neginf=1.0)
        adjusted = multipletests(pvalues, method="fdr_bh")[1]
        log2fc = (terminal_mean - previous_mean) / math.log(2.0)
        background = terminal_fraction >= float(response_fraction)
        positive = np.flatnonzero(background & (log2fc > 0))
        order = positive[np.argsort(-log2fc[positive], kind="mergesort")]
        selected_response = order[: min(int(response_top_n), len(order))]
        response = np.zeros(len(genes), dtype=bool)
        response[selected_response] = True
        if int(response.sum()) < 10:
            response_unavailable.append(
                {
                    "cell_type": cell_type,
                    "terminal_cells": int(terminal_mask.sum()),
                    "previous_cells": int(previous_mask.sum()),
                    "reason": "fewer_than_ten_positive_response_genes",
                }
            )
            continue
        for index in np.flatnonzero(background):
            receiver_rows.append(
                {
                    "dataset": dataset,
                    "receiver": cell_type,
                    "gene": genes[index],
                    "is_response": bool(response[index]),
                    "log2fc": float(log2fc[index]),
                    "fdr": float(adjusted[index]),
                    "terminal_fraction": float(terminal_fraction[index]),
                }
            )

    if not receiver_rows:
        raise ValueError(f"{dataset} has no receiver cell type eligible for NicheNet")

    expression = pd.DataFrame(expression_rows)
    ligands = expression[
        (expression["role"] == "ligand")
        & (expression["fraction"] >= float(expression_fraction))
    ].rename(columns={"cell_type": "sender", "fraction": "sender_fraction"})
    receptors = expression[
        (expression["role"] == "receptor")
        & (expression["fraction"] >= float(expression_fraction))
    ].rename(columns={"cell_type": "receiver", "fraction": "receiver_fraction"})
    candidates = (
        network.merge(
            ligands[["sender", "gene", "sender_fraction"]],
            left_on="ligand",
            right_on="gene",
        )
        .drop(columns="gene")
        .merge(
            receptors[["receiver", "gene", "receiver_fraction"]],
            left_on="receptor",
            right_on="gene",
        )
        .drop(columns="gene")
    )
    candidates.insert(0, "dataset", dataset)
    candidates = candidates.sort_values(
        ["sender", "receiver", "ligand", "receptor"], kind="mergesort"
    ).reset_index(drop=True)
    receivers = pd.DataFrame(receiver_rows).sort_values(
        ["receiver", "gene"], kind="mergesort"
    )
    expression = expression.sort_values(["cell_type", "role", "gene"], kind="mergesort")
    receivers.to_csv(output_dir / "receiver_gene_sets.csv", index=False)
    candidates.to_csv(output_dir / "sender_receiver_lr_candidates.csv", index=False)
    expression.to_csv(output_dir / "terminal_expression_fractions.csv", index=False)
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "cell_type_key": cell_type_key,
        "time_key": time_key,
        "terminal_time": float(terminal_time),
        "previous_time": float(previous_time),
        "expression_fraction": float(expression_fraction),
        "response_fraction": float(response_fraction),
        "response_top_n": int(response_top_n),
        "response_selection": "top positive terminal-minus-previous log2 expression changes",
        "n_cell_types": len(cell_type_values),
        "n_response_eligible_cell_types": int(
            len(cell_type_values) - len(response_unavailable)
        ),
        "response_unavailable_cell_types": response_unavailable,
        "n_lr_candidates": int(len(candidates)),
        "outputs": {},
    }
    for name in (
        "receiver_gene_sets.csv",
        "sender_receiver_lr_candidates.csv",
        "terminal_expression_fractions.csv",
    ):
        path = output_dir / name
        manifest["outputs"][name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def rank_percentile(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="raise").fillna(0.0)
    if len(values) <= 1:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return values.rank(method="average", pct=True)


def complete_directed_pairs(
    frame: pd.DataFrame, *, score_column: str, cell_types: Iterable[str]
) -> pd.DataFrame:
    types = sorted({str(value) for value in cell_types})
    grid = pd.MultiIndex.from_product(
        [types, types], names=["sender_type", "receiver_type"]
    ).to_frame(index=False)
    result = grid.merge(
        frame[["sender_type", "receiver_type", score_column]],
        on=["sender_type", "receiver_type"],
        how="left",
        validate="one_to_one",
    )
    result[score_column] = pd.to_numeric(result[score_column], errors="raise").fillna(
        0.0
    )
    return result


def pairwise_rank_metrics(
    long_scores: pd.DataFrame, *, top_fraction: float = 0.20
) -> pd.DataFrame:
    required = {"dataset", "sender_type", "receiver_type", "method", "score"}
    if not required.issubset(long_scores.columns):
        raise ValueError(
            f"long_scores is missing {sorted(required.difference(long_scores.columns))}"
        )
    rows: list[dict[str, object]] = []
    for dataset, table in long_scores.groupby("dataset", sort=True):
        pivot = table.pivot(
            index=["sender_type", "receiver_type"], columns="method", values="score"
        )
        if set(pivot.columns) != set(METHODS):
            raise ValueError(f"{dataset} does not contain all four methods")
        top_k = max(1, int(math.ceil(len(pivot) * top_fraction)))
        for left_index, left in enumerate(METHODS):
            for right in METHODS[left_index + 1 :]:
                rho = stats.spearmanr(pivot[left], pivot[right]).statistic
                tau = stats.kendalltau(pivot[left], pivot[right]).statistic
                left_top = set(pivot[left].nlargest(top_k).index)
                right_top = set(pivot[right].nlargest(top_k).index)
                intersection = len(left_top & right_top)
                union = len(left_top | right_top)
                rows.append(
                    {
                        "dataset": dataset,
                        "left_method": left,
                        "right_method": right,
                        "n_directed_pairs": len(pivot),
                        "spearman_rho": float(rho),
                        "kendall_tau_b": float(tau),
                        "top_fraction": float(top_fraction),
                        "top_k": top_k,
                        "top_k_intersection": intersection,
                        "top_k_jaccard": intersection / union if union else 0.0,
                    }
                )
    return pd.DataFrame(rows)
