"""Non-circular zebrafish interaction and ligand-receptor validation.

This module supplies the numerical contracts used by the reviewer-facing
zebrafish validation workflow.  It deliberately keeps three questions
separate:

* directed cell-type-pair reproducibility across methods;
* external ranks of pairs selected by CytoBridge alone; and
* post-hoc ligand-receptor (LR) interpretation of the complete CytoBridge
  pair field, with an explicit LR-expression-only baseline.

CytoBridge attention is a signed model gate, not an LR identifier or a
probability.  LR scores produced here therefore always combine a separately
measured ligand/receptor activity matrix with a declared CytoBridge modifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


PAIR_KEYS = ("sender_type", "receiver_type")
CYTOBRIDGE_VIEWS = {
    "attention": "G_AB_attention_mean_mean",
    "exact_message": "D_AB_joint_mean",
}
PRIMARY_TOP_FRACTION = 0.20
PRIMARY_PERMUTATIONS = 1000
PRIMARY_RANDOM_SEED = 20260816


@dataclass(frozen=True)
class RankMetrics:
    """Rank agreement on one explicitly shared candidate universe."""

    n_shared: int
    spearman_rho: float
    spearman_pvalue: float
    top_fraction: float
    top_n: int
    top_jaccard: float
    top_overlap_n: int


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, label: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _finite_numeric(values: pd.Series, *, label: str) -> np.ndarray:
    result = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def rank_percentile(values: Sequence[float] | pd.Series) -> np.ndarray:
    """Return average within-vector percentile ranks in [0, 1]."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("rank values must be a finite one-dimensional array")
    if array.size == 0:
        return np.asarray([], dtype=float)
    if array.size == 1:
        return np.asarray([1.0], dtype=float)
    return stats.rankdata(array, method="average") / float(array.size)


def positive_rank_weights(values: Sequence[float] | pd.Series) -> np.ndarray:
    """Rank positive values while leaving structural zero pairs at zero."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("positive-rank values must be finite and nonnegative")
    result = np.zeros_like(array, dtype=float)
    positive = array > 0
    if positive.any():
        result[positive] = rank_percentile(array[positive])
    return result


def _positive_top_indices(values: np.ndarray, fraction: float) -> set[int]:
    if not 0 < fraction <= 1:
        raise ValueError("top fraction must lie in (0, 1]")
    values = np.asarray(values, dtype=float)
    positive = np.flatnonzero(values > 0)
    if positive.size == 0:
        return set()
    n = max(1, int(math.ceil(float(fraction) * positive.size)))
    order = np.argsort(-values[positive], kind="mergesort")
    return set(positive[order[:n]].tolist())


def rank_metrics(
    left: Sequence[float],
    right: Sequence[float],
    *,
    top_fraction: float = PRIMARY_TOP_FRACTION,
) -> RankMetrics:
    """Compare two score vectors without comparing their raw units."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("rank-metric vectors must be one-dimensional and aligned")
    if left_array.size < 3:
        raise ValueError("rank metrics require at least three shared candidates")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("rank-metric vectors contain non-finite values")
    if np.unique(left_array).size < 2 or np.unique(right_array).size < 2:
        rho = float("nan")
        pvalue = float("nan")
    else:
        result = stats.spearmanr(left_array, right_array)
        rho = float(result.statistic)
        pvalue = float(result.pvalue)
    left_top = _positive_top_indices(left_array, top_fraction)
    right_top = _positive_top_indices(right_array, top_fraction)
    union = left_top | right_top
    overlap = left_top & right_top
    return RankMetrics(
        n_shared=int(left_array.size),
        spearman_rho=rho,
        spearman_pvalue=pvalue,
        top_fraction=float(top_fraction),
        top_n=max(len(left_top), len(right_top)),
        top_jaccard=float(len(overlap) / len(union)) if union else float("nan"),
        top_overlap_n=int(len(overlap)),
    )


def complete_directed_pair_table(
    frame: pd.DataFrame,
    *,
    score_column: str,
    cell_types: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Complete one score table onto an exact directed cell-type grid."""

    _require_columns(frame, (*PAIR_KEYS, score_column), label="pair table")
    local = frame[[*PAIR_KEYS, score_column]].copy()
    if local.duplicated(list(PAIR_KEYS)).any():
        raise ValueError("pair table contains duplicate directed pairs")
    if cell_types is None:
        types = sorted(
            set(local["sender_type"].astype(str))
            | set(local["receiver_type"].astype(str))
        )
    else:
        types = sorted({str(value) for value in cell_types})
    if not types:
        raise ValueError("directed pair grid has no cell types")
    grid = pd.MultiIndex.from_product([types, types], names=list(PAIR_KEYS)).to_frame(
        index=False
    )
    result = grid.merge(local, on=list(PAIR_KEYS), how="left", validate="one_to_one")
    result[score_column] = pd.to_numeric(result[score_column], errors="raise").fillna(
        0.0
    )
    _finite_numeric(result[score_column], label=score_column)
    return result


def pair_method_concordance(
    cytobridge: pd.DataFrame,
    external: pd.DataFrame,
    *,
    cytobridge_score: str,
    external_score: str,
    top_fraction: float = PRIMARY_TOP_FRACTION,
) -> RankMetrics:
    """Compare methods on the union-complete directed pair universe."""

    types = sorted(
        set(cytobridge["sender_type"].astype(str))
        | set(cytobridge["receiver_type"].astype(str))
        | set(external["sender_type"].astype(str))
        | set(external["receiver_type"].astype(str))
    )
    left = complete_directed_pair_table(
        cytobridge, score_column=cytobridge_score, cell_types=types
    )
    right = complete_directed_pair_table(
        external, score_column=external_score, cell_types=types
    )
    merged = left.merge(right, on=list(PAIR_KEYS), validate="one_to_one")
    return rank_metrics(
        merged[cytobridge_score],
        merged[external_score],
        top_fraction=top_fraction,
    )


def _pair_covariate_matrix(pair_grid: pd.DataFrame) -> np.ndarray:
    """Return the frozen nuisance design for pair-level rank residuals."""

    required = (
        "sender_type",
        "receiver_type",
        "n_sender_cells_mean",
        "n_receiver_cells_mean",
        "spatial_distance_mean_mean",
    )
    _require_columns(pair_grid, required, label="pair covariates")
    sender_n = _finite_numeric(
        pair_grid["n_sender_cells_mean"], label="n_sender_cells_mean"
    )
    receiver_n = _finite_numeric(
        pair_grid["n_receiver_cells_mean"], label="n_receiver_cells_mean"
    )
    if np.any(sender_n < 0) or np.any(receiver_n < 0):
        raise ValueError("pair abundances must be nonnegative")
    distance = pd.to_numeric(
        pair_grid["spatial_distance_mean_mean"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.isfinite(distance).any():
        fill = float(np.nanmedian(distance[np.isfinite(distance)]))
    else:
        fill = 0.0
    distance = np.where(np.isfinite(distance), distance, fill)
    self_pair = (
        pair_grid["sender_type"].astype(str) == pair_grid["receiver_type"].astype(str)
    ).to_numpy(dtype=float)
    return np.column_stack(
        [
            np.ones(len(pair_grid), dtype=float),
            np.log1p(sender_n),
            np.log1p(receiver_n),
            distance,
            self_pair,
        ]
    )


def _rank_residuals(values: Sequence[float], design: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.shape[0] != design.shape[0]:
        raise ValueError("rank-residual values do not match the design")
    if not np.isfinite(array).all() or not np.isfinite(design).all():
        raise ValueError("rank-residual inputs contain non-finite values")
    ranks = rank_percentile(array)
    coefficients, _, _, _ = np.linalg.lstsq(design, ranks, rcond=None)
    return ranks - design @ coefficients


def controlled_pair_concordance(
    pair_grid: pd.DataFrame,
    external: pd.DataFrame,
    *,
    cytobridge_score: str,
    external_score: str,
    permutations: int = PRIMARY_PERMUTATIONS,
    seed: int = PRIMARY_RANDOM_SEED,
) -> dict[str, float | int]:
    """Pair-rank agreement after abundance, distance, and self-pair adjustment.

    The empirical null permutes the CytoBridge score only within the frozen
    abundance/distance/self strata.  It therefore asks whether the observed
    rank agreement exceeds what those coarse nuisance structures can explain.
    """

    if permutations < 1:
        raise ValueError("controlled-pair permutations must be positive")
    _require_columns(pair_grid, (*PAIR_KEYS, cytobridge_score), label="pair grid")
    if pair_grid.duplicated(list(PAIR_KEYS)).any():
        raise ValueError("pair grid contains duplicate directed pairs")
    types = sorted(
        set(pair_grid["sender_type"].astype(str))
        | set(pair_grid["receiver_type"].astype(str))
        | set(external["sender_type"].astype(str))
        | set(external["receiver_type"].astype(str))
    )
    expected = complete_directed_pair_table(
        external, score_column=external_score, cell_types=types
    )
    merged = pair_grid.merge(expected, on=list(PAIR_KEYS), validate="one_to_one")
    if len(merged) != len(pair_grid):
        raise ValueError("external pair completion changed the frozen pair universe")
    design = _pair_covariate_matrix(merged)
    left = _finite_numeric(merged[cytobridge_score], label=cytobridge_score)
    right = _finite_numeric(merged[external_score], label=external_score)
    left_residual = _rank_residuals(left, design)
    right_residual = _rank_residuals(right, design)
    observed = stats.spearmanr(left_residual, right_residual)
    strata = adaptive_pair_strata(merged)
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(permutations), dtype=float)
    for iteration in range(int(permutations)):
        shuffled = left.copy()
        for indices in groups:
            if len(indices) > 1:
                shuffled[indices] = left[rng.permutation(indices)]
        shuffled_residual = _rank_residuals(shuffled, design)
        null[iteration] = float(
            stats.spearmanr(shuffled_residual, right_residual).statistic
        )
    finite = null[np.isfinite(null)]
    observed_rho = float(observed.statistic)
    if not np.isfinite(observed_rho) or finite.size == 0:
        raise ValueError("controlled pair concordance is degenerate")
    return {
        "n_pairs": int(len(merged)),
        "n_covariates_including_intercept": int(design.shape[1]),
        "n_strata": int(len(groups)),
        "n_permutations": int(permutations),
        "adjusted_spearman_rho": observed_rho,
        "adjusted_spearman_pvalue_asymptotic": float(observed.pvalue),
        "null_adjusted_spearman_mean": float(np.mean(finite)),
        "null_adjusted_spearman_q025": float(np.quantile(finite, 0.025)),
        "null_adjusted_spearman_q975": float(np.quantile(finite, 0.975)),
        "adjusted_spearman_empirical_p_upper": float(
            (1 + np.sum(finite >= observed_rho)) / (1 + len(finite))
        ),
    }


def select_pairs_by_cytobridge_only(
    cytobridge: pd.DataFrame,
    *,
    score_column: str,
    n_pairs: int,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """Freeze important pairs before any external score is inspected."""

    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    _require_columns(cytobridge, (*PAIR_KEYS, score_column), label="CytoBridge table")
    local = cytobridge[[*PAIR_KEYS, score_column]].copy()
    if local.duplicated(list(PAIR_KEYS)).any():
        raise ValueError("CytoBridge table contains duplicate directed pairs")
    local[score_column] = _finite_numeric(local[score_column], label=score_column)
    if exclude_self:
        local = local.loc[
            local["sender_type"].astype(str) != local["receiver_type"].astype(str)
        ].copy()
    local = local.loc[local[score_column] > 0].copy()
    local = local.sort_values(
        [score_column, "sender_type", "receiver_type"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(int(n_pairs))
    if local.empty:
        raise ValueError("no positive CytoBridge pairs are available for selection")
    local.insert(0, "cytobridge_selection_rank", np.arange(1, len(local) + 1))
    local["selection_rule"] = (
        f"top {len(local)} positive off-diagonal pairs by {score_column}; "
        "external scores were not used"
    )
    return local.reset_index(drop=True)


def external_ranks_for_selected_pairs(
    selected: pd.DataFrame,
    external: pd.DataFrame,
    *,
    external_score: str,
    external_method: str,
) -> pd.DataFrame:
    """Attach external ranks only after the CytoBridge selection is frozen."""

    _require_columns(
        selected, (*PAIR_KEYS, "cytobridge_selection_rank"), label="selection"
    )
    _require_columns(external, (*PAIR_KEYS, external_score), label=external_method)
    types = sorted(
        set(external["sender_type"].astype(str))
        | set(external["receiver_type"].astype(str))
        | set(selected["sender_type"].astype(str))
        | set(selected["receiver_type"].astype(str))
    )
    complete = complete_directed_pair_table(
        external, score_column=external_score, cell_types=types
    )
    complete["external_rank_percentile"] = rank_percentile(complete[external_score])
    complete["external_rank_descending"] = (
        complete[external_score].rank(method="min", ascending=False).astype(int)
    )
    result = selected.merge(complete, on=list(PAIR_KEYS), validate="one_to_one")
    result["external_method"] = str(external_method)
    result["external_score_column"] = str(external_score)
    return result


def collapse_lr_database(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse database duplicates to one case-insensitive LR candidate."""

    _require_columns(frame, ("ligand", "receptor"), label="LR database")
    local = frame.copy()
    for column in ("ligand", "receptor"):
        local[column] = local[column].fillna("").astype(str).str.strip()
    local = local.loc[local["ligand"].ne("") & local["receptor"].ne("")].copy()
    local["ligand_key"] = local["ligand"].str.casefold()
    local["receptor_key"] = local["receptor"].str.casefold()
    if "pathway" not in local:
        local["pathway"] = ""
    if "category" not in local:
        local["category"] = ""

    def joined(values: pd.Series) -> str:
        return ";".join(sorted({str(value) for value in values if str(value)}))

    result = (
        local.groupby(["ligand_key", "receptor_key"], sort=True, as_index=False)
        .agg(
            ligand=("ligand", "first"),
            receptor=("receptor", "first"),
            pathways=("pathway", joined),
            categories=("category", joined),
            database_rows=("ligand", "size"),
        )
        .reset_index(drop=True)
    )
    result["lr_id"] = result["ligand_key"] + "->" + result["receptor_key"]
    if result["lr_id"].duplicated().any():
        raise AssertionError("collapsed LR identifiers are not unique")
    return result


def scaled_expression_by_type(
    expression: pd.DataFrame,
    labels: Sequence[str],
    *,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """Mean expression per type, scaled per gene by its positive-cell quantile."""

    if not 0 < quantile <= 1:
        raise ValueError("expression quantile must lie in (0, 1]")
    if len(expression) != len(labels):
        raise ValueError("expression rows and labels have different lengths")
    values = expression.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("expression must be finite and nonnegative")
    scaled = np.zeros_like(values, dtype=float)
    for column in range(values.shape[1]):
        positive = values[:, column][values[:, column] > 0]
        scale = float(np.quantile(positive, quantile)) if positive.size else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        scaled[:, column] = np.clip(values[:, column] / scale, 0.0, 1.0)
    table = pd.DataFrame(scaled, columns=expression.columns)
    table.insert(0, "cell_type", [str(value) for value in labels])
    return table.groupby("cell_type", sort=True).mean(numeric_only=True)


def build_pair_lr_activity_matrix(
    pair_grid: pd.DataFrame,
    expression_by_type: pd.DataFrame,
    lr_candidates: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Map every directed type pair to every representable LR candidate."""

    _require_columns(pair_grid, PAIR_KEYS, label="pair grid")
    _require_columns(
        lr_candidates,
        ("ligand_key", "receptor_key", "lr_id"),
        label="LR candidates",
    )
    if pair_grid.duplicated(list(PAIR_KEYS)).any():
        raise ValueError("pair grid contains duplicates")
    gene_lookup = {str(column).casefold(): str(column) for column in expression_by_type}
    keep = lr_candidates["ligand_key"].isin(gene_lookup) & lr_candidates[
        "receptor_key"
    ].isin(gene_lookup)
    represented = lr_candidates.loc[keep].copy().reset_index(drop=True)
    if represented.empty:
        raise ValueError("no LR candidates are representable in expression")
    missing_types = (
        set(pair_grid["sender_type"].astype(str))
        | set(pair_grid["receiver_type"].astype(str))
    ).difference(expression_by_type.index.astype(str))
    if missing_types:
        raise ValueError(
            f"expression lacks pair-grid cell types: {sorted(missing_types)}"
        )
    sender = pair_grid["sender_type"].astype(str).tolist()
    receiver = pair_grid["receiver_type"].astype(str).tolist()
    matrix = np.empty((len(pair_grid), len(represented)), dtype=np.float64)
    for column, row in enumerate(represented.itertuples(index=False)):
        ligand = gene_lookup[str(row.ligand_key)]
        receptor = gene_lookup[str(row.receptor_key)]
        left = expression_by_type.loc[sender, ligand].to_numpy(dtype=float)
        right = expression_by_type.loc[receiver, receptor].to_numpy(dtype=float)
        matrix[:, column] = left * right
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise AssertionError("constructed LR activity matrix is invalid")
    return matrix, represented


def lr_scores_from_pair_modifiers(
    activity: np.ndarray,
    represented_lr: pd.DataFrame,
    modifiers: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Score LR candidates using LR-only and declared pair-level modifiers."""

    matrix = np.asarray(activity, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(represented_lr):
        raise ValueError("LR activity matrix shape does not match candidates")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("LR activity matrix is invalid")
    result = represented_lr.copy()
    result["lr_only_score"] = matrix.mean(axis=0)
    result["lr_only_rank_percentile"] = rank_percentile(result["lr_only_score"])
    for name, values in modifiers.items():
        weights = np.asarray(values, dtype=float)
        if weights.shape != (matrix.shape[0],) or not np.isfinite(weights).all():
            raise ValueError(f"modifier {name!r} is invalid")
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError(f"modifier {name!r} must be nonnegative and nonzero")
        score = np.average(matrix, axis=0, weights=weights)
        result[f"{name}_score"] = score
        result[f"{name}_rank_percentile"] = rank_percentile(score)
        result[f"{name}_minus_lr_only"] = score - result["lr_only_score"]
    return result


def collapse_commot_lr_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate COMMOT over all directed type pairs for each LR."""

    score = "abundance_controlled_distinct_cell_score"
    _require_columns(frame, ("ligand", "receptor", score), label="COMMOT LR")
    local = frame.copy()
    local["ligand_key"] = local["ligand"].astype(str).str.casefold()
    local["receptor_key"] = local["receptor"].astype(str).str.casefold()
    local[score] = _finite_numeric(local[score], label=score)
    result = local.groupby(
        ["ligand_key", "receptor_key"], as_index=False, sort=True
    ).agg(commot_score=(score, "sum"), commot_nonzero_contexts=(score, "size"))
    result["lr_id"] = result["ligand_key"] + "->" + result["receptor_key"]
    result["commot_rank_percentile"] = rank_percentile(result["commot_score"])
    return result


def collapse_nichenet_lr_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate NicheNet receiver-transition evidence for each LR.

    NicheNet activity is ligand/receiver-transition evidence.  Receptors enter
    through the explicit receptor-expression gate; the resulting value is not
    interpreted as a native cell-pair communication strength.
    """

    score = "lr_evidence"
    _require_columns(frame, ("ligand", "receptor", score), label="NicheNet LR")
    local = frame.copy()
    local["ligand_key"] = local["ligand"].astype(str).str.casefold()
    local["receptor_key"] = local["receptor"].astype(str).str.casefold()
    local[score] = _finite_numeric(local[score], label=score)
    result = local.groupby(
        ["ligand_key", "receptor_key"], as_index=False, sort=True
    ).agg(
        nichenet_score=(score, "mean"),
        nichenet_max_score=(score, "max"),
        nichenet_contexts=(score, "size"),
    )
    result["lr_id"] = result["ligand_key"] + "->" + result["receptor_key"]
    result["nichenet_rank_percentile"] = rank_percentile(result["nichenet_score"])
    return result


def shared_lr_rank_metrics(
    cytobridge_lr: pd.DataFrame,
    external_lr: pd.DataFrame,
    *,
    external_score: str,
    views: Sequence[str] = ("lr_only", "attention", "exact_message"),
    top_fraction: float = PRIMARY_TOP_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every CytoBridge-derived LR view on one shared universe."""

    _require_columns(cytobridge_lr, ("lr_id",), label="CytoBridge LR")
    _require_columns(external_lr, ("lr_id", external_score), label="external LR")
    merged = cytobridge_lr.merge(
        external_lr, on="lr_id", how="inner", validate="one_to_one"
    )
    if len(merged) < 3:
        raise ValueError("fewer than three LR candidates are shared")
    rows: list[dict[str, object]] = []
    for view in views:
        score = f"{view}_score"
        if score not in merged:
            raise ValueError(f"CytoBridge LR table lacks {score!r}")
        metrics = rank_metrics(
            merged[score], merged[external_score], top_fraction=top_fraction
        )
        rows.append({"cytobridge_view": view, **metrics.__dict__})
    return merged, pd.DataFrame(rows)


def adaptive_pair_strata(pair_grid: pd.DataFrame) -> np.ndarray:
    """Freeze broad geometry/abundance strata for modifier permutations."""

    required = (
        "sender_type",
        "receiver_type",
        "n_sender_cells_mean",
        "n_receiver_cells_mean",
        "spatial_distance_mean_mean",
    )
    _require_columns(pair_grid, required, label="pair grid")
    local = pair_grid.copy()
    abundance = np.sqrt(
        pd.to_numeric(local["n_sender_cells_mean"], errors="raise")
        * pd.to_numeric(local["n_receiver_cells_mean"], errors="raise")
    )
    distance = pd.to_numeric(local["spatial_distance_mean_mean"], errors="coerce")

    def bin_values(values: pd.Series, q: int = 4) -> pd.Series:
        finite = values.replace([np.inf, -np.inf], np.nan)
        filled = finite.fillna(finite.max() if finite.notna().any() else 0.0)
        if filled.nunique() < 2:
            return pd.Series(np.zeros(len(filled), dtype=int), index=filled.index)
        bins = min(q, int(filled.nunique()))
        return pd.qcut(
            filled.rank(method="first"), q=bins, labels=False, duplicates="drop"
        ).astype(int)

    self_pair = (
        local["sender_type"].astype(str) == local["receiver_type"].astype(str)
    ).astype(int)
    abundance_bin = bin_values(pd.Series(abundance, index=local.index))
    distance_bin = bin_values(pd.Series(distance, index=local.index))
    return np.asarray(
        [
            f"self={self_value}|abundance={a_value}|distance={d_value}"
            for self_value, a_value, d_value in zip(
                self_pair, abundance_bin, distance_bin
            )
        ],
        dtype=str,
    )


def modifier_permutation_test(
    activity: np.ndarray,
    modifier: Sequence[float],
    external_scores: Sequence[float],
    lr_indices: Sequence[int],
    strata: Sequence[str],
    *,
    permutations: int = PRIMARY_PERMUTATIONS,
    seed: int = PRIMARY_RANDOM_SEED,
    top_fraction: float = PRIMARY_TOP_FRACTION,
) -> dict[str, float | int]:
    """Test whether a CytoBridge modifier adds LR-rank agreement beyond chance."""

    matrix = np.asarray(activity, dtype=float)
    weights = np.asarray(modifier, dtype=float)
    external = np.asarray(external_scores, dtype=float)
    columns = np.asarray(lr_indices, dtype=int)
    groups = np.asarray(strata, dtype=str)
    if matrix.ndim != 2 or weights.shape != (matrix.shape[0],):
        raise ValueError("activity/modifier shape mismatch")
    if groups.shape != weights.shape:
        raise ValueError("permutation strata shape mismatch")
    if columns.ndim != 1 or external.shape != columns.shape:
        raise ValueError("shared LR indices and external scores are misaligned")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    observed_scores = np.average(matrix[:, columns], axis=0, weights=weights)
    observed = rank_metrics(observed_scores, external, top_fraction=top_fraction)
    unique_groups = [np.flatnonzero(groups == value) for value in np.unique(groups)]
    rng = np.random.default_rng(int(seed))
    null_rho = np.empty(int(permutations), dtype=float)
    null_jaccard = np.empty(int(permutations), dtype=float)
    for iteration in range(int(permutations)):
        shuffled = weights.copy()
        for indices in unique_groups:
            if len(indices) > 1:
                shuffled[indices] = weights[rng.permutation(indices)]
        score = np.average(matrix[:, columns], axis=0, weights=shuffled)
        metric = rank_metrics(score, external, top_fraction=top_fraction)
        null_rho[iteration] = metric.spearman_rho
        null_jaccard[iteration] = metric.top_jaccard
    finite_rho = null_rho[np.isfinite(null_rho)]
    finite_jaccard = null_jaccard[np.isfinite(null_jaccard)]
    if finite_rho.size == 0 or finite_jaccard.size == 0:
        raise ValueError("modifier permutation produced no finite null metrics")
    return {
        "n_shared_lr": int(len(columns)),
        "n_pairs": int(matrix.shape[0]),
        "n_strata": int(len(unique_groups)),
        "n_permutations": int(permutations),
        "observed_spearman_rho": float(observed.spearman_rho),
        "null_spearman_mean": float(np.mean(finite_rho)),
        "null_spearman_q025": float(np.quantile(finite_rho, 0.025)),
        "null_spearman_q975": float(np.quantile(finite_rho, 0.975)),
        "spearman_empirical_p_upper": float(
            (1 + np.sum(finite_rho >= observed.spearman_rho)) / (1 + len(finite_rho))
        ),
        "observed_top_jaccard": float(observed.top_jaccard),
        "null_top_jaccard_mean": float(np.mean(finite_jaccard)),
        "null_top_jaccard_q025": float(np.quantile(finite_jaccard, 0.025)),
        "null_top_jaccard_q975": float(np.quantile(finite_jaccard, 0.975)),
        "top_jaccard_empirical_p_upper": float(
            (1 + np.sum(finite_jaccard >= observed.top_jaccard))
            / (1 + len(finite_jaccard))
        ),
    }


def paper_reference_enrichment(
    scores: pd.DataFrame,
    paper_axes: pd.DataFrame,
    *,
    score_columns: Sequence[str],
    top_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the paper's frozen 21 LR axes against a declared background."""

    _require_columns(scores, ("lr_id", *score_columns), label="LR scores")
    _require_columns(paper_axes, ("ligand", "receptor"), label="paper LR axes")
    reference = paper_axes.copy()
    reference["lr_id"] = (
        reference["ligand"].astype(str).str.casefold()
        + "->"
        + reference["receptor"].astype(str).str.casefold()
    )
    if reference["lr_id"].duplicated().any():
        raise ValueError("paper LR reference contains duplicates")
    annotated = scores.copy()
    annotated["paper_2022_reference"] = annotated["lr_id"].isin(reference["lr_id"])
    n_reference = int(annotated["paper_2022_reference"].sum())
    if n_reference < 2:
        raise ValueError("fewer than two paper LR axes are present in the background")
    rows: list[dict[str, float | int | str]] = []
    labels = annotated["paper_2022_reference"].to_numpy(bool)
    for column in score_columns:
        values = _finite_numeric(annotated[column], label=column)
        positives = values[labels]
        negatives = values[~labels]
        if negatives.size < 2:
            raise ValueError(
                "paper LR enrichment requires at least two background axes"
            )
        u_result = stats.mannwhitneyu(positives, negatives, alternative="greater")
        auc = float(u_result.statistic / (len(positives) * len(negatives)))
        top = _positive_top_indices(values, top_fraction)
        top_reference = int(sum(labels[index] for index in top))
        table = np.asarray(
            [
                [top_reference, len(top) - top_reference],
                [
                    n_reference - top_reference,
                    len(labels) - n_reference - len(top) + top_reference,
                ],
            ]
        )
        fisher = stats.fisher_exact(table, alternative="greater")
        rows.append(
            {
                "score_column": column,
                "n_background": int(len(values)),
                "n_paper_reference_present": n_reference,
                "paper_reference_auc": auc,
                "mannwhitney_p_greater": float(u_result.pvalue),
                "top_fraction": float(top_fraction),
                "top_n": int(len(top)),
                "paper_reference_in_top_n": top_reference,
                "fisher_odds_ratio": float(fisher.statistic),
                "fisher_p_greater": float(fisher.pvalue),
            }
        )
    annotated = annotated.merge(
        reference[["lr_id", "paper_display_order"]]
        if "paper_display_order" in reference
        else reference[["lr_id"]],
        on="lr_id",
        how="left",
        validate="one_to_one",
    )
    return annotated, pd.DataFrame(rows)


def jointly_supported_lr_targets(
    cytobridge_lr: pd.DataFrame,
    commot_lr: pd.DataFrame,
    nichenet_lr: pd.DataFrame,
    nichenet_targets: pd.DataFrame,
    *,
    cytobridge_view: str = "exact_message",
    top_fraction: float = PRIMARY_TOP_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Connect shared top-ranked LR axes to independent NicheNet targets."""

    score = f"{cytobridge_view}_score"
    _require_columns(cytobridge_lr, ("lr_id", score), label="CytoBridge LR")
    _require_columns(commot_lr, ("lr_id", "commot_score"), label="COMMOT LR")
    _require_columns(nichenet_lr, ("lr_id", "nichenet_score"), label="NicheNet LR")
    _require_columns(
        nichenet_targets,
        ("ligand", "receptor", "target", "ligand_target_evidence"),
        label="NicheNet target links",
    )
    shared = (
        cytobridge_lr[["lr_id", "ligand", "receptor", "pathways", score]]
        .merge(commot_lr[["lr_id", "commot_score"]], on="lr_id", validate="one_to_one")
        .merge(
            nichenet_lr[["lr_id", "nichenet_score"]], on="lr_id", validate="one_to_one"
        )
    )
    for column in (score, "commot_score", "nichenet_score"):
        shared[f"{column}_rank_percentile"] = rank_percentile(shared[column])
        top_indices = _positive_top_indices(
            shared[column].to_numpy(dtype=float), float(top_fraction)
        )
        shared[f"{column}_top"] = [index in top_indices for index in range(len(shared))]
    shared["jointly_supported"] = shared[
        [f"{score}_top", "commot_score_top", "nichenet_score_top"]
    ].all(axis=1)
    target = nichenet_targets.copy()
    target["lr_id"] = (
        target["ligand"].astype(str).str.casefold()
        + "->"
        + target["receptor"].astype(str).str.casefold()
    )
    links = shared.loc[shared["jointly_supported"]].merge(
        target,
        on="lr_id",
        how="left",
        suffixes=("", "_nichenet"),
        validate="one_to_many",
    )
    links = links.loc[links["target"].fillna("").astype(str).ne("")].copy()
    if not links.empty:
        links["ligand_target_evidence"] = pd.to_numeric(
            links["ligand_target_evidence"], errors="raise"
        )
        links = links.sort_values(
            ["lr_id", "ligand_target_evidence", "target"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    return shared, links
