"""Results for the zebrafish attention and control comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import read_json, require_files, resolve_results_dir


CONDITION_ORDER = ("trained", "pre_interaction", "random")
EXTERNAL_METHOD_ORDER = ("COMMOT", "CellAgentChat")

_FILES = (
    "directed_pair_concordance.csv",
    "expression_detection_by_stage_type.csv",
    "jam_compatibility_percentile_summary.csv",
    "jam_quartile_compatibility.csv",
    "myog_association.csv",
    "somite_18hpf_spatial_cells.csv.gz",
    "somite_18hpf_spatial_null_iterations.csv.gz",
    "somite_18hpf_spatial_null_summary.csv",
    "trained_jam_display_edges.csv",
    "type_pair_raw_attention_ranks.csv",
    "manifest.json",
)

_PAIR_COLUMNS = (
    "cytobridge_view",
    "external_method",
    "n_pairs",
    "n_strata",
    "n_permutations",
    "adjusted_spearman_rho",
    "null_adjusted_spearman_mean",
    "null_adjusted_spearman_q025",
    "null_adjusted_spearman_q975",
    "adjusted_spearman_empirical_p_upper",
)
_COMPATIBILITY_COLUMNS = (
    "condition",
    "compatibility_class",
    "n_directed_edges",
    "mean_attention_percentile",
    "median_attention_percentile",
)
_QUARTILE_COLUMNS = (
    "condition",
    "top_n_edges_after_boundary_ties",
    "top_n_jam_compatible",
    "top_n_non_compatible",
    "top_compatibility_rate",
    "bottom_n_edges_after_boundary_ties",
    "bottom_n_jam_compatible",
    "bottom_n_non_compatible",
    "bottom_compatibility_rate",
    "top_vs_bottom_odds_ratio",
    "fisher_exact_two_sided_p",
)
_TYPE_PAIR_COLUMNS = (
    "condition",
    "sender_type",
    "receiver_type",
    "raw_attention_mean",
    "n_directed_edges",
    "zero_completed_no_edge",
    "rank_from_top",
    "n_complete_directed_type_pairs",
)
_NULL_SUMMARY_COLUMNS = (
    "stage_label",
    "cell_type",
    "n_cells",
    "observed_jam2a_jam3b_orientation_compatible_pairs",
    "n_permutations",
    "null_mean",
    "null_q025",
    "null_q975",
    "observed_over_null_mean",
    "n_null_at_least_observed",
    "monte_carlo_upper_tail_p_plus1",
)
_NULL_ITERATION_COLUMNS = (
    "iteration",
    "orientation_compatible_pair_count",
    "at_least_observed",
)
_ASSOCIATION_COLUMNS = (
    "stage_label",
    "cell_type",
    "gene_a",
    "gene_b",
    "n_cells",
    "both_detected",
    "gene_a_only",
    "gene_b_only",
    "neither_detected",
    "fisher_odds_ratio",
    "fisher_two_sided_p",
)
_DETECTION_COLUMNS = (
    "stage_label",
    "cell_type",
    "gene",
    "n_cells",
    "n_detected",
    "detected_fraction",
)
_CELL_COLUMNS = (
    "h5ad_index",
    "cell_type",
    "is_somite",
    "x",
    "y",
    "jam2a_positive",
    "jam3b_positive",
    "myog_positive",
)
_DISPLAY_EDGE_COLUMNS = (
    "display_rank",
    "jam_compatible_orientation",
    "source_x",
    "source_y",
    "target_x",
    "target_y",
)


@dataclass(frozen=True)
class ZebrafishAttentionPanels:
    """Tables calculated for the three figure panels."""

    external_agreement: pd.DataFrame
    jam_quartiles: pd.DataFrame
    somite_ranks: pd.DataFrame
    myog_detection: pd.DataFrame
    spatial_null: pd.DataFrame


@dataclass(frozen=True)
class ZebrafishAttentionResults:
    """Compact inputs and calculated tables for the zebrafish figure."""

    source_dir: Path
    manifest: dict[str, Any]
    directed_pair_concordance: pd.DataFrame
    compatibility_summary: pd.DataFrame
    quartile_compatibility: pd.DataFrame
    type_pair_ranks: pd.DataFrame
    spatial_null_summary: pd.DataFrame
    spatial_null_iterations: pd.DataFrame
    myog_association: pd.DataFrame
    expression_detection: pd.DataFrame
    spatial_cells: pd.DataFrame
    display_edges: pd.DataFrame
    panels: ZebrafishAttentionPanels


def _require_columns(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    source: Path,
) -> pd.DataFrame:
    missing = sorted(set(columns).difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    return table.copy()


def _numeric(
    table: pd.DataFrame,
    columns: tuple[str, ...],
    source: Path,
) -> pd.DataFrame:
    values = table.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains non-finite values in {list(columns)}")
    result = table.copy()
    result.loc[:, columns] = values
    return result


def _boolean(series: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.casefold()
    values = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )
    if values.isna().any():
        invalid = sorted(normalized.loc[values.isna()].unique())[:5]
        raise ValueError(f"{label} contains invalid boolean values: {invalid}")
    return values.astype(bool)


def _conditions(series: pd.Series, *, label: str) -> pd.Series:
    values = series.astype(str).str.strip().str.casefold()
    if set(values) != set(CONDITION_ORDER):
        raise ValueError(
            f"{label} must contain conditions {list(CONDITION_ORDER)}; "
            f"observed {sorted(set(values))}"
        )
    return values


def _validate_manifest(manifest: dict[str, Any], source: Path) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{source} has an unsupported schema version")
    if manifest.get("analysis") != "zebrafish_attention":
        raise ValueError(f"{source} does not describe zebrafish attention")
    if set(manifest.get("files", {})) != set(_FILES).difference({"manifest.json"}):
        raise ValueError(f"{source} has an unexpected file roster")
    calculation = manifest.get("calculation", {})
    external = calculation.get("external_agreement", {})
    if (
        external.get("cytobridge_view") != "attention"
        or external.get("directed_cell_type_pairs") != 361
        or tuple(external.get("external_methods", ())) != EXTERNAL_METHOD_ORDER
        or external.get("structured_null_permutations") != 1000
    ):
        raise ValueError(f"{source} has an unexpected external-agreement scope")
    jam = calculation.get("jam_compatibility", {})
    if (
        tuple(jam.get("conditions", ())) != CONDITION_ORDER
        or jam.get("stage") != "18hpf"
        or jam.get("cell_type") != "Somite"
        or jam.get("cells") != 375
        or jam.get("directed_edges_per_condition") != 677
    ):
        raise ValueError(f"{source} has an unexpected JAM comparison scope")
    spatial = calculation.get("spatial_context", {})
    expected = {
        "display_edges": 15,
        "label_permutations": 10000,
        "observed_complementary_neighbor_pairs": 396,
    }
    if any(spatial.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{source} has an unexpected spatial scope")
    numerical = {
        "null_mean": 286.2387,
        "observed_over_null_mean": 1.383460727008612,
        "plus_one_upper_tail_p": 9.999000099990002e-05,
    }
    if any(
        not np.isclose(float(spatial.get(name, np.nan)), value, rtol=1e-12, atol=1e-15)
        for name, value in numerical.items()
    ):
        raise ValueError(f"{source} has unexpected spatial summary values")


def _validate_pair_concordance(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _PAIR_COLUMNS, source)
    if (
        len(result) != 4
        or result.duplicated(["cytobridge_view", "external_method"]).any()
    ):
        raise ValueError(f"{source} must contain four unique view-method rows")
    expected = {
        (view, method)
        for view in ("attention", "exact_message")
        for method in EXTERNAL_METHOD_ORDER
    }
    observed = set(
        zip(result["cytobridge_view"], result["external_method"], strict=True)
    )
    if observed != expected:
        raise ValueError(f"{source} has an unexpected view-method grid")
    numeric_columns = tuple(
        name
        for name in _PAIR_COLUMNS
        if name not in {"cytobridge_view", "external_method"}
    )
    result = _numeric(result, numeric_columns, source)
    if not result["n_pairs"].eq(361).all():
        raise ValueError(f"{source} must use all 361 directed cell-type pairs")
    if not result["n_permutations"].eq(1000).all():
        raise ValueError(f"{source} must use 1,000 structured-null permutations")
    if (
        result["null_adjusted_spearman_q025"] > result["null_adjusted_spearman_q975"]
    ).any():
        raise ValueError(f"{source} has reversed null intervals")
    probabilities = result["adjusted_spearman_empirical_p_upper"]
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{source} has probabilities outside [0, 1]")
    return result


def _validate_compatibility(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _COMPATIBILITY_COLUMNS, source)
    result["condition"] = _conditions(result["condition"], label=str(source))
    labels = result["compatibility_class"].astype(str).str.strip().str.casefold()
    mapping = {"jam-compatible": True, "non-compatible": False}
    result["jam_compatible"] = labels.map(mapping)
    if result["jam_compatible"].isna().any():
        raise ValueError(f"{source} has unknown compatibility classes")
    if len(result) != 6 or result.duplicated(["condition", "jam_compatible"]).any():
        raise ValueError(f"{source} must contain two rows per condition")
    result = _numeric(
        result,
        (
            "n_directed_edges",
            "mean_attention_percentile",
            "median_attention_percentile",
        ),
        source,
    )
    edge_totals = result.groupby("condition", sort=False)["n_directed_edges"].sum()
    if not edge_totals.eq(677).all():
        raise ValueError(f"{source} must contain 677 edges per condition")
    percentiles = result[["mean_attention_percentile", "median_attention_percentile"]]
    if ((percentiles < 0) | (percentiles > 1)).any().any():
        raise ValueError(f"{source} has attention percentiles outside [0, 1]")
    return result


def _validate_quartiles(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _QUARTILE_COLUMNS, source)
    result["condition"] = _conditions(result["condition"], label=str(source))
    if len(result) != 3 or result["condition"].duplicated().any():
        raise ValueError(f"{source} must contain one row per condition")
    numeric_columns = tuple(name for name in _QUARTILE_COLUMNS if name != "condition")
    result = _numeric(result, numeric_columns, source)
    for prefix in ("top", "bottom"):
        total = result[f"{prefix}_n_edges_after_boundary_ties"]
        counted = (
            result[f"{prefix}_n_jam_compatible"] + result[f"{prefix}_n_non_compatible"]
        )
        if not np.array_equal(total.to_numpy(int), counted.to_numpy(int)):
            raise ValueError(f"{source} has inconsistent {prefix}-quartile counts")
        rate = result[f"{prefix}_n_jam_compatible"] / total
        if not np.allclose(
            rate,
            result[f"{prefix}_compatibility_rate"],
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(f"{source} has inconsistent {prefix}-quartile rates")
    if (result["top_vs_bottom_odds_ratio"] <= 0).any():
        raise ValueError(f"{source} has non-positive odds ratios")
    return result


def _validate_type_pairs(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _TYPE_PAIR_COLUMNS, source)
    result["condition"] = _conditions(result["condition"], label=str(source))
    if (
        len(result) != 588
        or result.duplicated(["condition", "sender_type", "receiver_type"]).any()
    ):
        raise ValueError(f"{source} must contain 196 directed type pairs per condition")
    result["zero_completed_no_edge"] = _boolean(
        result["zero_completed_no_edge"], label=f"{source} zero-completed flag"
    )
    result = _numeric(
        result,
        (
            "raw_attention_mean",
            "n_directed_edges",
            "rank_from_top",
            "n_complete_directed_type_pairs",
        ),
        source,
    )
    if not result["n_complete_directed_type_pairs"].eq(196).all():
        raise ValueError(f"{source} must rank a complete 196-pair field")
    somite = result.loc[
        result["sender_type"].astype(str).eq("Somite")
        & result["receiver_type"].astype(str).eq("Somite")
    ]
    if len(somite) != 3 or set(somite["condition"]) != set(CONDITION_ORDER):
        raise ValueError(f"{source} lacks one Somite-to-Somite row per condition")
    return result


def _validate_spatial_null(
    summary: pd.DataFrame,
    iterations: pd.DataFrame,
    summary_source: Path,
    iteration_source: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = _require_columns(summary, _NULL_SUMMARY_COLUMNS, summary_source)
    if len(result) != 1:
        raise ValueError(f"{summary_source} must contain one row")
    result = _numeric(
        result,
        tuple(
            name
            for name in _NULL_SUMMARY_COLUMNS
            if name not in {"stage_label", "cell_type"}
        ),
        summary_source,
    )
    row = result.iloc[0]
    if row["stage_label"] != "18hpf" or row["cell_type"] != "Somite":
        raise ValueError(f"{summary_source} must describe 18 hpf Somite cells")
    if int(row["n_cells"]) != 375 or int(row["n_permutations"]) != 10000:
        raise ValueError(
            f"{summary_source} has an unexpected cell or permutation count"
        )

    draws = _require_columns(iterations, _NULL_ITERATION_COLUMNS, iteration_source)
    if len(draws) != 10000 or draws["iteration"].duplicated().any():
        raise ValueError(f"{iteration_source} must contain 10,000 unique iterations")
    draws = _numeric(
        draws,
        ("iteration", "orientation_compatible_pair_count"),
        iteration_source,
    )
    draws["at_least_observed"] = _boolean(
        draws["at_least_observed"], label=f"{iteration_source} exceedance flag"
    )
    if not np.array_equal(
        np.sort(draws["iteration"].to_numpy(int)), np.arange(1, 10001)
    ):
        raise ValueError(
            f"{iteration_source} must use iteration numbers 1 through 10,000"
        )
    counts = draws["orientation_compatible_pair_count"].to_numpy(float)
    observed = int(row["observed_jam2a_jam3b_orientation_compatible_pairs"])
    exceedances = counts >= observed
    if not np.array_equal(exceedances, draws["at_least_observed"].to_numpy(bool)):
        raise ValueError(f"{iteration_source} has inconsistent exceedance flags")
    calculated = {
        "null_mean": float(np.mean(counts)),
        "null_q025": float(np.quantile(counts, 0.025)),
        "null_q975": float(np.quantile(counts, 0.975)),
        "observed_over_null_mean": float(observed / np.mean(counts)),
        "n_null_at_least_observed": int(exceedances.sum()),
        "monte_carlo_upper_tail_p_plus1": float(
            (exceedances.sum() + 1) / (counts.size + 1)
        ),
    }
    for name, value in calculated.items():
        if not np.isclose(float(row[name]), value, rtol=1e-12, atol=1e-15):
            raise ValueError(f"{summary_source} disagrees with permutations for {name}")
    return result, draws


def _validate_association(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _ASSOCIATION_COLUMNS, source)
    result["gene_key"] = result["gene_a"].astype(str).str.strip().str.casefold()
    if (
        len(result) != 2
        or set(result["gene_key"]) != {"jam2a", "jam3b"}
        or not result["gene_b"].astype(str).str.casefold().eq("myog").all()
        or not result["stage_label"].astype(str).eq("18hpf").all()
        or not result["cell_type"].astype(str).eq("Somite").all()
    ):
        raise ValueError(f"{source} must contain the two 18 hpf Somite JAM-myog rows")
    numeric_columns = tuple(
        name
        for name in _ASSOCIATION_COLUMNS
        if name not in {"stage_label", "cell_type", "gene_a", "gene_b"}
    )
    result = _numeric(result, numeric_columns, source)
    counts = result[["both_detected", "gene_a_only", "gene_b_only", "neither_detected"]]
    if (
        not counts.sum(axis=1).eq(result["n_cells"]).all()
        or not result["n_cells"].eq(375).all()
    ):
        raise ValueError(f"{source} has inconsistent cell counts")
    probabilities = result["fisher_two_sided_p"]
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{source} has probabilities outside [0, 1]")
    return result


def _validate_detection(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _DETECTION_COLUMNS, source)
    result = _numeric(
        result,
        ("n_cells", "n_detected", "detected_fraction"),
        source,
    )
    if result.duplicated(["stage_label", "cell_type", "gene"]).any():
        raise ValueError(f"{source} contains duplicate stage-type-gene rows")
    expected_fraction = result["n_detected"] / result["n_cells"]
    if not np.allclose(
        expected_fraction, result["detected_fraction"], rtol=1e-12, atol=1e-15
    ):
        raise ValueError(f"{source} has inconsistent detection fractions")
    somite = result.loc[
        result["stage_label"].astype(str).eq("18hpf")
        & result["cell_type"].astype(str).eq("Somite")
        & result["gene"].astype(str).str.casefold().isin({"jam2a", "jam3b", "myog"})
    ]
    if len(somite) != 3 or not somite["n_cells"].eq(375).all():
        raise ValueError(f"{source} lacks the three 18 hpf Somite detection rows")
    return result


def _validate_cells(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _CELL_COLUMNS, source)
    if len(result) != 3048 or result["h5ad_index"].duplicated().any():
        raise ValueError(f"{source} must contain 3,048 unique tissue cells")
    result = _numeric(result, ("h5ad_index", "x", "y"), source)
    for name in ("is_somite", "jam2a_positive", "jam3b_positive", "myog_positive"):
        result[name] = _boolean(result[name], label=f"{source} {name}")
    if int(result["is_somite"].sum()) != 375:
        raise ValueError(f"{source} must contain 375 Somite cells")
    return result


def _validate_display_edges(table: pd.DataFrame, source: Path) -> pd.DataFrame:
    result = _require_columns(table, _DISPLAY_EDGE_COLUMNS, source)
    if len(result) != 15 or result["display_rank"].duplicated().any():
        raise ValueError(f"{source} must contain the fixed 15-edge display roster")
    result = _numeric(
        result,
        ("display_rank", "source_x", "source_y", "target_x", "target_y"),
        source,
    )
    result = result.sort_values("display_rank", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(result["display_rank"].to_numpy(int), np.arange(1, 16)):
        raise ValueError(f"{source} must retain display ranks 1 through 15")
    orientation = (
        result["jam_compatible_orientation"].astype(str).str.strip().str.casefold()
    )
    result["jam_compatible"] = ~orientation.isin({"", "none", "nan"})
    if not result["jam_compatible"].all():
        raise ValueError(f"{source} must contain only JAM-compatible display edges")
    return result


def calculate_zebrafish_attention_panels(
    directed_pair_concordance: pd.DataFrame,
    quartile_compatibility: pd.DataFrame,
    type_pair_ranks: pd.DataFrame,
    spatial_null_summary: pd.DataFrame,
    spatial_null_iterations: pd.DataFrame,
    myog_association: pd.DataFrame,
) -> ZebrafishAttentionPanels:
    """Calculate the values drawn in the figure."""

    external = directed_pair_concordance.loc[
        directed_pair_concordance["cytobridge_view"].eq("attention")
    ].copy()
    external["external_method"] = pd.Categorical(
        external["external_method"], EXTERNAL_METHOD_ORDER, ordered=True
    )
    external = external.sort_values("external_method").reset_index(drop=True)
    external["external_method"] = external["external_method"].astype(str)

    quartiles = quartile_compatibility.copy()
    quartiles["condition"] = pd.Categorical(
        quartiles["condition"], CONDITION_ORDER, ordered=True
    )
    quartiles = quartiles.sort_values("condition").reset_index(drop=True)
    quartiles["condition"] = quartiles["condition"].astype(str)
    quartiles["top_compatibility_percent"] = quartiles["top_compatibility_rate"] * 100
    quartiles["bottom_compatibility_percent"] = (
        quartiles["bottom_compatibility_rate"] * 100
    )

    somite = type_pair_ranks.loc[
        type_pair_ranks["sender_type"].astype(str).eq("Somite")
        & type_pair_ranks["receiver_type"].astype(str).eq("Somite")
    ].copy()
    somite["condition"] = pd.Categorical(
        somite["condition"], CONDITION_ORDER, ordered=True
    )
    somite = somite.sort_values("condition").reset_index(drop=True)
    somite["condition"] = somite["condition"].astype(str)
    somite["top_rank_percentile"] = 1 - (
        (somite["rank_from_top"] - 1) / (somite["n_complete_directed_type_pairs"] - 1)
    )

    association = myog_association.set_index("gene_key").loc[["jam3b", "jam2a"]]
    positive_denominator = association["both_detected"] + association["gene_a_only"]
    negative_denominator = association["gene_b_only"] + association["neither_detected"]
    myog = pd.DataFrame(
        {
            "gene": ["jam3b", "jam2a"],
            "myog_percent_when_gene_positive": (
                association["both_detected"] / positive_denominator * 100
            ).to_numpy(float),
            "myog_percent_when_gene_negative": (
                association["gene_b_only"] / negative_denominator * 100
            ).to_numpy(float),
            "fisher_odds_ratio": association["fisher_odds_ratio"].to_numpy(float),
            "fisher_two_sided_p": association["fisher_two_sided_p"].to_numpy(float),
        }
    )

    counts = spatial_null_iterations["orientation_compatible_pair_count"].to_numpy(
        float
    )
    summary_row = spatial_null_summary.iloc[0]
    observed = int(summary_row["observed_jam2a_jam3b_orientation_compatible_pairs"])
    exceedances = int(np.count_nonzero(counts >= observed))
    spatial = pd.DataFrame(
        [
            {
                "n_permutations": int(counts.size),
                "observed_neighbor_pairs": observed,
                "null_mean": float(np.mean(counts)),
                "null_q025": float(np.quantile(counts, 0.025)),
                "null_q975": float(np.quantile(counts, 0.975)),
                "observed_over_null_mean": float(observed / np.mean(counts)),
                "n_null_at_least_observed": exceedances,
                "plus_one_upper_tail_p": float((exceedances + 1) / (counts.size + 1)),
            }
        ]
    )
    return ZebrafishAttentionPanels(
        external_agreement=external,
        jam_quartiles=quartiles,
        somite_ranks=somite,
        myog_detection=myog,
        spatial_null=spatial,
    )


def load_zebrafish_attention_results(
    results_dir: str | Path | None = None,
) -> ZebrafishAttentionResults:
    """Load the ten processed files and calculate the figure tables."""

    source_dir = resolve_results_dir(results_dir, slug="zebrafish_attention")
    paths = require_files(source_dir, _FILES)
    manifest = read_json(paths["manifest.json"])
    _validate_manifest(manifest, paths["manifest.json"])

    pair = _validate_pair_concordance(
        pd.read_csv(
            paths["directed_pair_concordance.csv"], float_precision="round_trip"
        ),
        paths["directed_pair_concordance.csv"],
    )
    compatibility = _validate_compatibility(
        pd.read_csv(
            paths["jam_compatibility_percentile_summary.csv"],
            float_precision="round_trip",
        ),
        paths["jam_compatibility_percentile_summary.csv"],
    )
    quartiles = _validate_quartiles(
        pd.read_csv(
            paths["jam_quartile_compatibility.csv"], float_precision="round_trip"
        ),
        paths["jam_quartile_compatibility.csv"],
    )
    type_pairs = _validate_type_pairs(
        pd.read_csv(
            paths["type_pair_raw_attention_ranks.csv"], float_precision="round_trip"
        ),
        paths["type_pair_raw_attention_ranks.csv"],
    )
    null_summary, null_iterations = _validate_spatial_null(
        pd.read_csv(
            paths["somite_18hpf_spatial_null_summary.csv"], float_precision="round_trip"
        ),
        pd.read_csv(
            paths["somite_18hpf_spatial_null_iterations.csv.gz"],
            float_precision="round_trip",
        ),
        paths["somite_18hpf_spatial_null_summary.csv"],
        paths["somite_18hpf_spatial_null_iterations.csv.gz"],
    )
    association = _validate_association(
        pd.read_csv(paths["myog_association.csv"], float_precision="round_trip"),
        paths["myog_association.csv"],
    )
    detection = _validate_detection(
        pd.read_csv(
            paths["expression_detection_by_stage_type.csv"],
            float_precision="round_trip",
        ),
        paths["expression_detection_by_stage_type.csv"],
    )
    cells = _validate_cells(
        pd.read_csv(
            paths["somite_18hpf_spatial_cells.csv.gz"], float_precision="round_trip"
        ),
        paths["somite_18hpf_spatial_cells.csv.gz"],
    )
    display_edges = _validate_display_edges(
        pd.read_csv(
            paths["trained_jam_display_edges.csv"], float_precision="round_trip"
        ),
        paths["trained_jam_display_edges.csv"],
    )
    panels = calculate_zebrafish_attention_panels(
        pair,
        quartiles,
        type_pairs,
        null_summary,
        null_iterations,
        association,
    )
    return ZebrafishAttentionResults(
        source_dir=source_dir,
        manifest=manifest,
        directed_pair_concordance=pair,
        compatibility_summary=compatibility,
        quartile_compatibility=quartiles,
        type_pair_ranks=type_pairs,
        spatial_null_summary=null_summary,
        spatial_null_iterations=null_iterations,
        myog_association=association,
        expression_detection=detection,
        spatial_cells=cells,
        display_edges=display_edges,
        panels=panels,
    )


def zebrafish_attention_statistics(
    results: ZebrafishAttentionResults,
) -> dict[str, object]:
    """Return numerical values used by the figure."""

    external = results.panels.external_agreement.set_index("external_method")
    spatial = results.panels.spatial_null.iloc[0]
    edge_totals = results.compatibility_summary.groupby("condition")[
        "n_directed_edges"
    ].sum()
    return {
        "conditions": list(CONDITION_ORDER),
        "directed_cell_type_pairs": int(external["n_pairs"].iloc[0]),
        "somite_cells": int(results.spatial_cells["is_somite"].sum()),
        "directed_edges_per_condition": int(edge_totals.iloc[0]),
        "display_edges": int(len(results.display_edges)),
        "spatial_null_permutations": int(spatial["n_permutations"]),
        "observed_neighbor_pairs": int(spatial["observed_neighbor_pairs"]),
        "spatial_null_mean": float(spatial["null_mean"]),
        "spatial_fold": float(spatial["observed_over_null_mean"]),
        "spatial_plus_one_p": float(spatial["plus_one_upper_tail_p"]),
        "commot_adjusted_spearman": float(
            external.loc["COMMOT", "adjusted_spearman_rho"]
        ),
        "cellagentchat_adjusted_spearman": float(
            external.loc["CellAgentChat", "adjusted_spearman_rho"]
        ),
    }


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(
        path,
        index=False,
        float_format="%.17g",
        na_rep="",
        lineterminator="\n",
    )


def write_zebrafish_attention_tables(
    results: ZebrafishAttentionResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the tables recalculated for the figure panels."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "external_agreement": results.panels.external_agreement,
        "jam_quartiles": results.panels.jam_quartiles,
        "somite_ranks": results.panels.somite_ranks,
        "myog_detection": results.panels.myog_detection,
        "spatial_null": results.panels.spatial_null,
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output / f"zebrafish_attention_{name}.csv"
        _write_csv(table, path)
        paths[name] = path
    return paths


def plot_zebrafish_attention(
    results: ZebrafishAttentionResults,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the zebrafish attention figure as PDF and PNG."""

    from ._zebrafish_attention_plot import plot_zebrafish_attention as _plot

    return _plot(results, output_dir)
