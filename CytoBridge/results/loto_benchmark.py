"""Processed results for the five-dataset LOTO benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._io import read_json, read_manifest, require_files, resolve_results_dir


DATASET_ORDER = (
    "zebrafish",
    "mosta",
    "arista",
    "admouse",
    "chicken_heart",
)
DATASET_LABELS = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "admouse": "AD mouse",
    "chicken_heart": "Chicken\nheart",
}
REFERENCE_METHOD = "CytoBridge-0.015"
METHOD_ORDER = (
    REFERENCE_METHOD,
    "stvcr",
    "moscot",
    "paste",
    "spateo",
    "random_independent_pairs",
    "mioflow",
    "stories",
    "wot",
)
COMPARISON_METHOD_ORDER = METHOD_ORDER[1:]

_TARGET_COLUMNS = (
    "dataset",
    "target",
    "method",
    "display_name",
    "space",
    "sliced_w2",
    "projection_sd",
    "n_projection_repeats",
)
_SUPPORT_COLUMNS = (
    "dataset",
    "target",
    "method",
    "display_name",
    "initial_source_roster_n",
    "native_output_n",
    "output_support_differs_from_initial",
    "output_support_policy",
    "sliced_w2_support",
    "sliced_w2_predicted_weights",
    "target_size_resampling",
    "native_vs_adapter",
    "output_scope",
)


@dataclass(frozen=True)
class LotoBenchmarkData:
    """Compact LOTO inputs and the tables calculated for the figure."""

    source_dir: Path
    manifest: dict[str, Any]
    protocol: dict[str, Any]
    target_means: pd.DataFrame
    native_support: pd.DataFrame
    paired_ratios: pd.DataFrame
    dataset_summary: pd.DataFrame


def _read_boolean(series: pd.Series, *, name: str, source: Path) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    values = normalized.map({"true": True, "false": False})
    if values.isna().any():
        invalid = sorted(normalized.loc[values.isna()].unique())[:5]
        raise ValueError(f"{source} has invalid {name} values: {invalid}")
    return values.astype(bool)


def _require_columns(
    table: pd.DataFrame,
    required: tuple[str, ...],
    *,
    source: Path,
) -> None:
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def _validate_protocol(protocol: dict[str, Any], source: Path) -> None:
    if protocol.get("analysis") != "loto_benchmark":
        raise ValueError(f"{source} is not a LOTO benchmark protocol")
    if tuple(protocol.get("datasets", ())) != DATASET_ORDER:
        raise ValueError(f"{source} has an unexpected dataset order")
    if tuple(protocol.get("method_order", ())) != METHOD_ORDER:
        raise ValueError(f"{source} has an unexpected method order")
    if tuple(protocol.get("comparison_method_order", ())) != COMPARISON_METHOD_ORDER:
        raise ValueError(f"{source} has an unexpected comparison-method order")
    if protocol.get("reference_method") != REFERENCE_METHOD:
        raise ValueError(f"{source} has an unexpected reference method")

    projection = protocol.get("projection", {})
    expected_projection = {
        "repeats_per_dataset_target_space": 5,
        "directions_per_repeat": 1024,
        "sharing_scope": "dataset-target-space-repeat",
        "applicable_methods_share_directions_within_repeat": True,
        "shared_across_datasets": False,
        "method_specific_directions": False,
    }
    if projection != expected_projection:
        raise ValueError(f"{source} has an unexpected projection protocol")

    support = protocol.get("support", {})
    if support.get("initial_source_roster_n") != 5000:
        raise ValueError(f"{source} must use the 5,000-point initial roster")
    if support.get("sliced_w2_support") != "all_native_predicted_points":
        raise ValueError(f"{source} has an unexpected Sliced-W2 support policy")
    if support.get("predicted_weights") != "normalized_before_metric":
        raise ValueError(f"{source} has an unexpected predicted-weight policy")
    if support.get("target_size_resampling") is not False:
        raise ValueError(f"{source} must disable target-size resampling")


def _validate_target_means(
    table: pd.DataFrame,
    *,
    protocol: dict[str, Any],
    source: Path,
) -> pd.DataFrame:
    _require_columns(table, _TARGET_COLUMNS, source=source)
    result = table.loc[:, _TARGET_COLUMNS].copy()
    result["target"] = pd.to_numeric(result["target"], errors="coerce")
    for column in ("sliced_w2", "projection_sd", "n_projection_repeats"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    numeric = result[
        ["target", "sliced_w2", "projection_sd", "n_projection_repeats"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{source} contains non-finite target-level values")
    if not np.equal(result["target"], np.floor(result["target"])).all():
        raise ValueError(f"{source} contains non-integer target indices")
    result["target"] = result["target"].astype(int)
    if not result["sliced_w2"].gt(0).all():
        raise ValueError(f"{source} contains non-positive Sliced-W2 values")
    if not result["projection_sd"].ge(0).all():
        raise ValueError(f"{source} contains negative projection standard deviations")
    if not result["n_projection_repeats"].eq(5).all():
        raise ValueError(f"{source} must contain five projection repeats per cell")
    result["n_projection_repeats"] = result["n_projection_repeats"].astype(int)

    key = ["dataset", "target", "method", "space"]
    if result.duplicated(key).any():
        raise ValueError(
            f"{source} contains duplicate dataset-target-method-space cells"
        )
    datasets = set(result["dataset"])
    unknown_datasets = sorted(datasets.difference(DATASET_ORDER))
    if unknown_datasets:
        raise ValueError(f"{source} contains unknown datasets: {unknown_datasets}")
    methods = set(result["method"])
    unknown_methods = sorted(methods.difference(METHOD_ORDER))
    if unknown_methods:
        raise ValueError(f"{source} contains unknown methods: {unknown_methods}")

    display_names = protocol["display_names"]
    expected_display = result["method"].map(display_names)
    if expected_display.isna().any() or not result["display_name"].equals(
        expected_display
    ):
        raise ValueError(f"{source} has inconsistent method display names")

    expected = {
        (dataset, int(target), method, space)
        for dataset in DATASET_ORDER
        for target in protocol["dataset_targets"][dataset]
        for method in METHOD_ORDER
        for space in protocol["spaces_by_method"][method]
    }
    actual = set(result[key].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(
            f"{source} does not match the dataset, target, method, and space contract"
        )
    return result.reset_index(drop=True)


def _validate_native_support(
    table: pd.DataFrame,
    *,
    protocol: dict[str, Any],
    source: Path,
) -> pd.DataFrame:
    _require_columns(table, _SUPPORT_COLUMNS, source=source)
    result = table.loc[:, _SUPPORT_COLUMNS].copy()
    for column in ("target", "initial_source_roster_n", "native_output_n"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    numeric = result[["target", "initial_source_roster_n", "native_output_n"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{source} contains invalid support counts")
    for column in ("target", "initial_source_roster_n", "native_output_n"):
        result[column] = result[column].astype(int)
    if not result["native_output_n"].gt(0).all():
        raise ValueError(f"{source} contains non-positive native output support")
    for column in (
        "output_support_differs_from_initial",
        "target_size_resampling",
    ):
        result[column] = _read_boolean(result[column], name=column, source=source)

    key = ["dataset", "target", "method"]
    if result.duplicated(key).any():
        raise ValueError(f"{source} contains duplicate dataset-target-method rows")
    expected = {
        (dataset, int(target), method)
        for dataset in DATASET_ORDER
        for target in protocol["dataset_targets"][dataset]
        for method in METHOD_ORDER
    }
    actual = set(result[key].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"{source} does not match the native-support row contract")

    display_names = protocol["display_names"]
    if not result["display_name"].equals(result["method"].map(display_names)):
        raise ValueError(f"{source} has inconsistent method display names")
    if not result["initial_source_roster_n"].eq(5000).all():
        raise ValueError(f"{source} must start every method from 5,000 points")
    differs = result["native_output_n"].ne(result["initial_source_roster_n"])
    if not result["output_support_differs_from_initial"].equals(differs):
        raise ValueError(f"{source} has inconsistent output-support flags")

    non_stvcr = result.loc[~result["method"].eq("stvcr")]
    if not non_stvcr["native_output_n"].eq(5000).all():
        raise ValueError(f"{source} changes fixed output support outside stVCR")
    if (
        not non_stvcr["output_support_policy"]
        .eq("fixed_initial_support_preserved")
        .all()
    ):
        raise ValueError(f"{source} has an unexpected fixed-support policy")
    stvcr = result.loc[result["method"].eq("stvcr")]
    if (
        not stvcr["output_support_policy"]
        .eq("growth_enabled_native_support_retained")
        .all()
    ):
        raise ValueError(f"{source} has an unexpected stVCR support policy")

    if not result["sliced_w2_support"].eq("all_native_predicted_points").all():
        raise ValueError(f"{source} does not use all method-native predicted points")
    if not result["sliced_w2_predicted_weights"].eq("normalized_before_metric").all():
        raise ValueError(f"{source} does not normalize predicted weights")
    if result["target_size_resampling"].any():
        raise ValueError(f"{source} enables target-size resampling")

    expected_output = protocol["output_contract_by_method"]
    for method, (native_vs_adapter, output_scope) in expected_output.items():
        subset = result.loc[result["method"].eq(method)]
        if (
            not subset["native_vs_adapter"].eq(native_vs_adapter).all()
            or not subset["output_scope"].eq(output_scope).all()
        ):
            raise ValueError(f"{source} has an unexpected output contract for {method}")
    return result.reset_index(drop=True)


def compute_paired_loto_ratios(
    target_means: pd.DataFrame,
    reference_method: str = REFERENCE_METHOD,
) -> pd.DataFrame:
    """Calculate method/reference ratios in matched target-space cells."""

    required = {
        "dataset",
        "target",
        "method",
        "display_name",
        "space",
        "sliced_w2",
    }
    missing = sorted(required.difference(target_means.columns))
    if missing:
        raise ValueError(f"target_means is missing columns: {missing}")
    if target_means.duplicated(["dataset", "target", "method", "space"]).any():
        raise ValueError("target_means contains duplicate matched cells")

    reference = target_means.loc[
        target_means["method"].eq(reference_method),
        ["dataset", "target", "space", "sliced_w2"],
    ].rename(columns={"sliced_w2": "cytobridge_sliced_w2"})
    if reference.empty:
        raise ValueError(
            f"target_means does not contain reference method {reference_method!r}"
        )

    present = set(target_means["method"])
    comparison_order = [
        method for method in COMPARISON_METHOD_ORDER if method in present
    ]
    unknown = sorted(present.difference({reference_method, *COMPARISON_METHOD_ORDER}))
    if unknown:
        raise ValueError(f"target_means contains unknown comparison methods: {unknown}")
    if not comparison_order:
        raise ValueError("target_means does not contain a comparison method")

    parts: list[pd.DataFrame] = []
    for method in comparison_order:
        comparison_rows = target_means.loc[target_means["method"].eq(method)]
        display_names = comparison_rows["display_name"].drop_duplicates()
        if len(display_names) != 1:
            raise ValueError(
                f"target_means has inconsistent display names for {method}"
            )
        comparison = comparison_rows.loc[
            :, ["dataset", "target", "space", "sliced_w2"]
        ].rename(columns={"sliced_w2": "method_sliced_w2"})
        paired = reference.merge(
            comparison,
            on=["dataset", "target", "space"],
            how="inner",
            validate="one_to_one",
        )
        if paired.empty:
            raise ValueError(f"target_means has no matched cells for {method}")
        paired.insert(0, "method", method)
        paired.insert(1, "display_name", str(display_names.iloc[0]))
        paired["method_to_cytobridge_ratio"] = (
            paired["method_sliced_w2"] / paired["cytobridge_sliced_w2"]
        )
        parts.append(paired)
    return pd.concat(parts, ignore_index=True)


def summarize_loto_ratios(paired_ratios: pd.DataFrame) -> pd.DataFrame:
    """Average matched cell ratios within each dataset and method."""

    required = {
        "method",
        "display_name",
        "dataset",
        "method_to_cytobridge_ratio",
    }
    missing = sorted(required.difference(paired_ratios.columns))
    if missing:
        raise ValueError(f"paired_ratios is missing columns: {missing}")
    ratios = pd.to_numeric(paired_ratios["method_to_cytobridge_ratio"], errors="coerce")
    if not np.isfinite(ratios).all() or not ratios.gt(0).all():
        raise ValueError("paired_ratios contains invalid relative Sliced-W2 values")
    return (
        paired_ratios.groupby(["method", "display_name", "dataset"], as_index=False)
        .agg(
            relative_sliced_w2=("method_to_cytobridge_ratio", "mean"),
            median_relative_sliced_w2=("method_to_cytobridge_ratio", "median"),
            n_paired_comparisons=("method_to_cytobridge_ratio", "size"),
        )
        .reset_index(drop=True)
    )


def load_loto_benchmark(
    results_dir: str | Path | None = None,
) -> LotoBenchmarkData:
    """Load compact LOTO results and calculate the figure tables.

    Parameters
    ----------
    results_dir
        Directory containing the target means, native support table, and
        protocol JSON. Packaged data are used when this argument is omitted.
    """

    source_dir = resolve_results_dir(results_dir, slug="loto_benchmark")
    paths = require_files(
        source_dir,
        (
            "loto_target_stage_means.csv",
            "native_output_support.csv",
            "protocol.json",
        ),
    )
    protocol = read_json(paths["protocol.json"])
    _validate_protocol(protocol, paths["protocol.json"])
    target_means = _validate_target_means(
        pd.read_csv(
            paths["loto_target_stage_means.csv"],
            float_precision="round_trip",
        ),
        protocol=protocol,
        source=paths["loto_target_stage_means.csv"],
    )
    native_support = _validate_native_support(
        pd.read_csv(
            paths["native_output_support.csv"],
            float_precision="round_trip",
        ),
        protocol=protocol,
        source=paths["native_output_support.csv"],
    )
    paired_ratios = compute_paired_loto_ratios(
        target_means,
        reference_method=protocol["reference_method"],
    )
    dataset_summary = summarize_loto_ratios(paired_ratios)
    return LotoBenchmarkData(
        source_dir=source_dir,
        manifest=read_manifest(source_dir),
        protocol=protocol,
        target_means=target_means,
        native_support=native_support,
        paired_ratios=paired_ratios,
        dataset_summary=dataset_summary,
    )


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(
        path,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )


def write_loto_benchmark_tables(
    data: LotoBenchmarkData,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the matched ratios and dataset summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "paired_ratios": output / "paired_loto_ratios.csv",
        "dataset_summary": output / "loto_dataset_summary.csv",
    }
    _write_csv(data.paired_ratios, paths["paired_ratios"])
    _write_csv(data.dataset_summary, paths["dataset_summary"])
    return paths


def plot_loto_benchmark(
    data: LotoBenchmarkData,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render the five-dataset LOTO benchmark as PDF and PNG."""

    from ._loto_benchmark_plot import plot_loto_benchmark as _plot

    return _plot(data, output_dir)
