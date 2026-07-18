#!/usr/bin/env python3
"""Create per-track benchmark tables and barplots without an overall score."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SPACES = ("joint", "state", "spatial")
METRIC_LABELS = {
    "sliced_w2": "Sliced Wasserstein-2",
    "exact_w1": "Exact OT Wasserstein-1",
    "exact_w2": "Exact OT Wasserstein-2",
}
TRACK_LABELS = {
    "loto": "Transductive leave-one-timepoint-out",
    "full_data": "Full-data reconstruction (in-sample)",
}


class SummaryError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_registry(path: Path | None, observed_methods: list[str]) -> list[dict[str, Any]]:
    if path is None:
        return [
            {
                "method": method,
                "display_name": method,
                "aliases": [],
                "spaces": list(SPACES),
                "status": "evaluated",
            }
            for method in observed_methods
        ]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read method registry {path}: {exc}") from exc
    records = payload.get("methods") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise SummaryError("method registry must contain a non-empty methods list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            raise SummaryError("each method registry entry must be an object")
        method = str(raw.get("method", "")).strip()
        if not method or method in seen:
            raise SummaryError(f"invalid or duplicate registry method {method!r}")
        seen.add(method)
        spaces = [str(value) for value in raw.get("spaces", [])]
        if any(space not in SPACES for space in spaces):
            raise SummaryError(f"{method}: invalid spaces {spaces}")
        result.append(
            {
                **raw,
                "method": method,
                "display_name": str(raw.get("display_name", method)),
                "aliases": [str(value) for value in raw.get("aliases", [])],
                "spaces": spaces,
                "status": str(raw.get("status", "evaluated")),
            }
        )
    return result


def _canonicalize_methods(
    metrics: pd.DataFrame, registry: list[dict[str, Any]]
) -> pd.DataFrame:
    alias_map: dict[str, str] = {}
    for record in registry:
        for alias in [record["method"], record["display_name"], *record["aliases"]]:
            key = str(alias).strip().casefold()
            if key in alias_map and alias_map[key] != record["method"]:
                raise SummaryError(f"method alias {alias!r} is ambiguous")
            alias_map[key] = record["method"]
    unknown = sorted(
        {
            str(value)
            for value in metrics["method"].unique()
            if str(value).strip().casefold() not in alias_map
        }
    )
    if unknown:
        raise SummaryError(f"metrics contain methods absent from registry: {unknown}")
    result = metrics.copy()
    result["method"] = [alias_map[str(value).strip().casefold()] for value in result["method"]]
    return result


def _canonical_method_names(
    values: list[Any], registry: list[dict[str, Any]]
) -> set[str]:
    alias_map: dict[str, str] = {}
    for record in registry:
        for alias in [record["method"], record["display_name"], *record["aliases"]]:
            alias_map[str(alias).strip().casefold()] = record["method"]
    missing = sorted(
        str(value)
        for value in values
        if str(value).strip().casefold() not in alias_map
    )
    if missing:
        raise SummaryError(f"evaluation manifest contains unknown methods: {missing}")
    return {alias_map[str(value).strip().casefold()] for value in values}


def _verify_evaluation_manifest(
    path: Path,
    metrics_path: Path,
    metrics: pd.DataFrame,
    registry: list[dict[str, Any]],
    track: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read evaluation manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise SummaryError("evaluation manifest is absent, invalid, or incomplete")
    if str(payload.get("track")) != track:
        raise SummaryError("evaluation manifest track differs from metrics CSV")
    declared_sha = str(payload.get("metrics_long_csv_sha256", "")).lower()
    if declared_sha != sha256_file(metrics_path):
        raise SummaryError("metrics CSV SHA-256 differs from evaluation manifest")
    expected_targets = {int(value) for value in payload.get("targets", [])}
    observed_targets = {int(value) for value in metrics["target"].unique()}
    if not expected_targets or observed_targets != expected_targets:
        raise SummaryError(
            f"metrics targets {sorted(observed_targets)} differ from evaluation manifest "
            f"{sorted(expected_targets)}"
        )
    expected_methods = _canonical_method_names(payload.get("methods", []), registry)
    observed_methods = {str(value) for value in metrics["method"].unique()}
    if not expected_methods or observed_methods != expected_methods:
        raise SummaryError(
            f"metrics methods {sorted(observed_methods)} differ from evaluation manifest "
            f"{sorted(expected_methods)}"
        )
    registry_primary = {
        record["method"] for record in registry if record.get("status") == "evaluated"
    }
    if expected_methods != registry_primary:
        raise SummaryError(
            f"evaluation methods {sorted(expected_methods)} do not equal the primary registry "
            f"{sorted(registry_primary)}"
        )
    observed_pairs = {
        (str(row.method), int(row.target))
        for row in metrics[["method", "target"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    expected_pairs = {
        (method, target)
        for method in expected_methods
        for target in expected_targets
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs.difference(observed_pairs))
        raise SummaryError(f"metrics lack complete method-by-target grid: {missing}")
    return payload


def _validate_metrics(metrics: pd.DataFrame) -> str:
    required = {
        "track",
        "target",
        "method",
        "space",
        "projection_repeat",
        "sliced_w2",
        "exact_w1",
        "exact_w2",
        "tmv_available",
        "tmv",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise SummaryError(f"metrics CSV is missing columns: {missing}")
    tracks = {str(value) for value in metrics["track"].unique()}
    if len(tracks) != 1:
        raise SummaryError(f"one summary call must contain exactly one track, found {tracks}")
    track = next(iter(tracks))
    if track not in TRACK_LABELS:
        raise SummaryError(f"unknown track {track!r}")
    invalid_spaces = sorted(set(metrics["space"]).difference(SPACES))
    if invalid_spaces:
        raise SummaryError(f"invalid spaces: {invalid_spaces}")
    for metric in METRIC_LABELS:
        values = pd.to_numeric(metrics[metric], errors="coerce")
        if values.isna().any() or (values < 0).any():
            raise SummaryError(f"{metric} must contain finite nonnegative values")
    if not pd.api.types.is_bool_dtype(metrics["tmv_available"]):
        normalized = metrics["tmv_available"].astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise SummaryError("tmv_available must contain only true/false values")
        metrics["tmv_available"] = normalized.eq("true")
    return track


def _target_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "track",
        "target",
        "source_time",
        "method",
        "space",
        "output_scope",
        "native_vs_adapter",
    ]
    available = [column for column in grouping if column in metrics]
    result = (
        metrics.groupby(available, dropna=False, sort=False)
        .agg(
            sliced_w2=("sliced_w2", "mean"),
            sliced_w2_projection_sd=("sliced_w2", "std"),
            exact_w1=("exact_w1", "first"),
            exact_w2=("exact_w2", "first"),
            tmv_available=("tmv_available", "first"),
            tmv=("tmv", "first"),
            tmv_absolute=("tmv_absolute", "first"),
            predicted_mass=("predicted_mass", "first"),
            observed_mass_relative=("observed_mass_relative", "first"),
            n_projection_repeats=("projection_repeat", "nunique"),
            n_predicted=("n_predicted", "first"),
            n_observed=("n_observed", "first"),
        )
        .reset_index()
    )
    result["sliced_w2_projection_sd"] = result["sliced_w2_projection_sd"].fillna(0.0)
    return result


def _method_summary(
    target: pd.DataFrame, registry: list[dict[str, Any]], track: str
) -> pd.DataFrame:
    observed = (
        target.groupby(["method", "space"], sort=False)
        .agg(
            n_targets=("target", "nunique"),
            sliced_w2_mean=("sliced_w2", "mean"),
            sliced_w2_target_sd=("sliced_w2", "std"),
            exact_w1_mean=("exact_w1", "mean"),
            exact_w1_target_sd=("exact_w1", "std"),
            exact_w2_mean=("exact_w2", "mean"),
            exact_w2_target_sd=("exact_w2", "std"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    observed_index = {
        (str(row.method), str(row.space)): row._asdict()
        for row in observed.itertuples(index=False)
    }
    for order, record in enumerate(registry):
        for space in SPACES:
            key = (record["method"], space)
            applicable = space in record["spaces"] and record["status"] == "evaluated"
            if key in observed_index:
                if not applicable:
                    raise SummaryError(
                        f"{record['method']} emitted {space}, but registry marks it ineligible"
                    )
                values = observed_index[key]
                status = "evaluated"
            else:
                values = {
                    "n_targets": 0,
                    "sliced_w2_mean": np.nan,
                    "sliced_w2_target_sd": np.nan,
                    "exact_w1_mean": np.nan,
                    "exact_w1_target_sd": np.nan,
                    "exact_w2_mean": np.nan,
                    "exact_w2_target_sd": np.nan,
                }
                status = "missing" if applicable else record["status"]
                if status == "evaluated":
                    status = "N/A"
            rows.append(
                {
                    "track": track,
                    "method": record["method"],
                    "display_name": record["display_name"],
                    "method_order": order,
                    "space": space,
                    "status": status,
                    "scope": record.get("scope"),
                    **{
                        key_name: values[key_name]
                        for key_name in (
                            "n_targets",
                            "sliced_w2_mean",
                            "sliced_w2_target_sd",
                            "exact_w1_mean",
                            "exact_w1_target_sd",
                            "exact_w2_mean",
                            "exact_w2_target_sd",
                        )
                    },
                }
            )
    result = pd.DataFrame.from_records(rows)
    result["rank_sliced_w2_within_space"] = np.nan
    eligible = result["status"].eq("evaluated") & result["sliced_w2_mean"].notna()
    result.loc[eligible, "rank_sliced_w2_within_space"] = (
        result.loc[eligible]
        .groupby("space")["sliced_w2_mean"]
        .rank(method="min", ascending=True)
    )
    return result.sort_values(["space", "method_order"], kind="stable").reset_index(drop=True)


def _colour_map(registry: list[dict[str, Any]]) -> dict[str, str]:
    palette = list(plt.get_cmap("tab20").colors)
    return {
        record["method"]: record.get("color") or matplotlib.colors.to_hex(palette[index % len(palette)])
        for index, record in enumerate(registry)
    }


def _barplot_metric(
    target: pd.DataFrame,
    method_summary: pd.DataFrame,
    registry: list[dict[str, Any]],
    *,
    metric: str,
    output: Path,
    track: str,
) -> None:
    colors = _colour_map(registry)
    methods = [record["method"] for record in registry]
    display = {record["method"]: record["display_name"] for record in registry}
    figure, axes = plt.subplots(1, 3, figsize=(max(14.0, len(methods) * 1.25), 5.4), squeeze=False)
    rng = np.random.default_rng(20260718)
    for axis, space in zip(axes[0], SPACES):
        table = method_summary[
            (method_summary["space"] == space) & method_summary["status"].eq("evaluated")
        ]
        mean_column = f"{metric}_mean" if metric != "sliced_w2" else "sliced_w2_mean"
        sd_column = f"{metric}_target_sd" if metric != "sliced_w2" else "sliced_w2_target_sd"
        values = table.set_index("method")[mean_column]
        errors = table.set_index("method")[sd_column].fillna(0.0)
        active = [method for method in methods if method in values.index]
        x = np.arange(len(active), dtype=float)
        axis.bar(
            x,
            [values[method] for method in active],
            yerr=[errors[method] for method in active],
            color=[colors[method] for method in active],
            edgecolor="black",
            linewidth=0.55,
            capsize=3,
            alpha=0.88,
        )
        stage_values = target[target["space"] == space]
        for index, method in enumerate(active):
            points = stage_values.loc[stage_values["method"] == method, metric].to_numpy(float)
            if points.size:
                jitter = rng.uniform(-0.12, 0.12, size=points.size)
                axis.scatter(
                    np.full(points.size, index) + jitter,
                    points,
                    s=22,
                    color="white",
                    edgecolor="black",
                    linewidth=0.65,
                    zorder=4,
                )
        axis.set_title(space.capitalize())
        axis.set_xticks(x, [display[method] for method in active], rotation=55, ha="right")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"{TRACK_LABELS[track]} — {METRIC_LABELS[metric]} (lower is better)")
    figure.text(
        0.5,
        0.005,
        "Bars: mean across target stages; error bars: target-stage SD; dots: individual targets. "
        "Projection repeats are numerical integration checks, not training-seed confidence intervals.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _applicability_plot(
    method_summary: pd.DataFrame,
    registry: list[dict[str, Any]],
    output: Path,
    track: str,
) -> None:
    methods = [record["method"] for record in registry]
    display = {record["method"]: record["display_name"] for record in registry}
    status = method_summary.set_index(["method", "space"])["status"]
    matrix = np.zeros((len(methods), len(SPACES)), dtype=float)
    labels = np.empty(matrix.shape, dtype=object)
    for i, method in enumerate(methods):
        for j, space in enumerate(SPACES):
            value = str(status.loc[(method, space)])
            if value == "evaluated":
                matrix[i, j], labels[i, j] = 2, "✓"
            elif value == "missing":
                matrix[i, j], labels[i, j] = 1, "missing"
            elif value == "sensitivity_only":
                matrix[i, j], labels[i, j] = 0, "sens."
            else:
                matrix[i, j], labels[i, j] = 0, "N/A"
    cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#f6bd60", "#5abf90"])
    figure, axis = plt.subplots(figsize=(6.8, max(4.0, 0.43 * len(methods))))
    axis.imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, labels[i, j], ha="center", va="center", fontsize=9)
    axis.set_xticks(range(len(SPACES)), [space.capitalize() for space in SPACES])
    axis.set_yticks(range(len(methods)), [display[method] for method in methods])
    axis.set_title(f"{TRACK_LABELS[track]} — primary-space applicability")
    axis.tick_params(length=0)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _tmv_plot(target: pd.DataFrame, registry: list[dict[str, Any]], output: Path, track: str) -> bool:
    values = target[target["tmv_available"].astype(bool)].drop_duplicates(
        ["method", "target"]
    )
    if values.empty:
        return False
    colors = _colour_map(registry)
    display = {record["method"]: record["display_name"] for record in registry}
    methods = [record["method"] for record in registry if record["method"] in set(values["method"])]
    summary = values.groupby("method")["tmv"].agg(["mean", "std"])
    x = np.arange(len(methods))
    figure, axis = plt.subplots(figsize=(max(5.5, len(methods) * 1.4), 4.8))
    axis.bar(
        x,
        [summary.loc[method, "mean"] for method in methods],
        yerr=[float(summary.loc[method, "std"]) if np.isfinite(summary.loc[method, "std"]) else 0 for method in methods],
        color=[colors[method] for method in methods],
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
    )
    rng = np.random.default_rng(20260718)
    for i, method in enumerate(methods):
        points = values.loc[values["method"] == method, "tmv"].to_numpy(float)
        axis.scatter(
            np.full(points.size, i) + rng.uniform(-0.1, 0.1, points.size),
            points,
            s=24,
            color="white",
            edgecolor="black",
            linewidth=0.65,
            zorder=4,
        )
    axis.set_xticks(x, [display[method] for method in methods], rotation=35, ha="right")
    axis.set_ylabel("Relative total-mass-variation error")
    axis.set_title(f"{TRACK_LABELS[track]} — native mass/growth methods only")
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return True


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    metrics_path = args.metrics_long.expanduser().resolve()
    metrics = pd.read_csv(metrics_path)
    track = _validate_metrics(metrics)
    registry_path = None if args.method_registry is None else args.method_registry.expanduser().resolve()
    registry = _load_registry(registry_path, sorted(str(value) for value in metrics["method"].unique()))
    metrics = _canonicalize_methods(metrics, registry)
    evaluation_path = args.evaluation_manifest.expanduser().resolve()
    evaluation = _verify_evaluation_manifest(
        evaluation_path, metrics_path, metrics, registry, track
    )
    target = _target_summary(metrics)
    method = _method_summary(target, registry, track)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_path = output / f"{track}_target_summary.csv"
    method_path = output / f"{track}_method_summary.csv"
    _atomic_csv(target_path, target)
    _atomic_csv(method_path, method)

    plot_paths: list[Path] = []
    for metric_name in METRIC_LABELS:
        plot = output / f"{track}_{metric_name}_barplot.png"
        _barplot_metric(target, method, registry, metric=metric_name, output=plot, track=track)
        plot_paths.extend([plot, plot.with_suffix(".pdf")])
    applicability = output / f"{track}_applicability_matrix.png"
    _applicability_plot(method, registry, applicability, track)
    plot_paths.extend([applicability, applicability.with_suffix(".pdf")])
    tmv_plot = output / f"{track}_tmv_native_mass.png"
    if _tmv_plot(target, registry, tmv_plot, track):
        plot_paths.extend([tmv_plot, tmv_plot.with_suffix(".pdf")])

    manifest = {
        "schema_version": "1.0.0",
        "status": "complete",
        "track": track,
        "track_label": TRACK_LABELS[track],
        "rank_policy": "within each feature space only; no cross-space or cross-track score",
        "full_data_warning": (
            "All stages participated in fitting; these are in-sample reconstruction results, not forecasting."
            if track == "full_data"
            else None
        ),
        "projection_repeat_warning": (
            "Projection repeats quantify sliced-Wasserstein numerical variability only; "
            "they are not independent training seeds or biological confidence intervals."
        ),
        "metrics_long": str(metrics_path),
        "metrics_long_sha256": sha256_file(metrics_path),
        "evaluation_manifest": str(evaluation_path),
        "evaluation_manifest_sha256": sha256_file(evaluation_path),
        "evaluation_methods": evaluation["methods"],
        "evaluation_targets": evaluation["targets"],
        "method_registry": None if registry_path is None else str(registry_path),
        "method_registry_sha256": None if registry_path is None else sha256_file(registry_path),
        "target_summary": str(target_path),
        "target_summary_sha256": sha256_file(target_path),
        "method_summary": str(method_path),
        "method_summary_sha256": sha256_file(method_path),
        "plots": [
            {"path": str(path), "sha256": sha256_file(path)} for path in plot_paths
        ],
    }
    _atomic_json(output / f"{track}_summary_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-long", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--method-registry", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = summarize(args)
    except (SummaryError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
