#!/usr/bin/env python3
"""Prepare result directories read by the S25, S39, and S40 figure commands.

The calculations that produce the source tables remain in their original
workflows.  This script only selects, checks, and copies those completed
tables into the directory layout expected by ``cytobridge figure``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from CytoBridge.results.interaction_evidence import load_lr_prior_stvcr_results
from CytoBridge.results.loto_benchmark import load_loto_benchmark
from CytoBridge.results.lr_complex_aggregation import (
    DATASET_ORDER as LR_COMPLEX_DATASETS,
    load_lr_complex_aggregation_results,
)


LOTO_DATASETS = ("zebrafish", "mosta", "arista", "admouse", "chicken_heart")
SPACE_ORDER = ("joint", "spatial", "state")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(path: Path, *, label: str) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return {
        "label": label,
        "path": str(resolved),
        "size": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _parse_assignments(
    values: Iterable[str], *, expected: tuple[str, ...], option: str
) -> dict[str, Path]:
    assignments: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"{option} expects DATASET=PATH, found {value!r}")
        if name not in expected:
            raise ValueError(
                f"{option} has unknown dataset {name!r}; expected {', '.join(expected)}"
            )
        if name in assignments:
            raise ValueError(f"{option} repeats dataset {name!r}")
        assignments[name] = Path(raw_path).expanduser().resolve()
    missing = [name for name in expected if name not in assignments]
    if missing:
        raise ValueError(f"{option} is missing: {', '.join(missing)}")
    return assignments


def _new_staging_dir(output_dir: Path) -> tuple[Path, Path]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new directory: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    return output, staging


def _finish(staging: Path, output: Path) -> Path:
    staging.rename(output)
    return output


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        float_format="%.17g",
        na_rep="",
        lineterminator="\n",
    )


def _write_manifest(
    directory: Path,
    *,
    analysis: str,
    paper_location: str,
    sources: list[dict[str, object]],
    descriptions: dict[str, str],
) -> Path:
    outputs: dict[str, dict[str, object]] = {}
    for relative_name, description in descriptions.items():
        path = directory / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"Collected output is missing: {path}")
        outputs[relative_name] = {
            "description": description,
            "size": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    manifest = {
        "schema_version": 1,
        "analysis": analysis,
        "paper_location": paper_location,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "select and validate completed result tables",
        "sources": sources,
        "files": outputs,
    }
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def collect_s25(dataset_results: dict[str, Path], output_dir: Path) -> Path:
    """Collect the four paired LR-complex score tables used by S25."""

    output, staging = _new_staging_dir(output_dir)
    sources: list[dict[str, object]] = []
    descriptions: dict[str, str] = {}
    try:
        for dataset in LR_COMPLEX_DATASETS:
            supplied = dataset_results[dataset]
            source = (
                supplied
                if supplied.is_file()
                else supplied / "comparison" / "paired_scores.csv"
            )
            record = _source_record(source, label=f"{dataset} paired scores")
            record["dataset"] = dataset
            sources.append(record)
            relative_name = f"{dataset}/paired_scores.csv"
            target = staging / relative_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            descriptions[
                relative_name
            ] = "paired minimum-subunit and geometric-mean LR scores"

        # The public reader checks columns, pair identities, score values, and
        # the required four-dataset roster.
        load_lr_complex_aggregation_results(staging)
        _write_manifest(
            staging,
            analysis="lr_complex_aggregation",
            paper_location="Supplementary Figure S25",
            sources=sources,
            descriptions=descriptions,
        )
        return _finish(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_protocol(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        protocol = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Protocol is not valid JSON: {source}") from error
    if not isinstance(protocol, dict):
        raise ValueError(f"Protocol must contain a JSON object: {source}")
    return protocol


def _method_aliases(protocol: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    display_names = protocol.get("display_names", {})
    for method in protocol.get("method_order", []):
        display = display_names.get(method)
        for value in (method, display):
            if value is not None:
                aliases[str(value).strip().casefold()] = str(method)
    return aliases


def _normalize_loto_summary(
    source: Path,
    *,
    dataset: str,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    table = pd.read_csv(source, float_precision="round_trip")
    required = {
        "track",
        "target",
        "method",
        "space",
        "status",
        "sliced_w2",
        "sliced_w2_projection_sd",
        "n_projection_repeats",
        "n_predicted",
        "native_vs_adapter",
        "output_scope",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    if "dataset" in table and not table["dataset"].astype(str).eq(dataset).all():
        raise ValueError(f"{source} contains rows from another dataset")
    if not table["track"].astype(str).eq("loto").all():
        raise ValueError(f"{source} must contain only LOTO rows")

    aliases = _method_aliases(protocol)
    result = table.copy()
    result["method"] = (
        result["method"].astype(str).str.strip().str.casefold().map(aliases)
    )
    result = result.loc[result["method"].notna()].copy()
    result = result.loc[result["status"].astype(str).eq("evaluated")].copy()
    if result.empty:
        raise ValueError(f"{source} has no evaluated methods used by S40")
    if "dataset" in result:
        result["dataset"] = dataset
    else:
        result.insert(0, "dataset", dataset)
    result["target"] = pd.to_numeric(result["target"], errors="coerce")
    for column in (
        "sliced_w2",
        "sliced_w2_projection_sd",
        "n_projection_repeats",
        "n_predicted",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    numeric = result[
        [
            "target",
            "sliced_w2",
            "sliced_w2_projection_sd",
            "n_projection_repeats",
            "n_predicted",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{source} contains missing or non-numeric figure values")
    for column in ("target", "n_projection_repeats", "n_predicted"):
        values = result[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{source} contains non-integer {column} values")
        result[column] = result[column].astype(int)
    return result


def _build_loto_tables(
    summaries: dict[str, Path], protocol: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [
        _normalize_loto_summary(summaries[dataset], dataset=dataset, protocol=protocol)
        for dataset in LOTO_DATASETS
    ]
    source = pd.concat(parts, ignore_index=True)
    display_names = protocol["display_names"]
    method_order = {
        method: index for index, method in enumerate(protocol["method_order"])
    }
    dataset_order = {name: index for index, name in enumerate(LOTO_DATASETS)}
    space_order = {name: index for index, name in enumerate(SPACE_ORDER)}

    target_means = pd.DataFrame(
        {
            "dataset": source["dataset"],
            "target": source["target"],
            "method": source["method"],
            "display_name": source["method"].map(display_names),
            "space": source["space"].astype(str),
            "sliced_w2": source["sliced_w2"],
            "projection_sd": source["sliced_w2_projection_sd"],
            "n_projection_repeats": source["n_projection_repeats"],
        }
    )
    target_means["__dataset"] = target_means["dataset"].map(dataset_order)
    target_means["__method"] = target_means["method"].map(method_order)
    target_means["__space"] = target_means["space"].map(space_order)
    target_means = (
        target_means.sort_values(
            ["__dataset", "target", "__method", "__space"], kind="mergesort"
        )
        .drop(columns=["__dataset", "__method", "__space"])
        .reset_index(drop=True)
    )

    metadata_columns = [
        "dataset",
        "target",
        "method",
        "n_predicted",
        "native_vs_adapter",
        "output_scope",
    ]
    metadata = source[metadata_columns].drop_duplicates()
    duplicated_metadata = metadata.duplicated(
        ["dataset", "target", "method"], keep=False
    )
    if duplicated_metadata.any():
        examples = (
            metadata.loc[duplicated_metadata, ["dataset", "target", "method"]]
            .drop_duplicates()
            .head()
            .to_dict(orient="records")
        )
        raise ValueError(
            "Native output metadata differs between spaces for the same result: "
            f"{examples}"
        )
    support = metadata.copy()
    initial_n = int(protocol["support"]["initial_source_roster_n"])
    support["display_name"] = support["method"].map(display_names)
    support["initial_source_roster_n"] = initial_n
    support["native_output_n"] = support.pop("n_predicted").astype(int)
    support["output_support_differs_from_initial"] = support["native_output_n"].ne(
        initial_n
    )
    support["output_support_policy"] = np.where(
        support["method"].eq("stvcr"),
        "growth_enabled_native_support_retained",
        "fixed_initial_support_preserved",
    )
    support["sliced_w2_support"] = protocol["support"]["sliced_w2_support"]
    support["sliced_w2_predicted_weights"] = protocol["support"]["predicted_weights"]
    support["target_size_resampling"] = bool(
        protocol["support"]["target_size_resampling"]
    )
    support["__dataset"] = support["dataset"].map(dataset_order)
    support["__method"] = support["method"].map(method_order)
    support = support.sort_values(
        ["__dataset", "target", "__method"], kind="mergesort"
    ).drop(columns=["__dataset", "__method"])
    support = support[
        [
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
        ]
    ].reset_index(drop=True)
    return target_means, support


def collect_s40(
    dataset_summaries: dict[str, Path], protocol_path: Path, output_dir: Path
) -> Path:
    """Collect five completed LOTO summaries into the S40 input directory."""

    protocol_path = protocol_path.expanduser().resolve()
    protocol = _read_protocol(protocol_path)
    output, staging = _new_staging_dir(output_dir)
    sources = [
        _source_record(protocol_path, label="S40 protocol"),
        *[
            {
                **_source_record(
                    dataset_summaries[dataset],
                    label=f"{dataset} LOTO target summary",
                ),
                "dataset": dataset,
            }
            for dataset in LOTO_DATASETS
        ],
    ]
    try:
        target_means, support = _build_loto_tables(dataset_summaries, protocol)
        _write_csv(target_means, staging / "loto_target_stage_means.csv")
        _write_csv(support, staging / "native_output_support.csv")
        shutil.copyfile(protocol_path, staging / "protocol.json")

        # This applies the same dataset, target, method, space, projection, and
        # native-support checks used by the public S40 plotting command.
        load_loto_benchmark(staging)
        _write_manifest(
            staging,
            analysis="loto_benchmark",
            paper_location="Supplementary Figure S40",
            sources=sources,
            descriptions={
                "loto_target_stage_means.csv": (
                    "target-level Sliced-W2 means from the five completed LOTO summaries"
                ),
                "native_output_support.csv": (
                    "method-native output counts and output type for every target"
                ),
                "protocol.json": "dataset, method, projection, and support settings",
            },
        )
        return _finish(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def collect_s40_tables(
    target_means_path: Path,
    native_support_path: Path,
    protocol_path: Path,
    output_dir: Path,
) -> Path:
    """Copy an already assembled, completed S40 table set after validation."""

    target_means_path = target_means_path.expanduser().resolve()
    native_support_path = native_support_path.expanduser().resolve()
    protocol_path = protocol_path.expanduser().resolve()
    sources = [
        _source_record(target_means_path, label="S40 target-stage means"),
        _source_record(native_support_path, label="S40 native output support"),
        _source_record(protocol_path, label="S40 protocol"),
    ]
    protocol = _read_protocol(protocol_path)
    output, staging = _new_staging_dir(output_dir)
    try:
        shutil.copyfile(target_means_path, staging / "loto_target_stage_means.csv")
        support = pd.read_csv(native_support_path, float_precision="round_trip")
        if "display_name" not in support.columns:
            if "method" not in support.columns:
                raise ValueError(f"{native_support_path} is missing column: method")
            display_names = support["method"].map(protocol.get("display_names", {}))
            if display_names.isna().any():
                unknown = sorted(
                    support.loc[display_names.isna(), "method"].astype(str).unique()
                )
                raise ValueError(
                    f"{native_support_path} has methods absent from the protocol: "
                    f"{unknown}"
                )
            support.insert(3, "display_name", display_names)
            _write_csv(support, staging / "native_output_support.csv")
        else:
            shutil.copyfile(native_support_path, staging / "native_output_support.csv")
        shutil.copyfile(protocol_path, staging / "protocol.json")
        load_loto_benchmark(staging)
        _write_manifest(
            staging,
            analysis="loto_benchmark",
            paper_location="Supplementary Figure S40",
            sources=sources,
            descriptions={
                "loto_target_stage_means.csv": (
                    "validated target-level Sliced-W2 means"
                ),
                "native_output_support.csv": (
                    "validated method-native output counts and output types"
                ),
                "protocol.json": "dataset, method, projection, and support settings",
            },
        )
        return _finish(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _select_no_lr_rows(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, float_precision="round_trip")
    required = [
        "dataset",
        "target",
        "space",
        "metric",
        "full",
        "no_lr_prior",
        "no_lr_prior_minus_full",
        "no_lr_prior_relative_to_full",
    ]
    missing = sorted(set(required).difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    result = table.loc[table["metric"].astype(str).eq("sliced_w2"), required].copy()
    if result.empty:
        raise ValueError(f"{path} contains no sliced_w2 target rows")
    return result


def _build_stvcr_rows(loto_dir: Path) -> pd.DataFrame:
    data = load_loto_benchmark(loto_dir)
    values = data.target_means.loc[
        data.target_means["method"].isin(["CytoBridge-0.015", "stvcr"]),
        ["dataset", "target", "method", "space", "sliced_w2"],
    ]
    wide = values.pivot(
        index=["dataset", "target", "space"],
        columns="method",
        values="sliced_w2",
    ).reset_index()
    if wide[["CytoBridge-0.015", "stvcr"]].isna().any().any():
        raise ValueError("S40 inputs do not contain a matched CytoBridge/stVCR pair")
    result = wide.rename(columns={"stvcr": "stVCR"})
    result["stvcr_minus_cytobridge"] = result["stVCR"] - result["CytoBridge-0.015"]
    result["stvcr_relative_to_cytobridge"] = (
        result["stvcr_minus_cytobridge"] / result["CytoBridge-0.015"]
    )
    return result[
        [
            "dataset",
            "target",
            "space",
            "CytoBridge-0.015",
            "stVCR",
            "stvcr_minus_cytobridge",
            "stvcr_relative_to_cytobridge",
        ]
    ]


def collect_s39(no_lr_table: Path, loto_dir: Path, output_dir: Path) -> Path:
    """Combine a matched No-LR report and S40 inputs for the S39 figure."""

    no_lr_table = no_lr_table.expanduser().resolve()
    loto_dir = loto_dir.expanduser().resolve()
    no_lr_record = _source_record(no_lr_table, label="matched Full and No-LR table")
    loto_sources = []
    for name in (
        "loto_target_stage_means.csv",
        "native_output_support.csv",
        "protocol.json",
    ):
        loto_sources.append(_source_record(loto_dir / name, label=f"S40 {name}"))
    output, staging = _new_staging_dir(output_dir)
    try:
        no_lr = _select_no_lr_rows(no_lr_table)
        stvcr = _build_stvcr_rows(loto_dir)
        _write_csv(no_lr, staging / "no_lr_paired_target_deltas.csv")
        _write_csv(stvcr, staging / "stvcr_paired_target_deltas.csv")

        load_lr_prior_stvcr_results(staging)
        _write_manifest(
            staging,
            analysis="interaction_evidence",
            paper_location="Supplementary Figure S39",
            sources=[no_lr_record, *loto_sources],
            descriptions={
                "no_lr_paired_target_deltas.csv": (
                    "paired Full and No-LR reconstruction errors"
                ),
                "stvcr_paired_target_deltas.csv": (
                    "paired CytoBridge and stVCR LOTO reconstruction errors"
                ),
            },
        )
        return _finish(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    s25 = commands.add_parser(
        "s25",
        help="collect four completed LR-complex sensitivity tables",
    )
    s25.add_argument(
        "--dataset-result",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help=(
            "repeat for zebrafish, mosta, arista, and chicken_heart; PATH may "
            "be paired_scores.csv or the sensitivity directory containing "
            "comparison/paired_scores.csv"
        ),
    )
    s25.add_argument("--output-dir", type=Path, required=True)

    s39 = commands.add_parser(
        "s39",
        help="combine the matched No-LR table with the completed S40 inputs",
    )
    s39.add_argument("--no-lr-table", type=Path, required=True)
    s39.add_argument(
        "--loto-results-dir",
        type=Path,
        required=True,
        help="the directory prepared by this script's s40 command",
    )
    s39.add_argument("--output-dir", type=Path, required=True)

    s40 = commands.add_parser(
        "s40",
        help="collect the five completed per-dataset LOTO target summaries",
    )
    s40.add_argument(
        "--dataset-summary",
        action="append",
        metavar="DATASET=PATH",
        help=(
            "repeat for all five datasets; PATH is the loto_target_summary.csv "
            "written by scripts.spatiotemporal_benchmark.summarize_results"
        ),
    )
    s40.add_argument(
        "--target-means",
        type=Path,
        help="an already assembled loto_target_stage_means.csv",
    )
    s40.add_argument(
        "--native-support",
        type=Path,
        help="native_output_support.csv paired with --target-means",
    )
    s40.add_argument("--protocol", type=Path, required=True)
    s40.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "s25":
            inputs = _parse_assignments(
                args.dataset_result,
                expected=LR_COMPLEX_DATASETS,
                option="--dataset-result",
            )
            output = collect_s25(inputs, args.output_dir)
        elif args.command == "s39":
            output = collect_s39(
                args.no_lr_table,
                args.loto_results_dir,
                args.output_dir,
            )
        else:
            if args.dataset_summary:
                if args.target_means is not None or args.native_support is not None:
                    raise ValueError(
                        "Use either five --dataset-summary values or "
                        "--target-means with --native-support, not both"
                    )
                inputs = _parse_assignments(
                    args.dataset_summary,
                    expected=LOTO_DATASETS,
                    option="--dataset-summary",
                )
                output = collect_s40(inputs, args.protocol, args.output_dir)
            else:
                if args.target_means is None or args.native_support is None:
                    raise ValueError(
                        "s40 requires five --dataset-summary values or both "
                        "--target-means and --native-support"
                    )
                output = collect_s40_tables(
                    args.target_means,
                    args.native_support,
                    args.protocol,
                    args.output_dir,
                )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as error:
        parser = build_parser()
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
