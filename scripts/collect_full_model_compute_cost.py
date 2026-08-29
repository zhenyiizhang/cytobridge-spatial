#!/usr/bin/env python3
"""Collect Supplementary Table 2 measurements from five training summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from CytoBridge.results.compute_cost import (
    DATASET_ORDER,
    RAW_COLUMNS,
    _DISPLAY_CONTRACT,
    _MEASUREMENT_CONTRACT,
    load_full_model_compute_cost,
)


PAPER_TIME_POINT_LABELS = {
    "admouse": "2.5, 5.7, 17.9 months",
    "arista": "2, 5, 10, 15, 20 DPI",
    "chicken_heart": "D4, D7, D10, D14",
    "mosta": "E12.5, E13.5, E14.5, E15.5",
    "zebrafish": "5.25, 10, 12, 18, 24 hpf",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_run(value: str) -> tuple[str, Path]:
    dataset, separator, path_text = value.partition("=")
    dataset = dataset.strip()
    if not separator or not path_text.strip():
        raise argparse.ArgumentTypeError(
            "--run must use DATASET=PATH, for example "
            "arista=outputs/arista/training/training_run_summary.json"
        )
    if dataset not in PAPER_TIME_POINT_LABELS:
        choices = ", ".join(DATASET_ORDER)
        raise argparse.ArgumentTypeError(
            f"unknown dataset {dataset!r}; choose from {choices}"
        )
    path = Path(path_text.strip()).expanduser()
    if path.is_dir():
        path = path / "training_run_summary.json"
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"training summary not found: {path}")
    return dataset, path.resolve()


def _positive_number(value: object, *, field: str, source: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} has an invalid {field}") from error
    if not number > 0:
        raise ValueError(f"{source} has a non-positive {field}")
    return number


def _positive_integer(value: object, *, field: str, source: Path) -> int:
    number = _positive_number(value, field=field, source=source)
    if not number.is_integer():
        raise ValueError(f"{source} has a non-integer {field}")
    return int(number)


def _read_summary(dataset: str, source: Path) -> dict[str, object]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{source} has an unsupported schema version")

    data = payload.get("data", {})
    timing = payload.get("timing", {})
    resources = payload.get("resources", {})
    environment = payload.get("environment", {})
    labels = PAPER_TIME_POINT_LABELS[dataset]
    label_count = len([part for part in labels.split(",") if part.strip()])
    n_timepoints = _positive_integer(
        data.get("n_timepoints"), field="data.n_timepoints", source=source
    )
    if n_timepoints != label_count:
        raise ValueError(
            f"{source} records {n_timepoints} time points, but the paper table "
            f"lists {label_count} for {dataset}"
        )

    expected_gpu = _MEASUREMENT_CONTRACT["hardware"]["gpu_model"]
    if environment.get("cuda_device_name") != expected_gpu:
        raise ValueError(
            f"{source} was measured on {environment.get('cuda_device_name')!r}; "
            f"Supplementary Table 2 uses {expected_gpu!r}"
        )

    return {
        "dataset": dataset,
        "time_points_used_for_training": n_timepoints,
        "training_time_point_labels": labels,
        "observed_cells_or_spots": _positive_integer(
            data.get("n_observations"),
            field="data.n_observations",
            source=source,
        ),
        "training_time_seconds": _positive_number(
            timing.get("run_wall_time_seconds"),
            field="timing.run_wall_time_seconds",
            source=source,
        ),
        "peak_host_memory_mib": _positive_number(
            resources.get("cpu_max_rss_mib"),
            field="resources.cpu_max_rss_mib",
            source=source,
        ),
        "peak_gpu_allocation_mib": _positive_number(
            resources.get("cuda_peak_allocated_mib"),
            field="resources.cuda_peak_allocated_mib",
            source=source,
        ),
    }


def collect(run_values: list[str], output_dir: Path) -> dict[str, str]:
    parsed = [_parse_run(value) for value in run_values]
    runs = dict(parsed)
    if len(runs) != len(parsed):
        raise ValueError("Each dataset may be passed to --run only once")
    missing = [dataset for dataset in DATASET_ORDER if dataset not in runs]
    if missing:
        raise ValueError("Missing --run entries for: " + ", ".join(missing))

    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = [_read_summary(dataset, runs[dataset]) for dataset in DATASET_ORDER]
    table_path = output / "full_model_compute_cost.csv"
    manifest_path = output / "manifest.json"
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(table_path, index=False)
    manifest = {
        "schema_version": 1,
        "analysis": "full_model_compute_cost",
        "manuscript_table": "Supplementary Table 2",
        "files": {
            table_path.name: "training time and peak memory by dataset"
        },
        "measurement": _MEASUREMENT_CONTRACT,
        "display": _DISPLAY_CONTRACT,
        "sources": {
            dataset: {
                "training_run_summary": str(path),
                "sha256": _sha256(path),
            }
            for dataset, path in runs.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    load_full_model_compute_cost(output)
    return {"table": str(table_path), "manifest": str(manifest_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help=(
            "completed training directory or training_run_summary.json; "
            "repeat for all five paper datasets"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.run, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
