#!/usr/bin/env python3
"""Build the S41 input tables from five completed training runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from CytoBridge.pl.training import summarize_training_history
from CytoBridge.results.training_histories import (
    CHECKPOINT_COLUMNS,
    DATASET_ORDER,
    HISTORY_COLUMNS,
    STAGES,
    load_training_history_results,
)


DATASET_LABELS = {
    "zebrafish": "Zebrafish",
    "mosta": "MOSTA",
    "arista": "ARISTA",
    "admouse": "AD mouse",
    "chicken_heart": "Chicken heart",
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
            "arista=outputs/arista/training"
        )
    if dataset not in DATASET_LABELS:
        choices = ", ".join(DATASET_LABELS)
        raise argparse.ArgumentTypeError(
            f"unknown dataset {dataset!r}; choose from {choices}"
        )
    path = Path(path_text.strip()).expanduser()
    history = path / "training_history.csv" if path.is_dir() else path
    if not history.is_file():
        raise argparse.ArgumentTypeError(f"training history not found: {history}")
    return dataset, history.resolve()


def _configured_epochs(frame: pd.DataFrame, *, source: Path) -> int:
    if "epochs" not in frame:
        return int(frame["epoch"].nunique())
    values = pd.to_numeric(frame["epochs"], errors="raise").drop_duplicates()
    if len(values) != 1 or not np.isfinite(values.iloc[0]):
        raise ValueError(f"{source} must record one configured epoch count per stage")
    value = float(values.iloc[0])
    if value <= 0 or value != int(value):
        raise ValueError(f"{source} contains an invalid configured epoch count")
    return int(value)


def _read_run(dataset: str, history_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    frame = pd.read_csv(history_path, float_precision="round_trip")
    summary = summarize_training_history(frame)
    stage_keys = list(
        frame[["stage_index", "stage"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected_keys = [(stage.stage_index, stage.stage) for stage in STAGES]
    if stage_keys != expected_keys:
        raise ValueError(
            f"{history_path} does not contain the expected six training stages"
        )

    rows: list[dict] = []
    for stage in STAGES:
        current = frame.loc[
            pd.to_numeric(frame["stage_index"], errors="raise").eq(stage.stage_index)
            & frame["stage"].astype(str).eq(stage.stage)
        ].sort_values("epoch")
        stage_summary = summary.loc[
            summary["stage_index"].eq(stage.stage_index)
            & summary["stage"].eq(stage.stage)
        ]
        if current.empty or len(stage_summary) != 1:
            raise ValueError(f"{history_path} is incomplete for stage {stage.stage}")
        first_metric = float(
            pd.to_numeric(current.iloc[0]["checkpoint_value"], errors="raise")
        )
        selected_metric = float(stage_summary.iloc[0]["selected_checkpoint_value"])
        selected_epoch = int(stage_summary.iloc[0]["selected_checkpoint_epoch"])
        configured_epochs = _configured_epochs(current, source=history_path)
        if not np.isfinite([first_metric, selected_metric]).all():
            raise ValueError(
                f"{history_path} contains a non-finite checkpoint metric for "
                f"{stage.stage}"
            )
        if first_metric == 0:
            raise ValueError(
                f"{history_path} cannot calculate a reduction from a zero first "
                f"checkpoint metric for {stage.stage}"
            )
        rows.append(
            {
                "dataset": DATASET_LABELS[dataset],
                "stage": stage.stage,
                "first_checkpoint_metric": first_metric,
                "selected_checkpoint_metric": selected_metric,
                "selected_epoch": selected_epoch,
                "configured_epochs": configured_epochs,
                "percent_reduction": 100.0
                * (1.0 - selected_metric / first_metric),
            }
        )

    history = frame.loc[:, HISTORY_COLUMNS].copy()
    history["stage_index"] = pd.to_numeric(
        history["stage_index"], errors="raise"
    ).astype(int)
    history["epoch"] = pd.to_numeric(history["epoch"], errors="raise").astype(int)
    history["loss"] = pd.to_numeric(history["loss"], errors="raise")
    return history, rows


def collect(run_values: list[str], output_dir: Path) -> dict[str, str]:
    parsed = [_parse_run(value) for value in run_values]
    runs = dict(parsed)
    if len(runs) != len(parsed):
        raise ValueError("Each dataset may be passed to --run only once")
    missing = [dataset for dataset in DATASET_LABELS if dataset not in runs]
    if missing:
        raise ValueError("Missing --run entries for: " + ", ".join(missing))

    output = output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    histories: dict[str, pd.DataFrame] = {}
    panel_rows: list[dict] = []
    for dataset in DATASET_LABELS:
        history, rows = _read_run(dataset, runs[dataset])
        histories[dataset] = history
        panel_rows.extend(rows)

    history_path = output / "arista_training_history.csv"
    panel_path = output / "panel_metrics.csv"
    manifest_path = output / "manifest.json"
    histories["arista"].to_csv(history_path, index=False)
    panel = pd.DataFrame(panel_rows, columns=CHECKPOINT_COLUMNS)
    panel.to_csv(panel_path, index=False)

    manifest = {
        "schema_version": 1,
        "analysis": "training_histories",
        "manuscript_figure": "Supplementary Figure S41",
        "displayed_dataset": "arista",
        "displayed_dataset_label": "ARISTA",
        "dataset_order": list(DATASET_ORDER),
        "history": {
            "file": history_path.name,
            "columns": list(HISTORY_COLUMNS),
            "rows": len(histories["arista"]),
        },
        "checkpoint_summary": {
            "file": panel_path.name,
            "columns": list(CHECKPOINT_COLUMNS),
            "rows": len(panel),
        },
        "stages": [asdict(stage) for stage in STAGES],
        "sources": {
            dataset: {
                "training_history": str(path),
                "sha256": _sha256(path),
            }
            for dataset, path in runs.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Reuse the public loader as a final schema and arithmetic check.
    load_training_history_results(output)
    return {
        "history": str(history_path),
        "checkpoint_summary": str(panel_path),
        "manifest": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help=(
            "completed training directory or training_history.csv; repeat for "
            "zebrafish, mosta, arista, admouse, and chicken_heart"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.run, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
