#!/usr/bin/env python3
"""Create reviewer-ready plots and summaries from training_history.csv."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=None,
        help=(
            "Optional training_run_summary.json. If omitted, use the file next "
            "to --history when present."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="CytoBridge training history")
    args = parser.parse_args(argv)

    history = args.history.expanduser().resolve()
    if not history.is_file():
        raise FileNotFoundError(history)
    run_summary_source = (
        args.run_summary.expanduser().resolve()
        if args.run_summary is not None
        else history.with_name("training_run_summary.json")
    )
    if args.run_summary is not None and not run_summary_source.is_file():
        raise FileNotFoundError(run_summary_source)
    run_summary = None
    run_summary_sha256 = None
    if run_summary_source.is_file():
        run_summary_sha256 = _sha256(run_summary_source)
        run_summary = json.loads(run_summary_source.read_text(encoding="utf-8"))
        if not isinstance(run_summary, dict):
            raise ValueError("Training run summary must contain a JSON object.")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = cb.pl.summarize_training_history(history)
    summary_path = output_dir / "training_stage_summary.csv"
    summary.to_csv(summary_path, index=False)
    resource_summary_path = None
    if run_summary is not None:
        resource_summary_path = output_dir / "training_resource_summary.json"
        resource_summary_path.write_text(
            json.dumps(run_summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    figures = []
    for suffix in ("png", "pdf"):
        figures.append(
            cb.pl.plot_training_history(
                history,
                output_dir / f"training_loss_by_stage.{suffix}",
                title=args.title,
            )
        )

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Training-history report",
                "",
                "Each panel uses the loss contract of one training stage. "
                "Curves from different stages must not be interpreted as one "
                "continuous objective.",
                "",
                "- `optimization loss` is the scalar differentiated in that epoch.",
                "- `checkpoint` is the metric used to select a best checkpoint; "
                "it may differ from the optimization loss.",
                "- `is_best` means the epoch set a new metric record; "
                "`is_selected_checkpoint` is the one state actually retained.",
                "- The star marks `is_selected_checkpoint` and therefore follows "
                "the configured `best` or `last` save strategy.",
                "- `training_stage_summary.csv` reports start/end/minimum loss and "
                "the checkpoint, learning-rate endpoints, measured wall time, "
                "batch size, and optimizer-step count for every stage.",
                "- For schema-v1 histories, selected checkpoints are explicitly "
                "labeled as inferred and learning-rate endpoints are limited to "
                "the first/last recorded epoch; missing resource values stay empty.",
                "- `training_resource_summary.json` is emitted when measured run "
                "metadata is available. CPU RSS is a process-lifetime high-water "
                "mark; CUDA peaks are allocator high-water marks. Unavailable "
                "measurements remain JSON `null`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    outputs = [summary_path, *figures, readme]
    if resource_summary_path is not None:
        outputs.append(resource_summary_path)
    manifest = {
        "schema_version": 1,
        "command": [str(value) for value in sys.argv],
        "git": _git_state(),
        "input": {
            "path": str(history),
            "sha256": _sha256(history),
            "training_run_summary": (
                {
                    "path": str(run_summary_source),
                    "sha256": run_summary_sha256,
                }
                if run_summary is not None
                else None
            ),
        },
        "parameters": {"title": str(args.title)},
        "outputs": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
