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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="CytoBridge training history")
    args = parser.parse_args(argv)

    history = args.history.expanduser().resolve()
    if not history.is_file():
        raise FileNotFoundError(history)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = cb.pl.summarize_training_history(history)
    summary_path = output_dir / "training_stage_summary.csv"
    summary.to_csv(summary_path, index=False)
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
                "- The star follows the configured `best` or `last` save strategy.",
                "- `training_stage_summary.csv` reports start/end/minimum loss and "
                "the checkpoint actually selected for every stage.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    outputs = [summary_path, *figures, readme]
    manifest = {
        "schema_version": 1,
        "command": [str(value) for value in sys.argv],
        "git": _git_state(),
        "input": {
            "path": str(history),
            "sha256": _sha256(history),
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
