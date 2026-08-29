"""Command-line helpers for paper-figure reproduction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._io import prepare_output_dir


def new_output_dir(path: str | Path) -> Path:
    """Create the empty output directory required by a CLI run."""

    return prepare_output_dir(path, require_empty=True)


def write_run_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write the small public run summary."""

    path = output_dir / "run_summary.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
