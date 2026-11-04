#!/usr/bin/env python3
"""Compatibility alias for the packaged ARISTA workflow.

New analyses should call ``cytobridge workflow --config arista`` directly.
This short alias remains for users who remember the historical script name; it
does not contain a second preprocessing or training implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from CytoBridge.cli import main as cytobridge_main


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unified package workflow with the ARISTA preset selected."""

    forwarded = list(sys.argv[1:] if argv is None else argv)
    return cytobridge_main(["workflow", "--config", "arista", *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
