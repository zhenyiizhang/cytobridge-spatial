#!/usr/bin/env python3
"""Compatibility entry point for the supported package workflow.

The former file was a machine-specific Zebrafish prototype with hard-coded
paths and obsolete model calls.  Keep the familiar script name, but delegate to
the installed, dataset-configured CLI so there is only one downstream
implementation to maintain.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Prefer this checkout when the compatibility script is run directly.  An
# installed ``cytobridge`` command continues to import its installed package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from CytoBridge.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["workflow"] + sys.argv[1:]))
