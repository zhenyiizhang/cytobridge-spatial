#!/usr/bin/env python3
"""Deprecated alias for the canonical clean-counts zebrafish runner.

This filename is retained so old links fail safely into the current CLI instead
of executing the former machine-specific script with hard-coded Lustre paths.
Use ``scripts/run_zebrafish_end_to_end.py`` directly in new workflows.
"""

from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_zebrafish_end_to_end import main as canonical_main  # noqa: E402


def main() -> int:
    print(
        "[deprecated] scripts/train_zebrafish.py is an alias; use "
        "scripts/run_zebrafish_end_to_end.py.",
        file=sys.stderr,
    )
    return canonical_main()


if __name__ == "__main__":
    raise SystemExit(main())
