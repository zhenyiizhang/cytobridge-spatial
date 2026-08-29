#!/usr/bin/env python3
"""Verify every file in the MOSTA reader release against CHECKSUMS.sha256."""
from __future__ import annotations
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
errors = []
for line in (root / "CHECKSUMS.sha256").read_text().splitlines():
    if not line.strip():
        continue
    expected, relative = line.split("  ", 1)
    path = root / relative
    if not path.is_file():
        errors.append(f"missing: {relative}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        errors.append(f"mismatch: {relative}: {digest} != {expected}")
if errors:
    raise SystemExit("\n".join(errors))
print("PASS: all MOSTA reader-release files match CHECKSUMS.sha256")
