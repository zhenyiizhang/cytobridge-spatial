#!/usr/bin/env python3
"""Bind caller-selected S39 producer outputs to their exact SHA-256 values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_zebrafish_attention_validation import (
    REQUIRED_INPUTS, SCHEMA_VERSION, WORKFLOW, _artifact, _load_spec,
)


def build(artifacts: list[str], output: Path) -> Path:
    bound = {}
    for value in artifacts:
        label, separator, path = value.partition("=")
        if not separator or label not in REQUIRED_INPUTS or label in bound:
            raise ValueError(f"Expected one unique REQUIRED_INPUT=PATH assignment, got {value!r}")
        bound[label] = _artifact(Path(path))
    missing = set(REQUIRED_INPUTS) - set(bound)
    if missing:
        raise ValueError(f"Missing producer outputs: {sorted(missing)}")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = dict(schema_version=SCHEMA_VERSION, workflow=WORKFLOW, dataset="zebrafish",
                   cell_type_key="Annotation", spatial_key="spatial_aligned",
                   permutations=1000, random_seed=20260816, artifacts=bound)
    text = json.dumps(payload, indent=2) + "\n"
    with tempfile.TemporaryDirectory(prefix="cytobridge-s39-spec-") as temporary:
        candidate = Path(temporary) / "input_spec.json"
        candidate.write_text(text)
        _load_spec(candidate)  # Includes the actual upstream acceptance PASS check.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(text)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True,
                        help="Repeat LABEL=PATH for: " + ", ".join(REQUIRED_INPUTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.artifact, args.output))
