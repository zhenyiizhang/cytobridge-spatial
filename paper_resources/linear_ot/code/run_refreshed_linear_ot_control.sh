#!/usr/bin/env bash
set -euo pipefail

BENCH=/data/cytobridge/projects/CytoBridge-ST-1104/software/cytobridge-release-838fece-s35-benchmark-r1
PY=/data/cytobridge/projects/CytoBridge-ST-1104/envs/arista-api/bin/python
ACCEPTED=/data/cytobridge/projects/CytoBridge-ST-1104/runs/rev03-arista-heart-all-method-loto-20260826-r1
CONTROL=/data/cytobridge/projects/CytoBridge-ST-1104/runs/rev03-arista-heart-linear-ot-control-20260901-r1

test ! -e "${CONTROL}"
mkdir -p "${CONTROL}/arista" "${CONTROL}/chicken_heart" "${CONTROL}/logs"
cp -a "${ACCEPTED}/arista/inputs" "${CONTROL}/arista/inputs"
cp -a "${ACCEPTED}/chicken_heart/inputs" "${CONTROL}/chicken_heart/inputs"

cd "${BENCH}"
"${PY}" scripts/spatiotemporal_benchmark/run_unified_benchmark.py \
  --datasets arista chicken_heart \
  --run-root "${CONTROL}" \
  run --methods exact_ot_displacement --tracks loto --timeout 3600 \
  2>&1 | tee "${CONTROL}/logs/run_exact_ot.log"

"${PY}" -m scripts.spatiotemporal_benchmark.evaluate_predictions \
  --input-manifest "${CONTROL}/arista/inputs/manifest.json" \
  --predictions-root "${CONTROL}/arista/predictions/loto" \
  --track loto --targets 1 2 3 --methods exact_ot_displacement \
  --n-projections 1024 --projection-repeats 5 --max-ot-points 800 \
  --output-dir "${CONTROL}/arista/evaluation/exact_ot_loto" \
  2>&1 | tee "${CONTROL}/logs/evaluate_arista.log"

"${PY}" -m scripts.spatiotemporal_benchmark.evaluate_predictions \
  --input-manifest "${CONTROL}/chicken_heart/inputs/manifest.json" \
  --predictions-root "${CONTROL}/chicken_heart/predictions/loto" \
  --track loto --targets 1 2 --methods exact_ot_displacement \
  --n-projections 1024 --projection-repeats 5 --max-ot-points 800 \
  --output-dir "${CONTROL}/chicken_heart/evaluation/exact_ot_loto" \
  2>&1 | tee "${CONTROL}/logs/evaluate_chicken_heart.log"

"${PY}" - "${CONTROL}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

root = Path(sys.argv[1])
expected = {
    "arista": {
        "targets": {1, 2, 3},
        "rows": 45,
        "input": "5491162645e69021470217b0178d64760be0b624fce0c97e92ff0d36f2f9bd11",
    },
    "chicken_heart": {
        "targets": {1, 2},
        "rows": 30,
        "input": "f2f90b51525c6c9dab8a0ec7864eb43e12333bd7d899be39c70f7528ea9d20e3",
    },
}
artifacts = {}
for dataset, contract in expected.items():
    path = root / dataset / "evaluation/exact_ot_loto/loto_metrics_long.csv"
    frame = pd.read_csv(path)
    assert len(frame) == contract["rows"]
    assert set(frame["target"]) == contract["targets"]
    assert set(frame["method"]) == {"exact_ot_displacement"}
    assert set(frame["space"]) == {"joint", "spatial", "state"}
    assert set(frame["projection_repeat"]) == set(range(5))
    assert set(frame["input_manifest_sha256"]) == {contract["input"]}
    assert not frame.duplicated(
        ["target", "method", "space", "projection_repeat"]
    ).any()
    artifacts[dataset] = {
        "metrics": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
    }

manifest = {
    "schema_version": "1.0.0",
    "status": "complete",
    "purpose": "Linear OT displacement control for the refreshed ARISTA and chicken-heart LOTO benchmark",
    "accepted_input_root": str(root),
    "benchmark_commit": "61f0b550678ed75e706638ceb7638a0818b7e033",
    "artifacts": artifacts,
}
(root / "RUN_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2, sort_keys=True))
PY
