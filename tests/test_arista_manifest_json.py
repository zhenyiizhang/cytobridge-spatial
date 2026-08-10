from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_runner_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_arista_end_to_end.py"
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("run_arista_end_to_end_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_ready_handles_h5ad_numpy_values() -> None:
    runner = _load_runner_module()
    value = {
        "integer": np.int64(3),
        "boolean": np.bool_(True),
        "array": np.asarray(["normalize_total", "log1p"]),
        "nested": {"fraction": np.float64(0.9999)},
    }

    converted = runner._json_ready(value)
    assert converted == {
        "integer": 3,
        "boolean": True,
        "array": ["normalize_total", "log1p"],
        "nested": {"fraction": 0.9999},
    }
    json.dumps(converted)
