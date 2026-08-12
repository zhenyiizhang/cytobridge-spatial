from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_runner_module():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_arista_end_to_end.py"
    )
    sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("run_arista_end_to_end_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arista_compatibility_runner_delegates_to_unified_workflow(monkeypatch) -> None:
    runner = _load_runner_module()
    captured = {}

    def fake_main(arguments):
        captured["arguments"] = arguments
        return 7

    monkeypatch.setattr(runner, "cytobridge_main", fake_main)
    result = runner.main(
        [
            "--train",
            "--input-h5ad",
            "Regeneration.h5ad",
            "--output-dir",
            "arista-output",
        ]
    )

    assert result == 7
    assert captured["arguments"] == [
        "workflow",
        "--config",
        "arista",
        "--train",
        "--input-h5ad",
        "Regeneration.h5ad",
        "--output-dir",
        "arista-output",
    ]
