from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "spatiotemporal_benchmark"
    / "summarize_results.py"
)
SPEC = importlib.util.spec_from_file_location("benchmark_reporting", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def _metrics() -> pd.DataFrame:
    rows = []
    for target in (1, 2):
        for method, spaces in (("joint_method", ("joint", "state", "spatial")), ("state_method", ("state",))):
            for space in spaces:
                for repeat in (0, 1):
                    rows.append(
                        {
                            "track": "loto",
                            "target": target,
                            "source_time": target - 1,
                            "method": method,
                            "space": space,
                            "projection_repeat": repeat,
                            "sliced_w2": 0.1 * target + 0.01 * repeat,
                            "exact_w1": 0.2 * target,
                            "exact_w2": 0.3 * target,
                            "tmv_available": method == "joint_method",
                            "tmv": 0.05 * target if method == "joint_method" else float("nan"),
                            "tmv_absolute": 0.01 if method == "joint_method" else float("nan"),
                            "predicted_mass": 1.0 if method == "joint_method" else float("nan"),
                            "observed_mass_relative": 1.0 if method == "joint_method" else float("nan"),
                            "output_scope": "native_joint" if method == "joint_method" else "native_state",
                            "native_vs_adapter": "native",
                            "n_predicted": 10,
                            "n_observed": 8,
                        }
                    )
    return pd.DataFrame(rows)


def _registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "methods": [
                    {
                        "method": "Joint",
                        "display_name": "Joint",
                        "aliases": ["joint_method"],
                        "spaces": ["joint", "state", "spatial"],
                        "scope": "native_joint",
                        "status": "evaluated",
                    },
                    {
                        "method": "State",
                        "display_name": "State",
                        "aliases": ["state_method"],
                        "spaces": ["state"],
                        "scope": "native_state",
                        "status": "evaluated",
                    },
                    {
                        "method": "Sensitivity",
                        "display_name": "Sensitivity",
                        "aliases": [],
                        "spaces": [],
                        "scope": "gene_sensitivity",
                        "status": "sensitivity_only",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _evaluation(path: Path, metrics: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "track": "loto",
                "metrics_long_csv_sha256": reporting.sha256_file(metrics),
                "methods": ["joint_method", "state_method"],
                "targets": [1, 2],
            }
        ),
        encoding="utf-8",
    )


def test_summary_preserves_na_and_never_creates_cross_space_score(tmp_path: Path) -> None:
    metrics = tmp_path / "loto_metrics_long.csv"
    _metrics().to_csv(metrics, index=False)
    registry = tmp_path / "registry.json"
    _registry(registry)
    evaluation = tmp_path / "evaluation.json"
    _evaluation(evaluation, metrics)
    output = tmp_path / "summary"

    manifest = reporting.summarize(
        Namespace(
            metrics_long=metrics,
            evaluation_manifest=evaluation,
            method_registry=registry,
            output_dir=output,
        )
    )

    method = pd.read_csv(output / "loto_method_summary.csv")
    state_spatial = method[(method["method"] == "State") & (method["space"] == "spatial")]
    assert state_spatial.iloc[0]["status"] != "evaluated"
    assert pd.isna(state_spatial.iloc[0]["sliced_w2_mean"])
    assert "overall" not in " ".join(method.columns).lower()
    assert manifest["rank_policy"].startswith("within each feature space")
    assert (output / "loto_sliced_w2_barplot.png").is_file()
    assert (output / "loto_applicability_matrix.png").is_file()


def test_projection_repeats_are_collapsed_before_target_aggregation() -> None:
    target = reporting._target_summary(_metrics())
    joint = target[
        (target["method"] == "joint_method")
        & (target["target"] == 1)
        & (target["space"] == "joint")
    ].iloc[0]
    assert joint["sliced_w2"] == pytest.approx(0.105)
    assert joint["n_projection_repeats"] == 2


def test_summary_rejects_metrics_truncated_after_evaluation(tmp_path: Path) -> None:
    complete = tmp_path / "complete.csv"
    _metrics().to_csv(complete, index=False)
    evaluation = tmp_path / "evaluation.json"
    _evaluation(evaluation, complete)
    truncated = tmp_path / "truncated.csv"
    _metrics().query("target == 1").to_csv(truncated, index=False)
    registry = tmp_path / "registry.json"
    _registry(registry)
    with pytest.raises(reporting.SummaryError, match="SHA-256|targets|grid"):
        reporting.summarize(
            Namespace(
                metrics_long=truncated,
                evaluation_manifest=evaluation,
                method_registry=registry,
                output_dir=tmp_path / "out",
            )
        )
