from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from CytoBridge.tl.downstream.evaluation import DistributionEvaluationResult


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_continuous_frozen_checkpoint_ablations.py"
SPEC = importlib.util.spec_from_file_location(
    "continuous_frozen_checkpoint_report", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _evaluation(weights: np.ndarray) -> DistributionEvaluationResult:
    points = np.arange(12, dtype=float).reshape(3, 4)
    return DistributionEvaluationResult(
        time_points=(0.0, 1.0),
        spatial_dim=2,
        predicted_points={0.0: points, 1.0: points + 0.25},
        predicted_weights={
            0.0: np.ones(3, dtype=float),
            1.0: np.asarray(weights, dtype=float),
        },
        observed_points={
            0.0: points[:2],
            1.0: np.vstack((points, points[:1] + 1.0)),
        },
        metrics=pd.DataFrame(),
        settings={},
    )


def test_output_guard_and_failure_log(tmp_path: Path) -> None:
    output = tmp_path / "report"
    assert report._prepare_output_dir(output, overwrite=False) == output
    marker = output / "keep.txt"
    marker.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        report._prepare_output_dir(output, overwrite=False)
    report._prepare_output_dir(output, overwrite=True)
    assert not marker.exists()

    try:
        raise RuntimeError("synthetic all-spatial failure")
    except RuntimeError as error:
        failure_json, failure_trace = report._write_seed_failure(
            output,
            seed=42,
            conditions=("full", "lr_gate_off"),
            error=error,
        )
    record = json.loads(failure_json.read_text(encoding="utf-8"))
    assert record["all_spatial_stress_condition_requested"] is True
    assert "synthetic all-spatial failure" in record["exception_message"]
    assert "RuntimeError" in failure_trace.read_text(encoding="utf-8")


def test_fixed_ot_order_mass_validation_and_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_metric(predicted, observed, **kwargs):
        calls.append(int(kwargs["random_seed"]))
        distance = float(np.mean(np.abs(predicted - observed[: len(predicted)])))
        return {
            "w1": distance,
            "w2": distance + 0.1,
            "ot_predicted_points": len(predicted),
            "ot_observed_points": len(observed),
        }

    monkeypatch.setattr(report, "compute_distribution_metrics", fake_metric)
    result = SimpleNamespace(
        evaluations={
            "full": _evaluation(np.asarray([0.5, 0.5, 0.5])),
            "interaction_off": _evaluation(np.asarray([0.6, 0.6, 0.6])),
            "lr_gate_off": _evaluation(np.asarray([1.0, 1.0, 1.0])),
        }
    )
    metrics, mass = report._evaluate_result_with_fixed_ot_seed(
        result,
        rollout_seed=42,
        primary_seed=42,
        max_ot_points=1024,
        ot_sampling_seed=42,
    )
    assert metrics["space"].drop_duplicates().tolist() == [
        "joint",
        "spatial",
        "state",
    ]
    assert calls[:3] == [143, 144, 145]
    full_mass = mass.query("condition == 'full'").iloc[0]
    assert full_mass["predicted_mass"] == pytest.approx(1.5)
    assert full_mass["observed_mass_relative"] == pytest.approx(2.0)
    assert full_mass["tmv"] == pytest.approx(0.25)

    invalid = SimpleNamespace(
        evaluations={"full": _evaluation(np.zeros(3, dtype=float))}
    )
    with pytest.raises(ValueError, match="finite and positive"):
        report._evaluate_result_with_fixed_ot_seed(
            invalid,
            rollout_seed=42,
            primary_seed=42,
            max_ot_points=1024,
            ot_sampling_seed=42,
        )

    plot_dir = tmp_path / "plots"
    wasserstein = report._plot_wasserstein_curves(
        metrics,
        conditions=("full", "interaction_off"),
        output_dir=plot_dir,
        stem="main",
        log_scale=False,
        title="test",
    )
    tmv = report._plot_tmv(
        mass,
        conditions=("full", "interaction_off", "lr_gate_off"),
        output_dir=plot_dir,
    )
    assert all(path.stat().st_size > 0 for path in [*wasserstein, *tmv])
