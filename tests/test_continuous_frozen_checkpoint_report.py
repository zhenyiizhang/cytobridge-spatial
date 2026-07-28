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
    stress = report._plot_wasserstein_curves(
        metrics,
        conditions=("full", "interaction_off", "lr_gate_off"),
        output_dir=plot_dir,
        stem="stress",
        log_scale=True,
        title="stress",
        central_tendency="median",
    )
    tmv = report._plot_tmv(
        mass,
        conditions=("full", "interaction_off", "lr_gate_off"),
        output_dir=plot_dir,
    )
    paired, paired_mass = report._paired_deltas(metrics, mass)
    summary = report._paired_delta_summary(paired, paired_mass)
    assert {
        "median_delta_vs_full",
        "min_delta_vs_full",
        "max_delta_vs_full",
        "median_percent_change_vs_full",
    }.issubset(summary.columns)
    sensitivity = report._sensitivity_summary(metrics, mass)
    assert "median" in sensitivity.columns
    assert np.allclose(sensitivity["median"], sensitivity["mean"])
    interpretation = report._interpretation_markdown(
        summary,
        conditions=("full", "interaction_off", "lr_gate_off"),
        seeds=(42,),
        primary_seed=42,
        ot_sampling_seed=42,
    )
    assert "1/1" in interpretation
    assert "5/5" not in interpretation
    assert "563" not in interpretation
    assert all(
        path.stat().st_size > 0
        for path in [*wasserstein, *stress, *tmv]
    )


def test_robust_outlier_statistics_and_report() -> None:
    metric_rows = []
    mass_rows = []
    for seed, value in zip((1, 2, 3), (1.0, 2.0, 100.0)):
        metric_rows.append(
            {
                "condition": "full",
                "rollout_seed": seed,
                "is_primary_seed": seed == 1,
                "time": 1.0,
                "space": "joint",
                "w1": value,
                "w2": value + 1.0,
            }
        )
        mass_rows.append(
            {
                "condition": "full",
                "rollout_seed": seed,
                "is_primary_seed": seed == 1,
                "time": 1.0,
                "tmv": value / 100.0,
            }
        )
    sensitivity = report._sensitivity_summary(
        pd.DataFrame(metric_rows),
        pd.DataFrame(mass_rows),
    )
    robust_row = sensitivity.query(
        "time == '1.0' and space == 'joint' and metric == 'w1'"
    ).iloc[0]
    assert robust_row["median"] == pytest.approx(2.0)
    assert robust_row["mean"] == pytest.approx(103.0 / 3.0)
    assert robust_row["min"] == pytest.approx(1.0)
    assert robust_row["max"] == pytest.approx(100.0)

    paired = pd.DataFrame(
        {
            "condition": ["lr_gate_off"] * 3,
            "rollout_seed": [1, 2, 3],
            "time": [4.0] * 3,
            "space": ["joint"] * 3,
            "w1": [2.0, 3.0, 100.0],
            "w2": [3.0, 4.0, 200.0],
            "full_w1": [1.0] * 3,
            "full_w2": [1.0] * 3,
            "w1_delta_vs_full": [1.0, 2.0, 99.0],
            "w1_percent_change_vs_full": [100.0, 200.0, 9900.0],
            "w2_delta_vs_full": [2.0, 3.0, 199.0],
            "w2_percent_change_vs_full": [200.0, 300.0, 19900.0],
        }
    )
    paired_mass = pd.DataFrame(
        {
            "condition": ["lr_gate_off"] * 3,
            "rollout_seed": [1, 2, 3],
            "time": [4.0] * 3,
            "tmv": [0.1, 0.2, 5.0],
            "full_tmv": [0.01] * 3,
            "tmv_delta_vs_full": [0.09, 0.19, 4.99],
            "tmv_percent_change_vs_full": [900.0, 1900.0, 49900.0],
        }
    )
    summary = report._paired_delta_summary(paired, paired_mass)
    robust_delta = summary.query(
        "time == '4.0' and space == 'joint' and metric == 'w1'"
    ).iloc[0]
    assert robust_delta["median_delta_vs_full"] == pytest.approx(2.0)
    assert robust_delta["min_delta_vs_full"] == pytest.approx(1.0)
    assert robust_delta["max_delta_vs_full"] == pytest.approx(99.0)
    markdown = report._interpretation_markdown(
        summary,
        conditions=("full", "lr_gate_off"),
        seeds=(1, 2, 3),
        primary_seed=1,
        ot_sampling_seed=42,
        paired=paired,
        paired_mass=paired_mass,
    )
    assert "rollout seed `3`" in markdown
    assert "其余 seeds 为 [3, 4]" in markdown
    tmv_line = next(
        line for line in markdown.splitlines() if "| mass | TMV |" in line
    )
    assert "—" in tmv_line
