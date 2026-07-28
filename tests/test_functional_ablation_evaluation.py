from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from CytoBridge.tl.downstream.functional_ablation_evaluation import (
    evaluate_frozen_ablation_distributions,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_frozen_checkpoint_distributions.py"
)


def _example_clouds() -> tuple[dict[str, np.ndarray], np.ndarray]:
    full = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [1.0, 1.0, 0.5, 0.5],
            [2.0, 1.0, 1.0, 0.5],
        ],
        dtype=float,
    )
    conditions = {
        "full": full,
        "interaction_off": full + np.asarray([0.0, 0.0, 1.0, 0.0]),
        "lr_gate_off": full + np.asarray([1.0, 0.0, 0.0, 2.0]),
    }
    observed = full + np.asarray([0.1, -0.2, 0.3, -0.1])
    return conditions, observed


def test_frozen_distribution_evaluation_reports_primary_and_percent_change() -> None:
    conditions, observed = _example_clouds()
    result = evaluate_frozen_ablation_distributions(
        conditions,
        observed,
        spatial_dim=2,
        max_ot_points=None,
        primary_seed=42,
        sensitivity_seeds=(7, 42),
    )

    observed_table = result.condition_vs_observed
    assert set(observed_table["sampling_seed"]) == {7, 42}
    assert observed_table["is_primary_seed"].sum() == 9
    primary = observed_table.loc[observed_table["is_primary_seed"]]
    for _, row in primary.iterrows():
        full = primary.loc[
            (primary["condition"] == "full")
            & (primary["space"] == row["space"])
        ].iloc[0]
        expected = 100.0 * (row["w1"] - full["w1"]) / full["w1"]
        assert row["w1_percent_change_vs_full"] == pytest.approx(expected)
    assert set(result.sensitivity_summary["comparison"]) == {
        "condition_vs_observed",
        "condition_vs_full",
    }
    assert result.settings["mass_policy"] == "uniform_empirical"


def test_condition_vs_full_uses_paired_support_and_identity_bounds() -> None:
    conditions, observed = _example_clouds()
    result = evaluate_frozen_ablation_distributions(
        conditions,
        observed,
        spatial_dim=2,
        max_ot_points=3,
        primary_seed=42,
        sensitivity_seeds=(42,),
    )
    table = result.condition_vs_full

    full = table.loc[table["condition"] == "full"]
    np.testing.assert_allclose(full[["w1", "w2"]], 0.0)
    nonfull = table.loc[table["condition"] != "full"]
    assert np.all(
        nonfull["w1"]
        <= nonfull["identity_coupling_w1_upper_bound"] + 1e-10
    )
    assert np.all(
        nonfull["w2"]
        <= nonfull["identity_coupling_w2_upper_bound"] + 1e-10
    )
    interaction_state = table.loc[
        (table["condition"] == "interaction_off")
        & (table["space"] == "state")
    ].iloc[0]
    assert interaction_state["w1"] == pytest.approx(1.0)
    assert interaction_state["w2"] == pytest.approx(1.0)
    assert interaction_state["paired_displacement_median"] == pytest.approx(1.0)
    assert interaction_state[
        "full_cohort_paired_displacement_median"
    ] == pytest.approx(1.0)
    assert interaction_state["subsampling_policy"] == (
        "shared_fixed_cohort_indices_without_replacement"
    )


def test_frozen_distribution_evaluation_rejects_unmatched_cohort() -> None:
    conditions, observed = _example_clouds()
    conditions["interaction_off"] = conditions["interaction_off"][:-1]
    with pytest.raises(ValueError, match="same fixed-cohort shape"):
        evaluate_frozen_ablation_distributions(
            conditions,
            observed,
            spatial_dim=2,
        )


def test_cli_endpoint_loader_and_barplots(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location(
        "evaluate_frozen_checkpoint_distributions",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    conditions, observed = _example_clouds()
    trajectory_path = tmp_path / "full.npz"
    np.savez_compressed(
        trajectory_path,
        times=np.asarray([3.0, 4.0]),
        points=np.stack((conditions["full"] - 0.1, conditions["full"])),
    )
    endpoint, metadata = module._load_condition_endpoint(
        trajectory_path,
        target_time=4.0,
    )
    np.testing.assert_array_equal(endpoint, conditions["full"])
    assert metadata["target_frame_index"] == 1

    result = evaluate_frozen_ablation_distributions(
        conditions,
        observed,
        spatial_dim=2,
        max_ot_points=None,
        primary_seed=42,
        sensitivity_seeds=(7, 42),
    )
    observed_paths = module._plot_condition_vs_observed(
        result.condition_vs_observed,
        condition_order=list(conditions),
        full_condition="full",
        primary_seed=42,
        target_time=4.0,
        output_dir=tmp_path,
    )
    full_paths = module._plot_condition_vs_full(
        result.condition_vs_full,
        condition_order=list(conditions),
        full_condition="full",
        primary_seed=42,
        output_dir=tmp_path,
    )
    assert all(path.is_file() and path.stat().st_size > 1000 for path in observed_paths)
    assert all(path.is_file() and path.stat().st_size > 1000 for path in full_paths)
