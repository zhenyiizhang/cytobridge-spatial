from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "spatiotemporal_benchmark"
    / "reviewer_shape_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("reviewer_shape_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
shape = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shape)

MATCHED_TEST = ROOT / "tests" / "test_spatiotemporal_benchmark_matched.py"
MATCHED_TEST_SPEC = importlib.util.spec_from_file_location(
    "matched_fixture_for_shape", MATCHED_TEST
)
assert MATCHED_TEST_SPEC is not None and MATCHED_TEST_SPEC.loader is not None
matched_fixture = importlib.util.module_from_spec(MATCHED_TEST_SPEC)
MATCHED_TEST_SPEC.loader.exec_module(matched_fixture)


def test_bures_identical_rank_deficient_covariance_is_zero() -> None:
    covariance = np.asarray(
        [
            [4.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        shape.covariance_bures_distance(covariance, covariance.copy()),
        0.0,
        atol=0.0,
    )


def test_centered_metrics_remove_translation_and_preserve_duplicate_mass() -> None:
    observed = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    predicted = np.asarray([[2.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    weights = np.asarray([0.25, 0.25, 0.50])
    observed_indices = np.arange(observed.shape[0], dtype=np.int64)

    metrics = shape._space_metrics(
        predicted=predicted,
        observed=observed,
        predicted_weights=weights,
        benchmark="synthetic",
        seed_split="matched_t1",
        space="state",
        n_projections=16,
        projection_repeats=2,
        max_ot_points=10,
        observed_indices=observed_indices,
    )

    np.testing.assert_allclose(metrics["centroid_error"], 2.0, atol=1e-12)
    np.testing.assert_allclose(metrics["centered_sliced_w2"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["centered_exact_w1"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["centered_exact_w2"], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        metrics["covariance_bures_distance"], 0.0, atol=1e-12
    )
    assert metrics["n_predicted_rows"] == 3
    assert metrics["n_predicted_unique_support"] == 2
    assert metrics["n_predicted_duplicate_rows_collapsed"] == 1
    np.testing.assert_allclose(metrics["predicted_effective_support_size"], 2.0)


def test_shape_plot_reserves_separate_title_and_legend_bands(
    tmp_path: Path, monkeypatch
) -> None:
    records = []
    for space in shape.SPACE_ORDER:
        for track, value in (("full_data", 0.1), ("loto", 0.2)):
            for target in (1.0, 2.0, 3.0):
                records.append(
                    {
                        "method": "method",
                        "method_display_name": "Method",
                        "space": space,
                        "track": track,
                        "target": target,
                        "centroid_error": value + 0.01 * target,
                    }
                )
    captured: dict[str, object] = {}
    pyplot = importlib.import_module("matplotlib.pyplot")
    monkeypatch.setattr(
        pyplot,
        "close",
        lambda figure: captured.setdefault("figure", figure),
    )

    shape._plot_metric(
        pd.DataFrame.from_records(records),
        metric="centroid_error",
        method_order=["method"],
        output_dir=tmp_path,
    )

    figure = captured["figure"]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    title_box = figure._suptitle.get_window_extent(renderer)
    legend_box = figure.legends[0].get_window_extent(renderer)
    assert legend_box.y1 < title_box.y0
    assert max(axis.get_window_extent(renderer).y1 for axis in figure.axes) < (
        legend_box.y0
    )


def test_end_to_end_uses_bound_matched_inventory_and_separates_tracks(
    tmp_path: Path,
) -> None:
    manifest, loto_root, full_root, _ = matched_fixture._build_contract(tmp_path)
    matched_output = tmp_path / "matched"
    matched_args = matched_fixture._args(
        manifest, loto_root, full_root, matched_output
    )
    _, _, matched_manifest = matched_fixture.matched.evaluate_matched(matched_args)
    matched_manifest_path = matched_output / "matched_evaluation_manifest.json"
    matched_fixture.matched._write_final_manifest(
        matched_manifest_path, matched_manifest
    )

    shape_output = tmp_path / "shape"
    args = Namespace(
        matched_manifest=matched_manifest_path,
        output_dir=shape_output,
        methods=None,
        tracks=None,
        targets=None,
        n_projections=12,
        projection_repeats=2,
        max_ot_points=3,
        no_plots=True,
    )
    metrics, summary, paired, shape_manifest = shape.run_analysis(args)

    assert len(metrics) == 16
    assert len(summary) == 8
    assert len(paired) == 8
    assert set(metrics["comparison_role"]) == {
        "loto_transductive_interpolation",
        "full_data_in_sample_oracle_control",
    }
    assert metrics.loc[metrics["track"] == "full_data", "is_in_sample"].all()
    assert metrics.loc[metrics["track"] == "full_data", "is_oracle_control"].all()
    assert not metrics.loc[metrics["track"] == "loto", "is_in_sample"].any()
    assert not metrics.loc[metrics["track"] == "loto", "is_oracle_control"].any()
    assert set(metrics.loc[metrics["method"] == "state_method", "space"]) == {
        "state"
    }
    assert not any(
        "rank" in column.casefold() or "overall" in column.casefold()
        for column in metrics.columns
    )
    for _, frame in metrics.groupby(["target", "space"], sort=True):
        assert frame["exact_ot_seed"].nunique() == 1
        assert frame["exact_ot_observed_indices_sha256"].nunique() == 1
        assert frame["projection_seeds_json"].nunique() == 1
        assert frame["projection_hashes_json"].nunique() == 1

    assert (paired["centroid_error_loto_minus_full_data"] > 0).all()
    np.testing.assert_allclose(metrics["centered_exact_w1"], 0.0, atol=1e-12)
    np.testing.assert_allclose(metrics["centered_exact_w2"], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        metrics["covariance_bures_distance"], 0.0, atol=1e-12
    )
    assert shape_manifest["primary_metrics_modified"] is False
    assert shape_manifest["reporting_policy"]["method_specific_tuning"] is False
    assert shape_manifest["reporting_policy"]["ranking"] is False
    assert (
        shape_manifest["tracks"]["loto"]["comparison_role"]
        == "loto_transductive_interpolation"
    )
    assert (
        shape_manifest["tracks"]["full_data"]["comparison_role"]
        == "full_data_in_sample_oracle_control"
    )
    for filename in (
        "shape_metrics_long.csv",
        "shape_metrics_summary.csv",
        "shape_metrics_paired_gaps.csv",
        "shape_analysis_manifest.json",
    ):
        assert (shape_output / filename).is_file()
    published = json.loads(
        (shape_output / "shape_analysis_manifest.json").read_text(encoding="utf-8")
    )
    assert published == shape_manifest


def test_legacy_manifest_reconstructs_missing_scope_metadata(
    tmp_path: Path,
) -> None:
    manifest, loto_root, full_root, _ = matched_fixture._build_contract(tmp_path)
    matched_output = tmp_path / "matched"
    matched_args = matched_fixture._args(
        manifest, loto_root, full_root, matched_output
    )
    _, _, matched_manifest = matched_fixture.matched.evaluate_matched(matched_args)
    assert matched_manifest.pop("scope_compatibility_audit", None) is not None

    inventory_path = Path(matched_manifest["prediction_inventory"])
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    for record in inventory["records"]:
        assert record.pop("scope_compatibility", None) is not None
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    inventory_sha = matched_fixture.matched.primary.sha256_file(inventory_path)
    matched_manifest["prediction_inventory_sha256"] = inventory_sha

    bound_path = Path(matched_manifest["bound_run_contract"])
    bound = json.loads(bound_path.read_text(encoding="utf-8"))
    bound["prediction_inventory_sha256"] = inventory_sha
    bound_path.write_text(
        json.dumps(bound, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    matched_manifest["bound_run_contract_sha256"] = (
        matched_fixture.matched.primary.sha256_file(bound_path)
    )
    matched_manifest_path = matched_output / "matched_evaluation_manifest.json"
    matched_fixture.matched._write_final_manifest(
        matched_manifest_path, matched_manifest
    )

    args = Namespace(
        matched_manifest=matched_manifest_path,
        output_dir=tmp_path / "legacy_shape",
        methods=None,
        tracks=None,
        targets=None,
        n_projections=12,
        projection_repeats=2,
        max_ot_points=3,
        no_plots=True,
    )
    _, _, _, shape_manifest = shape.run_analysis(args)

    audit = shape_manifest["scope_compatibility_verification"]
    assert audit["original_manifest_field_present"] is False
    assert audit["original_inventory_record_field_present"] is False
    assert audit["reconstructed_from_bound_inventory"] is True
    assert audit["reconstructed_inventory_record_fields"] is True
    assert audit["verified"] is True
    assert audit["n_records"] == 8
