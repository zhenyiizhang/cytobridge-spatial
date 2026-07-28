from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "spatiotemporal_benchmark"
    / "report_zebrafish_loto_spatial.py"
)
SPEC = importlib.util.spec_from_file_location("zebrafish_loto_spatial_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)


def test_prediction_weights_are_normalised_and_unweighted_control_is_uniform(
    tmp_path: Path,
) -> None:
    unweighted = tmp_path / "linear.npz"
    np.savez_compressed(
        unweighted,
        spatial=np.array([[0.0, 0.0], [1.0, 1.0]]),
        state=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    linear = report.load_prediction(unweighted)
    assert linear.weights.tolist() == pytest.approx([0.5, 0.5])

    weighted = tmp_path / "cytobridge.npz"
    np.savez_compressed(
        weighted,
        spatial=np.array([[0.0, 0.0], [1.0, 1.0]]),
        state=np.array([[0.0, 1.0], [1.0, 0.0]]),
        weights=np.array([1.0, 3.0]),
    )
    cytobridge = report.load_prediction(weighted)
    assert cytobridge.weights.tolist() == pytest.approx([0.25, 0.75])


def test_density_grid_has_unit_mass_and_contour_threshold_is_positive() -> None:
    points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    grid = report.density_grid(
        points,
        np.array([0.25, 0.25, 0.5]),
        (-1.0, 2.0, -1.0, 2.0),
        bins=40,
        smooth_sigma=1.0,
    )
    assert grid.shape == (40, 40)
    assert grid.sum() == pytest.approx(1.0)
    threshold = report.enclosed_mass_threshold(grid, 0.8)
    assert threshold > 0.0
    assert grid[grid >= threshold].sum() >= 0.8


def test_cell_type_selection_is_train_only_count_then_name() -> None:
    labels = ["B", "A", "C", "B", "A", "C", "D"]
    assert report.choose_cell_types(labels, 3) == ["A", "B", "C"]


def test_shared_cell_type_selection_is_observed_ranked_and_requires_all_methods() -> None:
    selected, counts, thresholds = report.choose_shared_cell_types(
        [
            ["A"] * 8 + ["B"] * 6 + ["C"] * 4,
            ["A"] * 8 + ["B"] * 1 + ["C"] * 6,
            ["A"] * 7 + ["B"] * 7 + ["C"] * 5,
        ],
        2,
        minimum_count=4,
        minimum_fraction=0.0,
    )
    assert selected == ["A", "C"]
    assert counts["A"]["Observed held-out"] == 8
    assert counts["C"]["Bracket centroid-shift"] == 6
    assert thresholds == [4, 4, 4]


def test_prediction_validation_rejects_bad_weight_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        spatial=np.zeros((3, 2)),
        state=np.zeros((3, 4)),
        weights=np.ones(2),
    )
    with pytest.raises(report.ReportError, match="weights"):
        report.load_prediction(path)


def test_reader_guide_calls_linear_bracketed_and_programs_state_readouts() -> None:
    guide = report._reader_guide(
        [1.0, 2.0],
        {"Her response": ["her4.1", "her4.3"]},
        linear_method="linear_centroid_shift",
        cytobridge_method="CytoBridge-0.015",
        selected_by_target={"1": ["A"], "2": ["B"]},
    )
    assert "uses both flanking stages" in guide
    assert "strong **transductive interpolation**" in guide
    assert "not direct gene" in guide
    assert "shared display support" in guide


def test_tiny_end_to_end_report_writes_reader_figures_and_contract(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    times = np.repeat([0.0, 1.0, 2.0], 12)
    state = rng.normal(size=(36, 3)) + times[:, None] * 0.2
    spatial = rng.normal(scale=0.15, size=(36, 2)) + times[:, None] * np.array([0.4, 0.2])
    labels = np.tile(np.array(["A", "B", "C"]), 12)
    source = ad.AnnData(
        X=np.maximum(rng.normal(loc=1.0, scale=0.3, size=(36, 3)), 0.0),
        obs=pd.DataFrame(
            {"time_point_processed": times, "Annotation": labels},
            index=[f"cell_{index}" for index in range(36)],
        ),
        var=pd.DataFrame(index=["dla", "dld", "her4.1"]),
    )
    source.obsm["X_latent"] = state
    source.obsm["spatial_aligned"] = spatial
    source_path = tmp_path / "source.h5ad"
    source.write_h5ad(source_path)

    train = source[times != 1.0].copy()
    train.obs["benchmark_time"] = train.obs["time_point_processed"].to_numpy()
    train.obs["benchmark_annotation"] = train.obs["Annotation"].astype(str).to_numpy()
    train.obsm["benchmark_state"] = np.asarray(train.obsm["X_latent"])
    input_root = tmp_path / "inputs"
    (input_root / "loto_t1").mkdir(parents=True)
    train.write_h5ad(input_root / "loto_t1" / "train.h5ad")

    prediction_root = tmp_path / "predictions"
    for method, offset, weighted in (
        ("linear_centroid_shift", 0.03, False),
        ("CytoBridge-0.015", 0.01, True),
    ):
        directory = prediction_root / method / "t1"
        directory.mkdir(parents=True)
        payload = {
            "spatial": spatial[times == 1.0] + offset,
            "state": state[times == 1.0] + offset,
        }
        if weighted:
            payload["weights"] = np.linspace(1.0, 2.0, 12)
        np.savez_compressed(directory / "prediction.npz", **payload)
    programs = tmp_path / "programs.json"
    programs.write_text(json.dumps({"Delta": ["dla", "dld"]}), encoding="utf-8")
    output = tmp_path / "report"
    manifest = report.build_report(
        Namespace(
            source_h5ad=source_path,
            loto_input_root=input_root,
            prediction_root=prediction_root,
            output_dir=output,
            targets=[1.0],
            linear_method="linear_centroid_shift",
            cytobridge_method="CytoBridge-0.015",
            source_time_key="time_point_processed",
            source_spatial_key="spatial_aligned",
            source_state_key="X_latent",
            source_annotation_key="Annotation",
            train_state_key="benchmark_state",
            train_annotation_key="benchmark_annotation",
            cell_types=None,
            max_cell_types=2,
            min_cell_type_count=1,
            min_cell_type_fraction=0.0,
            programs_json=programs,
            knn_neighbors=3,
            grid_bins=36,
            smooth_sigma=1.2,
            enclosed_fraction=0.8,
        )
    )
    assert manifest["stages"][0]["target_physically_absent_from_train"] is True
    assert manifest["methods"]["linear"]["uses_left_and_right_anchors"] is True
    assert (output / "t1_01_tissue_density.png").is_file()
    assert (output / "t1_02_boundary_overlay.png").is_file()
    assert (output / "t1_03_cell_type_territories.png").is_file()
    assert (output / "t1_04_gene_program_readout.png").is_file()
    assert "transductive interpolation" in (output / "START_HERE.md").read_text()
