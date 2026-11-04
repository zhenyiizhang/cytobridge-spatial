from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "reviewer_zebrafish_response"
    / "virtual_ablation_wasserstein.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reviewer_virtual_ablation_wasserstein", SCRIPT
)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_toy_trajectories(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    times = np.asarray([0.0, 0.5, 1.0], dtype=float)
    baseline_frame = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 1.0]], dtype=np.float32
    )
    baseline = np.stack([baseline_frame] * len(times))
    ysl = np.stack(
        [
            baseline_frame + np.asarray([time, 0.0, 0.0], dtype=np.float32)
            for time in times
        ]
    )
    evl = np.stack(
        [
            baseline_frame
            + np.asarray([2.0 * time, 0.0, 0.0], dtype=np.float32)
            for time in times
        ]
    )

    baseline_path = tmp_path / "baseline_points.npy"
    ysl_path = tmp_path / "remove_YSL_points.npy"
    evl_path = tmp_path / "remove_EVL_points.npy"
    time_path = tmp_path / "time_points.npy"
    np.save(baseline_path, baseline)
    np.save(ysl_path, ysl)
    np.save(evl_path, evl)
    np.save(time_path, times)
    return baseline_path, ysl_path, evl_path, time_path


def test_cli_writes_metrics_summary_plot_and_hash_manifest(tmp_path: Path) -> None:
    baseline, ysl, evl, times = _save_toy_trajectories(tmp_path)
    output = tmp_path / "reviewer_output"
    argv = [
        "--baseline-points",
        str(baseline),
        "--variant",
        f"remove_YSL={ysl}",
        "--variant",
        f"remove_EVL={evl}",
        "--time-grid",
        str(times),
        "--output-dir",
        str(output),
        "--spatial-dim",
        "2",
        "--max-ot-points",
        "none",
        "--random-seed",
        "17",
    ]

    assert analysis.main(argv) == 0

    metrics_path = output / analysis.METRICS_FILENAME
    summary_path = output / analysis.SUMMARY_FILENAME
    figure_path = output / analysis.FIGURE_FILENAME
    manifest_path = output / analysis.MANIFEST_FILENAME
    assert all(
        path.is_file()
        for path in (metrics_path, summary_path, figure_path, manifest_path)
    )

    metrics = pd.read_csv(metrics_path)
    assert len(metrics) == 2 * 3 * 3
    assert set(metrics["variant"]) == {"remove_YSL", "remove_EVL"}
    assert set(metrics["space"]) == {"joint", "spatial", "latent"}
    assert {"w1", "w2", "ot_ablation_points", "ot_baseline_points"}.issubset(
        metrics.columns
    )
    ysl_endpoint = metrics.query(
        "variant == 'remove_YSL' and space == 'spatial' and time == 1.0"
    ).iloc[0]
    evl_endpoint = metrics.query(
        "variant == 'remove_EVL' and space == 'spatial' and time == 1.0"
    ).iloc[0]
    assert ysl_endpoint["w1"] == pytest.approx(1.0)
    assert ysl_endpoint["w2"] == pytest.approx(1.0)
    assert evl_endpoint["w1"] == pytest.approx(2.0)
    assert evl_endpoint["w2"] == pytest.approx(2.0)

    summary = pd.read_csv(summary_path)
    assert len(summary) == 2 * 3 * 2
    ysl_w1 = summary.query(
        "variant == 'remove_YSL' and space == 'spatial' and metric == 'w1'"
    ).iloc[0]
    evl_w2 = summary.query(
        "variant == 'remove_EVL' and space == 'spatial' and metric == 'w2'"
    ).iloc[0]
    assert ysl_w1["value_t0"] == pytest.approx(0.0)
    assert ysl_w1["value_endpoint"] == pytest.approx(1.0)
    assert ysl_w1["endpoint_change_from_t0"] == pytest.approx(1.0)
    assert ysl_w1["auc"] == pytest.approx(0.5)
    assert ysl_w1["auc_change_from_t0"] == pytest.approx(0.5)
    assert evl_w2["value_endpoint"] == pytest.approx(2.0)
    assert evl_w2["auc"] == pytest.approx(1.0)

    with Image.open(figure_path) as image:
        assert image.format == "PNG"
        assert image.width >= 1000
        assert image.height >= 400

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["analysis"] == (
        "dataset_agnostic_virtual_ablation_wasserstein"
    )
    assert manifest["parameters"]["variant_order"] == [
        "remove_YSL",
        "remove_EVL",
    ]
    assert manifest["parameters"]["max_ot_points"] is None
    assert manifest["inputs"]["baseline_points"]["sha256"] == _sha256(baseline)
    assert manifest["inputs"]["variants"]["remove_YSL"]["sha256"] == _sha256(
        ysl
    )
    assert manifest["inputs"]["time_grid"]["file"]["sha256"] == _sha256(times)
    assert len(
        manifest["inputs"]["time_grid"]["values_sha256_float64_le"]
    ) == 64
    git_commit = manifest["code"]["git"]["commit"]
    # Source distributions intentionally do not contain a .git directory.
    # Reporting that boundary as null is more accurate than inventing an ID.
    assert git_commit is None or len(git_commit) == 40
    git_dirty = manifest["code"]["git"]["dirty"]
    assert git_dirty is None or isinstance(git_dirty, bool)
    assert "--variant" in manifest["command"]["argv"]
    for key, path in (
        ("metrics", metrics_path),
        ("summary", summary_path),
        ("spatial_time_curves", figure_path),
    ):
        assert manifest["outputs"][key]["sha256"] == _sha256(path)
    assert manifest["manifest"]["self_hash_omitted"] is True


def test_regular_time_grid_and_variant_validation(tmp_path: Path) -> None:
    baseline, ysl, _, _ = _save_toy_trajectories(tmp_path)
    parser = analysis.build_parser()
    args = parser.parse_args(
        [
            "--baseline",
            str(baseline),
            "--variant",
            f"remove_YSL={ysl}",
            "--time-start",
            "0",
            "--time-stop",
            "1",
            "--time-step",
            "0.5",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    values, source = analysis._resolve_time_grid(args)
    np.testing.assert_allclose(values, [0.0, 0.5, 1.0])
    assert source["kind"] == "regular_cli"
    assert len(source["values_sha256_float64_le"]) == 64

    with pytest.raises(ValueError, match="Duplicate"):
        analysis._parse_variant_specs(
            [f"remove_YSL={ysl}", f"remove_YSL={ysl}"]
        )
    with pytest.raises(ValueError, match="NAME=PATH"):
        analysis._parse_variant_specs([str(ysl)])


def test_cli_rejects_trajectory_time_length_mismatch(tmp_path: Path) -> None:
    baseline, ysl, _, _ = _save_toy_trajectories(tmp_path)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="length mismatch"):
        analysis.main(
            [
                "--baseline",
                str(baseline),
                "--variant",
                f"remove_YSL={ysl}",
                "--times",
                "0,1",
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()
