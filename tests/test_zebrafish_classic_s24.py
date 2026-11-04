"""Focused contracts for the classic unequal-population zebrafish S24 runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_zebrafish_classic_s24.py"
SPEC = importlib.util.spec_from_file_location("classic_s24", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixed_protocol_constants() -> None:
    np.testing.assert_array_equal(
        MODULE._time_grid(), np.linspace(0.0, 4.0, 81, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        MODULE._time_grid(3.0), np.linspace(0.0, 3.0, 61, dtype=np.float64)
    )
    assert MODULE.ABLATIONS == {
        "remove_YSL": ("Yolk Syncytial Layer",),
        "remove_EVL": ("EVL",),
    }
    assert MODULE.DT == 0.005
    assert MODULE.RESAMPLE_DT == 0.05
    assert MODULE.SIGMA == 0.03
    assert MODULE.GROWTH_ALPHA == 1.0
    assert MODULE.INTERACTION_M == 1024
    assert MODULE.MAX_PARTICLES == 100_000
    assert MODULE.TRAJECTORY_FILENAMES == {
        "baseline": "baseline_points.npy",
        "remove_YSL": "remove_YSL_points.npy",
        "remove_EVL": "remove_EVL_points.npy",
    }
    assert MODULE.PANEL_A_HEADER_Y == 0.985
    assert MODULE.PANEL_A_GRID_TOP == 0.85
    assert MODULE.COUNT_HEADROOM_MULTIPLIER == 1.12
    assert MODULE.MORPHOLOGY_COUNT_POSITION == (0.02, 0.02)


def test_trajectory_hashes_use_package_artifact_case(tmp_path: Path) -> None:
    for filename in MODULE.TRAJECTORY_FILENAMES.values():
        np.save(tmp_path / filename, np.asarray([1.0], dtype=np.float32))
    hashes = MODULE._trajectory_hashes(tmp_path)
    assert set(hashes) == {"baseline", "remove_YSL", "remove_EVL"}
    assert all(len(value) == 64 for value in hashes.values())


def test_support_time_grid_survives_formal_csv_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "support.csv"
    expected = MODULE._time_grid(3.0)
    pd.DataFrame({"time": expected}).to_csv(path, index=False, float_format="%.12g")
    actual = pd.read_csv(path)["time"].to_numpy(dtype=float)
    assert not np.array_equal(actual, expected)
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-12)


def _replay_frames() -> np.ndarray:
    frames = np.empty(2, dtype=object)
    frames[0] = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    frames[1] = np.asarray(
        [[0.1, 1.1, 2.1], [3.1, 4.1, 5.1], [6.1, 7.1, 8.1]],
        dtype=np.float32,
    )
    return frames


def _replay_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant": ["remove_YSL", "remove_EVL"] * 2,
            "time_index": [0, 0, 1, 1],
            "time": [0.0, 0.0, 0.05, 0.05],
            "space": ["spatial"] * 4,
            "w1": [0.1, 0.2, 0.11, 0.21],
            "w2": [0.12, 0.22, 0.13, 0.23],
            "centroid_shift": [0.01, 0.02, 0.011, 0.021],
            "n_baseline": [2, 2, 3, 3],
            "n_ablation": [2, 2, 3, 3],
            "count_ratio": [1.0, 1.0, 1.0, 1.0],
            "count_delta": [0, 0, 0, 0],
            "baseline_rms_radius": [1.0, 1.0, 1.1, 1.1],
            "ablation_rms_radius": [1.0, 1.0, 1.1, 1.1],
            "rms_radius_delta": [0.0, 0.0, 0.0, 0.0],
            "ot_baseline_points": [2, 2, 3, 3],
            "ot_ablation_points": [2, 2, 3, 3],
            "ot_random_seed": [42, 42, 43, 43],
        }
    )


def _write_replay_bundle(
    tmp_path: Path,
    *,
    state_delta: float = 0.0,
    shape_mismatch: bool = False,
    metric_delta: float = 0.0,
    discrete_delta: int = 0,
) -> tuple[Path, Path]:
    run_root = tmp_path / "run"
    replay_dir = tmp_path / "repeat"
    primary_experiment = run_root / "seeds" / "seed_42" / "experiment"
    replay_experiment = replay_dir / "experiment"
    primary_trajectories = primary_experiment / "trajectories"
    replay_trajectories = replay_experiment / "trajectories"
    primary_trajectories.mkdir(parents=True)
    replay_trajectories.mkdir(parents=True)
    for name, filename in MODULE.TRAJECTORY_FILENAMES.items():
        primary = _replay_frames()
        replay = _replay_frames()
        if name == "baseline":
            replay[1][0, 0] += np.float32(state_delta)
            if shape_mismatch:
                replay[1] = replay[1][:-1].copy()
        np.save(primary_trajectories / filename, primary)
        np.save(replay_trajectories / filename, replay)

    primary_metrics = _replay_metrics()
    replay_metrics = _replay_metrics()
    replay_metrics.loc[2, "time"] += 5e-13
    replay_metrics.loc[2, "w1"] += metric_delta
    replay_metrics.loc[2, "n_ablation"] += discrete_delta
    primary_metrics.to_csv(primary_experiment / "ablation_metrics.csv", index=False)
    replay_metrics.to_csv(replay_experiment / "ablation_metrics.csv", index=False)
    return run_root, replay_dir


def test_seed42_replay_accepts_tiny_gpu_drift_and_records_audit(
    tmp_path: Path,
) -> None:
    run_root, replay_dir = _write_replay_bundle(
        tmp_path, state_delta=2e-6, metric_delta=1e-10
    )
    audit = MODULE._require_seed42_replay(run_root, replay_dir)
    assert audit["status"] == "PASS"
    assert audit["trajectory_tolerance"] == {"rtol": 1e-6, "atol": 1e-5}
    assert audit["trajectories"]["baseline"]["byte_exact"] is False
    assert audit["trajectories"]["baseline"]["frame_shapes_exact"] is True
    assert audit["trajectories"]["baseline"]["particle_counts_exact"] is True
    assert audit["trajectories"]["baseline"]["max_abs_difference"] > 0.0
    assert audit["metrics"]["byte_exact"] is False
    assert audit["metrics"]["time"]["max_abs_difference"] <= 1e-12
    assert audit["metrics"]["numeric_columns"]["w1"]["max_abs_difference"] > 0


def test_seed42_replay_rejects_shape_or_particle_count_drift(tmp_path: Path) -> None:
    run_root, replay_dir = _write_replay_bundle(tmp_path, shape_mismatch=True)
    with pytest.raises(RuntimeError, match="shape/count differs"):
        MODULE._require_seed42_replay(run_root, replay_dir)


def test_seed42_replay_rejects_state_drift_beyond_tolerance(tmp_path: Path) -> None:
    run_root, replay_dir = _write_replay_bundle(tmp_path, state_delta=1e-3)
    with pytest.raises(RuntimeError, match="exceeds the numerical replay tolerance"):
        MODULE._require_seed42_replay(run_root, replay_dir)


def test_seed42_replay_rejects_continuous_metric_drift(tmp_path: Path) -> None:
    run_root, replay_dir = _write_replay_bundle(tmp_path, metric_delta=1e-4)
    with pytest.raises(RuntimeError, match="metric column 'w1' exceeds tolerance"):
        MODULE._require_seed42_replay(run_root, replay_dir)


def test_seed42_replay_rejects_discrete_metric_drift(tmp_path: Path) -> None:
    run_root, replay_dir = _write_replay_bundle(tmp_path, discrete_delta=1)
    with pytest.raises(
        RuntimeError, match="discrete metric column 'n_ablation' differs"
    ):
        MODULE._require_seed42_replay(run_root, replay_dir)


def test_direct_runner_binds_package_and_helper_to_same_release() -> None:
    MODULE._require_import_origins()
    assert Path(MODULE.cb.__file__).resolve().is_relative_to(ROOT)
    assert Path(MODULE.paper.__file__).resolve().is_relative_to(ROOT)


def test_labeler_rejects_main_classifier_with_same_feature_width(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wrong = SimpleNamespace(
        metadata={"cache_tag": "zebrafish-paper-main-spatial2-latent10"},
        feature_dim=12,
        include_time_feature=True,
        label_col="Annotation",
    )
    monkeypatch.setattr(
        MODULE.cb.tl, "load_cached_mlp_classifier", lambda *_args, **_kwargs: wrong
    )
    with pytest.raises(RuntimeError, match="formal ablation classifier"):
        MODULE._load_trajectory_labeler(
            tmp_path / "wrong.pt", tmp_path / "pca.npz", device="cpu"
        )


def _valid_result() -> SimpleNamespace:
    settings = {
        "mass_control": False,
        "growth_alpha": 1.0,
        "dt": 0.005,
        "resample_dt": 0.05,
        "sigma": 0.03,
        "interaction_m": 1024,
        "max_particles": 100_000,
        "random_seed": 42,
        "interaction_seed": 10_043,
        "common_random_seed": True,
        "n_initial": 563,
        "variant_initial_counts": {"remove_YSL": 534, "remove_EVL": 291},
    }
    points = np.empty(81, dtype=object)
    for index in range(81):
        points[index] = np.zeros((2, 52), dtype=np.float32)
    return SimpleNamespace(
        time_points=tuple(MODULE._time_grid()),
        settings=settings,
        ablation_points={"remove_YSL": points.copy(), "remove_EVL": points.copy()},
    )


def test_result_contract_accepts_exact_unequal_n_protocol() -> None:
    MODULE._validate_result_contract(_valid_result(), seed=42)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("mass_control", True),
        ("growth_alpha", 0.0),
        ("sigma", 0.0),
        ("dt", 0.05),
        ("resample_dt", 0.005),
        ("interaction_seed", 42),
    ],
)
def test_result_contract_rejects_estimand_drift(field: str, bad_value: object) -> None:
    result = _valid_result()
    result.settings[field] = bad_value
    with pytest.raises(RuntimeError, match=field):
        MODULE._validate_result_contract(result, seed=42)


def test_result_contract_rejects_equal_n_initialization() -> None:
    result = _valid_result()
    result.settings["variant_initial_counts"] = {
        "remove_YSL": 534,
        "remove_EVL": 563,
    }
    with pytest.raises(RuntimeError, match="full 563-cell baseline"):
        MODULE._validate_result_contract(result, seed=42)


def _audit(*, fail_t4: bool = False, fail_t3: bool = False) -> pd.DataFrame:
    rows = []
    for condition in ("baseline", "remove_YSL", "remove_EVL"):
        for time in MODULE._time_grid():
            passed = not (
                (fail_t4 and np.isclose(time, 4.0))
                or (fail_t3 and np.isclose(time, 3.0))
            )
            rows.append(
                {
                    "condition": condition,
                    "time": time,
                    "passes_publication_support_gate": passed,
                }
            )
    return pd.DataFrame(rows)


def test_latest_common_endpoint_uses_t4_only_when_all_15_pass() -> None:
    runs = {seed: {"audit": _audit()} for seed in MODULE.FORMAL_SEEDS}
    assert MODULE._latest_common_endpoint(runs) == 4.0
    runs[46] = {"audit": _audit(fail_t4=True)}
    assert MODULE._latest_common_endpoint(runs) == 3.0


def test_latest_common_endpoint_does_not_cherry_pick_seed_or_branch() -> None:
    runs = {seed: {"audit": _audit(fail_t4=True)} for seed in MODULE.FORMAL_SEEDS}
    rows = runs[44]["audit"]
    rows.loc[
        rows["condition"].eq("remove_EVL") & np.isclose(rows["time"], 3.0),
        "passes_publication_support_gate",
    ] = False
    assert MODULE._latest_common_endpoint(runs) == 2.0


def test_latest_common_endpoint_requires_every_prior_frame_to_pass() -> None:
    rows = []
    for condition in ("baseline", "remove_YSL", "remove_EVL"):
        for time in MODULE._time_grid():
            rows.append(
                {
                    "condition": condition,
                    "time": time,
                    "passes_publication_support_gate": not (
                        condition == "baseline" and np.isclose(time, 3.8)
                    ),
                }
            )
    dense = pd.DataFrame(rows)
    runs = {seed: {"audit": dense.copy()} for seed in MODULE.FORMAL_SEEDS}
    assert MODULE._latest_common_endpoint(runs) == 3.0


def test_cli_requires_canonical_seed_set(tmp_path: Path) -> None:
    args = SimpleNamespace(seed=41, output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="must be one of"):
        MODULE.run_seed(args)


def test_parser_has_parallel_seed_and_report_commands(tmp_path: Path) -> None:
    parser = MODULE.build_parser()
    seed = parser.parse_args(
        [
            "run-seed",
            "--aligned-h5ad",
            str(tmp_path / "aligned.h5ad"),
            "--model-dir",
            str(tmp_path / "model"),
            "--acceptance-report",
            str(tmp_path / "acceptance.json"),
            "--expected-acceptance-sha256",
            "a" * 64,
            "--seed",
            "42",
            "--end-time",
            "3",
            "--output-dir",
            str(tmp_path / "seed42"),
        ]
    )
    assert seed.command == "run-seed"
    assert seed.seed == 42
    report = parser.parse_args(
        [
            "report",
            "--run-root",
            str(tmp_path / "run"),
            "--seed42-repeat",
            str(tmp_path / "repeat"),
            "--classifier-cache",
            str(tmp_path / "classifier.pt"),
            "--classifier-pca",
            str(tmp_path / "classifier_pca.npz"),
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )
    assert report.command == "report"


def test_report_rejects_primary_seed_as_fake_replay(tmp_path: Path) -> None:
    primary = tmp_path / "run" / "seeds" / "seed_42"
    primary.mkdir(parents=True)
    args = SimpleNamespace(
        run_root=tmp_path / "run",
        seed42_repeat=primary,
        classifier_cache=tmp_path / "classifier.pt",
        classifier_pca=tmp_path / "classifier_pca.npz",
        classifier_device="cpu",
        output_dir=tmp_path / "report",
    )
    # Avoid creating an output directory before the replay-path contract is checked.
    with pytest.raises(ValueError, match="independently executed"):
        MODULE.report(args)


def test_quantitative_plot_uses_layered_zero_based_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectories = {}
    for name, scale in (("baseline", 1.0), ("remove_YSL", 1.1), ("remove_EVL", 1.2)):
        frames = np.empty(81, dtype=object)
        for index in range(81):
            frames[index] = np.asarray(
                [[-scale, -scale, 0.0], [scale, scale, 0.0]], dtype=np.float32
            )
        trajectories[name] = frames
    rows = []
    for seed in MODULE.FORMAL_SEEDS:
        for variant in ("remove_YSL", "remove_EVL"):
            for index, time in enumerate(MODULE._time_grid()):
                rows.append(
                    {
                        "variant": variant,
                        "time_index": index,
                        "time": time,
                        "space": "spatial",
                        "w1": 0.1 + time * 0.01,
                        "w2": 0.2 + time * 0.01,
                        "centroid_shift": 0.05 + time * 0.01,
                        "n_baseline": 2,
                        "n_ablation": 2,
                        "count_ratio": 1.0,
                    }
                )
    frame = pd.DataFrame(rows)
    runs = {
        seed: {
            "trajectories": trajectories,
            "metrics": frame.loc[frame.index % len(MODULE.FORMAL_SEEDS) == 0].copy(),
        }
        for seed in MODULE.FORMAL_SEEDS
    }
    # Give each seed one complete metric table; numeric identity is fine here.
    per_seed = frame.iloc[: 2 * 81].copy()
    for run in runs.values():
        run["metrics"] = per_seed.copy()
    captured: dict[str, object] = {}
    original_savefig = Figure.savefig

    def inspect_layout(figure: Figure, *args, **kwargs):
        if not captured:
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            header = next(
                text
                for text in figure.texts
                if text.get_text().startswith("Spatial distributions at t")
            )
            header_box = header.get_window_extent(renderer=renderer)
            title_boxes = [
                axis.title.get_window_extent(renderer=renderer)
                for axis in figure.axes[:3]
            ]
            ax_w1, ax_count = figure.axes[3], figure.axes[4]
            captured.update(
                {
                    "header_y": header.get_position()[1],
                    "subplot_top": figure.subplotpars.top,
                    "header_collision": any(
                        header_box.overlaps(title_box) for title_box in title_boxes
                    ),
                    "snapshot_xlim": figure.axes[0].get_xlim(),
                    "w1_ylim": ax_w1.get_ylim(),
                    "count_ylim": ax_count.get_ylim(),
                    "count_line_max": max(
                        float(np.max(line.get_ydata())) for line in ax_count.lines
                    ),
                    "count_title": ax_count.get_title(loc="left"),
                    "count_ylabel": ax_count.get_ylabel(),
                }
            )
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_layout)
    stem = tmp_path / "figure"
    MODULE._plot_quantitative(runs, 4.0, stem)
    assert stem.with_suffix(".pdf").stat().st_size > 0
    assert stem.with_suffix(".png").stat().st_size > 0
    assert captured["header_y"] == pytest.approx(MODULE.PANEL_A_HEADER_Y)
    assert captured["subplot_top"] == pytest.approx(MODULE.PANEL_A_GRID_TOP)
    assert captured["header_collision"] is False
    assert captured["snapshot_xlim"][0] < -1.2
    assert captured["snapshot_xlim"][1] > 1.2
    assert captured["w1_ylim"][0] == pytest.approx(0.0)
    assert captured["count_ylim"][0] == pytest.approx(0.0)
    assert captured["count_ylim"][1] == pytest.approx(
        captured["count_line_max"] * MODULE.COUNT_HEADROOM_MULTIPLIER
    )
    assert captured["count_title"] == "Growth-resampled particle count"
    assert captured["count_ylabel"] == "Simulated particle count"


def test_morphology_grid_uses_uniform_count_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectories = {}
    for name, scale in (("baseline", 1.0), ("remove_YSL", 1.1), ("remove_EVL", 1.2)):
        frames = np.empty(61, dtype=object)
        for index in range(61):
            frames[index] = np.asarray(
                [[-scale, -scale, 0.0], [scale, scale, 0.0]], dtype=np.float32
            )
        trajectories[name] = frames
    runs = {seed: {"trajectories": trajectories.copy()} for seed in MODULE.FORMAL_SEEDS}
    captured: dict[str, object] = {}
    original_savefig = Figure.savefig

    def inspect_layout(figure: Figure, *args, **kwargs):
        if not captured:
            count_positions = []
            time_positions = []
            time_axis_indices = []
            for axis_index, axis in enumerate(figure.axes):
                count_texts = [
                    text for text in axis.texts if text.get_text().startswith("n = ")
                ]
                assert len(count_texts) == 1
                count_positions.append(count_texts[0].get_position())
                time_texts = [
                    text for text in axis.texts if text.get_text().startswith("t = ")
                ]
                if time_texts:
                    assert len(time_texts) == 1
                    time_positions.append(time_texts[0].get_position())
                    time_axis_indices.append(axis_index)
            captured.update(
                {
                    "count_positions": count_positions,
                    "time_positions": time_positions,
                    "time_axis_indices": time_axis_indices,
                }
            )
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_layout)
    stem = tmp_path / "morphology"
    MODULE._plot_time_grid(runs, 3.0, stem)
    assert stem.with_suffix(".pdf").stat().st_size > 0
    assert stem.with_suffix(".png").stat().st_size > 0
    assert captured["count_positions"] == [MODULE.MORPHOLOGY_COUNT_POSITION] * 12
    assert captured["time_positions"] == [(-0.02, 0.5)] * 4
    assert captured["time_axis_indices"] == [0, 3, 6, 9]


def test_report_caption_disambiguates_classifier_and_simulated_counts() -> None:
    caption = MODULE._report_caption(3.0)
    assert caption["figure_roles"] == MODULE.FIGURE_ROLES
    assert "classifier-assigned" in caption["morphology_color_interpretation"]
    assert "not lineage identities" in caption["morphology_color_interpretation"]
    assert (
        "not observed biological abundance" in caption["population_axis_interpretation"]
    )
    assert "growth-resampled simulated particle count" in caption["text"]
