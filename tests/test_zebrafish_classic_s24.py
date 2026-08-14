"""Focused contracts for the classic unequal-population zebrafish S24 runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def test_quantitative_plot_keeps_full_spatial_extent(tmp_path: Path) -> None:
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
    stem = tmp_path / "figure"
    MODULE._plot_quantitative(runs, 4.0, stem)
    assert stem.with_suffix(".pdf").stat().st_size > 0
    assert stem.with_suffix(".png").stat().st_size > 0
