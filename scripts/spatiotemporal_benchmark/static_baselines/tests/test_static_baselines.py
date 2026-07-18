from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.spatiotemporal_benchmark.static_baselines.coupling import (
    CouplingDiagnostics,
    compose_row_plans,
    validate_and_row_normalize,
)
from scripts.spatiotemporal_benchmark.static_baselines.data import (
    InputKeys,
    _ranked_indices,
    build_source_roster,
    load_trajectory,
)
from scripts.spatiotemporal_benchmark.build_inputs import _ranked_support_indices
from scripts.spatiotemporal_benchmark.static_baselines.errors import (
    LeakageError,
    OfficialAPIError,
)
from scripts.spatiotemporal_benchmark.static_baselines.methods import fit_block_balance
from scripts.spatiotemporal_benchmark.static_baselines.registry import list_method_specs


run_module = importlib.import_module("scripts.spatiotemporal_benchmark.static_baselines.run")
methods_module = importlib.import_module(
    "scripts.spatiotemporal_benchmark.static_baselines.methods"
)


def write_fixture(
    path: Path,
    *,
    remove_time: int | None = None,
    prediction_n: int = 11,
    negative_expression: bool = False,
    builder_preprocess_proof: bool = False,
) -> Path:
    rng = np.random.default_rng(12)
    times = np.repeat(np.arange(5), [3, 4, 5, 4, 3])
    keep = np.ones(len(times), dtype=bool) if remove_time is None else times != remove_time
    times = times[keep]
    expression = rng.uniform(0.0, 2.0, size=(len(times), 4)).astype(np.float32)
    if negative_expression:
        expression[0, 0] = -1.0
    obs = pd.DataFrame(
        {
            "benchmark_time": times.astype(float),
            "row_id": [f"row_{index:03d}" for index in np.flatnonzero(keep)],
        },
        index=[f"cell_{index:03d}" for index in np.flatnonzero(keep)],
    )
    data = ad.AnnData(X=expression, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(4)]))
    # Nontrivial deterministic coordinates make composition assertions easier.
    data.obsm["benchmark_state"] = np.column_stack(
        [times + rng.normal(scale=0.01, size=len(times)), times**2, rng.normal(size=len(times))]
    ).astype(np.float32)
    data.obsm["benchmark_spatial"] = np.column_stack(
        [times * 2 + rng.normal(scale=0.01, size=len(times)), rng.normal(size=len(times))]
    ).astype(np.float32)
    data.uns["cytobridge_benchmark_contract"] = {
        "dataset_id": "fixture",
        "state_key": "benchmark_state",
        "spatial_key": "benchmark_spatial",
        "time_key": "benchmark_time",
        "row_id_key": "row_id",
        "expression_key": "X",
        "prediction_n": int(prediction_n),
        "source_roster_support_n": 20,
        "source_roster_seed": 20260718,
        "time_values": [0, 1, 2, 3, 4],
        "loto_targets": [1, 2, 3],
        "full_data_targets": [1, 2, 3, 4],
        "target_removed": remove_time is not None,
        "held_out_benchmark_time": "none" if remove_time is None else int(remove_time),
    }
    if builder_preprocess_proof:
        data.uns["cytobridge_benchmark_contract"]["preprocess_provenance_contract_passed"] = True
        data.uns["preprocess_info"] = {
            "transformation_sequence": ["normalize_total", "log1p"]
        }
    else:
        data.uns["cytobridge_benchmark_contract"][
            "expression_semantics"
        ] = "once_log_normalized_nonnegative"
    data.write_h5ad(path)
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    path: Path,
    train: Path,
    split: str,
    training_reference: Path | None = None,
) -> Path:
    train_record: dict[str, object] = {"h5ad": {"sha256": sha256(train)}}
    if training_reference is not None:
        train_record["training_reference_npz"] = {
            "path": str(training_reference),
            "sha256": sha256(training_reference),
        }
    data = ad.read_h5ad(train)
    contract = data.uns["cytobridge_benchmark_contract"]
    target = int(split.removeprefix("loto_t")) if split.startswith("loto_t") else None
    times = data.obs["benchmark_time"].to_numpy(dtype=float)
    source_time = float(np.min(times)) if target is None else float(
        max(value for value in np.unique(times) if value < target)
    )
    candidates = np.flatnonzero(np.isclose(times, source_time))
    roster_indices = np.resize(candidates, int(contract["prediction_n"]))
    roster = path.with_name(f"{split}_source_roster.npz")
    np.savez_compressed(
        roster,
        indices=roster_indices,
        row_id=data.obs["row_id"].astype(str).to_numpy(dtype=str)[roster_indices],
        source_time=np.asarray([source_time]),
        spatial=np.asarray(data.obsm["benchmark_spatial"])[roster_indices],
        state=np.asarray(data.obsm["benchmark_state"])[roster_indices],
    )
    train_record["source_roster_npz"] = {
        "path": str(roster),
        "sha256": sha256(roster),
    }
    path.write_text(
        json.dumps(
            {
                "dataset_id": "fixture",
                "splits": {split: {"train": train_record}},
            }
        ),
        encoding="utf-8",
    )
    return path


def common_args(input_path: Path, output: Path, mode: str, target: int | None = None) -> list[str]:
    args = [
        "run",
        "--input-h5ad",
        str(input_path),
        "--evaluation-mode",
        mode,
        "--output-dir",
        str(output),
        "--max-fit-n",
        "20",
    ]
    if target is not None:
        args += ["--target-time", str(target)]
    return args


def test_registry_exact_methods_and_scopes() -> None:
    specs = list_method_specs()
    assert set(specs) == {
        "moscot",
        "wot",
        "paste",
        "spateo",
        "spatrack",
        "linear_centroid_shift",
        "random_independent_pairs",
    }
    assert specs["wot"]["representations"]["matched_state_spatial"]["output_scope"] == "native_state"
    assert specs["paste"]["representations"]["matched_state_spatial"]["hybrid"] is True
    assert specs["spateo"]["representations"]["matched_state_spatial"]["hybrid"] is True
    assert specs["spatrack"]["representations"]["matched_state_spatial"]["applicable"] is False
    assert specs["spatrack"]["representations"]["native_gene_sensitivity"]["applicable"] is True


def test_loto_rejects_target_rows_before_method_fit(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "full.h5ad")
    with pytest.raises(LeakageError, match="target_removed|contains .* held-out rows"):
        load_trajectory(
            path,
            mode="loto",
            target_time=2,
            max_fit_n=20,
            seed=7,
            keys=InputKeys(),
        )


def test_loto_uses_target_removed_brackets_and_contract_prediction_n(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "loto.h5ad", remove_time=2, prediction_n=13)
    data = load_trajectory(path, mode="loto", target_time=2, max_fit_n=20, seed=7)
    pair = data.loto_pair()
    assert (pair.previous.time, pair.following.time, pair.interpolation_alpha) == (1.0, 3.0, 0.5)
    assert data.prediction_n == 13
    assert all(stage.time != 2 for stage in data.stages)


def test_no_holdout_returns_all_adjacent_pairs_and_full_t0_anchor(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "full.h5ad")
    data = load_trajectory(path, mode="no-holdout", target_time=None, max_fit_n=800, seed=7)
    assert [(pair.previous.time, pair.following.time) for pair in data.adjacent_pairs()] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
    ]
    assert data.stage(0).n_obs == 3


def test_source_roster_is_train_only_method_independent_and_fixed_size(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "full.h5ad", prediction_n=17)
    data = load_trajectory(path, mode="no-holdout", target_time=None, max_fit_n=20, seed=7)
    first_indices, first_ids = build_source_roster(data.stage(0), data.prediction_n, 91)
    second_indices, second_ids = build_source_roster(data.stage(0), data.prediction_n, 91)
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_ids, second_ids)
    assert len(first_indices) == 17
    assert set(first_ids).issubset(set(data.stage(0).row_ids))


def test_spateo_signed_pc_profile_disables_nn_initializer() -> None:
    spec = list_method_specs()["spateo"]
    params = spec["representations"]["matched_state_spatial"]["default_parameters"]
    assert params["nn_init"] is False


def test_static_anchor_ranking_matches_builder_for_integral_float_time() -> None:
    row_ids = np.asarray([f"row-{index:04d}" for index in range(1000)], dtype=str)
    builder = _ranked_support_indices(row_ids, 800, 20260718, 2)
    static = _ranked_indices(row_ids, 800, 20260718, 2.0)
    np.testing.assert_array_equal(static, builder)


def test_moscot_block_balance_is_fit_on_all_training_anchors(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "full.h5ad")
    data = load_trajectory(path, mode="no-holdout", target_time=None, max_fit_n=20, seed=7)
    transform = fit_block_balance(data.stages)
    transformed = np.vstack([transform.transform(stage) for stage in data.stages])
    state_dim = data.stage(0).state_pca.shape[1]
    state_energy = np.mean(np.sum(transformed[:, :state_dim] ** 2, axis=1))
    spatial_energy = np.mean(np.sum(transformed[:, state_dim:] ** 2, axis=1))
    assert transform.fitted_rows == sum(stage.n_obs for stage in data.stages)
    assert state_energy == pytest.approx(0.5, rel=1e-5)
    assert spatial_energy == pytest.approx(0.5, rel=1e-5)
    assert transform.manifest()["truth_rows_used"] == 0


def test_composition_is_p01_p12_not_direct_target_resample() -> None:
    p01 = np.array([[0.8, 0.2], [0.1, 0.9]])
    p12 = np.array([[0.3, 0.7], [0.6, 0.4]])
    composed, history = compose_row_plans([p01, p12])
    np.testing.assert_allclose(composed, p01 @ p12)
    assert len(history) == 2
    assert np.allclose(composed.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "bad,shape,message",
    [
        (np.ones((3, 2)), (2, 3), "Orientation is never guessed"),
        (np.array([[1.0, 0.0], [0.0, 0.0]]), (2, 2), "zero-mass"),
        (np.array([[1.0, -1.0], [0.5, 0.5]]), (2, 2), "negative"),
    ],
)
def test_invalid_couplings_fail_closed(bad: np.ndarray, shape: tuple[int, int], message: str) -> None:
    with pytest.raises(OfficialAPIError, match=message):
        validate_and_row_normalize(bad, shape)


def test_no_holdout_linear_writes_t1_to_t4_from_one_roster(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "full.h5ad", prediction_n=9)
    output = tmp_path / "linear"
    code = run_module.main(
        [
            *common_args(path, output, "no-holdout"),
            "--method",
            "linear_centroid_shift",
        ]
    )
    assert code == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["protocol"]["direct_previous_to_target_alpha_one_used"] is False
    assert set(manifest["outputs"]["prediction_by_time"]) == {"1.0", "2.0", "3.0", "4.0"}
    with np.load(output / "trajectory_prediction.npz", allow_pickle=False) as values:
        assert values["points"].shape == (36, 5)
        assert np.array_equal(np.unique(values["time"]), [1, 2, 3, 4])
    with np.load(output / "source_roster.npz", allow_pickle=False) as roster:
        assert len(roster["source_row_ids"]) == 9


def test_no_holdout_random_uses_all_adjacent_composed_plans(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "full.h5ad", prediction_n=7)
    output = tmp_path / "random"
    code = run_module.main(
        [
            *common_args(path, output, "full-data"),
            "--method",
            "random_independent_pairs",
            "--save-coupling",
        ]
    )
    assert code == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["protocol"]["mode"] == "no-holdout"
    assert "P01" in manifest["control_run"]["targets"]["1.0"]["formula"]
    assert manifest["control_run"]["targets"]["4.0"]["formula"] == "P01 @ P12 @ P23 @ P34"
    assert manifest["outputs"]["couplings"]["shapes"] == {
        "P_0_1": [3, 4],
        "P_1_2": [4, 5],
        "P_2_3": [5, 4],
        "P_3_4": [4, 3],
    }


def _mock_coupling(method_name, pair, representation, **kwargs):
    plan = np.full((pair.previous.n_obs, pair.following.n_obs), 1.0 / pair.following.n_obs)
    diagnostics = CouplingDiagnostics(
        shape=plan.shape,
        total_mass=float(plan.sum()),
        row_sum_min=1.0,
        row_sum_max=1.0,
        zero_rows=0,
    )
    return plan, diagnostics, {
        "dependency": {"available": True, "version": "mock", "git_commit": "abc"},
        "official_api": f"mock.{method_name}",
    }


def test_mocked_official_api_no_holdout_is_called_for_four_edges(tmp_path: Path, monkeypatch) -> None:
    path = write_fixture(tmp_path / "full.h5ad", prediction_n=6)
    output = tmp_path / "paste"
    calls: list[tuple[float, float]] = []

    def recording_mock(method_name, pair, representation, **kwargs):
        calls.append((pair.previous.time, pair.following.time))
        return _mock_coupling(method_name, pair, representation, **kwargs)

    monkeypatch.setattr(run_module, "run_official_coupling", recording_mock)
    code = run_module.main(
        [*common_args(path, output, "no-holdout"), "--method", "paste"]
    )
    assert code == 0
    assert calls == [(0, 1), (1, 2), (2, 3), (3, 4)]
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["composition"]["targets"]["4.0"]["formula"] == "P01 @ P12 @ P23 @ P34"
    assert manifest["output_scope"]["hybrid_adapter"] is True


def test_official_paste_api_mock_receives_signed_pc_euclidean_inputs(tmp_path: Path, monkeypatch) -> None:
    path = write_fixture(tmp_path / "loto.h5ad", remove_time=2)
    data = load_trajectory(path, mode="loto", target_time=2, max_fit_n=20, seed=7)
    pair = data.loto_pair()
    observed: dict[str, object] = {}

    def pairwise_align(previous, following, **kwargs):
        observed["left"] = np.asarray(previous.X)
        observed["right"] = np.asarray(following.X)
        observed["kwargs"] = kwargs
        return np.ones((previous.n_obs, following.n_obs), dtype=float)

    fake = SimpleNamespace(pairwise_align=pairwise_align, __name__="paste", __file__="/mock/paste.py")
    monkeypatch.setattr(
        methods_module,
        "import_official",
        lambda method, source_root: (fake, {"available": True, "version": "mock"}),
    )
    plan, diagnostics, metadata = methods_module.run_official_coupling(
        "paste", pair, "matched_state_spatial"
    )
    np.testing.assert_allclose(observed["left"], pair.previous.state_pca)
    assert observed["kwargs"]["dissimilarity"] == "euclidean"
    assert plan.shape == (pair.previous.n_obs, pair.following.n_obs)
    assert diagnostics.zero_rows == 0
    assert metadata["dependency"]["version"] == "mock"


def test_mocked_wot_emits_state_only_and_no_spatial(tmp_path: Path, monkeypatch) -> None:
    path = write_fixture(tmp_path / "loto.h5ad", remove_time=2, prediction_n=5)
    output = tmp_path / "wot"
    monkeypatch.setattr(run_module, "run_official_coupling", _mock_coupling)
    code = run_module.main(
        [*common_args(path, output, "loto", 2), "--method", "wot"]
    )
    assert code == 0
    with np.load(output / "prediction.npz", allow_pickle=False) as values:
        assert "state" in values
        assert "spatial" not in values
        assert "points" not in values
        assert values["state"].shape == (5, 3)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["output_scope"]["scope"] == "native_state"
    assert manifest["output_scope"]["spatial_dimensions"] == 0


def test_matched_spatrack_is_na_without_dependency_or_prediction(tmp_path: Path, monkeypatch) -> None:
    path = write_fixture(tmp_path / "full.h5ad")
    output = tmp_path / "spatrack_na"

    def forbidden(*args, **kwargs):
        raise AssertionError("dependency must not be imported for N/A representation")

    monkeypatch.setattr(run_module, "run_official_coupling", forbidden)
    code = run_module.main(
        [*common_args(path, output, "no-holdout"), "--method", "spatrack"]
    )
    assert code == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "not_applicable"
    assert not (output / "prediction.npz").exists()


def test_spatrack_native_sensitivity_rejects_negative_expression(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "bad.h5ad", negative_expression=True)
    output = tmp_path / "spatrack_bad"
    code = run_module.main(
        [
            *common_args(path, output, "no-holdout"),
            "--method",
            "spatrack",
            "--representation",
            "native_gene_sensitivity",
        ]
    )
    assert code == 2
    failure = json.loads((output / "failure_manifest.json").read_text())
    assert "nonnegative" in failure["error"]
    assert failure["surrogate_attempted"] is False


def test_builder_verified_once_log_metadata_allows_spatrack_sensitivity(
    tmp_path: Path, monkeypatch
) -> None:
    path = write_fixture(
        tmp_path / "builder_style.h5ad",
        remove_time=2,
        prediction_n=5,
        builder_preprocess_proof=True,
    )
    output = tmp_path / "spatrack_builder_proof"
    monkeypatch.setattr(run_module, "run_official_coupling", _mock_coupling)
    code = run_module.main(
        [
            *common_args(path, output, "loto", 2),
            "--method",
            "spatrack",
            "--representation",
            "native_gene_sensitivity",
        ]
    )
    assert code == 0
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["primary_benchmark_eligible"] is False
    assert manifest["output_scope"]["hybrid_adapter"] is True


def test_official_failure_writes_failure_manifest_without_surrogate(tmp_path: Path, monkeypatch) -> None:
    path = write_fixture(tmp_path / "loto.h5ad", remove_time=1)
    output = tmp_path / "failed"

    def fail(*args, **kwargs):
        raise OfficialAPIError("mock official failure")

    monkeypatch.setattr(run_module, "run_official_coupling", fail)
    code = run_module.main(
        [*common_args(path, output, "loto", 1), "--method", "paste"]
    )
    assert code == 2
    failure = json.loads((output / "failure_manifest.json").read_text())
    assert failure["surrogate_attempted"] is False
    assert failure["truth_artifact_opened"] is False
    assert not (output / "prediction.npz").exists()


def test_manifest_sha_is_verified_without_reading_truth(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "loto.h5ad", remove_time=1)
    reference = tmp_path / "training_reference.npz"
    np.savez_compressed(reference, state=np.ones((2, 3)), spatial=np.ones((2, 2)))
    manifest = write_manifest(
        tmp_path / "manifest.json", path, "loto_t1", training_reference=reference
    )
    output = tmp_path / "verified"
    code = run_module.main(
        [
            *common_args(path, output, "loto", 1),
            "--method",
            "linear_centroid_shift",
            "--input-manifest",
            str(manifest),
        ]
    )
    assert code == 0
    record = json.loads((output / "run_manifest.json").read_text())
    assert record["input"]["input_manifest_h5ad_verified"] is True
    assert record["input"]["input_manifest_training_reference_verified"] is True
    assert record["protocol"]["truth_artifact_opened"] is False
    assert record["protocol"]["target_n_used_for_prediction"] is False
    summary = json.loads((output / "summary.json").read_text())
    assert summary["target_time"] == 1.0
    assert summary["input_manifest_sha256"] == sha256(manifest)
    assert summary["training_reference_sha256"] == sha256(reference)


def test_manifest_sha_mismatch_fails(tmp_path: Path) -> None:
    path = write_fixture(tmp_path / "loto.h5ad", remove_time=1)
    manifest = tmp_path / "bad_manifest.json"
    manifest.write_text(
        json.dumps({"splits": {"loto_t1": {"train_h5ad_sha256": "0" * 64}}}),
        encoding="utf-8",
    )
    code = run_module.main(
        [
            *common_args(path, tmp_path / "bad", "loto", 1),
            "--method",
            "linear_centroid_shift",
            "--input-manifest",
            str(manifest),
        ]
    )
    assert code == 2
