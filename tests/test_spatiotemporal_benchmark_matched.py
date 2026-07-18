from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "spatiotemporal_benchmark"
    / "evaluate_matched_tracks.py"
)
SPEC = importlib.util.spec_from_file_location("matched_track_evaluator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
matched = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matched)


def _sha(path: Path) -> str:
    return matched.primary.sha256_file(path)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _reference_arrays() -> dict[str, np.ndarray]:
    time = np.repeat(np.arange(5, dtype=np.float64), 3)
    within = np.tile(np.arange(3, dtype=np.float64), 5)
    row_id = np.asarray(
        [f"cell_t{int(stage)}_{int(index)}" for stage, index in zip(time, within)]
    )
    state = np.column_stack(
        (
            time + 0.25 * within,
            (time + 1.0) * (within + 1.0),
        )
    ).astype(np.float64)
    spatial = np.column_stack(
        (
            2.0 * time + within,
            -time + 0.5 * within,
        )
    ).astype(np.float64)
    return {"state": state, "spatial": spatial, "time": time, "row_id": row_id}


def _subset(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[mask] for key, value in arrays.items()}


def _build_contract(
    tmp_path: Path,
    *,
    mutate_anchor_split: str | None = None,
    semantic_violation: str | None = None,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    reference = _reference_arrays()
    split_records: dict[str, dict[str, object]] = {}
    split_paths: dict[str, Path] = {}

    for split_id, held_out in (("loto_t1", 1), ("loto_t2", 2), ("full_data", None)):
        mask = np.ones(reference["time"].shape[0], dtype=bool)
        if held_out is not None:
            mask &= reference["time"] != float(held_out)
        training = _subset(reference, mask)
        if semantic_violation == "target_leak" and split_id == "loto_t1":
            leaked = _subset(reference, reference["time"] == 1.0)
            training = {
                key: np.concatenate((training[key], leaked[key][:1]), axis=0)
                for key in training
            }
        if split_id == mutate_anchor_split:
            training["state"] = training["state"].copy()
            anchor_row = np.flatnonzero(training["time"] == 0.0)[0]
            training["state"][anchor_row, 0] += 0.125
        split_dir = inputs / split_id
        training_path = split_dir / "training_reference.npz"
        roster_path = split_dir / "source_roster.npz"
        _write_npz(training_path, training)
        _write_npz(roster_path, _subset(training, np.arange(len(training["time"])) < 4))
        split_paths[split_id] = training_path

        truth_by_time: dict[str, object] = {}
        truth_targets = (held_out,) if held_out is not None else (1, 2, 4)
        for target in truth_targets:
            assert target is not None
            truth_path = split_dir / f"truth_t{target}.npz"
            _write_npz(truth_path, _subset(reference, reference["time"] == target))
            truth_by_time[str(target)] = _artifact(truth_path)
        split_records[split_id] = {
            "train": {
                "training_reference_npz": _artifact(training_path),
                "source_roster_npz": _artifact(roster_path),
            },
            "truth_by_time_npz": truth_by_time,
        }

    manifest_path = inputs / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "dataset_id": "synthetic_matched",
                "loto_targets": [1, 2],
                "full_data_targets": [1, 2, 4],
                "splits": split_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha = _sha(manifest_path)

    loto_root = tmp_path / "predictions" / "loto"
    full_root = tmp_path / "predictions" / "full_data"
    summaries: dict[str, Path] = {}
    for track, predictions_root in (("loto", loto_root), ("full_data", full_root)):
        for target in (1, 2):
            split_id = f"loto_t{target}" if track == "loto" else "full_data"
            split = split_records[split_id]
            training_record = split["train"]["training_reference_npz"]  # type: ignore[index]
            roster_record = split["train"]["source_roster_npz"]  # type: ignore[index]
            truth = _subset(reference, reference["time"] == target)
            for method, state_only in (("joint_method", False), ("state_method", True)):
                method_dir = predictions_root / method / f"t{target}"
                prediction_path = method_dir / "prediction.npz"
                offset = 0.20 if track == "loto" else 0.05
                predicted = {
                    "state": truth["state"] + offset,
                }
                if not state_only:
                    predicted["spatial"] = truth["spatial"] + offset
                _write_npz(prediction_path, predicted)
                summary_path = method_dir / "prediction.summary.json"
                summary_path.write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "track": track,
                            "target_time": target,
                            "source_time": 0,
                            "method": method,
                            "output_scope": (
                                "native_state" if state_only else "native_joint"
                            ),
                            "native_vs_adapter": "native",
                            "input_manifest_sha256": manifest_sha,
                            "training_reference_sha256": training_record["sha256"],  # type: ignore[index]
                            "source_roster_sha256": roster_record["sha256"],  # type: ignore[index]
                            "prediction_npz_sha256": _sha(prediction_path),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                summaries[f"{track}/{method}/t{target}"] = summary_path
    return manifest_path, loto_root, full_root, summaries


def _replace_prediction(
    summary_path: Path,
    *,
    state: np.ndarray,
    spatial: np.ndarray | None,
    weights: np.ndarray | None = None,
    summary_updates: dict[str, object] | None = None,
) -> None:
    prediction_path = summary_path.parent / "prediction.npz"
    arrays = {"state": np.asarray(state)}
    if spatial is not None:
        arrays["spatial"] = np.asarray(spatial)
    if weights is not None:
        arrays["weights"] = np.asarray(weights, dtype=np.float64)
    _write_npz(prediction_path, arrays)
    summary = json.loads(summary_path.read_text())
    summary["prediction_npz_sha256"] = _sha(prediction_path)
    if summary_updates:
        summary.update(summary_updates)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prediction_arrays(summary_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(summary_path.parent / "prediction.npz", allow_pickle=False) as archive:
        return np.asarray(archive["state"]), np.asarray(archive["spatial"])


def _args(
    manifest: Path,
    loto_root: Path,
    full_root: Path,
    output: Path,
) -> Namespace:
    return Namespace(
        input_manifest=manifest,
        loto_predictions_root=loto_root,
        full_data_predictions_root=full_root,
        output_dir=output,
        targets=None,
        anchor_times=[0.0, 4.0],
        methods=["joint_method", "state_method"],
        include_nonprimary=False,
        n_projections=12,
        projection_repeats=2,
        max_ot_points=2,
    )


def test_matched_evaluator_shares_transform_truth_and_random_keys(
    tmp_path: Path,
) -> None:
    manifest, loto_root, full_root, _ = _build_contract(tmp_path)
    output = tmp_path / "reports" / "matched"
    metrics, paired, evaluation = matched.evaluate_matched(
        _args(manifest, loto_root, full_root, output)
    )

    assert evaluation["targets"] == [1, 2]
    assert evaluation["tracks"]["full_data"]["evaluation_scope"] == "in_sample"
    assert evaluation["reporting_policy"]["cross_space_aggregation"] is False
    assert evaluation["reporting_policy"]["ranking"] is False
    assert metrics.loc[metrics["track"] == "full_data", "is_in_sample"].all()
    assert not metrics.loc[metrics["track"] == "loto", "is_in_sample"].any()
    assert metrics["transform_sha256"].nunique() == 1
    assert len(metrics) == 32
    assert evaluation["n_metrics_rows"] == 32
    assert evaluation["n_paired_rows"] == 8
    assert set(metrics["exact_ot_predicted_points"]) == {2}
    assert set(metrics["exact_ot_observed_points"]) == {2}

    for _, frame in metrics.groupby(
        ["method", "target", "space", "projection_repeat"], sort=True
    ):
        assert set(frame["track"]) == {"loto", "full_data"}
        assert frame["projection_seed"].nunique() == 1
        assert frame["projection_sha256"].nunique() == 1
        assert frame["exact_ot_seed"].nunique() == 1
        assert frame["exact_ot_observed_indices_sha256"].nunique() == 1
        assert frame["exact_ot_matched"].all()
        assert frame["exact_ot_separate_rng"].all()
        assert frame["truth_bundle_sha256"].nunique() == 1
    for _, frame in metrics.groupby(
        ["target", "space", "projection_repeat"], sort=True
    ):
        assert frame["projection_seed"].nunique() == 1
        assert frame["projection_sha256"].nunique() == 1
        assert frame["exact_ot_seed"].nunique() == 1
        assert frame["exact_ot_observed_indices_sha256"].nunique() == 1

    assert set(paired["comparison"]) == {"loto_held_out_minus_full_data_in_sample"}
    assert set(paired["comparison_type"]) == {"descriptive_paired_gap"}
    assert paired["full_data_is_in_sample"].all()
    assert set(paired.loc[paired["method"] == "state_method", "space"]) == {"state"}
    assert len(paired) == 8  # 2 targets * (3 joint spaces + 1 state-only space)
    assert not any(
        "rank" in column.lower() or "overall" in column.lower()
        for column in paired.columns
    )
    assert (
        paired["sliced_w2_mean_loto_minus_full_data"] > 0
    ).all(), "the deliberately worse held-out predictions should have a positive gap"

    audit = json.loads((output / "common_anchor_audit.json").read_text())
    assert audit["anchor_times"] == [0.0, 4.0]
    assert audit["anchor_rows"] == 6
    assert audit["all_participating_training_splits_byte_identical"] is True
    assert set(audit["participating_splits"]) == {"full_data", "loto_t1", "loto_t2"}
    assert len(audit["anchor_bundle_sha256"]) == 64
    assert len(audit["anchor_concatenated_c_order_sha256"]) == 64
    assert audit["anchor_concatenated_field_order"] == [
        "row_id",
        "state",
        "spatial",
        "time",
    ]
    assert all(item["byte_identical"] for item in audit["paired_truth"].values())
    split_audit = json.loads((output / "semantic_split_audit.json").read_text())
    assert split_audit["all_targets_semantically_verified"] is True
    for record in split_audit["targets"].values():
        assert record["loto_training_excludes_target"] is True
        assert record["loto_train_truth_row_ids_disjoint"] is True
        assert record["loto_training_is_byte_exact_full_data_complement"] is True
        assert record["full_data_truth_is_byte_exact_full_data_target_subset"] is True
        assert record["complement_rows_sum"] == record["full_data_rows"]
    exact_audit = json.loads((output / "matched_exact_ot_sampling.json").read_text())
    assert (
        exact_audit["observed_sampling_independent_of_predicted_rng_consumption"]
        is True
    )
    assert exact_audit["observed_indices_shared_across_methods_and_tracks"] is True
    assert len(exact_audit["records"]) == 6
    inventory = json.loads((output / "prediction_inventory.json").read_text())
    assert inventory["n_records"] == 8
    inventory_keys = [
        (
            record["method"],
            record["track"],
            record["target"],
            record["prediction_path"],
        )
        for record in inventory["records"]
    ]
    assert inventory_keys == sorted(inventory_keys)
    assert evaluation["prediction_inventory_sha256"] == _sha(
        output / "prediction_inventory.json"
    )
    assert evaluation["bound_run_contract_sha256"] == _sha(
        output / "bound_run_contract.json"
    )
    assert set(evaluation["code_dependencies"]) == {
        "matched_evaluator",
        "primary_evaluator",
        "benchmark_metrics",
        "distribution_metrics",
    }
    for dependency in evaluation["code_dependencies"].values():
        assert dependency["sha256"] == _sha(Path(dependency["path"]))
    assert set(evaluation["software_versions"]) == {
        "python",
        "python_implementation",
        "python_executable",
        "numpy",
        "pandas",
        "scipy",
        "POT",
    }
    assert all(evaluation["software_versions"].values())
    assert (output / "matched_metrics_long.csv").is_file()
    assert (output / "matched_paired_summary.csv").is_file()


def test_matched_evaluator_rejects_nonidentical_anchor_bytes(tmp_path: Path) -> None:
    manifest, loto_root, full_root, _ = _build_contract(
        tmp_path, mutate_anchor_split="loto_t1"
    )
    with pytest.raises(matched.ContractError, match="not byte-identical"):
        matched.evaluate_matched(
            _args(manifest, loto_root, full_root, tmp_path / "reports")
        )


def test_matched_evaluator_rejects_loto_target_membership_leak(tmp_path: Path) -> None:
    manifest, loto_root, full_root, _ = _build_contract(
        tmp_path, semantic_violation="target_leak"
    )
    with pytest.raises(matched.ContractError, match="still contains target t1"):
        matched.evaluate_matched(
            _args(manifest, loto_root, full_root, tmp_path / "reports")
        )


def test_matched_evaluator_preserves_prediction_provenance_checks(
    tmp_path: Path,
) -> None:
    manifest, loto_root, full_root, summaries = _build_contract(tmp_path)
    summary_path = summaries["full_data/joint_method/t2"]
    summary = json.loads(summary_path.read_text())
    summary["training_reference_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(matched.ContractError, match="training-reference SHA-256"):
        matched.evaluate_matched(
            _args(manifest, loto_root, full_root, tmp_path / "reports")
        )


def test_matched_evaluator_requires_complete_track_grid(tmp_path: Path) -> None:
    manifest, loto_root, full_root, _ = _build_contract(tmp_path)
    prediction = full_root / "state_method" / "t2" / "prediction.npz"
    prediction.unlink()
    with pytest.raises(matched.ContractError, match="incomplete matched.*grid"):
        matched.evaluate_matched(
            _args(manifest, loto_root, full_root, tmp_path / "reports")
        )


def test_only_targets_shared_by_both_tracks_are_eligible(tmp_path: Path) -> None:
    manifest, loto_root, full_root, _ = _build_contract(tmp_path)
    args = _args(manifest, loto_root, full_root, tmp_path / "reports")
    args.targets = [4]
    with pytest.raises(matched.ContractError, match="not shared"):
        matched.evaluate_matched(args)


def test_partial_run_is_resumable_but_different_contract_is_rejected(
    tmp_path: Path,
) -> None:
    manifest, loto_root, full_root, _ = _build_contract(tmp_path)
    output = tmp_path / "reports"
    args = _args(manifest, loto_root, full_root, output)
    matched.evaluate_matched(args)
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    before = {path.relative_to(output).as_posix(): _sha(path) for path in artifacts}
    mtimes_before = {
        path.relative_to(output).as_posix(): path.stat().st_mtime_ns
        for path in artifacts
    }

    matched.evaluate_matched(args)
    after = {
        path.relative_to(output).as_posix(): _sha(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert {
        path.relative_to(output).as_posix(): path.stat().st_mtime_ns
        for path in output.rglob("*")
        if path.is_file()
    } == mtimes_before

    changed = _args(manifest, loto_root, full_root, output)
    changed.n_projections += 1
    with pytest.raises(matched.ContractError, match="different matched run contract"):
        matched.evaluate_matched(changed)
    assert {
        path.relative_to(output).as_posix(): _sha(path)
        for path in output.rglob("*")
        if path.is_file()
    } == before


def test_partial_run_refuses_to_overwrite_changed_output(tmp_path: Path) -> None:
    manifest, loto_root, full_root, _ = _build_contract(tmp_path)
    output = tmp_path / "reports"
    args = _args(manifest, loto_root, full_root, output)
    matched.evaluate_matched(args)
    metrics_path = output / "matched_metrics_long.csv"
    metrics_path.write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(matched.ContractError, match="different audited file"):
        matched.evaluate_matched(args)
    assert metrics_path.read_text(encoding="utf-8") == "corrupted\n"


def test_nonempty_output_without_run_contract_is_not_adopted(tmp_path: Path) -> None:
    manifest, loto_root, full_root, _ = _build_contract(tmp_path)
    output = tmp_path / "reports"
    output.mkdir()
    foreign = output / "matched_metrics_long.csv"
    foreign.write_text("foreign\n", encoding="utf-8")
    with pytest.raises(matched.ContractError, match="non-empty output directory"):
        matched.evaluate_matched(_args(manifest, loto_root, full_root, output))
    assert foreign.read_text(encoding="utf-8") == "foreign\n"
    assert not (output / "run_contract.json").exists()


@pytest.mark.parametrize("artifact_kind", ["prediction", "summary"])
def test_bound_inventory_rejects_external_mutation_before_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    manifest, loto_root, full_root, summaries = _build_contract(tmp_path)
    summary_path = summaries["loto/joint_method/t1"]
    external_path = (
        summary_path.parent / "prediction.npz"
        if artifact_kind == "prediction"
        else summary_path
    )
    original_write = matched._write_immutable_json
    mutated = False

    def write_then_mutate(path: Path, payload: dict[str, object]) -> None:
        nonlocal mutated
        original_write(path, payload)
        if path.name == "bound_run_contract.json" and not mutated:
            with external_path.open("ab") as handle:
                handle.write(b"external-mutation")
            mutated = True

    def metrics_must_not_run(**kwargs):
        raise AssertionError("metrics ran after a bound external artifact changed")

    monkeypatch.setattr(matched, "_write_immutable_json", write_then_mutate)
    monkeypatch.setattr(matched, "_evaluate_predictions", metrics_must_not_run)
    with pytest.raises(matched.ContractError, match="changed after snapshot"):
        matched.evaluate_matched(
            _args(manifest, loto_root, full_root, tmp_path / "reports")
        )
    assert mutated is True


def test_final_publish_guard_rehashes_external_inventory(tmp_path: Path) -> None:
    manifest, loto_root, full_root, summaries = _build_contract(tmp_path)
    output = tmp_path / "reports"
    _, _, evaluation = matched.evaluate_matched(
        _args(manifest, loto_root, full_root, output)
    )
    with summaries["full_data/state_method/t2"].open("ab") as handle:
        handle.write(b"changed-before-final-publish")
    with pytest.raises(matched.ContractError, match="changed after snapshot"):
        matched._verify_bound_inventory_from_manifest(evaluation)
    assert not (output / "matched_evaluation_manifest.json").exists()


def test_exact_ot_shares_observed_indices_with_unequal_weighted_predictions(
    tmp_path: Path,
) -> None:
    manifest, loto_root, full_root, summaries = _build_contract(tmp_path)
    for track, extra_rows, weights in (
        ("loto", 2, np.asarray([0.70, 0.10, 0.05, 0.10, 0.05])),
        ("full_data", 1, np.asarray([0.05, 0.15, 0.20, 0.60])),
    ):
        summary_path = summaries[f"{track}/joint_method/t1"]
        state, spatial = _prediction_arrays(summary_path)
        state = np.vstack((state, state[:extra_rows] + 0.33))
        spatial = np.vstack((spatial, spatial[:extra_rows] - 0.27))
        _replace_prediction(
            summary_path,
            state=state,
            spatial=spatial,
            weights=weights,
        )

    output = tmp_path / "reports"
    metrics, _, _ = matched.evaluate_matched(
        _args(manifest, loto_root, full_root, output)
    )
    pair = metrics[
        (metrics["method"] == "joint_method")
        & (metrics["target"] == 1)
        & (metrics["space"] == "state")
    ]
    assert set(pair.groupby("track")["n_predicted"].first()) == {4, 5}
    assert pair["exact_ot_observed_indices_sha256"].nunique() == 1
    assert pair["exact_ot_observed_seed"].nunique() == 1
    assert pair["exact_ot_separate_rng"].all()
    assert pair["exact_ot_predicted_indices_sha256"].str.len().eq(64).all()

    exact_audit = json.loads((output / "matched_exact_ot_sampling.json").read_text())
    record = exact_audit["records"]["t1/state"]
    expected = np.random.default_rng(record["seed"]).choice(
        record["observed_rows"], size=2, replace=False
    )
    assert record["observed_indices"] == expected.tolist()
    assert record["observed_indices_sha256"] == matched._indices_sha256(expected)


def test_tmv_delta_requires_native_mass_and_matching_source_denominator(
    tmp_path: Path,
) -> None:
    manifest, loto_root, full_root, summaries = _build_contract(tmp_path)
    for target in (1, 2):
        for track in ("loto", "full_data"):
            summary_path = summaries[f"{track}/joint_method/t{target}"]
            state, spatial = _prediction_arrays(summary_path)
            weights = (
                np.full(state.shape[0], 0.5)
                if track == "loto"
                else np.ones(state.shape[0])
            )
            source_time = 1 if (track == "full_data" and target == 2) else 0
            _replace_prediction(
                summary_path,
                state=state,
                spatial=spatial,
                weights=weights,
                summary_updates={
                    "native_mass": True,
                    "weights_are_unnormalised": True,
                    "source_time": source_time,
                },
            )

    _, paired, _ = matched.evaluate_matched(
        _args(manifest, loto_root, full_root, tmp_path / "reports")
    )
    target1 = paired[(paired["method"] == "joint_method") & (paired["target"] == 1)]
    assert target1["tmv_available_loto"].all()
    assert target1["tmv_available_full_data"].all()
    assert target1["tmv_directly_comparable"].all()
    np.testing.assert_allclose(target1["tmv_loto"], 0.5)
    np.testing.assert_allclose(target1["tmv_full_data"], 2.0)
    np.testing.assert_allclose(target1["tmv_loto_minus_full_data"], -1.5)

    target2 = paired[(paired["method"] == "joint_method") & (paired["target"] == 2)]
    assert target2["tmv_available_loto"].all()
    assert target2["tmv_available_full_data"].all()
    assert not target2["tmv_directly_comparable"].any()
    assert target2["tmv_loto_minus_full_data"].isna().all()
    state_only = paired[paired["method"] == "state_method"]
    assert not state_only["tmv_available_loto"].any()
    assert not state_only["tmv_available_full_data"].any()


def test_matched_cli_writes_hash_bound_manifest(tmp_path: Path, capsys) -> None:
    manifest, loto_root, full_root, _ = _build_contract(tmp_path)
    output = tmp_path / "reports"
    status = matched.main(
        [
            "--input-manifest",
            str(manifest),
            "--loto-predictions-root",
            str(loto_root),
            "--full-data-predictions-root",
            str(full_root),
            "--output-dir",
            str(output),
            "--anchor-times",
            "0",
            "4",
            "--methods",
            "joint_method",
            "state_method",
            "--n-projections",
            "4",
            "--projection-repeats",
            "1",
            "--max-ot-points",
            "2",
        ]
    )
    assert status == 0
    capsys.readouterr()
    evaluation_path = output / "matched_evaluation_manifest.json"
    evaluation = json.loads(evaluation_path.read_text())
    assert evaluation["status"] == "complete"
    assert evaluation["anchor_rows"] == 6
    assert evaluation["all_participating_training_splits_byte_identical"] is True
    assert evaluation["metrics_long_csv_sha256"] == _sha(
        output / "matched_metrics_long.csv"
    )
    assert evaluation["paired_summary_csv_sha256"] == _sha(
        output / "matched_paired_summary.csv"
    )
    status = matched.main(
        [
            "--input-manifest",
            str(manifest),
            "--loto-predictions-root",
            str(loto_root),
            "--full-data-predictions-root",
            str(full_root),
            "--output-dir",
            str(output),
            "--anchor-times",
            "0",
            "4",
            "--methods",
            "joint_method",
            "state_method",
            "--n-projections",
            "4",
            "--projection-repeats",
            "1",
            "--max-ot-points",
            "2",
        ]
    )
    assert status == 2
    assert "final manifest exists" in capsys.readouterr().err
