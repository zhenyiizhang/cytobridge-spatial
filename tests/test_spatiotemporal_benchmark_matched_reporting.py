from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "spatiotemporal_benchmark"
    / "report_matched_tracks.py"
)
PRODUCTION_REGISTRY = SCRIPT.with_name("method_registry.json")
SPEC = importlib.util.spec_from_file_location("matched_track_reporting", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def _synthetic_registry_payload() -> dict[str, Any]:
    return {
        "schema_version": "test",
        "methods": [
            {
                "method": "Joint",
                "display_name": "Joint model",
                "aliases": ["joint_raw"],
                "scope": "native_joint",
                "spaces": ["joint", "state", "spatial"],
                "status": "evaluated",
                "color": "#123456",
            },
            {
                "method": "State",
                "display_name": "State model",
                "aliases": ["state_raw"],
                "scope": "native_state",
                "spaces": ["state"],
                "status": "evaluated",
                "color": "#654321",
            },
            {
                "method": "Sensitivity",
                "display_name": "Sensitivity only",
                "aliases": [],
                "scope": "sensitivity_only",
                "spaces": [],
                "status": "sensitivity_only",
            },
        ],
    }


def _write_registry(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _binding(
    registry_payload: dict[str, Any], raw_by_canonical: dict[str, str]
) -> dict[str, Any]:
    records = {
        str(record["method"]): record
        for record in registry_payload["methods"]
        if record.get("status") == "evaluated"
    }
    raw_to_canonical: dict[str, dict[str, Any]] = {}
    for canonical, raw_method in raw_by_canonical.items():
        record = records[canonical]
        raw_to_canonical[raw_method] = {
            "raw_method": raw_method,
            "canonical_method": canonical,
            "display_name": record["display_name"],
            "status": str(record["status"]).casefold(),
            "scope": str(record["scope"]).casefold(),
            "declared_spaces": list(record["spaces"]),
        }
    return {
        "raw_to_canonical": raw_to_canonical,
        "canonical_methods": list(raw_by_canonical),
    }


def _paired(
    registry_payload: dict[str, Any],
    binding: dict[str, Any],
    *,
    tmv_methods: set[str],
    targets: tuple[int, ...],
) -> pd.DataFrame:
    records = {str(record["method"]): record for record in registry_payload["methods"]}
    rows: list[dict[str, object]] = []
    for method_index, (raw_method, metadata) in enumerate(
        binding["raw_to_canonical"].items()
    ):
        canonical = str(metadata["canonical_method"])
        spaces = records[canonical]["spaces"]
        tmv_available = canonical in tmv_methods
        for target in targets:
            tmv_full = 0.05 * target + 0.001 * method_index
            tmv_loto = tmv_full + 0.025
            denominator = 1.0 + 0.1 * target
            for space_index, space in enumerate(spaces):
                full_sliced = 0.10 * target + 0.01 * space_index + 0.001 * method_index
                loto_sliced = full_sliced + 0.20
                full_w1 = 0.20 * target + 0.01 * space_index + 0.001 * method_index
                loto_w1 = full_w1 + 0.15
                full_w2 = 0.30 * target + 0.01 * space_index + 0.001 * method_index
                loto_w2 = full_w2 + 0.10
                rows.append(
                    {
                        "raw_method": raw_method,
                        "canonical_method": canonical,
                        "method": canonical,
                        "method_display_name": records[canonical]["display_name"],
                        "target": target,
                        "space": space,
                        "evaluation_scope_loto": "held_out",
                        "evaluation_scope_full_data": "in_sample",
                        "is_in_sample_loto": False,
                        "is_in_sample_full_data": True,
                        "projection_repeats_loto": 5,
                        "projection_repeats_full_data": 5,
                        "sliced_w2_mean_loto": loto_sliced,
                        "sliced_w2_std_loto": 0.01,
                        "sliced_w2_mean_full_data": full_sliced,
                        "sliced_w2_std_full_data": 0.008,
                        "sliced_w2_mean_loto_minus_full_data": (
                            loto_sliced - full_sliced
                        ),
                        "exact_w1_loto": loto_w1,
                        "exact_w1_full_data": full_w1,
                        "exact_w1_loto_minus_full_data": loto_w1 - full_w1,
                        "exact_w2_loto": loto_w2,
                        "exact_w2_full_data": full_w2,
                        "exact_w2_loto_minus_full_data": loto_w2 - full_w2,
                        "source_time_loto": target - 1,
                        "source_time_full_data": target - 1,
                        "tmv_available_loto": tmv_available,
                        "tmv_available_full_data": tmv_available,
                        "tmv_loto": tmv_loto if tmv_available else float("nan"),
                        "tmv_full_data": (tmv_full if tmv_available else float("nan")),
                        "observed_mass_relative_loto": (
                            denominator if tmv_available else float("nan")
                        ),
                        "observed_mass_relative_full_data": (
                            denominator if tmv_available else float("nan")
                        ),
                        "tmv_directly_comparable": tmv_available,
                        "tmv_loto_minus_full_data": (
                            tmv_loto - tmv_full if tmv_available else float("nan")
                        ),
                        "full_data_is_in_sample": True,
                        "comparison_type": "descriptive_paired_gap",
                        "comparison": ("loto_held_out_minus_full_data_in_sample"),
                        "exact_comparison_type": (
                            "matched_shared_observed_indices_separate_rng"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _inputs(
    tmp_path: Path,
    *,
    registry_payload: dict[str, Any] | None = None,
    raw_by_canonical: dict[str, str] | None = None,
    tmv_methods: set[str] | None = None,
    targets: tuple[int, ...] = (1, 2, 3),
) -> tuple[Path, Path, Path]:
    payload = registry_payload or _synthetic_registry_payload()
    raw_names = raw_by_canonical or {"Joint": "joint_raw", "State": "state_raw"}
    registry = tmp_path / "method_registry.json"
    _write_registry(registry, payload)
    method_binding = _binding(payload, raw_names)
    method_binding.update(
        {
            "path": str(registry.resolve()),
            "sha256": reporting.sha256_file(registry),
        }
    )
    frame = _paired(
        payload,
        method_binding,
        tmv_methods={"Joint"} if tmv_methods is None else tmv_methods,
        targets=targets,
    )
    paired = tmp_path / "matched_paired_summary.csv"
    frame.to_csv(paired, index=False)
    manifest = tmp_path / "matched_evaluation_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "status": "complete",
                "design": "matched_loto_vs_full_data",
                "dataset": "synthetic",
                "targets": list(targets),
                "tracks": {
                    "loto": {"evaluation_scope": "held_out"},
                    "full_data": {
                        "evaluation_scope": "in_sample",
                        "is_in_sample": True,
                    },
                },
                "methods": list(method_binding["canonical_methods"]),
                "method_registry": method_binding,
                "spaces": ["joint", "state", "spatial"],
                "projection_repeats": 5,
                "n_paired_rows": len(frame),
                "paired_summary_csv": str(paired.resolve()),
                "paired_summary_csv_sha256": reporting.sha256_file(paired),
                "reporting_policy": {
                    "full_data_is_in_sample": True,
                    "cross_space_aggregation": False,
                    "overall_score": False,
                    "ranking": False,
                    "statistical_inference": False,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, paired, registry


def _args(manifest: Path, paired: Path, registry: Path, output: Path) -> Namespace:
    return Namespace(
        matched_manifest=manifest,
        paired_summary=paired,
        method_registry=registry,
        output_dir=output,
    )


def _rewrite_manifest_for_paired(manifest: Path, paired: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["n_paired_rows"] = len(pd.read_csv(paired))
    payload["paired_summary_csv_sha256"] = reporting.sha256_file(paired)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _fast_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty_figure(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        return reporting.plt.figure(figsize=(1, 1))

    def save(figure: Any, png_path: Path) -> list[Path]:
        reporting.plt.close(figure)
        pdf_path = png_path.with_suffix(".pdf")
        reporting._immutable_bytes(png_path, b"png\n", label="test plot")
        reporting._immutable_bytes(pdf_path, b"pdf\n", label="test plot")
        return [png_path, pdf_path]

    monkeypatch.setattr(reporting, "_dumbbell_figure", empty_figure)
    monkeypatch.setattr(reporting, "_tmv_figure", empty_figure)
    monkeypatch.setattr(reporting, "_save_figure", save)


def test_matched_report_keeps_spaces_separate_hashes_and_preserves_identities(
    tmp_path: Path,
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    output = tmp_path / "report"
    result = reporting.report_matched(_args(manifest, paired, registry, output))

    target = pd.read_csv(output / "tables" / "matched_method_space_target_summary.csv")
    method_space = pd.read_csv(
        output / "tables" / "matched_method_space_across_targets_summary.csv"
    )
    tmv = pd.read_csv(output / "tables" / "matched_tmv_applicable_only.csv")
    assert len(target) == 12
    assert len(method_space) == 4
    assert set(method_space["n_targets"]) == {3}
    assert set(method_space["targets"]) == {"t1,t2,t3"}
    assert set(target.loc[target["canonical_method"] == "State", "space"]) == {"state"}
    for frame in (target, method_space, tmv):
        assert {"raw_method", "canonical_method", "method"}.issubset(frame.columns)
        assert (frame["canonical_method"] == frame["method"]).all()
    assert not any(
        "rank" in column.lower() or "overall" in column.lower()
        for column in [*target.columns, *method_space.columns, *tmv.columns]
    )
    assert len(tmv) == 3
    assert set(tmv["canonical_method"]) == {"Joint"}
    assert tmv["full_data_is_in_sample"].all()

    for metric in ("sliced_w2", "exact_w1", "exact_w2"):
        for space in ("state", "spatial", "joint"):
            base = output / "plots" / f"matched_{metric}_{space}_dumbbell"
            assert base.with_suffix(".png").is_file()
            assert base.with_suffix(".pdf").is_file()
    assert (output / "plots" / "matched_tmv_applicable_only_dumbbell.png").is_file()
    assert (output / "plots" / "matched_tmv_applicable_only_dumbbell.pdf").is_file()

    assert result["reporting_policy"]["ranking"] is False
    assert result["reporting_policy"]["cross_space_aggregation"] is False
    assert "not a" in result["reporting_policy"]["projection_repeat_interpretation"]
    assert len(result["primary_plots"]) == 6
    assert len(result["supplemental_plots"]) == 12
    assert len(result["tmv_plots"]) == 2
    run_contract = Path(result["inputs"]["report_run_contract"])
    assert (
        reporting.sha256_file(run_contract)
        == result["inputs"]["report_run_contract_sha256"]
    )
    bound = json.loads(run_contract.read_text(encoding="utf-8"))
    assert bound["status"] == "bound"
    assert bound["inputs"]["paired_summary_csv_sha256"] == reporting.sha256_file(paired)
    for artifact in [
        *result["tables"],
        *result["primary_plots"],
        *result["supplemental_plots"],
        *result["tmv_plots"],
    ]:
        path = Path(artifact["path"])
        assert reporting.sha256_file(path) == artifact["sha256"]
    published = json.loads(
        (output / "matched_report_manifest.json").read_text(encoding="utf-8")
    )
    assert published == result
    assert not (output / reporting._OUTPUT_CLAIM).exists()


def test_matched_report_rejects_paired_csv_changed_after_formal_manifest(
    tmp_path: Path,
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    frame = pd.read_csv(paired)
    frame.loc[0, "sliced_w2_mean_loto"] += 1.0
    frame.to_csv(paired, index=False)
    with pytest.raises(reporting.ReportError, match="SHA-256"):
        reporting.report_matched(_args(manifest, paired, registry, tmp_path / "report"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("in_sample", "in_sample"),
        ("ranking", "ranking"),
    ],
)
def test_matched_report_rejects_invalid_formal_policy(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "in_sample":
        payload["tracks"]["full_data"]["is_in_sample"] = False
    else:
        payload["reporting_policy"]["ranking"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(reporting.ReportError, match=message):
        reporting.report_matched(_args(manifest, paired, registry, tmp_path / "report"))


def test_matched_report_rejects_fabricated_state_only_spatial_row(
    tmp_path: Path,
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    frame = pd.read_csv(paired)
    fabricated = (
        frame[(frame["raw_method"] == "state_raw") & (frame["target"] == 1)]
        .iloc[0]
        .copy()
    )
    fabricated["space"] = "spatial"
    frame.loc[len(frame)] = fabricated
    frame.to_csv(paired, index=False)
    _rewrite_manifest_for_paired(manifest, paired)
    with pytest.raises(reporting.ReportError, match="applicability"):
        reporting.report_matched(_args(manifest, paired, registry, tmp_path / "report"))


def test_matched_report_rejects_alias_swapped_evaluator_binding(
    tmp_path: Path,
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mapping = payload["method_registry"]["raw_to_canonical"]
    joint_metadata = dict(mapping["joint_raw"])
    state_metadata = dict(mapping["state_raw"])
    mapping["joint_raw"] = {**state_metadata, "raw_method": "joint_raw"}
    mapping["state_raw"] = {**joint_metadata, "raw_method": "state_raw"}
    payload["method_registry"]["canonical_methods"] = ["State", "Joint"]
    payload["methods"] = ["State", "Joint"]
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(reporting.ReportError, match="alias"):
        reporting.report_matched(_args(manifest, paired, registry, tmp_path / "report"))


def test_matched_report_final_rehash_detects_post_snapshot_input_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    _fast_figures(monkeypatch)
    original = reporting._validate_paired

    def mutate_after_parse(*args: Any, **kwargs: Any) -> pd.DataFrame:
        frame = original(*args, **kwargs)
        paired.write_bytes(paired.read_bytes() + b"\n")
        return frame

    monkeypatch.setattr(reporting, "_validate_paired", mutate_after_parse)
    output = tmp_path / "report"
    with pytest.raises(
        reporting.ReportError, match="changed after its hashed byte snapshot"
    ):
        reporting.report_matched(_args(manifest, paired, registry, output))
    assert not (output / "matched_report_manifest.json").exists()
    assert (output / reporting._OUTPUT_CLAIM).is_file()
    assert (output / "report_run_contract.json").is_file()


@pytest.mark.parametrize(
    "foreign_name",
    ["foreign.txt", reporting._OUTPUT_CLAIM, "report_run_contract.json"],
)
def test_matched_report_rejects_foreign_or_partial_output_directory(
    tmp_path: Path, foreign_name: str
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    (output / foreign_name).write_text("foreign\n")
    with pytest.raises(reporting.ReportError, match="must be empty"):
        reporting.report_matched(_args(manifest, paired, registry, output))


def test_output_claim_is_exclusive_under_concurrent_attempts(tmp_path: Path) -> None:
    output = tmp_path / "report"

    def claim() -> Path | Exception:
        try:
            return reporting._claim_output(output)
        except Exception as exc:  # return the contender's formal failure
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: claim(), range(2)))
    winners = [result for result in results if isinstance(result, Path)]
    losers = [result for result in results if isinstance(result, Exception)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert isinstance(losers[0], reporting.ReportError)
    assert winners[0].is_file()


def test_completed_report_and_final_manifest_are_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    _fast_figures(monkeypatch)
    output = tmp_path / "report"
    reporting.report_matched(_args(manifest, paired, registry, output))
    final_path = output / "matched_report_manifest.json"
    before = final_path.read_bytes()
    with pytest.raises(reporting.ReportError, match="must be empty"):
        reporting.report_matched(_args(manifest, paired, registry, output))
    assert final_path.read_bytes() == before

    isolated = tmp_path / "isolated_manifest.json"
    reporting._write_final_manifest(isolated, {"version": 1})
    first = isolated.read_bytes()
    with pytest.raises(reporting.ReportError, match="refusing to overwrite"):
        reporting._write_final_manifest(isolated, {"version": 2})
    assert isolated.read_bytes() == first


def test_matched_report_rejects_false_negative_tmv_applicability(
    tmp_path: Path,
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    frame = pd.read_csv(paired)
    row = (
        frame["raw_method"].eq("joint_raw")
        & frame["target"].eq(1)
        & frame["space"].eq("joint")
    )
    frame.loc[row, "tmv_directly_comparable"] = False
    frame.loc[row, "tmv_loto_minus_full_data"] = np.nan
    frame.to_csv(paired, index=False)
    _rewrite_manifest_for_paired(manifest, paired)
    with pytest.raises(reporting.ReportError, match="exactly equal"):
        reporting.report_matched(_args(manifest, paired, registry, tmp_path / "report"))


def test_matched_report_rejects_cross_space_tmv_inconsistency(
    tmp_path: Path,
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    frame = pd.read_csv(paired)
    row = (
        frame["raw_method"].eq("joint_raw")
        & frame["target"].eq(1)
        & frame["space"].eq("spatial")
    )
    frame.loc[row, "tmv_loto"] += 0.1
    frame.loc[row, "tmv_loto_minus_full_data"] += 0.1
    frame.to_csv(paired, index=False)
    _rewrite_manifest_for_paired(manifest, paired)
    with pytest.raises(reporting.ReportError, match="varies across spaces"):
        reporting.report_matched(_args(manifest, paired, registry, tmp_path / "report"))


def test_matched_report_uses_validated_manifest_targets_without_hardcoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, paired, registry = _inputs(tmp_path, targets=(2, 5))
    _fast_figures(monkeypatch)
    output = tmp_path / "report"
    result = reporting.report_matched(_args(manifest, paired, registry, output))
    target = pd.read_csv(output / "tables" / "matched_method_space_target_summary.csv")
    method_space = pd.read_csv(
        output / "tables" / "matched_method_space_across_targets_summary.csv"
    )
    assert result["targets"] == [2, 5]
    assert result["n_target_rows"] == len(target) == 8
    assert result["n_method_space_rows"] == len(method_space) == 4
    assert set(target["target"]) == {2, 5}
    assert set(method_space["targets"]) == {"t2,t5"}
    assert (
        result["validated_contract"]["targets_validated_from_evaluator_manifest"]
        is True
    )


@pytest.mark.parametrize("invalid_targets", [[], [1, 1], [1, 2.5], [True, 2]])
def test_matched_report_rejects_invalid_manifest_targets(
    tmp_path: Path, invalid_targets: list[object]
) -> None:
    manifest, paired, registry = _inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["targets"] = invalid_targets
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(reporting.ReportError, match="targets"):
        reporting.report_matched(_args(manifest, paired, registry, tmp_path / "report"))


def test_production_registry_grid_has_72_target_and_24_method_space_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    production_payload = json.loads(PRODUCTION_REGISTRY.read_text(encoding="utf-8"))
    raw_by_canonical = {
        "CytoBridge-0.015": "CytoBridge-0.015",
        "stVCR": "stvcr",
        "STORIES": "stories",
        "MOSCOT": "moscot",
        "MIOFlow": "mioflow",
        "PASTE": "paste",
        "Spateo": "spateo",
        "Waddington-OT": "wot",
        "Linear interpolation": "linear_centroid_shift",
        "Random interpolation": "random_independent_pairs",
    }
    manifest, paired, registry = _inputs(
        tmp_path,
        registry_payload=production_payload,
        raw_by_canonical=raw_by_canonical,
        tmv_methods=set(),
    )
    _fast_figures(monkeypatch)
    output = tmp_path / "report"
    result = reporting.report_matched(_args(manifest, paired, registry, output))
    target = pd.read_csv(output / "tables" / "matched_method_space_target_summary.csv")
    method_space = pd.read_csv(
        output / "tables" / "matched_method_space_across_targets_summary.csv"
    )
    assert len(target) == result["n_target_rows"] == 72
    assert len(method_space) == result["n_method_space_rows"] == 24
    assert target.groupby("space").size().to_dict() == {
        "joint": 21,
        "spatial": 21,
        "state": 30,
    }
    state_only = {"STORIES", "MIOFlow", "Waddington-OT"}
    assert set(target.loc[target["canonical_method"].isin(state_only), "space"]) == {
        "state"
    }
    assert set(target["raw_method"]) == set(raw_by_canonical.values())
    assert set(target["canonical_method"]) == set(raw_by_canonical)
    wot_target = target[target["canonical_method"] == "Waddington-OT"]
    assert set(wot_target["scope"]) == {"state_coupling_barycenter_adapter"}
    wot_method_space = method_space[
        method_space["canonical_method"] == "Waddington-OT"
    ]
    assert set(wot_method_space["scope"]) == {
        "state_coupling_barycenter_adapter"
    }
