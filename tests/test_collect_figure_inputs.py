from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from CytoBridge.results.interaction_evidence import load_lr_prior_stvcr_results
from CytoBridge.results.loto_benchmark import load_loto_benchmark
from CytoBridge.results.lr_complex_aggregation import (
    DATASET_ORDER as LR_COMPLEX_DATASETS,
    load_lr_complex_aggregation_results,
)
from scripts.collect_figure_inputs import (
    LOTO_DATASETS,
    collect_s25,
    collect_s39,
    collect_s40,
    collect_s40_tables,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DATA = REPO_ROOT / "CytoBridge" / "results" / "data"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_upstream_loto_summaries(tmp_path: Path) -> dict[str, Path]:
    paper = load_loto_benchmark()
    summaries: dict[str, Path] = {}
    for dataset in LOTO_DATASETS:
        means = paper.target_means.loc[paper.target_means["dataset"].eq(dataset)]
        support = paper.native_support.loc[
            paper.native_support["dataset"].eq(dataset),
            [
                "dataset",
                "target",
                "method",
                "native_output_n",
                "native_vs_adapter",
                "output_scope",
            ],
        ]
        summary = means.merge(
            support,
            on=["dataset", "target", "method"],
            how="left",
            validate="many_to_one",
        ).rename(
            columns={
                "projection_sd": "sliced_w2_projection_sd",
                "native_output_n": "n_predicted",
            }
        )
        summary["track"] = "loto"
        summary["status"] = "evaluated"
        path = tmp_path / "upstream" / dataset / "loto_target_summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
        summaries[dataset] = path
    return summaries


def test_collect_s25_copies_and_validates_four_dataset_tables(tmp_path: Path) -> None:
    sources = {
        dataset: RESULT_DATA / "lr_complex_aggregation" / dataset / "paired_scores.csv"
        for dataset in LR_COMPLEX_DATASETS
    }
    output = collect_s25(sources, tmp_path / "s25")
    result = load_lr_complex_aggregation_results(output)

    assert set(result.paired_scores["dataset"]) == set(LR_COMPLEX_DATASETS)
    for dataset, source in sources.items():
        copied = output / dataset / "paired_scores.csv"
        assert _sha256(copied) == _sha256(source)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["analysis"] == "lr_complex_aggregation"
    assert len(manifest["sources"]) == 4


def test_collect_s40_uses_completed_target_summaries(tmp_path: Path) -> None:
    summaries = _write_upstream_loto_summaries(tmp_path)
    protocol = RESULT_DATA / "loto_benchmark" / "protocol.json"
    output = collect_s40(summaries, protocol, tmp_path / "s40")
    collected = load_loto_benchmark(output)
    paper = load_loto_benchmark()

    columns = ["dataset", "target", "method", "space", "sliced_w2"]
    keys = ["dataset", "target", "method", "space"]
    pd.testing.assert_frame_equal(
        collected.target_means[columns].sort_values(keys).reset_index(drop=True),
        paper.target_means[columns].sort_values(keys).reset_index(drop=True),
        check_exact=False,
        rtol=1e-14,
        atol=1e-15,
    )
    pd.testing.assert_frame_equal(
        collected.native_support.reset_index(drop=True),
        paper.native_support.reset_index(drop=True),
        check_dtype=False,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["analysis"] == "loto_benchmark"
    assert len(manifest["sources"]) == 6


def test_collect_s39_combines_no_lr_report_with_s40(tmp_path: Path) -> None:
    summaries = _write_upstream_loto_summaries(tmp_path)
    protocol = RESULT_DATA / "loto_benchmark" / "protocol.json"
    loto_dir = collect_s40(summaries, protocol, tmp_path / "s40")
    no_lr = RESULT_DATA / "interaction_evidence" / "no_lr_paired_target_deltas.csv"
    output = collect_s39(no_lr, loto_dir, tmp_path / "s39")
    collected = load_lr_prior_stvcr_results(output)
    paper = load_lr_prior_stvcr_results()

    pd.testing.assert_frame_equal(collected.no_lr, paper.no_lr)
    pd.testing.assert_frame_equal(
        collected.stvcr,
        paper.stvcr,
        check_exact=False,
        rtol=1e-13,
        atol=1e-15,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["analysis"] == "interaction_evidence"
    assert len(manifest["sources"]) == 4


def test_collect_s40_accepts_an_already_assembled_table_set(tmp_path: Path) -> None:
    source = RESULT_DATA / "loto_benchmark"
    output = collect_s40_tables(
        source / "loto_target_stage_means.csv",
        source / "native_output_support.csv",
        source / "protocol.json",
        tmp_path / "s40",
    )
    load_loto_benchmark(output)
    for name in (
        "loto_target_stage_means.csv",
        "native_output_support.csv",
        "protocol.json",
    ):
        assert _sha256(output / name) == _sha256(source / name)


def test_collect_s40_adds_display_names_to_an_analysis_support_table(
    tmp_path: Path,
) -> None:
    source = RESULT_DATA / "loto_benchmark"
    support = pd.read_csv(source / "native_output_support.csv").drop(
        columns="display_name"
    )
    support_path = tmp_path / "analysis_native_output_support.csv"
    support.to_csv(support_path, index=False)
    output = collect_s40_tables(
        source / "loto_target_stage_means.csv",
        support_path,
        source / "protocol.json",
        tmp_path / "s40",
    )
    collected = load_loto_benchmark(output)
    assert collected.native_support["display_name"].notna().all()


def test_collectors_do_not_replace_an_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sources = {
        dataset: RESULT_DATA / "lr_complex_aggregation" / dataset / "paired_scores.csv"
        for dataset in LR_COMPLEX_DATASETS
    }
    with pytest.raises(FileExistsError, match="choose a new directory"):
        collect_s25(sources, output)


def test_collection_script_is_in_source_distribution() -> None:
    manifest = (REPO_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include scripts/collect_figure_inputs.py" in manifest


def test_collection_script_runs_from_the_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/collect_figure_inputs.py", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{s41,s42,lr-prior-stvcr,s45}" in completed.stdout
