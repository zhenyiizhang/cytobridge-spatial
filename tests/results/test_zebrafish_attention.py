from __future__ import annotations

from importlib.resources import files
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image
import pymupdf
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results.zebrafish_attention import (  # noqa: E402
    CONDITION_ORDER,
    load_zebrafish_attention_results,
    plot_zebrafish_attention,
    zebrafish_attention_statistics,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_zebrafish_attention_results().source_dir
    target = tmp_path / "zebrafish_attention"
    shutil.copytree(source, target)
    return target


def test_packaged_zebrafish_attention_contract() -> None:
    results = load_zebrafish_attention_results()
    statistics = zebrafish_attention_statistics(results)

    assert results.source_dir.name == "zebrafish_attention"
    assert tuple(statistics["conditions"]) == CONDITION_ORDER
    assert statistics["directed_cell_type_pairs"] == 361
    assert statistics["somite_cells"] == 375
    assert statistics["directed_edges_per_condition"] == 677
    assert statistics["display_edges"] == 15
    assert statistics["spatial_null_permutations"] == 10_000
    assert statistics["observed_neighbor_pairs"] == 396
    assert np.isclose(statistics["spatial_null_mean"], 286.2387)
    assert np.isclose(statistics["spatial_fold"], 1.383460727008612)
    assert np.isclose(statistics["spatial_plus_one_p"], 1 / 10_001)
    assert results.display_edges["display_rank"].tolist() == list(range(1, 16))
    assert results.display_edges["jam_compatible"].all()

    quartiles = results.panels.jam_quartiles.set_index("condition")
    assert np.isclose(
        quartiles.loc["trained", "top_compatibility_percent"], 25.88235294117647
    )
    assert np.isclose(
        quartiles.loc["trained", "bottom_compatibility_percent"], 1.7647058823529411
    )
    assert np.isclose(quartiles.loc["random", "top_compatibility_percent"], 10.0)
    assert np.isclose(quartiles.loc["random", "bottom_compatibility_percent"], 10.0)


def test_zebrafish_attention_package_resources_are_compact() -> None:
    root = files("CytoBridge.results").joinpath("data", "zebrafish_attention")
    expected = {
        "directed_pair_concordance.csv",
        "expression_detection_by_stage_type.csv",
        "jam_compatibility_percentile_summary.csv",
        "jam_quartile_compatibility.csv",
        "myog_association.csv",
        "somite_18hpf_spatial_cells.csv.gz",
        "somite_18hpf_spatial_null_iterations.csv.gz",
        "somite_18hpf_spatial_null_summary.csv",
        "trained_jam_display_edges.csv",
        "type_pair_raw_attention_ranks.csv",
        "manifest.json",
    }
    assert {path.name for path in Path(str(root)).iterdir()} == expected

    manifest_text = root.joinpath("manifest.json").read_text(encoding="utf-8").lower()
    for marker in (
        "/users/",
        "sha256",
        "reviewer",
        "accepted",
        "claim_guardrail",
        "literature_direction_context",
        "descriptive_technical",
    ):
        assert marker not in manifest_text

    association = pd.read_csv(root.joinpath("myog_association.csv"))
    assert "claim_guardrail" not in association
    assert "literature_direction_context" not in association
    quartiles = pd.read_csv(root.joinpath("jam_quartile_compatibility.csv"))
    assert "fisher_exact_two_sided_p" in quartiles
    assert not any("descriptive" in name for name in quartiles.columns)


def test_zebrafish_attention_import_does_not_load_matplotlib(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import CytoBridge.results.zebrafish_attention; "
                "assert 'matplotlib' not in sys.modules"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    assert completed.stderr == ""


def test_zebrafish_attention_plot_is_agg_safe_and_local(tmp_path: Path) -> None:
    mpl.use("Agg", force=True)
    before = mpl.rcParams.copy()
    pdf, png = plot_zebrafish_attention(
        load_zebrafish_attention_results(),
        tmp_path,
    )
    assert pdf.is_file() and pdf.stat().st_size > 0
    with pymupdf.open(pdf) as document:
        assert document.page_count == 1
        page = document[0]
        assert np.isclose(page.rect.width, 595.276, atol=1.0)
        assert np.isclose(page.rect.height, 841.89, atol=1.0)
        assert page.get_drawings()
    with Image.open(png) as image:
        assert image.size == (2646, 3740)
        image.verify()
    assert mpl.rcParams["font.family"] == before["font.family"]
    assert mpl.rcParams["font.size"] == before["font.size"]


def test_zebrafish_attention_cli_summary(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/plot_zebrafish_attention.py"),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "PYTHONPATH": str(REPOSITORY_ROOT),
        },
    )
    summary = json.loads(completed.stdout)
    assert set(summary) == {"analysis", "input", "figure", "tables", "statistics"}
    assert summary["analysis"] == "zebrafish_attention"
    assert summary["input"] == "packaged data"
    assert not any("/" in value for value in summary["figure"].values())
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_zebrafish_attention_requires_fixed_conditions(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "jam_quartile_compatibility.csv"
    table = pd.read_csv(path)
    table.loc[table["condition"].eq("pre_interaction"), "condition"] = "init"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="must contain conditions"):
        load_zebrafish_attention_results(results_dir)


def test_zebrafish_attention_requires_fixed_display_roster(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "trained_jam_display_edges.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="fixed 15-edge display roster"):
        load_zebrafish_attention_results(results_dir)


def test_zebrafish_attention_recalculates_spatial_null(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "somite_18hpf_spatial_null_iterations.csv.gz"
    table = pd.read_csv(path)
    table.loc[0, "orientation_compatible_pair_count"] = 500
    table.loc[0, "at_least_observed"] = True
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagrees with permutations"):
        load_zebrafish_attention_results(results_dir)


def test_zebrafish_attention_rejects_missing_panel_file(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    (results_dir / "myog_association.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Missing processed result files"):
        load_zebrafish_attention_results(results_dir)
