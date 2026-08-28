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
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results.nonspatial_figures import (  # noqa: E402
    FIGURE_IDS,
    SCNT_CELL_TYPES,
    WEINREB_CELL_TYPES,
    calculate_nonspatial_panels,
    load_nonspatial_figures,
    nonspatial_statistics,
    plot_nonspatial_figures,
    write_nonspatial_tables,
)


RESOURCE_NAMES = {
    "external_inputs.json",
    "manifest.json",
    "scnt_cell_colors.json",
    "scnt_direction.csv",
    "scnt_distribution.csv",
    "scnt_interaction_vectors.csv",
    "scnt_metrics.json",
    "scnt_model_fields.npz",
    "scnt_network_edges.csv",
    "scnt_network_nodes.csv",
    "scnt_network_pairs.csv",
    "scnt_observed_cells.npz",
    "scnt_pathways.csv",
    "weinreb_cell_colors.json",
    "weinreb_clone_fate.csv",
    "weinreb_concordance.csv",
    "weinreb_distribution.csv",
    "weinreb_metrics.json",
    "weinreb_model_fields.npz",
    "weinreb_network_edges.csv",
    "weinreb_network_nodes.csv",
    "weinreb_observed_cells.npz",
    "weinreb_pathways.csv",
}


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_nonspatial_figures().source_dir
    target = tmp_path / "nonspatial_figures"
    shutil.copytree(source, target)
    return target


def test_packaged_nonspatial_contract() -> None:
    results = load_nonspatial_figures()

    assert results.source_dir.name == "nonspatial_figures"
    assert tuple(results.manifest["figures"]) == FIGURE_IDS
    assert tuple(results.weinreb.cells.label_names.astype(str)) == WEINREB_CELL_TYPES
    assert tuple(results.scnt.cells.label_names.astype(str)) == SCNT_CELL_TYPES
    assert len(results.weinreb.cells.times) == 49_302
    assert len(results.scnt.cells.times) == 20_547
    assert [len(results.weinreb.cells.frame(day)[0]) for day in (2, 4, 6)] == [
        4_638,
        14_985,
        29_679,
    ]
    assert [
        len(results.scnt.cells.frame(time)[0]) for time in (0, 0.25, 0.5, 1, 2)
    ] == [2_232, 4_031, 4_470, 6_814, 3_000]
    assert results.weinreb.fields["day6_lr_interaction_u"].shape == (50, 50)
    assert results.scnt.fields["interaction_u"].shape == (42, 42)
    assert not results.weinreb.cells.pc_xy.flags.writeable
    assert not results.scnt.fields["interaction_u"].flags.writeable

    assert results.weinreb.colors["Monocyte"] == "#2878B5"
    assert results.weinreb.colors["Neutrophil"] == "#E45756"
    assert results.scnt.colors["Ex"] == "#4C78A8"
    assert results.scnt.colors["RG"] == "#59A14F"


def test_nonspatial_package_resources_are_safe_compact_and_registered() -> None:
    resource = files("CytoBridge.results").joinpath("data", "nonspatial_figures")
    root = Path(str(resource))
    assert {path.name for path in root.iterdir()} == RESOURCE_NAMES
    assert sum(path.stat().st_size for path in root.iterdir()) < 1.25 * 1024**2

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == RESOURCE_NAMES - {"manifest.json"}
    for name, record in manifest["files"].items():
        assert record["bytes"] == (root / name).stat().st_size

    for path in root.glob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            assert payload.files
            for name in payload.files:
                assert not payload[name].dtype.hasobject
                assert payload[name].ndim >= 1

    registry = json.loads((root / "external_inputs.json").read_text(encoding="utf-8"))
    registered_paths: list[str] = []
    for record in registry["datasets"].values():
        registered_paths.extend([record["renderer"], *record["full_rerun_inputs"]])
    assert registered_paths
    assert all(not Path(value).is_absolute() for value in registered_paths)
    assert all(".." not in Path(value).parts for value in registered_paths)
    assert any(
        value.endswith("cytobridge_pathway_scores.csv") for value in registered_paths
    )
    assert any(
        value.endswith("full_paired_dense_trajectory.npz") for value in registered_paths
    )
    assert "cytobridge_pathway_scores.csv" not in RESOURCE_NAMES
    assert "full_paired_dense_trajectory.npz" not in RESOURCE_NAMES


def test_nonspatial_recalculates_formal_panel_values() -> None:
    results = load_nonspatial_figures()
    panels = calculate_nonspatial_panels(results)
    statistics = nonspatial_statistics(results, panels)

    assert np.isclose(
        statistics["weinreb_distribution_error_increase_pct"],
        3.68698928009914,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.isclose(statistics["weinreb_day6_spearman_rho"], 0.6640682592684947)
    assert np.isclose(statistics["scnt_network_spearman_rho"], 0.7523959827780977)
    assert statistics["scnt_direction_mean_wins"] == 3
    assert statistics["scnt_direction_median_wins"] == 3

    clone = panels.weinreb_clone_fate.set_index("metric")
    assert np.isclose(clone.loc["tv_agreement", "full"], 0.471617046630122)
    assert np.isclose(clone.loc["js_similarity", "full"], 0.5359076344381789)
    assert np.isclose(
        clone.loc["dominant_fate_match", "full"],
        0.4971705739692805,
    )
    assert panels.weinreb_pathways.iloc[0]["pathway"] == "GRN"
    assert np.isclose(
        panels.weinreb_pathways.iloc[0]["share_of_day6_score_pct"],
        40.12588366123648,
    )
    assert panels.scnt_pathways.iloc[0]["pathway"] == "PTN"
    assert np.isclose(panels.scnt_pathways.iloc[0]["share_pct"], 59.34256202082807)

    direction = panels.scnt_direction.set_index("condition")
    assert np.isclose(
        direction.loc["full_interaction_noise", "cell_cosine_mean"],
        0.009149566935620523,
    )
    assert np.isclose(
        direction.loc["no_interaction_noise", "cell_cosine_median"],
        0.009072096589338247,
    )


def test_nonspatial_tables_are_written(tmp_path: Path) -> None:
    panels = calculate_nonspatial_panels(load_nonspatial_figures())
    paths = write_nonspatial_tables(panels, tmp_path)

    assert set(paths) == {
        "weinreb_distribution",
        "weinreb_clone_fate",
        "weinreb_concordance",
        "weinreb_pathways",
        "scnt_distribution",
        "scnt_direction",
        "scnt_pathways",
        "summary",
    }
    assert pd.read_csv(paths["weinreb_distribution"]).shape == (2, 11)
    assert pd.read_csv(paths["scnt_direction"]).shape == (2, 4)
    assert pd.read_csv(paths["summary"]).shape == (7, 3)


def test_nonspatial_import_is_matplotlib_lazy(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import CytoBridge.results.nonspatial_figures; "
                "assert 'matplotlib' not in sys.modules"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    assert completed.stdout == ""


def test_nonspatial_renderer_writes_a4_pages_and_preserves_rc(tmp_path: Path) -> None:
    results = load_nonspatial_figures()
    panels = calculate_nonspatial_panels(results)
    before = {
        "font.family": list(mpl.rcParams["font.family"]),
        "font.size": mpl.rcParams["font.size"],
        "figure.figsize": list(mpl.rcParams["figure.figsize"]),
    }
    rendered = plot_nonspatial_figures(results, tmp_path, panels)

    assert tuple(rendered) == FIGURE_IDS
    assert before == {
        "font.family": list(mpl.rcParams["font.family"]),
        "font.size": mpl.rcParams["font.size"],
        "figure.figsize": list(mpl.rcParams["figure.figsize"]),
    }
    for pdf, png in rendered.values():
        assert pdf.stat().st_size > 40_000
        assert png.stat().st_size > 200_000
        with Image.open(png) as image:
            assert image.size == (2645, 3738)
            assert image.mode in {"RGB", "RGBA"}


def test_nonspatial_cli_supports_a_figure_subset(tmp_path: Path) -> None:
    output = tmp_path / "cli_output"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/plot_nonspatial_figures.py"),
            "--output-dir",
            str(output),
            "--figures",
            "s5",
        ],
        cwd=tmp_path,
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
    assert summary["analysis"] == "grouped_nonspatial_s4_s5"
    assert summary["input"] == "packaged data"
    assert tuple(summary["figures"]) == ("s5",)
    assert summary["statistics"]["figure_ids"] == list(FIGURE_IDS)
    assert len(summary["tables"]) == 8
    assert str(tmp_path) not in completed.stdout
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_nonspatial_notebook_documents_the_route_and_shows_outputs() -> None:
    path = REPOSITORY_ROOT / "docs/tutorials/paper_figures/nonspatial_figures.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = [
        "".join(cell["source"]).strip()
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ]
    assert any(text.startswith("## How this figure is made") for text in markdown)
    assert "## Load figure inputs" in markdown
    assert any(text.startswith("### 1.") for text in markdown)
    assert any(
        "Input:" in text and "Creates:" in text and "Continue with:" in text
        for text in markdown
    )
    assert "## Recalculate panel values" in markdown
    assert "## Draw and save the figure" in markdown
    assert any(text.startswith("## Preview the generated figures") for text in markdown)
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(isinstance(cell["execution_count"], int) for cell in code_cells)
    image_outputs = [
        output
        for cell in code_cells
        for output in cell["outputs"]
        if "image/png" in output.get("data", {})
    ]
    assert len(image_outputs) == 2
    assert sum(len(cell["outputs"]) for cell in code_cells) > 0


def test_nonspatial_rejects_an_object_array(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    np.savez_compressed(
        results_dir / "weinreb_observed_cells.npz",
        times=np.asarray([2], dtype=np.float32),
        label_id=np.asarray([0], dtype=np.uint8),
        label_names=np.asarray([{"name": "Undifferentiated"}], dtype=object),
        pc_xy=np.zeros((1, 2), dtype=np.float32),
        spring_xy=np.zeros((1, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="[Oo]bject arrays"):
        load_nonspatial_figures(results_dir)


def test_nonspatial_rejects_missing_inputs(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    (results_dir / "scnt_pathways.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Missing processed result files"):
        load_nonspatial_figures(results_dir)


def test_nonspatial_statistics_are_json_ready() -> None:
    summary = nonspatial_statistics(load_nonspatial_figures())
    assert summary["bundle_bytes"] < 1.25 * 1024**2
    assert summary["figure_ids"] == list(FIGURE_IDS)
    json.dumps(summary)
