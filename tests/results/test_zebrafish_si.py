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

from CytoBridge.results.zebrafish_si import (  # noqa: E402
    FIGURE_IDS,
    calculate_zebrafish_si_panels,
    load_zebrafish_si_results,
    plot_zebrafish_si,
    write_zebrafish_si_tables,
    zebrafish_si_statistics,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_zebrafish_si_results().source_dir
    target = tmp_path / "zebrafish_si"
    shutil.copytree(source, target)
    return target


def test_packaged_zebrafish_si_contract() -> None:
    results = load_zebrafish_si_results()

    assert results.source_dir.name == "zebrafish_si"
    assert tuple(results.manifest["figures"]) == FIGURE_IDS
    assert results.observed_generated.n_frames == 14
    assert results.observed_generated.frame_counts.tolist() == [
        563,
        1036,
        2081,
        3048,
        5271,
        563,
        563,
        563,
        563,
        563,
        563,
        563,
        563,
        563,
    ]
    assert results.virtual_removal.n_frames == 15
    assert len(results.virtual_removal.xy) == 29_214
    assert results.endpoint_baseline_xy.shape == (5177, 2)
    assert results.endpoint_ysl_xy.shape == (4546, 2)
    assert results.endpoint_evl_xy.shape == (3306, 2)
    assert results.growth.shape == (11_999, 4)
    assert results.gene_dynamics.shape == (2250, 3)
    assert results.loss_weight_metrics.shape == (60, 4)
    assert results.observed_expression.shape == (2584, 5)
    assert len(results.top_variable_genes) == 250

    observed_t0, observed_labels = results.observed_generated.frame("observed", 0)
    generated_t0, generated_labels = results.observed_generated.frame("generated", 0)
    assert np.array_equal(observed_t0, generated_t0)
    assert observed_labels.shape == generated_labels.shape == (563,)
    assert not results.observed_generated.xy.flags.writeable


def test_zebrafish_si_package_resources_are_safe_and_compact() -> None:
    resource = files("CytoBridge.results").joinpath("data", "zebrafish_si")
    root = Path(str(resource))
    expected = {
        "celltype_colors.json",
        "manifest.json",
        "s27_celltype_colors.json",
        "s27_observed_generated.npz",
        "s28_growth_per_cell.csv.gz",
        "s29_virtual_removal.npz",
        "s30_centroid_by_seed.csv",
        "s30_endpoint_spatial.npz",
        "s30_spatial_w1_curve.csv",
        "s31_gene_dynamics.csv.gz",
        "s32_loss_weight_metrics.csv",
        "s33_composition.csv.gz",
        "s33_lineage_pairs.csv",
        "s33_lineage_values.csv.gz",
        "s33_observed_composition.csv.gz",
        "s33_particle_counts.csv.gz",
        "s33_sensitivity.csv.gz",
        "s34_observed_expression.csv.gz",
        "s34_reconstructed_expression.csv.gz",
        "s34_reported_metrics.csv",
        "s34_top_genes.csv",
    }
    assert {path.name for path in root.iterdir()} == expected
    assert sum(path.stat().st_size for path in root.iterdir()) < 900 * 1024

    results = load_zebrafish_si_results()
    assert results.celltype_colors["Adaxial Cell"] == "#1f77b4"
    assert results.celltype_colors["Spinal Cord Ventral Region"] == "#9edae5"
    assert results.observed_generated_colors["Adaxial Cell"] == "#9467bd"
    assert results.observed_generated_colors["Spinal Cord Ventral Region"] == "#1f77b4"

    for path in root.glob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            assert payload.files
            for name in payload.files:
                assert not payload[name].dtype.hasobject
                assert payload[name].ndim >= 1


def test_zebrafish_si_recalculates_panel_values() -> None:
    results = load_zebrafish_si_results()
    panels = calculate_zebrafish_si_panels(results)

    assert np.allclose(
        panels.growth_quantiles["q05"],
        [0.213864002, 0.4641739825, 0.34094423, 0.273027997, 0.502341575],
        rtol=1e-12,
    )
    centroid = panels.ablation_centroid_summary.set_index("variant")
    assert np.isclose(centroid.loc["remove_YSL", "mean"], 0.07971487111444378)
    assert np.isclose(centroid.loc["remove_EVL", "mean"], 0.1744942608704382)
    assert np.isclose(centroid.loc["remove_EVL", "ci95_high"], 0.2453010333311179)

    assert panels.gene_zscores.shape == (250, 9)
    assert panels.gene_zscores.index[:5].tolist() == [
        "siva1",
        "aif1l",
        "alox5b.3",
        "ctsll",
        "isg15",
    ]
    assert panels.gene_zscores.index[-5:].tolist() == [
        "myhz1.3",
        "myl1",
        "postnb",
        "pvalb2",
        "ttn.2",
    ]
    assert np.allclose(panels.gene_zscores.to_numpy(float).mean(axis=1), 0, atol=2e-14)

    loss = panels.loss_weight_summary.set_index(["condition", "space"])
    assert np.isclose(
        loss.loc[("formal_alpha_control", "spatial"), "mean_w1"], 0.06807612909022664
    )
    assert np.isclose(
        loss.loc[("alpha_expr_005", "spatial"), "mean_w1"], 0.08529207710374524
    )
    assert np.isclose(loss.loc[("formal", "joint"), "mean_w1"], 4.127925992208147)
    assert np.isclose(
        loss.loc[("ot_mass_1_to_10", "pca"), "mean_w1"], 4.355810628542722
    )

    assert panels.daughter_top_celltypes == (
        "Segmental Plate, Tail Bud",
        "Nervous System",
        "Adaxial Cell",
        "Notochord",
        "Neural Rod",
        "Musculature System, Yolk Syncytial Layer",
    )
    endpoint = panels.daughter_sensitivity_summary.loc[
        np.isclose(panels.daughter_sensitivity_summary["time"], 4.0)
        & np.isclose(panels.daughter_sensitivity_summary["daughter_noise_std"], 0.06)
    ].iloc[0]
    assert np.isclose(
        endpoint["composition_tv_from_reference_mean"], 0.2839330886226318
    )
    assert np.isclose(
        endpoint["lineage_weighted_tv_from_reference_mean"], 0.348301333749273
    )

    inverse = panels.inverse_pca_metrics.set_index("time")
    assert np.isclose(inverse.loc[0.0, "pearson_r"], 0.8898811016835869)
    assert np.isclose(inverse.loc[4.0, "pearson_r"], 0.9933663612658927)
    assert np.isclose(inverse.loc[4.0, "rmse"], 0.02486244369631326)


def test_zebrafish_si_tables_are_written(tmp_path: Path) -> None:
    panels = calculate_zebrafish_si_panels(load_zebrafish_si_results())
    paths = write_zebrafish_si_tables(panels, tmp_path)
    assert set(paths) == {
        "growth_quantiles",
        "ablation_centroid",
        "gene_zscores",
        "loss_weight",
        "daughter_composition",
        "daughter_lineage",
        "daughter_sensitivity",
        "daughter_particles",
        "inverse_pca",
    }
    assert pd.read_csv(paths["growth_quantiles"]).shape == (5, 4)
    assert pd.read_csv(paths["gene_zscores"]).shape == (250, 10)
    assert pd.read_csv(paths["inverse_pca"]).shape == (5, 7)


def test_zebrafish_si_import_is_matplotlib_lazy(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import CytoBridge.results.zebrafish_si; "
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


def test_zebrafish_si_all_plots_are_agg_safe_and_rc_local(tmp_path: Path) -> None:
    mpl.use("Agg", force=True)
    before = mpl.rcParams.copy()
    results = load_zebrafish_si_results()
    panels = calculate_zebrafish_si_panels(results)
    rendered = plot_zebrafish_si(results, tmp_path, panels)
    assert tuple(rendered) == FIGURE_IDS
    assert len(list(tmp_path.glob("*.pdf"))) == 8
    assert len(list(tmp_path.glob("*.png"))) == 8

    expected_sizes = {
        "s31": (2646, 3740),
        "s32": (2646, 3740),
        "s33": (2646, 3740),
        "s34": (2620, 1824),
        "s35": (2646, 3740),
        "s36": (5611, 3969),
        "s38": (2646, 3740),
    }
    for figure_id, (pdf, png) in rendered.items():
        assert pdf.is_file() and pdf.stat().st_size > 20_000
        with Image.open(png) as image:
            if figure_id in expected_sizes:
                assert image.size == expected_sizes[figure_id]
            else:
                assert image.width > 2000 and image.height > 2000
            image.verify()
    assert mpl.rcParams["font.family"] == before["font.family"]
    assert mpl.rcParams["font.size"] == before["font.size"]


def test_zebrafish_si_cli_supports_a_figure_subset(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/plot_zebrafish_si.py"),
            "--output-dir",
            str(output),
            "--figures",
            "s34",
            "s38",
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
    assert summary["analysis"] == "zebrafish_si_s31_s38"
    assert summary["source"] == "packaged"
    assert tuple(summary["figures"]) == ("s34", "s38")
    assert summary["statistics"]["figure_ids"] == list(FIGURE_IDS)
    assert str(tmp_path) not in completed.stdout
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_zebrafish_si_notebook_documents_the_route_and_shows_outputs() -> None:
    path = REPOSITORY_ROOT / "docs/tutorials/paper_figures/zebrafish_si_s31_s38.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = [
        "".join(cell["source"]).strip()
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ]
    assert any(text.startswith("## Run the notebook") for text in markdown)
    assert "## Load figure inputs" in markdown
    assert any("reference/figure_sources/zebrafish-si.md" in text for text in markdown)
    guide = (REPOSITORY_ROOT / "docs/reference/figure_sources/zebrafish-si.md").read_text()
    assert "Start with:" in guide and "Writes:" in guide
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
    assert len(image_outputs) == 8


def test_zebrafish_si_rejects_an_object_array(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    np.savez_compressed(
        results_dir / "s27_observed_generated.npz",
        xy=np.asarray([{"x": 1}], dtype=object),
        label_id=np.asarray([0], dtype=np.uint8),
        offsets=np.asarray([0, 1], dtype=np.int32),
        groups=np.asarray(["observed"]),
        times=np.asarray([0], dtype=np.float32),
        label_names=np.asarray(["EVL"]),
    )
    with pytest.raises(ValueError, match="Object arrays"):
        load_zebrafish_si_results(results_dir)


def test_zebrafish_si_rejects_missing_inputs(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    (results_dir / "s34_top_genes.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Missing processed result files"):
        load_zebrafish_si_results(results_dir)


def test_zebrafish_si_statistics_are_json_ready() -> None:
    results = load_zebrafish_si_results()
    summary = zebrafish_si_statistics(results)
    assert summary["bundle_bytes"] < 900 * 1024
    assert summary["figure_ids"] == list(FIGURE_IDS)
    assert np.isclose(summary["s38_t4_pearson_r"], 0.9933663612658927)
    json.dumps(summary)
