from __future__ import annotations

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


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from CytoBridge.results.arista_local_domains import (  # noqa: E402
    DISPLAY_PROGRAMS,
    DOMAIN_COUNTS,
    DOMAIN_ORDER,
    calculate_arista_local_domain_panels,
    load_arista_local_domains,
    plot_arista_local_domains,
    write_arista_local_domain_tables,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_arista_local_domains().source_dir
    target = tmp_path / "arista_local_domains"
    shutil.copytree(source, target)
    return target


def test_packaged_arista_local_domain_contract() -> None:
    data = load_arista_local_domains()
    assert data.roi_assignments.shape == (1454, 30)
    assert data.domain_metadata.shape == (2, 9)
    assert data.celltype_edges.shape == (19, 7)
    assert data.attention_null.shape == (2, 18)
    assert data.pathway_null.shape == (160, 9)
    assert data.lr_pair_null.shape == (1062, 18)
    assert data.manifest["manuscript_figure"] == "Supplementary Figure S42"
    assert data.manifest["domains"]["cell_counts"] == DOMAIN_COUNTS
    assert DISPLAY_PROGRAMS[DOMAIN_ORDER[0]] == (
        "AGRN",
        "LAMININ",
        "TENASCIN",
        "FGF",
        "THBS",
    )
    assert data.attention_null["n_permutations"].eq(9999).all()
    assert data.pathway_null.groupby("module").size().to_dict() == {
        domain: 80 for domain in DOMAIN_ORDER
    }
    assert data.lr_pair_null.groupby("niche").size().to_dict() == {
        domain: 531 for domain in DOMAIN_ORDER
    }
    assert data.pathway_null["n_permutations"].eq(1999).all()
    assert data.lr_pair_null["n_permutations"].eq(1999).all()
    names = {path.name.lower() for path in data.source_dir.iterdir()}
    assert not any("h5ad" in name or "checkpoint" in name for name in names)


def test_arista_local_domain_panel_values_and_rosters() -> None:
    panels = calculate_arista_local_domain_panels(load_arista_local_domains())
    assert panels.attention.shape == (2, 6)
    assert np.allclose(
        panels.attention["fold_over_null"],
        [1.6869795096658176, 3.0622873648802242],
        rtol=1e-12,
    )
    assert np.allclose(
        panels.edge_structure["attention_percent"],
        [
            51.087105445821116,
            24.73018062405901,
            24.182713930119878,
            61.19665166433736,
            27.55294641552146,
            11.250401920141176,
        ],
        rtol=1e-12,
    )
    expected_programs = [
        program for domain in DOMAIN_ORDER for program in DISPLAY_PROGRAMS[domain]
    ]
    assert panels.pathways["pathway"].astype(str).tolist() == expected_programs
    assert panels.lr_axes["pair"].tolist() == [
        "AGRN_DAG1",
        "LAMA2_DAG1",
        "TNC_SDC4",
        "FGF7_FGFR1",
        "THBS1_SDC4",
        "GRN_SORT1",
        "L1CAM_L1CAM",
        "NRXN2_NLGN2",
        "SEMA3F_NRP2_PLXNA3",
        "FN1_ITGA5_ITGB1",
    ]
    n1_axes = panels.lr_axes.loc[panels.lr_axes["niche"].eq(DOMAIN_ORDER[0])]
    assert n1_axes["pathway"].tolist() == list(DISPLAY_PROGRAMS[DOMAIN_ORDER[0]])
    assert "NRG" not in set(n1_axes["pathway"])


def test_arista_local_domain_tables_are_written(tmp_path: Path) -> None:
    panels = calculate_arista_local_domain_panels(load_arista_local_domains())
    paths = write_arista_local_domain_tables(panels, tmp_path)
    assert {name: path.name for name, path in paths.items()} == {
        "attention": "panel_b_attention_matched_null.csv",
        "edge_structure": "panel_b_selected_edge_structure.csv",
        "pathways": "panel_c_lr_pathway_enrichment.csv",
        "lr_axes": "panel_d_candidate_lr_axes.csv",
    }
    assert pd.read_csv(paths["attention"]).shape == (2, 6)
    assert pd.read_csv(paths["pathways"]).shape == (10, 10)
    assert pd.read_csv(paths["lr_axes"])["pair"].tolist()[3] == "FGF7_FGFR1"


def test_arista_plot_is_agg_safe_and_rc_local(tmp_path: Path) -> None:
    mpl.use("Agg", force=True)
    before = mpl.rcParams.copy()
    data = load_arista_local_domains()
    panels = calculate_arista_local_domain_panels(data)
    pdf, png = plot_arista_local_domains(data, tmp_path, panels)
    assert pdf.name == "FigureS_ARISTA_Figure5c_local_interaction_niches_clean.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 20_000
    with Image.open(png) as image:
        assert image.size == (2646, 3740)
        image.verify()
    assert mpl.rcParams["font.family"] == before["font.family"]
    assert mpl.rcParams["font.size"] == before["font.size"]


def test_arista_module_import_is_matplotlib_lazy(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import CytoBridge.results.arista_local_domains; "
                "assert 'matplotlib' not in sys.modules"
            ),
        ],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)},
    )
    assert completed.stdout == ""


def test_arista_local_domain_cli_has_sanitized_summary(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(PACKAGE_ROOT / "scripts/results/plot_arista_local_domains.py"),
            "--output-dir",
            str(output),
        ],
        cwd=PACKAGE_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(tmp_path / "mpl"),
            "PYTHONPATH": str(PACKAGE_ROOT),
        },
    )
    summary = json.loads(completed.stdout)
    assert summary["analysis"] == "arista_local_domains"
    assert summary["source"] == "packaged"
    assert summary["rows"] == {
        "attention_null": 2,
        "celltype_edges": 19,
        "domain_metadata": 2,
        "lr_pair_null": 1062,
        "pathway_null": 160,
        "roi_assignments": 1454,
    }
    assert str(tmp_path) not in completed.stdout
    assert json.loads((output / "run_summary.json").read_text()) == summary


def test_arista_roi_missing_column_is_rejected(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "roi_assignments.csv"
    pd.read_csv(path).drop(columns=["paper_x"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_arista_local_domains(results_dir)


def test_arista_duplicate_cell_is_rejected(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "roi_assignments.csv"
    table = pd.read_csv(path)
    table.loc[1, "cell_index"] = table.loc[0, "cell_index"]
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate cell indices"):
        load_arista_local_domains(results_dir)


def test_arista_domain_count_is_rejected(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "roi_assignments.csv"
    table = pd.read_csv(path)
    row = table.index[table["two_niche_region"].eq(DOMAIN_ORDER[0])][0]
    table.loc[row, "two_niche_region"] = np.nan
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unexpected domain assignments"):
        load_arista_local_domains(results_dir)


def test_arista_attention_permutation_count_is_rejected(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "attention_null.csv"
    table = pd.read_csv(path)
    table.loc[0, "n_permutations"] = 9998
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="9,999 permutations"):
        load_arista_local_domains(results_dir)


def test_arista_bh_values_are_recalculated(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "pathway_null.csv"
    table = pd.read_csv(path)
    table.loc[0, "adjusted_p_value"] = 0.91
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent adjusted p-values"):
        load_arista_local_domains(results_dir)


def test_arista_manifest_roster_is_locked(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    programs = manifest["calculation"]["pathways"]["display_programs"]
    programs[DOMAIN_ORDER[0]][3] = "NRG"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pathway display roster"):
        load_arista_local_domains(results_dir)
