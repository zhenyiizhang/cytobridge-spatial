"""Check the AD source archive and its explicit-input population renderer."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "reproduction/admouse/plot_population.py"
SOURCE = ROOT / "reproduction/admouse/final_figures"


def test_ad_paper_sources_are_archived():
    for path in (
        "main/scripts/ad_main_figureb.py",
        "main/scripts/ad_main_figurecd.py",
        "main/scripts/ad_main_figuree.py",
        "main/scripts/ad_main_figuref.py",
        "main/scripts/ad_main_figureg.py",
        "supplementary/downstream/scripts/ad_supplementary1.py",
        "supplementary/downstream/scripts/ad_supplementary2.py",
        "supplementary/go/scripts/ad_supplementary_go_pattern1.R",
        "supplementary/go/scripts/ad_supplementary_go_pattern4.R",
        "supplementary/nichenet/scripts/nichenet1.R",
        "supplementary/nichenet/scripts/nichenet2.R",
        "supplementary/nichenet/pipeline/interpolation/run_admouse_interpolation_0p05.py",
        "supplementary/nichenet/pipeline/official_nichenet/01_prepare_model_expression_inputs.py",
        "supplementary/nichenet/pipeline/official_nichenet/02_run_official_nichenet_51_windows.R",
    ):
        assert (SOURCE / path).is_file(), path


def test_population_renderer_counts_input_labels(tmp_path):
    run = tmp_path / "input"
    states = run / "compat_base/01_interpolation"
    labels = run / "whole_tissue/baseline_labels_k1"
    states.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(25):
        t = i / 10
        celltypes = np.array(["Astrocytes"] * (i + 1) + ["Microglia", "OPC"])
        xy = np.column_stack([np.arange(len(celltypes)), np.arange(len(celltypes)) / 2])
        np.save(states / f"generated_t{t:g}.npy", xy.astype(float))
        np.save(labels / f"labels_t{t:g}.npy", celltypes)
    output = tmp_path / "output"
    subprocess.run([sys.executable, str(SCRIPT), "--run-dir", str(run),
                    "--output-dir", str(output), "--png-dpi", "72"],
                   check=True, capture_output=True, timeout=120)
    table = pd.read_csv(output / "celltype_counts_and_proportions.csv")
    assert len(table) == 200
    assert table.query("model_time == 2.4 and celltype == 'Astrocytes'")["count"].item() == 25
    assert table.query("model_time == 2.4")["total"].unique().tolist() == [27]
    np.testing.assert_allclose(table.groupby("model_time")["proportion_percent"].sum(), 100)
    assert (output / "ad_supplementary1.pdf").stat().st_size > 1000
    assert (output / "ad_supplementary1.png").stat().st_size > 1000


def test_population_renderer_rejects_writing_into_input(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--run-dir", str(tmp_path),
                             "--output-dir", str(tmp_path / "generated")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "outside the source run" in result.stderr
