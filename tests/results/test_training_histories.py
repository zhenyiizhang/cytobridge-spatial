from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from CytoBridge.results.training_histories import (  # noqa: E402
    DATASET_ORDER,
    STAGES,
    calculate_smoothed_training_history,
    centered_moving_mean,
    load_training_history_results,
    plot_training_histories,
)


def _fixture_copy(tmp_path: Path) -> Path:
    source = load_training_history_results().source_dir
    target = tmp_path / "training_histories"
    shutil.copytree(source, target)
    return target


def test_training_history_import_keeps_plotting_optional() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import CytoBridge.results.training_histories; "
                "print('matplotlib' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == "False"


def test_packaged_data_loads_from_another_working_directory(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from CytoBridge.results.training_histories import "
                "load_training_history_results; "
                "r = load_training_history_results(); "
                "print(len(r.history), len(r.checkpoint_summary))"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    assert completed.stdout.strip() == "5252 30"


def test_notebook_uses_the_installed_package() -> None:
    notebook = json.loads(
        (
            REPOSITORY_ROOT / "docs/tutorials/paper_figures/training_histories.ipynb"
        ).read_text()
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "from CytoBridge.results.training_histories import" in source
    assert "repo_root" not in source
    assert "sys.path" not in source
    assert 'output_dir = Path("outputs") / "training_histories_notebook"' in source


def test_packaged_training_history_contract() -> None:
    results = load_training_history_results()
    assert list(results.history.columns) == [
        "stage_index",
        "stage",
        "epoch",
        "loss",
    ]
    assert results.history.shape == (5252, 4)
    assert results.checkpoint_summary.shape == (30, 7)
    assert results.manifest["analysis"] == "training_histories"
    assert results.history.groupby("stage", sort=False).size().tolist() == [
        stage.configured_epochs for stage in STAGES
    ]


def test_centered_moving_mean_matches_reference_values() -> None:
    results = load_training_history_results()
    smoothed = calculate_smoothed_training_history(results)
    expected = {
        "Pretrain": (1.0708016583865339, 0.8506773378361355),
        "Refine": (0.9204799715768207, 0.8648124316876583),
        "Init_interaction": (9.184300056525638, 8.764288680894033),
        "Train_Score": (1.0127531349068826, 0.9962261950615603),
        "Finetune": (0.8896971318651649, 0.8142528881629308),
        "Score_Refine": (1.002138326663782, 0.9991884036819537),
    }
    for stage, (first, last) in expected.items():
        values = smoothed.loc[smoothed["stage"].eq(stage), "smoothed_loss"]
        assert np.isclose(float(values.iloc[0]), first, rtol=0.0, atol=1.0e-14)
        assert np.isclose(float(values.iloc[-1]), last, rtol=0.0, atol=1.0e-14)

    simple = centered_moving_mean(np.array([1.0, 2.0, 3.0, 4.0]), 3)
    assert np.allclose(simple, [4.0 / 3.0, 2.0, 3.0, 11.0 / 3.0])


def test_checkpoint_summary_arithmetic() -> None:
    summary = load_training_history_results().checkpoint_summary
    calculated = 100.0 * (
        1.0
        - summary["selected_checkpoint_metric"].to_numpy(dtype=float)
        / summary["first_checkpoint_metric"].to_numpy(dtype=float)
    )
    assert np.allclose(
        calculated,
        summary["percent_reduction"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_training_history_plot_is_agg_safe_and_local(tmp_path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    before = mpl.rcParams.copy()
    pdf, png = plot_training_histories(load_training_history_results(), tmp_path)
    assert pdf.is_file() and pdf.stat().st_size > 0
    with Image.open(png) as image:
        assert image.size == (4536, 2688)
        image.verify()
    assert mpl.rcParams["font.family"] == before["font.family"]
    assert mpl.rcParams["font.size"] == before["font.size"]


def test_training_history_cli(tmp_path: Path) -> None:
    output = tmp_path / "cli"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/results/plot_training_histories.py"),
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
    assert summary == json.loads((output / "run_summary.json").read_text())
    assert summary["analysis"] == "training_histories"
    assert summary["input"] == "packaged"
    assert set(summary["tables"]) == {"checkpoint_summary", "smoothed_history"}
    for name in (
        "representative_training_curves.pdf",
        "representative_training_curves.png",
        "arista_training_history_smoothed.csv",
        "panel_metrics.csv",
    ):
        assert (output / name).is_file()


def test_collect_training_history_inputs_from_five_runs(tmp_path: Path) -> None:
    slugs = ("zebrafish", "mosta", "arista", "admouse", "chicken_heart")
    run_arguments = []
    for slug in slugs:
        training = tmp_path / slug / "training"
        training.mkdir(parents=True)
        rows = []
        for stage in STAGES:
            epochs = (
                3001
                if slug == "admouse"
                and stage.stage in {"Train_Score", "Score_Refine"}
                else stage.configured_epochs
            )
            for epoch in range(1, epochs + 1):
                checkpoint = 10.0 - 5.0 * (epoch - 1) / max(epochs - 1, 1)
                rows.append(
                    {
                        "stage_index": stage.stage_index,
                        "stage": stage.stage,
                        "mode": stage.mode,
                        "epoch": epoch,
                        "epochs": epochs,
                        "loss": checkpoint,
                        "checkpoint_metric": "test_metric",
                        "checkpoint_value": checkpoint,
                        "is_best": epoch == epochs,
                        "is_selected_checkpoint": epoch == epochs,
                        "save_strategy": "last",
                    }
                )
        pd.DataFrame(rows).to_csv(training / "training_history.csv", index=False)
        run_arguments.extend(["--run", f"{slug}={training}"])

    output = tmp_path / "s41_inputs"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/collect_training_history_inputs.py"),
            *run_arguments,
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
    )
    written = json.loads(completed.stdout)
    assert set(written) == {"history", "checkpoint_summary", "manifest"}

    results = load_training_history_results(output)
    assert results.history.shape == (5252, 4)
    assert results.checkpoint_summary.shape == (30, 7)
    assert results.checkpoint_summary["dataset"].drop_duplicates().tolist() == list(
        DATASET_ORDER
    )
    assert np.allclose(results.checkpoint_summary["percent_reduction"], 50.0)


def test_training_history_missing_column(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "arista_training_history.csv"
    pd.read_csv(path).drop(columns=["loss"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_training_history_results(results_dir)


def test_training_history_duplicate_epoch(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "arista_training_history.csv"
    table = pd.read_csv(path)
    pd.concat([table, table.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate stage-epoch"):
        load_training_history_results(results_dir)


def test_training_history_incomplete_epoch_sequence(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "arista_training_history.csv"
    table = pd.read_csv(path).drop(index=[1]).reset_index(drop=True)
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="incomplete epoch sequence"):
        load_training_history_results(results_dir)


def test_training_history_nonfinite_loss(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "arista_training_history.csv"
    table = pd.read_csv(path)
    table.loc[0, "loss"] = np.nan
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite loss"):
        load_training_history_results(results_dir)


def test_checkpoint_summary_duplicate_key(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "panel_metrics.csv"
    table = pd.read_csv(path)
    pd.concat([table, table.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate dataset-stage"):
        load_training_history_results(results_dir)


def test_checkpoint_summary_reduction_mismatch(tmp_path: Path) -> None:
    results_dir = _fixture_copy(tmp_path)
    path = results_dir / "panel_metrics.csv"
    table = pd.read_csv(path)
    table.loc[0, "percent_reduction"] += 1.0
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="inconsistent percent reductions"):
        load_training_history_results(results_dir)
