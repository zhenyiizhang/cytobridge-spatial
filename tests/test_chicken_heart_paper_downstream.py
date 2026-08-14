from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_chicken_heart_paper_downstream.py"
SPEC = importlib.util.spec_from_file_location("chicken_heart_paper", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _complete_summary() -> dict[str, object]:
    required = (
        "velocity",
        "growth",
        "composition",
        "communication",
        "figures",
        "gene_dynamics",
        "ligand_receptor",
    )
    return {
        "dataset": "chicken_heart",
        "analyses": {name: {"status": "completed"} for name in required},
    }


def test_formal_perturbation_contract_is_global_d4_and_fixed_population():
    assert MODULE.TIME_POINTS == (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    assert MODULE.DISPLAY_TIMES == (0.0, 1.0, 2.0, 3.0)
    assert MODULE.CELLTYPE_ABLATIONS == {
        "remove_endocardial": "Endocardial cells",
        "remove_valve": "Valve cells",
        "remove_immature_myocardial": "Immature myocardial cells",
    }


def test_standard_downstream_requires_every_formal_analysis():
    MODULE._validate_standard_downstream_summary(_complete_summary())
    bad = _complete_summary()
    bad["analyses"]["ligand_receptor"]["status"] = "failed"
    with pytest.raises(RuntimeError, match="ligand_receptor"):
        MODULE._validate_standard_downstream_summary(bad)


def test_interaction_composition_rows_sum_to_one():
    frames = {
        "interaction_on": tuple(
            np.asarray(["A", "A", "B"]) for _ in MODULE.TIME_POINTS
        ),
        "interaction_off": tuple(
            np.asarray(["A", "B", "B"]) for _ in MODULE.TIME_POINTS
        ),
    }
    table = MODULE._composition_rows(frames)
    sums = table.groupby(["condition", "time"])["fraction"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)
    assert set(table.columns) == {
        "condition",
        "time_index",
        "time",
        "celltype",
        "count",
        "fraction",
    }


def test_lr_attention_renderer_uses_formal_tables(tmp_path):
    downstream = tmp_path / "downstream"
    lr = downstream / "ligand_receptor"
    communication = downstream / "communication"
    lr.mkdir(parents=True)
    communication.mkdir()
    pairs = [f"L{index}_R{index}" for index in range(9)]
    pd.DataFrame(
        [
            {"time": time, "pair": pair, "score": float(index + time)}
            for index, pair in enumerate(pairs)
            for time in MODULE.TIME_POINTS
        ]
    ).to_csv(lr / "pair_timecourse.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "auc": float(index + 1),
                "peak_time": 3.0,
                "peak_score": float(index + 3),
            }
            for index, pair in enumerate(pairs)
        ]
    ).to_csv(lr / "pattern_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_lr_pairs_database": 100,
                "n_lr_pairs_scored": 9,
                "n_active_lr_features": 20,
                "n_requested_lr_symbols": 40,
            }
        ]
    ).to_csv(lr / "coverage.csv", index=False)
    pd.DataFrame(
        [
            {
                "time": time,
                "source": source,
                "target": target,
                "attention_per_source": float(1 + int(source == target)),
            }
            for time in MODULE.DISPLAY_TIMES
            for source in ("Atria", "Ventricle")
            for target in ("Atria", "Ventricle")
        ]
    ).to_csv(communication / "communication_by_celltype.csv", index=False)

    files = MODULE._write_lr_attention_figures(downstream, tmp_path / "figures")
    assert len(files) == 6
    assert all(path.is_file() for path in files)
    selected = pd.read_csv(tmp_path / "figures" / "top_lr_pair_timecourses.csv")
    assert selected["pair"].nunique() == 8


def test_metric_summary_uses_public_ablation_w2_schema(tmp_path):
    table = pd.DataFrame(
        [
            {
                "variant": "remove_A",
                "time": time,
                "space": space,
                "w2": 0.1 + time,
                "centroid_shift": 0.2 + time,
            }
            for time in MODULE.TIME_POINTS
            for space in ("joint", "spatial", "latent")
        ]
    )
    output = tmp_path / "metric.pdf"
    MODULE._plot_metric_summary(table, output, "test")
    assert output.is_file()


def test_output_root_must_be_new_or_empty(tmp_path):
    fresh = MODULE._require_empty_output(tmp_path / "fresh")
    assert fresh.is_dir()
    (fresh / "evidence.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="non-empty"):
        MODULE._require_empty_output(fresh)


def test_cli_rejects_nonpositive_classifier_epochs(tmp_path):
    required = [
        "--run-root",
        str(tmp_path / "run"),
        "--input-h5ad",
        str(tmp_path / "input.h5ad"),
        "--model-dir",
        str(tmp_path / "model"),
        "--standard-downstream",
        str(tmp_path / "downstream"),
        "--output-dir",
        str(tmp_path / "output"),
        "--classifier-epochs",
        "0",
    ]
    with pytest.raises(SystemExit):
        MODULE.parse_args(required)
