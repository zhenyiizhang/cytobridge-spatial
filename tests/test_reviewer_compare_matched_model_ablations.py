from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "reviewer_zebrafish_response"
    / "compare_matched_model_ablations.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reviewer_compare_matched_model_ablations", SCRIPT
)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _write_metrics(path: Path, *, scale: float) -> None:
    rows = []
    for time in (1.0, 2.0):
        for space, base in (("joint", 2.0), ("spatial", 0.1), ("pca", 1.5)):
            rows.append(
                {
                    "time": time,
                    "space": space,
                    "w1": scale * base * time,
                    "w2": scale * (base + 0.2) * time,
                    "tmv": scale * 0.02 * time,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_cli_builds_matched_ablation_tables_figures_and_manifest(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full.csv"
    no_interaction = tmp_path / "no_interaction.csv"
    no_lr = tmp_path / "no_lr.csv"
    _write_metrics(full, scale=1.0)
    _write_metrics(no_interaction, scale=1.25)
    _write_metrics(no_lr, scale=1.10)
    output = tmp_path / "report"

    assert (
        analysis.main(
            [
                "--condition",
                f"full={full}",
                "--condition",
                f"no_interaction={no_interaction}",
                "--condition",
                f"no_lr_prior={no_lr}",
                "--reference",
                "full",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    summary = pd.read_csv(output / "matched_ablation_summary.csv")
    assert len(summary) == 3 * 3
    relative = pd.read_csv(output / "relative_to_reference.csv")
    row = relative.query(
        "condition == 'no_interaction' and space == 'spatial' "
        "and metric == 'mean_w2'"
    ).iloc[0]
    assert row["percent_change_vs_reference"] == pytest.approx(25.0)
    mass = pd.read_csv(output / "mass_summary.csv")
    assert len(mass) == 3

    with Image.open(output / "matched_ablation_w1_w2.png") as image:
        assert image.format == "PNG"
        assert image.width >= 1000
    with Image.open(output / "matched_ablation_tmv.png") as image:
        assert image.format == "PNG"

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["parameters"]["reference"] == "full"
    assert manifest["parameters"]["condition_order"] == [
        "full",
        "no_interaction",
        "no_lr_prior",
    ]
    assert manifest["interpretation"]["direction"] == (
        "Lower W1, W2, and TMV are better."
    )


def test_rejects_mismatched_evaluation_grid(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_metrics(first, scale=1.0)
    _write_metrics(second, scale=1.0)
    frame = pd.read_csv(second).iloc[:-1]
    frame.to_csv(second, index=False)
    paths = analysis._parse_named_paths(
        [f"full={first}", f"ablation={second}"]
    )
    with pytest.raises(ValueError, match="same time/space grid"):
        analysis._load_and_validate(paths)
