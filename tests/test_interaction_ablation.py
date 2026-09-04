import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from CytoBridge.results.interaction_ablation import (
    INTERACTION, interaction_ablation_statistics, load_interaction_ablation_results,
    pair_inference_errors, plot_interaction_ablation,
)
from scripts.collect_figure_inputs import collect_interaction_ablation


def test_published_numbers_and_target_uncertainty():
    result = load_interaction_ablation_results()
    stats = interaction_ablation_statistics(result)
    assert len(result.inference_metrics) == 1440
    assert len(result.paired_seeds) == 144
    assert stats["interaction_off_worse"] == 36
    assert stats["interaction_off_better"] == 12
    expected = {"zebrafish": 10.4209446468, "mosta": 2.5487477853,
                "arista": 30.0560863945, "admouse": 2.0183985746,
                "chicken_heart": 97.9157509253}
    for dataset, value in expected.items():
        assert stats["interaction_off_dataset_percent_increase"][dataset] == pytest.approx(value)
    for row in result.panel_summary.query("comparison == @INTERACTION").itertuples():
        values = result.interaction.query("dataset == @row.dataset and space == @row.space").off_relative_to_on
        assert row.sem == pytest.approx(values.std(ddof=1) / np.sqrt(len(values)))


def test_average_paired_ratios_not_ratio_of_seed_means():
    raw = load_interaction_ablation_results().inference_metrics.copy()
    mask = (raw.dataset == "zebrafish") & (raw.target == 1) & (raw.space == "joint")
    for seed, on, off in [(42, 1., 2.), (43, 2., 2.), (44, 4., 2.)]:
        raw.loc[mask & (raw.inference_seed == seed) & (raw.arm == "interaction_on"), "sliced_w2"] = on
        raw.loc[mask & (raw.inference_seed == seed) & (raw.arm == "interaction_off"), "sliced_w2"] = off
    _, targets = pair_inference_errors(raw)
    value = targets.query("dataset == 'zebrafish' and target == 1 and space == 'joint'").off_relative_to_on.iloc[0]
    assert value == pytest.approx((1 + 0 - .5) / 3)
    assert value != pytest.approx(6 / 7 - 1)


def test_incomplete_pairs_are_rejected():
    raw = load_interaction_ablation_results().inference_metrics
    with pytest.raises(ValueError, match="both arms"):
        pair_inference_errors(raw.iloc[1:])
    changed = raw.copy()
    changed.loc[0, "projection_id"] = "different"
    with pytest.raises(ValueError, match="Projection directions"):
        pair_inference_errors(changed)


@pytest.mark.parametrize("projection_column", ["projection_id", "projection_sha256"])
def test_collect_new_inference_runs(tmp_path, projection_column):
    result = load_interaction_ablation_results()
    records = json.loads((result.source_dir / "inference_run_manifests.json").read_text())
    run_root = tmp_path / "runs"
    for record in records:
        directory = run_root / record["dataset"]
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps(record))
        frame = result.inference_metrics.query("dataset == @record['dataset']")
        frame.rename(columns={"projection_id": projection_column}).to_csv(directory / "metrics.csv", index=False)
    collected = collect_interaction_ablation(result.source_dir / "no_lr_paired_target_deltas.csv", run_root, tmp_path / "collected")
    observed = load_interaction_ablation_results(collected)
    pd.testing.assert_frame_equal(observed.interaction, result.interaction, check_dtype=False)


def test_figure_is_vector_and_uses_new_comparison(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf, png = plot_interaction_ablation(load_interaction_ablation_results(), tmp_path)
    with fitz.open(pdf) as document:
        assert len(document) == 1
        assert document[0].get_images() == []
        text = document[0].get_text()
        assert "Without interaction" in text
        assert "stVCR" not in text
        spans = [span for block in document[0].get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"]]
        from matplotlib import font_manager
        if any(font.name == "Arial" for font in font_manager.fontManager.ttflist):
            assert all("Arial" in span["font"] for span in spans)
        assert all(span["color"] == 0 for span in spans)
    assert png.is_file()
