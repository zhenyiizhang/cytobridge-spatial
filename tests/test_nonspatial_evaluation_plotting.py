from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from CytoBridge.nonspatial.plotting import distribution_comparison, plot_nonspatial_evaluation


def example_metrics():
    return pd.DataFrame([
        {"time": time, "space": "pca", "condition": condition,
         "inference_seed": seed, "w1": time + offset + seed,
         "w2": time + offset + seed + 0.5}
        for time in (1., 2.) for condition, offset in (("full", 0.), ("no_interaction", 1.))
        for seed in (1, 2)
    ])


def test_distribution_pairs_and_averages_each_inference_seed():
    result = distribution_comparison(example_metrics())
    np.testing.assert_allclose(result["w1_full"], [2.5, 3.5])
    np.testing.assert_allclose(result["w1_no_interaction"], [3.5, 4.5])
    with pytest.raises(ValueError, match="matching times and seeds"):
        distribution_comparison(example_metrics().iloc[1:])
    with pytest.raises(ValueError, match="Duplicate"):
        distribution_comparison(pd.concat([example_metrics(), example_metrics().iloc[:1]]))


def test_plot_accepts_new_evaluation_without_saved_figures(tmp_path):
    source = tmp_path / "paired_distribution_metrics.csv"
    example_metrics().to_csv(source, index=False)
    output = tmp_path / "drawn"
    result = plot_nonspatial_evaluation("weinreb", source, output)
    assert set(result) == {"distribution_comparison_pdf", "distribution_comparison_png", "distribution_comparison_csv"}
    assert all(Path(path).is_file() for path in result.values())
    np.testing.assert_allclose(pd.read_csv(result["distribution_comparison_csv"])["w2_full"], [3., 4.])
    with pytest.raises(FileExistsError):
        plot_nonspatial_evaluation("weinreb", source, output)


def test_distribution_rejects_nonfinite_wide_input():
    result = distribution_comparison(example_metrics())
    result.loc[0, "w1_full"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        distribution_comparison(result)


def test_draw_clone_summaries_from_new_evaluation(tmp_path):
    source = tmp_path / "distribution.csv"
    example_metrics().to_csv(source, index=False)
    clone = tmp_path / "clone"
    for condition, offset in (("full", .05), ("no_interaction", 0)):
        folder = clone / condition
        folder.mkdir(parents=True)
        (folder / "summary.json").write_text(json.dumps({
            "clone_macro_tv_agreement": .7 + offset,
            "clone_macro_js_similarity": .8 + offset,
            "clone_macro_dominant_fate_match": .65 + offset,
        }))
    files = plot_nonspatial_evaluation("weinreb", source, tmp_path / "figures", clone_fate_dir=clone)
    assert files["clone_fate_comparison_pdf"].stat().st_size > 1000
    values = pd.read_csv(files["clone_fate_comparison_csv"])
    np.testing.assert_allclose(values.full - values.no_interaction, .05)
