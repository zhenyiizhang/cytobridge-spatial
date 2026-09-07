"""Changing a selected numerical input must reach the immediately following plot."""

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

from reproduction.paper_figures import ROOT, draw_supplementary


@pytest.mark.parametrize("name,selected", [
    ("zebrafish_attention", "results_dir=results.source_dir"),
    ("spatial_communication", "results_dir=data"),
    ("loto_benchmark_summary", "results_dir=source"),
])
def test_notebook_passes_its_selected_input(name, selected):
    notebook = json.loads((ROOT / "docs/tutorials/paper_figures" / f"{name}.ipynb").read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert selected in source
    if name == "zebrafish_attention":
        assert "commot_results_dir=commot_results_dir" in source


@pytest.mark.parametrize("number,source_name,filename", [
    (39, "CytoBridge/results/data/zebrafish_attention", "jam_quartile_compatibility.csv"),
    (43, "reproduction/supplementary_figures/data/communication", "global_pair_metrics.csv"),
    (44, "reproduction/supplementary_figures/data/benchmark", "loto_target_stage_means_with_spatrack.csv"),
])
def test_selected_input_changes_plot_even_when_output_exists(tmp_path, number, source_name, filename):
    source = ROOT / source_name
    original_bytes = (source / filename).read_bytes()
    inputs = tmp_path / "inputs"
    shutil.copytree(source, inputs)
    if number == 39:
        shutil.copytree(ROOT / "reproduction/supplementary_figures/data/commot_comparison",
                        inputs / "commot_comparison")
    output = tmp_path / "figure"
    _, png = draw_supplementary([number], output, results_dir=inputs)[f"s{number}"]
    initial_image = hashlib.sha256(png.read_bytes()).hexdigest()

    table = pd.read_csv(inputs / filename)
    if number == 39:
        selected = table.condition.eq("trained")
        table.loc[selected, "top_n_jam_compatible"] -= 10
        table.loc[selected, "top_n_non_compatible"] += 10
        table.loc[selected, "top_compatibility_rate"] = (
            table.loc[selected, "top_n_jam_compatible"]
            / table.loc[selected, "top_n_edges_after_boundary_ties"]
        )
        row = table.loc[selected].iloc[0]
        odds, p_value = fisher_exact([[row.top_n_jam_compatible, row.top_n_non_compatible],
                                     [row.bottom_n_jam_compatible, row.bottom_n_non_compatible]])
        table.loc[selected, "top_vs_bottom_odds_ratio"] = odds
        table.loc[selected, "fisher_exact_two_sided_p"] = p_value
    elif number == 43:
        table.loc[table.dataset.eq("mosta"), "spearman_rho"] *= .5
    else:
        table.loc[table.method.eq("CytoBridge-0.015"), "sliced_w2"] *= 1.25
    table.to_csv(inputs / filename, index=False)

    _, png = draw_supplementary([number], output, results_dir=inputs)[f"s{number}"]
    assert hashlib.sha256(png.read_bytes()).hexdigest() != initial_image
    assert (source / filename).read_bytes() == original_bytes
    if number == 39:
        plotted = pd.read_csv(output / "tables/zebrafish_attention_jam_quartiles.csv")
        actual = plotted.loc[plotted.condition.eq("trained"), "top_compatibility_percent"].iloc[0]
        assert actual == pytest.approx(100 * 34 / 170)
    elif number == 44:
        plotted = pd.read_csv(output / "tables/S44_relative_errors.csv")
        reference = table.loc[table.method.eq("CytoBridge-0.015")].set_index(["dataset", "target", "space"])
        expected = plotted.set_index(["dataset", "target", "space"]).join(
            reference[["sliced_w2"]].rename(columns={"sliced_w2": "selected_reference"})
        )
        np.testing.assert_allclose(expected.cytobridge_sliced_w2, expected.selected_reference)
        np.testing.assert_allclose(expected.relative_difference_pct,
                                   100 * (expected.sliced_w2 / expected.selected_reference - 1),
                                   atol=1e-10)


def test_s39_custom_results_cannot_silently_use_included_pair_scores(tmp_path):
    inputs = tmp_path / "inputs"
    shutil.copytree(ROOT / "CytoBridge/results/data/zebrafish_attention", inputs)
    output = tmp_path / "figure"
    with pytest.raises(RuntimeError, match="Figure calculation failed"):
        draw_supplementary([39], output, results_dir=inputs)
    assert str(inputs / "commot_comparison") in (output / "plotting.log").read_text()


def test_commot_directory_is_only_accepted_for_s39(tmp_path):
    with pytest.raises(ValueError, match="only used by S39"):
        draw_supplementary([43], tmp_path / "figure", commot_results_dir=tmp_path / "inputs")


def test_completed_s39_panel_tables_collect_and_load_without_packaged_manifest(tmp_path):
    from CytoBridge.results.zebrafish_attention import load_zebrafish_attention_results
    from scripts.collect_zebrafish_attention_inputs import collect_panel_data

    panel_data = tmp_path / "panel_data"
    source = ROOT / "CytoBridge/results/data/zebrafish_attention"
    shutil.copytree(source, panel_data, ignore=shutil.ignore_patterns("manifest.json"))
    path = panel_data / "directed_pair_concordance.csv"
    pair = pd.read_csv(path)
    selected = pair.cytobridge_view.eq("attention") & pair.external_method.eq("CellAgentChat")
    pair.loc[selected, "adjusted_spearman_rho"] = .123
    pair.to_csv(path, index=False)
    comparison = ROOT / "reproduction/supplementary_figures/data/commot_comparison"
    original_pairs = pd.read_csv(comparison / "cytobridge_type_pair_summary.csv")
    earlier_pairs = original_pairs.assign(stage=3.0)
    cytobridge_pairs = tmp_path / "type_pair_summary.csv"
    pd.concat([earlier_pairs, original_pairs], ignore_index=True).to_csv(cytobridge_pairs, index=False)
    commot_pairs = comparison / "commot_type_pair_scores.csv.gz"
    output = collect_panel_data(panel_data, cytobridge_pairs, commot_pairs, tmp_path / "collected")
    results = load_zebrafish_attention_results(output)
    plotted = results.panels.external_agreement.set_index("external_method")
    assert plotted.loc["CellAgentChat", "adjusted_spearman_rho"] == pytest.approx(.123)
    assert results.source_dir == output
    assert (output / "commot_comparison/commot_type_pair_scores.csv.gz").is_file()
    collected_pairs = pd.read_csv(output / "commot_comparison/cytobridge_type_pair_summary.csv")
    pd.testing.assert_frame_equal(collected_pairs, original_pairs, check_exact=False, check_dtype=False)
    for original in panel_data.iterdir():
        assert (output / original.name).read_bytes() == original.read_bytes()
    with pytest.raises(FileExistsError):
        collect_panel_data(panel_data, cytobridge_pairs, commot_pairs, output)
    invalid_summary = panel_data / "somite_18hpf_spatial_null_summary.csv"
    summary = pd.read_csv(invalid_summary)
    summary["null_mean"] += 1
    summary.to_csv(invalid_summary, index=False)
    with pytest.raises(ValueError, match="unexpected spatial summary values"):
        collect_panel_data(panel_data, cytobridge_pairs, commot_pairs, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
