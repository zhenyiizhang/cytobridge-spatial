"""Collectors use the newly selected, hash-bound producer outputs."""
import json

import pandas as pd
import pytest

from scripts.build_zebrafish_attention_spec import build
from scripts.collect_spatial_communication_inputs import collect
from scripts.run_zebrafish_attention_validation import REQUIRED_INPUTS, _artifact, _load_spec


def test_s39_spec_binds_every_selected_producer_and_requires_real_acceptance(tmp_path):
    inputs = []
    for label in REQUIRED_INPUTS:
        path = tmp_path / label
        path.write_text(json.dumps({"status": "PASS"}) if label == "matched_acceptance" else label)
        inputs.append(f"{label}={path}")
    output = build(inputs, tmp_path / "spec.json")
    payload, records = _load_spec(output)
    assert payload["permutations"] == 1000
    assert records["sample_h5ad"] == _artifact(tmp_path / "sample_h5ad")
    (tmp_path / "sample_h5ad").write_text("changed selected data")
    with pytest.raises(ValueError, match="SHA mismatch"):
        _load_spec(output)
    (tmp_path / "matched_acceptance").write_text(json.dumps({"status": "FAIL"}))
    with pytest.raises(ValueError, match="not PASS"):
        build(inputs, tmp_path / "rejected.json")
    assert not (tmp_path / "rejected.json").exists()


def _s43_inputs(tmp_path):
    directories = {name: tmp_path / name for name in ("aggregate", "selection", "molecular")}
    for directory in directories.values():
        directory.mkdir()
    datasets = ["zebrafish", "mosta", "arista", "admouse", "chicken_heart"]
    metrics, scores = [], []
    for dataset in datasets:
        for method, score in (("COMMOT", .125), ("CellAgentChat", .875)):
            metrics.append(dict(dataset=dataset, cytobridge_view="CytoBridge exact message",
                                external_method=method, spearman_rho=score))
            scores.append(dict(dataset=dataset, method=method, available=True, sender_type="A",
                               receiver_type="B", score=score, rank_percentile=score * 100))
    files = {
        "aggregate": {"cytobridge_external_metrics.csv": ("cytobridge_external_metrics.csv", metrics),
                      "directed_pair_method_scores.csv": ("directed_pair_method_scores.csv", scores)},
        "selection": {"external_support": ("model_linked_external_support.csv",
            [dict(dataset=dataset, sender_type="A", receiver_type="B") for dataset in datasets])},
        "molecular": {"panel": ("model_biology_molecular_panel.csv", [dict(new_calculated_value=17)]),
                      "model_first_nichenet_chains": ("model_first_nichenet_chains.csv", [dict(chain="new")])},
    }
    for name, directory in directories.items():
        outputs = {}
        for key, (filename, rows) in files[name].items():
            path = directory / filename
            pd.DataFrame(rows).to_csv(path, index=False)
            outputs[key] = _artifact(path)
        manifest = dict(status="complete", outputs=outputs)
        if name == "selection":
            manifest["workflow"] = "five_dataset_model_linked_lr_selection"
        elif name == "molecular":
            manifest["workflow"] = "five_dataset_model_biology_molecular_summary"
            manifest["inputs"] = {"selection_manifest": _artifact(directories["selection"] / "manifest.json")}
        (directory / "manifest.json").write_text(json.dumps(manifest))
    return directories


def test_s43_exports_new_calculated_values_and_refuses_cross_run_selection(tmp_path):
    inputs = _s43_inputs(tmp_path)
    output = collect(inputs["aggregate"], inputs["selection"], inputs["molecular"], tmp_path / "collected")
    support = pd.read_csv(output / "model_linked_external_support.csv")
    assert support.commot_pair_score.tolist() == [.125] * 5
    assert support.cellagentchat_pair_percentile.tolist() == [87.5] * 5
    assert pd.read_csv(output / "model_biology_molecular_panel.csv").new_calculated_value.iloc[0] == 17
    selection = inputs["selection"] / "manifest.json"
    selection.write_text(selection.read_text() + "\n")
    with pytest.raises(ValueError, match="does not match the frozen manifest"):
        collect(inputs["aggregate"], inputs["selection"], inputs["molecular"], tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_s43_refuses_changed_aggregate_bytes(tmp_path):
    inputs = _s43_inputs(tmp_path)
    metrics = inputs["aggregate"] / "cytobridge_external_metrics.csv"
    metrics.write_text(metrics.read_text().replace("0.125", "0.625"))
    with pytest.raises(ValueError, match="does not match the frozen manifest"):
        collect(inputs["aggregate"], inputs["selection"], inputs["molecular"], tmp_path / "invalid")
