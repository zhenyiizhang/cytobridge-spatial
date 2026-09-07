#!/usr/bin/env python3
"""Export current S43 inputs from completed external-method summary calculations.

This performs the table selection/merge used by plot_model_biology, without
rendering its older figure. External-method inference and LR selection remain
separate, manifest-bound calculations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_spatial_communication_consistency import _artifact, _verify_artifact_bytes


def collect(aggregate_dir, selection_dir, molecular_dir, output_dir):
    aggregate_dir, selection_dir, molecular_dir = map(Path, (aggregate_dir, selection_dir, molecular_dir))
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    manifests = {name: json.loads((directory / "manifest.json").read_text())
                 for name, directory in (("aggregate", aggregate_dir), ("selection", selection_dir), ("molecular", molecular_dir))}
    if manifests["molecular"].get("workflow") != "five_dataset_model_biology_molecular_summary" or manifests["molecular"].get("status") != "complete":
        raise ValueError("Molecular summary must be complete")
    if manifests["selection"].get("workflow") != "five_dataset_model_linked_lr_selection" or manifests["selection"].get("status") != "complete":
        raise ValueError("LR selection must be complete")
    _verify_artifact_bytes(selection_dir / "manifest.json",
        manifests["molecular"]["inputs"]["selection_manifest"], label="molecular summary selection manifest")
    reads = [
        ("aggregate", aggregate_dir, "cytobridge_external_metrics.csv", "cytobridge_external_metrics.csv"),
        ("aggregate", aggregate_dir, "directed_pair_method_scores.csv", "directed_pair_method_scores.csv"),
        ("selection", selection_dir, "model_linked_external_support.csv", "external_support"),
        ("molecular", molecular_dir, "model_biology_molecular_panel.csv", "panel"),
        ("molecular", molecular_dir, "model_first_nichenet_chains.csv", "model_first_nichenet_chains"),
    ]
    tables, sources = {}, []
    for source, directory, filename, key in reads:
        path = directory / filename
        _verify_artifact_bytes(path, manifests[source]["outputs"][key], label=filename)
        sources.append(_artifact(path))
        tables[filename] = pd.read_csv(path)
    metrics = tables["cytobridge_external_metrics.csv"]
    primary = metrics.loc[metrics.cytobridge_view.eq("CytoBridge exact message")
                          & metrics.external_method.isin(["COMMOT", "CellAgentChat"])].copy()
    if primary.groupby("external_method").dataset.nunique().to_dict() != {"COMMOT": 5, "CellAgentChat": 5}:
        raise ValueError("S43 aggregate must cover five datasets for both external methods")
    support = tables["model_linked_external_support.csv"]
    scores = tables["directed_pair_method_scores.csv"]
    for method, prefix in (("COMMOT", "commot"), ("CellAgentChat", "cellagentchat")):
        selected = scores.loc[scores.method.eq(method)
            & scores.available.astype(str).str.casefold().isin({"true", "1"}),
            ["dataset", "sender_type", "receiver_type", "score", "rank_percentile"]]
        support = support.merge(selected.rename(columns={"score": f"{prefix}_pair_score",
            "rank_percentile": f"{prefix}_pair_percentile"}),
            on=["dataset", "sender_type", "receiver_type"], how="left", validate="one_to_one")
    output.mkdir(parents=True)
    exports = {"global_pair_metrics.csv": primary, "model_linked_external_support.csv": support,
               "model_biology_molecular_panel.csv": tables["model_biology_molecular_panel.csv"],
               "model_first_nichenet_chains.csv": tables["model_first_nichenet_chains.csv"]}
    for filename, table in exports.items():
        table.to_csv(output / filename, index=False)
    (output / "manifest.json").write_text(json.dumps({
        "analysis": "spatial_communication_comparison_s43",
        "operation": "export calculated summary tables; no external-method inference",
        "sources": sources,
        "manifests": {name: _artifact(directory / "manifest.json") for name, directory in
                      (("aggregate", aggregate_dir), ("selection", selection_dir), ("molecular", molecular_dir))},
        "outputs": {filename: _artifact(output / filename) for filename in exports},
    }, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--molecular-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(collect(**vars(parser.parse_args())))
