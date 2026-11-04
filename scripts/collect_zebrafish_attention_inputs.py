"""Collect completed S39 analysis tables into the notebook's input format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from CytoBridge.results.zebrafish_attention import (
    CONDITION_ORDER,
    EXTERNAL_METHOD_ORDER,
    _FILES,
    load_zebrafish_attention_results,
)
from scripts.run_zebrafish_attention_validation import _terminal_stage


def collect_panel_data(panel_data_dir: Path, cytobridge_pairs: Path,
                       commot_pairs: Path, output_dir: Path) -> Path:
    """Copy computed tables and derive their manifest without recalculating scores.

    The existing S39 loader checks the paper's cell/edge scope and numerical
    consistency before the new directory is made available to the notebook.
    """
    source = Path(panel_data_dir).expanduser().resolve()
    comparison_files = {
        "cytobridge_type_pair_summary.csv": Path(cytobridge_pairs).expanduser().resolve(),
        "commot_type_pair_scores.csv.gz": Path(commot_pairs).expanduser().resolve(),
    }
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if source in output.parents:
        raise ValueError("Save collected results outside the input directories")
    filenames = tuple(name for name in _FILES if name != "manifest.json")
    required = [source / name for name in filenames]
    required.extend(comparison_files.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed analysis tables: {missing}")

    pair = pd.read_csv(source / "directed_pair_concordance.csv")
    pair = pair.loc[pair.cytobridge_view.eq("attention")]
    if set(pair.external_method) != set(EXTERNAL_METHOD_ORDER):
        raise ValueError("Attention comparisons must contain COMMOT and CellAgentChat")
    if pair.n_pairs.nunique() != 1 or pair.n_permutations.nunique() != 1:
        raise ValueError("Attention comparisons disagree on pair or permutation counts")
    compatibility = pd.read_csv(source / "jam_compatibility_percentile_summary.csv")
    if set(compatibility.condition) != set(CONDITION_ORDER):
        raise ValueError("JAM comparisons must contain all three model conditions")
    edge_counts = compatibility.groupby("condition").n_directed_edges.sum()
    if edge_counts.nunique() != 1:
        raise ValueError("JAM conditions must use the same directed edge scaffold")
    spatial = pd.read_csv(source / "somite_18hpf_spatial_null_summary.csv")
    if len(spatial) != 1:
        raise ValueError("Expected one spatial-null summary row")
    spatial = spatial.iloc[0]
    display = pd.read_csv(source / "trained_jam_display_edges.csv")
    manifest = {
        "schema_version": 1,
        "analysis": "zebrafish_attention",
        "manuscript_figure": "Supplementary Figure S39",
        "files": {name: name for name in filenames},
        "calculation": {
            "external_agreement": {
                "cytobridge_view": "attention",
                "directed_cell_type_pairs": int(pair.n_pairs.iloc[0]),
                "external_methods": list(EXTERNAL_METHOD_ORDER),
                "structured_null_permutations": int(pair.n_permutations.iloc[0]),
            },
            "jam_compatibility": {
                "stage": str(spatial.stage_label),
                "cell_type": str(spatial.cell_type),
                "conditions": list(CONDITION_ORDER),
                "cells": int(spatial.n_cells),
                "directed_edges_per_condition": int(edge_counts.iloc[0]),
            },
            "spatial_context": {
                "display_edges": len(display),
                "label_permutations": int(spatial.n_permutations),
                "observed_complementary_neighbor_pairs": int(
                    spatial.observed_jam2a_jam3b_orientation_compatible_pairs
                ),
                "null_mean": float(spatial.null_mean),
                "observed_over_null_mean": float(spatial.observed_over_null_mean),
                "plus_one_upper_tail_p": float(spatial.monte_carlo_upper_tail_p_plus1),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s39-collection-", dir=output.parent) as temporary:
        collected = Path(temporary) / "inputs"
        collected.mkdir()
        for name in filenames:
            shutil.copy2(source / name, collected / name)
        # The current native JAM producer spells out the technical/descriptive
        # scope in its Fisher-p column. Retain it and expose the historical
        # loader alias with identical values; this is not a new statistical test.
        quartile_path = collected / "jam_quartile_compatibility.csv"
        quartiles = pd.read_csv(quartile_path, float_precision="round_trip")
        native_p = "fisher_exact_two_sided_p_descriptive_technical"
        loader_p = "fisher_exact_two_sided_p"
        if native_p in quartiles and loader_p not in quartiles:
            quartiles[loader_p] = quartiles[native_p]
            quartiles.to_csv(quartile_path, index=False, float_format="%.17g")
            manifest["schema_aliases"] = {loader_p: native_p}
        (collected / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (collected / "commot_comparison").mkdir()
        for name, path in comparison_files.items():
            table = pd.read_csv(path, float_precision="round_trip")
            terminal = _terminal_stage(table, label=name)
            if len(terminal) == len(table):
                shutil.copy2(path, collected / "commot_comparison" / name)
            else:
                terminal.to_csv(collected / "commot_comparison" / name, index=False,
                                float_format="%.17g")
        load_zebrafish_attention_results(collected)
        collected.rename(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-data-dir", type=Path, required=True)
    parser.add_argument("--cytobridge-pairs", type=Path, required=True,
                        help="Native attribution type_pair_summary.csv")
    parser.add_argument("--commot-pairs", type=Path, required=True,
                        help="Native COMMOT commot_type_pair_scores.csv.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(collect_panel_data(args.panel_data_dir, args.cytobridge_pairs,
                            args.commot_pairs, args.output_dir))


if __name__ == "__main__":
    main()
