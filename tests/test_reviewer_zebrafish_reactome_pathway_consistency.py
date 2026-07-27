from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "reviewer_zebrafish_ccc"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "reactome_pathway_consistency.py"
SPEC = importlib.util.spec_from_file_location("reactome_pathway_consistency", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_components_split_receptor_complex_but_preserve_colon_symbol() -> None:
    assert analysis._components("ifngr1_ifngr2") == ["ifngr1", "ifngr2"]
    assert analysis._components("si:dkey-145c18.3") == ["si:dkey-145c18.3"]


def test_strict_one_to_one_conversion_excludes_ambiguous_and_failed() -> None:
    response = {
        "result": [
            {"incoming": "a", "converted": "ENSDARG1"},
            {"incoming": "b", "converted": "ENSDARG2"},
            {"incoming": "b", "converted": "ENSDARG3"},
            {"incoming": "c", "converted": "None"},
        ]
    }
    mapping, audit = analysis.strict_one_to_one_conversion(
        response, ["a", "b", "c"]
    )
    assert mapping == {"a": "ENSDARG1"}
    assert audit.set_index("gene_symbol")["status"].to_dict() == {
        "a": "one_to_one",
        "b": "ambiguous",
        "c": "failed",
    }


def test_query_manifest_keeps_positive_ties_and_splits_lr_genes() -> None:
    grid_rows = []
    for stage in analysis.STAGES:
        for axis, score in (
            ("l1->r1_r2", 3.0),
            ("l2->r3", 2.0),
            ("l3->r4", 2.0),
            ("l4->r5", 0.0),
        ):
            row = {"stage": stage, "axis": axis}
            for method in analysis.ALL_SCORE_METHODS:
                row[method] = score
            grid_rows.append(row)
    definitions = pd.DataFrame(
        {
            "axis": ["l1->r1_r2", "l2->r3", "l3->r4", "l4->r5"],
            "ligand": ["l1", "l2", "l3", "l4"],
            "receptor": ["r1_r2", "r3", "r4", "r5"],
        }
    )
    manifest, pairs = analysis.build_query_manifest(
        pd.DataFrame(grid_rows),
        definitions,
        methods=["CytoBridge attention x LR"],
        top_fraction=0.5,
    )
    first = manifest.loc[
        manifest["stage"].eq(0.0)
        & manifest["gene_mode"].eq("ligand_receptor")
    ].iloc[0]
    assert first["top_k_requested"] == 2
    assert first["top_k_after_positive_and_ties"] == 3
    assert set(first["query_gene_symbols"].split(";")) == {
        "l1",
        "l2",
        "l3",
        "r1",
        "r2",
        "r3",
        "r4",
    }
    assert len(pairs.loc[pairs["stage"].eq(0.0)]) == 3


def test_external_pathway_selection_never_uses_cytobridge() -> None:
    rows = []
    methods = [
        "CytoBridge attention x LR",
        "COMMOT",
        "CellChat triMean",
        "CellAgentChat significant",
    ]
    for stage in analysis.STAGES:
        for pathway_id, name in (("REAC:A", "A"), ("REAC:B", "B")):
            for method in methods:
                p_value = 1.0
                if pathway_id == "REAC:A" and method in (
                    "CytoBridge attention x LR",
                    "COMMOT",
                ):
                    p_value = 1e-4
                if pathway_id == "REAC:B" and method in (
                    "COMMOT",
                    "CellChat triMean",
                ):
                    p_value = 1e-4
                rows.append(
                    {
                        "profile_id": "p",
                        "method": method,
                        "stage": stage,
                        "reactome_id": pathway_id,
                        "reactome_name": name,
                        "adjusted_p_value": p_value,
                        "is_reactome_root": False,
                    }
                )
    selection = analysis.select_external_pathways(
        pd.DataFrame(rows),
        profile_id="p",
        external_methods=[
            "COMMOT",
            "CellChat triMean",
            "CellAgentChat significant",
        ],
        all_methods=methods,
        min_external_methods=2,
        max_pathways=10,
        analysis_id="test",
    )
    selected = set(
        selection.loc[selection["selected_for_heatmap"], "reactome_id"]
    )
    assert selected == {"REAC:B"}


def test_parse_profile_response_recovers_intersection_symbols() -> None:
    manifest = pd.DataFrame(
        [
            {
                "query_id": "q001",
                "gene_mode": "ligand_receptor",
                "method": "COMMOT",
                "stage": 0.0,
                "stage_label": "5.25 hpf",
            }
        ]
    )
    response = {
        "result": [
            {
                "query": "q001",
                "native": "REAC:R-DRE-1",
                "name": "Long biological pathway",
                "p_value": 0.005,
                "effective_domain_size": 3,
                "query_size": 2,
                "term_size": 5,
                "intersection_size": 1,
                "precision": 0.5,
                "recall": 0.2,
                "intersections": [["REAC"], []],
            }
        ],
        "meta": {
            "version": analysis.EXPECTED_GPROFILER_VERSION,
            "genes_metadata": {
                "failed": [],
                "ambiguous": {},
                "query": {"q001": {"ensgs": ["ENSDARG1", "ENSDARG2"]}},
            },
        },
    }
    result = analysis.parse_profile_response(
        response,
        manifest,
        profile_id="p",
        ensembl_to_symbol={"ENSDARG1": "gene1", "ENSDARG2": "gene2"},
        expected_background_size=3,
    )
    assert result.iloc[0]["intersection_gene_symbols"] == "gene1"
    assert bool(result.iloc[0]["significant_adjusted_p_lt_0_01"])
    assert np.isclose(result.iloc[0]["minus_log10_adjusted_p"], -np.log10(0.005))
