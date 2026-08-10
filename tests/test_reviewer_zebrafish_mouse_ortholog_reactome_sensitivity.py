from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "reviewer_zebrafish_ccc"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "mouse_ortholog_reactome_sensitivity.py"
SPEC = importlib.util.spec_from_file_location(
    "mouse_ortholog_reactome_sensitivity", SCRIPT
)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def _crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_ligand": ["wnt5b", "wnt5b"],
            "source_receptor": ["fzd7a", "ror2"],
            "mapped_ligand": ["Wnt5a", "Wnt5a"],
            "mapped_receptor": ["Fzd7", "Ror2"],
        }
    )


def test_crosswalk_gene_map_is_symbol_bijective() -> None:
    mapping, audit = analysis.build_strict_unique_gene_map(_crosswalk())
    assert mapping == {
        "fzd7a": "Fzd7",
        "ror2": "Ror2",
        "wnt5b": "Wnt5a",
    }
    assert audit["status"].eq("one_to_one").all()

    collision = pd.concat(
        [
            _crosswalk(),
            pd.DataFrame(
                {
                    "source_ligand": ["wnt5c"],
                    "source_receptor": ["fzd8a"],
                    "mapped_ligand": ["Wnt5a"],
                    "mapped_receptor": ["Fzd8"],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="symbol-bijective"):
        analysis.build_strict_unique_gene_map(collision)


def test_mouse_conversion_requires_unique_ensmusg_ids() -> None:
    response = {
        "result": [
            {"incoming": "A", "converted": "ENSMUSG00000001"},
            {"incoming": "B", "converted": "ENSMUSG00000002"},
            {"incoming": "B", "converted": "ENSMUSG00000003"},
            {"incoming": "C", "converted": "ENSDARG00000004"},
            {"incoming": "D", "converted": "ENSMUSG00000005"},
            {"incoming": "E", "converted": "ENSMUSG00000005"},
        ]
    }
    conversion, audit = analysis.strict_mouse_one_to_one_conversion(
        response, ["A", "B", "C", "D", "E"]
    )
    assert conversion == {"A": "ENSMUSG00000001"}
    assert audit.set_index("mouse_gene_symbol")["status"].to_dict() == {
        "A": "one_to_one",
        "B": "ambiguous_source",
        "C": "failed",
        "D": "ambiguous_target",
        "E": "ambiguous_target",
    }


def test_query_projection_and_optional_top_pair_validation() -> None:
    source_manifest = pd.DataFrame(
        [
            {
                "query_id": "q001",
                "gene_mode": "ligand_receptor",
                "method": "CytoBridge attention x LR",
                "stage": 0.0,
                "stage_label": "5.25 hpf",
                "query_gene_symbols": "wnt5b;fzd7a",
            },
            {
                "query_id": "q002",
                "gene_mode": "receptor_only",
                "method": "CytoBridge attention x LR",
                "stage": 0.0,
                "stage_label": "5.25 hpf",
                "query_gene_symbols": "fzd7a",
            },
            {
                "query_id": "q003",
                "gene_mode": "ligand_receptor",
                "method": "CytoBridge attention x LR",
                "stage": 1.0,
                "stage_label": "10 hpf",
                "query_gene_symbols": float("nan"),
            },
        ]
    )
    mapping, _ = analysis.build_strict_unique_gene_map(_crosswalk())
    projected = analysis.map_query_manifest(source_manifest, mapping)
    assert projected["query_id"].tolist() == ["q001", "q003"]
    assert projected.iloc[0]["query_gene_symbols_mouse"] == "Fzd7;Wnt5a"
    assert projected.iloc[1]["query_gene_symbols_mouse"] == ""

    top_pairs = pd.DataFrame(
        [
            {
                "method": "CytoBridge attention x LR",
                "stage": 0.0,
                "ligand": "wnt5b",
                "receptor": "fzd7a",
            }
        ]
    )
    top_audit = analysis.validate_top_lr_pairs(top_pairs, projected, _crosswalk())
    assert bool(top_audit.iloc[0]["source_gene_sets_match"])

    converted = analysis.add_mouse_ensembl_queries(
        projected,
        {
            "Fzd7": "ENSMUSG00000001",
            "Wnt5a": "ENSMUSG00000002",
        },
    )
    payload = analysis.build_mouse_profile_payload(
        converted,
        domain_scope="custom",
        correction="fdr",
        background=["ENSMUSG00000001", "ENSMUSG00000002"],
    )
    assert payload["organism"] == "mmusculus"
    assert payload["query"] == {"q001": ["ENSMUSG00000001", "ENSMUSG00000002"]}


def test_empty_selection_is_a_compact_diagnostic_panel(tmp_path: Path) -> None:
    profile_id = "mouse_custom_fdr_ligand_receptor"
    rows = []
    for stage in analysis.STAGES:
        for pathway_id, pathway_name in (
            ("REAC:R-MMU-1", "A short pathway"),
            (
                "REAC:R-MMU-2",
                "Assembly of collagen fibrils and other multimeric structures",
            ),
        ):
            for method in analysis.NATIVE_DISPLAY_METHODS:
                rows.append(
                    {
                        "profile_id": profile_id,
                        "method": method,
                        "stage": stage,
                        "reactome_id": pathway_id,
                        "reactome_name": pathway_name,
                        "adjusted_p_value": 0.2,
                        "is_reactome_root": False,
                    }
                )
    enrichment = pd.DataFrame(rows)
    selection = analysis.select_external_pathways(
        enrichment,
        profile_id=profile_id,
        external_methods=analysis.NATIVE_EXTERNAL,
        all_methods=analysis.NATIVE_DISPLAY_METHODS,
        min_external_methods=2,
        max_pathways=12,
        analysis_id="test",
    )
    query_manifest = pd.DataFrame(
        [
            {
                "stage": stage,
                "method": method,
                "api_queried": True,
            }
            for stage in analysis.STAGES
            for method in analysis.NATIVE_DISPLAY_METHODS
        ]
    )
    output = tmp_path / "empty_diagnostic"
    analysis.plot_sensitivity_heatmap(
        enrichment,
        query_manifest,
        selection,
        profile_id=profile_id,
        external_methods=analysis.NATIVE_EXTERNAL,
        display_methods=analysis.NATIVE_DISPLAY_METHODS,
        min_external_methods=2,
        title="Post-hoc mouse sensitivity",
        output_path=output,
    )
    assert output.with_suffix(".png").stat().st_size > 10_000
    assert output.with_suffix(".pdf").stat().st_size > 1_000
    assert "\n" in analysis.wrap_pathway_name(
        "Assembly of collagen fibrils and other multimeric structures",
        width=35,
    )
