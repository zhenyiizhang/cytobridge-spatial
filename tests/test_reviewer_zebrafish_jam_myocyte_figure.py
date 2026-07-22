from __future__ import annotations

import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from reviewer_zebrafish_ccc import make_jam_myocyte_biology_figure as FIGURE  # noqa: E402
from reviewer_zebrafish_ccc import make_delta_notch_biology_figure as DELTA  # noqa: E402


def _valid_render_tables() -> tuple[pd.DataFrame, ...]:
    spatial = pd.DataFrame(
        {
            "h5_index": [0, 1],
            "x": [0.0, 0.05],
            "y": [0.0, 0.0],
            "is_somite": [True, True],
            "jam2a": [1.0, 0.0],
            "jam3b": [0.0, 1.0],
        }
    )
    adjacency = pd.DataFrame(
        {
            "source_h5": [0],
            "target_h5": [1],
            "distance": [0.05],
            "jam_compatible": [True],
        }
    )
    display_edges = pd.DataFrame(
        {
            "_source_h5": [0],
            "_target_h5": [1],
            "seed_support": [3],
            "display_score": [0.2],
            "source_x": [0.0],
            "target_x": [0.05],
        }
    )
    ranks = pd.DataFrame(
        {
            "display_label": ["CytoBridge raw attention (18hpf)"],
            "rank": [1.0],
            "n_contexts": [196],
        }
    )
    marker_detection = pd.DataFrame(
        {
            "stage_label": ["18hpf"] * len(FIGURE.MARKER_GENES),
            "cell_type": ["Somite"] * len(FIGURE.MARKER_GENES),
            "gene": list(FIGURE.MARKER_GENES),
            "detected_fraction": [0.1] * len(FIGURE.MARKER_GENES),
        }
    )
    claims = pd.DataFrame(
        [
            {
                "group": "supported",
                "status": True,
                "claim": "Published experiments validate the fusion program",
                "evidence": "independent evidence",
            },
            {
                "group": "not_supported",
                "status": False,
                "claim": "Attention is a biochemical communication strength",
                "evidence": "generic attention",
            },
            {
                "group": "not_supported",
                "status": False,
                "claim": "The JAM pattern is training-specific or LR-causal",
                "evidence": "initialization-matched control",
            },
        ]
    )
    return spatial, adjacency, display_edges, ranks, marker_detection, claims


def test_render_contract_accepts_explicit_supported_and_unsupported_claims() -> None:
    FIGURE.validate_render_contract(*_valid_render_tables())


def test_render_contract_rejects_self_adjacency() -> None:
    tables = list(_valid_render_tables())
    tables[1] = tables[1].assign(target_h5=0)
    with pytest.raises(ValueError, match="distinct cells"):
        FIGURE.validate_render_contract(*tables)


def test_delta_ranks_use_formal_support_qualified_universe() -> None:
    rows = []
    for axis_id in DELTA.PAIR_ORDER:
        for sender, receiver, supported in (
            (DELTA.VENTRAL_SPINAL, DELTA.VENTRAL_SPINAL, True),
            ("Other", "Other", True),
            ("Unsupported", "Unsupported", False),
        ):
            rows.append(
                {
                    "stage": 4.0,
                    "axis_id": axis_id,
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "passes_context_support_filter": supported,
                    "is_evaluated_context": supported,
                    "n_evaluated_contexts": 2,
                    "attention_context_percentile": 1.0 if supported else np.nan,
                    "attention_context_rank_from_top": 1.0 if supported else np.nan,
                    "exact_message_context_percentile": 1.0 if supported else np.nan,
                    "exact_message_context_rank_from_top": 1.0 if supported else np.nan,
                    "commot_context_percentile": 0.5 if supported else np.nan,
                    "commot_context_rank_from_top": 2.0 if supported else np.nan,
                }
            )
    screen = pd.DataFrame(rows)

    result = DELTA.prepare_circuit_percentiles(
        screen,
        stage=4.0,
        sender_type=DELTA.VENTRAL_SPINAL,
        receiver_type=DELTA.VENTRAL_SPINAL,
        cytobridge_rank_score="attention",
    )

    assert len(result) == 8
    assert result["n_contexts"].eq(2).all()
    assert set(result.loc[result["method"].eq("COMMOT"), "rank_label"]) == {"2/2"}

    tampered = screen.copy()
    tampered.loc[tampered["passes_context_support_filter"], "n_evaluated_contexts"] = 3
    with pytest.raises(ValueError, match="Support-qualified context count disagrees"):
        DELTA.prepare_circuit_percentiles(
            tampered,
            stage=4.0,
            sender_type=DELTA.VENTRAL_SPINAL,
            receiver_type=DELTA.VENTRAL_SPINAL,
            cytobridge_rank_score="attention",
        )


def test_marker_adapter_adds_12h_somite_from_h5ad_and_verifies_formal_rows(
    tmp_path: Path,
) -> None:
    genes = list(FIGURE.MARKER_GENES)
    obs = pd.DataFrame(
        {
            "time_point_processed": [2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            "Annotation": [
                "Somite",
                "Somite",
                "Somite",
                "Somite",
                "Fast Muscle Cell",
                "Fast Muscle Cell",
            ],
        },
        index=[f"cell_{index}" for index in range(6)],
    )
    matrix = np.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    data = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=genes))
    table_dir = tmp_path / "tables"
    table_dir.mkdir()
    rows = []
    for stage, stage_label, cell_type, indices, formal_genes in (
        (3.0, "18hpf", "Somite", [2, 3], ["jam2a", "jam3b", "myog", "mymk"]),
        (4.0, "24hpf", "Fast Muscle Cell", [4, 5], genes),
    ):
        for gene in formal_genes:
            column = genes.index(gene)
            n_detected = int(np.sum(matrix[indices, column] > 0))
            rows.append(
                {
                    "stage": stage,
                    "stage_label": stage_label,
                    "cell_type": cell_type,
                    "gene": gene,
                    "n_cells": len(indices),
                    "n_detected": n_detected,
                    "detected_fraction": n_detected / len(indices),
                }
            )
    formal_path = table_dir / "expression_detection_by_stage_type.csv"
    pd.DataFrame(rows).to_csv(formal_path, index=False)

    result, source = FIGURE.load_marker_detection(
        tmp_path,
        data,
        time_key="time_point_processed",
        annotation_key="Annotation",
        stage=3.0,
        stage_label="18hpf",
        cell_type="Somite",
        comparison_stage=2.0,
        comparison_stage_label="12hpf",
        comparison_cell_type="Somite",
        later_stage=4.0,
        later_stage_label="24hpf",
        later_cell_type="Fast Muscle Cell",
    )

    assert source == formal_path
    assert len(result) == 3 * len(genes)
    lookup = result.set_index(["stage_label", "cell_type", "gene"])
    assert lookup.loc[("12hpf", "Somite", "jam2a"), "detected_fraction"] == 0.5
    assert not bool(
        lookup.loc[("12hpf", "Somite", "jam2a"), "formal_case_table_value_available"]
    )
    assert bool(
        lookup.loc[("18hpf", "Somite", "jam2a"), "formal_case_table_value_available"]
    )
    assert not bool(
        lookup.loc[("18hpf", "Somite", "acta1a"), "formal_case_table_value_available"]
    )

    tampered = pd.read_csv(formal_path)
    tampered.loc[
        tampered["stage"].eq(3.0) & tampered["gene"].eq("jam2a"),
        "detected_fraction",
    ] = 0.75
    tampered.to_csv(formal_path, index=False)
    with pytest.raises(ValueError, match="disagrees with H5AD"):
        FIGURE.load_marker_detection(
            tmp_path,
            data,
            time_key="time_point_processed",
            annotation_key="Annotation",
            stage=3.0,
            stage_label="18hpf",
            cell_type="Somite",
            comparison_stage=2.0,
            comparison_stage_label="12hpf",
            comparison_cell_type="Somite",
            later_stage=4.0,
            later_stage_label="24hpf",
            later_cell_type="Fast Muscle Cell",
        )


def test_commot_distinct_rank_uses_hash_verified_complete_total_square(
    tmp_path: Path,
) -> None:
    labels = ["A", "Somite"]
    rows = []
    for stage, stage_time, scores in (
        (2.0, 12.0, [4.0, 3.0, 2.0, 1.0]),
        (3.0, 18.0, [1.0, 2.0, 3.0, 4.0]),
    ):
        for (sender, receiver), score in zip(
            ((sender, receiver) for sender in labels for receiver in labels),
            scores,
        ):
            rows.append(
                {
                    "method": "COMMOT",
                    "stage": stage,
                    "stage_time": stage_time,
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "abundance_controlled_distinct_cell_score": score,
                    "score_mean_possible_distinct_cell_pairs": score,
                    "n_possible_distinct_cell_pairs": 10,
                    "interaction_id": "total",
                    "matrix_key": "commot-test-total-total",
                }
            )
    score_path = tmp_path / "commot_type_pair_scores.csv.gz"
    pd.DataFrame(rows).to_csv(score_path, index=False, compression="gzip")
    lr_rows = []
    availability_rows = []
    for stage, stage_time in ((2.0, 12.0), (3.0, 18.0)):
        for ligand, receptor in FIGURE.JAM_AXES:
            availability_rows.append(
                {
                    "stage": stage,
                    "stage_time": stage_time,
                    "ligand": ligand,
                    "receptor": receptor,
                    "method_available": True,
                }
            )
            lr_rows.append(
                {
                    "stage": stage,
                    "stage_time": stage_time,
                    "ligand": ligand,
                    "receptor": receptor,
                    "sender_type": "A",
                    "receiver_type": "A",
                    "abundance_controlled_distinct_cell_score": 1.0,
                }
            )
            if stage == 3.0:
                lr_rows.append(
                    {
                        "stage": stage,
                        "stage_time": stage_time,
                        "ligand": ligand,
                        "receptor": receptor,
                        "sender_type": "Somite",
                        "receiver_type": "Somite",
                        "abundance_controlled_distinct_cell_score": 0.5,
                    }
                )
    lr_path = tmp_path / "commot_lr_scores.csv.gz"
    availability_path = tmp_path / "commot_lr_axis_stage_availability.csv.gz"
    pd.DataFrame(lr_rows).to_csv(lr_path, index=False, compression="gzip")
    pd.DataFrame(availability_rows).to_csv(
        availability_path, index=False, compression="gzip"
    )
    manifest = {
        "method": "COMMOT",
        "design": {
            "type_pair_grid_export": {"complete_directed_stage_type_square": True}
        },
        "score_semantics": {
            "abundance_controlled_distinct_cell_score": "distinct score"
        },
        "artifacts": {
            "type_pair_scores": {
                "path": str(score_path),
                "bytes": score_path.stat().st_size,
                "sha256": FIGURE.sha256(score_path),
            },
            "lr_scores": {
                "path": str(lr_path),
                "bytes": lr_path.stat().st_size,
                "sha256": FIGURE.sha256(lr_path),
            },
            "lr_axis_stage_availability": {
                "path": str(availability_path),
                "bytes": availability_path.stat().st_size,
                "sha256": FIGURE.sha256(availability_path),
            },
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    ranked, sources = FIGURE.commot_distinct_type_pair_rank_table(
        tmp_path,
        stage=3.0,
        stage_label="18hpf",
        comparison_stage=2.0,
        comparison_stage_label="12hpf",
        somite_label="Somite",
    )

    lookup = ranked.set_index("stage_label")
    assert lookup.loc["12hpf", "rank"] == 4
    assert lookup.loc["18hpf", "rank"] == 1
    assert lookup["n_contexts"].eq(4).all()
    assert set(sources) == {"commot_type_pair_scores", "commot_manifest"}

    reciprocal, reciprocal_sources = FIGURE.commot_reciprocal_jam_rank_table(
        tmp_path,
        stage=3.0,
        stage_label="18hpf",
        comparison_stage=2.0,
        comparison_stage_label="12hpf",
        somite_label="Somite",
    )
    reciprocal_lookup = reciprocal.set_index("stage_label")
    assert reciprocal_lookup.loc["12hpf", "score"] == 0
    assert reciprocal_lookup.loc["12hpf", "status"] == "not detected"
    assert reciprocal_lookup.loc["12hpf", "tie_count"] == 3
    assert reciprocal_lookup.loc["18hpf", "score"] == 0.5
    assert set(reciprocal_sources) == {
        "commot_lr_scores",
        "commot_lr_axis_stage_availability",
        "commot_reciprocal_jam_manifest",
    }


def test_published_mechanism_uses_correct_authors_and_year() -> None:
    source = Path(FIGURE.__file__).read_text(encoding="utf-8")
    assert "Powell & Wright 2011" in source
    assert "Luo et al. 2022" in source
    assert "Duan et al. 2023" not in source
