from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_zebrafish_attention_validation.py"
SPEC = importlib.util.spec_from_file_location(
    "run_zebrafish_attention_validation", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def _analysis_record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _use_portable_report_font(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decouple report-logic tests from proprietary runner fonts.

    The publication command remains strict about Arial.  These tests exercise
    the frozen-data contract and rendered report structure, so use Matplotlib's
    bundled DejaVu Sans when the CI runner does not provide Microsoft fonts.
    """

    import matplotlib.font_manager as font_manager

    portable_font = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    monkeypatch.setattr(
        font_manager,
        "findfont",
        lambda *args, **kwargs: portable_font,
    )


def _write_fixture(tmp_path: Path) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    acceptance = inputs / "acceptance.json"
    _write_json(acceptance, {"status": "PASS"})

    genes = [
        "apoeb",
        "lrp5",
        "uba52",
        "bmpr1ba",
        "dla",
        "notch3",
        "ligx",
        "recx",
        "ligy",
        "recy",
    ]
    labels = ["A", "A", "B", "B", "C", "C"]
    matrix = np.asarray(
        [
            [4, 0, 1, 0, 1, 0, 3, 0, 0, 1],
            [3, 0, 2, 0, 0, 0, 2, 0, 0, 2],
            [0, 4, 0, 3, 0, 1, 0, 4, 2, 0],
            [0, 3, 0, 2, 0, 1, 0, 3, 1, 0],
            [1, 0, 0, 0, 4, 0, 1, 0, 3, 0],
            [1, 0, 0, 0, 3, 0, 2, 0, 2, 0],
        ],
        dtype=float,
    )
    h5ad = inputs / "sample.h5ad"
    sample = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame({"Annotation": labels}, index=[f"c{i}" for i in range(6)]),
        var=pd.DataFrame(index=genes),
    )
    sample.obsm["spatial_aligned"] = np.asarray(
        [[0, 0], [0.1, 0], [1, 0], [1.1, 0], [2, 0], [2.1, 0]], dtype=float
    )
    sample.write_h5ad(h5ad)

    pair_rows = []
    for sender_index, sender in enumerate(("A", "B", "C")):
        for receiver_index, receiver in enumerate(("A", "B", "C")):
            score = float((sender_index + 1) * (receiver_index + 2))
            pair_rows.append(
                {
                    "stage": 4,
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "G_AB_attention_mean_mean": score / 10,
                    "D_AB_joint_mean": score,
                    "n_sender_cells_mean": 2,
                    "n_receiver_cells_mean": 2,
                    "spatial_distance_mean_mean": 0.1 + score / 100,
                }
            )
    cytobridge = inputs / "cytobridge.csv"
    pd.DataFrame(pair_rows).to_csv(cytobridge, index=False)
    commot_pairs = inputs / "commot_pairs.csv"
    pd.DataFrame(
        [
            {
                "stage": 4,
                "sender_type": row["sender_type"],
                "receiver_type": row["receiver_type"],
                "abundance_controlled_distinct_cell_score": row["D_AB_joint_mean"]
                * 0.8,
            }
            for row in pair_rows
        ]
    ).to_csv(commot_pairs, index=False)
    cellagent = inputs / "cellagent.csv"
    pd.DataFrame(
        [
            {
                "stage": 4,
                "sender_type": row["sender_type"],
                "receiver_type": row["receiver_type"],
                "cellagentchat_native_primary_mean": row["D_AB_joint_mean"] * 0.5,
            }
            for row in pair_rows
        ]
    ).to_csv(cellagent, index=False)

    lr_rows = [
        ("apoeb", "lrp5", "LIPID"),
        ("uba52", "bmpr1ba", "BMP"),
        ("dla", "notch3", "NOTCH"),
        ("ligx", "recx", "PX"),
        ("ligy", "recy", "PY"),
    ]
    lr_database = inputs / "lr_database.csv"
    pd.DataFrame(lr_rows, columns=["ligand", "receptor", "pathway"]).assign(
        category="test"
    ).to_csv(lr_database, index=False)
    commot_lr = inputs / "commot_lr.csv"
    pd.DataFrame(
        [
            {
                "stage": 4,
                "ligand": ligand,
                "receptor": receptor,
                "sender_type": sender,
                "receiver_type": receiver,
                "abundance_controlled_distinct_cell_score": float(
                    (index + 1) * (sender_index + 1) * (receiver_index + 1)
                ),
            }
            for index, (ligand, receptor, _) in enumerate(lr_rows)
            for sender_index, sender in enumerate(sorted(set(labels)))
            for receiver_index, receiver in enumerate(sorted(set(labels)))
        ]
    ).to_csv(commot_lr, index=False)
    nichenet_lr = inputs / "nichenet_lr.csv"
    pd.DataFrame(
        [
            {
                "ligand": ligand,
                "receptor": receptor,
                "lr_evidence": float(index + 1) / 10,
            }
            for index, (ligand, receptor, _) in enumerate(lr_rows)
        ]
    ).to_csv(nichenet_lr, index=False)
    nichenet_targets = inputs / "nichenet_targets.csv"
    pd.DataFrame(
        [
            {
                "ligand": ligand,
                "receptor": receptor,
                "target": f"target_{index}",
                "ligand_target_evidence": float(index + 1) / 10,
            }
            for index, (ligand, receptor, _) in enumerate(lr_rows)
        ]
    ).to_csv(nichenet_targets, index=False)
    interaction = inputs / "interaction.csv"
    pd.DataFrame(
        {
            "dataset": ["zebrafish", "zebrafish"],
            "target": [1, 2],
            "space": ["joint", "joint"],
            "off_relative_to_on": [0.1, 0.2],
        }
    ).to_csv(interaction, index=False)

    sample_manifest = inputs / "sample_manifest.json"
    cytobridge_manifest = inputs / "cytobridge_manifest.json"
    commot_manifest = inputs / "commot_manifest.json"
    cellagent_manifest = inputs / "cellagent_manifest.json"
    nichenet_manifest = inputs / "nichenet_manifest.json"
    interaction_manifest = inputs / "interaction_manifest.json"
    for path in (
        sample_manifest,
        commot_manifest,
        cellagent_manifest,
        nichenet_manifest,
        interaction_manifest,
    ):
        _write_json(path, {"status": "complete", "name": path.stem})
    edge_path = inputs / "edges_seed_101.csv.gz"
    pd.DataFrame(
        [
            {
                "source_index_stage": source,
                "target_index_stage": target,
                "sender_type": labels[source],
                "receiver_type": labels[target],
                "attention_abs_mean": float(source + target + 1),
                "edge_message_norm_joint": float((source + 1) * (target + 1)),
            }
            for source in range(6)
            for target in range(6)
            if source != target
        ]
    ).to_csv(edge_path, index=False)
    _write_json(
        cytobridge_manifest,
        {
            "status": "complete",
            "runs": [
                {
                    "grouping_seed": 101,
                    "artifacts": {
                        "edges": {
                            "path": str(edge_path.resolve()),
                            "sha256": _sha(edge_path),
                            "bytes": edge_path.stat().st_size,
                        }
                    },
                }
            ],
        },
    )

    artifacts = {
        "matched_acceptance": acceptance,
        "sample_h5ad": h5ad,
        "sample_manifest": sample_manifest,
        "cytobridge_type_pair": cytobridge,
        "cytobridge_manifest": cytobridge_manifest,
        "commot_type_pair": commot_pairs,
        "commot_lr": commot_lr,
        "commot_manifest": commot_manifest,
        "cellagentchat_type_pair": cellagent,
        "cellagentchat_manifest": cellagent_manifest,
        "nichenet_lr": nichenet_lr,
        "nichenet_targets": nichenet_targets,
        "nichenet_manifest": nichenet_manifest,
        "lr_database": lr_database,
        "interaction_target_metrics": interaction,
        "interaction_manifest": interaction_manifest,
    }
    spec = tmp_path / "spec.json"
    _write_json(
        spec,
        {
            "schema_version": MODULE.SCHEMA_VERSION,
            "workflow": "zebrafish_attention_lr_independent_validation",
            "dataset": "zebrafish",
            "cell_type_key": "Annotation",
            "spatial_key": "spatial_aligned",
            "permutations": 10,
            "random_seed": 17,
            "artifacts": {key: _record(value) for key, value in artifacts.items()},
        },
    )
    return spec


def _write_submission_report_fixture(tmp_path: Path) -> Path:
    """Write a compact frozen analysis with the accepted formal report counts."""

    analysis = tmp_path / "formal-analysis"
    analysis.mkdir()

    pair_rows = [
        ("attention", "COMMOT", 0.845, 0.886, 0.661, 0.751, 0.000999),
        ("attention", "CellAgentChat", 0.311, 0.221, 0.131, 0.197, 0.000999),
        ("exact_message", "COMMOT", 0.884, 0.901, 0.695, 0.772, 0.000999),
        (
            "exact_message",
            "CellAgentChat",
            0.407,
            0.234,
            0.145,
            0.215,
            0.000999,
        ),
    ]
    tables: dict[str, pd.DataFrame] = {
        "directed_pair_concordance.csv": pd.DataFrame(
            [
                {
                    "cytobridge_view": view,
                    "external_method": method,
                    "n_shared": 361,
                    "n_pairs": 361,
                    "n_strata": 22,
                    "n_permutations": 1000,
                    "spearman_rho": raw,
                    "adjusted_spearman_rho": adjusted,
                    "null_adjusted_spearman_mean": (lower + upper) / 2,
                    "null_adjusted_spearman_q025": lower,
                    "null_adjusted_spearman_q975": upper,
                    "adjusted_spearman_empirical_p_upper": empirical_p,
                }
                for view, method, raw, adjusted, lower, upper, empirical_p in pair_rows
            ]
        )
    }

    reference = pd.read_csv(
        ROOT / "scripts/reviewer_zebrafish_ccc/original_paper_21_lr.csv"
    )
    represented = reference["ligand"].ne("fn1c")
    represented_reference = reference.loc[represented].copy().reset_index(drop=True)
    exact_percentiles = np.linspace(0.91, 0.995, len(represented_reference))
    expression_percentiles = np.linspace(0.91, 0.995, len(represented_reference))
    exact_percentiles[-2:] = [0.86, 0.84]
    expression_percentiles[-1] = 0.86
    mdka_sdc4 = represented_reference["ligand"].eq("mdka") & represented_reference[
        "receptor"
    ].eq("sdc4")
    exact_percentiles[mdka_sdc4.to_numpy()] = 1.0

    represented_reference["lr_id"] = (
        represented_reference["ligand"] + "->" + represented_reference["receptor"]
    )
    represented_reference["lr_only_score"] = expression_percentiles
    represented_reference["lr_only_rank_percentile"] = expression_percentiles
    represented_reference["exact_message_score"] = exact_percentiles
    represented_reference["exact_message_rank_percentile"] = exact_percentiles

    paper_scores = reference.rename(
        columns={"paper_display_order": "paper_display_order_paper"}
    ).copy()
    paper_scores["represented_in_current_expression"] = represented.to_numpy()
    paper_scores = paper_scores.merge(
        represented_reference[
            [
                "ligand",
                "receptor",
                "lr_id",
                "lr_only_score",
                "lr_only_rank_percentile",
                "exact_message_score",
                "exact_message_rank_percentile",
            ]
        ],
        on=["ligand", "receptor"],
        how="left",
        validate="one_to_one",
    )
    tables["original_paper_21_lr_scores.csv"] = paper_scores
    tables["original_paper_21_lr_enrichment.csv"] = pd.DataFrame(
        [
            {
                "score_column": "lr_only_score",
                "n_background": 993,
                "n_paper_reference_present": 20,
                "paper_reference_auc": 0.981,
                "mannwhitney_p_greater": 7.04e-14,
                "top_fraction": 0.1,
                "top_n": 80,
                "paper_reference_in_top_n": 19,
            },
            {
                "score_column": "exact_message_score",
                "n_background": 993,
                "n_paper_reference_present": 20,
                "paper_reference_auc": 0.980,
                "mannwhitney_p_greater": 7.83e-14,
                "top_fraction": 0.1,
                "top_n": 79,
                "paper_reference_in_top_n": 18,
            },
        ]
    )

    shared_axes = pd.DataFrame(
        [
            ("dla", "notch1a", 0.988),
            ("dla", "notch3", 0.987),
            ("mdka", "sdc4", 0.997),
            ("thbs1b", "sdc4", 0.978),
        ],
        columns=["ligand", "receptor", "commot_rank_percentile"],
    )
    shared_axes["lr_id"] = shared_axes["ligand"] + "->" + shared_axes["receptor"]
    shared_axes["commot_score"] = shared_axes["commot_rank_percentile"]
    tables["commot_lr_scores_collapsed.csv.gz"] = shared_axes

    cell_types = [f"type-{index:02d}" for index in range(19)]
    cells = pd.DataFrame(
        {
            "cell_index": np.arange(20),
            "cell_type": [cell_types[index % len(cell_types)] for index in range(20)],
            "spatial_x": np.linspace(0, 5, 20),
            "spatial_y": np.sin(np.linspace(0, np.pi, 20)),
            "ligand": "mdka",
            "receptor": "sdc4",
            "ligand_scaled_expression": np.linspace(0.1, 1.0, 20),
            "receptor_scaled_expression": np.linspace(1.0, 0.1, 20),
        }
    )
    tables["selected_paper_axis_spatial_cells.csv.gz"] = cells
    edge_rows = []
    for index in range(19):
        edge_rows.append(
            {
                "source_index_stage": index,
                "target_index_stage": index + 1,
                "sender_type": cell_types[index],
                "receiver_type": cell_types[(index + 1) % len(cell_types)],
                "source_x": cells.loc[index, "spatial_x"],
                "source_y": cells.loc[index, "spatial_y"],
                "target_x": cells.loc[index + 1, "spatial_x"],
                "target_y": cells.loc[index + 1, "spatial_y"],
                "lr_activity": 0.2 + index / 20,
                "edge_message_norm_joint": 1.0 + index,
                "exact_message_lr_score": 0.2 + index / 10,
                "top_exact_message_lr_edge": index >= 3,
            }
        )
    tables["selected_paper_axis_spatial_edges.csv.gz"] = pd.DataFrame(edge_rows)
    tables["selected_paper_axis_spatial_summary.csv"] = pd.DataFrame(
        [
            {
                "lr_id": "mdka->sdc4",
                "selection_rule": (
                    "highest CytoBridge exact-message score among the frozen "
                    "original-paper 21 LR reference axes"
                ),
                "n_unique_edges": 18_573,
                "n_positive_lr_edges": 6_226,
                "n_top_exact_message_lr_edges": 125,
                "cell_type_context_spearman_rho": 0.866,
            }
        ]
    )

    contexts = []
    rank = 0
    selected_count = 0
    for sender in cell_types:
        for receiver in cell_types:
            rank += 1
            selected_by_model = sender != receiver and selected_count < 8
            if selected_by_model:
                selected_count += 1
            contexts.append(
                {
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "cytobridge_exact_message_lr_score": rank / 361,
                    "commot_abundance_controlled_distinct_cell_score": rank / 400,
                    "cytobridge_rank_percentile": rank / 361,
                    "commot_rank_percentile": rank / 361,
                    "selected_by_cytobridge": selected_by_model,
                    "cytobridge_selection_rank": (
                        selected_count if selected_by_model else pd.NA
                    ),
                }
            )
    tables["selected_paper_axis_context_external_ranks.csv"] = pd.DataFrame(contexts)

    artifacts: dict[str, dict[str, str | int]] = {}
    for filename, table in tables.items():
        path = analysis / filename
        table.to_csv(
            path,
            index=False,
            compression="gzip" if filename.endswith(".gz") else None,
        )
        artifacts[filename] = _analysis_record(path)
    manifest = {
        "schema_version": MODULE.SCHEMA_VERSION,
        "workflow": MODULE.WORKFLOW,
        "status": "complete",
        "claim_contract": {
            "original_paper_21_scope": (
                "pre-specified source-paper axes from the same atlas; not an "
                "independent cohort or experimental ground truth"
            ),
            "interaction_on_off_scope": (
                "fixed-checkpoint inference sensitivity; not a ligand knockout"
            ),
        },
        "artifacts": artifacts,
    }
    _write_json(analysis / "analysis_manifest.json", manifest)
    return analysis


def _write_jam_report_manifests(
    tmp_path: Path,
) -> tuple[list[Path], list[str]]:
    jam_root = tmp_path / "current-jam"
    jam_root.mkdir()
    tables = jam_root / "tables"
    tables.mkdir()

    conditions = ["trained", "pre_interaction", "random"]
    compatibility_rows: list[dict[str, object]] = []
    for condition, compatible_median, other_median in zip(
        conditions,
        [0.76, 0.55, 0.49],
        [0.43, 0.51, 0.50],
        strict=True,
    ):
        compatibility_rows.extend(
            [
                {
                    "condition": condition,
                    "jam_compatible": True,
                    "n_edges": 24,
                    "attention_percentile_mean": compatible_median + 0.01,
                    "attention_percentile_median": compatible_median,
                },
                {
                    "condition": condition,
                    "jam_compatible": False,
                    "n_edges": 76,
                    "attention_percentile_mean": other_median + 0.01,
                    "attention_percentile_median": other_median,
                },
            ]
        )
    jam_tables: dict[str, pd.DataFrame] = {
        "jam_compatibility_percentile_summary.csv": pd.DataFrame(compatibility_rows),
        "jam_quartile_compatibility.csv": pd.DataFrame(
            {
                "condition": conditions,
                "top_n_edges_after_boundary_ties": [25, 25, 25],
                "top_n_jam_compatible": [8, 3, 2],
                "top_compatibility_rate": [0.32, 0.12, 0.08],
                "bottom_n_edges_after_boundary_ties": [25, 25, 25],
                "bottom_n_jam_compatible": [2, 2, 2],
                "bottom_compatibility_rate": [0.08, 0.08, 0.08],
                "top_vs_bottom_odds_ratio": [3.2, 1.15, 0.92],
                "fisher_exact_two_sided_p_descriptive_technical": [
                    0.004,
                    0.63,
                    0.81,
                ],
            }
        ),
        "type_pair_raw_attention_ranks.csv": pd.DataFrame(
            {
                "condition": conditions,
                "stage_label": ["18hpf"] * 3,
                "sender_type": ["Somite"] * 3,
                "receiver_type": ["Somite"] * 3,
                "rank_from_top": [2, 71, 155],
                "n_complete_directed_type_pairs": [361] * 3,
            }
        ),
        "somite_18hpf_spatial_null_summary.csv": pd.DataFrame(
            {
                "stage_label": ["18hpf"],
                "cell_type": ["Somite"],
                "n_cells": [6],
                "observed_jam2a_jam3b_orientation_compatible_pairs": [18],
                "null_mean": [9.0],
                "null_q025": [5.0],
                "null_q975": [13.0],
                "observed_over_null_mean": [2.0],
                "monte_carlo_upper_tail_p_plus1": [0.0099],
                "n_permutations": [1000],
            }
        ),
        "somite_18hpf_spatial_null_iterations.csv.gz": pd.DataFrame(
            {
                "iteration": np.arange(1, 11),
                "orientation_compatible_pair_count": [
                    8,
                    9,
                    11,
                    7,
                    10,
                    6,
                    13,
                    9,
                    8,
                    10,
                ],
            }
        ),
        "myog_association.csv": pd.DataFrame(
            {
                "stage_label": ["18hpf", "18hpf"],
                "cell_type": ["Somite", "Somite"],
                "gene_a": ["jam2a", "jam3b"],
                "gene_b": ["myog", "myog"],
                "n_cells": [6, 6],
                "both_detected": [2, 3],
                "gene_a_only": [1, 0],
                "gene_b_only": [2, 1],
                "neither_detected": [1, 2],
                "fisher_odds_ratio": [1.35, 3.10],
                "fisher_two_sided_p": [0.21, 0.003],
                "spearman_rho_expression": [0.12, 0.46],
            }
        ),
        "expression_detection_by_stage_type.csv": pd.DataFrame(
            {
                "stage_label": ["18hpf"] * 3,
                "cell_type": ["Somite"] * 3,
                "gene": ["jam2a", "jam3b", "myog"],
                "n_cells": [6] * 3,
                "n_detected": [3, 3, 4],
                "detected_fraction": [0.5, 0.5, 4 / 6],
            }
        ),
        "somite_18hpf_spatial_cells.csv.gz": pd.DataFrame(
            {
                "stage_label": ["18hpf"] * 8,
                "x": [0.0, 0.8, 1.6, 2.4, 3.2, 4.0, 1.1, 3.5],
                "y": [0.0, 0.3, 0.0, 0.4, 0.1, 0.4, 1.0, 1.2],
                "is_somite": [True] * 6 + [False] * 2,
                "jam2a_positive": [True, False, True, False, True, False, False, False],
                "jam3b_positive": [False, True, False, True, False, True, False, False],
                "myog_positive": [True, True, False, True, False, True, False, False],
            }
        ),
        "trained_jam_display_edges.csv": pd.DataFrame(
            {
                "condition": ["trained", "trained"],
                "display_rank": [1, 2],
                "source_x": [0.0, 1.6],
                "source_y": [0.0, 0.0],
                "target_x": [0.8, 2.4],
                "target_y": [0.3, 0.4],
                "jam_compatible": [True, True],
            }
        ),
    }
    records: dict[str, dict[str, str | int]] = {}
    for filename, table in jam_tables.items():
        path = tables / filename
        table.to_csv(
            path,
            index=False,
            compression="gzip" if filename.endswith(".gz") else None,
        )
        records[filename] = _analysis_record(path)

    control_keys = {
        "jam_compatibility_percentile_summary.csv",
        "jam_quartile_compatibility.csv",
        "type_pair_raw_attention_ranks.csv",
    }
    manifests = []
    for index, selected_keys in enumerate(
        [control_keys, set(records).difference(control_keys)], start=1
    ):
        path = jam_root / f"manifest_{index}.json"
        _write_json(
            path,
            {
                "schema_version": 1,
                "status": "complete",
                "artifacts": {key: records[key] for key in sorted(selected_keys)},
            },
        )
        manifests.append(path)
    return manifests, [_sha(path) for path in manifests]


def test_analyze_and_validate_real_writer_schemas(tmp_path: Path) -> None:
    spec = _write_fixture(tmp_path)
    output = tmp_path / "output"
    MODULE.analyze(spec, output, n_selected_pairs=3)
    MODULE.validate(output)
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["settings"]["n_directed_pairs"] == 9
    assert manifest["settings"]["n_represented_lr"] >= 5
    pair_metrics = pd.read_csv(output / "directed_pair_concordance.csv")
    assert set(pair_metrics.external_method) == {"COMMOT", "CellAgentChat"}
    assert set(pair_metrics.cytobridge_view) == {"attention", "exact_message"}
    permutation = pd.read_csv(output / "lr_modifier_permutation_tests.csv")
    assert len(permutation) == 4
    assert set(permutation.external_method) == {"COMMOT", "NicheNet"}
    controlled = pair_metrics.set_index(["cytobridge_view", "external_method"])
    assert np.isfinite(controlled["adjusted_spearman_rho"]).all()
    spatial = pd.read_csv(output / "selected_paper_axis_spatial_summary.csv")
    assert spatial.n_unique_edges.iloc[0] == 30
    assert spatial.n_top_exact_message_lr_edges.iloc[0] >= 1
    contexts = pd.read_csv(output / "selected_paper_axis_context_external_ranks.csv")
    assert len(contexts) == 9
    assert contexts.selected_by_cytobridge.sum() == 6


def test_validation_fails_after_table_tamper(tmp_path: Path) -> None:
    spec = _write_fixture(tmp_path)
    output = tmp_path / "output"
    MODULE.analyze(spec, output, n_selected_pairs=2)
    table = output / "directed_pair_concordance.csv"
    table.write_text(table.read_text() + "tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        MODULE.validate(output)


def test_report_rejects_nonformal_pair_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_portable_report_font(monkeypatch)
    spec = _write_fixture(tmp_path)
    analysis = tmp_path / "analysis"
    MODULE.analyze(spec, analysis, n_selected_pairs=3)
    manifest_sha = _sha(analysis / "analysis_manifest.json")
    jam_manifests, jam_manifest_shas = _write_jam_report_manifests(tmp_path)
    report = tmp_path / "report"
    with pytest.raises(ValueError, match="complete 19 x 19 pair field"):
        MODULE.report(
            analysis,
            report,
            expected_analysis_manifest_sha256=manifest_sha,
            jam_manifest_paths=jam_manifests,
            expected_jam_manifest_sha256s=jam_manifest_shas,
        )


def test_report_rejects_legacy_init_condition_label() -> None:
    with pytest.raises(ValueError, match="old 'init' label"):
        MODULE._canonical_conditions(
            pd.Series(["trained", "init", "random"]),
            label="JAM test fixture",
        )


def test_submission_report_is_three_panel_and_uses_strong_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_portable_report_font(monkeypatch)
    analysis = _write_submission_report_fixture(tmp_path)
    jam_manifests, jam_manifest_shas = _write_jam_report_manifests(tmp_path)
    output = tmp_path / "submission-report"
    MODULE.report(
        analysis,
        output,
        expected_analysis_manifest_sha256=_sha(analysis / "analysis_manifest.json"),
        jam_manifest_paths=jam_manifests,
        expected_jam_manifest_sha256s=jam_manifest_shas,
    )
    MODULE.validate_report(output)
    assert (output / "zebrafish_attention_validation_a4.pdf").stat().st_size > 1000
    assert (output / "zebrafish_attention_validation_a4.png").stat().st_size > 1000

    caption = (output / "caption.txt").read_text(encoding="utf-8")
    caption_folded = caption.casefold()
    response = (output / "reviewer_response.md").read_text(encoding="utf-8")
    response_folded = response.casefold()
    provenance = (output / "provenance.md").read_text(encoding="utf-8")
    report_source = inspect.getsource(MODULE.report)

    for required_column in (
        "adjusted_spearman_rho",
        "null_adjusted_spearman_q025",
        "null_adjusted_spearman_q975",
        "adjusted_spearman_empirical_p_upper",
        'dimensions["n_pairs"]',
        'dimensions["n_strata"]',
        'dimensions["n_permutations"]',
    ):
        assert required_column in report_source
    assert "361 directed cell-type pairs" in caption_folded
    assert "361" in response
    assert "22 strata" in caption_folded
    assert "1,000 structured permutations" in caption_folded
    assert "adjusted" in caption_folded
    assert "95%" in caption_folded
    assert "empirical" in caption_folded

    assert "jam-compatible" in caption_folded
    assert "32.0% of top-attention-quartile edges" in caption_folded
    assert "8.0% in the bottom quartile" in caption_folded
    assert "top_compatibility_rate" in report_source
    assert "bottom_compatibility_rate" in report_source
    assert "trained" in caption_folded
    assert "pre-interaction" in caption_folded
    assert "randomized" in caption_folded
    assert "6 somite cells at 18 hpf" in caption_folded
    assert "same 100 model edges" in caption_folded
    assert "before-learning control" in caption_folded
    assert "before the interaction learning stage" in caption_folded
    assert "all 8 cells for anatomical context" in caption_folded
    assert "statistics remain restricted to the 6 somite cells" in caption_folded
    assert "descriptive technical controls" in response_folded
    assert "edges and cells are not treated as biological replicates" in response_folded

    assert "jam3b and myog co-detection" in caption_folded
    assert "myog was detected in 100.0% of jam3b+ cells" in caption_folded
    assert "myog_given_positive" in report_source
    assert "spatially neighboring" in caption_folded
    assert "label-permutation null" in caption_folded
    assert "cross-sectional" in caption_folded
    assert "not evidence" in caption_folded
    assert "attention coefficient cannot be interpreted directly" in response_folded
    assert "old `init` label is not relabelled" in provenance.casefold()
    assert "figure pdf" in provenance.casefold()
    assert "plotting-script snapshot" in provenance.casefold()

    assert "aggregated into one model interaction rank" in caption_folded
    assert "exact message" not in caption_folded
    assert "exact-message" not in caption_folded
    assert '"attention", "commot"' in report_source.casefold()
    assert '"exact_message", "commot"' not in report_source.casefold()
    assert "figsize=(8.27, 11.69)" in report_source
    assert "purple" not in report_source.casefold()

    assert "(d)" not in caption_folded
    assert "(e)" not in caption_folded
    assert "nichenet" not in caption_folded
    assert "perturb" not in caption_folded
    assert "source-paper lr compatibility" not in caption_folded
    assert "18/20" not in caption_folded
    assert "19/20" not in caption_folded
    assert "mdka" not in caption_folded
    assert "sdc4" not in caption_folded

    panel_names = {path.name for path in (output / "panel_data").iterdir()}
    assert panel_names == {
        "directed_pair_concordance.csv",
        "jam_compatibility_percentile_summary.csv",
        "jam_quartile_compatibility.csv",
        "type_pair_raw_attention_ranks.csv",
        "somite_18hpf_spatial_null_summary.csv",
        "somite_18hpf_spatial_null_iterations.csv.gz",
        "myog_association.csv",
        "expression_detection_by_stage_type.csv",
        "somite_18hpf_spatial_cells.csv.gz",
        "trained_jam_display_edges.csv",
    }
    report_manifest = json.loads(
        (output / "report_manifest.json").read_text(encoding="utf-8")
    )
    assert set(report_manifest["figure_panels"]) == {"a", "b", "c"}
    assert {
        Path(path).name for path in report_manifest["panel_data_files"]
    } == panel_names
    claim_contract = report_manifest["claim_contract"]
    assert claim_contract["jam_controls_are_biological_replicates"] is False
    assert claim_contract["jam_statistics_scope"] == "descriptive technical controls"
    assert "not causal" in claim_contract["myog_scope"]
    assert claim_contract["legacy_jam_outputs_allowed"] is False
    assert claim_contract["attention_is_biochemical_probability"] is False
    assert claim_contract["figure_a_cytobridge_view"] == "attention"
    assert claim_contract["figure_canvas"] == "A4 portrait"
    assert "6 cells and 100 fixed edges" in claim_contract["panel_b_scope"]
    assert "8 tissue cells displayed" in claim_contract["panel_c_context_scope"]
    assert len(report_manifest["jam_manifests"]) == 2
    assert {record["sha256"] for record in report_manifest["jam_manifests"]} == set(
        jam_manifest_shas
    )

    pair_evidence = pd.read_csv(output / "panel_data" / "directed_pair_concordance.csv")
    assert set(pair_evidence["n_pairs"]) == {361}
    assert set(pair_evidence["n_strata"]) == {22}
    assert set(pair_evidence["n_permutations"]) == {1000}

    compatibility = pd.read_csv(
        output / "panel_data" / "jam_compatibility_percentile_summary.csv"
    )
    assert set(compatibility["condition"]) == {
        "trained",
        "pre_interaction",
        "random",
    }
    trained = compatibility.loc[compatibility["condition"].eq("trained")]
    trained_compatible = trained.loc[trained["jam_compatible"].astype(bool)]
    trained_other = trained.loc[~trained["jam_compatible"].astype(bool)]
    assert (
        trained_compatible["attention_percentile_median"].item()
        > trained_other["attention_percentile_median"].item()
    )

    type_pair = pd.read_csv(output / "panel_data" / "type_pair_raw_attention_ranks.csv")
    assert set(type_pair["condition"]) == {
        "trained",
        "pre_interaction",
        "random",
    }
    assert set(type_pair["n_complete_directed_type_pairs"]) == {361}

    spatial_summary = pd.read_csv(
        output / "panel_data" / "somite_18hpf_spatial_null_summary.csv"
    )
    assert spatial_summary.loc[0, "observed_over_null_mean"] == pytest.approx(2.0)
    assert spatial_summary.loc[0, "monte_carlo_upper_tail_p_plus1"] == pytest.approx(
        0.0099
    )
    associations = pd.read_csv(output / "panel_data" / "myog_association.csv")
    assert set(associations["gene_a"].str.casefold()) == {"jam2a", "jam3b"}
    display_edges = pd.read_csv(output / "panel_data" / "trained_jam_display_edges.csv")
    assert set(display_edges["condition"]) == {"trained"}
    assert display_edges["jam_compatible"].astype(bool).all()


def test_spec_rejects_wrong_input_sha(tmp_path: Path) -> None:
    spec = _write_fixture(tmp_path)
    payload = json.loads(spec.read_text())
    payload["artifacts"]["commot_lr"]["sha256"] = "0" * 64
    _write_json(spec, payload)
    with pytest.raises(ValueError, match="SHA mismatch"):
        MODULE.analyze(spec, tmp_path / "output", n_selected_pairs=2)
