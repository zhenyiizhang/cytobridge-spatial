from __future__ import annotations

import hashlib
import importlib.util
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


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
                "abundance_controlled_distinct_cell_score": float(index + 1),
            }
            for index, (ligand, receptor, _) in enumerate(lr_rows)
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
            "schema_version": 1,
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


def test_validation_fails_after_table_tamper(tmp_path: Path) -> None:
    spec = _write_fixture(tmp_path)
    output = tmp_path / "output"
    MODULE.analyze(spec, output, n_selected_pairs=2)
    table = output / "directed_pair_concordance.csv"
    table.write_text(table.read_text() + "tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        MODULE.validate(output)


def test_report_writer_and_validator(tmp_path: Path) -> None:
    spec = _write_fixture(tmp_path)
    analysis = tmp_path / "analysis"
    MODULE.analyze(spec, analysis, n_selected_pairs=3)
    manifest_sha = _sha(analysis / "analysis_manifest.json")
    report = tmp_path / "report"
    MODULE.report(
        analysis,
        report,
        expected_analysis_manifest_sha256=manifest_sha,
    )
    MODULE.validate_report(report)
    assert (report / "zebrafish_attention_validation_a4.pdf").stat().st_size > 1000
    assert (report / "zebrafish_attention_validation_a4.png").stat().st_size > 1000
    response = (report / "reviewer_response.md").read_text(encoding="utf-8")
    caption = (report / "caption.txt").read_text(encoding="utf-8")
    assert "attention weights alone" in response
    assert "not used as validation evidence" in response
    assert "fixed-checkpoint" not in caption.casefold()
    assert (report / "panel_data" / "cytobridge_lr_scores.csv.gz").is_file()
    assert (report / "panel_data" / "commot_lr_scores_collapsed.csv.gz").is_file()
    assert not (
        report / "panel_data" / "fixed_checkpoint_interaction_on_off.csv"
    ).exists()


def test_spec_rejects_wrong_input_sha(tmp_path: Path) -> None:
    spec = _write_fixture(tmp_path)
    payload = json.loads(spec.read_text())
    payload["artifacts"]["commot_lr"]["sha256"] = "0" * 64
    _write_json(spec, payload)
    with pytest.raises(ValueError, match="SHA mismatch"):
        MODULE.analyze(spec, tmp_path / "output", n_selected_pairs=2)
