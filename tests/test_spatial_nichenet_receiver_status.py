from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_spatial_communication_consistency.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "spatial_communication_consistency_receiver_status", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_nichenet_source(root: Path, status: pd.DataFrame | None) -> Path:
    source = root / "nichenet"
    official = source / "official"
    official.mkdir(parents=True)
    (source / "manifest.json").write_text("{}\n", encoding="utf-8")
    (official / "R_sessionInfo.txt").write_text("R fixture\n", encoding="utf-8")
    pd.DataFrame(
        {
            "dataset": ["fixture", "fixture"],
            "sender": ["A", "A"],
            "receiver": ["B", "C"],
            "ligand": ["L1", "L0"],
            "receptor": ["R1", "R0"],
            "sender_fraction": [1.0, 1.0],
            "receiver_fraction": [1.0, 1.0],
        }
    ).to_csv(source / "sender_receiver_lr_candidates.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["fixture"],
            "receiver": ["B"],
            "ligand": ["L1"],
            "aupr_corrected": [0.8],
        }
    ).to_csv(official / "ligand_activities.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["fixture"],
            "receiver": ["B"],
            "ligand": ["L1"],
            "target": ["T1"],
            "weight": [0.5],
        }
    ).to_csv(official / "ligand_target_links.csv", index=False)
    if status is not None:
        status.to_csv(official / "receiver_status.csv", index=False)
        pd.DataFrame(
            {
                "dataset": ["fixture", "fixture"],
                "receiver": ["B", "C"],
                "gene": ["G1", "G2"],
                "is_response": [True, False],
            }
        ).to_csv(source / "receiver_gene_sets.csv", index=False)
    return source


def _valid_status() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": ["fixture", "fixture"],
            "receiver": ["B", "C"],
            "status": ["complete", "skipped_no_potential_ligands"],
            "reason": [
                "",
                "no candidate ligand is represented in the frozen NicheNet ligand-target matrix",
            ],
            "n_response_genes": [20, 20],
            "n_background_genes": [100, 100],
            "n_potential_ligands": [1, 0],
        }
    )


def test_summarize_nichenet_binds_and_records_receiver_status(tmp_path: Path) -> None:
    module = _load_module()
    source = _write_nichenet_source(tmp_path, _valid_status())
    output = tmp_path / "summary"

    module.summarize_nichenet(
        SimpleNamespace(nichenet_dir=str(source), output_dir=str(output))
    )

    observed = pd.read_csv(output / "nichenet_receiver_status.csv")
    assert observed[["receiver", "status"]].values.tolist() == [
        ["B", "complete"],
        ["C", "skipped_no_potential_ligands"],
    ]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["receiver_status"] == {
        "present": True,
        "allowed_statuses": ["complete", "skipped_no_potential_ligands"],
        "counts": {"complete": 1, "skipped_no_potential_ligands": 1},
        "n_receivers": 2,
        "n_complete_receivers": 1,
        "n_skipped_no_potential_ligands": 1,
    }
    assert "receiver_status" in manifest["sources"]
    assert "receiver_gene_sets" in manifest["sources"]
    assert "nichenet_receiver_status.csv" in manifest["outputs"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda table: table.assign(status=["complete", "unexpected"]),
            "unsupported statuses",
        ),
        (
            lambda table: table.assign(n_potential_ligands=[1, 1]),
            "zero potential ligands",
        ),
        (
            lambda table: table.assign(
                status=["skipped_no_potential_ligands"] * 2,
                n_potential_ligands=[0, 0],
            ),
            "no complete receiver",
        ),
        (
            lambda table: table.assign(receiver=["B", "B"]),
            "unique receivers",
        ),
    ],
)
def test_summarize_nichenet_rejects_invalid_receiver_status(
    tmp_path: Path, mutate, message: str
) -> None:
    module = _load_module()
    source = _write_nichenet_source(tmp_path, mutate(_valid_status()))

    with pytest.raises(ValueError, match=message):
        module.summarize_nichenet(
            SimpleNamespace(
                nichenet_dir=str(source), output_dir=str(tmp_path / "summary")
            )
        )


def test_summarize_nichenet_keeps_legacy_status_optional(tmp_path: Path) -> None:
    module = _load_module()
    source = _write_nichenet_source(tmp_path, None)
    output = tmp_path / "summary"

    module.summarize_nichenet(
        SimpleNamespace(nichenet_dir=str(source), output_dir=str(output))
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["receiver_status"]["present"] is False
    assert not (output / "nichenet_receiver_status.csv").exists()
