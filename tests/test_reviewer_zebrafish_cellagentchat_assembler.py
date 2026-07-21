from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


PACKAGE_PARENT = (
    Path(__file__).resolve().parents[1] / "scripts" / "reviewer_zebrafish_ccc"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from cellagentchat import assemble_dual  # noqa: E402
from cellagentchat import common  # noqa: E402


STAGE_LABELS = {
    0.0: "5.25hpf",
    1.0: "10hpf",
    2.0: "12hpf",
    3.0: "18hpf",
    4.0: "24hpf",
}


def _write(path: Path, value: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _shared_records(tmp_path: Path) -> dict[str, dict]:
    shared = tmp_path / "shared"
    return {
        "preparation_manifest": common.artifact(
            _write(shared / "manifest.json", '{"status":"complete"}\n')
        ),
        "mapped_expression": common.artifact(_write(shared / "mapped.h5ad")),
        "sample_plan": common.artifact(_write(shared / "sample_plan.csv")),
    }


def _source_record(tmp_path: Path) -> dict:
    source_file = _write(tmp_path / "source" / "src" / "abm.py", "class CellModel: pass\n")
    return {
        "repository": "https://github.com/mcgilldinglab/CellAgentChat",
        "release": "v0.2.0",
        "expected_commit": common.PINNED_CELLAGENTCHAT_COMMIT,
        "observed_commit": common.PINNED_CELLAGENTCHAT_COMMIT,
        "pinned_source_verified": True,
        "files": {"abm.py": common.artifact(source_file)},
    }


def _design() -> dict:
    return {
        "species_prior": "mouse",
        "cross_species_interpretation": (
            "zebrafish expression projected into mouse ortholog space"
        ),
        "sampling_seeds": list(assemble_dual.EXPECTED_SAMPLING_SEEDS),
        "stages": list(assemble_dual.EXPECTED_STAGES),
        "epochs": assemble_dual.EXPECTED_EPOCHS,
        "learning_rate": 0.1,
        "batch_size": 256,
        "feature_shuffles": 1,
        "permutation_score_target": (
            assemble_dual.EXPECTED_PERMUTATION_SCORE_TARGET
        ),
        "multiple_testing": "CellAgentChat v0.2.0 Bonferroni",
        "bonferroni_threshold": 0.05,
        "spatial": True,
        "spatial_key": "spatial_aligned",
        "permutation_background_distance_scaled": True,
        "tau": 2.0,
        "delta": 1.0,
        "native_primary": (
            "number of Bonferroni-significant LR pairs per directed cell-type pair"
        ),
        "raw_score_sum_is_secondary": True,
        "device": "cuda:0",
        "torch_sparse_backend": "torch_native_sparse_compat_v1",
        "torch_sparse_backend_semantics": "tested compatibility backend",
    }


def _write_condition(
    tmp_path: Path,
    *,
    label: str,
    shared_records: dict[str, dict],
    source: dict,
) -> Path:
    directory = tmp_path / "conditions" / label
    database = _write(directory / "input_database.tsv", f"{label}\n")
    claims = {
        "orthology_policy": "one2one_bijective_all_confidence",
        "orthology_analysis_tier": "sensitivity",
        "primary_claim_allowed": False,
        "formal_primary": False,
        "allow_nonprimary_preparation": True,
        "nonprimary_preparation_explicitly_allowed": True,
    }
    shared = {
        **shared_records,
        "database": common.artifact(database),
        "preparation_claims": claims,
    }
    runs = []
    required_files = sorted(assemble_dual.REQUIRED_RUN_ARTIFACTS)
    for stage, seed in assemble_dual.EXPECTED_GRID:
        run_dir = directory / f"stage_{stage:g}_{STAGE_LABELS[stage]}" / f"seed_{seed}"
        artifacts = {}
        for name in required_files:
            value = "value\n"
            if name == "cellagentchat_lr_scores_significant.csv":
                value = "ligand,receptor\n"
            artifacts[name] = common.artifact(_write(run_dir / name, value))
        record = {
            "stage": stage,
            "stage_label": STAGE_LABELS[stage],
            "sampling_seed": seed,
            "n_cells": 1,
            "n_cell_types": 1,
            "n_genes": 2,
            "n_ligands": 1,
            "n_receptors": 1,
            "n_lr_pairs_tested": 1,
            "n_tfs": 1,
            "n_receptors_with_prior_tf": 1,
            "n_raw_lr_rows": 1,
            "n_significant_lr_rows": 0,
            "average_normalized_spatial_distance": 0.5,
            "spatial_distance_used": True,
            "distance_scaled_permutation_background": True,
            "embedding_used_as_spatial": False,
            "artifacts": artifacts,
        }
        common.write_json(run_dir / "run_manifest.json", record)
        runs.append(record)

    fifteen = pd.DataFrame({"value": range(15)})
    top_frames = {
        "cellagentchat_type_pair_scores_by_seed.csv.gz": fifteen,
        "cellagentchat_type_pair_scores.csv": pd.DataFrame({"value": [1]}),
        "cellagentchat_lr_scores_raw_by_seed.csv.gz": fifteen,
        "cellagentchat_lr_scores_significant_by_seed.csv.gz": pd.DataFrame(
            columns=["value"]
        ),
        "cellagentchat_cell_receiving_scores_by_seed.csv.gz": fifteen,
    }
    top_artifacts = {}
    for name, frame in top_frames.items():
        path = directory / name
        frame.to_csv(path, index=False, compression="gzip" if name.endswith(".gz") else None)
        top_artifacts[name] = common.artifact(path)
    manifest = {
        "schema_version": 1,
        "method": "official_cellagentchat_v0_2_0_spatial",
        "database_condition": label,
        "source": source,
        "shared_input": shared,
        "design": _design(),
        "runs": runs,
        "counts": {
            "n_runs": 15,
            "type_pair_rows_by_seed": 15,
            "raw_lr_rows": 15,
            "significant_lr_rows": 0,
            "cell_receiving_rows": 15,
        },
        "artifacts": top_artifacts,
    }
    common.write_json(directory / "manifest.json", manifest)
    return directory


@pytest.fixture()
def completed_conditions(tmp_path: Path) -> tuple[Path, Path]:
    shared = _shared_records(tmp_path)
    source = _source_record(tmp_path)
    official = _write_condition(
        tmp_path,
        label=common.OFFICIAL_DATABASE_LABEL,
        shared_records=shared,
        source=source,
    )
    custom = _write_condition(
        tmp_path,
        label=common.CUSTOM_DATABASE_LABEL,
        shared_records=shared,
        source=source,
    )
    return official, custom


def _args(official: Path, custom: Path, output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        official_run_dir=official,
        custom_run_dir=custom,
        output_dir=output,
        overwrite=False,
    )


def test_assembler_creates_strict_dual_parent(
    tmp_path: Path, completed_conditions: tuple[Path, Path]
) -> None:
    official, custom = completed_conditions
    output = tmp_path / "assembled"
    result = assemble_dual.run(_args(official, custom, output))

    assert result["status"] == "complete"
    assert result["exact_stage_seed_grid_verified"] is True
    assert result["formal_non_smoke_verified"] is True
    assert (output / common.OFFICIAL_DATABASE_LABEL).is_symlink()
    assert (output / common.CUSTOM_DATABASE_LABEL).is_symlink()
    summary = pd.read_csv(output / "dual_condition_run_summary.csv")
    assert summary["database_condition"].tolist() == list(common.CONDITION_LABELS)
    assert summary["n_runs"].tolist() == [15, 15]
    assert summary["non_smoke_verified"].tolist() == [True, True]
    parent = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert parent["formal_design"]["epochs"] == 50
    assert parent["database_sha256_are_distinct"] is True


def test_assembler_rejects_an_incomplete_grid_before_creating_output(
    tmp_path: Path, completed_conditions: tuple[Path, Path]
) -> None:
    official, custom = completed_conditions
    manifest_path = custom / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"].pop()
    common.write_json(manifest_path, manifest)
    output = tmp_path / "assembled"

    with pytest.raises(RuntimeError, match="exact formal stage-by-seed grid"):
        assemble_dual.run(_args(official, custom, output))
    assert not output.exists()


def test_assembler_rejects_unpinned_source_and_existing_output(
    tmp_path: Path, completed_conditions: tuple[Path, Path]
) -> None:
    official, custom = completed_conditions
    manifest_path = official / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["pinned_source_verified"] = False
    common.write_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="not verified at the pinned"):
        assemble_dual.run(_args(official, custom, tmp_path / "unpinned"))

    manifest["source"]["pinned_source_verified"] = True
    common.write_json(manifest_path, manifest)
    output = tmp_path / "already_exists"
    _write(output / "sentinel.txt")
    with pytest.raises(FileExistsError, match="not empty"):
        assemble_dual.run(_args(official, custom, output))
    assert (output / "sentinel.txt").is_file()
