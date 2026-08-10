from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


PACKAGE_PARENT = (
    Path(__file__).resolve().parents[1] / "scripts" / "reviewer_zebrafish_ccc"
)
sys.path.insert(0, str(PACKAGE_PARENT))

from cellagentchat import common  # noqa: E402
from cellagentchat import prepare_inputs  # noqa: E402
from cellagentchat import run_dual  # noqa: E402
from cellagentchat import run_spatial  # noqa: E402


def _mapping_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zebrafish_gene": ["a", "a", "b", "c", "d", "d", "e"],
            "mouse_gene": ["A", "A", "B", "B", "D1", "D2", "E"],
            "orthology_type": ["ortholog_one2one"] * 7,
            "orthology_confidence": [1, 1, 1, 1, 1, 1, 0],
        }
    )


def test_orthology_policy_is_explicit_and_auditable() -> None:
    strict, excluded, counts = common.select_orthology_mapping(
        _mapping_frame(),
        source_column="zebrafish_gene",
        target_column="mouse_gene",
        mapping_policy="strict_one_to_one",
    )
    assert strict[["source_gene", "target_gene"]].values.tolist() == [["a", "A"]]
    assert counts["selected_rows"] == 1
    assert {
        "duplicate_mapping_row",
        "target_has_multiple_sources",
        "source_maps_to_multiple_targets",
        "low_or_missing_confidence",
    }.issubset(set(excluded["exclusion_reason"]))

    many, _, many_counts = common.select_orthology_mapping(
        _mapping_frame(),
        source_column="zebrafish_gene",
        target_column="mouse_gene",
        mapping_policy="many_to_one_sum",
    )
    assert set(zip(many["source_gene"], many["target_gene"])) == {
        ("a", "A"),
        ("b", "B"),
        ("c", "B"),
    }
    assert many_counts["selected_target_genes"] == 2


def _write_ensembl_sensitivity_mapping(
    tmp_path: Path,
) -> tuple[Path, Path, pd.DataFrame]:
    mapping = pd.DataFrame(
        {
            "zebrafish_symbol": ["a", "b"],
            "mouse_symbol": ["A", "B"],
            "orthology_type": ["ortholog_one2one", "ortholog_one2one"],
            "orthology_confidence": [1, 0],
        }
    )
    mapping_path = (
        tmp_path
        / "ensembl_compara_drerio_to_mouse_one2one_bijective_all_confidence.csv"
    )
    mapping.to_csv(mapping_path, index=False)
    manifest_path = tmp_path / "orthology_manifest.json"
    manifest = {
        "schema_version": 2,
        "workflow": prepare_inputs.ORTHOLOGY_MANIFEST_WORKFLOW,
        "status": "complete",
        "ensembl_release": 116,
        "mapping_policy": "one2one_bijective_all_confidence",
        "analysis_tier": "sensitivity",
        "primary_claim_allowed": False,
        "mapping_file": mapping_path.name,
        "mapping_label": "sensitivity only",
        "source_mode": "frozen_raw_input",
        "source_input": {"md5": "1" * 32, "size_bytes": 123},
        "filter": {
            "orthology_type": "ortholog_one2one",
            "orthology_confidence_policy": "not_filtered",
            "nonempty_symbols": True,
            "symbol_level_bijection_after_casefold": True,
        },
        "counts": {"selected_bijective_symbol_pairs": len(mapping)},
        "output_md5": {
            "raw": "2" * 32,
            "mapping": prepare_inputs._md5_file(mapping_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return mapping_path, manifest_path, mapping


def test_all_confidence_manifest_is_verified_and_cannot_be_primary(
    tmp_path: Path,
) -> None:
    mapping_path, manifest_path, mapping = _write_ensembl_sensitivity_mapping(tmp_path)
    provenance = prepare_inputs.resolve_orthology_provenance(
        orthology_path=mapping_path,
        orthology_frame=mapping,
        selected_mapping=mapping.rename(
            columns={"zebrafish_symbol": "source_gene", "mouse_symbol": "target_gene"}
        ),
        mapping_counts={"selected_rows": len(mapping)},
        mapping_policy="strict_one_to_one",
        orthology_type_column="orthology_type",
        allowed_orthology_types=("ortholog_one2one",),
        confidence_column="orthology_confidence",
        minimum_confidence=0.0,
        requested_analysis_tier="sensitivity",
        manifest_path=manifest_path,
    )
    assert provenance["orthology_policy"] == "one2one_bijective_all_confidence"
    assert provenance["analysis_tier"] == "sensitivity"
    assert provenance["policy_primary_claim_allowed"] is False
    assert provenance["mapping_source_manifest"]["verified"] is True
    assert provenance["mapping_source_manifest"]["ensembl_release"] == 116
    assert not prepare_inputs.primary_claim_is_allowed(
        expression_projection_mode="strict_log1p_rename",
        mapping_policy="strict_one_to_one",
        confidence_column="orthology_confidence",
        minimum_confidence=0.0,
        orthology_provenance=provenance,
        pinned_source_verified=True,
    )

    with pytest.raises(ValueError, match="requires --minimum-confidence 0"):
        prepare_inputs.resolve_orthology_provenance(
            orthology_path=mapping_path,
            orthology_frame=mapping,
            selected_mapping=mapping,
            mapping_counts={"selected_rows": len(mapping)},
            mapping_policy="strict_one_to_one",
            orthology_type_column="orthology_type",
            allowed_orthology_types=("ortholog_one2one",),
            confidence_column="orthology_confidence",
            minimum_confidence=1.0,
            requested_analysis_tier="auto",
            manifest_path=manifest_path,
        )


def test_unverified_minimum_confidence_zero_is_automatically_sensitivity(
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "mapping.csv"
    mapping_path.write_text("zebrafish_symbol,mouse_symbol\na,A\n", encoding="utf-8")
    mapping = pd.DataFrame({"source_gene": ["a"], "target_gene": ["A"]})
    kwargs = dict(
        orthology_path=mapping_path,
        orthology_frame=mapping,
        selected_mapping=mapping,
        mapping_counts={"selected_rows": 1},
        mapping_policy="strict_one_to_one",
        orthology_type_column="orthology_type",
        allowed_orthology_types=("ortholog_one2one",),
        confidence_column="orthology_confidence",
        minimum_confidence=0.0,
        manifest_path=None,
    )
    provenance = prepare_inputs.resolve_orthology_provenance(
        **kwargs, requested_analysis_tier="auto"
    )
    assert provenance["analysis_tier"] == "sensitivity"
    assert provenance["policy_primary_claim_allowed"] is False
    with pytest.raises(ValueError, match="conflicts with the CLI-derived policy"):
        prepare_inputs.resolve_orthology_provenance(
            **kwargs, requested_analysis_tier="primary"
        )


def test_spatial_runner_requires_explicit_allowance_for_sensitivity() -> None:
    preparation = {
        "formal_primary": False,
        "orthology_policy": "one2one_bijective_all_confidence",
        "orthology_analysis_tier": "sensitivity",
        "primary_claim_allowed": False,
    }
    with pytest.raises(RuntimeError, match="--allow-nonprimary-preparation"):
        run_spatial.validate_preparation_claims(preparation, allow_nonprimary=False)
    record = run_spatial.validate_preparation_claims(preparation, allow_nonprimary=True)
    assert record["nonprimary_preparation_explicitly_allowed"] is True
    contradictory = {**preparation, "primary_claim_allowed": True}
    with pytest.raises(RuntimeError, match="Sensitivity orthology"):
        run_spatial.validate_preparation_claims(contradictory, allow_nonprimary=True)


def test_strict_projection_preserves_selected_single_log_values() -> None:
    expression = sparse.csr_matrix(
        np.array([[0.0, 1.2, 2.3], [3.4, 0.0, 4.5]], dtype=np.float32)
    )
    counts = sparse.csr_matrix(np.array([[0, 2, 3], [4, 0, 7]], dtype=np.int32))
    mapping = pd.DataFrame({"source_gene": ["b", "a"], "target_gene": ["B", "A"]})
    (
        projected_x,
        projected_counts,
        var,
        used,
        record,
    ) = common.project_expression_matrices(
        expression,
        counts,
        ["a", "b", "c"],
        mapping,
        mode="strict_log1p_rename",
    )
    assert list(var.index) == ["A", "B"]
    np.testing.assert_array_equal(
        projected_x.toarray(), expression[:, [0, 1]].toarray()
    )
    np.testing.assert_array_equal(
        projected_counts.toarray(), counts[:, [0, 1]].toarray()
    )
    assert record["normalization_target_sum"] is None
    assert record["selected_space_identity_max_abs_error"] == 0.0
    assert record["full_matrix_elementwise_comparison_applicable"] is False
    assert set(used["source_gene"]) == {"a", "b"}


def test_torch_sparse_compatibility_is_differentiable() -> None:
    torch = pytest.importorskip("torch")
    previous = sys.modules.pop("torch_sparse", None)
    try:
        backend = run_spatial._ensure_torch_sparse_backend()
        torch_sparse = __import__("torch_sparse")
        index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
        value = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
        matrix = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        observed = torch_sparse.spmm(index, value, 2, 3, matrix)
        assert torch.allclose(observed, torch.tensor([[4.0], [15.0]]))
        observed.sum().backward()
        assert value.grad is not None
        assert matrix.grad is not None
        assert backend in {"compiled_torch_sparse", "torch_native_sparse_compat_v1"}
    finally:
        sys.modules.pop("torch_sparse", None)
        if previous is not None:
            sys.modules["torch_sparse"] = previous


def test_many_to_one_projection_uses_frozen_1105_target() -> None:
    counts = sparse.csr_matrix(np.array([[1, 2, 3], [2, 2, 4]], dtype=np.int32))
    mapping = pd.DataFrame(
        {
            "source_gene": ["a", "b", "c"],
            "target_gene": ["A", "A", "C"],
        }
    )
    projected_x, projected_counts, var, _, record = common.project_expression_matrices(
        sparse.csr_matrix(counts.shape),
        counts,
        ["a", "b", "c"],
        mapping,
        mode="counts_sum_then_log1p",
        normalization_target_sum=1105.0,
    )
    np.testing.assert_array_equal(
        projected_counts.toarray(), np.array([[3, 3], [4, 4]], dtype=np.int32)
    )
    expected = np.log1p(np.array([[552.5, 552.5], [552.5, 552.5]]))
    np.testing.assert_allclose(projected_x.toarray(), expected, rtol=1e-6)
    assert list(var.index) == ["A", "C"]
    assert record["normalization_target_sum"] == 1105.0
    assert record["selected_space_identity_max_abs_error"] is None


def test_sampling_plan_is_frozen_before_database_choice() -> None:
    obs = pd.DataFrame(
        {
            "Annotation": ["A", "A", "A", "B", "B", "A"],
            "time_point_processed": [0, 0, 0, 0, 0, 1],
            "time_label": ["5hpf"] * 5 + ["10hpf"],
        }
    )
    kwargs = dict(
        cell_type_key="Annotation",
        time_key="time_point_processed",
        time_label_key="time_label",
        seeds=(101, 202),
        max_cells_per_type=2,
        minimum_cells_per_type=1,
    )
    first = common.build_sampling_plan(obs, [f"c{i}" for i in range(6)], **kwargs)
    second = common.build_sampling_plan(obs, [f"c{i}" for i in range(6)], **kwargs)
    pd.testing.assert_frame_equal(first, second)
    counts = first.groupby(["sampling_seed", "stage", "cell_type"]).size()
    assert counts.loc[(101, 0.0, "A")] == 2
    assert counts.loc[(101, 0.0, "B")] == 2
    assert not first.duplicated(["sampling_seed", "stage", "obs_name"]).any()


def test_custom_database_excludes_complexes_and_preserves_crosswalk(
    tmp_path: Path,
) -> None:
    official = pd.DataFrame(
        {
            "lr_pair": ["L1_R1"],
            "ligand_gene_symbol": ["L1"],
            "receptor_gene_symbol": ["R1"],
        }
    )
    official_path = tmp_path / "mouse_lr_pair.tsv"
    official.to_csv(official_path, sep="\t", index=False)
    custom = pd.DataFrame(
        {
            "0": ["l1", "l1", "l2_l3", "l2", "missing"],
            "1": ["r1", "r1", "r1", "r2", "r1"],
            "2": ["P1", "P2", "P3", "P4", "P5"],
            "3": ["Secreted", "Secreted", "Secreted", "Contact", "Contact"],
        }
    )
    mapping = pd.DataFrame(
        {
            "source_gene": ["l1", "r1", "l2", "r2"],
            "target_gene": ["L1", "R1", "L2", "R2"],
        }
    )
    record = common.build_lr_databases(
        official_database=official_path,
        custom_database=custom,
        mapping=mapping,
        output_dir=tmp_path / "out",
    )
    projected = pd.read_csv(
        tmp_path / "out" / "cytobridge_zebrafish_lr_projected_singletons.tsv",
        sep="\t",
    )
    assert set(projected["lr_pair"]) == {"L1_R1", "L2_R2"}
    crosswalk = pd.read_csv(tmp_path / "out" / "custom_lr_projection_crosswalk.csv")
    assert len(crosswalk) == 3  # duplicate source rows remain auditable
    excluded = pd.read_csv(tmp_path / "out" / "custom_lr_excluded_rows.csv")
    assert any(
        "ligand_complex_unrepresentable" in value
        for value in excluded["exclusion_reason"]
    )
    assert any(
        "ligand_not_in_selected_orthology" in value
        for value in excluded["exclusion_reason"]
    )
    assert record["counts"]["custom_mapped_target_pairs"] == 2
    assert record["counts"]["custom_excluded_rows"] == 2


def _write_fake_official_source(root: Path) -> None:
    src = root / "src"
    data = src / "cellagentchat_data"
    databases = data / "databases"
    databases.mkdir(parents=True)
    (src / "model_setup.py").write_text(
        "\n".join(
            [
                "def load_db(adata, file, sep): pass",
                "def load_tf_db(species, adata, rec_uni): pass",
                "def train(adata, lig_uni, rec_uni, tf_uni, rec_tf_uni, lr_pairs): pass",
                "def load_model(path, device): pass",
                "def feature_selection(model, mat, C, rec_uni): pass",
                "def add_rates(conversion_rates, rec_uni): pass",
            ]
        ),
        encoding="utf-8",
    )
    (src / "permutations.py").write_text(
        "def permutation_test(threshold, N, adata, lig_uni, rec_uni, rates, dist): pass\n",
        encoding="utf-8",
    )
    (src / "bckground_distribution.py").write_text(
        "def get_distribution(fin, dist, scaled): pass\n"
        "def get_significant_lr_pairs(lr1, fin, cutoff): pass\n",
        encoding="utf-8",
    )
    (src / "preprocessor.py").write_text(
        "def setup_adata(adata_or_path, coordinates_key, cell_type_label): pass\n",
        encoding="utf-8",
    )
    (src / "abm.py").write_text(
        "class CellModel:\n"
        "    def __init__(self): pass\n"
        "    def step(self): pass\n",
        encoding="utf-8",
    )
    (data / "mouse_lr_pair.tsv").write_text(
        "lr_pair\tligand_gene_symbol\treceptor_gene_symbol\nL_R\tL\tR\n",
        encoding="utf-8",
    )
    for filename in ("TF_TG_mouse.csv", "KEGG_mouse.csv", "REACTOME_mouse.csv"):
        (databases / filename).write_text("x\n1\n", encoding="utf-8")


def test_source_contract_is_checked_without_importing_heavy_dependencies(
    tmp_path: Path,
) -> None:
    _write_fake_official_source(tmp_path)
    record = common.inspect_official_source_api(tmp_path)
    assert "model_setup.py" in record["files"]
    assert "mouse_lr_pair.tsv" in record["files"]
    (tmp_path / "src" / "permutations.py").write_text(
        "def permutation_test(threshold, N): pass\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="lacks required parameters"):
        common.inspect_official_source_api(tmp_path)


def test_spatial_adapter_preserves_sender_receiver_direction_and_zeros() -> None:
    raw = run_spatial._flatten_raw_results(
        {"CT000_CT001": {"L-R": 2.5}},
        {"CT000": "A", "CT001": "B"},
        stage=1.0,
        stage_label="10hpf",
        sampling_seed=101,
    )
    assert raw.loc[0, ["sender_type", "receiver_type"]].tolist() == ["A", "B"]
    significant = run_spatial._flatten_significant_results(
        {"CT000_CT001": {"L-R": 2.5}},
        {"CT000_CT001": {"L-R": 0.01}},
        {"CT000": "A", "CT001": "B"},
        stage=1.0,
        stage_label="10hpf",
        sampling_seed=101,
    )
    complete = run_spatial._complete_type_pairs(
        raw,
        significant,
        ["A", "B"],
        {"A": 5, "B": 6},
        stage=1.0,
        stage_label="10hpf",
        sampling_seed=101,
        n_lr_pairs_tested=10,
    )
    assert len(complete) == 4
    target = complete[
        complete["sender_type"].eq("A") & complete["receiver_type"].eq("B")
    ].iloc[0]
    assert target["cellagentchat_native_primary"] == pytest.approx(2.5)
    assert target["cellagentchat_significant_score_sum"] == pytest.approx(2.5)
    assert target["cellagentchat_significant_lr_fraction"] == 0.1
    assert complete["cellagentchat_native_primary"].sum() == pytest.approx(2.5)


def test_lr_output_keys_are_decoded_from_loaded_universe() -> None:
    key_map = run_spatial._lr_key_map({"H2-Aa": ["Cd4"], "Lig": ["H2-Ab1"]})
    raw = run_spatial._flatten_raw_results(
        {
            "CT000_CT001": {
                "H2-Aa-Cd4": 1.0,
                "Lig-H2-Ab1": 2.0,
            }
        },
        {"CT000": "A", "CT001": "B"},
        stage=1.0,
        stage_label="10hpf",
        sampling_seed=101,
        lr_keys=key_map,
    )
    assert raw[["ligand", "receptor"]].values.tolist() == [
        ["H2-Aa", "Cd4"],
        ["Lig", "H2-Ab1"],
    ]


def _condition_manifest(label: str, database_hash: str) -> dict:
    shared = {
        "mapped_expression": {"sha256": "mapped"},
        "sample_plan": {"sha256": "samples"},
        "preparation_manifest": {"sha256": "prepare"},
        "database": {"sha256": database_hash},
        "preparation_claims": {
            "orthology_policy": "one2one_bijective_all_confidence",
            "orthology_analysis_tier": "sensitivity",
            "primary_claim_allowed": False,
        },
    }
    return {"database_condition": label, "shared_input": shared}


def test_dual_contract_requires_identical_expression_and_sample_plan() -> None:
    manifests = [
        _condition_manifest(common.CONDITION_LABELS[0], "db1"),
        _condition_manifest(common.CONDITION_LABELS[1], "db2"),
    ]
    assert run_dual.validate_paired_manifests(manifests) == {
        "mapped_expression": "mapped",
        "sample_plan": "samples",
        "preparation_manifest": "prepare",
    }
    manifests[1]["shared_input"]["sample_plan"]["sha256"] = "different"
    with pytest.raises(RuntimeError, match="same sample_plan"):
        run_dual.validate_paired_manifests(manifests)

    manifests = [
        _condition_manifest(common.CONDITION_LABELS[0], "db1"),
        _condition_manifest(common.CONDITION_LABELS[1], "db2"),
    ]
    manifests[1]["shared_input"]["preparation_claims"][
        "orthology_analysis_tier"
    ] = "primary"
    with pytest.raises(RuntimeError, match="identical preparation claims"):
        run_dual.validate_paired_manifests(manifests)
