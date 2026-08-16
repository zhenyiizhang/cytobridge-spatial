from __future__ import annotations

import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from reviewer_zebrafish_ccc.common import (
    prepare_inputs,
    sha256_file,
    stratified_subsample_indices,
    write_input_bundle,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, np.ndarray]:
    counts = np.asarray(
        [
            [2, 1, 3, 4],
            [0, 4, 2, 4],
            [3, 0, 1, 6],
            [1, 2, 0, 7],
        ],
        dtype=np.float32,
    )
    target_sum = 10.0
    normalized = np.log1p(counts * target_sum / counts.sum(axis=1, keepdims=True))
    obs = pd.DataFrame(
        {
            "Annotation": ["A", "A", "B", "B"],
            "time_point_processed": ["t0", "t0", "t1", "t1"],
            "time": [0.0, 0.0, 1.0, 1.0],
        },
        index=[f"cell_{idx}" for idx in range(4)],
    )
    data = ad.AnnData(
        X=sparse.csr_matrix(normalized),
        obs=obs,
        var=pd.DataFrame(index=["Lig", "REC", "a", "background"]),
    )
    data.layers["counts"] = sparse.csr_matrix(counts)
    data.obsm["spatial_aligned"] = np.asarray(
        [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float
    )
    data.uns["interaction_graph"] = {
        "neighborhood_threshold": 0.4,
        "recommended_spot_diameter": 0.1,
    }
    h5ad = tmp_path / "corrected.h5ad"
    data.write_h5ad(h5ad)

    database = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "0": ["lig_a", "missing"],
            "1": ["rec", "rec"],
            "2": ["P", "Q"],
            "3": ["Secreted Signaling", "Secreted Signaling"],
        }
    )
    lr_path = tmp_path / "lr.csv"
    database.to_csv(lr_path, index=False)
    audit = {
        "all_checks_passed": True,
        "normalization_and_log1p": {"resolved_target_sum": target_sum},
        "output_h5ad": {"sha256": sha256_file(h5ad)},
        "inputs": {"lr_database": {"sha256": sha256_file(lr_path)}},
    }
    audit_path = tmp_path / "preprocess_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return h5ad, lr_path, audit_path, normalized


def test_primary_preparation_uses_audit_target_and_matches_frozen_x(
    tmp_path: Path,
) -> None:
    h5ad, lr_path, audit_path, normalized = _fixture(tmp_path)
    prepared = prepare_inputs(
        h5ad, lr_path, preprocess_audit_path=audit_path, source_x_tolerance=2e-5
    )

    assert prepared.lr_database["database_row"].tolist() == [0]
    assert prepared.adata.var_names.tolist() == ["a", "lig", "rec"]
    np.testing.assert_allclose(
        prepared.adata.X.toarray(), normalized[:, [2, 0, 1]], rtol=0, atol=2e-5
    )
    assert prepared.diagnostics["resolved_target_sum"] == 10.0
    assert prepared.diagnostics["source_x_max_abs_residual"] < 2e-5
    assert (
        prepared.adata.uns["ccc_preprocessing"]["interaction_graph"][
            "neighborhood_threshold"
        ]
        == 0.4
    )
    assert prepared.stage_order == ["t0", "t1"]


def test_wrong_explicit_target_is_rejected_by_frozen_x_check(tmp_path: Path) -> None:
    h5ad, lr_path, audit_path, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="does not reproduce source X"):
        prepare_inputs(
            h5ad,
            lr_path,
            preprocess_audit_path=audit_path,
            target_sum=11.0,
            source_x_tolerance=2e-5,
        )


def test_exact_lowercase_feature_wins_over_casefold_duplicate(tmp_path: Path) -> None:
    counts = np.asarray([[9, 1, 2], [4, 2, 4]], dtype=np.float32)
    target_sum = 10.0
    normalized = np.log1p(counts * target_sum / counts.sum(axis=1, keepdims=True))
    obs = pd.DataFrame(
        {
            "Annotation": ["A", "B"],
            "time_point_processed": ["t0", "t1"],
            "time": [0.0, 1.0],
        },
        index=["cell_0", "cell_1"],
    )
    data = ad.AnnData(
        X=sparse.csr_matrix(normalized),
        obs=obs,
        var=pd.DataFrame(index=["LIG", "lig", "rec"]),
    )
    data.layers["counts"] = sparse.csr_matrix(counts)
    data.obsm["spatial_aligned"] = np.asarray([[0, 0], [1, 1]], dtype=float)
    h5ad = tmp_path / "casefold_duplicate.h5ad"
    data.write_h5ad(h5ad)

    lr_path = tmp_path / "lr.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0],
            "0": ["lig"],
            "1": ["rec"],
            "2": ["P"],
            "3": ["Secreted Signaling"],
        }
    ).to_csv(lr_path, index=False)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "all_checks_passed": True,
                "normalization_and_log1p": {"resolved_target_sum": target_sum},
                "output_h5ad": {"sha256": sha256_file(h5ad)},
                "inputs": {"lr_database": {"sha256": sha256_file(lr_path)}},
            }
        ),
        encoding="utf-8",
    )

    prepared = prepare_inputs(
        h5ad, lr_path, preprocess_audit_path=audit_path, source_x_tolerance=2e-5
    )
    assert prepared.adata.var_names.tolist() == ["lig", "rec"]
    np.testing.assert_allclose(
        prepared.adata.X.toarray(), normalized[:, [1, 2]], rtol=0, atol=2e-5
    )


def test_bundle_writes_same_cells_expression_and_spatial_for_all_methods(
    tmp_path: Path,
) -> None:
    h5ad, lr_path, audit_path, _ = _fixture(tmp_path)
    prepared = prepare_inputs(
        h5ad, lr_path, preprocess_audit_path=audit_path, source_x_tolerance=2e-5
    )
    out = tmp_path / "bundle"
    manifest = write_input_bundle(
        prepared,
        out,
        source_h5ad=h5ad,
        source_lr_database=lr_path,
        source_preprocess_audit=audit_path,
    )
    assert [record["stage"] for record in manifest["stages"]] == ["t0", "t1"]
    assert manifest["preprocessing"]["target_sum"] == 10.0
    assert (
        manifest["preprocessing"]["interaction_graph"]["recommended_spot_diameter"]
        == 0.1
    )
    for record in manifest["stages"]:
        stage_dir = out / "stages" / record["token"]
        metadata = pd.read_csv(stage_dir / "metadata.csv")
        spatial = pd.read_csv(stage_dir / "spatial_aligned.csv")
        assert metadata["cell_id"].tolist() == spatial["cell_id"].tolist()


def test_stratified_subsample_is_deterministic_and_retains_labels() -> None:
    labels = ["A"] * 8 + ["B"] * 3 + ["C"]
    first = stratified_subsample_indices(labels, max_cells=6, seed=7)
    second = stratified_subsample_indices(labels, max_cells=6, seed=7)
    assert np.array_equal(first, second)
    assert set(np.asarray(labels)[first]) == {"A", "B", "C"}


def test_developmental_time_labels_are_parsed_with_a_consistent_unit(
    tmp_path: Path,
) -> None:
    h5ad, lr_path, _, _ = _fixture(tmp_path)
    data = ad.read_h5ad(h5ad)
    data.obs["time"] = pd.Categorical(["5.25hpf", "5.25hpf", "10hpf", "10hpf"])
    data.write_h5ad(h5ad)
    audit_path = tmp_path / "preprocess_audit_hpf.json"
    audit_path.write_text(
        json.dumps(
            {
                "all_checks_passed": True,
                "normalization_and_log1p": {"resolved_target_sum": 10.0},
                "output_h5ad": {"sha256": sha256_file(h5ad)},
                "inputs": {"lr_database": {"sha256": sha256_file(lr_path)}},
            }
        ),
        encoding="utf-8",
    )

    prepared = prepare_inputs(
        h5ad, lr_path, preprocess_audit_path=audit_path, source_x_tolerance=2e-5
    )
    assert prepared.stage_order == ["t0", "t1"]
    assert prepared.stage_times == {"t0": 5.25, "t1": 10.0}


def test_preferred_species_projection_uses_exact_compound_feature_symbols(
    tmp_path: Path,
) -> None:
    counts = np.asarray([[3, 2, 5], [2, 4, 4]], dtype=np.float32)
    target_sum = 10.0
    normalized = np.log1p(counts * target_sum / counts.sum(axis=1, keepdims=True))
    data = ad.AnnData(
        X=sparse.csr_matrix(normalized),
        obs=pd.DataFrame(
            {
                "Annotation": ["A", "B"],
                "time_point_processed": ["t1", "t1"],
            },
            index=["cell_0", "cell_1"],
        ),
        var=pd.DataFrame(
            index=["FZD10 | AMEX60DD000016", "FGFR1[hs] | AMEX0002", "background"]
        ),
    )
    data.layers["counts"] = sparse.csr_matrix(counts)
    data.obsm["spatial_aligned"] = np.asarray([[0, 0], [1, 1]], dtype=float)
    h5ad = tmp_path / "compound.h5ad"
    data.write_h5ad(h5ad)
    lr_path = tmp_path / "human_lr.csv"
    pd.DataFrame(
        {
            "ligand": ["FZD10"],
            "receptor": ["FGFR1"],
            "pathway": ["FGF"],
            "category": ["Secreted Signaling"],
        }
    ).to_csv(lr_path, index=False)

    prepared = prepare_inputs(
        h5ad,
        lr_path,
        target_sum=target_sum,
        time_col=None,
        source_x_tolerance=2e-5,
        preferred_species_tag="hs",
    )
    assert prepared.adata.var_names.tolist() == ["fgfr1", "fzd10"]
    np.testing.assert_allclose(
        prepared.adata.X.toarray(), normalized[:, [1, 0]], rtol=0, atol=2e-5
    )
    assert prepared.diagnostics["preferred_species_tag"] == "hs"


def test_time_col_none_writes_float_nan_stage_times(tmp_path: Path) -> None:
    h5ad, lr_path, audit_path, _ = _fixture(tmp_path)
    prepared = prepare_inputs(
        h5ad,
        lr_path,
        preprocess_audit_path=audit_path,
        time_col=None,
        source_x_tolerance=2e-5,
    )
    out = tmp_path / "bundle_without_time"
    write_input_bundle(
        prepared,
        out,
        source_h5ad=h5ad,
        source_lr_database=lr_path,
        source_preprocess_audit=audit_path,
    )
    written = ad.read_h5ad(out / "normalized_lr_expression.h5ad")
    assert written.obs["ccc_stage_time"].dtype.kind == "f"
    assert written.obs["ccc_stage_time"].isna().all()
