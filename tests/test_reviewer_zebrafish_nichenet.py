from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "reviewer_zebrafish_ccc" / "nichenet"
PREPARE_SCRIPT = SCRIPT_DIR / "prepare_shared_inputs.py"
SPEC = importlib.util.spec_from_file_location(
    "reviewer_zebrafish_nichenet_prepare", PREPARE_SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_single_log_reconstruction_and_validation_rejects_relog():
    counts = sparse.csr_matrix(np.asarray([[1, 0], [3, 1]], dtype=float))
    normalized, logged, libraries = MODULE.single_log_from_counts(
        counts, target_sum=10.0
    )

    np.testing.assert_allclose(libraries, [1.0, 4.0])
    np.testing.assert_allclose(
        normalized.toarray(),
        [[10.0, 0.0], [7.5, 2.5]],
    )
    audit = MODULE.validate_single_log_x(logged, logged, tolerance=1e-12)
    assert audit["passed"] is True
    assert audit["support_mismatch_count"] == 0

    relogged = logged.copy()
    relogged.data = np.log1p(relogged.data)
    with pytest.raises(ValueError, match="not the expected single-log"):
        MODULE.validate_single_log_x(relogged, logged, tolerance=1e-12)


def test_strict_orthology_requires_confident_symbol_bijection(tmp_path):
    table = pd.DataFrame(
        {
            "external_gene_name": ["z1", "z2", "z3", "z4", "z5", "z6", "z7"],
            "mmusculus_homolog_associated_gene_name": [
                "M1",
                "M2",
                "M3",
                "M4",
                "MX",
                "MX",
                "M7",
            ],
            "mmusculus_homolog_orthology_type": [
                "ortholog_one2one",
                "ortholog_one2one",
                "ortholog_one2many",
                "ortholog_one2one",
                "ortholog_one2one",
                "ortholog_one2one",
                "ortholog_one2one",
            ],
            "mmusculus_homolog_orthology_confidence": [1, 1, 1, 0, 1, 1, 0.999999],
            "ensembl_gene_id": [f"ENSDARG{i}" for i in range(7)],
            "mmusculus_homolog_ensembl_gene": [f"ENSMUSG{i}" for i in range(7)],
        }
    )
    path = tmp_path / "orthology.csv"
    table.to_csv(path, index=False)

    strict, audit = MODULE.load_strict_one_to_one_orthology(path)
    assert strict[["zebrafish_symbol", "mouse_symbol"]].values.tolist() == [
        ["z1", "M1"],
        ["z2", "M2"],
    ]
    assert audit["strict_bijective_symbol_pairs"] == 2
    assert audit["filter"]["orthology_confidence"] == 1


def test_custom_lr_mapping_uses_receptor_components_and_rejects_composite_ligands(
    tmp_path,
):
    strict = pd.DataFrame(
        {
            "zebrafish_symbol": ["liga", "ligb", "reca", "r1", "r2"],
            "mouse_symbol": ["Liga", "Ligb", "Reca", "R1", "R2"],
            "orthology_type": ["ortholog_one2one"] * 5,
            "orthology_confidence": [1] * 5,
            "zebrafish_ensembl_gene": [""] * 5,
            "mouse_ensembl_gene": [""] * 5,
        }
    )
    lr = pd.DataFrame(
        {
            "0": ["liga", "liga", "liga_ligb", "missing", "liga"],
            "1": ["reca", "r1_r2", "reca", "reca", "reca"],
            "2": ["P1", "P2", "P3", "P4", "P1"],
            "3": ["Secreted"] * 5,
        }
    )
    path = tmp_path / "lr.csv"
    lr.to_csv(path, index=False)

    mapped, audit_frame, summary = MODULE.map_custom_lr_database(path, strict)
    assert set(mapped["receptor_mouse_components"]) == {"Reca", "R1;R2"}
    assert summary["eligible_unique_lr_pairs"] == 2
    assert summary["exact_duplicate_rows"] == 1
    composite = audit_frame[audit_frame["ligand_zebrafish"] == "liga_ligb"].iloc[0]
    assert composite["eligible_for_custom_nichenet_gate"] == np.bool_(False)
    assert (
        "unsupported_multisubunit_ligand_prior_column" in composite["exclusion_reason"]
    )


def test_formal_orthology_manifest_is_release_116_and_hash_bound(tmp_path):
    h5ad, orthology, lr = _write_synthetic_inputs(tmp_path)
    manifest_path = tmp_path / "orthology_manifest.json"
    payload = {
        "workflow": "ensembl_compara_zebrafish_mouse_strict_one2one_export",
        "status": "complete",
        "ensembl_release": 116,
        "source_mode": "frozen_raw_input",
        "filter": {
            "orthology_type": "ortholog_one2one",
            "orthology_confidence": 1,
            "nonempty_symbols": True,
            "symbol_level_bijection_after_casefold": True,
        },
        "output_md5": {"strict": MODULE._md5(orthology)},
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    config = MODULE.PrepareConfig(
        h5ad=h5ad,
        orthology_csv=orthology,
        orthology_manifest=manifest_path,
        custom_lr_db=lr,
        out_dir=tmp_path / "formal",
    )
    audit = MODULE.validate_orthology_provenance(config)
    assert audit["verified"] is True
    assert audit["ensembl_release"] == 116

    payload["ensembl_release"] = 115
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ensembl_release=116"):
        MODULE.validate_orthology_provenance(config)


def _write_synthetic_inputs(tmp_path: Path, *, corrupt_x: bool = False):
    genes = ["liga", "reca", "target", "background", "r1", "r2"]
    rows = []
    obs_rows = []
    for stage in (0.0, 1.0):
        for cell_type in ("Receiver", "Sender"):
            for _ in range(6):
                if cell_type == "Receiver":
                    row = [0, 3, 0 if stage == 0 else 20, 2, 2, 2]
                else:
                    row = [5, 0, 0, 2, 0, 0]
                rows.append(row)
                obs_rows.append(
                    {"time_point_processed": stage, "Annotation": cell_type}
                )
    counts = sparse.csr_matrix(np.asarray(rows, dtype=np.float64))
    _, logged, _ = MODULE.single_log_from_counts(counts, target_sum=100.0)
    x = logged.copy()
    if corrupt_x:
        x.data = np.log1p(x.data)
    adata = ad.AnnData(
        X=x,
        obs=pd.DataFrame(obs_rows, index=[f"cell_{i}" for i in range(len(rows))]),
        var=pd.DataFrame(index=genes),
    )
    adata.layers["counts"] = counts
    h5ad = tmp_path / ("bad.h5ad" if corrupt_x else "good.h5ad")
    adata.write_h5ad(h5ad)

    orthology = pd.DataFrame(
        {
            "zebrafish_symbol": genes,
            "mouse_symbol": ["Liga", "Reca", "Target", "Background", "R1", "R2"],
            "orthology_type": ["ortholog_one2one"] * len(genes),
            "orthology_confidence": [1] * len(genes),
        }
    )
    orthology_path = tmp_path / "orthology.csv"
    orthology.to_csv(orthology_path, index=False)

    lr = pd.DataFrame(
        {
            "0": ["liga", "liga"],
            "1": ["reca", "r1_r2"],
            "2": ["P1", "P2"],
            "3": ["Secreted", "Contact"],
        }
    )
    lr_path = tmp_path / "lr.csv"
    lr.to_csv(lr_path, index=False)
    return h5ad, orthology_path, lr_path


def test_prepare_shared_inputs_writes_auditable_transition_units(tmp_path):
    h5ad, orthology, lr = _write_synthetic_inputs(tmp_path)
    out = tmp_path / "prepared"
    config = MODULE.PrepareConfig(
        h5ad=h5ad,
        orthology_csv=orthology,
        custom_lr_db=lr,
        out_dir=out,
        formal_mode=False,
        transitions=(("0", "1"),),
        normalization_target_sum=100.0,
        normalized_x_tolerance=1e-10,
        min_cells_per_receiver_stage=5,
        min_expression_fraction=0.05,
        min_abs_log2fc=0.25,
        fdr_cutoff=1.0,
        min_target_genes=1,
        min_background_genes=1,
        de_chunk_size=2,
        stage_label_map={"0": "early", "1": "late"},
    )
    manifest = MODULE.prepare_shared_inputs(config)

    assert manifest["normalization"]["frozen_target_sum"] == 100.0
    assert (
        manifest["normalization"]["raw_counts"]["library_median_retained_cells"]
        != manifest["normalization"]["frozen_target_sum"]
    )
    assert manifest["normalization"]["x_validation"]["passed"] is True
    assert manifest["formal_mode"] is False
    assert all("md5" in record for record in manifest["output_files"])
    assert (out / "prepare_manifest.json").is_file()
    assert (out / "coverage_summary.csv").is_file()
    assert (out / "expression_by_stage_celltype.csv.gz").is_file()
    assert (out / "custom_lr_mapping_audit.csv").is_file()

    units = pd.read_csv(out / "units_manifest.csv")
    receiver = units[units["receiver"] == "Receiver"].iloc[0]
    assert receiver["status"] == "eligible"
    unit_dir = out / receiver["unit_dir"]
    geneset = pd.read_csv(unit_dir / "receiver_response_genes.csv")
    assert "Target" in set(geneset["gene_mouse"])
    metadata = pd.read_json(unit_dir / "unit_metadata.json", typ="series")
    assert metadata["source_stage_label"] == "early"
    assert metadata["target_stage_label"] == "late"


def test_prepare_shared_inputs_refuses_nonmatching_x(tmp_path):
    h5ad, orthology, lr = _write_synthetic_inputs(tmp_path, corrupt_x=True)
    config = MODULE.PrepareConfig(
        h5ad=h5ad,
        orthology_csv=orthology,
        custom_lr_db=lr,
        out_dir=tmp_path / "bad_prepared",
        formal_mode=False,
        transitions=(("0", "1"),),
        normalization_target_sum=100.0,
        min_cells_per_receiver_stage=1,
        min_target_genes=1,
        min_background_genes=1,
    )
    with pytest.raises(ValueError, match="not the expected single-log"):
        MODULE.prepare_shared_inputs(config)


def test_formal_mode_enforces_target_and_x_verification(tmp_path):
    h5ad, orthology, lr = _write_synthetic_inputs(tmp_path)
    wrong_target = MODULE.PrepareConfig(
        h5ad=h5ad,
        orthology_csv=orthology,
        custom_lr_db=lr,
        out_dir=tmp_path / "wrong_target",
        normalization_target_sum=100.0,
    )
    with pytest.raises(
        ValueError, match="requires normalization_target_sum exactly 1105"
    ):
        MODULE.prepare_shared_inputs(wrong_target)

    skipped_x = MODULE.PrepareConfig(
        h5ad=h5ad,
        orthology_csv=orthology,
        custom_lr_db=lr,
        out_dir=tmp_path / "skipped_x",
        normalization_target_sum=1105.0,
        verify_x=False,
    )
    with pytest.raises(ValueError, match="requires X verification"):
        MODULE.prepare_shared_inputs(skipped_x)

    missing_orthology_manifest = MODULE.PrepareConfig(
        h5ad=h5ad,
        orthology_csv=orthology,
        custom_lr_db=lr,
        out_dir=tmp_path / "missing_orthology_manifest",
        normalization_target_sum=1105.0,
    )
    with pytest.raises(ValueError, match="requires --orthology-manifest"):
        MODULE.prepare_shared_inputs(missing_orthology_manifest)


def test_prepare_shared_inputs_emits_zero_unit_manifest(tmp_path):
    h5ad, orthology, lr = _write_synthetic_inputs(tmp_path)
    adata = ad.read_h5ad(h5ad)
    target = adata.obs["time_point_processed"].astype(float).eq(1.0)
    adata.obs["Annotation"] = adata.obs["Annotation"].astype(object)
    adata.obs.loc[target, "Annotation"] = "LateOnly"
    disjoint_h5ad = tmp_path / "disjoint.h5ad"
    adata.write_h5ad(disjoint_h5ad)

    out = tmp_path / "zero_units"
    manifest = MODULE.prepare_shared_inputs(
        MODULE.PrepareConfig(
            h5ad=disjoint_h5ad,
            orthology_csv=orthology,
            custom_lr_db=lr,
            out_dir=out,
            formal_mode=False,
            transitions=(("0", "1"),),
            normalization_target_sum=100.0,
            min_cells_per_receiver_stage=1,
            min_target_genes=1,
            min_background_genes=1,
        )
    )
    units = pd.read_csv(out / "units_manifest.csv")
    assert units.empty
    assert "status" in units.columns
    assert manifest["receiver_units"]["n_rows"] == 0


def test_r_runner_uses_official_nichenetr_activity_and_fixed_prior():
    runner = (SCRIPT_DIR / "run_nichenet_v2.R").read_text(encoding="utf-8")
    orthology_exporter = (SCRIPT_DIR / "export_ensembl_one2one.R").read_text(
        encoding="utf-8"
    )

    assert "nichenetr::predict_ligand_activities" in runner
    assert "aupr_corrected" in runner
    assert "ligand_target_matrix_nsga2r_final_mouse.rds" in runner
    assert 'mode %in% c("default", "custom")' in runner
    assert "prepare_manifest <- fromJSON" in runner
    assert "shared_file_integrity" in runner
    assert "prepare_manifest$orthology_source$ensembl_release" in runner
    assert "valid_sha256" in runner
    assert "status = run_status" in runner
    assert "target_link_errors.csv" in runner
    assert "if (nrow(custom_lr) == 0)" in runner
    assert "all(matched$pct_detected >= min_expression_fraction)" in runner
    assert "ortholog_one2one" in orthology_exporter
    assert "version = ensembl_version" in orthology_exporter
