from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist

from CytoBridge.pp import validate_prepared_chicken_heart_input


SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_chicken_heart_input.py"
SPEC = importlib.util.spec_from_file_location("prepare_chicken_heart_input", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fixtures(monkeypatch):
    monkeypatch.setattr(
        MODULE, "EXPECTED_COUNTS", {time: 1 for time in MODULE.TIMEPOINTS}
    )
    names = [f"barcode-{time}_{time}" for time in MODULE.TIMEPOINTS]
    obs = pd.DataFrame(
        {
            "timepoint": list(MODULE.TIMEPOINTS),
            "region": ["Atria", "Atria", "Ventricle", "Ventricle"],
            "celltype_prediction": ["a", "b", "c", "d"],
        },
        index=names,
    )
    metadata = ad.AnnData(X=np.zeros((4, 2)), obs=obs.copy())
    metadata.obsm["spatial"] = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    aligned = ad.AnnData(X=np.zeros((4, 2)), obs=obs.copy())
    aligned.obsm["spatial_aligned"] = np.asarray(
        [[0.0, 0.0], [0.0, -1.0], [2.0, 0.0], [2.0, 2.0]]
    )
    raw = {}
    for index, timepoint in enumerate(MODULE.TIMEPOINTS):
        current = ad.AnnData(
            X=sparse.csr_matrix([[index + 1.0, index + 2.0]]),
            obs=pd.DataFrame(index=[f"barcode-{timepoint}"]),
            var=pd.DataFrame(index=["gene-a", "gene-b"]),
        )
        raw[timepoint] = current
    return raw, metadata, aligned


def test_assemble_reviewed_counts_preserves_reference_order_and_coordinates(
    monkeypatch,
):
    raw, metadata, aligned = _fixtures(monkeypatch)

    result = MODULE.assemble_reviewed_counts(raw, metadata, aligned)

    assert result.obs_names.tolist() == aligned.obs_names.tolist()
    assert result.obs["Annotation"].tolist() == [
        "a",
        "b",
        "c",
        "d",
    ]
    assert result.obs["region"].tolist() == [
        "Atria",
        "Atria",
        "Ventricle",
        "Ventricle",
    ]
    np.testing.assert_array_equal(
        result.layers["counts"].toarray(),
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
    )
    np.testing.assert_array_equal(
        result.obsm["spatial_aligned"], aligned.obsm["spatial_aligned"]
    )
    np.testing.assert_array_equal(
        result.obsm["spatial_original"], metadata.obsm["spatial"]
    )


def test_reference_contract_rejects_coordinate_or_metadata_drift(monkeypatch):
    _, metadata, aligned = _fixtures(monkeypatch)
    contract = MODULE._validate_reference(metadata, aligned)
    assert contract["coordinate_source"] == "reviewed_aligned_reference_exact_copy"
    assert len(contract["coordinate_sha256"]) == 64

    aligned.obs.loc[aligned.obs_names[1], "region"] = "wrong"
    with np.testing.assert_raises_regex(MODULE.ContractError, "changed metadata"):
        MODULE._validate_reference(metadata, aligned)


def test_orientation_qc_records_reflection_without_realigning(monkeypatch):
    raw, metadata, aligned = _fixtures(monkeypatch)
    # Use two non-collinear points per timepoint to make the affine orientation
    # test meaningful while retaining a tiny synthetic fixture.
    monkeypatch.setattr(
        MODULE, "EXPECTED_COUNTS", {time: 3 for time in MODULE.TIMEPOINTS}
    )
    names = []
    obs_rows = []
    raw = {}
    raw_coords = []
    aligned_coords = []
    for timepoint in MODULE.TIMEPOINTS:
        barcodes = [f"{timepoint}-a", f"{timepoint}-b", f"{timepoint}-c"]
        names.extend(f"{barcode}_{timepoint}" for barcode in barcodes)
        obs_rows.extend((timepoint, "Atria", "x") for _ in barcodes)
        raw[timepoint] = ad.AnnData(
            X=sparse.csr_matrix(np.ones((3, 2))),
            obs=pd.DataFrame(index=barcodes),
            var=pd.DataFrame(index=["gene-a", "gene-b"]),
        )
        coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        raw_coords.extend(coordinates)
        transformed = coordinates.copy()
        if timepoint == "D7":
            transformed[:, 1] *= -1.0
        aligned_coords.extend(transformed)
    obs = pd.DataFrame(
        obs_rows,
        columns=["timepoint", "region", "celltype_prediction"],
        index=names,
    )
    metadata = ad.AnnData(X=np.zeros((12, 1)), obs=obs.copy())
    metadata.obsm["spatial"] = np.asarray(raw_coords)
    aligned = ad.AnnData(X=np.zeros((12, 1)), obs=obs.copy())
    aligned.obsm["spatial_aligned"] = np.asarray(aligned_coords)
    assembled = MODULE.assemble_reviewed_counts(raw, metadata, aligned)

    qc = MODULE._orientation_qc(assembled)

    assert qc["D7"]["raw_to_reference_affine_orientation"] == "reflected"
    assert all(
        qc[time]["raw_to_reference_affine_orientation"] == "preserved"
        for time in ("D4", "D10", "D14")
    )


def _anatomical_fixture(*, mirrored_d7: bool = True) -> ad.AnnData:
    rows = []
    coordinates = []
    by_stage = {
        "D4": {
            "Atria": (0.0, 3.0),
            "Valves": (0.0, 2.0),
            "Ventricle": (0.0, 1.0),
        },
        "D7": {
            "Atria": (0.0, 3.0),
            "Valves": (0.0, 2.0),
            "Compact LV and \ninter-ventricular septum": (1.0, 1.0),
            "Right ventricle": (-1.0 if mirrored_d7 else 2.0, 1.0),
        },
        "D10": {
            "Atria": (0.0, 3.0),
            "Valves": (0.0, 2.0),
            "Compact LV and \ninter-ventricular septum": (-1.0, 1.0),
            "Right ventricle": (1.0, 1.0),
        },
        "D14": {
            "Atria": (0.0, 3.0),
            "Valves": (0.0, 2.0),
            "Compact LV and \ninter-ventricular septum": (-1.5, 1.0),
            "Right ventricle": (1.5, 1.0),
        },
    }
    for timepoint, regions in by_stage.items():
        for index, (region, coordinate) in enumerate(regions.items()):
            rows.append((f"{timepoint}-{index}", timepoint, region))
            coordinates.append(coordinate)
    obs = pd.DataFrame(rows, columns=["name", "timepoint", "region"]).set_index("name")
    result = ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    result.obsm["spatial_aligned"] = np.asarray(coordinates, dtype=np.float64)
    result.obsm["spatial"] = np.asarray(coordinates, dtype=np.float64)
    return result


def test_anatomical_contract_rejects_legacy_d7_mirror_by_default():
    fixture = _anatomical_fixture()
    qc = MODULE._anatomical_orientation_qc(fixture)
    assert qc["status"] == "fail"
    assert qc["failures"] == [
        "D7 Right ventricle is horizontally mirrored relative to Compact LV"
    ]

    with np.testing.assert_raises_regex(MODULE.ContractError, "D7 left/right mirror"):
        MODULE._apply_anatomical_coordinate_contract(
            fixture, repair_legacy_d7_left_right=False
        )


def test_explicit_d7_repair_preserves_distances_and_other_stages():
    fixture = _anatomical_fixture()
    before = np.asarray(fixture.obsm["spatial_aligned"]).copy()
    record = MODULE._apply_anatomical_coordinate_contract(
        fixture, repair_legacy_d7_left_right=True
    )
    after = np.asarray(fixture.obsm["spatial_aligned"])
    d7 = fixture.obs["timepoint"].astype(str).eq("D7").to_numpy()

    assert record["applied"] is True
    assert record["after_anatomical_qc"]["status"] == "pass"
    assert record["before_coordinate_sha256"] != record["after_coordinate_sha256"]
    assert record["pairwise_distance_max_abs_error"] == 0.0
    np.testing.assert_array_equal(after[~d7], before[~d7])
    np.testing.assert_allclose(np.sort(pdist(after[d7])), np.sort(pdist(before[d7])))


def test_d7_compatibility_repair_refuses_other_orientation_failures():
    fixture = _anatomical_fixture()
    d10_atria = fixture.obs["timepoint"].astype(str).eq("D10") & fixture.obs[
        "region"
    ].astype(str).eq("Atria")
    fixture.obsm["spatial_aligned"][d10_atria.to_numpy(), 1] = 0.0

    with np.testing.assert_raises_regex(MODULE.ContractError, "D10 Atria"):
        MODULE._apply_anatomical_coordinate_contract(
            fixture, repair_legacy_d7_left_right=True
        )


def _prepared_annotation_fixture() -> ad.AnnData:
    fixture = _anatomical_fixture(mirrored_d7=False)
    fixture.obs["celltype_prediction"] = [
        f"celltype-{index % 4}" for index in range(fixture.n_obs)
    ]
    fixture.obs["Annotation"] = fixture.obs["celltype_prediction"].astype(str)
    fixture.obs["time_point_processed"] = fixture.obs["timepoint"].map(
        MODULE.TIME_MAPPING
    )
    fixture.obsm["X_latent"] = np.zeros((fixture.n_obs, 50), dtype=np.float32)
    fixture.layers["counts"] = sparse.csr_matrix(
        np.ones((fixture.n_obs, fixture.n_vars), dtype=np.float32)
    )
    repair = MODULE._apply_anatomical_coordinate_contract(
        fixture, repair_legacy_d7_left_right=False
    )
    labels = fixture.obs["celltype_prediction"].astype(str).tolist()
    fixture.uns["chicken_heart_input_contract_json"] = json.dumps(
        {
            "schema_version": 3,
            "coordinate_repair": repair,
            "downstream_annotation": {
                "key": "celltype_prediction",
                "compatibility_key": "Annotation",
                "source": "metadata_h5ad",
                "n_classes": len(set(labels)),
                "ordered_label_sha256": MODULE._text_sha256(labels),
            },
        }
    )
    return fixture


def test_prepared_contract_binds_celltype_labels_separately_from_region():
    fixture = _prepared_annotation_fixture()

    contract = validate_prepared_chicken_heart_input(fixture)

    assert contract["schema_version"] == 3
    assert contract["downstream_annotation_key"] == "celltype_prediction"

    fixture.obs["Annotation"] = fixture.obs["region"].astype(str)
    with np.testing.assert_raises_regex(
        MODULE.ContractError, "must match.*celltype_prediction"
    ):
        validate_prepared_chicken_heart_input(fixture)


def test_prepared_contract_rejects_celltype_label_drift():
    fixture = _prepared_annotation_fixture()
    fixture.obs.loc[fixture.obs_names[0], "celltype_prediction"] = "changed"
    fixture.obs["Annotation"] = fixture.obs["celltype_prediction"].astype(str)

    with np.testing.assert_raises_regex(MODULE.ContractError, "labels do not match"):
        validate_prepared_chicken_heart_input(fixture)


def test_legacy_prepared_input_reuses_exact_state_but_ignores_region_alias():
    fixture = _prepared_annotation_fixture()
    contract = json.loads(fixture.uns["chicken_heart_input_contract_json"])
    contract["schema_version"] = 2
    contract.pop("downstream_annotation")
    fixture.uns["chicken_heart_input_contract_json"] = json.dumps(contract)
    fixture.obs["Annotation"] = fixture.obs["region"].astype(str)

    validated = validate_prepared_chicken_heart_input(fixture)

    assert validated["schema_version"] == 2
    assert validated["downstream_annotation_key"] == "celltype_prediction"
    assert validated["legacy_annotation_alias_ignored"] is True
