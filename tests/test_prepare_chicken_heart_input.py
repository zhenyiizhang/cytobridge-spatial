from __future__ import annotations

import inspect
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import pdist

import CytoBridge.pp.chicken_heart_input as heart_input
from CytoBridge.pp import (
    apply_chicken_heart_coordinate_validation,
    prepare_chicken_heart_input,
    validate_prepared_chicken_heart_input,
)


def test_public_preparation_api_defaults_to_no_legacy_repair():
    parameter = inspect.signature(prepare_chicken_heart_input).parameters[
        "repair_legacy_d7_left_right"
    ]
    assert parameter.default is False


def _count_fixtures(monkeypatch):
    monkeypatch.setattr(
        heart_input, "EXPECTED_COUNTS", {stage: 1 for stage in heart_input.TIMEPOINTS}
    )
    names = [f"barcode-{stage}_{stage}" for stage in heart_input.TIMEPOINTS]
    obs = pd.DataFrame(
        {
            "timepoint": list(heart_input.TIMEPOINTS),
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
    raw = {
        stage: ad.AnnData(
            X=sparse.csr_matrix([[index + 1.0, index + 2.0]]),
            obs=pd.DataFrame(index=[f"barcode-{stage}"]),
            var=pd.DataFrame(index=["gene-a", "gene-b"]),
        )
        for index, stage in enumerate(heart_input.TIMEPOINTS)
    }
    return raw, metadata, aligned


def test_reference_count_assembly_preserves_rows_counts_and_coordinates(monkeypatch):
    raw, metadata, aligned = _count_fixtures(monkeypatch)
    summary = heart_input._validate_reference_input(metadata, aligned)
    result = heart_input.assemble_chicken_heart_reference_counts(raw, metadata, aligned)

    assert summary == {
        "n_obs": 4,
        "timepoint_counts": {stage: 1 for stage in heart_input.TIMEPOINTS},
        "coordinate_shape": [4, 2],
        "coordinate_source": "aligned_reference",
    }
    assert result.obs_names.tolist() == aligned.obs_names.tolist()
    np.testing.assert_array_equal(
        result.layers["counts"].toarray(),
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
    )
    np.testing.assert_array_equal(
        result.obsm["spatial_aligned"], aligned.obsm["spatial_aligned"]
    )

    aligned.obs.loc[aligned.obs_names[1], "region"] = "different"
    with np.testing.assert_raises_regex(ValueError, "does not match"):
        heart_input._validate_reference_input(metadata, aligned)


def _anatomical_fixture(*, mirrored_d7: bool) -> ad.AnnData:
    rows: list[tuple[str, str, str]] = []
    coordinates: list[tuple[float, float]] = []
    stages = {
        "D4": {
            "Atria": (0.0, 3.0),
            "Valves": (0.0, 2.0),
            "Ventricle": (0.0, 1.0),
        },
        "D7": {
            "Atria": (0.0, 3.0),
            "Valves": (0.0, 2.0),
            "Compact LV and inter-ventricular septum": (1.0, 1.0),
            "Right ventricle": (-1.0 if mirrored_d7 else 2.0, 1.0),
        },
        "D10": {
            "Atria": (0.0, 3.0),
            "Valves": (0.0, 2.0),
            "Compact LV and inter-ventricular septum": (-1.0, 1.0),
            "Right ventricle": (1.0, 1.0),
        },
        "D14": {
            "Atria": (0.0, 3.0),
            "Valves": (0.0, 2.0),
            "Compact LV and inter-ventricular septum": (-1.5, 1.0),
            "Right ventricle": (1.5, 1.0),
        },
    }
    for stage, regions in stages.items():
        for index, (region, coordinate) in enumerate(regions.items()):
            rows.append((f"{stage}-{index}", stage, region))
            coordinates.append(coordinate)
    obs = pd.DataFrame(rows, columns=["name", "timepoint", "region"]).set_index("name")
    result = ad.AnnData(X=np.zeros((len(obs), 1)), obs=obs)
    result.obsm["spatial_aligned"] = np.asarray(coordinates, dtype=np.float64)
    result.obsm["spatial"] = np.asarray(coordinates, dtype=np.float64)
    return result


def test_public_coordinate_validation_has_no_digest_fields():
    fixture = _anatomical_fixture(mirrored_d7=True)
    before = np.asarray(fixture.obsm["spatial_aligned"]).copy()
    record = apply_chicken_heart_coordinate_validation(
        fixture, repair_legacy_d7_left_right=True
    )
    after = np.asarray(fixture.obsm["spatial_aligned"])
    d7 = fixture.obs["timepoint"].astype(str).eq("D7").to_numpy()

    assert record["applied"] is True
    assert record["policy"] == "legacy_d7_horizontal_reflection"
    assert not any("sha" in key.lower() for key in record)
    np.testing.assert_array_equal(after[~d7], before[~d7])
    np.testing.assert_allclose(np.sort(pdist(after[d7])), np.sort(pdist(before[d7])))


def test_current_prepared_input_validation_uses_counts_and_annotations():
    fixture = _anatomical_fixture(mirrored_d7=False)
    fixture.obs["celltype_prediction"] = [
        f"celltype-{index % 4}" for index in range(fixture.n_obs)
    ]
    fixture.obs["Annotation"] = fixture.obs["celltype_prediction"].astype(str)
    fixture.obs["time_point_processed"] = fixture.obs["timepoint"].map(
        heart_input.TIME_MAPPING
    )
    fixture.obsm["X_latent"] = np.zeros((fixture.n_obs, 50), dtype=np.float32)
    fixture.layers["counts"] = sparse.csr_matrix(np.ones((fixture.n_obs, 1)))
    counts = fixture.obs["timepoint"].astype(str).value_counts(sort=False).to_dict()
    fixture.uns["chicken_heart_input_validation_json"] = json.dumps(
        {
            "schema_version": 4,
            "reference": {"n_obs": fixture.n_obs, "timepoint_counts": counts},
            "coordinate_adjustment": {
                "applied": False,
                "policy": "reference_coordinates_validated",
            },
            "downstream_annotation": {
                "key": "celltype_prediction",
                "n_classes": 4,
            },
        }
    )

    result = validate_prepared_chicken_heart_input(fixture)

    assert result["schema_version"] == 4
    assert result["coordinate_policy"] == "reference_coordinates_validated"
    fixture.obs.loc[fixture.obs_names[0], "celltype_prediction"] = "new-class"
    fixture.obs["Annotation"] = fixture.obs["celltype_prediction"].astype(str)
    with np.testing.assert_raises_regex(ValueError, "class count"):
        validate_prepared_chicken_heart_input(fixture)
