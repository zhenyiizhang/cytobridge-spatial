"""Anatomy-aware coordinate contract for the GSE149457 chicken heart."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist


CHICKEN_HEART_TIMEPOINTS = ("D4", "D7", "D10", "D14")


class ChickenHeartContractError(ValueError):
    """Raised when fixed chicken-heart coordinates violate anatomy."""


def _coordinate_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(f"|{array.dtype.str}|".encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _clean_region(value: object) -> str:
    return " ".join(str(value).split())


def chicken_heart_anatomical_orientation_qc(adata) -> dict[str, Any]:
    """Recompute vertical and left/right anatomical invariants from labels."""

    if "timepoint" not in adata.obs or "region" not in adata.obs:
        raise ChickenHeartContractError(
            "Chicken-heart input requires obs['timepoint'] and obs['region']."
        )
    if "spatial_aligned" not in adata.obsm:
        raise ChickenHeartContractError(
            "Chicken-heart input requires obsm['spatial_aligned']."
        )
    coordinates = np.asarray(adata.obsm["spatial_aligned"], dtype=np.float64)
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ChickenHeartContractError(
            "Chicken-heart spatial_aligned coordinates must be finite Nx2."
        )
    time_values = adata.obs["timepoint"].astype(str).to_numpy()
    regions = adata.obs["region"].map(_clean_region).to_numpy()
    records: dict[str, Any] = {}
    failures: list[str] = []
    for timepoint in CHICKEN_HEART_TIMEPOINTS:
        mask = time_values == timepoint
        if not mask.any():
            failures.append(f"missing timepoint {timepoint}")
            records[timepoint] = {
                "region_counts": {},
                "region_centroids": {},
                "checks": {},
            }
            continue
        means: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for region in sorted(set(regions[mask])):
            selected = mask & (regions == region)
            means[region] = coordinates[selected].mean(axis=0).astype(float).tolist()
            counts[region] = int(selected.sum())

        checks: dict[str, Any] = {}
        if timepoint == "D4":
            required = ("Atria", "Valves", "Ventricle")
            missing = [region for region in required if region not in means]
            if missing:
                failures.append(f"D4 missing anatomical regions {missing}")
            else:
                atria_valves = means["Atria"][1] - means["Valves"][1]
                valves_ventricle = means["Valves"][1] - means["Ventricle"][1]
                checks["atria_above_valves_dy"] = atria_valves
                checks["valves_above_ventricle_dy"] = valves_ventricle
                if atria_valves <= 0:
                    failures.append("D4 Atria are not above Valves")
                if valves_ventricle <= 0:
                    failures.append("D4 Valves are not above Ventricle")
        else:
            right = "Right ventricle"
            left = "Compact LV and inter-ventricular septum"
            required = ("Atria", "Valves", right, left)
            missing = [region for region in required if region not in means]
            if missing:
                failures.append(f"{timepoint} missing anatomical regions {missing}")
            else:
                atria_valves = means["Atria"][1] - means["Valves"][1]
                right_left = means[right][0] - means[left][0]
                checks["atria_above_valves_dy"] = atria_valves
                checks["right_ventricle_right_of_compact_lv_dx"] = right_left
                if atria_valves <= 0:
                    failures.append(f"{timepoint} Atria are not above Valves")
                if right_left <= 0:
                    failures.append(
                        f"{timepoint} Right ventricle is horizontally mirrored "
                        "relative to Compact LV"
                    )
        records[timepoint] = {
            "region_counts": counts,
            "region_centroids": means,
            "checks": checks,
        }
    return {
        "coordinate_convention": "x-right_y-up",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "timepoints": records,
    }


def apply_chicken_heart_coordinate_contract(
    adata, *, repair_legacy_d7_left_right: bool
) -> dict[str, Any]:
    """Fail closed or apply only the audited legacy D7 x reflection."""

    before_coordinates = np.asarray(
        adata.obsm["spatial_aligned"], dtype=np.float64
    ).copy()
    before_qc = chicken_heart_anatomical_orientation_qc(adata)
    if before_qc["status"] == "pass":
        coordinate_hash = _coordinate_sha256(before_coordinates)
        return {
            "applied": False,
            "policy": "reviewed_reference_passed_anatomical_orientation_contract",
            "before_coordinate_sha256": coordinate_hash,
            "after_coordinate_sha256": coordinate_hash,
            "before_anatomical_qc": before_qc,
            "after_anatomical_qc": before_qc,
        }

    expected_failure = [
        "D7 Right ventricle is horizontally mirrored relative to Compact LV"
    ]
    if not repair_legacy_d7_left_right or before_qc["failures"] != expected_failure:
        raise ChickenHeartContractError(
            "Reviewed chicken-heart coordinates fail the anatomical orientation "
            f"contract: {before_qc['failures']}. Use a corrected reviewed reference; "
            "the explicit D7 compatibility repair is allowed only for the single "
            "known legacy D7 left/right mirror."
        )

    mask = adata.obs["timepoint"].astype(str).to_numpy() == "D7"
    d7_before = before_coordinates[mask].copy()
    center_x = float(d7_before[:, 0].mean())
    repaired = before_coordinates.copy()
    repaired[mask, 0] = (2.0 * center_x) - repaired[mask, 0]
    adata.obsm["spatial_aligned"] = repaired
    adata.obsm["spatial"] = repaired.copy()
    after_qc = chicken_heart_anatomical_orientation_qc(adata)
    if after_qc["status"] != "pass":
        raise ChickenHeartContractError(
            f"Explicit D7 reflection did not restore orientation: {after_qc['failures']}."
        )
    distance_error = float(np.max(np.abs(pdist(d7_before) - pdist(repaired[mask]))))
    if distance_error > 1e-12:
        raise ChickenHeartContractError(
            "D7 reflection changed within-stage pairwise distances by "
            f"{distance_error:.3g}."
        )
    if not np.array_equal(repaired[~mask], before_coordinates[~mask]):
        raise ChickenHeartContractError("D7 repair changed coordinates outside D7.")
    return {
        "applied": True,
        "policy": "explicit_legacy_d7_horizontal_reflection",
        "timepoint": "D7",
        "operation": "x_prime=2*stage_mean_x-x",
        "center_x": center_x,
        "linear_matrix": [[-1.0, 0.0], [0.0, 1.0]],
        "translation": [2.0 * center_x, 0.0],
        "pairwise_distance_max_abs_error": distance_error,
        "before_coordinate_sha256": _coordinate_sha256(before_coordinates),
        "after_coordinate_sha256": _coordinate_sha256(repaired),
        "before_anatomical_qc": before_qc,
        "after_anatomical_qc": after_qc,
    }


def apply_chicken_heart_coordinate_validation(
    adata, *, repair_legacy_d7_left_right: bool = False
) -> dict[str, Any]:
    """Validate the reference coordinates and optionally fix the legacy D7 mirror."""

    record = apply_chicken_heart_coordinate_contract(
        adata,
        repair_legacy_d7_left_right=repair_legacy_d7_left_right,
    )
    if not record["applied"]:
        return {
            "applied": False,
            "policy": "reference_coordinates_validated",
            "before_anatomical_qc": record["before_anatomical_qc"],
            "after_anatomical_qc": record["after_anatomical_qc"],
        }
    return {
        "applied": True,
        "policy": "legacy_d7_horizontal_reflection",
        "timepoint": record["timepoint"],
        "operation": record["operation"],
        "center_x": record["center_x"],
        "linear_matrix": record["linear_matrix"],
        "translation": record["translation"],
        "pairwise_distance_max_abs_error": record[
            "pairwise_distance_max_abs_error"
        ],
        "before_anatomical_qc": record["before_anatomical_qc"],
        "after_anatomical_qc": record["after_anatomical_qc"],
    }


def _remove_legacy_chicken_heart_validation_metadata(adata) -> None:
    adata.uns.pop("chicken_heart_input_contract_json", None)


def validate_prepared_chicken_heart_input(adata) -> dict[str, Any]:
    """Validate a prepared H5AD before workflow graph fitting or training."""

    raw_validation = adata.uns.get("chicken_heart_input_validation_json")
    legacy_metadata = False
    if isinstance(raw_validation, str):
        try:
            metadata = __import__("json").loads(raw_validation)
        except (TypeError, ValueError) as exc:
            raise ChickenHeartContractError(
                "Prepared chicken-heart input validation is not valid JSON."
            ) from exc
        schema_version = metadata.get("schema_version")
        if schema_version != 4:
            raise ChickenHeartContractError(
                "Prepared chicken-heart input validation requires schema_version 4."
            )
        coordinate_record = metadata.get("coordinate_adjustment")
        if not isinstance(coordinate_record, dict):
            raise ChickenHeartContractError(
                "Prepared chicken-heart input lacks coordinate adjustment metadata."
            )
    else:
        legacy_metadata = True
        raw_contract = adata.uns.get("chicken_heart_input_contract_json")
        if not isinstance(raw_contract, str):
            raise ChickenHeartContractError(
                "Prepared chicken-heart H5AD lacks input validation metadata."
            )
        try:
            metadata = __import__("json").loads(raw_contract)
        except (TypeError, ValueError) as exc:
            raise ChickenHeartContractError(
                "Prepared chicken-heart input metadata is not valid JSON."
            ) from exc
        schema_version = metadata.get("schema_version")
        if schema_version not in {2, 3}:
            raise ChickenHeartContractError(
                "Prepared chicken-heart legacy metadata requires schema_version 2 or 3."
            )
        coordinate_record = metadata.get("coordinate_repair")
        if not isinstance(coordinate_record, dict):
            raise ChickenHeartContractError(
                "Prepared chicken-heart input lacks coordinate metadata."
            )
        observed_digest = _coordinate_sha256(adata.obsm["spatial_aligned"])
        if coordinate_record.get("after_coordinate_sha256") != observed_digest:
            raise ChickenHeartContractError(
                "Prepared chicken-heart spatial coordinates do not match legacy metadata."
            )
    anatomical = chicken_heart_anatomical_orientation_qc(adata)
    if anatomical["status"] != "pass":
        raise ChickenHeartContractError(
            f"Prepared chicken-heart anatomy fails orientation: {anatomical['failures']}."
        )
    required_obsm = {"spatial_aligned": 2, "X_latent": 50}
    for key, width in required_obsm.items():
        if key not in adata.obsm or np.asarray(adata.obsm[key]).shape != (
            adata.n_obs,
            width,
        ):
            raise ChickenHeartContractError(
                f"Prepared chicken-heart obsm[{key!r}] must be ({adata.n_obs}, {width})."
            )
    if "counts" not in adata.layers:
        raise ChickenHeartContractError(
            "Prepared chicken-heart input lacks the raw counts layer."
        )
    for key in ("region", "celltype_prediction"):
        if key not in adata.obs:
            raise ChickenHeartContractError(
                f"Prepared chicken-heart input lacks obs[{key!r}]."
            )
        values = adata.obs[key]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ChickenHeartContractError(
                f"Prepared chicken-heart obs[{key!r}] contains missing labels."
            )
    if (
        schema_version in {3, 4}
        and "Annotation" in adata.obs
        and not np.array_equal(
            adata.obs["Annotation"].astype(str).to_numpy(),
            adata.obs["celltype_prediction"].astype(str).to_numpy(),
        )
    ):
        raise ChickenHeartContractError(
            "Prepared chicken-heart obs['Annotation'] must match "
            "obs['celltype_prediction']; obs['region'] is reserved for "
            "anatomical orientation QC."
        )
    labels = adata.obs["celltype_prediction"].astype(str).tolist()
    if schema_version in {3, 4}:
        annotation_metadata = metadata.get("downstream_annotation")
        if (
            not isinstance(annotation_metadata, dict)
            or annotation_metadata.get("key") != "celltype_prediction"
        ):
            raise ChickenHeartContractError(
                "Prepared chicken-heart input lacks cell-type annotation metadata."
            )
        if schema_version == 3:
            label_digest = hashlib.sha256()
            for value in labels:
                encoded = value.encode("utf-8")
                label_digest.update(len(encoded).to_bytes(8, "little"))
                label_digest.update(encoded)
            if annotation_metadata.get("ordered_label_sha256") != label_digest.hexdigest():
                raise ChickenHeartContractError(
                    "Prepared chicken-heart cell-type labels do not match legacy metadata."
                )
        if annotation_metadata.get("n_classes") != len(set(labels)):
            raise ChickenHeartContractError(
                "Prepared chicken-heart cell-type class count does not match metadata."
            )
    if schema_version == 4:
        reference = metadata.get("reference", {})
        time_counts = (
            adata.obs["timepoint"].astype(str).value_counts(sort=False).to_dict()
        )
        if reference.get("n_obs") != adata.n_obs or reference.get(
            "timepoint_counts"
        ) != time_counts:
            raise ChickenHeartContractError(
                "Prepared chicken-heart spot counts do not match reference metadata."
            )
    observed_times = sorted(
        np.unique(adata.obs["time_point_processed"].to_numpy(dtype=np.float64)).tolist()
    )
    if observed_times != [0.0, 1.0, 2.0, 3.0]:
        raise ChickenHeartContractError(
            f"Prepared chicken-heart times are {observed_times}, expected [0,1,2,3]."
        )
    return {
        "schema_version": int(schema_version),
        "coordinate_policy": coordinate_record.get("policy"),
        "downstream_annotation_key": "celltype_prediction",
        "legacy_annotation_alias_ignored": bool(
            legacy_metadata
            and schema_version == 2
            and "Annotation" in adata.obs
            and not np.array_equal(
                adata.obs["Annotation"].astype(str).to_numpy(),
                adata.obs["celltype_prediction"].astype(str).to_numpy(),
            )
        ),
        "anatomical_orientation_qc": anatomical,
    }


__all__ = [
    "CHICKEN_HEART_TIMEPOINTS",
    "ChickenHeartContractError",
    "apply_chicken_heart_coordinate_contract",
    "apply_chicken_heart_coordinate_validation",
    "chicken_heart_anatomical_orientation_qc",
    "validate_prepared_chicken_heart_input",
]
