"""Post-hoc LR annotation of one-layer state-space GNN interaction drift.

The functions in this module keep two quantities deliberately separate:

``D_AB``
    The mean norm of the *exact* GNN message contribution from sender cell
    type ``A`` to receiver cell type ``B``.  Sender labels are used only after
    the model has been trained.  For every receiver cell, contributions from
    all sender types sum to the model's interaction output up to floating
    point error.

``Q_AB,p``
    The observed, pathway-balanced ligand--receptor (LR) compatibility for
    sender type ``A``, receiver type ``B`` and LR row ``p``.  It reuses the
    normalization, strict-complex minimum, activity scales and row weights
    frozen in an LR edge-prior manifest.

Their product ``S_AB,p = D_AB * Q_AB,p`` is an LR-annotated drift score.  It is
not a transition probability, a causal effect, or a decomposition of the GNN
output by LR pathway.  The GNN sees PCA states and the frozen edge predictor;
cell-type labels and LR names enter only this post-hoc analysis.

Exact message decomposition is intentionally limited to the released
one-layer ``GNNInteraction`` state-space attention head.  Multi-layer message
passing mixes sender identities at intermediate nodes and is rejected rather
than approximated with attention magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch

from CytoBridge.pp.lr_edge_prior import (
    _complex_activity,
    _normalized_lr_expression,
    complex_subunits,
    load_lr_database,
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(json.dumps(list(values.shape)).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _resolve_manifest_artifact(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get(name), Mapping
    ):
        raise ValueError(f"Edge-prior manifest has no artifacts[{name!r}].")
    entry = artifacts[name]
    declared_text = str(entry.get("path", "")).strip()
    declared = Path(declared_text).expanduser() if declared_text else Path(name)
    # A relocated review bundle must prefer paths next to its manifest.  Do
    # not stop at the first existing candidate: a same-named file in the
    # caller's CWD is not the frozen artifact unless its digest also matches.
    candidates = (
        [declared]
        if declared.is_absolute()
        else [
            manifest_path.parent / declared,
            manifest_path.parent / declared.name,
            declared,
        ]
    )
    candidates.append(manifest_path.parent / name)
    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate.expanduser())
    existing = [path.resolve() for path in unique_candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"Could not resolve {name!r} from {manifest_path}; tried "
            + ", ".join(str(path) for path in unique_candidates)
        )
    expected = str(entry.get("sha256", "")).lower()
    if len(expected) != 64:
        raise ValueError(f"Edge-prior artifact {name!r} has no valid SHA256 digest.")
    resolved = next((path for path in existing if sha256_file(path) == expected), None)
    if resolved is None:
        observed = {str(path): sha256_file(path) for path in existing}
        raise ValueError(
            f"SHA256 mismatch for every resolved candidate of edge-prior artifact "
            f"{name!r}: manifest={expected}, observed={observed}."
        )
    return resolved


def _resolve_manifest_input_file(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    name: str,
) -> Path:
    """Resolve one immutable input by digest, including relocated bundles.

    Unlike an output artifact, an input is not necessarily copied next to the
    manifest.  We nevertheless accept a same-named colocated copy so a review
    bundle remains portable.  A path is never accepted on its name alone: the
    frozen SHA256 must match.
    """

    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping) or not isinstance(inputs.get(name), Mapping):
        raise ValueError(f"Edge-prior manifest has no inputs[{name!r}].")
    entry = inputs[name]
    declared_text = str(entry.get("path", "")).strip()
    if not declared_text:
        raise ValueError(f"Edge-prior input {name!r} has no declared path.")
    declared = Path(declared_text).expanduser()
    candidates = (
        [declared]
        if declared.is_absolute()
        else [manifest_path.parent / declared, declared]
    )
    candidates.extend(
        [
            manifest_path.parent / declared.name,
            manifest_path.parent / "inputs" / declared.name,
        ]
    )
    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.expanduser().resolve())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate.expanduser())
    existing = [path.resolve() for path in unique_candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"Could not resolve frozen input {name!r} from {manifest_path}; tried "
            + ", ".join(str(path) for path in unique_candidates)
        )
    expected = str(entry.get("sha256", "")).lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(f"Edge-prior input {name!r} has no valid SHA256 digest.")
    resolved = next((path for path in existing if sha256_file(path) == expected), None)
    if resolved is None:
        observed = {str(path): sha256_file(path) for path in existing}
        raise ValueError(
            f"SHA256 mismatch for every resolved candidate of edge-prior input "
            f"{name!r}: manifest={expected}, observed={observed}."
        )
    return resolved


_NORMALIZATION_FIELDS = {
    "input_semantics_required",
    "normalization",
    "target_sum",
    "normalize_total_applications",
    "log1p_applications",
    "library_sum_before",
    "n_requested_lr_genes",
    "n_present_lr_genes",
    "present_genes",
}


def _strict_manifest_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"Edge-prior manifest field {field!r} must be an integer >= {minimum}."
        )
    return int(value)


def _validate_expression_metadata_contract(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Validate the complete schema-v1 LR-expression semantics.

    This is deliberately stricter than checking only ``target_sum``.  Every
    normalization field produced by the builder is either validated here or
    recomputed exactly later; an unknown schema must receive an explicit
    implementation rather than silently inheriting schema-v1 assumptions.
    """

    if manifest.get("schema_version") != 1:
        raise ValueError(
            "LR attribution supports only edge-prior manifest schema_version=1."
        )
    configuration = manifest.get("configuration")
    filtering = manifest.get("database_filtering")
    normalization = manifest.get("normalization")
    data_usage = manifest.get("data_usage")
    if not all(
        isinstance(value, Mapping)
        for value in (configuration, filtering, normalization, data_usage)
    ):
        raise ValueError(
            "Edge-prior manifest lacks configuration/filtering/normalization/data-usage provenance."
        )
    if set(normalization) != _NORMALIZATION_FIELDS:
        missing = sorted(_NORMALIZATION_FIELDS.difference(normalization))
        unexpected = sorted(set(normalization).difference(_NORMALIZATION_FIELDS))
        raise ValueError(
            "Unsupported edge-prior normalization metadata schema: "
            f"missing={missing}, unexpected={unexpected}."
        )
    if normalization.get("input_semantics_required") != (
        "non-negative linear expression"
    ):
        raise ValueError("Unsupported edge-prior input-expression semantics.")
    if normalization.get("normalization") != "normalize_total_then_log1p":
        raise ValueError("Unsupported edge-prior expression normalization.")
    if (
        _strict_manifest_int(
            normalization.get("normalize_total_applications"),
            field="normalization.normalize_total_applications",
            minimum=1,
        )
        != 1
        or _strict_manifest_int(
            normalization.get("log1p_applications"),
            field="normalization.log1p_applications",
            minimum=1,
        )
        != 1
    ):
        raise ValueError("LR attribution requires exactly one normalize/log1p pass.")
    if isinstance(normalization.get("target_sum"), bool) or isinstance(
        configuration.get("target_sum"), bool
    ):
        raise ValueError("Edge-prior target_sum provenance is invalid.")
    try:
        target_sum = float(normalization["target_sum"])
        configured_target_sum = float(configuration["target_sum"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Edge-prior target_sum provenance is invalid.") from exc
    if (
        not math.isfinite(target_sum)
        or target_sum <= 0.0
        or target_sum != configured_target_sum
    ):
        raise ValueError(
            "Frozen normalization target_sum differs from the builder configuration."
        )
    requested_count = _strict_manifest_int(
        normalization.get("n_requested_lr_genes"),
        field="normalization.n_requested_lr_genes",
        minimum=1,
    )
    present_count = _strict_manifest_int(
        normalization.get("n_present_lr_genes"),
        field="normalization.n_present_lr_genes",
        minimum=1,
    )
    present_genes = normalization.get("present_genes")
    if (
        not isinstance(present_genes, list)
        or any(type(gene) is not str or not gene for gene in present_genes)
        or present_genes != sorted(set(present_genes))
        or len(present_genes) != present_count
        or present_count > requested_count
    ):
        raise ValueError(
            "Frozen present_genes must be a sorted unique list consistent with its counts."
        )
    library_sum = normalization.get("library_sum_before")
    if not isinstance(library_sum, Mapping) or set(library_sum) != {
        "min",
        "median",
        "max",
    }:
        raise ValueError("Frozen pre-normalization library-size summary is invalid.")
    try:
        if any(
            isinstance(library_sum[field], bool) for field in ("min", "median", "max")
        ):
            raise TypeError
        library_values = [
            float(library_sum[field]) for field in ("min", "median", "max")
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError("Frozen pre-normalization library sizes are invalid.") from exc
    if not all(
        math.isfinite(value) and value > 0.0 for value in library_values
    ) or library_values != sorted(library_values):
        raise ValueError("Frozen pre-normalization library sizes are invalid.")

    gene_symbol_key = configuration.get("gene_symbol_key")
    if type(gene_symbol_key) is not str or not gene_symbol_key:
        raise ValueError("Edge-prior gene_symbol_key provenance is invalid.")
    if data_usage.get("gene_symbol_source") != f"var[{gene_symbol_key!r}]":
        raise ValueError("Frozen gene-symbol source disagrees with gene_symbol_key.")
    if (
        filtering.get("strict_all_subunits") is not True
        or filtering.get("complex_delimiter") != "_"
        or filtering.get("complex_activity") != "minimum"
    ):
        raise ValueError(
            "LR attribution requires underscore-delimited strict all-subunit minimum complexes."
        )
    min_cells = _strict_manifest_int(
        filtering.get("min_cells_global_per_subunit"),
        field="database_filtering.min_cells_global_per_subunit",
        minimum=1,
    )
    if min_cells != _strict_manifest_int(
        configuration.get("min_cells"),
        field="configuration.min_cells",
        minimum=1,
    ):
        raise ValueError(
            "Frozen subunit min_cells differs from the builder configuration."
        )
    expected_filter_semantics = {
        "min_cells_scope": "all cells before cell split",
        "min_cells_filter_unsupervised": True,
        "min_cells_filter_uses_only_nonzero_expression_detection": True,
        "min_cells_filter_uses_outcomes_or_annotations": False,
    }
    for field, expected in expected_filter_semantics.items():
        if filtering.get(field) != expected:
            raise ValueError(
                f"Unsupported or ambiguous frozen filtering field {field!r}."
            )

    row_fields = (
        "rows_before_annotation_filter",
        "rows_after_annotation_filter",
        "rows_after_global_subunit_filter",
        "rows_final",
    )
    row_counts = {
        field: _strict_manifest_int(
            filtering.get(field), field=f"database_filtering.{field}"
        )
        for field in row_fields
    }
    reasons = filtering.get("filter_reason_counts")
    expected_reason_names = {
        "missing_subunit",
        "subunit_below_min_cells",
        "no_positive_train_complex_activity",
    }
    if not isinstance(reasons, Mapping) or set(reasons) != expected_reason_names:
        raise ValueError(
            "Frozen LR filter-reason counts are incomplete or unsupported."
        )
    reason_counts = {
        field: _strict_manifest_int(
            reasons.get(field), field=f"database_filtering.filter_reason_counts.{field}"
        )
        for field in expected_reason_names
    }
    if (
        row_counts["rows_after_annotation_filter"] + 0
        != reason_counts["missing_subunit"]
        + reason_counts["subunit_below_min_cells"]
        + row_counts["rows_after_global_subunit_filter"]
        or row_counts["rows_after_global_subunit_filter"]
        != reason_counts["no_positive_train_complex_activity"]
        + row_counts["rows_final"]
        or row_counts["rows_before_annotation_filter"]
        < row_counts["rows_after_annotation_filter"]
        or row_counts["rows_final"] < 1
    ):
        raise ValueError(
            "Frozen LR row/filter-reason counts are internally inconsistent."
        )
    return configuration, filtering, normalization


def _load_frozen_filtered_database(
    manifest: Mapping[str, Any], manifest_path: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Replay database loading and exact annotation filtering from the builder."""

    configuration, filtering, normalization = _validate_expression_metadata_contract(
        manifest
    )
    database_path = _resolve_manifest_input_file(manifest_path, manifest, "lr_database")
    database, observed_metadata = load_lr_database(database_path)
    frozen_metadata = manifest["inputs"]["lr_database"]
    for field in (
        "sha256",
        "columns",
        "rows_input",
        "rows_invalid",
        "rows_exact_duplicates",
        "rows_standardized",
    ):
        if observed_metadata.get(field) != frozen_metadata.get(field):
            raise ValueError(
                f"Re-loaded LR database disagrees with frozen field {field!r}."
            )
    expected_declared = {
        "source": configuration.get("database_source"),
        "version": configuration.get("database_version"),
        "commit": configuration.get("database_commit"),
        "caller_supplied": True,
    }
    if frozen_metadata.get("declared_provenance") != expected_declared:
        raise ValueError("Frozen LR-database declared provenance is inconsistent.")
    annotation_values = filtering.get("annotation_filter")
    if (
        filtering.get("annotation_match") != "exact"
        or annotation_values != list(configuration.get("annotation_filter", []))
        or annotation_values != ["Secreted Signaling"]
    ):
        raise ValueError(
            "Weinreb LR attribution requires exact Secreted Signaling rows."
        )
    if len(database) != int(filtering["rows_before_annotation_filter"]):
        raise ValueError(
            "Re-loaded LR database row count differs before annotation filtering."
        )
    database = database.loc[
        database["annotation"].isin(tuple(annotation_values))
    ].reset_index(drop=True)
    if len(database) != int(filtering["rows_after_annotation_filter"]):
        raise ValueError(
            "Re-loaded LR database row count differs after annotation filtering."
        )
    requested_genes = sorted(
        {
            subunit
            for token in pd.concat((database["ligand"], database["receptor"]))
            for subunit in complex_subunits(str(token))
        }
    )
    if len(requested_genes) != int(normalization["n_requested_lr_genes"]):
        raise ValueError(
            "Frozen n_requested_lr_genes disagrees with the exact filtered LR database."
        )
    return database, requested_genes


def load_edge_prior_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and validate the semantic fields needed for LR attribution."""

    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("method") != (
        "nonspatial_cellchatdb_expression_guided_directed_lr_edge_prior"
    ):
        raise ValueError(
            f"Unsupported LR edge-prior method: {manifest.get('method')!r}."
        )
    if bool(manifest.get("spatial_method_claimed", True)):
        raise ValueError("LR attribution requires an explicitly non-spatial prior.")
    configuration, filtering, _ = _validate_expression_metadata_contract(manifest)
    data_usage = manifest.get("data_usage", {})
    forbidden_usage = (
        "uses_clone",
        "uses_fate_or_cell_type",
        "uses_starting_population",
        "uses_spatial_or_visualization_coordinates",
    )
    for field in forbidden_usage:
        if data_usage.get(field) is not False:
            raise ValueError(
                f"Edge-prior no-leakage provenance is absent or false for {field!r}."
            )
    if data_usage.get("obs_keys_used") != [configuration.get("time_key")]:
        raise ValueError("Edge prior used obs columns other than its frozen time key.")
    _load_frozen_filtered_database(manifest, manifest_path)
    pair_sampling = manifest.get("pair_sampling", {})
    if (
        pair_sampling.get("directed") is not True
        or pair_sampling.get("self_edges") is not False
        or pair_sampling.get("candidate_rule")
        != "distance < radius and distance > 1e-6"
    ):
        raise ValueError(
            "Edge-prior directed candidate-pair semantics are unsupported."
        )
    predictor = manifest.get("predictor", {})
    predictor_input_dim = int(predictor.get("input_dim", -1))
    latent_dim = predictor_input_dim // 2
    expected_input_order = (
        f"concatenate(sender_{latent_dim}d_pca, receiver_{latent_dim}d_pca)"
    )
    if (
        predictor.get("architecture") != "LinkPredictorMLP"
        or predictor.get("state_dict_compatible_with_gnn_interaction") is not True
        or predictor_input_dim < 2
        or predictor_input_dim % 2 != 0
        or predictor.get("input_order") != expected_input_order
    ):
        raise ValueError(
            "Edge predictor architecture or sender/receiver order is unsupported."
        )
    _resolve_manifest_artifact(manifest_path, manifest, "lr_pair_metadata.csv")
    _resolve_manifest_artifact(manifest_path, manifest, "link_predictor.pt")
    split_path = _resolve_manifest_artifact(manifest_path, manifest, "cell_splits.npz")
    pair_path = _resolve_manifest_artifact(manifest_path, manifest, "pair_samples.npz")
    _validate_cell_disjoint_pair_artifacts(
        manifest,
        split_path=split_path,
        pair_path=pair_path,
    )
    return manifest, manifest_path


def _validate_cell_disjoint_pair_artifacts(
    manifest: Mapping[str, Any],
    *,
    split_path: Path,
    pair_path: Path,
) -> None:
    """Verify that every sampled pair stays inside one cell-disjoint split."""

    try:
        n_obs = int(manifest["inputs"]["latent_h5ad"]["shape"][0])
        split_manifest = manifest["cell_splits"]
        split_code_map = manifest["pair_sampling"]["split_code_map"]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("Edge-prior split provenance is incomplete.") from exc
    if n_obs < 2 or split_code_map != {"train": 0, "validation": 1, "test": 2}:
        raise ValueError("Edge-prior split dimensions/codes are unsupported.")
    time_codes = sorted(int(key) for key in split_manifest)
    if time_codes != list(range(len(time_codes))):
        raise ValueError("Edge-prior time split codes must be contiguous from zero.")

    membership: dict[tuple[int, int], np.ndarray] = {}
    all_cells: list[np.ndarray] = []
    with np.load(split_path, allow_pickle=False) as split_arrays:
        expected_names = {
            f"time_{time_code}_{split_name}"
            for time_code in time_codes
            for split_name in split_code_map
        }
        if set(split_arrays.files) != expected_names:
            raise ValueError(
                "cell_splits.npz keys disagree with the frozen split design."
            )
        for time_code in time_codes:
            time_entry = split_manifest[str(time_code)]
            for split_name, split_code in split_code_map.items():
                key = f"time_{time_code}_{split_name}"
                indices = np.asarray(split_arrays[key], dtype=np.int64)
                if (
                    indices.ndim != 1
                    or len(np.unique(indices)) != len(indices)
                    or np.any((indices < 0) | (indices >= n_obs))
                ):
                    raise ValueError(f"Invalid or duplicate cell indices in {key}.")
                declared = time_entry["splits"][split_name]
                if int(declared["n_cells"]) != len(indices) or str(
                    declared["global_indices_sha256"]
                ) != _sha256_array(indices):
                    raise ValueError(f"Split counts/digest disagree for {key}.")
                mask = np.zeros(n_obs, dtype=bool)
                mask[indices] = True
                membership[(time_code, int(split_code))] = mask
                all_cells.append(indices)
    concatenated = np.concatenate(all_cells)
    if len(concatenated) != n_obs or not np.array_equal(
        np.sort(concatenated), np.arange(n_obs, dtype=np.int64)
    ):
        raise ValueError(
            "Cell splits are not a disjoint exhaustive partition of cells."
        )

    with np.load(pair_path, allow_pickle=False) as pair_arrays:
        required = {
            "source",
            "target",
            "time_code",
            "split_code",
            "distance",
            "score",
            "label",
        }
        if not required.issubset(pair_arrays.files):
            raise ValueError("pair_samples.npz lacks required directed-pair arrays.")
        source_raw = np.asarray(pair_arrays["source"])
        target_raw = np.asarray(pair_arrays["target"])
        distance_raw = np.asarray(pair_arrays["distance"])
        score_raw = np.asarray(pair_arrays["score"])
        label_raw = np.asarray(pair_arrays["label"])
        source = source_raw.astype(np.int64, copy=False)
        target = target_raw.astype(np.int64, copy=False)
        time_code = np.asarray(pair_arrays["time_code"], dtype=np.int64)
        split_code = np.asarray(pair_arrays["split_code"], dtype=np.int64)
        distance = distance_raw.astype(np.float64, copy=False)
        score = score_raw.astype(np.float64, copy=False)
        label = label_raw.astype(np.int8, copy=False)
    n_pairs = len(source)
    if not all(
        len(values) == n_pairs
        for values in (target, time_code, split_code, distance, score, label)
    ):
        raise ValueError("Directed pair-sample arrays have inconsistent lengths.")
    radius = float(manifest["pair_sampling"]["candidate_radius"])
    if (
        np.any(source == target)
        or np.any((source < 0) | (source >= n_obs))
        or np.any((target < 0) | (target >= n_obs))
        or not np.isfinite(distance).all()
        or np.any((distance <= 1.0e-6) | (distance >= radius))
        or not np.isfinite(score).all()
        or np.any((score < 0.0) | (score > 1.0))
        or not np.isin(label, (0, 1)).all()
    ):
        raise ValueError(
            "Pair samples violate directed nonself candidate-radius semantics."
        )
    observed_strata = set(zip(time_code.tolist(), split_code.tolist()))
    if observed_strata != set(membership):
        raise ValueError(
            "Pair samples do not cover exactly the frozen time/split strata."
        )
    for stratum, allowed in membership.items():
        selected = (time_code == stratum[0]) & (split_code == stratum[1])
        if not allowed[source[selected]].all() or not allowed[target[selected]].all():
            raise ValueError(
                f"Pair endpoints cross cell-disjoint split boundaries for stratum {stratum}."
            )
        pair_ids = source[selected] * np.int64(n_obs) + target[selected]
        if len(np.unique(pair_ids)) != len(pair_ids):
            raise ValueError(
                f"Duplicate directed pair samples occur in stratum {stratum}."
            )

    threshold_records = manifest["lr_score_thresholds"]
    if sorted(int(code) for code in threshold_records) != time_codes:
        raise ValueError("LR score-threshold time codes disagree with cell splits.")
    threshold_by_time = np.empty(len(time_codes), dtype=np.float32)
    for code, record in threshold_records.items():
        threshold_by_time[int(code)] = np.float32(record["threshold"])
    expected_label = (score_raw > threshold_by_time[time_code]).astype(np.int8)
    if not np.array_equal(label, expected_label):
        raise ValueError(
            "Pair labels do not match frozen strict time-wise LR thresholds."
        )
    split_records = manifest["pair_sampling"]["splits"]
    for split_name, code in split_code_map.items():
        selected = split_code == int(code)
        record = split_records[split_name]
        expected_digests = {
            "source_indices_sha256": _sha256_array(source_raw[selected]),
            "target_indices_sha256": _sha256_array(target_raw[selected]),
            "distances_sha256": _sha256_array(distance_raw[selected]),
            "scores_sha256": _sha256_array(score_raw[selected]),
            "labels_sha256": _sha256_array(label_raw[selected]),
        }
        if int(record["n_pairs"]) != int(selected.sum()) or any(
            str(record[field]) != digest for field, digest in expected_digests.items()
        ):
            raise ValueError(f"Pair-split counts/digests disagree for {split_name!r}.")


def scaled_lr_activities_from_metadata(
    source: ad.AnnData,
    lr_metadata: pd.DataFrame,
    *,
    gene_symbol_key: str,
    target_sum: float,
    normalization_genes: Sequence[str] | None = None,
    training_indices: Sequence[int] | None = None,
    expected_activity_scale_quantile: float | None = None,
    filtered_database: pd.DataFrame | None = None,
    expected_min_cells: int | None = None,
    expected_filter_reason_counts: Mapping[str, Any] | None = None,
    expected_rows_after_global_filter: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct the frozen scaled ligand and receptor activity matrices.

    ``lr_metadata`` must be the immutable ``lr_pair_metadata.csv`` produced by
    :func:`CytoBridge.pp.lr_edge_prior.build_lr_edge_prior`.  Complexes are
    recomputed as the minimum across every underscore-delimited subunit, then
    divided by the training-positive scales recorded for that exact LR row.
    """

    required = {
        "database_row",
        "ligand",
        "receptor",
        "pathway",
        "annotation",
        "ligand_train_positive_activity_scale",
        "receptor_train_positive_activity_scale",
        "pair_weight",
        "pathway_pair_count",
        "activity_scale_quantile",
        "ligand_subunits",
        "receptor_subunits",
        "min_subunit_detected_cells",
        "ligand_positive_train_cells",
        "receptor_positive_train_cells",
    }
    missing = sorted(required.difference(lr_metadata.columns))
    if missing:
        raise ValueError(f"LR metadata is missing columns: {missing}.")
    if lr_metadata.empty:
        raise ValueError("LR metadata contains no rows.")
    if lr_metadata[["ligand", "receptor", "pathway", "annotation"]].isna().any().any():
        raise ValueError(
            "LR metadata contains missing ligand/receptor/pathway annotations."
        )
    database_rows_numeric = pd.to_numeric(
        lr_metadata["database_row"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(database_rows_numeric).all()
        or not np.equal(database_rows_numeric, np.floor(database_rows_numeric)).all()
    ):
        raise ValueError("Frozen database_row values must be finite integers.")
    database_row_ids = database_rows_numeric.astype(np.int64)
    if len(np.unique(database_row_ids)) != len(database_row_ids) or (
        len(database_row_ids) > 1 and np.any(np.diff(database_row_ids) <= 0)
    ):
        raise ValueError(
            "Frozen database_row values must be unique and preserve database order."
        )
    identity_columns = ["ligand", "receptor", "pathway", "annotation"]
    if lr_metadata.duplicated(identity_columns, keep=False).any():
        raise ValueError("Frozen LR metadata contains exact duplicate database rows.")

    pathway_counts = lr_metadata["pathway"].astype(str).value_counts().to_dict()
    n_pathways = len(pathway_counts)
    expected_counts = lr_metadata["pathway"].astype(str).map(pathway_counts).to_numpy()
    recorded_counts = lr_metadata["pathway_pair_count"].to_numpy(dtype=np.int64)
    if not np.array_equal(recorded_counts, expected_counts):
        raise ValueError(
            "Frozen pathway_pair_count values disagree with LR metadata rows."
        )
    expected_weights = np.asarray(
        [
            1.0 / (n_pathways * pathway_counts[str(pathway)])
            for pathway in lr_metadata["pathway"]
        ],
        dtype=np.float64,
    )
    weights = lr_metadata["pair_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or not np.allclose(
        weights, expected_weights, rtol=1e-12, atol=1e-15
    ):
        raise ValueError("Frozen pair weights are not exactly pathway balanced.")
    quantiles = lr_metadata["activity_scale_quantile"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(quantiles).all()
        or np.any((quantiles <= 0.0) | (quantiles >= 1.0))
        or not np.allclose(quantiles, quantiles[0], rtol=0.0, atol=1e-15)
    ):
        raise ValueError(
            "Frozen LR activity-scale quantiles are invalid or inconsistent."
        )
    for row in lr_metadata.itertuples(index=False):
        expected_ligand_subunits = "|".join(complex_subunits(str(row.ligand)))
        expected_receptor_subunits = "|".join(complex_subunits(str(row.receptor)))
        if str(row.ligand_subunits) != expected_ligand_subunits:
            raise ValueError(f"Frozen ligand subunits disagree for {row.ligand!r}.")
        if str(row.receptor_subunits) != expected_receptor_subunits:
            raise ValueError(f"Frozen receptor subunits disagree for {row.receptor!r}.")

    tokens = pd.concat((lr_metadata["ligand"], lr_metadata["receptor"]))
    activity_genes = sorted(
        {subunit for token in tokens.astype(str) for subunit in complex_subunits(token)}
    )
    if normalization_genes is None:
        genes = activity_genes
    else:
        genes = list(normalization_genes)
        if any(type(gene) is not str or not gene for gene in genes) or genes != sorted(
            set(genes)
        ):
            raise ValueError("normalization_genes must be a sorted unique gene list.")
        missing_from_universe = sorted(set(activity_genes).difference(genes))
        if missing_from_universe:
            raise ValueError(
                "Frozen normalization gene universe excludes retained LR subunits; "
                "examples: " + ", ".join(missing_from_universe[:10])
            )
    expression, detection, normalization = _normalized_lr_expression(
        source,
        gene_symbol_key=str(gene_symbol_key),
        genes=genes,
        target_sum=float(target_sum),
    )
    gene_to_column = {
        gene: index for index, gene in enumerate(normalization["present_genes"])
    }
    missing_genes = sorted(set(activity_genes).difference(gene_to_column))
    if missing_genes:
        raise ValueError(
            "Expression data no longer contain all frozen LR subunits; examples: "
            + ", ".join(missing_genes[:10])
        )

    cache: dict[str, np.ndarray] = {}
    if training_indices is not None:
        train_cells = np.asarray(training_indices, dtype=np.int64)
        if (
            train_cells.ndim != 1
            or not len(train_cells)
            or len(np.unique(train_cells)) != len(train_cells)
            or np.any((train_cells < 0) | (train_cells >= source.n_obs))
        ):
            raise ValueError(
                "training_indices must be unique valid source-cell indices."
            )
        if expected_activity_scale_quantile is None:
            raise ValueError(
                "expected_activity_scale_quantile is required with training_indices."
            )
        scale_quantile = float(expected_activity_scale_quantile)
        if not 0.0 < scale_quantile < 1.0:
            raise ValueError("expected_activity_scale_quantile must lie in (0, 1).")
    else:
        train_cells = None
        scale_quantile = None

    if filtered_database is not None:
        if train_cells is None or scale_quantile is None:
            raise ValueError(
                "Frozen database filtering can be replayed only with training_indices."
            )
        required_database_columns = {
            "database_row",
            "ligand",
            "receptor",
            "pathway",
            "annotation",
        }
        if set(filtered_database.columns) != required_database_columns:
            raise ValueError(
                "filtered_database does not have the standardized builder schema."
            )
        if (
            expected_min_cells is None
            or isinstance(expected_min_cells, bool)
            or int(expected_min_cells) != expected_min_cells
            or int(expected_min_cells) < 1
        ):
            raise ValueError("expected_min_cells must be a positive integer.")
        min_cells = int(expected_min_cells)
        replay_reasons = {
            "missing_subunit": 0,
            "subunit_below_min_cells": 0,
            "no_positive_train_complex_activity": 0,
        }
        replay_prelim_count = 0
        expected_final_database_rows: list[int] = []
        replay_cache: dict[str, np.ndarray] = {}
        for database_row in filtered_database.itertuples(index=False):
            ligand = str(database_row.ligand)
            receptor = str(database_row.receptor)
            subunits = (*complex_subunits(ligand), *complex_subunits(receptor))
            if any(gene not in gene_to_column for gene in subunits):
                replay_reasons["missing_subunit"] += 1
                continue
            if any(detection[gene] < min_cells for gene in subunits):
                replay_reasons["subunit_below_min_cells"] += 1
                continue
            replay_prelim_count += 1
            if ligand not in replay_cache:
                replay_cache[ligand] = _complex_activity(
                    ligand, expression=expression, gene_to_column=gene_to_column
                )
            if receptor not in replay_cache:
                replay_cache[receptor] = _complex_activity(
                    receptor, expression=expression, gene_to_column=gene_to_column
                )
            ligand_values = replay_cache[ligand][train_cells]
            receptor_values = replay_cache[receptor][train_cells]
            if not np.any(ligand_values > 0) or not np.any(receptor_values > 0):
                replay_reasons["no_positive_train_complex_activity"] += 1
                continue
            expected_final_database_rows.append(int(database_row.database_row))
        if not isinstance(expected_filter_reason_counts, Mapping) or set(
            expected_filter_reason_counts
        ) != set(replay_reasons):
            raise ValueError(
                "Expected frozen filter-reason counts are incomplete or unsupported."
            )
        try:
            frozen_reasons = {
                key: int(value) for key, value in expected_filter_reason_counts.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Expected frozen filter-reason counts are invalid."
            ) from exc
        if frozen_reasons != replay_reasons:
            raise ValueError(
                "Replayed strict subunit/positive-activity filter counts disagree "
                "with the frozen manifest."
            )
        if expected_rows_after_global_filter is None or replay_prelim_count != int(
            expected_rows_after_global_filter
        ):
            raise ValueError(
                "Replayed rows_after_global_subunit_filter disagrees with the frozen manifest."
            )
        if expected_final_database_rows != database_row_ids.tolist():
            raise ValueError(
                "Final LR metadata rows are not exactly those retained by replaying "
                "the builder's strict all-subunit filters."
            )
        database_by_id = filtered_database.set_index("database_row", drop=False)
        for metadata_row in lr_metadata.itertuples(index=False):
            database_row = database_by_id.loc[int(metadata_row.database_row)]
            for field in identity_columns:
                if str(getattr(metadata_row, field)) != str(database_row[field]):
                    raise ValueError(
                        f"Frozen LR metadata field {field!r} disagrees with "
                        f"database_row={int(metadata_row.database_row)}."
                    )

    ligand_columns: list[np.ndarray] = []
    receptor_columns: list[np.ndarray] = []
    for row in lr_metadata.itertuples(index=False):
        ligand = str(row.ligand)
        receptor = str(row.receptor)
        if ligand not in cache:
            cache[ligand] = _complex_activity(
                ligand, expression=expression, gene_to_column=gene_to_column
            )
        if receptor not in cache:
            cache[receptor] = _complex_activity(
                receptor, expression=expression, gene_to_column=gene_to_column
            )
        subunits = (*complex_subunits(ligand), *complex_subunits(receptor))
        observed_min_detection = min(detection[gene] for gene in subunits)
        if int(row.min_subunit_detected_cells) != int(observed_min_detection):
            raise ValueError(
                f"Frozen global subunit-detection count disagrees for {ligand!r}->{receptor!r}."
            )
        if expected_min_cells is not None and observed_min_detection < int(
            expected_min_cells
        ):
            raise ValueError(
                f"Frozen LR row {ligand!r}->{receptor!r} violates strict per-subunit min_cells."
            )
        ligand_scale = float(row.ligand_train_positive_activity_scale)
        receptor_scale = float(row.receptor_train_positive_activity_scale)
        if not np.isfinite(ligand_scale) or ligand_scale <= 0:
            raise ValueError(f"Invalid frozen ligand scale for {ligand!r}.")
        if not np.isfinite(receptor_scale) or receptor_scale <= 0:
            raise ValueError(f"Invalid frozen receptor scale for {receptor!r}.")
        if train_cells is not None:
            ligand_positive = cache[ligand][train_cells]
            ligand_positive = ligand_positive[ligand_positive > 0]
            receptor_positive = cache[receptor][train_cells]
            receptor_positive = receptor_positive[receptor_positive > 0]
            if int(row.ligand_positive_train_cells) != len(ligand_positive) or int(
                row.receptor_positive_train_cells
            ) != len(receptor_positive):
                raise ValueError(
                    f"Frozen positive training-cell counts disagree for {ligand!r}->{receptor!r}."
                )
            expected_ligand_scale = float(np.quantile(ligand_positive, scale_quantile))
            expected_receptor_scale = float(
                np.quantile(receptor_positive, scale_quantile)
            )
            if not np.isclose(
                ligand_scale, expected_ligand_scale, rtol=1e-12, atol=1e-12
            ) or not np.isclose(
                receptor_scale, expected_receptor_scale, rtol=1e-12, atol=1e-12
            ):
                raise ValueError(
                    f"Frozen training-positive activity scales disagree for "
                    f"{ligand!r}->{receptor!r}."
                )
        ligand_columns.append(np.clip(cache[ligand] / ligand_scale, 0.0, 1.0))
        receptor_columns.append(np.clip(cache[receptor] / receptor_scale, 0.0, 1.0))

    ligand_activity = np.column_stack(ligand_columns).astype(np.float32, copy=False)
    receptor_activity = np.column_stack(receptor_columns).astype(np.float32, copy=False)
    if not np.isclose(weights.sum(), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("Frozen pathway-balanced pair weights do not sum to one.")
    return ligand_activity, receptor_activity, normalization


def scaled_lr_activities_from_manifest(
    source: ad.AnnData,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Rebuild LR activities using only frozen manifest semantics/artifacts."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    configuration, filtering, normalization = _validate_expression_metadata_contract(
        manifest
    )
    filtered_database, requested_genes = _load_frozen_filtered_database(
        manifest, manifest_path
    )
    metadata_path = _resolve_manifest_artifact(
        manifest_path, manifest, "lr_pair_metadata.csv"
    )
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != int(filtering["rows_final"]):
        raise ValueError("Frozen LR metadata row count disagrees with the manifest.")
    observed_pathway_counts = {
        str(key): int(value)
        for key, value in metadata["pathway"]
        .astype(str)
        .value_counts()
        .sort_index()
        .items()
    }
    if (
        metadata["pathway"].nunique() != int(filtering["pathways_final"])
        or observed_pathway_counts != filtering["pathway_counts"]
    ):
        raise ValueError("Frozen final pathway counts disagree with LR metadata.")
    split_path = _resolve_manifest_artifact(manifest_path, manifest, "cell_splits.npz")
    with np.load(split_path, allow_pickle=False) as split_arrays:
        train_keys = sorted(key for key in split_arrays.files if key.endswith("_train"))
        if not train_keys:
            raise ValueError("Frozen cell-split artifact contains no training cells.")
        training_indices = np.concatenate(
            [np.asarray(split_arrays[key], dtype=np.int64) for key in train_keys]
        )
    ligand, receptor, observed_normalization = scaled_lr_activities_from_metadata(
        source,
        metadata,
        gene_symbol_key=str(configuration.get("gene_symbol_key", "gene")),
        target_sum=float(normalization["target_sum"]),
        # The builder records normalization metadata before the min-cells and
        # positive-training-activity LR-row filters.  Recover that complete
        # requested universe from the digest-bound database; using the
        # manifest's present_genes as input here would only verify a value
        # against itself and could conceal omitted-but-present DB genes.
        normalization_genes=requested_genes,
        training_indices=training_indices,
        expected_activity_scale_quantile=float(
            configuration["activity_scale_quantile"]
        ),
        filtered_database=filtered_database,
        expected_min_cells=int(filtering["min_cells_global_per_subunit"]),
        expected_filter_reason_counts=filtering["filter_reason_counts"],
        expected_rows_after_global_filter=int(
            filtering["rows_after_global_subunit_filter"]
        ),
    )
    frozen_quantile = float(configuration["activity_scale_quantile"])
    metadata_quantile = float(metadata["activity_scale_quantile"].iloc[0])
    if not np.isclose(frozen_quantile, metadata_quantile, rtol=0.0, atol=1e-15):
        raise ValueError(
            "LR metadata activity scaling differs from the edge-prior configuration."
        )
    if observed_normalization != normalization:
        differing = sorted(
            key
            for key in _NORMALIZATION_FIELDS
            if observed_normalization.get(key) != normalization.get(key)
        )
        raise ValueError(
            "Recomputed LR normalization metadata differs from the complete "
            f"frozen contract; fields={differing}."
        )
    return ligand, receptor, metadata, observed_normalization


def _nonself_type_pair_products(
    ligand: np.ndarray,
    receptor: np.ndarray,
    sender_indices: np.ndarray,
    receiver_indices: np.ndarray,
    *,
    same_population: bool,
) -> tuple[np.ndarray, int]:
    """Mean LR products over all directed type pairs, excluding self-pairs."""

    sender_indices = np.asarray(sender_indices, dtype=np.int64)
    receiver_indices = np.asarray(receiver_indices, dtype=np.int64)
    if same_population:
        n = int(len(sender_indices))
        if n < 2:
            return np.zeros(ligand.shape[1], dtype=np.float64), 0
        ligand_values = ligand[sender_indices].astype(np.float64, copy=False)
        receptor_values = receptor[sender_indices].astype(np.float64, copy=False)
        numerator = ligand_values.sum(axis=0) * receptor_values.sum(axis=0)
        numerator -= (ligand_values * receptor_values).sum(axis=0)
        return numerator / float(n * (n - 1)), n * (n - 1)
    n_pairs = int(len(sender_indices) * len(receiver_indices))
    if n_pairs == 0:
        return np.zeros(ligand.shape[1], dtype=np.float64), 0
    return (
        ligand[sender_indices].mean(axis=0, dtype=np.float64)
        * receptor[receiver_indices].mean(axis=0, dtype=np.float64),
        n_pairs,
    )


def compute_type_lr_scores(
    ligand_activity: np.ndarray,
    receptor_activity: np.ndarray,
    lr_metadata: pd.DataFrame,
    *,
    times: Sequence[Any],
    cell_types: Sequence[Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute observed type-level ``Q_AB,p`` for LR rows and pathways.

    Products are averaged over all directed, non-self cell pairs within one
    observed time point.  The frozen pathway-balanced ``pair_weight`` is then
    applied to every LR row.  Summing row scores within a pathway gives the
    pathway score; summing pathways gives the edge-prior's total LR score.
    """

    ligand = np.asarray(ligand_activity, dtype=np.float32)
    receptor = np.asarray(receptor_activity, dtype=np.float32)
    if ligand.shape != receptor.shape or ligand.ndim != 2:
        raise ValueError(
            "Ligand and receptor activities must have equal [cell, LR] shape."
        )
    if ligand.shape[1] != len(lr_metadata):
        raise ValueError("LR activity columns do not align with LR metadata rows.")
    time_values = np.asarray(times, dtype=object)
    type_values = np.asarray(cell_types, dtype=object)
    if len(time_values) != ligand.shape[0] or len(type_values) != ligand.shape[0]:
        raise ValueError("times/cell_types must align with LR activity rows.")
    if pd.isna(time_values).any() or pd.isna(type_values).any():
        raise ValueError("times and cell_types cannot contain missing values.")

    weights = lr_metadata["pair_weight"].to_numpy(dtype=np.float64)
    rows: list[pd.DataFrame] = []
    ordered_times = list(pd.unique(time_values))
    try:
        ordered_times = sorted(ordered_times)
    except TypeError:
        ordered_times = sorted(ordered_times, key=str)
    for time_value in ordered_times:
        time_mask = time_values == time_value
        types = sorted(set(map(str, type_values[time_mask])))
        indices_by_type = {
            label: np.flatnonzero(time_mask & (type_values.astype(str) == label))
            for label in types
        }
        for sender in types:
            for receiver in types:
                products, n_cell_pairs = _nonself_type_pair_products(
                    ligand,
                    receptor,
                    indices_by_type[sender],
                    indices_by_type[receiver],
                    same_population=sender == receiver,
                )
                q_values = products * weights
                frame = lr_metadata.copy()
                frame.insert(0, "time", time_value)
                frame.insert(1, "sender_type", sender)
                frame.insert(2, "receiver_type", receiver)
                frame.insert(3, "n_directed_cell_pairs", int(n_cell_pairs))
                frame["mean_unweighted_lr_product"] = products
                frame["Q_AB_lr_pair"] = q_values
                rows.append(frame)
    pair_scores = pd.concat(rows, ignore_index=True)
    pathway_scores = pair_scores.groupby(
        ["time", "sender_type", "receiver_type", "pathway", "annotation"],
        as_index=False,
        sort=True,
    ).agg(
        Q_AB_pathway=("Q_AB_lr_pair", "sum"),
        n_lr_rows=("Q_AB_lr_pair", "size"),
        n_directed_cell_pairs=("n_directed_cell_pairs", "first"),
    )
    total_scores = pathway_scores.groupby(
        ["time", "sender_type", "receiver_type"],
        as_index=False,
        sort=True,
    ).agg(
        Q_AB_total=("Q_AB_pathway", "sum"),
        n_pathways=("pathway", "size"),
        n_directed_cell_pairs=("n_directed_cell_pairs", "first"),
    )
    return pair_scores, pathway_scores, total_scores


@dataclass(frozen=True)
class ExactGroupDecomposition:
    """Exact sender-type message decomposition for one GNN group."""

    sender_types: tuple[str, ...]
    model_output: torch.Tensor
    sender_contributions: torch.Tensor
    edge_index: torch.Tensor
    edge_output_contributions: torch.Tensor
    edge_attention: torch.Tensor
    edge_mass_fraction: torch.Tensor
    reconstruction_max_abs: float
    reconstruction_rmse: float
    reconstruction_relative_l2: float


def _is_legacy_state_attention_architecture(gnn: torch.nn.Module) -> bool:
    """Recognize the frozen pre-``force_mode`` one-layer attention model.

    The Weinreb checkpoints were fitted with a frozen GNN implementation that
    predates the explicit ``force_mode``/``aggregation_mode`` configuration
    fields.  Missing metadata is accepted only when the instantiated module is
    unambiguously the old state-attention path.  Merely having a GNN-shaped
    object is insufficient, and any radial/potential head fails closed.
    """

    layers = getattr(gnn, "gnn_layers", None)
    if layers is None or len(layers) != 1:
        return False
    layer = layers[0]
    required_gnn_attributes = (
        "state_embed",
        "distance_projection",
        "rbf_expansion",
        "state_readout",
    )
    required_layer_attributes = (
        "layernorm",
        "q_proj",
        "k_proj",
        "v_proj",
        "dk_proj",
        "dv_proj",
        "attn_activation",
        "out_transform",
        "hidden_dim",
        "num_heads",
        "head_dim",
    )
    forbidden_heads = ("radial_force_net", "radial_potential_net", "radial_cutoff")
    return (
        layer.__class__.__name__ == "StateGraphAttentionLayer"
        and all(hasattr(gnn, name) for name in required_gnn_attributes)
        and all(hasattr(layer, name) for name in required_layer_attributes)
        and not any(hasattr(gnn, name) for name in forbidden_heads)
    )


def validate_exact_decomposition_model(gnn: torch.nn.Module) -> None:
    """Reject architectures for which sender identity is not exactly separable."""

    if not bool(getattr(gnn, "state_space", False)):
        raise ValueError("Exact LR drift attribution requires state_space=True.")
    if bool(getattr(gnn, "use_spatial", True)):
        raise ValueError("Exact LR drift attribution requires use_spatial=False.")
    if hasattr(gnn, "force_mode"):
        if str(gnn.force_mode).strip().lower() != "attention":
            raise ValueError(
                "Exact LR drift attribution currently supports only explicit "
                "attention force_mode."
            )
    elif not _is_legacy_state_attention_architecture(gnn):
        raise ValueError(
            "A model without force_mode is accepted only when it exactly matches "
            "the frozen legacy one-layer StateGraphAttentionLayer architecture."
        )
    layers = getattr(gnn, "gnn_layers", None)
    if layers is None or len(layers) != 1:
        raise ValueError(
            "Exact sender decomposition requires exactly one GNN layer; "
            "multi-layer sender identities are mixed and will not be approximated."
        )
    if (
        not hasattr(gnn, "state_readout")
        or getattr(gnn.state_readout, "bias", None) is not None
    ):
        raise ValueError("Exact decomposition requires a bias-free state readout.")
    if str(getattr(gnn, "edge_mode", "")) not in {"radius", "predictor"}:
        raise ValueError("Unsupported GNN edge mode for exact decomposition.")


def _exact_state_edges(
    gnn: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pairwise_distances = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), dim=2)
    rows, cols = torch.where(pairwise_distances < float(gnn.cutoff))
    if str(gnn.edge_mode) == "radius":
        connected = torch.ones(rows.shape[0], dtype=torch.bool, device=x.device)
    else:
        logits = gnn.link_predictor(torch.cat((x[rows], x[cols]), dim=1))
        connected = torch.sigmoid(logits).reshape(-1) >= float(gnn.edge_predictor_thre)
    edge_index = torch.stack((rows[connected], cols[connected]), dim=0)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    distances = pairwise_distances[edge_index[0], edge_index[1]]
    nonzero = distances > 1.0e-6
    return edge_index[:, nonzero], distances[nonzero]


def exact_state_gnn_group_decomposition(
    gnn: torch.nn.Module,
    x: torch.Tensor,
    lnw: torch.Tensor,
    sender_labels: Sequence[Any],
    *,
    t: torch.Tensor | None = None,
) -> ExactGroupDecomposition:
    """Decompose a one-layer state GNN output into exact sender-type messages.

    The calculation mirrors ``GNNInteraction.forward`` and
    ``StateGraphAttentionLayer.message/aggregate`` operation for operation.
    It uses signed transformed messages, particle masses and the exact
    connected-mass denominator.  Attention magnitudes are never substituted
    for message vectors.
    """

    validate_exact_decomposition_model(gnn)
    if x.ndim != 2 or x.shape[1] != int(gnn.in_out_dim):
        raise ValueError("x must have shape [group, gnn.in_out_dim].")
    if lnw.numel() != x.shape[0]:
        raise ValueError("lnw must have one value per group cell.")
    labels = np.asarray([str(value) for value in sender_labels], dtype=object)
    if len(labels) != x.shape[0]:
        raise ValueError("sender_labels must have one value per group cell.")
    sender_types = tuple(sorted(set(labels.tolist())))
    type_to_code = {label: index for index, label in enumerate(sender_types)}
    label_codes = torch.as_tensor(
        [type_to_code[label] for label in labels],
        dtype=torch.long,
        device=x.device,
    )
    if t is None:
        t = x.new_zeros(1)

    with torch.no_grad():
        edge_index, distances = _exact_state_edges(gnn, x)
        n_cells, output_dim = x.shape
        n_types = len(sender_types)
        if edge_index.numel() == 0:
            model_output = gnn(x, lnw, t)
            contributions = x.new_zeros((n_types, n_cells, output_dim))
            residual = contributions.sum(dim=0) - model_output
            num_heads = int(gnn.gnn_layers[0].num_heads)
            return ExactGroupDecomposition(
                sender_types=sender_types,
                model_output=model_output.detach(),
                sender_contributions=contributions,
                edge_index=edge_index.detach(),
                edge_output_contributions=x.new_zeros((0, output_dim)),
                edge_attention=x.new_zeros((0, num_heads)),
                edge_mass_fraction=x.new_zeros((0,)),
                reconstruction_max_abs=float(residual.abs().max().item()),
                reconstruction_rmse=float(
                    torch.sqrt(torch.mean(residual.square())).item()
                ),
                reconstruction_relative_l2=float(
                    torch.linalg.vector_norm(residual).item()
                    / max(torch.linalg.vector_norm(model_output).item(), 1.0e-12)
                ),
            )

        source, target = edge_index
        x_embed = gnn.state_embed(x)
        rbf_distance = torch.clamp(distances / float(gnn.cutoff), min=0.0, max=1.0)
        rbf = gnn.rbf_expansion(rbf_distance)
        edge_attr = (x_embed[source] + x_embed[target]) * gnn.distance_projection(rbf)
        layer = gnn.gnn_layers[0]
        normalized = layer.layernorm(x_embed)
        q = layer.q_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
        k = layer.k_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
        v = layer.v_proj(normalized).reshape(-1, layer.num_heads, layer.head_dim)
        dk = layer.dk_proj(edge_attr).reshape(-1, layer.num_heads, layer.head_dim)
        dv = layer.dv_proj(edge_attr).reshape(-1, layer.num_heads, layer.head_dim)
        attention = layer.attn_activation((q[target] * k[source] * dk).sum(dim=-1))
        hidden_message = layer.out_transform(v[source] * dv)
        hidden_message = (hidden_message * attention.unsqueeze(-1)).reshape(
            -1, layer.hidden_dim
        )
        particle_mass = torch.exp(lnw).reshape(-1) * lnw.shape[0]
        weighted_hidden_message = hidden_message * particle_mass[source].unsqueeze(-1)

        denominator = x.new_zeros((n_cells, 1))
        denominator.index_add_(0, target, particle_mass[source].unsqueeze(-1))
        edge_mass_fraction = particle_mass[source] / denominator[target].reshape(
            -1
        ).clamp_min(torch.finfo(x.dtype).eps)
        edge_hidden_contribution = weighted_hidden_message / denominator[
            target
        ].clamp_min(torch.finfo(x.dtype).eps)
        edge_output = gnn.state_readout(edge_hidden_contribution)

        contributions = x.new_zeros((n_types, n_cells, output_dim))
        source_codes = label_codes[source]
        for type_code in range(n_types):
            mask = source_codes == type_code
            if mask.any():
                contributions[type_code].index_add_(0, target[mask], edge_output[mask])

        model_output = gnn(x, lnw, t)
        reconstructed = contributions.sum(dim=0)
        residual = reconstructed - model_output
        maximum = float(residual.abs().max().item())
        rmse = float(torch.sqrt(torch.mean(residual.square())).item())
        relative = float(
            torch.linalg.vector_norm(residual).item()
            / max(torch.linalg.vector_norm(model_output).item(), 1.0e-12)
        )
    return ExactGroupDecomposition(
        sender_types=sender_types,
        model_output=model_output.detach(),
        sender_contributions=contributions.detach(),
        edge_index=edge_index.detach(),
        edge_output_contributions=edge_output.detach(),
        edge_attention=attention.detach(),
        edge_mass_fraction=edge_mass_fraction.detach(),
        reconstruction_max_abs=maximum,
        reconstruction_rmse=rmse,
        reconstruction_relative_l2=relative,
    )


def runtime_style_random_groups(
    indices: Sequence[int], *, group_size: int, seed: int
) -> list[np.ndarray]:
    """Partition cells like ``cal_interaction_gnn``, with a fixed RNG seed.

    When a remainder exists, the final full group is merged into it.  This
    avoids a one-cell remainder and matches the sizes used in model execution.
    """

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("At least two cells are required for interaction grouping.")
    if int(group_size) < 2:
        raise ValueError("group_size must be at least two.")
    permutation = np.random.default_rng(int(seed)).permutation(indices)
    n = len(permutation)
    if n % int(group_size) == 0:
        n_full = n // int(group_size)
    elif n < int(group_size):
        n_full = 0
    else:
        n_full = n // int(group_size) - 1
    groups = [
        permutation[start : start + int(group_size)]
        for start in range(0, n_full * int(group_size), int(group_size))
    ]
    remainder = permutation[n_full * int(group_size) :]
    if len(remainder):
        groups.append(remainder)
    if any(len(group) < 2 for group in groups):
        raise RuntimeError("Runtime-style grouping produced an isolated cell.")
    return groups


def analyze_exact_groupings(
    gnn: torch.nn.Module,
    latent: np.ndarray,
    *,
    observed_times: Sequence[Any],
    model_times: Sequence[float],
    cell_types: Sequence[Any],
    grouping_seeds: Sequence[int],
    training_seed: int,
    model_label: str,
    group_size: int = 16,
    device: str | torch.device = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run exact decompositions over fixed random partitions of observed cells.

    Each observed cell appears once per grouping seed within its time point.
    Missing sender types in a particular group contribute an exact zero and
    are included in the receiver denominator used for summary means.
    """

    validate_exact_decomposition_model(gnn)
    values = np.asarray(latent, dtype=np.float32)
    observed = np.asarray(observed_times, dtype=object)
    numerical_times = np.asarray(model_times, dtype=np.float32)
    labels = np.asarray([str(value) for value in cell_types], dtype=object)
    if not (len(values) == len(observed) == len(numerical_times) == len(labels)):
        raise ValueError(
            "latent, observed_times, model_times and cell_types must align."
        )
    if not np.isfinite(values).all() or not np.isfinite(numerical_times).all():
        raise ValueError("latent/model_times contain non-finite values.")
    all_types = tuple(sorted(set(labels.tolist())))
    dim = values.shape[1]
    selected_device = torch.device(device)
    gnn = gnn.to(selected_device)
    gnn.eval()
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    ordered_times = list(pd.unique(observed))
    try:
        ordered_times = sorted(ordered_times)
    except TypeError:
        ordered_times = sorted(ordered_times, key=str)
    for time_code, observed_time in enumerate(ordered_times):
        time_indices = np.flatnonzero(observed == observed_time).astype(np.int64)
        unique_model_times = np.unique(numerical_times[time_indices])
        if len(unique_model_times) != 1:
            raise ValueError(
                f"Observed time {observed_time!r} maps to {len(unique_model_times)} model times."
            )
        model_time = float(unique_model_times[0])
        receiver_counts = {
            receiver: int(np.sum(labels[time_indices] == receiver))
            for receiver in all_types
        }
        for grouping_seed in grouping_seeds:
            effective_seed = int(grouping_seed) + int(time_code) * 1_000_003
            groups = runtime_style_random_groups(
                time_indices, group_size=int(group_size), seed=effective_seed
            )
            vector_sum = {
                (sender, receiver): np.zeros(dim, dtype=np.float64)
                for sender in all_types
                for receiver in all_types
            }
            norm_sum = {(a, b): 0.0 for a in all_types for b in all_types}
            total_norm_sum = {receiver: 0.0 for receiver in all_types}
            cosine_sum = {(a, b): 0.0 for a in all_types for b in all_types}
            cosine_count = {(a, b): 0 for a in all_types for b in all_types}
            edge_count = {(a, b): 0 for a in all_types for b in all_types}
            attention_abs_sum = {(a, b): 0.0 for a in all_types for b in all_types}
            attention_signed_sum = {(a, b): 0.0 for a in all_types for b in all_types}
            attention_positive_sum = {(a, b): 0.0 for a in all_types for b in all_types}
            attention_negative_sum = {(a, b): 0.0 for a in all_types for b in all_types}
            attention_share_sum = {(a, b): 0.0 for a in all_types for b in all_types}
            attention_edge_abs_sum = {(a, b): 0.0 for a in all_types for b in all_types}
            attention_edge_signed_sum = {
                (a, b): 0.0 for a in all_types for b in all_types
            }
            attention_edge_abs_head_mean_sum = {
                (a, b): 0.0 for a in all_types for b in all_types
            }
            attention_positive_head_count = {
                (a, b): 0 for a in all_types for b in all_types
            }
            total_attention_abs_sum = {receiver: 0.0 for receiver in all_types}
            attention_covered_receiver_count = {receiver: 0 for receiver in all_types}
            maximum_residual = 0.0
            squared_residual_sum = 0.0
            residual_elements = 0
            output_squared_sum = 0.0
            group_sizes: list[int] = []

            for group_indices in groups:
                group_sizes.append(int(len(group_indices)))
                x = torch.as_tensor(
                    values[group_indices], dtype=torch.float32, device=selected_device
                )
                lnw = torch.full(
                    (len(group_indices), 1),
                    -math.log(float(len(group_indices))),
                    dtype=x.dtype,
                    device=x.device,
                )
                decomposition = exact_state_gnn_group_decomposition(
                    gnn,
                    x,
                    lnw,
                    labels[group_indices],
                    t=x.new_tensor([model_time]),
                )
                maximum_residual = max(
                    maximum_residual, decomposition.reconstruction_max_abs
                )
                reconstructed = decomposition.sender_contributions.sum(dim=0)
                residual = reconstructed - decomposition.model_output
                squared_residual_sum += float(residual.square().sum().item())
                residual_elements += int(residual.numel())
                output_squared_sum += float(
                    decomposition.model_output.square().sum().item()
                )
                local_types = {
                    sender: index
                    for index, sender in enumerate(decomposition.sender_types)
                }
                output = (
                    decomposition.model_output.cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                local_labels = labels[group_indices]
                for receiver in all_types:
                    receiver_local = np.flatnonzero(local_labels == receiver)
                    if not len(receiver_local):
                        continue
                    total_vectors = output[receiver_local]
                    total_norms = np.linalg.norm(total_vectors, axis=1)
                    total_norm_sum[receiver] += float(total_norms.sum())
                    for sender in all_types:
                        if sender in local_types:
                            vectors = (
                                decomposition.sender_contributions[
                                    local_types[sender], receiver_local
                                ]
                                .cpu()
                                .numpy()
                                .astype(np.float64, copy=False)
                            )
                        else:
                            vectors = np.zeros(
                                (len(receiver_local), dim), dtype=np.float64
                            )
                        vector_sum[(sender, receiver)] += vectors.sum(axis=0)
                        vector_norms = np.linalg.norm(vectors, axis=1)
                        norm_sum[(sender, receiver)] += float(vector_norms.sum())
                        valid = (vector_norms > 1.0e-12) & (total_norms > 1.0e-12)
                        if valid.any():
                            cosine = np.sum(
                                vectors[valid] * total_vectors[valid], axis=1
                            ) / (vector_norms[valid] * total_norms[valid])
                            cosine_sum[(sender, receiver)] += float(cosine.sum())
                            cosine_count[(sender, receiver)] += int(valid.sum())

                edges = decomposition.edge_index.cpu().numpy()
                edge_attention = (
                    decomposition.edge_attention.cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                edge_mass_fraction = (
                    decomposition.edge_mass_fraction.cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                if edge_attention.shape != (
                    edges.shape[1],
                    int(gnn.gnn_layers[0].num_heads),
                ):
                    raise RuntimeError(
                        "Edge attention tensor does not align with edge_index."
                    )
                if edge_mass_fraction.shape != (edges.shape[1],):
                    raise RuntimeError(
                        "Edge mass fractions do not align with edge_index."
                    )

                # These summaries deliberately stop before value projection and
                # state readout.  The native magnitude preserves the model's
                # connected-mass denominator; the receiver share is a separate
                # post-hoc normalization of absolute attention and is not a
                # softmax probability used by the fitted model.
                head_abs = np.mean(np.abs(edge_attention), axis=1)
                head_signed = np.mean(edge_attention, axis=1)
                head_positive = np.mean(np.maximum(edge_attention, 0.0), axis=1)
                head_negative = np.mean(np.maximum(-edge_attention, 0.0), axis=1)
                abs_head_mean = np.abs(head_signed)
                positive_head_count = np.sum(edge_attention > 0.0, axis=1)
                native_abs = edge_mass_fraction * head_abs
                native_signed = edge_mass_fraction * head_signed
                native_positive = edge_mass_fraction * head_positive
                native_negative = edge_mass_fraction * head_negative
                receiver_abs_total = np.zeros(len(group_indices), dtype=np.float64)
                if edges.shape[1]:
                    np.add.at(receiver_abs_total, edges[1], native_abs)
                receiver_share = np.divide(
                    native_abs,
                    receiver_abs_total[edges[1]],
                    out=np.zeros_like(native_abs),
                    where=receiver_abs_total[edges[1]] > 0.0,
                )

                for edge_position, (local_source, local_target) in enumerate(edges.T):
                    key = (local_labels[local_source], local_labels[local_target])
                    edge_count[key] += 1
                    attention_abs_sum[key] += float(native_abs[edge_position])
                    attention_signed_sum[key] += float(native_signed[edge_position])
                    attention_positive_sum[key] += float(native_positive[edge_position])
                    attention_negative_sum[key] += float(native_negative[edge_position])
                    attention_share_sum[key] += float(receiver_share[edge_position])
                    attention_edge_abs_sum[key] += float(head_abs[edge_position])
                    attention_edge_signed_sum[key] += float(head_signed[edge_position])
                    attention_edge_abs_head_mean_sum[key] += float(
                        abs_head_mean[edge_position]
                    )
                    attention_positive_head_count[key] += int(
                        positive_head_count[edge_position]
                    )
                for receiver in all_types:
                    receiver_local = np.flatnonzero(local_labels == receiver)
                    if not len(receiver_local):
                        continue
                    total_attention_abs_sum[receiver] += float(
                        receiver_abs_total[receiver_local].sum()
                    )
                    attention_covered_receiver_count[receiver] += int(
                        np.count_nonzero(receiver_abs_total[receiver_local] > 0.0)
                    )

            rmse = math.sqrt(squared_residual_sum / max(residual_elements, 1))
            relative_l2 = math.sqrt(squared_residual_sum) / max(
                math.sqrt(output_squared_sum), 1.0e-12
            )
            diagnostics.append(
                {
                    "model_label": str(model_label),
                    "training_seed": int(training_seed),
                    "grouping_seed": int(grouping_seed),
                    "effective_grouping_seed": int(effective_seed),
                    "time": observed_time,
                    "model_time": model_time,
                    "n_cells": int(len(time_indices)),
                    "n_groups": int(len(groups)),
                    "nominal_group_size": int(group_size),
                    "minimum_group_size": int(min(group_sizes)),
                    "maximum_group_size": int(max(group_sizes)),
                    "reconstruction_max_abs": float(maximum_residual),
                    "reconstruction_rmse": float(rmse),
                    "reconstruction_relative_l2": float(relative_l2),
                }
            )
            for sender in all_types:
                for receiver in all_types:
                    count = int(receiver_counts[receiver])
                    if count == 0:
                        continue
                    mean_vector = vector_sum[(sender, receiver)] / float(count)
                    valid_cosines = cosine_count[(sender, receiver)]
                    row: dict[str, Any] = {
                        "model_label": str(model_label),
                        "training_seed": int(training_seed),
                        "grouping_seed": int(grouping_seed),
                        "time": observed_time,
                        "model_time": model_time,
                        "sender_type": sender,
                        "receiver_type": receiver,
                        "n_receiver_cells": count,
                        "connected_edge_count": int(edge_count[(sender, receiver)]),
                        "G_AB_abs_mean_per_edge": float(
                            attention_edge_abs_sum[(sender, receiver)]
                            / edge_count[(sender, receiver)]
                            if edge_count[(sender, receiver)]
                            else 0.0
                        ),
                        "G_AB_signed_mean_per_edge": float(
                            attention_edge_signed_sum[(sender, receiver)]
                            / edge_count[(sender, receiver)]
                            if edge_count[(sender, receiver)]
                            else 0.0
                        ),
                        "G_AB_abs_of_head_mean_per_edge": float(
                            attention_edge_abs_head_mean_sum[(sender, receiver)]
                            / edge_count[(sender, receiver)]
                            if edge_count[(sender, receiver)]
                            else 0.0
                        ),
                        "G_AB_positive_head_fraction": float(
                            attention_positive_head_count[(sender, receiver)]
                            / (
                                edge_count[(sender, receiver)]
                                * int(gnn.gnn_layers[0].num_heads)
                            )
                            if edge_count[(sender, receiver)]
                            else 0.0
                        ),
                        "D_AB": float(norm_sum[(sender, receiver)] / count),
                        "A_AB_abs": float(
                            attention_abs_sum[(sender, receiver)] / count
                        ),
                        "A_AB_signed": float(
                            attention_signed_sum[(sender, receiver)] / count
                        ),
                        "A_AB_positive": float(
                            attention_positive_sum[(sender, receiver)] / count
                        ),
                        "A_AB_negative_magnitude": float(
                            attention_negative_sum[(sender, receiver)] / count
                        ),
                        "A_AB_receiver_share_all": float(
                            attention_share_sum[(sender, receiver)] / count
                        ),
                        "A_AB_receiver_share_supported": float(
                            attention_share_sum[(sender, receiver)]
                            / attention_covered_receiver_count[receiver]
                            if attention_covered_receiver_count[receiver]
                            else math.nan
                        ),
                        "mean_total_abs_attention": float(
                            total_attention_abs_sum[receiver] / count
                        ),
                        "attention_receiver_coverage": float(
                            attention_covered_receiver_count[receiver] / count
                        ),
                        "norm_mean_sender_drift": float(np.linalg.norm(mean_vector)),
                        "mean_total_interaction_norm": float(
                            total_norm_sum[receiver] / count
                        ),
                        "mean_cosine_to_total_interaction": (
                            float(cosine_sum[(sender, receiver)] / valid_cosines)
                            if valid_cosines
                            else 0.0
                        ),
                        "cosine_valid_receiver_count": int(valid_cosines),
                    }
                    row.update(
                        {
                            f"drift_pc_{index + 1}": float(value)
                            for index, value in enumerate(mean_vector)
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def summarize_drift_across_seeds(
    drift: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize grouping replicates hierarchically across training seeds.

    Grouping seeds are technical Monte-Carlo replicates of the random
    interaction partition, while training seeds are independently fitted
    models.  They are intentionally not pooled into one pseudo-replicate
    standard deviation: the output keeps within-model grouping variation and
    between-model training-seed variation separate.
    """

    keys = ["time", "model_time", "sender_type", "receiver_type"]
    required = {"training_seed", "grouping_seed", *keys, "D_AB"}
    missing = sorted(required.difference(drift.columns))
    if missing:
        raise ValueError(f"Drift table is missing columns: {missing}.")
    duplicate_keys = ["training_seed", "grouping_seed", *keys]
    if drift.duplicated(duplicate_keys).any():
        raise ValueError("Drift table contains duplicate seed/type/time rows.")
    design_counts = (
        drift.groupby(["training_seed", *keys], sort=True)["grouping_seed"]
        .nunique()
        .to_numpy(dtype=np.int64)
    )
    expected_grouping_count = int(drift["grouping_seed"].nunique())
    if not len(design_counts) or np.any(design_counts != expected_grouping_count):
        raise ValueError("Drift table has an incomplete grouping-seed design.")
    pc_columns = [column for column in drift.columns if column.startswith("drift_pc_")]
    scalar_columns = [
        "connected_edge_count",
        "D_AB",
        "norm_mean_sender_drift",
        "mean_total_interaction_norm",
        "mean_cosine_to_total_interaction",
        *pc_columns,
    ]
    optional_attention_columns = [
        "G_AB_abs_mean_per_edge",
        "G_AB_signed_mean_per_edge",
        "G_AB_abs_of_head_mean_per_edge",
        "G_AB_positive_head_fraction",
        "A_AB_abs",
        "A_AB_signed",
        "A_AB_positive",
        "A_AB_negative_magnitude",
        "A_AB_receiver_share_all",
        "A_AB_receiver_share_supported",
        "mean_total_abs_attention",
        "attention_receiver_coverage",
    ]
    present_attention_columns = [
        column for column in optional_attention_columns if column in drift.columns
    ]
    if present_attention_columns and len(present_attention_columns) != len(
        optional_attention_columns
    ):
        missing_attention = sorted(
            set(optional_attention_columns).difference(present_attention_columns)
        )
        raise ValueError(
            "Drift table contains an incomplete attention summary: "
            f"missing {missing_attention}."
        )
    scalar_columns.extend(present_attention_columns)
    missing_scalars = sorted(set(scalar_columns).difference(drift.columns))
    if missing_scalars:
        raise ValueError(f"Drift table is missing summary columns: {missing_scalars}.")
    per_seed_grouped = drift.groupby(
        ["training_seed", *keys], as_index=False, sort=True
    )
    by_seed = per_seed_grouped[scalar_columns].mean()
    grouping_std = (
        per_seed_grouped[scalar_columns]
        .std(ddof=1)
        .rename(
            columns={column: f"{column}_grouping_seed_std" for column in scalar_columns}
        )
    )
    by_seed = by_seed.merge(
        grouping_std,
        on=["training_seed", *keys],
        how="left",
        validate="one_to_one",
    )
    by_seed["n_grouping_seeds"] = expected_grouping_count
    grouped = by_seed.groupby(keys, as_index=False, sort=True)
    summary = grouped[scalar_columns].mean()
    summary = summary.rename(
        columns={column: f"{column}_mean" for column in scalar_columns}
    )
    std = (
        grouped[scalar_columns]
        .std(ddof=1)
        .rename(
            columns={column: f"{column}_training_seed_std" for column in scalar_columns}
        )
    )
    summary = summary.merge(std, on=keys, how="left", validate="one_to_one")
    grouping_std_columns = [f"{column}_grouping_seed_std" for column in scalar_columns]
    mean_grouping_std = (
        grouped[grouping_std_columns]
        .mean()
        .rename(columns={column: f"{column}_mean" for column in grouping_std_columns})
    )
    summary = summary.merge(
        mean_grouping_std, on=keys, how="left", validate="one_to_one"
    )
    seed_counts = (
        grouped["training_seed"]
        .nunique()
        .rename(columns={"training_seed": "n_training_seeds"})
    )
    summary = summary.merge(seed_counts, on=keys, how="left", validate="one_to_one")
    summary["n_grouping_seeds_per_training_seed"] = expected_grouping_count
    return by_seed, summary


__all__ = [
    "ExactGroupDecomposition",
    "analyze_exact_groupings",
    "compute_type_lr_scores",
    "exact_state_gnn_group_decomposition",
    "load_edge_prior_manifest",
    "runtime_style_random_groups",
    "scaled_lr_activities_from_manifest",
    "scaled_lr_activities_from_metadata",
    "sha256_file",
    "summarize_drift_across_seeds",
    "validate_exact_decomposition_model",
]
