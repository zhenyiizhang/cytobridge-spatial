#!/usr/bin/env python3
"""Validate the four corrected CytoBridge de novo production runs.

This is an artifact-level scientific acceptance check.  It reads the outputs
created by ``cytobridge workflow --train`` and verifies the shared protocol,
the complete six-stage fit, each dataset's declared edge prior, and the
numerical downstream products.  It does not modify the run directory.

Example
-------
python scripts/validate_corrected_de_novo_run.py \
    --run-root /path/to/corrected-de-novo-20260813-r2 \
    --report /path/to/corrected-de-novo-acceptance.json
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml


STAGES = (
    "Pretrain",
    "Refine",
    "Init_interaction",
    "Train_Score",
    "Finetune",
    "Score_Refine",
)


DATASETS = {
    "zebrafish": {
        "shape": (11999, 26628),
        "counts_layer": "counts",
        "annotation_key": "Annotation",
        "classifier_k": 10,
        "cutoff": 0.09606367405591873,
        "edge_prior_mode": "learned",
        "edge_predictor_threshold": 0.6063615679740906,
        "observed_counts": {0.0: 563, 1.0: 1036, 2.0: 2081, 3.0: 3048, 4.0: 5271},
        "interpolated": (0.5, 1.5, 2.5, 3.5),
        "score_epochs": 2001,
        "species": "zebrafish",
        "lr_database_name": "CellChatDB.ligrec.zebrafish.csv",
        "lr_database_sha256": "27fd0eb35da035a371ef68783d3e2dcf0729668fd58c2bb59f203173ea1b3f37",
    },
    "mosta": {
        "shape": (344603, 23761),
        "counts_layer": "count",
        "annotation_key": "Annotation",
        "classifier_k": 10,
        "cutoff": 0.02400244047956264,
        "edge_prior_mode": "learned",
        "edge_predictor_threshold": 0.1192110925912857,
        "observed_counts": {0.0: 51365, 1.0: 77369, 2.0: 102519, 3.0: 113350},
        "interpolated": (0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75),
        "score_epochs": 2001,
        "species": "mouse",
        "lr_database_name": "CellChatDB.ligrec.mouse.csv",
        "lr_database_sha256": "851c7ac12f4b2ba355eb991cda76646bf215fcab6e3819f7453bce2d3bc77673",
    },
    "arista": {
        "shape": (46209, 16379),
        "counts_layer": "counts",
        "annotation_key": "Annotation",
        "classifier_k": 10,
        "cutoff": 0.03154105148551745,
        "edge_prior_mode": "learned",
        "edge_predictor_threshold": 0.5884028673171997,
        "observed_counts": {0.0: 7668, 1.0: 8106, 2.0: 9440, 3.0: 9676, 4.0: 11319},
        "interpolated": (0.5, 1.5, 2.5, 3.5),
        "score_epochs": 2001,
        "species": "hs",
        "lr_database_name": "CellChatDB.ligrec.human.csv",
        "lr_database_sha256": "8bfb86da81206cc4c1d8ab15a0086e9fc6cd38a7206450a427e4e77bfb32731c",
    },
    "admouse": {
        "shape": (172092, 347),
        "counts_layer": "counts",
        "annotation_key": "major_annotation",
        "classifier_k": 1,
        "cutoff": 0.012106042891492197,
        "edge_prior_mode": "learned",
        "edge_predictor_threshold": 0.9956824779510498,
        "observed_counts": {0.0: 53615, 1.0: 58447, 2.0: 60030},
        "interpolated": tuple(
            round(value / 10, 1) for value in range(1, 26) if value not in (10, 20)
        ),
        "score_epochs": 3001,
        "species": "mouse",
        "lr_database_name": "CellChatDB.ligrec.mouse.csv",
        "lr_database_sha256": "851c7ac12f4b2ba355eb991cda76646bf215fcab6e3819f7453bce2d3bc77673",
        "strict_lr_pairs": 7,
    },
}

ABLATION_PROFILES = {
    "admouse_no_lr_prior": {
        **DATASETS["admouse"],
        "artifact_dataset": "admouse",
        "edge_prior_mode": "all_spatial",
        "edge_predictor_threshold": None,
        "run_role": "no-LR-prior ablation",
        "training_config": ("admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml"),
    }
}
VALIDATION_PROFILES = {**DATASETS, **ABLATION_PROFILES}


@dataclass
class Audit:
    dataset: str
    checks: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def check(self, condition: bool, name: str, detail: str) -> None:
        self.checks.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}
        )

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def errors(self) -> list[dict[str, str]]:
        return [item for item in self.checks if item["status"] == "FAIL"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS" if not self.errors else "FAIL",
            "checks": self.checks,
            "warnings": self.warnings,
        }


def close_enough(actual: float, expected: float, atol: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)


def safe_time_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "minus_").replace(".", "p")


def h5_shape(node: h5py.Dataset | h5py.Group) -> tuple[int, ...]:
    if isinstance(node, h5py.Dataset):
        return tuple(int(value) for value in node.shape)
    return tuple(int(value) for value in node.attrs["shape"])


def numeric_node(node: h5py.Dataset | h5py.Group) -> h5py.Dataset:
    """Return the stored numeric values for a dense or sparse AnnData matrix."""

    return node if isinstance(node, h5py.Dataset) else node["data"]


def numeric_stats(node: h5py.Dataset | h5py.Group) -> tuple[bool, int, float, float]:
    """Stream a numeric HDF5 array and return finite/count/min/max statistics."""

    values = numeric_node(node)
    count = 0
    minimum = math.inf
    maximum = -math.inf
    finite = True
    block_rows = 100_000
    if values.ndim == 0:
        blocks = (np.asarray(values[()]),)
    else:
        blocks = (
            np.asarray(values[start : start + block_rows])
            for start in range(0, values.shape[0], block_rows)
        )
    for block in blocks:
        if block.size == 0:
            continue
        finite = finite and bool(np.isfinite(block).all())
        count += int(block.size)
        minimum = min(minimum, float(np.nanmin(block)))
        maximum = max(maximum, float(np.nanmax(block)))
    return finite, count, minimum, maximum


def h5_sum_identity(
    handle: h5py.File,
    total_key: str,
    part_keys: tuple[str, ...],
    *,
    atol: float = 1e-5,
) -> bool:
    total = numeric_node(handle[total_key])
    parts = [numeric_node(handle[key]) for key in part_keys]
    if any(part.shape != total.shape for part in parts):
        return False
    for start in range(0, total.shape[0], 100_000):
        stop = start + 100_000
        expected = sum(np.asarray(part[start:stop]) for part in parts)
        if not np.allclose(
            np.asarray(total[start:stop]), expected, rtol=1e-5, atol=atol
        ):
            return False
    return True


def read_metadata(path: Path):
    data = ad.read_h5ad(path, backed="r")
    try:
        return data.shape, data.obs.copy(), data.obs_names.copy(), dict(data.uns)
    finally:
        data.file.close()


def expected_epochs(spec: dict[str, Any]) -> dict[str, int]:
    return {
        "Pretrain": 100,
        "Refine": 100,
        "Init_interaction": 50,
        "Train_Score": int(spec["score_epochs"]),
        "Finetune": 1000,
        "Score_Refine": int(spec["score_epochs"]),
    }


def _mapping(value: Any) -> dict[str, Any]:
    """Return a plain mapping when possible, otherwise an empty mapping."""

    return dict(value) if isinstance(value, Mapping) else {}


def _finite_probability(value: Any) -> float | None:
    """Normalize a strict probability, returning ``None`` for invalid values."""

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        return None
    return normalized


def _resolved_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def edge_threshold_provenance_contract(
    predictor_metadata: Any,
    *,
    expected_threshold: float,
) -> tuple[bool, str]:
    """Require the validation-selected threshold frozen for a production run."""

    metadata = _mapping(predictor_metadata)
    effective = _finite_probability(metadata.get("edge_predictor_threshold"))
    selected = _finite_probability(metadata.get("edge_predictor_threshold_selected"))
    expected = _finite_probability(expected_threshold)
    ok = (
        metadata.get("selection_source") == "validation"
        and effective is not None
        and selected is not None
        and expected is not None
        and close_enough(effective, selected)
        and close_enough(effective, expected)
    )
    return ok, (
        f"effective={metadata.get('edge_predictor_threshold')!r}, "
        f"selected={metadata.get('edge_predictor_threshold_selected')!r}, "
        f"expected={expected_threshold!r}, source={metadata.get('selection_source')!r}"
    )


def lr_database_provenance_contract(
    *,
    graph_metadata: Any,
    graph_metadata_present: bool,
    downstream_analysis: Any,
    spec: Mapping[str, Any],
    allow_ignored_input_graph: bool = False,
) -> tuple[bool, str]:
    """Validate the exact LR database and its role in graph and projection steps."""

    graph = _mapping(graph_metadata)
    graph_present = bool(graph_metadata_present or graph_metadata is not None)
    downstream = _mapping(downstream_analysis)
    expected_name = str(spec["lr_database_name"])
    expected_digest = str(spec["lr_database_sha256"])
    expected_species = str(spec["species"])
    mode = str(spec["edge_prior_mode"]).strip().lower()

    downstream_path = _resolved_optional_path(downstream.get("database"))
    downstream_digest = _file_sha256(downstream_path)
    downstream_ok = (
        downstream_path is not None
        and downstream_path.name == expected_name
        and downstream_digest == expected_digest
        and str(downstream.get("complex_mode", "")).lower() in {"min", "minimum"}
        and downstream.get("require_all_subunits") is True
        and str(downstream.get("preferred_species_tag", "")) == expected_species
    )

    graph_path = _resolved_optional_path(graph.get("lr_database_path"))
    graph_digest = _file_sha256(graph_path)
    resolved_pairs = graph.get("lr_unique_resolved_pairs")
    if mode == "learned":
        try:
            pair_count_ok = int(resolved_pairs) > 0
            if spec.get("strict_lr_pairs") is not None:
                pair_count_ok = pair_count_ok and int(resolved_pairs) == int(
                    spec["strict_lr_pairs"]
                )
        except (TypeError, ValueError):
            pair_count_ok = False
        graph_ok = (
            graph_present
            and graph_path is not None
            and graph_path.name == expected_name
            and graph_digest == expected_digest
            and graph.get("lr_matching_rule")
            == "selected_symbol_exact_case_insensitive_all_complex_subunits"
            and graph.get("lr_complex_expression_rule") == "minimum"
            and str(graph.get("preferred_species_tag", "")) == expected_species
            and pair_count_ok
        )
    elif mode == "all_spatial":
        # A matched no-LR ablation may deliberately reuse the main run's exact
        # aligned H5AD.  That input can retain immutable learned-graph provenance,
        # provided training records that it stripped/ignored the metadata and the
        # trained artifact itself is clean.  It is evidence about the shared input,
        # not an edge prior used by the ablation.
        if not graph_present:
            graph_ok = True
        elif allow_ignored_input_graph:
            try:
                ignored_pair_count_ok = int(resolved_pairs) > 0
                if spec.get("strict_lr_pairs") is not None:
                    ignored_pair_count_ok = int(resolved_pairs) == int(
                        spec["strict_lr_pairs"]
                    )
            except (TypeError, ValueError):
                ignored_pair_count_ok = False
            graph_ok = (
                graph_path is not None
                and graph_path.name == expected_name
                and graph_digest == expected_digest
                and graph.get("lr_matching_rule")
                == "selected_symbol_exact_case_insensitive_all_complex_subunits"
                and graph.get("lr_complex_expression_rule") == "minimum"
                and str(graph.get("preferred_species_tag", "")) == expected_species
                and ignored_pair_count_ok
            )
        else:
            graph_ok = False
    else:
        graph_ok = False

    ok = graph_ok and downstream_ok
    return ok, (
        f"mode={mode!r}, graph_present={graph_present}, "
        f"allow_ignored_input_graph={allow_ignored_input_graph}, "
        f"graph_database={None if graph_path is None else graph_path.name!r}, "
        f"graph_sha256={graph_digest}, graph_pairs={resolved_pairs!r}, "
        f"downstream_database={None if downstream_path is None else downstream_path.name!r}, "
        f"downstream_sha256={downstream_digest}, expected={expected_name!r}/"
        f"{expected_digest}"
    )


def trained_edge_prior_contract(
    all_model: Any,
    interaction_graph: Any,
    *,
    expected_mode: str,
    expected_threshold: float | None,
    expected_edge_path: Path | None,
    predictor_metadata: Any = None,
    interaction_graph_present: bool | None = None,
) -> tuple[bool, str]:
    """Validate the edge-prior provenance serialized in a trained AnnData.

    A learned prior must retain one internally consistent predictor identity in
    ``all_model``, the aligned interaction-graph metadata, and the predictor
    sidecar.  A radius-only model must serialize no inert predictor values and
    must not carry a stale ``interaction_graph`` into the trained artifact.
    """

    model_meta = _mapping(all_model)
    graph_meta = _mapping(interaction_graph)
    predictor_meta = _mapping(predictor_metadata)
    graph_present = bool(
        interaction_graph is not None
        or (interaction_graph_present is not None and bool(interaction_graph_present))
    )
    mode = str(model_meta.get("edge_prior_mode", "")).strip().lower()
    recorded_path = _resolved_optional_path(model_meta.get("edge_predictor_path"))
    recorded_threshold = _finite_probability(model_meta.get("edge_predictor_threshold"))
    detail = (
        f"mode={mode!r}, path={recorded_path}, "
        f"threshold={model_meta.get('edge_predictor_threshold')!r}, "
        f"interaction_graph_present={graph_present}"
    )

    if expected_mode == "all_spatial":
        clean = (
            mode == "all_spatial"
            and model_meta.get("edge_predictor_path") is None
            and model_meta.get("edge_predictor_threshold") is None
            and not graph_present
        )
        return clean, detail

    if expected_mode != "learned":
        return False, f"unsupported expected mode={expected_mode!r}; {detail}"

    expected_path = (
        None
        if expected_edge_path is None
        else expected_edge_path.expanduser().resolve()
    )
    expected_probability = _finite_probability(expected_threshold)
    sidecar_threshold = _finite_probability(
        predictor_meta.get("edge_predictor_threshold")
    )
    sidecar_selected = _finite_probability(
        predictor_meta.get("edge_predictor_threshold_selected")
    )
    graph_path = _resolved_optional_path(graph_meta.get("edge_predictor_path"))
    graph_model_path = _resolved_optional_path(
        graph_meta.get("edge_predictor_model_path")
    )
    graph_threshold = _finite_probability(graph_meta.get("edge_predictor_threshold"))
    graph_selected = _finite_probability(
        graph_meta.get("edge_predictor_threshold_selected")
    )
    learned_ok = (
        mode == "learned"
        and expected_path is not None
        and expected_path.is_file()
        and graph_present
        and recorded_path == expected_path
        and graph_path == expected_path
        and graph_model_path == expected_path
        and expected_probability is not None
        and recorded_threshold is not None
        and sidecar_threshold is not None
        and sidecar_selected is not None
        and graph_threshold is not None
        and graph_selected is not None
        and close_enough(recorded_threshold, expected_probability)
        and close_enough(sidecar_threshold, expected_probability)
        and close_enough(sidecar_selected, expected_probability)
        and close_enough(graph_threshold, expected_probability)
        and close_enough(graph_selected, expected_probability)
    )
    return learned_ok, detail


def classifier_split_contract(
    split: Any,
    *,
    expected_class_counts: Mapping[str, int] | None = None,
) -> tuple[bool, str]:
    """Validate strict class support in the cached classifier holdout."""

    split_meta = _mapping(split)
    per_class = _mapping(split_meta.get("per_class_counts"))
    listed_raw = split_meta.get("training_only_singleton_classes", [])
    listed = (
        [str(value) for value in listed_raw] if isinstance(listed_raw, list) else []
    )
    singleton_policy = split_meta.get("singleton_class_policy")
    expected_singletons: list[str] = []
    class_errors: list[str] = []
    recorded_totals: dict[str, int] = {}
    for class_name, raw_counts in per_class.items():
        counts = _mapping(raw_counts)
        try:
            total = int(counts["total"])
            train = int(counts["train"])
            validation = int(counts["validation"])
        except (KeyError, TypeError, ValueError):
            class_errors.append(f"{class_name}:malformed")
            continue
        recorded_totals[str(class_name)] = total
        if min(total, train, validation) < 0 or train + validation != total:
            class_errors.append(
                f"{class_name}:total={total},train={train},validation={validation}"
            )
        elif total == 1:
            expected_singletons.append(str(class_name))
            if (train, validation) != (1, 0):
                class_errors.append(
                    f"{class_name}:singleton train={train},validation={validation}"
                )
        elif total >= 2 and (train <= 0 or validation <= 0):
            class_errors.append(
                f"{class_name}:missing support train={train},validation={validation}"
            )
        elif total == 0:
            class_errors.append(f"{class_name}:empty class")

    singleton_listing_ok = len(listed) == len(set(listed)) and set(listed) == set(
        expected_singletons
    )
    expected_totals = (
        None
        if expected_class_counts is None
        else {str(key): int(value) for key, value in expected_class_counts.items()}
    )
    cohort_matches = expected_totals is None or recorded_totals == expected_totals
    ok = (
        split_meta.get("cache_protocol_version") == 8
        and bool(per_class)
        and not class_errors
        and singleton_listing_ok
        and cohort_matches
        and isinstance(singleton_policy, str)
        and bool(singleton_policy.strip())
    )
    detail = (
        f"protocol={split_meta.get('cache_protocol_version')!r}, "
        f"classes={len(per_class)}, listed_singletons={listed}, "
        f"expected_singletons={expected_singletons}, errors={class_errors}"
        f", cohort_matches={cohort_matches}, recorded_total={sum(recorded_totals.values())}, "
        f"expected_total={None if expected_totals is None else sum(expected_totals.values())}"
    )
    return ok, detail


def _time_count_mapping(value: Any) -> dict[float, int] | None:
    mapping = _mapping(value)
    normalized: dict[float, int] = {}
    try:
        for key, count in mapping.items():
            time_value = float(key)
            normalized_count = int(count)
            if not math.isfinite(time_value) or normalized_count <= 0:
                return None
            if time_value in normalized:
                return None
            normalized[time_value] = normalized_count
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized


def _time_text_mapping(value: Any) -> dict[float, str] | None:
    mapping = _mapping(value)
    normalized: dict[float, str] = {}
    try:
        for key, text in mapping.items():
            time_value = float(key)
            normalized_text = str(text)
            if (
                not math.isfinite(time_value)
                or not normalized_text
                or time_value in normalized
            ):
                return None
            normalized[time_value] = normalized_text
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized


def _time_float_mapping(value: Any) -> dict[float, float] | None:
    mapping = _mapping(value)
    normalized: dict[float, float] = {}
    try:
        for key, raw_value in mapping.items():
            time_value = float(key)
            normalized_value = float(raw_value)
            if (
                not math.isfinite(time_value)
                or not math.isfinite(normalized_value)
                or time_value in normalized
            ):
                return None
            normalized[time_value] = normalized_value
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized


def _zebrafish_scope_contract(scope: Any) -> bool:
    """Require language that rules out endpoint-conditioned/global trajectory claims."""

    normalized = " ".join(str(scope).lower().replace("_", "-").split())
    observed_anchor = "observed" in normalized and "anchor" in normalized
    interval_local = "interval-local" in normalized or "interval local" in normalized
    one_sided = "one-sided" in normalized or "one sided" in normalized
    following_endpoint = (
        "not conditioned on the following" in normalized and "endpoint" in normalized
    )
    not_global = (
        "not global-t0" in normalized
        or "not a global-t0" in normalized
        or "not global t0" in normalized
        or "not a global t0" in normalized
        or (
            "not a single lineage-continuous" in normalized
            and "or global-t0" in normalized
        )
    )
    not_lineage = (
        "not lineage-continuous" in normalized
        or "not a lineage-continuous" in normalized
        or "not a single lineage-continuous" in normalized
        or "not lineage continuous" in normalized
        or "not a lineage continuous" in normalized
        or "not a single lineage continuous" in normalized
    )
    return all(
        (
            observed_anchor,
            interval_local,
            one_sided,
            following_endpoint,
            not_global,
            not_lineage,
        )
    )


def slice_provenance_summary_contract(
    simulation: Any,
    *,
    slice_provenance: Mapping[float, Mapping[str, Any]],
    expected_times: Sequence[float],
) -> tuple[bool, str]:
    """Require the summary's origin/anchor maps to equal the emitted slices."""

    simulation_meta = _mapping(simulation)
    actual_origins = {
        float(time): str(provenance.get("origin"))
        for time, provenance in slice_provenance.items()
    }
    try:
        actual_anchors: dict[float, float] | None = {
            float(time): float(provenance.get("anchor_time"))
            for time, provenance in slice_provenance.items()
        }
    except (TypeError, ValueError, OverflowError):
        actual_anchors = None
    declared_origins = _time_text_mapping(simulation_meta.get("slice_origins_by_time"))
    declared_anchors = _time_float_mapping(
        simulation_meta.get("source_anchor_times_by_time")
    )
    expected = {float(time) for time in expected_times}
    ok = (
        declared_origins == actual_origins
        and declared_anchors == actual_anchors
        and set(actual_origins) == expected
    )
    return ok, (
        f"origins_match={declared_origins == actual_origins}, "
        f"anchors_match={declared_anchors == actual_anchors}, "
        f"times_match={set(actual_origins) == expected}"
    )


def zebrafish_split_sde_contract(
    simulation: Any,
    *,
    slice_counts: Mapping[float, int],
    expected_observed_counts: Mapping[float, int],
    analyses: Any,
) -> tuple[bool, str]:
    simulation_meta = _mapping(simulation)
    resample_dt = simulation_meta.get("split_resample_dt")
    ceiling = simulation_meta.get("split_particle_ceiling")
    trajectory_mode = simulation_meta.get("trajectory_mode")
    sample_mode = simulation_meta.get("piecewise_observed_sample_mode")
    include_end = simulation_meta.get("piecewise_include_end")
    scope = str(simulation_meta.get("trajectory_scope", ""))
    actual_counts = {float(time): int(count) for time, count in slice_counts.items()}
    expected_observed = {
        float(time): int(count) for time, count in expected_observed_counts.items()
    }
    actual_observed = {time: actual_counts.get(time) for time in expected_observed}
    actual_generated = {
        time: count
        for time, count in actual_counts.items()
        if time not in expected_observed
    }
    declared_all = _time_count_mapping(simulation_meta.get("particle_counts_by_time"))
    declared_observed = _time_count_mapping(
        simulation_meta.get("observed_particle_counts")
    )
    declared_generated = _time_count_mapping(
        simulation_meta.get("generated_particle_counts")
    )
    analysis_meta = _mapping(analyses)
    communication_scope = _mapping(analysis_meta.get("communication")).get(
        "trajectory_scope"
    )
    lr_scope = _mapping(analysis_meta.get("ligand_receptor")).get("trajectory_scope")
    reconstruction = _mapping(analysis_meta.get("reconstruction_diagnostic"))
    reconstruction_claim = str(reconstruction.get("claim", ""))
    reconstruction_claim_normalized = " ".join(
        reconstruction_claim.lower().replace("_", "-").split()
    )
    try:
        normalized_ceiling = int(ceiling)
        bounded_generated = bool(actual_generated) and all(
            0 < int(count) <= normalized_ceiling for count in actual_generated.values()
        )
        ok = (
            close_enough(simulation_meta.get("split_dt"), 0.05)
            and close_enough(resample_dt, 0.05)
            and close_enough(simulation_meta.get("sigma"), 0.03)
            and close_enough(simulation_meta.get("growth_alpha"), 1.0)
            and normalized_ceiling == 100_000
            and simulation_meta.get("configured_particle_cap") is None
            and simulation_meta.get("initial_particle_cap") is None
            and int(simulation_meta.get("initial_particles")) == 563
            and simulation_meta.get("non_split_lineage_rollout") is False
            and trajectory_mode
            == "piecewise_observed_anchored_interval_forward_simulation"
            and simulation_meta.get("split_sde_piecewise") is True
            and sample_mode == "per_timepoint"
            and include_end is False
            and _zebrafish_scope_contract(scope)
            and communication_scope == scope
            and lr_scope == scope
            and "global-t0" in reconstruction_claim_normalized
            and "not applicable" in reconstruction_claim_normalized
            and actual_observed == expected_observed
            and bounded_generated
            and declared_all == actual_counts
            and declared_observed == expected_observed
            and declared_generated == actual_generated
        )
    except (TypeError, ValueError, OverflowError):
        ok = False
    return (
        ok,
        f"trajectory_mode={trajectory_mode!r}, sample_mode={sample_mode!r}, "
        f"include_end={include_end!r}, split_dt={simulation_meta.get('split_dt')!r}, "
        f"split_resample_dt={resample_dt!r}, sigma={simulation_meta.get('sigma')!r}, "
        f"growth_alpha={simulation_meta.get('growth_alpha')!r}, "
        f"split_particle_ceiling={ceiling!r}, observed={actual_observed}, "
        f"generated={actual_generated}, declared_matches={declared_all == actual_counts}, "
        f"scope_contract={_zebrafish_scope_contract(scope)}, "
        f"analysis_scopes_match={communication_scope == scope and lr_scope == scope}, "
        f"reconstruction_claim={reconstruction_claim!r}",
    )


def retained_top_level_lr_pairs(frame: pd.DataFrame) -> tuple[int | None, str]:
    """Count retained pair trajectories, excluding pair-cell-type children."""

    required = {"trajectory_kind", "retained", "pair_id"}
    if not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        return None, f"missing_columns={missing}"
    retained = frame["retained"].astype(str).str.strip().str.lower().eq("true")
    top_level = frame["trajectory_kind"].astype(str).eq("pair")
    rows = frame.loc[retained & top_level, "pair_id"].astype(str)
    unique = int(rows.nunique())
    if unique != len(rows):
        return None, f"retained_pair_rows={len(rows)}, unique_pair_ids={unique}"
    return unique, f"retained_pair_rows={len(rows)}, unique_pair_ids={unique}"


def complete_lr_pair_time_grid(
    trajectory_coverage: pd.DataFrame,
    pair_timecourse: pd.DataFrame,
    *,
    expected_times: Sequence[float],
) -> tuple[bool, str]:
    """Require identical retained pair IDs and one row per pair and time."""

    coverage_columns = {"trajectory_kind", "retained", "pair_id"}
    timecourse_columns = {"pair_id", "time"}
    if not coverage_columns.issubset(trajectory_coverage.columns) or not (
        timecourse_columns.issubset(pair_timecourse.columns)
    ):
        return False, "missing pair identity or time columns"
    retained_mask = (
        trajectory_coverage["retained"].astype(str).str.strip().str.lower().eq("true")
    )
    top_level_mask = trajectory_coverage["trajectory_kind"].astype(str).eq("pair")
    retained_ids = set(
        trajectory_coverage.loc[retained_mask & top_level_mask, "pair_id"].astype(str)
    )
    pair_ids = pair_timecourse["pair_id"].astype(str)
    times = pd.to_numeric(pair_timecourse["time"], errors="coerce")
    keys = list(zip(pair_ids, times))
    expected_keys = {
        (pair_id, float(time)) for pair_id in retained_ids for time in expected_times
    }
    actual_keys = {(pair_id, float(time)) for pair_id, time in keys if pd.notna(time)}
    pair_id_match = retained_ids == set(pair_ids)
    unique_rows = len(keys) == len(actual_keys)
    complete = pair_id_match and unique_rows and actual_keys == expected_keys
    return complete, (
        f"retained_ids={len(retained_ids)}, timecourse_ids={pair_ids.nunique()}, "
        f"rows={len(keys)}, expected_rows={len(expected_keys)}, "
        f"pair_id_match={pair_id_match}, unique_rows={unique_rows}"
    )


def downstream_scope_contract(
    summary: Any,
    *,
    expected_edge_mode: str,
    expected_ad_lr_pairs: int | None = None,
) -> tuple[bool, str]:
    """Check that human-readable communication and LR scope matches the model."""

    summary_meta = _mapping(summary)
    analyses = _mapping(summary_meta.get("analyses"))
    communication = _mapping(analyses.get("communication"))
    communication_interpretation = communication.get("interpretation")
    training_scope = communication.get("training_interaction_scope")
    attention_scope = communication.get("attention_scope")
    communication_ok = (
        str(communication.get("edge_prior_mode", "")).lower() == expected_edge_mode
        and attention_scope == "full time-slice radius candidate graph"
        and isinstance(training_scope, str)
        and "stochastic interaction groups" in training_scope
        and "only within each group" in training_scope
        and "base size 1024" in training_scope
        and "at most 2047" in training_scope
    )

    lr_scope = _mapping(_mapping(analyses.get("ligand_receptor")).get("analysis_scope"))
    lr_ok = True
    if expected_ad_lr_pairs is not None:
        lr_interpretation = lr_scope.get("interpretation")
        communication_text = (
            communication_interpretation.lower()
            if isinstance(communication_interpretation, str)
            else ""
        )
        lr_text = (
            lr_interpretation.lower() if isinstance(lr_interpretation, str) else ""
        )
        communication_ok = communication_ok and (
            "within-cutoff" in communication_text
            and "global cell-cell communication" in communication_text
        )
        if expected_edge_mode == "learned":
            communication_ok = (
                communication_ok
                and "learned-predictor-gated" in communication_text
                and "trained from seven" in communication_text
                and "no-lr-prior ablation" not in communication_text
            )
        elif expected_edge_mode == "all_spatial":
            communication_ok = (
                communication_ok
                and "without a learned ligand-receptor edge gate" in communication_text
                and "no-lr-prior ablation" in communication_text
                and "not the production main model" in communication_text
                and "trained from seven" not in communication_text
            )
        else:
            communication_ok = False
        try:
            pair_count_ok = (
                int(lr_scope.get("strict_supported_pair_count")) == expected_ad_lr_pairs
            )
        except (TypeError, ValueError):
            pair_count_ok = False
        lr_ok = (
            pair_count_ok
            and ("seven" in lr_text or "7" in lr_text)
            and "expression panel" in lr_text
            and "global cci screen" in lr_text
        )
        if expected_edge_mode == "learned":
            lr_ok = lr_ok and "not a global cci screen" in lr_text
        elif expected_edge_mode == "all_spatial":
            lr_ok = lr_ok and "did not gate the all-spatial ablation model" in lr_text
    ok = communication_ok and lr_ok
    detail = (
        f"communication_mode={communication.get('edge_prior_mode')!r}, "
        f"attention_scope={attention_scope!r}, training_scope={training_scope!r}, "
        f"lr_scope={lr_scope}"
    )
    return ok, detail


def required_files(
    run_dir: Path,
    dataset: str,
    spec: dict[str, Any],
) -> dict[str, Path]:
    artifact_dataset = str(spec.get("artifact_dataset", dataset))
    training = run_dir / "training"
    downstream = run_dir / "downstream"
    paths = {
        "aligned H5AD": run_dir / "preprocess" / f"{artifact_dataset}_aligned.h5ad",
        "resolved training config": training / "config.yaml",
        "training history": training / "training_history.csv",
        "training run summary": training / "training_run_summary.json",
        "trained AnnData": training / "adata.h5ad",
        "downstream summary": downstream / "summary.json",
    }
    if spec["edge_prior_mode"] == "learned":
        edge_model = (
            run_dir
            / "preprocess"
            / "edge_classifier"
            / f"{artifact_dataset}_edge_model.pt"
        )
        paths["generated edge model"] = edge_model
        paths["generated edge metadata"] = edge_model.with_suffix(
            edge_model.suffix + ".meta.json"
        )
    return paths


def validate_aligned(
    audit: Audit,
    paths: dict[str, Path],
    spec: dict[str, Any],
) -> tuple[int, int, dict[str, int]]:
    aligned_path = paths["aligned H5AD"]
    shape, obs, obs_names, uns = read_metadata(aligned_path)
    audit.check(
        tuple(shape) == tuple(spec["shape"]), "analyzed cohort", f"shape={shape}"
    )
    audit.check(
        bool(obs_names.is_unique),
        "stable observation identities",
        f"unique={obs_names.is_unique}",
    )

    time_values = pd.to_numeric(obs["time_point_processed"], errors="raise").astype(
        float
    )
    counts = {
        float(key): int(value)
        for key, value in time_values.value_counts().sort_index().items()
    }
    audit.check(
        counts == spec["observed_counts"],
        "observed slice membership",
        f"counts={counts}",
    )
    annotation = obs[spec["annotation_key"]]
    valid_annotation = bool(
        annotation.notna().all() and annotation.astype(str).str.len().gt(0).all()
    )
    audit.check(
        valid_annotation, "cell annotations", f"column={spec['annotation_key']!r}"
    )
    annotation_counts = {
        str(key): int(value)
        for key, value in annotation.astype(str).value_counts().items()
    }

    preprocess_info = uns.get("preprocess_info", {})
    raw_layer = str(preprocess_info.get("raw_counts_layer", ""))
    audit.check(
        raw_layer == spec["counts_layer"],
        "raw expression layer",
        f"recorded={raw_layer!r}, expected={spec['counts_layer']!r}",
    )

    with h5py.File(aligned_path, "r") as handle:
        latent = handle["obsm/X_latent"]
        spatial = handle["obsm/spatial_aligned"]
        pcs = handle["varm/PCs"]
        center = handle["var/pca_center"]
        counts_node = handle[f"layers/{spec['counts_layer']}"]
        latent_stats = numeric_stats(latent)
        spatial_stats = numeric_stats(spatial)
        pcs_stats = numeric_stats(pcs)
        center_stats = numeric_stats(center)
        counts_stats = numeric_stats(counts_node)
        audit.check(
            h5_shape(latent) == (shape[0], 50) and latent_stats[0],
            "aligned latent state",
            f"shape={h5_shape(latent)}, range=({latent_stats[2]:.6g}, {latent_stats[3]:.6g})",
        )
        audit.check(
            h5_shape(spatial) == (shape[0], 2) and spatial_stats[0],
            "aligned spatial state",
            f"shape={h5_shape(spatial)}, range=({spatial_stats[2]:.6g}, {spatial_stats[3]:.6g})",
        )
        audit.check(
            h5_shape(pcs) == (shape[1], 50) and pcs_stats[0] and center_stats[0],
            "retained PCA transform",
            f"PCs={h5_shape(pcs)}, center={h5_shape(center)}",
        )
        audit.check(
            counts_stats[0] and counts_stats[1] > 0 and counts_stats[2] >= 0,
            "nonnegative finite raw counts",
            f"stored_values={counts_stats[1]}, range=({counts_stats[2]:.6g}, {counts_stats[3]:.6g})",
        )
    return int(shape[0]), 52, annotation_counts


def validate_edge_predictor(
    audit: Audit,
    run_dir: Path,
    paths: dict[str, Path],
    spec: dict[str, Any],
) -> float:
    meta = json.loads(paths["generated edge metadata"].read_text(encoding="utf-8"))
    threshold = float(meta["edge_predictor_threshold"])
    threshold_ok, threshold_detail = edge_threshold_provenance_contract(
        meta,
        expected_threshold=float(spec["edge_predictor_threshold"]),
    )
    audit.check(
        threshold_ok,
        "validation-selected corrected production edge threshold",
        threshold_detail,
    )
    audit.check(
        0.0 <= threshold <= 1.0
        and close_enough(meta["distance_threshold"], spec["cutoff"]),
        "edge distance and decision thresholds",
        f"cutoff={float(meta['distance_threshold']):.17g}, decision={threshold:.9g}",
    )
    audit.check(
        int(meta.get("random_seed", -1)) == 42
        and meta.get("split", {}).get("strategy") == "node_disjoint_holdout",
        "edge predictor split and seed",
        f"split={meta.get('split', {}).get('strategy')!r}, seed={meta.get('random_seed')}",
    )
    universe = meta["candidate_universe"]
    candidate_ok = (
        universe.get("definition") == "all directed pairs with 1e-6 < distance < cutoff"
        and int(universe.get("positive_edges", 0)) > 0
        and int(universe.get("negative_edges", 0)) > 0
        and int(universe.get("training_balanced_edges", 0)) > 0
    )
    audit.check(
        candidate_ok,
        "edge candidate universe",
        f"positive={universe.get('positive_edges')}, negative={universe.get('negative_edges')}",
    )
    expected_times = tuple(spec["observed_counts"])
    meta_times = tuple(
        float(value) for value in meta["split"]["time_values_by_local_index"]
    )
    graph_files = sorted(
        (run_dir / "preprocess" / "input_graph").glob("*/*_adjacency_records")
    )
    audit.check(
        meta_times == expected_times and len(graph_files) == len(expected_times),
        "per-time LR interaction graphs",
        f"times={meta_times}, graph_files={len(graph_files)}",
    )
    validation = meta["validation_metrics_at_selected_threshold"]
    test = meta["test_metrics_at_validation_threshold"]
    quality = (
        f"validation AP={validation.get('average_precision')}, F1={validation.get('f1')}; "
        f"test AP={test.get('average_precision')}, F1={test.get('f1')}"
    )
    audit.check(
        int(validation.get("n_candidates", 0)) > 0
        and int(test.get("n_candidates", 0)) > 0,
        "natural-prevalence edge holdouts",
        quality,
    )
    if float(validation.get("f1", 0.0)) < 0.1:
        audit.warn(
            "Low edge-predictor validation F1 under natural prevalence: " + quality
        )
    return threshold


def validate_training(
    audit: Audit,
    paths: dict[str, Path],
    spec: dict[str, Any],
    threshold: float | None,
    model_dim: int,
) -> None:
    config = yaml.safe_load(
        paths["resolved training config"].read_text(encoding="utf-8")
    )
    defaults = config["training"]["defaults"]
    interaction = config["model"]["interaction_net"]
    audit.check(
        int(config["seed"]) == 42
        and close_enough(defaults["alpha_express"], 0.015)
        and close_enough(defaults["alpha_spatial"], 10.0),
        "shared training constants",
        f"seed={config['seed']}, alpha_express={defaults['alpha_express']}, alpha_spatial={defaults['alpha_spatial']}",
    )
    actual_edge_prior = str(interaction.get("edge_prior_mode", "learned")).lower()
    if spec["edge_prior_mode"] == "learned":
        edge_path = paths["generated edge model"].resolve()
        configured_edge_path = Path(interaction["edge_predictor_path"]).resolve()
        edge_prior_ok = (
            threshold is not None
            and actual_edge_prior == "learned"
            and configured_edge_path == edge_path
            and close_enough(interaction["edge_predictor_thre"], threshold)
            and close_enough(interaction["cutoff"], spec["cutoff"])
        )
        edge_prior_detail = (
            f"mode={actual_edge_prior}, path={configured_edge_path}, "
            f"threshold={interaction['edge_predictor_thre']}"
        )
    else:
        edge_prior_ok = (
            threshold is None
            and actual_edge_prior == "all_spatial"
            and close_enough(interaction["cutoff"], spec["cutoff"])
            and "edge_predictor_path" not in interaction
            and "edge_predictor_thre" not in interaction
        )
        edge_prior_detail = (
            f"mode={actual_edge_prior}, cutoff={interaction.get('cutoff')}, "
            f"predictor_path={interaction.get('edge_predictor_path')}, "
            f"predictor_threshold={interaction.get('edge_predictor_thre')}"
        )
    audit.check(
        edge_prior_ok,
        "declared edge prior wired into training",
        edge_prior_detail,
    )

    plan = config["training"]["plan"]
    plan_names = tuple(str(stage["name"]) for stage in plan)
    epochs = expected_epochs(spec)
    audit.check(plan_names == STAGES, "six-stage training plan", f"stages={plan_names}")
    plan_epochs = {str(stage["name"]): int(stage["epochs"]) for stage in plan}
    audit.check(
        plan_epochs == epochs, "configured stage lengths", f"epochs={plan_epochs}"
    )

    history = pd.read_csv(paths["training history"])
    history_names = tuple(history["stage"].drop_duplicates().astype(str))
    complete = history_names == STAGES
    details = []
    for stage_name, stage_epochs in epochs.items():
        stage_rows = history.loc[history["stage"] == stage_name]
        epoch_values = stage_rows["epoch"].astype(int).to_numpy()
        selected = (
            stage_rows["is_selected_checkpoint"].astype(str).str.lower().eq("true")
        )
        stage_complete = (
            len(stage_rows) == stage_epochs
            and np.array_equal(epoch_values, np.arange(1, stage_epochs + 1))
            and int(selected.sum()) == 1
        )
        complete = complete and stage_complete
        details.append(
            f"{stage_name}:{len(stage_rows)}/{stage_epochs},selected={int(selected.sum())}"
        )
    core_numeric = history[["loss", "checkpoint_value", "learning_rate"]].to_numpy(
        dtype=float
    )
    complete = complete and bool(np.isfinite(core_numeric).all())
    audit.check(complete, "complete finite training history", "; ".join(details))

    for stage in plan:
        name = str(stage["name"])
        is_score = str(stage.get("mode", "")).lower() == "score_matching"
        strategy = str(
            stage.get("save_strategy", defaults.get("save_strategy", "best"))
        )
        filename = "score_model.pth" if is_score else f"{strategy}_model.pth"
        checkpoint = paths["resolved training config"].parent / name / filename
        audit.check(
            checkpoint.is_file() and checkpoint.stat().st_size > 0,
            f"{name} checkpoint",
            str(checkpoint),
        )

    run_summary = json.loads(paths["training run summary"].read_text(encoding="utf-8"))
    stage_summaries = run_summary["stages"]
    summary_complete = tuple(item["stage"] for item in stage_summaries) == STAGES
    summary_complete = summary_complete and all(
        int(item["recorded_epochs"]) == epochs[item["stage"]]
        and int(item["configured_epochs"]) == epochs[item["stage"]]
        and item["selected_checkpoint_epoch"] is not None
        for item in stage_summaries
    )
    audit.check(
        summary_complete and float(run_summary["timing"]["run_wall_time_seconds"]) > 0,
        "measured training run summary",
        f"wall_seconds={run_summary['timing']['run_wall_time_seconds']:.3f}",
    )

    trained_shape, _, _, trained_uns = read_metadata(paths["trained AnnData"])
    audit.check(
        tuple(trained_shape) == tuple(spec["shape"]),
        "trained AnnData cohort",
        f"shape={trained_shape}",
    )
    predictor_metadata = None
    expected_edge_path = None
    if spec["edge_prior_mode"] == "learned":
        expected_edge_path = paths["generated edge model"]
        predictor_metadata = json.loads(
            paths["generated edge metadata"].read_text(encoding="utf-8")
        )
    edge_artifact_ok, edge_artifact_detail = trained_edge_prior_contract(
        trained_uns.get("all_model"),
        trained_uns.get("interaction_graph"),
        expected_mode=spec["edge_prior_mode"],
        expected_threshold=threshold,
        expected_edge_path=expected_edge_path,
        predictor_metadata=predictor_metadata,
        interaction_graph_present="interaction_graph" in trained_uns,
    )
    audit.check(
        edge_artifact_ok,
        "trained AnnData edge-prior provenance",
        edge_artifact_detail,
    )
    if spec["edge_prior_mode"] == "all_spatial":
        all_model = _mapping(trained_uns.get("all_model"))
        audit.check(
            all_model.get("ignored_input_interaction_graph_metadata") is True,
            "matched no-LR input graph metadata was explicitly ignored",
            "ignored_input_interaction_graph_metadata="
            f"{all_model.get('ignored_input_interaction_graph_metadata')!r}",
        )
    with h5py.File(paths["trained AnnData"], "r") as handle:
        expected_vectors = (
            "obsm/velocity_model",
            "obsm/interaction_model",
            "obsm/score_gradient_model",
            "obsm/full_drift_model",
            "obsm/growth_rate",
        )
        vectors_ok = all(key in handle for key in expected_vectors)
        vector_details = []
        if vectors_ok:
            for key in expected_vectors:
                stats = numeric_stats(handle[key])
                vectors_ok = vectors_ok and stats[0] and stats[1] > 0
                vector_details.append(
                    f"{key.rsplit('/', 1)[-1]}={h5_shape(handle[key])}"
                )
            vectors_ok = vectors_ok and h5_shape(handle["obsm/full_drift_model"]) == (
                trained_shape[0],
                model_dim,
            )
            vectors_ok = vectors_ok and h5_sum_identity(
                handle,
                "obsm/full_drift_model",
                (
                    "obsm/velocity_model",
                    "obsm/interaction_model",
                    "obsm/score_gradient_model",
                ),
            )
        audit.check(
            vectors_ok, "finite fitted vector components", ", ".join(vector_details)
        )

    import CytoBridge as cb
    from CytoBridge.workflow import (
        WorkflowOptions,
        _loaded_model_scientific_contract,
        load_workflow_config,
    )

    load_options = {}
    if spec["edge_prior_mode"] == "learned":
        load_options["edge_predictor_path"] = paths["generated edge model"]
    loaded = cb.tl.load_dynamical_model_from_dir(
        paths["resolved training config"].parent,
        dim=model_dim,
        device="cpu",
        **load_options,
    )
    preset_name = str(spec.get("artifact_dataset", audit.dataset))
    preset, _ = load_workflow_config(preset_name)
    training_config = spec.get("training_config")
    contract = _loaded_model_scientific_contract(
        loaded,
        config=preset,
        options=WorkflowOptions(train=True, training_config=training_config),
    )
    parameters_finite = all(
        bool(np.isfinite(value.detach().cpu().numpy()).all())
        for value in loaded.model.state_dict().values()
    )
    audit.check(
        loaded.weight_stage == "Finetune"
        and loaded.score_stage == "Score_Refine"
        and parameters_finite
        and contract["status"] == "matches requested preset",
        "strict final checkpoint load",
        f"weight={loaded.weight_stage}, score={loaded.score_stage}, contract={contract['status']}",
    )


def validate_slice(
    audit: Audit,
    path: Path,
    annotation_key: str,
    model_dim: int,
    *,
    time_value: float,
    observed_time: bool,
    aligned: ad.AnnData | None = None,
    aligned_time_key: str = "time_point_processed",
) -> tuple[int, set[str], dict[str, Any]]:
    shape, obs, _, uns = read_metadata(path)
    annotations_ok = annotation_key in obs and obs[annotation_key].notna().all()
    with h5py.File(path, "r") as handle:
        state = handle["X"]
        spatial = handle["obsm/spatial"]
        state_stats = numeric_stats(state)
        spatial_stats = numeric_stats(spatial)
        consistent = h5_shape(state) == (shape[0], model_dim) and h5_shape(spatial) == (
            shape[0],
            2,
        )
        if consistent:
            state_values = numeric_node(state)
            spatial_values = numeric_node(spatial)
            for start in range(0, shape[0], 100_000):
                stop = start + 100_000
                if not np.allclose(
                    np.asarray(state_values[start:stop, :2]),
                    np.asarray(spatial_values[start:stop]),
                    rtol=0,
                    atol=0,
                ):
                    consistent = False
                    break
    audit.check(
        bool(
            shape[0] > 0
            and annotations_ok
            and state_stats[0]
            and spatial_stats[0]
            and consistent
        ),
        f"finite state {path.stem}",
        f"shape={shape}",
    )
    labels = set(obs[annotation_key].astype(str)) if annotation_key in obs else set()
    provenance = {
        "origin": uns.get("slice_origin"),
        "anchor_time": uns.get("source_anchor_time"),
    }
    if aligned is not None:
        if observed_time:
            aligned_times = pd.to_numeric(
                aligned.obs[aligned_time_key], errors="coerce"
            ).to_numpy(dtype=float)
            aligned_mask = np.isclose(
                aligned_times, float(time_value), rtol=0, atol=1e-12
            )
            aligned_subset = aligned[aligned_mask]
            expected_ids = aligned_subset.obs_names.astype(str).to_numpy()
            source_ids = (
                obs["source_obs_id"].astype(str).to_numpy()
                if "source_obs_id" in obs
                else np.asarray([], dtype=str)
            )
            expected_annotations = (
                aligned_subset.obs[annotation_key].astype(str).to_numpy()
            )
            actual_annotations = (
                obs[annotation_key].astype(str).to_numpy()
                if annotation_key in obs
                else np.asarray([], dtype=str)
            )
            expected_state = np.hstack(
                (
                    np.asarray(aligned_subset.obsm["spatial_aligned"]),
                    np.asarray(aligned_subset.obsm["X_latent"]),
                )
            )
            with h5py.File(path, "r") as handle:
                actual_state = np.asarray(numeric_node(handle["X"]))
            try:
                anchor_matches = close_enough(provenance["anchor_time"], time_value)
            except (TypeError, ValueError, OverflowError):
                anchor_matches = False
            observed_provenance_ok = (
                provenance["origin"] == "observed_real"
                and anchor_matches
                and "source_obs_id" in obs
                and np.array_equal(source_ids, expected_ids)
                and np.array_equal(actual_annotations, expected_annotations)
                and actual_state.shape == expected_state.shape
                and np.allclose(
                    actual_state,
                    expected_state,
                    rtol=0,
                    atol=1e-6,
                )
            )
            audit.check(
                observed_provenance_ok,
                f"exact observed slice provenance t={time_value:g}",
                f"origin={provenance['origin']!r}, anchor={provenance['anchor_time']!r}, "
                f"source_ids={len(source_ids)}, expected={len(expected_ids)}",
            )
        else:
            earlier_anchors = [
                float(value)
                for value in DATASETS["zebrafish"]["observed_counts"]
                if float(value) < float(time_value)
            ]
            expected_anchor = max(earlier_anchors) if earlier_anchors else None
            aligned_labels = set(aligned.obs[annotation_key].astype(str))
            source_ids_absent = "source_obs_id" not in obs or bool(
                obs["source_obs_id"].isna().all()
            )
            try:
                anchor_matches = expected_anchor is not None and close_enough(
                    provenance["anchor_time"], expected_anchor
                )
            except (TypeError, ValueError, OverflowError):
                anchor_matches = False
            generated_provenance_ok = (
                provenance["origin"] == "generated_interval_local"
                and anchor_matches
                and source_ids_absent
                and annotations_ok
                and bool(labels)
                and labels.issubset(aligned_labels)
            )
            audit.check(
                generated_provenance_ok,
                f"interval-local generated slice provenance t={time_value:g}",
                f"origin={provenance['origin']!r}, anchor={provenance['anchor_time']!r}, "
                f"expected_anchor={expected_anchor!r}, labels={len(labels)}",
            )
    return int(shape[0]), labels, provenance


def valid_color_map(
    value: Any,
    *,
    required_labels: set[str],
) -> tuple[bool, str]:
    """Reject missing, incomplete, invalid, or single-color label palettes."""

    mapping = _mapping(value)
    colors = [str(mapping.get(label, "")) for label in sorted(required_labels)]
    valid_hex = all(
        len(color) == 7
        and color.startswith("#")
        and all(character in "0123456789abcdefABCDEF" for character in color[1:])
        for color in colors
    )
    distinct = len({color.lower() for color in colors})
    ok = (
        len(required_labels) > 1
        and set(required_labels).issubset(mapping)
        and valid_hex
        and distinct > 1
    )
    return ok, (
        f"required_labels={len(required_labels)}, mapped={len(mapping)}, "
        f"distinct_required_colors={distinct}"
    )


def finite_numeric_frame(frame: pd.DataFrame, *, allow_na: bool = False) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return False
    values = numeric.to_numpy(dtype=float)
    if allow_na:
        return bool(
            np.isfinite(values[~np.isnan(values)]).all() and np.isfinite(values).any()
        )
    return bool(np.isfinite(values).all())


def validate_downstream(
    audit: Audit,
    run_dir: Path,
    paths: dict[str, Path],
    spec: dict[str, Any],
    threshold: float | None,
    model_dim: int,
    aligned_class_counts: Mapping[str, int],
) -> None:
    downstream = run_dir / "downstream"
    summary = json.loads(paths["downstream summary"].read_text(encoding="utf-8"))
    observed = tuple(spec["observed_counts"])
    expected_times = tuple(sorted(set(observed + tuple(spec["interpolated"]))))
    actual_times = tuple(float(value) for value in summary["time_points"])
    audit.check(
        summary["dataset"] == str(spec.get("artifact_dataset", audit.dataset))
        and int(summary["seed"]) == 42
        and close_enough(summary["alpha_express"], 0.015)
        and int(summary["classifier_k"]) == int(spec["classifier_k"]),
        "downstream scientific constants",
        f"seed={summary['seed']}, alpha={summary['alpha_express']}, k={summary['classifier_k']}",
    )
    audit.check(
        actual_times == expected_times, "downstream time grid", f"times={actual_times}"
    )
    model_contract = summary["model"]["scientific_contract"]
    recorded_threshold = model_contract.get("edge_predictor_threshold")
    threshold_matches = (
        recorded_threshold is None
        if threshold is None
        else recorded_threshold is not None
        and close_enough(recorded_threshold, threshold)
    )
    try:
        group_size_matches = (
            int(model_contract.get("interaction_group_size")) == 1024
            and int(model_contract.get("interaction_group_max_size")) == 2047
            and int(summary.get("simulation", {}).get("interaction_group_size"))
            == int(model_contract.get("interaction_group_size"))
            and "remainder"
            in str(model_contract.get("interaction_group_remainder_policy", ""))
        )
    except (TypeError, ValueError):
        group_size_matches = False
    audit.check(
        model_contract["status"] == "matches requested preset"
        and close_enough(model_contract["interaction_cutoff"], spec["cutoff"])
        and model_contract.get("edge_prior_mode") == spec["edge_prior_mode"]
        and group_size_matches
        and threshold_matches
        and summary["model"]["weight_stage"] == "Finetune"
        and summary["model"]["score_stage"] == "Score_Refine",
        "downstream loaded corrected checkpoint",
        f"mode={model_contract.get('edge_prior_mode')}, cutoff={model_contract['interaction_cutoff']}, "
        f"group_size={model_contract.get('interaction_group_size')}, "
        f"simulation_group_size={summary.get('simulation', {}).get('interaction_group_size')}, "
        f"threshold={recorded_threshold}",
    )
    split_ok, split_detail = classifier_split_contract(
        summary.get("classifier_split"),
        expected_class_counts=aligned_class_counts,
    )
    audit.check(split_ok, "classifier holdout class support", split_detail)
    classifier_metrics = (
        float(summary["classifier_accuracy"]),
        float(summary["classifier_balanced_accuracy"]),
    )
    audit.check(
        all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in classifier_metrics
        ),
        "classifier metrics",
        f"accuracy={classifier_metrics[0]:.6g}, balanced_accuracy={classifier_metrics[1]:.6g}",
    )

    expected_analyses = (
        "velocity",
        "growth",
        "composition",
        "communication",
        "figures",
        "gene_dynamics",
        "ligand_receptor",
    )
    analyses = summary["analyses"]
    analysis_ok = all(
        analyses.get(name, {}).get("status") == "completed"
        for name in expected_analyses
    )
    audit.check(
        analysis_ok,
        "complete standard downstream analyses",
        ", ".join(
            f"{name}={analyses.get(name, {}).get('status')}"
            for name in expected_analyses
        ),
    )
    _, _, _, aligned_uns = read_metadata(paths["aligned H5AD"])
    trained_all_model = _mapping(
        read_metadata(paths["trained AnnData"])[3].get("all_model")
    )
    ignored_input_graph = (
        spec["edge_prior_mode"] == "all_spatial"
        and trained_all_model.get("ignored_input_interaction_graph_metadata") is True
    )
    database_ok, database_detail = lr_database_provenance_contract(
        graph_metadata=aligned_uns.get("interaction_graph"),
        graph_metadata_present="interaction_graph" in aligned_uns,
        downstream_analysis=analyses.get("ligand_receptor"),
        spec=spec,
        allow_ignored_input_graph=ignored_input_graph,
    )
    audit.check(
        database_ok,
        "species-matched LR database provenance and role",
        database_detail,
    )
    scope_ok, scope_detail = downstream_scope_contract(
        summary,
        expected_edge_mode=spec["edge_prior_mode"],
        expected_ad_lr_pairs=spec.get("strict_lr_pairs"),
    )
    audit.check(
        scope_ok,
        "readable downstream scientific scope",
        scope_detail,
    )

    slice_counts: dict[float, int] = {}
    slice_labels: dict[float, set[str]] = {}
    slice_provenance: dict[float, dict[str, Any]] = {}
    aligned_for_slice_provenance = (
        ad.read_h5ad(paths["aligned H5AD"], backed="r")
        if audit.dataset == "zebrafish"
        else None
    )
    for time_value in expected_times:
        path = downstream / "slice_data" / f"time_{safe_time_name(time_value)}.h5ad"
        if not path.is_file():
            audit.check(False, f"slice file t={time_value:g}", str(path))
            continue
        (
            slice_counts[time_value],
            slice_labels[time_value],
            slice_provenance[time_value],
        ) = validate_slice(
            audit,
            path,
            spec["annotation_key"],
            model_dim,
            time_value=time_value,
            observed_time=time_value in spec["observed_counts"],
            aligned=aligned_for_slice_provenance,
        )
    if aligned_for_slice_provenance is not None:
        aligned_for_slice_provenance.file.close()
    observed_slice_counts = {time: slice_counts.get(time) for time in observed}
    audit.check(
        observed_slice_counts == spec["observed_counts"],
        "observed states retain real cells",
        f"counts={observed_slice_counts}",
    )
    observed_labels = set().union(*(slice_labels.get(time, set()) for time in observed))
    generated_times = tuple(
        time for time in expected_times if time not in spec["observed_counts"]
    )
    generated_labels = set().union(
        *(slice_labels.get(time, set()) for time in generated_times)
    )
    generated_label_counts = {
        time: len(slice_labels.get(time, set())) for time in generated_times
    }
    audit.check(
        len(observed_labels) > 1
        and bool(generated_label_counts)
        and all(count > 1 for count in generated_label_counts.values()),
        "noncollapsed observed and generated labels",
        f"observed_labels={len(observed_labels)}, generated_per_slice={generated_label_counts}",
    )
    if audit.dataset == "zebrafish":
        provenance_ok, provenance_detail = slice_provenance_summary_contract(
            summary.get("simulation"),
            slice_provenance=slice_provenance,
            expected_times=expected_times,
        )
        audit.check(
            provenance_ok,
            "zebrafish summary matches per-slice provenance",
            provenance_detail,
        )
        split_sde_ok, split_sde_detail = zebrafish_split_sde_contract(
            summary.get("simulation"),
            slice_counts=slice_counts,
            expected_observed_counts=spec["observed_counts"],
            analyses=analyses,
        )
        audit.check(
            split_sde_ok,
            "observed-anchored zebrafish split-SDE contract",
            split_sde_detail,
        )
    color_path = downstream / "label_to_color.json"
    color_mapping = (
        json.loads(color_path.read_text(encoding="utf-8"))
        if color_path.is_file()
        else None
    )
    color_ok, color_detail = valid_color_map(
        color_mapping,
        required_labels=observed_labels | generated_labels,
    )
    audit.check(color_ok, "nondegenerate label color mapping", color_detail)

    velocity_path = downstream / "velocity" / "velocity_components.npz"
    velocity_ok = velocity_path.is_file()
    velocity_detail = "missing"
    if velocity_ok:
        with np.load(velocity_path) as velocity:
            expected_keys = {
                "drift",
                "interaction",
                "score",
                "full",
                "times",
                "features",
            }
            velocity_ok = set(velocity.files) == expected_keys
            arrays = {key: np.asarray(velocity[key]) for key in expected_keys}
            component_shape = arrays["full"].shape
            velocity_ok = velocity_ok and component_shape == (
                sum(spec["observed_counts"].values()),
                model_dim,
            )
            velocity_ok = velocity_ok and all(
                np.isfinite(value).all() for value in arrays.values()
            )
            velocity_ok = velocity_ok and np.allclose(
                arrays["full"],
                arrays["drift"] + arrays["interaction"] + arrays["score"],
                rtol=1e-5,
                atol=1e-5,
            )
            velocity_ok = velocity_ok and all(
                float(np.ptp(arrays[key])) > 0.0
                for key in ("drift", "interaction", "score", "full")
            )
            velocity_detail = (
                f"shape={component_shape}, times={tuple(np.unique(arrays['times']))}"
            )
    audit.check(
        velocity_ok, "finite nondegenerate velocity decomposition", velocity_detail
    )

    growth = pd.read_csv(downstream / "growth" / "growth_by_cell.csv")
    growth_ok = (
        len(growth) == sum(slice_counts.values())
        and finite_numeric_frame(growth)
        and float(growth["growth"].max() - growth["growth"].min()) > 0.0
    )
    audit.check(
        growth_ok,
        "finite nondegenerate growth",
        f"rows={len(growth)}, range=({growth['growth'].min():.6g}, {growth['growth'].max():.6g})",
    )

    composition = pd.read_csv(downstream / "composition" / "celltype_composition.csv")
    fraction_sums = composition.groupby("time")["fraction"].sum()
    composition_ok = (
        not composition.empty
        and finite_numeric_frame(composition)
        and bool((composition["count"] > 0).all())
        and bool(np.allclose(fraction_sums.to_numpy(), 1.0, atol=1e-8))
    )
    audit.check(composition_ok, "cell-type composition", f"rows={len(composition)}")

    communication_dir = downstream / "communication"
    communication = pd.read_csv(communication_dir / "communication_by_celltype.csv")
    communication_ok = not communication.empty and finite_numeric_frame(communication)
    communication_ok = communication_ok and bool(
        (communication["attention_per_source"] >= 0).all()
    )
    communication_ok = (
        communication_ok and float(communication["attention_per_source"].max()) > 0.0
    )
    sparse_details = []
    for time_value in expected_times:
        attention_path = (
            communication_dir
            / "sparse_attention"
            / f"attn_mean_interp_t{time_value}.npy"
        )
        edge_path = (
            communication_dir
            / "sparse_attention"
            / f"edge_index_interp_t{time_value}.npy"
        )
        if not attention_path.is_file() or not edge_path.is_file():
            communication_ok = False
            continue
        attention = np.load(attention_path, mmap_mode="r")
        edge_index = np.load(edge_path, mmap_mode="r")
        aligned = (
            edge_index.ndim == 2
            and edge_index.shape[0] == 2
            and edge_index.shape[1] == attention.shape[0]
        )
        valid = (
            aligned
            and attention.size > 0
            and np.isfinite(attention).all()
            and bool((attention >= 0).all())
        )
        communication_ok = communication_ok and valid
        sparse_details.append(f"t={time_value:g}:edges={attention.size}")
    dense_attention = list(
        (communication_dir / "sparse_attention").glob("attn_interp_t*.npy")
    )
    communication_ok = communication_ok and not dense_attention
    audit.check(
        communication_ok,
        "sparse nondegenerate communication",
        ", ".join(sparse_details),
    )

    gene_dir = downstream / "gene_dynamics"
    mean_expression = pd.read_csv(gene_dir / "mean_expression.csv", index_col=0)
    signed_expression = pd.read_csv(
        gene_dir / "signed_mean_expression.csv", index_col=0
    )
    diagnostics = pd.read_csv(gene_dir / "reconstruction_diagnostics.csv")
    top_genes = pd.read_csv(gene_dir / "top_variable_genes.csv")
    prototypes = pd.read_csv(gene_dir / "cluster_prototypes.csv", index_col=0)
    gene_ok = (
        not mean_expression.empty
        and mean_expression.shape[1] == len(expected_times)
        and finite_numeric_frame(mean_expression)
        and bool((mean_expression.to_numpy(dtype=float) >= 0).all())
        and finite_numeric_frame(signed_expression)
        and finite_numeric_frame(diagnostics)
        and not top_genes.empty
        and finite_numeric_frame(top_genes)
        and finite_numeric_frame(prototypes)
        and float(np.ptp(mean_expression.to_numpy(dtype=float))) > 0.0
    )
    audit.check(
        gene_ok,
        "finite nondegenerate gene dynamics",
        f"genes={mean_expression.shape[0]}, times={mean_expression.shape[1]}",
    )

    lr_dir = downstream / "ligand_receptor"
    lr_tables = {
        name: pd.read_csv(lr_dir / f"{name}.csv")
        for name in (
            "pair_timecourse",
            "celltype_timecourse",
            "pattern_summary",
            "coverage",
            "trajectory_coverage",
        )
    }
    lr_ok = all(not frame.empty for frame in lr_tables.values())
    lr_ok = lr_ok and all(
        finite_numeric_frame(frame, allow_na=True) for frame in lr_tables.values()
    )
    pair_times = tuple(
        sorted(
            pd.to_numeric(lr_tables["pair_timecourse"]["time"]).astype(float).unique()
        )
    )
    lr_ok = lr_ok and pair_times == expected_times
    retained = lr_tables["trajectory_coverage"].get("retained")
    lr_ok = (
        lr_ok
        and retained is not None
        and bool(retained.astype(str).str.lower().eq("true").any())
    )
    retained_pair_count, retained_pair_detail = retained_top_level_lr_pairs(
        lr_tables["trajectory_coverage"]
    )
    if "strict_lr_pairs" in spec:
        lr_ok = lr_ok and retained_pair_count == int(spec["strict_lr_pairs"])
        complete_pair_times, pair_grid_detail = complete_lr_pair_time_grid(
            lr_tables["trajectory_coverage"],
            lr_tables["pair_timecourse"],
            expected_times=expected_times,
        )
        lr_ok = lr_ok and complete_pair_times
        retained_pair_detail += (
            f", complete_pair_times={complete_pair_times}, {pair_grid_detail}"
        )
    audit.check(
        lr_ok,
        "strict complete-trajectory LR dynamics",
        f"pair_rows={len(lr_tables['pair_timecourse'])}, "
        f"celltype_rows={len(lr_tables['celltype_timecourse'])}, {retained_pair_detail}",
    )

    expected_visuals = (
        downstream / "growth" / "growth_timepoint_grid.pdf",
        downstream / "composition" / "celltype_composition.pdf",
        downstream / "gene_dynamics" / "temporal_gene_programs.pdf",
        downstream / "figures" / "spatiotemporal_communication_3d.html",
    )
    visuals_ok = all(
        path.is_file() and path.stat().st_size > 0 for path in expected_visuals
    )
    audit.check(
        visuals_ok,
        "standard figure artifacts",
        ", ".join(path.name for path in expected_visuals),
    )


def validate_dataset(run_root: Path, dataset: str) -> Audit:
    audit = Audit(dataset)
    spec = VALIDATION_PROFILES[dataset]
    run_dir = run_root / dataset
    paths = required_files(run_dir, dataset, spec)
    missing = False
    for label, path in paths.items():
        present = path.is_file() and path.stat().st_size > 0
        audit.check(present, label, str(path))
        missing = missing or not present
    if spec["edge_prior_mode"] == "all_spatial":
        clean, cleanliness_detail = validate_no_lr_ablation_artifact_cleanliness(
            run_dir
        )
        audit.check(
            clean,
            "no learned LR graph or predictor artifacts for all-spatial run",
            cleanliness_detail,
        )
    if missing:
        return audit

    n_cells, model_dim, aligned_class_counts = validate_aligned(audit, paths, spec)
    audit.check(
        n_cells == sum(spec["observed_counts"].values()),
        "cohort count total",
        f"n_cells={n_cells}",
    )
    threshold = None
    if spec["edge_prior_mode"] == "learned":
        threshold = validate_edge_predictor(audit, run_dir, paths, spec)
    validate_training(audit, paths, spec, threshold, model_dim)
    validate_downstream(
        audit,
        run_dir,
        paths,
        spec,
        threshold,
        model_dim,
        aligned_class_counts,
    )
    return audit


def validate_no_lr_ablation_artifact_cleanliness(run_dir: Path) -> tuple[bool, str]:
    """Reject newly generated learned artifacts in an all-spatial ablation tree.

    The matched input ``admouse_aligned.h5ad`` may itself retain immutable graph
    provenance from the main run; fit strips that metadata in memory and records
    the action in the trained AnnData.  Separate graph/model files in the
    ablation run remain forbidden.
    """

    preprocess = run_dir / "preprocess"
    artifacts = sorted(
        path
        for relative in ("edge_classifier", "input_graph", "metadata")
        for root in (preprocess / relative,)
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
    )
    return (not artifacts), (
        "none"
        if not artifacts
        else ", ".join(str(path.relative_to(preprocess)) for path in artifacts)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(VALIDATION_PROFILES),
        default=tuple(DATASETS),
        help=(
            "Datasets or explicit ablation profiles to validate "
            "(default: four production main runs)."
        ),
    )
    parser.add_argument(
        "--report", type=Path, help="Optional human-readable JSON report."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    audits = []
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        try:
            audit = validate_dataset(run_root, dataset)
        except Exception as error:
            audit = Audit(dataset)
            audit.check(
                False, "acceptance execution", f"{type(error).__name__}: {error}"
            )
        audits.append(audit)
        for item in audit.checks:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")
        for warning in audit.warnings:
            print(f"[WARN] {warning}")

    report = {
        "run_root": str(run_root),
        "status": "PASS" if all(not audit.errors for audit in audits) else "FAIL",
        "datasets": {audit.dataset: audit.as_dict() for audit in audits},
    }
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"\nReport: {report_path}")
    print(f"\nOverall: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
