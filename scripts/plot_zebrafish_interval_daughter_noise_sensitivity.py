#!/usr/bin/env python3
"""Render the signed zebrafish interval-local daughter-noise analysis.

This publication plotter accepts only a complete, signed manifest produced by
``run_zebrafish_interval_daughter_noise_sensitivity.py``.  Before creating an
output directory it verifies the serialized-manifest sidecar, canonical JSON
signature, frozen scientific settings, the hashes and schemas of every table,
all retained raw-state hashes, and the separately supplied canonical
acceptance report.  The figure reports paired midpoint changes from the
daughter-noise-zero run; it never turns an endpoint or global-t0 result into an
interval-local claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SCHEMA_VERSION = 1
ANALYSIS_ID = "zebrafish_interval_local_daughter_noise_sensitivity"
FIGURE_ANALYSIS_ID = f"{ANALYSIS_ID}_publication_figure"
TRAJECTORY_SCOPE = (
    "independent observed-anchored interval-local one-sided forecasts; each "
    "interval starts from all real cells at its left observed anchor; not "
    "conditioned on the following observed endpoint; not global-t0; not "
    "lineage-continuous across intervals; no spatial warp"
)
OBSERVED_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0)
INTERVALS = tuple(zip(OBSERVED_TIMES[:-1], OBSERVED_TIMES[1:]))
NOISE_VALUES = (0.0, 0.01, 0.03, 0.06)
PAIRED_SEEDS = (42, 43, 44, 45, 46)
MIDPOINT_ROLE = "midpoint_one_sided_forecast"
TABLE_NAMES = (
    "anchor_roster",
    "composition_long",
    "particle_counts",
    "lineage_descendant_counts",
    "lineage_transition_long",
    "noise0_paired_deltas",
)
PAIRED_METRICS = (
    "composition_total_variation",
    "particle_count_relative_delta",
    "joint_w2_from_noise0",
    "spatial_w2_from_noise0",
    "lineage_fate_mean_total_variation_from_noise0",
)
REQUIRED_TABLE_COLUMNS = {
    "anchor_roster": {
        "interval_start",
        "interval_end",
        "source_lineage_id",
        "lineage_namespace",
        "source_obs_id",
        "source_celltype",
        "source_state",
    },
    "composition_long": {
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "forecast_time",
        "forecast_role",
        "state_source",
        "following_endpoint_conditioned",
        "celltype",
        "count",
        "fraction",
        "n_particles",
        "population_empty",
    },
    "particle_counts": {
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "time",
        "frame_role",
        "state_source",
        "n_particles",
    },
    "lineage_descendant_counts": {
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "forecast_time",
        "forecast_role",
        "state_source",
        "following_endpoint_conditioned",
        "source_lineage_id",
        "lineage_namespace",
        "source_obs_id",
        "source_celltype",
        "descendant_count",
        "lineage_alive",
    },
    "lineage_transition_long": {
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "forecast_time",
        "forecast_role",
        "state_source",
        "following_endpoint_conditioned",
        "source_lineage_id",
        "lineage_namespace",
        "source_obs_id",
        "source_celltype",
        "target_celltype",
        "descendant_count",
        "fraction_within_lineage",
    },
    "noise0_paired_deltas": {
        "baseline_daughter_noise_std",
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "forecast_time",
        "forecast_role",
        "n_source_lineages",
        "baseline_n_particles",
        "n_particles",
        "particle_count_delta",
        "particle_count_relative_delta",
        "composition_total_variation",
        "lineage_fate_mean_total_variation_from_noise0",
        "lineage_fate_max_total_variation_from_noise0",
        "paired_common_seed",
        "ot_max_points",
        "joint_w1_from_noise0",
        "joint_w2_from_noise0",
        "joint_ot_noise0_points",
        "joint_ot_noise_points",
        "joint_ot_status",
        "spatial_w1_from_noise0",
        "spatial_w2_from_noise0",
        "spatial_ot_noise0_points",
        "spatial_ot_noise_points",
        "spatial_ot_status",
    },
}
EXPECTED_TABLE_COLUMNS = {
    "anchor_roster": (
        "interval_start",
        "interval_end",
        "source_lineage_id",
        "lineage_namespace",
        "source_obs_id",
        "source_celltype",
        "source_state",
    ),
    "composition_long": (
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "forecast_time",
        "forecast_role",
        "state_source",
        "following_endpoint_conditioned",
        "celltype",
        "count",
        "fraction",
        "n_particles",
        "population_empty",
    ),
    "particle_counts": (
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "time",
        "frame_role",
        "state_source",
        "n_particles",
    ),
    "lineage_descendant_counts": (
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "forecast_time",
        "forecast_role",
        "state_source",
        "following_endpoint_conditioned",
        "source_lineage_id",
        "lineage_namespace",
        "source_obs_id",
        "source_celltype",
        "descendant_count",
        "lineage_alive",
    ),
    "lineage_transition_long": (
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "forecast_time",
        "forecast_role",
        "state_source",
        "following_endpoint_conditioned",
        "source_lineage_id",
        "lineage_namespace",
        "source_obs_id",
        "source_celltype",
        "target_celltype",
        "descendant_count",
        "fraction_within_lineage",
    ),
    "noise0_paired_deltas": (
        "baseline_daughter_noise_std",
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "forecast_time",
        "forecast_role",
        "n_source_lineages",
        "baseline_n_particles",
        "n_particles",
        "particle_count_delta",
        "particle_count_relative_delta",
        "composition_total_variation",
        "mean_absolute_lineage_descendant_count_delta",
        "max_absolute_lineage_descendant_count_delta",
        "fraction_lineages_same_descendant_count",
        "lineage_alive_status_agreement",
        "lineage_survival_jaccard",
        "lineage_fate_mean_total_variation_from_noise0",
        "lineage_fate_max_total_variation_from_noise0",
        "paired_common_seed",
        "ot_max_points",
        "joint_ot_random_seed",
        "joint_w1_from_noise0",
        "joint_w2_from_noise0",
        "joint_ot_noise0_points",
        "joint_ot_noise_points",
        "joint_ot_status",
        "spatial_ot_random_seed",
        "spatial_w1_from_noise0",
        "spatial_w2_from_noise0",
        "spatial_ot_noise0_points",
        "spatial_ot_noise_points",
        "spatial_ot_status",
    ),
}
NOISE_COLORS = {0.01: "#07838B", 0.03: "#7A6BBE", 0.06: "#CC6677"}
NOISE_MARKERS = {0.01: "o", 0.03: "s", 0.06: "D"}
TEXT_COLOR = "#24313A"
GRID_COLOR = "#D7DDE2"
REFERENCE_COLOR = "#59616A"
FIGURE_BASENAME = "zebrafish_interval_daughter_noise_sensitivity"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_digest(value: str, *, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 digest")
    return digest


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verified_file(
    path: str | Path, expected_sha256: str, *, description: str
) -> tuple[Path, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {description}: {resolved}")
    expected = _normalise_digest(expected_sha256, name=f"{description} SHA-256")
    observed = _sha256(resolved)
    if observed != expected:
        raise RuntimeError(
            f"{description} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return resolved, observed


def _read_json(path: Path, *, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {description} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{description} must contain a JSON object")
    return payload


def _require_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{description} must be a JSON object")
    return value


def _require_exact(
    payload: Mapping[str, Any], keys: Sequence[str], expected: Any
) -> None:
    current: Any = payload
    dotted = ".".join(keys)
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise RuntimeError(f"Signed manifest is missing required setting {dotted}")
        current = current[key]
    if current != expected:
        raise RuntimeError(
            f"Signed manifest setting {dotted} must equal {expected!r}; "
            f"observed {current!r}"
        )


def _verify_manifest_signature(manifest: Mapping[str, Any]) -> None:
    signature = _require_mapping(
        manifest.get("signature"), description="manifest.signature"
    )
    if signature.get("algorithm") != "sha256-canonical-json":
        raise RuntimeError("Manifest signature algorithm is not sha256-canonical-json")
    fields = signature.get("covered_top_level_fields")
    if (
        not isinstance(fields, list)
        or not fields
        or not all(isinstance(field, str) for field in fields)
        or len(fields) != len(set(fields))
    ):
        raise RuntimeError("Manifest signature coverage list is invalid")
    unsigned_fields = set(manifest) - {"status", "completed_at", "signature"}
    if set(fields) != unsigned_fields:
        raise RuntimeError(
            "Manifest signature must cover every scientific top-level field"
        )
    covered = {field: manifest[field] for field in fields}
    expected = _normalise_digest(
        str(signature.get("value", "")), name="manifest canonical signature"
    )
    observed = _stable_json_sha256(covered)
    if observed != expected:
        raise RuntimeError(
            "Manifest canonical signature mismatch: signed scientific content "
            "has changed"
        )


def _verify_manifest_sidecar(manifest_path: Path, manifest_sha256: str) -> None:
    sidecar = manifest_path.with_name("run_manifest.sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing signed-manifest sidecar: {sidecar}")
    lines = [line.strip() for line in sidecar.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) != 1:
        raise RuntimeError("run_manifest.sha256 must contain exactly one record")
    fields = lines[0].split()
    if fields != [manifest_sha256, "run_manifest.json"]:
        raise RuntimeError(
            "run_manifest.sha256 does not bind the supplied run_manifest.json"
        )


def _verify_frozen_settings(manifest: Mapping[str, Any]) -> None:
    exact = (
        (("schema_version",), SCHEMA_VERSION),
        (("analysis",), ANALYSIS_ID),
        (("status",), "complete"),
        (("trajectory_scope",), TRAJECTORY_SCOPE),
        (("claim_guardrails", "following_endpoint_conditioned"), False),
        (("claim_guardrails", "global_t0_rollout"), False),
        (("claim_guardrails", "lineage_continuous_across_intervals"), False),
        (("claim_guardrails", "spatial_warp_applied"), False),
        (("claim_guardrails", "endpoint_is_observed_when_included"), False),
        (("data_contract", "observed_times"), list(OBSERVED_TIMES)),
        (("data_contract", "intervals"), [list(interval) for interval in INTERVALS]),
        (("data_contract", "fresh_lineage_roster_per_interval"), True),
        (("data_contract", "joint_feature_dim"), 52),
        (
            ("data_contract", "lineage_namespace_fields"),
            ["anchor_time", "source_obs_id"],
        ),
        (
            ("data_contract", "initial_roster"),
            "all real observed cells at each interval's left anchor",
        ),
        (("simulation", "daughter_noise_std"), list(NOISE_VALUES)),
        (("simulation", "paired_seeds"), list(PAIRED_SEEDS)),
        (("simulation", "paired_common_seed_with_noise0"), True),
        (("simulation", "midpoint_forecast"), True),
        (("simulation", "dt"), 0.05),
        (("simulation", "resample_dt"), 0.05),
        (("simulation", "continuous_diffusion_sigma"), 0.03),
        (("simulation", "growth_alpha"), 1.0),
        (("simulation", "interaction_m"), 1024),
        (("simulation", "max_particles"), 100000),
        (("simulation", "spatial_warp"), False),
        (("simulation", "classifier_feature_dim"), 52),
        (("simulation", "classifier_knn_neighbors"), 10),
        (
            ("simulation", "interaction_grouping_rng", "stream"),
            "dedicated_torch_generator",
        ),
        (
            ("simulation", "interaction_grouping_rng", "paired_across_daughter_noise"),
            True,
        ),
        (
            ("simulation", "interaction_grouping_rng", "seed_formula"),
            "paired_seed + interaction_seed_offset",
        ),
        (
            ("simulation", "interaction_grouping_rng", "interaction_seed_offset"),
            10000,
        ),
        (("metric_contract", "wasserstein", "metrics"), ["W1", "W2"]),
        (("metric_contract", "wasserstein", "max_points_per_cloud"), 1024),
        (("model_contract", "interaction_type"), "gnn"),
        (("model_contract", "edge_prior_mode"), "learned"),
        (
            ("model_contract", "edge_predictor_source"),
            "embedded_in_weight_checkpoint",
        ),
        (("model_contract", "interaction_group_size"), 1024),
        (("model_contract", "weight_stage"), "Finetune"),
    )
    for keys, expected in exact:
        _require_exact(manifest, keys, expected)
    expected_seed_map = {str(seed): seed + 10000 for seed in PAIRED_SEEDS}
    _require_exact(
        manifest,
        (
            "simulation",
            "interaction_grouping_rng",
            "interaction_seed_by_paired_seed",
        ),
        expected_seed_map,
    )
    model_contract = _require_mapping(
        manifest.get("model_contract"), description="manifest.model_contract"
    )
    components = model_contract.get("components")
    if not isinstance(components, list) or not {
        "velocity",
        "growth",
        "score",
        "interaction",
    }.issubset(map(str, components)):
        raise RuntimeError("Signed model contract lacks a required learned component")
    threshold = model_contract.get("edge_predictor_threshold")
    cutoff = model_contract.get("spatial_cutoff")
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise RuntimeError("Signed model contract has an invalid edge threshold")
    if (
        not isinstance(cutoff, (int, float))
        or not math.isfinite(float(cutoff))
        or float(cutoff) <= 0.0
    ):
        raise RuntimeError("Signed model contract has an invalid spatial cutoff")
    if (
        not isinstance(model_contract.get("score_stage"), str)
        or not model_contract["score_stage"]
    ):
        raise RuntimeError("Signed model contract lacks a score-checkpoint stage")
    include_end = manifest["simulation"].get("end_forecast_included")
    if not isinstance(include_end, bool):
        raise RuntimeError("simulation.end_forecast_included must be Boolean")
    frames_per_run = 2 if include_end else 1
    expected_counts = {
        "independent_interval_noise_seed_runs": 80,
        "forecast_frames": 80 * frames_per_run,
        "noise0_paired_delta_rows": 60 * frames_per_run,
        "raw_state_files": 80,
    }
    _require_exact(manifest, ("run_counts",), expected_counts)


def _verify_acceptance_binding(
    manifest: Mapping[str, Any], path: str | Path, expected_sha256: str
) -> tuple[Path, str, Mapping[str, Any]]:
    record = _require_mapping(
        _require_mapping(manifest.get("inputs"), description="manifest.inputs").get(
            "canonical_acceptance_report"
        ),
        description="manifest canonical acceptance record",
    )
    recorded = _normalise_digest(
        str(record.get("sha256", "")), name="manifest acceptance SHA-256"
    )
    expected = _normalise_digest(expected_sha256, name="supplied acceptance SHA-256")
    if expected != recorded:
        raise RuntimeError(
            "Supplied acceptance SHA-256 is not the acceptance artifact bound by "
            "the signed sensitivity manifest"
        )
    report_path, digest = _verified_file(
        path, expected, description="canonical four-dataset acceptance report"
    )
    report = _read_json(report_path, description="acceptance report")
    datasets = report.get("datasets")
    zebrafish = datasets.get("zebrafish") if isinstance(datasets, Mapping) else None
    if report.get("status") != "PASS" or not isinstance(zebrafish, Mapping):
        raise RuntimeError(
            "Acceptance report must record overall PASS and datasets.zebrafish"
        )
    if zebrafish.get("status") != "PASS":
        raise RuntimeError("Acceptance report does not record zebrafish status PASS")
    required_exact = record.get("required_exact")
    if required_exact != {
        "status": "PASS",
        "datasets": {"zebrafish": {"status": "PASS"}},
    }:
        raise RuntimeError("Manifest acceptance requirement is not the frozen contract")
    observed_run_root = record.get("observed_run_root")
    if not isinstance(observed_run_root, str) or not observed_run_root:
        raise RuntimeError("Signed acceptance binding lacks a non-empty run_root")
    if observed_run_root != report.get("run_root"):
        raise RuntimeError(
            "Acceptance run_root differs from the value signed into the sensitivity "
            "manifest"
        )
    return report_path, digest, report


def _artifact_path(root: Path, relative: Any, *, description: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"{description} path must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{description} path escapes the signed run root") from exc
    return candidate


def _verify_artifact(root: Path, record: Any, *, description: str) -> tuple[Path, str]:
    artifact = _require_mapping(record, description=f"{description} record")
    path = _artifact_path(root, artifact.get("path"), description=description)
    expected_size = artifact.get("size_bytes")
    if not path.is_file() or not isinstance(expected_size, int) or expected_size <= 0:
        raise RuntimeError(f"Missing, empty, or invalid {description}: {path}")
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise RuntimeError(
            f"{description} size mismatch: expected {expected_size}, "
            f"observed {observed_size}"
        )
    expected_hash = _normalise_digest(
        str(artifact.get("sha256", "")), name=f"{description} SHA-256"
    )
    observed_hash = _sha256(path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"{description} SHA-256 mismatch: expected {expected_hash}, "
            f"observed {observed_hash}"
        )
    return path, observed_hash


def _verify_tables_and_raw_states(
    manifest: Mapping[str, Any], root: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    outputs = _require_mapping(manifest.get("outputs"), description="manifest.outputs")
    table_records = _require_mapping(
        outputs.get("tables"), description="manifest.outputs.tables"
    )
    if set(table_records) != set(TABLE_NAMES):
        raise RuntimeError(
            "Signed manifest must contain exactly the six producer-owned tables"
        )
    tables: dict[str, pd.DataFrame] = {}
    artifact_hashes: dict[str, str] = {}
    seen_paths: set[Path] = set()
    for name in TABLE_NAMES:
        record = _require_mapping(
            table_records[name], description=f"table {name} record"
        )
        path, digest = _verify_artifact(
            root, record, description=f"signed table {name}"
        )
        if path in seen_paths:
            raise RuntimeError(f"Duplicate signed artifact path: {path}")
        seen_paths.add(path)
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise RuntimeError(f"Cannot read signed table {name}: {exc}") from exc
        if record.get("row_count") != len(frame):
            raise RuntimeError(f"Signed row count mismatch for table {name}")
        if record.get("columns") != list(frame.columns):
            raise RuntimeError(f"Signed column order mismatch for table {name}")
        if list(frame.columns) != list(EXPECTED_TABLE_COLUMNS[name]):
            raise RuntimeError(f"Signed table {name} does not match producer schema v1")
        missing = REQUIRED_TABLE_COLUMNS[name] - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"Signed table {name} is missing required columns: {sorted(missing)}"
            )
        tables[name] = frame
        artifact_hashes[name] = digest
    if outputs.get("raw_states_saved") is not True:
        raise RuntimeError("Sensitivity manifest must record raw_states_saved=true")
    raw_records = outputs.get("raw_states")
    if not isinstance(raw_records, list) or len(raw_records) != 80:
        raise RuntimeError("Sensitivity manifest must bind exactly 80 raw-state files")
    for index, record in enumerate(raw_records):
        path, digest = _verify_artifact(
            root, record, description=f"signed raw state {index}"
        )
        if path.suffix != ".npz":
            raise RuntimeError(f"Raw-state artifact is not NPZ: {path}")
        if path in seen_paths:
            raise RuntimeError(f"Duplicate signed artifact path: {path}")
        seen_paths.add(path)
        artifact_hashes[f"raw_state_{index:03d}"] = digest
    return tables, artifact_hashes


def _boolean_values(series: pd.Series, *, description: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
    }
    converted = series.map(mapping)
    if converted.isna().any():
        raise RuntimeError(f"{description} contains a non-Boolean value")
    return converted.astype(bool)


def _assert_numeric_finite(
    frame: pd.DataFrame, columns: Sequence[str], *, description: str
) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise RuntimeError(f"{description}.{column} contains non-finite values")
        frame[column] = numeric


def _validate_common_generated_table(frame: pd.DataFrame, *, name: str) -> None:
    if frame.empty:
        raise RuntimeError(f"Signed table {name} is empty")
    if set(frame["state_source"].astype(str)) != {"generated_interval_local_one_sided"}:
        raise RuntimeError(f"Signed table {name} contains a non-generated state")
    if _boolean_values(
        frame["following_endpoint_conditioned"],
        description=f"{name}.following_endpoint_conditioned",
    ).any():
        raise RuntimeError(f"Signed table {name} contains endpoint-conditioned rows")


def _validate_csv_semantics(
    manifest: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    roster = tables["anchor_roster"].copy()
    if roster.empty or set(roster["source_state"].astype(str)) != {
        "observed_real_left_anchor"
    }:
        raise RuntimeError("Anchor roster is empty or does not contain real anchors")
    roster_intervals = {
        (float(start), float(end))
        for start, end in roster[["interval_start", "interval_end"]].itertuples(
            index=False, name=None
        )
    }
    if roster_intervals != set(INTERVALS):
        raise RuntimeError("Anchor roster does not cover the frozen four intervals")

    for name in (
        "composition_long",
        "lineage_descendant_counts",
        "lineage_transition_long",
    ):
        _validate_common_generated_table(tables[name], name=name)

    composition = tables["composition_long"].copy()
    _assert_numeric_finite(
        composition,
        ("count", "fraction", "n_particles"),
        description="composition_long",
    )
    if (
        (composition["count"] < 0).any()
        or (composition["fraction"] < 0).any()
        or (composition["fraction"] > 1).any()
        or (composition["n_particles"] < 0).any()
    ):
        raise RuntimeError("Composition table contains out-of-range values")
    composition_groups = composition.groupby(
        [
            "daughter_noise_std",
            "seed",
            "interval_start",
            "forecast_time",
            "forecast_role",
        ],
        sort=False,
    )
    for _, group in composition_groups:
        population_empty = _boolean_values(
            group["population_empty"], description="composition_long.population_empty"
        )
        n_particles = int(group["n_particles"].iloc[0])
        if group["n_particles"].nunique() != 1:
            raise RuntimeError("Composition frame disagrees on n_particles")
        if n_particles == 0:
            if not population_empty.all() or float(group["fraction"].sum()) != 0.0:
                raise RuntimeError("Empty composition frame has invalid fractions")
        elif (
            population_empty.any()
            or int(group["count"].sum()) != n_particles
            or not math.isclose(
                float(group["fraction"].sum()), 1.0, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise RuntimeError("Composition counts/fractions do not close to one frame")

    particles = tables["particle_counts"].copy()
    _assert_numeric_finite(
        particles,
        (
            "daughter_noise_std",
            "seed",
            "interaction_seed",
            "interval_start",
            "interval_end",
            "time",
            "n_particles",
        ),
        description="particle_counts",
    )
    if (particles["n_particles"] < 0).any():
        raise RuntimeError("Particle-count table contains a negative population")
    if set(particles["daughter_noise_std"].astype(float)) != set(NOISE_VALUES):
        raise RuntimeError(
            "Particle-count table does not contain the frozen noise grid"
        )

    paired = tables["noise0_paired_deltas"].copy()
    numeric_columns = (
        "baseline_daughter_noise_std",
        "daughter_noise_std",
        "seed",
        "interaction_seed",
        "interval_start",
        "interval_end",
        "forecast_time",
        "n_source_lineages",
        "baseline_n_particles",
        "n_particles",
        "particle_count_delta",
        "particle_count_relative_delta",
        "composition_total_variation",
        "lineage_fate_mean_total_variation_from_noise0",
        "lineage_fate_max_total_variation_from_noise0",
        "ot_max_points",
        "joint_w1_from_noise0",
        "joint_w2_from_noise0",
        "joint_ot_noise0_points",
        "joint_ot_noise_points",
        "spatial_w1_from_noise0",
        "spatial_w2_from_noise0",
        "spatial_ot_noise0_points",
        "spatial_ot_noise_points",
    )
    _assert_numeric_finite(paired, numeric_columns, description="noise0_paired_deltas")
    if not (paired["baseline_daughter_noise_std"] == 0.0).all():
        raise RuntimeError("Paired table contains a non-zero baseline noise")
    if set(paired["daughter_noise_std"].astype(float)) != set(NOISE_VALUES[1:]):
        raise RuntimeError("Paired table does not contain the three nonzero noises")
    if set(paired["seed"].astype(int)) != set(PAIRED_SEEDS):
        raise RuntimeError("Paired table does not contain the five frozen seeds")
    if not _boolean_values(
        paired["paired_common_seed"], description="paired_common_seed"
    ).all():
        raise RuntimeError("Paired table contains an unpaired simulation seed")
    if not (
        paired["interaction_seed"].astype(int) == paired["seed"].astype(int) + 10000
    ).all():
        raise RuntimeError(
            "Paired table violates the dedicated interaction RNG binding"
        )
    if not (
        paired["particle_count_delta"].astype(int)
        == paired["n_particles"].astype(int)
        - paired["baseline_n_particles"].astype(int)
    ).all():
        raise RuntimeError("Paired table particle-count delta is inconsistent")
    if (paired["baseline_n_particles"] <= 0).any():
        raise RuntimeError("Paired table contains an empty noise-zero baseline")
    expected_relative = paired["particle_count_delta"] / paired["baseline_n_particles"]
    if not np.allclose(
        paired["particle_count_relative_delta"], expected_relative, rtol=0.0, atol=1e-12
    ):
        raise RuntimeError("Paired relative particle-count delta is inconsistent")
    for column in (
        "composition_total_variation",
        "lineage_fate_mean_total_variation_from_noise0",
        "lineage_fate_max_total_variation_from_noise0",
    ):
        if ((paired[column] < 0) | (paired[column] > 1)).any():
            raise RuntimeError(f"Paired metric {column} is outside [0, 1]")
    for column in (
        "joint_w1_from_noise0",
        "joint_w2_from_noise0",
        "spatial_w1_from_noise0",
        "spatial_w2_from_noise0",
    ):
        if (paired[column] < 0).any():
            raise RuntimeError(f"Paired metric {column} is negative")
    if set(paired["joint_ot_status"].astype(str)) != {"complete"} or set(
        paired["spatial_ot_status"].astype(str)
    ) != {"complete"}:
        raise RuntimeError("Paired OT table contains an incomplete comparison")
    if not (paired["ot_max_points"].astype(int) == 1024).all():
        raise RuntimeError("Paired table changed the retained OT support cap")

    include_end = bool(manifest["simulation"]["end_forecast_included"])
    roles = {MIDPOINT_ROLE}
    if include_end:
        roles.add("endpoint_one_sided_forecast")
    if set(paired["forecast_role"].astype(str)) != roles:
        raise RuntimeError("Paired table forecast roles differ from the signed setting")
    midpoint = paired.loc[paired["forecast_role"].astype(str) == MIDPOINT_ROLE].copy()
    identity_columns = [
        "daughter_noise_std",
        "seed",
        "interval_start",
        "interval_end",
        "forecast_time",
    ]
    if midpoint.duplicated(identity_columns).any():
        raise RuntimeError("Paired midpoint table contains duplicate run identities")
    expected_identities = {
        (noise, seed, start, end, (start + end) / 2.0)
        for noise in NOISE_VALUES[1:]
        for seed in PAIRED_SEEDS
        for start, end in INTERVALS
    }
    observed_identities = {
        tuple(
            float(value) if index != 1 else int(value)
            for index, value in enumerate(row)
        )
        for row in midpoint[identity_columns].itertuples(index=False, name=None)
    }
    if observed_identities != expected_identities:
        raise RuntimeError(
            "Paired midpoint table does not contain the exact 60-run grid"
        )
    if len(midpoint) != 60:
        raise RuntimeError("Paired midpoint table must contain exactly 60 rows")
    return midpoint


def _summarise_midpoints(midpoint: pd.DataFrame) -> pd.DataFrame:
    grouped = midpoint.groupby(
        [
            "daughter_noise_std",
            "interval_start",
            "interval_end",
            "forecast_time",
        ],
        sort=True,
    )
    rows: list[dict[str, object]] = []
    for identity, frame in grouped:
        noise, start, end, forecast_time = identity
        seeds = sorted(frame["seed"].astype(int).unique().tolist())
        if seeds != list(PAIRED_SEEDS) or len(frame) != len(PAIRED_SEEDS):
            raise RuntimeError(
                "Each plotted interval/noise point must contain the five paired seeds"
            )
        row: dict[str, object] = {
            "daughter_noise_std": float(noise),
            "interval_start": float(start),
            "interval_end": float(end),
            "forecast_time": float(forecast_time),
            "n_paired_seeds": len(seeds),
        }
        for metric in PAIRED_METRICS:
            values = frame[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sem"] = float(values.std(ddof=1) / math.sqrt(len(values)))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        ["daughter_noise_std", "interval_start"], ignore_index=True
    )
    if len(summary) != 12:
        raise RuntimeError("Expected exactly 12 plotted interval/noise summaries")
    return summary


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _style_axis(ax: plt.Axes, *, panel: str, title: str) -> None:
    ax.text(
        -0.07 if panel == "e" else -0.12,
        1.12,
        panel,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(["0–1", "1–2", "2–3", "3–4"])
    ax.set_xlabel("Observed interval (midpoint forecast)")


def _plot_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    metric: str,
    panel: str,
    title: str,
    ylabel: str,
    percent: bool = False,
    zero_reference: bool = False,
) -> None:
    _style_axis(ax, panel=panel, title=title)
    if zero_reference:
        ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8, zorder=0)
    for noise in NOISE_VALUES[1:]:
        group = summary.loc[
            np.isclose(summary["daughter_noise_std"], noise)
        ].sort_values("forecast_time")
        x = group["forecast_time"].to_numpy(dtype=float)
        mean = group[f"{metric}_mean"].to_numpy(dtype=float)
        sem = group[f"{metric}_sem"].to_numpy(dtype=float)
        color = NOISE_COLORS[noise]
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)
        ax.plot(
            x,
            mean,
            color=color,
            marker=NOISE_MARKERS[noise],
            markerfacecolor="white",
            markeredgewidth=1.1,
        )
    ax.set_ylabel(ylabel)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.margins(x=0.08, y=0.14)
    if not zero_reference:
        ax.set_ylim(bottom=0.0)


def _save_figure(summary: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    _apply_style()
    figure = plt.figure(figsize=(8.27, 11.69), constrained_layout=False)
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1.0, 1.05),
        left=0.10,
        right=0.96,
        top=0.91,
        bottom=0.08,
        hspace=0.48,
        wspace=0.34,
    )
    axes = (
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, :]),
    )
    specifications = (
        (
            "composition_total_variation",
            "a",
            "Cell-type composition",
            "Total variation from noise = 0",
            True,
            False,
        ),
        (
            "particle_count_relative_delta",
            "b",
            "Population size",
            "Relative particle-count change",
            True,
            True,
        ),
        (
            "joint_w2_from_noise0",
            "c",
            "Joint-state distribution",
            "Empirical W2 from noise = 0",
            False,
            False,
        ),
        (
            "spatial_w2_from_noise0",
            "d",
            "Spatial distribution",
            "Empirical W2 from noise = 0",
            False,
            False,
        ),
        (
            "lineage_fate_mean_total_variation_from_noise0",
            "e",
            "Interval-local lineage fates",
            "Mean per-lineage fate TV from noise = 0",
            True,
            False,
        ),
    )
    for axis, specification in zip(axes, specifications):
        metric, panel, title, ylabel, percent, zero_reference = specification
        _plot_metric(
            axis,
            summary,
            metric=metric,
            panel=panel,
            title=title,
            ylabel=ylabel,
            percent=percent,
            zero_reference=zero_reference,
        )
    handles = [
        Line2D(
            [0],
            [0],
            color=NOISE_COLORS[noise],
            marker=NOISE_MARKERS[noise],
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=f"Daughter-noise SD {noise:g}",
        )
        for noise in NOISE_VALUES[1:]
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.972),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=2.0,
    )
    figure.text(
        0.53,
        0.032,
        "Paired with daughter noise = 0; n = 5 common seeds; mean ± SEM.",
        ha="center",
        va="center",
        fontsize=9,
        color=REFERENCE_COLOR,
    )
    metadata = {
        "Title": "Zebrafish interval-local daughter-noise sensitivity",
        "Author": "CytoBridge",
        "Subject": "Paired observed-anchor midpoint sensitivity analysis",
        "Keywords": "CytoBridge, zebrafish, daughter noise, interval-local",
    }
    temporary_pdf = pdf_path.with_name(f".{pdf_path.name}.{os.getpid()}.tmp")
    temporary_png = png_path.with_name(f".{png_path.name}.{os.getpid()}.tmp")
    try:
        figure.savefig(temporary_pdf, format="pdf", metadata=metadata)
        figure.savefig(temporary_png, format="png", dpi=320, metadata=metadata)
        os.replace(temporary_pdf, pdf_path)
        os.replace(temporary_png, png_path)
    finally:
        plt.close(figure)
        temporary_pdf.unlink(missing_ok=True)
        temporary_png.unlink(missing_ok=True)


def _atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    _atomic_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        path,
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, float_format="%.12g")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(
                "Output directory must be new or empty; refusing to overwrite a "
                f"publication bundle: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)
    return output_dir


def _artifact(path: Path, *, root: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty figure artifact: {path}")
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _caption_text(summary: pd.DataFrame) -> str:
    late = summary.loc[np.isclose(summary["interval_start"], 3.0)].sort_values(
        "daughter_noise_std"
    )
    late_parts = []
    for row in late.itertuples(index=False):
        late_parts.append(
            "SD {noise:g}: composition TV {composition}, particle change "
            "{particles}, joint/spatial W2 {joint:.3g}/{spatial:.3g}, lineage-fate "
            "TV {lineage}".format(
                noise=row.daughter_noise_std,
                composition=_format_percent(row.composition_total_variation_mean),
                particles=_format_percent(row.particle_count_relative_delta_mean),
                joint=row.joint_w2_from_noise0_mean,
                spatial=row.spatial_w2_from_noise0_mean,
                lineage=_format_percent(
                    row.lineage_fate_mean_total_variation_from_noise0_mean
                ),
            )
        )
    return (
        "**Zebrafish interval-local daughter-noise sensitivity.** Each observed "
        "interval was simulated independently from all real cells at its left "
        "anchor, without conditioning on the following observed endpoint, a "
        "global-t0 rollout, lineage continuity across intervals, or a spatial "
        "warp. All panels show paired midpoint differences from daughter noise "
        "SD = 0 using the same five simulation and interaction-grouping seeds; "
        "lines and ribbons are the mean and SEM across seeds. **a,** Cell-type "
        "composition total-variation distance. **b,** Relative particle-count "
        "change; the gray line marks zero change. **c,d,** Exact empirical W2 "
        "distance on the retained joint-state and spatial supports, respectively. "
        "**e,** Mean per-source-lineage fate total-variation distance, with "
        "extinction treated as an explicit fate. The generated right endpoint, "
        "when present in the source run, is not plotted. Final interval [3, 4] "
        "midpoint summaries were "
        + "; ".join(late_parts)
        + ". This is an inference-time sensitivity analysis of one frozen learned "
        "checkpoint, not a training-seed hypothesis test.\n"
    )


def _git_state(repo_root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _rebuild_command(
    *,
    manifest_path: Path,
    manifest_sha: str,
    acceptance_path: Path,
    acceptance_sha: str,
    output_dir: Path,
) -> str:
    values = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-manifest",
        str(manifest_path),
        "--expected-manifest-sha256",
        manifest_sha,
        "--acceptance-report",
        str(acceptance_path),
        "--expected-acceptance-sha256",
        acceptance_sha,
        "--output-dir",
        str(output_dir.with_name(f"{output_dir.name}-rebuild")),
    )
    return " ".join(shlex.quote(value) for value in values)


def _provenance_text(
    *,
    manifest_path: Path,
    manifest_sha: str,
    acceptance_path: Path,
    acceptance_sha: str,
    summary_path: Path,
    figure_artifacts: Mapping[str, Mapping[str, object]],
    output_dir: Path,
) -> str:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    rows = [
        "# Figure provenance",
        "",
        f"Generated (UTC): `{_utc_now()}`",
        "",
        "## Scientific scope",
        "",
        f"- Analysis: `{FIGURE_ANALYSIS_ID}`",
        f"- Trajectory contract: {TRAJECTORY_SCOPE}.",
        "- Plotted estimator: mean ± SEM across the five common paired seeds at "
        "each observed-interval midpoint.",
        "- Non-claim: this is inference-time sensitivity of one frozen checkpoint; "
        "it is not a training-seed hypothesis test.",
        "",
        "## Source paths and bound inputs",
        "",
        f"- Producer manifest: `{manifest_path}`",
        f"- Producer manifest SHA-256: `{manifest_sha}`",
        f"- Canonical acceptance report: `{acceptance_path}`",
        f"- Acceptance report SHA-256: `{acceptance_sha}`",
        f"- Plotted source table: `tables/noise0_paired_deltas.csv`",
        "- All six CSV and all 80 raw-state artifact hashes were revalidated before "
        "rendering.",
        "",
        "## Figure contract",
        "",
        "- Page: A4 portrait; five panels; Arial-first sans-serif typography.",
        "- Panel labels: lower-case bold 14 pt; panel titles 12 pt; axes, ticks, and "
        "legend 9 pt.",
        "- Outputs: vector PDF and 320-dpi PNG.",
        f"- Derived plotted values: `{summary_path.name}`",
        f"- Plotter SHA-256: `{_sha256(script_path)}`",
        f"- Git state: `{json.dumps(_git_state(repo_root), sort_keys=True)}`",
        "",
        "## Output hashes",
        "",
    ]
    for name, record in figure_artifacts.items():
        rows.append(f"- `{record['path']}`: `{record['sha256']}` ({name})")
    rows.extend(
        [
            "",
            "## Rebuild",
            "",
            "```text",
            _rebuild_command(
                manifest_path=manifest_path,
                manifest_sha=manifest_sha,
                acceptance_path=acceptance_path,
                acceptance_sha=acceptance_sha,
                output_dir=output_dir,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--acceptance-report", required=True, type=Path)
    parser.add_argument("--expected-acceptance-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def run(args: argparse.Namespace) -> Path:
    manifest_path, manifest_sha = _verified_file(
        args.run_manifest,
        args.expected_manifest_sha256,
        description="signed daughter-noise run manifest",
    )
    if manifest_path.name != "run_manifest.json":
        raise RuntimeError("Producer manifest must be named run_manifest.json")
    _verify_manifest_sidecar(manifest_path, manifest_sha)
    manifest = _read_json(manifest_path, description="producer run manifest")
    _verify_manifest_signature(manifest)
    _verify_frozen_settings(manifest)
    acceptance_path, acceptance_sha, _ = _verify_acceptance_binding(
        manifest,
        args.acceptance_report,
        args.expected_acceptance_sha256,
    )
    tables, source_artifact_hashes = _verify_tables_and_raw_states(
        manifest, manifest_path.parent
    )
    midpoint = _validate_csv_semantics(manifest, tables)
    summary = _summarise_midpoints(midpoint)

    output_dir = _prepare_output_dir(args.output_dir)
    summary_path = output_dir / "figure_metrics_by_interval.csv"
    _atomic_csv(summary, summary_path)
    pdf_path = output_dir / f"{FIGURE_BASENAME}.pdf"
    png_path = output_dir / f"{FIGURE_BASENAME}.png"
    _save_figure(summary, pdf_path, png_path)
    caption_path = output_dir / "figure_caption.md"
    _atomic_text(_caption_text(summary), caption_path)

    pre_provenance_artifacts = {
        "figure_pdf": _artifact(pdf_path, root=output_dir),
        "figure_png": _artifact(png_path, root=output_dir),
        "figure_caption": _artifact(caption_path, root=output_dir),
        "figure_metrics_by_interval": _artifact(summary_path, root=output_dir),
    }
    provenance_path = output_dir / "PROVENANCE.md"
    _atomic_text(
        _provenance_text(
            manifest_path=manifest_path,
            manifest_sha=manifest_sha,
            acceptance_path=acceptance_path,
            acceptance_sha=acceptance_sha,
            summary_path=summary_path,
            figure_artifacts=pre_provenance_artifacts,
            output_dir=output_dir,
        ),
        provenance_path,
    )
    output_artifacts = {
        **pre_provenance_artifacts,
        "provenance": _artifact(provenance_path, root=output_dir),
    }
    covered_payload = {
        "schema_version": 1,
        "analysis": FIGURE_ANALYSIS_ID,
        "source": {
            "run_manifest_path": str(manifest_path),
            "run_manifest_sha256": manifest_sha,
            "run_manifest_signature": manifest["signature"]["value"],
            "acceptance_report_path": str(acceptance_path),
            "acceptance_report_sha256": acceptance_sha,
            "verified_source_artifact_sha256": source_artifact_hashes,
        },
        "plot_contract": {
            "forecast_role": MIDPOINT_ROLE,
            "baseline_daughter_noise_std": 0.0,
            "daughter_noise_std": list(NOISE_VALUES[1:]),
            "paired_seeds": list(PAIRED_SEEDS),
            "estimator": "arithmetic mean plus/minus SEM across paired seeds",
            "panels": list(PAIRED_METRICS),
            "figure_size_inches": [8.27, 11.69],
            "png_dpi": 320,
        },
        "outputs": output_artifacts,
        "code": {
            "plotter_path": str(Path(__file__).resolve()),
            "plotter_sha256": _sha256(Path(__file__).resolve()),
            "git": _git_state(Path(__file__).resolve().parents[1]),
        },
    }
    bundle_manifest = {
        **covered_payload,
        "status": "complete",
        "completed_at": _utc_now(),
        "signature": {
            "algorithm": "sha256-canonical-json",
            "value": _stable_json_sha256(covered_payload),
            "covered_top_level_fields": list(covered_payload),
            "excludes": [
                "status",
                "completed_at",
                "signature",
                "figure_manifest.json self hash",
            ],
        },
    }
    bundle_manifest_path = output_dir / "figure_manifest.json"
    _atomic_json(bundle_manifest, bundle_manifest_path)
    bundle_sha = _sha256(bundle_manifest_path)
    _atomic_text(
        f"{bundle_sha}  figure_manifest.json\n",
        output_dir / "figure_manifest.sha256",
    )
    print(
        f"Saved publication bundle {output_dir} "
        f"(figure_manifest_sha256={bundle_sha})",
        flush=True,
    )
    return bundle_manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
