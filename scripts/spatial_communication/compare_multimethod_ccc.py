#!/usr/bin/env python3
"""Compare directed cell-type communication rankings across reviewer methods.

The methods in this workflow do not share a numerical scale.  This script
therefore never pools or contrasts their raw scores.  It standardizes only the
directed ``stage, sender_type, receiver_type`` key, ranks scores within each
method/condition/stage, and reports rank concordance, top-edge overlap,
reciprocal-direction agreement, and stage stability.

CytoBridge attention is an internal signed-gate magnitude, not a calibrated
cell-cell-communication probability.  Its exact one-layer message contribution
is retained as a separate score view.  NicheNet is likewise represented by a
clearly labelled, derived sender-associated ligand-activity summary rather than
as a native spatial type-pair probability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import md5, sha256
from itertools import product
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["stage", "sender_type", "receiver_type"]
EXPECTED_FULL_GRID_STAGES = (0.0, 1.0, 2.0, 3.0, 4.0)
STRICT_PERMUTATION_STRATA = "stage+sender_type+receiver_type+distance_bin+state_bin"
CELLAGENTCHAT_OFFICIAL = "official_mouse_default_celltalkdb"
CELLAGENTCHAT_CUSTOM = "cytobridge_zebrafish_lr_projected_singletons"
PINNED_CELLAGENTCHAT_COMMIT = "310cfc03df91c5ec917f110801e0c2ae4ab57800"
PINNED_CELLCHAT_COMMIT = "75253cd0c9e68410e6e721a6d3a0419a1d7e358f"
PINNED_NICHENETR_COMMIT = "66f90d5eeafef280b2b2f339b3fd70ffec1781dd"


@dataclass(frozen=True)
class ScoreView:
    view_id: str
    display_label: str
    method: str
    database_condition: str
    score_view: str
    path: Path
    loader: Callable[[Path], pd.DataFrame]
    raw_units_cross_method_comparable: bool = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cytobridge-dir", type=Path)
    parser.add_argument("--controls-trained-dir", type=Path)
    parser.add_argument("--controls-init-dir", type=Path)
    parser.add_argument("--controls-random-dir", type=Path)
    parser.add_argument("--commot-dir", type=Path)
    parser.add_argument("--cellchat-dir", type=Path)
    parser.add_argument("--nichenet-default-dir", type=Path)
    parser.add_argument("--nichenet-custom-dir", type=Path)
    parser.add_argument("--cellagentchat-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Diagnostic only: record and skip missing/malformed method outputs. "
            "The resulting manifest is partial_diagnostic, not reviewer-ready."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")


def _numeric(values: pd.Series, *, name: str, path: Path) -> pd.Series:
    result = pd.to_numeric(values, errors="coerce")
    if result.isna().any() or not np.isfinite(result.to_numpy(float)).all():
        raise ValueError(f"{path} contains non-finite/non-numeric {name} values")
    return result.astype(float)


def _stage_values(values: pd.Series, *, path: Path) -> pd.Series:
    result = _numeric(values, name="stage", path=path)
    rounded = np.round(result.to_numpy(float), 12)
    return pd.Series(rounded, index=values.index, dtype=float)


def _clean_labels(values: pd.Series, *, name: str, path: Path) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{path} contains missing {name}")
    result = values.astype(str).str.strip()
    if (result == "").any():
        raise ValueError(f"{path} contains empty {name}")
    return result


def _validate_manifest_identity(
    directory: Path,
    *,
    filename: str,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest = _read_json(directory / filename)
    for key, wanted in expected.items():
        observed = manifest.get(key)
        if observed != wanted:
            raise ValueError(
                f"{directory / filename}: {key!r}={observed!r}, expected {wanted!r}"
            )
    return manifest


def _verify_manifest_artifact(
    directory: Path,
    record: Mapping[str, Any],
    *,
    expected_name: str,
) -> Path:
    """Resolve a colocated manifest artifact and verify its recorded identity."""

    if not isinstance(record, Mapping):
        raise ValueError(f"Manifest artifact {expected_name!r} is not an object")
    recorded_path = Path(str(record.get("path", "")))
    if recorded_path.name != expected_name:
        raise ValueError(
            f"Manifest artifact names {recorded_path.name!r}; expected {expected_name!r}"
        )
    path = directory / expected_name
    if not path.is_file():
        raise FileNotFoundError(path)
    recorded_bytes = record.get("bytes", record.get("size_bytes"))
    if recorded_bytes is None or int(recorded_bytes) != path.stat().st_size:
        raise ValueError(f"Manifest byte count does not match {path}")
    recorded_sha256 = str(record.get("sha256", "")).casefold()
    if not recorded_sha256 or recorded_sha256 != _sha256(path).casefold():
        raise ValueError(f"Manifest SHA256 does not match {path}")
    return path


def _verify_manifest_artifact_key(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    artifact_key: str,
    expected_name: str,
) -> tuple[Path, dict[str, Any]]:
    """Fail closed on a colocated SHA256-bound manifest artifact."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"{directory} manifest lacks an artifacts object")
    path = _verify_manifest_artifact(
        directory,
        artifacts.get(artifact_key, {}),
        expected_name=expected_name,
    )
    return path, {
        "manifest_artifact_key": artifact_key,
        "hash_algorithm": "sha256",
        "verified": True,
        **_file_record(path),
    }


def _verify_recorded_artifact(
    record: Mapping[str, Any], *, expected_name: str | None = None
) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError("Manifest artifact record is not an object")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if expected_name is not None and path.name != expected_name:
        raise ValueError(
            f"Manifest artifact names {path.name!r}; expected {expected_name!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    recorded_bytes = record.get("bytes", record.get("size_bytes"))
    if recorded_bytes is None or int(recorded_bytes) != path.stat().st_size:
        raise ValueError(f"Manifest byte count does not match {path}")
    recorded_sha256 = str(record.get("sha256", "")).casefold()
    if not recorded_sha256 or recorded_sha256 != _sha256(path).casefold():
        raise ValueError(f"Manifest SHA256 does not match {path}")
    return path


def _format_hpf(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("Developmental time must be finite")
    return f"{value:g}hpf"


def _verified_external_type_pair_universe(
    manifest: Mapping[str, Any],
) -> tuple[Path, pd.DataFrame]:
    """Build the evaluated directed grid from the hash-bound shared input.

    COMMOT and CellChat evaluate every sender/receiver cell-type combination at
    every prepared stage.  Their historical long tables emitted positive
    aggregates only, so the only defensible zeros are missing rows in this
    independently verified stage-specific ``cell_type_counts`` universe.
    """

    input_manifest_path = _verify_recorded_artifact(
        manifest.get("input_manifest", {}), expected_name="input_manifest.json"
    )
    shared = _read_json(input_manifest_path)
    stages = shared.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"{input_manifest_path} has no stage records")
    rows: list[dict[str, Any]] = []
    observed_stages: list[float] = []
    for record in stages:
        if not isinstance(record, Mapping):
            raise ValueError(
                f"{input_manifest_path} contains a non-object stage record"
            )
        try:
            stage_id = float(record.get("stage"))
            stage_time = float(record.get("stage_time"))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{input_manifest_path} has an invalid stage mapping"
            ) from error
        if not np.isfinite(stage_id) or not np.isfinite(stage_time):
            raise ValueError(f"{input_manifest_path} has a non-finite stage mapping")
        stage_id = float(np.round(stage_id, 12))
        if stage_id in observed_stages:
            raise ValueError(f"{input_manifest_path} repeats stage {stage_id:g}")
        observed_stages.append(stage_id)
        counts = record.get("cell_type_counts")
        if not isinstance(counts, Mapping) or not counts:
            raise ValueError(
                f"{input_manifest_path} stage {stage_id:g} lacks cell_type_counts"
            )
        cleaned_counts: dict[str, int] = {}
        for raw_label, raw_count in counts.items():
            label = str(raw_label).strip()
            if not label:
                raise ValueError(
                    f"{input_manifest_path} stage {stage_id:g} has an empty cell type"
                )
            if isinstance(raw_count, bool):
                raise ValueError(
                    f"{input_manifest_path} stage {stage_id:g} has an invalid count"
                )
            try:
                count = int(raw_count)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{input_manifest_path} stage {stage_id:g} has an invalid count"
                ) from error
            if count <= 0 or float(raw_count) != float(count):
                raise ValueError(
                    f"{input_manifest_path} stage {stage_id:g} has a non-positive/non-integer count"
                )
            if label in cleaned_counts:
                raise ValueError(
                    f"{input_manifest_path} stage {stage_id:g} repeats cell type {label!r}"
                )
            cleaned_counts[label] = count
        recorded_n_types = record.get("n_cell_types")
        if recorded_n_types is not None and int(recorded_n_types) != len(
            cleaned_counts
        ):
            raise ValueError(
                f"{input_manifest_path} stage {stage_id:g} n_cell_types disagrees with cell_type_counts"
            )
        recorded_n_cells = record.get("n_cells")
        if recorded_n_cells is not None and int(recorded_n_cells) != sum(
            cleaned_counts.values()
        ):
            raise ValueError(
                f"{input_manifest_path} stage {stage_id:g} n_cells disagrees with cell_type_counts"
            )
        for sender, receiver in product(sorted(cleaned_counts), repeat=2):
            rows.append(
                {
                    "stage": stage_id,
                    "stage_time": stage_time,
                    "stage_label": _format_hpf(stage_time),
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "n_sender_cells": cleaned_counts[sender],
                    "n_receiver_cells": cleaned_counts[receiver],
                    "n_cell_types": len(cleaned_counts),
                }
            )
    if tuple(sorted(observed_stages)) != EXPECTED_FULL_GRID_STAGES:
        raise ValueError(
            f"{input_manifest_path} stages={sorted(observed_stages)!r}; expected the five "
            f"observed stages {list(EXPECTED_FULL_GRID_STAGES)!r}"
        )
    universe = pd.DataFrame(rows)
    if universe.duplicated(KEYS).any():
        raise ValueError(
            f"{input_manifest_path} creates duplicate directed type-pair keys"
        )
    return input_manifest_path, universe.sort_values(KEYS).reset_index(drop=True)


def _external_stage_contract(
    universe: pd.DataFrame, frame: pd.DataFrame, *, score_path: Path
) -> tuple[pd.Series, pd.Series]:
    """Map emitted runner stage IDs to verified manuscript hpf labels."""

    stage_mapping = universe[["stage", "stage_time", "stage_label"]].drop_duplicates()
    mapping = stage_mapping.set_index("stage")["stage_time"]
    labels = stage_mapping.set_index("stage")["stage_label"]
    stage = _stage_values(frame["stage"], path=score_path)
    observed_time = _numeric(frame["stage_time"], name="stage_time", path=score_path)
    expected_time = stage.map(mapping)
    if expected_time.isna().any():
        unknown = sorted(set(stage.loc[expected_time.isna()]))
        raise ValueError(
            f"{score_path} contains stages absent from shared inputs: {unknown}"
        )
    if not np.allclose(
        observed_time.to_numpy(float),
        expected_time.to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{score_path} stage_time values disagree with the verified shared input manifest"
        )
    return stage, stage.map(labels)


def _complete_external_positive_only_grid(
    emitted: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    universe: pd.DataFrame,
    input_manifest_path: Path,
    score_path: Path,
) -> pd.DataFrame:
    """Complete only verified positive-only aggregation gaps with native zero."""

    if (emitted["native_score"] < 0).any():
        raise ValueError(f"{score_path} contains a negative communication score")
    unknown = emitted[KEYS].merge(universe[KEYS], on=KEYS, how="left", indicator=True)
    unknown = unknown.loc[unknown["_merge"] == "left_only", KEYS]
    if not unknown.empty:
        raise ValueError(
            f"{score_path} contains keys outside the verified stage/type universe: "
            f"{unknown.head(5).to_dict('records')}"
        )
    template = universe[["stage", "stage_label", "sender_type", "receiver_type"]]
    completed = template.merge(
        emitted.drop(columns=["stage_label", "heterotypic"]),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    missing = completed["native_score"].isna()
    export_contract = manifest.get("design", {}).get("type_pair_grid_export", {})
    declared_complete = bool(
        isinstance(export_contract, Mapping)
        and export_contract.get("complete_directed_stage_type_square") is True
    )
    positive_only_policy = str(
        manifest.get("design", {}).get("long_table_zero_policy", "")
    ).casefold()
    explicitly_positive_only = (
        "structural zeros omitted" in positive_only_policy
        and "fill zero for comparisons" in positive_only_policy
    )
    if missing.any() and declared_complete:
        raise ValueError(
            f"{score_path} is missing {int(missing.sum())} rows despite declaring a complete grid"
        )
    if missing.any() and not explicitly_positive_only:
        raise ValueError(
            f"{score_path} is missing {int(missing.sum())} evaluated type-pair rows, but "
            "the runner manifest does not explicitly declare positive-only aggregation"
        )
    completed["native_score"] = completed["native_score"].fillna(0.0)
    completed["structural_zero_filled"] = missing.to_numpy(bool)
    completed["heterotypic"] = completed["sender_type"] != completed["receiver_type"]
    metadata_columns = [
        "view_id",
        "display_label",
        "method",
        "database_condition",
        "score_view",
    ]
    for column in metadata_columns:
        values = emitted[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(f"{score_path} has no unique {column} metadata value")
        completed[column] = values.iloc[0]
    completed = (
        completed[
            metadata_columns
            + [
                "stage",
                "stage_label",
                "sender_type",
                "receiver_type",
                "native_score",
                "structural_zero_filled",
                "heterotypic",
            ]
        ]
        .sort_values(KEYS, kind="mergesort")
        .reset_index(drop=True)
    )

    stage_audit: list[dict[str, Any]] = []
    for stage, grid in completed.groupby("stage", sort=True):
        source = emitted.loc[emitted["stage"] == stage]
        n_types = int(universe.loc[universe["stage"] == stage, "n_cell_types"].iloc[0])
        stage_audit.append(
            {
                "stage": float(stage),
                "stage_label": str(grid["stage_label"].iloc[0]),
                "receiver_unit": "",
                "n_cell_types": n_types,
                "expected_directed_rows": int(n_types**2),
                "native_emitted_rows": int(len(source)),
                "native_emitted_positive_rows": int((source["native_score"] > 0).sum()),
                "native_emitted_zero_rows": int((source["native_score"] == 0).sum()),
                "structural_zero_filled_rows": int(
                    grid["structural_zero_filled"].sum()
                ),
                "verified_complete_evaluated_universe": True,
            }
        )
    completed.attrs["zero_completion"] = {
        "universe_scope": "verified_stage_specific_cell_type_square",
        "universe_source": _file_record(input_manifest_path),
        "runner_export_contract": (
            "complete_directed_grid"
            if declared_complete
            else "positive_only_aggregation"
        ),
        "full_stage_type_square_required": True,
        "expected_stage_count": len(EXPECTED_FULL_GRID_STAGES),
        "observed_stage_count": int(completed["stage"].nunique()),
        "expected_rows": int(len(completed)),
        "native_emitted_rows": int(len(emitted)),
        "structural_zero_filled_rows": int(missing.sum()),
        "verified_complete_evaluated_universe": True,
        "unevaluated_units_zero_filled": False,
        "method_unavailable_lr_rows_zero_filled": False,
        "audit_rows": stage_audit,
    }
    return completed


def _canonical(
    frame: pd.DataFrame,
    *,
    path: Path,
    method: str,
    database_condition: str,
    score_view: str,
    display_label: str,
    view_id: str,
    score: pd.Series,
    stage: pd.Series,
    stage_label: pd.Series,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "view_id": view_id,
            "display_label": display_label,
            "method": method,
            "database_condition": database_condition,
            "score_view": score_view,
            "stage": _stage_values(stage, path=path),
            "stage_label": _clean_labels(stage_label, name="stage_label", path=path),
            "sender_type": _clean_labels(
                frame["sender_type"], name="sender_type", path=path
            ),
            "receiver_type": _clean_labels(
                frame["receiver_type"], name="receiver_type", path=path
            ),
            "native_score": _numeric(score, name="native_score", path=path),
        }
    )
    duplicate = result.duplicated(KEYS, keep=False)
    if duplicate.any():
        examples = result.loc[duplicate, KEYS].head(5).to_dict("records")
        raise ValueError(
            f"{path} has duplicate directed keys after adaptation: {examples}"
        )
    labels_per_stage = result.groupby("stage", sort=False)["stage_label"].nunique()
    if (labels_per_stage != 1).any():
        raise ValueError(f"{path} maps a numeric stage to multiple labels")
    result["heterotypic"] = result["sender_type"] != result["receiver_type"]
    return result.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def _attach_native_no_completion_audit(
    frame: pd.DataFrame, *, universe_scope: str
) -> pd.DataFrame:
    """Record that a score view was retained exactly as natively emitted."""

    result = frame.copy()
    result["structural_zero_filled"] = False
    audit_rows: list[dict[str, Any]] = []
    for stage, stage_frame in result.groupby("stage", sort=True):
        audit_rows.append(
            {
                "stage": float(stage),
                "stage_label": str(stage_frame["stage_label"].iloc[0]),
                "receiver_unit": "",
                "n_cell_types": int(
                    len(
                        set(stage_frame["sender_type"])
                        | set(stage_frame["receiver_type"])
                    )
                ),
                "expected_directed_rows": int(len(stage_frame)),
                "native_emitted_rows": int(len(stage_frame)),
                "native_emitted_positive_rows": int(
                    (stage_frame["native_score"] > 0).sum()
                ),
                "native_emitted_zero_rows": int(
                    (stage_frame["native_score"] == 0).sum()
                ),
                "structural_zero_filled_rows": 0,
                "verified_complete_evaluated_universe": False,
            }
        )
    result.attrs["zero_completion"] = {
        "universe_scope": universe_scope,
        "universe_source": None,
        "runner_export_contract": "native_emitted_rows_no_loader_completion",
        "full_stage_type_square_required": False,
        "expected_stage_count": None,
        "observed_stage_count": int(result["stage"].nunique()),
        "expected_rows": int(len(result)),
        "native_emitted_rows": int(len(result)),
        "structural_zero_filled_rows": 0,
        "verified_complete_evaluated_universe": False,
        "unevaluated_units_zero_filled": False,
        "method_unavailable_lr_rows_zero_filled": False,
        "audit_rows": audit_rows,
    }
    return result


def _load_cytobridge_views(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = _validate_manifest_identity(
        directory,
        filename="run_manifest.json",
        expected={"method": "cytobridge_one_layer_spatial_attention_and_exact_message"},
    )
    if manifest.get("interpretation", {}).get("probability_claim") is not False:
        raise ValueError(
            "CytoBridge manifest must explicitly set probability_claim=false"
        )
    path, score_artifact = _verify_manifest_artifact_key(
        directory,
        manifest,
        artifact_key="type_pair_summary",
        expected_name="type_pair_summary.csv",
    )
    frame = pd.read_csv(path)
    required = {
        "stage",
        "stage_label",
        "sender_type",
        "receiver_type",
        "G_AB_attention_mean_mean",
        "D_AB_joint_mean",
    }
    _require_columns(frame, required, path)
    attention = _canonical(
        frame,
        path=path,
        method="CytoBridge",
        database_condition="trained_project_lr_edge_prior",
        score_view="mean_absolute_attention_gate",
        display_label="CytoBridge attention",
        view_id="cytobridge__trained__attention",
        score=frame["G_AB_attention_mean_mean"],
        stage=frame["stage"],
        stage_label=frame["stage_label"],
    )
    message = _canonical(
        frame,
        path=path,
        method="CytoBridge",
        database_condition="trained_project_lr_edge_prior",
        score_view="exact_joint_message_contribution",
        display_label="CytoBridge exact message",
        view_id="cytobridge__trained__exact_message",
        score=frame["D_AB_joint_mean"],
        stage=frame["stage"],
        stage_label=frame["stage_label"],
    )
    attention = _attach_native_no_completion_audit(
        attention, universe_scope="native_cytobridge_type_pair_summary"
    )
    message = _attach_native_no_completion_audit(
        message, universe_scope="native_cytobridge_type_pair_summary"
    )
    attention.attrs["verified_primary_score_artifacts"] = [score_artifact]
    message.attrs["verified_primary_score_artifacts"] = [score_artifact]
    return attention, message


def _load_commot(directory: Path) -> pd.DataFrame:
    manifest = _validate_manifest_identity(
        directory,
        filename="manifest.json",
        expected={
            "method": "COMMOT",
            "database_variant": "current_zebrafish_lr_database",
        },
    )
    path, score_artifact = _verify_manifest_artifact_key(
        directory,
        manifest,
        artifact_key="type_pair_scores",
        expected_name="commot_type_pair_scores.csv.gz",
    )
    frame = pd.read_csv(path)
    required = {
        "method",
        "database_variant",
        "stage",
        "stage_time",
        "sender_type",
        "receiver_type",
        "abundance_controlled_score",
    }
    _require_columns(frame, required, path)
    if set(frame["method"].astype(str)) != {"COMMOT"}:
        raise ValueError(f"{path} contains an unexpected method label")
    if set(frame["database_variant"].astype(str)) != {"current_zebrafish_lr_database"}:
        raise ValueError(f"{path} contains an unexpected database variant")
    input_manifest_path, universe = _verified_external_type_pair_universe(manifest)
    stage, stage_label = _external_stage_contract(universe, frame, score_path=path)
    emitted = _canonical(
        frame,
        path=path,
        method="COMMOT",
        database_condition="current_zebrafish_lr_database",
        score_view="mean_communication_mass_per_possible_cell_pair",
        display_label="COMMOT | project LR",
        view_id="commot__project_lr",
        score=frame["abundance_controlled_score"],
        stage=stage,
        stage_label=stage_label,
    )
    result = _complete_external_positive_only_grid(
        emitted,
        manifest=manifest,
        universe=universe,
        input_manifest_path=input_manifest_path,
        score_path=path,
    )
    result.attrs["verified_primary_score_artifacts"] = [score_artifact]
    return result


def _load_cellchat(directory: Path) -> pd.DataFrame:
    manifest = _validate_manifest_identity(
        directory,
        filename="manifest.json",
        expected={
            "method": "CellChat",
            "database_variant": "current_zebrafish_lr_database",
        },
    )
    design = manifest.get("design")
    software = manifest.get("software")
    if not isinstance(design, Mapping) or not isinstance(software, Mapping):
        raise ValueError("CellChat manifest lacks design/software provenance")
    if (
        design.get("population_size") is not False
        or int(design.get("nboot", -1)) != 100
        or str(design.get("mean_method", "")) != "triMean"
        or design.get("raw_use") is not True
    ):
        raise ValueError(
            "CellChat score label requires population_size=false, nboot=100, "
            "mean_method=triMean, and raw_use=true"
        )
    cellchat_commit = str(software.get("CellChat_source_commit", "")).casefold()
    if (
        str(software.get("CellChat_load_mode", "")) != "pinned official core R source"
        or str(software.get("CellChat", "")) != "2.2.0.9001"
        or cellchat_commit != PINNED_CELLCHAT_COMMIT
    ):
        raise ValueError(
            "CellChat manifest does not identify the pinned official source"
        )
    validation = manifest.get("database_validation")
    if not isinstance(validation, Mapping):
        raise ValueError("CellChat manifest lacks database_validation")
    requested = int(validation.get("rows_requested", -1))
    eligible = int(validation.get("rows_eligible", -1))
    excluded = int(validation.get("rows_excluded", -1))
    if min(requested, eligible, excluded) < 0 or requested != eligible + excluded:
        raise ValueError(
            "CellChat manifest has inconsistent requested/eligible/excluded counts"
        )
    if (
        validation.get("excluded_rows_are_method_unavailable_not_biological_zero")
        is not True
    ):
        raise ValueError(
            "CellChat manifest must mark excluded rows as method-unavailable, not zero"
        )
    policy = str(manifest.get("design", {}).get("method_unavailable_policy", ""))
    if "never zero-filled" not in policy:
        raise ValueError(
            "CellChat manifest lost the method-unavailable no-zero-fill policy"
        )
    exclusion_path = _verify_manifest_artifact(
        directory,
        validation.get("exclusion_table", {}),
        expected_name="excluded_lr_rows.csv",
    )
    exclusion = pd.read_csv(exclusion_path)
    exclusion_columns = {
        "database_row",
        "interaction_id",
        "current_ligand",
        "current_receptor",
        "cellchat_ligand_token",
        "cellchat_receptor_token",
        "eligible",
        "exclusion_reason",
    }
    _require_columns(exclusion, exclusion_columns, exclusion_path)
    if len(exclusion) != excluded:
        raise ValueError(
            f"{exclusion_path} has {len(exclusion)} rows but manifest records {excluded}"
        )
    if exclusion["database_row"].duplicated().any():
        raise ValueError(f"{exclusion_path} has duplicate database_row values")
    eligible_text = exclusion["eligible"].astype(str).str.casefold()
    if len(exclusion) and not eligible_text.isin({"false", "0"}).all():
        raise ValueError(f"{exclusion_path} contains an eligible row")
    path, score_artifact = _verify_manifest_artifact_key(
        directory,
        manifest,
        artifact_key="cellchat_type_pair_scores.csv.gz",
        expected_name="cellchat_type_pair_scores.csv.gz",
    )
    frame = pd.read_csv(path)
    required = {
        "method",
        "database_variant",
        "stage",
        "stage_time",
        "sender_type",
        "receiver_type",
        "abundance_controlled_score",
    }
    _require_columns(frame, required, path)
    if set(frame["method"].astype(str)) != {"CellChat"}:
        raise ValueError(f"{path} contains an unexpected method label")
    if set(frame["database_variant"].astype(str)) != {"current_zebrafish_lr_database"}:
        raise ValueError(f"{path} contains an unexpected database variant")
    input_manifest_path, universe = _verified_external_type_pair_universe(manifest)
    stage, stage_label = _external_stage_contract(universe, frame, score_path=path)
    emitted = _canonical(
        frame,
        path=path,
        method="CellChat",
        database_condition="current_zebrafish_lr_database",
        score_view="population_size_false_total_probability",
        display_label="CellChat | project LR",
        view_id="cellchat__project_lr",
        score=frame["abundance_controlled_score"],
        stage=stage,
        stage_label=stage_label,
    )
    result = _complete_external_positive_only_grid(
        emitted,
        manifest=manifest,
        universe=universe,
        input_manifest_path=input_manifest_path,
        score_path=path,
    )

    unavailable = exclusion.rename(
        columns={
            "current_ligand": "ligand",
            "current_receptor": "receptor",
            "cellchat_ligand_token": "method_ligand_token",
            "cellchat_receptor_token": "method_receptor_token",
            "exclusion_reason": "reason",
        }
    )[
        [
            "database_row",
            "interaction_id",
            "ligand",
            "receptor",
            "method_ligand_token",
            "method_receptor_token",
            "reason",
        ]
    ].copy()
    unavailable.insert(0, "database_condition", "current_zebrafish_lr_database")
    unavailable.insert(0, "method", "CellChat")
    unavailable.insert(0, "view_id", "cellchat__project_lr")
    unavailable["status"] = "method_unavailable_excluded_not_biological_zero"
    unavailable["zero_filled"] = False
    result.attrs["method_unavailable_lr_rows"] = unavailable
    result.attrs["cellchat_lr_universe"] = {
        "rows_requested": requested,
        "rows_eligible": eligible,
        "rows_method_unavailable": excluded,
        "method_unavailable_rows_zero_filled": False,
    }
    result.attrs["verified_primary_score_artifacts"] = [score_artifact]
    return result


def _verify_nichenet_output(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    key: str,
    filename: str,
) -> Path:
    output_files = manifest.get("output_files")
    output_md5 = manifest.get("output_md5")
    if not isinstance(output_files, Mapping) or not isinstance(output_md5, Mapping):
        raise ValueError("NicheNet manifest lacks output_files/output_md5")
    recorded = Path(str(output_files.get(key, "")))
    if recorded.name != filename:
        raise ValueError(f"NicheNet manifest does not bind {key!r} to {filename!r}")
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_md5 = str(output_md5.get(key, "")).casefold()
    if len(expected_md5) != 32 or expected_md5 != _md5(path).casefold():
        raise ValueError(f"NicheNet manifest MD5 does not match {path}")
    return path


def _verify_nichenet_shared_inputs(
    manifest: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any], Path]:
    record = manifest.get("shared_prepare_manifest")
    if not isinstance(record, Mapping):
        raise ValueError("NicheNet manifest lacks shared_prepare_manifest")
    prepare_path = Path(str(record.get("path", ""))).expanduser().resolve()
    if prepare_path.name != "prepare_manifest.json" or not prepare_path.is_file():
        raise FileNotFoundError(prepare_path)
    recorded_md5 = str(record.get("md5", "")).casefold()
    if len(recorded_md5) != 32 or recorded_md5 != _md5(prepare_path).casefold():
        raise ValueError("NicheNet shared prepare manifest MD5 mismatch")
    prepare = _read_json(prepare_path)
    if (
        prepare.get("workflow")
        != "reviewer_zebrafish_nichenet_shared_input_preparation"
        or prepare.get("status") != "complete"
    ):
        raise ValueError(
            "NicheNet shared prepare manifest is not a completed preparation"
        )
    for field in ("orthology_policy", "analysis_tier", "primary_claim_allowed"):
        if prepare.get(field) != manifest.get(field):
            raise ValueError(f"NicheNet run and shared preparation disagree on {field}")
    records = prepare.get("output_files")
    if not isinstance(records, list):
        raise ValueError("NicheNet shared prepare manifest lacks output_files")
    matches = [
        item
        for item in records
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).name
        == "expression_by_stage_celltype.csv.gz"
    ]
    if len(matches) != 1:
        raise ValueError(
            "NicheNet shared prepare manifest must inventory expression_by_stage_celltype.csv.gz once"
        )
    item = matches[0]
    relative = Path(str(item.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("NicheNet shared input artifact path escapes shared directory")
    expression_path = (prepare_path.parent / relative).resolve()
    try:
        expression_path.relative_to(prepare_path.parent.resolve())
    except ValueError as error:
        raise ValueError(
            "NicheNet shared input artifact path escapes shared directory"
        ) from error
    if not expression_path.is_file():
        raise FileNotFoundError(expression_path)
    if int(item.get("size_bytes", -1)) != expression_path.stat().st_size:
        raise ValueError("NicheNet shared expression size mismatch")
    if str(item.get("sha256", "")).casefold() != _sha256(expression_path).casefold():
        raise ValueError("NicheNet shared expression SHA256 mismatch")
    if str(item.get("md5", "")).casefold() != _md5(expression_path).casefold():
        raise ValueError("NicheNet shared expression MD5 mismatch")
    return prepare_path, prepare, expression_path


def _load_nichenet(directory: Path, *, mode: str) -> pd.DataFrame:
    manifest = _validate_manifest_identity(
        directory,
        filename="run_manifest.json",
        expected={
            "workflow": "reviewer_zebrafish_cross_species_nichenet_v2",
            "status": "complete",
            "mode": mode,
        },
    )
    if "sender-specific" not in str(manifest.get("activity_semantics", "")):
        raise ValueError("NicheNet manifest must retain its sender-specificity caveat")
    orthology_policy = str(manifest.get("orthology_policy", ""))
    policy_contract = {
        "strict_confidence1": {
            "analysis_tier": "primary",
            "primary_claim_allowed": True,
            "condition_suffix": "strict_confidence1_orthology",
            "label_suffix": "strict confidence=1 orthology",
        },
        "one2one_bijective_all_confidence": {
            "analysis_tier": "sensitivity",
            "primary_claim_allowed": False,
            "condition_suffix": "one2one_all_confidence_sensitivity",
            "label_suffix": "all-confidence orthology sensitivity",
        },
    }
    if orthology_policy not in policy_contract:
        raise ValueError(
            f"NicheNet manifest has unsupported orthology_policy={orthology_policy!r}"
        )
    policy = policy_contract[orthology_policy]
    if manifest.get("analysis_tier") != policy["analysis_tier"]:
        raise ValueError("NicheNet analysis_tier disagrees with orthology_policy")
    if manifest.get("primary_claim_allowed") is not policy["primary_claim_allowed"]:
        raise ValueError(
            "NicheNet primary_claim_allowed disagrees with orthology_policy"
        )
    method_label = str(manifest.get("method_label", ""))
    if orthology_policy == "one2one_bijective_all_confidence" and (
        "orthology sensitivity: confidence unfiltered" not in method_label
    ):
        raise ValueError(
            "All-confidence NicheNet output must explicitly label its orthology sensitivity"
        )
    prior = manifest.get("official_prior")
    engine = manifest.get("software", {}).get("nichenetr")
    if not isinstance(prior, Mapping) or not isinstance(engine, Mapping):
        raise ValueError("NicheNet manifest lacks frozen prior/source provenance")
    if prior.get("md5_verified") is not True or prior.get("expected_md5") != prior.get(
        "observed_md5"
    ):
        raise ValueError("NicheNet official prior MD5 contract is not verified")
    if (
        engine.get("mode") != "pinned_core_source"
        or str(engine.get("version", "")) != "2.2.1.1"
        or engine.get("version_verified") is not True
        or str(engine.get("git_commit", "")).casefold() != PINNED_NICHENETR_COMMIT
        or str(engine.get("expected_git_commit", "")).casefold()
        != PINNED_NICHENETR_COMMIT
        or engine.get("commit_verified") is not True
        or engine.get("core_md5_verified") is not True
    ):
        raise ValueError("NicheNet pinned source contract is not verified")
    prior_source_signature = sha256(
        json.dumps(
            {"official_prior": prior, "nichenetr": engine},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    prepare_path, _, expression_path = _verify_nichenet_shared_inputs(manifest)
    path = _verify_nichenet_output(
        directory,
        manifest,
        key="sender_ligand_activity",
        filename="sender_ligand_activity.csv",
    )
    unit_status_path = _verify_nichenet_output(
        directory,
        manifest,
        key="unit_status",
        filename="unit_status.csv",
    )
    frame = pd.read_csv(path)
    required = {
        "unit_id",
        "source_stage_id",
        "source_stage_label",
        "receiver",
        "mode",
        "sender",
        "ligand",
        "aupr_corrected",
        "activity_scope",
    }
    _require_columns(frame, required, path)
    if frame.empty:
        raise ValueError(f"{path} contains no completed sender-ligand activities")
    if set(frame["mode"].astype(str)) != {mode}:
        raise ValueError(f"{path} mode differs from requested {mode!r}")
    if (
        not frame["activity_scope"]
        .astype(str)
        .str.contains("not_sender_specific", regex=False)
        .all()
    ):
        raise ValueError(f"{path} lost the NicheNet activity-scope caveat")

    unit_status = pd.read_csv(unit_status_path)
    unit_columns = {
        "unit_id",
        "source_stage_id",
        "source_stage_label",
        "receiver",
        "mode",
        "status",
    }
    _require_columns(unit_status, unit_columns, unit_status_path)
    if unit_status["unit_id"].astype(str).duplicated().any():
        raise ValueError(f"{unit_status_path} contains duplicate unit_id values")
    if set(unit_status["mode"].astype(str)) != {mode}:
        raise ValueError(f"{unit_status_path} mode differs from requested {mode!r}")
    complete_units = unit_status.loc[
        unit_status["status"].astype(str) == "complete"
    ].copy()
    recorded_complete = int(manifest.get("counts", {}).get("units_complete", -1))
    if recorded_complete != len(complete_units) or complete_units.empty:
        raise ValueError(
            f"{unit_status_path} complete-unit count disagrees with run manifest"
        )
    if complete_units.duplicated(["source_stage_id", "receiver"]).any():
        raise ValueError(
            "NicheNet has multiple completed transitions for one source-stage/receiver; "
            "the directed stage/type key cannot represent the target transition"
        )
    activity_units = set(frame["unit_id"].astype(str))
    complete_ids = set(complete_units["unit_id"].astype(str))
    if not activity_units.issubset(complete_ids):
        raise ValueError(
            f"{path} contains sender activities from skipped, ineligible, or failed units"
        )

    expression = pd.read_csv(expression_path)
    expression_columns = {"stage_id", "stage_label", "cell_type", "n_cells"}
    _require_columns(expression, expression_columns, expression_path)
    expression["stage_id"] = _stage_values(expression["stage_id"], path=expression_path)
    expression["stage_label"] = _clean_labels(
        expression["stage_label"], name="stage_label", path=expression_path
    )
    expression["cell_type"] = _clean_labels(
        expression["cell_type"], name="cell_type", path=expression_path
    )
    expression["n_cells"] = _numeric(
        expression["n_cells"], name="n_cells", path=expression_path
    )
    if (expression["n_cells"] <= 0).any():
        raise ValueError(f"{expression_path} contains non-positive cell counts")
    group_check = expression.groupby(["stage_id", "cell_type"], sort=False).agg(
        n_counts=("n_cells", "nunique"),
        n_labels=("stage_label", "nunique"),
    )
    if (group_check[["n_counts", "n_labels"]] != 1).any().any():
        raise ValueError(
            f"{expression_path} has inconsistent stage/type count or label metadata"
        )
    stage_types = expression[
        ["stage_id", "stage_label", "cell_type", "n_cells"]
    ].drop_duplicates()

    work = frame.rename(
        columns={"sender": "sender_type", "receiver": "receiver_type"}
    ).copy()
    work["aupr_corrected"] = _numeric(
        work["aupr_corrected"], name="aupr_corrected", path=path
    )
    work["positive_aupr_corrected"] = work["aupr_corrected"].clip(lower=0.0)
    keys = [
        "unit_id",
        "source_stage_id",
        "source_stage_label",
        "sender_type",
        "receiver_type",
    ]
    aggregated = (
        work.groupby(keys, sort=True, dropna=False)["positive_aupr_corrected"]
        .sum()
        .reset_index()
    )
    aggregated["source_stage_id"] = _stage_values(
        aggregated["source_stage_id"], path=path
    )
    complete_units["source_stage_id"] = _stage_values(
        complete_units["source_stage_id"], path=unit_status_path
    )
    completed_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for unit in complete_units.itertuples(index=False):
        unit_id = str(unit.unit_id)
        stage = float(unit.source_stage_id)
        receiver = str(unit.receiver).strip()
        unit_label = str(unit.source_stage_label).strip()
        available = stage_types.loc[stage_types["stage_id"] == stage]
        if available.empty:
            raise ValueError(
                f"{unit_status_path} complete unit {unit_id!r} has no verified source-stage expression"
            )
        labels = available["stage_label"].drop_duplicates().tolist()
        if len(labels) != 1 or labels[0] != unit_label:
            raise ValueError(
                f"{unit_status_path} unit {unit_id!r} stage label disagrees with shared expression"
            )
        if receiver not in set(available["cell_type"]):
            raise ValueError(
                f"{unit_status_path} unit {unit_id!r} receiver is absent from source-stage expression"
            )
        native = aggregated.loc[aggregated["unit_id"].astype(str) == unit_id].copy()
        if not native.empty and (
            set(native["source_stage_id"].astype(float)) != {stage}
            or set(native["source_stage_label"].astype(str)) != {unit_label}
            or set(native["receiver_type"].astype(str)) != {receiver}
        ):
            raise ValueError(
                f"{path} metadata disagrees with completed unit {unit_id!r}"
            )
        if not set(native["sender_type"].astype(str)).issubset(
            set(available["cell_type"].astype(str))
        ):
            raise ValueError(
                f"{path} contains an unverified sender for unit {unit_id!r}"
            )
        native_by_sender = native.set_index("sender_type")[
            "positive_aupr_corrected"
        ].to_dict()
        for sender in sorted(available["cell_type"].astype(str)):
            filled = sender not in native_by_sender
            completed_rows.append(
                {
                    "unit_id": unit_id,
                    "source_stage_id": stage,
                    "source_stage_label": unit_label,
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "positive_aupr_corrected": float(native_by_sender.get(sender, 0.0)),
                    "structural_zero_filled": filled,
                }
            )
        audit_rows.append(
            {
                "stage": stage,
                "stage_label": unit_label,
                "receiver_unit": receiver,
                "n_cell_types": int(available["cell_type"].nunique()),
                "expected_directed_rows": int(available["cell_type"].nunique()),
                "native_emitted_rows": int(len(native)),
                "native_emitted_positive_rows": int(
                    (native["positive_aupr_corrected"] > 0).sum()
                ),
                "native_emitted_zero_rows": int(
                    (native["positive_aupr_corrected"] == 0).sum()
                ),
                "structural_zero_filled_rows": int(
                    available["cell_type"].nunique() - len(native)
                ),
                "verified_complete_evaluated_universe": True,
            }
        )
    completed = pd.DataFrame(completed_rows)
    condition_prefix = (
        "official_mouse_lr_prior" if mode == "default" else "project_lr_mouse_gate"
    )
    condition = f"{condition_prefix}__{policy['condition_suffix']}"
    label_prefix = (
        "NicheNet-v2 | official mouse LR"
        if mode == "default"
        else "NicheNet-v2 | project LR gate"
    )
    label = f"{label_prefix} | {policy['label_suffix']}"
    view_id = (
        "nichenet_v2__official_mouse_lr"
        if mode == "default"
        else "nichenet_v2__project_lr_gate"
    )
    result = _canonical(
        completed,
        path=path,
        method="NicheNet-v2 (cross-species)",
        database_condition=condition,
        score_view="sum_positive_sender_associated_aupr_corrected",
        display_label=label,
        view_id=view_id,
        score=completed["positive_aupr_corrected"],
        stage=completed["source_stage_id"],
        stage_label=completed["source_stage_label"],
    )
    completion_flags = completed.rename(columns={"source_stage_id": "stage"})[
        KEYS + ["structural_zero_filled"]
    ]
    result = result.merge(
        completion_flags,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    result["orthology_policy"] = orthology_policy
    result["analysis_tier"] = str(policy["analysis_tier"])
    result["primary_claim_allowed"] = bool(policy["primary_claim_allowed"])
    result.attrs["nichenet_orthology"] = {
        "orthology_policy": orthology_policy,
        "analysis_tier": policy["analysis_tier"],
        "primary_claim_allowed": policy["primary_claim_allowed"],
        "method_label": method_label,
        "shared_prepare_manifest_md5": _md5(prepare_path),
        "prior_source_signature_sha256": prior_source_signature,
    }
    result.attrs["zero_completion"] = {
        "universe_scope": "verified_complete_nichenet_unit_source_sender_types",
        "universe_source": _file_record(expression_path),
        "runner_export_contract": "complete_units_only_sender_assignment_completion",
        "full_stage_type_square_required": False,
        "expected_stage_count": None,
        "observed_stage_count": int(result["stage"].nunique()),
        "expected_rows": int(len(result)),
        "native_emitted_rows": int(len(aggregated)),
        "structural_zero_filled_rows": int(result["structural_zero_filled"].sum()),
        "verified_complete_evaluated_universe": True,
        "unevaluated_units_zero_filled": False,
        "method_unavailable_lr_rows_zero_filled": False,
        "completed_units": int(len(complete_units)),
        "skipped_or_ineligible_units": int(len(unit_status) - len(complete_units)),
        "audit_rows": audit_rows,
    }
    result.attrs["verified_primary_score_artifacts"] = [
        {
            "manifest_artifact_key": "output_files.sender_ligand_activity",
            "hash_algorithm": "md5",
            "verified": True,
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "md5": _md5(path),
        }
    ]
    return result


def _load_cellagentchat(directory: Path, *, condition: str) -> pd.DataFrame:
    manifest = _validate_manifest_identity(
        directory,
        filename="manifest.json",
        expected={
            "method": "official_cellagentchat_v0_2_0_spatial",
            "database_condition": condition,
        },
    )
    native_primary = str(manifest.get("design", {}).get("native_primary", ""))
    paper_ctps_manifest = (
        "sum of Bonferroni-significant" in native_primary
        and "interaction scores" in native_primary
    )
    legacy_count_manifest = (
        "number of Bonferroni-significant LR pairs" in native_primary
    )
    if not (paper_ctps_manifest or legacy_count_manifest):
        raise ValueError(
            "CellAgentChat manifest has unexpected native-primary semantics"
        )
    claims = manifest.get("shared_input", {}).get("preparation_claims")
    if not isinstance(claims, Mapping):
        raise ValueError("CellAgentChat manifest lacks shared_input.preparation_claims")
    orthology_policy = str(claims.get("orthology_policy", ""))
    analysis_tier = str(claims.get("orthology_analysis_tier", ""))
    primary_claim_allowed = claims.get("primary_claim_allowed")
    policy_contract = {
        "strict_confidence1": {
            "analysis_tier": "primary",
            "primary_claim_allowed": True,
            "condition_suffix": "strict_confidence1_orthology",
            "label_suffix": "strict confidence=1 orthology",
        },
        "one2one_bijective_all_confidence": {
            "analysis_tier": "sensitivity",
            "primary_claim_allowed": False,
            "condition_suffix": "one2one_all_confidence_sensitivity",
            "label_suffix": "all-confidence orthology sensitivity",
        },
    }
    if orthology_policy not in policy_contract:
        raise ValueError(
            "CellAgentChat manifest has unsupported preparation orthology_policy="
            f"{orthology_policy!r}"
        )
    policy = policy_contract[orthology_policy]
    if analysis_tier != policy["analysis_tier"]:
        raise ValueError(
            "CellAgentChat preparation orthology_analysis_tier conflicts with policy"
        )
    if primary_claim_allowed is not policy["primary_claim_allowed"]:
        raise ValueError(
            "CellAgentChat preparation primary_claim_allowed conflicts with policy"
        )
    shared_input = manifest.get("shared_input")
    if not isinstance(shared_input, Mapping):
        raise ValueError("CellAgentChat manifest lacks shared_input")
    sample_plan_path = _verify_recorded_artifact(
        shared_input.get("sample_plan", {}),
        expected_name="shared_sampled_cells.csv.gz",
    )
    mapped_expression = _verify_recorded_artifact(
        shared_input.get("mapped_expression", {})
    )
    preparation_manifest = _verify_recorded_artifact(
        shared_input.get("preparation_manifest", {}), expected_name="manifest.json"
    )
    path, score_artifact = _verify_manifest_artifact_key(
        directory,
        manifest,
        artifact_key="cellagentchat_type_pair_scores.csv",
        expected_name="cellagentchat_type_pair_scores.csv",
    )
    design = manifest.get("design")
    if not isinstance(design, Mapping):
        raise ValueError("CellAgentChat manifest lacks design")
    design_stages = tuple(sorted(float(value) for value in design.get("stages", [])))
    if design_stages != EXPECTED_FULL_GRID_STAGES:
        raise ValueError(
            "CellAgentChat formal comparison requires exactly the five observed stages"
        )
    design_seeds = tuple(
        sorted(int(value) for value in design.get("sampling_seeds", []))
    )
    if design_seeds != (101, 202, 303):
        raise ValueError(
            "CellAgentChat formal comparison requires sampling seeds 101,202,303"
        )
    if (
        int(design.get("epochs", -1)) != 50
        or int(design.get("permutation_score_target", -1)) != 10_000
        or design.get("spatial") is not True
        or design.get("permutation_background_distance_scaled") is not True
    ):
        raise ValueError(
            "CellAgentChat formal comparison requires epochs=50, permutation target=10000, "
            "and the spatial distance-scaled design"
        )
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("CellAgentChat manifest lacks source provenance")
    if (
        str(source.get("release", "")) != "v0.2.0"
        or str(source.get("expected_commit", "")).casefold()
        != PINNED_CELLAGENTCHAT_COMMIT
        or str(source.get("observed_commit", "")).casefold()
        != PINNED_CELLAGENTCHAT_COMMIT
        or source.get("pinned_source_verified") is not True
    ):
        raise ValueError(
            "CellAgentChat source is not the pinned official v0.2.0 commit"
        )
    source_files = source.get("files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise ValueError("CellAgentChat source provenance has no frozen file artifacts")
    source_signature: list[tuple[str, str]] = []
    for name, record in sorted(source_files.items()):
        source_path = _verify_recorded_artifact(record)
        source_signature.append((str(name), _sha256(source_path)))
    sample_plan = pd.read_csv(sample_plan_path)
    plan_columns = {"stage", "stage_label", "sampling_seed", "cell_type", "obs_name"}
    _require_columns(sample_plan, plan_columns, sample_plan_path)
    sample_plan["stage"] = _stage_values(sample_plan["stage"], path=sample_plan_path)
    sample_plan["stage_label"] = _clean_labels(
        sample_plan["stage_label"], name="stage_label", path=sample_plan_path
    )
    sample_plan["cell_type"] = _clean_labels(
        sample_plan["cell_type"], name="cell_type", path=sample_plan_path
    )
    numeric_seeds = _numeric(
        sample_plan["sampling_seed"], name="sampling_seed", path=sample_plan_path
    )
    if not np.equal(numeric_seeds, np.floor(numeric_seeds)).all():
        raise ValueError("CellAgentChat sample plan has a non-integer sampling seed")
    sample_plan["sampling_seed"] = numeric_seeds.astype(int)
    selected_plan = sample_plan.loc[
        sample_plan["stage"].isin(design_stages)
        & sample_plan["sampling_seed"].isin(design_seeds)
    ].copy()
    if (
        selected_plan.empty
        or selected_plan.assign(obs_name=selected_plan["obs_name"].astype(str))
        .duplicated(["stage", "sampling_seed", "obs_name"])
        .any()
    ):
        raise ValueError(
            "CellAgentChat sample plan is empty or repeats cells within a run"
        )
    expected_rows: list[dict[str, Any]] = []
    stage_grid_meta: dict[float, dict[str, Any]] = {}
    for stage in design_stages:
        stage_plan = selected_plan.loc[selected_plan["stage"] == stage]
        if set(stage_plan["sampling_seed"].astype(int)) != set(design_seeds):
            raise ValueError(
                f"CellAgentChat sample plan lacks a seed at stage {stage:g}"
            )
        type_sets = []
        label_sets = []
        for seed in design_seeds:
            run_plan = stage_plan.loc[stage_plan["sampling_seed"] == seed]
            type_sets.append(tuple(sorted(run_plan["cell_type"].unique())))
            label_sets.append(tuple(sorted(run_plan["stage_label"].unique())))
        if len(set(type_sets)) != 1 or not type_sets[0]:
            raise ValueError(
                f"CellAgentChat sampled cell-type universe differs across seeds at stage {stage:g}"
            )
        if len(set(label_sets)) != 1 or len(label_sets[0]) != 1:
            raise ValueError(
                f"CellAgentChat stage label differs across seeds at stage {stage:g}"
            )
        types = type_sets[0]
        stage_label = label_sets[0][0]
        stage_grid_meta[stage] = {
            "stage_label": stage_label,
            "n_cell_types": len(types),
        }
        for sender, receiver in product(types, repeat=2):
            expected_rows.append(
                {
                    "stage": stage,
                    "stage_label": stage_label,
                    "sender_type": sender,
                    "receiver_type": receiver,
                }
            )
    expected_grid = pd.DataFrame(expected_rows)
    frame = pd.read_csv(path)
    required = {
        "stage",
        "stage_label",
        "sender_type",
        "receiver_type",
        "cellagentchat_native_primary_mean",
        "cellagentchat_significant_score_sum_mean",
    }
    _require_columns(frame, required, path)
    observed_keys = frame[
        ["stage", "stage_label", "sender_type", "receiver_type"]
    ].copy()
    observed_keys["stage"] = _stage_values(observed_keys["stage"], path=path)
    observed_keys["stage_label"] = _clean_labels(
        observed_keys["stage_label"], name="stage_label", path=path
    )
    observed_keys["sender_type"] = _clean_labels(
        observed_keys["sender_type"], name="sender_type", path=path
    )
    observed_keys["receiver_type"] = _clean_labels(
        observed_keys["receiver_type"], name="receiver_type", path=path
    )
    if observed_keys.duplicated(KEYS).any():
        raise ValueError(f"{path} contains duplicate stage/type-pair rows")
    grid_check = expected_grid.merge(
        observed_keys,
        on=["stage", "stage_label", "sender_type", "receiver_type"],
        how="outer",
        indicator=True,
    )
    if not (grid_check["_merge"] == "both").all():
        counts = grid_check["_merge"].value_counts().to_dict()
        raise ValueError(
            f"{path} is not the complete verified stage/type square: {counts}"
        )
    if (
        "n_sampling_seeds" not in frame
        or not (
            pd.to_numeric(frame["n_sampling_seeds"], errors="coerce")
            == len(design_seeds)
        ).all()
    ):
        raise ValueError(f"{path} is not summarized over every declared sampling seed")
    expected_by_seed = int(len(expected_grid) * len(design_seeds))
    counts_manifest = manifest.get("counts", {})
    if int(
        counts_manifest.get("type_pair_rows_by_seed", -1)
    ) != expected_by_seed or int(counts_manifest.get("n_runs", -1)) != len(
        EXPECTED_FULL_GRID_STAGES
    ) * len(
        design_seeds
    ):
        raise ValueError(
            "CellAgentChat manifest type-pair row count violates the full-grid contract"
        )
    official = condition == CELLAGENTCHAT_OFFICIAL
    label_prefix = (
        "CellAgentChat | official mouse DB"
        if official
        else "CellAgentChat | project LR"
    )
    label = f"{label_prefix} | {policy['label_suffix']}"
    database_condition = f"{condition}__{policy['condition_suffix']}"
    paper_ctps = pd.to_numeric(
        frame["cellagentchat_significant_score_sum_mean"], errors="coerce"
    )
    if not np.isfinite(paper_ctps).all() or (paper_ctps < 0).any():
        raise ValueError(f"{path} contains invalid CellAgentChat CTPS values")
    if paper_ctps_manifest:
        declared_primary = pd.to_numeric(
            frame["cellagentchat_native_primary_mean"], errors="coerce"
        )
        if not np.allclose(
            declared_primary.to_numpy(dtype=float),
            paper_ctps.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "CellAgentChat native primary does not equal the Methods Eq. 8 CTPS"
            )
    result = _canonical(
        frame,
        path=path,
        method="CellAgentChat (cross-species)",
        database_condition=database_condition,
        score_view="mean_ctps_sum_bonferroni_significant_interaction_scores",
        display_label=label,
        view_id=(
            "cellagentchat__official_mouse_default"
            if official
            else "cellagentchat__project_lr"
        ),
        # Historical 2026-07-22 manifests mislabeled significant-pair count as
        # native primary, but preserved the actual Eq. 8 CTPS column.  Consume
        # that immutable score column and record the correction below; never
        # rewrite the source manifest or output table in place.
        score=paper_ctps,
        stage=frame["stage"],
        stage_label=frame["stage_label"],
    )
    result["structural_zero_filled"] = False
    result["orthology_policy"] = orthology_policy
    result["analysis_tier"] = analysis_tier
    result["primary_claim_allowed"] = bool(primary_claim_allowed)
    result.attrs["cellagentchat_orthology"] = {
        "orthology_policy": orthology_policy,
        "analysis_tier": analysis_tier,
        "primary_claim_allowed": bool(primary_claim_allowed),
        "sample_plan_sha256": _sha256(sample_plan_path),
        "mapped_expression_sha256": _sha256(mapped_expression),
        "preparation_manifest_sha256": _sha256(preparation_manifest),
        "source_commit": PINNED_CELLAGENTCHAT_COMMIT,
        "source_signature": source_signature,
        "paper_ctps_definition": (
            "sum of Bonferroni-significant CellAgentChat interaction scores "
            "per directed cell-type pair (Methods Eq. 8)"
        ),
        "source_manifest_native_primary": native_primary,
        "legacy_native_primary_mislabel_corrected": bool(legacy_count_manifest),
        "ctps_source_column": "cellagentchat_significant_score_sum_mean",
    }
    audit_rows = []
    for stage in design_stages:
        stage_frame = result.loc[result["stage"] == stage]
        n_types = int(stage_grid_meta[stage]["n_cell_types"])
        audit_rows.append(
            {
                "stage": float(stage),
                "stage_label": str(stage_grid_meta[stage]["stage_label"]),
                "receiver_unit": "",
                "n_cell_types": n_types,
                "expected_directed_rows": int(n_types**2),
                "native_emitted_rows": int(len(stage_frame)),
                "native_emitted_positive_rows": int(
                    (stage_frame["native_score"] > 0).sum()
                ),
                "native_emitted_zero_rows": int(
                    (stage_frame["native_score"] == 0).sum()
                ),
                "structural_zero_filled_rows": 0,
                "verified_complete_evaluated_universe": True,
            }
        )
    result.attrs["zero_completion"] = {
        "universe_scope": "verified_cellagentchat_sample_plan_stage_type_square",
        "universe_source": _file_record(sample_plan_path),
        "runner_export_contract": "native_complete_grid_no_loader_completion",
        "full_stage_type_square_required": True,
        "expected_stage_count": len(EXPECTED_FULL_GRID_STAGES),
        "observed_stage_count": int(result["stage"].nunique()),
        "expected_rows": int(len(expected_grid)),
        "native_emitted_rows": int(len(result)),
        "structural_zero_filled_rows": 0,
        "verified_complete_evaluated_universe": True,
        "unevaluated_units_zero_filled": False,
        "method_unavailable_lr_rows_zero_filled": False,
        "audit_rows": audit_rows,
    }
    result.attrs["verified_primary_score_artifacts"] = [score_artifact]
    return result


def _load_cellagentchat_pair(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent = _validate_manifest_identity(
        directory,
        filename="manifest.json",
        expected={
            "workflow": "official_cellagentchat_spatial_dual_lr_database",
            "status": "complete",
            "conditions": [CELLAGENTCHAT_OFFICIAL, CELLAGENTCHAT_CUSTOM],
            "same_mapped_expression_and_sample_plan_verified": True,
            "same_preparation_manifest_and_orthology_claims_verified": True,
            "same_formal_design_and_pinned_source_verified": True,
            "exact_stage_seed_grid_verified": True,
            "formal_non_smoke_verified": True,
            "database_sha256_are_distinct": True,
        },
    )
    formal_design = parent.get("formal_design")
    if not isinstance(formal_design, Mapping) or (
        tuple(float(value) for value in formal_design.get("stages", []))
        != EXPECTED_FULL_GRID_STAGES
        or tuple(int(value) for value in formal_design.get("sampling_seeds", []))
        != (101, 202, 303)
        or int(formal_design.get("epochs", -1)) != 50
        or int(formal_design.get("permutation_score_target", -1)) != 10_000
        or str(formal_design.get("source_commit", "")).casefold()
        != PINNED_CELLAGENTCHAT_COMMIT
    ):
        raise ValueError("CellAgentChat dual manifest lacks the exact formal design")
    condition_records = parent.get("condition_manifests")
    if not isinstance(condition_records, Mapping):
        raise ValueError("CellAgentChat dual manifest lacks condition_manifests")
    for condition in (CELLAGENTCHAT_OFFICIAL, CELLAGENTCHAT_CUSTOM):
        path = _verify_recorded_artifact(
            condition_records.get(condition, {}), expected_name="manifest.json"
        )
        expected_path = (directory / condition / "manifest.json").resolve()
        if path.resolve() != expected_path:
            raise ValueError(
                f"CellAgentChat dual manifest points {condition!r} at the wrong condition"
            )
    official = _load_cellagentchat(
        directory / CELLAGENTCHAT_OFFICIAL,
        condition=CELLAGENTCHAT_OFFICIAL,
    )
    custom = _load_cellagentchat(
        directory / CELLAGENTCHAT_CUSTOM,
        condition=CELLAGENTCHAT_CUSTOM,
    )
    return official, custom


def _add_stage_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby(["view_id", "stage"], sort=False)["native_score"]
    result["within_stage_rank_high"] = grouped.rank(method="average", ascending=False)
    counts = grouped.transform("size").astype(int)
    result["n_directed_pairs_in_view_stage"] = counts
    denominator = (counts - 1).replace(0, 1)
    result["within_stage_rank_percentile_high"] = (
        1.0 - (result["within_stage_rank_high"] - 1.0) / denominator
    )
    result.loc[counts == 1, "within_stage_rank_percentile_high"] = 1.0
    return result


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.corr(right, method="spearman"))


def _positive_top_selection(frame: pd.DataFrame, score: str, k: int) -> dict[str, Any]:
    """Select positive top edges and expand every tie at the kth boundary."""

    positive = frame.loc[frame[score] > 0].copy()
    effective_k = min(int(k), int(len(positive)))
    if effective_k < 1:
        return {
            "keys": set(),
            "boundary_score": float("nan"),
            "boundary_tie_count": 0,
            "realized_size": 0,
            "tie_expanded": False,
        }
    ordered_scores = positive[score].sort_values(ascending=False, kind="mergesort")
    boundary = float(ordered_scores.iloc[effective_k - 1])
    selected = positive.loc[positive[score] >= boundary]
    keys = set(zip(selected["sender_type"], selected["receiver_type"]))
    tie_count = int(np.isclose(positive[score], boundary, rtol=0.0, atol=0.0).sum())
    return {
        "keys": keys,
        "boundary_score": boundary,
        "boundary_tie_count": tie_count,
        "realized_size": len(keys),
        "tie_expanded": len(keys) > effective_k,
    }


def pairwise_consistency(
    scores: pd.DataFrame, *, top_k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    views = scores[["view_id", "display_label"]].drop_duplicates()
    ordered_views = views["view_id"].tolist()
    labels = dict(zip(views["view_id"], views["display_label"]))
    for left_index, left_id in enumerate(ordered_views):
        left_all = scores.loc[scores["view_id"] == left_id]
        for right_id in ordered_views[left_index + 1 :]:
            right_all = scores.loc[scores["view_id"] == right_id]
            stages = sorted(set(left_all["stage"]) & set(right_all["stage"]))
            for stage in stages:
                left = left_all.loc[left_all["stage"] == stage]
                right = right_all.loc[right_all["stage"] == stage]
                merged = left.merge(
                    right,
                    on=KEYS,
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_left", "_right"),
                )
                n_shared = int(len(merged))
                n_positive_left = int((merged["native_score_left"] > 0).sum())
                n_positive_right = int((merged["native_score_right"] > 0).sum())
                effective_k = min(
                    int(top_k), n_shared, n_positive_left, n_positive_right
                )
                if effective_k:
                    common_left = merged[
                        ["sender_type", "receiver_type", "native_score_left"]
                    ].rename(columns={"native_score_left": "score"})
                    common_right = merged[
                        ["sender_type", "receiver_type", "native_score_right"]
                    ].rename(columns={"native_score_right": "score"})
                    left_selection = _positive_top_selection(
                        common_left, "score", effective_k
                    )
                    right_selection = _positive_top_selection(
                        common_right, "score", effective_k
                    )
                    left_top = left_selection["keys"]
                    right_top = right_selection["keys"]
                    intersection = len(left_top & right_top)
                    union = len(left_top | right_top)
                    jaccard = intersection / union if union else float("nan")
                else:
                    left_selection = _positive_top_selection(
                        merged.rename(columns={"native_score_left": "score"}),
                        "score",
                        0,
                    )
                    right_selection = _positive_top_selection(
                        merged.rename(columns={"native_score_right": "score"}),
                        "score",
                        0,
                    )
                    intersection = 0
                    union = 0
                    jaccard = float("nan")
                top_k_informative = bool(
                    effective_k > 0
                    and left_selection["realized_size"] < n_shared
                    and right_selection["realized_size"] < n_shared
                )
                overlap_denominator = min(
                    left_selection["realized_size"],
                    right_selection["realized_size"],
                )
                rows.append(
                    {
                        "view_id_left": left_id,
                        "display_label_left": labels[left_id],
                        "view_id_right": right_id,
                        "display_label_right": labels[right_id],
                        "stage": float(stage),
                        "stage_label": (
                            merged["stage_label_left"].iloc[0] if n_shared else ""
                        ),
                        "n_shared_directed_pairs": n_shared,
                        "spearman_rank_concordance": _safe_spearman(
                            merged["native_score_left"], merged["native_score_right"]
                        ),
                        "requested_top_k": int(top_k),
                        "n_positive_left": n_positive_left,
                        "n_positive_right": n_positive_right,
                        "effective_top_k": effective_k,
                        "top_k_informative": top_k_informative,
                        "top_k_left_realized_set_size": left_selection["realized_size"],
                        "top_k_right_realized_set_size": right_selection[
                            "realized_size"
                        ],
                        "top_k_left_boundary_score": left_selection["boundary_score"],
                        "top_k_right_boundary_score": right_selection["boundary_score"],
                        "top_k_left_boundary_tie_count": left_selection[
                            "boundary_tie_count"
                        ],
                        "top_k_right_boundary_tie_count": right_selection[
                            "boundary_tie_count"
                        ],
                        "top_k_left_boundary_tie_expanded": left_selection[
                            "tie_expanded"
                        ],
                        "top_k_right_boundary_tie_expanded": right_selection[
                            "tie_expanded"
                        ],
                        "top_k_intersection": intersection,
                        "top_k_union": union,
                        "top_k_overlap_fraction": (
                            intersection / overlap_denominator
                            if overlap_denominator
                            else float("nan")
                        ),
                        "top_k_jaccard": jaccard,
                        "top_k_selection_rule": (
                            "positive_support_kth_score_boundary_tie_expanded"
                        ),
                        "comparison_universe": "inner_join_on_stage_sender_type_receiver_type",
                    }
                )
    by_stage = pd.DataFrame(rows)
    if by_stage.empty:
        summary = pd.DataFrame(
            columns=[
                "view_id_left",
                "display_label_left",
                "view_id_right",
                "display_label_right",
                "n_stages_compared",
                "n_finite_spearman_stages",
                "n_shared_directed_pairs_total",
                "mean_stage_spearman",
                "median_stage_spearman",
                "n_top_k_informative_stages",
                "mean_stage_top_k_jaccard_all_stages",
                "median_stage_top_k_jaccard_all_stages",
                "mean_stage_top_k_jaccard_informative_only",
                "median_stage_top_k_jaccard_informative_only",
            ]
        )
        return by_stage, summary
    by_stage["top_k_jaccard_informative_only"] = by_stage["top_k_jaccard"].where(
        by_stage["top_k_informative"]
    )
    keys = [
        "view_id_left",
        "display_label_left",
        "view_id_right",
        "display_label_right",
    ]
    summary = (
        by_stage.groupby(keys, sort=False, dropna=False)
        .agg(
            n_stages_compared=("stage", "nunique"),
            n_finite_spearman_stages=(
                "spearman_rank_concordance",
                lambda values: int(pd.Series(values).notna().sum()),
            ),
            n_shared_directed_pairs_total=("n_shared_directed_pairs", "sum"),
            mean_stage_spearman=("spearman_rank_concordance", "mean"),
            median_stage_spearman=("spearman_rank_concordance", "median"),
            n_top_k_informative_stages=("top_k_informative", "sum"),
            mean_stage_top_k_jaccard_all_stages=("top_k_jaccard", "mean"),
            median_stage_top_k_jaccard_all_stages=("top_k_jaccard", "median"),
            mean_stage_top_k_jaccard_informative_only=(
                "top_k_jaccard_informative_only",
                "mean",
            ),
            median_stage_top_k_jaccard_informative_only=(
                "top_k_jaccard_informative_only",
                "median",
            ),
        )
        .reset_index()
    )
    return by_stage, summary


def reciprocal_asymmetry(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for (_, _), frame in scores.groupby(["view_id", "stage"], sort=False):
        reverse = frame[
            [
                "sender_type",
                "receiver_type",
                "native_score",
                "within_stage_rank_percentile_high",
            ]
        ].rename(
            columns={
                "sender_type": "receiver_type",
                "receiver_type": "sender_type",
                "native_score": "reverse_native_score",
                "within_stage_rank_percentile_high": "reverse_rank_percentile_high",
            }
        )
        paired = frame.merge(
            reverse,
            on=["sender_type", "receiver_type"],
            how="inner",
            validate="one_to_one",
        )
        paired = paired.loc[paired["sender_type"] < paired["receiver_type"]].copy()
        if paired.empty:
            continue
        paired["cell_type_a"] = paired["sender_type"]
        paired["cell_type_b"] = paired["receiver_type"]
        paired["rank_asymmetry_a_to_b"] = (
            paired["within_stage_rank_percentile_high"]
            - paired["reverse_rank_percentile_high"]
        )
        paired["native_score_difference_a_to_b"] = (
            paired["native_score"] - paired["reverse_native_score"]
        )
        rows.append(
            paired[
                [
                    "view_id",
                    "display_label",
                    "method",
                    "database_condition",
                    "score_view",
                    "stage",
                    "stage_label",
                    "cell_type_a",
                    "cell_type_b",
                    "rank_asymmetry_a_to_b",
                    "native_score_difference_a_to_b",
                ]
            ]
        )
    asymmetry = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "view_id",
                "display_label",
                "method",
                "database_condition",
                "score_view",
                "stage",
                "stage_label",
                "cell_type_a",
                "cell_type_b",
                "rank_asymmetry_a_to_b",
                "native_score_difference_a_to_b",
            ]
        )
    )
    comparison_rows: list[dict[str, Any]] = []
    view_order = scores["view_id"].drop_duplicates().tolist()
    labels = scores.drop_duplicates("view_id").set_index("view_id")["display_label"]
    for left_index, left_id in enumerate(view_order):
        for right_id in view_order[left_index + 1 :]:
            left = asymmetry.loc[asymmetry["view_id"] == left_id]
            right = asymmetry.loc[asymmetry["view_id"] == right_id]
            for stage in sorted(set(left["stage"]) & set(right["stage"])):
                merged = left.loc[left["stage"] == stage].merge(
                    right.loc[right["stage"] == stage],
                    on=["stage", "cell_type_a", "cell_type_b"],
                    how="inner",
                    validate="one_to_one",
                    suffixes=("_left", "_right"),
                )
                comparison_rows.append(
                    {
                        "view_id_left": left_id,
                        "display_label_left": labels[left_id],
                        "view_id_right": right_id,
                        "display_label_right": labels[right_id],
                        "stage": float(stage),
                        "n_shared_reciprocal_type_pairs": int(len(merged)),
                        "spearman_rank_asymmetry": _safe_spearman(
                            merged["rank_asymmetry_a_to_b_left"],
                            merged["rank_asymmetry_a_to_b_right"],
                        ),
                    }
                )
    by_stage = pd.DataFrame(comparison_rows)
    if by_stage.empty:
        summary = pd.DataFrame()
    else:
        group_keys = [
            "view_id_left",
            "display_label_left",
            "view_id_right",
            "display_label_right",
        ]
        summary = (
            by_stage.groupby(group_keys, sort=False, dropna=False)
            .agg(
                n_stages_compared=("stage", "nunique"),
                n_shared_reciprocal_type_pairs_total=(
                    "n_shared_reciprocal_type_pairs",
                    "sum",
                ),
                mean_stage_spearman_rank_asymmetry=(
                    "spearman_rank_asymmetry",
                    "mean",
                ),
                median_stage_spearman_rank_asymmetry=(
                    "spearman_rank_asymmetry",
                    "median",
                ),
            )
            .reset_index()
        )
    return asymmetry, by_stage, summary


def stage_stability(
    scores: pd.DataFrame,
    *,
    top_k: int,
    global_stages: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = scores.drop_duplicates("view_id").set_index("view_id")["display_label"]
    for view_id in scores["view_id"].drop_duplicates():
        view = scores.loc[scores["view_id"] == view_id]
        for stage_from, stage_to in zip(global_stages[:-1], global_stages[1:]):
            left = view.loc[view["stage"] == stage_from]
            right = view.loc[view["stage"] == stage_to]
            if left.empty or right.empty:
                rows.append(
                    {
                        "view_id": view_id,
                        "display_label": labels[view_id],
                        "stage_from": float(stage_from),
                        "stage_to": float(stage_to),
                        "status": "missing_stage",
                        "n_shared_directed_pairs": 0,
                        "spearman_rank_stability": float("nan"),
                        "n_positive_from": 0,
                        "n_positive_to": 0,
                        "effective_top_k": 0,
                        "top_k_informative": False,
                        "top_k_from_realized_set_size": 0,
                        "top_k_to_realized_set_size": 0,
                        "top_k_from_boundary_score": float("nan"),
                        "top_k_to_boundary_score": float("nan"),
                        "top_k_from_boundary_tie_count": 0,
                        "top_k_to_boundary_tie_count": 0,
                        "top_k_from_boundary_tie_expanded": False,
                        "top_k_to_boundary_tie_expanded": False,
                        "top_k_jaccard": float("nan"),
                    }
                )
                continue
            merged = left.merge(
                right,
                on=["sender_type", "receiver_type"],
                how="inner",
                validate="one_to_one",
                suffixes=("_from", "_to"),
            )
            n_positive_from = int((merged["native_score_from"] > 0).sum())
            n_positive_to = int((merged["native_score_to"] > 0).sum())
            effective_k = min(int(top_k), len(merged), n_positive_from, n_positive_to)
            if effective_k:
                left_common = merged[
                    ["sender_type", "receiver_type", "native_score_from"]
                ].rename(columns={"native_score_from": "score"})
                right_common = merged[
                    ["sender_type", "receiver_type", "native_score_to"]
                ].rename(columns={"native_score_to": "score"})
                left_selection = _positive_top_selection(
                    left_common, "score", effective_k
                )
                right_selection = _positive_top_selection(
                    right_common, "score", effective_k
                )
                left_top = left_selection["keys"]
                right_top = right_selection["keys"]
                union = len(left_top | right_top)
                jaccard = len(left_top & right_top) / union if union else float("nan")
            else:
                left_selection = _positive_top_selection(
                    merged.rename(columns={"native_score_from": "score"}),
                    "score",
                    0,
                )
                right_selection = _positive_top_selection(
                    merged.rename(columns={"native_score_to": "score"}),
                    "score",
                    0,
                )
                jaccard = float("nan")
            top_k_informative = bool(
                effective_k > 0
                and left_selection["realized_size"] < len(merged)
                and right_selection["realized_size"] < len(merged)
            )
            rows.append(
                {
                    "view_id": view_id,
                    "display_label": labels[view_id],
                    "stage_from": float(stage_from),
                    "stage_to": float(stage_to),
                    "status": "complete",
                    "n_shared_directed_pairs": int(len(merged)),
                    "spearman_rank_stability": _safe_spearman(
                        merged["native_score_from"], merged["native_score_to"]
                    ),
                    "n_positive_from": n_positive_from,
                    "n_positive_to": n_positive_to,
                    "effective_top_k": int(effective_k),
                    "top_k_informative": top_k_informative,
                    "top_k_from_realized_set_size": left_selection["realized_size"],
                    "top_k_to_realized_set_size": right_selection["realized_size"],
                    "top_k_from_boundary_score": left_selection["boundary_score"],
                    "top_k_to_boundary_score": right_selection["boundary_score"],
                    "top_k_from_boundary_tie_count": left_selection[
                        "boundary_tie_count"
                    ],
                    "top_k_to_boundary_tie_count": right_selection[
                        "boundary_tie_count"
                    ],
                    "top_k_from_boundary_tie_expanded": left_selection["tie_expanded"],
                    "top_k_to_boundary_tie_expanded": right_selection["tie_expanded"],
                    "top_k_jaccard": jaccard,
                    "top_k_selection_rule": (
                        "positive_support_kth_score_boundary_tie_expanded"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _extract_one(
    frame: pd.DataFrame,
    filters: Mapping[str, Any],
    *,
    path: Path,
) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        if column not in selected:
            raise ValueError(f"{path} lacks filter column {column!r}")
        selected = selected.loc[selected[column].astype(str) == str(value)]
    if len(selected) != 1:
        raise ValueError(
            f"{path}: expected one row for {dict(filters)}, observed {len(selected)}"
        )
    return selected.iloc[0]


def load_cytobridge_control(
    directory: Path, *, control: str, display_label: str
) -> pd.DataFrame:
    manifest = _read_json(directory / "run_manifest.json")
    permutation_path, permutation_artifact = _verify_manifest_artifact_key(
        directory,
        manifest,
        artifact_key="conditional_permutations",
        expected_name="conditional_permutation_tests.csv",
    )
    nested_path, nested_artifact = _verify_manifest_artifact_key(
        directory,
        manifest,
        artifact_key="nested_grouped_cv",
        expected_name="nested_grouped_cv_metrics.csv",
    )
    reciprocal_path, reciprocal_artifact = _verify_manifest_artifact_key(
        directory,
        manifest,
        artifact_key="reciprocal_edge_direction_tests",
        expected_name="reciprocal_edge_direction_tests.csv",
    )
    permutation = pd.read_csv(permutation_path)
    nested = pd.read_csv(nested_path)
    reciprocal = pd.read_csv(reciprocal_path, dtype={"stage": str})
    rows: list[dict[str, Any]] = []
    for target, target_label in (
        ("log1p_attention", "attention"),
        ("log1p_edge_message_joint", "exact message"),
    ):
        conditional = _extract_one(
            permutation,
            {
                "score": "lr_compatibility_forward",
                "strata": STRICT_PERMUTATION_STRATA,
                "target": target,
            },
            path=permutation_path,
        )
        rows.append(
            {
                "control": control,
                "control_label": display_label,
                "target": target_label,
                "metric": "conditional_residual_spearman_forward_lr",
                "estimate": float(conditional["observed_spearman"]),
                "p_value": float(conditional["empirical_p_greater"]),
                "n_observations": int(conditional["n_edges_retained"]),
                "selection": STRICT_PERMUTATION_STRATA,
            }
        )
        nested_row = _extract_one(
            nested,
            {"target": target, "model": "confounders_plus_forward_lr"},
            path=nested_path,
        )
        rows.append(
            {
                "control": control,
                "control_label": display_label,
                "target": target_label,
                "metric": "delta_r2_forward_lr_over_confounders",
                "estimate": float(nested_row["delta_r2_vs_confounders"]),
                "p_value": float("nan"),
                "n_observations": int(nested_row["n_edges"]),
                "selection": "receiver-grouped out-of-fold CV",
            }
        )
        direction_name = (
            "attention_direction_delta"
            if target == "log1p_attention"
            else "message_direction_delta"
        )
        reciprocal_row = _extract_one(
            reciprocal,
            {"stage": "all", "learned_direction_delta": direction_name},
            path=reciprocal_path,
        )
        rows.append(
            {
                "control": control,
                "control_label": display_label,
                "target": target_label,
                "metric": "reciprocal_direction_spearman_all_stage",
                "estimate": float(reciprocal_row["spearman_with_lr_direction_delta"]),
                "p_value": float(reciprocal_row["empirical_p_greater"]),
                "n_observations": int(reciprocal_row["n_reciprocal_pairs"]),
                "selection": "reciprocal directed edges, all stages",
            }
        )
    result = pd.DataFrame(rows)
    result.attrs["verified_primary_score_artifacts"] = [
        permutation_artifact,
        nested_artifact,
        reciprocal_artifact,
    ]
    return result


def condition_coverage(
    scores: pd.DataFrame,
    expected: Sequence[Mapping[str, str]],
    issues: Sequence[Mapping[str, str]],
    global_stages: Sequence[float],
    stage_labels: Mapping[float, str],
) -> pd.DataFrame:
    issue_by_view = {str(item["view_id"]): str(item["error"]) for item in issues}
    union_by_stage = {
        float(stage): set(
            zip(
                scores.loc[scores["stage"] == stage, "sender_type"],
                scores.loc[scores["stage"] == stage, "receiver_type"],
            )
        )
        for stage in global_stages
    }
    rows: list[dict[str, Any]] = []
    for spec in expected:
        view_id = spec["view_id"]
        view = scores.loc[scores["view_id"] == view_id]
        if view.empty:
            rows.append(
                {
                    **spec,
                    "stage": float("nan"),
                    "stage_label": "",
                    "status": "missing_or_invalid_method",
                    "n_directed_pairs": 0,
                    "n_sender_types": 0,
                    "n_receiver_types": 0,
                    "n_positive_directed_pairs": 0,
                    "n_structural_zero_filled": 0,
                    "fraction_of_loaded_union_keys": float("nan"),
                    "diagnostic": issue_by_view.get(view_id, "not loaded"),
                }
            )
            continue
        for stage in global_stages:
            stage_view = view.loc[view["stage"] == stage]
            union_count = len(union_by_stage[float(stage)])
            rows.append(
                {
                    **spec,
                    "stage": float(stage),
                    "stage_label": stage_labels[float(stage)],
                    "status": "complete" if len(stage_view) else "stage_not_emitted",
                    "n_directed_pairs": int(len(stage_view)),
                    "n_sender_types": int(stage_view["sender_type"].nunique()),
                    "n_receiver_types": int(stage_view["receiver_type"].nunique()),
                    "n_positive_directed_pairs": int(
                        (stage_view["native_score"] > 0).sum()
                    ),
                    "n_structural_zero_filled": int(
                        stage_view.get(
                            "structural_zero_filled",
                            pd.Series(False, index=stage_view.index),
                        )
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    ),
                    "fraction_of_loaded_union_keys": (
                        len(stage_view) / union_count if union_count else float("nan")
                    ),
                    "diagnostic": "",
                }
            )
    return pd.DataFrame(rows)


def _matrix_from_summary(
    summary: pd.DataFrame,
    *,
    views: pd.DataFrame,
    value_column: str,
) -> np.ndarray:
    ids = views["view_id"].tolist()
    matrix = np.full((len(ids), len(ids)), np.nan, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    index = {view_id: position for position, view_id in enumerate(ids)}
    if not summary.empty and value_column in summary:
        for row in summary.itertuples(index=False):
            left = index[str(row.view_id_left)]
            right = index[str(row.view_id_right)]
            value = float(getattr(row, value_column))
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def _heatmap(
    matrix: np.ndarray,
    labels: Sequence[str],
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    output_base: Path,
    diagonal_note: str | None = None,
) -> None:
    size = max(7.0, 0.75 * len(labels) + 3.5)
    figure, axis = plt.subplots(figsize=(size, size * 0.9))
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("#E5E7EB")
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, cmap=color_map, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title(title, loc="left", fontweight="bold")
    threshold = (vmin + vmax) / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text = "NA" if not np.isfinite(value) else f"{value:.2f}"
            color = "white" if np.isfinite(value) and value > threshold else "#111827"
            axis.text(
                column, row, text, ha="center", va="center", fontsize=7, color=color
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    if diagonal_note:
        figure.text(0.01, 0.01, diagonal_note, fontsize=8, color="#4B5563")
    figure.tight_layout(rect=(0, 0.035 if diagonal_note else 0, 1, 1))
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_coverage(
    coverage: pd.DataFrame,
    *,
    expected: Sequence[Mapping[str, str]],
    global_stages: Sequence[float],
    stage_labels: Mapping[float, str],
    output_base: Path,
) -> None:
    labels = [spec["display_label"] for spec in expected]
    matrix = np.full((len(expected), len(global_stages)), np.nan)
    annotations = np.full(matrix.shape, "missing", dtype=object)
    for row_index, spec in enumerate(expected):
        subset = coverage.loc[coverage["view_id"] == spec["view_id"]]
        for column_index, stage in enumerate(global_stages):
            match = subset.loc[np.isclose(subset["stage"], stage, equal_nan=False)]
            if match.empty:
                continue
            count = int(match["n_directed_pairs"].iloc[0])
            matrix[row_index, column_index] = math.log1p(count)
            annotations[row_index, column_index] = str(count)
    width = max(7.5, 1.2 * len(global_stages) + 4)
    height = max(5.5, 0.62 * len(expected) + 2.5)
    figure, axis = plt.subplots(figsize=(width, height))
    color_map = plt.get_cmap("Blues").copy()
    color_map.set_bad("#D1D5DB")
    image = axis.imshow(np.ma.masked_invalid(matrix), cmap=color_map, aspect="auto")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xticks(
        np.arange(len(global_stages)),
        [stage_labels[float(stage)] for stage in global_stages],
        rotation=30,
        ha="right",
    )
    axis.set_title(
        "Condition and directed-key coverage",
        loc="left",
        fontweight="bold",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                annotations[row, column],
                ha="center",
                va="center",
                fontsize=7,
                color="#111827",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label(
        "log(1 + canonical evaluated directed keys); cell text is raw count"
    )
    figure.text(
        0.01,
        0.01,
        "Counts include explicitly provenance-verified structural zeros. Grey = unevaluated, "
        "unavailable, or invalid condition/stage; coverage precedes pairwise inner joins.",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_stage_stability(
    stability: pd.DataFrame,
    *,
    views: pd.DataFrame,
    global_stages: Sequence[float],
    stage_labels: Mapping[float, str],
    output_base: Path,
) -> None:
    transitions = list(zip(global_stages[:-1], global_stages[1:]))
    shape = (len(views), len(transitions))
    rho = np.full(shape, np.nan)
    jaccard = np.full(shape, np.nan)
    index = {view: pos for pos, view in enumerate(views["view_id"])}
    transition_index = {pair: pos for pos, pair in enumerate(transitions)}
    for row in stability.itertuples(index=False):
        position = (float(row.stage_from), float(row.stage_to))
        rho[
            index[row.view_id], transition_index[position]
        ] = row.spearman_rank_stability
        jaccard[index[row.view_id], transition_index[position]] = (
            row.top_k_jaccard if row.top_k_informative else np.nan
        )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(max(10, 1.7 * len(transitions) + 6), max(5.5, 0.6 * len(views) + 2.5)),
        sharey=True,
    )
    transition_labels = [
        f"{stage_labels[float(left)]} → {stage_labels[float(right)]}"
        for left, right in transitions
    ]
    for axis, matrix, title, cmap, vmin, vmax in (
        (axes[0], rho, "Adjacent-stage rank stability", "coolwarm", -1, 1),
        (
            axes[1],
            jaccard,
            "Adjacent-stage positive-support top-k stability (tie-aware informative sets)",
            "viridis",
            0,
            1,
        ),
    ):
        color_map = plt.get_cmap(cmap).copy()
        color_map.set_bad("#E5E7EB")
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap=color_map,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        axis.set_xticks(
            np.arange(len(transitions)), transition_labels, rotation=35, ha="right"
        )
        axis.set_title(title, loc="left", fontweight="bold")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    "NA" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    axes[0].set_yticks(np.arange(len(views)), views["display_label"])
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_controls(controls: pd.DataFrame, output_base: Path) -> None:
    metrics = [
        (
            "conditional_residual_spearman_forward_lr",
            "Conditional residual association\n(forward LR compatibility, Spearman ρ)",
        ),
        (
            "delta_r2_forward_lr_over_confounders",
            "Incremental predictive value\n(ΔR² over confounders)",
        ),
        (
            "reciprocal_direction_spearman_all_stage",
            "Reciprocal direction agreement\n(all-stage Spearman ρ)",
        ),
    ]
    controls_order = controls[["control", "control_label"]].drop_duplicates()
    targets = ["attention", "exact message"]
    colors = {"attention": "#2563EB", "exact message": "#F97316"}
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    width = 0.34
    x = np.arange(len(controls_order))
    for axis, (metric, title) in zip(axes, metrics):
        metric_frame = controls.loc[controls["metric"] == metric]
        for target_index, target in enumerate(targets):
            positions = x + (target_index - 0.5) * width
            values = []
            pvalues = []
            for control in controls_order["control"]:
                row = metric_frame.loc[
                    (metric_frame["control"] == control)
                    & (metric_frame["target"] == target)
                ]
                values.append(float(row["estimate"].iloc[0]) if len(row) else np.nan)
                pvalues.append(float(row["p_value"].iloc[0]) if len(row) else np.nan)
            bars = axis.bar(
                positions,
                values,
                width=width,
                color=colors[target],
                label=target,
                edgecolor="white",
            )
            for bar, value, pvalue in zip(bars, values, pvalues):
                if not np.isfinite(value):
                    continue
                vertical = 3 if value >= 0 else -11
                annotation = f"{value:.4f}"
                if np.isfinite(pvalue):
                    annotation += f"\np={pvalue:.3g}"
                axis.annotate(
                    annotation,
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords="offset points",
                    xytext=(0, vertical),
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=7,
                )
        axis.axhline(0, color="#374151", linewidth=0.8)
        axis.set_xticks(x, controls_order["control_label"], rotation=22, ha="right")
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(
        "CytoBridge learned-edge controls — attention is not a CCC probability",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.02,
        0.01,
        "Conditional test uses the strict stage + sender + receiver + distance + state strata. "
        "Direction tests compare reciprocal edge deltas; one-sided p values are shown where defined.",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.92))
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_readme(
    path: Path,
    *,
    status: str,
    top_k: int,
    expected: Sequence[Mapping[str, str]],
    issues: Sequence[Mapping[str, str]],
    zero_completion: Mapping[str, Mapping[str, Any]],
    readiness_checks: Mapping[str, bool],
) -> None:
    condition_lines = "\n".join(
        f"- `{item['view_id']}`: {item['display_label']} — "
        f"{item['database_condition']} / {item['score_view']}"
        for item in expected
    )
    issue_text = (
        "\n".join(f"- `{item['view_id']}`: {item['error']}" for item in issues)
        if issues
        else "- None. All required inputs passed validation."
    )
    zero_lines = "\n".join(
        f"- `{view_id}`: scope `{record.get('universe_scope')}`; "
        f"native rows {record.get('native_emitted_rows')}; verified structural zeros added "
        f"{record.get('structural_zero_filled_rows')}; unevaluated units zero-filled = "
        f"`{record.get('unevaluated_units_zero_filled')}`."
        for view_id, record in zero_completion.items()
    )
    readiness_lines = "\n".join(
        f"- `{key}`: `{str(bool(value)).lower()}`"
        for key, value in readiness_checks.items()
    )
    path.write_text(
        f"""# Zebrafish directed-CCC comparison

Status: **{status}**

This directory compares methods only after ranking each method/condition within
each observed stage. Raw COMMOT mass, CellChat probability, NicheNet ligand
activity, CellAgentChat CTPS, CytoBridge attention, and exact
CytoBridge message norms are not treated as a shared numerical unit.

## Conditions and score views

{condition_lines}

The two NicheNet and two CellAgentChat database conditions are deliberately
separate. NicheNet uses a mouse prior after the manifest-declared bijective
one-to-one orthology mapping;
the exact NicheNet orthology policy and primary/sensitivity tier are read from
and validated against each run manifest. Its type-pair score here is a derived
sum of positive sender-associated ligand
activities and is not native sender-specific, receptor-specific, spatial, or a
biochemical strength. CytoBridge attention is an internal gate magnitude, not
a calibrated CCC probability. The exact message contribution is shown as a
separate model-internal view. CellAgentChat's two database conditions must also
share identical preparation orthology claims; an all-confidence mapping is
always labelled as a sensitivity and never as primary.

## Comparison contract

- Exact join key: `stage, sender_type, receiver_type`.
- Rank and percentile: computed within each method/condition/stage.
- Pairwise universe: exact-key inner join after method-specific evaluated-grid
  completion. COMMOT/CellChat gaps are zero only when the hash-verified
  stage-specific type square and runner positive-only policy both authorize it.
  NicheNet is completed only across source-stage sender types for `complete`
  receiver units; skipped/ineligible units and absent transitions stay unavailable.
- Top-edge overlap: requested top-{top_k} is capped by positive support in both
  views. Selection is restricted to scores > 0 and expands all ties at the kth
  score boundary, so alphabetical tie-breaking cannot manufacture overlap from
  an all-zero or zero-tail grid. Zero support gives NA. A selection covering the
  whole shared universe is audit-only/non-informative.
- Directionality: difference of within-stage rank percentiles for reciprocal
  A→B and B→A edges.
- Stage stability: adjacent global observed stages only.

## Main artifacts

- `canonical_type_pair_scores.csv.gz`: native values plus within-stage ranks.
- `pairwise_consistency_by_stage.csv` and `pairwise_consistency_summary.csv`.
- `reciprocal_rank_asymmetry.csv.gz` and directionality concordance summaries.
- `stage_stability.csv` and `condition_coverage.csv`.
- `structural_zero_audit.csv`: per-view/stage (and per NicheNet completed
  receiver unit) evaluated-universe, native-row, and zero-completion counts.
- `method_unavailable_lr_rows.csv`: CellChat-incompatible requested LR rows;
  these are excluded from its method universe and are never zero-filled.
- `cytobridge_control_metrics.csv`: trained, Init_interaction, and randomized
  interaction controls.
- PNG/PDF panels for rank concordance, top-edge overlap, condition coverage,
  reciprocal directionality, stage stability, and CytoBridge controls.

## Diagnostics

{issue_text}

## Structural-zero provenance

{zero_lines}

## Formal readiness checks

{readiness_lines}

`partial_diagnostic` output was generated with `--allow-partial` and must not be
used as the formal reviewer comparison until every required condition is
available and the script completes without that flag.
""",
        encoding="utf-8",
    )


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.run_root.expanduser().resolve()
    return {
        "cytobridge": (args.cytobridge_dir or root / "01_cytobridge")
        .expanduser()
        .resolve(),
        "control_trained": (args.controls_trained_dir or root / "02_attention_controls")
        .expanduser()
        .resolve(),
        "control_init": (
            args.controls_init_dir or root / "02_attention_controls_init_interaction"
        )
        .expanduser()
        .resolve(),
        "control_random": (
            args.controls_random_dir or root / "02_attention_controls_random_seed17"
        )
        .expanduser()
        .resolve(),
        "commot": (args.commot_dir or root / "03_external_ccc" / "commot_current_lr")
        .expanduser()
        .resolve(),
        "cellchat": (
            args.cellchat_dir or root / "03_external_ccc" / "cellchat_current_lr"
        )
        .expanduser()
        .resolve(),
        "nichenet_default": (
            args.nichenet_default_dir or root / "04_nichenet" / "02_default_mouse_v2"
        )
        .expanduser()
        .resolve(),
        "nichenet_custom": (
            args.nichenet_custom_dir or root / "04_nichenet" / "03_custom_zebrafish_lr"
        )
        .expanduser()
        .resolve(),
        "cellagentchat": (args.cellagentchat_dir or root / "05_cellagentchat")
        .expanduser()
        .resolve(),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if int(args.top_k) < 1:
        raise ValueError("--top-k must be positive")
    output = _prepare_output(args.output_dir, bool(args.overwrite))
    paths = _resolve_paths(args)

    expected = [
        {
            "view_id": "cytobridge__trained__attention",
            "display_label": "CytoBridge attention",
            "method": "CytoBridge",
            "database_condition": "trained_project_lr_edge_prior",
            "score_view": "mean_absolute_attention_gate",
        },
        {
            "view_id": "cytobridge__trained__exact_message",
            "display_label": "CytoBridge exact message",
            "method": "CytoBridge",
            "database_condition": "trained_project_lr_edge_prior",
            "score_view": "exact_joint_message_contribution",
        },
        {
            "view_id": "commot__project_lr",
            "display_label": "COMMOT | project LR",
            "method": "COMMOT",
            "database_condition": "current_zebrafish_lr_database",
            "score_view": "mean_communication_mass_per_possible_cell_pair",
        },
        {
            "view_id": "cellchat__project_lr",
            "display_label": "CellChat | project LR",
            "method": "CellChat",
            "database_condition": "current_zebrafish_lr_database",
            "score_view": "population_size_false_total_probability",
        },
        {
            "view_id": "nichenet_v2__official_mouse_lr",
            "display_label": "NicheNet-v2 | official mouse LR | manifest policy",
            "method": "NicheNet-v2 (cross-species)",
            "database_condition": "from_run_manifest",
            "score_view": "sum_positive_sender_associated_aupr_corrected",
        },
        {
            "view_id": "nichenet_v2__project_lr_gate",
            "display_label": "NicheNet-v2 | project LR gate | manifest policy",
            "method": "NicheNet-v2 (cross-species)",
            "database_condition": "from_run_manifest",
            "score_view": "sum_positive_sender_associated_aupr_corrected",
        },
        {
            "view_id": "cellagentchat__official_mouse_default",
            "display_label": "CellAgentChat | official mouse DB | manifest policy",
            "method": "CellAgentChat (cross-species)",
            "database_condition": "from_run_manifest",
            "score_view": "mean_ctps_sum_bonferroni_significant_interaction_scores",
        },
        {
            "view_id": "cellagentchat__project_lr",
            "display_label": "CellAgentChat | project LR | manifest policy",
            "method": "CellAgentChat (cross-species)",
            "database_condition": "from_run_manifest",
            "score_view": "mean_ctps_sum_bonferroni_significant_interaction_scores",
        },
    ]
    loaded: list[pd.DataFrame] = []
    issues: list[dict[str, str]] = []
    method_unavailable_frames: list[pd.DataFrame] = []
    nichenet_orthology_records: dict[str, Mapping[str, Any]] = {}
    cellagentchat_orthology_records: dict[str, Mapping[str, Any]] = {}
    zero_completion_records: dict[str, Mapping[str, Any]] = {}
    zero_audit_frames: list[pd.DataFrame] = []
    cellchat_lr_universe: Mapping[str, Any] | None = None
    primary_score_artifact_verification: dict[str, list[Mapping[str, Any]]] = {}
    control_artifact_verification: dict[str, list[Mapping[str, Any]]] = {}

    loaders: list[tuple[list[str], Callable[[], Sequence[pd.DataFrame]]]] = [
        (
            ["cytobridge__trained__attention", "cytobridge__trained__exact_message"],
            lambda: _load_cytobridge_views(paths["cytobridge"]),
        ),
        (["commot__project_lr"], lambda: (_load_commot(paths["commot"]),)),
        (["cellchat__project_lr"], lambda: (_load_cellchat(paths["cellchat"]),)),
        (
            ["nichenet_v2__official_mouse_lr"],
            lambda: (_load_nichenet(paths["nichenet_default"], mode="default"),),
        ),
        (
            ["nichenet_v2__project_lr_gate"],
            lambda: (_load_nichenet(paths["nichenet_custom"], mode="custom"),),
        ),
        (
            [
                "cellagentchat__official_mouse_default",
                "cellagentchat__project_lr",
            ],
            lambda: _load_cellagentchat_pair(paths["cellagentchat"]),
        ),
    ]
    for view_ids, loader in loaders:
        try:
            result = tuple(loader())
            if len(result) != len(view_ids):
                raise RuntimeError(
                    "Loader returned an unexpected number of score views"
                )
            for view_id, frame in zip(view_ids, result):
                score_artifacts = frame.attrs.get("verified_primary_score_artifacts")
                if not isinstance(score_artifacts, list) or not score_artifacts:
                    raise ValueError(
                        f"Score view {view_id} lacks verified primary-score artifact provenance"
                    )
                if not all(
                    isinstance(record, Mapping) and record.get("verified") is True
                    for record in score_artifacts
                ):
                    raise ValueError(
                        f"Score view {view_id} has invalid primary-score artifact provenance"
                    )
                primary_score_artifact_verification[view_id] = [
                    dict(record) for record in score_artifacts
                ]
                unavailable = frame.attrs.get("method_unavailable_lr_rows")
                if isinstance(unavailable, pd.DataFrame):
                    method_unavailable_frames.append(unavailable.copy())
                universe = frame.attrs.get("cellchat_lr_universe")
                if isinstance(universe, Mapping):
                    cellchat_lr_universe = dict(universe)
                orthology = frame.attrs.get("nichenet_orthology")
                if isinstance(orthology, Mapping):
                    nichenet_orthology_records[view_id] = dict(orthology)
                cellagentchat_orthology = frame.attrs.get("cellagentchat_orthology")
                if isinstance(cellagentchat_orthology, Mapping):
                    cellagentchat_orthology_records[view_id] = dict(
                        cellagentchat_orthology
                    )
                zero_completion = frame.attrs.get("zero_completion")
                if not isinstance(zero_completion, Mapping):
                    raise ValueError(
                        f"Score view {view_id} lacks structural-zero provenance"
                    )
                zero_record = dict(zero_completion)
                zero_completion_records[view_id] = {
                    key: value
                    for key, value in zero_record.items()
                    if key != "audit_rows"
                }
                audit_rows = zero_record.get("audit_rows")
                if not isinstance(audit_rows, list) or not audit_rows:
                    raise ValueError(
                        f"Score view {view_id} has no structural-zero audit rows"
                    )
                metadata = frame[
                    [
                        "view_id",
                        "display_label",
                        "method",
                        "database_condition",
                    ]
                ].drop_duplicates()
                if len(metadata) != 1:
                    raise ValueError(f"Score view {view_id} has ambiguous metadata")
                audit = pd.DataFrame(audit_rows)
                audit.insert(0, "universe_scope", str(zero_record["universe_scope"]))
                audit.insert(
                    0,
                    "database_condition",
                    str(metadata["database_condition"].iloc[0]),
                )
                audit.insert(0, "method", str(metadata["method"].iloc[0]))
                audit.insert(0, "display_label", str(metadata["display_label"].iloc[0]))
                audit.insert(0, "view_id", view_id)
                audit["unevaluated_units_zero_filled"] = bool(
                    zero_record.get("unevaluated_units_zero_filled")
                )
                audit["method_unavailable_lr_rows_zero_filled"] = bool(
                    zero_record.get("method_unavailable_lr_rows_zero_filled")
                )
                source = zero_record.get("universe_source")
                audit["provenance_manifest_path"] = (
                    str(source.get("path", "")) if isinstance(source, Mapping) else ""
                )
                zero_audit_frames.append(audit)
                loaded.append(frame)
        except Exception as error:
            if not args.allow_partial:
                raise
            for view_id in view_ids:
                issues.append(
                    {
                        "component": "score_view",
                        "view_id": view_id,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    if not loaded:
        raise RuntimeError(
            "No valid score view is available, even for partial diagnostics"
        )
    combined_scores = pd.concat(loaded, ignore_index=True)
    for spec in expected:
        observed = combined_scores.loc[combined_scores["view_id"] == spec["view_id"]]
        if observed.empty:
            continue
        metadata = observed[
            ["display_label", "method", "database_condition", "score_view"]
        ].drop_duplicates()
        if len(metadata) != 1:
            raise ValueError(f"Score view {spec['view_id']} has inconsistent metadata")
        for column in ("display_label", "method", "database_condition", "score_view"):
            spec[column] = str(metadata[column].iloc[0])

    nichenet_policies = {
        str(record["orthology_policy"])
        for record in nichenet_orthology_records.values()
    }
    nichenet_tiers = {
        str(record["analysis_tier"]) for record in nichenet_orthology_records.values()
    }
    nichenet_shared_preparations = {
        str(record.get("shared_prepare_manifest_md5", ""))
        for record in nichenet_orthology_records.values()
    }
    nichenet_prior_sources = {
        str(record.get("prior_source_signature_sha256", ""))
        for record in nichenet_orthology_records.values()
    }
    nichenet_pair_contract_ok = len(nichenet_orthology_records) == 2 and (
        len(nichenet_policies) == 1
        and len(nichenet_tiers) == 1
        and len(nichenet_shared_preparations) == 1
        and "" not in nichenet_shared_preparations
        and len(nichenet_prior_sources) == 1
        and "" not in nichenet_prior_sources
    )
    if len(nichenet_orthology_records) == 2 and not nichenet_pair_contract_ok:
        error = ValueError(
            "The official/default and project-LR NicheNet conditions do not share "
            "the same orthology_policy, analysis_tier, and shared preparation"
        )
        if not args.allow_partial:
            raise error
        issues.append(
            {
                "component": "nichenet_condition_pair",
                "view_id": "nichenet_v2__official_and_project_lr",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    cellagentchat_policies = {
        str(record["orthology_policy"])
        for record in cellagentchat_orthology_records.values()
    }
    cellagentchat_tiers = {
        str(record["analysis_tier"])
        for record in cellagentchat_orthology_records.values()
    }
    cellagentchat_shared_contracts = {
        (
            str(record.get("sample_plan_sha256", "")),
            str(record.get("mapped_expression_sha256", "")),
            str(record.get("preparation_manifest_sha256", "")),
            str(record.get("source_commit", "")),
            json.dumps(record.get("source_signature", []), sort_keys=True),
        )
        for record in cellagentchat_orthology_records.values()
    }
    cellagentchat_pair_contract_ok = len(cellagentchat_orthology_records) == 2 and (
        len(cellagentchat_policies) == 1
        and len(cellagentchat_tiers) == 1
        and len(cellagentchat_shared_contracts) == 1
        and all(next(iter(cellagentchat_shared_contracts), ()))
    )
    if len(cellagentchat_orthology_records) == 2 and not cellagentchat_pair_contract_ok:
        error = ValueError(
            "The official/default and project-LR CellAgentChat conditions do not "
            "share the same expression/sample preparation, orthology_policy, and analysis_tier"
        )
        if not args.allow_partial:
            raise error
        issues.append(
            {
                "component": "cellagentchat_condition_pair",
                "view_id": "cellagentchat__official_and_project_lr",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    scores = _add_stage_ranks(combined_scores)
    if scores["view_id"].nunique() < 2 and not args.allow_partial:
        raise RuntimeError("At least two complete score views are required")

    global_stage_map = scores[["stage", "stage_label"]].drop_duplicates()
    inconsistent = global_stage_map.groupby("stage")["stage_label"].nunique()
    if (inconsistent != 1).any():
        conflicts = inconsistent[inconsistent != 1].index.tolist()
        raise ValueError(f"Stage labels disagree across methods at stages: {conflicts}")
    stage_labels = (
        global_stage_map.sort_values("stage")
        .drop_duplicates("stage")
        .set_index("stage")["stage_label"]
        .to_dict()
    )
    global_stages = sorted(float(value) for value in stage_labels)

    control_specs = [
        ("trained", "Trained", paths["control_trained"]),
        ("init_interaction", "Init interaction", paths["control_init"]),
        (
            "randomized_interaction_seed17",
            "Randomized interaction",
            paths["control_random"],
        ),
    ]
    control_frames: list[pd.DataFrame] = []
    for control, label, directory in control_specs:
        try:
            control_frame = load_cytobridge_control(
                directory, control=control, display_label=label
            )
            verified = control_frame.attrs.get("verified_primary_score_artifacts")
            if (
                not isinstance(verified, list)
                or len(verified) != 3
                or not all(
                    isinstance(record, Mapping) and record.get("verified") is True
                    for record in verified
                )
            ):
                raise ValueError(
                    f"CytoBridge control {control!r} lacks three verified score artifacts"
                )
            control_artifact_verification[control] = [
                dict(record) for record in verified
            ]
            control_frames.append(control_frame)
        except Exception as error:
            if not args.allow_partial:
                raise
            issues.append(
                {
                    "component": "cytobridge_control",
                    "view_id": control,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    controls = (
        pd.concat(control_frames, ignore_index=True)
        if control_frames
        else pd.DataFrame(
            columns=[
                "control",
                "control_label",
                "target",
                "metric",
                "estimate",
                "p_value",
                "n_observations",
                "selection",
            ]
        )
    )

    by_stage, summary = pairwise_consistency(scores, top_k=int(args.top_k))
    asymmetry, direction_by_stage, direction_summary = reciprocal_asymmetry(scores)
    stability = stage_stability(
        scores,
        top_k=int(args.top_k),
        global_stages=global_stages,
    )
    coverage = condition_coverage(
        scores,
        expected,
        issues,
        global_stages,
        stage_labels,
    )

    artifacts: list[Path] = []

    def write_csv(frame: pd.DataFrame, name: str, *, gzip: bool = False) -> Path:
        path = output / name
        frame.to_csv(path, index=False, compression="gzip" if gzip else None)
        artifacts.append(path)
        return path

    write_csv(scores, "canonical_type_pair_scores.csv.gz", gzip=True)
    write_csv(by_stage, "pairwise_consistency_by_stage.csv")
    write_csv(summary, "pairwise_consistency_summary.csv")
    write_csv(asymmetry, "reciprocal_rank_asymmetry.csv.gz", gzip=True)
    write_csv(direction_by_stage, "directionality_concordance_by_stage.csv")
    write_csv(direction_summary, "directionality_concordance_summary.csv")
    write_csv(stability, "stage_stability.csv")
    write_csv(coverage, "condition_coverage.csv")
    write_csv(controls, "cytobridge_control_metrics.csv")
    structural_zero_audit = (
        pd.concat(zero_audit_frames, ignore_index=True)
        if zero_audit_frames
        else pd.DataFrame()
    )
    structural_columns = [
        "view_id",
        "display_label",
        "method",
        "database_condition",
        "universe_scope",
        "stage",
        "stage_label",
        "receiver_unit",
        "n_cell_types",
        "expected_directed_rows",
        "native_emitted_rows",
        "native_emitted_positive_rows",
        "native_emitted_zero_rows",
        "structural_zero_filled_rows",
        "verified_complete_evaluated_universe",
        "unevaluated_units_zero_filled",
        "method_unavailable_lr_rows_zero_filled",
        "provenance_manifest_path",
    ]
    for column in structural_columns:
        if column not in structural_zero_audit:
            structural_zero_audit[column] = (
                ""
                if column
                in {
                    "view_id",
                    "display_label",
                    "method",
                    "database_condition",
                    "universe_scope",
                    "stage_label",
                    "receiver_unit",
                    "provenance_manifest_path",
                }
                else np.nan
            )
    structural_zero_audit = structural_zero_audit[structural_columns]
    structural_zero_path = write_csv(
        structural_zero_audit,
        "structural_zero_audit.csv",
    )
    method_unavailable = (
        pd.concat(method_unavailable_frames, ignore_index=True)
        if method_unavailable_frames
        else pd.DataFrame(
            columns=[
                "view_id",
                "method",
                "database_condition",
                "database_row",
                "interaction_id",
                "ligand",
                "receptor",
                "method_ligand_token",
                "method_receptor_token",
                "reason",
                "status",
                "zero_filled",
            ]
        )
    )
    unavailable_path = write_csv(
        method_unavailable,
        "method_unavailable_lr_rows.csv",
    )
    diagnostic_rows: list[dict[str, Any]] = [
        {
            "component": item["component"],
            "view_id": item["view_id"],
            "severity": "error",
            "status": "missing_or_invalid",
            "database_row": np.nan,
            "interaction_id": "",
            "ligand": "",
            "receptor": "",
            "error": item["error"],
            "zero_filled": np.nan,
        }
        for item in issues
    ]
    for row in method_unavailable.itertuples(index=False):
        diagnostic_rows.append(
            {
                "component": "cellchat_lr_method_unavailable",
                "view_id": row.view_id,
                "severity": "warning",
                "status": row.status,
                "database_row": int(row.database_row),
                "interaction_id": row.interaction_id,
                "ligand": row.ligand,
                "receptor": row.receptor,
                "error": row.reason,
                "zero_filled": False,
            }
        )
    issues_path = write_csv(
        pd.DataFrame(
            diagnostic_rows,
            columns=[
                "component",
                "view_id",
                "severity",
                "status",
                "database_row",
                "interaction_id",
                "ligand",
                "receptor",
                "error",
                "zero_filled",
            ],
        ),
        "input_diagnostics.csv",
    )

    views = scores[["view_id", "display_label"]].drop_duplicates()
    rank_matrix = _matrix_from_summary(
        summary,
        views=views,
        value_column="mean_stage_spearman",
    )
    top_matrix = _matrix_from_summary(
        summary,
        views=views,
        value_column="mean_stage_top_k_jaccard_informative_only",
    )
    direction_matrix = _matrix_from_summary(
        direction_summary,
        views=views,
        value_column="mean_stage_spearman_rank_asymmetry",
    )
    rank_base = output / "rank_concordance"
    _heatmap(
        rank_matrix,
        views["display_label"].tolist(),
        title="Directed type-pair rank concordance across methods",
        colorbar_label="Mean per-stage Spearman ρ",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        output_base=rank_base,
        diagonal_note=(
            "Raw score units are never compared. Off-diagonal values are means of stage-wise "
            "Spearman correlations on the exact shared directed-key universe."
        ),
    )
    top_base = output / "top_edge_overlap"
    _heatmap(
        top_matrix,
        views["display_label"].tolist(),
        title=(
            f"Directed top-edge overlap (requested k={int(args.top_k)}; "
            "informative stages only)"
        ),
        colorbar_label="Mean per-stage Jaccard where k < n shared",
        cmap="viridis",
        vmin=0,
        vmax=1,
        output_base=top_base,
        diagonal_note=(
            "Selection uses positive scores only, caps k by both positive supports, and expands "
            "every kth-score boundary tie. All-zero support is NA; a tie-expanded selection "
            "covering the whole shared universe is audit-only and excluded here."
        ),
    )
    direction_base = output / "directionality_concordance"
    _heatmap(
        direction_matrix,
        views["display_label"].tolist(),
        title="Reciprocal-direction rank-asymmetry concordance",
        colorbar_label="Mean per-stage Spearman ρ",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        output_base=direction_base,
        diagonal_note=(
            "Asymmetry is percentile-rank(A→B) − percentile-rank(B→A); native method units "
            "are not subtracted across methods."
        ),
    )
    coverage_base = output / "condition_coverage"
    _plot_coverage(
        coverage,
        expected=expected,
        global_stages=global_stages,
        stage_labels=stage_labels,
        output_base=coverage_base,
    )
    stability_base = output / "stage_stability"
    _plot_stage_stability(
        stability,
        views=views,
        global_stages=global_stages,
        stage_labels=stage_labels,
        output_base=stability_base,
    )
    for base in (rank_base, top_base, direction_base, coverage_base, stability_base):
        artifacts.extend([base.with_suffix(".png"), base.with_suffix(".pdf")])
    if not controls.empty:
        control_base = output / "cytobridge_control_panel"
        _plot_controls(controls, control_base)
        artifacts.extend(
            [control_base.with_suffix(".png"), control_base.with_suffix(".pdf")]
        )

    expected_view_ids = {str(item["view_id"]) for item in expected}
    loaded_view_ids = set(scores["view_id"].astype(str).unique())
    expected_control_ids = {item[0] for item in control_specs}
    all_primary_score_artifacts_hash_verified = bool(
        set(primary_score_artifact_verification) == expected_view_ids
        and set(control_artifact_verification) == expected_control_ids
        and all(primary_score_artifact_verification.values())
        and all(
            len(records) == 3 and records
            for records in control_artifact_verification.values()
        )
    )
    external_condition_ids = {
        "commot__project_lr",
        "cellchat__project_lr",
        "nichenet_v2__official_mouse_lr",
        "nichenet_v2__project_lr_gate",
        "cellagentchat__official_mouse_default",
        "cellagentchat__project_lr",
    }
    full_grid_view_ids = {
        "commot__project_lr",
        "cellchat__project_lr",
        "cellagentchat__official_mouse_default",
        "cellagentchat__project_lr",
    }
    nichenet_view_ids = {
        "nichenet_v2__official_mouse_lr",
        "nichenet_v2__project_lr_gate",
    }
    full_grid_stages_ok = all(
        set(scores.loc[scores["view_id"] == view_id, "stage"].astype(float).unique())
        == set(EXPECTED_FULL_GRID_STAGES)
        for view_id in full_grid_view_ids
        if view_id in loaded_view_ids
    ) and full_grid_view_ids.issubset(loaded_view_ids)
    full_grid_zero_contracts_ok = all(
        bool(
            zero_completion_records.get(view_id, {}).get(
                "verified_complete_evaluated_universe"
            )
        )
        and bool(
            zero_completion_records.get(view_id, {}).get(
                "full_stage_type_square_required"
            )
        )
        and int(
            zero_completion_records.get(view_id, {}).get("observed_stage_count", -1)
        )
        == len(EXPECTED_FULL_GRID_STAGES)
        and zero_completion_records.get(view_id, {}).get(
            "unevaluated_units_zero_filled"
        )
        is False
        and zero_completion_records.get(view_id, {}).get(
            "method_unavailable_lr_rows_zero_filled"
        )
        is False
        for view_id in full_grid_view_ids
    )
    nichenet_zero_contracts_ok = all(
        bool(
            zero_completion_records.get(view_id, {}).get(
                "verified_complete_evaluated_universe"
            )
        )
        and zero_completion_records.get(view_id, {}).get(
            "full_stage_type_square_required"
        )
        is False
        and zero_completion_records.get(view_id, {}).get(
            "unevaluated_units_zero_filled"
        )
        is False
        for view_id in nichenet_view_ids
    )
    controls_contract_ok = bool(
        set(controls["control"].astype(str))
        == {"trained", "init_interaction", "randomized_interaction_seed17"}
        and set(controls["target"].astype(str)) == {"attention", "exact message"}
        and set(controls["metric"].astype(str))
        == {
            "conditional_residual_spearman_forward_lr",
            "delta_r2_forward_lr_over_confounders",
            "reciprocal_direction_spearman_all_stage",
        }
        and controls.groupby("control").size().eq(6).all()
    )
    six_condition_execution_complete = (
        external_condition_ids.issubset(loaded_view_ids)
        and not bool(issues)
        and nichenet_pair_contract_ok
        and cellagentchat_pair_contract_ok
        and full_grid_stages_ok
        and full_grid_zero_contracts_ok
        and nichenet_zero_contracts_ok
    )
    readiness_checks = {
        "exact_eight_score_views_loaded": (
            loaded_view_ids == expected_view_ids and len(loaded_view_ids) == 8
        ),
        "no_input_issues": not bool(issues),
        "global_observed_stages_are_exactly_five": set(global_stages)
        == set(EXPECTED_FULL_GRID_STAGES),
        "full_grid_methods_have_all_five_stages": full_grid_stages_ok,
        "all_eight_views_have_zero_completion_provenance": set(zero_completion_records)
        == expected_view_ids,
        "full_grid_zero_completion_contracts_verified": full_grid_zero_contracts_ok,
        "nichenet_complete_unit_zero_contracts_verified": nichenet_zero_contracts_ok,
        "cellchat_method_unavailable_rows_not_zero_filled": (
            cellchat_lr_universe is not None
            and cellchat_lr_universe.get("method_unavailable_rows_zero_filled") is False
        ),
        "nichenet_condition_pair_contract_verified": nichenet_pair_contract_ok,
        "cellagentchat_condition_pair_contract_verified": cellagentchat_pair_contract_ok,
        "cytobridge_controls_contract_verified": controls_contract_ok,
        "all_primary_score_artifacts_hash_verified": (
            all_primary_score_artifacts_hash_verified
        ),
        "six_condition_execution_complete": six_condition_execution_complete,
    }
    status = "partial_diagnostic" if args.allow_partial or issues else "complete"
    reviewer_reporting_ready = bool(
        status == "complete"
        and not args.allow_partial
        and all(readiness_checks.values())
    )
    if status == "complete" and not args.allow_partial and not reviewer_reporting_ready:
        failed = sorted(key for key, value in readiness_checks.items() if not value)
        raise RuntimeError(f"Formal reviewer readiness checks failed: {failed}")
    readme_path = output / "README.md"
    _write_readme(
        readme_path,
        status=status,
        top_k=int(args.top_k),
        expected=expected,
        issues=issues,
        zero_completion=zero_completion_records,
        readiness_checks=readiness_checks,
    )
    artifacts.append(readme_path)

    input_records: dict[str, Any] = {}
    for name, path in paths.items():
        input_records[name] = {"path": str(path), "exists": path.exists()}
    manifest = {
        "schema_version": 2,
        "created_at_utc": _utc_now(),
        "workflow": "reviewer_zebrafish_multimethod_directed_ccc_rank_comparison",
        "status": status,
        "formal_reviewer_ready": reviewer_reporting_ready,
        "reviewer_reporting_ready": reviewer_reporting_ready,
        "six_condition_execution_complete": six_condition_execution_complete,
        "formal_readiness_checks": readiness_checks,
        "primary_score_artifact_hash_verification": {
            "score_views": primary_score_artifact_verification,
            "cytobridge_controls": control_artifact_verification,
        },
        "readiness_semantics": {
            "six_condition_execution_complete": (
                "all six requested external method/database conditions passed execution, "
                "pairing, evaluated-universe, and stage contracts"
            ),
            "reviewer_reporting_ready": (
                "all eight score views plus controls and provenance passed report-generation contracts"
            ),
            "readiness_is_not_primary_claim_permission": True,
            "condition_level_primary_claim_allowed_remains_authoritative": True,
        },
        "allow_partial": bool(args.allow_partial),
        "contract": {
            "exact_key": KEYS,
            "rank_scope": "within method/database condition/score view/stage",
            "raw_cross_method_units_compared": False,
            "pairwise_universe": (
                "inner join on exact key after provenance-verified completion of "
                "evaluated structural zeros only"
            ),
            "structural_zero_policy": (
                "COMMOT/CellChat: hash-bound stage cell_type_counts type-square; "
                "NicheNet: complete units only across verified source-stage sender types; "
                "CellAgentChat: validate native complete grid; unavailable units are never zero"
            ),
            "top_k": int(args.top_k),
            "top_k_selection_rule": (
                "positive scores only; effective k=min(requested,n_shared,n_positive_left,"
                "n_positive_right); include every kth-score boundary tie"
            ),
            "top_k_tie_break": "none; kth-score boundary ties are expanded",
            "top_k_informative_rule": (
                "effective k > 0 and neither tie-expanded selection equals the whole "
                "shared directed universe"
            ),
            "primary_top_k_summary": "informative stages only; NA when none",
            "directionality": "within-stage rank percentile A->B minus B->A",
            "stage_stability": "adjacent global observed stages",
            "cytobridge_attention_is_ccc_probability": False,
            "nichenet_type_pair_score_is_native": False,
            "cellchat_method_unavailable_lr_rows_zero_filled": False,
            "nichenet_skipped_or_ineligible_units_zero_filled": False,
            "cellagentchat_native_full_grid_required": True,
            "formal_expected_full_grid_stages": list(EXPECTED_FULL_GRID_STAGES),
        },
        "expected_score_views": expected,
        "loaded_score_views": scores["view_id"].drop_duplicates().tolist(),
        "global_stages": [
            {"stage": stage, "stage_label": stage_labels[stage]}
            for stage in global_stages
        ],
        "inputs": input_records,
        "issues": issues,
        "score_view_zero_completion": zero_completion_records,
        "cellchat_lr_universe": (
            dict(cellchat_lr_universe) if cellchat_lr_universe is not None else None
        ),
        "cellchat_method_unavailable_lr_rows": {
            "count": int(len(method_unavailable)),
            "status": "excluded_from_method_universe_not_biological_zero",
            "zero_filled": False,
            "rows": method_unavailable.to_dict("records"),
        },
        "nichenet_orthology_conditions": nichenet_orthology_records,
        "cellagentchat_orthology_conditions": cellagentchat_orthology_records,
        "artifacts": {
            path.name: _file_record(path)
            for path in artifacts
            if path.is_file() and path != issues_path
        },
    }
    manifest["artifacts"][unavailable_path.name] = _file_record(unavailable_path)
    manifest["artifacts"][issues_path.name] = _file_record(issues_path)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "formal_reviewer_ready": manifest["formal_reviewer_ready"],
                "loaded_score_views": manifest["loaded_score_views"],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
