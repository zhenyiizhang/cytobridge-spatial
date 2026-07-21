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
from hashlib import sha256
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
STRICT_PERMUTATION_STRATA = (
    "stage+sender_type+receiver_type+distance_bin+state_bin"
)
CELLAGENTCHAT_OFFICIAL = "official_mouse_default_celltalkdb"
CELLAGENTCHAT_CUSTOM = "cytobridge_zebrafish_lr_projected_singletons"


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


def _external_stage_contract(
    manifest: Mapping[str, Any], frame: pd.DataFrame, *, score_path: Path
) -> tuple[pd.Series, pd.Series]:
    """Map runner stage IDs to manuscript hpf labels through shared inputs."""

    input_manifest_path = _verify_recorded_artifact(
        manifest.get("input_manifest", {}), expected_name="input_manifest.json"
    )
    shared = _read_json(input_manifest_path)
    stages = shared.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"{input_manifest_path} has no stage records")
    mapping: dict[float, float] = {}
    for record in stages:
        if not isinstance(record, Mapping):
            raise ValueError(f"{input_manifest_path} contains a non-object stage record")
        stage_id = float(record.get("stage"))
        stage_time = float(record.get("stage_time"))
        if not np.isfinite(stage_id) or not np.isfinite(stage_time):
            raise ValueError(f"{input_manifest_path} has a non-finite stage mapping")
        stage_id = float(np.round(stage_id, 12))
        previous = mapping.get(stage_id)
        if previous is not None and not np.isclose(
            previous, stage_time, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{input_manifest_path} maps one stage ID to two times")
        mapping[stage_id] = stage_time

    stage = _stage_values(frame["stage"], path=score_path)
    observed_time = _numeric(frame["stage_time"], name="stage_time", path=score_path)
    expected_time = stage.map(mapping)
    if expected_time.isna().any():
        unknown = sorted(set(stage.loc[expected_time.isna()]))
        raise ValueError(f"{score_path} contains stages absent from shared inputs: {unknown}")
    if not np.allclose(
        observed_time.to_numpy(float),
        expected_time.to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{score_path} stage_time values disagree with the verified shared input manifest"
        )
    stage_label = expected_time.map(lambda value: _format_hpf(float(value)))
    return stage, stage_label


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
            "sender_type": _clean_labels(frame["sender_type"], name="sender_type", path=path),
            "receiver_type": _clean_labels(
                frame["receiver_type"], name="receiver_type", path=path
            ),
            "native_score": _numeric(score, name="native_score", path=path),
        }
    )
    duplicate = result.duplicated(KEYS, keep=False)
    if duplicate.any():
        examples = result.loc[duplicate, KEYS].head(5).to_dict("records")
        raise ValueError(f"{path} has duplicate directed keys after adaptation: {examples}")
    labels_per_stage = result.groupby("stage", sort=False)["stage_label"].nunique()
    if (labels_per_stage != 1).any():
        raise ValueError(f"{path} maps a numeric stage to multiple labels")
    result["heterotypic"] = result["sender_type"] != result["receiver_type"]
    return result.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def _load_cytobridge_views(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = _validate_manifest_identity(
        directory,
        filename="run_manifest.json",
        expected={"method": "cytobridge_one_layer_spatial_attention_and_exact_message"},
    )
    if manifest.get("interpretation", {}).get("probability_claim") is not False:
        raise ValueError("CytoBridge manifest must explicitly set probability_claim=false")
    path = directory / "type_pair_summary.csv"
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
    path = directory / "commot_type_pair_scores.csv.gz"
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
    if set(frame["database_variant"].astype(str)) != {
        "current_zebrafish_lr_database"
    }:
        raise ValueError(f"{path} contains an unexpected database variant")
    stage, stage_label = _external_stage_contract(manifest, frame, score_path=path)
    return _canonical(
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


def _load_cellchat(directory: Path) -> pd.DataFrame:
    manifest = _validate_manifest_identity(
        directory,
        filename="manifest.json",
        expected={
            "method": "CellChat",
            "database_variant": "current_zebrafish_lr_database",
        },
    )
    validation = manifest.get("database_validation")
    if not isinstance(validation, Mapping):
        raise ValueError("CellChat manifest lacks database_validation")
    requested = int(validation.get("rows_requested", -1))
    eligible = int(validation.get("rows_eligible", -1))
    excluded = int(validation.get("rows_excluded", -1))
    if min(requested, eligible, excluded) < 0 or requested != eligible + excluded:
        raise ValueError("CellChat manifest has inconsistent requested/eligible/excluded counts")
    if validation.get("excluded_rows_are_method_unavailable_not_biological_zero") is not True:
        raise ValueError(
            "CellChat manifest must mark excluded rows as method-unavailable, not zero"
        )
    policy = str(manifest.get("design", {}).get("method_unavailable_policy", ""))
    if "never zero-filled" not in policy:
        raise ValueError("CellChat manifest lost the method-unavailable no-zero-fill policy")
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
    path = directory / "cellchat_type_pair_scores.csv.gz"
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
    if set(frame["database_variant"].astype(str)) != {
        "current_zebrafish_lr_database"
    }:
        raise ValueError(f"{path} contains an unexpected database variant")
    stage, stage_label = _external_stage_contract(manifest, frame, score_path=path)
    result = _canonical(
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
    return result


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
        raise ValueError("NicheNet primary_claim_allowed disagrees with orthology_policy")
    method_label = str(manifest.get("method_label", ""))
    if orthology_policy == "one2one_bijective_all_confidence" and (
        "orthology sensitivity: confidence unfiltered" not in method_label
    ):
        raise ValueError(
            "All-confidence NicheNet output must explicitly label its orthology sensitivity"
        )
    path = directory / "sender_ligand_activity.csv"
    frame = pd.read_csv(path)
    required = {
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
    if not frame["activity_scope"].astype(str).str.contains(
        "not_sender_specific", regex=False
    ).all():
        raise ValueError(f"{path} lost the NicheNet activity-scope caveat")
    work = frame.rename(
        columns={"sender": "sender_type", "receiver": "receiver_type"}
    ).copy()
    work["aupr_corrected"] = _numeric(
        work["aupr_corrected"], name="aupr_corrected", path=path
    )
    work["positive_aupr_corrected"] = work["aupr_corrected"].clip(lower=0.0)
    keys = [
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
        aggregated,
        path=path,
        method="NicheNet-v2 (cross-species)",
        database_condition=condition,
        score_view="sum_positive_sender_associated_aupr_corrected",
        display_label=label,
        view_id=view_id,
        score=aggregated["positive_aupr_corrected"],
        stage=aggregated["source_stage_id"],
        stage_label=aggregated["source_stage_label"],
    )
    result["orthology_policy"] = orthology_policy
    result["analysis_tier"] = str(policy["analysis_tier"])
    result["primary_claim_allowed"] = bool(policy["primary_claim_allowed"])
    result.attrs["nichenet_orthology"] = {
        "orthology_policy": orthology_policy,
        "analysis_tier": policy["analysis_tier"],
        "primary_claim_allowed": policy["primary_claim_allowed"],
        "method_label": method_label,
    }
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
    native_primary = manifest.get("design", {}).get("native_primary")
    if "Bonferroni-significant LR pairs" not in str(native_primary):
        raise ValueError("CellAgentChat manifest has unexpected native-primary semantics")
    path = directory / "cellagentchat_type_pair_scores.csv"
    frame = pd.read_csv(path)
    required = {
        "stage",
        "stage_label",
        "sender_type",
        "receiver_type",
        "cellagentchat_native_primary_mean",
    }
    _require_columns(frame, required, path)
    official = condition == CELLAGENTCHAT_OFFICIAL
    label = (
        "CellAgentChat | official mouse DB"
        if official
        else "CellAgentChat | project LR"
    )
    return _canonical(
        frame,
        path=path,
        method="CellAgentChat (cross-species)",
        database_condition=condition,
        score_view="mean_bonferroni_significant_lr_count",
        display_label=label,
        view_id=(
            "cellagentchat__official_mouse_default"
            if official
            else "cellagentchat__project_lr"
        ),
        score=frame["cellagentchat_native_primary_mean"],
        stage=frame["stage"],
        stage_label=frame["stage_label"],
    )


def _add_stage_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby(["view_id", "stage"], sort=False)["native_score"]
    result["within_stage_rank_high"] = grouped.rank(
        method="average", ascending=False
    )
    counts = grouped.transform("size").astype(int)
    result["n_directed_pairs_in_view_stage"] = counts
    denominator = (counts - 1).replace(0, 1)
    result["within_stage_rank_percentile_high"] = 1.0 - (
        result["within_stage_rank_high"] - 1.0
    ) / denominator
    result.loc[counts == 1, "within_stage_rank_percentile_high"] = 1.0
    return result


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return float("nan")
    return float(left.corr(right, method="spearman"))


def _top_keys(frame: pd.DataFrame, score: str, k: int) -> set[tuple[str, str]]:
    ordered = frame.sort_values(
        [score, "sender_type", "receiver_type"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(k)
    return set(zip(ordered["sender_type"], ordered["receiver_type"]))


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
                effective_k = min(int(top_k), n_shared)
                if effective_k:
                    common_left = merged[
                        ["sender_type", "receiver_type", "native_score_left"]
                    ].rename(columns={"native_score_left": "score"})
                    common_right = merged[
                        ["sender_type", "receiver_type", "native_score_right"]
                    ].rename(columns={"native_score_right": "score"})
                    left_top = _top_keys(common_left, "score", effective_k)
                    right_top = _top_keys(common_right, "score", effective_k)
                    intersection = len(left_top & right_top)
                    union = len(left_top | right_top)
                    jaccard = intersection / union if union else float("nan")
                else:
                    intersection = 0
                    union = 0
                    jaccard = float("nan")
                rows.append(
                    {
                        "view_id_left": left_id,
                        "display_label_left": labels[left_id],
                        "view_id_right": right_id,
                        "display_label_right": labels[right_id],
                        "stage": float(stage),
                        "stage_label": (
                            merged["stage_label_left"].iloc[0]
                            if n_shared
                            else ""
                        ),
                        "n_shared_directed_pairs": n_shared,
                        "spearman_rank_concordance": _safe_spearman(
                            merged["native_score_left"], merged["native_score_right"]
                        ),
                        "requested_top_k": int(top_k),
                        "effective_top_k": effective_k,
                        "top_k_intersection": intersection,
                        "top_k_union": union,
                        "top_k_overlap_fraction": (
                            intersection / effective_k if effective_k else float("nan")
                        ),
                        "top_k_jaccard": jaccard,
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
                "n_shared_directed_pairs_total",
                "mean_stage_spearman",
                "median_stage_spearman",
                "mean_stage_top_k_jaccard",
                "median_stage_top_k_jaccard",
            ]
        )
        return by_stage, summary
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
            n_shared_directed_pairs_total=("n_shared_directed_pairs", "sum"),
            mean_stage_spearman=("spearman_rank_concordance", "mean"),
            median_stage_spearman=("spearman_rank_concordance", "median"),
            mean_stage_top_k_jaccard=("top_k_jaccard", "mean"),
            median_stage_top_k_jaccard=("top_k_jaccard", "median"),
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
            ["sender_type", "receiver_type", "native_score", "within_stage_rank_percentile_high"]
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
                        "effective_top_k": 0,
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
            effective_k = min(int(top_k), len(merged))
            if effective_k:
                left_common = merged[
                    ["sender_type", "receiver_type", "native_score_from"]
                ].rename(columns={"native_score_from": "score"})
                right_common = merged[
                    ["sender_type", "receiver_type", "native_score_to"]
                ].rename(columns={"native_score_to": "score"})
                left_top = _top_keys(left_common, "score", effective_k)
                right_top = _top_keys(right_common, "score", effective_k)
                union = len(left_top | right_top)
                jaccard = len(left_top & right_top) / union if union else float("nan")
            else:
                jaccard = float("nan")
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
                    "effective_top_k": int(effective_k),
                    "top_k_jaccard": jaccard,
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
    _read_json(directory / "run_manifest.json")
    permutation_path = directory / "conditional_permutation_tests.csv"
    nested_path = directory / "nested_grouped_cv_metrics.csv"
    reciprocal_path = directory / "reciprocal_edge_direction_tests.csv"
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
                "estimate": float(
                    reciprocal_row["spearman_with_lr_direction_delta"]
                ),
                "p_value": float(reciprocal_row["empirical_p_greater"]),
                "n_observations": int(reciprocal_row["n_reciprocal_pairs"]),
                "selection": "reciprocal directed edges, all stages",
            }
        )
    return pd.DataFrame(rows)


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
            axis.text(column, row, text, ha="center", va="center", fontsize=7, color=color)
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
    colorbar.set_label("log(1 + emitted directed keys); cell text is raw count")
    figure.text(
        0.01,
        0.01,
        "Grey = method/condition or stage not emitted. Coverage is reported before any pairwise inner join.",
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
        rho[index[row.view_id], transition_index[position]] = row.spearman_rank_stability
        jaccard[index[row.view_id], transition_index[position]] = row.top_k_jaccard
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
        (axes[1], jaccard, "Adjacent-stage top-k stability", "viridis", 0, 1),
    ):
        color_map = plt.get_cmap(cmap).copy()
        color_map.set_bad("#E5E7EB")
        image = axis.imshow(
            np.ma.masked_invalid(matrix), cmap=color_map, vmin=vmin, vmax=vmax, aspect="auto"
        )
        axis.set_xticks(np.arange(len(transitions)), transition_labels, rotation=35, ha="right")
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
    path.write_text(
        f"""# Zebrafish directed-CCC comparison

Status: **{status}**

This directory compares methods only after ranking each method/condition within
each observed stage. Raw COMMOT mass, CellChat probability, NicheNet ligand
activity, CellAgentChat significant-pair count, CytoBridge attention, and exact
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
separate model-internal view.

## Comparison contract

- Exact join key: `stage, sender_type, receiver_type`.
- Rank and percentile: computed within each method/condition/stage.
- Pairwise universe: inner join of emitted directed keys; coverage is reported
  separately so missing keys are never silently converted to evidence.
- Top-edge overlap: deterministic top-{top_k} on the shared key universe; the
  effective k is reduced when fewer shared keys exist.
- Directionality: difference of within-stage rank percentiles for reciprocal
  A→B and B→A edges.
- Stage stability: adjacent global observed stages only.

## Main artifacts

- `canonical_type_pair_scores.csv.gz`: native values plus within-stage ranks.
- `pairwise_consistency_by_stage.csv` and `pairwise_consistency_summary.csv`.
- `reciprocal_rank_asymmetry.csv.gz` and directionality concordance summaries.
- `stage_stability.csv` and `condition_coverage.csv`.
- `method_unavailable_lr_rows.csv`: CellChat-incompatible requested LR rows;
  these are excluded from its method universe and are never zero-filled.
- `cytobridge_control_metrics.csv`: trained, Init_interaction, and randomized
  interaction controls.
- PNG/PDF panels for rank concordance, top-edge overlap, condition coverage,
  reciprocal directionality, stage stability, and CytoBridge controls.

## Diagnostics

{issue_text}

`partial_diagnostic` output was generated with `--allow-partial` and must not be
used as the formal reviewer comparison until every required condition is
available and the script completes without that flag.
""",
        encoding="utf-8",
    )


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.run_root.expanduser().resolve()
    return {
        "cytobridge": (args.cytobridge_dir or root / "01_cytobridge").expanduser().resolve(),
        "control_trained": (
            args.controls_trained_dir or root / "02_attention_controls"
        ).expanduser().resolve(),
        "control_init": (
            args.controls_init_dir or root / "02_attention_controls_init_interaction"
        ).expanduser().resolve(),
        "control_random": (
            args.controls_random_dir or root / "02_attention_controls_random_seed17"
        ).expanduser().resolve(),
        "commot": (
            args.commot_dir or root / "03_external_ccc" / "commot_current_lr"
        ).expanduser().resolve(),
        "cellchat": (
            args.cellchat_dir or root / "03_external_ccc" / "cellchat_current_lr"
        ).expanduser().resolve(),
        "nichenet_default": (
            args.nichenet_default_dir
            or root / "04_nichenet" / "02_default_mouse_v2"
        ).expanduser().resolve(),
        "nichenet_custom": (
            args.nichenet_custom_dir
            or root / "04_nichenet" / "03_custom_zebrafish_lr"
        ).expanduser().resolve(),
        "cellagentchat": (
            args.cellagentchat_dir or root / "05_cellagentchat"
        ).expanduser().resolve(),
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
            "display_label": "CellAgentChat | official mouse DB",
            "method": "CellAgentChat (cross-species)",
            "database_condition": CELLAGENTCHAT_OFFICIAL,
            "score_view": "mean_bonferroni_significant_lr_count",
        },
        {
            "view_id": "cellagentchat__project_lr",
            "display_label": "CellAgentChat | project LR",
            "method": "CellAgentChat (cross-species)",
            "database_condition": CELLAGENTCHAT_CUSTOM,
            "score_view": "mean_bonferroni_significant_lr_count",
        },
    ]
    loaded: list[pd.DataFrame] = []
    issues: list[dict[str, str]] = []
    method_unavailable_frames: list[pd.DataFrame] = []
    nichenet_orthology_records: dict[str, Mapping[str, Any]] = {}
    cellchat_lr_universe: Mapping[str, Any] | None = None

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
            ["cellagentchat__official_mouse_default"],
            lambda: (
                _load_cellagentchat(
                    paths["cellagentchat"] / CELLAGENTCHAT_OFFICIAL,
                    condition=CELLAGENTCHAT_OFFICIAL,
                ),
            ),
        ),
        (
            ["cellagentchat__project_lr"],
            lambda: (
                _load_cellagentchat(
                    paths["cellagentchat"] / CELLAGENTCHAT_CUSTOM,
                    condition=CELLAGENTCHAT_CUSTOM,
                ),
            ),
        ),
    ]
    for view_ids, loader in loaders:
        try:
            result = tuple(loader())
            if len(result) != len(view_ids):
                raise RuntimeError("Loader returned an unexpected number of score views")
            for view_id, frame in zip(view_ids, result):
                unavailable = frame.attrs.get("method_unavailable_lr_rows")
                if isinstance(unavailable, pd.DataFrame):
                    method_unavailable_frames.append(unavailable.copy())
                universe = frame.attrs.get("cellchat_lr_universe")
                if isinstance(universe, Mapping):
                    cellchat_lr_universe = dict(universe)
                orthology = frame.attrs.get("nichenet_orthology")
                if isinstance(orthology, Mapping):
                    nichenet_orthology_records[view_id] = dict(orthology)
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
        raise RuntimeError("No valid score view is available, even for partial diagnostics")
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
        str(record["analysis_tier"])
        for record in nichenet_orthology_records.values()
    }
    if len(nichenet_orthology_records) == 2 and (
        len(nichenet_policies) != 1 or len(nichenet_tiers) != 1
    ):
        error = ValueError(
            "The official/default and project-LR NicheNet conditions do not share "
            "the same orthology_policy and analysis_tier"
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
        ("randomized_interaction_seed17", "Randomized interaction", paths["control_random"]),
    ]
    control_frames: list[pd.DataFrame] = []
    for control, label, directory in control_specs:
        try:
            control_frames.append(
                load_cytobridge_control(directory, control=control, display_label=label)
            )
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
        value_column="mean_stage_top_k_jaccard",
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
        title=f"Directed top-edge overlap (requested k={int(args.top_k)})",
        colorbar_label="Mean per-stage Jaccard",
        cmap="viridis",
        vmin=0,
        vmax=1,
        output_base=top_base,
        diagonal_note=(
            "Top-k is selected independently within each method on the shared stage/type-pair "
            "universe; effective k is recorded when fewer than requested keys exist."
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
        artifacts.extend([control_base.with_suffix(".png"), control_base.with_suffix(".pdf")])

    status = "partial_diagnostic" if args.allow_partial or issues else "complete"
    readme_path = output / "README.md"
    _write_readme(
        readme_path,
        status=status,
        top_k=int(args.top_k),
        expected=expected,
        issues=issues,
    )
    artifacts.append(readme_path)

    input_records: dict[str, Any] = {}
    for name, path in paths.items():
        input_records[name] = {"path": str(path), "exists": path.exists()}
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "workflow": "reviewer_zebrafish_multimethod_directed_ccc_rank_comparison",
        "status": status,
        "formal_reviewer_ready": status == "complete" and not bool(args.allow_partial),
        "allow_partial": bool(args.allow_partial),
        "contract": {
            "exact_key": KEYS,
            "rank_scope": "within method/database condition/score view/stage",
            "raw_cross_method_units_compared": False,
            "pairwise_universe": "inner join on exact key; no silent zero fill",
            "top_k": int(args.top_k),
            "top_k_tie_break": "score descending, then sender_type and receiver_type ascending",
            "directionality": "within-stage rank percentile A->B minus B->A",
            "stage_stability": "adjacent global observed stages",
            "cytobridge_attention_is_ccc_probability": False,
            "nichenet_type_pair_score_is_native": False,
            "cellchat_method_unavailable_lr_rows_zero_filled": False,
        },
        "expected_score_views": expected,
        "loaded_score_views": scores["view_id"].drop_duplicates().tolist(),
        "global_stages": [
            {"stage": stage, "stage_label": stage_labels[stage]}
            for stage in global_stages
        ],
        "inputs": input_records,
        "issues": issues,
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
        "artifacts": {
            path.name: _file_record(path)
            for path in artifacts
            if path.is_file() and path != issues_path
        },
    }
    manifest["artifacts"][unavailable_path.name] = _file_record(unavailable_path)
    manifest["artifacts"][issues_path.name] = _file_record(issues_path)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
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
