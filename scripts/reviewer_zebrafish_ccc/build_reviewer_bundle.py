#!/usr/bin/env python3
"""Build a self-contained reviewer bundle for the zebrafish CCC analysis.

The bundle is intentionally a reporting layer.  It does not recompute any
method and it never changes an upstream result directory.  It validates the
formal multi-method comparison, reviewer-axis validation, and the six external
method/database condition manifests before copying their key audit artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


CELLAGENTCHAT_OFFICIAL = "official_mouse_default_celltalkdb"
CELLAGENTCHAT_CUSTOM = "cytobridge_zebrafish_lr_projected_singletons"

INTERNAL_VIEW_IDS = (
    "cytobridge__trained__attention",
    "cytobridge__trained__exact_message",
)
EXTERNAL_VIEW_IDS = (
    "commot__project_lr",
    "cellchat__project_lr",
    "nichenet_v2__official_mouse_lr",
    "nichenet_v2__project_lr_gate",
    "cellagentchat__official_mouse_default",
    "cellagentchat__project_lr",
)
EXPECTED_VIEW_IDS = INTERNAL_VIEW_IDS + EXTERNAL_VIEW_IDS
FULL_STAGE_TYPE_SQUARE_VIEW_IDS = (
    "commot__project_lr",
    "cellchat__project_lr",
    "cellagentchat__official_mouse_default",
    "cellagentchat__project_lr",
)
NICHENET_VIEW_IDS = (
    "nichenet_v2__official_mouse_lr",
    "nichenet_v2__project_lr_gate",
)
EXPECTED_FORMAL_READINESS_CHECKS = {
    "exact_eight_score_views_loaded",
    "no_input_issues",
    "global_observed_stages_are_exactly_five",
    "full_grid_methods_have_all_five_stages",
    "all_eight_views_have_zero_completion_provenance",
    "full_grid_zero_completion_contracts_verified",
    "nichenet_complete_unit_zero_contracts_verified",
    "cellchat_method_unavailable_rows_not_zero_filled",
    "nichenet_condition_pair_contract_verified",
    "cellagentchat_condition_pair_contract_verified",
    "cytobridge_controls_contract_verified",
    "all_primary_score_artifacts_hash_verified",
    "six_condition_execution_complete",
}

COMPARISON_TABLES = (
    "canonical_type_pair_scores.csv.gz",
    "pairwise_consistency_by_stage.csv",
    "pairwise_consistency_summary.csv",
    "reciprocal_rank_asymmetry.csv.gz",
    "directionality_concordance_by_stage.csv",
    "directionality_concordance_summary.csv",
    "stage_stability.csv",
    "condition_coverage.csv",
    "cytobridge_control_metrics.csv",
    "structural_zero_audit.csv",
    "method_unavailable_lr_rows.csv",
    "input_diagnostics.csv",
)
COMPARISON_FIGURES = tuple(
    f"{stem}.{suffix}"
    for stem in (
        "rank_concordance",
        "top_edge_overlap",
        "condition_coverage",
        "directionality_concordance",
        "stage_stability",
        "cytobridge_control_panel",
    )
    for suffix in ("png", "pdf")
)
VALIDATION_TABLES = (
    "sender_receiver_contexts.csv",
    "context_enrichment_tests.csv",
    "degree_matched_conditional_tests.csv",
    "degree_matching_strata_audit.csv.gz",
    "lr_axis_availability_audit.csv",
    "lr_axis_stage_scores.csv.gz",
    "top_identifiable_lr_axes.csv",
    "known_axis_database_provenance.csv",
    "known_axis_stage_scores.csv",
)
OPTIONAL_VALIDATION_TABLES = (
    "virtual_ablation_summary.csv",
    "virtual_ablation_observed_stages.csv",
)
VALIDATION_FIGURES = (
    "reviewer_validation_axes.png",
    "reviewer_validation_axes.pdf",
)
POSITIVE_CONSISTENCY_TABLES = (
    "harmonized_type_pair_scores.csv.gz",
    "consensus_by_stage.csv",
    "consensus_summary.csv",
    "pairwise_sensitivity_by_stage.csv",
    "pairwise_sensitivity_summary.csv",
    "top_signal_overlap_by_stage.csv",
    "top_signal_overlap_summary.csv",
    "spatial_proximity_by_stage.csv",
    "spatial_proximity_summary.csv",
    "spatial_beyond_proximity_conditional_tests.csv",
    "pathway_enrichment.csv",
    "nichenet_downstream_ligand_detail.csv",
    "nichenet_downstream_consistency_summary.csv",
)
POSITIVE_CONSISTENCY_FIGURES = tuple(
    f"{stem}.{suffix}"
    for stem in ("positive_consistency_overview", "top_signal_biology")
    for suffix in ("png", "pdf")
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--validation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--commot-dir", type=Path)
    parser.add_argument("--cellchat-dir", type=Path)
    parser.add_argument("--nichenet-default-dir", type=Path)
    parser.add_argument("--nichenet-custom-dir", type=Path)
    parser.add_argument("--cellagentchat-dir", type=Path)
    parser.add_argument(
        "--positive-consistency-dir",
        type=Path,
        help=(
            "Optional hash-verified output from paper_style_positive_consistency.py. "
            "When supplied, its figures, tables, and reviewer-response notes are bundled."
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _series_is_false(values: pd.Series) -> bool:
    normalized = values.astype(str).str.strip().str.casefold()
    return bool(normalized.isin({"false", "0", "0.0"}).all())


def _manifest_artifacts_by_name(
    manifest: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    records = manifest.get("artifacts", {})
    if not isinstance(records, Mapping):
        raise ValueError("Manifest artifacts field is not an object")
    result: dict[str, Mapping[str, Any]] = {}
    for key, record in records.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"Artifact record {key!r} is not an object")
        recorded_path = Path(str(record.get("path", key)))
        result[recorded_path.name] = record
    return result


def _verify_artifact(path: Path, record: Mapping[str, Any] | None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if record is None:
        raise ValueError(f"No manifest artifact record for {path.name}")
    recorded_name = Path(str(record.get("path", path.name))).name
    if recorded_name != path.name:
        raise ValueError(
            f"Artifact record names {recorded_name!r}, expected {path.name!r}"
        )
    recorded_size = record.get("size_bytes", record.get("bytes"))
    if recorded_size is None or int(recorded_size) != path.stat().st_size:
        raise ValueError(f"Manifest byte count does not match {path}")
    recorded_hash = str(record.get("sha256", "")).casefold()
    if not recorded_hash or recorded_hash != _sha256(path).casefold():
        raise ValueError(f"Manifest SHA256 does not match {path}")


def _validate_positive_consistency(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    manifest = _read_json(directory / "manifest.json")
    _require(
        manifest.get("workflow")
        == "zebrafish_paper_style_positive_communication_consistency",
        "Unexpected positive-consistency workflow",
    )
    primary = manifest.get("primary_design", {})
    _require(
        isinstance(primary, Mapping)
        and primary.get("name") == "external_only_native_primary"
        and primary.get("cytobridge_excluded") is True,
        "Positive-consistency primary design is not external-only",
    )
    supporting = manifest.get("supporting_design", {})
    _require(
        isinstance(supporting, Mapping)
        and supporting.get("self_inclusion_disclosed") is True,
        "Positive-consistency all-method self-inclusion is not disclosed",
    )
    correction = manifest.get("cellagentchat_ctps_correction", {})
    _require(
        isinstance(correction, Mapping)
        and correction.get("source_column")
        == "cellagentchat_significant_score_sum_mean"
        and correction.get("source_files_mutated") is False,
        "Positive-consistency CellAgentChat CTPS correction contract is missing",
    )
    records = _manifest_artifacts_by_name(manifest)
    for filename in (*POSITIVE_CONSISTENCY_TABLES, *POSITIVE_CONSISTENCY_FIGURES):
        _verify_artifact(directory / filename, records.get(filename))
    return manifest, records


def _prepare_output(path: Path, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_method_dirs(
    comparison_manifest: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Path]:
    inputs = comparison_manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Comparison manifest lacks its input-directory records")

    def resolve(key: str, override: Path | None) -> Path:
        if override is not None:
            return override.expanduser().resolve()
        record = inputs.get(key)
        if not isinstance(record, Mapping) or not record.get("path"):
            raise ValueError(
                f"Comparison manifest lacks inputs.{key}.path; pass an explicit override"
            )
        return Path(str(record["path"])).expanduser().resolve()

    return {
        "commot": resolve("commot", args.commot_dir),
        "cellchat": resolve("cellchat", args.cellchat_dir),
        "nichenet_default": resolve("nichenet_default", args.nichenet_default_dir),
        "nichenet_custom": resolve("nichenet_custom", args.nichenet_custom_dir),
        "cellagentchat": resolve("cellagentchat", args.cellagentchat_dir),
    }


def _validate_comparison(directory: Path) -> dict[str, Any]:
    manifest = _read_json(directory / "manifest.json")
    _require(
        manifest.get("workflow")
        == "reviewer_zebrafish_multimethod_directed_ccc_rank_comparison",
        "Unexpected comparison workflow",
    )
    _require(manifest.get("status") == "complete", "Comparison is not complete")
    _require(
        manifest.get("formal_reviewer_ready") is True,
        "Comparison is not marked formal_reviewer_ready",
    )
    _require(
        manifest.get("reviewer_reporting_ready") is True,
        "Comparison is not marked reviewer_reporting_ready",
    )
    _require(
        manifest.get("six_condition_execution_complete") is True,
        "The six external method/database executions are not complete",
    )
    _require(
        manifest.get("allow_partial") is False, "Partial comparison is not allowed"
    )
    _require(
        manifest.get("issues") in (None, []),
        "Formal comparison has unresolved input issues",
    )
    loaded = tuple(str(value) for value in manifest.get("loaded_score_views", []))
    _require(
        set(loaded) == set(EXPECTED_VIEW_IDS) and len(loaded) == len(EXPECTED_VIEW_IDS),
        "Comparison must contain exactly the two CytoBridge views and six external conditions",
    )
    expected = manifest.get("expected_score_views")
    _require(isinstance(expected, list), "Comparison lacks expected_score_views")
    expected_ids = [
        str(item.get("view_id")) for item in expected if isinstance(item, Mapping)
    ]
    _require(
        set(expected_ids) == set(EXPECTED_VIEW_IDS)
        and len(expected_ids) == len(EXPECTED_VIEW_IDS),
        "Comparison expected_score_views does not match the eight-view contract",
    )
    contract = manifest.get("contract", {})
    _require(
        isinstance(contract, Mapping)
        and contract.get("raw_cross_method_units_compared") is False,
        "Comparison must explicitly reject raw cross-method score comparisons",
    )
    _require(
        contract.get("cytobridge_attention_is_ccc_probability") is False,
        "Comparison must explicitly state that attention is not a CCC probability",
    )
    _require(
        contract.get("nichenet_type_pair_score_is_native") is False,
        "Comparison must retain the derived NicheNet score caveat",
    )
    _require(
        contract.get("cellchat_method_unavailable_lr_rows_zero_filled") is False,
        "CellChat method-unavailable LR rows must not be zero-filled",
    )
    _require(
        bool(str(contract.get("structural_zero_policy", "")).strip()),
        "Comparison lacks an evaluated-universe structural-zero policy",
    )
    top_k_rule = str(contract.get("top_k_selection_rule", "")).casefold()
    _require(
        "positive" in top_k_rule and "boundary tie" in top_k_rule,
        "Comparison top-k contract is not positive-support/boundary-tie aware",
    )
    _require(
        "ties are expanded" in str(contract.get("top_k_tie_break", "")).casefold(),
        "Comparison must expand rather than arbitrarily break kth-boundary ties",
    )
    _require(
        [float(value) for value in contract.get("formal_expected_full_grid_stages", [])]
        == [0.0, 1.0, 2.0, 3.0, 4.0],
        "Comparison full-grid stage contract is not the five observed stages",
    )
    readiness = manifest.get("formal_readiness_checks")
    _require(
        isinstance(readiness, Mapping),
        "Comparison lacks formal_readiness_checks",
    )
    missing_readiness = sorted(EXPECTED_FORMAL_READINESS_CHECKS - set(readiness))
    _require(
        not missing_readiness,
        f"Comparison lacks formal readiness checks: {missing_readiness}",
    )
    _require(
        all(isinstance(value, bool) and value for value in readiness.values()),
        "At least one formal comparison readiness check failed or is not boolean",
    )
    semantics = manifest.get("readiness_semantics")
    _require(
        isinstance(semantics, Mapping)
        and semantics.get("readiness_is_not_primary_claim_permission") is True
        and semantics.get("condition_level_primary_claim_allowed_remains_authoritative")
        is True,
        "Comparison readiness must remain separate from primary-claim permission",
    )
    unavailable = manifest.get("cellchat_method_unavailable_lr_rows")
    _require(
        isinstance(unavailable, Mapping)
        and unavailable.get("zero_filled") is False
        and unavailable.get("status")
        == "excluded_from_method_universe_not_biological_zero",
        "Comparison lacks the CellChat method-unavailable/no-zero-fill audit",
    )
    return manifest


def _validate_method_unavailable_rows(
    directory: Path, manifest: Mapping[str, Any]
) -> None:
    path = directory / "method_unavailable_lr_rows.csv"
    artifacts = _manifest_artifacts_by_name(manifest)
    _verify_artifact(path, artifacts.get(path.name))
    frame = pd.read_csv(path)
    record = manifest["cellchat_method_unavailable_lr_rows"]
    _require(
        int(record.get("count", -1)) == len(frame),
        "CellChat method-unavailable count disagrees with its audit table",
    )
    if frame.empty:
        return
    _require("zero_filled" in frame, "CellChat unavailable table lacks zero_filled")
    _require(
        _series_is_false(frame["zero_filled"]),
        "A CellChat method-unavailable LR row was zero-filled",
    )
    if "status" in frame:
        _require(
            frame["status"]
            .astype(str)
            .eq("method_unavailable_excluded_not_biological_zero")
            .all(),
            "CellChat unavailable table has an invalid status",
        )


def _validate_structural_zero_completion(
    directory: Path, manifest: Mapping[str, Any]
) -> pd.DataFrame:
    """Validate zero completion only within each method's evaluated universe."""

    completion = manifest.get("score_view_zero_completion")
    _require(
        isinstance(completion, Mapping),
        "Comparison lacks score_view_zero_completion provenance",
    )
    missing_views = sorted(set(EXPECTED_VIEW_IDS) - set(completion))
    _require(
        not missing_views,
        f"Zero-completion provenance is missing score views: {missing_views}",
    )
    required_record_fields = {
        "universe_scope",
        "universe_source",
        "runner_export_contract",
        "full_stage_type_square_required",
        "expected_stage_count",
        "observed_stage_count",
        "expected_rows",
        "native_emitted_rows",
        "structural_zero_filled_rows",
        "verified_complete_evaluated_universe",
        "unevaluated_units_zero_filled",
        "method_unavailable_lr_rows_zero_filled",
    }
    for view_id in EXPECTED_VIEW_IDS:
        record = completion[view_id]
        _require(
            isinstance(record, Mapping),
            f"Zero-completion record for {view_id} is not an object",
        )
        missing = sorted(required_record_fields - set(record))
        _require(
            not missing,
            f"Zero-completion record for {view_id} lacks fields: {missing}",
        )
        _require(
            bool(str(record["universe_scope"]).strip())
            and bool(str(record["runner_export_contract"]).strip()),
            f"Zero-completion universe provenance is empty for {view_id}",
        )
        if view_id in EXTERNAL_VIEW_IDS:
            _require(
                isinstance(record["universe_source"], Mapping),
                f"External zero-completion source is not an artifact record for {view_id}",
            )
        else:
            _require(
                record["universe_source"] is None,
                f"Native CytoBridge rows unexpectedly claim an independent universe source: {view_id}",
            )
        expected_stages_raw = record["expected_stage_count"]
        observed_stages = int(record["observed_stage_count"])
        expected_rows = int(record["expected_rows"])
        emitted_rows = int(record["native_emitted_rows"])
        zero_rows = int(record["structural_zero_filled_rows"])
        if view_id in FULL_STAGE_TYPE_SQUARE_VIEW_IDS:
            expected_stages = int(expected_stages_raw)
            _require(
                expected_stages == 5 and observed_stages == expected_stages,
                f"Five-stage full-grid coverage is incomplete for {view_id}",
            )
        else:
            _require(
                expected_stages_raw is None and observed_stages > 0,
                f"Native/complete-unit stage accounting is invalid for {view_id}",
            )
        _require(
            expected_rows >= 0
            and emitted_rows >= 0
            and zero_rows >= 0
            and emitted_rows + zero_rows == expected_rows,
            f"Structural-zero row accounting is inconsistent for {view_id}",
        )
        _require(
            isinstance(record["full_stage_type_square_required"], bool),
            f"Full-grid requirement is not explicitly boolean for {view_id}",
        )
        if view_id in FULL_STAGE_TYPE_SQUARE_VIEW_IDS:
            _require(
                record["full_stage_type_square_required"] is True
                and record["verified_complete_evaluated_universe"] is True,
                f"Full stage/type-square universe is not verified for {view_id}",
            )
        elif view_id in NICHENET_VIEW_IDS:
            _require(
                record["full_stage_type_square_required"] is False
                and record["verified_complete_evaluated_universe"] is True,
                f"NicheNet complete-unit sender universe is not verified for {view_id}",
            )
            _require(
                int(record.get("completed_units", 0)) > 0
                and int(record.get("skipped_or_ineligible_units", -1)) >= 0,
                f"NicheNet completed/skipped unit audit is missing for {view_id}",
            )
        else:
            _require(
                record["full_stage_type_square_required"] is False
                and record["verified_complete_evaluated_universe"] is False
                and zero_rows == 0,
                f"Native CytoBridge rows must not masquerade as an independently verified full grid: {view_id}",
            )
        _require(
            record["unevaluated_units_zero_filled"] is False,
            f"Unevaluated units were zero-filled for {view_id}",
        )
        _require(
            record["method_unavailable_lr_rows_zero_filled"] is False,
            f"Method-unavailable LR rows were zero-filled for {view_id}",
        )

    path = directory / "structural_zero_audit.csv"
    artifacts = _manifest_artifacts_by_name(manifest)
    _verify_artifact(path, artifacts.get(path.name))
    audit = pd.read_csv(path)
    required_columns = {
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
    }
    missing_columns = sorted(required_columns - set(audit.columns))
    _require(
        not missing_columns,
        f"Structural-zero audit lacks columns: {missing_columns}",
    )
    _require(
        set(audit["view_id"].astype(str)) == set(EXPECTED_VIEW_IDS),
        "Structural-zero audit must cover exactly the eight score views",
    )
    numeric_columns = (
        "expected_directed_rows",
        "native_emitted_rows",
        "native_emitted_positive_rows",
        "native_emitted_zero_rows",
        "structural_zero_filled_rows",
    )
    numeric = {
        column: pd.to_numeric(audit[column], errors="raise").astype(int)
        for column in numeric_columns
    }
    _require(
        (
            numeric["native_emitted_rows"] + numeric["structural_zero_filled_rows"]
            == numeric["expected_directed_rows"]
        ).all(),
        "Structural-zero audit has inconsistent expected-row accounting",
    )
    _require(
        (
            numeric["native_emitted_positive_rows"]
            + numeric["native_emitted_zero_rows"]
            == numeric["native_emitted_rows"]
        ).all(),
        "Structural-zero audit has inconsistent native positive/zero accounting",
    )
    verified = audit["verified_complete_evaluated_universe"].astype(str).str.casefold()
    external_mask = audit["view_id"].astype(str).isin(EXTERNAL_VIEW_IDS)
    internal_mask = audit["view_id"].astype(str).isin(INTERNAL_VIEW_IDS)
    _require(
        verified.loc[external_mask].eq("true").all()
        and verified.loc[internal_mask].eq("false").all(),
        "Structural-zero audit must verify external evaluated universes without relabelling native CytoBridge rows as a verified grid",
    )
    _require(
        _series_is_false(audit["unevaluated_units_zero_filled"]),
        "Structural-zero audit zero-filled an unevaluated unit",
    )
    _require(
        _series_is_false(audit["method_unavailable_lr_rows_zero_filled"]),
        "Structural-zero audit zero-filled a method-unavailable LR row",
    )
    for view_id, frame in audit.groupby("view_id", sort=False):
        record = completion[str(view_id)]
        _require(
            int(numeric["expected_directed_rows"].loc[frame.index].sum())
            == int(record["expected_rows"])
            and int(numeric["native_emitted_rows"].loc[frame.index].sum())
            == int(record["native_emitted_rows"])
            and int(numeric["structural_zero_filled_rows"].loc[frame.index].sum())
            == int(record["structural_zero_filled_rows"]),
            f"Structural-zero table totals disagree with the manifest for {view_id}",
        )
    return audit


def _validate_positive_support_top_k(
    directory: Path, manifest: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fail closed unless top-k is positive-support and boundary-tie aware."""

    artifacts = _manifest_artifacts_by_name(manifest)
    by_stage_path = directory / "pairwise_consistency_by_stage.csv"
    summary_path = directory / "pairwise_consistency_summary.csv"
    _verify_artifact(by_stage_path, artifacts.get(by_stage_path.name))
    _verify_artifact(summary_path, artifacts.get(summary_path.name))
    by_stage = pd.read_csv(by_stage_path)
    summary = pd.read_csv(summary_path)
    required_by_stage = {
        "view_id_left",
        "view_id_right",
        "stage",
        "n_positive_left",
        "n_positive_right",
        "effective_top_k",
        "top_k_left_realized_set_size",
        "top_k_right_realized_set_size",
        "top_k_left_boundary_score",
        "top_k_right_boundary_score",
        "top_k_left_boundary_tie_count",
        "top_k_right_boundary_tie_count",
        "top_k_left_boundary_tie_expanded",
        "top_k_right_boundary_tie_expanded",
        "top_k_selection_rule",
        "top_k_jaccard",
    }
    missing = sorted(required_by_stage - set(by_stage.columns))
    _require(not missing, f"Pairwise top-k audit lacks columns: {missing}")
    _require(not by_stage.empty, "Pairwise top-k audit is empty")
    selection_rules = by_stage["top_k_selection_rule"].astype(str).str.strip()
    _require(
        selection_rules.ne("").all()
        and selection_rules.str.casefold().str.contains("positive").all()
        and selection_rules.str.casefold().str.contains("tie").all(),
        "Top-k selection rule must explicitly be positive-support and boundary-tie aware",
    )
    integer_columns = (
        "n_positive_left",
        "n_positive_right",
        "effective_top_k",
        "top_k_left_realized_set_size",
        "top_k_right_realized_set_size",
        "top_k_left_boundary_tie_count",
        "top_k_right_boundary_tie_count",
    )
    values = {
        column: pd.to_numeric(by_stage[column], errors="raise").astype(int)
        for column in integer_columns
    }
    _require(
        all((series >= 0).all() for series in values.values()),
        "Pairwise positive-support/tie counts must be non-negative",
    )
    _require(
        (values["top_k_left_realized_set_size"] <= values["n_positive_left"]).all()
        and (
            values["top_k_right_realized_set_size"] <= values["n_positive_right"]
        ).all(),
        "A realized top set exceeds its positive-support universe",
    )
    for side in ("left", "right"):
        expanded = (
            by_stage[f"top_k_{side}_boundary_tie_expanded"].astype(str).str.casefold()
        )
        _require(
            expanded.isin({"true", "false"}).all(),
            f"Top-k {side} tie-expansion flag is not boolean",
        )
        realized = values[f"top_k_{side}_realized_set_size"]
        effective = values["effective_top_k"]
        _require(
            (expanded.eq("true") == (realized > effective)).all(),
            f"Top-k {side} tie-expansion flag disagrees with realized set size",
        )
    zero_support = (values["n_positive_left"] == 0) | (values["n_positive_right"] == 0)
    jaccard = pd.to_numeric(by_stage["top_k_jaccard"], errors="coerce")
    _require(
        jaccard.loc[zero_support].isna().all(),
        "Top-k Jaccard must be NA when either view has zero positive support",
    )
    required_summary = {
        "view_id_left",
        "view_id_right",
        "n_stages_compared",
        "n_finite_spearman_stages",
        "n_top_k_informative_stages",
        "mean_stage_spearman",
        "mean_stage_top_k_jaccard_informative_only",
    }
    missing_summary = sorted(required_summary - set(summary.columns))
    _require(
        not missing_summary,
        f"Pairwise summary lacks positive-support readiness fields: {missing_summary}",
    )
    n_compared = pd.to_numeric(summary["n_stages_compared"], errors="raise").astype(int)
    n_finite = pd.to_numeric(
        summary["n_finite_spearman_stages"], errors="raise"
    ).astype(int)
    _require(
        ((n_finite >= 0) & (n_finite <= n_compared)).all(),
        "Finite-Spearman stage counts are inconsistent",
    )
    return by_stage, summary


def _validate_validation_axes(directory: Path) -> dict[str, Any]:
    manifest = _read_json(directory / "run_manifest.json")
    _require(
        manifest.get("method") == "zebrafish_reviewer_ccc_internal_consistency_axes",
        "Unexpected reviewer-axis validation workflow",
    )
    claims = manifest.get("claims")
    _require(isinstance(claims, Mapping), "Reviewer-axis manifest lacks claims")
    required_false_claims = (
        "attention_is_ccc_probability",
        "exact_message_is_biochemical_flux",
        "lr_database_agreement_is_independent_validation",
        "virtual_ablation_is_causal_perturbation",
        "known_axis_literature_claim",
    )
    for claim in required_false_claims:
        _require(
            claims.get(claim) is False, f"Reviewer-axis claim must be false: {claim}"
        )
    return manifest


def _orthology_tier(
    *, policy: str, tier: str, primary_claim_allowed: Any, method: str
) -> str:
    contracts = {
        "strict_confidence1": ("primary", True),
        "one2one_bijective_all_confidence": ("sensitivity", False),
    }
    _require(
        policy in contracts, f"{method} uses unsupported orthology policy {policy!r}"
    )
    expected_tier, expected_primary = contracts[policy]
    _require(tier == expected_tier, f"{method} orthology tier conflicts with policy")
    _require(
        primary_claim_allowed is expected_primary,
        f"{method} primary_claim_allowed conflicts with policy",
    )
    return (
        "primary (strict confidence=1 one-to-one orthology)"
        if expected_primary
        else "sensitivity only (all-confidence one-to-one orthology; no primary claim)"
    )


def _validate_methods(
    paths: Mapping[str, Path]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifests: dict[str, Any] = {}

    commot_path = paths["commot"] / "manifest.json"
    commot = _read_json(commot_path)
    _require(commot.get("method") == "COMMOT", "Unexpected COMMOT manifest")
    _require(
        commot.get("database_variant") == "current_zebrafish_lr_database",
        "COMMOT must use the current zebrafish LR database",
    )
    _require(
        commot.get("score_semantics", {}).get("raw_cross_method_units_comparable")
        is False,
        "COMMOT manifest must reject raw cross-method score comparability",
    )
    manifests["commot"] = {"path": commot_path, "content": commot}

    cellchat_path = paths["cellchat"] / "manifest.json"
    cellchat = _read_json(cellchat_path)
    _require(cellchat.get("method") == "CellChat", "Unexpected CellChat manifest")
    _require(
        cellchat.get("database_variant") == "current_zebrafish_lr_database",
        "CellChat must use the current zebrafish LR database",
    )
    validation = cellchat.get("database_validation", {})
    _require(
        isinstance(validation, Mapping)
        and validation.get("excluded_rows_are_method_unavailable_not_biological_zero")
        is True,
        "CellChat exclusions must be labelled method-unavailable, not biological zero",
    )
    _require(
        "never zero-filled"
        in str(cellchat.get("design", {}).get("method_unavailable_policy", "")),
        "CellChat method-unavailable rows must never be zero-filled",
    )
    requested = int(validation.get("rows_requested", -1))
    eligible = int(validation.get("rows_eligible", -1))
    excluded = int(validation.get("rows_excluded", -1))
    _require(
        min(requested, eligible, excluded) >= 0 and requested == eligible + excluded,
        "CellChat requested/eligible/method-unavailable counts are inconsistent",
    )
    manifests["cellchat"] = {"path": cellchat_path, "content": cellchat}

    nichenet_records: list[tuple[str, str, Path]] = [
        (
            "nichenet_default",
            "default",
            paths["nichenet_default"] / "run_manifest.json",
        ),
        ("nichenet_custom", "custom", paths["nichenet_custom"] / "run_manifest.json"),
    ]
    nichenet_tiers: list[str] = []
    nichenet_policies: list[str] = []
    for key, mode, path in nichenet_records:
        manifest = _read_json(path)
        _require(
            manifest.get("workflow") == "reviewer_zebrafish_cross_species_nichenet_v2"
            and manifest.get("status") == "complete"
            and manifest.get("mode") == mode,
            f"Unexpected or incomplete NicheNet {mode} manifest",
        )
        activity_semantics = str(manifest.get("activity_semantics", "")).casefold()
        _require(
            "sender-specific" in activity_semantics
            and (
                "do not make" in activity_semantics
                or "not sender-specific" in activity_semantics
            ),
            f"NicheNet {mode} manifest lost its sender-specificity caveat",
        )
        policy = str(manifest.get("orthology_policy", ""))
        tier = str(manifest.get("analysis_tier", ""))
        tier_label = _orthology_tier(
            policy=policy,
            tier=tier,
            primary_claim_allowed=manifest.get("primary_claim_allowed"),
            method=f"NicheNet {mode}",
        )
        nichenet_tiers.append(tier_label)
        nichenet_policies.append(policy)
        manifests[key] = {"path": path, "content": manifest, "tier_label": tier_label}
    _require(
        len(set(nichenet_policies)) == 1 and len(set(nichenet_tiers)) == 1,
        "The two NicheNet conditions must share one orthology policy/tier",
    )

    cellagent_dir = paths["cellagentchat"]
    dual_path = cellagent_dir / "manifest.json"
    dual = _read_json(dual_path)
    _require(
        dual.get("workflow") == "official_cellagentchat_spatial_dual_lr_database",
        "Unexpected CellAgentChat dual-run manifest",
    )
    _require(
        set(dual.get("conditions", []))
        == {CELLAGENTCHAT_OFFICIAL, CELLAGENTCHAT_CUSTOM},
        "CellAgentChat dual run must contain the official and project-LR conditions",
    )
    _require(
        dual.get("same_mapped_expression_and_sample_plan_verified") is True
        and dual.get("same_preparation_manifest_and_orthology_claims_verified") is True,
        "CellAgentChat conditions must share mapped expression, sampling, and orthology claims",
    )
    manifests["cellagentchat_dual"] = {"path": dual_path, "content": dual}

    cellagent_tiers: list[str] = []
    cellagent_policies: list[str] = []
    recorded_condition_manifests = dual.get("condition_manifests")
    _require(
        isinstance(recorded_condition_manifests, Mapping),
        "CellAgentChat dual manifest lacks condition manifest records",
    )
    child_claims: list[Mapping[str, Any]] = []
    for key, condition in (
        ("cellagentchat_official", CELLAGENTCHAT_OFFICIAL),
        ("cellagentchat_custom", CELLAGENTCHAT_CUSTOM),
    ):
        path = cellagent_dir / condition / "manifest.json"
        record = recorded_condition_manifests.get(condition)
        _require(
            isinstance(record, Mapping),
            f"CellAgentChat dual manifest lacks the {condition} artifact record",
        )
        _verify_artifact(path, record)
        manifest = _read_json(path)
        _require(
            manifest.get("method") == "official_cellagentchat_v0_2_0_spatial"
            and manifest.get("database_condition") == condition,
            f"Unexpected CellAgentChat manifest for {condition}",
        )
        native_primary = str(manifest.get("design", {}).get("native_primary", ""))
        _require(
            (
                "sum of Bonferroni-significant" in native_primary
                and "interaction scores" in native_primary
            )
            or "number of Bonferroni-significant LR pairs" in native_primary,
            "CellAgentChat native score semantics are missing",
        )
        claims = manifest.get("shared_input", {}).get("preparation_claims", {})
        _require(isinstance(claims, Mapping), "CellAgentChat lacks preparation claims")
        child_claims.append(claims)
        policy = str(claims.get("orthology_policy", ""))
        tier = str(claims.get("orthology_analysis_tier", ""))
        tier_label = _orthology_tier(
            policy=policy,
            tier=tier,
            primary_claim_allowed=claims.get("primary_claim_allowed"),
            method=f"CellAgentChat {condition}",
        )
        cellagent_tiers.append(tier_label)
        cellagent_policies.append(policy)
        manifests[key] = {"path": path, "content": manifest, "tier_label": tier_label}
    _require(
        len(set(cellagent_policies)) == 1 and len(set(cellagent_tiers)) == 1,
        "The two CellAgentChat conditions must share one orthology policy/tier",
    )
    _require(
        dual.get("preparation_claims") in (None, child_claims[0])
        and child_claims[0] == child_claims[1],
        "CellAgentChat dual/condition preparation claims disagree",
    )

    conditions = [
        {
            "view_id": "commot__project_lr",
            "condition": "COMMOT — project zebrafish LR",
            "database": "current project zebrafish LR database",
            "score": "mean native OT communication mass per possible sender/receiver cell pair",
            "space": "spatial",
            "tier": "direct zebrafish/project-LR run (not orthology-tiered)",
        },
        {
            "view_id": "cellchat__project_lr",
            "condition": "CellChat — project zebrafish LR",
            "database": "current project zebrafish LR database",
            "score": "sum of unthresholded type-level probabilities; population.size=false",
            "space": "non-spatial in this run (coordinates audited but not used by CellChat)",
            "tier": "direct zebrafish/project-LR run (not orthology-tiered)",
        },
        {
            "view_id": "nichenet_v2__official_mouse_lr",
            "condition": "NicheNet-v2 — official/default",
            "database": "official mouse LR candidates and mouse ligand-target prior",
            "score": "derived sum of positive sender-associated aupr_corrected activity",
            "space": "non-spatial receiver-transition ligand activity",
            "tier": nichenet_tiers[0],
        },
        {
            "view_id": "nichenet_v2__project_lr_gate",
            "condition": "NicheNet-v2 — project LR gate",
            "database": "project zebrafish LR projected to mouse as candidate gate; mouse ligand-target prior unchanged",
            "score": "derived sum of positive sender-associated aupr_corrected activity",
            "space": "non-spatial receiver-transition ligand activity",
            "tier": nichenet_tiers[0],
        },
        {
            "view_id": "cellagentchat__official_mouse_default",
            "condition": "CellAgentChat — official/default",
            "database": "official mouse CellTalkDB default",
            "score": "mean CTPS: sum of Bonferroni-significant interaction scores across sampling seeds",
            "space": "spatial CellAgentChat",
            "tier": cellagent_tiers[0],
        },
        {
            "view_id": "cellagentchat__project_lr",
            "condition": "CellAgentChat — project LR",
            "database": "project zebrafish LR projected to supported mouse singleton pairs",
            "score": "mean CTPS: sum of Bonferroni-significant interaction scores across sampling seeds",
            "space": "spatial CellAgentChat",
            "tier": cellagent_tiers[0],
        },
    ]
    return manifests, conditions


def _crosscheck_orthology_records(
    comparison: Mapping[str, Any], manifests: Mapping[str, Any]
) -> None:
    checks = (
        (
            "nichenet_orthology_conditions",
            {
                "nichenet_v2__official_mouse_lr": manifests["nichenet_default"],
                "nichenet_v2__project_lr_gate": manifests["nichenet_custom"],
            },
            lambda entry: entry["content"],
        ),
        (
            "cellagentchat_orthology_conditions",
            {
                "cellagentchat__official_mouse_default": manifests[
                    "cellagentchat_official"
                ],
                "cellagentchat__project_lr": manifests["cellagentchat_custom"],
            },
            lambda entry: entry["content"]["shared_input"]["preparation_claims"],
        ),
    )
    for field, expected, source in checks:
        records = comparison.get(field)
        _require(isinstance(records, Mapping), f"Comparison lacks {field}")
        for view_id, entry in expected.items():
            record = records.get(view_id)
            _require(isinstance(record, Mapping), f"Comparison lacks {field}.{view_id}")
            source_claims = source(entry)
            source_policy = str(source_claims.get("orthology_policy", ""))
            source_tier = str(
                source_claims.get(
                    "analysis_tier", source_claims.get("orthology_analysis_tier", "")
                )
            )
            _require(
                str(record.get("orthology_policy")) == source_policy
                and str(record.get("analysis_tier")) == source_tier
                and record.get("primary_claim_allowed")
                is source_claims.get("primary_claim_allowed"),
                f"Comparison orthology claims disagree with source manifest for {view_id}",
            )


def _copy_verified(
    *,
    source_dir: Path,
    filename: str,
    records: Mapping[str, Mapping[str, Any]],
    destination_dir: Path,
) -> Path:
    source = source_dir / filename
    _verify_artifact(source, records.get(filename))
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    shutil.copy2(source, destination)
    return destination


def _copy_plain(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _format_number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    if number == 0:
        return "0"
    if abs(number) < 10 ** (-(digits + 1)):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(_markdown_escape(value) for value in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    return lines


def _pairwise_row(summary: pd.DataFrame, left: str, right: str) -> pd.Series | None:
    selected = summary.loc[
        ((summary["view_id_left"] == left) & (summary["view_id_right"] == right))
        | ((summary["view_id_left"] == right) & (summary["view_id_right"] == left))
    ]
    if len(selected) != 1:
        return None
    return selected.iloc[0]


def _coverage_rows(coverage: pd.DataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for view_id in EXPECTED_VIEW_IDS:
        frame = coverage.loc[coverage["view_id"] == view_id]
        complete = frame.loc[frame["status"] == "complete"]
        if complete.empty:
            rows.append([view_id, 0, 0, "NA", "missing"])
            continue
        counts = pd.to_numeric(complete["n_directed_pairs"], errors="raise")
        rows.append(
            [
                view_id,
                int(len(complete)),
                int(counts.sum()),
                _format_number(counts.median(), digits=1),
                "complete",
            ]
        )
    return rows


def _structural_completion_rows(comparison: Mapping[str, Any]) -> list[list[Any]]:
    records = comparison["score_view_zero_completion"]
    rows: list[list[Any]] = []
    for view_id in EXPECTED_VIEW_IDS:
        record = records[view_id]
        expected = int(record["expected_rows"])
        filled = int(record["structural_zero_filled_rows"])
        expected_stages = record["expected_stage_count"]
        rows.append(
            [
                view_id,
                record["universe_scope"],
                (
                    f"{int(record['observed_stage_count'])}/"
                    f"{int(expected_stages) if expected_stages is not None else 'NA'}"
                ),
                expected,
                int(record["native_emitted_rows"]),
                filled,
                _format_number(filled / expected if expected else float("nan")),
                (
                    f"completed={int(record['completed_units'])}; "
                    f"skipped/ineligible={int(record['skipped_or_ineligible_units'])}"
                    if "completed_units" in record
                    else "not unit-scoped"
                ),
                str(bool(record["verified_complete_evaluated_universe"])).lower(),
                str(bool(record["unevaluated_units_zero_filled"])).lower(),
            ]
        )
    return rows


def _top_k_audit_rows(by_stage: pd.DataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for internal in INTERNAL_VIEW_IDS:
        for external in EXTERNAL_VIEW_IDS:
            frame = by_stage.loc[
                (
                    (by_stage["view_id_left"] == internal)
                    & (by_stage["view_id_right"] == external)
                )
                | (
                    (by_stage["view_id_left"] == external)
                    & (by_stage["view_id_right"] == internal)
                )
            ].copy()
            if frame.empty:
                rows.append([internal, external, 0, "NA", "NA", 0, 0, "missing"])
                continue
            internal_is_left = frame["view_id_left"].eq(internal)
            internal_positive = pd.Series(
                data=frame["n_positive_right"].to_numpy(), index=frame.index
            )
            external_positive = pd.Series(
                data=frame["n_positive_left"].to_numpy(), index=frame.index
            )
            internal_positive.loc[internal_is_left] = frame.loc[
                internal_is_left, "n_positive_left"
            ]
            external_positive.loc[internal_is_left] = frame.loc[
                internal_is_left, "n_positive_right"
            ]
            zero_support = (pd.to_numeric(internal_positive) == 0) | (
                pd.to_numeric(external_positive) == 0
            )
            left_expanded = (
                frame["top_k_left_boundary_tie_expanded"]
                .astype(str)
                .str.casefold()
                .eq("true")
            )
            right_expanded = (
                frame["top_k_right_boundary_tie_expanded"]
                .astype(str)
                .str.casefold()
                .eq("true")
            )
            selection_rules = frame["top_k_selection_rule"].astype(str).unique()
            rows.append(
                [
                    internal,
                    external,
                    len(frame),
                    _format_number(pd.to_numeric(internal_positive).median(), digits=1),
                    _format_number(pd.to_numeric(external_positive).median(), digits=1),
                    int(zero_support.sum()),
                    int((left_expanded | right_expanded).sum()),
                    "; ".join(selection_rules),
                ]
            )
    return rows


def _manifest_count_summary(manifests: Mapping[str, Any]) -> list[list[Any]]:
    cellchat_validation = manifests["cellchat"]["content"]["database_validation"]
    rows: list[list[Any]] = [
        [
            "COMMOT — project LR",
            "complete",
            "all prepared observed stages; see stage_diagnostics",
        ],
        [
            "CellChat — project LR",
            "complete",
            (
                f"{cellchat_validation.get('rows_eligible', 'NA')} executable / "
                f"{cellchat_validation.get('rows_requested', 'NA')} requested LR rows; "
                f"{cellchat_validation.get('rows_excluded', 'NA')} method-unavailable"
            ),
        ],
    ]
    for key, label in (
        ("nichenet_default", "NicheNet-v2 — official/default"),
        ("nichenet_custom", "NicheNet-v2 — project LR gate"),
    ):
        counts = manifests[key]["content"].get("counts", {})
        rows.append(
            [
                label,
                manifests[key]["content"].get("status", "NA"),
                (
                    f"{counts.get('units_complete', 'NA')} receiver-transition units; "
                    f"{counts.get('sender_ligand_activity_rows', 'NA')} sender-activity rows"
                ),
            ]
        )
    for key, label in (
        ("cellagentchat_official", "CellAgentChat — official/default"),
        ("cellagentchat_custom", "CellAgentChat — project LR"),
    ):
        counts = manifests[key]["content"].get("counts", {})
        rows.append(
            [
                label,
                "complete",
                (
                    f"{counts.get('n_runs', 'NA')} sampled runs; "
                    f"{counts.get('significant_lr_rows', 'NA')} significant LR rows"
                ),
            ]
        )
    return rows


def _build_readme(
    *,
    output: Path,
    comparison: Mapping[str, Any],
    validation: Mapping[str, Any],
    manifests: Mapping[str, Any],
    conditions: Sequence[Mapping[str, str]],
) -> Path:
    tables = output / "tables"
    pairwise = pd.read_csv(tables / "pairwise_consistency_summary.csv")
    pairwise_by_stage = pd.read_csv(tables / "pairwise_consistency_by_stage.csv")
    coverage = pd.read_csv(tables / "condition_coverage.csv")
    controls = pd.read_csv(tables / "cytobridge_control_metrics.csv")
    context = pd.read_csv(tables / "context_enrichment_tests.csv")
    matched = pd.read_csv(tables / "degree_matched_conditional_tests.csv")
    unavailable = pd.read_csv(tables / "method_unavailable_lr_rows.csv")
    ablation_path = tables / "virtual_ablation_summary.csv"
    ablation = pd.read_csv(ablation_path) if ablation_path.is_file() else None

    lines = [
        "# Zebrafish CCC reviewer bundle",
        "",
        "Status: **formal comparison complete; reporting bundle validated**.",
        "",
        (
            "All primary score artifacts were re-hashed and matched their source "
            "manifest records before this report was accepted."
        ),
        "",
        "## Bottom line",
        "",
        (
            "This bundle tests whether two CytoBridge model-internal readouts agree with "
            "six separately labelled external method/database conditions. It does not "
            "treat agreement as ground truth. CytoBridge attention is an internal gate "
            "magnitude, **not a calibrated cell-cell communication (CCC) probability**; "
            "the exact message is a model contribution, not biochemical flux. Raw scores "
            "are never compared across methods because their units differ."
        ),
        "",
        (
            "`formal_reviewer_ready=true` means the declared files, conditions, joins, "
            "and provenance passed the reporting contract. It does not promote a "
            "cross-species sensitivity analysis into primary biological evidence."
        ),
        "",
        "## Exact benchmark design: six external conditions",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "condition",
                "database/prior",
                "score used",
                "spatial scope",
                "evidence tier",
            ],
            [
                [
                    item["condition"],
                    item["database"],
                    item["score"],
                    item["space"],
                    item["tier"],
                ]
                for item in conditions
            ],
        )
    )
    lines.extend(
        [
            "",
            (
                "The two NicheNet conditions use the same official mouse ligand-target/"
                "signaling/GRN prior. The project-LR condition changes only the candidate "
                "LR gate. NicheNet activity is native to a receiver transition and ligand; "
                "the sender/type-pair score in this comparison is derived, non-spatial, "
                "not receptor-specific, and not a biochemical communication strength."
            ),
            "",
            (
                "Both CellAgentChat conditions use the same zebrafish-to-mouse projected "
                "expression and sample plan. They differ only in the official mouse "
                "CellTalkDB versus project-LR-projected database. CellAgentChat remains a "
                "cross-species mouse-prior analysis even in the project-LR condition."
            ),
            "",
            "## CytoBridge readouts (not two additional external methods)",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["view", "meaning", "forbidden interpretation"],
            [
                [
                    "CytoBridge trained attention",
                    "mean magnitude of the trained spatial-GNN attention gate",
                    "CCC probability or communication rate",
                ],
                [
                    "CytoBridge trained exact message",
                    "exact one-layer joint message contribution reconstructed from the model",
                    "biochemical ligand-receptor flux",
                ],
            ],
        )
    )
    lines.extend(["", "## Completion and coverage", ""])
    lines.extend(
        _markdown_table(
            ["condition", "status", "method-native count audit"],
            _manifest_count_summary(manifests),
        )
    )
    lines.extend(["", "Emitted directed-key coverage:", ""])
    lines.extend(
        _markdown_table(
            [
                "view_id",
                "stages emitted",
                "directed-key rows",
                "median keys/stage",
                "status",
            ],
            _coverage_rows(coverage),
        )
    )
    lines.extend(
        [
            "",
            (
                "Positive-only runner exports are completed with explicit structural zeros "
                "only inside a manifest-verified evaluated universe. Skipped or ineligible "
                "receiver units and method-unavailable LR rows remain outside that universe "
                "and are never zero-filled. Every pairwise statistic uses the exact shared "
                "`stage, sender_type, receiver_type` universe after this audited completion."
            ),
            "",
            "Structural-zero provenance:",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "view_id",
                "evaluated universe",
                "stages observed/expected",
                "expected rows",
                "native rows",
                "structural zeros added",
                "fill fraction",
                "receiver-unit audit",
                "external universe verified",
                "unevaluated units zero-filled",
            ],
            _structural_completion_rows(comparison),
        )
    )
    lines.extend(
        [
            "",
            (
                "The two CytoBridge entries report their native type-pair summary and add "
                "no rows; they are deliberately not relabelled as an independently verified "
                "full grid. COMMOT, CellChat, and both CellAgentChat conditions require a "
                "complete evaluated stage/type grid. NicheNet completion is restricted to "
                "the sender grid of completed receiver-transition units; skipped/ineligible "
                "units are not filled."
            ),
            "",
            "## CytoBridge versus each external condition",
            "",
            (
                "Spearman correlations use the complete shared evaluated grid within each "
                "stage and are then averaged over finite stages. Top-k uses positive support "
                "with boundary-tie expansion; `NA` means the statistic was undefined, not "
                "zero agreement."
            ),
            "",
        ]
    )
    comparison_rows: list[list[Any]] = []
    for internal in INTERNAL_VIEW_IDS:
        for external in EXTERNAL_VIEW_IDS:
            row = _pairwise_row(pairwise, internal, external)
            comparison_rows.append(
                [
                    internal,
                    external,
                    _format_number(row["mean_stage_spearman"])
                    if row is not None
                    else "NA",
                    int(row["n_stages_compared"]) if row is not None else 0,
                    int(row["n_finite_spearman_stages"]) if row is not None else 0,
                    int(row["n_shared_directed_pairs_total"]) if row is not None else 0,
                    int(row["n_top_k_informative_stages"]) if row is not None else 0,
                    (
                        _format_number(row["mean_stage_top_k_jaccard_informative_only"])
                        if row is not None
                        else "NA"
                    ),
                ]
            )
    lines.extend(
        _markdown_table(
            [
                "CytoBridge view",
                "external condition",
                "mean stage rho",
                "stages compared",
                "finite-rho stages",
                "shared keys total",
                "informative top-k stages",
                "mean informative Jaccard",
            ],
            comparison_rows,
        )
    )
    lines.extend(
        [
            "",
            "Positive-support and boundary-tie audit for these comparisons:",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "CytoBridge view",
                "external condition",
                "stage rows",
                "median CytoBridge positive keys",
                "median external positive keys",
                "zero-support stages",
                "boundary-tie-expanded stages",
                "selection rule",
            ],
            _top_k_audit_rows(pairwise_by_stage),
        )
    )
    lines.extend(
        [
            "",
            (
                "Top-k is selected from strictly positive support. All keys tied at the "
                "positive kth boundary are retained, so realized set sizes can exceed the "
                "requested/effective k. Jaccard is `NA`, not zero, if either side has no "
                "positive support. Spearman may likewise be undefined for a constant stage; "
                "the finite-rho stage count is reported explicitly."
            ),
        ]
    )

    lines.extend(["", "## Model-internal controls", ""])
    if controls.empty:
        lines.append("No control rows were emitted.")
    else:
        lines.extend(
            _markdown_table(
                ["control", "target", "metric", "estimate", "p", "n"],
                [
                    [
                        row.control_label,
                        row.target,
                        row.metric,
                        _format_number(row.estimate),
                        _format_number(row.p_value),
                        int(row.n_observations),
                    ]
                    for row in controls.itertuples(index=False)
                ],
            )
        )

    lines.extend(
        [
            "",
            "## LR association and spatial/degree-matched reviewer checks",
            "",
            (
                "The frozen CytoBridge edge classifier is LR-informed, so these LR "
                "associations are partly circular and are internal consistency checks, "
                "not independent validation. Forward and reverse LR orientations must be "
                "read together. Edge rows share cells and are not independent biological replicates."
            ),
            "",
            "Context-level stage-stratified tests:",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["target", "LR orientation", "high-low", "within-stage rho", "p", "BH q"],
            [
                [
                    row.target,
                    row.score,
                    _format_number(row.high_minus_low_score),
                    _format_number(row.within_stage_rank_correlation),
                    _format_number(row.empirical_p_greater),
                    _format_number(row.bh_q_within_context_family),
                ]
                for row in context.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            "Conditional tests matched on time, cell types, distance, non-LR state, source out-degree, and target in-degree:",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "target residual",
                "LR orientation",
                "conditional rho",
                "rho-null",
                "p",
                "BH q",
                "retained edges",
            ],
            [
                [
                    row.target,
                    row.score,
                    _format_number(row.conditional_rank_correlation),
                    _format_number(row.observed_minus_null_mean),
                    _format_number(row.empirical_p_greater),
                    _format_number(row.bh_q_within_degree_matched_family),
                    int(row.n_edges_retained),
                ]
                for row in matched.itertuples(index=False)
            ],
        )
    )

    lines.extend(["", "## CellChat method-unavailable rows", ""])
    if unavailable.empty:
        lines.append("No requested LR row was method-unavailable in this run.")
    else:
        lines.append(
            f"{len(unavailable)} requested LR rows were not representable by the pinned CellChat implementation. They were excluded from its comparison universe and **never zero-filled**."
        )
        lines.append("")
        display_columns = [
            column
            for column in (
                "database_row",
                "interaction_id",
                "ligand",
                "receptor",
                "reason",
            )
            if column in unavailable.columns
        ]
        lines.extend(
            _markdown_table(
                display_columns,
                unavailable.loc[:, display_columns]
                .head(20)
                .astype(str)
                .values.tolist(),
            )
        )

    lines.extend(["", "## Database-identifiable axes and virtual removal", ""])
    lines.append(
        "The LR-axis tables use labels present in the supplied database. Developmental literature can support pathway or gene relevance, but does not validate the exact LR pair, sender-to-receiver direction, or CytoBridge rank."
    )
    if ablation is None:
        lines.append("")
        lines.append("No compatible virtual-removal summary was included.")
    else:
        lines.extend(
            [
                "",
                (
                    "Virtual removal is a one-model/one-seed sensitivity analysis. It "
                    "changes the initial cohort and particle count and is **not a causal "
                    "genetic perturbation estimate**. Endpoint-minus-t0 should be read "
                    "alongside the immediate t0 shift."
                ),
                "",
            ]
        )
        selected_columns = [
            column
            for column in (
                "variant",
                "space",
                "endpoint_normalized_centroid_shift",
                "endpoint_normalized_shift_minus_t0",
                "endpoint_composition_total_variation",
            )
            if column in ablation.columns
        ]
        lines.extend(
            _markdown_table(
                selected_columns,
                [
                    [
                        _format_number(value)
                        if isinstance(value, (float, int))
                        else value
                        for value in row
                    ]
                    for row in ablation.loc[:, selected_columns].values.tolist()
                ],
            )
        )

    unavailable_info = comparison.get("cellchat_method_unavailable_lr_rows", {})
    lines.extend(
        [
            "",
            "## What can and cannot be claimed",
            "",
            "Supported reporting:",
            "",
            "- Reproducible rank concordance, informative top-edge overlap, reciprocal-direction agreement, temporal stability, and coverage across the declared score views.",
            "- Model-internal LR, confounder, directionality, identifiable-axis, and virtual-removal sensitivity checks with their stated nulls and limitations.",
            "- Separate official/default versus project-LR results for NicheNet and CellAgentChat.",
            "",
            "Not supported by this bundle:",
            "",
            "- Calling attention a CCC probability, or the exact message a biochemical flux.",
            "- Treating agreement between methods as experimental truth or treating missing/method-unavailable keys as biological zeroes.",
            "- Calling NicheNet's derived type-pair value a native spatial sender-receptor score.",
            "- Calling mouse-prior NicheNet or CellAgentChat native zebrafish methods, or promoting all-confidence orthology sensitivity results to primary evidence.",
            "- Calling one-seed virtual removal a causal perturbation experiment.",
            "",
            "## Bundle layout",
            "",
            "- `figures/`: comparison and reviewer-validation panels in PNG and PDF.",
            "- `tables/`: exact score, coverage, concordance, control, axis, and ablation audit tables.",
            "- `manifests/`: renamed copies of every source manifest used to validate the six conditions.",
            "- `notes/`: the upstream comparison and validation interpretation notes.",
            "- `bundle_manifest.json`: SHA256 inventory and claim guardrails for this bundle.",
            "",
            (
                f"CellChat unavailable-row audit in the source manifest: count="
                f"{unavailable_info.get('count', len(unavailable))}, zero_filled="
                f"{unavailable_info.get('zero_filled', False)}."
            ),
            "",
            "![Cross-method rank concordance](figures/rank_concordance.png)",
            "",
            "![Reviewer validation axes](figures/reviewer_validation_axes.png)",
            "",
        ]
    )
    path = output / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _build_readme_cn(
    *,
    output: Path,
    comparison: Mapping[str, Any],
    conditions: Sequence[Mapping[str, str]],
) -> Path:
    """Write a concise, CSV-backed Chinese handoff for lab presentation."""

    tables = output / "tables"
    pairwise = pd.read_csv(tables / "pairwise_consistency_summary.csv")
    matched = pd.read_csv(tables / "degree_matched_conditional_tests.csv")
    unavailable = pd.read_csv(tables / "method_unavailable_lr_rows.csv")
    lines = [
        "# 斑马鱼 CCC reviewer bundle（中文速查）",
        "",
        "状态：**正式 comparison 与 provenance 检查通过**。这里的“正式”指文件、运行条件和统计契约完整，不代表所有跨物种结果都能作为 primary biological claim。",
        "",
        "所有 primary score artifacts 均已重新计算 hash，并与各自 source manifest 的记录一致。",
        "",
        "## 六个外部条件",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["条件", "数据库/先验", "比较分数", "空间属性", "证据层级"],
            [
                [
                    item["condition"],
                    item["database"],
                    item["score"],
                    item["space"],
                    item["tier"],
                ]
                for item in conditions
            ],
        )
    )
    lines.extend(
        [
            "",
            "另外有两个 CytoBridge 内部读出：trained attention gate magnitude 和 exact one-layer message contribution；它们不是两个外部方法。",
            "",
            "## CytoBridge 对六个条件的关键 rank 指标",
            "",
        ]
    )
    metric_rows: list[list[Any]] = []
    for internal in INTERNAL_VIEW_IDS:
        for external in EXTERNAL_VIEW_IDS:
            row = _pairwise_row(pairwise, internal, external)
            metric_rows.append(
                [
                    internal,
                    external,
                    _format_number(row["mean_stage_spearman"])
                    if row is not None
                    else "NA",
                    int(row["n_finite_spearman_stages"]) if row is not None else 0,
                    int(row["n_top_k_informative_stages"]) if row is not None else 0,
                    (
                        _format_number(row["mean_stage_top_k_jaccard_informative_only"])
                        if row is not None
                        else "NA"
                    ),
                ]
            )
    lines.extend(
        _markdown_table(
            [
                "CytoBridge 读出",
                "外部条件",
                "平均 stage Spearman rho",
                "有限 rho 的 stage 数",
                "有效 top-k stage 数",
                "平均有效 Jaccard",
            ],
            metric_rows,
        )
    )
    lines.extend(
        [
            "",
            "这些 rho 是每个 stage 内基于共同 evaluated universe 的 rank correlation；不同方法的 raw score 单位不一样，不能直接比数值大小。Top-k 只从正支持中选，并纳入 kth boundary 的全部 ties；任何一侧没有正支持时 Jaccard 记为 NA。",
            "",
            "## structural zero 审计",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "view_id",
                "evaluated universe",
                "expected rows",
                "native rows",
                "补入的 structural zeros",
                "补零比例",
                "receiver-unit 审计",
                "unevaluated units 是否补零",
            ],
            [
                [row[0], row[1], row[3], row[4], row[5], row[6], row[7], row[9]]
                for row in _structural_completion_rows(comparison)
            ],
        )
    )
    lines.extend(
        [
            "",
            "这里只对 manifest 验证过的 evaluated universe 补 structural zero。NicheNet 被跳过/不合格的 receiver-transition units 不补零；CellChat 无法表示的 LR rows 也不补零。",
            "",
            f"本次 CellChat method-unavailable LR rows：**{len(unavailable)}**；它们从 CellChat universe 排除，不等于 biological zero。",
            "",
            "## 空间/degree-matched 条件检验",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["残差目标", "LR 方向", "conditional rho", "rho-null", "p", "BH q", "保留 edges"],
            [
                [
                    row.target,
                    row.score,
                    _format_number(row.conditional_rank_correlation),
                    _format_number(row.observed_minus_null_mean),
                    _format_number(row.empirical_p_greater),
                    _format_number(row.bh_q_within_degree_matched_family),
                    int(row.n_edges_retained),
                ]
                for row in matched.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 汇报时必须保留的解释边界",
            "",
            "- attention 是模型内部 gate magnitude，不是 CCC probability；exact message 是模型贡献，不是 biochemical flux。",
            "- edge classifier 本身使用了 LR 信息，因此 LR association 有部分 circularity，不是独立验证；forward 和 reverse 必须一起报告。",
            "- NicheNet 两个条件都使用 mouse ligand-target prior；project-LR 条件只替换 candidate gate。其 type-pair score 是 derived、non-spatial 的。",
            "- CellAgentChat 使用 zebrafish-to-mouse projected expression 和 mouse-prior method；all-confidence orthology 只能叫 sensitivity。",
            "- database-identifiable axes 不证明精确 LR pair 或 sender-to-receiver direction；one-seed virtual removal 不是 causal perturbation。",
            "",
            "完整英文 reviewer 报告见 `README.md`，逐 stage 数值见 `tables/`。",
            "",
            "![跨方法 rank concordance](figures/rank_concordance.png)",
            "",
            "![Reviewer validation axes](figures/reviewer_validation_axes.png)",
            "",
        ]
    )
    path = output / "README_CN.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_bundle(args: argparse.Namespace) -> Mapping[str, Any]:
    comparison_dir = args.comparison_dir.expanduser().resolve()
    validation_dir = args.validation_dir.expanduser().resolve()
    comparison = _validate_comparison(comparison_dir)
    _validate_method_unavailable_rows(comparison_dir, comparison)
    _validate_structural_zero_completion(comparison_dir, comparison)
    _validate_positive_support_top_k(comparison_dir, comparison)
    validation = _validate_validation_axes(validation_dir)
    positive_dir = (
        getattr(args, "positive_consistency_dir", None).expanduser().resolve()
        if getattr(args, "positive_consistency_dir", None) is not None
        else None
    )
    positive_manifest: Mapping[str, Any] | None = None
    positive_records: Mapping[str, Mapping[str, Any]] = {}
    if positive_dir is not None:
        positive_manifest, positive_records = _validate_positive_consistency(
            positive_dir
        )
    method_dirs = _resolve_method_dirs(comparison, args)
    manifests, conditions = _validate_methods(method_dirs)
    _crosscheck_orthology_records(comparison, manifests)

    output_path = args.output_dir.expanduser().resolve()
    _require(
        output_path
        not in {
            comparison_dir,
            validation_dir,
            *method_dirs.values(),
            *([positive_dir] if positive_dir is not None else []),
        },
        "Output directory must differ from every source result directory",
    )
    output = _prepare_output(output_path, bool(args.overwrite))
    comparison_records = _manifest_artifacts_by_name(comparison)
    validation_records = _manifest_artifacts_by_name(validation)

    optional_validation_present = [
        (validation_dir / filename).is_file() for filename in OPTIONAL_VALIDATION_TABLES
    ]
    _require(
        all(optional_validation_present) or not any(optional_validation_present),
        "Virtual-ablation tables must be present together or both absent",
    )

    copied: list[Path] = []
    for filename in COMPARISON_TABLES:
        copied.append(
            _copy_verified(
                source_dir=comparison_dir,
                filename=filename,
                records=comparison_records,
                destination_dir=output / "tables",
            )
        )
    for filename in COMPARISON_FIGURES:
        copied.append(
            _copy_verified(
                source_dir=comparison_dir,
                filename=filename,
                records=comparison_records,
                destination_dir=output / "figures",
            )
        )
    for filename in VALIDATION_TABLES:
        copied.append(
            _copy_verified(
                source_dir=validation_dir,
                filename=filename,
                records=validation_records,
                destination_dir=output / "tables",
            )
        )
    for filename in OPTIONAL_VALIDATION_TABLES:
        if (validation_dir / filename).is_file():
            copied.append(
                _copy_verified(
                    source_dir=validation_dir,
                    filename=filename,
                    records=validation_records,
                    destination_dir=output / "tables",
                )
            )
    for filename in VALIDATION_FIGURES:
        copied.append(
            _copy_verified(
                source_dir=validation_dir,
                filename=filename,
                records=validation_records,
                destination_dir=output / "figures",
            )
        )
    if positive_dir is not None:
        for filename in POSITIVE_CONSISTENCY_TABLES:
            copied.append(
                _copy_verified(
                    source_dir=positive_dir,
                    filename=filename,
                    records=positive_records,
                    destination_dir=output / "tables",
                )
            )
        for filename in POSITIVE_CONSISTENCY_FIGURES:
            copied.append(
                _copy_verified(
                    source_dir=positive_dir,
                    filename=filename,
                    records=positive_records,
                    destination_dir=output / "figures",
                )
            )

    note_sources = (
        (
            comparison_dir / "README.md",
            output / "notes" / "comparison_interpretation.md",
        ),
        (
            validation_dir / "reviewer_validation_summary.md",
            output / "notes" / "reviewer_validation_summary.md",
        ),
    )
    for source, destination in note_sources:
        source_records = (
            comparison_records
            if source.parent == comparison_dir
            else validation_records
        )
        _verify_artifact(source, source_records.get(source.name))
        copied.append(_copy_plain(source, destination))
    if positive_dir is not None:
        positive_notes = {
            "README.md": "positive_consistency_interpretation.md",
            "reviewer_response_draft.md": "reviewer_response_draft.md",
            "汇报说明.md": "positive_consistency_CN.md",
        }
        for source_name, destination_name in positive_notes.items():
            source = positive_dir / source_name
            _verify_artifact(source, positive_records.get(source_name))
            copied.append(_copy_plain(source, output / "notes" / destination_name))

    manifest_sources = {
        "comparison_manifest.json": comparison_dir / "manifest.json",
        "validation_axes_manifest.json": validation_dir / "run_manifest.json",
        "commot_manifest.json": manifests["commot"]["path"],
        "cellchat_manifest.json": manifests["cellchat"]["path"],
        "nichenet_default_manifest.json": manifests["nichenet_default"]["path"],
        "nichenet_project_lr_manifest.json": manifests["nichenet_custom"]["path"],
        "cellagentchat_dual_manifest.json": manifests["cellagentchat_dual"]["path"],
        "cellagentchat_official_manifest.json": manifests["cellagentchat_official"][
            "path"
        ],
        "cellagentchat_project_lr_manifest.json": manifests["cellagentchat_custom"][
            "path"
        ],
    }
    if positive_dir is not None:
        manifest_sources["positive_consistency_manifest.json"] = (
            positive_dir / "manifest.json"
        )
    for destination_name, source in manifest_sources.items():
        copied.append(_copy_plain(source, output / "manifests" / destination_name))

    readme = _build_readme(
        output=output,
        comparison=comparison,
        validation=validation,
        manifests=manifests,
        conditions=conditions,
    )
    copied.append(readme)
    readme_cn = _build_readme_cn(
        output=output,
        comparison=comparison,
        conditions=conditions,
    )
    copied.append(readme_cn)
    if positive_dir is not None:
        with readme.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## Positive-consistency addendum\n\n"
                "The primary consensus in this addendum excludes CytoBridge; the "
                "CellAgentChat-style all-method ensemble is explicitly self-included.\n\n"
                "![Positive communication consistency](figures/positive_consistency_overview.png)\n\n"
                "![Top-signal biology](figures/top_signal_biology.png)\n\n"
                "See `notes/reviewer_response_draft.md` and "
                "`notes/positive_consistency_interpretation.md`.\n"
            )
        with readme_cn.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## 正向 communication consistency 补充分析\n\n"
                "主分析的 external-only consensus 不包含 CytoBridge；包含我们"
                "自身的 all-method ensemble 仅作论文式 supporting comparison。\n\n"
                "![正向一致性](figures/positive_consistency_overview.png)\n\n"
                "![Top-signal biology](figures/top_signal_biology.png)\n\n"
                "详细解释见 `notes/positive_consistency_CN.md`。\n"
            )

    bundle_manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "workflow": "zebrafish_ccc_reviewer_ready_report_bundle",
        "status": "complete",
        "formal_reviewer_ready": True,
        "formal_status_scope": (
            "input/result/provenance contract complete; does not override per-condition "
            "primary versus sensitivity evidence tiers"
        ),
        "score_view_contract": {
            "n_internal_cytobridge_views": 2,
            "n_external_method_database_conditions": 6,
            "expected_view_ids": list(EXPECTED_VIEW_IDS),
            "raw_cross_method_units_compared": False,
            "structural_zero_completion_is_evaluated_universe_only": True,
            "positive_support_boundary_tie_aware_top_k": True,
        },
        "upstream_formal_readiness_checks": comparison["formal_readiness_checks"],
        "score_view_zero_completion": comparison["score_view_zero_completion"],
        "cellchat_method_unavailable_lr_rows": comparison[
            "cellchat_method_unavailable_lr_rows"
        ],
        "conditions": list(conditions),
        "claims": {
            "cytobridge_attention_is_ccc_probability": False,
            "cytobridge_exact_message_is_biochemical_flux": False,
            "method_agreement_is_ground_truth": False,
            "nichenet_derived_type_pair_score_is_native_spatial_ccc": False,
            "all_confidence_cross_species_sensitivity_is_primary": False,
            "cellchat_method_unavailable_rows_are_biological_zero": False,
            "unevaluated_or_skipped_units_are_structural_zero": False,
            "virtual_removal_is_causal_perturbation": False,
            "all_method_self_included_ensemble_is_independent_validation": False,
            "nichenet_downstream_consistency_is_direct_spatial_ccc_strength": False,
        },
        "positive_consistency_included": positive_dir is not None,
        "positive_consistency_contract": (
            positive_manifest.get("primary_design")
            if positive_manifest is not None
            else None
        ),
        "source_manifests": {
            name: _file_record(source) for name, source in manifest_sources.items()
        },
        "artifacts": {
            str(path.relative_to(output)): _file_record(path) for path in copied
        },
    }
    manifest_path = output / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return bundle_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_bundle(args)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "formal_reviewer_ready": manifest["formal_reviewer_ready"],
                "n_external_conditions": manifest["score_view_contract"][
                    "n_external_method_database_conditions"
                ],
                "output_dir": str(args.output_dir.expanduser().resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
