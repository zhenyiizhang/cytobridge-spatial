#!/usr/bin/env python3
"""Strictly assemble two completed parallel CellAgentChat spatial runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

try:
    from .common import (
        CONDITION_LABELS,
        CUSTOM_DATABASE_LABEL,
        OFFICIAL_DATABASE_LABEL,
        PINNED_CELLAGENTCHAT_COMMIT,
        artifact,
        json_value,
        prepare_output,
        utc_now,
        verify_artifact,
        write_json,
    )
    from .run_dual import validate_paired_manifests
except ImportError:  # pragma: no cover - direct CLI execution
    from common import (  # type: ignore
        CONDITION_LABELS,
        CUSTOM_DATABASE_LABEL,
        OFFICIAL_DATABASE_LABEL,
        PINNED_CELLAGENTCHAT_COMMIT,
        artifact,
        json_value,
        prepare_output,
        utc_now,
        verify_artifact,
        write_json,
    )
    from run_dual import validate_paired_manifests  # type: ignore


EXPECTED_STAGES = (0.0, 1.0, 2.0, 3.0, 4.0)
EXPECTED_SAMPLING_SEEDS = (101, 202, 303)
EXPECTED_EPOCHS = 50
EXPECTED_PERMUTATION_SCORE_TARGET = 10_000
EXPECTED_GRID = tuple(
    (stage, seed)
    for stage in EXPECTED_STAGES
    for seed in EXPECTED_SAMPLING_SEEDS
)

REQUIRED_TOP_ARTIFACTS = {
    "cellagentchat_type_pair_scores_by_seed.csv.gz",
    "cellagentchat_type_pair_scores.csv",
    "cellagentchat_lr_scores_raw_by_seed.csv.gz",
    "cellagentchat_lr_scores_significant_by_seed.csv.gz",
    "cellagentchat_cell_receiving_scores_by_seed.csv.gz",
}
REQUIRED_RUN_ARTIFACTS = {
    "background_distribution.csv",
    "cell_type_token_map.csv",
    "sampled_cells.csv",
    "conversion_model.pt",
    "conversion_rates.csv",
    "cellagentchat_lr_scores_raw.csv.gz",
    "cellagentchat_lr_scores_significant.csv",
    "cellagentchat_type_pair_scores.csv",
    "cellagentchat_cell_receiving_scores.csv",
}
SHARED_DESIGN_FIELDS = (
    "species_prior",
    "cross_species_interpretation",
    "sampling_seeds",
    "stages",
    "epochs",
    "learning_rate",
    "batch_size",
    "feature_shuffles",
    "permutation_score_target",
    "multiple_testing",
    "bonferroni_threshold",
    "spatial",
    "spatial_key",
    "permutation_background_distance_scaled",
    "tau",
    "delta",
    "native_primary",
    "significant_lr_count_is_diagnostic",
    "unthresholded_raw_score_sum_is_secondary",
    "torch_sparse_backend",
    "torch_sparse_backend_semantics",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-run-dir", required=True, type=Path)
    parser.add_argument("--custom-run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a JSON object.")
    return value


def _read_manifest(directory: Path) -> tuple[Path, dict[str, Any]]:
    resolved = directory.expanduser().resolve()
    manifest_path = resolved / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Condition manifest must be a JSON object: {manifest_path}")
    return resolved, payload


def _verify_artifact_within(
    record: Mapping[str, Any], *, root: Path, context: str
) -> Path:
    path = verify_artifact(record)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"{context} artifact escapes its condition directory: {path}"
        ) from error
    return path


def _validate_pinned_source(source_value: Any) -> tuple[str, tuple[tuple[str, str], ...]]:
    source = _mapping(source_value, context="CellAgentChat source provenance")
    observed = str(source.get("observed_commit", ""))
    expected = str(source.get("expected_commit", ""))
    if (
        observed != PINNED_CELLAGENTCHAT_COMMIT
        or expected != PINNED_CELLAGENTCHAT_COMMIT
        or source.get("pinned_source_verified") is not True
    ):
        raise RuntimeError(
            "CellAgentChat source is not verified at the pinned v0.2.0 commit."
        )
    if source.get("release") != "v0.2.0":
        raise RuntimeError("CellAgentChat source release must be exactly v0.2.0.")
    files = _mapping(source.get("files"), context="CellAgentChat source files")
    if not files:
        raise RuntimeError("CellAgentChat source provenance contains no files.")
    signatures: list[tuple[str, str]] = []
    for label, raw_record in sorted(files.items()):
        record = _mapping(raw_record, context=f"source artifact {label}")
        verify_artifact(record)
        digest = str(record.get("sha256", ""))
        if not digest:
            raise RuntimeError(f"Source artifact {label} lacks a SHA256.")
        signatures.append((str(label), digest))
    return observed, tuple(signatures)


def _validate_formal_design(design_value: Any) -> Mapping[str, Any]:
    design = _mapping(design_value, context="CellAgentChat design")
    if tuple(design.get("stages", ())) != EXPECTED_STAGES:
        raise RuntimeError(
            f"Formal CellAgentChat stages must be exactly {EXPECTED_STAGES}."
        )
    if tuple(design.get("sampling_seeds", ())) != EXPECTED_SAMPLING_SEEDS:
        raise RuntimeError(
            "Formal CellAgentChat sampling seeds must be exactly "
            f"{EXPECTED_SAMPLING_SEEDS}."
        )
    if design.get("epochs") != EXPECTED_EPOCHS:
        raise RuntimeError(
            f"Formal CellAgentChat epochs must be exactly {EXPECTED_EPOCHS}."
        )
    if design.get("permutation_score_target") != EXPECTED_PERMUTATION_SCORE_TARGET:
        raise RuntimeError(
            "Formal CellAgentChat permutation score target must be exactly "
            f"{EXPECTED_PERMUTATION_SCORE_TARGET}."
        )
    if design.get("spatial") is not True:
        raise RuntimeError("Formal CellAgentChat assembly requires spatial=True.")
    if design.get("permutation_background_distance_scaled") is not True:
        raise RuntimeError(
            "Formal CellAgentChat assembly requires a distance-scaled permutation background."
        )
    return design


def _validate_run_records(
    manifest: Mapping[str, Any], *, directory: Path
) -> tuple[dict[float, str], dict[tuple[float, int], tuple[int, int, int]]]:
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise RuntimeError("CellAgentChat condition manifest runs must be a list.")
    observed_grid: list[tuple[float, int]] = []
    stage_labels: dict[float, str] = {}
    shared_dimensions: dict[tuple[float, int], tuple[int, int, int]] = {}
    raw_total = 0
    significant_total = 0
    receiving_total = 0
    type_pair_total = 0
    for index, raw_run in enumerate(runs):
        run = _mapping(raw_run, context=f"run record {index}")
        stage = float(run.get("stage"))
        seed = int(run.get("sampling_seed"))
        observed_grid.append((stage, seed))
        label = str(run.get("stage_label", "")).strip()
        if not label:
            raise RuntimeError(f"Run stage={stage:g}, seed={seed} lacks a stage label.")
        previous_label = stage_labels.setdefault(stage, label)
        if previous_label != label:
            raise RuntimeError(f"Stage {stage:g} has inconsistent labels within a condition.")
        for field, expected in (
            ("spatial_distance_used", True),
            ("distance_scaled_permutation_background", True),
            ("embedding_used_as_spatial", False),
        ):
            if run.get(field) is not expected:
                raise RuntimeError(
                    f"Run stage={stage:g}, seed={seed} has invalid {field}."
                )
        n_cells = int(run.get("n_cells", 0))
        n_cell_types = int(run.get("n_cell_types", 0))
        n_genes = int(run.get("n_genes", 0))
        if min(n_cells, n_cell_types, n_genes) < 1:
            raise RuntimeError(
                f"Run stage={stage:g}, seed={seed} has invalid dimensions."
            )
        shared_dimensions[(stage, seed)] = (n_cells, n_cell_types, n_genes)
        raw_total += int(run.get("n_raw_lr_rows", -1))
        significant_total += int(run.get("n_significant_lr_rows", -1))
        receiving_total += n_cells
        type_pair_total += n_cell_types**2

        artifacts = _mapping(
            run.get("artifacts"), context=f"run stage={stage:g}, seed={seed} artifacts"
        )
        missing = sorted(REQUIRED_RUN_ARTIFACTS.difference(artifacts))
        if missing:
            raise RuntimeError(
                f"Run stage={stage:g}, seed={seed} is incomplete; missing artifacts: {missing}."
            )
        artifact_paths = []
        for name in sorted(REQUIRED_RUN_ARTIFACTS):
            artifact_paths.append(
                _verify_artifact_within(
                    _mapping(artifacts[name], context=f"run artifact {name}"),
                    root=directory,
                    context=f"run stage={stage:g}, seed={seed}",
                )
            )
        parents = {path.parent for path in artifact_paths}
        if len(parents) != 1:
            raise RuntimeError(
                f"Run stage={stage:g}, seed={seed} artifacts span multiple directories."
            )
        run_manifest_path = parents.pop() / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise FileNotFoundError(run_manifest_path)
        run_on_disk = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_on_disk != run:
            raise RuntimeError(
                f"Embedded and on-disk run manifests disagree: {run_manifest_path}"
            )

    if tuple(observed_grid) != EXPECTED_GRID:
        raise RuntimeError(
            "CellAgentChat condition does not contain the exact formal stage-by-seed "
            f"grid in stage-major order: expected {EXPECTED_GRID}, observed {tuple(observed_grid)}."
        )
    counts = _mapping(manifest.get("counts"), context="CellAgentChat counts")
    expected_counts = {
        "n_runs": len(EXPECTED_GRID),
        "type_pair_rows_by_seed": type_pair_total,
        "raw_lr_rows": raw_total,
        "significant_lr_rows": significant_total,
        "cell_receiving_rows": receiving_total,
    }
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            raise RuntimeError(
                f"CellAgentChat count {field} disagrees with completed run records: "
                f"expected {expected}, observed {counts.get(field)!r}."
            )
    return stage_labels, shared_dimensions


def _csv_row_count(path: Path) -> int:
    try:
        return int(len(pd.read_csv(path)))
    except pd.errors.EmptyDataError:
        return 0


def _validate_condition(
    directory: Path, *, expected_label: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved, manifest = _read_manifest(directory)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("CellAgentChat condition schema_version must be exactly 1.")
    if manifest.get("method") != "official_cellagentchat_v0_2_0_spatial":
        raise RuntimeError("Unexpected CellAgentChat condition method identifier.")
    if manifest.get("database_condition") != expected_label:
        raise RuntimeError(
            f"Expected database condition {expected_label!r}, observed "
            f"{manifest.get('database_condition')!r}."
        )
    source_commit, source_signature = _validate_pinned_source(manifest.get("source"))
    design = _validate_formal_design(manifest.get("design"))

    shared = _mapping(manifest.get("shared_input"), context="CellAgentChat shared_input")
    for name in ("preparation_manifest", "mapped_expression", "sample_plan", "database"):
        record = _mapping(shared.get(name), context=f"shared input {name}")
        verify_artifact(record)
        if not str(record.get("sha256", "")):
            raise RuntimeError(f"Shared input {name} lacks a SHA256.")
    claims = _mapping(
        shared.get("preparation_claims"), context="CellAgentChat preparation claims"
    )
    for name in (
        "orthology_policy",
        "orthology_analysis_tier",
        "primary_claim_allowed",
        "formal_primary",
    ):
        if name not in claims:
            raise RuntimeError(f"Preparation claims lack required field {name!r}.")

    stage_labels, dimensions = _validate_run_records(manifest, directory=resolved)
    top_artifacts = _mapping(
        manifest.get("artifacts"), context="CellAgentChat top-level artifacts"
    )
    missing_top = sorted(REQUIRED_TOP_ARTIFACTS.difference(top_artifacts))
    if missing_top:
        raise RuntimeError(f"CellAgentChat output is incomplete: {missing_top}.")
    verified_top = {
        name: _verify_artifact_within(
            _mapping(top_artifacts[name], context=f"top-level artifact {name}"),
            root=resolved,
            context="top-level",
        )
        for name in sorted(REQUIRED_TOP_ARTIFACTS)
    }
    counts = _mapping(manifest["counts"], context="CellAgentChat counts")
    row_contract = {
        "cellagentchat_type_pair_scores_by_seed.csv.gz": "type_pair_rows_by_seed",
        "cellagentchat_lr_scores_raw_by_seed.csv.gz": "raw_lr_rows",
        "cellagentchat_lr_scores_significant_by_seed.csv.gz": "significant_lr_rows",
        "cellagentchat_cell_receiving_scores_by_seed.csv.gz": "cell_receiving_rows",
    }
    for filename, count_field in row_contract.items():
        observed_rows = _csv_row_count(verified_top[filename])
        if observed_rows != counts[count_field]:
            raise RuntimeError(
                f"{filename} has {observed_rows} rows but manifest {count_field} is "
                f"{counts[count_field]}."
            )
    return resolved, manifest, {
        "source_commit": source_commit,
        "source_signature": source_signature,
        "design": {field: design.get(field) for field in SHARED_DESIGN_FIELDS},
        "stage_labels": stage_labels,
        "dimensions": dimensions,
    }


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _summary_rows(manifests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        counts = manifest["counts"]
        shared = manifest["shared_input"]
        claims = shared["preparation_claims"]
        rows.append(
            {
                "database_condition": manifest["database_condition"],
                "n_runs": counts["n_runs"],
                "type_pair_rows_by_seed": counts["type_pair_rows_by_seed"],
                "raw_lr_rows": counts["raw_lr_rows"],
                "significant_lr_rows": counts["significant_lr_rows"],
                "sample_plan_sha256": shared["sample_plan"]["sha256"],
                "mapped_expression_sha256": shared["mapped_expression"]["sha256"],
                "database_sha256": shared["database"]["sha256"],
                "orthology_policy": claims["orthology_policy"],
                "orthology_analysis_tier": claims["orthology_analysis_tier"],
                "primary_claim_allowed": claims["primary_claim_allowed"],
                "source_commit": manifest["source"]["observed_commit"],
                "formal_full_grid_verified": True,
                "non_smoke_verified": True,
            }
        )
    return rows


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    condition_inputs = (
        (args.official_run_dir, OFFICIAL_DATABASE_LABEL),
        (args.custom_run_dir, CUSTOM_DATABASE_LABEL),
    )
    validated = [
        _validate_condition(path, expected_label=label)
        for path, label in condition_inputs
    ]
    directories = [item[0] for item in validated]
    manifests = [item[1] for item in validated]
    metadata = [item[2] for item in validated]
    if directories[0] == directories[1]:
        raise RuntimeError("Official and custom condition directories must be distinct.")

    shared_hashes = validate_paired_manifests(manifests)
    if metadata[0]["source_commit"] != metadata[1]["source_commit"]:
        raise RuntimeError("CellAgentChat conditions used different source commits.")
    if metadata[0]["source_signature"] != metadata[1]["source_signature"]:
        raise RuntimeError("CellAgentChat conditions used different pinned source files.")
    if metadata[0]["design"] != metadata[1]["design"]:
        raise RuntimeError("CellAgentChat conditions do not share the same formal design.")
    if metadata[0]["stage_labels"] != metadata[1]["stage_labels"]:
        raise RuntimeError("CellAgentChat conditions have different stage labels.")
    if metadata[0]["dimensions"] != metadata[1]["dimensions"]:
        raise RuntimeError(
            "CellAgentChat conditions do not share the same per-run cell/gene dimensions."
        )

    output_path = args.output_dir.expanduser().resolve()
    for directory in directories:
        if _paths_overlap(output_path, directory):
            raise RuntimeError(
                "Assembly output must not contain, equal, or be contained by a condition "
                f"directory: output={output_path}, condition={directory}."
            )
    output = prepare_output(output_path, bool(args.overwrite))
    links: dict[str, dict[str, str]] = {}
    for label, directory in zip(CONDITION_LABELS, directories):
        link = output / label
        link.symlink_to(directory, target_is_directory=True)
        links[label] = {
            "path": str(link),
            "target": str(directory),
            "kind": "directory_symlink",
        }

    readiness_path = output / "dual_condition_run_summary.csv"
    pd.DataFrame(_summary_rows(manifests)).to_csv(readiness_path, index=False)
    manifest_paths = [output / label / "manifest.json" for label in CONDITION_LABELS]
    dual_manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workflow": "official_cellagentchat_spatial_dual_lr_database",
        "status": "complete",
        "assembly_mode": "validated_parallel_run_directory_symlinks",
        "conditions": list(CONDITION_LABELS),
        "same_mapped_expression_and_sample_plan_verified": True,
        "same_preparation_manifest_and_orthology_claims_verified": True,
        "same_formal_design_and_pinned_source_verified": True,
        "exact_stage_seed_grid_verified": True,
        "formal_non_smoke_verified": True,
        "shared_sha256": shared_hashes,
        "preparation_claims": manifests[0]["shared_input"]["preparation_claims"],
        "database_sha256_are_distinct": True,
        "formal_design": {
            "stages": list(EXPECTED_STAGES),
            "sampling_seeds": list(EXPECTED_SAMPLING_SEEDS),
            "epochs": EXPECTED_EPOCHS,
            "permutation_score_target": EXPECTED_PERMUTATION_SCORE_TARGET,
            "source_commit": PINNED_CELLAGENTCHAT_COMMIT,
        },
        "condition_links": links,
        "condition_manifests": {
            label: artifact(path)
            for label, path in zip(CONDITION_LABELS, manifest_paths)
        },
        "artifacts": {readiness_path.name: artifact(readiness_path)},
    }
    write_json(output / "manifest.json", dual_manifest)
    return dual_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(json.dumps(json_value({"status": "ok", **manifest}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
