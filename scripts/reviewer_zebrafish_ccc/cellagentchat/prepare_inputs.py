#!/usr/bin/env python3
"""Prepare one shared mouse-ortholog input for both CellAgentChat conditions.

The formal primary mode uses only confidence>=1 reciprocal one-to-one
orthologs and subsets the already single-log normalized expression matrix
without changing its values.  Confidence-unfiltered and many-to-one adapters
are separately labelled sensitivities.  Both LR database conditions and every
sampling seed consume the exact same mapped H5AD and frozen cell-sampling plan
produced here.
"""

from __future__ import annotations

import argparse
from hashlib import md5
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ORTHOLOGY_MANIFEST_WORKFLOW = "ensembl_compara_zebrafish_mouse_one2one_bijective_export"
ORTHOLOGY_POLICY_TIERS = {
    "strict_confidence1": ("primary", True, "require_equal_1"),
    "one2one_bijective_all_confidence": (
        "sensitivity",
        False,
        "not_filtered",
    ),
}

try:
    from .common import (
        PINNED_CELLAGENTCHAT_COMMIT,
        artifact,
        build_lr_databases,
        build_sampling_plan,
        csv_ints,
        csv_strings,
        prepare_output,
        project_expression_matrices,
        read_table,
        select_orthology_mapping,
        sha256_file,
        utc_now,
        validate_official_source,
        write_json,
    )
except ImportError:  # pragma: no cover - direct CLI execution
    from common import (  # type: ignore
        PINNED_CELLAGENTCHAT_COMMIT,
        artifact,
        build_lr_databases,
        build_sampling_plan,
        csv_ints,
        csv_strings,
        prepare_output,
        project_expression_matrices,
        read_table,
        select_orthology_mapping,
        sha256_file,
        utc_now,
        validate_official_source,
        write_json,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression-h5ad", required=True, type=Path)
    parser.add_argument("--orthology-map", required=True, type=Path)
    parser.add_argument(
        "--orthology-manifest",
        type=Path,
        help=(
            "Optional manifest from export_ensembl_one2one.R. When supplied, "
            "the mapping filename, MD5, Ensembl release, policy, tier, and "
            "filter contract are verified before preparation."
        ),
    )
    parser.add_argument(
        "--orthology-analysis-tier",
        choices=("auto", "primary", "sensitivity"),
        default="auto",
        help=(
            "Audit label for the mapping. auto derives it from a verified "
            "manifest or the CLI confidence filter."
        ),
    )
    parser.add_argument("--custom-lr-database", required=True, type=Path)
    parser.add_argument("--cellagentchat-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-expression-sha256")
    parser.add_argument("--expected-custom-lr-sha256")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--cell-type-key", default="Annotation")
    parser.add_argument("--time-key", default="time_point_processed")
    parser.add_argument("--time-label-key", default="time_label")
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument(
        "--copy-obsm-keys",
        type=csv_strings,
        default=("spatial_aligned", "spatial_original", "spatial"),
    )
    parser.add_argument("--orthology-separator", default="auto")
    parser.add_argument("--source-gene-column", default="zebrafish_gene")
    parser.add_argument("--target-gene-column", default="mouse_gene")
    parser.add_argument("--orthology-type-column", default="orthology_type")
    parser.add_argument(
        "--allowed-orthology-types",
        type=csv_strings,
        default=("ortholog_one2one",),
    )
    parser.add_argument("--confidence-column", default="orthology_confidence")
    parser.add_argument("--minimum-confidence", type=float, default=1.0)
    parser.add_argument(
        "--mapping-policy",
        choices=("strict_one_to_one", "many_to_one_sum"),
        default="strict_one_to_one",
    )
    parser.add_argument(
        "--expression-projection-mode",
        choices=("strict_log1p_rename", "counts_sum_then_log1p"),
        default="strict_log1p_rename",
    )
    parser.add_argument(
        "--normalization-target-sum",
        type=float,
        default=1105.0,
        help=(
            "Used only by counts_sum_then_log1p. The default is the frozen "
            "zebrafish preprocessing target, not the post-filter library median."
        ),
    )
    parser.add_argument("--custom-ligand-column", default="0")
    parser.add_argument("--custom-receptor-column", default="1")
    parser.add_argument("--custom-pathway-column", default="2")
    parser.add_argument("--custom-category-column", default="3")
    parser.add_argument("--custom-lr-separator", default="auto")
    parser.add_argument("--sampling-seeds", type=csv_ints, default=(101, 202, 303))
    parser.add_argument("--max-cells-per-type", type=int, default=500)
    parser.add_argument("--minimum-cells-per-type", type=int, default=1)
    parser.add_argument("--allow-unpinned-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _optional_column(value: str) -> str | None:
    value = str(value).strip()
    return value or None


def _md5_file(path: Path) -> str:
    digest = md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_md5(value: object) -> bool:
    return re.fullmatch(r"[0-9a-fA-F]{32}", str(value)) is not None


def _derived_unverified_orthology_policy(
    *,
    mapping_policy: str,
    orthology_type_column: str | None,
    allowed_orthology_types: Sequence[str],
    confidence_column: str | None,
    minimum_confidence: float,
) -> tuple[str, str, bool]:
    strict_one2one = (
        mapping_policy == "strict_one_to_one"
        and orthology_type_column is not None
        and set(allowed_orthology_types) == {"ortholog_one2one"}
    )
    if strict_one2one and confidence_column is not None and minimum_confidence >= 1:
        return "strict_confidence1_unverified_source", "primary", True
    return (
        f"{mapping_policy}_minimum_confidence_{minimum_confidence:g}_unverified_source",
        "sensitivity",
        False,
    )


def resolve_orthology_provenance(
    *,
    orthology_path: Path,
    orthology_frame: pd.DataFrame,
    selected_mapping: pd.DataFrame,
    mapping_counts: Mapping[str, Any],
    mapping_policy: str,
    orthology_type_column: str | None,
    allowed_orthology_types: Sequence[str],
    confidence_column: str | None,
    minimum_confidence: float,
    requested_analysis_tier: str,
    manifest_path: Path | None,
) -> dict[str, Any]:
    """Resolve and verify the orthology policy without upgrading sensitivities.

    A confidence-unfiltered bijective mapping is useful for coverage, but it is
    not the formal primary mapping.  When an Ensembl exporter manifest is
    supplied, this function binds the selected CSV to that manifest and makes
    its policy/tier authoritative.  Without a manifest, provenance is labelled
    unverified and only a confidence threshold of at least one can be primary-
    eligible.
    """

    if requested_analysis_tier not in {"auto", "primary", "sensitivity"}:
        raise ValueError(
            "orthology analysis tier must be auto, primary, or sensitivity."
        )
    if not math.isfinite(float(minimum_confidence)):
        raise ValueError("--minimum-confidence must be finite.")
    if not 0 <= float(minimum_confidence) <= 1:
        raise ValueError("--minimum-confidence must lie in [0, 1].")

    if manifest_path is None:
        policy, policy_tier, policy_primary = _derived_unverified_orthology_policy(
            mapping_policy=mapping_policy,
            orthology_type_column=orthology_type_column,
            allowed_orthology_types=allowed_orthology_types,
            confidence_column=confidence_column,
            minimum_confidence=float(minimum_confidence),
        )
        if requested_analysis_tier != "auto" and requested_analysis_tier != policy_tier:
            raise ValueError(
                "Requested orthology analysis tier conflicts with the CLI-derived "
                f"policy: requested={requested_analysis_tier}, derived={policy_tier}."
            )
        return {
            "orthology_policy": policy,
            "analysis_tier": policy_tier,
            "policy_primary_claim_allowed": policy_primary,
            "mapping_source_manifest": {
                "provided": False,
                "verified": False,
            },
        }

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not parse orthology manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Orthology manifest must contain a JSON object.")

    policy = str(payload.get("mapping_policy", ""))
    if policy not in ORTHOLOGY_POLICY_TIERS:
        raise ValueError(f"Unsupported orthology manifest mapping_policy: {policy!r}.")
    expected_tier, expected_primary, confidence_policy = ORTHOLOGY_POLICY_TIERS[policy]
    required_values = {
        "schema_version": 2,
        "workflow": ORTHOLOGY_MANIFEST_WORKFLOW,
        "status": "complete",
        "ensembl_release": 116,
        "mapping_policy": policy,
        "analysis_tier": expected_tier,
        "primary_claim_allowed": expected_primary,
        "mapping_file": orthology_path.name,
    }
    for key, expected in required_values.items():
        if payload.get(key) != expected:
            raise ValueError(
                "Orthology manifest contract mismatch: "
                f"expected {key}={expected!r}, observed {payload.get(key)!r}."
            )
    filters = payload.get("filter")
    if not isinstance(filters, dict):
        raise ValueError("Orthology manifest lacks a filter object.")
    required_filters = {
        "orthology_type": "ortholog_one2one",
        "orthology_confidence_policy": confidence_policy,
        "nonempty_symbols": True,
        "symbol_level_bijection_after_casefold": True,
    }
    for key, expected in required_filters.items():
        if filters.get(key) != expected:
            raise ValueError(
                "Orthology manifest filter mismatch: "
                f"expected {key}={expected!r}, observed {filters.get(key)!r}."
            )
    output_md5 = payload.get("output_md5")
    if not isinstance(output_md5, dict):
        raise ValueError("Orthology manifest lacks an output_md5 object.")
    recorded_mapping_md5 = str(output_md5.get("mapping", ""))
    observed_mapping_md5 = _md5_file(orthology_path)
    if (
        not _valid_md5(recorded_mapping_md5)
        or recorded_mapping_md5.casefold() != observed_mapping_md5
    ):
        raise ValueError(
            "Orthology map MD5 does not match its source manifest: "
            f"recorded={recorded_mapping_md5 or '<missing>'}, "
            f"observed={observed_mapping_md5}."
        )
    raw_md5 = str(output_md5.get("raw", ""))
    if not _valid_md5(raw_md5):
        raise ValueError("Orthology manifest lacks a valid output_md5.raw value.")
    manifest_counts = payload.get("counts")
    if not isinstance(manifest_counts, dict):
        raise ValueError("Orthology manifest lacks a counts object.")
    manifest_count = manifest_counts.get("selected_bijective_symbol_pairs")
    if not isinstance(manifest_count, int) or manifest_count <= 0:
        raise ValueError(
            "Orthology manifest lacks a positive selected_bijective_symbol_pairs count."
        )
    if manifest_count != len(orthology_frame):
        raise ValueError(
            "Orthology manifest selected count does not equal mapping CSV rows: "
            f"manifest={manifest_count}, CSV={len(orthology_frame)}."
        )
    if int(mapping_counts.get("selected_rows", -1)) != len(orthology_frame) or len(
        selected_mapping
    ) != len(orthology_frame):
        raise ValueError(
            "CellAgentChat CLI filters changed the manifest-selected orthology map; "
            "all manifest rows must remain selected."
        )
    if mapping_policy != "strict_one_to_one":
        raise ValueError(
            "Ensembl bijective manifests require --mapping-policy strict_one_to_one."
        )
    if orthology_type_column is None or set(allowed_orthology_types) != {
        "ortholog_one2one"
    }:
        raise ValueError(
            "Ensembl manifests require an orthology_type column filtered only to "
            "ortholog_one2one."
        )
    if confidence_column is None:
        raise ValueError("Ensembl manifests require an orthology confidence column.")
    if policy == "strict_confidence1":
        if not math.isclose(float(minimum_confidence), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "strict_confidence1 manifest requires --minimum-confidence 1."
            )
    else:
        if not math.isclose(float(minimum_confidence), 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "one2one_bijective_all_confidence manifest requires "
                "--minimum-confidence 0."
            )
        confidence = pd.to_numeric(orthology_frame[confidence_column], errors="coerce")
        if confidence.isna().any() or (confidence < 0).any():
            raise ValueError(
                "Confidence-unfiltered CellAgentChat sensitivity requires finite "
                "nonnegative confidence values so minimum-confidence=0 selects all rows."
            )
        source_input = payload.get("source_input")
        if payload.get("source_mode") != "frozen_raw_input" or not isinstance(
            source_input, dict
        ):
            raise ValueError(
                "Formal all-confidence sensitivity manifest must be replayed from "
                "a frozen Ensembl-116 raw input."
            )
        if (
            not _valid_md5(source_input.get("md5"))
            or not isinstance(source_input.get("size_bytes"), (int, float))
            or float(source_input["size_bytes"]) <= 0
        ):
            raise ValueError(
                "All-confidence sensitivity manifest lacks a complete frozen raw "
                "source size/MD5 record."
            )
    if requested_analysis_tier != "auto" and requested_analysis_tier != expected_tier:
        raise ValueError(
            "Requested orthology analysis tier conflicts with the verified manifest: "
            f"requested={requested_analysis_tier}, manifest={expected_tier}."
        )

    return {
        "orthology_policy": policy,
        "analysis_tier": expected_tier,
        "policy_primary_claim_allowed": expected_primary,
        "mapping_source_manifest": {
            "provided": True,
            "verified": True,
            "artifact": artifact(manifest_path),
            "schema_version": payload["schema_version"],
            "workflow": payload["workflow"],
            "status": payload["status"],
            "ensembl_release": payload["ensembl_release"],
            "source_mode": payload.get("source_mode"),
            "mapping_file": payload["mapping_file"],
            "mapping_md5": observed_mapping_md5,
            "raw_md5": raw_md5.casefold(),
            "mapping_label": payload.get("mapping_label"),
            "analysis_tier": expected_tier,
            "primary_claim_allowed": expected_primary,
        },
    }


def primary_claim_is_allowed(
    *,
    expression_projection_mode: str,
    mapping_policy: str,
    confidence_column: str | None,
    minimum_confidence: float,
    orthology_provenance: Mapping[str, Any],
    pinned_source_verified: bool,
) -> bool:
    """Return the complete formal-primary gate for a preparation."""

    confidence_primary_eligible = (
        confidence_column is not None and float(minimum_confidence) >= 1.0
    )
    return bool(
        expression_projection_mode == "strict_log1p_rename"
        and mapping_policy == "strict_one_to_one"
        and confidence_primary_eligible
        and orthology_provenance.get("analysis_tier") == "primary"
        and orthology_provenance.get("policy_primary_claim_allowed") is True
        and pinned_source_verified
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    import anndata as ad

    expression_path = args.expression_h5ad.expanduser().resolve()
    orthology_path = args.orthology_map.expanduser().resolve()
    orthology_manifest_path = (
        args.orthology_manifest.expanduser().resolve()
        if args.orthology_manifest is not None
        else None
    )
    custom_lr_path = args.custom_lr_database.expanduser().resolve()
    source = args.cellagentchat_source.expanduser().resolve()
    for path in (expression_path, orthology_path, custom_lr_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expression_sha256 = sha256_file(expression_path)
    custom_lr_sha256 = sha256_file(custom_lr_path)
    expected_hashes = (
        ("expression H5AD", args.expected_expression_sha256, expression_sha256),
        ("custom LR database", args.expected_custom_lr_sha256, custom_lr_sha256),
    )
    for label, expected, observed in expected_hashes:
        if expected and str(expected).lower() != observed:
            raise ValueError(
                f"{label} SHA256 mismatch: expected {expected}, observed {observed}."
            )
    if (
        args.mapping_policy == "many_to_one_sum"
        and args.expression_projection_mode != "counts_sum_then_log1p"
    ):
        raise ValueError(
            "many_to_one_sum requires --expression-projection-mode counts_sum_then_log1p."
        )
    if (
        args.mapping_policy == "strict_one_to_one"
        and args.expression_projection_mode == "counts_sum_then_log1p"
    ):
        # Allowed as an explicit scale sensitivity, but never mistaken for the primary.
        pass
    source_record = validate_official_source(
        source, allow_unpinned=bool(args.allow_unpinned_source)
    )

    data = ad.read_h5ad(expression_path)
    for key in (args.cell_type_key, args.time_key, args.time_label_key):
        if key not in data.obs:
            raise KeyError(f"Missing adata.obs[{key!r}].")
    if args.counts_layer not in data.layers:
        raise KeyError(f"Missing adata.layers[{args.counts_layer!r}].")
    if args.spatial_key not in data.obsm:
        raise KeyError(f"Missing adata.obsm[{args.spatial_key!r}].")
    spatial = np.asarray(data.obsm[args.spatial_key])
    if spatial.shape != (data.n_obs, 2) or not np.isfinite(spatial).all():
        raise ValueError(
            f"adata.obsm[{args.spatial_key!r}] must be a finite N x 2 matrix."
        )

    orthology = read_table(orthology_path, args.orthology_separator)
    selected_mapping, excluded_mapping, mapping_counts = select_orthology_mapping(
        orthology,
        source_column=args.source_gene_column,
        target_column=args.target_gene_column,
        mapping_policy=args.mapping_policy,
        orthology_type_column=_optional_column(args.orthology_type_column),
        allowed_orthology_types=args.allowed_orthology_types,
        confidence_column=_optional_column(args.confidence_column),
        minimum_confidence=float(args.minimum_confidence),
    )
    orthology_provenance = resolve_orthology_provenance(
        orthology_path=orthology_path,
        orthology_frame=orthology,
        selected_mapping=selected_mapping,
        mapping_counts=mapping_counts,
        mapping_policy=args.mapping_policy,
        orthology_type_column=_optional_column(args.orthology_type_column),
        allowed_orthology_types=args.allowed_orthology_types,
        confidence_column=_optional_column(args.confidence_column),
        minimum_confidence=float(args.minimum_confidence),
        requested_analysis_tier=args.orthology_analysis_tier,
        manifest_path=orthology_manifest_path,
    )
    (
        projected_x,
        projected_counts,
        projected_var,
        present_mapping,
        expression_record,
    ) = project_expression_matrices(
        data.X,
        data.layers[args.counts_layer],
        data.var_names,
        selected_mapping,
        mode=args.expression_projection_mode,
        normalization_target_sum=float(args.normalization_target_sum),
    )
    present_sources = set(present_mapping["source_gene"])
    selected_mapping = selected_mapping.copy()
    selected_mapping["present_in_expression"] = selected_mapping["source_gene"].isin(
        present_sources
    )
    present_mapping = selected_mapping.loc[
        selected_mapping["present_in_expression"]
    ].copy()
    if (
        args.expression_projection_mode == "strict_log1p_rename"
        and present_mapping["target_gene"].duplicated().any()
    ):
        raise RuntimeError(
            "Primary strict projection unexpectedly has target collisions."
        )

    obsm: dict[str, np.ndarray] = {}
    for key in args.copy_obsm_keys:
        if key in data.obsm:
            values = np.asarray(data.obsm[key])
            if values.shape[0] != data.n_obs or not np.isfinite(values).all():
                raise ValueError(f"Cannot copy invalid adata.obsm[{key!r}].")
            obsm[key] = values.copy()
    if args.spatial_key not in obsm:
        obsm[args.spatial_key] = spatial.copy()

    mapped = ad.AnnData(
        X=projected_x,
        obs=data.obs.copy(),
        var=projected_var,
        obsm=obsm,
        layers={"counts": projected_counts},
    )
    mapped.obs_names = data.obs_names.astype(str)
    mapped.var_names = projected_var.index.astype(str)
    confidence_primary_eligible = (
        _optional_column(args.confidence_column) is not None
        and float(args.minimum_confidence) >= 1.0
    )
    primary_claim_allowed = primary_claim_is_allowed(
        expression_projection_mode=args.expression_projection_mode,
        mapping_policy=args.mapping_policy,
        confidence_column=_optional_column(args.confidence_column),
        minimum_confidence=float(args.minimum_confidence),
        orthology_provenance=orthology_provenance,
        pinned_source_verified=bool(source_record["pinned_source_verified"]),
    )
    formal_primary = bool(primary_claim_allowed)
    mapped.uns["cellagentchat_projection"] = {
        "schema_version": 1,
        "source_species": "danio_rerio",
        "target_species": "mus_musculus",
        "mapping_policy": str(args.mapping_policy),
        "expression_projection_mode": str(args.expression_projection_mode),
        "normalization_target_sum": expression_record["normalization_target_sum"],
        "source_normalization_target_sum": str(
            data.uns.get("normalization_target_sum", "not_recorded")
        ),
        "selected_space_identity_max_abs_error": expression_record[
            "selected_space_identity_max_abs_error"
        ],
        "orthology_policy": orthology_provenance["orthology_policy"],
        "orthology_analysis_tier": orthology_provenance["analysis_tier"],
        "primary_claim_allowed": formal_primary,
        "formal_primary": formal_primary,
    }

    output = prepare_output(args.output_dir, bool(args.overwrite))
    mapped_path = output / "zebrafish_mouse_ortholog_expression.h5ad"
    mapped.write_h5ad(mapped_path, compression="gzip")

    mapping_used_path = output / "orthology_mapping_used.csv"
    selected_mapping.to_csv(mapping_used_path, index=False)
    mapping_excluded_path = output / "orthology_mapping_excluded.csv"
    excluded_mapping.to_csv(mapping_excluded_path, index=False)
    source_unmapped_path = output / "expression_source_genes_not_mapped.csv"
    pd.DataFrame(
        {
            "source_gene": sorted(
                set(data.var_names.astype(str)).difference(present_sources)
            )
        }
    ).to_csv(source_unmapped_path, index=False)

    sample_plan = build_sampling_plan(
        mapped.obs,
        mapped.obs_names,
        cell_type_key=args.cell_type_key,
        time_key=args.time_key,
        time_label_key=args.time_label_key,
        seeds=args.sampling_seeds,
        max_cells_per_type=int(args.max_cells_per_type),
        minimum_cells_per_type=int(args.minimum_cells_per_type),
    )
    sample_plan_path = output / "shared_sampled_cells.csv.gz"
    sample_plan.to_csv(sample_plan_path, index=False, compression="gzip")
    sample_counts_path = output / "shared_sampling_counts.csv"
    (
        sample_plan.groupby(
            ["sampling_seed", "stage", "stage_label", "cell_type"],
            sort=True,
            as_index=False,
        )
        .agg(
            n_cells_sampled=("obs_name", "size"),
            n_cells_available=("n_type_cells_available", "first"),
        )
        .to_csv(sample_counts_path, index=False)
    )

    custom_lr = read_table(custom_lr_path, args.custom_lr_separator)
    lr_dir = output / "lr_databases"
    official_database = source / "src" / "cellagentchat_data" / "mouse_lr_pair.tsv"
    lr_record = build_lr_databases(
        official_database=official_database,
        custom_database=custom_lr,
        mapping=present_mapping,
        output_dir=lr_dir,
        ligand_column=args.custom_ligand_column,
        receptor_column=args.custom_receptor_column,
        pathway_column=_optional_column(args.custom_pathway_column),
        category_column=_optional_column(args.custom_category_column),
    )

    artifacts = {
        "mapped_expression": artifact(mapped_path),
        "orthology_mapping_used": artifact(mapping_used_path),
        "orthology_mapping_excluded": artifact(mapping_excluded_path),
        "expression_source_genes_not_mapped": artifact(source_unmapped_path),
        "shared_sampled_cells": artifact(sample_plan_path),
        "shared_sampling_counts": artifact(sample_counts_path),
    }
    if orthology_manifest_path is not None:
        artifacts["orthology_source_manifest"] = artifact(orthology_manifest_path)
    artifacts.update(lr_record["artifacts"])
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workflow": "zebrafish_cellagentchat_dual_database_shared_input",
        "formal_primary": formal_primary,
        "orthology_policy": orthology_provenance["orthology_policy"],
        "orthology_analysis_tier": orthology_provenance["analysis_tier"],
        "analysis_tier": orthology_provenance["analysis_tier"],
        "primary_claim_allowed": formal_primary,
        "input": {
            "expression_h5ad": {
                "path": str(expression_path),
                "sha256": expression_sha256,
                "expected_sha256": args.expected_expression_sha256,
                "shape": [int(data.n_obs), int(data.n_vars)],
                "counts_layer": args.counts_layer,
                "source_normalization_target_sum": str(
                    data.uns.get("normalization_target_sum", "not_recorded")
                ),
            },
            "orthology_map": {
                "path": str(orthology_path),
                "sha256": sha256_file(orthology_path),
                "source_gene_column": args.source_gene_column,
                "target_gene_column": args.target_gene_column,
                "source_manifest": orthology_provenance["mapping_source_manifest"],
            },
            "custom_lr_database": {
                "path": str(custom_lr_path),
                "sha256": custom_lr_sha256,
                "expected_sha256": args.expected_custom_lr_sha256,
            },
        },
        "cellagentchat_source": source_record,
        "projection": {
            **expression_record,
            "mapping_policy": args.mapping_policy,
            "orthology_counts": mapping_counts,
            "normalization_guardrail": (
                "primary preserves original single-log values exactly"
                if args.expression_projection_mode == "strict_log1p_rename"
                else "secondary uses frozen target_sum=1105 unless explicitly overridden"
            ),
        },
        "orthology": {
            "orthology_policy": orthology_provenance["orthology_policy"],
            "analysis_tier": orthology_provenance["analysis_tier"],
            "policy_primary_claim_allowed": bool(
                orthology_provenance["policy_primary_claim_allowed"]
            ),
            "primary_claim_allowed": formal_primary,
            "mapping_policy": args.mapping_policy,
            "orthology_type_column": _optional_column(args.orthology_type_column),
            "allowed_orthology_types": list(args.allowed_orthology_types),
            "confidence_column": _optional_column(args.confidence_column),
            "minimum_confidence": float(args.minimum_confidence),
            "confidence_threshold_primary_eligible": confidence_primary_eligible,
            "mapping_source_manifest": orthology_provenance["mapping_source_manifest"],
            "confidence_unfiltered_runs_are_sensitivity": True,
        },
        "sampling": {
            "seeds": list(args.sampling_seeds),
            "max_cells_per_type": int(args.max_cells_per_type),
            "minimum_cells_per_type": int(args.minimum_cells_per_type),
            "n_plan_rows": int(len(sample_plan)),
            "shared_across_database_conditions": True,
        },
        "keys": {
            "cell_type": args.cell_type_key,
            "time": args.time_key,
            "time_label": args.time_label_key,
            "spatial": args.spatial_key,
        },
        "lr_databases": lr_record["databases"],
        "lr_coverage": lr_record["counts"],
        "artifacts": artifacts,
        "pinned_cellagentchat_commit": PINNED_CELLAGENTCHAT_COMMIT,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = run(args)
    print(
        {
            "status": "ok",
            "formal_primary": manifest["formal_primary"],
            "mapped_shape": manifest["artifacts"]["mapped_expression"]["path"],
            "lr_coverage": manifest["lr_coverage"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
