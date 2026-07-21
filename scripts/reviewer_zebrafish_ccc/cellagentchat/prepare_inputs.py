#!/usr/bin/env python3
"""Prepare one shared mouse-ortholog input for both CellAgentChat conditions.

The formal primary mode uses only reciprocal one-to-one orthologs and subsets
the already single-log normalized expression matrix without changing its
values.  A separately labelled many-to-one adapter can instead sum raw counts
and reproduce the frozen preprocessing target sum (1105) before ``log1p``.
Both LR database conditions and every sampling seed consume the exact same
mapped H5AD and frozen cell-sampling plan produced here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

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
    parser.add_argument(
        "--sampling-seeds", type=csv_ints, default=(101, 202, 303)
    )
    parser.add_argument("--max-cells-per-type", type=int, default=500)
    parser.add_argument("--minimum-cells-per-type", type=int, default=1)
    parser.add_argument("--allow-unpinned-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _optional_column(value: str) -> str | None:
    value = str(value).strip()
    return value or None


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    import anndata as ad

    expression_path = args.expression_h5ad.expanduser().resolve()
    orthology_path = args.orthology_map.expanduser().resolve()
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
    if args.mapping_policy == "many_to_one_sum" and args.expression_projection_mode != "counts_sum_then_log1p":
        raise ValueError(
            "many_to_one_sum requires --expression-projection-mode counts_sum_then_log1p."
        )
    if args.mapping_policy == "strict_one_to_one" and args.expression_projection_mode == "counts_sum_then_log1p":
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
    projected_x, projected_counts, projected_var, present_mapping, expression_record = (
        project_expression_matrices(
            data.X,
            data.layers[args.counts_layer],
            data.var_names,
            selected_mapping,
            mode=args.expression_projection_mode,
            normalization_target_sum=float(args.normalization_target_sum),
        )
    )
    present_sources = set(present_mapping["source_gene"])
    selected_mapping = selected_mapping.copy()
    selected_mapping["present_in_expression"] = selected_mapping["source_gene"].isin(
        present_sources
    )
    present_mapping = selected_mapping.loc[
        selected_mapping["present_in_expression"]
    ].copy()
    if args.expression_projection_mode == "strict_log1p_rename" and present_mapping[
        "target_gene"
    ].duplicated().any():
        raise RuntimeError("Primary strict projection unexpectedly has target collisions.")

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
    official_database = (
        source
        / "src"
        / "cellagentchat_data"
        / "mouse_lr_pair.tsv"
    )
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
    artifacts.update(lr_record["artifacts"])
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workflow": "zebrafish_cellagentchat_dual_database_shared_input",
        "formal_primary": args.expression_projection_mode == "strict_log1p_rename"
        and args.mapping_policy == "strict_one_to_one"
        and bool(source_record["pinned_source_verified"]),
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
