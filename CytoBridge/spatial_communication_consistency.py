"""Shared contracts for five-dataset spatial communication consistency.

The analysis compares directed sender-cell-type to receiver-cell-type rankings.
Native method scores are deliberately never pooled because CytoBridge exact
messages, attention gates, CellChat probabilities, COMMOT transport mass,
CellAgentChat CTPS, and NicheNet ligand activity do not share a unit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

from .graph_database import selected_feature_symbol


ANALYSIS_SEED = 20260816
TERMINAL_SAMPLE_N = 3000
TOP_FRACTION = 0.20
SPATIAL_PROXY_SAMPLING_SEEDS = (101, 202, 303)
CURRENT_LR_DATABASE_LABEL = "cytobridge_current_lr_representable_singletons"

# Frozen before the five-dataset result matrix was produced.  These thresholds
# decide main-figure inclusion only; every attempted method remains in the
# complete audit tables regardless of outcome.
MAIN_FIGURE_GATE = {
    "minimum_valid_datasets": 4,
    "minimum_positive_datasets": 4,
    "minimum_median_spearman_rho": 0.20,
    "minimum_median_top_fraction_jaccard": 0.15,
    "primary_cytobridge_view": "CytoBridge exact message",
}

FORMAL_DATASET_CONTRACTS: dict[str, dict[str, object]] = {
    "zebrafish": {
        "display_name": "Zebrafish",
        "cell_type_key": "Annotation",
        "time_key": "time_point_processed",
        "counts_layer": "counts",
        "normalization_target_sum": 1105.0,
        "terminal_time": 4.0,
        "previous_time": 3.0,
        "cellchat_species": "zebrafish",
        "database_file": "CellChatDB.ligrec.zebrafish.csv",
        "database_scope": "species-matched zebrafish CellChatDB",
    },
    "mosta": {
        "display_name": "MOSTA",
        "cell_type_key": "Annotation",
        "time_key": "time_point_processed",
        "counts_layer": "count",
        "normalization_target_sum": 10000.0,
        "terminal_time": 3.0,
        "previous_time": 2.0,
        "cellchat_species": "mouse",
        "database_file": "CellChatDB.ligrec.mouse.csv",
        "database_scope": "species-matched mouse CellChatDB",
    },
    "arista": {
        "display_name": "ARISTA",
        "cell_type_key": "Annotation",
        "time_key": "time_point_processed",
        "counts_layer": "counts",
        "normalization_target_sum": 2841.0,
        "terminal_time": 4.0,
        "previous_time": 3.0,
        "cellchat_species": "human",
        "database_file": "CellChatDB.ligrec.human.csv",
        "database_scope": "species-matched human CellChatDB",
    },
    "admouse": {
        "display_name": "AdMouse",
        "cell_type_key": "major_annotation",
        "time_key": "time_point_processed",
        "counts_layer": "counts",
        "normalization_target_sum": 10000.0,
        "terminal_time": 2.0,
        "previous_time": 1.0,
        "cellchat_species": "mouse",
        "database_file": "CellChatDB.ligrec.mouse.csv",
        "database_scope": "species-matched mouse CellChatDB; seven complete LR pairs in the 347-gene panel",
    },
    "chicken_heart": {
        "display_name": "Chicken heart",
        "cell_type_key": "celltype_prediction",
        "time_key": "time_point_processed",
        "counts_layer": "counts",
        "normalization_target_sum": 9743.5,
        "terminal_time": 3.0,
        "previous_time": 2.0,
        "cellchat_species": "human",
        "database_file": "CellChatDB.ligrec.human.csv",
        "database_scope": "human conserved-symbol proxy; not a species-complete Gallus gallus screen",
    },
}


# Cross-method species adapters are deliberately frozen separately from the
# native CytoBridge preprocessing contract.  The LR candidate universe always
# starts from each dataset's accepted ``filtered_lr_database.csv``; only the
# representation needed by CellAgentChat/NicheNet changes here.
SPATIAL_PROXY_CONTRACTS: dict[str, dict[str, object]] = {
    "zebrafish": {
        "target_species": "mouse",
        "preferred_species_tag": "zebrafish",
        "projection": "ensembl116_strict_one_to_one",
        "analysis_tier": "sensitivity",
        "interpretation": "strict zebrafish-to-mouse ortholog proxy",
    },
    "mosta": {
        "target_species": "mouse",
        "preferred_species_tag": "mouse",
        "projection": "direct_species_symbol",
        "analysis_tier": "primary",
        "interpretation": "species-matched mouse prior",
    },
    "arista": {
        "target_species": "human",
        "preferred_species_tag": "hs",
        "projection": "direct_species_symbol",
        "analysis_tier": "primary",
        "interpretation": "species-matched human prior",
    },
    "admouse": {
        "target_species": "mouse",
        "preferred_species_tag": "mouse",
        "projection": "direct_species_symbol",
        "analysis_tier": "primary",
        "interpretation": "species-matched mouse prior",
    },
    "chicken_heart": {
        "target_species": "human",
        "preferred_species_tag": "chicken",
        "projection": "direct_conserved_symbol_proxy",
        "analysis_tier": "sensitivity",
        "interpretation": "human conserved-symbol proxy for Gallus gallus",
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _strict_zebrafish_mouse_orthology(
    mapping_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the accepted Ensembl-116 high-confidence one-to-one map."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "workflow": "ensembl_compara_zebrafish_mouse_strict_one2one_export",
        "status": "complete",
        "ensembl_release": 116,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"orthology manifest requires {key}={expected!r}; "
                f"observed {manifest.get(key)!r}"
            )
    filters = manifest.get("filter")
    if not isinstance(filters, dict) or filters != {
        "orthology_type": "ortholog_one2one",
        "orthology_confidence": 1,
        "nonempty_symbols": True,
        "symbol_level_bijection_after_casefold": True,
    }:
        raise ValueError("orthology manifest does not declare the strict frozen filter")
    frame = pd.read_csv(mapping_path)
    required_columns = {
        "zebrafish_symbol",
        "mouse_symbol",
        "orthology_type",
        "orthology_confidence",
    }
    if not required_columns.issubset(frame.columns):
        raise ValueError(
            "orthology map lacks "
            f"{sorted(required_columns.difference(frame.columns))}"
        )
    frame = frame.copy()
    for column in ("zebrafish_symbol", "mouse_symbol", "orthology_type"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    confidence = pd.to_numeric(frame["orthology_confidence"], errors="raise")
    if (
        frame[["zebrafish_symbol", "mouse_symbol"]].eq("").any(axis=None)
        or not frame["orthology_type"].eq("ortholog_one2one").all()
        or not confidence.eq(1).all()
    ):
        raise ValueError("orthology CSV violates the strict one-to-one contract")
    source_key = frame["zebrafish_symbol"].str.casefold()
    target_key = frame["mouse_symbol"].str.casefold()
    if source_key.duplicated().any() or target_key.duplicated().any():
        raise ValueError("orthology CSV is not symbol-bijective after case folding")
    expected_count = int(manifest["counts"]["strict_bijective_symbol_pairs"])
    if len(frame) != expected_count:
        raise ValueError("orthology CSV row count differs from its manifest")
    observed_md5 = hashlib.md5(mapping_path.read_bytes()).hexdigest()
    if observed_md5 != str(manifest["output_md5"]["strict"]).casefold():
        raise ValueError("orthology CSV MD5 differs from its manifest")
    return frame, {
        "policy": "Ensembl 116 ortholog_one2one, confidence=1, symbol-bijective",
        "mapping": _artifact_record(mapping_path),
        "manifest": _artifact_record(manifest_path),
        "n_pairs": int(len(frame)),
    }


def _all_confidence_zebrafish_mouse_orthology(
    mapping_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the archived all-confidence one-to-one sensitivity mapping."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 2,
        "workflow": "ensembl_compara_zebrafish_mouse_one2one_bijective_export",
        "status": "complete",
        "ensembl_release": 116,
        "mapping_policy": "one2one_bijective_all_confidence",
        "analysis_tier": "sensitivity",
        "primary_claim_allowed": False,
        "mapping_file": mapping_path.name,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"orthology sensitivity manifest requires {key}={expected!r}; "
                f"observed {manifest.get(key)!r}"
            )
    filters = manifest.get("filter")
    if not isinstance(filters, dict) or filters != {
        "orthology_type": "ortholog_one2one",
        "orthology_confidence_policy": "not_filtered",
        "nonempty_symbols": True,
        "symbol_level_bijection_after_casefold": True,
    }:
        raise ValueError(
            "orthology sensitivity manifest does not declare the frozen filter"
        )
    frame = pd.read_csv(mapping_path)
    required_columns = {
        "zebrafish_symbol",
        "mouse_symbol",
        "orthology_type",
        "orthology_confidence",
    }
    if not required_columns.issubset(frame.columns):
        raise ValueError(
            "orthology sensitivity map lacks "
            f"{sorted(required_columns.difference(frame.columns))}"
        )
    frame = frame.copy()
    for column in ("zebrafish_symbol", "mouse_symbol", "orthology_type"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    confidence = pd.to_numeric(frame["orthology_confidence"], errors="raise")
    if (
        frame[["zebrafish_symbol", "mouse_symbol"]].eq("").any(axis=None)
        or not frame["orthology_type"].eq("ortholog_one2one").all()
        or not confidence.isin([0, 1]).all()
    ):
        raise ValueError("orthology sensitivity CSV violates its frozen contract")
    source_key = frame["zebrafish_symbol"].str.casefold()
    target_key = frame["mouse_symbol"].str.casefold()
    if source_key.duplicated().any() or target_key.duplicated().any():
        raise ValueError(
            "orthology sensitivity CSV is not symbol-bijective after case folding"
        )
    expected_count = int(manifest["counts"]["selected_bijective_symbol_pairs"])
    if len(frame) != expected_count:
        raise ValueError(
            "orthology sensitivity CSV row count differs from its manifest"
        )
    observed_md5 = hashlib.md5(mapping_path.read_bytes()).hexdigest()
    if observed_md5 != str(manifest["output_md5"]["mapping"]).casefold():
        raise ValueError("orthology sensitivity CSV MD5 differs from its manifest")
    return frame, {
        "policy": (
            "Ensembl 116 ortholog_one2one, confidence unfiltered, "
            "symbol-bijective sensitivity"
        ),
        "mapping": _artifact_record(mapping_path),
        "manifest": _artifact_record(manifest_path),
        "n_pairs": int(len(frame)),
        "primary_claim_allowed": False,
    }


def prepare_spatial_proxy_inputs(
    input_h5ad: str | Path,
    filtered_lr_database: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    expected_h5ad_sha256: str,
    expected_database_sha256: str,
    orthology_map: str | Path | None = None,
    orthology_manifest: str | Path | None = None,
    orthology_policy: str = "strict_confidence1",
    sampling_seeds: Sequence[int] = SPATIAL_PROXY_SAMPLING_SEEDS,
) -> dict[str, object]:
    """Adapt the accepted CytoBridge LR universe for CAG and NicheNet.

    The output expression matrix retains the accepted log-normalized values;
    columns are only subset and renamed.  Both external methods receive the
    same unique, monomeric, expression-representable subset of the dataset's
    already frozen ``filtered_lr_database.csv``.  NicheNet still uses its own
    official ligand-to-target matrix downstream because that matrix defines
    NicheNet's regulatory model rather than its LR candidate gate.
    """

    if dataset not in SPATIAL_PROXY_CONTRACTS:
        raise KeyError(f"unknown formal spatial dataset: {dataset}")
    source = Path(input_h5ad).expanduser().resolve()
    database_path = Path(filtered_lr_database).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    for label, path, expected in (
        ("input H5AD", source, expected_h5ad_sha256),
        ("filtered LR database", database_path, expected_database_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != str(expected).casefold():
            raise ValueError(
                f"{label} SHA256 mismatch: expected {expected}, observed {observed}"
            )

    contract = SPATIAL_PROXY_CONTRACTS[dataset]
    dataset_contract = FORMAL_DATASET_CONTRACTS[dataset]
    data = ad.read_h5ad(source)
    required_obs = {"ccc_cell_type", "ccc_stage", "ccc_stage_label"}
    if not required_obs.issubset(data.obs.columns):
        raise ValueError(
            f"input H5AD lacks {sorted(required_obs.difference(data.obs.columns))}"
        )
    if "spatial_aligned" not in data.obsm:
        raise ValueError("input H5AD lacks obsm['spatial_aligned']")
    spatial = np.asarray(data.obsm["spatial_aligned"], dtype=float)
    if spatial.shape != (data.n_obs, 2) or not np.isfinite(spatial).all():
        raise ValueError("spatial_aligned must be a finite N x 2 matrix")
    count_layer = str(dataset_contract["counts_layer"])
    if count_layer not in data.layers:
        raise ValueError(f"input H5AD lacks layers[{count_layer!r}]")
    expression_values = _matrix_values(data.X)
    if expression_values.size and (
        not np.isfinite(expression_values).all() or float(expression_values.min()) < 0
    ):
        raise ValueError("accepted expression must be finite and nonnegative")

    projection = str(contract["projection"])
    orthology_record: dict[str, object] = {
        "provided": False,
        "policy": "not applicable",
    }
    orthology_lookup: dict[str, str] = {}
    if projection == "ensembl116_strict_one_to_one":
        if orthology_map is None or orthology_manifest is None:
            raise ValueError("zebrafish proxy requires orthology map and manifest")
        mapping_path = Path(orthology_map).expanduser().resolve()
        mapping_manifest_path = Path(orthology_manifest).expanduser().resolve()
        if orthology_policy == "strict_confidence1":
            frame, verified = _strict_zebrafish_mouse_orthology(
                mapping_path, mapping_manifest_path
            )
            resolved_projection = "ensembl116_strict_confidence1_one_to_one"
        elif orthology_policy == "one2one_bijective_all_confidence":
            frame, verified = _all_confidence_zebrafish_mouse_orthology(
                mapping_path, mapping_manifest_path
            )
            resolved_projection = "ensembl116_all_confidence_one_to_one_sensitivity"
        else:
            raise ValueError(
                f"unsupported zebrafish orthology policy: {orthology_policy}"
            )
        orthology_lookup = dict(
            zip(
                frame["zebrafish_symbol"].str.casefold(),
                frame["mouse_symbol"].astype(str),
                strict=True,
            )
        )
        orthology_record = {"provided": True, **verified}
    elif orthology_map is not None or orthology_manifest is not None:
        raise ValueError("orthology inputs are only valid for the zebrafish proxy")
    else:
        if orthology_policy != "strict_confidence1":
            raise ValueError("non-zebrafish datasets do not accept an orthology policy")
        resolved_projection = projection

    preferred_tag = str(contract["preferred_species_tag"])
    feature_rows: list[dict[str, object]] = []
    for index, raw_name in enumerate(data.var_names.astype(str)):
        source_symbol = selected_feature_symbol(
            raw_name, preferred_species_tag=preferred_tag
        )
        target_symbol: str | None
        if source_symbol is None:
            target_symbol = None
        elif projection == "ensembl116_strict_one_to_one":
            target_symbol = orthology_lookup.get(source_symbol.casefold())
        elif str(contract["target_species"]) == "human":
            target_symbol = source_symbol.upper()
        else:
            target_symbol = source_symbol
        feature_rows.append(
            {
                "source_index": int(index),
                "source_feature": raw_name,
                "source_symbol": source_symbol or "",
                "target_symbol": target_symbol or "",
            }
        )
    feature_map = pd.DataFrame(feature_rows)
    feature_map["exclusion_reason"] = ""
    feature_map.loc[
        feature_map.source_symbol.eq(""), "exclusion_reason"
    ] = "no_selected_source_symbol"
    feature_map.loc[
        feature_map.target_symbol.eq("") & feature_map.exclusion_reason.eq(""),
        "exclusion_reason",
    ] = "no_target_species_mapping"
    eligible = feature_map.exclusion_reason.eq("")
    source_key = feature_map.source_symbol.str.casefold()
    duplicated_source = eligible & source_key.duplicated(keep=False)
    feature_map.loc[
        duplicated_source, "exclusion_reason"
    ] = "ambiguous_source_symbol_after_casefold"
    eligible = feature_map.exclusion_reason.eq("")
    target_key = feature_map.target_symbol.str.casefold()
    duplicated_target = eligible & target_key.duplicated(keep=False)
    feature_map.loc[
        duplicated_target, "exclusion_reason"
    ] = "ambiguous_target_symbol_after_casefold"
    selected = feature_map.loc[feature_map.exclusion_reason.eq("")].copy()
    if selected.empty:
        raise ValueError("no expression features survive the species projection")
    selected = selected.sort_values(
        ["target_symbol", "source_index"], kind="mergesort"
    ).reset_index(drop=True)
    if selected.target_symbol.str.casefold().duplicated().any():
        raise RuntimeError("projected target gene symbols are not unique")

    indices = selected.source_index.to_numpy(dtype=int)
    mapped = data[:, indices].copy()
    mapped.var = pd.DataFrame(
        {
            "source_feature": selected.source_feature.to_numpy(),
            "source_symbol": selected.source_symbol.to_numpy(),
        },
        index=pd.Index(selected.target_symbol.astype(str), name="target_gene"),
    )
    mapped.var_names = selected.target_symbol.astype(str).to_numpy()
    mapped.layers["counts"] = mapped.layers[count_layer].copy()
    mapped.uns["spatial_proxy_projection"] = {
        "schema_version": 1,
        "dataset": dataset,
        "target_species": str(contract["target_species"]),
        "projection": resolved_projection,
        "analysis_tier": str(contract["analysis_tier"]),
        "values_changed": False,
        "current_cytobridge_lr_database_required": True,
    }

    database = pd.read_csv(database_path)
    if not {"database_row", "ligand", "receptor"}.issubset(database.columns):
        raise ValueError("filtered LR database lacks database_row/ligand/receptor")
    database = database.copy()
    database["ligand"] = database.ligand.fillna("").astype(str).str.strip()
    database["receptor"] = database.receptor.fillna("").astype(str).str.strip()
    source_to_target = dict(
        zip(
            selected.source_symbol.str.casefold(),
            selected.target_symbol.astype(str),
            strict=True,
        )
    )
    mapped_ligands: list[str] = []
    mapped_receptors: list[str] = []
    reasons: list[str] = []
    for ligand, receptor in zip(database.ligand, database.receptor, strict=True):
        row_reasons: list[str] = []
        if not ligand:
            row_reasons.append("missing_ligand")
        if not receptor:
            row_reasons.append("missing_receptor")
        if "_" in ligand:
            row_reasons.append("ligand_complex_not_gene_level_representable")
        if "_" in receptor:
            row_reasons.append("receptor_complex_not_gene_level_representable")
        mapped_ligand = source_to_target.get(ligand.casefold(), "")
        mapped_receptor = source_to_target.get(receptor.casefold(), "")
        if ligand and "_" not in ligand and not mapped_ligand:
            row_reasons.append("ligand_not_in_projected_expression")
        if receptor and "_" not in receptor and not mapped_receptor:
            row_reasons.append("receptor_not_in_projected_expression")
        if "_" in mapped_ligand:
            row_reasons.append("mapped_ligand_contains_underscore")
        if "_" in mapped_receptor:
            row_reasons.append("mapped_receptor_contains_underscore")
        mapped_ligands.append(mapped_ligand)
        mapped_receptors.append(mapped_receptor)
        reasons.append(";".join(row_reasons))
    crosswalk = database.copy()
    crosswalk["mapped_ligand"] = mapped_ligands
    crosswalk["mapped_receptor"] = mapped_receptors
    crosswalk["exclusion_reason"] = reasons
    represented = crosswalk.loc[crosswalk.exclusion_reason.eq("")].copy()
    pairs = (
        represented[["mapped_ligand", "mapped_receptor"]]
        .drop_duplicates()
        .sort_values(["mapped_ligand", "mapped_receptor"], kind="mergesort")
        .reset_index(drop=True)
    )
    if pairs.empty:
        raise ValueError("no current-database LR pair is representable by both methods")

    output.mkdir(parents=True)
    mapped_path = output / "projected_terminal_previous.h5ad"
    mapped.write_h5ad(mapped_path, compression="gzip")
    feature_path = output / "feature_projection_crosswalk.csv.gz"
    feature_map.to_csv(feature_path, index=False, compression="gzip")
    lr_crosswalk_path = output / "current_lr_projection_crosswalk.csv"
    crosswalk.to_csv(lr_crosswalk_path, index=False)
    cag_path = output / "cellagentchat_current_lr_pairs.tsv"
    cag = pairs.rename(
        columns={
            "mapped_ligand": "ligand_gene_symbol",
            "mapped_receptor": "receptor_gene_symbol",
        }
    )
    cag.insert(
        0,
        "lr_pair",
        cag.ligand_gene_symbol + "_" + cag.receptor_gene_symbol,
    )
    cag.to_csv(cag_path, sep="\t", index=False)
    nichenet_path = output / "nichenet_current_lr_network.csv"
    pairs.rename(columns={"mapped_ligand": "from", "mapped_receptor": "to"}).to_csv(
        nichenet_path, index=False
    )

    plan_rows: list[pd.DataFrame] = []
    base = pd.DataFrame(
        {
            "stage": pd.to_numeric(mapped.obs["ccc_stage"], errors="raise").to_numpy(
                float
            ),
            "stage_label": mapped.obs["ccc_stage_label"].astype(str).to_numpy(),
            "cell_type": mapped.obs["ccc_cell_type"].astype(str).to_numpy(),
            "obs_name": mapped.obs_names.astype(str),
            "original_index": np.arange(mapped.n_obs, dtype=int),
        }
    )
    stage_labels = base.groupby("stage", sort=True).stage_label.nunique()
    if not stage_labels.eq(1).all():
        raise ValueError("each numeric stage must have exactly one stage label")
    base["within_type_sample_order"] = base.groupby(
        ["stage", "cell_type"], sort=False
    ).cumcount()
    base["n_type_cells_available"] = base.groupby(
        ["stage", "cell_type"], sort=False
    ).obs_name.transform("size")
    base["n_type_cells_sampled"] = base["n_type_cells_available"]
    for seed in sampling_seeds:
        if int(seed) < 0:
            raise ValueError("sampling seeds must be nonnegative")
        plan_rows.append(base.assign(sampling_seed=int(seed)))
    plan = pd.concat(plan_rows, ignore_index=True)
    plan = plan[
        [
            "sampling_seed",
            "stage",
            "stage_label",
            "cell_type",
            "obs_name",
            "original_index",
            "within_type_sample_order",
            "n_type_cells_available",
            "n_type_cells_sampled",
        ]
    ].sort_values(
        ["sampling_seed", "stage", "cell_type", "original_index"],
        kind="mergesort",
    )
    plan_path = output / "shared_sampled_cells.csv.gz"
    plan.to_csv(plan_path, index=False, compression="gzip")

    artifacts = {
        "mapped_expression": _artifact_record(mapped_path),
        "shared_sampled_cells": _artifact_record(plan_path),
        "feature_projection_crosswalk": _artifact_record(feature_path),
        "current_lr_projection_crosswalk": _artifact_record(lr_crosswalk_path),
        "cellagentchat_current_lr_pairs": _artifact_record(cag_path),
        "nichenet_current_lr_network": _artifact_record(nichenet_path),
    }
    analysis_tier = str(contract["analysis_tier"])
    manifest: dict[str, object] = {
        "schema_version": 1,
        "workflow": "five_dataset_spatial_communication_shared_database_proxy_input",
        "dataset": dataset,
        "formal_primary": analysis_tier == "primary",
        "primary_claim_allowed": analysis_tier == "primary",
        "orthology_policy": (
            str(orthology_record["policy"])
            if orthology_record["provided"]
            else "direct accepted feature-symbol projection"
        ),
        "orthology_analysis_tier": analysis_tier,
        "analysis_tier": analysis_tier,
        "input": {
            "expression_h5ad": _artifact_record(source),
            "filtered_current_cytobridge_lr_database": _artifact_record(database_path),
        },
        "projection": {
            **contract,
            "projection": resolved_projection,
            "n_source_features": int(data.n_vars),
            "n_projected_features": int(mapped.n_vars),
            "expression_values_changed": False,
            "selection_only_then_rename": True,
        },
        "orthology": orthology_record,
        "keys": {
            "cell_type": "ccc_cell_type",
            "time": "ccc_stage",
            "time_label": "ccc_stage_label",
            "spatial": "spatial_aligned",
        },
        "sampling": {
            "seeds": [int(value) for value in sampling_seeds],
            "all_accepted_shared_sample_cells_used": True,
            "n_plan_rows": int(len(plan)),
        },
        "target_species_prior": str(contract["target_species"]),
        "cross_species_interpretation": str(contract["interpretation"]),
        "lr_database_contract": {
            "authoritative_candidate_source": "current accepted CytoBridge filtered LR database",
            "complex_policy": "exclude without decomposition for both gene-level methods",
            "n_current_database_rows": int(len(database)),
            "n_representable_source_rows": int(len(represented)),
            "n_unique_representable_pairs": int(len(pairs)),
            "same_pair_universe_for_cellagentchat_and_nichenet": True,
            "nichenet_target_prior": (
                f"official NicheNet v2 {contract['target_species']} ligand-target matrix; "
                "not replaced by the LR candidate database"
            ),
        },
        "lr_databases": {
            CURRENT_LR_DATABASE_LABEL: artifacts["cellagentchat_current_lr_pairs"]
        },
        "artifacts": artifacts,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _matrix_values(matrix: object) -> np.ndarray:
    return np.asarray(matrix.data if sparse.issparse(matrix) else matrix).ravel()


def stratified_sample_indices(
    labels: Iterable[object], *, total: int, seed: int
) -> np.ndarray:
    """Select a deterministic near-proportional sample retaining every type."""

    values = np.asarray([str(value) for value in labels], dtype=object)
    if total <= 0:
        raise ValueError("total must be positive")
    if len(values) <= total:
        return np.arange(len(values), dtype=np.int64)
    groups, counts = np.unique(values, return_counts=True)
    if total < len(groups):
        raise ValueError("sample size is smaller than the terminal cell-type universe")
    ideal = total * counts.astype(float) / len(values)
    allocation = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while int(allocation.sum()) < total:
        candidates = np.flatnonzero(allocation < counts)
        chosen = candidates[np.argmax(ideal[candidates] - allocation[candidates])]
        allocation[chosen] += 1
    while int(allocation.sum()) > total:
        candidates = np.flatnonzero(allocation > 1)
        chosen = candidates[np.argmax(allocation[candidates] - ideal[candidates])]
        allocation[chosen] -= 1
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for group, amount in zip(groups, allocation, strict=True):
        candidates = np.flatnonzero(values == group)
        selected.append(np.sort(rng.choice(candidates, int(amount), replace=False)))
    return np.sort(np.concatenate(selected)).astype(np.int64, copy=False)


def _verify_reconstructed_x(
    data: ad.AnnData,
    rows: np.ndarray,
    *,
    counts_layer: str,
    target_sum: float,
    tolerance: float,
) -> float:
    counts = data.layers[counts_layer][rows]
    counts_csr = sparse.csr_matrix(counts, dtype=np.float64)
    raw_values = _matrix_values(counts_csr)
    if raw_values.size and (
        not np.isfinite(raw_values).all()
        or float(raw_values.min()) < 0
        or float(np.max(np.abs(raw_values - np.rint(raw_values)))) > 1e-5
    ):
        raise ValueError(
            "declared count layer is not finite nonnegative integer-like data"
        )
    library = np.asarray(counts_csr.sum(axis=1)).ravel()
    if np.any(~np.isfinite(library)) or np.any(library <= 0):
        raise ValueError("sample contains a nonpositive or nonfinite raw library size")
    reconstructed = counts_csr.multiply((float(target_sum) / library)[:, None]).tocsr()
    reconstructed.data = np.log1p(reconstructed.data)
    observed = sparse.csr_matrix(data.X[rows], dtype=np.float64)
    residual = (reconstructed - observed).tocsr()
    maximum = float(np.max(np.abs(residual.data))) if residual.nnz else 0.0
    if maximum > float(tolerance):
        raise ValueError(
            "accepted X does not match counts-derived normalize_total+log1p: "
            f"max residual={maximum:.6g}, tolerance={tolerance:.6g}"
        )
    return maximum


def prepare_shared_samples(
    input_h5ad: str | Path,
    output_dir: str | Path,
    *,
    dataset: str,
    expected_h5ad_sha256: str,
    sample_n: int = TERMINAL_SAMPLE_N,
    seed: int = ANALYSIS_SEED,
    source_x_tolerance: float = 1e-5,
) -> dict[str, object]:
    """Freeze terminal and preceding-stage cells shared by all methods.

    The terminal roster is identical in both emitted H5ADs.  The two-stage
    H5AD adds a separately stratified preceding-stage roster for NicheNet's
    receiver-response calculation; terminal pair-score methods consume only
    ``terminal_sample.h5ad``.
    """

    if dataset not in FORMAL_DATASET_CONTRACTS:
        raise KeyError(f"unknown formal spatial dataset: {dataset}")
    contract = FORMAL_DATASET_CONTRACTS[dataset]
    source = Path(input_h5ad).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    if sha256_file(source) != str(expected_h5ad_sha256).lower():
        raise ValueError("input H5AD SHA256 differs from the accepted binding")
    data = ad.read_h5ad(source)
    label_key = str(contract["cell_type_key"])
    time_key = str(contract["time_key"])
    counts_layer = str(contract["counts_layer"])
    for key in (label_key, time_key):
        if key not in data.obs:
            raise KeyError(f"accepted H5AD lacks obs[{key!r}]")
    if counts_layer not in data.layers:
        raise KeyError(f"accepted H5AD lacks layers[{counts_layer!r}]")
    for key in ("spatial_aligned", "X_latent"):
        if key not in data.obsm:
            raise KeyError(f"accepted H5AD lacks obsm[{key!r}]")
    times = pd.to_numeric(data.obs[time_key], errors="raise").to_numpy(float)
    terminal_time = float(contract["terminal_time"])
    previous_time = float(contract["previous_time"])
    labels = data.obs[label_key].astype(str).to_numpy()
    terminal_all = np.flatnonzero(np.isclose(times, terminal_time, rtol=0, atol=1e-8))
    previous_all = np.flatnonzero(np.isclose(times, previous_time, rtol=0, atol=1e-8))
    if not len(terminal_all) or not len(previous_all):
        raise ValueError("accepted H5AD lacks the declared terminal or previous stage")
    terminal_local = stratified_sample_indices(
        labels[terminal_all], total=int(sample_n), seed=int(seed)
    )
    previous_local = stratified_sample_indices(
        labels[previous_all], total=int(sample_n), seed=int(seed) + 1
    )
    terminal_rows = terminal_all[terminal_local]
    previous_rows = previous_all[previous_local]
    if len(set(data.obs_names[terminal_rows].astype(str))) != len(terminal_rows):
        raise ValueError("terminal sample has duplicate observation names")
    checked_rows = np.unique(np.concatenate((terminal_rows, previous_rows)))
    max_residual = _verify_reconstructed_x(
        data,
        checked_rows,
        counts_layer=counts_layer,
        target_sum=float(contract["normalization_target_sum"]),
        tolerance=float(source_x_tolerance),
    )
    output.mkdir(parents=True)

    terminal = data[terminal_rows].copy()
    terminal.obs["ccc_cell_type"] = terminal.obs[label_key].astype(str)
    terminal.obs["ccc_stage"] = terminal.obs[time_key].astype(float)
    terminal.obs["ccc_stage_label"] = f"terminal_{terminal_time:g}"
    terminal.obs["time_label"] = terminal.obs["ccc_stage_label"]
    terminal_path = output / "terminal_sample.h5ad"
    terminal.write_h5ad(terminal_path, compression="gzip")

    two_stage_rows = np.concatenate((previous_rows, terminal_rows))
    two_stage = data[two_stage_rows].copy()
    two_stage.obs["ccc_cell_type"] = two_stage.obs[label_key].astype(str)
    two_stage.obs["ccc_stage"] = two_stage.obs[time_key].astype(float)
    two_stage.obs["ccc_stage_label"] = np.where(
        np.isclose(two_stage.obs["ccc_stage"].astype(float), terminal_time),
        f"terminal_{terminal_time:g}",
        f"previous_{previous_time:g}",
    )
    two_stage.obs["time_label"] = two_stage.obs["ccc_stage_label"]
    two_stage_path = output / "terminal_previous_sample.h5ad"
    two_stage.write_h5ad(two_stage_path, compression="gzip")

    roster = pd.DataFrame(
        {
            "dataset": dataset,
            "stage_role": ["terminal"] * len(terminal_rows)
            + ["previous"] * len(previous_rows),
            "source_row": np.concatenate((terminal_rows, previous_rows)),
            "obs_name": np.concatenate(
                (
                    data.obs_names[terminal_rows].astype(str),
                    data.obs_names[previous_rows].astype(str),
                )
            ),
            "cell_type": np.concatenate((labels[terminal_rows], labels[previous_rows])),
            "stage": np.concatenate((times[terminal_rows], times[previous_rows])),
        }
    )
    roster_path = output / "sample_roster.csv"
    roster.to_csv(roster_path, index=False)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "workflow": "five_dataset_spatial_communication_shared_sample",
        "dataset": dataset,
        "contract": contract,
        "selection": {
            "seed": int(seed),
            "maximum_cells_per_stage": int(sample_n),
            "rule": "deterministic near-proportional cell-type-stratified sample retaining every observed type",
            "terminal_cells_available": int(len(terminal_all)),
            "terminal_cells_selected": int(len(terminal_rows)),
            "previous_cells_available": int(len(previous_all)),
            "previous_cells_selected": int(len(previous_rows)),
            "terminal_cell_types": sorted(set(labels[terminal_rows])),
        },
        "expression": {
            "source": f"layers[{counts_layer!r}]",
            "transform": "normalize_total over all genes, then log1p exactly once",
            "target_sum": float(contract["normalization_target_sum"]),
            "accepted_x_reconstruction_max_abs_residual": max_residual,
            "accepted_x_reconstruction_tolerance": float(source_x_tolerance),
        },
        "source_h5ad": {
            "path": str(source),
            "sha256": str(expected_h5ad_sha256).lower(),
            "size_bytes": int(source.stat().st_size),
        },
        "artifacts": {},
    }
    for path in (terminal_path, two_stage_path, roster_path):
        manifest["artifacts"][path.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        }
    _write_json(output / "manifest.json", manifest)
    return manifest


def rank_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("rank input contains missing or nonfinite scores")
    if len(numeric) <= 1:
        return pd.Series(np.ones(len(numeric)), index=numeric.index, dtype=float)
    return numeric.rank(method="average", pct=True)


def _positive_top_keys(
    frame: pd.DataFrame, *, score_column: str, top_fraction: float
) -> set[tuple[str, str]]:
    positive = frame.loc[pd.to_numeric(frame[score_column], errors="raise") > 0].copy()
    if positive.empty:
        return set()
    requested = max(1, int(math.ceil(len(frame) * float(top_fraction))))
    k = min(requested, len(positive))
    threshold = positive[score_column].nlargest(k).min()
    selected = positive.loc[positive[score_column] >= threshold]
    return set(
        zip(selected.sender_type.astype(str), selected.receiver_type.astype(str))
    )


def pairwise_cytobridge_metrics(
    long_scores: pd.DataFrame,
    *,
    cytobridge_views: Sequence[str] = (
        "CytoBridge exact message",
        "CytoBridge attention",
    ),
    top_fraction: float = TOP_FRACTION,
) -> pd.DataFrame:
    """Compare each external method to each CytoBridge view on shared keys."""

    required = {
        "dataset",
        "sender_type",
        "receiver_type",
        "method",
        "score",
        "available",
    }
    if not required.issubset(long_scores.columns):
        raise ValueError(
            f"score table lacks {sorted(required.difference(long_scores.columns))}"
        )
    external = sorted(set(long_scores.method.astype(str)).difference(cytobridge_views))
    rows: list[dict[str, object]] = []
    keys = ["sender_type", "receiver_type"]
    for dataset, dataset_table in long_scores.groupby("dataset", sort=True):
        for view in cytobridge_views:
            left = dataset_table.loc[
                dataset_table.method.astype(str).eq(view)
                & dataset_table.available.astype(bool),
                keys + ["score"],
            ].rename(columns={"score": "left_score"})
            if left.duplicated(keys).any():
                raise ValueError(f"duplicate {view} directed keys for {dataset}")
            for method in external:
                right = dataset_table.loc[
                    dataset_table.method.astype(str).eq(method)
                    & dataset_table.available.astype(bool),
                    keys + ["score"],
                ].rename(columns={"score": "right_score"})
                if right.duplicated(keys).any():
                    raise ValueError(f"duplicate {method} directed keys for {dataset}")
                merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
                valid = (
                    len(merged) >= 4
                    and merged.left_score.nunique() > 1
                    and merged.right_score.nunique() > 1
                )
                rho = (
                    float(
                        stats.spearmanr(merged.left_score, merged.right_score).statistic
                    )
                    if valid
                    else np.nan
                )
                left_top = _positive_top_keys(
                    merged.rename(columns={"left_score": "score"}),
                    score_column="score",
                    top_fraction=top_fraction,
                )
                right_top = _positive_top_keys(
                    merged.rename(columns={"right_score": "score"}),
                    score_column="score",
                    top_fraction=top_fraction,
                )
                union = left_top | right_top
                jaccard = len(left_top & right_top) / len(union) if union else np.nan
                rows.append(
                    {
                        "dataset": str(dataset),
                        "cytobridge_view": view,
                        "external_method": method,
                        "n_shared_directed_pairs": int(len(merged)),
                        "spearman_rho": rho,
                        "top_fraction": float(top_fraction),
                        "top_left_n": int(len(left_top)),
                        "top_right_n": int(len(right_top)),
                        "top_intersection_n": int(len(left_top & right_top)),
                        "top_jaccard": float(jaccard)
                        if np.isfinite(jaccard)
                        else np.nan,
                        "metric_available": bool(valid),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_main_figure_gate(
    metrics: pd.DataFrame,
    *,
    gate: Mapping[str, object] = MAIN_FIGURE_GATE,
) -> pd.DataFrame:
    """Apply the frozen cross-dataset inclusion gate without hiding failures."""

    primary = str(gate["primary_cytobridge_view"])
    selected = metrics.loc[metrics.cytobridge_view.astype(str).eq(primary)].copy()
    rows: list[dict[str, object]] = []
    for method, table in selected.groupby("external_method", sort=True):
        valid = table.loc[table.metric_available.astype(bool)].copy()
        rho = pd.to_numeric(valid.spearman_rho, errors="coerce").dropna()
        jaccard = pd.to_numeric(valid.top_jaccard, errors="coerce").dropna()
        n_valid = int(len(rho))
        n_positive = int((rho > 0).sum())
        median_rho = float(rho.median()) if len(rho) else np.nan
        median_jaccard = float(jaccard.median()) if len(jaccard) else np.nan
        checks = {
            "valid_dataset_count": n_valid >= int(gate["minimum_valid_datasets"]),
            "positive_dataset_count": n_positive
            >= int(gate["minimum_positive_datasets"]),
            "median_spearman": np.isfinite(median_rho)
            and median_rho >= float(gate["minimum_median_spearman_rho"]),
            "median_top_jaccard": np.isfinite(median_jaccard)
            and median_jaccard >= float(gate["minimum_median_top_fraction_jaccard"]),
        }
        rows.append(
            {
                "external_method": method,
                "n_valid_datasets": n_valid,
                "n_positive_spearman_datasets": n_positive,
                "median_spearman_rho": median_rho,
                "median_top_jaccard": median_jaccard,
                **{f"passes_{key}": bool(value) for key, value in checks.items()},
                "include_in_main_figure": bool(all(checks.values())),
                "decision_rule_frozen_before_results": True,
            }
        )
    return pd.DataFrame(rows)
