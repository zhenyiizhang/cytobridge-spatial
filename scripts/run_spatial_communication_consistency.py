#!/usr/bin/env python3
"""Prepare, aggregate, and plot five-dataset spatial CCC consistency evidence.

External methods are executed by their pinned adapters. This orchestrator
freezes the shared sample, combines only manifest-bound outputs, applies the
predeclared main-figure gate, and produces an A4 submission figure without
silently hiding weak or unavailable methods from the audit tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import anndata as ad

from CytoBridge.spatial_communication_consistency import (
    FORMAL_DATASET_CONTRACTS,
    MAIN_FIGURE_GATE,
    SPATIAL_PROXY_SAMPLING_SEEDS,
    TOP_FRACTION,
    evaluate_main_figure_gate,
    pairwise_cytobridge_metrics,
    prepare_shared_samples,
    prepare_spatial_proxy_inputs,
    rank_percentile,
    model_linked_spatial_colocalization,
    select_global_model_linked_lr_example,
    select_model_linked_lr_example,
    sha256_file,
)


METHOD_COLORS = {
    "COMMOT": "#2A9D8F",
    "CellChat": "#4C78A8",
    "CellAgentChat": "#E9C46A",
    "NicheNet": "#E76F51",
}
DATASET_COLORS = {
    "zebrafish": "#5B4B8A",
    "mosta": "#4C78A8",
    "arista": "#2A9D8F",
    "admouse": "#E9C46A",
    "chicken_heart": "#E76F51",
}
BIOLOGICAL_PROGRAMS = {
    "zebrafish": "Neuroepithelial patterning and ECM-guided forebrain development",
    "mosta": "Mesenchymal signaling linked to chondrogenic maturation",
    "arista": "Regenerative neuroglial niche and stress-responsive remodeling",
    "admouse": "Neuron–astrocyte signaling and reactive structural programs",
    "chicken_heart": "Valve ECM remodeling and endothelial–mesenchymal maturation",
}
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_BIOLOGY_IMPLEMENTATION_FILES = (
    "CytoBridge/spatial_communication_consistency.py",
    "scripts/run_spatial_communication_consistency.py",
    "scripts/reviewer_zebrafish_ccc/run_selected_commot_flows.py",
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _model_biology_implementation() -> dict[str, object]:
    files = {
        relative: _artifact(REPO_ROOT / relative)
        for relative in MODEL_BIOLOGY_IMPLEMENTATION_FILES
    }
    digest = hashlib.sha256()
    for relative in sorted(files):
        digest.update(f"{relative}:{files[relative]['sha256']}\n".encode())
    return {"files": files, "aggregate_sha256": digest.hexdigest()}


def _read_table(path: str | Path, *, label: str) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {label}: {resolved}")
    frame = pd.read_csv(resolved)
    if frame.empty:
        raise ValueError(f"{label} is empty: {resolved}")
    return frame


def _complete(
    frame: pd.DataFrame,
    *,
    method: str,
    dataset: str,
    types: Iterable[str],
    score_column: str,
    available: bool = True,
) -> pd.DataFrame:
    keys = ["sender_type", "receiver_type"]
    required = {*keys, score_column}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{dataset}/{method} lacks {sorted(required.difference(frame.columns))}"
        )
    local = frame[keys + [score_column]].copy()
    if local.duplicated(keys).any():
        raise ValueError(f"{dataset}/{method} has duplicate directed type pairs")
    grid = pd.MultiIndex.from_product(
        [sorted(set(types)), sorted(set(types))], names=keys
    ).to_frame(index=False)
    merged = grid.merge(local, on=keys, how="left", validate="one_to_one")
    merged[score_column] = pd.to_numeric(merged[score_column], errors="raise").fillna(
        0.0
    )
    if not np.isfinite(merged[score_column].to_numpy(float)).all():
        raise ValueError(f"{dataset}/{method} has nonfinite scores")
    merged = merged.rename(columns={score_column: "score"})
    merged.insert(0, "dataset", dataset)
    merged["method"] = method
    merged["available"] = bool(available)
    merged["rank_percentile"] = rank_percentile(merged["score"])
    return merged


def _terminal_types(sample_manifest: Path) -> list[str]:
    manifest = json.loads(sample_manifest.read_text(encoding="utf-8"))
    if manifest.get("workflow") != "five_dataset_spatial_communication_shared_sample":
        raise ValueError(f"unexpected shared-sample manifest: {sample_manifest}")
    return [str(value) for value in manifest["selection"]["terminal_cell_types"]]


def _load_cytobridge(
    spec: dict[str, object], dataset: str, types: list[str]
) -> list[pd.DataFrame]:
    frame = _read_table(spec["cytobridge_type_pair_csv"], label=f"{dataset} CytoBridge")
    terminal = float(FORMAL_DATASET_CONTRACTS[dataset]["terminal_time"])
    frame = frame.loc[np.isclose(pd.to_numeric(frame.stage), terminal)].copy()
    return [
        _complete(
            frame,
            method="CytoBridge exact message",
            dataset=dataset,
            types=types,
            score_column="D_AB_joint_mean",
        ),
        _complete(
            frame,
            method="CytoBridge attention",
            dataset=dataset,
            types=types,
            score_column="G_AB_attention_mean_mean",
        ),
    ]


def _load_external(
    spec: dict[str, object], dataset: str, types: list[str]
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    tables: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    definitions = {
        "COMMOT": ("commot_type_pair_csv", "abundance_controlled_distinct_cell_score"),
        "CellChat": ("cellchat_type_pair_csv", "abundance_controlled_score"),
        "CellAgentChat": ("cellagentchat_type_pair_csv", "cellagentchat_native_ctps"),
        "NicheNet": ("nichenet_type_pair_csv", "nichenet_support_score"),
    }
    for method, (path_key, score_column) in definitions.items():
        if method == "CellAgentChat":
            score_column = str(spec.get("cellagentchat_score_column", score_column))
        method_status = dict(spec.get("method_status", {})).get(method, "complete")
        method_reason = str(dict(spec.get("method_reason", {})).get(method, ""))
        path_value = spec.get(path_key)
        if method_status != "complete" or not path_value:
            status_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "status": str(method_status),
                    "included_in_score_table": False,
                    "reason": method_reason,
                }
            )
            continue
        frame = _read_table(path_value, label=f"{dataset} {method}")
        if "stage" in frame:
            terminal = float(FORMAL_DATASET_CONTRACTS[dataset]["terminal_time"])
            frame = frame.loc[np.isclose(pd.to_numeric(frame.stage), terminal)].copy()
        tables.append(
            _complete(
                frame,
                method=method,
                dataset=dataset,
                types=types,
                score_column=score_column,
            )
        )
        status_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "status": "complete",
                "included_in_score_table": True,
                "reason": method_reason,
            }
        )
    return tables, status_rows


def _select_pairs(scores: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    included = (
        decisions.loc[decisions.include_in_main_figure, "external_method"]
        .astype(str)
        .tolist()
    )
    methods = ["CytoBridge exact message", *included]
    local = scores.loc[
        scores.method.isin(methods) & scores.available.astype(bool)
    ].copy()
    rows: list[pd.Series] = []
    for dataset, table in local.groupby("dataset", sort=True):
        methods_here = [
            method for method in methods if method in set(table.method.astype(str))
        ]
        wide = table.pivot(
            index=["sender_type", "receiver_type"],
            columns="method",
            values="rank_percentile",
        )
        wide = wide.loc[:, methods_here].dropna()
        if "CytoBridge exact message" not in wide or wide.empty:
            continue
        wide = wide.loc[
            wide.index.get_level_values(0) != wide.index.get_level_values(1)
        ]
        if wide.empty:
            continue
        included_here = [method for method in included if method in methods_here]
        wide["consensus_rank"] = wide[methods_here].mean(axis=1)
        wide["external_top_support"] = (
            (wide[included_here] >= 1 - TOP_FRACTION).sum(axis=1)
            if included_here
            else 0
        )
        eligible = wide.loc[wide["CytoBridge exact message"] >= 1 - TOP_FRACTION]
        if eligible.empty:
            eligible = wide
        best = eligible.sort_values(
            ["external_top_support", "consensus_rank"],
            ascending=False,
            kind="mergesort",
        ).iloc[0]
        record = best.copy()
        record["dataset"] = dataset
        record["sender_type"] = best.name[0]
        record["receiver_type"] = best.name[1]
        rows.append(record)
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_nichenet(args: argparse.Namespace) -> None:
    source = Path(args.nichenet_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    required = {
        "prepare_manifest": source / "manifest.json",
        "candidates": source / "sender_receiver_lr_candidates.csv",
        "activities": source / "official" / "ligand_activities.csv",
        "targets": source / "official" / "ligand_target_links.csv",
        "r_session": source / "official" / "R_sessionInfo.txt",
    }
    receiver_status_path = source / "official" / "receiver_status.csv"
    if receiver_status_path.is_file():
        required["receiver_status"] = receiver_status_path
        required["receiver_gene_sets"] = source / "receiver_gene_sets.csv"
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing NicheNet {label}: {path}")
    candidates = pd.read_csv(required["candidates"])
    activities = pd.read_csv(required["activities"])
    targets = pd.read_csv(required["targets"])
    dataset_sets: dict[str, set[str]] = {}
    for label, table in (
        ("candidates", candidates),
        ("activities", activities),
        ("targets", targets),
    ):
        if "dataset" not in table:
            raise ValueError(f"NicheNet {label} table lacks dataset")
        dataset_sets[label] = set(table["dataset"].astype(str))
        if len(dataset_sets[label]) != 1:
            raise ValueError(f"NicheNet {label} table must contain one dataset")
    if len({next(iter(values)) for values in dataset_sets.values()}) != 1:
        raise ValueError("NicheNet candidates, activities, and targets datasets differ")
    receiver_status: pd.DataFrame | None = None
    receiver_status_summary: dict[str, object] = {
        "present": False,
        "allowed_statuses": ["complete", "skipped_no_potential_ligands"],
    }
    if "receiver_status" in required:
        receiver_status = pd.read_csv(required["receiver_status"])
        receiver_gene_sets = pd.read_csv(required["receiver_gene_sets"])
        required_status_columns = {
            "dataset",
            "receiver",
            "status",
            "reason",
            "n_response_genes",
            "n_background_genes",
            "n_potential_ligands",
        }
        missing_status_columns = sorted(
            required_status_columns.difference(receiver_status.columns)
        )
        if missing_status_columns:
            raise ValueError(
                "NicheNet receiver status is missing columns: "
                + ", ".join(missing_status_columns)
            )
        if receiver_status.empty:
            raise ValueError("NicheNet receiver status must not be empty")
        required_gene_set_columns = {"dataset", "receiver", "gene", "is_response"}
        missing_gene_set_columns = sorted(
            required_gene_set_columns.difference(receiver_gene_sets.columns)
        )
        if missing_gene_set_columns:
            raise ValueError(
                "NicheNet receiver gene sets are missing columns: "
                + ", ".join(missing_gene_set_columns)
            )
        if receiver_gene_sets.empty:
            raise ValueError("NicheNet receiver gene sets must not be empty")
        if receiver_status["receiver"].astype(str).duplicated().any():
            raise ValueError("NicheNet receiver status must contain unique receivers")
        allowed_statuses = {"complete", "skipped_no_potential_ligands"}
        observed_statuses = set(receiver_status["status"].astype(str))
        unexpected_statuses = sorted(observed_statuses.difference(allowed_statuses))
        if unexpected_statuses:
            raise ValueError(
                "NicheNet receiver status contains unsupported statuses: "
                + ", ".join(unexpected_statuses)
            )
        complete_status = receiver_status.loc[
            receiver_status["status"].astype(str).eq("complete")
        ]
        if complete_status.empty:
            raise ValueError("NicheNet receiver status has no complete receiver")
        skipped_status = receiver_status.loc[
            receiver_status["status"].astype(str).eq("skipped_no_potential_ligands")
        ]
        skipped_potential_ligands = pd.to_numeric(
            skipped_status["n_potential_ligands"], errors="coerce"
        )
        if (
            skipped_potential_ligands.isna().any()
            or not skipped_potential_ligands.eq(0).all()
        ):
            raise ValueError(
                "skipped NicheNet receivers must have zero potential ligands"
            )
        if skipped_status["reason"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError("skipped NicheNet receivers must record a reason")
        complete_receivers = set(complete_status["receiver"].astype(str))
        activity_receivers = set(activities["receiver"].astype(str))
        if complete_receivers != activity_receivers:
            raise ValueError(
                "NicheNet complete receiver status must match activity receivers"
            )
        status_datasets = set(receiver_status["dataset"].astype(str))
        activity_datasets = set(activities["dataset"].astype(str))
        gene_set_datasets = set(receiver_gene_sets["dataset"].astype(str))
        candidate_datasets = dataset_sets["candidates"]
        target_datasets = dataset_sets["targets"]
        if (
            len(status_datasets) != 1
            or status_datasets != activity_datasets
            or status_datasets != gene_set_datasets
            or status_datasets != candidate_datasets
            or status_datasets != target_datasets
        ):
            raise ValueError(
                "NicheNet candidate, activity, target, receiver-status, and "
                "receiver-gene-set datasets must match"
            )
        status_receivers = set(receiver_status["receiver"].astype(str))
        gene_set_receivers = set(receiver_gene_sets["receiver"].astype(str))
        if status_receivers != gene_set_receivers:
            raise ValueError(
                "NicheNet receiver-status receivers must match receiver gene sets"
            )
        target_receivers = set(targets["receiver"].astype(str))
        if not target_receivers.issubset(complete_receivers):
            raise ValueError(
                "NicheNet target receivers must be complete receiver-status entries"
            )
        activity_pairs = set(
            zip(
                activities["receiver"].astype(str),
                activities["ligand"].astype(str),
                strict=False,
            )
        )
        target_pairs = set(
            zip(
                targets["receiver"].astype(str),
                targets["ligand"].astype(str),
                strict=False,
            )
        )
        if not target_pairs.issubset(activity_pairs):
            raise ValueError(
                "NicheNet target receiver-ligand pairs must be present in activities"
            )
        receiver_status_summary = {
            "present": True,
            "allowed_statuses": sorted(allowed_statuses),
            "counts": {
                str(status): int(count)
                for status, count in receiver_status["status"].value_counts().items()
            },
            "n_receivers": int(len(receiver_status)),
            "n_complete_receivers": int(len(complete_status)),
            "n_skipped_no_potential_ligands": int(len(skipped_status)),
        }
    activities["activity_rank"] = activities.groupby("receiver", sort=False)[
        "aupr_corrected"
    ].transform(rank_percentile)
    evidence = candidates.merge(
        activities[["receiver", "ligand", "aupr_corrected", "activity_rank"]],
        on=["receiver", "ligand"],
        how="inner",
        validate="many_to_one",
    )
    evidence["lr_evidence"] = evidence["activity_rank"] * np.sqrt(
        evidence["sender_fraction"] * evidence["receiver_fraction"]
    )
    top = (
        evidence.sort_values("lr_evidence", ascending=False, kind="mergesort")
        .groupby(["dataset", "sender", "receiver"], sort=False)
        .head(5)
    )
    pair_scores = (
        top.groupby(["dataset", "sender", "receiver"], as_index=False)
        .agg(nichenet_support_score=("lr_evidence", "mean"))
        .rename(columns={"sender": "sender_type", "receiver": "receiver_type"})
    )
    detailed = evidence.merge(
        targets,
        on=["dataset", "receiver", "ligand"],
        how="inner",
        validate="many_to_many",
    )
    detailed["ligand_target_evidence"] = detailed["lr_evidence"] * detailed["weight"]
    output.mkdir(parents=True)
    pair_path = output / "nichenet_type_pair_scores.csv"
    evidence_path = output / "nichenet_lr_evidence.csv.gz"
    target_path = output / "nichenet_ligand_target_evidence.csv.gz"
    receiver_status_output = output / "nichenet_receiver_status.csv"
    pair_scores.to_csv(pair_path, index=False)
    evidence.to_csv(evidence_path, index=False, compression="gzip")
    detailed.to_csv(target_path, index=False, compression="gzip")
    if receiver_status is not None:
        receiver_status.to_csv(receiver_status_output, index=False)
    output_paths = [pair_path, evidence_path, target_path]
    if receiver_status is not None:
        output_paths.append(receiver_status_output)
    manifest = {
        "schema_version": 3,
        "workflow": "spatial_communication_consistency_nichenet_summary",
        "pair_score": "mean of the top five activity-rank × sqrt(sender-expression-fraction × receiver-expression-fraction) LR evidences",
        "sources": {label: _artifact(path) for label, path in required.items()},
        "receiver_status": receiver_status_summary,
        "outputs": {path.name: _artifact(path) for path in output_paths},
    }
    _write_json(output / "manifest.json", manifest)


def _requires_positive_receiver_aupr(spec: dict[str, object]) -> bool:
    """Apply the strict cross-species receiver-response gate only where declared."""

    scope = str(
        spec.get("nichenet_target_evidence_scope", "pair_level_receiver_response")
    )
    return "strict_confidence1" in scope.casefold()


def _selected_molecular_evidence(
    selected: pd.DataFrame, datasets: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pathway_rows: list[dict[str, object]] = []
    lr_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    score_column = "abundance_controlled_distinct_cell_score"
    for pair in selected.itertuples(index=False):
        dataset = str(pair.dataset)
        sender = str(pair.sender_type)
        receiver = str(pair.receiver_type)
        spec = dict(datasets[dataset])
        for kind, key, label_columns, destination in (
            ("pathway", "commot_pathway_csv", ["pathway"], pathway_rows),
            (
                "ligand_receptor",
                "commot_lr_csv",
                ["ligand", "receptor", "pathway"],
                lr_rows,
            ),
        ):
            if not spec.get(key):
                continue
            table = _read_table(spec[key], label=f"{dataset} {kind}")
            table = table.loc[
                table.sender_type.astype(str).eq(sender)
                & table.receiver_type.astype(str).eq(receiver)
            ].copy()
            if "stage" in table:
                terminal = float(FORMAL_DATASET_CONTRACTS[dataset]["terminal_time"])
                table = table.loc[
                    np.isclose(pd.to_numeric(table.stage, errors="raise"), terminal)
                ].copy()
            table[score_column] = pd.to_numeric(table[score_column], errors="raise")
            table = table.loc[table[score_column] > 0].sort_values(
                score_column, ascending=False, kind="mergesort"
            )
            table = table.drop_duplicates(label_columns)
            total_score = float(table[score_column].sum())
            table["fraction_of_pair_evidence"] = (
                table[score_column] / total_score if total_score > 0 else 0.0
            )
            table = table.head(3)
            top_score = float(table[score_column].max()) if not table.empty else 0.0
            for rank, record in enumerate(table.itertuples(index=False), start=1):
                row = {
                    "dataset": dataset,
                    "sender_type": sender,
                    "receiver_type": receiver,
                    "rank_within_pair": rank,
                    "commot_score": float(getattr(record, score_column)),
                    "relative_to_pair_top": float(getattr(record, score_column))
                    / top_score,
                    "fraction_of_pair_evidence": float(
                        getattr(record, "fraction_of_pair_evidence")
                    ),
                }
                row.update(
                    {column: str(getattr(record, column)) for column in label_columns}
                )
                destination.append(row)
        target_path = spec.get("nichenet_target_csv")
        if target_path:
            evidence_scope = str(
                spec.get("nichenet_target_evidence_scope", "primary_species_prior")
            )
            table = _read_table(target_path, label=f"{dataset} NicheNet targets")
            table = table.loc[
                table.sender.astype(str).eq(sender)
                & table.receiver.astype(str).eq(receiver)
            ].copy()
            table["ligand_target_evidence"] = pd.to_numeric(
                table["ligand_target_evidence"], errors="raise"
            )
            if "aupr_corrected" in table and _requires_positive_receiver_aupr(spec):
                table["aupr_corrected"] = pd.to_numeric(
                    table["aupr_corrected"], errors="raise"
                )
                table = table.loc[table["aupr_corrected"] > 0].copy()
            table = (
                table.loc[table.ligand_target_evidence > 0]
                .sort_values(
                    "ligand_target_evidence", ascending=False, kind="mergesort"
                )
                .drop_duplicates(["ligand", "receptor", "target"])
            )
            representative_lr = (
                table.groupby("target", as_index=False, sort=False)
                .first()[["target", "ligand", "receptor"]]
                .rename(
                    columns={
                        "ligand": "representative_ligand",
                        "receptor": "representative_receptor",
                    }
                )
            )
            table = (
                table.groupby("target", as_index=False, sort=False)
                .agg(
                    nichenet_evidence=("ligand_target_evidence", "sum"),
                    supporting_ligand_receptor_count=(
                        "ligand_target_evidence",
                        "size",
                    ),
                )
                .merge(representative_lr, on="target", validate="one_to_one")
                .sort_values("nichenet_evidence", ascending=False, kind="mergesort")
            )
            total_evidence = float(table.nichenet_evidence.sum())
            table["fraction_of_pair_evidence"] = (
                table.nichenet_evidence / total_evidence if total_evidence > 0 else 0.0
            )
            table = table.head(3)
            for rank, record in enumerate(table.itertuples(index=False), start=1):
                target_rows.append(
                    {
                        "dataset": dataset,
                        "sender_type": sender,
                        "receiver_type": receiver,
                        "rank_within_pair": rank,
                        "target": str(record.target),
                        "nichenet_evidence": float(record.nichenet_evidence),
                        "fraction_of_pair_evidence": float(
                            record.fraction_of_pair_evidence
                        ),
                        "supporting_ligand_receptor_count": int(
                            record.supporting_ligand_receptor_count
                        ),
                        "representative_ligand": str(record.representative_ligand),
                        "representative_receptor": str(record.representative_receptor),
                        "evidence_scope": evidence_scope,
                    }
                )
    pathway_columns = [
        "dataset",
        "sender_type",
        "receiver_type",
        "rank_within_pair",
        "commot_score",
        "relative_to_pair_top",
        "fraction_of_pair_evidence",
        "pathway",
    ]
    lr_columns = [
        "dataset",
        "sender_type",
        "receiver_type",
        "rank_within_pair",
        "commot_score",
        "relative_to_pair_top",
        "fraction_of_pair_evidence",
        "ligand",
        "receptor",
        "pathway",
    ]
    target_columns = [
        "dataset",
        "sender_type",
        "receiver_type",
        "rank_within_pair",
        "target",
        "nichenet_evidence",
        "fraction_of_pair_evidence",
        "supporting_ligand_receptor_count",
        "representative_ligand",
        "representative_receptor",
        "evidence_scope",
    ]
    return (
        pd.DataFrame(pathway_rows, columns=pathway_columns),
        pd.DataFrame(lr_rows, columns=lr_columns),
        pd.DataFrame(target_rows, columns=target_columns),
    )


def _verify_artifact_bytes(
    path: Path, record: dict[str, object], *, label: str
) -> None:
    """Require a local artifact to match its frozen manifest record."""

    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    expected_size = int(record["size_bytes"])
    expected_sha = str(record["sha256"])
    if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
        raise ValueError(f"{label} does not match the frozen manifest: {path}")


MOLECULAR_CONFIG_OVERLAY_KEYS = frozenset(
    {
        "nichenet_target_csv",
        "nichenet_target_evidence_scope",
        "nichenet_summary_manifest",
        "nichenet_run_manifest",
    }
)


def _verify_molecular_config_overlay(
    config_path: Path,
    config: dict[str, object],
    selection_config_record: dict[str, object],
) -> Path:
    """Bind molecular-only NicheNet updates to a frozen selection config.

    LR-axis selection does not consume the NicheNet target table. A later
    molecular summary may therefore bind a newly frozen target table without
    reselecting the CytoBridge axes, but every selection-driving config field
    must remain equivalent at the JSON-value level.
    """

    try:
        _verify_artifact_bytes(
            config_path,
            selection_config_record,
            label="model-biology config",
        )
        return config_path
    except ValueError as exact_error:
        frozen_path = Path(str(selection_config_record["path"])).expanduser().resolve()
        _verify_artifact_bytes(
            frozen_path,
            selection_config_record,
            label="frozen selection config",
        )
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

        def selection_projection(payload: dict[str, object]) -> dict[str, object]:
            projected = json.loads(json.dumps(payload))
            datasets = projected.get("datasets")
            if not isinstance(datasets, dict):
                raise ValueError("model-biology config lacks a datasets mapping")
            for spec in datasets.values():
                if not isinstance(spec, dict):
                    raise ValueError("model-biology dataset spec must be a mapping")
                for key in MOLECULAR_CONFIG_OVERLAY_KEYS:
                    spec.pop(key, None)
            return projected

        if selection_projection(config) != selection_projection(frozen):
            raise ValueError(
                "model-biology config differs from the frozen selection config "
                "outside the allowed molecular-only NicheNet overlay fields"
            ) from exact_error
        return frozen_path


def summarize_model_biology_molecular(args: argparse.Namespace) -> None:
    """Resolve model-first LR selections into computed pathway/target evidence."""

    config_path, config = _model_biology_config(args.config)
    selection_dir = Path(args.selection_dir).expanduser().resolve()
    selection_manifest_path = selection_dir / "manifest.json"
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    if selection_manifest.get("workflow") != "five_dataset_model_linked_lr_selection":
        raise ValueError("selection manifest has the wrong workflow")
    if selection_manifest.get("status") != "complete":
        raise ValueError("selection manifest is not complete")
    frozen_selection_config_path = _verify_molecular_config_overlay(
        config_path,
        config,
        dict(selection_manifest["inputs"]["config"]),
    )
    file_records = {
        "candidates": ("model_linked_lr_candidates.csv", "candidates"),
        "selected": ("selected_model_linked_lr.csv", "selected"),
        "support": ("model_linked_external_support.csv", "external_support"),
        "status": ("model_linked_lr_selection_status.csv", "status"),
    }
    paths: dict[str, Path] = {}
    for label, (filename, manifest_key) in file_records.items():
        path = selection_dir / filename
        _verify_artifact_bytes(
            path,
            dict(selection_manifest["outputs"][manifest_key]),
            label=f"selection {label}",
        )
        paths[label] = path
    candidates = pd.read_csv(paths["candidates"])
    selected = pd.read_csv(paths["selected"])
    support = pd.read_csv(paths["support"]).set_index("dataset")
    status = pd.read_csv(paths["status"]).set_index("dataset")
    dataset_order = list(FORMAL_DATASET_CONTRACTS)
    if set(status.index.astype(str)) != set(dataset_order):
        raise ValueError("selection status does not cover the five datasets")

    selection_sources = selection_manifest["inputs"]["datasets"]
    molecular_specs: dict[str, dict[str, object]] = {}
    molecular_sources: dict[str, dict[str, object]] = {}
    for dataset in dataset_order:
        spec = dict(config["datasets"][dataset])
        strict_nichenet_scope = _requires_positive_receiver_aupr(spec)
        if strict_nichenet_scope and (
            not spec.get("nichenet_summary_manifest")
            or not spec.get("nichenet_run_manifest")
        ):
            raise ValueError(
                f"{dataset} strict_confidence1 NicheNet scope requires both "
                "nichenet_summary_manifest and nichenet_run_manifest"
            )
        commot_lr_path = Path(spec["commot_lr_csv"]).expanduser().resolve()
        commot_pathway_path = commot_lr_path.with_name("commot_pathway_scores.csv.gz")
        source_record = dict(selection_sources[dataset])
        _verify_artifact_bytes(
            commot_lr_path,
            dict(source_record["commot_lr"]),
            label=f"{dataset} COMMOT LR table",
        )
        nichenet_path: Path | None = None
        nichenet_lr_path: Path | None = None
        nichenet_summary_manifest_path: Path | None = None
        nichenet_run_manifest_path: Path | None = None
        nichenet_record = source_record.get("nichenet_targets")
        if spec.get("nichenet_target_csv"):
            nichenet_path = Path(spec["nichenet_target_csv"]).expanduser().resolve()
            frozen_nichenet_path = (
                Path(str(nichenet_record["path"])).expanduser().resolve()
                if nichenet_record
                else None
            )
            if nichenet_record and nichenet_path == frozen_nichenet_path:
                _verify_artifact_bytes(
                    nichenet_path,
                    dict(nichenet_record),
                    label=f"{dataset} NicheNet target table",
                )
            nichenet_lr_path = nichenet_path.with_name("nichenet_lr_evidence.csv.gz")
            if not nichenet_lr_path.is_file():
                raise FileNotFoundError(
                    f"missing {dataset} NicheNet LR evidence: {nichenet_lr_path}"
                )
            if spec.get("nichenet_summary_manifest"):
                nichenet_summary_manifest_path = (
                    Path(str(spec["nichenet_summary_manifest"])).expanduser().resolve()
                )
                summary_manifest = json.loads(
                    nichenet_summary_manifest_path.read_text(encoding="utf-8")
                )
                if summary_manifest.get("workflow") != (
                    "spatial_communication_consistency_nichenet_summary"
                ):
                    raise ValueError(f"{dataset} NicheNet summary manifest is invalid")
                if strict_nichenet_scope:
                    receiver_summary = dict(summary_manifest.get("receiver_status", {}))
                    if (
                        not bool(receiver_summary.get("present"))
                        or int(receiver_summary.get("n_complete_receivers", 0)) < 1
                    ):
                        raise ValueError(
                            f"{dataset} strict NicheNet summary must bind at least "
                            "one complete receiver status"
                        )
                summary_outputs = dict(summary_manifest.get("outputs", {}))
                target_record = summary_outputs.get(nichenet_path.name)
                if not target_record:
                    raise ValueError(
                        f"{dataset} NicheNet summary does not bind {nichenet_path.name}"
                    )
                _verify_artifact_bytes(
                    nichenet_path,
                    dict(target_record),
                    label=f"{dataset} NicheNet summary target output",
                )
                if nichenet_lr_path.is_file():
                    lr_record = summary_outputs.get(nichenet_lr_path.name)
                    if not lr_record:
                        raise ValueError(
                            f"{dataset} NicheNet summary does not bind "
                            f"{nichenet_lr_path.name}"
                        )
                    _verify_artifact_bytes(
                        nichenet_lr_path,
                        dict(lr_record),
                        label=f"{dataset} NicheNet summary LR output",
                    )
            if spec.get("nichenet_run_manifest"):
                nichenet_run_manifest_path = (
                    Path(str(spec["nichenet_run_manifest"])).expanduser().resolve()
                )
                if not nichenet_run_manifest_path.is_file():
                    raise FileNotFoundError(
                        f"missing {dataset} NicheNet run manifest: "
                        f"{nichenet_run_manifest_path}"
                    )
                run_manifest = json.loads(
                    nichenet_run_manifest_path.read_text(encoding="utf-8")
                )
                if strict_nichenet_scope and (
                    run_manifest.get("formal_primary") is not False
                    or str(run_manifest.get("analysis_tier")) != "sensitivity"
                ):
                    raise ValueError(
                        f"{dataset} strict NicheNet run manifest must declare "
                        "formal_primary=false and analysis_tier=sensitivity"
                    )
        molecular_specs[dataset] = {
            "commot_lr_csv": str(commot_lr_path),
            "commot_pathway_csv": str(commot_pathway_path),
            "nichenet_target_csv": str(nichenet_path) if nichenet_path else None,
            "nichenet_target_evidence_scope": str(
                spec.get(
                    "nichenet_target_evidence_scope",
                    "pair_level_receiver_response",
                )
            ),
        }
        molecular_sources[dataset] = {
            "commot_lr": _artifact(commot_lr_path),
            "commot_pathway": _artifact(commot_pathway_path),
        }
        if nichenet_path is not None:
            molecular_sources[dataset]["nichenet_targets"] = _artifact(nichenet_path)
            molecular_sources[dataset]["nichenet_target_evidence_scope"] = str(
                molecular_specs[dataset].get(
                    "nichenet_target_evidence_scope", "pair_level_receiver_response"
                )
            )
            assert nichenet_lr_path is not None
            if nichenet_lr_path.is_file():
                molecular_sources[dataset]["nichenet_lr_evidence"] = _artifact(
                    nichenet_lr_path
                )
            if nichenet_summary_manifest_path is not None:
                molecular_sources[dataset]["nichenet_summary_manifest"] = _artifact(
                    nichenet_summary_manifest_path
                )
            if nichenet_run_manifest_path is not None:
                molecular_sources[dataset]["nichenet_run_manifest"] = _artifact(
                    nichenet_run_manifest_path
                )

    commot_pathways, commot_lrs, nichenet_targets = _selected_molecular_evidence(
        selected, molecular_specs
    )

    pathway_candidates = candidates.loc[
        pd.to_numeric(candidates["cytobridge_message_lr_score"], errors="raise").gt(0)
    ].copy()
    pathway_candidates["cytobridge_message_lr_score"] = pd.to_numeric(
        pathway_candidates["cytobridge_message_lr_score"], errors="raise"
    )
    pathway_candidates["cytobridge_pathway"] = pathway_candidates["pathways"].map(
        lambda value: [item.strip() for item in str(value).split(";") if item.strip()]
    )
    pathway_candidates = pathway_candidates.explode("cytobridge_pathway")
    pathway_scores = (
        pathway_candidates.groupby(
            ["dataset", "cytobridge_pathway"], as_index=False, sort=False
        )
        .agg(
            cytobridge_pathway_score=("cytobridge_message_lr_score", "sum"),
            contributing_lr_pair_axes=("cytobridge_message_lr_score", "size"),
        )
        .sort_values(
            ["dataset", "cytobridge_pathway_score", "cytobridge_pathway"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    pathway_scores["cytobridge_pathway_rank"] = (
        pathway_scores.groupby("dataset", sort=False).cumcount() + 1
    )
    pathway_scores["cytobridge_pathway_count"] = pathway_scores.groupby(
        "dataset", sort=False
    )["cytobridge_pathway"].transform("size")
    pathway_scores["cytobridge_pathway_percentile"] = pathway_scores.groupby(
        "dataset", sort=False
    )["cytobridge_pathway_score"].transform(rank_percentile)

    def rank_consistency(
        table: pd.DataFrame,
        *,
        dataset: str,
        method: str,
        external_column: str,
    ) -> dict[str, object]:
        local = table.loc[
            pd.to_numeric(table["cytobridge_message_lr_score"], errors="raise").gt(0)
            & pd.to_numeric(table[external_column], errors="raise").gt(0)
        ].copy()
        if len(local) < 4:
            return {
                "dataset": dataset,
                "external_method": method,
                "available": False,
                "n_jointly_positive_axes": int(len(local)),
                "spearman_rho": np.nan,
                "top_fraction": TOP_FRACTION,
                "top_jaccard": np.nan,
            }
        local = local.reset_index(drop=True)
        rho = float(
            local["cytobridge_message_lr_score"].corr(
                local[external_column], method="spearman"
            )
        )
        top_n = max(1, int(np.ceil(len(local) * TOP_FRACTION)))
        cb_top = set(
            local.sort_values(
                ["cytobridge_message_lr_score", "sender_type", "receiver_type"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .head(top_n)
            .index
        )
        external_top = set(
            local.sort_values(
                [external_column, "sender_type", "receiver_type"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .head(top_n)
            .index
        )
        return {
            "dataset": dataset,
            "external_method": method,
            "available": True,
            "n_jointly_positive_axes": int(len(local)),
            "spearman_rho": rho,
            "top_fraction": TOP_FRACTION,
            "top_jaccard": len(cb_top & external_top) / len(cb_top | external_top),
        }

    molecular_metric_rows: list[dict[str, object]] = []
    nichenet_evidence_tables: dict[str, pd.DataFrame] = {}
    nichenet_target_tables: dict[str, pd.DataFrame] = {}
    for dataset in dataset_order:
        dataset_candidates = candidates.loc[candidates["dataset"].eq(dataset)].copy()
        if dataset_candidates.empty:
            continue
        if "commot_abundance_score" in dataset_candidates:
            molecular_metric_rows.append(
                rank_consistency(
                    dataset_candidates,
                    dataset=dataset,
                    method="COMMOT",
                    external_column="commot_abundance_score",
                )
            )
        spec = molecular_specs[dataset]
        target_value = spec.get("nichenet_target_csv")
        if not target_value:
            continue
        target_path = Path(str(target_value)).expanduser().resolve()
        evidence_path = target_path.with_name("nichenet_lr_evidence.csv.gz")
        targets_full = pd.read_csv(target_path)
        nichenet_target_tables[dataset] = targets_full
        if not evidence_path.is_file():
            continue
        evidence = pd.read_csv(evidence_path)
        nichenet_evidence_tables[dataset] = evidence
        evidence_keys = evidence.rename(
            columns={"sender": "sender_type", "receiver": "receiver_type"}
        ).copy()
        for column in ("ligand", "receptor"):
            evidence_keys[f"{column}_key"] = (
                evidence_keys[column].astype(str).str.casefold()
            )
            dataset_candidates[f"{column}_key"] = (
                dataset_candidates[column].astype(str).str.casefold()
            )
        evidence_keys = evidence_keys.groupby(
            [
                "sender_type",
                "receiver_type",
                "ligand_key",
                "receptor_key",
            ],
            as_index=False,
        ).agg(nichenet_lr_evidence=("lr_evidence", "max"))
        merged = dataset_candidates.merge(
            evidence_keys,
            on=[
                "sender_type",
                "receiver_type",
                "ligand_key",
                "receptor_key",
            ],
            how="inner",
            validate="many_to_one",
        )
        molecular_metric_rows.append(
            rank_consistency(
                merged,
                dataset=dataset,
                method="NicheNet",
                external_column="nichenet_lr_evidence",
            )
        )
    molecular_metrics = pd.DataFrame(
        molecular_metric_rows,
        columns=[
            "dataset",
            "external_method",
            "available",
            "n_jointly_positive_axes",
            "spearman_rho",
            "top_fraction",
            "top_jaccard",
        ],
    )

    chain_rows: list[dict[str, object]] = []
    for dataset, targets_full in nichenet_target_tables.items():
        evidence_scope = str(
            molecular_specs[dataset].get(
                "nichenet_target_evidence_scope", "pair_level_receiver_response"
            )
        )
        passes_support = (
            candidates["passes_support"].astype(str).str.casefold().isin({"true", "1"})
            if "passes_support" in candidates
            else (
                pd.to_numeric(candidates["cytobridge_message_lr_score"], errors="raise")
                > 0
            )
            & (
                pd.to_numeric(candidates["n_model_lr_active_edges"], errors="raise")
                >= 10
            )
        )
        local = candidates.loc[
            candidates["dataset"].eq(dataset) & passes_support
        ].copy()
        if local.empty:
            continue
        local = local.sort_values(
            [
                "cytobridge_message_lr_score",
                "sender_type",
                "receiver_type",
                "ligand",
                "receptor",
            ],
            ascending=[False, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        local["cytobridge_global_rank"] = np.arange(1, len(local) + 1)
        target_keys = targets_full.copy()
        for table in (local, target_keys):
            for column in ("ligand", "receptor"):
                table[f"{column}_key"] = table[column].astype(str).str.casefold()
        target_keys = target_keys.rename(
            columns={"sender": "sender_type", "receiver": "receiver_type"}
        )
        merged = local.merge(
            target_keys,
            on=[
                "sender_type",
                "receiver_type",
                "ligand_key",
                "receptor_key",
            ],
            how="inner",
            suffixes=("", "_nichenet"),
        )
        merged["ligand_target_evidence"] = pd.to_numeric(
            merged["ligand_target_evidence"], errors="raise"
        )
        if "aupr_corrected" in merged and _requires_positive_receiver_aupr(
            molecular_specs[dataset]
        ):
            merged["aupr_corrected"] = pd.to_numeric(
                merged["aupr_corrected"], errors="raise"
            )
            merged = merged.loc[merged["aupr_corrected"] > 0].copy()
        merged = merged.loc[merged["ligand_target_evidence"] > 0]
        if merged.empty:
            continue
        chosen_rank = int(merged["cytobridge_global_rank"].min())
        chosen = merged.loc[
            merged["cytobridge_global_rank"].eq(chosen_rank)
        ].sort_values(
            ["ligand_target_evidence", "target"],
            ascending=[False, True],
            kind="mergesort",
        )
        chosen = chosen.drop_duplicates("target").head(3)
        for target_rank, record in enumerate(chosen.itertuples(index=False), start=1):
            aupr_value = getattr(record, "aupr_corrected", np.nan)
            chain_rows.append(
                {
                    "dataset": dataset,
                    "cytobridge_global_rank": chosen_rank,
                    "sender_type": str(record.sender_type),
                    "receiver_type": str(record.receiver_type),
                    "ligand": str(record.ligand),
                    "receptor": str(record.receptor),
                    "pathways": str(record.pathways),
                    "cytobridge_percentile": float(record.cytobridge_percentile),
                    "commot_percentile": float(record.commot_percentile),
                    "receiver_target_rank": target_rank,
                    "receiver_target": str(record.target),
                    "nichenet_ligand_target_evidence": float(
                        record.ligand_target_evidence
                    ),
                    "nichenet_corrected_aupr": (
                        float(aupr_value) if pd.notna(aupr_value) else np.nan
                    ),
                    "nichenet_evidence_scope": evidence_scope,
                }
            )
    model_first_chains = pd.DataFrame(
        chain_rows,
        columns=[
            "dataset",
            "cytobridge_global_rank",
            "sender_type",
            "receiver_type",
            "ligand",
            "receptor",
            "pathways",
            "cytobridge_percentile",
            "commot_percentile",
            "receiver_target_rank",
            "receiver_target",
            "nichenet_ligand_target_evidence",
            "nichenet_corrected_aupr",
            "nichenet_evidence_scope",
        ],
    )

    panel_rows: list[dict[str, object]] = []
    for dataset in dataset_order:
        dataset_status = status.loc[dataset]
        if str(dataset_status.status) != "complete":
            panel_rows.append(
                {
                    "dataset": dataset,
                    "status": "not_evaluable",
                    "reason": str(dataset_status.reason),
                }
            )
            continue
        row = selected.loc[selected["dataset"].eq(dataset)]
        if len(row) != 1:
            raise ValueError(f"{dataset} must have exactly one selected LR axis")
        row = row.iloc[0]
        support_row = support.loc[dataset]
        within_pair = candidates.loc[
            candidates["dataset"].eq(dataset)
            & candidates["sender_type"].astype(str).eq(str(row.sender_type))
            & candidates["receiver_type"].astype(str).eq(str(row.receiver_type))
        ].copy()
        within_pair["cytobridge_within_pair_rank"] = within_pair[
            "cytobridge_message_lr_score"
        ].rank(method="min", ascending=False)
        commot_rank_source = (
            "commot_abundance_score"
            if "commot_abundance_score" in within_pair
            else "commot_percentile"
        )
        within_pair["commot_within_pair_rank"] = within_pair[commot_rank_source].rank(
            method="min", ascending=False
        )
        selected_in_pair = within_pair.loc[
            within_pair["ligand"]
            .astype(str)
            .str.casefold()
            .eq(str(row.ligand).casefold())
            & within_pair["receptor"]
            .astype(str)
            .str.casefold()
            .eq(str(row.receptor).casefold())
        ]
        if len(selected_in_pair) != 1:
            raise ValueError(f"{dataset} selected LR is not unique within its pair")
        selected_in_pair = selected_in_pair.iloc[0]
        selected_pathways = {
            item.strip() for item in str(row.pathways).split(";") if item.strip()
        }
        ranked_pathways = pathway_scores.loc[
            pathway_scores["dataset"].eq(dataset)
            & pathway_scores["cytobridge_pathway"].isin(selected_pathways)
        ].sort_values(
            ["cytobridge_pathway_score", "cytobridge_pathway"],
            ascending=[False, True],
            kind="mergesort",
        )
        if ranked_pathways.empty:
            raise ValueError(f"{dataset} selected pathway lacks a computed model rank")
        pathway = ranked_pathways.iloc[0]
        top_commot_pathway = commot_pathways.loc[
            (commot_pathways["dataset"] == dataset)
            & (commot_pathways["rank_within_pair"] == 1)
        ]
        top_commot_lr = commot_lrs.loc[
            (commot_lrs["dataset"] == dataset) & (commot_lrs["rank_within_pair"] == 1)
        ]
        top_target = nichenet_targets.loc[
            (nichenet_targets["dataset"] == dataset)
            & (nichenet_targets["rank_within_pair"] == 1)
        ]
        target_record = top_target.iloc[0] if not top_target.empty else None
        panel_rows.append(
            {
                "dataset": dataset,
                "status": "complete",
                "reason": "",
                "stage": float(row.stage),
                "sender_type": str(row.sender_type),
                "receiver_type": str(row.receiver_type),
                "ligand": str(row.ligand),
                "receptor": str(row.receptor),
                "pathways": str(row.pathways),
                "n_model_lr_active_edges": int(row.n_model_lr_active_edges),
                "cytobridge_lr_percentile": float(row.cytobridge_percentile),
                "cytobridge_within_pair_rank": int(
                    selected_in_pair.cytobridge_within_pair_rank
                ),
                "commot_within_pair_rank": int(
                    selected_in_pair.commot_within_pair_rank
                ),
                "within_pair_lr_count": int(len(within_pair)),
                "cytobridge_pathway": str(pathway.cytobridge_pathway),
                "cytobridge_pathway_percentile": float(
                    pathway.cytobridge_pathway_percentile
                ),
                "cytobridge_pathway_rank": int(pathway.cytobridge_pathway_rank),
                "cytobridge_pathway_count": int(pathway.cytobridge_pathway_count),
                "commot_exact_axis_percentile": float(support_row.commot_percentile),
                "commot_top_pathway": (
                    str(top_commot_pathway.iloc[0].pathway)
                    if not top_commot_pathway.empty
                    else np.nan
                ),
                "commot_top_lr": (
                    f"{top_commot_lr.iloc[0].ligand}–{top_commot_lr.iloc[0].receptor}"
                    if not top_commot_lr.empty
                    else np.nan
                ),
                "nichenet_top_receiver_target": (
                    str(target_record.target) if target_record is not None else np.nan
                ),
                "nichenet_top_receiver_target_fraction": (
                    float(target_record.fraction_of_pair_evidence)
                    if target_record is not None
                    else np.nan
                ),
                "nichenet_scope": (
                    str(target_record.evidence_scope)
                    if target_record is not None
                    else "not_evaluable"
                ),
            }
        )
    panel_columns = [
        "dataset",
        "status",
        "reason",
        "stage",
        "sender_type",
        "receiver_type",
        "ligand",
        "receptor",
        "pathways",
        "n_model_lr_active_edges",
        "cytobridge_lr_percentile",
        "cytobridge_within_pair_rank",
        "commot_within_pair_rank",
        "within_pair_lr_count",
        "cytobridge_pathway",
        "cytobridge_pathway_percentile",
        "cytobridge_pathway_rank",
        "cytobridge_pathway_count",
        "commot_exact_axis_percentile",
        "commot_top_pathway",
        "commot_top_lr",
        "nichenet_top_receiver_target",
        "nichenet_top_receiver_target_fraction",
        "nichenet_scope",
    ]
    panel = pd.DataFrame(panel_rows).reindex(columns=panel_columns)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    output_paths = {
        "panel": output / "model_biology_molecular_panel.csv",
        "pathway_ranks": output / "cytobridge_pathway_ranks.csv",
        "rank_consistency": output / "molecular_rank_consistency.csv",
        "model_first_nichenet_chains": output / "model_first_nichenet_chains.csv",
        "commot_pathways": output / "selected_pair_commot_pathways.csv",
        "commot_lrs": output / "selected_pair_commot_lr.csv",
        "nichenet_targets": output / "selected_pair_nichenet_targets.csv",
    }
    panel.to_csv(output_paths["panel"], index=False)
    pathway_scores.to_csv(output_paths["pathway_ranks"], index=False)
    molecular_metrics.to_csv(output_paths["rank_consistency"], index=False)
    model_first_chains.to_csv(output_paths["model_first_nichenet_chains"], index=False)
    commot_pathways.to_csv(output_paths["commot_pathways"], index=False)
    commot_lrs.to_csv(output_paths["commot_lrs"], index=False)
    nichenet_targets.to_csv(output_paths["nichenet_targets"], index=False)
    manifest = {
        "schema_version": 2,
        "workflow": "five_dataset_model_biology_molecular_summary",
        "status": "complete",
        "selection_rule": (
            "CytoBridge selects the LR axis before external evaluation; pathway "
            "scores sum abundance-normalized exact-message x LR-activity scores "
            "over all eligible directed model-pair/LR axes assigned to a pathway"
        ),
        "nichenet_chain_rule": (
            "Within each dataset, require positive ligand-target evidence; for the "
            "strict_confidence1 cross-species zebrafish scope additionally require "
            "corrected receiver AUPR > 0. Then select the first CytoBridge-ranked "
            "supported LR x directed-pair axis with a matched NicheNet receiver target."
        ),
        "implementation": _model_biology_implementation(),
        "inputs": {
            "config": _artifact(config_path),
            "frozen_selection_config": _artifact(frozen_selection_config_path),
            "selection_manifest": _artifact(selection_manifest_path),
            "selection_outputs": {
                label: _artifact(path) for label, path in paths.items()
            },
            "molecular_sources": molecular_sources,
        },
        "outputs": {label: _artifact(path) for label, path in output_paths.items()},
    }
    _write_json(output / "manifest.json", manifest)


def aggregate(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    datasets = config.get("datasets", {})
    if set(datasets) != set(FORMAL_DATASET_CONTRACTS):
        raise ValueError("aggregate config must bind exactly the five formal datasets")
    score_tables: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    for dataset in FORMAL_DATASET_CONTRACTS:
        spec = dict(datasets[dataset])
        sample_manifest = Path(spec["sample_manifest"]).expanduser().resolve()
        types = _terminal_types(sample_manifest)
        score_tables.extend(_load_cytobridge(spec, dataset, types))
        external, status = _load_external(spec, dataset, types)
        score_tables.extend(external)
        status_rows.extend(status)
        sources[dataset] = {
            "sample_manifest": _artifact(sample_manifest),
            "bound_files": {
                key: _artifact(Path(value))
                for key, value in spec.items()
                if key.endswith("_csv") and value
            },
        }
    scores = pd.concat(score_tables, ignore_index=True)
    metrics = pairwise_cytobridge_metrics(scores)
    decisions = evaluate_main_figure_gate(metrics)
    selected = _select_pairs(scores, decisions)
    pathways, ligand_receptors, targets = _selected_molecular_evidence(
        selected, datasets
    )
    status = pd.DataFrame(status_rows)
    scores.to_csv(output / "directed_pair_method_scores.csv", index=False)
    metrics.to_csv(output / "cytobridge_external_metrics.csv", index=False)
    decisions.to_csv(output / "main_figure_method_decisions.csv", index=False)
    status.to_csv(output / "method_execution_status.csv", index=False)
    selected.to_csv(output / "selected_biological_pairs.csv", index=False)
    pathways.to_csv(output / "selected_pair_commot_pathways.csv", index=False)
    ligand_receptors.to_csv(output / "selected_pair_commot_lr.csv", index=False)
    targets.to_csv(output / "selected_pair_nichenet_targets.csv", index=False)
    manifest = {
        "schema_version": 1,
        "workflow": "five_dataset_spatial_communication_consistency_aggregate",
        "claim_scope": "shared-input descriptive computational consistency; not causal or independent experimental validation",
        "comparison": {
            "unit": "complete directed terminal-stage cell-type-pair grid",
            "native_units_pooled": False,
            "primary_cytobridge_view": "CytoBridge exact message",
            "secondary_cytobridge_view": "CytoBridge attention",
            "top_fraction": TOP_FRACTION,
        },
        "main_figure_gate": MAIN_FIGURE_GATE,
        "aggregate_config": _artifact(config_path),
        "sources": sources,
        "outputs": {},
    }
    for path in sorted(output.glob("*.csv")):
        manifest["outputs"][path.name] = _artifact(path)
    _write_json(output / "manifest.json", manifest)


def _model_biology_config(path: str | Path) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("model-biology config requires schema_version=1")
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(FORMAL_DATASET_CONTRACTS):
        raise ValueError("model-biology config must bind exactly five datasets")
    return resolved, config


def _cellchat_selected_axis_support(
    data: ad.AnnData,
    selected: pd.Series,
    lr_path: Path,
    eligibility_path: Path,
) -> dict[str, object]:
    """Rank one selected axis in the complete eligible CellChat zero-filled grid."""

    eligibility = pd.read_csv(eligibility_path)
    eligible_flag = (
        eligibility["eligible"].astype(str).str.casefold().isin({"true", "1"})
    )
    axes = eligibility.loc[
        eligible_flag, ["current_ligand", "current_receptor"]
    ].drop_duplicates()
    axis_keys = set(
        zip(
            axes["current_ligand"].astype(str).str.casefold(),
            axes["current_receptor"].astype(str).str.casefold(),
        )
    )
    selected_axis = (
        str(selected.ligand).casefold(),
        str(selected.receptor).casefold(),
    )
    terminal = data.obs.loc[
        np.isclose(
            pd.to_numeric(data.obs["ccc_stage"], errors="coerce"),
            float(selected.stage),
        )
    ]
    counts = terminal["ccc_cell_type"].astype(str).value_counts()
    eligible_types = sorted(counts.loc[counts.ge(10)].index.astype(str))
    total = len(axis_keys) * len(eligible_types) * max(0, len(eligible_types) - 1)
    if selected_axis not in axis_keys:
        return {
            "cellchat_available": False,
            "cellchat_reason": "selected LR is not eligible in the frozen CellChat database mapping",
            "cellchat_score": 0.0,
            "cellchat_percentile": np.nan,
        }
    if (
        str(selected.sender_type) not in eligible_types
        or str(selected.receiver_type) not in eligible_types
    ):
        return {
            "cellchat_available": False,
            "cellchat_reason": "selected sender or receiver has fewer than 10 terminal cells",
            "cellchat_score": 0.0,
            "cellchat_percentile": np.nan,
        }
    values = pd.read_csv(lr_path)
    values = values.loc[
        np.isclose(
            pd.to_numeric(values["stage"], errors="coerce"), float(selected.stage)
        )
        & values["sender_type"].astype(str).ne(values["receiver_type"].astype(str))
    ].copy()
    values["score_numeric"] = pd.to_numeric(
        values["abundance_controlled_score"], errors="raise"
    )
    values = values.groupby(
        ["sender_type", "receiver_type", "ligand", "receptor"], as_index=False
    ).agg(score_numeric=("score_numeric", "max"), pathway=("pathway", "first"))
    match = values.loc[
        values["sender_type"].astype(str).eq(str(selected.sender_type))
        & values["receiver_type"].astype(str).eq(str(selected.receiver_type))
        & values["ligand"].astype(str).str.casefold().eq(selected_axis[0])
        & values["receptor"].astype(str).str.casefold().eq(selected_axis[1])
    ]
    score = float(match["score_numeric"].max()) if not match.empty else 0.0
    positive = values["score_numeric"].to_numpy(float)
    zero_count = max(0, total - len(positive))
    if total <= 0:
        raise ValueError("CellChat complete ranking universe is empty")
    if score > 0:
        less = zero_count + int(np.sum(positive < score))
        equal = int(np.sum(np.isclose(positive, score, rtol=0, atol=0)))
        percentile = (less + (equal + 1) / 2) / total
    else:
        percentile = (zero_count + 1) / (2 * total)
    return {
        "cellchat_available": True,
        "cellchat_reason": "",
        "cellchat_score": score,
        "cellchat_percentile": float(percentile),
    }


def select_model_biology(args: argparse.Namespace) -> None:
    """Select one model-linked LR axis per frozen biological type pair."""

    config_path, config = _model_biology_config(args.config)
    aggregate_dir = Path(args.aggregate_dir).expanduser().resolve()
    aggregate_manifest = aggregate_dir / "manifest.json"
    selected_pairs_path = aggregate_dir / "selected_biological_pairs.csv"
    pairs = pd.read_csv(selected_pairs_path).set_index("dataset")
    if set(pairs.index.astype(str)) != set(FORMAL_DATASET_CONTRACTS):
        raise ValueError("aggregate selected pairs do not cover the five datasets")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    candidate_tables: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    excluded_tables: list[pd.DataFrame] = []
    status_rows: list[dict[str, object]] = []
    external_support_rows: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    for dataset, contract in FORMAL_DATASET_CONTRACTS.items():
        spec = dict(config["datasets"][dataset])
        h5ad_path = Path(spec["h5ad"]).expanduser().resolve()
        expression_h5ad_path = Path(spec["expression_h5ad"]).expanduser().resolve()
        attribution_dir = Path(spec["attribution_dir"]).expanduser().resolve()
        commot_lr_path = Path(spec["commot_lr_csv"]).expanduser().resolve()
        lr_database_path = (
            Path(spec["commot_input_dir"]).expanduser().resolve()
            / "filtered_lr_database.csv"
        )
        data = ad.read_h5ad(h5ad_path)
        activity_data = ad.read_h5ad(expression_h5ad_path)
        try:
            candidates, selected, excluded = select_global_model_linked_lr_example(
                data,
                attribution_dir,
                pd.read_csv(commot_lr_path),
                activity_data=activity_data,
                lr_database=pd.read_csv(lr_database_path),
                dataset=dataset,
                terminal_time=float(contract["terminal_time"]),
                minimum_active_edges=int(args.minimum_active_edges),
            )
        except ValueError as error:
            status_rows.append(
                {
                    "dataset": dataset,
                    "status": "not_evaluable",
                    "reason": str(error),
                }
            )
        else:
            candidate_tables.append(candidates)
            selected_rows.append(selected.to_dict())
            status_rows.append({"dataset": dataset, "status": "complete", "reason": ""})
            if not excluded.empty:
                excluded.insert(0, "dataset", dataset)
                excluded_tables.append(excluded)
            cellchat_lr_path = Path(spec["cellchat_lr_csv"]).expanduser().resolve()
            cellchat_eligibility_path = (
                Path(spec["cellchat_eligibility_csv"]).expanduser().resolve()
            )
            support = {
                "dataset": dataset,
                "stage": float(selected.stage),
                "sender_type": str(selected.sender_type),
                "receiver_type": str(selected.receiver_type),
                "ligand": str(selected.ligand),
                "receptor": str(selected.receptor),
                "pathways": str(selected.pathways),
                "cytobridge_percentile": float(selected.cytobridge_percentile),
                "commot_percentile": float(selected.commot_percentile),
                **_cellchat_selected_axis_support(
                    data,
                    selected,
                    cellchat_lr_path,
                    cellchat_eligibility_path,
                ),
                "nichenet_available": False,
                "nichenet_target": "",
                "nichenet_ligand_target_evidence": np.nan,
            }
            nichenet_value = spec.get("nichenet_target_csv")
            if nichenet_value:
                nichenet_path = Path(nichenet_value).expanduser().resolve()
                nichenet = pd.read_csv(nichenet_path)
                match = nichenet.loc[
                    nichenet["sender"].astype(str).eq(str(selected.sender_type))
                    & nichenet["receiver"].astype(str).eq(str(selected.receiver_type))
                    & nichenet["ligand"]
                    .astype(str)
                    .str.casefold()
                    .eq(str(selected.ligand).casefold())
                    & nichenet["receptor"]
                    .astype(str)
                    .str.casefold()
                    .eq(str(selected.receptor).casefold())
                ].copy()
                if not match.empty:
                    match["evidence_numeric"] = pd.to_numeric(
                        match["ligand_target_evidence"], errors="raise"
                    )
                    best = match.sort_values(
                        ["evidence_numeric", "target"], ascending=[False, True]
                    ).iloc[0]
                    support["nichenet_available"] = True
                    support["nichenet_target"] = str(best.target)
                    support["nichenet_ligand_target_evidence"] = float(
                        best.evidence_numeric
                    )
            external_support_rows.append(support)
        attribution_manifest = attribution_dir / "run_manifest.json"
        sources[dataset] = {
            "h5ad": _artifact(h5ad_path),
            "expression_h5ad": _artifact(expression_h5ad_path),
            "attribution_manifest": _artifact(attribution_manifest),
            "commot_lr": _artifact(commot_lr_path),
            "filtered_lr_database": _artifact(lr_database_path),
            "commot_input_dir": str(
                Path(spec["commot_input_dir"]).expanduser().resolve()
            ),
            "cellchat_lr": _artifact(
                Path(spec["cellchat_lr_csv"]).expanduser().resolve()
            ),
            "cellchat_eligibility": _artifact(
                Path(spec["cellchat_eligibility_csv"]).expanduser().resolve()
            ),
        }
        if spec.get("nichenet_target_csv"):
            sources[dataset]["nichenet_targets"] = _artifact(
                Path(spec["nichenet_target_csv"]).expanduser().resolve()
            )
    candidates_path = output / "model_linked_lr_candidates.csv"
    selected_path = output / "selected_model_linked_lr.csv"
    excluded_path = output / "unrepresentable_lr_candidates.csv"
    if not candidate_tables:
        raise ValueError("No dataset yielded a model-linked LR example")
    pd.concat(candidate_tables, ignore_index=True).to_csv(candidates_path, index=False)
    selected_frame = pd.DataFrame(selected_rows)
    required_flow_columns = [
        "example_id",
        "stage",
        "stage_label",
        "ligand",
        "receptor",
        "pathways",
        "categories",
    ]
    remaining = [
        column
        for column in selected_frame.columns
        if column not in required_flow_columns
    ]
    selected_frame[required_flow_columns + remaining].to_csv(selected_path, index=False)
    for dataset, table in selected_frame.groupby("dataset", sort=False):
        table[required_flow_columns + remaining].to_csv(
            output / f"selected_model_linked_lr_{dataset}.csv", index=False
        )
    (
        pd.concat(excluded_tables, ignore_index=True)
        if excluded_tables
        else pd.DataFrame(columns=["dataset", "ligand", "receptor", "reason"])
    ).to_csv(excluded_path, index=False)
    status_path = output / "model_linked_lr_selection_status.csv"
    pd.DataFrame(status_rows).to_csv(status_path, index=False)
    external_support_path = output / "model_linked_external_support.csv"
    pd.DataFrame(external_support_rows).to_csv(external_support_path, index=False)
    manifest = {
        "schema_version": 1,
        "workflow": "five_dataset_model_linked_lr_selection",
        "status": "complete",
        "claim_scope": (
            "post-hoc LR-compatible decomposition of exact learned GNN messages; "
            "not native LR identity inference or causal validation"
        ),
        "selection_rule": str(selected_frame["selection_rule"].iloc[0]),
        "minimum_active_edges": int(args.minimum_active_edges),
        "attempted_datasets": len(FORMAL_DATASET_CONTRACTS),
        "evaluable_datasets": int(len(selected_frame)),
        "implementation": _model_biology_implementation(),
        "inputs": {
            "config": _artifact(config_path),
            "aggregate_manifest": _artifact(aggregate_manifest),
            "selected_biological_pairs": _artifact(selected_pairs_path),
            "datasets": sources,
        },
        "outputs": {
            "candidates": _artifact(candidates_path),
            "selected": _artifact(selected_path),
            "unrepresentable": _artifact(excluded_path),
            "status": _artifact(status_path),
            "external_support": _artifact(external_support_path),
        },
    }
    _write_json(output / "manifest.json", manifest)


def score_model_biology(args: argparse.Namespace) -> None:
    """Score five selected LR axes against cell-level COMMOT spatial flows."""

    config_path, config = _model_biology_config(args.config)
    selection_dir = Path(args.selection_dir).expanduser().resolve()
    selection_manifest_path = selection_dir / "manifest.json"
    selection = pd.read_csv(selection_dir / "selected_model_linked_lr.csv")
    selection_status = pd.read_csv(
        selection_dir / "model_linked_lr_selection_status.csv"
    ).set_index("dataset")
    if set(selection_status.index.astype(str)) != set(FORMAL_DATASET_CONTRACTS):
        raise ValueError("selected LR status does not cover the five datasets")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    result_rows: list[dict[str, object]] = []
    cb_edge_tables: list[pd.DataFrame] = []
    commot_edge_tables: list[pd.DataFrame] = []
    null_tables: list[pd.DataFrame] = []
    sources: dict[str, object] = {}
    for index, dataset in enumerate(FORMAL_DATASET_CONTRACTS):
        spec = dict(config["datasets"][dataset])
        status = selection_status.loc[dataset]
        if str(status.status) != "complete":
            result_rows.append(
                {
                    "dataset": dataset,
                    "available": False,
                    "reason": str(status.reason),
                }
            )
            continue
        h5ad_path = Path(spec["h5ad"]).expanduser().resolve()
        expression_h5ad_path = Path(spec["expression_h5ad"]).expanduser().resolve()
        attribution_dir = Path(spec["attribution_dir"]).expanduser().resolve()
        flow_dir = Path(spec["selected_commot_flow_dir"]).expanduser().resolve()
        flow_path = flow_dir / "selected_commot_cell_flows.csv.gz"
        flow_manifest_path = flow_dir / "manifest.json"
        selected = selection.loc[selection["dataset"].eq(dataset)].iloc[0].to_dict()
        data = ad.read_h5ad(h5ad_path)
        activity_data = ad.read_h5ad(expression_h5ad_path)
        attribution_manifest_path = attribution_dir / "run_manifest.json"
        attribution_manifest = json.loads(
            attribution_manifest_path.read_text(encoding="utf-8")
        )
        cutoff = float(attribution_manifest["checkpoint"]["spatial_cutoff"])
        result, cb_top, commot_top, null = model_linked_spatial_colocalization(
            data,
            attribution_dir,
            pd.read_csv(flow_path),
            selected,
            activity_data=activity_data,
            match_radius=cutoff / 2.0,
            top_fraction=float(args.top_fraction),
            permutations=int(args.permutations),
            seed=int(args.seed) + index,
        )
        result_rows.append(result)
        result_rows[-1]["available"] = True
        result_rows[-1]["reason"] = ""
        cb_top.insert(0, "dataset", dataset)
        commot_top.insert(0, "dataset", dataset)
        cb_edge_tables.append(cb_top)
        commot_edge_tables.append(commot_top)
        null.insert(0, "dataset", dataset)
        null.insert(2, "seed", int(args.seed) + index)
        null_tables.append(null)
        sources[dataset] = {
            "h5ad": _artifact(h5ad_path),
            "expression_h5ad": _artifact(expression_h5ad_path),
            "attribution_manifest": _artifact(attribution_manifest_path),
            "selected_commot_flow_manifest": _artifact(flow_manifest_path),
            "selected_commot_cell_flows": _artifact(flow_path),
        }
    summary_path = output / "model_linked_spatial_colocalization.csv"
    cb_path = output / "cytobridge_top_model_linked_edges.csv.gz"
    commot_path = output / "commot_top_lr_flows.csv.gz"
    null_path = output / "permutation_seed_audit.csv.gz"
    pd.DataFrame(result_rows).to_csv(summary_path, index=False)
    pd.concat(cb_edge_tables, ignore_index=True).to_csv(
        cb_path, index=False, compression="gzip"
    )
    pd.concat(commot_edge_tables, ignore_index=True).to_csv(
        commot_path, index=False, compression="gzip"
    )
    pd.concat(null_tables, ignore_index=True).to_csv(
        null_path, index=False, compression="gzip"
    )
    manifest = {
        "schema_version": 1,
        "workflow": "five_dataset_model_linked_lr_spatial_colocalization",
        "status": "complete",
        "claim_scope": (
            "descriptive spatial consistency of top LR-compatible edge midpoints; "
            "not exact edge accuracy, native LR inference, or causal validation"
        ),
        "design": {
            "top_fraction": float(args.top_fraction),
            "match_radius": "half each frozen CytoBridge graph cutoff",
            "permutations": int(args.permutations),
            "seed": int(args.seed),
        },
        "implementation": _model_biology_implementation(),
        "inputs": {
            "config": _artifact(config_path),
            "selection_manifest": _artifact(selection_manifest_path),
            "datasets": sources,
        },
        "outputs": {
            "summary": _artifact(summary_path),
            "cytobridge_top_edges": _artifact(cb_path),
            "commot_top_flows": _artifact(commot_path),
            "permutation_seed_audit": _artifact(null_path),
        },
    }
    _write_json(output / "manifest.json", manifest)


def _heading(axis, panel: str, title: str) -> None:
    axis.set_axis_off()
    axis.text(
        0, 0.55, panel, fontsize=14, fontweight="bold", va="center", color="black"
    )
    axis.text(
        0.055, 0.55, title, fontsize=12, fontweight="bold", va="center", color="black"
    )


def plot(args: argparse.Namespace) -> None:
    from CytoBridge.nonspatial import scnt_figure_style as style

    source = Path(args.aggregate_dir).expanduser().resolve()
    table_source = source / "panel_data" if (source / "panel_data").is_dir() else source
    source_manifest_path = next(
        (
            candidate
            for candidate in (
                source / "manifest.json",
                source / "aggregate_manifest.json",
                source / "figure_manifest.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if source_manifest_path is None:
        raise FileNotFoundError(f"no source manifest found under {source}")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    metrics = pd.read_csv(table_source / "cytobridge_external_metrics.csv")
    decisions = pd.read_csv(table_source / "main_figure_method_decisions.csv")
    selected = pd.read_csv(table_source / "selected_biological_pairs.csv")
    pathways = pd.read_csv(table_source / "selected_pair_commot_pathways.csv")
    ligand_receptors = pd.read_csv(table_source / "selected_pair_commot_lr.csv")
    target_evidence = pd.read_csv(table_source / "selected_pair_nichenet_targets.csv")
    included = (
        decisions.loc[decisions.include_in_main_figure, "external_method"]
        .astype(str)
        .tolist()
    )
    if not included:
        raise ValueError("no external method passed the frozen main-figure gate")
    included = [
        method
        for method in ("COMMOT", "CellAgentChat", "CellChat", "NicheNet")
        if method in included
    ]
    primary = metrics.loc[
        metrics.cytobridge_view.eq("CytoBridge exact message")
        & metrics.external_method.isin(included)
    ].copy()
    style.apply_style()
    plt.rcParams.update(
        {
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    plt.rcParams.update(
        {
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    dataset_order = list(FORMAL_DATASET_CONTRACTS)
    dataset_labels = {
        key: str(value["display_name"])
        for key, value in FORMAL_DATASET_CONTRACTS.items()
    }
    method_styles = {
        "CytoBridge exact message": ("#5B4B8A", "o", "CytoBridge"),
        "COMMOT": (METHOD_COLORS["COMMOT"], "s", "COMMOT"),
        "CellAgentChat": (METHOD_COLORS["CellAgentChat"], "D", "CellAgentChat"),
        "CellChat": (METHOD_COLORS["CellChat"], "^", "CellChat"),
        "NicheNet": (METHOD_COLORS["NicheNet"], "v", "NicheNet"),
    }
    fig = plt.figure(figsize=(11.69, 8.27))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.48],
        left=0.055,
        right=0.975,
        top=0.985,
        bottom=0.075,
        hspace=0.27,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[0.48, 0.52], wspace=0.22)
    panel_a = top[0].subgridspec(2, 1, height_ratios=[0.14, 1.0], hspace=0.04)
    panel_b = top[1].subgridspec(2, 1, height_ratios=[0.14, 1.0], hspace=0.04)
    panel_c = outer[1].subgridspec(2, 1, height_ratios=[0.12, 1.0], hspace=0.04)

    head_a = fig.add_subplot(panel_a[0])
    _heading(head_a, "a", "Cross-method concordance")
    axes_a = panel_a[1].subgridspec(1, 3, width_ratios=[0.22, 0.39, 0.39], wspace=0.24)
    ax_a_labels = fig.add_subplot(axes_a[0])
    ax_rho = fig.add_subplot(axes_a[1])
    ax_j = fig.add_subplot(axes_a[2])
    method_offsets = (
        np.asarray([0.0])
        if len(included) == 1
        else np.linspace(-0.15, 0.15, len(included))
    )
    y_positions = np.arange(len(dataset_order), dtype=float)
    for method_index, method in enumerate(included):
        table = (
            primary.loc[primary.external_method.eq(method)]
            .set_index("dataset")
            .reindex(dataset_order)
        )
        if table.metric_available.isna().any() or not table.metric_available.all():
            raise ValueError(f"included method lacks a complete metric grid: {method}")
        color, marker, label = method_styles[method]
        ys = y_positions + method_offsets[method_index]
        for axis, values in (
            (ax_rho, table.spearman_rho),
            (ax_j, table.top_jaccard),
        ):
            axis.scatter(
                values,
                ys,
                s=42,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=label,
                zorder=3,
            )
    for axis, xlabel in (
        (ax_rho, "Spearman rank correlation (ρ)"),
        (ax_j, "Top-20% directed-pair Jaccard"),
    ):
        axis.axvline(0, color="#AAB2B8", lw=0.7)
        axis.set_yticks(y_positions, [""] * len(dataset_order))
        axis.set_ylim(len(dataset_order) - 0.55, -0.55)
        axis.set_xlabel(xlabel)
        style.clean_axis(axis, grid=True)
    ax_a_labels.set_axis_off()
    ax_a_labels.set_xlim(0, 1)
    ax_a_labels.set_ylim(len(dataset_order) - 0.55, -0.55)
    for row_index, dataset in enumerate(dataset_order):
        ax_a_labels.text(
            0.98,
            row_index,
            dataset_labels[dataset],
            ha="right",
            va="center",
            fontsize=7.4,
            color="black",
        )
    head_a.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker=method_styles[method][1],
                color="none",
                markerfacecolor=method_styles[method][0],
                markeredgecolor="white",
                label=method_styles[method][2],
            )
            for method in included
        ],
        frameon=False,
        fontsize=7.5,
        ncol=min(2, len(included)),
        loc="center right",
        bbox_to_anchor=(1.0, 0.52),
    )
    head_b = fig.add_subplot(panel_b[0])
    _heading(head_b, "b", "Representative interactions")
    axes_b = panel_b[1].subgridspec(1, 2, width_ratios=[0.34, 0.66], wspace=0.025)
    ax_b_labels = fig.add_subplot(axes_b[0])
    ax_b = fig.add_subplot(axes_b[1])
    selected = selected.set_index("dataset").loc[dataset_order].reset_index()
    comparison_methods = ["CytoBridge exact message", *included]
    for row_index, row in selected.iterrows():
        values = [float(row[method]) for method in comparison_methods]
        ax_b.plot(
            [min(values), max(values)],
            [row_index, row_index],
            color="#C2C7CC",
            lw=1.4,
            zorder=1,
        )
        for method, value in zip(comparison_methods, values, strict=True):
            color, marker, _ = method_styles[method]
            ax_b.scatter(
                value,
                row_index,
                s=44,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    ax_b_labels.set_axis_off()
    ax_b_labels.set_xlim(0, 1)
    ax_b_labels.set_ylim(len(selected) - 0.55, -0.55)
    for row_index, row in enumerate(selected.itertuples()):
        ax_b_labels.text(
            0.98,
            row_index - 0.10,
            dataset_labels[row.dataset],
            ha="right",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="black",
        )
        ax_b_labels.text(
            0.98,
            row_index + 0.10,
            textwrap.fill(f"{row.sender_type} → {row.receiver_type}", width=27),
            ha="right",
            va="top",
            fontsize=6.2,
            color="black",
        )
    ax_b.set_yticks(range(len(selected)), [""] * len(selected))
    ax_b.set_ylim(len(selected) - 0.55, -0.55)
    minimum_rank = min(float(selected[method].min()) for method in comparison_methods)
    ax_b.set_xlim(max(0.0, minimum_rank - 0.06), 1.015)
    ax_b.axvline(0.8, color="#8A949C", lw=0.8, ls="--")
    ax_b.set_xlabel("Within-method directed-pair rank percentile")
    style.clean_axis(ax_b, grid=True)
    ax_b.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                label=label,
            )
            for color, marker, label in [
                method_styles[method] for method in comparison_methods
            ]
        ],
        frameon=False,
        ncol=len(comparison_methods),
        loc="upper left",
        fontsize=7.2,
    )
    head_c = fig.add_subplot(panel_c[0])
    _heading(head_c, "c", "Molecular resolution of selected programs")
    axes_c = panel_c[1].subgridspec(
        1, 4, width_ratios=[0.25, 0.82, 1.02, 0.82], wspace=0.34
    )
    ax_c_labels = fig.add_subplot(axes_c[0])
    ax_pathway = fig.add_subplot(axes_c[1])
    ax_lr = fig.add_subplot(axes_c[2])
    ax_target = fig.add_subplot(axes_c[3])
    molecular_axes = (ax_pathway, ax_lr, ax_target)
    group_bases = {
        dataset: 2.2 * (len(dataset_order) - 1 - index)
        for index, dataset in enumerate(dataset_order)
    }
    y_min, y_max = -0.72, max(group_bases.values()) + 0.72
    ax_c_labels.set_axis_off()
    ax_c_labels.set_xlim(0, 1)
    ax_c_labels.set_ylim(y_min, y_max)
    pathway_plot_rows: list[dict[str, object]] = []
    lr_plot_rows: list[dict[str, object]] = []
    target_plot_rows: list[dict[str, object]] = []
    evidence_rows = []

    for dataset in dataset_order:
        base = group_bases[dataset]
        pair = selected.loc[selected.dataset.eq(dataset)].iloc[0]
        ax_c_labels.text(
            0.98,
            base + 0.10,
            dataset_labels[dataset],
            ha="right",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="black",
        )
        ax_c_labels.text(
            0.98,
            base - 0.10,
            textwrap.fill(f"{pair.sender_type} → {pair.receiver_type}", width=22),
            ha="right",
            va="top",
            fontsize=5.8,
            color="black",
        )
        pair_rank = float(pair["CytoBridge exact message"])
        pathway_values = (
            pathways.loc[pathways.dataset.eq(dataset)]
            .sort_values("rank_within_pair")
            .head(2)
            .copy()
        )
        lr_values = (
            ligand_receptors.loc[ligand_receptors.dataset.eq(dataset)]
            .sort_values("rank_within_pair")
            .head(2)
            .copy()
        )
        target_values = (
            target_evidence.loc[target_evidence.dataset.eq(dataset)]
            .sort_values("rank_within_pair")
            .drop_duplicates("target")
            .head(2)
            .copy()
        )
        if target_values.empty:
            target_scope = "not_evaluable"
            target_text = "not evaluable"
        else:
            target_scope = (
                str(target_values.evidence_scope.iloc[0])
                if "evidence_scope" in target_values
                else "unspecified_legacy"
            )
            target_text = ", ".join(
                f"{int(row.rank_within_pair)}. {row.target}"
                for row in target_values.itertuples()
            )
        offsets = (0.26, -0.26)
        for offset, row in zip(offsets, pathway_values.itertuples(), strict=False):
            pathway_plot_rows.append(
                {
                    "y": base + offset,
                    "label": str(row.pathway),
                    "value": float(row.rank_within_pair),
                }
            )
        for offset, row in zip(offsets, lr_values.itertuples(), strict=False):
            lr_plot_rows.append(
                {
                    "y": base + offset,
                    "label": f"{row.ligand}-{row.receptor}",
                    "value": float(row.rank_within_pair),
                }
            )
        for offset, row in zip(offsets, target_values.itertuples(), strict=False):
            target_plot_rows.append(
                {
                    "y": base + offset,
                    "label": str(row.target),
                    "value": float(row.rank_within_pair),
                }
            )
        pathway_text = ", ".join(
            f"{int(row.rank_within_pair)}. {row.pathway}"
            for row in pathway_values.itertuples()
        )
        lr_text = ", ".join(
            f"{int(row.rank_within_pair)}. {row.ligand}-{row.receptor}"
            for row in lr_values.itertuples()
        )
        evidence_rows.append(
            {
                "dataset": dataset,
                "sender_type": str(pair.sender_type),
                "receiver_type": str(pair.receiver_type),
                "cytobridge_rank_percentile": pair_rank,
                "ordinal_rank_definition": "rank among positive external-method entries within the CytoBridge-selected cell-type pair; 1 is highest",
                "commot_pathway_ranks": pathway_text,
                "commot_ligand_receptor_ranks": lr_text,
                "nichenet_target_ranks": target_text,
                "nichenet_evidence_scope": target_scope,
                "biological_process": BIOLOGICAL_PROGRAMS[dataset],
            }
        )

    def _draw_molecular_axis(
        axis,
        rows: list[dict[str, object]],
        *,
        title: str,
        xlabel: str,
        color: str,
        marker: str,
    ) -> None:
        axis.set_xlim(0.72, 2.28)
        axis.set_ylim(y_min, y_max)
        axis.set_title(title, fontsize=9, color="black", pad=7)
        axis.set_xlabel(xlabel, fontsize=7.2, color="black")
        for row in rows:
            y_value = float(row["y"])
            value = float(row["value"])
            axis.scatter(
                value,
                y_value,
                s=25,
                marker=marker,
                color=color,
                edgecolor="black",
                linewidth=0.35,
                zorder=3,
            )
        axis.set_yticks(
            [float(row["y"]) for row in rows],
            [str(row["label"]) for row in rows],
            fontsize=7.0,
            color="black",
        )
        for index in range(len(dataset_order) - 1):
            upper = group_bases[dataset_order[index]]
            lower = group_bases[dataset_order[index + 1]]
            axis.axhline((upper + lower) / 2.0, color="#D8DDE1", lw=0.55)
        axis.tick_params(axis="x", labelsize=7.2, colors="black")
        axis.tick_params(axis="y", length=0, pad=2, colors="black")
        axis.set_xticks([1, 2], ["1", "2"])
        axis.axvline(1, color="#E4E7E9", lw=0.6, zorder=0)
        axis.axvline(2, color="#E4E7E9", lw=0.6, zorder=0)
        style.clean_axis(axis, grid=False)

    _draw_molecular_axis(
        ax_pathway,
        pathway_plot_rows,
        title="COMMOT pathways",
        xlabel="Rank within selected pair",
        color=METHOD_COLORS["COMMOT"],
        marker="s",
    )
    _draw_molecular_axis(
        ax_lr,
        lr_plot_rows,
        title="COMMOT ligand–receptor pairs",
        xlabel="Rank within selected pair",
        color="#4C78A8",
        marker="o",
    )
    _draw_molecular_axis(
        ax_target,
        target_plot_rows,
        title="NicheNet receiver targets",
        xlabel="Rank within selected pair",
        color=METHOD_COLORS["NicheNet"],
        marker="D",
    )
    evidence_path = output / "panel_c_model_to_biology.csv"
    pd.DataFrame(evidence_rows).to_csv(evidence_path, index=False)
    panel_filenames = (
        "cytobridge_external_metrics.csv",
        "main_figure_method_decisions.csv",
        "method_execution_status.csv",
        "selected_biological_pairs.csv",
        "selected_pair_commot_pathways.csv",
        "selected_pair_commot_lr.csv",
        "selected_pair_nichenet_targets.csv",
    )
    panel_dir = output / "panel_data"
    panel_dir.mkdir()
    for name in panel_filenames:
        shutil.copy2(table_source / name, panel_dir / name)
    pdf = output / "spatial_communication_consistency_a4.pdf"
    png = output / "spatial_communication_consistency_a4.png"
    style.save_figure(fig, pdf, png, dpi=320)
    plt.close(fig)
    decision_index = decisions.set_index("external_method")
    summary_text = ", ".join(
        f"{method} (median ρ={float(decision_index.loc[method, 'median_spearman_rho']):.2f}, median top-20% Jaccard={float(decision_index.loc[method, 'median_top_jaccard']):.2f})"
        for method in included
    )
    included_text = " and ".join(included)
    evidence_table = pd.DataFrame(evidence_rows)
    zebrafish_scope = str(
        evidence_table.loc[
            evidence_table.dataset.eq("zebrafish"), "nichenet_evidence_scope"
        ].item()
    )
    if zebrafish_scope == "one2one_bijective_all_confidence_sensitivity":
        zebrafish_note = " Zebrafish target annotations use the prespecified one-to-one all-confidence orthology sensitivity because the confidence-1 mapping was not complete across all receiver populations."
    elif zebrafish_scope == "not_evaluable":
        zebrafish_note = " Zebrafish NicheNet target evidence was not evaluable under the declared orthology contract."
    else:
        zebrafish_note = ""
    caption = f"**Five-dataset spatial communication consistency.** (a) Terminal-stage concordance compares the exact CytoBridge interaction contribution with {included_text} over complete directed cell-type-pair grids: {summary_text}. CellAgentChat used spatial mode, three fixed sampling seeds, and the gene-level representable subset of each dataset's accepted CytoBridge LR database. (b) Connected points show within-method ranks for one representative high-ranking off-diagonal interaction per dataset. (c) Molecular resolution of the selected interactions. CytoBridge specifies the sender-receiver interaction, COMMOT ranks pathway and ligand-receptor programs within that interaction, and NicheNet links candidate ligands from the same frozen LR universe to receiver target programs. Only ordinal positions 1 and 2 are shown; method-specific score magnitudes are deliberately omitted because they are not comparable across methods or datasets. The resulting programs are consistent with neuroepithelial patterning and extracellular-matrix-guided forebrain development in zebrafish, mesenchymal chondrogenic maturation in MOSTA, regenerative neuroglial remodeling in ARISTA, neuron-astrocyte structural and reactive programs in AdMouse, and valve extracellular-matrix remodeling in chicken heart. COMMOT and NicheNet provide molecular and regulatory resolution rather than native ligand-receptor parameters learned end-to-end by CytoBridge.{zebrafish_note} ARISTA and chicken are conserved-human-symbol proxy analyses. The agreement supports biological interpretability but is not a causal validation."
    (output / "caption.md").write_text(caption + "\n", encoding="utf-8")
    provenance = (
        "# Figure provenance\n\n"
        "## Source paths\n\n"
        f"- Frozen source directory: `{source}`\n"
        f"- Source manifest: `{source_manifest_path}`\n"
        f"- Source manifest SHA-256: `{sha256_file(source_manifest_path)}`\n"
        f"- Plotting script: `{Path(__file__).resolve()}`\n"
        f"- Plotting script SHA-256: `{sha256_file(Path(__file__).resolve())}`\n"
        "- Comparison unit: complete directed terminal-stage cell-type-pair grid.\n"
        "- CytoBridge view: exact learned interaction-message contribution.\n"
        "- Molecular interpretation: COMMOT pathway and LR scores plus NicheNet receiver-target evidence within the selected pair.\n"
        "\n## Rebuild\n\n"
        "```text\n"
        "python scripts/run_spatial_communication_consistency.py plot \\\n"
        f"  --aggregate-dir {source} \\\n"
        f"  --output-dir {output}\n"
        "```\n"
    )
    (output / "provenance.md").write_text(provenance, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source_manifest": _artifact(source_manifest_path),
        "included_methods": included,
        "gate": MAIN_FIGURE_GATE,
        "figure": {pdf.name: _artifact(pdf), png.name: _artifact(png)},
        "caption": _artifact(output / "caption.md"),
        "provenance": _artifact(output / "provenance.md"),
        "panel_data": {name: _artifact(panel_dir / name) for name in panel_filenames},
        "panel_c_model_to_biology": _artifact(evidence_path),
    }
    _write_json(output / "figure_manifest.json", manifest)


def _plot_model_biology_heatmap_legacy(args: argparse.Namespace) -> None:
    """Draw the reviewer-facing model-linked five-dataset figure."""

    from CytoBridge.nonspatial import scnt_figure_style as style

    aggregate_dir = Path(args.aggregate_dir).expanduser().resolve()
    selection_dir = Path(args.selection_dir).expanduser().resolve()
    molecular_dir = Path(args.molecular_panel_data_dir).expanduser().resolve()
    molecular_manifest_path = molecular_dir / "manifest.json"
    molecular_manifest = json.loads(molecular_manifest_path.read_text(encoding="utf-8"))
    if (
        molecular_manifest.get("workflow")
        != "five_dataset_model_biology_molecular_summary"
        or molecular_manifest.get("status") != "complete"
    ):
        raise ValueError("molecular panel manifest is not a complete formal summary")
    molecular_panel_path = molecular_dir / "model_biology_molecular_panel.csv"
    _verify_artifact_bytes(
        molecular_panel_path,
        dict(molecular_manifest["outputs"]["panel"]),
        label="model-biology molecular panel",
    )
    molecular_panel = pd.read_csv(molecular_panel_path).set_index("dataset")
    for manifest_key, filename in (
        ("rank_consistency", "molecular_rank_consistency.csv"),
        ("model_first_nichenet_chains", "model_first_nichenet_chains.csv"),
    ):
        _verify_artifact_bytes(
            molecular_dir / filename,
            dict(molecular_manifest["outputs"][manifest_key]),
            label=f"model-biology {manifest_key}",
        )
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    metrics = pd.read_csv(aggregate_dir / "cytobridge_external_metrics.csv")
    primary_metrics = metrics.loc[
        metrics["cytobridge_view"].eq("CytoBridge exact message")
        & metrics["external_method"].isin(["COMMOT", "CellAgentChat"])
    ].copy()
    commot = primary_metrics.loc[
        primary_metrics["external_method"].eq("COMMOT")
    ].set_index("dataset")
    cellagent = primary_metrics.loc[
        primary_metrics["external_method"].eq("CellAgentChat")
    ].set_index("dataset")
    support = pd.read_csv(
        selection_dir / "model_linked_external_support.csv"
    ).set_index("dataset")
    status = pd.read_csv(
        selection_dir / "model_linked_lr_selection_status.csv"
    ).set_index("dataset")
    pair_scores_path = aggregate_dir / "directed_pair_method_scores.csv"
    pair_scores = pd.read_csv(pair_scores_path)
    support_rows = support.reset_index()
    for method, prefix in (("COMMOT", "commot"), ("CellAgentChat", "cellagentchat")):
        method_pairs = pair_scores.loc[
            pair_scores["method"].eq(method)
            & pair_scores["available"].astype(str).str.casefold().isin({"true", "1"})
        ][["dataset", "sender_type", "receiver_type", "score", "rank_percentile"]]
        support_rows = support_rows.merge(
            method_pairs.rename(
                columns={
                    "score": f"{prefix}_pair_score",
                    "rank_percentile": f"{prefix}_pair_percentile",
                }
            ),
            on=["dataset", "sender_type", "receiver_type"],
            how="left",
        )
    support = support_rows.set_index("dataset")
    dataset_order = list(FORMAL_DATASET_CONTRACTS)
    labels = {
        key: str(value["display_name"])
        for key, value in FORMAL_DATASET_CONTRACTS.items()
    }
    if set(commot.index.astype(str)) != set(dataset_order) or set(
        cellagent.index.astype(str)
    ) != set(dataset_order):
        raise ValueError(
            "COMMOT/CellAgentChat metric tables do not cover five datasets"
        )
    if set(molecular_panel.index.astype(str)) != set(dataset_order):
        raise ValueError("molecular panel does not cover the five datasets")
    style.apply_style()
    plt.rcParams.update(
        {
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )

    def value_matrix(
        axis: plt.Axes,
        values: np.ndarray,
        row_labels: list[str],
        column_labels: list[str],
        *,
        colors: tuple[str, str, str],
        percentage: bool = False,
        cell_labels: np.ndarray | None = None,
        row_fontsize: float = 7.5,
        column_fontsize: float = 7.2,
        cell_fontsize: float = 7.2,
    ) -> None:
        """Draw a compact numeric matrix without decorative chart furniture."""

        colormap = LinearSegmentedColormap.from_list("figure_matrix", list(colors))
        colormap.set_bad("#F1F2F3")
        masked = np.ma.masked_invalid(np.asarray(values, dtype=float))
        axis.pcolormesh(
            np.arange(values.shape[1] + 1, dtype=float) - 0.5,
            np.arange(values.shape[0] + 1, dtype=float) - 0.5,
            masked,
            cmap=colormap,
            vmin=0,
            vmax=1,
            shading="flat",
            edgecolors="white",
            linewidth=0.8,
            antialiased=True,
        )
        axis.set_xlim(-0.5, values.shape[1] - 0.5)
        axis.set_ylim(values.shape[0] - 0.5, -0.5)
        axis.set_xticks(
            np.arange(len(column_labels)),
            column_labels,
            fontsize=column_fontsize,
        )
        axis.xaxis.tick_top()
        axis.set_yticks(
            np.arange(len(row_labels)),
            row_labels,
            fontsize=row_fontsize,
        )
        axis.tick_params(axis="both", which="major", length=0, pad=3)
        axis.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = float(values[row_index, column_index])
                if np.isfinite(value):
                    if cell_labels is not None:
                        label = str(cell_labels[row_index, column_index])
                    else:
                        label = f"{100 * value:.0f}" if percentage else f"{value:.2f}"
                    fontweight = "bold" if value >= 0.95 else "normal"
                    color = "black"
                else:
                    label = "N/A"
                    fontweight = "normal"
                    color = "#687078"
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    fontweight=fontweight,
                    color=color,
                )

    rank_metrics = pd.read_csv(molecular_dir / "molecular_rank_consistency.csv")
    chains = pd.read_csv(molecular_dir / "model_first_nichenet_chains.csv")

    def panel_heading(axis: plt.Axes, label: str, title: str) -> None:
        axis.set_axis_off()
        axis.text(
            0.0,
            0.54,
            label,
            fontsize=14,
            fontweight="bold",
            color="black",
            va="center",
        )
        axis.text(
            0.09,
            0.54,
            title,
            fontsize=12,
            fontweight="bold",
            color="black",
            va="center",
        )

    fig = plt.figure(figsize=style.A4_PORTRAIT)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.44, 0.56),
        left=0.095,
        right=0.97,
        top=0.965,
        bottom=0.065,
        hspace=0.19,
    )
    top = outer[0].subgridspec(
        2,
        2,
        height_ratios=(0.13, 0.87),
        width_ratios=(0.50, 0.50),
        hspace=0.06,
        wspace=0.36,
    )
    panel_heading(fig.add_subplot(top[0, 0]), "a", "Directed-pair consistency")
    panel_heading(fig.add_subplot(top[0, 1]), "b", "Model-first interaction axes")

    ax_a = fig.add_subplot(top[1, 0])
    a_values = np.full((len(dataset_order), 4), np.nan, dtype=float)
    for row_index, dataset in enumerate(dataset_order):
        for method_index, table in enumerate((commot, cellagent)):
            row = table.loc[dataset]
            available = str(row.metric_available).casefold() in {"true", "1"}
            if available:
                a_values[row_index, method_index] = float(row.spearman_rho)
                a_values[row_index, method_index + 2] = float(row.top_jaccard)
    value_matrix(
        ax_a,
        a_values,
        [labels[dataset] for dataset in dataset_order],
        ["COMMOT", "CellAgentChat\nproxy", "COMMOT", "CellAgentChat\nproxy"],
        colors=("#F5F9F8", "#B9DDD8", "#4BA99E"),
        row_fontsize=7.8,
        column_fontsize=7.2,
        cell_fontsize=7.4,
    )
    ax_a.axvline(1.5, color="white", linewidth=3.0)
    ax_a.text(
        0.25,
        1.08,
        "Rank correlation (ρ)",
        transform=ax_a.transAxes,
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="black",
    )
    ax_a.text(
        0.75,
        1.08,
        "Top-20% Jaccard",
        transform=ax_a.transAxes,
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="black",
    )
    ax_b = fig.add_subplot(top[1, 1])
    b_values = np.full((len(dataset_order), 4), np.nan, dtype=float)
    b_cell_labels = np.full((len(dataset_order), 4), "N/A", dtype=object)
    b_labels: list[str] = []
    for row_index, dataset in enumerate(dataset_order):
        if dataset not in support.index:
            b_labels.append(f"{labels[dataset]}\nnot evaluable")
            continue
        row = support.loc[dataset]
        panel_row = molecular_panel.loc[dataset]
        lr_count = int(panel_row.within_pair_lr_count)
        cytobridge_rank = int(panel_row.cytobridge_within_pair_rank)
        commot_rank = int(panel_row.commot_within_pair_rank)

        def within_pair_percentile(rank: int) -> float:
            if lr_count <= 1:
                return 1.0
            return 1.0 - (rank - 1) / (lr_count - 1)

        b_values[row_index] = [
            within_pair_percentile(cytobridge_rank),
            within_pair_percentile(commot_rank),
            float(row.commot_pair_percentile),
            float(row.cellagentchat_pair_percentile),
        ]
        b_cell_labels[row_index] = [
            f"{cytobridge_rank}/{lr_count}",
            f"{commot_rank}/{lr_count}",
            f"{100 * float(row.commot_pair_percentile):.0f}%",
            f"{100 * float(row.cellagentchat_pair_percentile):.0f}%",
        ]
        b_labels.append(
            f"{labels[dataset]}\n{str(row.ligand).upper()}–{str(row.receptor).upper()}"
            f" · {panel_row.cytobridge_pathway}"
        )
    value_matrix(
        ax_b,
        b_values,
        b_labels,
        [
            "CytoBridge\nLR within pair",
            "COMMOT\nsame LR",
            "COMMOT\nsame pair",
            "CellAgentChat proxy\nsame pair",
        ],
        colors=("#F8F6FA", "#D9D1E8", "#8B79B6"),
        percentage=True,
        cell_labels=b_cell_labels,
        row_fontsize=7.0,
        column_fontsize=6.9,
        cell_fontsize=7.0,
    )
    ax_b.text(
        0.5,
        -0.10,
        "Cell shading: within-method percentile (100 = highest)",
        transform=ax_b.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        color="black",
    )
    c_block = outer[1].subgridspec(2, 1, height_ratios=(0.16, 0.84), hspace=0.08)
    panel_heading(
        fig.add_subplot(c_block[0]),
        "c",
        "Molecular rank consistency and NicheNet receiver-target support",
    )
    c_body = c_block[1].subgridspec(1, 2, width_ratios=(0.46, 0.54), wspace=0.18)

    ax_molecular = fig.add_subplot(c_body[0])
    molecular_values = np.full((len(dataset_order), 4), np.nan, dtype=float)
    for row_index, dataset in enumerate(dataset_order):
        for method_index, method in enumerate(("COMMOT", "NicheNet")):
            local = rank_metrics.loc[
                rank_metrics["dataset"].eq(dataset)
                & rank_metrics["external_method"].eq(method)
                & rank_metrics["available"]
                .astype(str)
                .str.casefold()
                .isin({"true", "1"})
            ]
            if local.empty:
                continue
            molecular_values[row_index, method_index] = float(
                local.iloc[0].spearman_rho
            )
            molecular_values[row_index, method_index + 2] = float(
                local.iloc[0].top_jaccard
            )
    value_matrix(
        ax_molecular,
        molecular_values,
        [labels[dataset] for dataset in dataset_order],
        ["COMMOT", "NicheNet", "COMMOT", "NicheNet"],
        colors=("#FBF7F5", "#F3C8BB", "#E48162"),
        row_fontsize=7.8,
        column_fontsize=7.2,
        cell_fontsize=7.4,
    )
    ax_molecular.axvline(1.5, color="white", linewidth=3.0)
    ax_molecular.text(
        0.25,
        1.06,
        "LR×pair rank correlation (ρ)",
        transform=ax_molecular.transAxes,
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="black",
    )
    ax_molecular.text(
        0.75,
        1.06,
        "Top-20% overlap",
        transform=ax_molecular.transAxes,
        ha="center",
        fontsize=8.0,
        fontweight="bold",
        color="black",
    )
    ax_chain = fig.add_subplot(c_body[1])
    ax_chain.set_axis_off()
    ax_chain.text(
        0.00,
        1.02,
        "NicheNet receiver-target support for model-ranked axes",
        transform=ax_chain.transAxes,
        fontsize=9.0,
        fontweight="bold",
        color="black",
        va="bottom",
    )
    ax_chain.text(
        0.22,
        0.945,
        "CytoBridge LR / pathway",
        transform=ax_chain.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color="black",
        va="center",
    )
    ax_chain.text(
        0.60,
        0.935,
        "COMMOT",
        transform=ax_chain.transAxes,
        fontsize=7.0,
        fontweight="bold",
        color="black",
        va="center",
    )
    ax_chain.text(
        0.79,
        0.945,
        "NicheNet\nreceiver target",
        transform=ax_chain.transAxes,
        fontsize=6.4,
        fontweight="bold",
        color="black",
        va="center",
    )
    y_positions = np.linspace(0.82, 0.10, len(dataset_order))
    for row_index, (y_value, dataset) in enumerate(
        zip(y_positions, dataset_order, strict=True)
    ):
        local = chains.loc[chains["dataset"].eq(dataset)].sort_values(
            "receiver_target_rank"
        )
        ax_chain.axhline(
            0.91 - row_index * 0.18,
            color="#D4D9DD",
            lw=0.55,
            xmin=0,
            xmax=1,
        )
        ax_chain.text(
            0.00,
            y_value,
            labels[dataset],
            transform=ax_chain.transAxes,
            fontsize=7.4,
            fontweight="bold",
            color="black",
            va="center",
        )
        if dataset == "admouse":
            ax_chain.text(
                0.22,
                y_value,
                "N/A",
                transform=ax_chain.transAxes,
                fontsize=7.2,
                color="#687078",
                va="center",
            )
            continue
        if local.empty:
            panel_row = molecular_panel.loc[dataset]
            ax_chain.text(
                0.22,
                y_value + 0.023,
                (
                    f"rank 1 · {str(panel_row.ligand).upper()}–"
                    f"{str(panel_row.receptor).upper()}"
                ),
                transform=ax_chain.transAxes,
                fontsize=7.1,
                fontweight="bold",
                color="#5B4B8A",
                va="center",
            )
            ax_chain.text(
                0.22,
                y_value - 0.035,
                str(panel_row.cytobridge_pathway),
                transform=ax_chain.transAxes,
                fontsize=6.6,
                color="black",
                va="center",
            )
            commot_percentile = float(panel_row.commot_exact_axis_percentile)
            targets = "N/A"
            receiver = "NicheNet unavailable"
        else:
            first = local.iloc[0]
            targets = " / ".join(local["receiver_target"].astype(str).head(3))
            receiver = str(first.receiver_type)
            commot_percentile = float(first.commot_percentile)
            ax_chain.text(
                0.22,
                y_value + 0.023,
                (
                    f"rank {int(first.cytobridge_global_rank)} · "
                    f"{str(first.ligand).upper()}–{str(first.receptor).upper()}"
                ),
                transform=ax_chain.transAxes,
                fontsize=7.1,
                fontweight="bold",
                color="#5B4B8A",
                va="center",
            )
            ax_chain.text(
                0.22,
                y_value - 0.035,
                str(first.pathways),
                transform=ax_chain.transAxes,
                fontsize=6.6,
                color="black",
                va="center",
            )
        ax_chain.scatter(
            [0.61],
            [y_value + 0.010],
            transform=ax_chain.transAxes,
            s=23,
            marker="s",
            color="#2A9D8F",
            edgecolor="white",
            linewidth=0.4,
            clip_on=False,
        )
        ax_chain.text(
            0.64,
            y_value + 0.010,
            f"{100 * commot_percentile:.1f}%",
            transform=ax_chain.transAxes,
            fontsize=6.8,
            color="black",
            va="center",
        )
        ax_chain.annotate(
            "",
            xy=(0.78, y_value),
            xytext=(0.72, y_value),
            xycoords=ax_chain.transAxes,
            textcoords=ax_chain.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#E76F51",
                "linewidth": 0.9,
                "mutation_scale": 7,
            },
        )
        ax_chain.text(
            0.80,
            y_value + 0.022,
            targets,
            transform=ax_chain.transAxes,
            fontsize=6.8,
            color="#C95840",
            fontweight="bold",
            va="center",
        )
        ax_chain.text(
            0.80,
            y_value - 0.038,
            receiver,
            transform=ax_chain.transAxes,
            fontsize=6.1,
            color="#4E565D" if local.empty else "black",
            va="center",
        )
    ax_chain.axhline(0.01, color="#D4D9DD", lw=0.55, xmin=0, xmax=1)

    pdf = output / "spatial_communication_model_biology_a4.pdf"
    png = output / "spatial_communication_model_biology_a4.png"
    style.save_figure(fig, pdf, png, dpi=320)
    plt.close(fig)
    panel_output = output / "panel_data"
    panel_output.mkdir()
    panel_metrics_path = panel_output / "global_pair_metrics.csv"
    panel_support_path = panel_output / "model_linked_external_support.csv"
    panel_status_path = panel_output / "model_linked_lr_selection_status.csv"
    panel_molecular_path = panel_output / "model_biology_molecular_panel.csv"
    panel_rank_path = panel_output / "molecular_rank_consistency.csv"
    panel_chain_path = panel_output / "model_first_nichenet_chains.csv"
    primary_metrics.to_csv(panel_metrics_path, index=False)
    support.reset_index().to_csv(panel_support_path, index=False)
    status.reset_index().to_csv(panel_status_path, index=False)
    molecular_panel.reset_index().to_csv(panel_molecular_path, index=False)
    rank_metrics.to_csv(panel_rank_path, index=False)
    chains.to_csv(panel_chain_path, index=False)
    caption = (
        "**Five-dataset model-linked communication consistency.** (a) Terminal-stage "
        "directed cell-type-pair concordance is shown for COMMOT and for the frozen "
        "current-database CellAgentChat proxy across all five datasets. (b) For each "
        "dataset, CytoBridge alone selected the highest "
        "abundance-normalized exact-message magnitude × sender-ligand × "
        "receiver-receptor activity axis from the frozen LR database crossed with "
        "every off-diagonal model pair; neither COMMOT nor the CellAgentChat proxy entered "
        "selection. COMMOT was evaluated on the complete zero-filled LR×pair "
        "universe; the CellAgentChat proxy shows its native CTPS rank for the same selected "
        "cell-type pair. AdMouse had no "
        "axis with at least 10 active model-linked edges under its 347-gene panel and "
        "learned-edge threshold. (c) Molecular consistency is measured over the "
        "complete jointly positive LR×directed-cell-pair universe. Matrix cells "
        "report Spearman rank agreement and top-20% Jaccard overlap with COMMOT or "
        "NicheNet. The right-hand rows begin from the highest-ranked CytoBridge axis "
        "with formal NicheNet coverage in each dataset and connect its LR/pathway "
        "annotation to NicheNet-ranked candidate receiver targets; where NicheNet is not "
        "available, the model-first axis and its COMMOT percentile remain visible. "
        "External methods are evaluated only after CytoBridge ranking."
    )
    caption_path = output / "caption.md"
    caption_path.write_text(caption + "\n", encoding="utf-8")
    implementation = _model_biology_implementation()
    provenance_path = output / "provenance.md"
    provenance_path.write_text(
        "\n".join(
            [
                "# Five-dataset model-linked communication figure provenance",
                "",
                "## Scope",
                "",
                "CytoBridge alone selects each LR-compatible axis; COMMOT and the "
                "frozen current-database CellAgentChat proxy are evaluated only after "
                "selection. AdMouse remains "
                "explicitly not evaluable under the shared minimum-active-edge rule.",
                "",
                "## Source paths",
                "",
                f"- aggregate manifest: `{aggregate_dir / 'manifest.json'}`",
                f"- selection manifest: `{selection_dir / 'manifest.json'}`",
                f"- molecular summary manifest: `{molecular_manifest_path}`",
                "- compact visual inputs: `panel_data/` in this bundle",
                "",
                "## Figure implementation",
                "",
                *[
                    f"- `{relative}`: `{record['sha256']}`"
                    for relative, record in implementation["files"].items()
                ],
                f"- aggregate SHA-256: `{implementation['aggregate_sha256']}`",
                f"- PDF SHA-256: `{sha256_file(pdf)}`",
                f"- PNG SHA-256: `{sha256_file(png)}`",
                "",
                "## Rebuild",
                "",
                "Run `scripts/run_spatial_communication_consistency.py "
                "plot-model-biology` with the aggregate, selection, and molecular "
                "summary directories. The figure does not require spatial "
                "coordinates or the original large H5AD files.",
                "",
                "## Interpretation boundary",
                "",
                "LR labels are post-hoc molecular compatibility annotations of exact "
                "learned GNN messages. They do not assert that the GNN natively "
                "identifies a unique biochemical ligand-receptor pair, and the figure "
                "does not establish causal signaling.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "workflow": "five_dataset_model_linked_communication_figure",
        "implementation": implementation,
        "inputs": {
            "aggregate_manifest": _artifact(aggregate_dir / "manifest.json"),
            "selection_manifest": _artifact(selection_dir / "manifest.json"),
            "molecular_summary_manifest": _artifact(molecular_manifest_path),
        },
        "figure": {pdf.name: _artifact(pdf), png.name: _artifact(png)},
        "caption": _artifact(caption_path),
        "provenance": _artifact(provenance_path),
        "panel_data": {
            "global_metrics": _artifact(panel_metrics_path),
            "external_support": _artifact(panel_support_path),
            "selection_status": _artifact(panel_status_path),
            "molecular_summary": _artifact(panel_molecular_path),
            "molecular_rank_consistency": _artifact(panel_rank_path),
            "model_first_nichenet_chains": _artifact(panel_chain_path),
        },
    }
    _write_json(output / "figure_manifest.json", manifest)


def plot_model_biology(args: argparse.Namespace) -> None:
    """Draw a four-dataset figure from the frozen five-dataset audit evidence."""

    from matplotlib.lines import Line2D

    from CytoBridge.nonspatial import scnt_figure_style as style

    aggregate_dir = Path(args.aggregate_dir).expanduser().resolve()
    selection_dir = Path(args.selection_dir).expanduser().resolve()
    molecular_dir = Path(args.molecular_panel_data_dir).expanduser().resolve()
    molecular_manifest_path = molecular_dir / "manifest.json"
    molecular_manifest = json.loads(molecular_manifest_path.read_text(encoding="utf-8"))
    if (
        molecular_manifest.get("workflow")
        != "five_dataset_model_biology_molecular_summary"
        or molecular_manifest.get("status") != "complete"
    ):
        raise ValueError("molecular panel manifest is not a complete formal summary")
    for manifest_key, filename in (
        ("panel", "model_biology_molecular_panel.csv"),
        ("rank_consistency", "molecular_rank_consistency.csv"),
        ("model_first_nichenet_chains", "model_first_nichenet_chains.csv"),
    ):
        _verify_artifact_bytes(
            molecular_dir / filename,
            dict(molecular_manifest["outputs"][manifest_key]),
            label=f"model-biology {manifest_key}",
        )

    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)

    metrics = pd.read_csv(aggregate_dir / "cytobridge_external_metrics.csv")
    primary_metrics = metrics.loc[
        metrics["cytobridge_view"].eq("CytoBridge exact message")
        & metrics["external_method"].isin(["COMMOT", "CellAgentChat"])
    ].copy()
    commot = primary_metrics.loc[
        primary_metrics["external_method"].eq("COMMOT")
    ].set_index("dataset")
    cellagent = primary_metrics.loc[
        primary_metrics["external_method"].eq("CellAgentChat")
    ].set_index("dataset")

    support = pd.read_csv(selection_dir / "model_linked_external_support.csv")
    status = pd.read_csv(selection_dir / "model_linked_lr_selection_status.csv")
    pair_scores = pd.read_csv(aggregate_dir / "directed_pair_method_scores.csv")
    for method, prefix in (("COMMOT", "commot"), ("CellAgentChat", "cellagentchat")):
        method_pairs = pair_scores.loc[
            pair_scores["method"].eq(method)
            & pair_scores["available"].astype(str).str.casefold().isin({"true", "1"})
        ][["dataset", "sender_type", "receiver_type", "score", "rank_percentile"]]
        support = support.merge(
            method_pairs.rename(
                columns={
                    "score": f"{prefix}_pair_score",
                    "rank_percentile": f"{prefix}_pair_percentile",
                }
            ),
            on=["dataset", "sender_type", "receiver_type"],
            how="left",
            validate="one_to_one",
        )
    support = support.set_index("dataset")
    status = status.set_index("dataset")
    molecular_panel = pd.read_csv(
        molecular_dir / "model_biology_molecular_panel.csv"
    ).set_index("dataset")
    rank_metrics = pd.read_csv(molecular_dir / "molecular_rank_consistency.csv")
    chains = pd.read_csv(molecular_dir / "model_first_nichenet_chains.csv")
    zebrafish_chains = chains.loc[
        chains["dataset"].astype(str).eq("zebrafish")
    ].sort_values("receiver_target_rank")
    if zebrafish_chains.empty:
        raise ValueError("NicheNet chain is missing for zebrafish")
    zebrafish_scope = str(zebrafish_chains.iloc[0].get("nichenet_evidence_scope", ""))
    zebrafish_is_strict_proxy = "strict_confidence1" in zebrafish_scope.casefold()
    zebrafish_is_sensitivity = (
        zebrafish_is_strict_proxy or "sensitivity" in zebrafish_scope.casefold()
    )
    zebrafish_aupr = pd.to_numeric(
        zebrafish_chains.get(
            "nichenet_corrected_aupr",
            pd.Series(np.nan, index=zebrafish_chains.index),
        ),
        errors="coerce",
    ).dropna()
    zebrafish_corrected_aupr = (
        float(zebrafish_aupr.iloc[0]) if not zebrafish_aupr.empty else np.nan
    )
    if zebrafish_is_strict_proxy:
        aupr_parenthetical = (
            f" (corrected AUPR = {zebrafish_corrected_aupr:.3f})"
            if np.isfinite(zebrafish_corrected_aupr)
            else ""
        )
        zebrafish_artwork_note = (
            "† Zebrafish NicheNet uses confidence-1 one-to-one orthology with the "
            "mouse prior (cross-species sensitivity)."
        )
        zebrafish_caption_clause = (
            "Zebrafish is a cross-species sensitivity analysis using a prespecified "
            "Ensembl 116 one-to-one, confidence-1, symbol-bijective mapping and the "
            "official mouse NicheNet prior"
            "; it is excluded from pooled and primary NicheNet claims and is not a "
            "native zebrafish regulatory prior. The corrected AUPR"
            f"{aupr_parenthetical.replace(' (corrected AUPR = ', ' = ').rstrip(')')} "
            "measures how well mapped ligand activity recovers receiver-response "
            "genes; it is not an LR-axis confidence score or a cross-dataset effect "
            "size. NicheNet supports the ligand-to-receiver-target annotation, not "
            "independent validation of the receptor or the complete LR axis. "
        )
    elif zebrafish_is_sensitivity:
        aupr_parenthetical = (
            f" (corrected AUPR = {zebrafish_corrected_aupr:.3f})"
            if np.isfinite(zebrafish_corrected_aupr)
            else ""
        )
        zebrafish_artwork_note = (
            "† Cross-species NicheNet-v2 orthology sensitivity"
            f"{aupr_parenthetical}; excluded from pooled NicheNet claims."
        )
        zebrafish_caption_clause = (
            "Zebrafish is a cross-species sensitivity analysis using a prespecified "
            "Ensembl 116 one-to-one, confidence-unfiltered, symbol-bijective mapping "
            "and the official mouse NicheNet prior"
            f"{aupr_parenthetical}; it is excluded from pooled NicheNet claims and is "
            "not a native zebrafish regulatory prior or a primary NicheNet claim. "
        )
    else:
        zebrafish_artwork_note = ""
        zebrafish_caption_clause = (
            "Zebrafish uses the prespecified cross-species NicheNet evidence scope "
            f"'{zebrafish_scope}' and the official mouse prior; it is not a native "
            "zebrafish regulatory prior. "
        )

    audit_dataset_order = list(FORMAL_DATASET_CONTRACTS)
    dataset_order = [dataset for dataset in audit_dataset_order if dataset != "admouse"]
    labels = {
        key: str(value["display_name"])
        for key, value in FORMAL_DATASET_CONTRACTS.items()
    }
    if set(commot.index.astype(str)) != set(audit_dataset_order) or set(
        cellagent.index.astype(str)
    ) != set(audit_dataset_order):
        raise ValueError("COMMOT/CellAgentChat metrics do not cover five datasets")
    if set(molecular_panel.index.astype(str)) != set(audit_dataset_order):
        raise ValueError("molecular panel does not cover five datasets")
    if not set(dataset_order).issubset(set(support.index.astype(str))):
        raise ValueError("external support does not cover the four displayed datasets")

    style.apply_style()
    plt.rcParams.update(
        {
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    accent = "#277F7A"
    secondary_grey = "#9AA1A6"
    dark_grey = "#555B60"
    grid_color = "#D9DEE2"

    def panel_heading(axis: plt.Axes, label: str, title: str) -> None:
        axis.set_axis_off()
        axis.text(0.0, 0.52, label, fontsize=14, fontweight="bold", va="center")
        axis.text(0.065, 0.52, title, fontsize=12, fontweight="bold", va="center")

    def dot_axis(
        axis: plt.Axes,
        *,
        metric: str,
        external_methods: tuple[str, ...],
        tables: dict[str, pd.DataFrame],
        title: str,
        x_label: str,
        x_limits: tuple[float, float],
        show_y: bool,
    ) -> None:
        y_base = np.arange(len(dataset_order), dtype=float)
        offsets = np.linspace(-0.10, 0.10, len(external_methods))
        marker_map = {"COMMOT": "s", "CellAgentChat": "D", "NicheNet": "^"}
        color_map = {
            "COMMOT": accent,
            "CellAgentChat": secondary_grey,
            "NicheNet": dark_grey,
        }
        for offset, method in zip(offsets, external_methods, strict=True):
            table = tables[method]
            xs: list[float] = []
            ys: list[float] = []
            for row_index, dataset in enumerate(dataset_order):
                if dataset not in table.index:
                    continue
                row = table.loc[dataset]
                if "metric_available" in row.index:
                    available = str(row.metric_available).casefold() in {"true", "1"}
                else:
                    available = str(row.available).casefold() in {"true", "1"}
                value = float(row[metric]) if available else float("nan")
                if np.isfinite(value):
                    xs.append(value)
                    ys.append(row_index + offset)
            axis.scatter(
                xs,
                ys,
                s=42,
                marker=marker_map[method],
                color=color_map[method],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label="CellAgentChat proxy" if method == "CellAgentChat" else method,
            )
        axis.set_xlim(*x_limits)
        axis.set_ylim(len(dataset_order) - 0.5, -0.5)
        axis.set_yticks(
            y_base,
            [labels[dataset] for dataset in dataset_order] if show_y else [],
        )
        axis.set_title(title, fontsize=9.2, pad=7)
        axis.set_xlabel(x_label, fontsize=8.5)
        axis.grid(axis="x", color=grid_color, linewidth=0.6)
        axis.axvline(0.0, color="#9AA3AA", linewidth=0.8, zorder=1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7.5)

    fig = plt.figure(figsize=style.A4_PORTRAIT)
    outer = fig.add_gridspec(
        6,
        1,
        height_ratios=(0.065, 0.225, 0.065, 0.255, 0.065, 0.305),
        left=0.12,
        right=0.965,
        top=0.97,
        bottom=0.07,
        hspace=0.28,
    )

    panel_heading(fig.add_subplot(outer[0]), "a", "Directed-pair consistency")
    a_grid = outer[1].subgridspec(1, 2, wspace=0.27)
    ax_a_rho = fig.add_subplot(a_grid[0])
    ax_a_jaccard = fig.add_subplot(a_grid[1])
    global_tables = {"COMMOT": commot, "CellAgentChat": cellagent}
    dot_axis(
        ax_a_rho,
        metric="spearman_rho",
        external_methods=("COMMOT", "CellAgentChat"),
        tables=global_tables,
        title="Rank agreement",
        x_label="Spearman rank correlation (ρ)",
        x_limits=(-0.03, 1.05),
        show_y=True,
    )
    dot_axis(
        ax_a_jaccard,
        metric="top_jaccard",
        external_methods=("COMMOT", "CellAgentChat"),
        tables=global_tables,
        title="Top-ranked pair overlap",
        x_label="Top-20% directed-pair Jaccard",
        x_limits=(-0.03, 1.05),
        show_y=False,
    )
    ax_a_rho.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.11),
        ncol=2,
        fontsize=7.2,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    panel_heading(fig.add_subplot(outer[2]), "b", "CytoBridge-selected LR axes")
    ax_b = fig.add_subplot(outer[3])
    ax_b.set_axis_off()
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(-0.55, 4.62)
    ax_b.text(
        0.44,
        4.38,
        "CytoBridge selection",
        fontsize=7.8,
        fontweight="bold",
        ha="center",
    )
    ax_b.text(
        0.885,
        4.38,
        "COMMOT comparison",
        fontsize=7.8,
        fontweight="bold",
        ha="center",
    )
    ax_b.plot([0.13, 0.745], [4.18, 4.18], color="black", linewidth=0.55)
    ax_b.plot([0.77, 1.0], [4.18, 4.18], color="black", linewidth=0.55)
    ax_b.text(0.00, 3.91, "Dataset", fontsize=7.1, fontweight="bold")
    ax_b.text(0.13, 3.91, "Sender", fontsize=7.1, fontweight="bold")
    ax_b.text(0.29, 3.91, "Receiver", fontsize=7.1, fontweight="bold")
    ax_b.text(
        0.565,
        3.99,
        "Selected\nLR axis",
        fontsize=6.8,
        fontweight="bold",
        ha="center",
        va="top",
        linespacing=1.1,
    )
    ax_b.text(
        0.705,
        3.99,
        "Database\npathway",
        fontsize=6.8,
        fontweight="bold",
        ha="center",
        va="top",
        linespacing=1.1,
    )
    ax_b.text(
        0.835,
        3.99,
        "Same-pair\nLR rank",
        fontsize=6.8,
        fontweight="bold",
        ha="center",
        va="top",
        linespacing=1.1,
    )
    ax_b.text(
        0.95,
        3.99,
        "Pair\npercentile",
        fontsize=6.8,
        fontweight="bold",
        ha="center",
        va="top",
        linespacing=1.1,
    )
    ax_b.plot([0.0, 1.0], [3.50, 3.50], color="black", linewidth=0.75)
    receiver_display = {
        "zebrafish": "Musculature / YSL",
        "mosta": "Connective tissue",
        "arista": "tlNBL",
        "chicken_heart": "Valve cells",
    }
    for row_index, dataset in enumerate(dataset_order):
        row = support.loc[dataset]
        panel_row = molecular_panel.loc[dataset]
        y = float(len(dataset_order) - 1 - row_index)
        if row_index:
            ax_b.plot(
                [0.0, 1.0],
                [y + 0.50, y + 0.50],
                color=grid_color,
                linewidth=0.55,
                zorder=0,
            )
        ax_b.text(
            0.00,
            y,
            labels[dataset],
            fontsize=7.5,
            fontweight="bold",
            va="center",
            color="black",
        )
        ax_b.text(
            0.13,
            y,
            str(row.sender_type),
            fontsize=7.2,
            va="center",
            color="black",
        )
        ax_b.text(
            0.29,
            y,
            receiver_display[dataset],
            fontsize=7.2,
            va="center",
            color="black",
        )
        ax_b.text(
            0.51,
            y,
            f"{str(row.ligand).upper()}–{str(row.receptor).upper()}",
            fontsize=7.5,
            fontweight="bold",
            color="black",
            va="center",
        )
        ax_b.text(
            0.66,
            y,
            str(panel_row.cytobridge_pathway),
            fontsize=7.2,
            color="black",
            va="center",
        )
        ax_b.text(
            0.835,
            y,
            f"{int(panel_row.commot_within_pair_rank)} / "
            f"{int(panel_row.within_pair_lr_count):,}",
            fontsize=7.2,
            ha="center",
            va="center",
            color="black",
        )
        ax_b.text(
            0.95,
            y,
            f"{100 * float(row.commot_pair_percentile):.1f}",
            fontsize=7.2,
            ha="center",
            color="black",
            va="center",
        )
    ax_b.plot([0.0, 1.0], [-0.50, -0.50], color="black", linewidth=0.75)

    panel_heading(
        fig.add_subplot(outer[4]),
        "c",
        "NicheNet receiver targets for CytoBridge-ranked axes",
    )
    ax_c_chain = fig.add_subplot(outer[5])
    ax_c_chain.set_axis_off()
    ax_c_chain.set_xlim(0.0, 1.0)
    ax_c_chain.set_ylim(-1.04, 4.38)
    ax_c_chain.text(
        0.42,
        4.18,
        "CytoBridge ranking",
        fontsize=7.8,
        fontweight="bold",
        ha="center",
    )
    ax_c_chain.text(
        0.865,
        4.18,
        "COMMOT and NicheNet results",
        fontsize=7.8,
        fontweight="bold",
        ha="center",
    )
    ax_c_chain.plot([0.14, 0.70], [3.99, 3.99], color="black", linewidth=0.55)
    ax_c_chain.plot([0.72, 1.0], [3.99, 3.99], color="black", linewidth=0.55)
    ax_c_chain.text(0.00, 3.72, "Dataset", fontsize=7.0, fontweight="bold")
    ax_c_chain.text(
        0.14,
        3.79,
        "Directed pair\n(sender; receiver)",
        fontsize=6.8,
        fontweight="bold",
        va="top",
        linespacing=1.1,
    )
    ax_c_chain.text(
        0.40,
        3.79,
        "LR axis\nDatabase pathway",
        fontsize=6.8,
        fontweight="bold",
        va="top",
        linespacing=1.1,
    )
    ax_c_chain.text(
        0.665,
        3.86,
        "CytoBridge\nLR × directed-\npair rank*",
        fontsize=6.5,
        fontweight="bold",
        ha="center",
        va="top",
        linespacing=1.0,
    )
    ax_c_chain.text(
        0.78,
        3.86,
        "COMMOT\nsame-axis\npercentile",
        fontsize=6.5,
        fontweight="bold",
        ha="center",
        va="top",
        linespacing=1.0,
    )
    ax_c_chain.text(
        0.925,
        3.86,
        "NicheNet-predicted\nreceiver\ntargets",
        fontsize=6.5,
        fontweight="bold",
        ha="center",
        va="top",
        linespacing=1.0,
    )
    ax_c_chain.plot([0.0, 1.0], [3.34, 3.34], color="black", linewidth=0.75)
    chain_order = ["zebrafish", "mosta", "arista", "chicken_heart"]
    for row_index, dataset in enumerate(chain_order):
        group = chains.loc[chains["dataset"].astype(str).eq(dataset)].sort_values(
            "receiver_target_rank"
        )
        if group.empty:
            raise ValueError(f"NicheNet chain is missing for {dataset}")
        first = group.iloc[0]
        receiver_targets = ", ".join(group["receiver_target"].astype(str).tolist())
        if dataset == "zebrafish" and not zebrafish_is_strict_proxy:
            receiver_targets = ", ".join(
                group["receiver_target"].astype(str).str.casefold().tolist()
            )
        if dataset == "zebrafish" and zebrafish_is_sensitivity:
            receiver_targets += "†"
        y = float(2.95 - row_index)
        if row_index:
            ax_c_chain.plot(
                [0.0, 1.0],
                [y + 0.50, y + 0.50],
                color=grid_color,
                linewidth=0.55,
            )
        ax_c_chain.text(
            0.0,
            y,
            labels[dataset],
            fontsize=7.5,
            fontweight="bold",
            va="center",
            color="black",
        )
        ax_c_chain.text(
            0.14,
            y + 0.12,
            f"Sender: {str(first.sender_type)}",
            fontsize=7.2,
            color="black",
            va="center",
        )
        ax_c_chain.text(
            0.14,
            y - 0.14,
            f"Receiver: {str(first.receiver_type)}",
            fontsize=7.2,
            color="black",
            va="center",
        )
        ax_c_chain.text(
            0.40,
            y + 0.12,
            f"{str(first.ligand).upper()}–{str(first.receptor).upper()}",
            fontsize=7.5,
            color="black",
            fontweight="bold",
            va="center",
        )
        ax_c_chain.text(
            0.40,
            y - 0.14,
            str(first.pathways),
            fontsize=7.2,
            color="black",
            va="center",
        )
        ax_c_chain.text(
            0.665,
            y,
            f"{int(first.cytobridge_global_rank)}",
            fontsize=7.5,
            color="black",
            ha="center",
            va="center",
        )
        ax_c_chain.text(
            0.78,
            y,
            f"{100 * float(first.commot_percentile):.1f}",
            fontsize=7.5,
            color="black",
            ha="center",
            va="center",
        )
        ax_c_chain.text(
            0.855,
            y,
            receiver_targets,
            fontsize=7.4,
            color="black",
            va="center",
        )
    ax_c_chain.plot([0.0, 1.0], [-0.50, -0.50], color="black", linewidth=0.75)
    ax_c_chain.text(
        0.0,
        -0.67,
        "*First CytoBridge-ranked LR × directed-pair axis with a matched NicheNet-predicted receiver target.",
        fontsize=7.0,
        color="black",
        va="top",
    )
    if zebrafish_artwork_note:
        ax_c_chain.text(
            0.0,
            -0.87,
            zebrafish_artwork_note,
            fontsize=7.0,
            color="black",
            va="top",
        )
    pdf = output / "spatial_communication_model_biology_a4.pdf"
    png = output / "spatial_communication_model_biology_a4.png"
    style.save_figure(fig, pdf, png, dpi=320)
    plt.close(fig)

    panel_output = output / "panel_data"
    panel_output.mkdir()
    panel_metrics_path = panel_output / "global_pair_metrics.csv"
    panel_support_path = panel_output / "model_linked_external_support.csv"
    panel_status_path = panel_output / "model_linked_lr_selection_status.csv"
    panel_molecular_path = panel_output / "model_biology_molecular_panel.csv"
    panel_rank_path = panel_output / "molecular_rank_consistency.csv"
    panel_chain_path = panel_output / "model_first_nichenet_chains.csv"
    primary_metrics.to_csv(panel_metrics_path, index=False)
    support.reset_index().to_csv(panel_support_path, index=False)
    status.reset_index().to_csv(panel_status_path, index=False)
    molecular_panel.reset_index().to_csv(panel_molecular_path, index=False)
    rank_metrics.to_csv(panel_rank_path, index=False)
    chains.to_csv(panel_chain_path, index=False)

    def molecular_metric_range(method: str) -> str:
        local = rank_metrics.loc[
            rank_metrics["dataset"].astype(str).isin(dataset_order)
            & rank_metrics["external_method"].astype(str).eq(method)
            & rank_metrics["available"].astype(str).str.casefold().isin({"true", "1"})
        ].copy()
        if method == "NicheNet" and zebrafish_is_sensitivity:
            local = local.loc[~local["dataset"].astype(str).eq("zebrafish")]
        rho = pd.to_numeric(local["spearman_rho"], errors="coerce").dropna()
        jaccard = pd.to_numeric(local["top_jaccard"], errors="coerce").dropna()
        if rho.empty or jaccard.empty:
            return f"{method} not evaluable"
        return (
            f"{method} Spearman ρ={rho.min():.2f}–{rho.max():.2f}, "
            f"top-20% Jaccard={jaccard.min():.2f}–{jaccard.max():.2f} "
            f"(n={len(local)})"
        )

    molecular_metric_summary = (
        f"{molecular_metric_range('COMMOT')}; " f"{molecular_metric_range('NicheNet')}"
    )

    caption = (
        "**Four-dataset interaction consistency.** (a) CytoBridge interaction-"
        "contribution scores are compared with COMMOT and the frozen current-database "
        "CellAgentChat proxy over each complete, zero-filled directed cell-type-pair "
        "grid. Points report Spearman rank correlation and top-20% Jaccard overlap; "
        "weak datasets remain visible. (b) CytoBridge alone selected the highest "
        "abundance-normalized exact-message × ligand × receptor axis before any "
        "external lookup. Rows connect the sender, LR pathway annotation, and receiver "
        "and report the subsequent COMMOT LR rank within that same directed cell pair "
        "and the pair-level percentile. (c) For each dataset, the highest globally "
        "ranked CytoBridge LR × directed-pair axis with matched positive NicheNet "
        "ligand-target evidence is listed. Ranks are calculated over all "
        "supported CytoBridge LR × directed-pair axes before NicheNet matching. The "
        "COMMOT column is "
        "the same-axis percentile, and the final column contains NicheNet-predicted "
        "receiver target genes. These targets are not CytoBridge outputs or "
        "experimentally measured responses. "
        f"{zebrafish_caption_clause}"
        "Cell-type names "
        "follow the source-atlas "
        "annotations. "
        "Across jointly positive LR × directed-pair candidates, the frozen "
        f"molecular-rank summaries are {molecular_metric_summary}. These are "
        "shared-input computational "
        "consistency analyses, not independent-cohort or causal validation."
    )
    caption_path = output / "caption.md"
    caption_path.write_text(caption + "\n", encoding="utf-8")
    implementation = _model_biology_implementation()
    provenance_path = output / "provenance.md"
    provenance_path.write_text(
        "\n".join(
            [
                "# Four-dataset interaction-consistency figure provenance",
                "",
                "## Scope",
                "",
                "The artwork shows the four datasets with a complete model-linked "
                "molecular axis. CytoBridge selects each model-first axis before "
                "external lookup. The frozen panel tables retain the complete "
                "five-dataset audit records for provenance.",
                "",
                "## Source paths",
                "",
                f"- aggregate manifest: `{aggregate_dir / 'manifest.json'}`",
                f"- selection manifest: `{selection_dir / 'manifest.json'}`",
                f"- molecular summary manifest: `{molecular_manifest_path}`",
                "- compact visual inputs: `panel_data/` in this bundle",
                "",
                "## Figure implementation",
                "",
                *[
                    f"- `{relative}`: `{record['sha256']}`"
                    for relative, record in implementation["files"].items()
                ],
                f"- aggregate SHA-256: `{implementation['aggregate_sha256']}`",
                "",
                "## Interpretation boundary",
                "",
                "LR identities and pathway labels are database-guided post-hoc "
                "decompositions of learned interaction contributions, not native "
                "biochemical probabilities. COMMOT is queried after CytoBridge "
                "selection. In panel c, NicheNet supplies the displayed predicted "
                "receiver targets; those genes are neither CytoBridge outputs nor "
                "experimentally observed responses. "
                f"{zebrafish_caption_clause}"
                "CellAgentChat is a frozen "
                "shared-database proxy analysis. Cross-method agreement is descriptive "
                "and does not establish causality or independent-cohort validation.",
                "",
                "## Rebuild",
                "",
                "Run the following command with a new empty output directory. The "
                "command revalidates every frozen input artifact before rendering.",
                "",
                f"`python {REPO_ROOT / 'scripts/run_spatial_communication_consistency.py'} "
                f"plot-model-biology --aggregate-dir {aggregate_dir} "
                f"--selection-dir {selection_dir} "
                f"--molecular-panel-data-dir {molecular_dir} "
                "--output-dir <new-empty-output-dir>`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 6,
        "workflow": "four_dataset_interaction_biology_figure",
        "displayed_datasets": dataset_order,
        "audit_datasets": audit_dataset_order,
        "implementation": implementation,
        "inputs": {
            "aggregate_manifest": _artifact(aggregate_dir / "manifest.json"),
            "selection_manifest": _artifact(selection_dir / "manifest.json"),
            "molecular_summary_manifest": _artifact(molecular_manifest_path),
        },
        "figure": {pdf.name: _artifact(pdf), png.name: _artifact(png)},
        "caption": _artifact(caption_path),
        "provenance": _artifact(provenance_path),
        "panel_semantics": {
            "b": {
                "selection": (
                    "CytoBridge selects the displayed LR-compatible axis before "
                    "external lookup"
                ),
                "commot": (
                    "post-selection within-pair LR rank and directed-pair percentile"
                ),
            },
            "c": {
                "selection": (
                    "for each dataset, the highest ranked supported CytoBridge LR x "
                    "directed-pair axis with matched positive NicheNet target evidence"
                ),
                "commot": "post-selection same-axis percentile",
                "nichenet": (
                    "predicted receiver targets; not CytoBridge outputs or "
                    "experimentally measured responses"
                ),
                "zebrafish_nichenet_scope": zebrafish_scope,
                "zebrafish_nichenet_corrected_aupr": (
                    zebrafish_corrected_aupr
                    if np.isfinite(zebrafish_corrected_aupr)
                    else None
                ),
                "zebrafish_included_in_pooled_nichenet_claims": (
                    not zebrafish_is_sensitivity
                ),
            },
        },
        "panel_data": {
            "global_metrics": _artifact(panel_metrics_path),
            "external_support": _artifact(panel_support_path),
            "selection_status": _artifact(panel_status_path),
            "molecular_summary": _artifact(panel_molecular_path),
            "molecular_rank_consistency": _artifact(panel_rank_path),
            "model_first_nichenet_chains": _artifact(panel_chain_path),
        },
    }
    _write_json(output / "figure_manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-sample")
    prepare.add_argument("--dataset", choices=FORMAL_DATASET_CONTRACTS, required=True)
    prepare.add_argument("--input-h5ad", required=True)
    prepare.add_argument("--expected-h5ad-sha256", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--sample-n", type=int, default=3000)
    prepare.set_defaults(
        function=lambda args: prepare_shared_samples(
            args.input_h5ad,
            args.output_dir,
            dataset=args.dataset,
            expected_h5ad_sha256=args.expected_h5ad_sha256,
            sample_n=args.sample_n,
        )
    )
    proxy = sub.add_parser("prepare-shared-database-proxy")
    proxy.add_argument("--dataset", choices=FORMAL_DATASET_CONTRACTS, required=True)
    proxy.add_argument("--input-h5ad", required=True)
    proxy.add_argument("--expected-h5ad-sha256", required=True)
    proxy.add_argument("--filtered-lr-database", required=True)
    proxy.add_argument("--expected-database-sha256", required=True)
    proxy.add_argument("--orthology-map")
    proxy.add_argument("--orthology-manifest")
    proxy.add_argument(
        "--orthology-policy",
        choices=("strict_confidence1", "one2one_bijective_all_confidence"),
        default="strict_confidence1",
    )
    proxy.add_argument(
        "--sampling-seeds",
        default=",".join(str(value) for value in SPATIAL_PROXY_SAMPLING_SEEDS),
    )
    proxy.add_argument("--output-dir", required=True)
    proxy.set_defaults(
        function=lambda args: prepare_spatial_proxy_inputs(
            args.input_h5ad,
            args.filtered_lr_database,
            args.output_dir,
            dataset=args.dataset,
            expected_h5ad_sha256=args.expected_h5ad_sha256,
            expected_database_sha256=args.expected_database_sha256,
            orthology_map=args.orthology_map,
            orthology_manifest=args.orthology_manifest,
            orthology_policy=args.orthology_policy,
            sampling_seeds=tuple(
                int(value.strip())
                for value in args.sampling_seeds.split(",")
                if value.strip()
            ),
        )
    )
    nichenet = sub.add_parser("summarize-nichenet")
    nichenet.add_argument("--nichenet-dir", required=True)
    nichenet.add_argument("--output-dir", required=True)
    nichenet.set_defaults(function=summarize_nichenet)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--config", required=True)
    aggregate_parser.add_argument("--output-dir", required=True)
    aggregate_parser.set_defaults(function=aggregate)
    select_biology = sub.add_parser("select-model-biology")
    select_biology.add_argument("--config", required=True)
    select_biology.add_argument("--aggregate-dir", required=True)
    select_biology.add_argument("--output-dir", required=True)
    select_biology.add_argument("--minimum-active-edges", type=int, default=10)
    select_biology.set_defaults(function=select_model_biology)
    score_biology = sub.add_parser("score-model-biology")
    score_biology.add_argument("--config", required=True)
    score_biology.add_argument("--selection-dir", required=True)
    score_biology.add_argument("--output-dir", required=True)
    score_biology.add_argument("--top-fraction", type=float, default=TOP_FRACTION)
    score_biology.add_argument("--permutations", type=int, default=1000)
    score_biology.add_argument("--seed", type=int, default=20260816)
    score_biology.set_defaults(function=score_model_biology)
    summarize_biology = sub.add_parser("summarize-model-biology-molecular")
    summarize_biology.add_argument("--config", required=True)
    summarize_biology.add_argument("--selection-dir", required=True)
    summarize_biology.add_argument("--output-dir", required=True)
    summarize_biology.set_defaults(function=summarize_model_biology_molecular)
    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("--aggregate-dir", required=True)
    plot_parser.add_argument("--output-dir", required=True)
    plot_parser.set_defaults(function=plot)
    model_figure = sub.add_parser("plot-model-biology")
    model_figure.add_argument("--config")
    model_figure.add_argument("--spatial-panel-data-dir")
    model_figure.add_argument("--aggregate-dir", required=True)
    model_figure.add_argument("--selection-dir", required=True)
    model_figure.add_argument("--edge-dir")
    model_figure.add_argument("--molecular-panel-data-dir", required=True)
    model_figure.add_argument("--output-dir", required=True)
    model_figure.add_argument("--maximum-display-edges", type=int, default=60)
    model_figure.set_defaults(function=plot_model_biology)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
