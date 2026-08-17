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
MODEL_LINKED_BIOLOGICAL_PROGRAMS = {
    "zebrafish": (
        "Collagen-linked extracellular-matrix organization at the "
        "pronephric–yolk/muscle interface"
    ),
    "mosta": (
        "COL6A3–CD44 matrix adhesion in connective-tissue maturation; "
        "Cd44 is independently linked to the receiver response by NicheNet"
    ),
    "arista": (
        "PSAP–GPR37L1 trophic and lipid-stress signaling in a regenerative "
        "neuroglial niche"
    ),
    "chicken_heart": (
        "CD99 homophilic cell contact during fibroblast-to-valve tissue remodeling"
    ),
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
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing NicheNet {label}: {path}")
    candidates = pd.read_csv(required["candidates"])
    activities = pd.read_csv(required["activities"])
    targets = pd.read_csv(required["targets"])
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
    pair_scores.to_csv(pair_path, index=False)
    evidence.to_csv(evidence_path, index=False, compression="gzip")
    detailed.to_csv(target_path, index=False, compression="gzip")
    manifest = {
        "schema_version": 1,
        "workflow": "spatial_communication_consistency_nichenet_summary",
        "pair_score": "mean of the top five activity-rank × sqrt(sender-expression-fraction × receiver-expression-fraction) LR evidences",
        "sources": {label: _artifact(path) for label, path in required.items()},
        "outputs": {
            path.name: _artifact(path)
            for path in (pair_path, evidence_path, target_path)
        },
    }
    _write_json(output / "manifest.json", manifest)


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


def plot_model_biology(args: argparse.Namespace) -> None:
    """Draw the reviewer-facing model-linked five-dataset figure."""

    from CytoBridge.nonspatial import scnt_figure_style as style

    aggregate_dir = Path(args.aggregate_dir).expanduser().resolve()
    selection_dir = Path(args.selection_dir).expanduser().resolve()
    edge_dir = Path(args.edge_dir).expanduser().resolve()
    panel_input = (
        Path(args.spatial_panel_data_dir).expanduser().resolve()
        if args.spatial_panel_data_dir
        else None
    )
    if panel_input is None:
        config_path, config = _model_biology_config(args.config)
        input_cells = None
        input_edges = None
    else:
        config_path = None
        config = None
        input_cells = _read_table(
            panel_input / "spatial_map_cells.csv.gz", label="spatial map cells"
        )
        input_edges = _read_table(
            panel_input / "spatial_map_edges.csv.gz", label="spatial map edges"
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
    cellagent_pairs = pair_scores.loc[
        pair_scores["method"].eq("CellAgentChat")
        & pair_scores["available"].astype(str).str.casefold().isin({"true", "1"})
    ][["dataset", "sender_type", "receiver_type", "score", "rank_percentile"]]
    support = (
        support.reset_index()
        .merge(
            cellagent_pairs.rename(
                columns={
                    "score": "cellagentchat_pair_score",
                    "rank_percentile": "cellagentchat_pair_percentile",
                }
            ),
            on=["dataset", "sender_type", "receiver_type"],
            how="left",
        )
        .set_index("dataset")
    )
    edge_path = edge_dir / "cytobridge_top_model_linked_edges.csv.gz"
    edges = pd.read_csv(edge_path)
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
    style.apply_style()
    fig = plt.figure(figsize=style.A4_PORTRAIT)
    grid = fig.add_gridspec(
        7,
        1,
        height_ratios=[0.12, 1.15, 0.12, 1.25, 0.12, 0.05, 2.65],
        left=0.21,
        right=0.965,
        top=0.975,
        bottom=0.055,
        hspace=0.29,
    )

    head_a = fig.add_subplot(grid[0])
    _heading(head_a, "a", "Global directed-pair concordance")
    axes_a = grid[1].subgridspec(1, 2, wspace=0.30)
    ax_rho = fig.add_subplot(axes_a[0])
    ax_jaccard = fig.add_subplot(axes_a[1])
    y = np.arange(len(dataset_order), dtype=float)
    metric_specs = (
        ("COMMOT", commot.reindex(dataset_order), "s", -0.09),
        ("CellAgentChat", cellagent.reindex(dataset_order), "D", 0.09),
    )
    for axis, column, xlabel in (
        (ax_rho, "spearman_rho", "Spearman rank correlation (ρ)"),
        (ax_jaccard, "top_jaccard", "Top-20% directed-pair Jaccard"),
    ):
        for method, method_table, marker, offset in metric_specs:
            available = (
                method_table["metric_available"]
                .astype(str)
                .str.casefold()
                .isin({"true", "1"})
                & method_table[column].notna()
            )
            axis.scatter(
                method_table.loc[available, column],
                y[available.to_numpy()] + offset,
                s=43,
                marker=marker,
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.6,
                label=method,
                zorder=3,
            )
        axis.axvline(0, color="#AAB2B8", lw=0.7)
        axis.set_yticks(y, [labels[name] for name in dataset_order])
        axis.set_ylim(len(dataset_order) - 0.55, -0.55)
        axis.set_xlim(-0.05, 1.04)
        axis.set_xlabel(xlabel)
        style.clean_axis(axis, grid=True)
    ax_jaccard.set_yticklabels([])
    ax_rho.legend(
        frameon=False,
        fontsize=6.8,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
    )

    head_b = fig.add_subplot(grid[2])
    _heading(head_b, "b", "External support for CytoBridge-selected programs")
    ax_b = fig.add_subplot(grid[3])
    method_specs = (
        ("cytobridge_percentile", "CytoBridge selection", "#5B4B8A", "o"),
        ("commot_percentile", "COMMOT evaluation", METHOD_COLORS["COMMOT"], "s"),
        (
            "cellagentchat_pair_percentile",
            "CellAgentChat pair rank",
            METHOD_COLORS["CellAgentChat"],
            "D",
        ),
    )
    for row_index, dataset in enumerate(dataset_order):
        if dataset not in support.index:
            ax_b.text(
                0.51,
                row_index,
                "no positive LR-compatible model axis",
                fontsize=6.5,
                color="#7A848C",
                va="center",
            )
            continue
        row = support.loc[dataset]
        for column, _, color, marker in method_specs:
            value = float(row[column])
            if np.isfinite(value):
                ax_b.scatter(
                    value,
                    row_index,
                    s=43,
                    marker=marker,
                    color=color,
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=3,
                )
    ylabels: list[str] = []
    for dataset in dataset_order:
        if dataset in support.index:
            row = support.loc[dataset]
            pair = textwrap.fill(f"{row.sender_type} → {row.receiver_type}", width=31)
            ylabels.append(
                f"{labels[dataset]} · {str(row.ligand).upper()}–{str(row.receptor).upper()}\n{pair}"
            )
        else:
            ylabels.append(f"{labels[dataset]}\nno positive model-linked LR axis")
    ax_b.set_yticks(np.arange(len(dataset_order)), ylabels, fontsize=6.4)
    ax_b.set_ylim(len(dataset_order) - 0.55, -0.55)
    ax_b.set_xlim(0.45, 1.015)
    ax_b.axvline(0.95, color="#AAB2B8", lw=0.75, ls="--")
    ax_b.set_xlabel(
        "Percentile in each method's complete zero-filled candidate universe"
    )
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
            for _, label, color, marker in method_specs
        ],
        frameon=False,
        fontsize=6.8,
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
    )

    head_c = fig.add_subplot(grid[4])
    _heading(
        head_c,
        "c",
        "Model-selected spatial LR circuits and biological programs",
    )
    legend_c = fig.add_subplot(grid[5])
    legend_c.set_axis_off()
    c_body = grid[6].subgridspec(1, 2, width_ratios=(0.43, 0.57), wspace=0.10)
    ax_c_map = fig.add_subplot(c_body[0])
    ax_c_biology = fig.add_subplot(c_body[1])
    ax_c_biology.set_axis_off()
    map_datasets = ["zebrafish", "mosta", "arista", "chicken_heart"]
    panel_cell_frames: list[pd.DataFrame] = []
    panel_edge_frames: list[pd.DataFrame] = []
    map_payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for dataset in map_datasets:
        row = support.loc[dataset]
        if panel_input is None:
            assert config is not None
            spec = dict(config["datasets"][dataset])
            data = ad.read_h5ad(Path(spec["h5ad"]).expanduser().resolve())
            stage_mask = np.isclose(
                pd.to_numeric(data.obs["ccc_stage"], errors="coerce"),
                float(row.stage),
            )
            coordinates = np.asarray(data.obsm["spatial_aligned"], dtype=float)
            types = data.obs["ccc_cell_type"].astype(str).to_numpy()
            cell_indices = np.flatnonzero(stage_mask)
            map_cells = pd.DataFrame(
                {
                    "dataset": dataset,
                    "cell_index": cell_indices,
                    "x": coordinates[cell_indices, 0],
                    "y": coordinates[cell_indices, 1],
                    "cell_type": types[cell_indices],
                }
            )
            selected_edges = edges.loc[edges["dataset"].eq(dataset)].sort_values(
                "cytobridge_message_lr_flow", ascending=False
            )
            source_index = selected_edges["source_index"].to_numpy(int)
            target_index = selected_edges["target_index"].to_numpy(int)
            map_edges = pd.DataFrame(
                {
                    "dataset": dataset,
                    "source_index": source_index,
                    "target_index": target_index,
                    "source_x": coordinates[source_index, 0],
                    "source_y": coordinates[source_index, 1],
                    "target_x": coordinates[target_index, 0],
                    "target_y": coordinates[target_index, 1],
                    "cytobridge_message_lr_flow": selected_edges[
                        "cytobridge_message_lr_flow"
                    ].to_numpy(float),
                }
            )
        else:
            assert input_cells is not None and input_edges is not None
            map_cells = input_cells.loc[input_cells["dataset"].eq(dataset)].copy()
            map_edges = input_edges.loc[input_edges["dataset"].eq(dataset)].copy()
            if map_cells.empty or map_edges.empty:
                raise ValueError(f"spatial panel data is incomplete for {dataset}")
        panel_cell_frames.append(map_cells)
        panel_edge_frames.append(map_edges)
        map_payload[dataset] = (map_cells, map_edges)

    # A single enlarged spatial exemplar makes individual learned edges legible at
    # final A4 size.  The adjacent molecular table retains all four evaluable
    # datasets, so this is a spatial example rather than a cherry-picked summary.
    map_dataset = "zebrafish"
    row = support.loc[map_dataset]
    map_cells, map_edges = map_payload[map_dataset]
    coordinates = map_cells[["x", "y"]].to_numpy(float)
    types = map_cells["cell_type"].astype(str).to_numpy()
    ax_c_map.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=1.0,
        color="#D8DEE2",
        alpha=0.48,
        linewidths=0,
        rasterized=False,
    )
    sender = types == str(row.sender_type)
    receiver = types == str(row.receiver_type)
    ax_c_map.scatter(
        coordinates[sender, 0],
        coordinates[sender, 1],
        s=5.5,
        color="#3A86A8",
        alpha=0.82,
        linewidths=0,
        zorder=2,
    )
    ax_c_map.scatter(
        coordinates[receiver, 0],
        coordinates[receiver, 1],
        s=5.5,
        color="#E08C46",
        alpha=0.82,
        linewidths=0,
        zorder=2,
    )
    display_edges = map_edges.sort_values(
        "cytobridge_message_lr_flow", ascending=False
    ).head(min(int(args.maximum_display_edges), 12))
    score = display_edges["cytobridge_message_lr_flow"].to_numpy(float)
    scaled = score / max(float(np.max(score)), np.finfo(float).eps)
    for edge, relative_score in zip(
        display_edges.itertuples(index=False), scaled, strict=True
    ):
        ax_c_map.annotate(
            "",
            xy=(float(edge.target_x), float(edge.target_y)),
            xytext=(float(edge.source_x), float(edge.source_y)),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#5C3E91",
                "linewidth": 0.9 + 1.7 * float(relative_score),
                "alpha": 0.72 + 0.22 * float(relative_score),
                "mutation_scale": 7.0 + 4.0 * float(relative_score),
                "shrinkA": 0.0,
                "shrinkB": 0.0,
            },
            zorder=4,
        )
    endpoint_x = np.concatenate(
        [
            display_edges["source_x"].to_numpy(float),
            display_edges["target_x"].to_numpy(float),
        ]
    )
    endpoint_y = np.concatenate(
        [
            display_edges["source_y"].to_numpy(float),
            display_edges["target_y"].to_numpy(float),
        ]
    )
    x_span = max(float(np.ptp(endpoint_x)), 0.08)
    y_span = max(float(np.ptp(endpoint_y)), 0.08)
    x_limits = (
        float(endpoint_x.min() - 0.30 * x_span),
        float(endpoint_x.max() + 0.30 * x_span),
    )
    y_limits = (
        float(endpoint_y.min() - 0.30 * y_span),
        float(endpoint_y.max() + 0.30 * y_span),
    )
    ax_c_map.set_xlim(*x_limits)
    ax_c_map.set_ylim(*y_limits)
    context = ax_c_map.inset_axes([0.015, 0.015, 0.30, 0.28])
    context.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=0.35,
        color="#CDD4D9",
        alpha=0.60,
        linewidths=0,
        rasterized=False,
    )
    context.plot(
        [x_limits[0], x_limits[1], x_limits[1], x_limits[0], x_limits[0]],
        [y_limits[0], y_limits[0], y_limits[1], y_limits[1], y_limits[0]],
        color="#5C3E91",
        linewidth=0.8,
    )
    context.set_aspect("equal", adjustable="datalim")
    context.set_axis_off()
    ax_c_map.set_aspect("equal", adjustable="datalim")
    ax_c_map.set_axis_off()
    ax_c_map.set_title(
        "Zebrafish spatial exemplar\n"
        f"{str(row.sender_type)} → {str(row.receiver_type)}",
        fontsize=7.8,
        fontweight="bold",
        color="black",
        pad=2,
    )
    ax_c_map.text(
        0.5,
        -0.018,
        (
            f"{str(row.ligand).upper()}–{str(row.receptor).upper()} "
            f"({row.pathways}); top {len(display_edges)} of {len(map_edges)} "
            "positive model-linked edges"
        ),
        transform=ax_c_map.transAxes,
        fontsize=5.8,
        color="black",
        va="top",
        ha="center",
    )

    # Molecular interpretation is presented as a compact scientific table.  LR
    # and pathway labels come from the frozen shared database; the last column
    # states the tissue program supported by the selected circuit.
    ax_c_biology.text(
        0.00,
        1.01,
        "Cell circuit",
        fontsize=6.4,
        fontweight="bold",
        color="black",
        va="bottom",
    )
    ax_c_biology.text(
        0.37,
        1.01,
        "LR / pathway",
        fontsize=6.4,
        fontweight="bold",
        color="black",
        va="bottom",
    )
    ax_c_biology.text(
        0.66,
        1.01,
        "Biological interpretation",
        fontsize=6.4,
        fontweight="bold",
        color="black",
        va="bottom",
    )
    biology_rows: list[dict[str, object]] = []
    y_positions = np.linspace(0.88, 0.10, len(dataset_order))
    for y_value, dataset in zip(y_positions, dataset_order, strict=True):
        if dataset not in support.index:
            ax_c_biology.axhline(y_value + 0.085, color="#D7DDE2", lw=0.55)
            ax_c_biology.text(
                0.00,
                y_value,
                labels[dataset],
                fontsize=7.0,
                fontweight="bold",
                color="black",
                va="center",
            )
            ax_c_biology.text(
                0.37,
                y_value,
                "No positive\nmodel-linked LR axis",
                fontsize=5.9,
                color="#3E4A52",
                va="center",
                linespacing=1.05,
            )
            ax_c_biology.text(
                0.66,
                y_value,
                "No molecular interpretation\nat the learned-edge threshold",
                fontsize=5.8,
                color="#3E4A52",
                va="center",
                linespacing=1.05,
            )
            biology_rows.append(
                {
                    "dataset": dataset,
                    "sender_type": "not_evaluable",
                    "receiver_type": "not_evaluable",
                    "ligand": "not_evaluable",
                    "receptor": "not_evaluable",
                    "pathway": "not_evaluable",
                    "commot_rank_percentile": np.nan,
                    "biological_program": "not_evaluable",
                    "nichenet_target": "not_evaluable",
                }
            )
            continue
        row = support.loc[dataset]
        ax_c_biology.axhline(y_value + 0.085, color="#D7DDE2", lw=0.55)
        ax_c_biology.text(
            0.00,
            y_value + 0.035,
            labels[dataset],
            fontsize=7.0,
            fontweight="bold",
            color="black",
            va="center",
        )
        ax_c_biology.text(
            0.00,
            y_value - 0.035,
            textwrap.fill(f"{row.sender_type} → {row.receiver_type}", width=25),
            fontsize=5.8,
            color="#3E4A52",
            va="center",
            linespacing=1.05,
        )
        ax_c_biology.text(
            0.37,
            y_value + 0.025,
            f"{str(row.ligand).upper()}–{str(row.receptor).upper()}",
            fontsize=6.2,
            fontweight="bold",
            color="black",
            va="center",
        )
        ax_c_biology.text(
            0.37,
            y_value - 0.040,
            f"{row.pathways}\nCOMMOT percentile {100 * float(row.commot_percentile):.1f}",
            fontsize=5.5,
            color="#3E4A52",
            va="center",
            linespacing=1.05,
        )
        program = MODEL_LINKED_BIOLOGICAL_PROGRAMS[dataset]
        ax_c_biology.text(
            0.66,
            y_value,
            textwrap.fill(program, width=32),
            fontsize=5.9,
            color="black",
            va="center",
            linespacing=1.10,
        )
        biology_rows.append(
            {
                "dataset": dataset,
                "sender_type": str(row.sender_type),
                "receiver_type": str(row.receiver_type),
                "ligand": str(row.ligand),
                "receptor": str(row.receptor),
                "pathway": str(row.pathways),
                "commot_rank_percentile": float(row.commot_percentile),
                "biological_program": program,
                "nichenet_target": (
                    str(row.nichenet_target)
                    if bool(row.nichenet_available)
                    else "not_evaluable_for_exact_axis"
                ),
            }
        )
    ax_c_biology.set_xlim(0, 1)
    ax_c_biology.set_ylim(0, 1)
    ax_c_biology.axhline(0.035, color="#D7DDE2", lw=0.55)
    legend_c.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#3A86A8",
                markeredgecolor="none",
                label="sender type",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="#E08C46",
                markeredgecolor="none",
                label="receiver type",
            ),
            plt.Line2D(
                [0],
                [0],
                color="#6A51A3",
                lw=1.6,
                marker=">",
                markevery=[1],
                label="model-linked exact-message × LR-activity edge",
            ),
        ],
        frameon=False,
        fontsize=6.5,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.5, 0.45),
    )

    pdf = output / "spatial_communication_model_biology_a4.pdf"
    png = output / "spatial_communication_model_biology_a4.png"
    style.save_figure(fig, pdf, png, dpi=320)
    plt.close(fig)
    panel_output = output / "panel_data"
    panel_output.mkdir()
    panel_cells_path = panel_output / "spatial_map_cells.csv.gz"
    panel_edges_path = panel_output / "spatial_map_edges.csv.gz"
    panel_metrics_path = panel_output / "global_pair_metrics.csv"
    panel_support_path = panel_output / "model_linked_external_support.csv"
    panel_status_path = panel_output / "model_linked_lr_selection_status.csv"
    panel_biology_path = panel_output / "model_linked_biological_programs.csv"
    pd.concat(panel_cell_frames, ignore_index=True).to_csv(
        panel_cells_path, index=False
    )
    pd.concat(panel_edge_frames, ignore_index=True).to_csv(
        panel_edges_path, index=False
    )
    primary_metrics.to_csv(panel_metrics_path, index=False)
    support.reset_index().to_csv(panel_support_path, index=False)
    status.reset_index().to_csv(panel_status_path, index=False)
    pd.DataFrame(biology_rows).to_csv(panel_biology_path, index=False)
    caption = (
        "**Five-dataset model-linked communication consistency.** (a) Terminal-stage "
        "directed cell-type-pair concordance is shown for COMMOT and for the frozen "
        "current-database CellAgentChat proxy across all five datasets. (b) For each "
        "dataset, CytoBridge alone selected the highest "
        "abundance-normalized exact-message magnitude × sender-ligand × "
        "receiver-receptor activity axis from the frozen LR database crossed with "
        "every off-diagonal model pair; neither COMMOT nor CellAgentChat entered "
        "selection. COMMOT was evaluated on the complete zero-filled LR×pair "
        "universe; CellAgentChat shows its native CTPS rank for the same selected "
        "cell-type pair. AdMouse had no "
        "axis with at least 10 active model-linked edges under its 347-gene panel and "
        "learned-edge threshold. (c) The enlarged zebrafish spatial exemplar "
        "shows observed terminal coordinates, sender/receiver populations, and the "
        "strongest positive CytoBridge exact-message × LR-activity edges with "
        "directional arrowheads; the inset locates the magnified field within the "
        "whole tissue. The adjacent table resolves the four evaluable model-first "
        "selections into "
        "COL1A2–SDC4 extracellular-matrix circuit in zebrafish, a COL6A3–CD44 "
        "matrix-adhesion circuit in MOSTA, a PSAP–GPR37L1 neuroglial circuit in "
        "ARISTA, and a CD99 homophilic cell-contact circuit in chicken valve tissue. "
        "The LR label is a post-hoc molecular compatibility annotation of learned "
        "model edges, not a claim that the GNN natively identifies a unique "
        "biochemical LR pair. Receiver-response support is evaluated separately in "
        "the zebrafish deep-validation figure; NicheNet is not shown here because "
        "exact-axis support was incomplete across the five datasets."
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
                f"- edge manifest: `{edge_dir / 'manifest.json'}`",
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
                "## Rebuild",
                "",
                "Run `scripts/run_spatial_communication_consistency.py "
                "plot-model-biology` with the aggregate, selection, edge, and either "
                "the five-dataset config or this bundle's compact `panel_data` "
                "directory. The compact route reproduces the visual without the "
                "original large H5AD files.",
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
        "workflow": "five_dataset_model_linked_spatial_communication_figure",
        "implementation": implementation,
        "inputs": {
            "aggregate_manifest": _artifact(aggregate_dir / "manifest.json"),
            "selection_manifest": _artifact(selection_dir / "manifest.json"),
            "edge_manifest": _artifact(edge_dir / "manifest.json"),
        },
        "figure": {pdf.name: _artifact(pdf), png.name: _artifact(png)},
        "caption": _artifact(caption_path),
        "provenance": _artifact(provenance_path),
        "panel_data": {
            "global_metrics": _artifact(panel_metrics_path),
            "external_support": _artifact(panel_support_path),
            "selection_status": _artifact(panel_status_path),
            "biological_programs": _artifact(panel_biology_path),
            "spatial_map_cells": _artifact(panel_cells_path),
            "spatial_map_edges": _artifact(panel_edges_path),
        },
    }
    if config_path is not None:
        manifest["inputs"]["config"] = _artifact(config_path)
    else:
        assert panel_input is not None
        manifest["inputs"]["spatial_panel_data"] = {
            "cells": _artifact(panel_input / "spatial_map_cells.csv.gz"),
            "edges": _artifact(panel_input / "spatial_map_edges.csv.gz"),
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
    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("--aggregate-dir", required=True)
    plot_parser.add_argument("--output-dir", required=True)
    plot_parser.set_defaults(function=plot)
    model_figure = sub.add_parser("plot-model-biology")
    model_figure_input = model_figure.add_mutually_exclusive_group(required=True)
    model_figure_input.add_argument("--config")
    model_figure_input.add_argument("--spatial-panel-data-dir")
    model_figure.add_argument("--aggregate-dir", required=True)
    model_figure.add_argument("--selection-dir", required=True)
    model_figure.add_argument("--edge-dir", required=True)
    model_figure.add_argument("--output-dir", required=True)
    model_figure.add_argument("--maximum-display-edges", type=int, default=60)
    model_figure.set_defaults(function=plot_model_biology)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
