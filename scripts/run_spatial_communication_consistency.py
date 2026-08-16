#!/usr/bin/env python3
"""Prepare, aggregate, and plot five-dataset spatial CCC consistency evidence.

External methods are executed by their pinned adapters. This orchestrator
freezes the shared sample, combines only manifest-bound outputs, applies the
predeclared main-figure gate, and produces an A4 submission figure without
silently hiding weak or unavailable methods from the audit tables.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
            table = table.drop_duplicates(label_columns).head(3)
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
                }
                row.update(
                    {column: str(getattr(record, column)) for column in label_columns}
                )
                destination.append(row)
        target_path = spec.get("nichenet_target_csv")
        if target_path:
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
                .head(3)
            )
            for rank, record in enumerate(table.itertuples(index=False), start=1):
                target_rows.append(
                    {
                        "dataset": dataset,
                        "sender_type": sender,
                        "receiver_type": receiver,
                        "rank_within_pair": rank,
                        "ligand": str(record.ligand),
                        "receptor": str(record.receptor),
                        "target": str(record.target),
                        "nichenet_evidence": float(record.ligand_target_evidence),
                    }
                )
    pathway_columns = [
        "dataset",
        "sender_type",
        "receiver_type",
        "rank_within_pair",
        "commot_score",
        "relative_to_pair_top",
        "pathway",
    ]
    lr_columns = [
        "dataset",
        "sender_type",
        "receiver_type",
        "rank_within_pair",
        "commot_score",
        "relative_to_pair_top",
        "ligand",
        "receptor",
        "pathway",
    ]
    target_columns = [
        "dataset",
        "sender_type",
        "receiver_type",
        "rank_within_pair",
        "ligand",
        "receptor",
        "target",
        "nichenet_evidence",
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
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.mkdir(parents=True)
    metrics = pd.read_csv(source / "cytobridge_external_metrics.csv")
    decisions = pd.read_csv(source / "main_figure_method_decisions.csv")
    selected = pd.read_csv(source / "selected_biological_pairs.csv")
    pathways = pd.read_csv(source / "selected_pair_commot_pathways.csv")
    ligand_receptors = pd.read_csv(source / "selected_pair_commot_lr.csv")
    target_evidence = pd.read_csv(source / "selected_pair_nichenet_targets.csv")
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
    fig = plt.figure(figsize=style.A4_PORTRAIT)
    grid = fig.add_gridspec(
        6,
        1,
        height_ratios=[0.18, 1.58, 0.18, 1.62, 0.18, 2.48],
        left=0.255,
        right=0.97,
        top=0.98,
        bottom=0.045,
        hspace=0.34,
    )
    head_a = fig.add_subplot(grid[0])
    _heading(head_a, "a", "Cross-method concordance")
    axes_a = grid[1].subgridspec(1, 2, wspace=0.32)
    ax_rho = fig.add_subplot(axes_a[0])
    ax_j = fig.add_subplot(axes_a[1])
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
        axis.set_yticks(y_positions, [dataset_labels[d] for d in dataset_order])
        axis.set_ylim(len(dataset_order) - 0.55, -0.55)
        axis.set_xlabel(xlabel)
        style.clean_axis(axis, grid=True)
    ax_j.set_yticklabels([])
    ax_rho.legend(
        frameon=False,
        fontsize=7.0,
        ncol=min(2, len(included)),
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
    )
    head_b = fig.add_subplot(grid[2])
    _heading(head_b, "b", "Shared interaction programs")
    ax_b = fig.add_subplot(grid[3])
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
    ax_b.set_yticks(
        range(len(selected)),
        [
            f"{dataset_labels[row.dataset]}\n"
            + textwrap.fill(f"{row.sender_type} → {row.receiver_type}", width=30)
            for row in selected.itertuples()
        ],
        fontsize=6.7,
    )
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
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        fontsize=6.9,
    )
    head_c = fig.add_subplot(grid[4])
    _heading(head_c, "c", "Biological annotations")
    ax_c = fig.add_subplot(grid[5])
    ax_c.set_axis_off()
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(-0.55, len(dataset_order) + 0.70)
    columns = {"program": 0.00, "pathway": 0.31, "lr": 0.55, "target": 0.76}
    for x, label in (
        (columns["program"], "Dataset · directed program"),
        (columns["pathway"], "COMMOT pathways"),
        (columns["lr"], "Top COMMOT LR"),
        (columns["target"], "NicheNet ligand → target"),
    ):
        ax_c.text(x, len(dataset_order) + 0.32, label, fontsize=7.0, fontweight="bold")
    for separator in (0.295, 0.535, 0.745):
        ax_c.vlines(
            separator,
            -0.45,
            len(dataset_order) + 0.47,
            color="#D7DCE0",
            lw=0.65,
        )
    for row_index, dataset in enumerate(dataset_order):
        y = len(dataset_order) - 1 - row_index
        if row_index % 2 == 0:
            ax_c.axhspan(y - 0.44, y + 0.44, color="#F6F7F8", zorder=0)
        pair = selected.loc[selected.dataset.eq(dataset)].iloc[0]
        pair_text = textwrap.fill(
            f"{pair.sender_type} → {pair.receiver_type}", width=27
        )
        ax_c.text(
            columns["program"],
            y + 0.14,
            dataset_labels[dataset],
            fontsize=6.8,
            fontweight="bold",
            color=DATASET_COLORS[dataset],
            va="center",
        )
        ax_c.text(
            columns["program"],
            y - 0.14,
            pair_text,
            fontsize=6.2,
            color="#26343F",
            va="center",
        )
        pathway_values = pathways.loc[pathways.dataset.eq(dataset)].sort_values(
            "rank_within_pair"
        )
        pathway_text = " · ".join(pathway_values.pathway.astype(str).head(3))
        ax_c.text(
            columns["pathway"],
            y,
            textwrap.fill(pathway_text or "not available", width=21),
            fontsize=6.4,
            color="#26343F",
            va="center",
        )
        lr_values = ligand_receptors.loc[
            ligand_receptors.dataset.eq(dataset)
        ].sort_values("rank_within_pair")
        lr_text = "\n".join(
            f"{row.ligand}–{row.receptor}" for row in lr_values.head(2).itertuples()
        )
        ax_c.text(
            columns["lr"],
            y,
            lr_text or "not available",
            fontsize=6.3,
            color="#26343F",
            va="center",
        )
        target_values = target_evidence.loc[
            target_evidence.dataset.eq(dataset)
        ].sort_values("rank_within_pair")
        if target_values.empty:
            target_text = "not evaluable\n(strict orthology)"
            target_color = "#78828A"
        else:
            target = target_values.iloc[0]
            target_text = textwrap.fill(
                f"{target.ligand}–{target.receptor} → {target.target}", width=24
            )
            target_color = "#26343F"
        ax_c.text(
            columns["target"],
            y,
            target_text,
            fontsize=6.3,
            color=target_color,
            va="center",
        )
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
    caption = f"**Five-dataset spatial communication consistency.** (a) Terminal-stage concordance compares the exact CytoBridge interaction contribution with {included_text} over complete directed cell-type-pair grids. The included methods passed the prespecified cross-dataset gate: {summary_text}. CellAgentChat used spatial mode, three fixed sampling seeds, and the gene-level representable subset of each dataset's accepted CytoBridge LR database. (b) Connected points show within-method ranks for one jointly high-ranking off-diagonal program per dataset. (c) COMMOT supplies pathway and LR annotations, and NicheNet supplies ligand-to-target support where its species prior yielded a valid result. Zebrafish NicheNet was not evaluable under strict orthology. ARISTA and chicken are conserved-human-symbol proxy analyses. These results show descriptive computational consistency rather than causal validation."
    (output / "caption.md").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source_manifest": _artifact(source / "manifest.json"),
        "included_methods": included,
        "gate": MAIN_FIGURE_GATE,
        "figure": {pdf.name: _artifact(pdf), png.name: _artifact(png)},
        "caption": _artifact(output / "caption.md"),
        "panel_data": {
            name: _artifact(source / name)
            for name in (
                "cytobridge_external_metrics.csv",
                "main_figure_method_decisions.csv",
                "method_execution_status.csv",
                "selected_biological_pairs.csv",
                "selected_pair_commot_pathways.csv",
                "selected_pair_commot_lr.csv",
                "selected_pair_nichenet_targets.csv",
            )
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
    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("--aggregate-dir", required=True)
    plot_parser.add_argument("--output-dir", required=True)
    plot_parser.set_defaults(function=plot)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
