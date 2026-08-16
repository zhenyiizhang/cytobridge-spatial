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
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from CytoBridge.spatial_communication_consistency import (
    FORMAL_DATASET_CONTRACTS,
    MAIN_FIGURE_GATE,
    TOP_FRACTION,
    evaluate_main_figure_gate,
    pairwise_cytobridge_metrics,
    prepare_shared_samples,
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
        method_status = dict(spec.get("method_status", {})).get(method, "complete")
        path_value = spec.get(path_key)
        if method_status != "complete" or not path_value:
            status_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "status": str(method_status),
                    "included_in_score_table": False,
                    "reason": str(dict(spec.get("method_reason", {})).get(method, "")),
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
                "reason": "",
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
    status = pd.DataFrame(status_rows)
    scores.to_csv(output / "directed_pair_method_scores.csv", index=False)
    metrics.to_csv(output / "cytobridge_external_metrics.csv", index=False)
    decisions.to_csv(output / "main_figure_method_decisions.csv", index=False)
    status.to_csv(output / "method_execution_status.csv", index=False)
    selected.to_csv(output / "selected_biological_pairs.csv", index=False)
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
    included = (
        decisions.loc[decisions.include_in_main_figure, "external_method"]
        .astype(str)
        .tolist()
    )
    if not included:
        raise ValueError("no external method passed the frozen main-figure gate")
    primary = metrics.loc[
        metrics.cytobridge_view.eq("CytoBridge exact message")
        & metrics.external_method.isin(included)
    ].copy()
    style.apply_style()
    fig = plt.figure(figsize=style.A4_PORTRAIT)
    grid = fig.add_gridspec(
        6,
        1,
        height_ratios=[0.20, 1.5, 0.20, 1.25, 0.20, 2.25],
        left=0.19,
        right=0.96,
        top=0.975,
        bottom=0.07,
        hspace=0.42,
    )
    head_a = fig.add_subplot(grid[0])
    _heading(head_a, "a", "Cross-dataset rank concordance")
    axes_a = grid[1].subgridspec(1, 2, wspace=0.38)
    ax_rho = fig.add_subplot(axes_a[0])
    ax_j = fig.add_subplot(axes_a[1])
    method_y = {method: index for index, method in enumerate(reversed(included))}
    offsets = np.linspace(-0.22, 0.22, len(FORMAL_DATASET_CONTRACTS))
    for offset, dataset in zip(offsets, FORMAL_DATASET_CONTRACTS, strict=True):
        table = primary.loc[primary.dataset.eq(dataset)]
        for row in table.itertuples():
            y = method_y[row.external_method] + offset
            ax_rho.scatter(
                row.spearman_rho,
                y,
                s=34,
                color=DATASET_COLORS[dataset],
                edgecolor="white",
                linewidth=0.5,
            )
            ax_j.scatter(
                row.top_jaccard,
                y,
                s=34,
                color=DATASET_COLORS[dataset],
                edgecolor="white",
                linewidth=0.5,
            )
    for axis, xlabel in ((ax_rho, "Spearman ρ"), (ax_j, "Top-20% Jaccard")):
        axis.axvline(0, color="#AAB2B8", lw=0.7)
        axis.set_yticks(list(method_y.values()), list(method_y.keys()))
        axis.set_xlabel(xlabel)
        style.clean_axis(axis, grid=True)
    ax_rho.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=DATASET_COLORS[d],
                markeredgecolor="white",
                label=str(FORMAL_DATASET_CONTRACTS[d]["display_name"]),
            )
            for d in FORMAL_DATASET_CONTRACTS
        ],
        frameon=False,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        fontsize=7.2,
    )
    head_b = fig.add_subplot(grid[2])
    _heading(head_b, "b", "Frozen inclusion decision")
    ax_b = fig.add_subplot(grid[3])
    decision = decisions.set_index("external_method").loc[included]
    x = np.arange(len(included))
    ax_b.bar(
        x - 0.18,
        decision.median_spearman_rho,
        width=0.36,
        color=[METHOD_COLORS[m] for m in included],
        alpha=0.90,
        label="Median ρ",
    )
    ax_b.bar(
        x + 0.18,
        decision.median_top_jaccard,
        width=0.36,
        color=[METHOD_COLORS[m] for m in included],
        alpha=0.45,
        hatch="///",
        label="Median Jaccard",
    )
    ax_b.axhline(
        float(MAIN_FIGURE_GATE["minimum_median_spearman_rho"]),
        color="#59616A",
        lw=0.8,
        ls="--",
    )
    ax_b.set_xticks(x, included)
    ax_b.set_ylabel("Across-dataset median")
    style.clean_axis(ax_b, grid=True)
    ax_b.legend(frameon=False, ncol=2)
    head_c = fig.add_subplot(grid[4])
    _heading(head_c, "c", "Highest shared terminal-stage communication programs")
    ax_c = fig.add_subplot(grid[5])
    display_methods = ["CytoBridge exact message", *included]
    offsets = np.linspace(-0.25, 0.25, len(display_methods))
    for row_index, row in selected.iterrows():
        for offset, method in zip(offsets, display_methods, strict=True):
            if method not in row or pd.isna(row[method]):
                continue
            color = (
                "#5B4B8A" if method.startswith("CytoBridge") else METHOD_COLORS[method]
            )
            ax_c.scatter(
                float(row[method]),
                row_index + offset,
                s=42,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
    labels = [
        f"{FORMAL_DATASET_CONTRACTS[row.dataset]['display_name']} · {row.sender_type} → {row.receiver_type}"
        for row in selected.itertuples()
    ]
    ax_c.set_yticks(range(len(labels)), labels, fontsize=7.4)
    ax_c.set_ylim(len(labels) - 0.55, -0.55)
    ax_c.set_xlim(0, 1.02)
    ax_c.axvline(0.8, color="#8A949C", lw=0.8, ls="--")
    ax_c.set_xlabel("Within-method rank percentile")
    style.clean_axis(ax_c, grid=True)
    ax_c.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=(
                    "#5B4B8A" if m.startswith("CytoBridge") else METHOD_COLORS[m]
                ),
                label=m,
            )
            for m in display_methods
        ],
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=7.2,
    )
    pdf = output / "spatial_communication_consistency_a4.pdf"
    png = output / "spatial_communication_consistency_a4.png"
    style.save_figure(fig, pdf, png, dpi=320)
    plt.close(fig)
    caption = "**Five-dataset spatial communication consistency.** (a) Terminal-stage rank concordance and top-20% directed-pair overlap between the exact CytoBridge interaction message contribution and external methods that passed a gate frozen before the five-dataset results were computed. (b) Across-dataset medians for the included methods. (c) One off-diagonal terminal-stage pair per dataset selected by a deterministic cross-method percentile rule. Native score units were never pooled. All attempted methods, including unavailable or weak methods omitted from the main panel, remain in the audit tables. Ligand–receptor, pathway, and target-gene interpretations must be attributed to the external molecular method that supplies them; a cell-type-pair rank alone does not identify a molecular mechanism. Chicken heart uses the human CellChatDB only as an exact conserved-symbol proxy. These results are descriptive computational consistency, not causal or independent experimental validation."
    (output / "caption.md").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source_manifest": _artifact(source / "manifest.json"),
        "included_methods": included,
        "gate": MAIN_FIGURE_GATE,
        "figure": {pdf.name: _artifact(pdf), png.name: _artifact(png)},
        "caption": _artifact(output / "caption.md"),
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
