#!/usr/bin/env python3
"""Build a self-contained, plain-language guide to the zebrafish CCC results.

This script deliberately leaves the frozen reviewer bundle untouched.  It
turns the same tables into three more direct figures and a Chinese reading
guide that separates positive evidence, limitations, and audit-only panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


STAGE_LABELS = {
    0.0: "5.25 hpf",
    1.0: "10 hpf",
    2.0: "12 hpf",
    3.0: "18 hpf",
    4.0: "24 hpf",
}
ORIGINAL_FIGURES = [
    "rank_concordance",
    "top_edge_overlap",
    "condition_coverage",
    "directionality_concordance",
    "stage_stability",
    "cytobridge_control_panel",
    "positive_consistency_overview",
    "top_signal_biology",
    "reviewer_validation_axes",
    "spatial_lr_interaction_maps",
    "ccc_circle_comparison",
    "known_lr_temporal_consistency_bubble",
]

SPATIAL_AUDIT_FIGURES = {
    "spatial_hotspot_consistency": "06_spatial_hotspot_consistency",
    "spatial_null_sensitivity": "07_spatial_null_sensitivity",
    "spatial_component_control": "08_spatial_component_control",
    "spatial_sender_receiver_consistency": "09_spatial_sender_receiver_consistency",
}
SPATIAL_AUDIT_TABLES = [
    "permutation_strata_diagnostics.csv",
    "spatial_primary_metrics.csv",
    "spatial_null_sensitivity.csv.gz",
    "spatial_component_control_metrics.csv",
    "spatial_sender_receiver_metrics.csv",
]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bundle-dir", required=True, type=Path)
    result.add_argument(
        "--spatial-consistency-dir",
        type=Path,
        help=(
            "Optional output from spatial_coordinate_consistency.py. When supplied, "
            "its manifest is verified and the density-controlled spatial audit is "
            "incorporated into the guide."
        ),
    )
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--overwrite", action="store_true")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def require(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def read_table(bundle: Path, name: str, required: list[str]) -> pd.DataFrame:
    path = bundle / "tables" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require(frame, required, path)
    return frame


def load_spatial_audit(directory: Path) -> dict[str, Any]:
    """Verify and load a formal coordinate-level spatial-consistency output."""
    root = directory.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing spatial audit manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("workflow") != "zebrafish_spatial_coordinate_consistency":
        raise ValueError("Unexpected spatial audit workflow identity")
    claims = manifest.get("claims", {})
    required_claims = {
        "spatial_consistency_not_ground_truth",
        "midpoint_overlap_not_direction_accuracy",
        "component_control_required_for_attention_increment",
        "selected_examples_not_all_lr_axes",
    }
    if any(claims.get(name) is not True for name in required_claims):
        raise ValueError("Spatial audit manifest is missing required claim guardrails")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Spatial audit manifest has no artifact inventory")
    inventory: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        relative = Path(str(artifact.get("path", "")))
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"Spatial artifact path escapes source directory: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(artifact.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Spatial artifact byte count mismatch: {relative}")
        if str(artifact.get("sha256", "")) != sha256(path):
            raise ValueError(f"Spatial artifact SHA256 mismatch: {relative}")
        inventory[str(relative)] = artifact

    required = set(SPATIAL_AUDIT_TABLES) | {"README_CN.md"}
    required |= {
        f"{name}.{suffix}"
        for name in SPATIAL_AUDIT_FIGURES
        for suffix in ("png", "pdf")
    }
    missing = sorted(required.difference(inventory))
    if missing:
        raise ValueError(f"Spatial audit manifest is missing artifacts: {missing}")

    parameters = manifest.get("parameters", {})
    if int(parameters.get("permutations", 0)) < 1:
        raise ValueError("Spatial audit must contain at least one permutation")
    primary = pd.read_csv(root / "spatial_primary_metrics.csv")
    null = pd.read_csv(root / "spatial_null_sensitivity.csv.gz")
    components = pd.read_csv(root / "spatial_component_control_metrics.csv")
    direction = pd.read_csv(root / "spatial_sender_receiver_metrics.csv")
    strata = pd.read_csv(root / "permutation_strata_diagnostics.csv")
    require(
        primary,
        [
            "example_id",
            "stage_label",
            "ligand",
            "receptor",
            "top_fraction",
            "scale_factor",
            "field_overlap_ovl",
            "hdr80_dice",
            "spatial_match_f1",
        ],
        root / "spatial_primary_metrics.csv",
    )
    require(
        null,
        [
            "example_id",
            "top_fraction",
            "scale_factor",
            "metric",
            "observed",
            "null_mean",
            "null_ci_low",
            "null_ci_high",
            "empirical_p_greater_equal",
            "n_permutations",
        ],
        root / "spatial_null_sensitivity.csv.gz",
    )
    require(
        components,
        [
            "example_id",
            "component",
            "field_overlap_ovl",
            "observed_minus_null_mean",
            "empirical_p_greater_equal",
            "delta_vs_lr_only",
        ],
        root / "spatial_component_control_metrics.csv",
    )
    require(
        direction,
        [
            "example_id",
            "direction",
            "cell_mass_overlap_ovl",
            "spearman_active_union_cells",
            "positive_cell_support_jaccard",
            "top20_positive_cell_jaccard",
        ],
        root / "spatial_sender_receiver_metrics.csv",
    )
    require(
        strata,
        [
            "example_id",
            "analysis",
            "method",
            "coarsening_level",
            "fraction_edges",
            "n_strata",
            "min_realized_stratum_size",
            "movable_edge_fraction_overall",
            "assignment_sha256",
        ],
        root / "permutation_strata_diagnostics.csv",
    )
    global_rows = strata.loc[strata["coarsening_level"].eq("global")]
    max_allowed = float(parameters.get("max_global_fallback_fraction", 0.0))
    if not global_rows.empty and global_rows["fraction_edges"].max() > max_allowed:
        raise ValueError("Spatial audit exceeds its declared global-fallback limit")
    if strata["movable_edge_fraction_overall"].min() < 0.95:
        raise ValueError("Spatial audit contains a mostly immovable permutation assignment")
    if primary["example_id"].nunique() != 3:
        raise ValueError("Expected exactly three frozen LR examples in spatial audit")
    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_path),
        "primary": primary,
        "null": null,
        "components": components,
        "direction": direction,
        "strata": strata,
    }


def top_set(group: pd.DataFrame, column: str, requested: int) -> set[int]:
    values = pd.to_numeric(group[column], errors="coerce")
    supported = values[np.isfinite(values) & values.gt(0)]
    if supported.empty:
        return set()
    boundary = supported.nlargest(min(requested, len(supported))).iloc[-1]
    return set(group.index[np.isfinite(values) & values.gt(0) & (values >= boundary)])


def prepare_stage_table(scores: pd.DataFrame, reported: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    formal = reported.loc[
        reported["target"].eq("CytoBridge attention")
        & reported["reference"].eq("External native consensus")
    ].set_index("stage")
    for stage, group in scores.groupby("stage", sort=True):
        group = group.copy()
        requested = max(1, int(round(len(group) * 0.20)))
        left = top_set(group, "cytobridge_attention", requested)
        right = top_set(group, "external_native_consensus", requested)
        shared = left & right
        group["attention_top20"] = group.index.isin(left)
        group["external_top20"] = group.index.isin(right)
        group["shared_top20"] = group.index.isin(shared)
        group["external_consensus_rank"] = group["external_native_consensus"].rank(
            method="average", pct=True
        )
        expected = formal.loc[float(stage)]
        observed = (len(left), len(right), len(shared))
        target = (
            int(expected["target_set_size_after_boundary_ties"]),
            int(expected["reference_set_size_after_boundary_ties"]),
            int(expected["intersection"]),
        )
        if observed != target:
            raise AssertionError(
                f"Top-set reconstruction differs at stage {stage}: {observed} != {target}"
            )
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def stage_summary(stage_table: pd.DataFrame, consensus: pd.DataFrame, top: pd.DataFrame) -> pd.DataFrame:
    consensus = consensus.loc[
        consensus["design"].eq("external_only_native_primary")
        & consensus["target"].eq("CytoBridge attention")
    ].copy()
    top = top.loc[
        top["target"].eq("CytoBridge attention")
        & top["reference"].eq("External native consensus")
    ].copy()
    selected = [
        "stage",
        "stage_label",
        "n_directed_pairs",
        "spearman",
    ]
    result = consensus[selected].merge(
        top[
            [
                "stage",
                "top_k_requested",
                "target_set_size_after_boundary_ties",
                "reference_set_size_after_boundary_ties",
                "intersection",
                "overlap_fraction_of_smaller_set",
                "overlap_enrichment_over_random",
                "bh_q_within_target_reference_family",
            ]
        ],
        on="stage",
        validate="one_to_one",
    )
    return result.sort_values("stage").reset_index(drop=True)


def plot_rank_scatter(stage_table: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18.0, 4.4), sharex=True, sharey=True)
    for ax, row in zip(axes, summary.itertuples(index=False)):
        group = stage_table.loc[stage_table["stage"].eq(row.stage)]
        other = group.loc[~group["shared_top20"]]
        shared = group.loc[group["shared_top20"]]
        ax.add_patch(
            Rectangle((0.8, 0.8), 0.2, 0.2, facecolor="#EEE8FF", edgecolor="none", zorder=0)
        )
        ax.scatter(
            other["external_consensus_rank"],
            other["cytobridge_attention_rank"],
            s=18,
            color="#C7CBD1",
            alpha=0.72,
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            shared["external_consensus_rank"],
            shared["cytobridge_attention_rank"],
            s=48,
            color="#6F42C1",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.axvline(0.8, color="#8E7CC3", lw=0.9, ls="--")
        ax.axhline(0.8, color="#8E7CC3", lw=0.9, ls="--")
        ax.plot([0, 1], [0, 1], color="#8B9198", lw=0.8, ls=":")
        ax.set_title(str(row.stage_label), fontsize=12, weight="bold")
        ax.text(
            0.04,
            0.96,
            f"rank correlation = {row.spearman:.2f}\nshared top 20% = {int(row.intersection)}/{int(row.target_set_size_after_boundary_ties)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#D5D8DC", "alpha": 0.94},
        )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 0.8, 1.0])
        ax.set_yticks([0, 0.5, 0.8, 1.0])
        ax.grid(color="#ECEFF1", lw=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("CytoBridge attention rank\nweak  →  strong", fontsize=11)
    fig.supxlabel("External-only consensus rank   weak  →  strong", fontsize=11, y=0.025)
    fig.suptitle(
        "Do CytoBridge and independent external methods rank the same cell-type arrows highly?",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.075,
        "Each dot is one directed sender→receiver cell-type pair. Purple dots are in both top-20% sets.",
        ha="center",
        fontsize=10,
        color="#4A4F55",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def pair_label(sender: str, receiver: str) -> str:
    if sender == receiver:
        return f"{sender} → {receiver} (homotypic type pair)"
    return f"{sender} → {receiver}"


def make_pair_checklist(stage_table: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for stage in (1.0, 4.0):
        group = stage_table.loc[stage_table["stage"].eq(stage)].copy()
        group["pair"] = [
            pair_label(str(sender), str(receiver))
            for sender, receiver in zip(group["sender_type"], group["receiver_type"])
        ]
        group["combined_rank"] = (
            group["cytobridge_attention_rank"] + group["external_consensus_rank"]
        ) / 2
        shared = group.loc[group["shared_top20"]].nlargest(8, "combined_rank").copy()
        shared["category"] = "shared_top20"
        attention_only = group.loc[
            group["attention_top20"] & ~group["external_top20"]
        ].nlargest(3, "cytobridge_attention_rank").copy()
        attention_only["category"] = "cytobridge_only_top20"
        external_only = group.loc[
            group["external_top20"] & ~group["attention_top20"]
        ].nlargest(3, "external_consensus_rank").copy()
        external_only["category"] = "external_only_top20"
        outputs.extend([shared, attention_only, external_only])
    selected = pd.concat(outputs, ignore_index=True)
    return selected[
        [
            "stage",
            "sender_type",
            "receiver_type",
            "pair",
            "category",
            "cytobridge_attention_rank",
            "external_consensus_rank",
        ]
    ]


def plot_pair_checklist(checklist: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 10.0))
    category_meta = {
        "shared_top20": ("AGREE", "#E8F5E9", "#207245"),
        "cytobridge_only_top20": ("CB ONLY", "#FFF3E0", "#A05A00"),
        "external_only_top20": ("EXT ONLY", "#E3F2FD", "#1C6395"),
    }
    for ax, stage in zip(axes, (1.0, 4.0)):
        ax.set_axis_off()
        row = summary.loc[summary["stage"].eq(stage)].iloc[0]
        local = checklist.loc[checklist["stage"].eq(stage)].copy()
        ax.text(
            0.0,
            1.04,
            STAGE_LABELS[stage],
            transform=ax.transAxes,
            fontsize=17,
            weight="bold",
            va="top",
        )
        ax.text(
            0.0,
            0.99,
            (
                f"Both rank {int(row.intersection)} arrows in their top 20% "
                f"({row.overlap_enrichment_over_random:.2f}× random expectation)."
            ),
            transform=ax.transAxes,
            fontsize=10.5,
            color="#40464D",
            va="top",
        )
        y = 0.92
        for category in ("shared_top20", "cytobridge_only_top20", "external_only_top20"):
            block = local.loc[local["category"].eq(category)]
            badge, background, color = category_meta[category]
            if category == "shared_top20":
                heading = "Concrete arrows both methods call high"
            elif category == "cytobridge_only_top20":
                heading = "Examples high only in CytoBridge"
            else:
                heading = "Examples high only in external consensus"
            ax.text(0.0, y, heading, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
            y -= 0.038
            for item in block.itertuples(index=False):
                height = 0.067 if category == "shared_top20" else 0.061
                ax.add_patch(
                    Rectangle(
                        (0.0, y - height + 0.006),
                        1.0,
                        height,
                        transform=ax.transAxes,
                        facecolor=background,
                        edgecolor="white",
                        linewidth=1.0,
                    )
                )
                ax.text(
                    0.012,
                    y - 0.012,
                    badge,
                    transform=ax.transAxes,
                    fontsize=8.4,
                    weight="bold",
                    color=color,
                    va="top",
                )
                wrapped = textwrap.fill(str(item.pair), width=47)
                ax.text(
                    0.125,
                    y - 0.006,
                    wrapped,
                    transform=ax.transAxes,
                    fontsize=8.7,
                    color="#25292D",
                    va="top",
                )
                ax.text(
                    0.985,
                    y - 0.006,
                    f"CB {item.cytobridge_attention_rank:.2f} | EXT {item.external_consensus_rank:.2f}",
                    transform=ax.transAxes,
                    fontsize=7.8,
                    color="#4F555B",
                    va="top",
                    ha="right",
                )
                y -= height
            y -= 0.026
    fig.suptitle(
        "Which sender→receiver arrows are actually consistent?",
        fontsize=16,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Ranks run from 0 (weak) to 1 (strong). The formal top-20% rule includes homotypic type pairs and all boundary ties.",
        ha="center",
        fontsize=9.5,
        color="#4A4F55",
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def headline_metrics(
    direct: pd.DataFrame,
    top: pd.DataFrame,
) -> dict[str, float]:
    required_direct = {
        "attention_vs_commot_rho",
        "exact_message_vs_commot_rho",
        "attention_vs_external_consensus_rho",
        "exact_message_vs_external_consensus_rho",
    }
    missing_direct = required_direct.difference(direct.columns)
    if missing_direct:
        raise ValueError(
            "Direct comparison table is missing headline columns: "
            f"{sorted(missing_direct)}"
        )
    attention_external_top = top.loc[
        top["target"].eq("CytoBridge attention")
        & top["reference"].eq("External native consensus")
    ]
    if attention_external_top.empty:
        raise ValueError(
            "Missing CytoBridge attention versus external-native top-signal rows"
        )
    return {
        "attention_commot": float(direct["attention_vs_commot_rho"].mean()),
        "exact_message_commot": float(
            direct["exact_message_vs_commot_rho"].mean()
        ),
        "attention_external": float(
            direct["attention_vs_external_consensus_rho"].mean()
        ),
        "exact_message_external": float(
            direct["exact_message_vs_external_consensus_rho"].mean()
        ),
        "top_enrichment": float(
            attention_external_top["overlap_enrichment_over_random"].mean()
        ),
    }


def plot_evidence_map(
    out: Path,
    *,
    headline: dict[str, float],
    spatial_audit: dict[str, Any] | None = None,
) -> None:
    if spatial_audit is not None:
        primary = spatial_audit["primary"]
        null = spatial_audit["null"]
        null_primary = null.loc[null["metric"].eq("field_overlap_ovl")].merge(
            primary[["example_id", "top_fraction", "scale_factor"]],
            on=["example_id", "top_fraction", "scale_factor"],
            validate="one_to_one",
        )
        audited = primary.merge(
            null_primary[["example_id", "null_ci_high"]],
            on="example_id",
            validate="one_to_one",
        )
        n_above = int(
            (audited["field_overlap_ovl"] > audited["null_ci_high"]).sum()
        )
        spatial_status = (
            "PARTIAL",
            "Raw LR hotspot overlap is visible, but "
            f"{n_above}/{len(audited)} examples exceed the fixed-support "
            "permutation-null interval; use as a diagnostic unless this test is positive.",
        )
    else:
        spatial_status = (
            "PARTIAL",
            "Raw asymmetric midpoint coverage is descriptive; no spatial-null or threshold-sensitivity test was supplied.",
        )
    rows = [
        (
            "Independent methods rank cell-type arrows similarly",
            "SUPPORTED",
            "External-only consensus vs attention: mean rank correlation = "
            f"{headline['attention_external']:.3f}; all five stages are positive.",
        ),
        (
            "Direct cell-type-pair rank agreement with a spatial CCC method",
            "SUPPORTED",
            "COMMOT is the strongest direct external comparison: mean stage "
            f"correlation = {headline['attention_commot']:.3f}.",
        ),
        (
            "The strongest signals overlap more than random",
            "SUPPORTED",
            "Top-20% overlap averages "
            f"{headline['top_enrichment']:.2f}× random expectation, but strength varies by stage.",
        ),
        (
            "Known signaling biology appears near the top",
            "SUPPORTED",
            "CXCL, NOTCH and non-canonical WNT pathways are enriched; NicheNet overlap is 50–100% in tested units.",
        ),
        (
            "LR-specific hotspot maps show raw overlap, not above-null validation",
            spatial_status[0],
            spatial_status[1],
        ),
        (
            "The result is not merely 'near cells score higher'",
            "PARTIAL",
            "Raw attention vs inverse distance is near zero, but the graph is already spatially local by construction.",
        ),
        (
            "Attention gives the exact ligand→receptor direction",
            "NOT SHOWN",
            "Forward LR residual association is positive, but reverse is not weaker; direction specificity is not established.",
        ),
        (
            "Virtual removal proves a causal perturbation response",
            "NOT SHOWN",
            "It is a one-model sensitivity analysis, not an experimental perturbation or causal test.",
        ),
    ]
    styles = {
        "SUPPORTED": ("#E7F5EC", "#176B3A"),
        "PARTIAL": ("#FFF4D6", "#8A5A00"),
        "NOT SHOWN": ("#FDE9E7", "#A5322A"),
    }
    fig, ax = plt.subplots(figsize=(15.5, 9.0))
    ax.set_axis_off()
    ax.text(
        0.0,
        1.03,
        "Can the current results answer the reviewer's concern?",
        transform=ax.transAxes,
        fontsize=20,
        weight="bold",
        va="top",
    )
    ax.text(
        0.0,
        0.975,
        "Short answer: yes for communication-relevant organization; no for literal biochemical strength, exact direction, or causality.",
        transform=ax.transAxes,
        fontsize=12,
        color="#3F454B",
        va="top",
    )
    y, height = 0.91, 0.101
    for question, status, evidence in rows:
        background, color = styles[status]
        ax.add_patch(
            Rectangle(
                (0.0, y - height + 0.008),
                1.0,
                height - 0.008,
                transform=ax.transAxes,
                facecolor=background,
                edgecolor="white",
                linewidth=1.0,
            )
        )
        ax.text(
            0.018,
            y - 0.018,
            status,
            transform=ax.transAxes,
            fontsize=10.5,
            weight="bold",
            color=color,
            va="top",
        )
        ax.text(
            0.165,
            y - 0.012,
            question,
            transform=ax.transAxes,
            fontsize=11.2,
            weight="bold",
            color="#202428",
            va="top",
        )
        ax.text(
            0.165,
            y - 0.050,
            evidence,
            transform=ax.transAxes,
            fontsize=9.7,
            color="#454A50",
            va="top",
        )
        y -= height
    ax.text(
        0.0,
        0.012,
        "Recommended wording: attention captures communication-relevant, biologically coherent interaction organization.",
        transform=ax.transAxes,
        fontsize=11,
        weight="bold",
        color="#32205F",
        va="bottom",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def interaction_term_dictionary() -> pd.DataFrame:
    """Return the small vocabulary needed to interpret every result."""

    rows = [
        (
            "candidate_edge",
            "Candidate spatial edge i→j",
            "cell edge",
            "spatial cutoff + learned link-predictor threshold",
            True,
            False,
            "Defines which neighboring cells can exchange a model message",
            "Not used as the primary CCC score",
        ),
        (
            "attention",
            "Attention gate |αᵢⱼ|",
            "cell edge",
            "absolute mean of signed, non-softmax gates across heads",
            True,
            False,
            "Model edge importance/gating; reviewer-target quantity",
            "Not a probability; magnitude discards the sign and has no ligand/receptor identity",
        ),
        (
            "exact_message",
            "Exact edge message ‖mᵢⱼ‖",
            "cell edge",
            "norm of the complete edge-contribution vector reconstructed exactly from the trained layer",
            True,
            False,
            "Deterministic decomposition/readout of the native one-layer interaction output",
            "Exact means algebraically exact reconstruction, not biological truth or biochemical flux",
        ),
        (
            "type_attention",
            "G_AB attention",
            "cell-type pair",
            "mean |αᵢⱼ| over edges with sender type A and receiver type B",
            False,
            False,
            "Primary direct comparison of the attention quantity with CCC methods",
            "A type-pair ranking, not an LR-specific signal",
        ),
        (
            "type_exact_message",
            "D_AB exact message",
            "cell-type pair",
            "per receiver, sum complete messages from sender type A; take norm; average over all type-B receivers",
            False,
            False,
            "Most functional CytoBridge quantity in the direct CCC comparison",
            "Not the mean of edge norms: vectors may cancel before the norm; not biochemical flux",
        ),
        (
            "lr_compatible_attention",
            "LR-compatible attention",
            "LR-specific cell edge",
            "|αᵢⱼ| × scaled ligand expression in i × scaled receptor expression in j",
            False,
            True,
            "Specific LR spatial examples and LR-axis/pathway ranking",
            "Post-hoc composite; not a native LR-specific attention head",
        ),
        (
            "lr_compatible_message",
            "LR-compatible exact message",
            "LR-specific cell edge",
            "‖mᵢⱼ‖ × scaled ligand expression in i × scaled receptor expression in j",
            False,
            True,
            "Sensitivity analysis for identifiable LR axes",
            "Post-hoc composite; not biochemical flux",
        ),
        (
            "virtual_removal",
            "Virtual-removal response",
            "trajectory-level sensitivity",
            "difference after removing a starting cell type and re-simulating",
            False,
            False,
            "Tests whether a predicted trajectory depends on a population",
            "Not an interaction score and not causal perturbation evidence",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "term_id",
            "display_name",
            "level",
            "formula_or_definition",
            "model_native",
            "lr_specific",
            "used_for",
            "claim_boundary",
        ],
    )


def result_inventory(*, spatial_audit_available: bool = False) -> pd.DataFrame:
    rows = [
        (
            "Direct CCC rank consistency",
            "G_AB attention; D_AB exact message",
            "COMMOT, CellChat, CellAgentChat type-pair scores",
            "pairwise_consistency_by_stage.csv",
            "02_direct_ccc_comparison",
            "Main direct evidence",
        ),
        (
            "External-only consensus",
            "G_AB attention; D_AB exact message",
            "mean within-stage rank of COMMOT + CellAgentChat CTPS + CellChat triMean",
            "consensus_by_stage.csv",
            "04_external_consensus_rank_scatter",
            "Main robustness evidence",
        ),
        (
            "Top-signal overlap",
            "top 20% G_AB or D_AB arrows",
            "top 20% external-consensus arrows",
            "top_signal_overlap_by_stage.csv",
            "05_top_communication_arrows_checklist",
            "Supporting evidence; stage dependent",
        ),
        (
            "LR spatial examples",
            "LR-compatible attention",
            "COMMOT LR-specific cell flow",
            (
                "spatial_primary_metrics.csv + spatial_null_sensitivity.csv.gz"
                if spatial_audit_available
                else "spatial_display_audit.csv"
            ),
            (
                "06_spatial_hotspot_consistency + 07_spatial_null_sensitivity"
                if spatial_audit_available
                else "03_spatial_location_coverage"
            ),
            (
                "Diagnostic/limitation: raw overlap does not exceed fixed-support null"
                if spatial_audit_available
                else "Illustrative spatial support; no null-model inference"
            ),
        ),
        (
            "Pathway enrichment",
            "top LR-compatible attention axes",
            "full project LR database background",
            "pathway_enrichment.csv",
            "top_signal_biology",
            "Biological interpretability; partly prior-conditioned",
        ),
        (
            "NicheNet downstream consistency",
            "top LR-compatible attention ligands",
            "NicheNet ligand→target rankings",
            "nichenet_downstream_consistency_summary.csv",
            "top_signal_biology",
            "Complementary downstream support, not direct CCC agreement",
        ),
        (
            "Proximity diagnostic",
            "G_AB attention; D_AB exact message",
            "inverse mean spatial distance",
            "spatial_proximity_by_stage.csv",
            "positive_consistency_overview",
            "Confounder audit; negative values are not negative CCC consistency",
        ),
        (
            "Conditional LR audit",
            "confounder-residual attention/message",
            "forward and reverse LR compatibility",
            "spatial_beyond_proximity_conditional_tests.csv",
            "cytobridge_control_panel",
            "Internal consistency; does not establish direction",
        ),
        (
            "Virtual removal",
            "trajectory response after population removal",
            "baseline simulation",
            "virtual_ablation_summary.csv",
            "reviewer_validation_axes",
            "Model sensitivity only; not causal",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "result",
            "cytobridge_quantity",
            "compared_with",
            "authoritative_table",
            "figure",
            "role",
        ],
    )


def direct_comparison_table(
    pairwise: pd.DataFrame, consensus: pd.DataFrame, scores: pd.DataFrame
) -> pd.DataFrame:
    required = [
        "display_label_left",
        "display_label_right",
        "stage",
        "spearman_rank_concordance",
        "top_k_intersection",
        "effective_top_k",
    ]
    require(pairwise, required, Path("pairwise_consistency_by_stage.csv"))
    commot = pairwise.loc[
        pairwise["display_label_right"].eq("COMMOT | project LR")
        & pairwise["display_label_left"].isin(
            ["CytoBridge attention", "CytoBridge exact message"]
        )
    ].copy()
    if len(commot) != 10:
        raise ValueError("Expected two CytoBridge×COMMOT rows at each of five stages")
    direct = commot.pivot(
        index="stage", columns="display_label_left", values="spearman_rank_concordance"
    ).rename(
        columns={
            "CytoBridge attention": "attention_vs_commot_rho",
            "CytoBridge exact message": "exact_message_vs_commot_rho",
        }
    )
    overlap = commot.pivot(
        index="stage", columns="display_label_left", values="top_k_intersection"
    ).rename(
        columns={
            "CytoBridge attention": "attention_commot_top10_intersection",
            "CytoBridge exact message": "exact_message_commot_top10_intersection",
        }
    )
    external = consensus.loc[
        consensus["design"].eq("external_only_native_primary")
        & consensus["target"].isin(
            ["CytoBridge attention", "CytoBridge exact message"]
        )
    ].pivot(index="stage", columns="target", values="spearman").rename(
        columns={
            "CytoBridge attention": "attention_vs_external_consensus_rho",
            "CytoBridge exact message": "exact_message_vs_external_consensus_rho",
        }
    )
    result = direct.join(overlap).join(external).reset_index()
    result.insert(1, "stage_label", result["stage"].map(STAGE_LABELS))
    result["commot_top_k"] = 10

    for row in result.itertuples(index=False):
        stage = scores.loc[scores["stage"].eq(row.stage)]
        for column, expected in (
            ("cytobridge_attention", row.attention_vs_commot_rho),
            ("cytobridge_exact_message", row.exact_message_vs_commot_rho),
        ):
            observed = stage[column].corr(stage["commot"], method="spearman")
            if not np.isclose(observed, expected, atol=1e-12, rtol=1e-10):
                raise AssertionError(
                    f"Direct COMMOT correlation mismatch at stage {row.stage}: "
                    f"{column} {observed} != {expected}"
                )
    return result.sort_values("stage").reset_index(drop=True)


def plot_computation_map(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.0, 8.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.012",
                facecolor=fc,
                edgecolor=ec,
                linewidth=1.3,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.2)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color="#72777D",
                connectionstyle="arc3,rad=0",
            )
        )

    def curved_arrow(
        x1: float, y1: float, x2: float, y2: float, radius: float
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.2,
                color="#72777D",
                connectionstyle=f"arc3,rad={radius}",
            )
        )

    ax.text(
        0.5,
        0.975,
        "What CytoBridge computes — and which quantity each analysis actually uses",
        ha="center",
        va="top",
        fontsize=17,
        weight="bold",
    )
    ax.text(
        0.5,
        0.93,
        "There is no single object called 'the CytoBridge CCC score'.",
        ha="center",
        va="top",
        fontsize=11.5,
        color="#4B5056",
    )

    box(0.02, 0.69, 0.15, 0.15, "Observed cells\nstate + aligned space", "#F2F3F4", "#AEB4BA")
    box(0.21, 0.69, 0.16, 0.15, "Candidate edge i→j\ncutoff + link predictor", "#F2F3F4", "#AEB4BA")
    box(0.42, 0.78, 0.16, 0.12, "Attention magnitude |αᵢⱼ|\n(sign discarded)", "#EEE8FF", "#7453B3")
    box(0.42, 0.61, 0.16, 0.12, "Exact edge contribution\n‖mᵢⱼ‖ (algebraic readout)", "#E7F2FD", "#3477A9")
    box(0.63, 0.69, 0.16, 0.15, "Aggregate by cell type\nG_AB and D_AB", "#E8F5E9", "#2E7D4F")
    box(0.83, 0.69, 0.15, 0.15, "DIRECT CCC TEST\nrank vs COMMOT /\nCellChat / CellAgentChat", "#DDF3E4", "#176B3A")
    arrow(0.17, 0.765, 0.21, 0.765)
    arrow(0.37, 0.765, 0.42, 0.84)
    arrow(0.37, 0.765, 0.42, 0.67)
    arrow(0.58, 0.84, 0.63, 0.78)
    arrow(0.58, 0.67, 0.63, 0.74)
    arrow(0.79, 0.765, 0.83, 0.765)

    box(0.21, 0.32, 0.16, 0.15, "Ligand in sender ×\nreceptor in receiver", "#FFF4DF", "#B26A12")
    box(0.42, 0.32, 0.16, 0.15, "LR-compatible score\n|α|×L×R or ‖m‖×L×R", "#FFF0D9", "#B26A12")
    box(0.63, 0.32, 0.16, 0.15, "LR-specific spatial map\n+ pathway ranking", "#FFF4DF", "#B26A12")
    box(0.83, 0.32, 0.15, 0.15, "BIOLOGICAL SUPPORT\nWNT / NOTCH / CXCL\nNicheNet overlap", "#FFF4DF", "#B26A12")
    arrow(0.37, 0.395, 0.42, 0.395)
    arrow(0.50, 0.61, 0.50, 0.47)
    curved_arrow(0.56, 0.78, 0.56, 0.47, -0.42)
    arrow(0.58, 0.395, 0.63, 0.395)
    arrow(0.79, 0.395, 0.83, 0.395)

    box(0.42, 0.07, 0.16, 0.13, "Distance / state /\ndegree controls", "#F4F4F4", "#888D92")
    box(0.63, 0.07, 0.16, 0.13, "Proximity diagnostic\ncan be near zero/negative", "#F4F4F4", "#888D92")
    box(0.83, 0.07, 0.15, 0.13, "Virtual removal\ntrajectory sensitivity", "#FDEBE9", "#A94B43")
    arrow(0.58, 0.135, 0.63, 0.135)
    ax.text(
        0.02,
        0.52,
        "MODEL GRAPH + INTERACTION READOUTS",
        fontsize=10,
        weight="bold",
        color="#563C8C",
    )
    ax.text(
        0.02,
        0.26,
        "POST-HOC LR OVERLAY",
        fontsize=10,
        weight="bold",
        color="#94550B",
    )
    ax.text(
        0.02,
        0.105,
        "CONTROLS / SENSITIVITY",
        fontsize=10,
        weight="bold",
        color="#555A60",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_direct_ccc_scatter(
    scores: pd.DataFrame, direct: pd.DataFrame, out: Path
) -> None:
    rows = [
        ("cytobridge_attention_rank", "attention_vs_commot_rho", "CytoBridge attention", "#C56A16"),
        ("cytobridge_exact_message_rank", "exact_message_vs_commot_rho", "CytoBridge exact message", "#356F9F"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(18.0, 7.4), sharex=True, sharey=True)
    for row_index, (score_col, rho_col, label, color) in enumerate(rows):
        for column_index, stage in enumerate(sorted(scores["stage"].unique())):
            ax = axes[row_index, column_index]
            group = scores.loc[scores["stage"].eq(stage)]
            shared_high = group[score_col].ge(0.8) & group["commot_rank"].ge(0.8)
            ax.add_patch(
                Rectangle((0.8, 0.8), 0.2, 0.2, facecolor="#EEF2F5", edgecolor="none")
            )
            ax.scatter(
                group.loc[~shared_high, "commot_rank"],
                group.loc[~shared_high, score_col],
                s=15,
                color="#C9CDD2",
                alpha=0.68,
                linewidths=0,
            )
            ax.scatter(
                group.loc[shared_high, "commot_rank"],
                group.loc[shared_high, score_col],
                s=35,
                color=color,
                alpha=0.92,
                edgecolor="white",
                linewidth=0.6,
            )
            ax.axvline(0.8, color="#A7ADB3", lw=0.8, ls="--")
            ax.axhline(0.8, color="#A7ADB3", lw=0.8, ls="--")
            ax.plot([0, 1], [0, 1], color="#858B91", lw=0.7, ls=":")
            rho = direct.loc[direct["stage"].eq(stage), rho_col].item()
            ax.text(
                0.04,
                0.95,
                f"rho = {rho:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D4D8DC"},
            )
            if row_index == 0:
                ax.set_title(STAGE_LABELS[float(stage)], fontsize=11.5, weight="bold")
            if column_index == 0:
                ax.set_ylabel(f"{label} rank\nweak → strong", fontsize=10)
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.set_xticks([0, 0.5, 0.8, 1.0])
            ax.set_yticks([0, 0.5, 0.8, 1.0])
            ax.grid(color="#ECEFF1", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "The direct CCC comparison: the same sender→receiver cell-type pair in CytoBridge and COMMOT",
        fontsize=15,
        weight="bold",
        y=1.015,
    )
    fig.supxlabel("COMMOT rank within the same stage   weak → strong", fontsize=11, y=0.025)
    fig.text(
        0.5,
        -0.035,
        "Each dot is one identical directed cell-type pair. Colored = top 20% in both methods; gray = all other pairs.",
        ha="center",
        fontsize=10,
        color="#4B5056",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_coverage(audit: pd.DataFrame, out: Path) -> None:
    audit = audit.copy()
    audit["axis"] = (
        audit["ligand"].str.upper()
        + "→"
        + audit["receptor"].str.upper()
        + "  ("
        + audit["stage"].map(STAGE_LABELS)
        + ")"
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    specs = [
        (
            "cytobridge_midpoints_near_commot_fraction",
            "Of CytoBridge high-score locations,\nhow many have COMMOT nearby?",
            "#C56A16",
            "n_cytobridge_top_fraction_midpoints",
        ),
        (
            "commot_midpoints_near_cytobridge_fraction",
            "Of COMMOT high-score locations,\nhow many have CytoBridge nearby?",
            "#19847A",
            "n_commot_top_fraction_midpoints",
        ),
    ]
    y = np.arange(len(audit))
    for ax, (column, title, color, n_column) in zip(axes, specs):
        values = audit[column].to_numpy(float)
        ax.barh(y, np.ones(len(values)), color="#ECEFF1", height=0.58)
        ax.barh(y, values, color=color, height=0.58)
        for yi, value, n_value in zip(y, values, audit[n_column]):
            ax.text(
                min(value + 0.025, 0.96),
                yi,
                f"{value:.1%}  (n={int(n_value)})",
                va="center",
                ha="left" if value < 0.82 else "right",
                fontsize=10,
                color="#202428",
                weight="bold",
            )
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(title, fontsize=11.5, weight="bold")
        ax.grid(axis="x", color="#E4E7EA", lw=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, audit["axis"], fontsize=10.5)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Spatial location coverage — a quantitative replacement for visually counting arrows",
        fontsize=14,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "n = top-20% cell edges (one midpoint per edge). A match lies within half the frozen graph cutoff; this is descriptive, not exact-edge accuracy.",
        ha="center",
        fontsize=9.5,
        color="#4B5056",
    )
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_figure_index(
    *,
    spatial_audit_available: bool = False,
    headline: dict[str, float] | None = None,
    spatial_audit: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if headline is None:
        direct_conclusion = "从当前结果表动态读取 attention 与 exact-message 的平均相关"
    else:
        direct_conclusion = (
            f"attention 平均 rho={headline['attention_commot']:.3f}；"
            f"exact message={headline['exact_message_commot']:.3f}"
        )
    if spatial_audit is not None:
        spatial_audit_available = True
        primary = spatial_audit["primary"]
        null = spatial_audit["null"]
        null_primary = null.loc[null["metric"].eq("field_overlap_ovl")].merge(
            primary[["example_id", "top_fraction", "scale_factor"]],
            on=["example_id", "top_fraction", "scale_factor"],
            validate="one_to_one",
        )
        audited = primary.merge(
            null_primary[["example_id", "null_ci_high"]],
            on="example_id",
            validate="one_to_one",
        )
        ovl_min = float(primary["field_overlap_ovl"].min())
        ovl_max = float(primary["field_overlap_ovl"].max())
        n_above = int(
            (audited["field_overlap_ovl"] > audited["null_ci_high"]).sum()
        )
        spatial_description = (
            f"raw OVL 为 {ovl_min:.3f}–{ovl_max:.3f}；"
            "只能先说明大区域部分重合"
        )
        spatial_null_conclusion = (
            f"{n_above}/{len(audited)} 条轴高于 null 区间；"
            "本结果不构成额外独立空间验证"
        )
    else:
        spatial_description = "raw OVL 由正式空间表读取；只能先说明大区域重合"
        spatial_null_conclusion = (
            "需结合正式 null 表判断；未通过时不构成额外独立空间验证"
        )
    rows = [
        ("00_computation_to_result_map", "第一张", "CytoBridge 到底计算了哪些 interaction 量，每项分析用了哪一个", "先建立唯一术语和计算主线"),
        ("02_direct_ccc_comparison", "最主要直接证据", "同一个 sender→receiver type pair 在 CytoBridge 与 COMMOT 中的排名是否一致", direct_conclusion),
        ("03_spatial_location_coverage", "旧空间描述/审计", "一个方法的高位中点附近能否找到另一个方法的点", "受点密度影响，只能描述 raw coverage，不能作一致性推断"),
        ("01_reviewer_evidence_map", "新主图", "审稿人的各项疑问目前分别得到什么答案", "结论总览；先看这一张"),
        ("04_external_consensus_rank_scatter", "稳健性主图", "每个细胞类型箭头在双方排名中是否同时靠前", "外部多方法共识的总体一致性"),
        ("05_top_communication_arrows_checklist", "补充直观图", "具体哪些 sender→receiver 箭头一致，哪些不一致", "把抽象 overlap 还原成具体生物学对象"),
        ("spatial_lr_interaction_maps", "旧空间示意", "三个已知 LR 轴在真实空间中的箭头分布", "只显示组织背景；肉眼箭头不承担一致性验证"),
        ("known_lr_temporal_consistency_bubble", "生物学补充证据", "已知 LR 轴随时间在两种方法中的相对强弱", "同一信号轴是否被双方同时列为高位"),
        ("top_signal_biology", "生物学补充证据", "高 attention 信号富集哪些通路，并与 NicheNet 下游结果重合多少", "支持已知生物通路，但不是因果证据"),
        ("ccc_circle_comparison", "补充证据", "10 和 24 hpf 的高位细胞类型箭头网络", "适合展示整体网络；不适合精确数边"),
        ("positive_consistency_overview", "补充摘要", "共识、top overlap、self-inclusion 和距离关系的统计摘要", "信息全但较抽象，建议不作为第一张"),
        ("rank_concordance", "补充/审计", "所有方法两两排序相关", "说明 COMMOT 与 attention 最一致；矩阵本身较密"),
        ("top_edge_overlap", "补充/审计", "各方法 top-k 集合的 Jaccard 重叠", "方法定义差异大，不能单独据此判定成败"),
        ("stage_stability", "生物学审计", "同一方法相邻发育时点是否稳定", "反映发育变化，不是跨方法验证"),
        ("condition_coverage", "质量审计", "每种方法实际评估了多少细胞类型方向对", "只看覆盖范围，不是性能图"),
        ("directionality_concordance", "限制/审计", "A→B 与 B→A 的方向差是否跨方法一致", "稀疏且混合，不支持精确方向主张"),
        ("cytobridge_control_panel", "限制/审计", "距离、状态、度数、初始化和随机化控制", "attention 有部分正证据，但 exact-message 控制混合"),
        ("reviewer_validation_axes", "限制/审计", "内部 LR、方向、空间及虚拟移除测试", "不能把虚拟移除写成实验因果"),
    ]
    if spatial_audit_available:
        rows[6:6] = [
            ("06_spatial_hotspot_consistency", "空间描述", "两种方法在同一坐标上的单位质量热点与 80% 高密度区重合多少", spatial_description),
            ("07_spatial_null_sensitivity", "空间主审计", "热点重合是否超过固定 edge-support 的分数置换基线", spatial_null_conclusion),
            ("08_spatial_component_control", "空间关键控制", "attention×LR 是否比 LR-only 更接近 COMMOT", "三条轴增量均为负；空间结构不能归因于 attention 增量"),
            ("09_spatial_sender_receiver_consistency", "方向限制/审计", "双方是否把相同 sender/receiver 细胞排在高位", "active-cell rho 接近零或混合，不能由 midpoint 重合推出方向一致"),
        ]
    return pd.DataFrame(rows, columns=["figure", "recommended_role", "plain_question", "plain_conclusion"])


def spatial_audit_report_section(spatial_audit: dict[str, Any]) -> str:
    """Build one plain-language, CSV-backed explanation of all spatial panels."""
    primary = spatial_audit["primary"].copy()
    null = spatial_audit["null"].copy()
    components = spatial_audit["components"].copy()
    direction = spatial_audit["direction"].copy()
    strata = spatial_audit["strata"].copy()
    parameters = spatial_audit["manifest"].get("parameters", {})

    null_primary = null.loc[null["metric"].eq("field_overlap_ovl")].merge(
        primary[["example_id", "top_fraction", "scale_factor"]],
        on=["example_id", "top_fraction", "scale_factor"],
        how="inner",
        validate="one_to_one",
    )
    if len(null_primary) != len(primary):
        raise ValueError("Spatial null table does not contain every primary OVL setting")
    merged = primary.merge(
        null_primary[
            [
                "example_id",
                "null_mean",
                "null_ci_low",
                "null_ci_high",
                "observed_minus_null_mean",
                "empirical_p_greater_equal",
                "n_permutations",
            ]
        ],
        on="example_id",
        validate="one_to_one",
    )
    primary_rows = "\n".join(
        (
            f"| {row.ligand.upper()}→{row.receptor.upper()} ({row.stage_label}) | "
            f"{row.field_overlap_ovl:.3f} | {row.hdr80_dice:.3f} | "
            f"{row.spatial_match_f1:.3f} | {int(row.n_cytobridge_top_edges)} / "
            f"{int(row.n_commot_top_edges)} |"
        )
        for row in merged.itertuples(index=False)
    )
    null_rows = "\n".join(
        (
            f"| {row.ligand.upper()}→{row.receptor.upper()} | {row.field_overlap_ovl:.3f} | "
            f"{row.null_mean:.3f} [{row.null_ci_low:.3f}, {row.null_ci_high:.3f}] | "
            f"{row.observed_minus_null_mean:+.3f} | {row.empirical_p_greater_equal:.4f} |"
        )
        for row in merged.itertuples(index=False)
    )

    component_pivot = components.pivot(
        index="example_id", columns="component", values="field_overlap_ovl"
    )
    component_delta = components.loc[
        components["component"].eq("attention_lr"),
        ["example_id", "delta_vs_lr_only"],
    ].set_index("example_id")
    component_rows = "\n".join(
        (
            f"| {row.ligand.upper()}→{row.receptor.upper()} | "
            f"{component_pivot.loc[row.example_id, 'lr_only']:.3f} | "
            f"{component_pivot.loc[row.example_id, 'attention_lr']:.3f} | "
            f"{component_delta.loc[row.example_id, 'delta_vs_lr_only']:+.3f} | "
            f"{component_pivot.loc[row.example_id, 'exact_message_lr']:.3f} |"
        )
        for row in merged.itertuples(index=False)
    )

    direction_pivot = direction.pivot(
        index="example_id",
        columns="direction",
        values=["spearman_active_union_cells", "cell_mass_overlap_ovl", "top20_positive_cell_jaccard"],
    )
    direction_rows = "\n".join(
        (
            f"| {row.ligand.upper()}→{row.receptor.upper()} | "
            f"{direction_pivot.loc[row.example_id, ('spearman_active_union_cells', 'outgoing')]:.3f} | "
            f"{direction_pivot.loc[row.example_id, ('cell_mass_overlap_ovl', 'outgoing')]:.3f} | "
            f"{direction_pivot.loc[row.example_id, ('spearman_active_union_cells', 'incoming')]:.3f} | "
            f"{direction_pivot.loc[row.example_id, ('cell_mass_overlap_ovl', 'incoming')]:.3f} |"
        )
        for row in merged.itertuples(index=False)
    )
    permutations = int(merged["n_permutations"].min())
    n_positive_delta = int((component_delta["delta_vs_lr_only"] > 0).sum())
    exact_positive = components.loc[
        components["component"].eq("exact_message_lr")
        & components["observed_minus_null_mean"].gt(0)
        & components["empirical_p_greater_equal"].lt(0.05)
    ]
    if len(exact_positive) == 1:
        exact_row = exact_positive.iloc[0]
        exact_message_note = (
            f"唯一超过 modifier-permutation null 的探索性分量是 "
            f"{str(exact_row.ligand).upper()}→{str(exact_row.receptor).upper()} 的 "
            f"exact-message×LR（OVL={exact_row.field_overlap_ovl:.3f}，"
            f"null mean={exact_row.null_mean:.3f}，未校正单侧 P={exact_row.empirical_p_greater_equal:.4f}）。"
            "它是 exact-message 的轴特异结果，不能改写成 attention 获得了空间验证。"
        )
    elif exact_positive.empty:
        exact_message_note = "没有 exact-message×LR 轴超过对应 modifier-permutation null。"
    else:
        exact_message_note = (
            f"有 {len(exact_positive)} 条 exact-message×LR 轴超过对应 null；"
            "这些仍是分量级探索结果，不能替代 attention 的增量检验。"
        )
    global_fraction = float(
        strata.loc[strata["coarsening_level"].eq("global"), "fraction_edges"].max()
        if strata["coarsening_level"].eq("global").any()
        else 0.0
    )
    movable_fraction = float(strata["movable_edge_fraction_overall"].min())
    permutation_bins = int(parameters.get("permutation_bins", 0))
    min_stratum = int(parameters.get("min_permutation_stratum", 0))

    return f"""> **空间部分先读这一句：三条轴都有部分 raw 大区域重合，但 0/3 超过 fixed-support null，attention×LR 也 0/3 优于 LR-only；所以空间图是诊断，不是正向主证据。**

这三条轴不是看完空间图后凭外观挑的。流程先固定 ncWNT、CXCL、NOTCH 三个 pathway family，再在每个 family 内选择 CytoBridge attention×LR 与 COMMOT stage-percentile 均值最高、且双方正分 support 与 `n_active_edges≥10` 的轴；选择发生在 cell-level COMMOT flow 重建之前。这个规则避免“看哪张图漂亮就挑哪张”，但三条轴仍是规则筛选的代表例子，不能代表全部 LR axes。

### 先看旧覆盖图为什么会显得很强

[旧覆盖图](figures/03_spatial_location_coverage.png) 统计“每个点附近是否至少有另一个方法的点”。它允许同一个密集 COMMOT 点被很多 CytoBridge 点重复命中，所以 92.8% 或 99.1% 会随点密度升高。这个数字可以描述位置覆盖，**不能用来推断高分空间排序一致**，也不能证明方向正确。

### 空间图 A：同一坐标上的热点到底重合多少？

![Spatial hotspot consistency](figures/06_spatial_hotspot_consistency.png)

每行是一条预先固定的 LR 轴。前两列分别是 CytoBridge 和 COMMOT 在同一 `spatial_aligned` 坐标上的 top-edge midpoint 热点；第三列叠加双方 50%/80% 高密度区轮廓；第四列直接涂出双方共有、CytoBridge-only 和 COMMOT-only 区域。Raw score 只负责选择 top edges；入选后每条 edge 在 hotspot histogram 中等权计数，不再按两个方法不可比的 raw score 加权。

- `OVL`：两个单位总质量热点场的重叠，0=完全分开，1=完全一样；
- `Dice80`：双方 80% 高密度区域的面积重叠；
- `MatchF1`：半径内最大一对一中点匹配，一个 COMMOT 点不能被重复使用。

| LR 轴 | raw OVL | Dice80 | one-to-one MatchF1 | top edges CB / COMMOT |
|---|---:|---:|---:|---:|
{primary_rows}

这一步只能说“大组织区域看起来有多少重合”。它还不能回答这种重合是否只是组织形状、可用 edge support 和 LR expression geography 共同造成的。

### 空间图 B：重合有没有超过控制共同 edge support 和坐标的置换基线？

![Spatial null and sensitivity](figures/07_spatial_null_sensitivity.png)

实线是观察值；虚线和淡色带是固定双方正分 edge support、细胞坐标和 top-edge 数量后得到的 null 均值和 95% 区间。置换采用可审计的自适应层级：先尝试保留 sender→receiver type 与 {permutation_bins} 档距离/LR activity；不足 {min_stratum} 条边时逐级合并到 covariate-only、distance-only，最后才允许少量 global fallback。正式结果使用 {permutations} 次置换，并同时改变 top fraction 和空间尺度检查敏感性。本次最大 global fallback 为 {global_fraction:.1%}，最小 movable-edge fraction 为 {movable_fraction:.1%}；完整 realized strata 见 [permutation_strata_diagnostics.csv](tables/permutation_strata_diagnostics.csv)。

| LR 轴 | observed OVL | null mean [95% interval] | observed−null | 单侧 enrichment P |
|---|---:|---:|---:|---:|
{null_rows}

三条轴在主设置下都低于 null 区间。因此最直白的解释是：**能看到共同组织区域，但没有证据表明双方对“哪些边最强”的空间排序比共同 support geography 更一致。** 观察值低于这个单侧 enrichment null 也不等于证明生物学关系“相反”；这里只能说没有得到高于基线的空间一致性证据。

### 空间图 C：空间重合来自 attention，还是来自 LR expression 本身？

![Spatial component control](figures/08_spatial_component_control.png)

这张图把同一批 LR-positive CytoBridge edges 分别用 `LR-only`、`attention-only`、`attention×LR` 和 `exact-message×LR` 排序，再与 COMMOT 比。判断 attention 是否带来额外信息，要看 `attention×LR − LR-only`，而不是只看 `attention×LR` 的绝对 OVL。

| LR 轴 | LR-only OVL | attention×LR OVL | 增量 | exact-message×LR OVL |
|---|---:|---:|---:|---:|
{component_rows}

attention×LR 相对 LR-only 为正的轴数是 **{n_positive_delta}/3**。因此这三个例子不能支持“attention 在 LR expression geography 之外增加了空间一致性”。{exact_message_note}

### 空间图 D：发送端和接收端细胞也一致吗？

![Spatial sender and receiver consistency](figures/09_spatial_sender_receiver_consistency.png)

midpoint 会把 sender 与 receiver 混在一起。这张图分别把每个细胞的 outgoing 和 incoming 分数求和，并只在双方至少一方 active 的细胞并集中算 Spearman，避免几千个共同零值把相关性人为抬高。

| LR 轴 | outgoing active-cell rho | outgoing mass OVL | incoming active-cell rho | incoming mass OVL |
|---|---:|---:|---:|---:|
{direction_rows}

active-cell 排名相关接近零或正负混合，所以不能由 midpoint 的大区域重合推出“相同 sender/receiver 被排高”或“方向一致”。

当前坐标输入只提供 aligned x/y、stage 和 cell type，没有可验证的胚胎前后/背腹方向 landmark；因此这些图回答“两个方法是否占据相似相对区域”，不应擅自把某个热点命名为具体解剖方位。

### 空间部分一句话结论

**raw hotspot 图显示部分共同组织区域；严格 fixed-support null、LR-only control 和 sender/receiver audit 均不支持把它升级为 attention 特异的独立空间验证。** 因此空间图应作为透明的诊断与限制；审稿回复的主要正向证据仍是完整 cell-type-pair 排名与 COMMOT / external-only consensus 的一致性。
"""


def report_text(
    summary: pd.DataFrame,
    checklist: pd.DataFrame,
    direct: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    consensus: pd.DataFrame,
    top: pd.DataFrame,
    pathway: pd.DataFrame,
    spatial_audit: dict[str, Any] | None = None,
) -> str:
    stage_rows = "\n".join(
        (
            f"| {row.stage_label} | {row.spearman:.3f} | "
            f"{int(row.intersection)}/{int(row.target_set_size_after_boundary_ties)} | "
            f"{row.overlap_enrichment_over_random:.2f}× | {row.bh_q_within_target_reference_family:.3g} |"
        )
        for row in summary.itertuples(index=False)
    )
    shared_10 = checklist.loc[
        checklist["stage"].eq(1.0) & checklist["category"].eq("shared_top20"), "pair"
    ].tolist()
    shared_24 = checklist.loc[
        checklist["stage"].eq(4.0) & checklist["category"].eq("shared_top20"), "pair"
    ].tolist()
    bullets_10 = "\n".join(f"- {item}" for item in shared_10)
    bullets_24 = "\n".join(f"- {item}" for item in shared_24)
    spatial_section = (
        spatial_audit_report_section(spatial_audit)
        if spatial_audit is not None
        else """空间正式 null/sensitivity 结果未提供。本报告只能把旧箭头与最近邻覆盖图作为描述，不能据此声称 WNT/NOTCH 空间一致或独立验证。"""
    )

    attention_external = float(
        direct["attention_vs_external_consensus_rho"].mean()
    )
    attention_commot = float(direct["attention_vs_commot_rho"].mean())

    def pairwise_mean(left: str, right_contains: str) -> tuple[float, int]:
        selected = pairwise_summary.loc[
            pairwise_summary["display_label_left"].eq(left)
            & pairwise_summary["display_label_right"].str.contains(
                right_contains, regex=False, na=False
            )
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one pairwise summary for {left!r} vs {right_contains!r}"
            )
        row = selected.iloc[0]
        return (
            float(row["mean_stage_spearman"]),
            int(row["n_finite_spearman_stages"]),
        )

    attention_cag_project, _ = pairwise_mean(
        "CytoBridge attention", "CellAgentChat | project LR"
    )
    attention_cellchat, attention_cellchat_n = pairwise_mean(
        "CytoBridge attention", "CellChat | project LR"
    )
    self_included = consensus.loc[
        consensus["design"].eq("article_style_all_method_native_primary")
        & consensus["target"].eq("CytoBridge attention"),
        "spearman",
    ]
    if self_included.empty:
        raise ValueError("Missing article-style self-included attention consensus")
    self_included_mean = float(self_included.mean())
    attention_external_top = top.loc[
        top["target"].eq("CytoBridge attention")
        & top["reference"].eq("External native consensus")
    ]
    if attention_external_top.empty:
        raise ValueError("Missing attention versus external-native top-signal rows")
    attention_external_top_enrichment = float(
        attention_external_top["overlap_enrichment_over_random"].mean()
    )

    def pathway_fold(name: str) -> float:
        selected = pathway.loc[pathway["pathway"].eq(name)]
        if len(selected) != 1:
            raise ValueError(f"Expected one pathway enrichment row for {name!r}")
        return float(selected["fold_enrichment"].iloc[0])

    cxcl_fold = pathway_fold("CXCL")
    notch_fold = pathway_fold("NOTCH")
    ncwnt_fold = pathway_fold("ncWNT")
    reviewer_paragraph = reviewer_reply_text(
        direct,
        top,
        spatial_audit,
    ).split("\n\n", 1)[1].strip()
    if spatial_audit is not None:
        final_spatial_summary = (
            "坐标级热点审计没有超过 null，所以我们主动把它保留为限制"
        )
    else:
        final_spatial_summary = (
            "当前未提供 formal fixed-support null，所以空间覆盖图只作为描述"
        )
    return f"""# 斑马鱼 CCC 结果：一份讲人话的读图说明

## 先看结论：这些结果能不能回复审稿人？

**能，但必须把论点说准。** 现有结果支持的是：

> CytoBridge attention / exact message 在完整 cell-type-pair 排名上与 COMMOT 及 external-only 多方法共识呈稳定正相关；已知 LR/通路提供补充生物学解释。坐标级空间图则是透明的诊断，而不是额外独立验证。

现有结果**不支持**把 attention 直接写成：

- 生化意义上的“通信强度”或“通信概率”；
- 精确的 ligand→receptor 方向；
- 实验扰动意义上的因果效应。

这一区分非常重要。审稿人的核心担忧不是“attention 是否完美复刻每一个 CCC 软件”，而是 attention 会不会只反映距离、共同表达或模型拟合。当前结果已经提供了多层正向证据，足以回应这个担忧；但不能越界宣称 attention 就是真实信号通量。

建议先看 [图 1：审稿问题证据地图](figures/01_reviewer_evidence_map.png)，30 秒就能知道哪些结论成立、哪些只成立一部分、哪些还没证明。

![Reviewer evidence map](figures/01_reviewer_evidence_map.png)

## 审稿人究竟问了什么？

审稿人担心：高 attention 可能只是因为细胞靠得近、转录状态相似或者模型恰好这样拟合，不一定是真正的 signaling。因此她希望看到至少一种外部或生物学证据：已知 ligand–receptor、空间局部信号，或 perturbation-sensitive communication program。

我们现在给出三层正向证据和一层空间审计：

1. **总体排序一致：** 不把 CytoBridge 放进共识，三个外部方法仍与 attention 正相关。
2. **具体高位信号一致：** top 20% 的 cell-type arrows 超过随机预期地重叠。
3. **生物学内容一致：** 高位信号富集 CXCL、NOTCH、ncWNT 等已知通路，并与 NicheNet 的下游 ligand 排名重合。
4. **空间审计：** 三条预选 LR 轴在同一坐标上有部分 raw hotspot overlap，但还要经过 fixed-support null、LR-only control 和 sender/receiver audit，不能只靠肉眼认定一致。

## 四个词先讲明白

### 一个“点”或一条“箭头”是什么？

它代表一个有方向的细胞类型对，例如 `Notochord → Spinal Cord Ventral Region`。方向相反的箭头是另一个对象。`A→A` 表示同一 cell type 中不同细胞之间的 homotypic type pair；底层的单细胞 `i→i` self-edge 已删除，不能把 `A→A` 叫成 literal self-loop。

### CytoBridge attention 是什么？

它是模型在图边上学习到的权重摘要。数值大表示这类 sender→receiver interaction 对模型更重要。它本身不是某个特定 LR 分子的生化通量。

### External-only consensus 是什么？

先在每个发育时点内，把 COMMOT、CellAgentChat CTPS 和 CellChat triMean 各自的 cell-type arrow 分数转成 0–1 排名，再取三个排名的平均。**这里完全不包含 CytoBridge**，所以不会因为把自己的结果放进共识而人为提高相关性。

### rank correlation 怎么理解？

它只问“双方把同一批箭头排出的先后顺序像不像”：1 表示顺序完全相同，0 表示没有稳定关系，负数表示相反。它不要求两个软件的原始分数单位相同。

## 图 2：最直观的总体一致性

![External-only rank scatter](figures/04_external_consensus_rank_scatter.png)

### 这是什么？

每个小图对应一个时点；每个点是一条 sender→receiver 细胞类型箭头。横轴是三个外部方法的共识排名，纵轴是 CytoBridge attention 排名。

### 怎么画的？

所有分数只在同一时点内转成百分位排名。右上角紫色区域表示双方都把该箭头放进 top 20%；紫点是实际落在双方 top 20% 交集中的箭头。没有把五个时点的原始分数混在一起，也没有平均不同软件的原始单位。

### 怎么看？

- 点云越呈左下到右上的趋势，双方整体排序越一致。
- 紫点越多，双方对“最强的一批箭头”越有具体共识。
- 相关系数是总体排序指标；紫点交集是 top-signal 指标。两者回答不同问题，不能互相替代。

### 结果是什么？

| 时点 | 全部箭头排序相关 rho | 双方 top 20% 交集 | 相对随机预期 | 多重校正 q |
|---|---:|---:|---:|---:|
{stage_rows}

五个时点相关性全部为正，平均为 **{attention_external:.3f}**。10 hpf 和 24 hpf 的 top overlap 最清楚；早期 5.25 hpf 的 top overlap 较弱。因此正确说法是“总体稳定正相关、具体 top signal 的一致性具有阶段差异”，而不是“每个时点都完全一致”。

## 图 3：不要只看数字，具体哪些箭头一致？

![Concrete arrow checklist](figures/05_top_communication_arrows_checklist.png)

### 这是什么？

它把抽象的 overlap 数字还原成具体 sender→receiver 箭头。绿色 `AGREE` 是双方都排进 top 20% 的例子；橙色和蓝色行故意保留“不一致例子”，防止把结果画成只有成功案例。

### 怎么画的？

用与正式统计完全相同的 top-20% 规则：包含 homotypic type pair，并保留落在第 k 名边界上的全部并列项。每行末尾直接写双方 0–1 排名。

### 怎么解读？

你不需要先理解 Jaccard。直接看一条生物学箭头是否同时被双方排高，以及双方排名差多大即可。

10 hpf 的代表性共同高位箭头包括：

{bullets_10}

24 hpf 的代表性共同高位箭头包括：

{bullets_24}

## 图 4：同一空间坐标上的结果到底一致到什么程度？

{spatial_section}

## 图 5：已知 LR 信号随时间是否被双方同时认为重要？

![Known LR temporal bubble](figures/known_lr_temporal_consistency_bubble.png)

### 这是什么？

每一行是一个已知 LR 轴，每一列是发育时点；圆点和方块分别代表 CytoBridge 与 COMMOT。点越大/越深，表示它在该方法、该时点内的相对排名越高。黑色外框表示双方都进入较高分位。

### 怎么画的？

先按预先登记的已知 zebrafish 轴筛选可识别 LR，然后在每个方法内部转成排名。原始 score 不直接相减，因为 COMMOT 和 attention 的单位没有可比性。

### 结论是什么？

它回答的是“已知信号轴是否在相似发育阶段被双方同时列为高位”，不是“两个方法给出的绝对强度相等”。WNT、NOTCH 等轴提供了可讲清楚的正向生物学例子。

## 图 6：高 attention 中是什么生物学通路？

![Top signal biology](figures/top_signal_biology.png)

### 左图怎么读？

取每个时点 **LR-compatible attention score** 排名前 20 的 LR 轴，用完整 project LR database 作为超几何检验背景。fold enrichment = 1 表示与该背景期望一样；大于 1 表示某通路在高位信号中出现得更多。显著富集包括：CXCL {cxcl_fold:.2f}×、NOTCH {notch_fold:.2f}×、ncWNT {ncwnt_fold:.2f}×，均通过多重检验。由于训练图本身使用了 LR-informed edge prior，这一结果只能作为补充生物学解释，不能算完全独立验证。

### 右图怎么读？

NicheNet 不直接预测空间 cell–cell communication，它问“哪些 ligand 更能解释 receiver 的下游靶基因”。因此这里比较的是**下游生物学一致性**：被 NicheNet 排高的 ligand 有多少也出现在 attention 的高位 LR 轴中。五个可测单元的重合比例为 50%–100%。

### 结论是什么？

高 attention 不是一堆无法解释的边，它集中在已知 signaling programs；NicheNet 从不同目标函数给出下游侧支持。但这不是对空间通信强度的一对一复刻，也不是完全独立于 LR 先验的验证。

## 原来的折线图、barplot 和矩阵到底是什么意思？

下面按文件逐张说明，并标注它在论文叙事中的正确位置。

### `positive_consistency_overview`

- **是什么：** 四块统计摘要：各阶段 external-only correlation、把自身放进共识导致的升高、top-20% overlap enrichment，以及与距离的关系。
- **怎么看：** A 是最重要的；B 告诉读者 self-included consensus 会把均值从 {attention_external:.3f} 抬到 {self_included_mean:.3f}，因此正式结论必须使用 external-only；C 表示 top signals 平均为随机预期的 {attention_external_top_enrichment:.2f} 倍；D 表示 attention 原始分数并不是简单“越近越高”。
- **结论：** 有用但过于压缩，适合作为补充统计摘要，不适合作为读者第一张图。

### `rank_concordance`

- **是什么：** 所有方法两两之间的平均 stage-wise Spearman 矩阵。
- **怎么看：** 红色接近 1 表示两种方法把 cell-type arrows 排得相似，0 表示没稳定关系，负值表示相反。对角线永远是自己和自己，信息量为零。
- **结论：** attention 与 COMMOT 的直接平均相关为 {attention_commot:.3f}；与 CellAgentChat project-LR 为 {attention_cag_project:.3f}；CellChat 只有 {attention_cellchat_n} 个 stage 有有限分数，均值 {attention_cellchat:.3f}。它说明不同 CCC 方法并不等价，不能要求所有格子都红。
- **建议：** 补充材料或方法审计；正文改用图 2 的点云。

### `top_edge_overlap`

- **是什么：** 每种方法 top-k sender→receiver 集合之间的 Jaccard，即“交集/并集”。
- **怎么看：** 0.1 不是“10 条里相同 1 条”，而是交集占两个集合并集的 10%。零值很多也可能来自阈值、结构零和方法目标不同。
- **结论：** 可显示严格 top set 的差异，但很抽象，也容易被误读。不要单独用它回答审稿人，改看图 3 的具体箭头。

### `condition_coverage`

- **是什么：** 每种方法、每个时点实际有多少 directed type pairs 可比较。
- **结论：** 这是输入覆盖质量检查，不是方法性能。NicheNet 的空格来自它只分析特定 source→target stage/receiver unit，不代表算法失败。

### `directionality_concordance`

- **是什么：** 比较一种方法中 A→B 相对 B→A 的排名差，是否被另一方法重复。
- **结论：** 结果稀疏且混合，不足以证明精确方向。它应该作为限制或审计图，不宜主打。

### `stage_stability`

- **是什么：** 同一种方法在相邻发育时点的排序相关和 top-k 重合。
- **结论：** 它描述发育过程中网络是否变化，不是跨方法一致性。早期变化大、后期较稳定完全可能是合理生物学，不应简单理解成“越高越好”。

### `cytobridge_control_panel`

- **是什么：** 在控制 stage、cell type、距离、细胞状态和图度数后测试 LR association，并与初始化/随机化模型比较。
- **结论：** trained attention 的残差 LR association 为正且随机化后消失，是部分正证据；但初始化也有非零信号，exact-message 某些 control 甚至不更好，所以它不能作为唯一或最强证据。

### `reviewer_validation_axes`

- **是什么：** 把 LR 匹配、方向、空间和 virtual removal 放在一张内部验证图里。
- **结论：** 适合作为完整审计。virtual removal 只能说明模型对删除某些相互作用敏感，不能写成真实 perturbation causality。

### `ccc_circle_comparison`

- **是什么：** 10 和 24 hpf 的高位 cell-type communication 网络。节点是 cell type，箭头是高位 sender→receiver。
- **怎么看：** 用于看网络结构和双方共同出现的箭头，不适合逐条精确对数。
- **注意：** 为了画面可读，这张图去掉 homotypic type pair (`A→A`)，并只显示有限条边；因此它的边数不会与正式 top-20% 统计完全相同。正式数字以图 2/图 3 和 CSV 为准。

## 为什么 CellChat、CellAgentChat、NicheNet 的相关性不都很高？

- **COMMOT** 是最接近的外部对照：它也是空间 CCC 方法，并使用当前项目 LR 数据库，所以直接排序相关最高。
- **CellChat** 主要是非空间的群体表达统计；当前 native triMean 还在多个 stage 产生大量并列零，仅两个 stage 有可计算相关，因此均值低不奇怪。
- **CellAgentChat** 使用不同的统计和数据库/orthology 路径，project-LR 条件比 official-default 更接近 CytoBridge，但相关仍只有中低水平。
- **NicheNet** 的目标是 ligand→target gene regulation，而不是当前空间 sender→receiver strength。用它的 raw score 与 attention 做直接排序相关并不公平；它更适合图 6 右侧的 downstream ligand overlap。

因此最有说服力的论证不是“CytoBridge 与所有软件都高度相关”，而是：“与目标最接近的空间方法 COMMOT 在完整 type-pair 排名上稳定正相关；多个方法组成的 external-only consensus 在五个时点均为正；已知 LR/pathway 提供补充生物学解释。”坐标级热点审计没有给出超出共同 support geography 的额外正证据，应该如实报告。

## 哪些证据有一定循环性？

CytoBridge 的 LR edge prior 使用了 LR 信息，所以“高 attention 中出现更多已知 LR”不是完全独立的盲验证。它仍能说明训练后模型把权重组织到了可解释信号上，但不能单独证明真实通信。为降低循环性，正式回复应把重点放在：

1. 不含 CytoBridge 的 external-only consensus；
2. 与 COMMOT 的直接空间比较；
3. NicheNet 下游 ligand consistency；
4. 将坐标级空间结果明确写为未超过 null 的诊断；
5. 对方向性与因果性的明确限制。

## 建议给审稿人的英文回复

> {reviewer_paragraph}

## 写稿时可以说 / 不要说

可以说：

- `communication-relevant interaction organization`
- `biologically coherent interaction-associated weights`
- `consistent with external spatial CCC rankings and known signaling programs`
- `partially shared raw tissue regions in descriptive spatial maps`

不要说：

- `attention is the true communication strength`
- `attention is a calibrated CCC probability`
- `the analysis proves the ligand-to-receptor direction`
- `virtual ablation proves a causal perturbation response`
- `all external methods agree strongly with CytoBridge`
- `LR hotspot maps independently validate attention`

## 最后给汇报者的一句话

**最强的故事不是“所有方法给出一模一样的网络”，而是“在完全排除自身的外部共识中，CytoBridge 在五个时点都呈正向排序一致，并能落到具体 cell-type arrows 与已知 WNT/NOTCH/CXCL 通路；{final_spatial_summary}，而不把 attention 等同于真实生化通信强度。”**
"""


def from_zero_report_text(
    direct: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    spatial: pd.DataFrame,
    proximity: pd.DataFrame,
    spatial_audit: dict[str, Any] | None = None,
) -> str:
    direct_rows = "\n".join(
        (
            f"| {row.stage_label} | {row.attention_vs_commot_rho:.3f} | "
            f"{row.exact_message_vs_commot_rho:.3f} | "
            f"{row.attention_vs_external_consensus_rho:.3f} | "
            f"{row.exact_message_vs_external_consensus_rho:.3f} |"
        )
        for row in direct.itertuples(index=False)
    )
    top_rows = "\n".join(
        (
            f"| {row.stage_label} | {int(row.attention_commot_top10_intersection)}/10 | "
            f"{int(row.exact_message_commot_top10_intersection)}/10 |"
        )
        for row in direct.itertuples(index=False)
    )

    def pair_mean(left: str, contains: str) -> tuple[float, int]:
        selected = pairwise_summary.loc[
            pairwise_summary["display_label_left"].eq(left)
            & pairwise_summary["display_label_right"].str.contains(
                contains, regex=False, na=False
            )
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one {left} vs {contains} summary row")
        row = selected.iloc[0]
        return float(row["mean_stage_spearman"]), int(row["n_finite_spearman_stages"])

    attention_commot, _ = pair_mean("CytoBridge attention", "COMMOT | project LR")
    message_commot, _ = pair_mean("CytoBridge exact message", "COMMOT | project LR")
    attention_cellchat, attention_cellchat_n = pair_mean(
        "CytoBridge attention", "CellChat | project LR"
    )
    message_cellchat, message_cellchat_n = pair_mean(
        "CytoBridge exact message", "CellChat | project LR"
    )
    attention_cag_project, _ = pair_mean(
        "CytoBridge attention", "CellAgentChat | project LR"
    )
    message_cag_project, _ = pair_mean(
        "CytoBridge exact message", "CellAgentChat | project LR"
    )
    attention_cag_official, _ = pair_mean(
        "CytoBridge attention", "CellAgentChat | official mouse DB"
    )
    message_cag_official, _ = pair_mean(
        "CytoBridge exact message", "CellAgentChat | official mouse DB"
    )
    attention_nichenet_project, attention_nichenet_n = pair_mean(
        "CytoBridge attention", "NicheNet-v2 | project LR gate"
    )
    message_nichenet_project, message_nichenet_n = pair_mean(
        "CytoBridge exact message", "NicheNet-v2 | project LR gate"
    )

    spatial_rows = "\n".join(
        (
            f"| {row.ligand.upper()}→{row.receptor.upper()} ({STAGE_LABELS[float(row.stage)]}) | "
            f"{row.cytobridge_midpoints_near_commot_fraction:.1%} | "
            f"{row.commot_midpoints_near_cytobridge_fraction:.1%} |"
        )
        for row in spatial.itertuples(index=False)
    )
    prox = proximity.loc[
        proximity["method"].isin(
            ["CytoBridge attention", "CytoBridge exact message"]
        )
    ].pivot(index="stage", columns="method", values="spearman_score_vs_inverse_mean_spatial_distance")
    proximity_rows = "\n".join(
        (
            f"| {STAGE_LABELS[float(stage)]} | "
            f"{row['CytoBridge attention']:.3f} | "
            f"{row['CytoBridge exact message']:.3f} |"
        )
        for stage, row in prox.sort_index().iterrows()
    )
    if spatial_audit is not None:
        spatial_section = spatial_audit_report_section(spatial_audit)
        spatial_inventory_role = (
            "raw hotspot 描述 + fixed-support null / LR-only / direction 审计；"
            "未形成额外独立正向证据"
        )
        spatial_conclusion = (
            "三条预选 LR 轴的 raw hotspot OVL 为部分重合，但主设置均低于自适应"
            " fixed-support null；attention×LR 相对 LR-only 的增量为 0/3 个正值，"
            "active-cell sender/receiver 排名也接近零或混合。因此空间图是诊断与限制，"
            "不能作为 attention 特异的独立空间验证。"
        )
        spatial_reply_point = (
            "透明报告空间审计：raw 组织区域部分重合，但未超过 fixed-support null，"
            "所以不把它包装成额外正向证据；"
        )
        spatial_short = (
            "同坐标 hotspot 虽有部分 raw overlap，但三条轴都未超过 fixed-support null，"
            "attention×LR 也没有优于 LR-only；因此空间图保留为诚实的诊断，而不是主证据。"
        )
        spatial_reviewer_claim = (
            "LR-specific hotspot maps are reported as descriptive diagnostics "
            "because they did not exceed the audited fixed-support null."
        )
    else:
        spatial_section = f"""### 旧覆盖图只能作描述

![Spatial coverage](figures/03_spatial_location_coverage.png)

| LR 轴 | CytoBridge 高位位置附近有 COMMOT | COMMOT 高位位置附近有 CytoBridge |
|---|---:|---:|
{spatial_rows}

这是 many-to-one 最近邻覆盖：同一个密集 COMMOT 点可以命中多个 CytoBridge 点，而且没有随机位置基线、阈值敏感性或 LR-only control。它只能说明 raw coverage，不能据此声称 WNT/NOTCH 空间一致很强。若要作空间推断，请用 `spatial_coordinate_consistency.py` 生成正式 fixed-support null。"""
        spatial_inventory_role = "旧 many-to-one coverage；描述性审计，无 null-model inference"
        spatial_conclusion = (
            "旧空间覆盖率受 COMMOT 点密度和 many-to-one 匹配影响；在没有正式 null 时，"
            "不能把 WNT/NOTCH 的高覆盖写成强空间一致。"
        )
        spatial_reply_point = "不把旧空间覆盖率作为正式证据；"
        spatial_short = "旧空间覆盖图只有描述性意义，不能主张独立空间验证。"
        spatial_reviewer_claim = (
            "LR-specific hotspot maps are reported as descriptive diagnostics "
            "because a formal fixed-support null was not supplied for this report."
        )

    return f"""# 从零开始讲懂斑马鱼 CytoBridge–CCC 分析

## 0. 先给最终答案

之前之所以越看越乱，是因为我们把四类完全不同的东西都叫成了“interaction”或“CCC”：模型建图、attention、真实进入动力学的 message，以及后处理得到的 LR-specific score。

最关键的纠正是：

> **CytoBridge 并不存在一个唯一的、原生的“CCC 分数”。**

CytoBridge 原生给出的是一个空间图动力学模型。我们可以从模型中读出两种 interaction 量：

1. **attention：** 这条边在消息传递中被门控得多强；
2. **exact message：** 这条边实际对模型输出贡献了多大的向量。

然后才有两类后续分析：

- 把这些细胞边按 sender cell type → receiver cell type 汇总，与 COMMOT、CellChat、CellAgentChat 做**直接 CCC 一致性比较**；
- 再乘上 sender ligand 和 receiver receptor 表达，得到 **LR-compatible score**，用于画 WNT/NOTCH/CXCL 空间例子和通路富集。

所以，真正最直接的 CCC 对比是下面的 [CytoBridge–COMMOT 直接对比](figures/02_direct_ccc_comparison.png)，不是那张空间箭头图。

### 先统一五个最常用的词

- **sender（发送细胞）：** 一条有方向的边 `i→j` 中的细胞 i；
- **receiver（接收细胞）：** 同一条边中的细胞 j；
- **LR：** ligand–receptor，即发送细胞中的配体和接收细胞中的受体；
- **CCC：** cell–cell communication，细胞间通讯；
- **hpf：** hours post fertilization，受精后的小时数，例如 24 hpf。

## 1. 整个计算流程只有这一张图

![Computation map](figures/00_computation_to_result_map.png)

图中绿色上半支是**直接 CCC 对比**；橙色中间分支是**LR 生物学解释**；灰色下半支是**控制分析**。三类结果回答不同问题，不能混在一起解释。

## 2. 用一个具体例子理解所有 interaction 名词

假设空间图中有一条细胞边：

```text
细胞 i → 细胞 j
```

模型按以下顺序处理：

### 第一步：这两个细胞之间有没有候选边？

它们必须在预处理给出的空间 cutoff 内，并通过模型的 link-predictor threshold。这个步骤只决定“允许不允许传消息”，不是最终 CCC 分数。

### 第二步：attention `|αᵢⱼ|`

这是一个 signed、non-softmax gate 的绝对值跨 head 平均。通俗说：模型给这条边开了多大的“阀门”。

原始每个 head 的 gate 可以有正负号，但本文报告的 `|αᵢⱼ|` 先取绝对值再跨 head 平均，**正负号已经被丢弃**。因此它只能表示 gate magnitude，不能据此判断激活/抑制方向。它也不包含具体 ligand/receptor 名称，更不是 0–1 通信概率。

### 第三步：exact message `‖mᵢⱼ‖`

完整 message `mᵢⱼ` 是一个向量，不只是 attention；它还包括 source state、target state、距离特征、value/message transformation 和质量归一化。我们把每条边的完整向量贡献从训练好的 interaction layer 中确定性地拆出来，并验证所有边加回去能够重建模型 interaction output。这里的 **exact 只表示代数重建精确**，不表示它是真实生物学作用或实验真值。

通俗说：

- attention 是“阀门开多大”；
- exact message 是“最终通过这条管道，对动力学输出造成了多大贡献”。

因此 edge-level `‖mᵢⱼ‖` 比 attention 多包含了模型实际消息传递中的 source/target state、距离和 transformation，但它仍然只是模型贡献，不是生化 flux，也不能预先称为“更接近生物学真值”。模型原生的是 interaction output；把它拆成单边向量、取 norm、再按 cell type 汇总，是对该输出的确定性读出。本数据中它与 COMMOT 的相关确实更高，这是后面的实证结果，而不是由定义保证的。

### 第四步：如果要看 `Wnt5b→Fzd7a`

attention 本身不知道 Wnt5b 或 Fzd7a。我们在模型输出以后计算：

```text
LR activity(i→j)
= scaled Wnt5b expression in sender i
× scaled Fzd7a expression in receiver j

LR-compatible attention
= |attention(i→j)| × LR activity(i→j)
```

这就是原图写的 `attention×LR`。更准确的名字是 **LR-compatible attention score**。

它高，表示 sender ligand、receiver receptor 和模型 attention 三项都不能太低；乘积中某一项更高也可以补偿另一项处于中等水平。它是 post-hoc composite，不是模型原生输出的“Wnt5b→Fzd7a attention”。

最简单的数字例子：某条边的 attention magnitude 为 0.8，sender 的 Wnt5b scaled expression 为 0.5，receiver 的 Fzd7a scaled expression 为 0.25，那么这条边的 LR-compatible attention 是 `0.8×0.5×0.25=0.10`。只要 ligand 或 receptor 任一为 0，这条特定 LR 的分数就是 0，即使模型 attention 很高。

这里的 `scaled expression` 也有固定定义：单基因直接取表达；多亚基 complex 取所有亚基中的最小值；随后除以全数据正值的 95% 分位数并截断到 0–1。这样做只用于让 LR activity 的量纲稳定，不会把它变成通信概率。

## 3. 现在到底有几个 CytoBridge interaction 量？

| 名称 | 层级 | 是不是模型原生 | 是否 LR-specific | 用在哪里 |
|---|---|---:|---:|---|
| candidate edge | 单细胞边 | 是 | 否 | 决定哪些邻近细胞能传消息 |
| attention `|αᵢⱼ|` | 单细胞边 | 是 | 否 | reviewer 关心的模型 gate |
| exact message `‖mᵢⱼ‖` | 单细胞边 | 原生 output 的确定性拆解 | 否 | 进入 interaction output 的边贡献大小 |
| `G_AB attention` | cell-type pair | 由 attention 汇总 | 否 | 与外部 CCC 做直接比较 |
| `D_AB exact message` | cell-type pair | 由 message 汇总 | 否 | 与外部 CCC 做直接比较 |
| LR-compatible attention/message | LR-specific cell edge | 否，后处理 | 是 | 空间 LR 图、LR 轴和通路分析 |
| virtual-removal response | 整条模拟轨迹 | 否 | 否 | 模型敏感性，不是 interaction 分数 |

完整、可机器读取的定义见 [interaction_term_dictionary.csv](tables/interaction_term_dictionary.csv)。

## 4. “我们推断的 CCC”和其他方法到底怎么直接比较？

### 4.1 先把所有方法放到同一个比较对象上

外部 CCC 方法的原始单位完全不同：

- COMMOT：cell-level OT communication mass；
- CellChat：群体层面的 LR communication probability；
- CellAgentChat：显著 interaction score 的和；
- NicheNet：ligand 解释 receiver target genes 的能力。

因此不能直接比较原始数值。我们把真正可以对齐的方法都整理成同一个对象：

```text
同一时点、同一条有方向的 cell-type pair
A sender type → B receiver type
```

CytoBridge 给这条箭头两个分数：

- `G_AB attention`：A→B 所有模型边的平均 attention；
- `D_AB exact message`：对每个 B receiver，先把所有 A sender 提供的 message **向量相加**，再对合向量取 norm，最后对全部 B cells（包括没有 A 输入、贡献为零的 receiver）平均。

因此 `D_AB` 不是简单平均所有单边 `‖mᵢⱼ‖`：不同 edge message 方向相反时，会先发生向量抵消。这个定义测的是“来自 A 的完整合成贡献”，不是边数或总流量。

比较使用每个时点的完整有向 type-pair 方阵：5 个时点依次有 7²、7²、11²、14² 和 19² 条箭头，共 776 条。`A→A` 保留，表示 **homotypic type pair**，不能把它叫作 literal self-loop；CytoBridge 底层 graph 的 cell-level `i→i` self-edge 已经排除，外部工具则保留各自的 native score definition。

Direct comparison 的 CytoBridge 原始 type-pair 分数先在 5 个 technical grouping seeds 上逐项平均；这些 seed 只是为了降低分组计算的技术波动，不是 5 次独立训练。COMMOT 则把 A→B block 的 cell-level OT mass 除以 `n_sender × n_receiver`，避免细胞数量多的 type pair 仅因组合数多而得分更大。

最后，各方法都在**同一时点内部**对 49/121/196/361 条箭头排序，再计算 Spearman correlation；并列值使用普通 average rank。这里比较的是“谁把同一批 cell-type arrows 排得相似”，不是比较原始单位。文中的五时点均值是五个 stage correlation 的等权算术平均，不是 replicate 置信区间。

### 4.2 这张才是最直接的 CCC comparison

![Direct CCC comparison](figures/02_direct_ccc_comparison.png)

每一个点就是同一条 sender type→receiver type 箭头：

- 横轴：COMMOT 对它的排名；
- 上排纵轴：CytoBridge attention 排名；
- 下排纵轴：CytoBridge exact-message 排名；
- 越呈左下→右上，整体排序越一致；
- 右上角表示双方都把它排得较高。

### 4.3 直接比较的实际数字

| 时点 | attention vs COMMOT | exact message vs COMMOT | attention vs external-only consensus | exact message vs external-only consensus |
|---|---:|---:|---:|---:|
{direct_rows}

五时点平均：

- attention vs COMMOT：**{attention_commot:.3f}**；
- exact message vs COMMOT：**{message_commot:.3f}**；
- attention vs external-only consensus：**{direct['attention_vs_external_consensus_rho'].mean():.3f}**；
- exact message vs external-only consensus：**{direct['exact_message_vs_external_consensus_rho'].mean():.3f}**。

这给出一个很清楚的结果：

> **CytoBridge 与最接近的空间 CCC 方法 COMMOT 在五个时点均呈正相关；在本数据和当前 type-pair 汇总定义下，exact-message 的 rank correlation 高于 attention。**

这应该是汇报和回复审稿人的第一组结果。空间 LR 箭头图只能放在后面作为具体例子。

`external-only consensus` 的计算是：先在每个时点分别把 COMMOT、CellAgentChat CTPS 和 CellChat triMean 转成百分位排名，再把三者**等权平均**；其中不包含 CytoBridge，避免 self-inclusion。它并不是实验 gold standard。外部方法仍与 CytoBridge 共用表达矩阵、cell-type annotation，并且部分条件共用 project LR database。

### 4.4 “总体排序一致”不等于“前十名完全相同”

严格 top-10 的直接重合是：

| 时点 | attention 与 COMMOT 共同 top 10 | exact message 与 COMMOT 共同 top 10 |
|---|---:|---:|
{top_rows}

例如 24 hpf 的 attention–COMMOT 全矩阵相关达到 0.831，但严格 top 10 交集为 0。这不矛盾：两种方法可以把几百条箭头的总体强弱顺序排得相似，却在最顶端十条的精确名次上不同。

因此结果支持的是 **global interaction organization consistency**，不支持“CytoBridge 精确复刻 COMMOT 的 strongest individual edges”。

这里的 direct top-10 来自限制正分并处理边界并列的正式 pairwise 表。不要使用通用 top-20% 表中 CellChat triMean 或 CellAgentChat CTPS 的某些 per-method overlap 行：当大量零分在阈值处并列时，整个矩阵都可能被选入 top set，产生误导性的高 overlap。External-consensus top-20% 是稠密共识排名，不受这个零尾并列问题影响。

## 5. 其他方法为什么有的低、有的负？

| 对照 | attention 平均 rho | exact-message 平均 rho | 应该怎么解释 |
|---|---:|---:|---|
| COMMOT + project LR | {attention_commot:.3f} | {message_commot:.3f} | 最直接、最合理的空间 CCC 对照；五个 stage 都可比 |
| CellChat + project LR | {attention_cellchat:.3f}（{attention_cellchat_n} stage） | {message_cellchat:.3f}（{message_cellchat_n} stage） | 非空间、群体表达统计，而且 triMean 在多个 stage 大量并列零 |
| CellAgentChat + project LR | {attention_cag_project:.3f} | {message_cag_project:.3f} | 中低度正相关；跨物种 orthology sensitivity |
| CellAgentChat official DB | {attention_cag_official:.3f} | {message_cag_official:.3f} | 仍为正，但数据库更加不同；只作 sensitivity |
| NicheNet + project gate | {attention_nichenet_project:.3f}（{attention_nichenet_n} unit） | {message_nichenet_project:.3f}（{message_nichenet_n} unit） | 负值，但它比较的是 ligand→target regulation，不是空间 CCC strength |

NicheNet 的负相关不能解释成“CytoBridge 与空间 CCC 相反”。NicheNet 的任务和比较单位不同，所以 raw rank correlation 本来就不应作为主要验证。正确用法是看 NicheNet top ligand 与我们的 LR-compatible top ligand 是否重合；现有五个可评估 receiver units 的 overlap 为 50%–100%。

## 6. 你提到的“spatial 那个负数”到底是什么？

那不是 CytoBridge 和某个 spatial CCC 方法的相关性。它计算的是：

```text
cell-type interaction score
vs
inverse mean spatial distance
```

横向理解：inverse distance 越大，表示该 type pair 的模型边平均更近。如果 rho 为正，表示平均更近的 type pair 往往分数越高；如果接近 0，表示在已经允许连接的局部邻居中，type-pair 分数不再由平均距离单调决定；如果略为负，表示该时点中平均距离较大、但仍在 cutoff 内的 type pair 反而略高。它是 cell-type-pair 层面的聚合关系，不能改写成“每条较长的单细胞边都更强”。

| 时点 | attention vs inverse distance | exact message vs inverse distance |
|---|---:|---:|
{proximity_rows}

attention 五时点平均为 **{prox['CytoBridge attention'].mean():.3f}**，基本就是零。24 hpf 的 −0.192 不能说成“负的 CCC consistency”；它只说明在预先限制为局部邻居以后，高 attention 并不是简单等于更短距离。

这个分析是**距离混杂审计**，不是验证结果，也不应放在主结果图中。由于图本身仍由空间 cutoff 构建，我们也不能进一步声称 attention 完全不依赖空间。

## 7. 同一空间坐标上的结果到底一致到什么程度？

{spatial_section}

这里还有一个容易混淆的 seed 差异：图 02 的 direct type-pair comparison 对 5 个 technical grouping seeds 的 type-pair 分数逐项平均；图 06–09 对同一 cell edge 在它实际出现的 grouping seeds 中取均值，未出现的 seed 不补零；旧 `reviewer_validation_axes` 的固定 edge-level confounder/LR audit 则主要使用 seed 101。它们来自同一 checkpoint，但不是独立训练重复，也不是同一张分数表；seed 只降低/审计 grouping 计算的技术波动。

## 8. 现在一共有哪些结果？每个结果回答什么问题？

| 结果 | 用的 CytoBridge 量 | 回答的问题 | 在叙事中的位置 |
|---|---|---|---|
| Direct COMMOT consistency | G_AB attention、D_AB exact message | 全部 cell-type arrows 的总体排序像不像 | **最主要直接证据** |
| External-only consensus | G_AB、D_AB | 不依赖单个工具时是否仍一致 | **主要稳健性证据** |
| Top-signal overlap | top-20% G_AB/D_AB | 最强一批箭头是否超过随机重合 | 支持证据，且有 stage 差异 |
| LR spatial examples | LR-compatible attention | WNT/NOTCH/CXCL 的 raw hotspot 与 null/control 结果 | {spatial_inventory_role} |
| Pathway enrichment | top LR-compatible attention axes | 高位信号是否富集已知通路 | 生物学可解释性；超几何检验以完整 project LR database 为背景，且受 LR prior 影响 |
| NicheNet overlap | top LR-compatible ligands | receiver 下游 ligand 是否一致 | 不同目标函数的补充支持 |
| Proximity diagnostic | G_AB/D_AB vs inverse distance | 是否只是“越近越高” | 混杂审计，不是主验证 |
| Conditional LR audit | residual attention/message | 控制距离、state、degree 后是否仍有 LR association | 内部一致性；方向未证明 |
| Virtual removal | 模拟轨迹差异 | 模型是否对某类细胞的存在敏感 | 敏感性，不是因果 |

机器可读版本见 [result_inventory.csv](tables/result_inventory.csv)。

## 9. 最终生物学结论应该怎么说？

把全部结果合起来，最稳妥的结论是：

1. CytoBridge 的 cell-type interaction organization 与外部 CCC 方法存在正向一致性；最强直接对照是 COMMOT。这里的“稳定”只指五个发育时点的相关均为正，不代表独立训练重复或置信区间验证。
2. 在本数据和当前 type-pair 汇总定义下，exact-message 与 COMMOT 的 rank correlation 高于 attention；这是本次实证结果，不等于证明 exact-message 普遍更优或更接近生物学真值。
3. 高位 LR-compatible signals 富集 CXCL、NOTCH、ncWNT 等已知 pathway。
4. {spatial_conclusion}
5. attention 与距离的平均相关接近零，说明它不是一个简单的距离排序；但空间 graph 本身仍限制在局部邻居。
6. 当前结果没有证明精确 ligand→receptor 方向、真实生化通信强度或实验因果性。

另外，训练配置中的 `alpha_spatial` 和 `alpha_express` 是 OT/loss 权重，不是这里任何 attention、message 或 CCC score 的组成部分。

## 10. 这能不能回复审稿人？

**可以，但回复应以直接 type-pair comparison 为主，空间坐标结果作为透明审计。** 推荐论证顺序：

1. 先承认 attention 不等于真实 CCC strength；
2. 给出 attention/exact message 与 COMMOT 的五时点直接相关；
3. 给出 external-only consensus，排除 self-inclusion；
4. {spatial_reply_point}
5. 给出 pathway/NicheNet 生物学支持；其中 pathway enrichment 使用完整 project LR database 作超几何背景，但因为训练图已使用 LR-informed edge prior，只能作为补充而非独立验证；
6. 明确方向性和因果性的边界。

不要把论点写成“CytoBridge 与所有 CCC 方法都高度一致”。准确说法是：

> The learned interaction organization shows consistent positive concordance with the closest spatial CCC reference, COMMOT, and with an external-only multi-method consensus. {spatial_reviewer_claim}

## 11. 如果只汇报五分钟，可以这样讲

> CytoBridge 本身不是一个传统 LR 打分软件，而是一个空间图动力学模型。它在每条细胞边上给出 attention gate，并且我们可以精确拆出这条边对动力学输出的完整 message。为了和 CCC 软件公平比较，我们把两种模型量按同一时点的 sender cell type→receiver cell type 汇总，再只比较排名。最直接的结果是：attention 与 COMMOT 五时点平均 rho 为 {attention_commot:.3f}，exact message 为 {message_commot:.3f}，五个时点均为正；排除 CytoBridge 本身的外部共识结果也分别为 {direct['attention_vs_external_consensus_rho'].mean():.3f} 和 {direct['exact_message_vs_external_consensus_rho'].mean():.3f}。这说明总体 interaction organization 有稳定一致性，但严格 top-10 并不完全相同。{spatial_short} 综合起来，正向结论来自 type-pair rank concordance 与已知 pathway/downstream consistency；attention 仍不能解释为真实通信概率、精确 LR 方向或因果信号。
"""


def reviewer_reply_text(
    direct: pd.DataFrame,
    top: pd.DataFrame,
    spatial_audit: dict[str, Any] | None = None,
) -> str:
    headline = headline_metrics(direct, top)
    attention_commot = headline["attention_commot"]
    message_commot = headline["exact_message_commot"]
    attention_external = headline["attention_external"]
    message_external = headline["exact_message_external"]
    top_enrichment = headline["top_enrichment"]

    if spatial_audit is not None:
        primary = spatial_audit["primary"]
        null = spatial_audit["null"]
        null_primary = null.loc[null["metric"].eq("field_overlap_ovl")].merge(
            primary[["example_id", "top_fraction", "scale_factor"]],
            on=["example_id", "top_fraction", "scale_factor"],
            validate="one_to_one",
        )
        merged = primary.merge(
            null_primary[["example_id", "null_mean", "n_permutations"]],
            on="example_id",
            validate="one_to_one",
        )
        axis_numbers = ", ".join(
            f"{row.ligand.upper()}–{row.receptor.upper()} {row.field_overlap_ovl:.3f} "
            f"(null mean {row.null_mean:.3f})"
            for row in merged.itertuples(index=False)
        )
        spatial_sentence = (
            " We additionally mapped three preselected LR examples on the same aligned "
            f"spatial coordinates: {axis_numbers}. None exceeded the audited adaptive "
            f"fixed-support score-permutation null ({int(merged['n_permutations'].min())} "
            "permutations), attention×LR did not improve spatial overlap over LR activity "
            "alone in any example, and sender/receiver rank agreement among active cells "
            "was low or mixed. We therefore treat these spatial maps as descriptive "
            "diagnostics rather than independent validation."
        )
    else:
        spatial_sentence = (
            " LR-specific spatial maps are treated as descriptive only because a formal "
            "fixed-support null was not supplied to this report."
        )
    return f"""# Reviewer-response wording (plain and bounded)

We agree that attention weights should not be interpreted as direct biochemical communication strengths. We therefore revised the manuscript to describe them as interaction-associated gates and added external and biological consistency analyses. We first aligned the complete directed sender-cell-type→receiver-cell-type matrices at each developmental stage and compared within-stage ranks. Against the closest spatial CCC reference, COMMOT, CytoBridge attention was positively concordant at all five stages (mean stage-wise Spearman rho = {attention_commot:.3f}); the exactly reconstructed complete message contribution showed stronger concordance (mean rho = {message_commot:.3f}). An external-only consensus built by equally averaging within-stage percentile ranks from COMMOT, CellAgentChat CTPS, and CellChat triMean, without CytoBridge, was also positively concordant with attention (mean rho = {attention_external:.3f}) and exact message (mean rho = {message_external:.3f}). Top-20% attention interactions overlapped this external consensus above random expectation on average ({top_enrichment:.2f}-fold), although strict top-edge identities varied by stage.{spatial_sentence} Pathway enrichment and NicheNet ligand overlap provided complementary, but not fully independent, biological support. These results support communication-relevant organization in the learned interaction structure, while we do not interpret attention as a calibrated CCC probability, activating/inhibitory sign, exact ligand-to-receptor direction, or causal perturbation evidence.
"""


def main() -> None:
    args = parser().parse_args()
    bundle = args.bundle_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not (bundle / "bundle_manifest.json").is_file():
        raise FileNotFoundError(f"Not a reviewer bundle: {bundle}")
    spatial_audit = (
        load_spatial_audit(args.spatial_consistency_dir)
        if args.spatial_consistency_dir is not None
        else None
    )
    if spatial_audit is not None:
        spatial_bundle = (
            spatial_audit["manifest"].get("inputs", {}).get("bundle_manifest", {})
        )
        if spatial_bundle.get("sha256") != sha256(bundle / "bundle_manifest.json"):
            raise ValueError(
                "Spatial audit was not generated from the supplied reviewer bundle manifest"
            )
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {output}")
        shutil.rmtree(output)
    figures = output / "figures"
    tables = output / "tables"
    originals = figures / "original"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)
    originals.mkdir(parents=True)

    scores = read_table(
        bundle,
        "harmonized_type_pair_scores.csv.gz",
        [
            "stage",
            "sender_type",
            "receiver_type",
            "cytobridge_attention",
            "cytobridge_attention_rank",
            "cytobridge_exact_message",
            "cytobridge_exact_message_rank",
            "commot",
            "commot_rank",
            "external_native_consensus",
        ],
    )
    consensus = read_table(
        bundle,
        "consensus_by_stage.csv",
        ["design", "target", "stage", "stage_label", "spearman"],
    )
    top = read_table(
        bundle,
        "top_signal_overlap_by_stage.csv",
        [
            "target",
            "reference",
            "stage",
            "intersection",
            "target_set_size_after_boundary_ties",
            "reference_set_size_after_boundary_ties",
            "overlap_enrichment_over_random",
            "bh_q_within_target_reference_family",
        ],
    )
    pathway = read_table(
        bundle,
        "pathway_enrichment.csv",
        ["pathway", "fold_enrichment", "bh_q"],
    )
    pairwise = read_table(
        bundle,
        "pairwise_consistency_by_stage.csv",
        [
            "display_label_left",
            "display_label_right",
            "stage",
            "spearman_rank_concordance",
            "top_k_intersection",
            "effective_top_k",
        ],
    )
    pairwise_summary = read_table(
        bundle,
        "pairwise_consistency_summary.csv",
        [
            "display_label_left",
            "display_label_right",
            "n_finite_spearman_stages",
            "mean_stage_spearman",
        ],
    )
    spatial = read_table(
        bundle,
        "spatial_display_audit.csv",
        [
            "stage",
            "ligand",
            "receptor",
            "n_cytobridge_top_fraction_midpoints",
            "n_commot_top_fraction_midpoints",
            "cytobridge_midpoints_near_commot_fraction",
            "commot_midpoints_near_cytobridge_fraction",
        ],
    )
    proximity = read_table(
        bundle,
        "spatial_proximity_by_stage.csv",
        [
            "method",
            "stage",
            "spearman_score_vs_inverse_mean_spatial_distance",
        ],
    )
    stage_table = prepare_stage_table(scores, top)
    summary = stage_summary(stage_table, consensus, top)
    checklist = make_pair_checklist(stage_table)
    direct = direct_comparison_table(pairwise, consensus, scores)
    headline = headline_metrics(direct, top)

    summary.to_csv(tables / "plain_language_stage_summary.csv", index=False)
    checklist.to_csv(tables / "top_pair_checklist.csv", index=False)
    build_figure_index(
        headline=headline,
        spatial_audit=spatial_audit,
    ).to_csv(
        tables / "figure_reading_index.csv", index=False
    )
    interaction_term_dictionary().to_csv(
        tables / "interaction_term_dictionary.csv", index=False
    )
    result_inventory(spatial_audit_available=spatial_audit is not None).to_csv(
        tables / "result_inventory.csv", index=False
    )
    direct.to_csv(tables / "direct_ccc_consistency_by_stage.csv", index=False)

    plot_computation_map(figures / "00_computation_to_result_map")
    plot_evidence_map(
        figures / "01_reviewer_evidence_map",
        headline=headline,
        spatial_audit=spatial_audit,
    )
    plot_direct_ccc_scatter(scores, direct, figures / "02_direct_ccc_comparison")
    plot_spatial_coverage(spatial, figures / "03_spatial_location_coverage")
    plot_rank_scatter(stage_table, summary, figures / "04_external_consensus_rank_scatter")
    plot_pair_checklist(checklist, summary, figures / "05_top_communication_arrows_checklist")

    for name in ORIGINAL_FIGURES:
        for suffix in ("png", "pdf"):
            source = bundle / "figures" / f"{name}.{suffix}"
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, originals / source.name)
            if name in {
                "spatial_lr_interaction_maps",
                "known_lr_temporal_consistency_bubble",
                "top_signal_biology",
                "ccc_circle_comparison",
                "positive_consistency_overview",
            }:
                shutil.copy2(source, figures / source.name)

    if spatial_audit is not None:
        spatial_root: Path = spatial_audit["root"]
        for source_name, destination_name in SPATIAL_AUDIT_FIGURES.items():
            for suffix in ("png", "pdf"):
                shutil.copy2(
                    spatial_root / f"{source_name}.{suffix}",
                    figures / f"{destination_name}.{suffix}",
                )
        for name in SPATIAL_AUDIT_TABLES:
            shutil.copy2(spatial_root / name, tables / name)
        shutil.copy2(
            spatial_root / "README_CN.md",
            output / "SPATIAL_COORDINATE_CONSISTENCY_CN.md",
        )
        shutil.copy2(
            spatial_root / "manifest.json",
            output / "SPATIAL_AUDIT_MANIFEST.json",
        )

    guide = output / "START_HERE_CN.md"
    guide.write_text(
        from_zero_report_text(
            direct, pairwise_summary, spatial, proximity, spatial_audit
        ),
        encoding="utf-8",
    )
    (output / "DETAILED_FIGURE_GUIDE_CN.md").write_text(
        report_text(
            summary,
            checklist,
            direct,
            pairwise_summary,
            consensus,
            top,
            pathway,
            spatial_audit,
        ),
        encoding="utf-8",
    )
    (output / "REVIEWER_RESPONSE_PLAIN_EN.md").write_text(
        reviewer_reply_text(direct, top, spatial_audit), encoding="utf-8"
    )

    artifacts = [path for path in output.rglob("*") if path.is_file()]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bundle": str(bundle),
        "source_bundle_manifest_sha256": sha256(bundle / "bundle_manifest.json"),
        "source_spatial_audit": (
            str(spatial_audit["root"]) if spatial_audit is not None else None
        ),
        "source_spatial_audit_manifest_sha256": (
            spatial_audit["manifest_sha256"] if spatial_audit is not None else None
        ),
        "notes": [
            "The frozen source reviewer bundle was not modified.",
            "SUPPORTED/PARTIAL/NOT SHOWN are narrative evidence labels, not statistical scores.",
            "Top-20% sets reproduce the source tie-inclusive formal analysis exactly.",
            (
                "The coordinate-level spatial audit was manifest/hash verified and is "
                "treated as a diagnostic rather than independent positive validation."
                if spatial_audit is not None
                else "No formal coordinate-level spatial audit was supplied."
            ),
        ],
        "artifacts": [record(path, output) for path in sorted(artifacts)],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "n_artifacts": len(artifacts) + 1}, indent=2))


if __name__ == "__main__":
    main()
