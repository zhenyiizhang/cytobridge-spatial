#!/usr/bin/env python3
"""Render a clean, biology-first response to the ARISTA Figure 5c review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "output/arista_package_native_spatialqc_z50_retrain_20260824_r1"
ANALYSIS_ROOT = RUN_ROOT / "figure5c_two_niche_timecourse_v1"
LR_AXES_ROOT = RUN_ROOT / "figure5c_two_niche_lr_axes_v1"
STYLE_PATH = (
    Path.home()
    / ".codex/skills/cytobridge-scientific-figures/assets/cytobridge-paper.mplstyle"
)

TEXT = "#111111"
GRID = "#D9E8F0"
REFERENCE = "#214761"
NULL = "#7FA9C4"
N1 = "#CC6677"
N2 = "#07838B"

NICHE_ORDER = ("N1_sfrpEGC_VLMC", "N2_reaEGC_wntEGC")
NICHE_SHORT = {
    "N1_sfrpEGC_VLMC": "sfrpEGC–VLMC",
    "N2_reaEGC_wntEGC": "reaEGC–wntEGC",
}
NICHE_ROLE = {
    "N1_sfrpEGC_VLMC": "Matrix/trophic program",
    "N2_reaEGC_wntEGC": "Reactive/adhesion program",
}
NICHE_COLOR = {"N1_sfrpEGC_VLMC": N1, "N2_reaEGC_wntEGC": N2}

DISPLAY_PROGRAMS = {
    "N1_sfrpEGC_VLMC": ("AGRN", "LAMININ", "TENASCIN", "FGF", "THBS"),
    "N2_reaEGC_wntEGC": ("GRN", "L1CAM", "NRXN", "SEMA3", "FN1"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_style() -> None:
    if STYLE_PATH.is_file():
        plt.style.use(STYLE_PATH)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _heading(fig: plt.Figure, label: str, title: str, x: float, y: float) -> None:
    fig.text(x, y, label, fontsize=14, fontweight="bold", ha="left", va="center", color=TEXT)
    fig.text(x + 0.037, y, title, fontsize=12, fontweight="bold", ha="left", va="center", color=TEXT)


def _clean_axis(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _outline(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    if points.shape[0] < 3:
        return
    hull = ConvexHull(points)
    polygon = np.vstack((points[hull.vertices], points[hull.vertices[0]]))
    ax.plot(polygon[:, 0], polygon[:, 1], color="white", linewidth=3.1, zorder=3)
    ax.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=1.8, zorder=4)


def _plot_spatial_map(ax: plt.Axes, cells: pd.DataFrame) -> None:
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    scatter = ax.scatter(
        cells["paper_x"],
        cells["paper_y"],
        c=cells["cosine_full_vs_interaction"],
        cmap="plasma",
        norm=norm,
        s=10,
        linewidths=0,
        alpha=0.96,
        zorder=1,
    )
    for index, niche in enumerate(NICHE_ORDER, start=1):
        subset = cells.loc[cells["two_niche_region"].fillna("").eq(niche)]
        points = subset[["paper_x", "paper_y"]].to_numpy(float)
        _outline(ax, points, NICHE_COLOR[niche])
        centroid = np.median(points, axis=0)
        ax.text(
            centroid[0],
            centroid[1],
            f"N{index}",
            fontsize=9,
            fontweight="bold",
            color=NICHE_COLOR[niche],
            ha="center",
            va="center",
            zorder=5,
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    cax = inset_axes(ax, width="54%", height="3.6%", loc="lower left", borderpad=0.7)
    colorbar = plt.colorbar(scatter, cax=cax, orientation="horizontal")
    colorbar.set_ticks([-1, 0, 1])
    colorbar.ax.tick_params(labelsize=8, length=2, pad=1)
    colorbar.set_label("Full–interaction spatial-velocity cosine", fontsize=8, labelpad=1)


def _plot_attention_null(ax: plt.Axes, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_positions = np.array([1.0, 0.0])
    for y, niche in zip(y_positions, NICHE_ORDER):
        row = summary.loc[summary["module"].eq(niche)].iloc[0]
        observed = float(row["observed_attention_per_cell"])
        null_mean = float(row["null_attention_per_cell_mean"])
        null_sd = float(row["null_attention_per_cell_sd"])
        fold = observed / null_mean
        rows.append(
            {
                "niche": niche,
                "observed_attention_per_cell": observed,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "fold_over_null": fold,
                "empirical_p": float(row["attention_per_cell_empirical_p_greater"]),
            }
        )
        ax.errorbar(
            null_mean,
            y,
            xerr=null_sd,
            fmt="o",
            markersize=5.5,
            markerfacecolor=NULL,
            markeredgecolor=REFERENCE,
            markeredgewidth=0.6,
            ecolor=NULL,
            elinewidth=1.2,
            capsize=3,
            zorder=2,
        )
        ax.scatter(
            observed,
            y,
            marker="D",
            s=42,
            color=NICHE_COLOR[niche],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(observed + 0.10, y, f"{fold:.2f}×", fontsize=8.5, va="center", color=NICHE_COLOR[niche])
    ax.set_yticks(y_positions, ["N1", "N2"])
    ax.set_xlim(0.55, 4.15)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("Selected attention per domain cell")
    ax.set_title("Above composition-matched null", loc="left", pad=7)
    ax.tick_params(axis="y", length=0)
    _clean_axis(ax, grid_axis="x")
    ax.scatter([], [], marker="D", s=35, color=REFERENCE, label="Observed")
    ax.errorbar([], [], xerr=[], fmt="o", color=NULL, label="Matched null mean ± s.d.")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        ["Observed domain", "Random same-composition cells"],
        loc="lower right",
        frameon=False,
        fontsize=8,
        handletextpad=0.5,
    )
    return pd.DataFrame(rows)


def _edge_summary(edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for niche in NICHE_ORDER:
        table = edges.loc[edges["niche"].eq(niche)].copy()
        total = float(table["attention_sum"].sum())
        if niche.startswith("N1"):
            definitions = (
                ("sfrpEGC→sfrpEGC", (table["sender"].eq("sfrpEGC") & table["receiver"].eq("sfrpEGC"))),
                (
                    "VLMC ↔ sfrpEGC",
                    (
                        (table["sender"].eq("VLMC") & table["receiver"].eq("sfrpEGC"))
                        | (table["sender"].eq("sfrpEGC") & table["receiver"].eq("VLMC"))
                    ),
                ),
            )
        else:
            definitions = (
                ("wntEGC→wntEGC", (table["sender"].eq("wntEGC") & table["receiver"].eq("wntEGC"))),
                ("reaEGC→reaEGC", (table["sender"].eq("reaEGC") & table["receiver"].eq("reaEGC"))),
                (
                    "reaEGC ↔ wntEGC",
                    (
                        (table["sender"].eq("reaEGC") & table["receiver"].eq("wntEGC"))
                        | (table["sender"].eq("wntEGC") & table["receiver"].eq("reaEGC"))
                    ),
                ),
            )
        used = np.zeros(len(table), dtype=bool)
        for label, mask in definitions:
            mask_array = mask.to_numpy(bool)
            used |= mask_array
            attention = float(table.loc[mask_array, "attention_sum"].sum())
            rows.append(
                {
                    "niche": niche,
                    "edge_class": label,
                    "attention_sum": attention,
                    "attention_percent": 100.0 * attention / total,
                }
            )
        if niche.startswith("N1"):
            attention = float(table.loc[~used, "attention_sum"].sum())
            rows.append(
                {
                    "niche": niche,
                    "edge_class": "Other selected edges",
                    "attention_sum": attention,
                    "attention_percent": 100.0 * attention / total,
                }
            )
    return pd.DataFrame(rows)


def _plot_edge_summary(ax: plt.Axes, edges: pd.DataFrame) -> pd.DataFrame:
    summary = _edge_summary(edges)
    ordered = pd.concat(
        [
            summary.loc[summary["niche"].eq(NICHE_ORDER[0])],
            summary.loc[summary["niche"].eq(NICHE_ORDER[1])],
        ],
        ignore_index=True,
    )
    y = np.arange(len(ordered))[::-1]
    colors = [NICHE_COLOR[niche] for niche in ordered["niche"]]
    ax.barh(y, ordered["attention_percent"], height=0.58, color=colors, alpha=0.88)
    for yi, value in zip(y, ordered["attention_percent"].to_numpy(float)):
        ax.text(value + 1.2, yi, f"{value:.0f}%", va="center", fontsize=8)
    labels = ordered["edge_class"].tolist()
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 68)
    ax.set_xlabel("Share of selected attention (%)")
    ax.set_title("Dominant sender–receiver structure", loc="left", pad=7)
    ax.tick_params(axis="y", length=0, labelsize=8)
    _clean_axis(ax, grid_axis="x")
    ax.axhline(2.5, color=GRID, linewidth=0.8)
    ax.text(0.98, 0.97, "N1", transform=ax.transAxes, ha="right", va="top", color=N1, fontweight="bold")
    ax.text(0.98, 0.40, "N2", transform=ax.transAxes, ha="right", va="top", color=N2, fontweight="bold")
    return ordered


def _plot_pathways(ax: plt.Axes, matched: pd.DataFrame, niche: str) -> pd.DataFrame:
    programs = DISPLAY_PROGRAMS[niche]
    table = matched.loc[matched["module"].eq(niche) & matched["pathway"].isin(programs)].copy()
    table["pathway"] = pd.Categorical(table["pathway"], categories=programs, ordered=True)
    table = table.sort_values("pathway", kind="mergesort")
    if len(table) != len(programs) or not table["adjusted_p_value"].lt(0.05).all():
        raise AssertionError(f"Displayed pathway set is incomplete or not FDR-significant: {niche}")
    table["log2_fold_over_null"] = np.log2(table["fold_over_null_mean"].to_numpy(float))
    y = np.arange(len(table))[::-1]
    color = NICHE_COLOR[niche]
    for yi, value in zip(y, table["log2_fold_over_null"].to_numpy(float)):
        ax.plot([0, value], [yi, yi], color=color, linewidth=1.6, alpha=0.78)
        ax.scatter(value, yi, s=38, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0, color=REFERENCE, linewidth=0.8)
    ax.set_yticks(y, table["pathway"].astype(str))
    ax.set_xlim(-0.08, 2.75)
    ax.set_xlabel("log2 enrichment over same-composition null")
    ax.set_title(f"{NICHE_SHORT[niche]}  ·  {NICHE_ROLE[niche]}", loc="left", color=color, pad=7)
    ax.tick_params(axis="y", length=0)
    _clean_axis(ax, grid_axis="x")
    return table


def _plot_lr_axis_panel(ax: plt.Axes, candidate_axes: pd.DataFrame, niche: str) -> pd.DataFrame:
    programs = DISPLAY_PROGRAMS[niche]
    selected: list[pd.DataFrame] = []
    for pathway in programs:
        candidates = candidate_axes.loc[
            candidate_axes["niche"].eq(niche)
            & candidate_axes["pathway"].eq(pathway)
            & candidate_axes["adjusted_p_value"].lt(0.05)
            & candidate_axes["observed_pair_score"].gt(0)
        ].copy()
        candidates = candidates.sort_values(
            ["adjusted_p_value", "observed_pair_score", "fold_over_null_mean"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        if candidates.empty:
            raise AssertionError(f"No pair-level FDR-significant axis for {niche}: {pathway}")
        selected.append(candidates.iloc[[0]])
    table = pd.concat(selected, ignore_index=True)
    table["pathway"] = pd.Categorical(table["pathway"], categories=programs, ordered=True)
    table = table.sort_values("pathway", kind="mergesort")
    if len(table) != len(programs) or not table["adjusted_p_value"].lt(0.05).all():
        raise AssertionError(f"Panel d pair axes are incomplete or fail pair-level FDR: {niche}")
    y = np.arange(len(table))[::-1]
    values = table["log2_fold_over_null"].to_numpy(float)
    color = NICHE_COLOR[niche]
    ax.barh(y, values, height=0.55, color=color, alpha=0.88)
    labels = []
    for row in table.itertuples(index=False):
        receptor = str(row.receptor).replace("_", "/")
        labels.append(
            f"{row.ligand}–{receptor}\n{row.dominant_sender}→{row.dominant_receiver}"
        )
    for yi, value, fold in zip(y, values, table["fold_over_null_mean"].to_numpy(float)):
        ax.text(value + 0.08, yi, f"{fold:.1f}×", va="center", fontsize=8, color=color)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 5.65)
    ax.set_xlabel("log2 pair enrichment over same-composition null")
    ax.set_title(f"{NICHE_SHORT[niche]}-associated domain", loc="left", color=color, pad=7)
    ax.tick_params(axis="y", length=0, labelsize=8)
    _clean_axis(ax, grid_axis="x")
    return table


def _plot_lr_axes(
    axes: tuple[plt.Axes, plt.Axes], candidate_axes: pd.DataFrame
) -> pd.DataFrame:
    displayed = [
        _plot_lr_axis_panel(ax, candidate_axes, niche)
        for ax, niche in zip(axes, NICHE_ORDER)
    ]
    return pd.concat(displayed, ignore_index=True)


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        inputs = {
            "cells": ANALYSIS_ROOT / "tables/roi_two_niche_assignments.csv",
            "edges": ANALYSIS_ROOT / "tables/two_niche_t1_celltype_edges.csv",
            "attention_null": ANALYSIS_ROOT / "tables/two_niche_attention_matched_null.csv",
            "pathway_null": ANALYSIS_ROOT / "tables/two_niche_lr_pathway_matched_null.csv",
            "pair_axes": LR_AXES_ROOT / "two_niche_lr_pair_matched_null.csv.gz",
            "pair_axes_manifest": LR_AXES_ROOT / "manifest.json",
            "analysis_manifest": ANALYSIS_ROOT / "manifest.json",
        }
        for path in inputs.values():
            if not path.is_file():
                raise FileNotFoundError(path)

        cells = pd.read_csv(inputs["cells"])
        edges = pd.read_csv(inputs["edges"])
        attention_null = pd.read_csv(inputs["attention_null"])
        pathway_null = pd.read_csv(inputs["pathway_null"])
        pair_axes = pd.read_csv(inputs["pair_axes"])

        _apply_style()
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        _heading(fig, "a", "Spatial interaction domains", 0.065, 0.956)
        _heading(fig, "b", "Organized cell-state interactions", 0.625, 0.956)
        _heading(fig, "c", "Domain-specific candidate repair programs", 0.065, 0.535)
        _heading(fig, "d", "Candidate ligand–receptor axes", 0.065, 0.276)

        ax_a = fig.add_axes([0.070, 0.595, 0.500, 0.315])
        ax_b1 = fig.add_axes([0.655, 0.765, 0.305, 0.125])
        ax_b2 = fig.add_axes([0.655, 0.560, 0.305, 0.135])
        ax_c1 = fig.add_axes([0.105, 0.330, 0.365, 0.165])
        ax_c2 = fig.add_axes([0.585, 0.330, 0.365, 0.165])
        # Pair labels carry both molecular and cell-state direction; reserve a
        # wider left gutter in each column so the mechanism remains legible.
        ax_d1 = fig.add_axes([0.165, 0.080, 0.300, 0.155])
        ax_d2 = fig.add_axes([0.665, 0.080, 0.285, 0.155])

        _plot_spatial_map(ax_a, cells)
        displayed_attention = _plot_attention_null(ax_b1, attention_null)
        displayed_edges = _plot_edge_summary(ax_b2, edges)
        displayed_pathways = pd.concat(
            [
                _plot_pathways(ax_c1, pathway_null, NICHE_ORDER[0]),
                _plot_pathways(ax_c2, pathway_null, NICHE_ORDER[1]),
            ],
            ignore_index=True,
        )
        displayed_pair_axes = _plot_lr_axes((ax_d1, ax_d2), pair_axes)

        stem = "FigureS_ARISTA_Figure5c_local_interaction_niches_clean"
        fig.savefig(stage / f"{stem}.pdf")
        fig.savefig(stage / f"{stem}.svg")
        fig.savefig(stage / f"{stem}.png", dpi=320)
        plt.close(fig)

        tables = stage / "tables"
        tables.mkdir()
        displayed_attention.to_csv(tables / "panel_b_attention_matched_null.csv", index=False)
        displayed_edges.to_csv(tables / "panel_b_selected_edge_structure.csv", index=False)
        displayed_pathways.to_csv(tables / "panel_c_lr_pathway_enrichment.csv", index=False)
        displayed_pair_axes.to_csv(tables / "panel_d_candidate_lr_axes.csv", index=False)

        caption = (
            "**Supplementary Figure. Local interaction domains underlying the heterogeneous spatial-velocity pattern.** "
            "(a) Full-versus-interaction spatial-velocity cosine in the frozen 5-DPI Figure 5c region. "
            "The upper-quartile cells were connected on the trained physical-radius graph, and components containing at least 20 cells defined two spatial domains; selected model edges were evaluated afterward. "
            "(b) Selected attention per domain cell compared with 9,999 random cell sets drawn from the same ROI while preserving the exact number of every cell type in each domain. This null estimates the attention expected from cell-type composition alone. Diamonds show the observed domains and circles show null means plus or minus s.d. "
            "The sfrpEGC–VLMC- and reaEGC–wntEGC-associated domains showed 1.69-fold and 3.06-fold enrichment, respectively (both empirical P = 0.0001). Dominant sender–receiver structures summarize the selected attention within each domain. "
            "(c) Representative ligand–receptor pathway scores at 5 DPI relative to 1,999 cell-type-composition-matched permutations. All displayed pathways passed Benjamini–Hochberg correction at q < 0.05. "
            "The sfrpEGC–VLMC-associated domain was enriched for matrix and trophic programs, whereas the reaEGC–wntEGC-associated domain was enriched for reactive, adhesion, and guidance programs. "
            "(d) Candidate cell-state-resolved ligand–receptor axes calculated inside the exact spatial components. Within each pathway displayed in (c), the pair with the minimum pair-level BH q value was selected, followed by the largest observed pair score and fold enrichment. "
            "All displayed pairs passed BH correction across 531 tested pairs within their domain. Bars show enrichment over 1,999 random ROI cell sets preserving the exact domain cell-type composition. These model-supported axes generate mechanistic hypotheses and do not establish causal signaling."
        )
        (stage / "caption.md").write_text(caption + "\n", encoding="utf-8")

        response = (
            "# Figure 5c reviewer response\n\n"
            "The added analysis addresses the reviewer in four linked steps. Panel a pulls out two spatially connected cell populations using a fixed cosine, physical-neighbor, and component-size rule. Panel b tests organization against cell-type-composition-matched random sets and identifies the dominant sender–receiver structures. Panel c uses CytoBridge ligand–receptor scoring to identify distinct candidate repair programs. Panel d resolves those programs into statistically supported sender-to-receiver ligand–receptor axes inside the exact spatial components.\n\n"
            "The result supports a specific conclusion: the heterogeneous Figure 5c pattern contains two localized, organized ependymoglial interaction domains with distinct candidate molecular programs. A pixel-level wound boundary was not available, so the figure does not claim direct geometric alignment with the wound edge or causal signaling.\n"
        )
        (stage / "reviewer_response_summary.md").write_text(response, encoding="utf-8")

        script_snapshot = stage / Path(__file__).name
        shutil.copy2(Path(__file__), script_snapshot)
        final_pdf = output_dir / f"{stem}.pdf"
        final_png = output_dir / f"{stem}.png"
        provenance = f"""# Figure provenance

Archived on: `2026-08-24`

Manuscript figure: `ARISTA Figure 5c reviewer-response supplement`

Scientific claim: `The heterogeneous Figure 5c field contains two localized, model-edge-supported ependymoglial interaction domains with distinct candidate pathway and ligand–receptor axes.`

## Files

- Vector figure: `{final_pdf}`
- PNG preview: `{final_png}`
- Plotting script: `{output_dir / script_snapshot.name}`
- Caption source: `{output_dir / 'caption.md'}`
- Compiled manuscript or SI: `not yet integrated`

## Source paths

- Figure analysis: `{ANALYSIS_ROOT}`
- Accepted package-native run: `{RUN_ROOT}`
- Server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/arista-package-native-spatialqc-z50-20260824-r1`

## Selected experiment

- Local run: `{RUN_ROOT}`
- Server run: `/data/cytobridge/projects/CytoBridge-ST-1104/runs/arista-package-native-spatialqc-z50-20260824-r1`
- Configuration: `recorded by the selected run and analysis manifest`
- Manifest: `{inputs['analysis_manifest']}`
- Checkpoint and SHA-256: `inherited from the selected package-native run; exact downstream input hashes are recorded in the analysis manifest`
- Training stages and epoch counts: `not changed by this post hoc figure analysis`

## Panel sources

| Panel | Content | Source files | Calculation |
|---|---|---|---|
| a | Spatial interaction niches | `{inputs['cells']}` | Frozen ROI cosine map plus fixed physical connected-component rule |
| b | Organized cell-state interactions | `{inputs['attention_null']}`, `{inputs['edges']}` | Observed selected attention versus 9,999 cell-type-matched null regions; selected-edge attention fractions |
| c | Niche-specific repair programs | `{inputs['pathway_null']}` | Package-native LR pathway scores versus 1,999 cell-type-matched permutations with BH correction |
| d | Candidate ligand–receptor axes | `{inputs['pair_axes']}` | Exact-component pair-level LR scores versus 1,999 cell-type-composition-matched permutations with BH correction across 531 pairs; one pair per panel-c program selected by minimum q, then score, then fold |

## Evaluation protocol

- Initial cells or particles: `frozen 1,454-cell Figure 5c ROI at 5 DPI`
- Evaluation weights: `package-native selected attention`
- Growth handling: `not applicable`
- Time step and diffusion scale: `5-DPI observed component; no new simulation`
- Seeds: `permutation seeds recorded in the pathway and pair-axis analysis manifests`
- Uncertainty summary: `matched-null mean plus or minus s.d.; empirical permutation P values in the caption; the composition-matched null does not preserve spatial geometry`

## Rebuild command

```bash
python {Path(__file__).resolve()} --output-dir {output_dir}
```

## Interpretation

`The figure supports spatially organized interaction domains with distinct candidate LR programs and cell-state-resolved axes. The cross-species LR database and model-derived attention make these mechanistic hypotheses rather than causal signaling evidence. No pixel-level wound boundary is available.`

## SHA-256

- Figure PDF: `{_sha256(stage / f'{stem}.pdf')}`
- Figure PNG: `{_sha256(stage / f'{stem}.png')}`
- Plotting script: `{_sha256(script_snapshot)}`
"""
        (stage / "PROVENANCE.md").write_text(provenance, encoding="utf-8")
        manifest = {
            "schema": "cytobridge.arista.figure5c-local-interaction-niches.v2",
            "claim": "The heterogeneous Figure 5c field contains two localized, model-edge-supported ependymoglial interaction domains with distinct candidate pathway and ligand-receptor axes.",
            "style": {
                "font": "Arial",
                "palette": {"N1": N1, "N2": N2, "reference": REFERENCE, "text": TEXT},
                "cosine_colormap": "plasma, retained from accepted Figure 5c",
                "heatmap": False,
            },
            "inputs": {key: {"path": str(path), "sha256": _sha256(path)} for key, path in inputs.items()},
            "limitations": [
                "No pixel-level wound-boundary annotation was available.",
                "Attention is model influence rather than a measured biophysical rate.",
                "Human-symbol-mapped axolotl features are scored against a human CellChat LR database.",
                "Composition-matched null sets do not preserve spatial geometry.",
                "The analysis supports organized interaction hypotheses, not causal signaling.",
            ],
            "outputs": {},
        }
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                manifest["outputs"][str(path.relative_to(stage))] = {
                    "sha256": _sha256(path),
                    "size_bytes": int(path.stat().st_size),
                }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        stage.rename(output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
