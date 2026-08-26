#!/usr/bin/env python3
"""Render corrected MOSTA S9/S10 with the submitted GO/wave visual grammar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


# Submitted MOSTA GO panels use the paper's five-stop teal-to-coral
# p.adjust grammar.  Define it directly from the submitted-panel colors
# instead of substituting a Matplotlib stock colormap.
GO_PALETTE = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]
GO_CMAP = LinearSegmentedColormap.from_list(
    "mosta_submitted_go_p_adjust",
    GO_PALETTE,
)
HEAT_CMAP = "viridis"
HEAT_LIMIT = 1.8
PHASE_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
GRID_COLOR = "#e5e5e5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def wrap_term(value: object, width: int) -> str:
    return textwrap.fill(str(value), width=width, break_long_words=False, break_on_hyphens=False)


def scientific_or_decimal(value: float, _position: int) -> str:
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.0e}"
    return f"{value:.3f}"


def vector_colorbar(
    fig: mpl.figure.Figure,
    box: list[float],
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    title: str,
    label: str | None = None,
    title_size: float = 8.0,
    tick_size: float = 7.0,
) -> mpl.axes.Axes:
    cax = fig.add_axes(box)
    values = np.linspace(vmin, vmax, 256)
    grid_x, grid_y = np.meshgrid(np.asarray([0.0, 1.0]), values)
    color_grid = np.repeat(values[:, None], 2, axis=1)
    cax.pcolormesh(
        grid_x,
        grid_y,
        color_grid,
        cmap=cmap,
        norm=Normalize(vmin, vmax),
        shading="gouraud",
        rasterized=False,
    )
    cax.set_xlim(0, 1)
    cax.set_ylim(vmin, vmax)
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_title(title, fontsize=title_size, loc="left", pad=5)
    if cmap == HEAT_CMAP and np.isclose(vmin, -vmax):
        cax.set_yticks([-1.0, 0.0, 1.0])
    else:
        cax.set_yticks(np.linspace(vmin, vmax, 4)[1:])
    cax.yaxis.set_major_formatter(FuncFormatter(scientific_or_decimal))
    cax.tick_params(axis="y", labelsize=tick_size, length=2.5)
    if label:
        cax.set_ylabel(label, fontsize=title_size, labelpad=5)
    for spine in cax.spines.values():
        spine.set_visible(False)
    return cax


def load_display_table(path: Path, expected_query: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"ID", "Description", "pvalue", "p.adjust", "Count", "ONTOLOGY", "query_id"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")
    if not table["query_id"].eq(expected_query).all():
        raise ValueError(f"{path} query_id differs from {expected_query}")
    if not table["p.adjust"].lt(0.05).all():
        raise ValueError(f"{path} contains a non-significant display term")
    expected = table.sort_values(
        ["p.adjust", "pvalue", "Count", "Description"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    if table["ID"].tolist() != expected["ID"].tolist():
        raise ValueError(f"{path} is not in the audited display order")
    if len(table) > 20:
        raise ValueError(f"{path} contains more than 20 display terms")
    return table


def plot_go_axes(
    ax: mpl.axes.Axes,
    table: pd.DataFrame,
    *,
    title: str,
    wrap_width: int,
    title_size: float,
    tick_size: float,
    tick_line_spacing: float = 1.2,
) -> tuple[float, float]:
    values = table["p.adjust"].to_numpy(float)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin or (vmax - vmin) <= np.finfo(float).eps * max(abs(vmin), abs(vmax), 1e-300):
        vmax = max(vmin * 1.01, np.nextafter(vmin, np.inf))
    colors = plt.get_cmap(GO_CMAP)(Normalize(vmin, vmax)(values))
    y = np.arange(len(table))
    # Keep submitted per-bar thickness when a corrected query has fewer than
    # 20 significant terms (S9 Pattern 2 has 11); do not add non-significant
    # filler terms merely to recover the old bar count.
    bar_height = 0.8 * min(1.0, len(table) / 20.0)
    ax.barh(y, table["Count"].to_numpy(float), color=colors, edgecolor="none", height=bar_height)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_term(value, wrap_width) for value in table["Description"]], fontsize=tick_size)
    for label in ax.get_yticklabels():
        label.set_linespacing(tick_line_spacing)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=title_size, pad=7)
    ax.set_xlabel("Count", fontsize=9)
    ax.set_ylabel("")
    ax.set_xlim(0, max(float(table["Count"].max()) * 1.05, 1.0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(axis="x", labelsize=7.5, length=0)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(True, color=GRID_COLOR, linewidth=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return vmin, vmax


def save_figure(fig: mpl.figure.Figure, output_dir: Path, stem: str, dpi: int = 200) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "pdf": output_dir / f"{stem}.pdf",
        "svg": output_dir / f"{stem}.svg",
        "png": output_dir / f"{stem}.png",
    }
    fig.savefig(outputs["pdf"], format="pdf", facecolor="white", bbox_inches=None)
    fig.savefig(outputs["svg"], format="svg", facecolor="white", bbox_inches=None)
    fig.savefig(outputs["png"], format="png", dpi=dpi, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return outputs


def render_s9(go_dir: Path, output_dir: Path) -> tuple[dict[str, Path], dict[str, object]]:
    p1 = load_display_table(go_dir / "tables" / "s9_pattern_1_enrichGO_display_top20.csv", "s9_pattern_1")
    p2 = load_display_table(go_dir / "tables" / "s9_pattern_2_enrichGO_display_top20.csv", "s9_pattern_2")
    fig = plt.figure(figsize=(7.31, 11.0), facecolor="white")

    ax1 = fig.add_axes([0.296, 0.555, 0.526, 0.405])
    vmin1, vmax1 = plot_go_axes(
        ax1,
        p1,
        title="brain_pattern_1_genes.txt - GO (ALL) - Barplot",
        wrap_width=30,
        title_size=13.0,
        tick_size=8.0,
    )
    vector_colorbar(fig, [0.862, 0.703, 0.025, 0.095], vmin=vmin1, vmax=vmax1, cmap=GO_CMAP, title="p.adjust")
    fig.text(0.002, 0.995, "a", fontsize=16, fontweight="bold", ha="left", va="top")

    ax2 = fig.add_axes([0.296, 0.040, 0.551, 0.425])
    vmin2, vmax2 = plot_go_axes(
        ax2,
        p2,
        title="brain_pattern_2_genes.txt - GO (ALL) - Barplot",
        wrap_width=30,
        title_size=13.0,
        tick_size=8.0,
    )
    vector_colorbar(fig, [0.885, 0.197, 0.025, 0.095], vmin=vmin2, vmax=vmax2, cmap=GO_CMAP, title="p.adjust")
    fig.text(0.002, 0.515, "b", fontsize=16, fontweight="bold", ha="left", va="top")

    outputs = save_figure(
        fig,
        output_dir,
        "Figure_S9_MOSTA_latest_package_clusterProfiler_GO_exact_submitted_style",
    )
    return outputs, {"pattern_1_display_terms": len(p1), "pattern_2_display_terms": len(p2)}


def render_wave_axes(
    fig: mpl.figure.Figure,
    shared: Path,
) -> dict[str, object]:
    wave_dir = shared / "s10_developmental_wave"
    matrix = pd.read_csv(wave_dir / "s10_top1000_peak_ordered_profiles.csv", index_col=0)
    matrix.columns = [float(value) for value in matrix.columns]
    assignments = pd.read_csv(wave_dir / "s10_top1000_dp3_assignments.csv")
    if matrix.shape != (1000, 13):
        raise ValueError(f"Unexpected S10 wave shape: {matrix.shape}")
    if assignments["profile"].astype(str).tolist() != matrix.index.astype(str).tolist():
        raise ValueError("S10 assignments and peak-ordered matrix differ")
    phases = assignments["phase"].to_numpy(int)
    if not np.array_equal(np.unique(phases), np.asarray([1, 2, 3])) or np.any(np.diff(phases) < 0):
        raise ValueError("S10 phases are not contiguous 1/2/3")
    counts = pd.Series(phases).value_counts().sort_index().to_dict()
    if counts != {1: 483, 2: 299, 3: 218}:
        raise ValueError(f"Unexpected corrected S10 phase sizes: {counts}")

    ax_strip = fig.add_axes([0.164, 0.655, 0.023, 0.327])
    strip = (phases - 1).reshape(-1, 1)
    ax_strip.pcolormesh(
        np.arange(2),
        np.arange(1001),
        strip,
        cmap=ListedColormap(PHASE_COLORS),
        shading="flat",
        rasterized=False,
    )
    ax_strip.invert_yaxis()
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    ax_strip.set_title("Phase", fontsize=7.5, pad=5, color="#625b56")
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    ax = fig.add_axes([0.201, 0.655, 0.647, 0.327])
    values = matrix.to_numpy(float)
    mesh = ax.pcolormesh(
        np.arange(14),
        np.arange(1001),
        values,
        cmap=HEAT_CMAP,
        vmin=-HEAT_LIMIT,
        vmax=HEAT_LIMIT,
        shading="flat",
        rasterized=False,
    )
    ax.invert_yaxis()
    ax.set_yticks([])
    times = matrix.columns.to_numpy(float)
    tick_indices = np.arange(0, len(times), 2)
    ax.set_xticks(tick_indices + 0.5)
    ax.set_xticklabels([f"{times[index]:.2f}" for index in tick_indices], rotation=35, ha="right", fontsize=7)
    ax.set_xlabel("Time", fontsize=8)
    ax.set_title(
        "Developmental wave map ordered by peak time with phase annotation (k=3)",
        fontsize=10.8,
        loc="left",
        pad=7,
    )
    for peak_time in sorted(assignments["peak_time"].unique()):
        ax.axvline(np.searchsorted(times, peak_time), color="#efe9df", linewidth=0.55, alpha=0.85)
    for boundary in np.cumsum(list(counts.values()))[:-1]:
        ax.axhline(boundary, color="#ece6dd", linewidth=0.8)
        ax_strip.axhline(boundary, color="#ece6dd", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.126, 0.817, "Top 1000 dynamic genes", rotation=90, ha="center", va="center", fontsize=8.5, color="#625b56")
    vector_colorbar(
        fig,
        [0.864, 0.765, 0.010, 0.105],
        vmin=-HEAT_LIMIT,
        vmax=HEAT_LIMIT,
        cmap=HEAT_CMAP,
        title="",
        label="Gene-wise z-score",
        title_size=7,
        tick_size=6.5,
    )
    return {"phase_sizes": {str(key): int(value) for key, value in counts.items()}, "heat_mesh": mesh}


def render_s10(shared: Path, go_dir: Path, output_dir: Path) -> tuple[dict[str, Path], dict[str, object]]:
    tables = {
        phase: load_display_table(
            go_dir / "tables" / f"s10_phase_{phase}_enrichGO_display_top20.csv",
            f"s10_phase_{phase}",
        )
        for phase in (1, 2, 3)
    }
    fig = plt.figure(figsize=(7.98, 11.0), facecolor="white")
    wave_details = render_wave_axes(fig, shared)
    fig.text(0.002, 0.995, "a", fontsize=15, fontweight="bold", ha="left", va="top")

    go_layout = {
        1: ([0.142, 0.353, 0.280, 0.225], [0.439, 0.438, 0.012, 0.090], "b", 0.606),
        2: ([0.645, 0.353, 0.290, 0.225], [0.950, 0.438, 0.012, 0.090], "c", 0.603),
        3: ([0.171, 0.030, 0.331, 0.277], [0.517, 0.118, 0.012, 0.090], "d", 0.323),
    }
    for phase in (1, 2, 3):
        axes_box, colorbar_box, letter, letter_y = go_layout[phase]
        ax = fig.add_axes(axes_box)
        vmin, vmax = plot_go_axes(
            ax,
            tables[phase],
            title=f"brain_wave_phase_{phase}_genes.txt - GO (ALL) - Barplot",
            wrap_width=32,
            title_size=8.6 if phase < 3 else 10.0,
            tick_size=4.3 if phase < 3 else 6.0,
            tick_line_spacing=0.82 if phase < 3 else 0.90,
        )
        vector_colorbar(
            fig,
            colorbar_box,
            vmin=vmin,
            vmax=vmax,
            cmap=GO_CMAP,
            title="p.adjust",
            title_size=6.8 if phase < 3 else 7.5,
            tick_size=5.5 if phase < 3 else 6.5,
        )
        fig.text(0.003 if phase != 2 else 0.527, letter_y, letter, fontsize=15, fontweight="bold", ha="left", va="top")

    outputs = save_figure(
        fig,
        output_dir,
        "Figure_S10_MOSTA_latest_package_DP3_clusterProfiler_GO_exact_submitted_style",
    )
    details = {**wave_details, **{f"phase_{phase}_display_terms": len(table) for phase, table in tables.items()}}
    details.pop("heat_mesh", None)
    return outputs, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-run", required=True, type=Path)
    parser.add_argument("--clusterprofiler-run", required=True, type=Path)
    parser.add_argument("--s9-output-dir", required=True, type=Path)
    parser.add_argument("--s10-output-dir", required=True, type=Path)
    parser.add_argument("--s9-style-reference", required=True, type=Path)
    parser.add_argument("--s10-style-reference", required=True, type=Path)
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    shared = args.shared_run.resolve()
    go_dir = args.clusterprofiler_run.resolve()
    s9_outputs, s9_details = render_s9(go_dir, args.s9_output_dir.resolve())
    s10_outputs, s10_details = render_s10(shared, go_dir, args.s10_output_dir.resolve())
    manifest = {
        "schema_version": 1,
        "dataset": "MOSTA",
        "numerical_authority": {
            "shared_latest_package_run": identity(shared / "summary.json"),
            "clusterprofiler_manifest": identity(go_dir / "manifest.json"),
            "numerical_audit": "must be PASS before archival",
        },
        "style_authority": {
            "s9": identity(args.s9_style_reference.resolve()),
            "s10": identity(args.s10_style_reference.resolve()),
            "visual_grammar": "submitted JPEG geometry, direct raw p.adjust five-stop teal-to-coral scale, horizontal Count bars, original filename-form titles",
        },
        "scientific_contract": {
            "go_display": "all and only FDR<0.05 terms, ordered by audited clusterProfiler ranking, up to 20; no filler or cherry picking",
            "go_colormap": {"name": GO_CMAP.name, "colors": GO_PALETTE},
            "wave_colormap": HEAT_CMAP,
            "wave_limits": [-HEAT_LIMIT, HEAT_LIMIT],
            "phase_colors": list(PHASE_COLORS),
            "coordinate_warp": False,
        },
        "s9": {"outputs": {key: identity(path) for key, path in s9_outputs.items()}, "details": s9_details},
        "s10": {"outputs": {key: identity(path) for key, path in s10_outputs.items()}, "details": s10_details},
        "software": {
            "matplotlib": mpl.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    manifest_path = args.s9_output_dir.resolve().parent / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "s9": {key: str(value) for key, value in s9_outputs.items()}, "s10": {key: str(value) for key, value in s10_outputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
