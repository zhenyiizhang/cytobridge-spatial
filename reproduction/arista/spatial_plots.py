"""Spatial population, growth and composition plots used for ARISTA."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


OBSERVED_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0)
MIDPOINT_TIMES = (0.5, 1.5, 2.5, 3.5)
S12_LAYOUT = (
    (0.0, 'Observed'), (0.0, 'Generated'), (0.5, 'Generated'),
    (1.0, 'Observed'), (1.0, 'Generated'), (1.5, 'Generated'),
    (2.0, 'Observed'), (2.0, 'Generated'), (2.5, 'Generated'),
    (3.0, 'Observed'), (3.0, 'Generated'), (3.5, 'Generated'),
    (4.0, 'Observed'), (4.0, 'Generated'), None, None,
)

DENSE_TIMES = tuple(sorted(OBSERVED_TIMES + MIDPOINT_TIMES))

FIXED_TIMESTAMP = "2026-08-23T00:00:00+00:00"

FIXED_PDF_DATE = datetime(2026, 8, 23, tzinfo=timezone.utc)

S12_SUBMITTED_CANVAS_PT = (505.44, 502.262564)

S13_SUBMITTED_CANVAS_PT = (900.132969, 865.422001)

S12_SUBMITTED_RASTER_PX = (2106, 2093)

S13_SUBMITTED_RASTER_PX = (3751, 3606)

@dataclass
class SpatialPanel:
    time: float
    source: str
    x: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    ids: np.ndarray
    input_path: Path
    coordinate_basis: str

    @property
    def key(self) -> tuple[float, str]:
        return (self.time, self.source)

def validate_labels(labels: Iterable[str], palette: dict[str, str], context: str) -> None:
    missing = sorted(set(map(str, labels)) - set(palette))
    if missing:
        raise KeyError(f"Labels absent from the canonical palette in {context}: {missing}")

def configure_s12_legacy_style(font_family: str = "DejaVu Sans") -> None:
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(
        {
            "font.family": str(font_family),
            "savefig.dpi": 300,
            "svg.fonttype": "path",
            "svg.hashsalt": "arista-s12-s14-legacy-style-3c87a3e",
        }
    )

def configure_review_legacy_style(font_family: str = "DejaVu Sans") -> None:
    matplotlib.rcdefaults()
    sns.set_theme(style="white", context="paper")
    matplotlib.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "font.size": 10,
            "font.family": str(font_family),
            "svg.fonttype": "path",
            "svg.hashsalt": "arista-s12-s14-legacy-style-3c87a3e",
        }
    )

def save_jpeg(png_path: Path, jpeg_path: Path) -> None:
    image = Image.open(png_path).convert("RGB")
    image.save(jpeg_path, format="JPEG", quality=95, subsampling=0, dpi=(300, 300), optimize=False)

def submitted_canvas_bbox(fig: plt.Figure, canvas_pt: tuple[float, float]) -> Bbox:
    """Return a fixed legacy canvas around the unchanged tight plot content.

    The historical renderers used ``bbox_inches='tight'``.  A tight bounding
    box is data-dependent when equal-aspect spatial axes receive corrected
    coordinates, so the same renderer can otherwise change the outer canvas.
    We preserve the tight content at one point per point and only add symmetric
    white margin up to the physical canvas measured from the submitted SVG.
    """

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    target_width = float(canvas_pt[0]) / 72.0
    target_height = float(canvas_pt[1]) / 72.0
    # Font rendering differs slightly across platforms. Extend white margins
    # when needed, keeping every axis and label at its original size.
    target_width = max(target_width, tight.width)
    target_height = max(target_height, tight.height)
    return Bbox.from_bounds(
        tight.x0 - (target_width - tight.width) / 2.0,
        tight.y0 - (target_height - tight.height) / 2.0,
        target_width,
        target_height,
    )

def submitted_raster_bbox(fig: plt.Figure, raster_px: tuple[int, int], dpi: int = 300) -> Bbox:
    """Return an exact raster-sized canvas around the unchanged tight content."""

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
    # A tiny epsilon avoids platform-dependent truncation of an integer-valued
    # floating-point pixel extent inside Matplotlib's Agg backend.
    target_width = (float(raster_px[0]) + 1e-6) / float(dpi)
    target_height = (float(raster_px[1]) + 1e-6) / float(dpi)
    target_width = max(target_width, tight.width)
    target_height = max(target_height, tight.height)
    return Bbox.from_bounds(
        tight.x0 - (target_width - tight.width) / 2.0,
        tight.y0 - (target_height - tight.height) / 2.0,
        target_width,
        target_height,
    )

def save_mpl_figure(
    fig: plt.Figure,
    name: str,
    directories: dict[str, Path],
    *,
    submitted_canvas_pt: tuple[float, float] | None = None,
    submitted_raster_px: tuple[int, int] | None = None,
) -> dict[str, Path]:
    paths = {
        "svg": directories["vector"] / f"{name}.svg",
        "pdf": directories["pdf"] / f"{name}.pdf",
        "png": directories["png"] / f"{name}.png",
        "jpg": directories["jpeg"] / f"{name}.jpg",
    }
    bbox_inches: str | Bbox = (
        submitted_canvas_bbox(fig, submitted_canvas_pt)
        if submitted_canvas_pt is not None
        else "tight"
    )
    fig.savefig(
        paths["svg"],
        format="svg",
        facecolor="white",
        bbox_inches=bbox_inches,
        metadata={"Date": FIXED_TIMESTAMP, "Creator": "ARISTA corrected legacy-style renderer"},
    )
    fig.savefig(
        paths["pdf"],
        format="pdf",
        facecolor="white",
        bbox_inches=bbox_inches,
        metadata={
            "Creator": "ARISTA corrected legacy-style renderer",
            "CreationDate": FIXED_PDF_DATE,
            "ModDate": FIXED_PDF_DATE,
        },
    )
    raster_bbox: str | Bbox = (
        submitted_raster_bbox(fig, submitted_raster_px)
        if submitted_raster_px is not None
        else bbox_inches
    )
    fig.savefig(paths["png"], format="png", facecolor="white", bbox_inches=raster_bbox, dpi=300)
    save_jpeg(paths["png"], paths["jpg"])
    return paths

def plot_s12(
    panels: dict[tuple[float, str], SpatialPanel],
    palette: dict[str, str],
    flag_keys: set[tuple[float, str]],
    hide_objective_flags: bool,
    name: str,
    directories: dict[str, Path],
    *,
    font_family: str = "DejaVu Sans",
) -> tuple[dict[str, Path], pd.DataFrame]:
    configure_s12_legacy_style(font_family)
    fig, axes = plt.subplots(4, 4, figsize=(8.8, 8.8), dpi=300)
    inventory: list[dict] = []
    for layout_index, (ax, item) in enumerate(zip(axes.flat, S12_LAYOUT)):
        if item is None:
            ax.axis("off")
            ax.set_facecolor("white")
            continue
        panel = panels[item]
        keep = np.ones(len(panel.x), dtype=bool)
        if hide_objective_flags:
            keep = np.asarray([(panel.time, str(source_id)) not in flag_keys for source_id in panel.ids])
        colors = [palette.get(str(label), "#888888") for label in panel.labels[keep]]
        ax.set_facecolor("white")
        ax.scatter(
            panel.x[keep],
            panel.y[keep],
            s=2.5,
            c=colors,
            linewidths=0,
            alpha=0.9,
            rasterized=False,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"t = {panel.time:.1f} | {panel.source}", color="#1a1a1a", fontsize=8, pad=3)
        inventory.append(
            {
                "layout_index": int(layout_index),
                "time": float(panel.time),
                "source": panel.source,
                "n_compute": int(len(panel.x)),
                "n_display": int(keep.sum()),
                "n_objective_display_hidden": int((~keep).sum()),
                "input_path": str(panel.input_path.resolve()),
                "coordinate_basis": panel.coordinate_basis,
            }
        )
    paths = save_mpl_figure(
        fig,
        name,
        directories,
        submitted_canvas_pt=S12_SUBMITTED_CANVAS_PT,
        submitted_raster_px=S12_SUBMITTED_RASTER_PX,
    )
    plt.close(fig)
    return paths, pd.DataFrame(inventory)

def plot_s13(
    sample: pd.DataFrame,
    hide_objective_flags: bool,
    name: str,
    directories: dict[str, Path],
    tables_dir: Path,
    *,
    fit_package_native_canvas: bool = False,
    annotate_injury_reference: bool = False,
    annotate_injury_all_panels: bool = False,
    font_family: str = "DejaVu Sans",
) -> tuple[dict[str, Path], pd.DataFrame]:
    configure_review_legacy_style(font_family)
    # The new package-native spatial ranges make equal-aspect panels extend
    # about one point beyond the submitted fixed canvas.  A 0.02-inch vertical
    # figure adjustment restores the measured outer canvas without changing
    # fonts, markers, colormap, normalization, or panel grammar.  Historical
    # callers retain the exact original 12.6 x 12.6-inch figure.
    if fit_package_native_canvas:
        # Arial has a slightly taller tight bounding box than DejaVu Sans for
        # this fixed legacy page.  Compensate by exactly 0.02 inch so the
        # scientific axes and submitted outer canvas remain unchanged.
        figure_height = 12.5535 if str(font_family).lower() == "arial" else 12.58
    else:
        figure_height = 12.6
    fig, axes = plt.subplots(3, 3, figsize=(12.6, figure_height), squeeze=False)
    scale_rows: list[dict] = []
    for ax in axes.flat:
        ax.axis("off")
    for ax, (time, sub) in zip(axes.flat, sample.groupby("time", sort=True)):
        shown = sub[~sub["objective_isolation_flag"]].copy() if hide_objective_flags else sub.copy()
        # Policy B is display-only: use the exact same old seed-42 sample to
        # freeze normalization, then suppress flagged glyphs.  This keeps the
        # colormap contract identical between A and B.
        values = sub["growth"].to_numpy(dtype=float)
        q05, q95 = np.percentile(values, [5, 95])
        if q05 == q95:
            q95 = q05 + np.finfo(float).eps
        norm = Normalize(vmin=float(q05), vmax=float(q95), clip=True)
        scatter = ax.scatter(
            shown["x"],
            shown["y"],
            c=shown["growth"],
            cmap="viridis",
            s=2.0,
            linewidths=0,
            alpha=0.85,
            norm=norm,
        )
        source = "observed" if float(time) in OBSERVED_TIMES else "simulated"
        ax.set_title(f"t={float(time):.1f} ({source})", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        annotate_this_panel = bool(
            annotate_injury_reference
            and (
                annotate_injury_all_panels
                or np.isclose(float(time), 0.0)
            )
        )
        if annotate_this_panel:
            # Anatomical locator only: the visible orientation places the
            # injured half-brain on the right, with the resection in its lower
            # portion.  The outline is deliberately schematic and is not
            # estimated from the growth-rate colors.
            injury = Ellipse(
                (0.69, 0.17),
                width=0.28,
                height=0.18,
                angle=-8.0,
                transform=ax.transAxes,
                fill=False,
                edgecolor="#242b83",
                linewidth=1.4,
                linestyle=(0, (4, 3)),
                zorder=20,
            )
            ax.add_patch(injury)
            if np.isclose(float(time), 0.0):
                ax.annotate(
                    "Right hemisphere",
                    xy=(0.76, 0.69),
                    xycoords="axes fraction",
                    xytext=(0.58, 0.92),
                    textcoords="axes fraction",
                    fontsize=8,
                    color="#1f1f1f",
                    ha="left",
                    va="top",
                    arrowprops={"arrowstyle": "-|>", "color": "#1f1f1f", "lw": 0.9},
                    annotation_clip=False,
                    zorder=21,
                )
            ax.annotate(
                "Injury region",
                xy=(0.69, 0.17),
                xycoords="axes fraction",
                xytext=(0.46, 0.035),
                textcoords="axes fraction",
                fontsize=8,
                fontweight="bold",
                color="#242b83",
                ha="left",
                va="bottom",
                arrowprops={"arrowstyle": "-|>", "color": "#242b83", "lw": 1.0},
                annotation_clip=False,
                zorder=21,
            )
        scale_rows.append(
            {
                "time": float(time),
                "source": source,
                "n_compute": int(sub["n_compute_panel"].iloc[0]),
                "n_seed42_sample": int(len(sub)),
                "n_display": int(len(shown)),
                "n_objective_display_hidden": int(len(sub) - len(shown)),
                "q05_display_sample": float(q05),
                "q95_display_sample": float(q95),
            }
        )
    fig.suptitle("Arista growth-rate maps across dense time grid", fontsize=13)
    fig.tight_layout()
    paths = save_mpl_figure(
        fig,
        name,
        directories,
        submitted_canvas_pt=S13_SUBMITTED_CANVAS_PT,
        submitted_raster_px=S13_SUBMITTED_RASTER_PX,
    )
    plt.close(fig)
    scales = pd.DataFrame(scale_rows)
    suffix = "B_nnmad20" if hide_objective_flags else "A_all_valid"
    scales.to_csv(tables_dir / f"s13_scale_and_display_counts_{suffix}.csv", index=False)
    return paths, scales

def plot_s14b(
    fraction_table: pd.DataFrame,
    palette: dict[str, str],
    name: str,
    directories: dict[str, Path],
    tables_dir: Path,
) -> tuple[dict[str, Path], pd.DataFrame]:
    configure_review_legacy_style()
    validate_labels(fraction_table.columns, palette, "corrected S14b composition")
    global_order = fraction_table.mean(axis=0).sort_values(ascending=False)
    selected = list(global_order.head(min(15, len(global_order))).index)
    display = fraction_table.copy()
    if len(selected) < display.shape[1]:
        display["Other"] = display.drop(columns=selected).sum(axis=1)
        display = display[selected + ["Other"]]
    else:
        display = display[selected]
    display_pct = display * 100.0
    colors = [palette.get(label, "#c9c3b8" if label == "Other" else "#808080") for label in display_pct.columns]
    fig, ax = plt.subplots(figsize=(11.0, 4.8), facecolor="white")
    bottom = np.zeros(display_pct.shape[0], dtype=float)
    x = np.arange(display_pct.shape[0], dtype=float)
    for cell_type, color in zip(display_pct.columns, colors):
        values = display_pct[cell_type].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.76,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=cell_type,
        )
        bottom += values
    ax.set_xlim(-0.55, len(x) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cell proportion (%)")
    ax.set_xlabel("Time")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{time:.2f}" for time in display_pct.index], rotation=0)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#e9e3d8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Arista cell composition across observed and interpolated time points",
        loc="left",
        fontsize=12.5,
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Cell type",
        fontsize=8,
        title_fontsize=8.5,
    )
    fig.tight_layout()
    paths = save_mpl_figure(fig, name, directories)
    plt.close(fig)
    display_pct.to_csv(tables_dir / "s14b_corrected_top15_other_percent.csv", index_label="time")
    return paths, display_pct
