#!/usr/bin/env python3
"""Render a biology-first CXCL12a->CXCR4a counterfactual mechanism figure.

The input is the strict bundle written by ``lr_gene_counterfactual.py``.  This
renderer does not rerun or alter the model.  It follows the fixed sender and
receiver cells through the stored deterministic baseline and counterfactual
endpoints, maps the baseline gated GNN edges on the expression-compatible
CXCL12a/CXCR4a support, and contrasts interaction-on with the same checkpoint
evaluated interaction-off.

The resulting figure is intentionally bounded: it visualizes trained-model
sensitivity and does not claim experimental causality or target specificity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MechanismData:
    cells: pd.DataFrame
    target_edges: pd.DataFrame
    baseline_on: np.ndarray
    counterfactual_on: np.ndarray
    baseline_off: np.ndarray
    counterfactual_off: np.ndarray
    sham_state_mediation: np.ndarray
    primary_state_mediation: float
    primary_relative_rank: float
    anchor_id: str
    grouping_seed: int
    knockdown_fraction: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--anchor-id", default="3_to_4")
    parser.add_argument("--grouping-seed", default=101, type=int)
    parser.add_argument("--knockdown-fraction", default=1.0, type=float)
    parser.add_argument(
        "--h5ad",
        type=Path,
        help=(
            "Optional source h5ad. When supplied, sender/receiver cohorts and "
            "baseline edges are shown at the observed anchor-start coordinates. "
            "Without it, the renderer makes an explicitly labelled endpoint-"
            "geometry preview that must not be used as the manuscript edge map."
        ),
    )
    parser.add_argument("--spatial-key", default="spatial_aligned")
    parser.add_argument("--arrow-magnification", default=10.0, type=float)
    parser.add_argument("--n-arrows", default=40, type=int)
    parser.add_argument("--dpi", default=300, type=int)
    return parser


def _fraction_token(value: float) -> str:
    if not 0.0 < value <= 1.0:
        raise ValueError("knockdown_fraction must be in (0, 1].")
    return f"{value:g}".replace(".", "_")


def _array_key(
    anchor_id: str,
    grouping_seed: int,
    stem: str,
) -> str:
    return f"{anchor_id}__seed_{grouping_seed}__{stem}"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mechanism_data(
    bundle_dir: Path,
    *,
    anchor_id: str = "3_to_4",
    grouping_seed: int = 101,
    knockdown_fraction: float = 1.0,
) -> MechanismData:
    """Load and validate the fixed-cohort primary counterfactual artifacts."""
    bundle_dir = Path(bundle_dir)
    cells = pd.read_csv(bundle_dir / "cohort_cells.csv.gz")
    cells = cells[cells["anchor_id"].astype(str) == str(anchor_id)].copy()
    cells = cells.sort_values("local_index", kind="stable").reset_index(drop=True)
    expected_local = np.arange(len(cells), dtype=np.int64)
    if not np.array_equal(cells["local_index"].to_numpy(dtype=np.int64), expected_local):
        raise ValueError("cohort local_index must be contiguous and endpoint-aligned.")
    for column in ("fixed_primary_ligand_positive_sender", "fixed_receiver"):
        cells[column] = cells[column].astype(bool)
        if not cells[column].any():
            raise ValueError(f"{column} is empty for {anchor_id}.")

    token = _fraction_token(float(knockdown_fraction))
    arrays_path = bundle_dir / "primary_counterfactual_arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as payload:
        keys = {
            "baseline_on": _array_key(
                anchor_id, grouping_seed, "baseline_on_endpoint"
            ),
            "baseline_off": _array_key(
                anchor_id, grouping_seed, "baseline_off_endpoint"
            ),
            "counterfactual_on": _array_key(
                anchor_id,
                grouping_seed,
                f"ligand__kd_{token}__on_endpoint",
            ),
            "counterfactual_off": _array_key(
                anchor_id,
                grouping_seed,
                f"ligand__kd_{token}__off_endpoint",
            ),
        }
        missing = sorted(set(keys.values()).difference(payload.files))
        if missing:
            raise KeyError(f"Missing counterfactual arrays: {missing}")
        loaded = {name: np.asarray(payload[key], dtype=np.float64) for name, key in keys.items()}

    endpoint_shapes = {array.shape for array in loaded.values()}
    if len(endpoint_shapes) != 1:
        raise ValueError(f"Endpoint arrays do not share one shape: {endpoint_shapes}")
    endpoint_shape = next(iter(endpoint_shapes))
    if endpoint_shape[0] != len(cells) or endpoint_shape[1] < 3:
        raise ValueError(
            "Endpoint arrays must align to cohort rows and contain 2 spatial "
            "coordinates plus at least one state coordinate."
        )
    if not all(np.isfinite(array).all() for array in loaded.values()):
        raise ValueError("Counterfactual endpoint arrays contain non-finite values.")

    edges = pd.read_csv(
        bundle_dir / "primary_edge_diagnostics.csv.gz",
        low_memory=False,
    )
    edges = edges[
        (edges["anchor_id"].astype(str) == str(anchor_id))
        & (edges["condition"].astype(str) == "baseline")
        & (edges["grouping_seed"].astype(int) == int(grouping_seed))
    ].copy()
    sender_by_index = cells.set_index("local_index")[
        "fixed_primary_ligand_positive_sender"
    ]
    receiver_by_index = cells.set_index("local_index")["fixed_receiver"]
    source_sender = edges["source_index"].map(sender_by_index).fillna(False)
    target_receiver = edges["target_index"].map(receiver_by_index).fillna(False)
    target_edges = edges[source_sender & target_receiver].copy()
    if target_edges.empty:
        raise ValueError("No baseline gated edge exists on the fixed LR support.")

    mediation = pd.read_csv(bundle_dir / "interaction_mediation.csv")
    mediation_mask = (
        (mediation["anchor_id"].astype(str) == str(anchor_id))
        & np.isclose(
            mediation["knockdown_fraction"].to_numpy(dtype=float),
            float(knockdown_fraction),
        )
        & (mediation["grouping_seed"].astype(int) == int(grouping_seed))
        & (
            mediation["cohort"].astype(str)
            == "fixed_receptor_positive_ligand_negative"
        )
        & (mediation["space"].astype(str) == "state")
    )
    selected_mediation = mediation[mediation_mask].copy()
    primary_rows = selected_mediation[
        selected_mediation["condition"].astype(str) == "ligand"
    ]
    sham_rows = selected_mediation[selected_mediation["is_sham"].fillna(False).astype(bool)]
    if len(primary_rows) != 1 or sham_rows.empty:
        raise ValueError(
            "Expected one primary ligand row and at least one matched-sham row "
            "for receiver-state interaction mediation."
        )
    primary_state_mediation = float(
        primary_rows.iloc[0]["interaction_mediated_centroid_norm"]
    )
    sham_state_mediation = sham_rows[
        "interaction_mediated_centroid_norm"
    ].to_numpy(dtype=float)
    primary_relative_rank = float(
        (np.count_nonzero(sham_state_mediation <= primary_state_mediation) + 1)
        / (len(sham_state_mediation) + 1)
    )

    return MechanismData(
        cells=cells,
        target_edges=target_edges,
        baseline_on=loaded["baseline_on"],
        counterfactual_on=loaded["counterfactual_on"],
        baseline_off=loaded["baseline_off"],
        counterfactual_off=loaded["counterfactual_off"],
        sham_state_mediation=sham_state_mediation,
        primary_state_mediation=primary_state_mediation,
        primary_relative_rank=primary_relative_rank,
        anchor_id=str(anchor_id),
        grouping_seed=int(grouping_seed),
        knockdown_fraction=float(knockdown_fraction),
    )


def _observed_start_coordinates(
    h5ad: Path,
    cells: pd.DataFrame,
    *,
    spatial_key: str,
) -> np.ndarray:
    try:
        import anndata as ad
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("anndata is required when --h5ad is supplied.") from error

    data = ad.read_h5ad(h5ad, backed="r")
    try:
        if spatial_key not in data.obsm:
            raise KeyError(f"{spatial_key!r} is absent from h5ad.obsm.")
        indices = cells["global_index"].to_numpy(dtype=np.int64)
        if indices.min() < 0 or indices.max() >= data.n_obs:
            raise IndexError("cohort global_index is outside the h5ad observation range.")
        observed_names = np.asarray(data.obs_names[indices]).astype(str)
        expected_names = cells["obs_name"].astype(str).to_numpy()
        if not np.array_equal(observed_names, expected_names):
            mismatch = int(np.count_nonzero(observed_names != expected_names))
            raise ValueError(
                f"h5ad/cohort identity mismatch for {mismatch} observations."
            )
        coords = np.asarray(data.obsm[spatial_key])[indices, :2].astype(np.float64)
    finally:
        data.file.close()
    if coords.shape != (len(cells), 2) or not np.isfinite(coords).all():
        raise ValueError("Observed start coordinates are not finite n_cells x 2.")
    return coords


def _set_spatial_axis(ax: plt.Axes, coords: np.ndarray) -> None:
    ax.set_aspect("equal", adjustable="box")
    margin = 0.04 * max(float(np.ptp(coords[:, 0])), float(np.ptp(coords[:, 1])))
    margin = max(margin, 1e-3)
    ax.set_xlim(float(coords[:, 0].min() - margin), float(coords[:, 0].max() + margin))
    ax.set_ylim(float(coords[:, 1].min() - margin), float(coords[:, 1].max() + margin))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _cohort_composition_text(cells: pd.DataFrame, mask: np.ndarray) -> str:
    counts = cells.loc[mask, "cell_type"].astype(str).value_counts()
    top = counts.head(3)
    return "\n".join(f"{label}: {int(count)}" for label, count in top.items())


def _plot_tissue(
    ax: plt.Axes,
    coords: np.ndarray,
    *,
    background_size: float = 5.0,
) -> None:
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=background_size,
        c="#d8d8d8",
        linewidths=0,
        alpha=0.65,
        rasterized=True,
        zorder=0,
    )


def _add_lr_support(
    ax: plt.Axes,
    *,
    coords: np.ndarray,
    data: MechanismData,
    sender: np.ndarray,
    receiver: np.ndarray,
) -> None:
    """Overlay the fixed LR-compatible gated support without implying LR messages."""
    edge_source = data.target_edges["source_index"].to_numpy(dtype=int)
    edge_target = data.target_edges["target_index"].to_numpy(dtype=int)
    segments = np.stack((coords[edge_source], coords[edge_target]), axis=1)
    strength = data.target_edges["complete_message_norm_joint"].to_numpy(dtype=float)
    strength_scale = max(
        float(np.quantile(strength, 0.9)),
        float(np.finfo(float).eps),
    )
    line_widths = 0.3 + 1.6 * np.clip(strength / strength_scale, 0.0, 1.0)
    ax.add_collection(
        LineCollection(
            segments,
            colors="#ef6c00",
            linewidths=line_widths,
            alpha=0.45,
            rasterized=True,
            zorder=1,
        )
    )
    ax.scatter(
        coords[sender, 0],
        coords[sender, 1],
        s=15,
        c="#d81b60",
        linewidths=0,
        alpha=0.9,
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        coords[receiver, 0],
        coords[receiver, 1],
        s=17,
        c="#1565c0",
        linewidths=0,
        alpha=0.9,
        rasterized=True,
        zorder=2,
    )


def _draw_endpoint_arrows(
    ax: plt.Axes,
    *,
    endpoint_coords: np.ndarray,
    delta: np.ndarray,
    receiver: np.ndarray,
    arrow_magnification: float,
    n_arrows: int,
) -> int:
    """Draw the largest fixed-receiver endpoint shifts and return their count."""
    receiver_indices = np.flatnonzero(receiver)
    spatial_norm = np.linalg.norm(delta[:, :2], axis=1)
    ranked = receiver_indices[np.argsort(spatial_norm[receiver_indices])[::-1]]
    nonzero = ranked[spatial_norm[ranked] > np.finfo(float).eps]
    arrow_indices = nonzero[: min(int(n_arrows), len(nonzero))]
    if len(arrow_indices):
        ax.quiver(
            endpoint_coords[arrow_indices, 0],
            endpoint_coords[arrow_indices, 1],
            delta[arrow_indices, 0] * float(arrow_magnification),
            delta[arrow_indices, 1] * float(arrow_magnification),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="#7b1fa2",
            width=0.0042,
            headwidth=3.7,
            headlength=4.8,
            alpha=0.85,
            zorder=3,
        )
    return int(len(arrow_indices))


def _render_reader_main_figure(
    data: MechanismData,
    *,
    start_coords: np.ndarray,
    output: Path,
    geometry_source: str,
    geometry_warning: str | None,
    arrow_magnification: float,
    n_arrows: int,
    dpi: int,
) -> tuple[Path, Path, dict[str, Any]]:
    """Render the compact biology-reader figure; keep the six-panel audit separate."""
    sender = data.cells["fixed_primary_ligand_positive_sender"].to_numpy(dtype=bool)
    receiver = data.cells["fixed_receiver"].to_numpy(dtype=bool)
    endpoint_coords = np.asarray(data.baseline_on[:, :2], dtype=np.float64)
    delta_on = data.counterfactual_on - data.baseline_on
    delta_off = data.counterfactual_off - data.baseline_off
    state_norm_on = np.linalg.norm(delta_on[:, 2:], axis=1)
    state_norm_off = np.linalg.norm(delta_off[:, 2:], axis=1)
    spatial_norm_on = np.linalg.norm(delta_on[:, :2], axis=1)
    spatial_norm_off = np.linalg.norm(delta_off[:, :2], axis=1)

    state_vmax = max(
        float(np.quantile(state_norm_on[receiver], 0.95)),
        float(np.finfo(float).eps),
    )
    state_norm = Normalize(vmin=0.0, vmax=state_vmax, clip=True)

    fig = plt.figure(figsize=(8.5, 7.4), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.28, 1.0, 1.0),
        height_ratios=(1.0, 1.0, 0.46),
        left=0.055,
        right=0.975,
        bottom=0.105,
        top=0.88,
        wspace=0.17,
        hspace=0.32,
    )
    ax_context = fig.add_subplot(grid[:2, 0])
    ax_state_on = fig.add_subplot(grid[0, 1])
    ax_state_off = fig.add_subplot(grid[0, 2])
    ax_spatial_on = fig.add_subplot(grid[1, 1])
    ax_spatial_off = fig.add_subplot(grid[1, 2])
    ax_sham = fig.add_subplot(grid[2, :])

    # A: the biological context in one observed t3 spatial panel.
    _plot_tissue(ax_context, start_coords, background_size=5.5)
    _add_lr_support(
        ax_context,
        coords=start_coords,
        data=data,
        sender=sender,
        receiver=receiver,
    )
    _set_spatial_axis(ax_context, start_coords)
    ax_context.set_title(
        "A  Observed t3 signaling neighborhood",
        loc="left",
        fontsize=9.5,
        fontweight="bold",
        pad=7,
    )
    context_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#d81b60",
            markersize=6,
            label=f"CXCL12a+ senders  n={int(sender.sum())}",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#1565c0",
            markersize=6,
            label=f"CXCR4a+ receivers  n={int(receiver.sum())}",
        ),
        Line2D(
            [0],
            [0],
            color="#ef6c00",
            linewidth=1.6,
            alpha=0.65,
            label=(
                "sender → receiver gated edges  "
                f"n={len(data.target_edges)}"
            ),
        ),
    ]
    ax_context.legend(
        handles=context_handles,
        loc="upper left",
        frameon=False,
        fontsize=7.2,
        handlelength=1.6,
        borderaxespad=0.4,
    )

    # B/C: same fixed receivers, identical coordinates and color scale.
    state_axes = (
        (ax_state_on, state_norm_on, "B  State response - interaction ON"),
        (ax_state_off, state_norm_off, "C  State response - interaction OFF"),
    )
    state_points = None
    for ax, values, title in state_axes:
        _plot_tissue(ax, endpoint_coords, background_size=4.2)
        state_points = ax.scatter(
            endpoint_coords[receiver, 0],
            endpoint_coords[receiver, 1],
            s=25,
            c=values[receiver],
            cmap="magma",
            norm=state_norm,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        _set_spatial_axis(ax, endpoint_coords)
        ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=4)
    assert state_points is not None
    colorbar = fig.colorbar(
        state_points,
        ax=[ax_state_on, ax_state_off],
        orientation="horizontal",
        fraction=0.047,
        pad=0.027,
        aspect=30,
    )
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.set_label(
        "Receiver state response (shared scale; 95% capped)",
        fontsize=6.3,
        labelpad=1,
    )

    # D/E: spatial endpoint changes for exactly the same receiver identities.
    spatial_axes = (
        (
            ax_spatial_on,
            delta_on,
            spatial_norm_on,
            "D  Spatial shift - interaction ON",
        ),
        (
            ax_spatial_off,
            delta_off,
            spatial_norm_off,
            "E  Spatial shift - interaction OFF",
        ),
    )
    arrow_counts: dict[str, int] = {}
    for ax, delta, values, title in spatial_axes:
        _plot_tissue(ax, endpoint_coords, background_size=4.2)
        ax.scatter(
            endpoint_coords[receiver, 0],
            endpoint_coords[receiver, 1],
            s=17,
            c="#1565c0",
            linewidths=0,
            alpha=0.78,
            rasterized=True,
            zorder=2,
        )
        count = _draw_endpoint_arrows(
            ax,
            endpoint_coords=endpoint_coords,
            delta=delta,
            receiver=receiver,
            arrow_magnification=arrow_magnification,
            n_arrows=n_arrows,
        )
        arrow_counts[title] = count
        _set_spatial_axis(ax, endpoint_coords)
        median = float(np.median(values[receiver]))
        ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", pad=4)
        ax.text(
            0.02,
            0.965,
            f"median actual |Δspatial| = {median:.2e}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=5.7,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": "none",
                "alpha": 0.82,
            },
        )
    ax_spatial_on.text(
        0.98,
        0.89,
        f"arrows ×{arrow_magnification:g}",
        transform=ax_spatial_on.transAxes,
        va="top",
        ha="right",
        fontsize=5.7,
        color="#7b1fa2",
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": "none",
            "alpha": 0.82,
        },
    )

    # F: a compact, non-significance-coded placement within the matched-gene range.
    sham = np.asarray(data.sham_state_mediation, dtype=float)
    order = np.argsort(sham, kind="stable")
    y_offsets = np.tile(np.linspace(-0.18, 0.18, 7), int(np.ceil(len(sham) / 7)))[
        : len(sham)
    ]
    y = np.empty_like(y_offsets)
    y[order] = y_offsets
    q_low, q_high = np.quantile(sham, [0.025, 0.975])
    ax_sham.axvspan(q_low, q_high, color="#bdbdbd", alpha=0.18, zorder=0)
    ax_sham.scatter(
        sham,
        y,
        s=17,
        c="#8d8d8d",
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )
    ax_sham.scatter(
        [data.primary_state_mediation],
        [0.30],
        s=64,
        marker="D",
        c="#d81b60",
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )
    x_span = max(float(np.ptp(sham)), float(np.finfo(float).eps))
    x_min = min(float(sham.min()), float(data.primary_state_mediation)) - 0.04 * x_span
    x_max = max(float(sham.max()), float(data.primary_state_mediation)) + 0.18 * x_span
    ax_sham.set_xlim(x_min, x_max)
    ax_sham.set_ylim(-0.31, 0.47)
    ax_sham.set_yticks([])
    ax_sham.tick_params(axis="x", labelsize=6.5)
    ax_sham.set_xlabel(
        "Interaction-mediated receiver-state response",
        fontsize=6.8,
        labelpad=1,
    )
    ax_sham.set_title(
        "F  Matched-gene specificity check",
        loc="left",
        fontsize=9,
        fontweight="bold",
        pad=5,
    )
    ax_sham.text(
        data.primary_state_mediation,
        0.36,
        f"CXCL12a  {100 * data.primary_relative_rank:.1f}th percentile",
        va="bottom",
        ha="center",
        fontsize=6.9,
        color="#333333",
    )
    ax_sham.text(
        0.99,
        0.80,
        "Inside matched-gene range  →  target specificity not established",
        transform=ax_sham.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color="#444444",
        bbox={
            "boxstyle": "round,pad=0.18",
            "fc": "white",
            "ec": "none",
            "alpha": 0.88,
        },
    )
    ax_sham.text(
        q_low,
        -0.27,
        "matched HVGs",
        va="bottom",
        ha="left",
        fontsize=6.5,
        color="#555555",
    )
    for spine in ("top", "right", "left"):
        ax_sham.spines[spine].set_visible(False)

    fig.text(
        0.69,
        0.94,
        f"Same {int(receiver.sum())} fixed receivers, 100% CXCL12a input knockdown",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
    )
    footer = (
        "t3→predicted t4; deterministic same-checkpoint counterfactual. "
        "Interaction OFF is inference-time. Model sensitivity, not an experimental "
        "perturbation or causal mechanism proof."
    )
    if geometry_warning is not None:
        footer += " PREVIEW ONLY: observed t3 geometry was not supplied."
    fig.text(0.055, 0.022, footer, ha="left", va="bottom", fontsize=6.6)

    png_path = output / "cxcl12a_cxcr4a_reader_main.png"
    pdf_path = output / "cxcl12a_cxcr4a_reader_main.pdf"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    contract = {
        "role": "biology_reader_main_figure",
        "geometry_source": geometry_source,
        "same_fixed_receiver_identity": True,
        "state_shared_color_scale": True,
        "state_color_cap_quantile": 0.95,
        "spatial_arrow_magnification": float(arrow_magnification),
        "spatial_arrow_counts": arrow_counts,
        "sham_display": "descriptive matched-HVG strip; no formal p-value",
        "target_specificity_supported": False,
    }
    return png_path, pdf_path, contract


def _save_receiver_effects(
    output: Path,
    data: MechanismData,
    start_coords: np.ndarray,
) -> Path:
    on_delta = data.counterfactual_on - data.baseline_on
    off_delta = data.counterfactual_off - data.baseline_off
    table = data.cells.copy()
    table["anchor_start_x"] = start_coords[:, 0]
    table["anchor_start_y"] = start_coords[:, 1]
    table["baseline_endpoint_x"] = data.baseline_on[:, 0]
    table["baseline_endpoint_y"] = data.baseline_on[:, 1]
    table["ligand_kd_endpoint_x"] = data.counterfactual_on[:, 0]
    table["ligand_kd_endpoint_y"] = data.counterfactual_on[:, 1]
    table["interaction_on_spatial_delta_x"] = on_delta[:, 0]
    table["interaction_on_spatial_delta_y"] = on_delta[:, 1]
    table["interaction_on_spatial_delta_norm"] = np.linalg.norm(
        on_delta[:, :2], axis=1
    )
    table["interaction_on_state_delta_norm"] = np.linalg.norm(
        on_delta[:, 2:], axis=1
    )
    table["interaction_off_spatial_delta_norm"] = np.linalg.norm(
        off_delta[:, :2], axis=1
    )
    table["interaction_off_state_delta_norm"] = np.linalg.norm(
        off_delta[:, 2:], axis=1
    )
    path = output / "receiver_cell_effects.csv.gz"
    table.to_csv(path, index=False, compression="gzip")
    return path


def render_spatial_mechanism(
    data: MechanismData,
    *,
    output_dir: Path,
    h5ad: Path | None = None,
    spatial_key: str = "spatial_aligned",
    arrow_magnification: float = 10.0,
    n_arrows: int = 40,
    dpi: int = 300,
) -> dict[str, Any]:
    """Render the figure, export cell/edge tables, and write an audit manifest."""
    if arrow_magnification <= 0:
        raise ValueError("arrow_magnification must be positive.")
    if n_arrows < 1:
        raise ValueError("n_arrows must be positive.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if h5ad is None:
        start_coords = np.asarray(data.baseline_on[:, :2], dtype=np.float64)
        geometry_source = "baseline_predicted_endpoint_preview"
        geometry_warning = (
            "Observed anchor-start h5ad was not supplied. Panel A/B use baseline "
            "predicted endpoint positions only as a cell-identity preview; this "
            "preview is not a manuscript-ready map of anchor-start edges."
        )
    else:
        start_coords = _observed_start_coordinates(
            Path(h5ad), data.cells, spatial_key=spatial_key
        )
        geometry_source = "observed_anchor_start_spatial_coordinates"
        geometry_warning = None

    sender = data.cells["fixed_primary_ligand_positive_sender"].to_numpy(dtype=bool)
    receiver = data.cells["fixed_receiver"].to_numpy(dtype=bool)
    endpoint_coords = data.baseline_on[:, :2]
    delta_on = data.counterfactual_on - data.baseline_on
    delta_off = data.counterfactual_off - data.baseline_off
    state_norm_on = np.linalg.norm(delta_on[:, 2:], axis=1)
    state_norm_off = np.linalg.norm(delta_off[:, 2:], axis=1)
    spatial_norm_on = np.linalg.norm(delta_on[:, :2], axis=1)
    spatial_norm_off = np.linalg.norm(delta_off[:, :2], axis=1)

    state_vmax = float(np.quantile(state_norm_on[receiver], 0.95))
    state_vmax = max(state_vmax, float(np.finfo(float).eps))
    state_norm = Normalize(vmin=0.0, vmax=state_vmax, clip=True)
    spatial_vmax = float(np.quantile(spatial_norm_on[receiver], 0.95))
    spatial_vmax = max(spatial_vmax, float(np.finfo(float).eps))
    spatial_norm = Normalize(vmin=0.0, vmax=spatial_vmax, clip=True)

    fig = plt.figure(figsize=(17.2, 10.8), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.045,
        right=0.985,
        bottom=0.105,
        top=0.905,
        wspace=0.18,
        hspace=0.24,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, column]) for column in range(3)],
            [fig.add_subplot(grid[1, column]) for column in range(3)],
        ]
    )

    # A: where the fixed sender and receiver cells are.
    ax = axes[0, 0]
    _plot_tissue(ax, start_coords)
    ax.scatter(
        start_coords[sender, 0],
        start_coords[sender, 1],
        s=15,
        c="#d81b60",
        linewidths=0,
        alpha=0.9,
        label=f"CXCL12a+ fixed senders (n={sender.sum()})",
        rasterized=True,
    )
    ax.scatter(
        start_coords[receiver, 0],
        start_coords[receiver, 1],
        s=18,
        c="#1565c0",
        linewidths=0,
        alpha=0.9,
        label=f"CXCR4a+ / CXCL12a- fixed receivers (n={receiver.sum()})",
        rasterized=True,
    )
    ax.legend(loc="upper right", frameon=True, fontsize=8, markerscale=1.4)
    composition = (
        "Top sender annotations\n"
        + _cohort_composition_text(data.cells, sender)
        + "\n\nTop receiver annotations\n"
        + _cohort_composition_text(data.cells, receiver)
    )
    ax.text(
        0.015,
        0.015,
        composition,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.4,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#aaaaaa", "alpha": 0.9},
    )
    _set_spatial_axis(ax, start_coords)
    ax.set_title(
        "A  Where are the fixed sender and receiver cells?\n"
        + (
            "Observed t3 aligned spatial coordinates"
            if h5ad is not None
            else "Preview at baseline-predicted t4 geometry"
        ),
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    # B: exact baseline gated edges on the fixed LR-expression support.
    ax = axes[0, 1]
    _plot_tissue(ax, start_coords)
    edge_source = data.target_edges["source_index"].to_numpy(dtype=int)
    edge_target = data.target_edges["target_index"].to_numpy(dtype=int)
    segments = np.stack((start_coords[edge_source], start_coords[edge_target]), axis=1)
    strength = data.target_edges["complete_message_norm_joint"].to_numpy(dtype=float)
    strength_scale = np.quantile(strength, 0.9)
    strength_scale = max(float(strength_scale), float(np.finfo(float).eps))
    line_widths = 0.25 + 1.8 * np.clip(strength / strength_scale, 0.0, 1.0)
    collection = LineCollection(
        segments,
        colors="#ef6c00",
        linewidths=line_widths,
        alpha=0.42,
        rasterized=True,
        zorder=1,
    )
    ax.add_collection(collection)
    ax.scatter(
        start_coords[sender, 0],
        start_coords[sender, 1],
        s=11,
        c="#d81b60",
        linewidths=0,
        alpha=0.85,
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        start_coords[receiver, 0],
        start_coords[receiver, 1],
        s=13,
        c="#1565c0",
        linewidths=0,
        alpha=0.85,
        rasterized=True,
        zorder=2,
    )
    _set_spatial_axis(ax, start_coords)
    ax.set_title(
        "B  Baseline interaction support\n"
        f"{len(data.target_edges)} gated GNN edges on CXCL12a/CXCR4a-compatible support",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.015,
        0.015,
        "Orange = generic complete GNN message support\n"
        "(expression-compatible, not an LR-specific message component)",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#aaaaaa", "alpha": 0.9},
    )

    # C/D: same receiver identities, interaction on versus off.
    for ax, values, label, panel in (
        (
            axes[0, 2],
            state_norm_on,
            "Interaction ON",
            "C",
        ),
        (
            axes[1, 0],
            state_norm_off,
            "Same checkpoint, interaction OFF",
            "D",
        ),
    ):
        _plot_tissue(ax, endpoint_coords)
        points = ax.scatter(
            endpoint_coords[receiver, 0],
            endpoint_coords[receiver, 1],
            s=24,
            c=values[receiver],
            cmap="magma",
            norm=state_norm,
            linewidths=0.15,
            edgecolors="white",
            rasterized=True,
        )
        _set_spatial_axis(ax, endpoint_coords)
        ax.set_title(
            f"{panel}  Receiver-state response after 100% CXCL12a input KD\n{label}",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        colorbar = fig.colorbar(points, ax=ax, fraction=0.035, pad=0.01)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(
            r"$\|\Delta$ PCA state$\|$ (95% cap)",
            fontsize=7.5,
        )

    # E: where spatial endpoint changes occur; arrows are magnified and subset.
    ax = axes[1, 1]
    _plot_tissue(ax, endpoint_coords)
    receiver_indices = np.flatnonzero(receiver)
    ranked = receiver_indices[np.argsort(spatial_norm_on[receiver_indices])[::-1]]
    arrow_indices = ranked[: min(int(n_arrows), len(ranked))]
    points = ax.scatter(
        endpoint_coords[receiver, 0],
        endpoint_coords[receiver, 1],
        s=22,
        c=spatial_norm_on[receiver],
        cmap="viridis",
        norm=spatial_norm,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.quiver(
        endpoint_coords[arrow_indices, 0],
        endpoint_coords[arrow_indices, 1],
        delta_on[arrow_indices, 0] * float(arrow_magnification),
        delta_on[arrow_indices, 1] * float(arrow_magnification),
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color="#7b1fa2",
        width=0.004,
        headwidth=3.5,
        headlength=4.5,
        alpha=0.8,
        zorder=3,
    )
    _set_spatial_axis(ax, endpoint_coords)
    colorbar = fig.colorbar(points, ax=ax, fraction=0.035, pad=0.01)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.set_label(
        r"actual $\|\Delta$ spatial$\|$ (95% cap)",
        fontsize=7.5,
    )
    paired = ax.inset_axes([0.65, 0.68, 0.32, 0.28])
    paired.boxplot(
        [spatial_norm_on[receiver], spatial_norm_off[receiver]],
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        boxprops={"facecolor": "#b39ddb", "edgecolor": "#5e35b1"},
        medianprops={"color": "#212121", "linewidth": 1.4},
        whiskerprops={"color": "#5e35b1"},
        capprops={"color": "#5e35b1"},
    )
    paired.set_xticks([1, 2], ["ON", "OFF"])
    paired.set_yscale("symlog", linthresh=1e-6)
    paired.tick_params(labelsize=6.5, length=2)
    paired.set_title("same fixed receivers\nspatial endpoint |Δ|", fontsize=6.8)
    paired.spines["top"].set_visible(False)
    paired.spines["right"].set_visible(False)
    ax.set_title(
        "E  Where does the predicted endpoint move?\n"
        f"Same receivers; top {len(arrow_indices)} arrows shown at x{arrow_magnification:g}",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    # F: honest matched-sham positioning of the primary target.
    ax = axes[1, 2]
    values = np.asarray(data.sham_state_mediation, dtype=float)
    bins = min(16, max(6, int(np.sqrt(len(values)) * 1.4)))
    ax.hist(
        values,
        bins=bins,
        color="#bdbdbd",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
        label=f"{len(values)} matched-HVG shams",
    )
    ax.axvline(
        data.primary_state_mediation,
        color="#d81b60",
        linewidth=2.5,
        label="CXCL12a primary",
    )
    q_low, q_high = np.quantile(values, [0.025, 0.975])
    ax.axvspan(q_low, q_high, color="#616161", alpha=0.08)
    ax.set_xlabel(
        "Interaction-mediated receiver-state centroid change",
        fontsize=9,
    )
    ax.set_ylabel("Matched-HVG count", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title(
        "F  Is the CXCL12a response target-specific?\n"
        f"Primary rank = {100 * data.primary_relative_rank:.1f}th percentile (within sham range)",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.97,
        "Interaction-mediated model sensitivity is present,\n"
        "but the primary remains inside the sham range.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#aaaaaa", "alpha": 0.92},
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        "CXCL12a -> CXCR4a computational perturbation: where the trained model changes",
        fontsize=17,
        fontweight="bold",
        x=0.045,
        ha="left",
    )
    footer = (
        "Fixed cohorts were defined at t3 and followed to predicted t4 with the same "
        "trained checkpoint (deterministic rollout, sigma=0). Interaction-off is an "
        "inference-time matched-model control. This is exploratory trained-model "
        "sensitivity, not an experimental perturbation, mechanism proof, or causal claim."
    )
    if geometry_warning is not None:
        footer += " PREVIEW WARNING: " + geometry_warning
    fig.text(0.045, 0.025, footer, ha="left", va="bottom", fontsize=8.5, wrap=True)

    png_path = output / "cxcl12a_cxcr4a_spatial_mechanism.png"
    pdf_path = output / "cxcl12a_cxcr4a_spatial_mechanism.pdf"
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    reader_png_path, reader_pdf_path, reader_visual_contract = (
        _render_reader_main_figure(
            data,
            start_coords=start_coords,
            output=output,
            geometry_source=geometry_source,
            geometry_warning=geometry_warning,
            arrow_magnification=arrow_magnification,
            n_arrows=n_arrows,
            dpi=dpi,
        )
    )

    effects_path = _save_receiver_effects(output, data, start_coords)
    edges_path = output / "baseline_lr_compatible_edges.csv.gz"
    data.target_edges.to_csv(edges_path, index=False, compression="gzip")

    receiver_on_state = state_norm_on[receiver]
    receiver_off_state = state_norm_off[receiver]
    receiver_on_spatial = spatial_norm_on[receiver]
    receiver_off_spatial = spatial_norm_off[receiver]
    manifest: dict[str, Any] = {
        "analysis": "cxcl12a_cxcr4a_biology_first_spatial_mechanism_renderer",
        "input_semantics": {
            "anchor_id": data.anchor_id,
            "grouping_seed": data.grouping_seed,
            "knockdown_fraction": data.knockdown_fraction,
            "fixed_sender_definition": "baseline cxcl12a-positive cells",
            "fixed_receiver_definition": (
                "baseline cxcr4a-positive and cxcl12a-negative cells"
            ),
            "interaction_off": "same trained checkpoint; interaction term disabled at inference",
            "geometry_source": geometry_source,
            "geometry_warning": geometry_warning,
        },
        "counts": {
            "cells": int(len(data.cells)),
            "fixed_senders": int(sender.sum()),
            "fixed_receivers": int(receiver.sum()),
            "baseline_lr_compatible_gated_edges": int(len(data.target_edges)),
            "matched_hvg_shams": int(len(data.sham_state_mediation)),
        },
        "direct_effect_summary": {
            "receiver_state_delta_norm_interaction_on_median": float(
                np.median(receiver_on_state)
            ),
            "receiver_state_delta_norm_interaction_off_median": float(
                np.median(receiver_off_state)
            ),
            "receiver_spatial_delta_norm_interaction_on_median": float(
                np.median(receiver_on_spatial)
            ),
            "receiver_spatial_delta_norm_interaction_off_median": float(
                np.median(receiver_off_spatial)
            ),
            "interaction_mediated_receiver_state_centroid_norm": float(
                data.primary_state_mediation
            ),
            "primary_descriptive_relative_rank_among_matched_shams": float(
                data.primary_relative_rank
            ),
            "primary_within_sham_central_95_percent": bool(
                np.quantile(data.sham_state_mediation, 0.025)
                <= data.primary_state_mediation
                <= np.quantile(data.sham_state_mediation, 0.975)
            ),
        },
        "visual_contract": {
            "state_color_cap_quantile": 0.95,
            "spatial_color_cap_quantile": 0.95,
            "spatial_arrow_magnification": float(arrow_magnification),
            "spatial_arrows_show_largest_receiver_effects": int(
                min(n_arrows, receiver.sum())
            ),
            "reader_main": reader_visual_contract,
        },
        "claim_bounds": {
            "trained_model_sensitivity": True,
            "post_hoc_exploratory_descriptive": True,
            "experimental_perturbation": False,
            "experimental_causality": False,
            "biological_mechanism_proven": False,
            "target_specificity_supported_by_matched_shams": False,
        },
        "artifacts": {},
    }
    guide_path = output / "FIGURE_GUIDE_CN.md"
    guide_path.write_text(
        "# CXCL12a→CXCR4a 空间机制图怎么读\n\n"
        "## 主图：`cxcl12a_cxcr4a_reader_main`\n\n"
        "主图按生物学问题从左到右读，不需要先理解指标：\n\n"
        "- **A** 把 observed t3 的 CXCL12a 阳性 sender、CXCR4a 阳性 receiver "
        "和模型实际保留的表达相容 gated edge 放在同一张组织空间图上，回答“哪里、"
        "谁对谁”。橙线只是落在 CXCL12a/CXCR4a 表达相容支持上的 generic GNN "
        "edge，不能称为配体受体特异 message。\n"
        "- **B/C** 在同一批固定 receiver 上对比 100% CXCL12a 输入 knockdown："
        "interaction ON 时出现局部 state response，使用同一 checkpoint 只在推理时"
        "关掉 interaction 后响应消失。两图使用完全相同的颜色范围和空间位置。\n"
        "- **D/E** 显示同一批 receiver 的 predicted spatial endpoint shift。紫色"
        f"箭头为最大的非零位移并放大 {arrow_magnification:g} 倍；图内同时写出未放大"
        "的实际中位位移。interaction OFF 时没有可见位移箭头。\n"
        "- **F** 把 CXCL12a 直接放进 100 个 matched-HVG sham 的分布中。它位于约 "
        f"{100 * data.primary_relative_rank:.1f} 百分位且仍在 matched-gene 范围内，"
        "因此主图明确区分“interaction-dependent model sensitivity”和“target-specific "
        "biological evidence”；后者目前不成立。\n\n"
        "## 详细补充图：`cxcl12a_cxcr4a_spatial_mechanism`\n\n"
        "六联详细图保留颜色刻度、箭头筛选、receiver-level effect 和 sham histogram，"
        "用于方法审计与 supplement。它与主图使用同一份固定 cohort、同一组 endpoint "
        "array 和 observed t3 geometry。\n\n"
        "## 六联详细图逐面板说明\n\n"
        "- **A**：先回答“谁在给谁发信号、它们在胚胎哪里”。粉色是基线 "
        "CXCL12a 阳性的固定 sender，蓝色是 CXCR4a 阳性且 CXCL12a 阴性的固定 receiver。\n"
        "- **B**：橙线是模型在基线时真正保留的、落在上述表达相容 sender→receiver "
        "支持上的 GNN 边。它是 generic complete GNN message 的支持，不能叫作 "
        "CXCL12a→CXCR4a 特异 message。\n"
        "- **C**：把 CXCL12a 输入完全敲低后，同一批 receiver 的 PCA-state 改变量在 "
        "predicted t4 的空间位置上着色，显示模型响应发生在哪里。\n"
        "- **D**：使用同一个训练模型、只在推理时关掉 interaction 项。当前正式 bundle "
        "中该改变量降为零，说明 C 中的数值响应由 interaction 通路传递，而不是重新训练差异。\n"
        "- **E**：同一批 receiver 的 predicted spatial endpoint 位移。颜色是实际位移量；"
        f"为了看清方向，只画最大的 {min(n_arrows, int(receiver.sum()))} 个箭头，箭头放大 "
        f"{arrow_magnification:g} 倍，不能把箭头长度当作原始坐标尺度。\n"
        "- **F**：关键限制。粉线是 CXCL12a，灰色是 100 个按表达和 PCA loading "
        f"匹配的 HVG sham。CXCL12a 位于约 {100 * data.primary_relative_rank:.1f} "
        "百分位且仍在 sham 范围内。因此可说“模型存在 interaction-mediated sensitivity”，"
        "但不能说“已经证明 CXCL12a 靶点特异性或真实生物学因果机制”。\n\n"
        "这张图把空间位置、边、同一 receiver 的 state/spatial 变化以及 same-model "
        "interaction-off 对照放在一起；所有限制都保留在图内，不用指标替代生物学叙事。\n",
        encoding="utf-8",
    )

    artifact_paths = (
        reader_png_path,
        reader_pdf_path,
        png_path,
        pdf_path,
        effects_path,
        edges_path,
        guide_path,
    )
    manifest["artifacts"] = {
        path.name: {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in artifact_paths
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = _parser().parse_args()
    data = load_mechanism_data(
        args.bundle_dir,
        anchor_id=args.anchor_id,
        grouping_seed=args.grouping_seed,
        knockdown_fraction=args.knockdown_fraction,
    )
    manifest = render_spatial_mechanism(
        data,
        output_dir=args.output_dir,
        h5ad=args.h5ad,
        spatial_key=args.spatial_key,
        arrow_magnification=args.arrow_magnification,
        n_arrows=args.n_arrows,
        dpi=args.dpi,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
