"""Render corrected MOSTA velocity/communication in the archived notebook style.

This module deliberately separates scientific inputs from presentation.  It
does not load a model, simulate cells, recompute a velocity component, or
invent missing intrinsic/cell-self-edge terms.  The renderer accepts an audited
two-dimensional velocity field and a complete observed background, then
reuses the visual grammar of
``mosta_velocity_communication_focus_brain_t3.ipynb``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LegacyMostaVelocityCommunicationStyle:
    """Frozen style constants from the archived MOSTA notebook."""

    figsize: tuple[float, float] = (16.0, 20.0)
    density: float = 2.0
    background_alpha: float = 0.35
    full_background_alpha: float = 0.20
    point_size: float = 50.0
    smooth: float = 0.8
    min_mass: float = 1.0
    cutoff_perc: float = 3.0
    streamline_width: float = 1.5
    streamline_arrow_size: float = 1.2
    title_fontsize: float = 20.0
    title_pad: float = 20.0
    communication_node_size: float = 2000.0
    communication_node_edge_width: float = 2.5
    communication_arrow_alpha: float = 0.9
    communication_arrow_curvature: float = 0.15
    communication_arrow_mutation_scale: float = 20.0
    communication_arrow_min_width: float = 1.0
    communication_arrow_width_range: float = 10.0
    centroid_top_n_y: int = 200


LEGACY_FIG4D_STYLE = LegacyMostaVelocityCommunicationStyle()
LEGACY_FOCUS_LABEL = "Brain"
LEGACY_EDGE_TOP_K = 3
LEGACY_CENTROID_TOP_Y_EXCLUSIONS = (
    "Brain",
    "Meninges",
    "Choroid plexus",
)


def validate_palette(
    labels: Sequence[object], label_to_color: Mapping[str, str]
) -> dict[str, str]:
    """Return a complete, valid palette for the displayed annotations."""

    from matplotlib.colors import is_color_like

    categories = sorted(set(map(str, labels)))
    missing = [label for label in categories if label not in label_to_color]
    if missing:
        raise ValueError(
            "label-color JSON is missing displayed Annotation values: " f"{missing}."
        )
    invalid = [
        label for label in categories if not is_color_like(label_to_color[label])
    ]
    if invalid:
        raise ValueError(f"Invalid colors for Annotation values: {invalid}.")
    return {label: str(label_to_color[label]) for label in categories}


def select_brain_focus_top_edges(
    communication: pd.DataFrame,
    *,
    focus_label: str = LEGACY_FOCUS_LABEL,
    top_k: int = LEGACY_EDGE_TOP_K,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select legacy top-K non-diagonal edges plus the saved Brain type loop.

    The archived notebook retained positive directed edges incident to Brain,
    ranked the non-diagonal entries by ``M_per_source``, kept the first three,
    and then appended the Brain-to-Brain type-level diagonal.  In the corrected
    API, cell ``i -> i`` edges were removed before type aggregation; a positive
    Brain-to-Brain row therefore represents communication between *different*
    Brain cells and is not a fabricated cell self-edge.
    """

    required = {"source", "target", "is_self_edge", "weight_per_source"}
    missing = sorted(required - set(communication.columns))
    if missing:
        raise KeyError(f"Communication CSV is missing columns: {missing}.")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive.")
    source = communication["source"].astype(str)
    target = communication["target"].astype(str)
    weights = communication["weight_per_source"].to_numpy(dtype=float)
    same_type = communication["is_self_edge"].astype(bool).to_numpy()
    label_diagonal = (source == target).to_numpy()
    if not np.array_equal(same_type, label_diagonal):
        raise ValueError(
            "is_self_edge must encode the type-level source==target diagonal; "
            "it does not describe upstream cell i->i edges."
        )
    focus = str(focus_label)
    candidates = communication.loc[
        ((source == focus) | (target == focus)).to_numpy()
        & ~same_type
        & np.isfinite(weights)
        & (weights > 0.0)
    ].copy()
    candidates = candidates.sort_values(
        ["weight_per_source", "source", "target"],
        ascending=[False, True, True],
        kind="stable",
    )
    if candidates.empty:
        raise ValueError(
            f"No positive non-same-type communication edge touches {focus!r}."
        )
    nonself_kept = candidates.head(int(top_k)).copy().reset_index(drop=True)
    if len(nonself_kept) != int(top_k):
        raise ValueError(
            f"Expected {int(top_k)} positive non-same-type Brain edges; "
            f"found {len(nonself_kept)}."
        )
    focus_loop_candidates = communication.loc[
        (source == focus).to_numpy()
        & (target == focus).to_numpy()
        & same_type
        & np.isfinite(weights)
        & (weights > 0.0)
    ].copy()
    if len(focus_loop_candidates) != 1:
        raise ValueError(
            "Expected exactly one positive saved Brain-to-Brain type-level row; "
            f"found {len(focus_loop_candidates)}."
        )
    nonself_kept["legacy_display_role"] = "non_same_type_top_k"
    nonself_kept["nonself_rank"] = np.arange(1, len(nonself_kept) + 1)
    focus_loop = focus_loop_candidates.copy().reset_index(drop=True)
    focus_loop["legacy_display_role"] = "focus_same_type_loop"
    focus_loop["nonself_rank"] = pd.Series([pd.NA], dtype="Int64")
    kept = pd.concat([nonself_kept, focus_loop], ignore_index=True)
    kept["nonself_rank"] = kept["nonself_rank"].astype("Int64")
    kept.insert(0, "draw_order", np.arange(1, len(kept) + 1))
    # Preserve the historical column name for existing consumers while making
    # its meaning explicit through draw_order and legacy_display_role.
    kept.insert(1, "legacy_display_rank", np.arange(1, len(kept) + 1))
    candidate_weight = float(candidates["weight_per_source"].sum())
    nonself_kept_weight = float(nonself_kept["weight_per_source"].sum())
    same_type_rows = communication.loc[same_type]
    positive_same_type_rows = same_type_rows.loc[
        np.isfinite(same_type_rows["weight_per_source"].to_numpy(dtype=float))
        & (same_type_rows["weight_per_source"].to_numpy(dtype=float) > 0.0)
    ]
    audit = {
        "focus_label": focus,
        "top_k": int(top_k),
        "matrix": "M_per_source / weight_per_source",
        "candidate_rule": (
            "finite positive directed non-same-type edge with source==Brain or "
            "target==Brain"
        ),
        "focus_loop_rule": (
            "append the unique finite positive saved Brain-to-Brain type-level "
            "diagonal after the top-K non-same-type edges"
        ),
        "ordering": "descending weight_per_source; lexical source/target ties",
        "n_candidates": int(len(candidates)),
        "n_kept": int(len(kept)),
        "n_non_same_type_kept": int(len(nonself_kept)),
        "n_focus_same_type_loops_kept": 1,
        "candidate_weight_sum": candidate_weight,
        "non_same_type_kept_weight_sum": nonself_kept_weight,
        "displayed_weight_sum": float(kept["weight_per_source"].sum()),
        "kept_candidate_weight_fraction": (
            float(nonself_kept_weight / candidate_weight)
            if candidate_weight
            else 0.0
        ),
        "saved_positive_same_type_rows": int(len(positive_same_type_rows)),
        "focus_same_type_loop_weight": float(
            focus_loop["weight_per_source"].iloc[0]
        ),
        "cell_self_edges_fabricated": 0,
        "same_type_loops_drawn": 1,
        "edge_semantics": (
            "is_self_edge in this flattened table means source label equals "
            "target label; upstream cell i->i edges remain removed"
        ),
    }
    return kept, audit


def compute_legacy_communication_centroids(
    coordinates: np.ndarray,
    labels: Sequence[object],
    node_labels: Sequence[object],
    *,
    top_n_y: int = LEGACY_FIG4D_STYLE.centroid_top_n_y,
    top_n_y_exclusions: Sequence[str] = LEGACY_CENTROID_TOP_Y_EXCLUSIONS,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Compute notebook centroids, including its optional top-Y placement."""

    coords = np.asarray(coordinates, dtype=float)
    values = np.asarray(labels).astype(str)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != len(values):
        raise ValueError("Coordinates and labels must align with shape (n, 2).")
    exclusions = set(map(str, top_n_y_exclusions))
    centroids: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for label in sorted(set(map(str, node_labels))):
        subset = coords[values == label]
        subset = subset[np.isfinite(subset).all(axis=1)]
        if len(subset) == 0:
            raise ValueError(
                f"No finite observed coordinates exist for node {label!r}."
            )
        use_top_y = label not in exclusions and int(top_n_y) > 0
        if use_top_y and len(subset) > int(top_n_y):
            order = np.argsort(subset[:, 1], kind="stable")[-int(top_n_y) :]
            centroid_rows = subset[order]
        else:
            centroid_rows = subset
        centroid = centroid_rows.mean(axis=0)
        centroids[label] = centroid
        rows.append(
            {
                "node": label,
                "n_available": int(len(subset)),
                "n_used": int(len(centroid_rows)),
                "top_n_y_rule_applied": bool(use_top_y and len(subset) > int(top_n_y)),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
            }
        )
    return centroids, pd.DataFrame(rows)


def _remove_y_outliers(
    coordinates: np.ndarray,
    velocity: np.ndarray,
    labels: np.ndarray,
    *,
    reference_coordinates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Apply the notebook's 1.5-IQR Y filter using an explicit reference."""

    reference = (
        coordinates
        if reference_coordinates is None
        else np.asarray(reference_coordinates, dtype=float)
    )
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError("reference_coordinates must have shape (n, 2).")
    reference = reference[np.isfinite(reference).all(axis=1)]
    if len(reference) < 3:
        raise ValueError("Y-IQR reference has fewer than three finite coordinates.")
    reference_y = reference[:, 1]
    q1, q3 = np.percentile(reference_y, [25, 75])
    iqr = float(q3 - q1)
    lower = float(q1 - 1.5 * iqr)
    upper = float(q3 + 1.5 * iqr)
    coordinate_y = coordinates[:, 1]
    mask = (coordinate_y >= lower) & (coordinate_y <= upper)
    return coordinates[mask], velocity[mask], labels[mask], (lower, upper)


def render_legacy_velocity_communication_panel(
    *,
    velocity_coordinates: np.ndarray,
    velocity_vectors: np.ndarray,
    velocity_labels: Sequence[object],
    full_background_coordinates: np.ndarray,
    full_background_labels: Sequence[object],
    kept_edges: pd.DataFrame,
    label_to_color: Mapping[str, str],
    title: str | None,
    out_paths: Sequence[str | Path],
    style: LegacyMostaVelocityCommunicationStyle = LEGACY_FIG4D_STYLE,
    legend_loc: str | None = "right margin",
    y_iqr_reference: str = "velocity",
    full_background_point_size: float | None = None,
    full_background_alpha: float | None = None,
    velocity_point_size: float | None = None,
    axis_decorator: Callable[
        [Any, Any, Mapping[str, np.ndarray], Mapping[str, str]],
        Mapping[str, Any] | None,
    ]
    | None = None,
) -> dict[str, Any]:
    """Render one saved velocity component with legacy notebook presentation.

    ``legend_loc`` and ``axis_decorator`` are presentation-only hooks.  They
    allow a manuscript compositor to remove the raw notebook legend/title and
    add annotations without changing the validated velocity field,
    communication edges, centroids, or source-color encoding.
    """

    import anndata as ad
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import scvelo as scv

    coordinates = np.asarray(velocity_coordinates, dtype=np.float32)
    velocity = np.asarray(velocity_vectors, dtype=np.float32)
    labels = np.asarray(velocity_labels).astype(str)
    background = np.asarray(full_background_coordinates, dtype=np.float32)
    background_labels = np.asarray(full_background_labels).astype(str)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("velocity_coordinates must have shape (n, 2).")
    if velocity.shape != coordinates.shape or len(labels) != len(coordinates):
        raise ValueError("Velocity vectors and labels must align with coordinates.")
    if background.ndim != 2 or background.shape[1] != 2:
        raise ValueError("full_background_coordinates must have shape (n, 2).")
    if len(background_labels) != len(background):
        raise ValueError("Full-background labels do not align with coordinates.")
    finite = (
        np.isfinite(coordinates).all(axis=1)
        & np.isfinite(velocity).all(axis=1)
        & (np.linalg.norm(velocity, axis=1) > 1e-12)
    )
    if int(finite.sum()) < 3:
        raise ValueError("Velocity field has fewer than three finite nonzero vectors.")
    coordinates = coordinates[finite]
    velocity = velocity[finite]
    labels = labels[finite]
    if y_iqr_reference not in {"velocity", "full_background"}:
        raise ValueError("y_iqr_reference must be 'velocity' or 'full_background'.")
    reference_coordinates = background if y_iqr_reference == "full_background" else None
    coordinates, velocity, labels, y_bounds = _remove_y_outliers(
        coordinates,
        velocity,
        labels,
        reference_coordinates=reference_coordinates,
    )
    if len(coordinates) < 3:
        raise ValueError("Notebook Y-outlier filtering removed the velocity cohort.")

    palette = validate_palette(background_labels, label_to_color)
    missing_compute = sorted(set(labels) - set(palette))
    if missing_compute:
        raise ValueError(
            f"Velocity labels are missing from background palette: {missing_compute}."
        )
    categories = sorted(palette)
    categorical = pd.Categorical(labels, categories=categories)
    velocity_adata = ad.AnnData(X=np.zeros((len(coordinates), 1), dtype=np.float32))
    velocity_adata.obsm["X_spatial"] = coordinates
    velocity_adata.obsm["velocity_interaction_spatial"] = velocity
    velocity_adata.obs["Annotation"] = categorical

    centroids, centroid_table = compute_legacy_communication_centroids(
        background,
        background_labels,
        set(kept_edges["source"].astype(str)) | set(kept_edges["target"].astype(str)),
        top_n_y=style.centroid_top_n_y,
        top_n_y_exclusions=LEGACY_CENTROID_TOP_Y_EXCLUSIONS,
    )

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=style.figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # The corrected velocity artifact contains the audited 8k compute cohort,
    # while the aligned H5AD contains the complete observed E15.5 background.
    # Draw the latter as label-only context; no velocity is imputed to it.
    background_finite = np.isfinite(background).all(axis=1)
    background_in_y = (
        background_finite
        & (background[:, 1] >= y_bounds[0])
        & (background[:, 1] <= y_bounds[1])
    )
    resolved_background_point_size = (
        style.point_size
        if full_background_point_size is None
        else float(full_background_point_size)
    )
    resolved_background_alpha = (
        style.full_background_alpha
        if full_background_alpha is None
        else float(full_background_alpha)
    )
    resolved_velocity_point_size = (
        style.point_size if velocity_point_size is None else float(velocity_point_size)
    )
    if min(resolved_background_point_size, resolved_velocity_point_size) < 0:
        raise ValueError("Point sizes must be nonnegative.")
    if not 0.0 <= resolved_background_alpha <= 1.0:
        raise ValueError("full_background_alpha must be within [0, 1].")
    for label in categories:
        mask = background_in_y & (background_labels == label)
        if np.any(mask):
            ax.scatter(
                background[mask, 0],
                background[mask, 1],
                c=palette[label],
                s=resolved_background_point_size,
                alpha=resolved_background_alpha,
                linewidths=0,
                rasterized=True,
                zorder=0,
            )

    palette_sequence = [palette[label] for label in categories]
    stream_n_neighbors = max(1, min(30, int(len(coordinates)) - 1))
    scv.pl.velocity_embedding_stream(
        velocity_adata,
        basis="spatial",
        vkey="velocity_interaction",
        color="Annotation",
        palette=palette_sequence,
        ax=ax,
        show=False,
        density=style.density,
        smooth=style.smooth,
        min_mass=style.min_mass,
        cutoff_perc=style.cutoff_perc,
        linewidth=style.streamline_width,
        arrow_size=style.streamline_arrow_size,
        n_neighbors=stream_n_neighbors,
        alpha=style.background_alpha,
        size=resolved_velocity_point_size,
        legend_loc="none" if legend_loc is None else str(legend_loc),
        title="",
        frameon=False,
    )

    if kept_edges.empty:
        raise ValueError("No communication edges were supplied for display.")
    max_weight = float(kept_edges["weight_per_source"].max())
    if not np.isfinite(max_weight) or max_weight <= 0:
        raise ValueError("Displayed communication weights must be positive and finite.")
    drawn_nodes: set[str] = set()
    arrow_records: list[dict[str, Any]] = []
    for row in kept_edges.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)
        weight = float(row.weight_per_source)
        p1 = centroids[source]
        p2 = centroids[target]
        width = style.communication_arrow_min_width + (
            style.communication_arrow_width_range * np.sqrt(weight / max_weight)
        )
        is_same_type_loop = source == target
        if is_same_type_loop:
            arrow_start = (float(p1[0] - 0.03), float(p1[1] + 0.05))
            arrow_end = (float(p1[0] + 0.06), float(p1[1] + 0.02))
            connectionstyle = "arc3,rad=-10.0"
            mutation_scale = 8.0
        else:
            arrow_start = (float(p1[0]), float(p1[1]))
            arrow_end = (float(p2[0]), float(p2[1]))
            connectionstyle = f"arc3,rad={style.communication_arrow_curvature}"
            mutation_scale = style.communication_arrow_mutation_scale
        arrow = mpatches.FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle="-|>,head_length=0.8,head_width=0.5",
            mutation_scale=mutation_scale,
            connectionstyle=connectionstyle,
            color=palette[source],
            linewidth=width,
            alpha=style.communication_arrow_alpha,
            zorder=30,
        )
        ax.add_patch(arrow)
        arrow_records.append(
            {
                "source": source,
                "target": target,
                "legacy_display_role": str(
                    getattr(row, "legacy_display_role", "unspecified")
                ),
                "draw_order": int(getattr(row, "draw_order", len(arrow_records) + 1)),
                "weight_per_source": weight,
                "linewidth": float(width),
                "start": [float(arrow_start[0]), float(arrow_start[1])],
                "end": [float(arrow_end[0]), float(arrow_end[1])],
                "connectionstyle": connectionstyle,
                "mutation_scale": float(mutation_scale),
                "cell_self_edge_fabricated": False,
                "same_type_loop": bool(is_same_type_loop),
            }
        )
        for node, position in ((source, p1), (target, p2)):
            if node in drawn_nodes:
                continue
            ax.scatter(
                position[0],
                position[1],
                s=style.communication_node_size,
                c=palette[node],
                edgecolors="white",
                linewidth=style.communication_node_edge_width,
                zorder=31,
            )
            drawn_nodes.add(node)

    if title:
        ax.set_title(
            str(title),
            fontsize=style.title_fontsize,
            fontweight="bold",
            color="black",
            pad=style.title_pad,
        )
    else:
        ax.set_title("")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(colors="black", labelsize=12)

    decoration_audit: dict[str, Any] = {}
    if axis_decorator is not None:
        decoration = axis_decorator(fig, ax, centroids, palette)
        if decoration is not None:
            if not isinstance(decoration, Mapping):
                raise TypeError("axis_decorator must return a mapping or None.")
            decoration_audit = dict(decoration)

    written: list[str] = []
    for value in out_paths:
        path = Path(value).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".svg":
            fig.savefig(path, format="svg", bbox_inches="tight")
        elif path.suffix.lower() == ".png":
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        elif path.suffix.lower() == ".pdf":
            fig.savefig(path, format="pdf", bbox_inches="tight", facecolor="white")
        else:
            raise ValueError(f"Unsupported output format: {path.suffix!r}.")
        written.append(str(path))
    plt.close(fig)
    return {
        "style": asdict(style),
        "n_velocity_cells_input": int(len(velocity_coordinates)),
        "n_velocity_cells_rendered": int(len(coordinates)),
        "n_full_background_cells": int(len(background)),
        "n_full_background_cells_rendered": int(background_in_y.sum()),
        "velocity_y_iqr_bounds": [float(y_bounds[0]), float(y_bounds[1])],
        "velocity_y_iqr_reference": y_iqr_reference,
        "resolved_point_style": {
            "full_background_point_size": resolved_background_point_size,
            "full_background_alpha": resolved_background_alpha,
            "velocity_compute_point_size": resolved_velocity_point_size,
        },
        "n_annotation_categories": int(len(categories)),
        "annotation_categories": categories,
        "n_communication_edges": int(len(kept_edges)),
        "communication_arrows": arrow_records,
        "communication_nodes": sorted(drawn_nodes),
        "centroids": centroid_table.to_dict(orient="records"),
        "title": None if title is None else str(title),
        "legend_loc": "none" if legend_loc is None else str(legend_loc),
        "axis_decoration": decoration_audit,
        "outputs": written,
    }


__all__ = [
    "LEGACY_CENTROID_TOP_Y_EXCLUSIONS",
    "LEGACY_EDGE_TOP_K",
    "LEGACY_FIG4D_STYLE",
    "LEGACY_FOCUS_LABEL",
    "LegacyMostaVelocityCommunicationStyle",
    "compute_legacy_communication_centroids",
    "render_legacy_velocity_communication_panel",
    "select_brain_focus_top_edges",
    "validate_palette",
]
