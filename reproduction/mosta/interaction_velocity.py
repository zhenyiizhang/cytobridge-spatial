"""Draw interaction-induced gene velocity and communication in Figure 4d."""
from __future__ import annotations
from typing import Any
LABEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "Choroid plexus": {
        "text": "Choroid Plexus",
        "offset": (2.2, -16.3),
        "ha": "center",
        "va": "top",
    },
    "Brain": {
        "text": "Brain",
        "offset": (24.7, 8.9),
        "ha": "left",
        "va": "center",
    },
    "Meninges": {
        "text": "Meninges",
        "offset": (12.0, -7.0),
        "ha": "left",
        "va": "center",
    },
}
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scvelo as scv
from .communication import compute_legacy_communication_centroids

EXPECTED_STAGE = "E15.5"

PANEL_WIDTH_PT = 290.0

PANEL_HEIGHT_PT = 378.0

PANEL_PAGE_TOP_PT = 464.0

AXES_RECT = (9.5512 / PANEL_WIDTH_PT, 28.078 / PANEL_HEIGHT_PT, 250.124 / PANEL_WIDTH_PT, 310.301 / PANEL_HEIGHT_PT)

ROI_BOUNDS = (-1.3, -0.5, 3.3, 4.2)

EDGE_IDENTITIES = (
    ("Choroid plexus", "Brain"),
    ("Meninges", "Brain"),
    ("Brain", "Meninges"),
    ("Brain", "Brain"),
)

def select_edges(table: pd.DataFrame) -> pd.DataFrame:
    required = {"source", "target", "is_same_type", "weight_per_source"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"Communication table is missing columns: {missing}")
    rows = []
    for draw_order, (source, target) in enumerate(EDGE_IDENTITIES, start=1):
        match = table.loc[(table["source"].astype(str) == source) & (table["target"].astype(str) == target)]
        if len(match) != 1:
            raise RuntimeError(f"Expected one communication row for {source} -> {target}; found {len(match)}.")
        row = match.iloc[0].copy()
        weight = float(row["weight_per_source"])
        if not np.isfinite(weight) or weight <= 0.0:
            raise RuntimeError(f"Displayed communication edge is not finite positive: {source} -> {target}.")
        row["draw_order"] = draw_order
        row["legacy_display_role"] = "focus_same_type_loop" if source == target else "original_AI_cross_type_identity"
        rows.append(row)
    result = pd.DataFrame(rows).reset_index(drop=True)
    result["is_self_edge"] = result["is_same_type"].astype(bool)
    return result

def draw_interaction_velocity(numeric, communication, palette, output_dir):
    with np.load(numeric, allow_pickle=False) as archive:
        full_xy = np.asarray(archive["full_background_spatial"], dtype=np.float32)
        full_labels = np.asarray(archive["full_background_labels"]).astype(str)
        compute_xy = np.asarray(archive["compute_spatial"], dtype=np.float32)
        compute_labels = np.asarray(archive["compute_labels"]).astype(str)
        velocity = np.asarray(archive["gene_interaction_projected_spatial"], dtype=np.float32)
    edges = select_edges(pd.read_csv(communication))
    categories = sorted(set(full_labels))
    # Historical notebook Y-IQR gate, referenced to the complete observed state.
    q1, q3 = np.percentile(full_xy[:, 1], [25.0, 75.0])
    iqr = float(q3 - q1)
    y_bounds = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
    finite_velocity = np.isfinite(compute_xy).all(axis=1) & np.isfinite(velocity).all(axis=1) & (np.linalg.norm(velocity, axis=1) > 1e-12)
    in_y = (compute_xy[:, 1] >= y_bounds[0]) & (compute_xy[:, 1] <= y_bounds[1])
    keep = finite_velocity & in_y
    if int(keep.sum()) < 3:
        raise RuntimeError("Legacy Y-IQR display gate removed the velocity cohort.")
    velocity_adata = ad.AnnData(X=np.zeros((int(keep.sum()), 1), dtype=np.float32))
    velocity_adata.obsm["X_spatial"] = compute_xy[keep]
    velocity_adata.obsm["velocity_interaction_spatial"] = velocity[keep]
    categories = sorted(set(full_labels))
    velocity_adata.obs["Annotation"] = pd.Categorical(compute_labels[keep], categories=categories)

    centroids, centroid_table = compute_legacy_communication_centroids(
        full_xy,
        full_labels,
        ["Brain", "Choroid plexus", "Meninges"],
        top_n_y=200,
        top_n_y_exclusions=("Brain", "Meninges", "Choroid plexus"),
    )

    fig = plt.figure(figsize=(PANEL_WIDTH_PT / 72.0, PANEL_HEIGHT_PT / 72.0), facecolor="white")
    ax = fig.add_axes(AXES_RECT, facecolor="white")
    for label in categories:
        mask = full_labels == label
        ax.scatter(
            full_xy[mask, 0],
            full_xy[mask, 1],
            c=palette[label],
            s=2.35,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            zorder=0,
        )
    scv.pl.velocity_embedding_stream(
        velocity_adata,
        basis="spatial",
        vkey="velocity_interaction",
        color="Annotation",
        palette=[palette[label] for label in categories],
        ax=ax,
        show=False,
        density=2.0,
        smooth=0.8,
        min_mass=1.0,
        cutoff_perc=3.0,
        linewidth=0.55,
        arrow_size=0.26,
        n_neighbors=30,
        alpha=0.35,
        size=0.01,
        legend_loc="none",
        title="",
        frameon=False,
    )

    max_weight = float(edges["weight_per_source"].max())
    arrow_records = []
    drawn_nodes: set[str] = set()
    for row in edges.itertuples(index=False):
        source, target = str(row.source), str(row.target)
        p1, p2 = centroids[source], centroids[target]
        weight = float(row.weight_per_source)
        width = 1.0 + 10.0 * np.sqrt(weight / max_weight)
        if source == target:
            start = (float(p1[0] - 0.03), float(p1[1] + 0.05))
            end = (float(p1[0] + 0.06), float(p1[1] + 0.02))
            connectionstyle, mutation_scale = "arc3,rad=-10.0", 1.75
        else:
            start = (float(p1[0]), float(p1[1]))
            end = (float(p2[0]), float(p2[1]))
            connectionstyle, mutation_scale = "arc3,rad=0.15", 4.35
        ax.add_patch(
            mpatches.FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>,head_length=0.8,head_width=0.5",
                mutation_scale=mutation_scale,
                connectionstyle=connectionstyle,
                color=palette[source],
                linewidth=width,
                alpha=1.0,
                zorder=30,
            )
        )
        arrow_records.append(
            {
                "source": source,
                "target": target,
                "weight_per_source": weight,
                "linewidth_pt": float(width),
                "same_type_loop": source == target,
            }
        )
        for node, position in ((source, p1), (target, p2)):
            if node not in drawn_nodes:
                ax.scatter(position[0], position[1], s=162.0, c=palette[node], edgecolors="white", linewidth=2.5, zorder=31)
                drawn_nodes.add(node)

    ax.add_patch(
        mpatches.Rectangle(
            (ROI_BOUNDS[0], ROI_BOUNDS[2]),
            ROI_BOUNDS[1] - ROI_BOUNDS[0],
            ROI_BOUNDS[3] - ROI_BOUNDS[2],
            fill=False,
            edgecolor="black",
            linewidth=1.6,
            zorder=36,
        )
    )
    for node, spec in LABEL_SPECS.items():
        p = centroids[node]
        ax.annotate(
            str(spec["text"]),
            xy=(float(p[0]), float(p[1])),
            xytext=spec["offset"],
            textcoords="offset points",
            ha=str(spec["ha"]),
            va=str(spec["va"]),
            fontsize=9.14134,
            fontweight="bold",
            color="black",
            bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 1.0},
            zorder=40,
            clip_on=False,
        )

    # The archived notebook uses plt.subplots(figsize=(16, 20)) and never
    # overrides Matplotlib's default aspect.  Preserve that original panel
    # grammar while keeping every stored coordinate value byte-identical.
    ax.set_aspect("auto")
    ax.set_axis_off()
    fig.text(3.167236 / PANEL_WIDTH_PT, 1.0 - (469.115 - PANEL_PAGE_TOP_PT) / PANEL_HEIGHT_PT, "d", ha="left", va="top", fontsize=14.0, fontweight="bold")
    fig.text(13.6812 / PANEL_WIDTH_PT, 1.0 - (468.835 - PANEL_PAGE_TOP_PT) / PANEL_HEIGHT_PT, "Interaction-induced gene velocity", ha="left", va="top", fontsize=14.0, fontweight="bold")
    fig.text(32.8916 / PANEL_WIDTH_PT, 1.0 - (491.79 - PANEL_PAGE_TOP_PT) / PANEL_HEIGHT_PT, EXPECTED_STAGE, ha="left", va="top", fontsize=12.0, fontweight="bold")

    stem = "Figure4d_interaction_gene_velocity"
    outputs = {suffix: output_dir / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png")}
    fig.savefig(outputs["pdf"], format="pdf", dpi=400, facecolor="white")
    fig.savefig(outputs["svg"], format="svg", dpi=400, facecolor="white")
    fig.savefig(outputs["png"], format="png", dpi=600, facecolor="white")
    plt.close(fig)

    edges.to_csv(output_dir / "communication_displayed_original_AI_identities.csv", index=False)
    centroid_table.to_csv(output_dir / "communication_centroids.csv", index=False)
    return list(outputs.values())
