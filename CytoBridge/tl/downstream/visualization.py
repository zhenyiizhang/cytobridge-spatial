"""Visualization-oriented downstream helpers."""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "load_label_to_color",
    "save_timepoint_snapshots",
]


def load_label_to_color(
    labels: np.ndarray,
    *,
    label_color_json: Optional[str] = None,
    color_h5ad: Optional[str] = None,
    annotation_key: str = "Annotation",
) -> Dict[str, str]:
    """Resolve a label-to-color mapping from JSON, AnnData metadata, or a fallback palette."""
    if label_color_json and os.path.exists(label_color_json):
        import json

        with open(label_color_json, "r", encoding="utf-8") as handle:
            return json.load(handle)

    if color_h5ad and os.path.exists(color_h5ad):
        try:
            import anndata as ad

            adata = ad.read_h5ad(color_h5ad, backed="r")
            try:
                key = annotation_key if annotation_key in adata.obs else None
                if key is None and annotation_key.lower() in adata.obs:
                    key = annotation_key.lower()
                if key:
                    colors_key = f"{key}_colors"
                    colors = adata.uns.get(colors_key)
                    if colors is not None:
                        categories = (
                            adata.obs[key].cat.categories
                            if hasattr(adata.obs[key], "cat")
                            else sorted(adata.obs[key].unique())
                        )
                        return {str(c): str(col) for c, col in zip(categories, colors)}
            finally:
                try:
                    adata.file.close()
                except Exception:
                    pass
        except Exception as exc:
            print(f"Color map load failed from {color_h5ad}: {exc}")

    import matplotlib.pyplot as plt

    unique_labels = list(dict.fromkeys([str(x) for x in labels]))
    cmap = plt.get_cmap("tab20")
    out = {}
    for idx, lab in enumerate(unique_labels):
        rgb = cmap(idx % cmap.N)[:3]
        out[str(lab)] = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    return out


def save_timepoint_snapshots(
    *,
    adata_dict,
    time_keys: Sequence[str],
    annotation_key: str,
    label_to_color: Dict[str, str],
    observed_variants: Optional[Dict[float, Dict[str, tuple[np.ndarray, np.ndarray]]]] = None,
    snapshot_dir: str,
    background_color: Optional[str],
    font_color: str,
    snapshot_point_size: float,
    snapshot_alpha: float,
    mosaic_cols: int,
    mosaic_cell_size: float,
    mosaic_show_title: bool,
    save_pdf: bool = True,
) -> None:
    """Save 2D per-timepoint scatter snapshots and a simple mosaic/legend bundle."""
    import math

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    bg = background_color or "white"
    os.makedirs(snapshot_dir, exist_ok=True)

    panels: list[dict] = []
    for tk in time_keys:
        t_val = float(tk)
        if observed_variants is not None and (t_val in observed_variants):
            variants = observed_variants[t_val]
            for suffix, (coords, labels) in variants.items():
                panels.append(
                    {
                        "tk": tk,
                        "suffix": suffix,
                        "title": f"t = {tk} ({suffix})",
                        "coords": np.asarray(coords),
                        "labels": np.asarray(labels).astype(str),
                    }
                )
        else:
            ad = adata_dict[tk]
            coords = np.asarray(ad.obsm["spatial"])
            labels = ad.obs[annotation_key].astype(str).values
            panels.append(
                {
                    "tk": tk,
                    "suffix": None,
                    "title": f"t = {tk}",
                    "coords": coords,
                    "labels": np.asarray(labels).astype(str),
                }
            )

    for panel in panels:
        coords = np.asarray(panel["coords"])
        labels = np.asarray(panel["labels"]).astype(str)
        colors = [label_to_color.get(str(l), "#888888") for l in labels]
        rasterized = coords.shape[0] > 30000

        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(snapshot_point_size),
            c=colors,
            linewidths=0,
            alpha=float(snapshot_alpha),
            rasterized=rasterized,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(panel["title"], color=font_color, fontsize=12, pad=6)

        tk = panel["tk"]
        suffix = panel["suffix"]
        stem = f"time_{tk}" if suffix is None else f"time_{tk}__{suffix}"
        out_path = os.path.join(snapshot_dir, f"{stem}.svg")
        fig.savefig(out_path, format="svg", facecolor=bg, bbox_inches="tight")
        if save_pdf:
            fig.savefig(
                os.path.join(snapshot_dir, f"{stem}.pdf"),
                format="pdf",
                facecolor=bg,
                bbox_inches="tight",
            )
        plt.close(fig)

    n_panels = len(panels)
    cols = max(1, int(mosaic_cols))
    rows = math.ceil(n_panels / cols)
    fig_w = cols * float(mosaic_cell_size)
    fig_h = rows * float(mosaic_cell_size)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor(bg)
    axes = axes if isinstance(axes, np.ndarray) else np.array([[axes]])
    axes = axes.reshape(rows, cols)

    for idx, panel in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        coords = np.asarray(panel["coords"])
        labels = np.asarray(panel["labels"]).astype(str)
        colors = [label_to_color.get(str(l), "#888888") for l in labels]
        rasterized = coords.shape[0] > 30000
        ax.set_facecolor(bg)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=float(snapshot_point_size),
            c=colors,
            linewidths=0,
            alpha=float(snapshot_alpha),
            rasterized=rasterized,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if mosaic_show_title:
            title = panel["tk"] if panel.get("suffix") is None else f"{panel['tk']} ({panel['suffix']})"
            ax.set_title(title, color=font_color, fontsize=8, pad=3)

    for idx in range(n_panels, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
        axes[r, c].set_facecolor(bg)

    fig.savefig(
        os.path.join(snapshot_dir, "timepoint_mosaic.svg"),
        format="svg",
        facecolor=bg,
        bbox_inches="tight",
    )
    plt.close(fig)

    legend_path = os.path.join(snapshot_dir, "label_legend.svg")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=label_to_color[k], markersize=6, label=k)
        for k in label_to_color.keys()
    ]
    fig, ax = plt.subplots(figsize=(4, 6), facecolor=bg)
    ax.set_facecolor(bg)
    ax.legend(handles=handles, loc="center left", frameon=False, labelcolor=font_color)
    ax.axis("off")
    fig.savefig(legend_path, format="svg", facecolor=bg, bbox_inches="tight")
    plt.close(fig)
