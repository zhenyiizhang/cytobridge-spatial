"""Submitted MOSTA video style, with optional local cell-edge arrow anchors."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np

def apply_nature_methods_mpl_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.0,
            "legend.title_fontsize": 8.0,
            "figure.titlesize": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )

def get_nature_scatter_defaults() -> dict[str, float]:
    return {
        "frame_dpi": 300.0,
        "point_size": 2.5,
        "point_alpha": 0.9,
        "title_fontsize": 10.0,
        "suptitle_fontsize": 12.0,
        "legend_fontsize": 7.0,
    }

def _compute_xy_limits(
    points: Sequence[np.ndarray],
    *,
    pad_frac: float = 0.03,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    for arr in points:
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0:
            continue
        x_min = min(x_min, float(np.min(a[:, 0])))
        x_max = max(x_max, float(np.max(a[:, 0])))
        y_min = min(y_min, float(np.min(a[:, 1])))
        y_max = max(y_max, float(np.max(a[:, 1])))

    if not np.isfinite(x_min) or not np.isfinite(y_min):
        raise ValueError("Failed to compute axis limits from simulated points.")

    x_span = max(1e-8, x_max - x_min)
    y_span = max(1e-8, y_max - y_min)
    x_pad = x_span * float(pad_frac)
    y_pad = y_span * float(pad_frac)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = str(hex_color).strip().lstrip("#")
    if len(s) == 8:
        s = s[:6]
    if len(s) != 6:
        return (136, 136, 136)
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (136, 136, 136)

def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(v))) for v in rgb])

def _blend_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((r1 * (1 - t) + r2 * t, g1 * (1 - t) + g2 * t, b1 * (1 - t) + b2 * t))

def _desaturate_hex(color: str, amount: float = 0.55) -> str:
    a = float(np.clip(amount, 0.0, 1.0))
    return _blend_hex(color, "#c0c0c0", a)

def _edge_contrast_color(color: str, darken: float = 0.35) -> str:
    rgb = np.asarray(_hex_to_rgb(color), dtype=np.float32) / 255.0
    d = float(np.clip(darken, 0.0, 0.85))
    rgb = np.clip(rgb * (1.0 - d), 0.0, 1.0)
    return _rgb_to_hex((rgb[0] * 255.0, rgb[1] * 255.0, rgb[2] * 255.0))

def _label_centroids(coords: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    labels_np = np.asarray(labels).astype(str)
    out: Dict[str, np.ndarray] = {}
    for lab in np.unique(labels_np):
        mask = labels_np == lab
        if int(np.sum(mask)) > 0:
            out[str(lab)] = np.asarray(coords, dtype=np.float32)[mask].mean(axis=0)
    return out

def _focus_neighbor_centroids(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    focus_type: str,
    neighbor_pct: float = 0.1,
) -> Dict[str, np.ndarray]:
    labels_np = np.asarray(labels).astype(str)
    coords_np = np.asarray(coords, dtype=np.float32)
    if focus_type not in set(labels_np.tolist()):
        return _label_centroids(coords_np, labels_np)

    focus_mask = labels_np == str(focus_type)
    focus_centroid = coords_np[focus_mask].mean(axis=0)
    out: Dict[str, np.ndarray] = {}
    pct = float(neighbor_pct)
    if pct > 1.0:
        pct = pct / 100.0
    pct = float(np.clip(pct, 0.0, 1.0))

    for lab in np.unique(labels_np):
        mask = labels_np == lab
        if int(np.sum(mask)) == 0:
            continue
        if str(lab) == str(focus_type):
            out[str(lab)] = coords_np[mask].mean(axis=0)
            continue
        coords_l = coords_np[mask]
        d = np.linalg.norm(coords_l - focus_centroid, axis=1)
        k = max(1, int(np.ceil(pct * len(coords_l))))
        idx = np.argsort(d)[:k]
        out[str(lab)] = coords_l[idx].mean(axis=0)
    return out

def _get_focus_edges(
    comm_result: Optional[dict],
    *,
    focus_type: str,
    mode: str = "both",
    top_k: int = 4,
    min_weight: float = 0.0,
    quantile: Optional[float] = None,
) -> list[tuple[float, str, str]]:
    if not comm_result:
        return []
    M = comm_result.get("M_per_source")
    types = comm_result.get("types")
    if M is None or types is None:
        return []
    types_list = [str(x) for x in list(types)]
    if str(focus_type) not in types_list:
        return []

    idx = types_list.index(str(focus_type))
    flows: list[tuple[float, str, str]] = []
    for j, tname in enumerate(types_list):
        if j == idx:
            continue
        if mode in ("outgoing", "both"):
            flows.append((float(M[idx, j]), str(focus_type), str(tname)))
        if mode in ("incoming", "both"):
            flows.append((float(M[j, idx]), str(tname), str(focus_type)))
    flows = [(w, s, d) for (w, s, d) in flows if float(w) > float(min_weight)]
    if not flows:
        return []
    if quantile is not None:
        weights = np.array([f[0] for f in flows], dtype=float)
        q = float(np.quantile(weights, float(quantile)))
        flows = [f for f in flows if float(f[0]) >= q]
    flows.sort(key=lambda x: float(x[0]), reverse=True)
    selected = flows[: max(1, int(top_k))]
    # Optional display filter, applied after ranking without promoting other pairs.
    # The original score matrix and all unfiltered results remain unchanged.
    minimum = comm_result.get('display_min_edges')
    if minimum is not None:
        support = {(row['source'], row['target']): row['cell_edges']
                   for row in comm_result['display_edge_support']}
        selected = [(w, s, d) for w, s, d in selected
                    if support[(s, d)] >= int(minimum)]
    return selected

def _draw_edge(ax, p1, p2, color: str, lw: float, alpha: float, rad: float = 0.15, local: bool = False) -> None:
    from matplotlib import patches
    import matplotlib.patheffects as pe

    edge_color = _edge_contrast_color(color)
    if local:
        # A schematic glyph centered on the contact, not a distance measurement.
        # Preserve the local sender-to-receiver direction and enlarge only its
        # display length so that it remains visible at whole-tissue scale.
        start, end = ax.transData.transform([p1, p2])
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            raise ValueError('A zero-length contact has no drawable direction')
        midpoint = (start + end) / 2
        half = direction / length * max(length, 24 * ax.figure.dpi / 72) / 2
        p1, p2 = ax.transData.inverted().transform([midpoint - half, midpoint + half])
    head = 10 + float(lw) * 1.5
    width = float(lw)
    arrow = patches.FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=head,
        lw=width,
        color=str(edge_color),
        alpha=float(alpha),
        zorder=1,
        connectionstyle=f"arc3,rad={float(rad)}",
        shrinkA=4,
        shrinkB=4,
    )
    arrow.set_path_effects(
        [
            pe.Stroke(linewidth=width + 1.5, foreground="white", alpha=min(1.0, 0.95 * float(alpha))),
            pe.Normal(),
        ]
    )
    ax.add_patch(arrow)
    return p1, p2

def _render_frame_single_2d(
    *,
    xy: np.ndarray,
    labels: np.ndarray,
    time_value: float,
    label_to_color: Dict[str, str],
    out_png: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    point_size: float,
    alpha: float,
    dpi: int,
    panel_size: float,
    title: str,
    show_titles: bool,
    comm_result: Optional[dict],
    config: MostaBaselineVideoConfig,
) -> None:
    import matplotlib.pyplot as plt

    panel_size = float(max(2.0, panel_size))
    fig, ax = plt.subplots(1, 1, figsize=(panel_size, panel_size), dpi=int(dpi))
    fig.patch.set_facecolor("white")

    labs = np.asarray(labels).astype(str)
    base_colors = [label_to_color.get(str(l), "#888888") for l in labs]
    if bool(config.draw_focus_interactions):
        colors = [
            (
                label_to_color.get(str(l), "#888888")
                if str(l) == str(config.focus_celltype)
                else _desaturate_hex(label_to_color.get(str(l), "#888888"), config.focus_other_desaturate)
            )
            for l in labs
        ]
    else:
        colors = base_colors

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=float(point_size),
        c=colors,
        linewidths=0,
        alpha=float(alpha),
        rasterized=xy.shape[0] > 30000,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if bool(show_titles):
        ax.set_title(f"{title}\nt={time_value:.1f}", fontsize=float(NATURE_SCATTER_DEFAULTS["title_fontsize"]))
        fig.suptitle("MOSTA Baseline Trajectory", fontsize=float(NATURE_SCATTER_DEFAULTS["suptitle_fontsize"]))

    if bool(config.draw_focus_interactions):
        local_anchors = None if comm_result is None else comm_result.get('local_anchors')
        anchor_lookup = {(a['source'], a['target']): a for a in local_anchors or []}
        # Establish the final axes transform before sizing short local arrows.
        fig.tight_layout()
        fig.canvas.draw()
        centroids = _focus_neighbor_centroids(
            xy,
            labs,
            focus_type=str(config.focus_celltype),
            neighbor_pct=float(config.focus_neighbor_pct),
        )
        flows = _get_focus_edges(
            comm_result,
            focus_type=str(config.focus_celltype),
            mode=str(config.focus_mode),
            top_k=int(config.focus_top_k),
            min_weight=float(config.focus_min_weight),
            quantile=config.focus_edge_quantile,
        )
        if flows:
            directed_pairs = {(s, d) for _, s, d in flows}
            bidirectional_pairs = {tuple(sorted((s, d))) for s, d in directed_pairs if (d, s) in directed_pairs}
            max_w = max(float(w) for w, _, _ in flows)
            highlighted = []
            for w, src, dst in flows:
                use_local = (src, dst) == ('Spinal cord', 'Brain') and (src, dst) in anchor_lookup
                if not use_local and (src not in centroids or dst not in centroids):
                    continue
                t = float(w) / max_w if max_w > 0 else 0.0
                lw = float(config.focus_edge_width_min) + t * (
                    float(config.focus_edge_width_max) - float(config.focus_edge_width_min)
                )
                a = float(config.focus_edge_alpha_min) + t * (
                    float(config.focus_edge_alpha_max) - float(config.focus_edge_alpha_min)
                )
                edge_color = label_to_color.get(str(dst), label_to_color.get(str(src), "#111111"))
                pair = tuple(sorted((src, dst)))
                if pair in bidirectional_pairs:
                    rad = float(config.focus_edge_curve_bi)
                else:
                    rad = float(config.focus_edge_curve) if str(src) < str(dst) else -float(config.focus_edge_curve)
                if use_local:
                    anchor = anchor_lookup[(src, dst)]
                    if not np.isclose(w, anchor['pair_score']):
                        raise ValueError('Local anchors do not match the ranked pair score')
                    p1, p2 = _draw_edge(ax, anchor['start'], anchor['end'], edge_color,
                                       lw=lw, alpha=a, rad=rad, local=True)
                else:
                    p1, p2 = _draw_edge(ax, centroids[src], centroids[dst], edge_color, lw=lw, alpha=a, rad=rad)
                highlighted.extend([(src, p1), (dst, p2)])

            if bool(config.focus_show_endpoints):
                if highlighted:
                    unique = {(label, *map(float, position)): position for label, position in highlighted}
                    node_xy = np.asarray(list(unique.values()), dtype=np.float32)
                    node_colors = [label_to_color.get(key[0], "#111111") for key in unique]
                    ax.scatter(
                        node_xy[:, 0],
                        node_xy[:, 1],
                        s=float(max(1.0, config.focus_endpoint_size)),
                        c=node_colors,
                        linewidths=float(max(0.0, config.focus_endpoint_linewidth)),
                        edgecolors=str(config.focus_endpoint_edgecolor),
                        alpha=float(np.clip(config.focus_endpoint_alpha, 0.0, 1.0)),
                        zorder=6,
                    )

        ax.text(
            0.02,
            0.98,
            f"Focus: {config.focus_celltype} ({config.focus_mode})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=float(NATURE_SCATTER_DEFAULTS["legend_fontsize"]),
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

NATURE_SCATTER_DEFAULTS=get_nature_scatter_defaults()
apply_nature_methods_mpl_style()
