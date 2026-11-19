import glob
import shutil
from pathlib import Path
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from reproduction.videos._mosta_style import apply_nature_methods_mpl_style, get_nature_scatter_defaults

apply_nature_methods_mpl_style()
NATURE_STYLE = get_nature_scatter_defaults()

point_size = float(globals().get("point_size", NATURE_STYLE["point_size"]))
point_alpha = float(globals().get("point_alpha", NATURE_STYLE["point_alpha"]))

rng = np.random.default_rng(rng_seed)

focus_show_endpoints = globals().get("focus_show_endpoints", False)
focus_endpoint_size = globals().get("focus_endpoint_size", 10)
focus_endpoint_alpha = globals().get("focus_endpoint_alpha", 0.9)

focus_edge_curve = globals().get("focus_edge_curve", 0.18)
focus_edge_curve_bi = globals().get("focus_edge_curve_bi", 0.35)

focus_zoom_enabled = bool(globals().get("focus_zoom_enabled", True))
focus_zoom_mode = str(globals().get("focus_zoom_mode", "box_inset"))
focus_zoom_padding_frac = float(globals().get("focus_zoom_padding_frac", 0.12))
focus_zoom_position = str(globals().get("focus_zoom_position", "right_upper")).lower()
focus_zoom_size = str(globals().get("focus_zoom_size", "40%"))
focus_zoom_outside = bool(globals().get("focus_zoom_outside", True))
focus_zoom_outside_pad = float(globals().get("focus_zoom_outside_pad", 0.08))
focus_zoom_fixed = bool(globals().get("focus_zoom_fixed", True))
focus_zoom_fixed_mode = str(globals().get("focus_zoom_fixed_mode", "global")).lower()

# Axis limits shared across frames for stable view
all_coords = np.concatenate([adata.obsm["spatial"][:, :2] for adata in adata_dict.values()], axis=0)
min_xy = all_coords.min(axis=0)
max_xy = all_coords.max(axis=0)
pad = 0.05 * (max_xy - min_xy)
pad[pad == 0] = 1.0
lims = (min_xy - pad, max_xy + pad)


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (136, 136, 136)
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(v))) for v in rgb])


def _blend(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    return _rgb_to_hex((r1 * (1 - t) + r2 * t, g1 * (1 - t) + g2 * t, b1 * (1 - t) + b2 * t))


def _desaturate(color, amount=0.6):
    return _blend(color, "#bdbdbd", amount)


def label_centroids(coords, labels):
    cent = {}
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() > 0:
            cent[str(lab)] = coords[mask].mean(axis=0)
    return cent


def focus_neighbor_centroids(coords, labels, focus_type, neighbor_pct=0.2):
    labels = np.asarray(labels)
    coords = np.asarray(coords)
    if focus_type not in labels:
        return label_centroids(coords, labels)
    focus_mask = labels == focus_type
    focus_centroid = coords[focus_mask].mean(axis=0)
    cent = {}
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() == 0:
            continue
        if lab == focus_type:
            cent[str(lab)] = coords[mask].mean(axis=0)
            continue
        coords_l = coords[mask]
        d = np.linalg.norm(coords_l - focus_centroid, axis=1)
        if neighbor_pct is None:
            k = len(coords_l)
        else:
            pct = float(neighbor_pct)
            if pct > 1.0:
                pct = pct / 100.0
            pct = max(0.0, min(1.0, pct))
            k = max(1, int(np.ceil(pct * len(coords_l))))
        idx = np.argsort(d)[:k]
        cent[str(lab)] = coords_l[idx].mean(axis=0)
    return cent


def top_flows_for_frame(comm_result, top_k=6, min_weight=0.0, quantile=None):
    if not comm_result:
        return []
    M = comm_result.get("M_per_source")
    types = comm_result.get("types")
    if M is None or types is None:
        return []
    flows = []
    for i, src in enumerate(types):
        for j, dst in enumerate(types):
            w = float(M[i, j])
            if i != j and w > min_weight:
                flows.append((w, str(src), str(dst)))
    if not flows:
        return []
    if quantile is not None:
        weights = np.array([f[0] for f in flows])
        q = np.quantile(weights, quantile)
        flows = [f for f in flows if f[0] >= q]
    flows.sort(key=lambda x: x[0], reverse=True)
    return flows[:top_k]


def _get_focus_edges(comm_result, focus_type, mode="outgoing", top_k=6, min_weight=0.0, quantile=None):
    if not comm_result:
        return []
    M = comm_result.get("M_per_source")
    types = comm_result.get("types")
    if M is None or types is None:
        return []
    if focus_type not in types:
        return []
    idx = list(types).index(focus_type)
    flows = []
    for j, tname in enumerate(types):
        if j == idx:
            continue
        if mode in ("outgoing", "both"):
            flows.append((float(M[idx, j]), focus_type, str(tname)))
        if mode in ("incoming", "both"):
            flows.append((float(M[j, idx]), str(tname), focus_type))
    flows = [(w, s, d) for (w, s, d) in flows if w > min_weight]
    if not flows:
        return []
    if quantile is not None:
        weights = np.array([f[0] for f in flows])
        q = np.quantile(weights, quantile)
        flows = [f for f in flows if f[0] >= q]
    flows.sort(key=lambda x: x[0], reverse=True)
    return flows[:top_k]


def _draw_edge(ax, p1, p2, color, lw, alpha, rad=0.15):
    arrow = patches.FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=10 + lw * 1.5,
        lw=lw,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def _compute_roi_bbox(points_xy, limits, pad_frac=0.12):
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] == 0:
        return None

    xmin = float(np.min(pts[:, 0]))
    xmax = float(np.max(pts[:, 0]))
    ymin = float(np.min(pts[:, 1]))
    ymax = float(np.max(pts[:, 1]))

    xr = max(1e-6, xmax - xmin)
    yr = max(1e-6, ymax - ymin)
    px = xr * float(pad_frac)
    py = yr * float(pad_frac)

    x0 = max(float(limits[0][0]), xmin - px)
    x1 = min(float(limits[1][0]), xmax + px)
    y0 = max(float(limits[0][1]), ymin - py)
    y1 = min(float(limits[1][1]), ymax + py)

    if (x1 - x0) <= 1e-6 or (y1 - y0) <= 1e-6:
        return None
    return (x0, x1, y0, y1)



def _collect_focus_roi_points_for_time(time_key):
    adata_t = adata_dict[time_key]
    coords_t = adata_t.obsm["spatial"][:, :2]
    labels_t = adata_t.obs["Annotation"].astype(str).values

    centroids_t = focus_neighbor_centroids(coords_t, labels_t, focus_celltype, neighbor_pct=focus_neighbor_pct)
    flows_t = []
    if draw_focus_interactions and time_key in all_time_communications:
        comm_t = all_time_communications.get(time_key)
        flows_t = _get_focus_edges(
            comm_t,
            focus_type=focus_celltype,
            mode=focus_mode,
            top_k=focus_top_k,
            min_weight=focus_min_weight,
            quantile=focus_edge_quantile,
        )

    pts = []
    if len(flows_t) > 0 and len(centroids_t) > 0:
        for _, src, dst in flows_t:
            if src in centroids_t:
                pts.append(centroids_t[src])
            if dst in centroids_t:
                pts.append(centroids_t[dst])

    if len(pts) == 0 and draw_focus_interactions and (focus_celltype in labels_t):
        pts = coords_t[labels_t == focus_celltype].tolist()

    return pts


fixed_roi_bbox = None
if focus_zoom_enabled and focus_zoom_mode == "box_inset" and focus_zoom_fixed:
    if focus_zoom_fixed_mode == "first_valid":
        for tk in time_keys:
            pts = _collect_focus_roi_points_for_time(tk)
            cand = _compute_roi_bbox(pts, lims, pad_frac=focus_zoom_padding_frac)
            if cand is not None:
                fixed_roi_bbox = cand
                break
    else:
        all_pts = []
        for tk in time_keys:
            all_pts.extend(_collect_focus_roi_points_for_time(tk))
        fixed_roi_bbox = _compute_roi_bbox(all_pts, lims, pad_frac=focus_zoom_padding_frac)


frame_files = []
for idx, key in enumerate(time_keys):
    adata = adata_dict[key]
    coords = adata.obsm["spatial"][:, :2]
    labels = adata.obs["Annotation"].astype(str).values

    if frame_point_subsample and frame_point_subsample < len(coords):
        take = rng.choice(len(coords), size=frame_point_subsample, replace=False)
        coords = coords[take]
        labels = labels[take]

    base_colors = [label_to_color.get(lab, "#555555") for lab in labels]
    if draw_focus_interactions:
        colors = [
            (label_to_color.get(lab, "#555555") if lab == focus_celltype else _desaturate(label_to_color.get(lab, "#555555"), focus_other_desaturate))
            for lab in labels
        ]
    else:
        colors = base_colors

    use_right_inset = focus_zoom_enabled and (focus_zoom_position in ("right_upper", "right_lower"))
    fig_w = 7.6 if (use_right_inset and focus_zoom_outside) else 6.0
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    if use_right_inset and focus_zoom_outside:
        fig.subplots_adjust(right=0.78)
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=point_size, alpha=point_alpha, edgecolors="none")

    # Halo highlight for focus cells
    if draw_focus_interactions and focus_celltype in labels:
        focus_mask = labels == focus_celltype
        if focus_mask.sum() > 0:
            halo_size = point_size * focus_halo_scale * (1.0 + 0.15 * np.sin(2 * np.pi * idx / max(len(time_keys), 1)))
            ax.scatter(
                coords[focus_mask, 0],
                coords[focus_mask, 1],
                s=halo_size,
                c=label_to_color.get(focus_celltype, "#111111"),
                alpha=0.2,
                edgecolors="none",
            )

    ax.set_title(f"t = {float(key):.2f}", fontsize=float(NATURE_STYLE["title_fontsize"]))
    ax.set_facecolor(background_color)
    ax.set_xlim(lims[0][0], lims[1][0])
    ax.set_ylim(lims[0][1], lims[1][1])
    ax.set_aspect("equal")
    ax.axis("off")

    # All-cell-type arrows (optional legacy)
    if draw_comm_arrows and key in all_time_communications:
        comm = all_time_communications.get(key)
        flows = top_flows_for_frame(comm, top_k=top_flow_edges, min_weight=comm_edge_threshold)
        if flows:
            centroids = label_centroids(coords, labels)
            max_w = flows[0][0]
            for w, src, dst in flows:
                if src not in centroids or dst not in centroids:
                    continue
                alpha = 0.35 + 0.65 * (w / max_w if max_w > 0 else 0.0)
                ax.annotate(
                    "",
                    xy=centroids[dst],
                    xytext=centroids[src],
                    arrowprops=dict(
                        arrowstyle="->",
                        color="black",
                        lw=2.0,
                        alpha=alpha,
                        shrinkA=5,
                        shrinkB=5,
                    ),
                )

    focus_flows = []
    focus_centroids = {}

    # Focused interaction overlay
    if draw_focus_interactions and key in all_time_communications:
        comm = all_time_communications.get(key)
        focus_centroids = focus_neighbor_centroids(coords, labels, focus_celltype, neighbor_pct=focus_neighbor_pct)
        focus_flows = _get_focus_edges(
            comm,
            focus_type=focus_celltype,
            mode=focus_mode,
            top_k=focus_top_k,
            min_weight=focus_min_weight,
            quantile=focus_edge_quantile,
        )
        if focus_flows:
            directed_pairs = {(s, d) for _, s, d in focus_flows}
            bidirectional_pairs = {tuple(sorted((s, d))) for s, d in directed_pairs if (d, s) in directed_pairs}
            max_w = max([w for w, _, _ in focus_flows])
            for w, src, dst in focus_flows:
                if src not in focus_centroids or dst not in focus_centroids:
                    continue
                t = w / max_w if max_w > 0 else 0.0
                lw = focus_edge_width[0] + t * (focus_edge_width[1] - focus_edge_width[0])
                alpha = focus_edge_alpha[0] + t * (focus_edge_alpha[1] - focus_edge_alpha[0])
                edge_color = label_to_color.get(dst, label_to_color.get(src, "#111111"))
                pair = tuple(sorted((src, dst)))
                if pair in bidirectional_pairs:
                    rad = focus_edge_curve_bi
                else:
                    rad = focus_edge_curve if src < dst else -focus_edge_curve
                _draw_edge(ax, focus_centroids[src], focus_centroids[dst], edge_color, lw=lw, alpha=alpha, rad=rad)
                if focus_show_endpoints:
                    ax.scatter(
                        [focus_centroids[src][0], focus_centroids[dst][0]],
                        [focus_centroids[src][1], focus_centroids[dst][1]],
                        s=focus_endpoint_size,
                        c=[label_to_color.get(src, "#111111"), label_to_color.get(dst, "#111111")],
                        alpha=focus_endpoint_alpha,
                        edgecolors="none",
                        zorder=3,
                    )

        # Label focus type
        ax.text(
            0.02,
            0.98,
            f"Focus: {focus_celltype} ({focus_mode})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=float(NATURE_STYLE["legend_fontsize"]),
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8),
        )

    # Focus ROI box + inset zoom
    roi_points = []
    if len(focus_flows) > 0 and len(focus_centroids) > 0:
        for _, src, dst in focus_flows:
            if src in focus_centroids:
                roi_points.append(focus_centroids[src])
            if dst in focus_centroids:
                roi_points.append(focus_centroids[dst])
    if len(roi_points) == 0 and draw_focus_interactions and (focus_celltype in labels):
        focus_mask = labels == focus_celltype
        roi_points = coords[focus_mask].tolist()

    roi_bbox_dynamic = _compute_roi_bbox(roi_points, lims, pad_frac=focus_zoom_padding_frac)
    roi_bbox = fixed_roi_bbox if (focus_zoom_fixed and fixed_roi_bbox is not None) else roi_bbox_dynamic

    if focus_zoom_enabled and focus_zoom_mode == "box_inset" and roi_bbox is not None:
        x0, x1, y0, y1 = roi_bbox
        rect = patches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            linestyle="--",
            linewidth=1.1,
            edgecolor="#222222",
            alpha=0.95,
            zorder=5,
        )
        ax.add_patch(rect)

        loc_map = {
            "right_upper": "upper right",
            "right_lower": "lower right",
            "left_upper": "upper left",
            "left_lower": "lower left",
        }
        ins_loc = loc_map.get(focus_zoom_position, "upper right")

        if focus_zoom_outside:
            if focus_zoom_position == "right_upper":
                inset_loc = "upper left"
                bbox = (1.0 + focus_zoom_outside_pad, 0.0, 1.0, 1.0)
            elif focus_zoom_position == "right_lower":
                inset_loc = "lower left"
                bbox = (1.0 + focus_zoom_outside_pad, 0.0, 1.0, 1.0)
            elif focus_zoom_position == "left_upper":
                inset_loc = "upper right"
                bbox = (-focus_zoom_outside_pad, 0.0, 1.0, 1.0)
            else:
                inset_loc = "lower right"
                bbox = (-focus_zoom_outside_pad, 0.0, 1.0, 1.0)

            axins = inset_axes(
                ax,
                width=focus_zoom_size,
                height=focus_zoom_size,
                loc=inset_loc,
                bbox_to_anchor=bbox,
                bbox_transform=ax.transAxes,
                borderpad=0.0,
            )
        else:
            axins = inset_axes(ax, width=focus_zoom_size, height=focus_zoom_size, loc=ins_loc, borderpad=0.8)

        axins.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=max(0.6, point_size * 0.75),
            alpha=min(1.0, point_alpha + 0.1),
            edgecolors="none",
        )

        if len(focus_flows) > 0 and len(focus_centroids) > 0:
            directed_pairs = {(s, d) for _, s, d in focus_flows}
            bidirectional_pairs = {tuple(sorted((s, d))) for s, d in directed_pairs if (d, s) in directed_pairs}
            max_w = max([w for w, _, _ in focus_flows])
            for w, src, dst in focus_flows:
                if src not in focus_centroids or dst not in focus_centroids:
                    continue
                t = w / max_w if max_w > 0 else 0.0
                lw = focus_edge_width[0] + t * (focus_edge_width[1] - focus_edge_width[0])
                alpha = focus_edge_alpha[0] + t * (focus_edge_alpha[1] - focus_edge_alpha[0])
                edge_color = label_to_color.get(dst, label_to_color.get(src, "#111111"))
                pair = tuple(sorted((src, dst)))
                if pair in bidirectional_pairs:
                    rad = focus_edge_curve_bi
                else:
                    rad = focus_edge_curve if src < dst else -focus_edge_curve
                _draw_edge(axins, focus_centroids[src], focus_centroids[dst], edge_color, lw=lw, alpha=alpha, rad=rad)

        axins.set_xlim(x0, x1)
        axins.set_ylim(y0, y1)
        axins.set_aspect("equal")
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_facecolor("white")
        for s in axins.spines.values():
            s.set_edgecolor("#111111")
            s.set_linewidth(1.0)

        if focus_zoom_position in ("right_upper", "right_lower"):
            c_loc1, c_loc2 = 2, 4
        else:
            c_loc1, c_loc2 = 1, 3
        mark_inset(ax, axins, loc1=c_loc1, loc2=c_loc2, fc="none", ec="#222222", lw=0.8, alpha=0.75)

    out_path = os.path.join(frames_dir, f"frame_{idx:03d}.png")
    fig.savefig(out_path, dpi=int(NATURE_STYLE["frame_dpi"]), facecolor="white")
    plt.close(fig)
    frame_files.append(out_path)

print(f"Saved {len(frame_files)} frames to {frames_dir}")
