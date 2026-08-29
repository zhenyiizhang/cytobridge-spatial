"""3D spatiotemporal Sankey-style visualization (Plotly).

This module is ported from the ST-1104 downstream analysis code and provides a
single, feature-complete function for rendering spatiotemporal communication
and lineage ribbons in 3D.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "plot_3d_spatial_sankey_style",
]


def plot_3d_spatial_sankey_style(
    adata_dict,
    all_time_communications,
    time_keys,
    label_to_color,
    predicted_labels_list,
    spatial_key="spatial",
    annotation_key: str = "Annotation",
    intra_threshold=1.0,
    ribbon_resolution=20,
    ribbon_width_scale=0.01,
    z_spacing=3.0,
    focus_celltype=None,
    edge_focus_celltype=None,
    edge_top_k=None,
    edge_top_k_focus_label: Optional[str] = None,
    edge_weight_quantile=None,
    edge_width_scale=1.0,
    edge_global_top_k=None,
    edge_render_mode="line",
    edge_line_width_base=1.0,
    edge_line_width_scale=1.0,
    edge_center_highlight: bool = False,
    edge_center_highlight_width_scale: float = 0.45,
    edge_center_highlight_alpha: float = 0.9,
    edge_color: Optional[str] = None,
    edge_show_arrows: bool = False,
    edge_arrow_length_scale: float = 0.14,
    edge_arrow_width_scale: float = 0.5,
    edge_arrow_position: float = 0.78,
    edge_arrow_in_slice_plane: bool = False,
    ribbon_top_k=None,
    ribbon_count_quantile=None,
    ribbon_min_count: Optional[float] = None,
    ribbon_keep_source_cumfrac: Optional[float] = None,
    ribbon_focus_celltype=None,
    ribbon_focus_target_only=False,
    ribbon_render_mode="line",
    ribbon_line_width_base=1.0,
    ribbon_line_width_scale=1.0,
    ribbon_line_alpha: float = 0.6,
    ribbon_line_curve: float = 0.0,
    ribbon_line_points: int = 16,
    ribbon_center_highlight: bool = False,
    ribbon_center_highlight_width_scale: float = 0.5,
    ribbon_center_highlight_alpha: float = 0.9,
    point_size=1.0,
    point_subsample: Optional[Union[int, float]] = None,
    observed_point_subsample: Optional[Union[int, float]] = None,
    generated_point_subsample: Optional[Union[int, float]] = None,
    point_alpha: float = 0.6,
    observed_point_alpha: Optional[float] = None,
    generated_point_alpha: Optional[float] = None,
    point_line_width: float = 0.0,
    point_line_color: Optional[str] = None,
    observed_point_line_width: Optional[float] = None,
    observed_point_line_color: Optional[str] = None,
    generated_point_line_width: Optional[float] = None,
    generated_point_line_color: Optional[str] = None,
    show_centroid_nodes=False,
    centroid_node_size=6.0,
    centroid_node_opacity=0.9,
    highlight_endpoints=False,
    endpoint_size=8.0,
    endpoint_opacity=0.95,
    background_color="white",
    font_color="black",
    reverse_time_order=False,
    anchor_mode: str = "centroid",
    anchor_subsample: Optional[int] = 1000,
    slices_only: bool = False,
    show_time_axis: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    width: int = 1400,
    height: int = 1000,
    include_self_loops=False,
    self_loop_radius_scale=0.6,
    self_loop_line_width_base=1.0,
    self_loop_line_width_scale=1.0,
    self_loop_points=36,
    self_loop_focus_label: Optional[str] = None,
    flow_normalize_mode: Optional[str] = None,
    lineage_anchor_mode: bool = False,
    anchor_time_index: int = 0,
    ribbon_focus_source_only: bool = False,
    out_html=None,
    show_slice_border: bool = False,
    slice_border_color: str = "black",
    slice_border_width: float = 2.0,
    slice_border_color_observed: Optional[str] = None,
    slice_border_color_generated: Optional[str] = None,
    slice_fill_color_observed: Optional[str] = None,
    slice_fill_color_generated: Optional[str] = None,
    slice_fill_opacity: float = 0.0,
    observed_time_points: Optional[Sequence] = None,
    generated_time_points: Optional[Sequence] = None,
    focus_anchor_label: Optional[str] = None,
    focus_anchor_k: int = 200,
    focus_anchor_frac: Optional[float] = None,
    focus_anchor_radius: Optional[float] = None,
    focus_anchor_min_count: int = 10,
    bidirectional_offset: float = 0.0,
    bidirectional_curve: bool = False,
    bidirectional_curve_points: int = 16,
    exclude_celltypes: Optional[Sequence[str]] = None,
):
    """
    Variant of plot_3d_spatial_sankey_style with optional focus-centric anchors.

    If focus_anchor_label is set, then any edge/ribbon involving that label keeps the
    focus label's centroid, while the other endpoint uses the centroid of its cells
    nearest to the focus label (per time slice).
    """
    import pandas as pd
    import plotly.graph_objects as go

    def to_valid_color(color_str, default_alpha=1.0):
        if not isinstance(color_str, str):
            return "rgba(136,136,136,1)"
        color_str = color_str.strip()
        if color_str.startswith("#") and len(color_str) == 9:
            r = int(color_str[1:3], 16)
            g = int(color_str[3:5], 16)
            b = int(color_str[5:7], 16)
            a = int(color_str[7:9], 16) / 255.0
            return f"rgba({r},{g},{b},{a:.3f})"
        if color_str.startswith("#") and len(color_str) == 7:
            if default_alpha < 1.0:
                r = int(color_str[1:3], 16)
                g = int(color_str[3:5], 16)
                b = int(color_str[5:7], 16)
                return f"rgba({r},{g},{b},{default_alpha})"
            return color_str
        return color_str

    def get_rgb_tuple(color_str):
        valid_c = to_valid_color(color_str)
        if valid_c.startswith("#"):
            return (int(valid_c[1:3], 16), int(valid_c[3:5], 16), int(valid_c[5:7], 16))
        if valid_c.startswith("rgba") or valid_c.startswith("rgb"):
            import re

            nums = [float(x) for x in re.findall(r"[\d\.]+", valid_c)]
            return (int(nums[0]), int(nums[1]), int(nums[2]))
        return (136, 136, 136)

    def _rgba_from_color(color_str, default_alpha=1.0):
        c = to_valid_color(color_str, default_alpha=default_alpha)
        if c.startswith("rgba(") and c.endswith(")"):
            parts = [p.strip() for p in c[5:-1].split(",")]
            if len(parts) == 4:
                try:
                    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    return None
        if c.startswith("rgb(") and c.endswith(")"):
            parts = [p.strip() for p in c[4:-1].split(",")]
            if len(parts) == 3:
                try:
                    return float(parts[0]), float(parts[1]), float(parts[2]), float(default_alpha)
                except ValueError:
                    return None
        return None

    def _lighten_color(color_str, amount=0.6, alpha: Optional[float] = None, default_alpha=1.0):
        rgba = _rgba_from_color(color_str, default_alpha=default_alpha)
        if rgba is None:
            a = alpha if alpha is not None else default_alpha
            return f"rgba(255,255,255,{a:.3f})"
        r, g, b, a = rgba
        r = int(round(r + (255.0 - r) * amount))
        g = int(round(g + (255.0 - g) * amount))
        b = int(round(b + (255.0 - b) * amount))
        a = alpha if alpha is not None else a
        return f"rgba({r},{g},{b},{a:.3f})"

    def cylinder_between(p1, p2, radius=0.3, n=16):
        p1, p2 = np.array(p1), np.array(p2)
        v = p2 - p1
        length = np.linalg.norm(v)
        if length < 1e-6:
            return None
        v = v / length
        not_v = np.array([1, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        theta = np.linspace(0, 2 * np.pi, n)
        circle = radius * (np.outer(np.cos(theta), n1) + np.outer(np.sin(theta), n2))
        p1_ring, p2_ring = circle + p1, circle + p2
        x = np.hstack([p1_ring[:, 0], p2_ring[:, 0]])
        y = np.hstack([p1_ring[:, 1], p2_ring[:, 1]])
        z = np.hstack([p1_ring[:, 2], p2_ring[:, 2]])
        i, j, k = [], [], []
        n_theta = len(theta)
        for t in range(n_theta - 1):
            base = t
            next_t = t + 1
            i.extend([base, next_t, base + n_theta])
            j.extend([next_t, next_t + n_theta, next_t + n_theta])
            k.extend([base + n_theta, base + n_theta, base])
        i.extend([n_theta - 1, 0, 2 * n_theta - 1])
        j.extend([0, n_theta, n_theta])
        k.extend([2 * n_theta - 1, 2 * n_theta - 1, n_theta - 1])
        return x, y, z, i, j, k

    def create_ribbon_surface(c1, c2, count, color_from, color_to, n_points=20):
        width = ribbon_width_scale * count
        width = min(width, 10.0)
        t = np.linspace(0, 1, n_points)
        x_path = c1["x"] + t * (c2["x"] - c1["x"])
        y_path = c1["y"] + t * (c2["y"] - c1["y"])
        z_path = c1["z"] + t * (c2["z"] - c1["z"])
        bend = 0.2 * np.sin(np.pi * t) * (abs(c2["x"] - c1["x"]) + abs(c2["y"] - c1["y"]))
        x_path = x_path + bend * 0.1
        dx = np.gradient(x_path)
        dy = np.gradient(y_path)
        norm = np.sqrt(dx**2 + dy**2) + 1e-6
        perp_x = -dy / norm
        perp_y = dx / norm
        x_left, x_right = x_path + perp_x * width, x_path - perp_x * width
        y_left, y_right = y_path + perp_y * width, y_path - perp_y * width
        x_mesh = np.column_stack([x_left, x_right]).flatten()
        y_mesh = np.column_stack([y_left, y_right]).flatten()
        z_mesh = np.column_stack([z_path, z_path]).flatten()
        i_idx, j_idx, k_idx = [], [], []
        for idx in range(n_points - 1):
            base = idx * 2
            i_idx.extend([base, base + 1])
            j_idx.extend([base + 1, base + 3])
            k_idx.extend([base + 2, base + 2])
        rgb_from = get_rgb_tuple(color_from)
        rgb_to = get_rgb_tuple(color_to)
        vertex_colors = []
        for idx in range(n_points):
            blend = t[idx]
            r, g, b = [int(s * (1 - blend) + e * blend) for s, e in zip(rgb_from, rgb_to)]
            c = f"rgb({r},{g},{b})"
            vertex_colors.extend([c, c])
        return {
            "x": x_mesh,
            "y": y_mesh,
            "z": z_mesh,
            "i": i_idx,
            "j": j_idx,
            "k": k_idx,
            "colors": vertex_colors,
        }

    def create_arrowhead_segments(
        p_prev,
        p_tip,
        *,
        length_scale: float,
        width_scale: float,
        plane_normal=None,
        reference_length: Optional[float] = None,
    ):
        p_prev = np.asarray(p_prev, dtype=float)
        p_tip = np.asarray(p_tip, dtype=float)
        direction = p_tip - p_prev
        direction_len = float(np.linalg.norm(direction))
        if direction_len < 1e-6:
            return None
        direction /= direction_len
        edge_len = direction_len if reference_length is None else float(reference_length)
        arrow_len = min(edge_len * max(float(length_scale), 0.0), edge_len * 0.35)
        if arrow_len < 1e-6:
            return None
        arrow_width = arrow_len * max(float(width_scale), 0.0)

        if plane_normal is not None:
            normal = np.asarray(plane_normal, dtype=float)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm >= 1e-6:
                normal /= normal_norm
                perpendiculars = [np.cross(normal, direction)]
            else:
                perpendiculars = []
        else:
            reference = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(direction, reference))) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            first = np.cross(direction, reference)
            second = np.cross(direction, first)
            perpendiculars = [first, second]

        wings = []
        base = p_tip - direction * arrow_len
        for perpendicular in perpendiculars:
            perpendicular_norm = float(np.linalg.norm(perpendicular))
            if perpendicular_norm < 1e-6:
                continue
            perpendicular = perpendicular / perpendicular_norm
            wings.extend(
                [
                    base + perpendicular * arrow_width,
                    base - perpendicular * arrow_width,
                ]
            )
        if not wings:
            return None
        xs, ys, zs = [], [], []
        for wing in wings:
            xs.extend([wing[0], p_tip[0], None])
            ys.extend([wing[1], p_tip[1], None])
            zs.extend([wing[2], p_tip[2], None])
        return xs, ys, zs

    if anchor_mode not in ("centroid", "nearest"):
        raise ValueError("anchor_mode must be 'centroid' or 'nearest'")
    if slices_only:
        show_centroid_nodes = False
        highlight_endpoints = False

    exclude_set = set(exclude_celltypes or [])

    def _resolve_annotation_col(ad):
        if annotation_key in ad.obs.columns:
            return annotation_key
        for col in ("Annotation", "annotation", "bin_annotation"):
            if col in ad.obs.columns:
                return col
        raise KeyError(
            "No annotation column found for 3D sankey. "
            f"Tried: '{annotation_key}', 'Annotation', 'annotation', 'bin_annotation'."
        )
    all_types = set()
    for tk in time_keys:
        ad = adata_dict[tk]
        ann_col = _resolve_annotation_col(ad)
        all_types.update([ct for ct in ad.obs[ann_col].unique() if ct not in exclude_set])
    all_types = sorted(all_types)
    if reverse_time_order:
        z_values = [(len(time_keys) - 1 - i) * z_spacing for i in range(len(time_keys))]
    else:
        z_values = [i * z_spacing for i in range(len(time_keys))]

    fig = go.Figure()
    centroids = {}
    type_coords_by_time = {}
    endpoint_nodes = {}
    anchor_cache_same: Dict[tuple, Optional[tuple]] = {}
    anchor_cache_cross: Dict[tuple, Optional[tuple]] = {}
    focus_cache: Dict[tuple, Optional[dict]] = {}
    rng = np.random.default_rng(0)

    def _subsample_coords(coords, max_n):
        if max_n is None or coords.shape[0] <= max_n:
            return coords
        idx = rng.choice(coords.shape[0], size=int(max_n), replace=False)
        return coords[idx]

    def _normalize_focus_labels(value):
        if value is None:
            return None
        if isinstance(value, str):
            labels = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
            labels = [str(item).strip() for item in value if str(item).strip()]
        else:
            labels = [str(value).strip()]
        return set(labels) if labels else None

    def _filter_transitions_keep_source_cumfrac(df, cumfrac: float):
        cumfrac = float(cumfrac)
        if not 0.0 < cumfrac <= 1.0:
            raise ValueError("ribbon_keep_source_cumfrac must be in (0, 1].")
        keep_indices = []
        for _, group in df.groupby("source", sort=False):
            group = group.sort_values("count", ascending=False)
            total = float(group["count"].sum())
            if total <= 0:
                continue
            cumulative = (group["count"].cumsum() / total).to_numpy()
            keep_n = int(np.searchsorted(cumulative, cumfrac, side="left")) + 1
            keep_indices.extend(group.head(max(1, min(len(group), keep_n))).index.tolist())
        return df.loc[keep_indices] if keep_indices else df.iloc[0:0]

    def _canonical_time_value(value):
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    observed_set = None
    generated_set = None
    if observed_time_points:
        observed_set = {_canonical_time_value(v) for v in observed_time_points}
    if generated_time_points:
        generated_set = {_canonical_time_value(v) for v in generated_time_points}

    def _is_generated(time_key):
        if generated_set is not None:
            return time_key in generated_set
        if observed_set is not None:
            return time_key not in observed_set
        return False

    def _resolve_point_alpha(is_generated):
        if is_generated and generated_point_alpha is not None:
            return generated_point_alpha
        if (not is_generated) and observed_point_alpha is not None:
            return observed_point_alpha
        return point_alpha

    def _resolve_point_line_width(is_generated):
        if is_generated and generated_point_line_width is not None:
            return generated_point_line_width
        if (not is_generated) and observed_point_line_width is not None:
            return observed_point_line_width
        return point_line_width

    def _resolve_point_line_color(is_generated):
        if is_generated and generated_point_line_color:
            return generated_point_line_color
        if (not is_generated) and observed_point_line_color:
            return observed_point_line_color
        return point_line_color

    def _resolve_subsample_indices(n, spec):
        if spec is None:
            return None
        try:
            val = float(spec)
        except (TypeError, ValueError):
            return None
        if val <= 0:
            return None
        if 0 < val < 1.0:
            size = int(round(n * val))
        else:
            size = int(round(val))
        size = max(1, min(n, size))
        if size >= n:
            return None
        return rng.choice(n, size=size, replace=False)

    def _resolve_subsample_spec(is_generated):
        if is_generated:
            if generated_point_subsample is not None:
                return generated_point_subsample
        else:
            if observed_point_subsample is not None:
                return observed_point_subsample
        return point_subsample

    def _resolve_slice_border_color(is_generated):
        if is_generated and slice_border_color_generated:
            return slice_border_color_generated
        if (not is_generated) and slice_border_color_observed:
            return slice_border_color_observed
        return slice_border_color

    def _resolve_slice_fill_color(is_generated):
        if is_generated and slice_fill_color_generated:
            return slice_fill_color_generated
        if (not is_generated) and slice_fill_color_observed:
            return slice_fill_color_observed
        return None

    def _nearest_pair(coords_a, coords_b):
        if coords_a.size == 0 or coords_b.size == 0:
            return None
        coords_a = _subsample_coords(coords_a, anchor_subsample)
        coords_b = _subsample_coords(coords_b, anchor_subsample)
        if coords_a.size == 0 or coords_b.size == 0:
            return None
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(coords_b)
            dists, idx = tree.query(coords_a, k=1)
            min_idx = int(np.argmin(dists))
            return coords_a[min_idx], coords_b[int(idx[min_idx])]
        except Exception:
            dmat = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=2)
            min_idx = np.unravel_index(np.argmin(dmat), dmat.shape)
            return coords_a[min_idx[0]], coords_b[min_idx[1]]

    def _local_centroid_near_focus(tk, ct, z_val):
        if focus_anchor_label is None:
            return None
        if ct == focus_anchor_label:
            return centroids.get(tk, {}).get(ct)
        key = (tk, ct)
        if key in focus_cache:
            return focus_cache[key]
        coords_map = type_coords_by_time.get(tk, {})
        focus_coords = coords_map.get(focus_anchor_label)
        target_coords = coords_map.get(ct)
        if focus_coords is None or target_coords is None:
            focus_cache[key] = None
            return None
        if focus_coords.size == 0 or target_coords.size == 0:
            focus_cache[key] = None
            return None
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(focus_coords)
            dists, _ = tree.query(target_coords, k=1)
        except Exception:
            dmat = np.linalg.norm(target_coords[:, None, :] - focus_coords[None, :, :], axis=2)
            dists = dmat.min(axis=1)

        subset = None
        if focus_anchor_radius is not None:
            subset = target_coords[dists <= float(focus_anchor_radius)]
        elif focus_anchor_frac is not None:
            frac = float(focus_anchor_frac)
            if not (0 < frac <= 1.0):
                frac = 0.2
            k = max(1, int(round(len(dists) * frac)))
            idx = np.argsort(dists)[:k]
            subset = target_coords[idx]
        else:
            k = int(focus_anchor_k) if focus_anchor_k is not None else 0
            if k <= 0:
                k = max(1, int(round(len(dists) * 0.2)))
            idx = np.argsort(dists)[: min(k, len(dists))]
            subset = target_coords[idx]

        if subset is None or subset.size == 0:
            focus_cache[key] = None
            return None
        if focus_anchor_min_count is not None and subset.shape[0] < int(focus_anchor_min_count):
            focus_cache[key] = None
            return None

        centroid = subset.mean(axis=0)
        val = {"x": float(centroid[0]), "y": float(centroid[1]), "z": z_val}
        focus_cache[key] = val
        return val

    def _get_anchor_same_layer(tk, type_from, type_to, z_val):
        if anchor_mode != "nearest":
            layer_cents = centroids.get(tk, {})
            if type_from not in layer_cents or type_to not in layer_cents:
                return None
            if focus_anchor_label and (type_from == focus_anchor_label or type_to == focus_anchor_label):
                if type_from == focus_anchor_label:
                    p1 = layer_cents[type_from]
                    p2 = _local_centroid_near_focus(tk, type_to, z_val) or layer_cents[type_to]
                else:
                    p1 = _local_centroid_near_focus(tk, type_from, z_val) or layer_cents[type_from]
                    p2 = layer_cents[type_to]
                return p1, p2
            return layer_cents[type_from], layer_cents[type_to]
        key = (tk, type_from, type_to)
        if key in anchor_cache_same:
            cached = anchor_cache_same[key]
            if cached is None:
                return None
            p1, p2 = cached
            return {"x": float(p1[0]), "y": float(p1[1]), "z": z_val}, {
                "x": float(p2[0]),
                "y": float(p2[1]),
                "z": z_val,
            }
        coords_map = type_coords_by_time.get(tk, {})
        if type_from not in coords_map or type_to not in coords_map:
            anchor_cache_same[key] = None
            return None
        pair = _nearest_pair(coords_map[type_from], coords_map[type_to])
        if pair is None:
            anchor_cache_same[key] = None
            return None
        p1, p2 = pair
        anchor_cache_same[key] = (p1, p2)
        return {"x": float(p1[0]), "y": float(p1[1]), "z": z_val}, {
            "x": float(p2[0]),
            "y": float(p2[1]),
            "z": z_val,
        }

    def _get_anchor_cross_layer(t1_key, t2_key, src, tgt, z1, z2):
        if anchor_mode != "nearest":
            if src not in centroids.get(t1_key, {}) or tgt not in centroids.get(t2_key, {}):
                return None
            if focus_anchor_label and (src == focus_anchor_label or tgt == focus_anchor_label):
                if src == focus_anchor_label:
                    p1 = centroids[t1_key][src]
                    p2 = _local_centroid_near_focus(t2_key, tgt, z2) or centroids[t2_key][tgt]
                else:
                    p1 = _local_centroid_near_focus(t1_key, src, z1) or centroids[t1_key][src]
                    p2 = centroids[t2_key][tgt]
                return p1, p2
            return centroids[t1_key][src], centroids[t2_key][tgt]
        key = (t1_key, t2_key, src, tgt)
        if key in anchor_cache_cross:
            cached = anchor_cache_cross[key]
            if cached is None:
                return None
            p1, p2 = cached
            return {"x": float(p1[0]), "y": float(p1[1]), "z": z1}, {
                "x": float(p2[0]),
                "y": float(p2[1]),
                "z": z2,
            }
        coords_map_1 = type_coords_by_time.get(t1_key, {})
        coords_map_2 = type_coords_by_time.get(t2_key, {})
        if src not in coords_map_1 or tgt not in coords_map_2:
            anchor_cache_cross[key] = None
            return None
        pair = _nearest_pair(coords_map_1[src], coords_map_2[tgt])
        if pair is None:
            anchor_cache_cross[key] = None
            return None
        p1, p2 = pair
        anchor_cache_cross[key] = (p1, p2)
        return {"x": float(p1[0]), "y": float(p1[1]), "z": z1}, {
            "x": float(p2[0]),
            "y": float(p2[1]),
            "z": z2,
        }

    def _align_labels_for_flow(lbl_list: Sequence[Sequence[str]], anchor_idx: int = 0):
        arrs = [np.asarray(list(x)).astype(str) for x in lbl_list]
        if not arrs:
            return [], None
        min_len = min(len(a) for a in arrs)
        arrs = [a[:min_len] for a in arrs]
        if anchor_idx < 0 or anchor_idx >= len(arrs):
            raise ValueError("anchor_time_index out of range.")
        return arrs, arrs[anchor_idx]

    labels_flow: List[np.ndarray] = []
    anchor_labels: Optional[np.ndarray] = None
    if predicted_labels_list:
        labels_flow, anchor_labels = _align_labels_for_flow(predicted_labels_list, anchor_time_index)
    focus_labels = _normalize_focus_labels(
        ribbon_focus_celltype if ribbon_focus_celltype is not None else focus_celltype
    )

    for layer_idx, (tk, z) in enumerate(zip(time_keys, z_values)):
        ad = adata_dict[tk]
        ann_col = _resolve_annotation_col(ad)
        if spatial_key not in ad.obsm:
            continue
        coords = np.asarray(ad.obsm[spatial_key])
        labels = ad.obs[ann_col].values
        if exclude_set:
            keep_mask = np.array([lab not in exclude_set for lab in labels], dtype=bool)
            coords = coords[keep_mask]
            labels = labels[keep_mask]
        time_key = _canonical_time_value(tk)
        is_generated = _is_generated(time_key)
        layer_centroids = {}
        layer_type_coords = {}
        for ct in all_types:
            mask = labels == ct
            if np.any(mask):
                layer_type_coords[ct] = coords[mask]
                centroid = coords[mask].mean(axis=0)
                layer_centroids[ct] = {"x": float(centroid[0]), "y": float(centroid[1]), "z": z}
        centroids[tk] = layer_centroids
        type_coords_by_time[tk] = layer_type_coords

        sample_spec = _resolve_subsample_spec(is_generated)
        idx = _resolve_subsample_indices(coords.shape[0], sample_spec)
        if idx is not None:
            coords_plot = coords[idx]
            labels_plot = labels[idx]
        else:
            coords_plot = coords
            labels_plot = labels

        point_alpha_val = _resolve_point_alpha(is_generated)
        point_colors = [
            to_valid_color(label_to_color.get(l, "#888888"), default_alpha=point_alpha_val)
            for l in labels_plot
        ]
        line_width = _resolve_point_line_width(is_generated)
        line_color = _resolve_point_line_color(is_generated)
        marker_kwargs = dict(size=point_size, color=point_colors)
        if line_width and line_width > 0:
            if not line_color:
                line_color = "#333333"
            marker_kwargs["line"] = dict(
                width=float(line_width),
                color=to_valid_color(line_color, default_alpha=1.0),
            )

        if (show_slice_border or slice_fill_opacity > 0.0) and coords.size:
            x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
            x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
            span = max(x_max - x_min, y_max - y_min)
            pad = span * 0.02 if span > 0 else 1.0
            x_min -= pad
            x_max += pad
            y_min -= pad
            y_max += pad

            if slice_fill_opacity > 0.0:
                fill_color = _resolve_slice_fill_color(is_generated)
                if fill_color is not None:
                    fig.add_trace(
                        go.Mesh3d(
                            x=[x_min, x_max, x_max, x_min],
                            y=[y_min, y_min, y_max, y_max],
                            z=[z, z, z, z],
                            i=[0, 0],
                            j=[1, 2],
                            k=[2, 3],
                            color=to_valid_color(fill_color, default_alpha=1.0),
                            opacity=slice_fill_opacity,
                            flatshading=True,
                            showscale=False,
                            hoverinfo="skip",
                        )
                    )

        fig.add_trace(
            go.Scatter3d(
                x=coords_plot[:, 0],
                y=coords_plot[:, 1],
                z=[z] * len(coords_plot),
                mode="markers",
                marker=marker_kwargs,
                name=tk,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        if show_slice_border and coords.size:
            dash_style = "dash" if is_generated else "solid"
            border_color = to_valid_color(_resolve_slice_border_color(is_generated), default_alpha=1.0)
            fig.add_trace(
                go.Scatter3d(
                    x=[x_min, x_max, x_max, x_min, x_min],
                    y=[y_min, y_min, y_max, y_max, y_min],
                    z=[z, z, z, z, z],
                    mode="lines",
                    line=dict(color=border_color, width=slice_border_width, dash=dash_style),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if show_centroid_nodes and layer_centroids:
            cent_x = []
            cent_y = []
            cent_z = []
            cent_c = []
            cent_t = []
            for ct, vals in layer_centroids.items():
                cent_x.append(vals["x"])
                cent_y.append(vals["y"])
                cent_z.append(vals["z"])
                cent_c.append(
                    to_valid_color(label_to_color.get(ct, "#888888"), default_alpha=centroid_node_opacity)
                )
                cent_t.append(f"{ct} ({tk})")
            fig.add_trace(
                go.Scatter3d(
                    x=cent_x,
                    y=cent_y,
                    z=cent_z,
                    mode="markers",
                    marker=dict(size=centroid_node_size, color=cent_c, line=dict(width=0.5, color="white")),
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=cent_t,
                )
            )

    def _add_endpoint_node(tk, ct, use_focus_local=False):
        if not highlight_endpoints:
            return
        if tk not in centroids:
            return
        key = (tk, ct)
        if key in endpoint_nodes and not use_focus_local:
            return

        vals = None
        if use_focus_local and focus_anchor_label and ct != focus_anchor_label:
            z_val = centroids.get(tk, {}).get(ct, {}).get("z")
            if z_val is None:
                z_val = 0.0
            vals = _local_centroid_near_focus(tk, ct, z_val)
        if vals is None:
            vals = centroids.get(tk, {}).get(ct)
        if vals is None:
            return

        endpoint_nodes[key] = {
            "x": vals["x"],
            "y": vals["y"],
            "z": vals["z"],
            "color": to_valid_color(label_to_color.get(ct, "#888888"), default_alpha=endpoint_opacity),
            "label": f"{ct} ({tk})",
        }

    if not slices_only:
        edges_by_time = {}
        edges_all = []
        self_loops_by_time = {}
        for layer_idx, (tk, z_val) in enumerate(zip(time_keys, z_values)):
            if tk not in all_time_communications:
                continue
            comm_data = all_time_communications[tk]
            M_comm, types = comm_data["M_per_source"], comm_data["types"]
            type_to_idx = {t: i for i, t in enumerate(types)}
            layer_cents = centroids.get(tk, {})
            edges_this_tk = []
            self_loops_this_tk = []
            for type_from in types:
                if type_from in exclude_set:
                    continue
                for type_to in types:
                    if type_to in exclude_set:
                        continue
                    if type_from == type_to:
                        if include_self_loops:
                            i = type_to_idx.get(type_from)
                            if i is None:
                                continue
                            if type_from not in layer_cents:
                                continue
                            weight = float(M_comm[i, i])
                            if weight > intra_threshold:
                                self_loops_this_tk.append({"weight": weight, "type_from": type_from})
                        continue
                    focus_edge = edge_focus_celltype if edge_focus_celltype is not None else focus_celltype
                    if focus_edge and type_from != focus_edge and type_to != focus_edge:
                        continue
                    i, j = type_to_idx.get(type_from), type_to_idx.get(type_to)
                    if i is None or j is None:
                        continue
                    weight = float(M_comm[i, j])
                    if weight <= intra_threshold:
                        continue
                    edges_this_tk.append({"weight": weight, "type_from": type_from, "type_to": type_to})

            if edges_this_tk:
                weights = np.array([e["weight"] for e in edges_this_tk], dtype=float)
                thresh = intra_threshold
                if edge_weight_quantile is not None and len(weights) > 1:
                    try:
                        quant_cut = np.quantile(weights, edge_weight_quantile)
                        thresh = max(thresh, float(quant_cut))
                    except Exception:
                        pass
                edges_this_tk = [e for e in edges_this_tk if e["weight"] >= thresh]
                edges_this_tk = sorted(edges_this_tk, key=lambda x: x["weight"], reverse=True)
                total_after_thresh = len(edges_this_tk)
                if edge_top_k_focus_label:
                    edges_this_tk = [
                        e
                        for e in edges_this_tk
                        if e["type_from"] == edge_top_k_focus_label or e["type_to"] == edge_top_k_focus_label
                    ]
                total_after_focus = len(edges_this_tk)
                if edge_top_k is not None and edge_global_top_k is None:
                    edges_this_tk = edges_this_tk[: int(edge_top_k)]
                kept_after_topk = len(edges_this_tk)
                print(
                    f"[edges] {tk}: total_after_thresh={total_after_thresh}, "
                    f"after_focus={total_after_focus}, kept={kept_after_topk}"
                )
            else:
                print(f"[edges] {tk}: total_after_thresh=0, after_focus=0, kept=0")

            edges_by_time[tk] = edges_this_tk
            self_loops_by_time[tk] = self_loops_this_tk
            for e in edges_this_tk:
                edges_all.append({"tk": tk, "z": z_val, **e})

        if edge_global_top_k is not None and edges_all:
            edges_all = sorted(edges_all, key=lambda x: x["weight"], reverse=True)
            selected = edges_all[: int(edge_global_top_k)]
            edges_by_time = {tk: [] for tk in time_keys}
            for e in selected:
                edges_by_time[e["tk"]].append(e)

        for tk, z_val in zip(time_keys, z_values):
            if tk not in all_time_communications:
                continue
            layer_cents = centroids.get(tk, {})
            drawn_count = 0
            skipped_missing_anchor = 0
            undirected_pairs = set()
            edges_list = edges_by_time.get(tk, [])
            pair_counts = {}
            if bidirectional_offset:
                for e in edges_list:
                    pair = tuple(sorted((e["type_from"], e["type_to"])))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            for e in edges_list:
                w = e["weight"]
                t_f, t_t = e["type_from"], e["type_to"]
                anchor_pair = _get_anchor_same_layer(tk, t_f, t_t, z_val)
                if anchor_pair is None:
                    skipped_missing_anchor += 1
                    continue
                a1, a2 = anchor_pair
                p1 = [a1["x"], a1["y"], a1["z"]]
                p2 = [a2["x"], a2["y"], a2["z"]]
                curve_points = None
                if bidirectional_offset:
                    pair = tuple(sorted((t_f, t_t)))
                    if pair_counts.get(pair, 0) > 1:
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        norm = float(np.hypot(dx, dy))
                        if norm > 1e-6:
                            perp_x = -dy / norm
                            perp_y = dx / norm
                            off = float(bidirectional_offset)
                            if bidirectional_curve:
                                mid_x = 0.5 * (p1[0] + p2[0]) + perp_x * off
                                mid_y = 0.5 * (p1[1] + p2[1]) + perp_y * off
                                mid_z = 0.5 * (p1[2] + p2[2])
                                t_vals = np.linspace(0.0, 1.0, max(4, int(bidirectional_curve_points)))
                                curve_points = []
                                for t in t_vals:
                                    omt = 1.0 - t
                                    x = omt * omt * p1[0] + 2.0 * omt * t * mid_x + t * t * p2[0]
                                    y = omt * omt * p1[1] + 2.0 * omt * t * mid_y + t * t * p2[1]
                                    z = omt * omt * p1[2] + 2.0 * omt * t * mid_z + t * t * p2[2]
                                    curve_points.append((x, y, z))
                            else:
                                p1[0] += perp_x * off
                                p1[1] += perp_y * off
                                p2[0] += perp_x * off
                                p2[1] += perp_y * off
                use_focus_local = focus_anchor_label and (t_f == focus_anchor_label or t_t == focus_anchor_label)
                _add_endpoint_node(tk, t_f, use_focus_local=use_focus_local)
                _add_endpoint_node(tk, t_t, use_focus_local=use_focus_local)
                if edge_render_mode == "line":
                    line_w = edge_line_width_base + np.log1p(w * 10) * edge_line_width_scale
                    line_w = max(0.5, float(line_w))
                    if edge_color is not None:
                        line_color = to_valid_color(edge_color, default_alpha=0.85)
                    else:
                        line_color = to_valid_color(label_to_color.get(t_f, "#cccccc"), default_alpha=0.85)
                    if curve_points is not None:
                        xs, ys, zs = zip(*curve_points)
                        segments = np.diff(np.asarray(curve_points, dtype=float), axis=0)
                        arrow_reference_length = float(np.linalg.norm(segments, axis=1).sum())
                        arrow_idx = min(
                            len(curve_points) - 1,
                            max(1, int(round(float(edge_arrow_position) * (len(curve_points) - 1)))),
                        )
                        arrow_prev = curve_points[arrow_idx - 1]
                        arrow_tip = curve_points[arrow_idx]
                    else:
                        xs, ys, zs = [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]]
                        arrow_reference_length = float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))
                        arrow_fraction = min(0.95, max(0.15, float(edge_arrow_position)))
                        previous_fraction = max(0.0, arrow_fraction - 0.12)
                        arrow_prev = np.asarray(p1) + (np.asarray(p2) - np.asarray(p1)) * previous_fraction
                        arrow_tip = np.asarray(p1) + (np.asarray(p2) - np.asarray(p1)) * arrow_fraction
                    fig.add_trace(
                        go.Scatter3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            mode="lines",
                            line=dict(width=line_w, color=line_color),
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=f"Comm: {t_f} -> {t_t}<br>{tk}<br>Val: {w:.3f}",
                        )
                    )
                    if edge_show_arrows:
                        arrow_segments = create_arrowhead_segments(
                            arrow_prev,
                            arrow_tip,
                            length_scale=edge_arrow_length_scale,
                            width_scale=edge_arrow_width_scale,
                            plane_normal=np.array([0.0, 0.0, 1.0]) if edge_arrow_in_slice_plane else None,
                            reference_length=arrow_reference_length,
                        )
                        if arrow_segments is not None:
                            fig.add_trace(
                                go.Scatter3d(
                                    x=arrow_segments[0],
                                    y=arrow_segments[1],
                                    z=arrow_segments[2],
                                    mode="lines",
                                    line=dict(width=max(0.8, float(line_w)), color=line_color),
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )
                    if edge_center_highlight:
                        hl_color = _lighten_color(
                            line_color,
                            amount=0.6,
                            alpha=edge_center_highlight_alpha,
                            default_alpha=0.9,
                        )
                        hl_w = max(0.3, float(line_w) * edge_center_highlight_width_scale)
                        fig.add_trace(
                            go.Scatter3d(
                                x=xs,
                                y=ys,
                                z=zs,
                                mode="lines",
                                line=dict(width=hl_w, color=hl_color),
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
                else:
                    line_w = 0.8 + np.log1p(w * 10) * edge_width_scale
                    rad = min(0.04 * line_w, 0.35)
                    cyl = cylinder_between(p1, p2, radius=rad, n=20)
                    if cyl:
                        mesh_color = (
                            to_valid_color(edge_color, default_alpha=0.9)
                            if edge_color is not None
                            else to_valid_color(label_to_color.get(t_f, "#cccccc"), default_alpha=0.9)
                        )
                        fig.add_trace(
                            go.Mesh3d(
                                x=cyl[0],
                                y=cyl[1],
                                z=cyl[2],
                                i=cyl[3],
                                j=cyl[4],
                                k=cyl[5],
                                color=mesh_color,
                                opacity=1.0,
                                showscale=False,
                                hoverinfo="text",
                                hovertext=f"Comm: {t_f} -> {t_t}<br>{tk}<br>Val: {w:.3f}",
                            )
                        )
                drawn_count += 1
                undirected_pairs.add(tuple(sorted((t_f, t_t))))
            if edges_by_time.get(tk, []):
                print(
                    f"[edges_drawn] {tk}: drawn={drawn_count}, "
                    f"skipped_missing_anchor={skipped_missing_anchor}, "
                    f"unique_undirected_pairs={len(undirected_pairs)}"
                )

            if include_self_loops:
                for loop in self_loops_by_time.get(tk, []):
                    t_f = loop["type_from"]
                    if self_loop_focus_label is not None and t_f != self_loop_focus_label:
                        continue
                    w = loop["weight"]
                    if t_f not in layer_cents:
                        continue
                    center = layer_cents[t_f]
                    radius = self_loop_radius_scale * (0.6 + np.log1p(w))
                    theta = np.linspace(0, 2 * np.pi, self_loop_points)
                    x_loop = center["x"] + radius * np.cos(theta)
                    y_loop = center["y"] + radius * np.sin(theta)
                    z_loop = [center["z"]] * len(theta)
                    line_w = self_loop_line_width_base + np.log1p(w) * self_loop_line_width_scale
                    if edge_color is not None:
                        loop_color = to_valid_color(edge_color, default_alpha=0.9)
                    else:
                        loop_color = to_valid_color(label_to_color.get(t_f, "#cccccc"), default_alpha=0.9)
                    fig.add_trace(
                        go.Scatter3d(
                            x=x_loop,
                            y=y_loop,
                            z=z_loop,
                            mode="lines",
                            line=dict(width=line_w, color=loop_color),
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=f"Comm: {t_f} -> {t_f}<br>{tk}<br>Val: {w:.3f}",
                        )
                    )

        if flow_normalize_mode not in (None, "source", "global"):
            raise ValueError("flow_normalize_mode must be None, 'source', or 'global'")

        flow_pairs = range(len(time_keys) - 1) if ribbon_render_mode != "none" else ()
        for t_idx in flow_pairs:
            t1_key = time_keys[t_idx]
            t2_key = time_keys[t_idx + 1]
            labels_t = labels_flow[t_idx] if labels_flow else predicted_labels_list[t_idx]
            labels_next = labels_flow[t_idx + 1] if labels_flow else predicted_labels_list[t_idx + 1]
            min_len = min(len(labels_t), len(labels_next))
            if min_len == 0:
                continue
            if lineage_anchor_mode and anchor_labels is not None:
                anc = anchor_labels[:min_len]
                df_flow = pd.DataFrame({"ancestor": anc, "source": labels_t[:min_len], "target": labels_next[:min_len]})
                if exclude_set:
                    df_flow = df_flow[
                        (~df_flow["source"].isin(exclude_set))
                        & (~df_flow["target"].isin(exclude_set))
                        & (~df_flow["ancestor"].isin(exclude_set))
                    ]
                transitions = df_flow.groupby(["ancestor", "source", "target"]).size().reset_index(name="count")
                if flow_normalize_mode == "source":
                    transitions["value"] = transitions["count"] / transitions.groupby(["ancestor", "source"])[
                        "count"
                    ].transform("sum")
                elif flow_normalize_mode == "global":
                    total = transitions["count"].sum()
                    transitions["value"] = transitions["count"] / total if total > 0 else transitions["count"]
                else:
                    transitions["value"] = transitions["count"]
            else:
                df_flow = pd.DataFrame({"source": labels_t[:min_len], "target": labels_next[:min_len]})
                if exclude_set:
                    df_flow = df_flow[
                        (~df_flow["source"].isin(exclude_set))
                        & (~df_flow["target"].isin(exclude_set))
                    ]
                transitions = df_flow.groupby(["source", "target"]).size().reset_index(name="count")
                if flow_normalize_mode == "source":
                    transitions["value"] = transitions["count"] / transitions.groupby("source")["count"].transform("sum")
                elif flow_normalize_mode == "global":
                    total = transitions["count"].sum()
                    transitions["value"] = transitions["count"] / total if total > 0 else transitions["count"]
                else:
                    transitions["value"] = transitions["count"]
            if transitions.empty:
                continue
            if ribbon_min_count is not None:
                transitions = transitions[transitions["count"] >= ribbon_min_count]
                if transitions.empty:
                    continue
            if ribbon_keep_source_cumfrac is not None:
                transitions = _filter_transitions_keep_source_cumfrac(
                    transitions,
                    ribbon_keep_source_cumfrac,
                )
                if transitions.empty:
                    continue
            if ribbon_count_quantile is not None and len(transitions) > 1:
                try:
                    ribbon_cut = float(transitions["count"].quantile(ribbon_count_quantile))
                except Exception:
                    ribbon_cut = None
            else:
                ribbon_cut = None
            if ribbon_top_k is not None:
                transitions = transitions.sort_values("count", ascending=False).head(int(ribbon_top_k))
            for _, row in transitions.iterrows():
                anc = row["ancestor"] if (lineage_anchor_mode and "ancestor" in row) else None
                src, tgt, count = row["source"], row["target"], row["count"]
                flow_val = row["value"]
                if ribbon_cut is not None and count < ribbon_cut:
                    continue
                if focus_labels:
                    if ribbon_focus_source_only:
                        if src not in focus_labels:
                            continue
                    elif ribbon_focus_target_only:
                        if tgt not in focus_labels:
                            continue
                    elif src not in focus_labels and tgt not in focus_labels:
                        continue
                anchor_pair = _get_anchor_cross_layer(t1_key, t2_key, src, tgt, z_values[t_idx], z_values[t_idx + 1])
                if anchor_pair is None:
                    continue
                c1, c2 = anchor_pair
                use_focus_local = focus_anchor_label and (src == focus_anchor_label or tgt == focus_anchor_label)
                _add_endpoint_node(t1_key, src, use_focus_local=use_focus_local)
                _add_endpoint_node(t2_key, tgt, use_focus_local=use_focus_local)
                base_color_key = anc if anc is not None else src
                c_src = label_to_color.get(base_color_key, "#888")
                c_tgt = label_to_color.get(tgt, "#888")
                if ribbon_render_mode == "line":
                    line_w = ribbon_line_width_base + np.log1p(float(flow_val)) * ribbon_line_width_scale
                    line_w = max(0.5, float(line_w))
                    line_color = to_valid_color(c_src, default_alpha=ribbon_line_alpha)
                    if ribbon_line_curve and ribbon_line_curve > 0:
                        n_pts = max(2, int(ribbon_line_points))
                        t_vals = np.linspace(0.0, 1.0, n_pts)
                        x_path = c1["x"] + t_vals * (c2["x"] - c1["x"])
                        y_path = c1["y"] + t_vals * (c2["y"] - c1["y"])
                        z_path = c1["z"] + t_vals * (c2["z"] - c1["z"])
                        dx = c2["x"] - c1["x"]
                        dy = c2["y"] - c1["y"]
                        norm = np.hypot(dx, dy) + 1e-6
                        px, py = -dy / norm, dx / norm
                        bend = np.sin(np.pi * t_vals) * ribbon_line_curve * norm
                        x_path = x_path + px * bend
                        y_path = y_path + py * bend
                        fig.add_trace(
                            go.Scatter3d(
                                x=x_path,
                                y=y_path,
                                z=z_path,
                                mode="lines",
                                line=dict(width=line_w, color=line_color),
                                showlegend=False,
                                hoverinfo="text",
                                hovertext=(
                                    f"Ancestor: {anc if anc is not None else 'NA'}<br>"
                                    f"Fate: {src} -> {tgt}<br>Flow: {flow_val:.4f}"
                                ),
                            )
                        )
                        if ribbon_center_highlight:
                            hl_color = _lighten_color(
                                line_color,
                                amount=0.6,
                                alpha=ribbon_center_highlight_alpha,
                                default_alpha=0.9,
                            )
                            hl_w = max(0.3, float(line_w) * ribbon_center_highlight_width_scale)
                            fig.add_trace(
                                go.Scatter3d(
                                    x=x_path,
                                    y=y_path,
                                    z=z_path,
                                    mode="lines",
                                    line=dict(width=hl_w, color=hl_color),
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )
                    else:
                        fig.add_trace(
                            go.Scatter3d(
                                x=[c1["x"], c2["x"]],
                                y=[c1["y"], c2["y"]],
                                z=[c1["z"], c2["z"]],
                                mode="lines",
                                line=dict(width=line_w, color=line_color),
                                showlegend=False,
                                hoverinfo="text",
                                hovertext=(
                                    f"Ancestor: {anc if anc is not None else 'NA'}<br>"
                                    f"Fate: {src} -> {tgt}<br>Flow: {flow_val:.4f}"
                                ),
                            )
                        )
                        if ribbon_center_highlight:
                            hl_color = _lighten_color(
                                line_color,
                                amount=0.6,
                                alpha=ribbon_center_highlight_alpha,
                                default_alpha=0.9,
                            )
                            hl_w = max(0.3, float(line_w) * ribbon_center_highlight_width_scale)
                            fig.add_trace(
                                go.Scatter3d(
                                    x=[c1["x"], c2["x"]],
                                    y=[c1["y"], c2["y"]],
                                    z=[c1["z"], c2["z"]],
                                    mode="lines",
                                    line=dict(width=hl_w, color=hl_color),
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )
                else:
                    ribbon = create_ribbon_surface(c1, c2, flow_val, c_src, c_tgt, n_points=ribbon_resolution)
                    fig.add_trace(
                        go.Mesh3d(
                            x=ribbon["x"],
                            y=ribbon["y"],
                            z=ribbon["z"],
                            i=ribbon["i"],
                            j=ribbon["j"],
                            k=ribbon["k"],
                            vertexcolor=ribbon["colors"],
                            opacity=0.85,
                            showlegend=False,
                            hoverinfo="text",
                            hovertext=(
                                f"Ancestor: {anc if anc is not None else 'NA'}<br>"
                                f"Fate: {src} -> {tgt}<br>Flow: {flow_val:.4f}"
                            ),
                        )
                    )

    if highlight_endpoints and endpoint_nodes:
        pts = list(endpoint_nodes.values())
        fig.add_trace(
            go.Scatter3d(
                x=[p["x"] for p in pts],
                y=[p["y"] for p in pts],
                z=[p["z"] for p in pts],
                mode="markers",
                marker=dict(size=endpoint_size, color=[p["color"] for p in pts], line=dict(width=0.6, color="white")),
                showlegend=False,
                hoverinfo="text",
                hovertext=[p["label"] for p in pts],
            )
        )

    if show_legend:
        for ct in all_types:
            raw_c = label_to_color.get(ct, "#888888")
            clean_c = to_valid_color(raw_c, default_alpha=1.0)
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=10, color=clean_c),
                    name=ct,
                    showlegend=True,
                )
            )

    zaxis_cfg = dict(
        showbackground=False,
        showgrid=False,
        showline=False,
        tickvals=z_values if show_time_axis else [],
        ticktext=time_keys if show_time_axis else [],
        showticklabels=show_time_axis,
        title=dict(text="Time", font=dict(color=font_color)) if show_time_axis else dict(text=""),
        tickfont=dict(color=font_color),
    )
    fig.update_layout(
        scene_camera={"projection": {"type": "orthographic"}},
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font=dict(color=font_color),
        title=dict(text="3D Spatial Fate & Communication", x=0.5, y=0.9) if show_title else None,
        showlegend=show_legend,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=zaxis_cfg,
            bgcolor=background_color,
            camera=dict(eye=dict(x=1.8, y=1.2, z=1.0)),
            aspectmode="manual",
            aspectratio=dict(x=1.5, y=1, z=1.2),
        ),
        width=width,
        height=height,
    )

    if out_html:
        fig.write_html(out_html)
    return fig


# Backwards-compatible alias (historical name from ST-1104 notebooks/scripts)
plot_3d_spatial_sankey_style_focus_anchor = plot_3d_spatial_sankey_style
