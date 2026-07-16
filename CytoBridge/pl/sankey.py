"""Sankey diagram and cell lineage visualization.

This module provides functions for creating Sankey diagrams and 3D spatiotemporal
communication plots using Plotly.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .spatiotemporal_sankey import plot_3d_spatial_sankey_style

__all__ = [
    "plot_sankey",
    "plot_3d_spatial_sankey",
    "plot_3d_spatial_sankey_style",
]


def plot_sankey(
    predicted_labels_list: Sequence[Sequence[str]],
    out_html: Optional[str] = None,
    start_index: int = 0,
    focus_source_label: Optional[str] = None,
    focus_target_label: Optional[str] = None,
    include_labels: Optional[Sequence[str]] = None,
    time_keys: Optional[Sequence[str]] = None,
    show_time_axis: bool = False,
    time_axis_y: float = -0.08,
    normalize_mode: Optional[str] = None,
    min_flow: Optional[float] = None,
    keep_source_cumfrac: Optional[float] = None,
    label_to_color: Optional[Dict[str, str]] = None,
    lineage_anchor_mode: bool = False,
    anchor_time_index: int = 0,
    style: str = "default",
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    """Create a Sankey diagram showing cell lineage transitions.
    
    Parameters
    ----------
    predicted_labels_list : Sequence[Sequence[str]]
        List of label arrays for each time point.
    out_html : str, optional
        Path to save HTML output.
    start_index : int
        Starting time index for the diagram.
    focus_source_label : str, optional
        Filter to only show transitions from this source label.
    focus_target_label : str, optional
        Filter to only show transitions to this target label.
    include_labels : Sequence[str], optional
        Only include these labels in the diagram.
    time_keys : Sequence[str], optional
        Labels for time points.
    show_time_axis : bool
        Whether to display time axis.
    time_axis_y : float
        Y position for time axis.
    normalize_mode : str, optional
        Normalization mode: None, 'source', or 'global'.
    min_flow : float, optional
        Minimum flow threshold to display.
    label_to_color : Dict[str, str], optional
        Mapping of labels to colors.
    lineage_anchor_mode : bool
        Whether to use lineage anchor mode.
    anchor_time_index : int
        Time index to use as anchor.
    title : str
        Title for the diagram.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object.
    """
    import pandas as pd
    import plotly.graph_objects as go
    
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    
    labels_list = [list(x) for x in predicted_labels_list]
    time_keys_list = list(time_keys) if time_keys is not None else None
    if time_keys_list is not None:
        min_len = min(len(time_keys_list), len(labels_list))
        time_keys_list = time_keys_list[:min_len]
        labels_list = labels_list[:min_len]
    
    if normalize_mode not in (None, "source", "global"):
        raise ValueError("normalize_mode must be None, 'source', or 'global'")
    if min_flow is not None and min_flow < 0:
        raise ValueError("min_flow must be >= 0")
    if keep_source_cumfrac is not None:
        keep_source_cumfrac = float(keep_source_cumfrac)
        if not 0.0 < keep_source_cumfrac <= 1.0:
            raise ValueError("keep_source_cumfrac must be in (0, 1].")
    if style not in {"default", "nature-methods"}:
        raise ValueError("style must be 'default' or 'nature-methods'.")
    
    def _align_for_anchor(lbl_list: Sequence[Sequence[str]], anchor_idx: int):
        arrs = [pd.Series(list(x)).astype(str).values for x in lbl_list]
        if not arrs:
            return [], None
        min_len_arr = min(len(a) for a in arrs)
        arrs = [a[:min_len_arr] for a in arrs]
        if anchor_idx < 0 or anchor_idx >= len(arrs):
            raise ValueError("anchor_time_index out of range.")
        return arrs, arrs[anchor_idx]
    
    links = []
    num_timepoints = len(labels_list)
    if start_index >= num_timepoints - 1:
        raise ValueError("start_index must be <= len(timepoints) - 2")
    
    anchor_labels = None
    labels_aligned = labels_list
    if lineage_anchor_mode:
        labels_aligned, anchor_labels = _align_for_anchor(labels_list, anchor_time_index)
        if anchor_labels is None or len(anchor_labels) == 0:
            raise ValueError("No labels available for ancestor-coupled Sankey.")
    else:
        labels_aligned = [list(x) for x in labels_list]

    def _filter_keep_source_cumfrac(df, cumfrac: float, group_cols: Sequence[str]):
        keep_indices = []
        for _, group in df.groupby(list(group_cols), sort=False):
            group = group.sort_values("value", ascending=False)
            total = float(group["value"].sum())
            if total <= 0:
                continue
            cumulative = (group["value"].cumsum() / total).to_numpy()
            keep_n = int(np.searchsorted(cumulative, cumfrac, side="left")) + 1
            keep_indices.extend(group.head(max(1, min(len(group), keep_n))).index.tolist())
        return df.loc[keep_indices] if keep_indices else df.iloc[0:0]
    
    if focus_source_label is not None:
        min_len_focus = min(len(labels_aligned[t]) for t in range(start_index, num_timepoints))
        if min_len_focus == 0:
            raise ValueError("No labels available for focus_source_label filtering.")
        base = pd.Series(labels_aligned[start_index][:min_len_focus])
        mask = base == focus_source_label
        if not mask.any():
            raise ValueError(f"No entries found for focus_source_label='{focus_source_label}'.")
        for t in range(start_index, num_timepoints):
            labels_t = pd.Series(labels_aligned[t][:min_len_focus])
            labels_aligned[t] = labels_t[mask].tolist()
        if lineage_anchor_mode and anchor_labels is not None:
            anchor_labels = anchor_labels[:min_len_focus][mask.values]
    
    # Build link data
    for t in range(start_index, num_timepoints - 1):
        src = list(labels_aligned[t])
        tgt = list(labels_aligned[t + 1])
        min_len_t = min(len(src), len(tgt))
        if min_len_t == 0:
            continue
        
        if lineage_anchor_mode:
            anc = anchor_labels[:min_len_t] if anchor_labels is not None else ["anc"] * min_len_t
            df = pd.DataFrame({"ancestor": anc, "source": src[:min_len_t], "target": tgt[:min_len_t]})
            if focus_target_label is not None:
                df = df[df["target"] == focus_target_label]
            if include_labels:
                df = df[df["source"].isin(include_labels) | df["target"].isin(include_labels)]
            if df.empty:
                continue
            group_cols = ["ancestor", "source", "target"]
            counts = df.groupby(group_cols).size().reset_index(name="value")
            if normalize_mode == "source":
                counts["value"] = counts["value"] / counts.groupby(["ancestor", "source"])["value"].transform("sum")
            elif normalize_mode == "global":
                total = counts["value"].sum()
                if total > 0:
                    counts["value"] = counts["value"] / total
            if keep_source_cumfrac is not None:
                counts = _filter_keep_source_cumfrac(
                    counts,
                    keep_source_cumfrac,
                    group_cols=["ancestor", "source"],
                )
                if counts.empty:
                    continue
            if min_flow is not None:
                counts = counts[counts["value"] >= min_flow]
                if counts.empty:
                    continue
            counts["source"] = counts.apply(
                lambda r: f"{r['ancestor']}->{r['source']}__T{t + 1}", axis=1
            )
            counts["target"] = counts.apply(
                lambda r: f"{r['ancestor']}->{r['target']}__T{t + 2}", axis=1
            )
        else:
            df = pd.DataFrame({"source": src[:min_len_t], "target": tgt[:min_len_t]})
            if focus_target_label is not None:
                df = df[df["target"] == focus_target_label]
            if include_labels:
                df = df[df["source"].isin(include_labels) | df["target"].isin(include_labels)]
            if df.empty:
                continue
            counts = df.groupby(["source", "target"]).size().reset_index(name="value")
            if normalize_mode == "source":
                counts["value"] = counts["value"] / counts.groupby("source")["value"].transform("sum")
            elif normalize_mode == "global":
                total = counts["value"].sum()
                if total > 0:
                    counts["value"] = counts["value"] / total
            if keep_source_cumfrac is not None:
                counts = _filter_keep_source_cumfrac(
                    counts,
                    keep_source_cumfrac,
                    group_cols=["source"],
                )
                if counts.empty:
                    continue
            if min_flow is not None:
                counts = counts[counts["value"] >= min_flow]
                if counts.empty:
                    continue
            counts["source"] = counts["source"].astype(str) + f"__T{t + 1}"
            counts["target"] = counts["target"].astype(str) + f"__T{t + 2}"
        links.append(counts)
    
    if not links:
        raise ValueError("No valid transitions to plot: all timepoint label arrays are empty.")
    
    all_links_df = pd.concat(links, axis=0)
    all_nodes = pd.unique(all_links_df[["source", "target"]].values.ravel("K"))
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    all_links_df["source_idx"] = all_links_df["source"].map(node_indices)
    all_links_df["target_idx"] = all_links_df["target"].map(node_indices)
    
    def _split_node(node_id: str) -> str:
        return node_id.rsplit("__T", 1)[0] if "__T" in node_id else node_id
    
    def _parse_node(node_id: str):
        core = _split_node(node_id)
        if "->" in core:
            anc, curr = core.split("->", 1)
        else:
            anc, curr = core, core
        return anc, curr
    
    base_types = sorted({_split_node(node) for node in all_nodes})
    if lineage_anchor_mode:
        base_types = sorted({_parse_node(node)[0] for node in all_nodes})
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    color_map = {base_type: color_palette[i % len(color_palette)] for i, base_type in enumerate(base_types)}
    
    def _base_color(node_label: str) -> str:
        anc, _ = _parse_node(node_label)
        base = anc if lineage_anchor_mode else _split_node(node_label)
        if label_to_color and base in label_to_color:
            return label_to_color[base]
        return color_map.get(base, "#888888")
    
    def _sanitize_color(color: str, alpha: float) -> str:
        if not isinstance(color, str):
            return f"rgba(136,136,136,{alpha})"
        c = color.strip()
        if c.startswith("#") and len(c) == 9:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            a = int(c[7:9], 16) / 255.0
            if alpha < 1.0:
                a = alpha
            return f"rgba({r},{g},{b},{a})"
        if c.startswith("#") and len(c) == 7:
            if alpha >= 1.0:
                return c
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"
        return c
    
    node_labels = []
    for node in all_nodes:
        if lineage_anchor_mode:
            anc, curr = _parse_node(node)
            node_labels.append(f"{curr} | anc {anc}")
        else:
            node_labels.append(_split_node(node))
    node_colors = [_sanitize_color(_base_color(node), alpha=1.0) for node in all_nodes]
    
    # Position nodes
    time_for_node: Dict[str, int] = {}
    for node in all_nodes:
        if "__T" in node:
            time_for_node[node] = int(node.rsplit("__T", 1)[1])
        else:
            time_for_node[node] = start_index + 1
    time_groups: Dict[int, List[str]] = {}
    for node, t_idx in time_for_node.items():
        time_groups.setdefault(t_idx, []).append(node)
    node_y = []
    for node in all_nodes:
        t_idx = time_for_node[node]
        group = time_groups.get(t_idx, [node])
        pos = group.index(node)
        if len(group) == 1:
            node_y.append(0.5)
        else:
            node_y.append(pos / (len(group) - 1))
    
    link_colors = []
    link_alpha = 0.4 if style == "nature-methods" else 0.5
    for src_idx in all_links_df["source_idx"]:
        source_node_label = all_nodes[src_idx]
        base_color = _base_color(source_node_label)
        link_colors.append(_sanitize_color(base_color, alpha=link_alpha))
    
    displayed_steps = num_timepoints - start_index
    if displayed_steps < 2:
        displayed_steps = 2
    
    def _node_x(node_id: str) -> float:
        if "__T" in node_id:
            time_idx = int(node_id.rsplit("__T", 1)[1]) - (start_index + 1)
        else:
            time_idx = 0
        if style == "nature-methods":
            return 0.05 + (time_idx / (displayed_steps - 1)) * 0.9
        return time_idx / (displayed_steps - 1)
    
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap" if style == "nature-methods" else "fixed",
                node=dict(
                    pad=15 if style == "nature-methods" else 20,
                    thickness=20 if style == "nature-methods" else 25,
                    line=dict(color="black", width=0.5),
                    label=node_labels,
                    color=node_colors,
                    x=[_node_x(node) for node in all_nodes],
                    **({} if style == "nature-methods" else {"y": node_y}),
                ),
                link=dict(
                    source=all_links_df["source_idx"],
                    target=all_links_df["target_idx"],
                    value=all_links_df["value"],
                    color=link_colors,
                ),
            )
        ]
    )
    
    if title is None:
        title = "Cell Fate Transitions" if style == "nature-methods" else "Cell Lineage Sankey"
    layout_kwargs = dict(
        font_family="Arial",
        font_size=12,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    if style == "nature-methods":
        layout_kwargs["title"] = dict(text=title, x=0.5, xanchor="center")
        layout_kwargs["width"] = 1600 if width is None else int(width)
        layout_kwargs["height"] = 1000 if height is None else int(height)
    else:
        layout_kwargs["title_text"] = title
        if width is not None:
            layout_kwargs["width"] = int(width)
        if height is not None:
            layout_kwargs["height"] = int(height)
    fig.update_layout(**layout_kwargs)
    
    if show_time_axis:
        if time_keys_list is None:
            time_keys_list = [f"T{i + 1}" for i in range(num_timepoints)]
        shown_keys = time_keys_list[start_index:]
        if len(shown_keys) >= 1:
            for idx, tk in enumerate(shown_keys):
                x_pos = idx / max(1, len(shown_keys) - 1)
                fig.add_annotation(
                    x=x_pos,
                    y=time_axis_y,
                    xref="paper",
                    yref="paper",
                    text=str(tk),
                    showarrow=False,
                    font=dict(size=12, color="black"),
                )
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=time_axis_y + 0.02,
                y1=time_axis_y + 0.02,
                xref="paper",
                yref="paper",
                line=dict(color="black", width=1),
            )
            fig.update_layout(margin=dict(b=80))
    
    if out_html:
        fig.write_html(out_html)
    return fig


def plot_3d_spatial_sankey(
    adata_dict: Dict,
    communications: Dict,
    time_keys: Sequence,
    label_to_color: Dict[str, str],
    predicted_labels_list: Sequence[Sequence[str]],
    spatial_key: str = "spatial",
    z_spacing: float = 3.0,
    point_size: float = 1.0,
    point_alpha: float = 0.6,
    ribbon_min_count: Optional[float] = None,
    ribbon_top_k: Optional[int] = None,
    width: int = 1400,
    height: int = 1000,
    background_color: str = "white",
    out_html: Optional[str] = None,
    title: str = "3D Spatiotemporal Communication",
):
    """Create a 3D spatiotemporal communication plot (ST-1104 style wrapper)."""
    fig = plot_3d_spatial_sankey_style(
        adata_dict=adata_dict,
        all_time_communications=communications,
        time_keys=time_keys,
        label_to_color=label_to_color,
        predicted_labels_list=predicted_labels_list,
        spatial_key=spatial_key,
        z_spacing=z_spacing,
        point_size=point_size,
        point_alpha=point_alpha,
        ribbon_min_count=ribbon_min_count,
        ribbon_top_k=ribbon_top_k,
        width=width,
        height=height,
        background_color=background_color,
        out_html=out_html,
        show_time_axis=True,
        show_legend=True,
        show_title=bool(title),
    )
    if title:
        try:
            fig.update_layout(title=title)
        except Exception:
            pass
    return fig
