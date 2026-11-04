"""Port of `evaluation/mosta_code/mosta_velocity.ipynb` onto the Arista dataset.

This module keeps the original notebook's logical steps (config/model/data load,
annotation merge, velocity decomposition, stream plots with optional communication
overlay) but exposes them as functions for reuse in scripts and notebooks.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import seaborn as sns
import torch
import anndata as ad

from evaluation.arista_code import arista_helpers as helpers
from DeepRUOT.interaction import cal_interaction


def load_velocity_inputs(
    config_path: str,
    annotation_csv: str,
    color_h5ad: Optional[str] = None,
    label_to_color_json: Optional[str] = None,
    annotation_key: str = "Annotation",
    annotation_mapping: Optional[Dict[str, str]] = None,
    mapped_column: str = "telencephalon",
) -> Dict:
    """Load config, dataset (with annotation), model weights, and optional palette."""
    config = helpers.load_config(config_path)
    df, csv_path = helpers.load_arista_df(config)
    dim = config["data"]["dim"]
    df = df.iloc[:, : dim + 1].copy()

    if "samples" in df.columns:
        df["samples"] = df["samples"].astype(float)

    ann_used = None
    if annotation_csv and os.path.exists(annotation_csv):
        ann_df = pd.read_csv(annotation_csv)
        if len(ann_df) != len(df):
            raise ValueError(
                f"Annotation CSV rows ({len(ann_df)}) do not match base df rows ({len(df)})."
            )
        df = df.copy()
        df[annotation_key] = ann_df[annotation_key].values
        ann_used = annotation_csv
    elif annotation_key not in df.columns:
        raise FileNotFoundError(f"Annotation column '{annotation_key}' missing and CSV not found: {annotation_csv}")

    if annotation_mapping is not None and annotation_key in df.columns:
        df[mapped_column] = df[annotation_key].map(annotation_mapping).fillna("Other")

    label_to_color = None
    if label_to_color_json and os.path.exists(label_to_color_json):
        try:
            with open(label_to_color_json, "r", encoding="utf-8") as handle:
                label_to_color = json.load(handle)
        except Exception as exc:  # pragma: no cover - only logs in notebook
            print(f"Color map load failed from {label_to_color_json}: {exc}")

    if color_h5ad and os.path.exists(color_h5ad):
        try:
            adata_color = sc.read(color_h5ad)
            colors_key = f"{annotation_key}_colors"
            if annotation_key in adata_color.obs and colors_key in adata_color.uns:
                cats = list(adata_color.obs[annotation_key].cat.categories)
                cols = list(adata_color.uns[colors_key])
                if label_to_color is None:
                    label_to_color = dict(zip(cats, cols))
            if label_to_color is None:
                fallback_keys = [
                    annotation_key,
                    annotation_key.lower(),
                    "bin_annotation",
                    "annotation",
                    "Annotation",
                ]
                color_key = next((k for k in fallback_keys if k in adata_color.obs), None)
                value_key = next((k for k in ("colors", "color", "Color") if k in adata_color.obs), None)
                if color_key and value_key:
                    series = adata_color.obs[color_key].astype(str)
                    if hasattr(adata_color.obs[color_key], "cat"):
                        categories = list(adata_color.obs[color_key].cat.categories.astype(str))
                    else:
                        categories = sorted(series.unique())
                    label_to_color = {}
                    for cat in categories:
                        mask = series == cat
                        if not mask.any():
                            continue
                        label_to_color[cat] = str(adata_color.obs.loc[mask, value_key].iloc[0])
        except Exception as exc:  # pragma: no cover - only logs in notebook
            print(f"Color map load failed from {color_h5ad}: {exc}")

    f_net, score_net, exp_dir, device = helpers.load_models(config)

    return {
        "config": config,
        "df": df,
        "csv_path": csv_path,
        "annotation_csv": ann_used,
        "label_to_color": label_to_color,
        "dim": dim,
        "f_net": f_net,
        "score_net": score_net,
        "exp_dir": exp_dir,
        "device": device,
    }


def ensure_valid_palette(adata, color_key: str, provided_palette: Optional[Dict[str, str]] = None):
    """Ensure every category in `color_key` has a color entry."""
    if color_key not in adata.obs:
        return None

    categories = adata.obs[color_key].unique()
    categories = categories[~pd.isna(categories)]

    if provided_palette is not None and isinstance(provided_palette, dict):
        missing = [cat for cat in categories if cat not in provided_palette]
        if not missing:
            return provided_palette
        print(f"Warning: Provided palette missing keys for {missing}. Generating new palette.")

    n_cats = len(categories)
    if n_cats <= 20:
        colors = sc.pl.palettes.vega_20
    elif n_cats <= 28:
        colors = sc.pl.palettes.zeileis_28
    else:
        colors = sc.pl.palettes.godsnot_102

    if n_cats > len(colors):
        colors = list(colors) * (n_cats // len(colors) + 1)

    return {cat: colors[i] for i, cat in enumerate(categories)}


def plot_single_velocity_field(
    adata,
    velocity_key: str,
    density: float,
    figsize: Tuple[int, int],
    flip_y: bool,
    flip_x: bool,
    title: str,
    color_key: str,
    mode: str = "default",
    remove_outliers: bool = True,
    timepoint_str: Optional[str] = None,
    plot_region: Optional[Sequence[float]] = None,
    palette: Optional[Dict[str, str]] = None,
    scvelo_default_style: bool = False,
    **kwargs,
):
    """Helper to stream-plot a single velocity field (intrinsic or interaction)."""
    alpha_val = kwargs.get("alpha", 0.25)

    if mode == "black":
        plt.style.use("dark_background")
        background_color, text_color = "black", "white"
    else:
        plt.style.use("default")
        background_color, text_color = "white", "black"

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    adata_plot = adata.copy()
    if remove_outliers:
        y = adata_plot.obsm["X_spatial"][:, 1]
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
        adata_plot = adata_plot[mask].copy()

    if plot_region is not None:
        x_min, x_max, y_min, y_max = plot_region
        X = adata_plot.obsm["X_spatial"]
        mask = np.ones(len(X), dtype=bool)
        if x_min is not None:
            mask &= X[:, 0] > x_min
        if x_max is not None:
            mask &= X[:, 0] < x_max
        if y_min is not None:
            mask &= X[:, 1] > y_min
        if y_max is not None:
            mask &= X[:, 1] < y_max
        adata_plot = adata_plot[mask].copy()
        print(f"  Zoom-in subset: {len(adata_plot)} cells remaining.")

    point_size = 50 if plot_region is None else 60
    # scvelo's internal default chooses n_neighbors ~ int(n_obs/50), which can become 0
    # for very small subsets (e.g. after filtering/zoom). Guard to keep it >= 1.
    stream_n_neighbors = max(1, min(30, int(adata_plot.n_obs) - 1)) if int(adata_plot.n_obs) > 1 else 1

    if scvelo_default_style:
        # Keep scVelo defaults for a more canonical look.
        # Still pass a safe `n_neighbors` to avoid scvelo's internal default becoming 0
        # on very small subsets (which raises in sklearn).
        scv.pl.velocity_embedding_stream(
            adata_plot,
            basis="spatial",
            vkey=velocity_key,
            color=color_key,
            palette=palette,
            ax=ax,
            show=False,
            density=density,
            n_neighbors=stream_n_neighbors,
            title="",
        )
    else:
        scv.pl.velocity_embedding_stream(
            adata_plot,
            basis="spatial",
            vkey=velocity_key,
            color=color_key,
            palette=palette,
            ax=ax,
            show=False,
            density=density,
            smooth=0.8,
            min_mass=1,
            cutoff_perc=3,
            linewidth=1.5,
            arrow_size=1.2,
            n_neighbors=stream_n_neighbors,
            alpha=alpha_val,
            size=point_size,
            legend_loc="right margin",
            title="",
            frameon=False,
        )

    if flip_y:
        ax.invert_yaxis()
    if flip_x:
        ax.invert_xaxis()

    if plot_region is not None:
        x_min, x_max, y_min, y_max = plot_region
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        ax.set_xlim(x_min if x_min is not None else cur_xlim[0], x_max if x_max is not None else cur_xlim[1])
        target_ymin = y_min if y_min is not None else (cur_ylim[0] if not flip_y else cur_ylim[1])
        target_ymax = y_max if y_max is not None else (cur_ylim[1] if not flip_y else cur_ylim[0])
        if flip_y:
            ax.set_ylim(target_ymax, target_ymin)
        else:
            ax.set_ylim(target_ymin, target_ymax)

    full_title = f"{title} - {timepoint_str}" if timepoint_str else title
    ax.set_title(full_title, fontsize=20, fontweight="bold", color=text_color, pad=20)
    for spine in ax.spines.values():
        spine.set_color(text_color)
    ax.tick_params(colors=text_color, labelsize=12)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)

    return fig, ax


def velocity_fingerprint_stream_simple(
    df: pd.DataFrame,
    f_net,
    sf2m_score_model,
    timepoint_idx: float,
    dim: int = 52,
    space: str = "physical",
    velocity_type: str = "full",
    basis: str = "spatial",
    density: float = 2,
    figsize: Tuple[int, int] = (12, 10),
    flip_y: bool = True,
    flip_x: bool = False,
    n_neighbors: int = 30,
    mode: str = "default",
    remove_outliers: bool = True,
    timepoint_str: Optional[str] = None,
    plot_region: Optional[Sequence[float]] = None,
    cell_type: Optional[str] = None,
    color: str = "Annotation",
    palette: Optional[Dict[str, str]] = None,
    scvelo_default_style: bool = False,
    interaction_m: int = 1024,
    interaction_threshold: int = 1000,
    device: str = "cpu",
    **kwargs,
):
    """Compute intrinsic/interaction velocity and plot streamlines for one timepoint."""
    print(f"Processing timepoint {timepoint_idx} ({space} space, {velocity_type})")

    df_t = df[df["samples"] == timepoint_idx].copy()

    if color == "telencephalon":
        color_col_for_adata = "telencephalon"
    else:
        color_col_for_adata = color

    if cell_type is not None:
        if "Annotation" in df_t.columns and cell_type in df_t["Annotation"].values:
            df_t = df_t[df_t["Annotation"] == cell_type].copy()
            print(f"Filter: Found '{cell_type}' in column 'Annotation'")

    if color_col_for_adata not in df_t.columns:
        obs_color = ["Unspecified"] * len(df_t)
        df_t["_temp_color"] = "Unspecified"
        color_col_for_adata = "_temp_color"
    else:
        obs_color = df_t[color_col_for_adata].values

    all_data = df_t[[f"x{i}" for i in range(1, dim + 1)]].values
    coords = all_data[:, 0:2]
    X_expression = all_data[:, 2:]

    t_tensor = torch.full((all_data.shape[0], 1), fill_value=timepoint_idx, dtype=torch.float32, device=device)
    data_tensor = torch.tensor(all_data, dtype=torch.float32, device=device)
    with torch.no_grad():
        drift_full = f_net.v_net(t_tensor, data_tensor).detach().cpu().numpy()
    lnw = torch.log(torch.ones(data_tensor.shape[0], 1, device=device) / data_tensor.shape[0])
    with torch.no_grad():
        interaction_full = cal_interaction(
            data_tensor,
            lnw,
            f_net.interaction_net,
            torch.tensor([timepoint_idx], dtype=torch.float32, device=device),
            m=interaction_m,
            threshold=interaction_threshold,
        ).detach().cpu().numpy()

    V3_intr = drift_full[:, 0:2]
    V1_rna_intr = drift_full[:, 2:]
    V4_inter = interaction_full[:, 0:2]
    V2_rna_inter = interaction_full[:, 2:]

    if X_expression.shape[1] != V1_rna_intr.shape[1]:
        raise ValueError("Shape mismatch between expression matrix and velocity components.")

    if space == "physical":
        V_intr = V3_intr
        V_inter = V4_inter
        space_title = "Physical Space"
    elif space == "gene":
        print("  Gene space: Projecting high-dim velocity to 2D using scVelo...")
        space_title = "Gene Space"

        def run_scvelo_projection(V_high_dim):
            tmp_ad = ad.AnnData(X=X_expression)
            tmp_ad.layers["Ms"] = X_expression.copy()
            tmp_ad.layers["velocity"] = V_high_dim.copy()
            tmp_ad.obsm["X_spatial"] = coords.copy()
            sc.pp.neighbors(tmp_ad, n_neighbors=n_neighbors, use_rep="X")
            scv.tl.velocity_graph(tmp_ad, vkey="velocity", xkey="Ms", n_jobs=-1)
            scv.tl.velocity_embedding(tmp_ad, basis="spatial", vkey="velocity")
            return tmp_ad.obsm["velocity_spatial"]

        V_intr = run_scvelo_projection(V1_rna_intr)
        V_inter = run_scvelo_projection(V2_rna_inter)
    else:
        raise ValueError(f"Unknown space: {space}")

    if not np.isfinite(V_intr).all():
        nan_count = np.sum(~np.isfinite(V_intr))
        print(f"Warning: Found {nan_count} non-finite values in Intrinsic Velocity. Replacing with 0.")
        V_intr = np.nan_to_num(V_intr, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.isfinite(V_inter).all():
        nan_count = np.sum(~np.isfinite(V_inter))
        print(f"Warning: Found {nan_count} non-finite values in Interaction Velocity. Replacing with 0.")
        V_inter = np.nan_to_num(V_inter, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.isfinite(coords).all():
        print("Warning: Found non-finite values in Spatial Coords. Cleaning...")
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)

    adata = ad.AnnData(X=all_data)
    adata.obsm["X_spatial"] = coords
    adata.obsm["velocity_intrinsic_spatial"] = V_intr
    adata.obsm["velocity_interaction_spatial"] = V_inter

    adata.obs[color_col_for_adata] = obs_color
    adata.obs[color_col_for_adata] = adata.obs[color_col_for_adata].astype("category")

    final_palette = ensure_valid_palette(adata, color_col_for_adata, palette)

    fig1, ax1 = plot_single_velocity_field(
        adata,
        "velocity_intrinsic",
        density,
        figsize,
        flip_y,
        flip_x,
        f"{space_title} - Intrinsic Velocity",
        color_col_for_adata,
        mode,
        remove_outliers,
        timepoint_str,
        plot_region,
        final_palette,
        scvelo_default_style=scvelo_default_style,
        **kwargs,
    )

    fig2, ax2 = plot_single_velocity_field(
        adata,
        "velocity_interaction",
        density,
        figsize,
        flip_y,
        flip_x,
        f"{space_title} - Interaction Velocity",
        color_col_for_adata,
        mode,
        remove_outliers,
        timepoint_str,
        plot_region,
        final_palette,
        scvelo_default_style=scvelo_default_style,
        **kwargs,
    )

    return adata, [ax1, ax2], [fig1, fig2]


class VelocityAnalyzer:
    """Wraps velocity plotting and optional communication overlay."""

    def __init__(self, df: pd.DataFrame, f_net, sf2m_score_model, dim: int = 52):
        self.df = df
        self.f_net = f_net
        self.sf2m_score_model = sf2m_score_model
        self.dim = dim
        if "Annotation" not in df.columns:
            print("Warning: 'Annotation' column not found.")

    def plot_fingerprint(
        self,
        adata,
        timepoint: float,
        timepoint_str: str,
        all_time_communication: Optional[Dict] = None,
        space: str = "physical",
        cell_type: str = "celltype",
        background_cell_type: Optional[str] = None,
        save_path: Optional[str] = None,
        density: float = 2,
        figsize: Tuple[int, int] = (7, 5),
        n_neighbors: int = 30,
        mode: str = "default",
        remove_outliers: bool = True,
        plot_region: Optional[Sequence[float]] = None,
        flip_x: bool = False,
        communication: bool = True,
        color: str = "Annotation",
        label_to_color: Optional[Dict[str, str]] = None,
        comm_edge_threshold: Optional[float] = None,
        comm_edge_top_k: Optional[int] = None,
        comm_edge_top_k_focus_label: Optional[str] = None,
        comm_edge_weight_quantile: Optional[float] = None,
        comm_include_self_loops: bool = True,
        comm_centroid_top_n_y: Optional[int] = None,
        comm_centroid_top_n_y_exclude_types: Optional[Sequence[str]] = None,
        scvelo_default_style: bool = False,
        interaction_m: int = 1024,
        interaction_threshold: int = 1000,
        device: str = "cpu",
        **kwargs,
    ):
        target_focus_cell = cell_type
        if target_focus_cell == "celltype":
            target_focus_cell = None

        target_background_cell = background_cell_type
        print(f"\n=== Plotting Velocity (Bg: {target_background_cell}) & Comm (Focus: {target_focus_cell}) ===")

        bg_alpha = 0.35 if communication else 0.4

        ad_res, axes, figs = velocity_fingerprint_stream_simple(
            df=self.df,
            f_net=self.f_net,
            sf2m_score_model=self.sf2m_score_model,
            timepoint_idx=timepoint,
            dim=self.dim,
            space=space,
            basis="spatial",
            density=density,
            figsize=figsize,
            flip_y=False,
            flip_x=flip_x,
            n_neighbors=n_neighbors,
            cell_type=target_background_cell,
            mode=mode,
            remove_outliers=remove_outliers,
            timepoint_str=timepoint_str,
            plot_region=plot_region,
            color=color,
            palette=label_to_color,
            scvelo_default_style=scvelo_default_style,
            alpha=bg_alpha,
            interaction_m=interaction_m,
            interaction_threshold=interaction_threshold,
            device=device,
            **kwargs,
        )

        if communication and all_time_communication is not None:
            comm_data = all_time_communication.get(timepoint_str)
            if isinstance(comm_data, list) and len(comm_data) > 0:
                comm_data = comm_data[0]

            matrix = comm_data.get("M_per_source") if comm_data else None

            if matrix is not None and "types" in comm_data:
                use_focus_anchor_style = (
                    comm_edge_threshold is not None
                    or comm_edge_top_k is not None
                    or comm_edge_top_k_focus_label is not None
                    or comm_edge_weight_quantile is not None
                    or (not comm_include_self_loops)
                )

                types = [str(t) for t in list(comm_data["types"])]
                mat = np.asarray(matrix, dtype=float)
                type_to_idx = {t: i for i, t in enumerate(types)}

                df_ann = self.df.get("Annotation", pd.Series([], dtype=str)).astype(str)
                ann_set = set(df_ann.unique())
                valid_types = [t for t in types if t in ann_set]

                def _edges_focus_anchor_style() -> list[dict]:
                    threshold = float(comm_edge_threshold) if comm_edge_threshold is not None else 0.0
                    focus_edge = target_focus_cell
                    edges_this: list[dict] = []
                    self_loops: list[dict] = []
                    for src in valid_types:
                        i = type_to_idx.get(src)
                        if i is None:
                            continue
                        for tgt in valid_types:
                            j = type_to_idx.get(tgt)
                            if j is None:
                                continue

                            if focus_edge and src != focus_edge and tgt != focus_edge:
                                continue

                            if src == tgt:
                                if not comm_include_self_loops:
                                    continue

                            weight = float(mat[i, j])
                            if weight <= threshold:
                                continue
                            if src == tgt:
                                # Keep self-loops separate so they do not compete with top-K edge selection
                                # (matches focus-anchor behavior).
                                self_loops.append({"weight": weight, "type_from": src, "type_to": tgt})
                            else:
                                edges_this.append({"weight": weight, "type_from": src, "type_to": tgt})

                    if not edges_this and not self_loops:
                        return []

                    weights = np.asarray([e["weight"] for e in edges_this], dtype=float)
                    thresh = threshold
                    if comm_edge_weight_quantile is not None and len(weights) > 1:
                        try:
                            quant_cut = np.quantile(weights, float(comm_edge_weight_quantile))
                            thresh = max(thresh, float(quant_cut))
                        except Exception:
                            pass

                    edges_this = [e for e in edges_this if e["weight"] >= thresh]
                    edges_this = sorted(edges_this, key=lambda x: x["weight"], reverse=True)

                    if comm_edge_top_k_focus_label:
                        focus_lab = str(comm_edge_top_k_focus_label)
                        edges_this = [
                            e
                            for e in edges_this
                            if e["type_from"] == focus_lab or e["type_to"] == focus_lab
                        ]

                    if comm_edge_top_k is not None:
                        edges_this = edges_this[: int(comm_edge_top_k)]

                    # Always append self-loops (already threshold-filtered) after top-K selection.
                    # Ordering does not matter for drawing; linewidth normalizes by global max.
                    return edges_this + self_loops

                if use_focus_anchor_style:
                    edges_to_draw = _edges_focus_anchor_style()
                    nodes_to_draw = sorted({e["type_from"] for e in edges_to_draw} | {e["type_to"] for e in edges_to_draw})
                else:
                    comm_df = pd.DataFrame(mat, index=types, columns=types)
                    valid_nodes = [ct for ct in comm_df.index if str(ct) in ann_set]
                    if target_focus_cell is not None and target_focus_cell in comm_df.index:
                        weight_thresh = 1
                        keep_nodes = {target_focus_cell}
                        for ct in valid_nodes:
                            if ct == target_focus_cell:
                                continue
                            if max(
                                float(comm_df.loc[target_focus_cell, ct]),
                                float(comm_df.loc[ct, target_focus_cell]),
                            ) > weight_thresh:
                                keep_nodes.add(ct)
                        valid_nodes = [ct for ct in valid_nodes if ct in keep_nodes]
                    edges_to_draw = []
                    for src in valid_nodes:
                        for tgt in valid_nodes:
                            w = float(comm_df.loc[src, tgt])
                            if w <= 0:
                                continue
                            if target_focus_cell and (src != target_focus_cell and tgt != target_focus_cell):
                                continue
                            edges_to_draw.append({"weight": w, "type_from": str(src), "type_to": str(tgt)})
                    nodes_to_draw = sorted({e["type_from"] for e in edges_to_draw} | {e["type_to"] for e in edges_to_draw})

                if not edges_to_draw:
                    # Nothing to overlay; keep velocity plots as-is.
                    edges_to_draw = []

                custom_centroids = {}
                time_df = self.df[self.df["samples"] == timepoint]
                exclude_types = (
                    {str(x) for x in comm_centroid_top_n_y_exclude_types}
                    if comm_centroid_top_n_y_exclude_types is not None
                    else set()
                )
                for ct in nodes_to_draw:
                    ct_subs = time_df[time_df["Annotation"].astype(str) == str(ct)]
                    if len(ct_subs) == 0:
                        continue
                    # Optionally compute centroid using the top-N highest-y points (x2),
                    # optionally within the plot region. This can produce more visually
                    # stable node placement when cell distributions are vertically skewed.
                    if plot_region is not None:
                        x_min, x_max, y_min, y_max = plot_region
                        if x_min is not None:
                            ct_subs = ct_subs[ct_subs["x1"] > float(x_min)]
                        if x_max is not None:
                            ct_subs = ct_subs[ct_subs["x1"] < float(x_max)]
                        if y_min is not None:
                            ct_subs = ct_subs[ct_subs["x2"] > float(y_min)]
                        if y_max is not None:
                            ct_subs = ct_subs[ct_subs["x2"] < float(y_max)]
                        if len(ct_subs) == 0:
                            continue

                    if comm_centroid_top_n_y is not None and str(ct) not in exclude_types:
                        n_top = int(comm_centroid_top_n_y)
                        if n_top > 0 and len(ct_subs) > n_top:
                            ct_subs = ct_subs.nlargest(n_top, "x2")

                    custom_centroids[str(ct)] = ct_subs[["x1", "x2"]].mean().values

                # Drop edges whose endpoints do not exist at this timepoint.
                edges_to_draw = [
                    e
                    for e in edges_to_draw
                    if str(e["type_from"]) in custom_centroids and str(e["type_to"]) in custom_centroids
                ]
                if not edges_to_draw:
                    # Nothing to overlay after centroid filtering.
                    edges_to_draw = []

                if plot_region:
                    zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax = plot_region
                else:
                    zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax = None, None, None, None

                def _is_in_region(p):
                    if zoom_xmin is not None and (p[0] < zoom_xmin or p[0] > zoom_xmax):
                        return False
                    if zoom_ymin is not None and (p[1] < zoom_ymin or p[1] > zoom_ymax):
                        return False
                    return True

                if edges_to_draw:
                    max_weight = max(float(e["weight"]) for e in edges_to_draw)
                    max_weight = max(max_weight, 1e-12)

                    for ax in axes:
                        drawn_nodes = set()
                        for e in edges_to_draw:
                            src = str(e["type_from"])
                            tgt = str(e["type_to"])
                            weight = float(e["weight"])

                            p1, p2 = custom_centroids[src], custom_centroids[tgt]
                            if not (np.isfinite(p1).all() and np.isfinite(p2).all()):
                                continue
                            if not (_is_in_region(p1) and _is_in_region(p2)):
                                continue

                            lw = 1 + 10 * np.sqrt(weight / max_weight)
                            edge_color = label_to_color.get(src, "#333333") if label_to_color else "#333333"

                            if src == tgt:
                                start = (p1[0] - 0.03, p1[1] + 0.05)
                                end = (p1[0] + 0.06, p1[1] + 0.02)
                                rad, curr_mutation_scale = -10.0, 8
                            else:
                                start, end = (p1[0], p1[1]), (p2[0], p2[1])
                                rad, curr_mutation_scale = 0.15, 20

                            arrow = mpatches.FancyArrowPatch(
                                start,
                                end,
                                arrowstyle="-|>,head_length=0.8,head_width=0.5",
                                mutation_scale=curr_mutation_scale,
                                connectionstyle=f"arc3,rad={rad}",
                                color=edge_color,
                                linewidth=lw,
                                alpha=0.9,
                                zorder=30,
                            )
                            ax.add_patch(arrow)

                            for node_name, pos in [(src, p1), (tgt, p2)]:
                                if node_name not in drawn_nodes:
                                    node_c = label_to_color.get(node_name, "grey") if label_to_color else "grey"
                                    ax.scatter(
                                        pos[0],
                                        pos[1],
                                        s=2000,
                                        c=node_c,
                                        edgecolors="white",
                                        linewidth=2.5,
                                        zorder=31,
                                    )
                                    drawn_nodes.add(node_name)

        if save_path:
            base = save_path.replace(".pdf", "").replace(".svg", "")

            bg_s = f"_bg-{target_background_cell}" if target_background_cell else ""
            focus_s = f"_focus-{target_focus_cell}" if target_focus_cell else ""
            zoom_s = "_zoom" if plot_region else ""
            os.makedirs(os.path.dirname(base), exist_ok=True)

            figs[0].savefig(f"{base}{bg_s}{focus_s}{zoom_s}_intrinsic.pdf", dpi=300, bbox_inches="tight")
            figs[1].savefig(f"{base}{bg_s}{focus_s}{zoom_s}_interaction.pdf", dpi=300, bbox_inches="tight")
            print(f"Saved figures to {base}{bg_s}{focus_s}{zoom_s}_*.pdf")

        return ad_res, axes, figs


def save_palette_legend(palette: Dict[str, str], save_path: str):
    """Save a standalone legend PDF for a palette dictionary."""
    fig = plt.figure(figsize=(3, max(2, len(palette) * 0.35)))
    ax = fig.add_subplot(111)
    ax.axis("off")

    patches = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    ax.legend(
        handles=patches,
        loc="center",
        frameon=False,
        fontsize=10,
        handlelength=1.5,
        labelspacing=0.8,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close()
    return save_path


def run_velocity_workflow(
    config_path: str,
    output_dir: str,
    annotation_csv: str,
    color_h5ad: Optional[str] = None,
    label_to_color_json: Optional[str] = None,
    annotation_mapping: Optional[Dict[str, str]] = None,
    communication_data: Optional[Dict] = None,
    spaces: Sequence[str] = ("gene", "physical"),
    focus_celltype: Optional[str] = None,
    zoom_region: Optional[Sequence[float]] = None,
    interaction_m: int = 1024,
    interaction_threshold: int = 1000,
    scvelo_stream: bool = True,
) -> Dict:
    """Run the velocity analysis pipeline and return a manifest of outputs."""
    os.makedirs(output_dir, exist_ok=True)

    inputs = load_velocity_inputs(
        config_path=config_path,
        annotation_csv=annotation_csv,
        color_h5ad=color_h5ad,
        label_to_color_json=label_to_color_json,
        annotation_mapping=annotation_mapping,
    )
    df = inputs["df"]
    dim = inputs["dim"]
    f_net = inputs["f_net"]
    score_net = inputs["score_net"]
    device = inputs["device"]

    annotated_csv_out = os.path.join(output_dir, "velocity_input_with_annotation.csv")
    df.to_csv(annotated_csv_out, index=False)

    analyzer = VelocityAnalyzer(df, f_net, score_net, dim=dim)

    timepoints = sorted(df["samples"].unique())
    palette = inputs["label_to_color"]
    legend_path = None
    plot_paths: List[str] = []

    if palette:
        legend_path = save_palette_legend(palette, os.path.join(output_dir, "annotation_legend.pdf"))

    for idx, t in enumerate(timepoints):
        tp_str = str(t)
        for space in spaces:
            save_base = os.path.join(output_dir, f"velocity_{space}_t{idx}")

    scvelo_dir = os.path.join(output_dir, "scvelo_streams")
    scvelo_plots: List[str] = []
    if scvelo_stream:
        scvelo_plots = _plot_scvelo_streams(
            df=df,
            dim=dim,
            f_net=f_net,
            score_net=score_net,
            out_dir=scvelo_dir,
            label_to_color=palette,
            device=device,
        )

    manifest = {
        "config": config_path,
        "data_csv": inputs["csv_path"],
        "annotation_csv": annotation_csv,
        "annotated_csv": annotated_csv_out,
        "model_dir": inputs["exp_dir"],
        "plots": plot_paths,
        "scvelo_plots": scvelo_plots,
        "legend": legend_path,
        "timepoints": [str(t) for t in timepoints],
        "device": device,
    }
    return manifest


def _plot_scvelo_streams(
    df: pd.DataFrame,
    dim: int,
    f_net,
    score_net,
    out_dir: str,
    label_to_color: Optional[Dict[str, str]] = None,
    device: str = "cpu",
) -> List[str]:
    """Plot scVelo-style streamlines per timepoint using helper logic from arista_downstream_analysis."""
    os.makedirs(out_dir, exist_ok=True)
    plots: List[str] = []
    timepoints = sorted(df["samples"].unique())
    if label_to_color is None and "Annotation" in df.columns:
        all_labels = sorted(df["Annotation"].astype(str).unique())
        if len(all_labels) <= 20:
            colors = sc.pl.palettes.vega_20
        elif len(all_labels) <= 28:
            colors = sc.pl.palettes.zeileis_28
        else:
            colors = sc.pl.palettes.godsnot_102
        if len(all_labels) > len(colors):
            colors = list(colors) * (len(all_labels) // len(colors) + 1)
        label_to_color = {lab: colors[i] for i, lab in enumerate(all_labels)}
    elif label_to_color is not None and "Annotation" in df.columns:
        all_labels = set(df["Annotation"].astype(str).unique())
        missing = [lab for lab in all_labels if lab not in label_to_color]
        if missing:
            if len(missing) <= 20:
                colors = sc.pl.palettes.vega_20
            elif len(missing) <= 28:
                colors = sc.pl.palettes.zeileis_28
            else:
                colors = sc.pl.palettes.godsnot_102
            if len(missing) > len(colors):
                colors = list(colors) * (len(missing) // len(colors) + 1)
            for idx, lab in enumerate(sorted(missing)):
                label_to_color[lab] = colors[idx]

    for idx, t in enumerate(timepoints):
        data_t = df[df["samples"] == t].iloc[:, 1 : dim + 1].values
        coords = data_t[:, :2]
        labels_t = df[df["samples"] == t]["Annotation"].values if "Annotation" in df.columns else None
        vel = helpers.compute_velocity_components(data_t, float(t), f_net, score_net, device=device)

        def _save_component(arr: np.ndarray, name: str, use_labels=True):
            path = os.path.join(out_dir, f"velocity_scvelo_{name}_t{idx}.svg")
            helpers.plot_velocity_component(
                coords=coords,
                velocity=arr,
                labels=labels_t if use_labels else None,
                label_to_color=label_to_color,
                title=f"{name.replace('_', ' ').title()} (t={t})",
                out_path=path,
                basis="spatial",
                show_legend=True,
            )
            plots.append(path)

        _save_component(vel["drift"][:, :2], "spatial_intrinsic")
        _save_component(vel["interaction"][:, :2], "spatial_interaction")
        _save_component(vel["full"][:, :2], "spatial_full")

        _save_component(vel["drift"][:, 2:4], "gene_intrinsic")
        _save_component(vel["interaction"][:, 2:4], "gene_interaction")
        _save_component(vel["full"][:, 2:4], "gene_full")

    return plots
