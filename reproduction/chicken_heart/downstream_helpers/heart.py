"""Plotting functions copied from the collaborator heart analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

HEART_REPO_ROOT = Path.cwd()

import random
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_data_for_side_by_side_2d(
    adata_dict,
    time_keys,
    spatial_key='spatial',
    spacing=80,
    target_size=1000,
    label_to_color=None
):
    import numpy as np
    import anndata as ad

    all_x = []
    all_y = []
    temp_adata_dict = {}

    # 1) 收集所有切片，用于计算统一缩放比例
    for tk in time_keys:
        ada = adata_dict[tk].copy()
        coords = np.asarray(ada.obsm[spatial_key])
        all_x.extend(coords[:, 0])
        all_y.extend(coords[:, 1])
        temp_adata_dict[tk] = ada

    # 全局范围只用于计算统一 scale
    global_x_min = min(all_x)
    global_y_min = min(all_y)
    global_x_max = max(all_x)
    global_y_max = max(all_y)

    global_span = max(global_x_max - global_x_min, global_y_max - global_y_min)
    global_scale = target_size / global_span
    print(f"全局最大跨度: {global_span:.2f}, 统一缩放因子: {global_scale:.4f}")

    # 2) 逐切片处理：x 独立去左边距，y 做中心对齐
    cumulative_offset = 0.0
    final_adata_list = []

    for tk in time_keys:
        ada = temp_adata_dict[tk]
        coords = np.asarray(ada.obsm[spatial_key])

        x = coords[:, 0]
        y = coords[:, 1]

        # 统一缩放，但 x 先按每个切片自己的 xmin 归零
        scaled_x = (x - x.min()) * global_scale

        # y 也统一缩放，然后按每个切片自己的中心对齐
        scaled_y = y * global_scale
        current_y_center = 0.5 * (scaled_y.min() + scaled_y.max())
        final_y = scaled_y - current_y_center

        # 横向依次拼接
        final_x = scaled_x + cumulative_offset

        ada.obsm[spatial_key] = np.column_stack([final_x, final_y])

        current_x_span = scaled_x.max() - scaled_x.min()
        cumulative_offset += current_x_span + spacing

        ada.obs['time_key_for_concat'] = tk
        final_adata_list.append(ada)

    # 3) 合并
    adata_combined = ad.concat(
        final_adata_list,
        join="outer",
        axis=0,
        label="original_timepoint",
        keys=time_keys,
        fill_value=0
    )

    # 4) 绑定颜色
    if label_to_color is not None:
        adata_combined.obs['celltype_prediction'] = adata_combined.obs['celltype_prediction'].astype('category')
        merged_cats = adata_combined.obs['celltype_prediction'].cat.categories
        new_colors = [label_to_color[cat] for cat in merged_cats]
        adata_combined.uns['celltype_prediction_colors'] = np.array(new_colors)

    return adata_combined

def plot_side_by_side_spatial_with_custom_colors(
    adata_combined,
    color_by='celltype_prediction',
    time_key='timepoint',
    spot_size=50,
    figsize=(15, 7),
    save_path=None
):
    # --- 1. 检查 ---
    if color_by not in adata_combined.obs.columns:
        raise ValueError(f"列 '{color_by}' 不存在于 adata.obs 中。")
    if 'spatial' not in adata_combined.obsm:
        raise ValueError("obsm['spatial'] 坐标缺失。")
    if time_key not in adata_combined.obs.columns:
        raise ValueError(f"列 '{time_key}' 不存在于 adata.obs 中。")

    # --- 2. 创建 Figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- 3. 绘制空间图（关键：不显示标题） ---
    sc.pl.spatial(
        adata_combined,
        color=color_by,
        spot_size=spot_size,
        frameon=False,
        show=False,
        ax=ax,
        title=None          # ★ 关键：不显示 celltype_prediction
    )

    # 再保险一次，彻底清空标题
    ax.set_title("")

    ax.set_xlabel("X (Scaled and Shifted Coordinate)")
    ax.set_ylabel("Y (Scaled Coordinate)")

    # ====================================================
    # ★ 4. 顶部统一时间标注（字体区分 real / interp）
    # ====================================================

    real_timepoints = {"D4", "D7", "D10", "D14"}

    coords = adata_combined.obsm["spatial"]
    times = adata_combined.obs[time_key].values

    unique_times = sorted(np.unique(times), key=lambda x: int(x[1:]) if x[1:].isdigit() else float(x[1:]))

    y_max = coords[:, 1].max()
    y_min = coords[:, 1].min()
    y_range = y_max - y_min

    text_y = y_max + 0.05 * y_range

    for tp in unique_times:
        mask = times == tp
        if mask.sum() == 0:
            continue

        x_center = np.median(coords[mask, 0])

        is_real = tp in real_timepoints

        ax.text(
            x_center,
            text_y,
            tp,
            ha="center",
            va="bottom",
            fontsize=13,
            color="black" if is_real else "gray",
            fontweight="bold" if is_real else "normal",
            alpha=1.0 if is_real else 0.7
        )

    # --- 5. 扩展 y 轴，防止顶部被裁 ---
    ax.set_ylim(y_min, y_max + 0.12 * y_range)

    # --- 6. 保存 / 显示 ---
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"图像已保存到: {save_path}")

    plt.show()

def plot_celltype_stackbar(adata_dict, time_keys, label_to_color, annotation_key='celltype_prediction', save_path=None):
    """
    绘制每个时间点 cell type 构成比例的堆叠条形图，并给插值 stackbar 添加黑色虚线边框
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        style="white",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
        }
    )
    # --- 💡 关键参数设置 ---
    # width 控制柱子之间的距离。1.0 表示完全挨在一起，0.8-0.9 比较紧凑。
    custom_bar_width = 0.85
    # linewidth 控制块与块之间的白色边线。0.1 或 0 可以消除空隙。
    custom_linewidth = 0.15

    # 汇总 cell type 计数
    count_data = []
    for tk in time_keys:
        if tk not in adata_dict:
            continue
        ad = adata_dict[tk]
        ann_col = annotation_key if annotation_key in ad.obs.columns else 'annotation'
        if ann_col not in ad.obs.columns:
            print(f"Warning: Column '{ann_col}' not found in adata for time {tk}. Skipping.")
            continue

        ct_counts = ad.obs[ann_col].value_counts(normalize=True)
        for ct, frac in ct_counts.items():
            count_data.append({'time': tk, 'cell_type': ct, 'fraction': frac})

    if not count_data:
        print("No data collected for stackbar plot.")
        return

    df = pd.DataFrame(count_data)
    cell_types = sorted(df['cell_type'].unique()) # 所有的细胞类型

    plt.figure(figsize=(1.8*len(time_keys), 8))
    bottom_vals = pd.Series(0.0, index=time_keys)

    # 手动堆叠绘制条形图
    for ct in cell_types:
        subset = df[df['cell_type'] == ct].set_index('time').reindex(time_keys).fillna(0)

        color = label_to_color.get(ct, '#333333') # Default fallback

        plt.bar(time_keys,
                subset['fraction'],
                bottom=bottom_vals,
                label=ct,
                color=color,
                edgecolor='white',
                linewidth=custom_linewidth, # <--- 减小色块间隙
                width=custom_bar_width      # <--- 减小柱子间距离
        )
        bottom_vals += subset['fraction'].values

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel('Fraction', fontsize=12)
    plt.title('Cell Type Composition Across Time Points', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=7,title_fontsize=8, loc='upper left', title='Cell Type')
    sns.despine()
    plt.tight_layout()

    if save_path:
        # Auto add extension if needed, but usually provided
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stackbar plot saved to {save_path}")

    plt.show()

HEART_LABEL_TO_COLOR = {
    "Cardiomyocytes-1": "#FF8C00",
    "Cardiomyocytes-2": "#808000",
    "MT-enriched cardiomyocytes": "#004EB0",
    "Immature myocardial cells": "#FF90C9",
    "Vascular endothelial cells": "#8983BF",
    "Endocardial cells": "#f5cac3",
    "Fibroblast cells": "#8FBC8F",
    "Mural cells": "#A52A2A",
    "Macrophages": "#1CE6FF",
    "Erythrocytes": "#FF4A46",
    "Valve cells": "#71c33a",
    "Epi-epithelial cells": "#FFFF00",
    "TMSB4X high cells": "#00868B",
    "Cardiomyocytes": "#e8612C",
}

def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (HEART_REPO_ROOT / p).resolve()

def _day_from_label(label: str) -> float:
    return float(str(label).replace("D", ""))

def _prepare_g_heatmap_matrix(
    df_with_g: Optional[pd.DataFrame] = None,
    panels: Optional[Sequence[Dict[str, object]]] = None,
    *,
    time_col: str = "timepoint",
    label_col: str = "celltype_prediction",
    agg: str = "median",
    exclude_celltypes: Optional[Sequence[str]] = None,
    include_celltypes: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    exclude_celltypes = set(str(x) for x in (exclude_celltypes or []))
    include_celltypes = set(str(x) for x in (include_celltypes or []))

    if df_with_g is not None:
        df_long = df_with_g.copy()
        value_col = "g_value" if "g_value" in df_long.columns else "g"
        if value_col not in df_long.columns:
            raise ValueError("df_with_g must contain 'g_value' or 'g' column.")
        if label_col not in df_long.columns:
            if "celltype" in df_long.columns:
                df_long[label_col] = df_long["celltype"].astype(str)
            else:
                raise ValueError(f"df_with_g must contain '{label_col}' column.")
        if time_col not in df_long.columns:
            if "timepoint" in df_long.columns:
                df_long[time_col] = df_long["timepoint"].astype(str)
            else:
                raise ValueError(f"df_with_g must contain '{time_col}' column.")
        df_long = df_long[[label_col, time_col, value_col]].rename(columns={value_col: "g_value"}).copy()
    elif panels is not None:
        rows = []
        for panel in panels:
            tp = str(panel["label"])
            cts = np.asarray(panel["celltype"], dtype=str)
            gs = np.asarray(panel["g"], dtype=float)
            for ct, g in zip(cts, gs):
                rows.append({label_col: ct, time_col: tp, "g_value": float(g)})
        df_long = pd.DataFrame(rows)
    else:
        raise ValueError("Either df_with_g or panels must be provided.")

    if df_long.empty:
        raise ValueError("No g-value data available for heatmap plotting.")

    df_long[label_col] = df_long[label_col].astype(str)
    df_long[time_col] = df_long[time_col].astype(str)
    if include_celltypes:
        df_long = df_long[df_long[label_col].isin(include_celltypes)].copy()
    if exclude_celltypes:
        df_long = df_long[~df_long[label_col].isin(exclude_celltypes)].copy()

    if df_long.empty:
        raise ValueError("Filtering removed all data. Please check exclude_celltypes/include_celltypes.")

    agg = str(agg).lower()
    if agg not in {"median", "mean"}:
        raise ValueError(f"agg must be 'median' or 'mean', got {agg!r}")

    if agg == "median":
        df_stat = (
            df_long.groupby([label_col, time_col], as_index=False)["g_value"]
            .median()
        )
    else:
        df_stat = (
            df_long.groupby([label_col, time_col], as_index=False)["g_value"]
            .mean()
        )

    time_order = sorted(df_stat[time_col].unique(), key=_day_from_label)
    ct_order = (
        df_stat.groupby(label_col)["g_value"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    return (
        df_stat.pivot(index=label_col, columns=time_col, values="g_value")
        .reindex(index=ct_order, columns=time_order)
    )

def plot_network_evolution_global(
    adata_dict: Dict[str, ad.AnnData],
    all_time_communications: Dict[str, dict],
    time_keys: Sequence[str],
    label_to_color: Dict[str, str],
    exclude_types: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
):
    from matplotlib.patches import Circle, FancyArrowPatch

    exclude_types = [str(x) for x in (exclude_types or [])]
    n_times = len(time_keys)
    fig, axes = plt.subplots(1, n_times, figsize=(6 * n_times, 6))
    if n_times == 1:
        axes = [axes]

    target_order = [
        "Valve cells",
        "Epi-epithelial cells",
        "Mural cells",
        "Vascular endothelial cells",
        "Erythrocytes",
        "Immature myocardial cells",
        "Macrophages",
        "Cardiomyocytes-1",
        "Cardiomyocytes-2",
        "MT-enriched cardiomyocytes",
        "TMSB4X high cells",
        "Endocardial cells",
        "Fibroblast cells",
    ]

    all_types_raw = set()
    for tk in time_keys:
        all_types_raw.update(str(x) for x in all_time_communications[tk]["types"])

    available_types = [ct for ct in all_types_raw if ct not in exclude_types]
    all_types = [ct for ct in target_order if ct in available_types]
    others = [ct for ct in available_types if ct not in target_order]
    all_types.extend(sorted(others))
    if not all_types:
        return None

    angles = np.linspace(0, 2 * np.pi, len(all_types), endpoint=False)
    pos_global = {ct: (np.cos(a), np.sin(a)) for ct, a in zip(all_types, angles)}
    type_colors = {ct: label_to_color.get(ct, "#888888") for ct in all_types}

    thre_x = 0.25
    thre_y = 0.8
    min_x = 15
    max_x = 20
    min_y = 25
    max_y = 40

    for ax, tk in zip(axes, time_keys):
        comm = all_time_communications[tk]
        types_raw = [str(x) for x in comm["types"]]
        valid_indices = [i for i, ct in enumerate(types_raw) if ct in all_types]
        types = [types_raw[i] for i in valid_indices]
        matrix = comm["M_per_source"][np.ix_(valid_indices, valid_indices)]

        ad = adata_dict[tk]
        counts = ad.obs["celltype_prediction"].astype(str).value_counts().reindex(types).fillna(0)
        log_counts = np.log1p(counts)
        log_min = float(log_counts.min()) if len(log_counts) else 0.0
        log_max = float(log_counts.max()) if len(log_counts) else 1.0
        min_r, max_r = 0.05, 0.135

        matrix_flat = matrix.flatten()
        non_zero = sorted(matrix_flat[matrix_flat > 0], reverse=True)
        num_non_zero = len(non_zero)
        if num_non_zero > 0:
            ideal_x_count = int(num_non_zero * thre_x)
            actual_x_count = max(min_x, min(ideal_x_count, max_x))
            idx_x = max(0, min(actual_x_count, num_non_zero) - 1)
            weight_threshold_x = non_zero[idx_x]

            ideal_y_count = int(num_non_zero * thre_y)
            actual_y_count = max(min_y, min(ideal_y_count, max_y))
            actual_y_count = max(actual_y_count, actual_x_count)
            idx_y = max(0, min(actual_y_count, num_non_zero) - 1)
            weight_threshold_y = non_zero[idx_y]
        else:
            weight_threshold_x = 1e-9
            weight_threshold_y = 1e-9

        current_max = float(max(matrix.max(), 1e-9)) if matrix.size else 1e-9
        for i, src in enumerate(types):
            for j, tgt in enumerate(types):
                weight = float(matrix[i, j])
                if weight < weight_threshold_y:
                    continue

                if weight >= weight_threshold_x:
                    w_norm = np.clip((weight - weight_threshold_x) / (current_max - weight_threshold_x + 1e-9), 0, 1)
                    alpha = 0.2 + 0.6 * (2 * w_norm - w_norm**2)
                    width = 0.8 + 2.0 * w_norm
                    mutation_scale = 0.0
                    z_order = 3 + w_norm
                else:
                    alpha, width, mutation_scale, z_order = 0.12, 0.6, 0.0, 2

                if i != j:
                    arrow = FancyArrowPatch(
                        pos_global[src],
                        pos_global[tgt],
                        arrowstyle="->,head_length=0.4,head_width=0.2" if mutation_scale > 0 else "-",
                        connectionstyle="arc3,rad=0.18",
                        color=type_colors[src],
                        linewidth=width,
                        mutation_scale=mutation_scale,
                        alpha=alpha,
                        zorder=z_order,
                    )
                    ax.add_patch(arrow)
                else:
                    angle = angles[all_types.index(src)]
                    pos = pos_global[src]
                    cx = pos[0] + np.cos(angle) * 0.11
                    cy = pos[1] + np.sin(angle) * 0.11
                    circle = Circle(
                        (cx, cy),
                        0.14,
                        fill=False,
                        color=type_colors[src],
                        linewidth=width,
                        alpha=alpha,
                        zorder=z_order,
                    )
                    ax.add_patch(circle)

        for ct in types:
            pos = pos_global[ct]
            n_cells = float(counts[ct])
            if log_max > log_min:
                radius = min_r + (np.log1p(n_cells) - log_min) / (log_max - log_min) * (max_r - min_r)
            else:
                radius = min_r
            ax.add_patch(
                Circle(
                    pos,
                    radius,
                    color=type_colors[ct],
                    alpha=1.0,
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=5,
                )
            )
            ax.text(
                pos[0] * 1.2,
                pos[1] * 1.2,
                ct.replace(" ", "\n"),
                ha="center",
                va="center",
                ma="center",
                fontsize=10,
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
            )

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{tk}\nNetwork Topology", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path is not None:
        save_path = _resolve_repo_path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format=save_path.suffix.lstrip(".") or "svg", bbox_inches="tight", transparent=True)
    plt.show()
    return fig

def plot_segment_network_evolution(
    segment_name: str,
    segment_labels: Sequence[str],
    adata_dict: Dict[str, ad.AnnData],
    all_time_communications: Dict[str, dict],
    label_to_color: Dict[str, str],
    exclude_types: Optional[Sequence[str]],
    output_dir: str | Path,
):
    segment_dir = _resolve_repo_path(output_dir) / segment_name
    segment_dir.mkdir(parents=True, exist_ok=True)
    save_path = segment_dir / "network_evolution_global.svg"
    plot_network_evolution_global(
        adata_dict=adata_dict,
        all_time_communications=all_time_communications,
        time_keys=list(segment_labels),
        label_to_color=label_to_color,
        exclude_types=exclude_types,
        save_path=save_path,
    )
    return save_path

set_seed(42)
