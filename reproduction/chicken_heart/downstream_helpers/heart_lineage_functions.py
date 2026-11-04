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


def prepare_multi_timepoint_adata(
    data_list,
    labels_list,
    timepoint_labels,
    gap_scale=0.05,
    space=0.2,
    align_bottom=True  # 新增开关：是否底边对齐
):
    """
    Build side-by-side AnnData for multiple timepoints.

    Parameters:
        data_list: List[np.ndarray] - coordinates/features for each step
        labels_list: List[np.ndarray] - cell types for each step
        timepoint_labels: List[str] - names e.g. ['D4', 'D5', 'D6']
    """
    spatial_list = []
    current_shift = 0.0

    all_X = []
    all_labels = []
    all_tp = []

    for i, (data, labels, tp) in enumerate(zip(data_list, labels_list, timepoint_labels)):
        spatial = data[:, :2].copy()

        if align_bottom:
            # 将当前切片的 Y 轴起点归零，实现底边对齐
            spatial[:, 1] -= spatial[:, 1].min()

        # # Calculate shift based on previous
        # if i > 0:
        #     prev_spatial = data_list[i-1][:, :2]
        #     width = prev_spatial[:, 0].max() - prev_spatial[:, 0].min()
        #     if width == 0: width = 1.0
        #     gap = width * gap_scale
        #     current_shift += (width + gap)

        # spatial[:, 0] += current_shift
        # spatial_list.append(spatial)

        # 2. X轴先归零（重要：消除原始坐标自带的位移影响）
        spatial[:, 0] -= spatial[:, 0].min()

        # 3. 计算【当前切片本身】的宽度（在应用偏移之前计算！）
        width = spatial[:, 0].max() - spatial[:, 0].min()
        if width == 0: width = 1.0

        # 4. 应用累积偏移量
        spatial[:, 0] += current_shift
        spatial_list.append(spatial)

        # 5. 更新偏移量：下一个切片的起点 = 当前切片终点 + 固定间距
        current_shift += (width + space)

        all_X.append(data)
        all_labels.append(labels)
        all_tp.extend([tp] * len(data))

    X_combined = np.vstack(all_X)
    spatial_combined = np.vstack(spatial_list)
    labels_combined = np.concatenate(all_labels)

    adata = ad.AnnData(X=X_combined)
    adata.obsm['spatial'] = spatial_combined
    adata.obsm['X_pca'] = X_combined[:, 2:]
    adata.obs['celltype_prediction'] = labels_combined
    adata.obs['timepoint'] = pd.Categorical(all_tp, categories=timepoint_labels, ordered=True)

    print(f"X range: {adata.obsm['spatial'][:,0].min()} to {adata.obsm['spatial'][:,0].max()}")

    return adata

def plot_lineage_transition(
    adata_combined,
    transition_matrices,
    time_keys,
    label_to_color=None,
    step_configs=None,
    only_show_focustype_transitions=True,
    show_self_transitions=True,
    node_position_mode='centroid',
    node_jitter=0.0,
    node_separation=0.0,
    node_separation_iters=80,
    node_separation_spring=0.05,
    fixed_ct_order=None,
    figsize=(15,6),
    save_path=None,
    margin_ratio=0.05,
    node_size_factor=1200,
    dot_size=20
):
    """
    Plot spatial lineage transition diagram for multiple steps.

    Parameters:
        transition_matrices: dict {(src_tp, tgt_tp): pd.DataFrame}
        focus_mode: 'either' | 'source' | 'target'
    """

    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path

    print(f"DEBUG: time_keys received = {time_keys}")
    print(f"DEBUG: adata timepoints = {adata_combined.obs['timepoint'].unique()}")


    if label_to_color is None:
        label_to_color = {}
    if node_position_mode not in ('centroid', 'spread'):
        raise ValueError("node_position_mode must be one of: 'centroid', 'spread'")

    time_key = 'timepoint'
    color_by = 'celltype_prediction'
    spatial_key = 'spatial'

    fig, ax = plt.subplots(figsize=figsize)

    node_positions = {}
    node_sizes = {}
    active_nodes = set()
    edge_list = []
    node_styles = {}

# --- 1. Edge Calculation (带分段逻辑) ---
    for (src_tp, tgt_tp), df in transition_matrices.items():
        # 获取当前时间段的定制配置，若无则使用默认
        config = step_configs.get((src_tp, tgt_tp), {}) if step_configs else {}
        f_types = config.get('focus', [])
        f_mode = config.get('mode', 'either')
        e_types = config.get('exclude', [])

        obs_src = adata_combined.obs[adata_combined.obs[time_key] == src_tp]
        counts_t0 = obs_src[color_by].value_counts()
        inflow_totals = {tgt: sum(counts_t0.get(s, 0) * df.loc[s, tgt] for s in df.index) for tgt in df.columns}

        # --- 在 plot_lineage_transition 的循环内部 ---
        for src in df.index:
            for tgt in df.columns:
                if src in e_types or tgt in e_types: continue

                # 获取精准配置
                f_srcs = config.get('focus_source', [])
                f_tgts = config.get('focus_target', [])

                # 实现你的核心需求逻辑：
                # 1. 如果设置了 focus_source，src 必须在里面
                # 2. 如果设置了 focus_target，tgt 必须在里面
                # 3. 如果两者都设置了，满足其一即可 (OR 关系)

                keep = False
                if not f_srcs and not f_tgts:
                    keep = True # 如果都没设，显示所有（除非被 exclude）
                else:
                    if src in f_srcs: keep = True  # 符合“Epi 的出边”
                    if tgt in f_tgts: keep = True  # 符合“Fib 的入边”

                if not keep: continue

                p_out = df.loc[src, tgt]
                flow_vol = counts_t0.get(src, 0) * p_out

                if flow_vol > 0:
                    s_contrib = flow_vol / (inflow_totals[tgt] + 1e-9)
                    edge_list.append({
                        'src_tp': src_tp, 'src': src,
                        'tgt_tp': tgt_tp, 'tgt': tgt,
                        'p_out': p_out, 'p_in': s_contrib, 'flow': flow_vol
                    })
                    active_nodes.add((src_tp, src))
                    active_nodes.add((tgt_tp, tgt))

    # --- 2. 核心改动：Node Positioning & Background (同步 v7 逻辑) ---
    for tp in time_keys:
        mask = adata_combined.obs[time_key] == tp
        tp_coords = adata_combined.obsm[spatial_key][mask]
        tp_obs = adata_combined.obs[mask]
        if tp_coords.shape[0] == 0: continue

        # 凸包与路径
        hull = ConvexHull(tp_coords)
        hull_path = Path(tp_coords[hull.vertices])

        # 背景点绘制逻辑 (同步 v7: 区分活跃点颜色与透明度)
        for ct in tp_obs[color_by].unique():
            ct_mask = (tp_obs[color_by] == ct)
            is_active = (tp, ct) in active_nodes
            ax.scatter(
                tp_coords[ct_mask.values, 0], tp_coords[ct_mask.values, 1],
                color=label_to_color.get(ct, "#999999") if is_active else "lightgrey",
                s=dot_size,
                alpha=(0.7 if is_active else 0.3), # v7 风格 alpha
                edgecolors='none', zorder=1
            )

        # 评分系统定位 (同步 v7 核心评分算法)
        tp_active_cts = [ct for ct in tp_obs[color_by].unique() if (tp, ct) in active_nodes]
        if not tp_active_cts: continue

        y_min, y_max = tp_coords[:, 1].min(), tp_coords[:, 1].max()
        x_min, x_max = tp_coords[:, 0].min(), tp_coords[:, 0].max()
        x_center = (x_min + x_max) / 2

        # 应用 Margin 定义安全搜索空间
        safe_y_min, safe_y_max = y_min + (y_max-y_min)*margin_ratio, y_max - (y_max-y_min)*margin_ratio

        # 分区逻辑：按 Y 轴中位数对活跃类别排序
        # fixed_ct_order: dict, 越小越靠上（rank 更高）
        # 例如 {'Immature myocardial cells': 0, 'Cardiomyocytes-1': 1}
        if fixed_ct_order is None:
            fixed_ct_order = {}

        ct_centers_y = {
            ct: np.median(tp_coords[(tp_obs[color_by] == ct).values, 1])
            for ct in tp_active_cts
        }

        def ct_sort_key(ct):
            # 先按固定rank排序（没指定的放后面），再按y中位数降序作为tie-break
            return (fixed_ct_order.get(ct, 10**9), -ct_centers_y[ct])

        sorted_cts = sorted(tp_active_cts, key=ct_sort_key)

        partition_edges = np.linspace(safe_y_max, safe_y_min, len(sorted_cts) + 1)

        counts = tp_obs[color_by].value_counts()
        total = len(tp_obs)

        for i, ct in enumerate(sorted_cts):
            p_top, p_bottom = partition_edges[i], partition_edges[i+1]
            p_center_y = (p_top + p_bottom) / 2

            # 采样与评分
            pts = np.column_stack([
                np.random.uniform(x_min, x_max, 1500),
                np.random.uniform(p_bottom, p_top, 1500)
            ])
            candidates = pts[hull_path.contains_points(pts)]

            if len(candidates) == 0:
                node_positions[(tp, ct)] = np.array([x_center, p_center_y])
            else:
                # v7 评分：倾向于 X 轴居中和 Y 轴分区中心
                score_x = (1.0 - (np.abs(candidates[:, 0] - x_center) / ((x_max-x_min)/2 + 1e-9))) * 0.8
                score_y = (1.0 - (np.abs(candidates[:, 1] - p_center_y) / ((p_top-p_bottom)/2 + 1e-9))) * 0.8
                node_positions[(tp, ct)] = candidates[np.argmax(score_x + score_y)]

            # 方案 A：标准对数处理 (推荐)
            # 使用 np.log1p (即 log(1+x)) 避免针对零值的报错，并平滑小数值
            node_styles[(tp, ct)] = {
                's': np.log1p(counts[ct]) * (node_size_factor / 5),
                'c': label_to_color.get(ct)
            }

            # node_styles[(tp, ct)] = {
            #     's': np.sqrt(counts[ct]/total)*node_size_factor,
            #     'c': label_to_color.get(ct)
            # }

        # 时间标签居中 (同步 v7 样式)
        ax.text(x_center, y_max + (y_max-y_min)*0.05, f"Time: {tp}", ha="center", fontsize=12, fontweight='bold')

    # --- 3. Draw Edges ---
    def draw_edge(p1, p2, c1, c2, p_out, p_in):
        mid = (p1 + p2) / 2
        direction = p2 - p1
        normal = np.array([-direction[1], direction[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        control = mid + normal * (np.linalg.norm(direction) * 0.2)

        t = np.linspace(0, 1, 50)
        curve = (1-t)[:, None]**2 * p1 + 2*(1-t)[:, None]*t[:, None]*control + t[:, None]**2 * p2

        segments = np.concatenate([curve[:-1, None, :], curve[1:, None, :]], axis=1)
        lc = LineCollection(
             segments,
             cmap=LinearSegmentedColormap.from_list('g', [c1, c2]),
             linewidth=0.5 + p_out * 3,
             alpha=0.6,
             zorder=2
        )
        lc.set_array(t)
        ax.add_collection(lc)

        # Labels
        p_start = 0.8*p1 + 0.2*control
        ax.text(p_start[0], p_start[1], f"{p_out:.1%}", fontsize=7, ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))

        p_end = 0.2*control + 0.8*p2
        ax.text(p_end[0], p_end[1], f"{p_in:.1%}", fontsize=7, ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))

    for e in edge_list:
        start_key = (e['src_tp'], e['src'])
        end_key = (e['tgt_tp'], e['tgt'])

        if start_key in node_positions and end_key in node_positions:
            c1 = label_to_color.get(e['src'], "#999999")
            c2 = label_to_color.get(e['tgt'], "#999999")
            draw_edge(
                node_positions[start_key],
                node_positions[end_key],
                c1, c2,
                e['p_out'], e['p_in']
            )

    # --- 4. Draw Nodes (增加“孤立节点”过滤) ---

    # 1. 首先收集所有在边中出现过的节点（有连接的节点）
    connected_nodes = set()
    for e in edge_list:
        connected_nodes.add((e['src_tp'], e['src']))
        connected_nodes.add((e['tgt_tp'], e['tgt']))

    # 2. 只遍历那些有连接的节点进行绘制
    # sorted 确保大圆在下，小圆在上，不遮挡文字
    nodes_to_draw = [k for k in node_positions.keys() if k in connected_nodes]

    for k in sorted(nodes_to_draw, key=lambda x: node_styles[x]['s'], reverse=True):
        pos, style = node_positions[k], node_styles[k]

        # 绘制节点圆圈
        ax.scatter(
            pos[0], pos[1],
            s=style['s'],
            color=style['c'],
            edgecolors='white',
            linewidth=1.5,
            zorder=10
        )

        # 绘制类别文字标签
        ax.text(
            pos[0], pos[1],
            k[1], # k[1] 是 celltype 名称
            fontsize=8,
            ha='center',
            va='center',
            fontweight='bold',
            zorder=11 # 确保文字在最上方
        )

    # 在所有绘图逻辑完成后调用
    ax.set_aspect('equal')
    ax.axis('off')  # 彻底关掉外框、刻度和标签

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
