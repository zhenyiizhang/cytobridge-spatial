"""Numerical renderers from the original AGIST Figure 2 evaluation notebook.

Source: evaluation/mosta_simulation_evaluation_viz_notebook.ipynb, cells
7 (growth), 8 (attention), 15/16 (velocity), and 26 (observed snapshots).
The original coordinate choices and plot settings are retained. In particular,
the gene display uses the first two gene-state coordinates, not a fitted UMAP.
No panel image or default numerical table is loaded by these functions.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 14, "axes.linewidth": 1.2,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def evaluate_growth_attention(data_csv, config, model_dir, edge_predictor, output_dir, *, device="cuda"):
    """Evaluate cell growth and the original full time-zero attention graph.

    Unlike the random interaction groups used for population simulation, the
    Figure 2c graph contains all time-zero cells in one forward pass. Store
    sparse edges instead of the original quadratic dense attention matrix.
    """
    import pandas as pd
    import torch
    from scripts.run_agist_split_sde_replicates import load_models

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(data_csv)
    _, model, _, _, _ = load_models(
        Path(__file__).resolve().parents[2], Path(config), Path(model_dir), device,
        edge_predictor=Path(edge_predictor), load_score=False,
    )
    growth = np.empty(len(frame), dtype=np.float32)
    features = frame[[f"x{i}" for i in range(1, 53)]].to_numpy(np.float32)
    with torch.no_grad():
        for time in sorted(frame["samples"].unique()):
            rows = np.flatnonzero(frame["samples"].to_numpy() == time)
            data = torch.tensor(features[rows], device=device)
            t = torch.tensor([float(time)], device=device)
            growth[rows] = model.g_net(t, data).cpu().numpy().reshape(-1)
        rows = np.flatnonzero(frame["samples"].to_numpy() == 0)
        data = torch.tensor(features[rows], device=device)
        edges = _attention_edges(model, data, 0., device)
    np.savez_compressed(output / "predicted_attention_time0.npz", **edges, row_index=rows)
    np.save(output / "predicted_growth.npy", growth)
    return growth, edges


def _attention_edges(model, data, time, device):
    import torch
    with torch.no_grad():
        log_weight = torch.log(torch.ones(len(data), 1, device=device) / len(data))
        model.interaction_net(data, log_weight, torch.tensor([float(time)], device=device), return_attn=True)
        values = model.interaction_net.gnn_layers[0].attn.abs().mean(dim=1).cpu().numpy()
        indices = model.interaction_net.edge_index.cpu().numpy()
    return dict(source=indices[0], target=indices[1], attention=values)


def attention_strength_correlation(edges, truth):
    """Compare outgoing row means on the same nonzero cells as the original analysis.

    The original notebook removed zero rows independently. Require identical
    masks before doing so, to prevent comparisons between different cells.
    """
    truth = np.asarray(truth)
    if truth.ndim != 2 or truth.shape[0] != truth.shape[1] or not np.isfinite(truth).all():
        raise ValueError("Reference attention must be a finite square matrix")
    n = len(truth)
    rows, cols = np.asarray(edges["source"]), np.asarray(edges["target"])
    values = np.asarray(edges["attention"], dtype=float)
    if rows.shape != cols.shape or rows.shape != values.shape or values.ndim != 1:
        raise ValueError("Attention edge arrays must have the same one-dimensional shape")
    if (not np.isfinite(values).all() or np.any(values < 0) or np.any(truth < 0)
            or not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(cols.dtype, np.integer)):
        raise ValueError("Attention values must be finite and nonnegative, with integer indices")
    if len(rows) and (min(rows.min(), cols.min()) < 0 or max(rows.max(), cols.max()) >= n):
        raise ValueError("Attention edge index is outside the reference matrix")
    if len(np.unique(rows * n + cols)) != len(rows):
        raise ValueError("Duplicate attention edges")
    predicted = np.bincount(rows, weights=values, minlength=n) / n
    reference = truth.mean(axis=1)
    mask = reference != 0
    if not np.array_equal(predicted != 0, mask):
        raise ValueError("Prediction and reference have different nonzero cells")
    if mask.sum() < 2 or np.ptp(predicted[mask]) == 0 or np.ptp(reference[mask]) == 0:
        raise ValueError("At least two nonconstant paired strengths are required")
    result = dict(n_cells=n, n_compared=int(mask.sum()),
                  spearman=float(stats.spearmanr(predicted[mask], reference[mask]).statistic))
    return result, predicted, reference, mask


def evaluate_attention_recovery(data_csv, config, model_dir, edge_predictor, reference_dir,
                                output_dir, *, device="cuda"):
    """Infer attention at each observed time and compare with generator matrices.

    This is the separate 6,707-cell experiment (0210 model / 0207 generator).
    The table contains one Spearman correlation per time. Its arithmetic mean
    is not a pooled-cell correlation or the Figure 2c time-zero statistic.
    """
    import pandas as pd
    import torch
    from scripts.run_agist_split_sde_replicates import load_models
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(data_csv)
    _, model, _, _, _ = load_models(
        Path(__file__).resolve().parents[2], Path(config), Path(model_dir), device,
        edge_predictor=Path(edge_predictor), load_score=False,
    )
    records = []
    for time in sorted(frame["samples"].unique()):
        rows = np.flatnonzero(frame["samples"].eq(time).to_numpy())
        data = torch.tensor(frame.iloc[rows, 1:].to_numpy(np.float32), device=device)
        edges = _attention_edges(model, data, time, device)
        np.savez_compressed(output / f"attention_time{int(time)}.npz", **edges, row_index=rows)
        truth = np.load(Path(reference_dir) / f"attn_matrix_time{int(time)}_brain_gt.npy", allow_pickle=False)
        if truth.shape != (len(rows), len(rows)):
            raise ValueError("Reference attention does not match the observed cells")
        metrics, predicted, reference, mask = attention_strength_correlation(edges, truth)
        pd.DataFrame(dict(row_index=rows, predicted=predicted, reference=reference,
                          compared=mask)).to_csv(output / f"strength_time{int(time)}.csv", index=False)
        records.append(dict(time=float(time), **metrics))
    table = pd.DataFrame(records)
    table.to_csv(output / "attention_by_time.csv", index=False)
    summary = dict(mean_spearman=float(table["spearman"].mean()), n_times=len(table),
                   summary="Equal-weight arithmetic mean of per-time Spearman correlations")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return table, summary


def calculate_velocity_display(frame, predicted, reference_fields, output_dir, *, n_jobs=16):
    """Calculate the original neighbor/velocity graphs on shared seed-0 rows.

    Return the four AnnData objects consumed directly by draw_velocity_display.
    Save the selected rows and projected numerical velocities for inspection.
    """
    import anndata
    import scanpy as sc
    import scvelo as scv

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(reference_fields, allow_pickle=False) as reference:
        truth = sum(reference[f"truth_{name}"] for name in ("base", "interaction", "score"))
        if not np.array_equal(reference["time"], frame["samples"].to_numpy(np.float32)):
            raise ValueError("Reference fields and observations have different time order")
    full = np.asarray(predicted["full"])
    if full.shape != truth.shape or full.shape != (len(frame), 52):
        raise ValueError("Velocity arrays must match all observed cells and 52 features")
    result = {}
    exported = {}
    for space, columns, row_mask in (
        ("gene", slice(2, 52), np.ones(len(frame), dtype=bool)),
        ("physical", slice(0, 2), frame["samples"].eq(0).to_numpy()),
    ):
        observed_rows = np.flatnonzero(row_mask)
        states = frame.iloc[observed_rows, 1:].to_numpy()[:, columns]
        selected = np.random.default_rng(0).choice(len(states), size=int(len(states) * .3), replace=False)
        exported[f"{space}_row_index"] = observed_rows[selected]
        for label, values in (("truth", truth), ("predicted", full)):
            adata = anndata.AnnData(X=states.copy())
            adata.layers["Ms"] = states.copy()
            adata.layers["velocity"] = values[row_mask, columns]
            adata.obsm["X_umap"] = states[:, :2].copy()
            adata.obs["time"] = frame["samples"].to_numpy()[row_mask]
            adata = adata[selected].copy()
            sc.pp.neighbors(adata, n_neighbors=30, use_rep="X")
            scv.tl.velocity_graph(adata, vkey="velocity", n_jobs=n_jobs)
            scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")
            result[f"{space}_{label}"] = adata
            exported[f"{space}_{label}_velocity"] = np.asarray(adata.obsm["velocity_umap"])
        exported[f"{space}_coordinates"] = states[selected, :2]
    np.savez_compressed(output / "velocity_display.npz", **exported)
    return result


def draw_velocity_display(display_data, output_dir):
    """Draw the original gene and time-zero spatial velocity comparisons."""
    import scvelo as scv

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    with mpl.rc_context(STYLE):
        scv.settings.set_figure_params(style="scvelo", dpi=100, frameon=False)
        for space in ("gene", "physical"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            plt.subplots_adjust(wspace=.3)
            gene = space == "gene"
            kwargs = dict(
                basis="umap", color="time" if gene else "lightgrey",
                palette="blues" if gene else "viridis", legend_loc="none",
                density=2. if gene else 1., linewidth=2. if gene else 1.,
                arrow_size=2. if gene else 1., arrow_style="-|>",
                alpha=.8, size=50, show=False,
            )
            if gene:
                kwargs["color_map"] = "coolwarm"
            for ax, label, title in zip(axes, ("predicted", "truth"), ("Prediction", "Ground Truth")):
                selected_kwargs = dict(kwargs)
                if label == "predicted":
                    selected_kwargs.update(legend_loc="right margin" if gene else "none", colorbar=True)
                scv.pl.velocity_embedding_stream(display_data[f"{space}_{label}"], ax=ax, title="", **selected_kwargs)
                # Current Matplotlib rejects the NaN width that scVelo leaves
                # on a segment crossing an unsupported grid cell. That segment
                # is invisible in the original raster renderer; use zero width
                # for PDF too. All finite widths and all field values stay put.
                from matplotlib.collections import LineCollection
                for artist in ax.collections:
                    if isinstance(artist, LineCollection):
                        widths = np.asarray(artist.get_linewidths())
                        if not np.isfinite(widths).all():
                            artist.set_linewidths(np.where(np.isfinite(widths), widths, 0.))
                ax.set_title(title, fontsize=18, fontweight="bold", pad=15)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_aspect("equal")
            # The original gene panel was saved as PNG. scVelo can retain NaN
            # colors on unsupported stream cells, which PDF cannot represent.
            path = output / f"velocity_comparison_{space}.{'png' if gene else 'pdf'}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            paths[space] = path
    return paths


def draw_observed_snapshots(frame, output_dir):
    """Draw panel a's four numerical snapshots directly from the simulation CSV."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    times = sorted(frame["samples"].unique())
    with mpl.rc_context(STYLE):
        # Match the red time progression in the assembled paper figure.
        colors = plt.get_cmap("Reds")(np.linspace(.25, .85, len(times)))
        fig, axes = plt.subplots(1, len(times), figsize=(14, 4))
        for ax, time, color in zip(axes, times, colors):
            xy = frame.loc[frame["samples"].eq(time)].iloc[:, 1:3].to_numpy()
            ax.scatter(xy[::2, 0], xy[::2, 1], color=color, s=8, alpha=.8)
            ax.set_title(f"T={time}", fontsize=13, fontweight="bold")
            ax.set(xticks=[], yticks=[], xlabel="", ylabel="", aspect="equal")
            ax.spines[:].set_visible(False)
        path = output / "dataset.pdf"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return path


def calculate_attention_display(attention, spatial_coord, *, top_k=5000):
    """Original positive-edge top-5000 flow, accepting dense or sparse inputs.

    Sparse inputs have source, target and attention arrays. Row-major ordering
    reproduces np.where(dense > 0), including the original np.argsort tie order.
    """
    xy = np.asarray(spatial_coord)
    if isinstance(attention, np.ndarray):
        if attention.shape != (len(xy), len(xy)):
            raise ValueError("Attention matrix must match the time-zero cells")
        strength = attention.sum(axis=1)
        rows, cols = np.where(attention > 0)
        values = attention[rows, cols]
    else:
        rows = np.asarray(attention["source"], dtype=int)
        cols = np.asarray(attention["target"], dtype=int)
        values = np.asarray(attention["attention"], dtype=float)
        if not (rows.shape == cols.shape == values.shape):
            raise ValueError("Sparse attention arrays must have matching lengths")
        if len(rows) and (min(rows.min(), cols.min()) < 0 or max(rows.max(), cols.max()) >= len(xy)):
            raise ValueError("Attention edge refers to an absent cell")
        if len(np.unique(rows * len(xy) + cols)) != len(rows):
            raise ValueError("Duplicate edges cannot represent the original dense attention assignment")
        strength = np.bincount(rows, weights=values, minlength=len(xy))
        positive = values > 0
        rows, cols, values = rows[positive], cols[positive], values[positive]
        order = np.lexsort((cols, rows))
        rows, cols, values = rows[order], cols[order], values[order]
    if not len(values):
        raise ValueError("No positive attention edges")
    selected = np.argsort(values)[-top_k:] if len(values) > top_k else np.arange(len(values))
    rows, cols, scores = rows[selected], cols[selected], values[selected]
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() > scores.min() else np.zeros_like(scores)
    x_grid = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 50)
    y_grid = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    U, V, W = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(X)
    for source, target, score in zip(rows, cols, norm_scores):
        dx, dy = xy[target] - xy[source]
        xi = np.abs(x_grid - xy[source, 0]).argmin()
        yi = np.abs(y_grid - xy[source, 1]).argmin()
        U[yi, xi] += dx * score
        V[yi, xi] += dy * score
        W[yi, xi] += score
    mask = W > 0
    U[mask] /= W[mask]
    V[mask] /= W[mask]
    return dict(strength=strength, X=X, Y=Y, U=U, V=V, W=W)


def draw_attention_display(spatial_coord, predicted, truth, output_dir):
    """Draw c from calculated strengths and flow grids; both references required."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=.1, right=.9)
        for ax, data, title in zip(axes, (predicted, truth), ("Prediction - t=0", "Ground Truth - t=0")):
            strength = data["strength"]
            sc = ax.scatter(spatial_coord[:, 0], spatial_coord[:, 1], c=strength,
                cmap="Reds", s=15, alpha=.9, vmin=0,
                vmax=np.percentile(strength, 95), edgecolors="none", rasterized=True)
            ax.streamplot(data["X"], data["Y"], data["U"], data["V"],
                density=1., color="black", linewidth=1.5, arrowsize=1.5, arrowstyle="-|>")
            ax.set_title(title, pad=12, fontweight="bold", fontsize=16)
            ax.axis("off")
            ax.set_aspect("equal")
        cax = fig.add_axes([.92, .15, .02, .7])
        cbar = fig.colorbar(sc, cax=cax)
        cbar.outline.set_visible(False)
        cbar.set_ticks([])
        cax.text(.5, 1.02, "High", transform=cax.transAxes, ha="center", va="bottom", fontsize=14, fontweight="bold")
        cax.text(.5, -.02, "Low", transform=cax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold")
        cbar.set_label("Attention Strength", rotation=270, labelpad=25, fontsize=16)
        path = output / "attention_flow_t0.pdf"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return path


def draw_distance_panels(data, output_dir):
    """Draw only panel e from the supplied tables, without loading panels a–d."""
    from CytoBridge.results._main_figure_2_plot import FIGURE_2_RC, _draw_w2_panel
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(FIGURE_2_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5),
            gridspec_kw={"width_ratios": [196, 135]})
        _draw_w2_panel(axes[0], data, space="gene", show_legend=True)
        _draw_w2_panel(axes[1], data, space="physical", show_legend=False)
        axes[0].set_ylabel("Wasserstein-2 distance")
        axes[1].set_yticklabels([])
        axes[1].tick_params(axis="y", length=0)
        fig.tight_layout()
        pdf, png = output / "panel_e.pdf", output / "panel_e.png"
        fig.savefig(pdf, dpi=300)
        fig.savefig(png, dpi=300)
        plt.close(fig)
    return pdf, png


def normalize_growth(values):
    """Original all-times 1st/99th-percentile clipping and min-max scaling."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Growth values must be finite")
    lower, upper = np.percentile(values, [1, 99])
    if upper - lower < 1e-12:
        return np.zeros_like(values)
    clipped = np.clip(values, lower, upper)
    return (clipped - lower) / (upper - lower)


def draw_growth_correlation(predicted_growth, truth_growth, output_dir):
    """Calculate normalization and regression from the paired cell-level rates."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction, truth = normalize_growth(predicted_growth), normalize_growth(truth_growth)
    if prediction.shape != truth.shape:
        raise ValueError("Growth reference and prediction must describe the same cells")
    fit = stats.linregress(truth, prediction)
    metrics = dict(pearson_r=float(np.corrcoef(truth, prediction)[0, 1]),
        slope=float(fit.slope), intercept=float(fit.intercept), n=len(truth))
    np.savez_compressed(output / "growth_correlation.npz", predicted=prediction, truth=truth)
    (output / "growth_correlation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    path = output / "correlation_final_box.pdf"
    with mpl.rc_context({**STYLE, "axes.spines.top": True, "axes.spines.right": True}):
        plot_growth_correlation_box_style(prediction, truth, path)
    return path, metrics


def plot_growth_correlation_box_style(g_pred, g_gt, out_path):
    # 数据扁平化
    x = np.array(g_gt).flatten()
    y = np.array(g_pred).flatten()

    # 计算统计
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    pearson_r = np.corrcoef(x, y)[0, 1]

    # 1. 创建画布 (标准单图比例)
    fig, ax = plt.subplots(figsize=(6, 5.2)) # 稍微留高一点给 title 或紧凑布局

    # 2. 绘制 Hexbin (无边框平滑风格)
    hb = ax.hexbin(
        x, y,
        gridsize=60,
        cmap='YlOrRd',  # Paper's final growth-density color scale.
        mincnt=1,
        linewidths=0,  # 关键：无描边，平滑过渡
        edgecolors='none'
    )

    # 3. 绘制参考线
    # 灰虚线对角线
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
    
    # 红实线拟合线
    x_vals = np.array([0, 1])
    y_vals = slope * x_vals + intercept
    y_vals = np.clip(y_vals, 0, 1) # 防止画出界
    ax.plot(x_vals, y_vals, color='#C1272D', linewidth=3, alpha=0.95, zorder=2)

    # 4. 统计信息图例 (左上角)
    # 使用 Patch 创建纯文本图例，更整洁
    legend = ax.legend(
        handles=[
            mpatches.Patch(color='none', label=f"Pearson's r = {pearson_r:.2f}"),
            mpatches.Patch(color='none', label=f'Slope = {slope:.2f}')
        ],
        loc='upper left',
        frameon=False,
        fontsize=16,
        handlelength=0,
        handletextpad=0
    )
    # 强制左对齐
    for text in legend.get_texts():
        text.set_ha('left')

    # 5. 坐标轴设置 (带边框)
    ax.set_xlabel("Ground Truth Growth", fontweight='bold', fontsize=14, labelpad=10)
    ax.set_ylabel("Predicted Growth", fontweight='bold', fontsize=14, labelpad=10)
    
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect('equal') # 保证正方形

    # 6. Colorbar 设置 (标准附着模式)
    # fraction: colorbar 占图的比例
    # pad: colorbar 距离图的间距
    cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    
    cbar.outline.set_visible(False) # 去掉 colorbar 自己的外框
    cbar.set_ticks([]) # 不显示具体数字
    cbar.set_label('Point Density', rotation=270, labelpad=20, fontsize=14)

    # 添加 High / Low
    # 直接在 colorbar 的轴上用 text 标注，坐标系用 transAxes (0在底, 1在顶)
    cbar.ax.text(0.5, 1.02, 'High', transform=cbar.ax.transAxes, 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    cbar.ax.text(0.5, -0.065, 'Low', transform=cbar.ax.transAxes,
                 ha='center', va='top', fontsize=12, fontweight='bold')

    # 保存
    print(f"Saving to {out_path}...")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
