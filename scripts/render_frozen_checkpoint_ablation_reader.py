#!/usr/bin/env python3
"""Render biology-first views of a frozen-checkpoint functional ablation.

The input is produced by ``run_frozen_checkpoint_ablations.py``.  This
renderer deliberately emphasizes where cells end up and which labelled
territories move.  Scalar summaries are exported for audit, but are not the
main figure.

Important semantics
-------------------
``lr_gate_off`` is the same fitted checkpoint with the learned LR-informed
edge gate replaced by an all-spatial, within-cutoff gate at inference.  The
number of admitted edges is therefore allowed to change.  It is neither a
retrained no-LR model nor a matched-edge-density shuffle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.ndimage import gaussian_filter


CONDITIONS = ("full", "interaction_off", "lr_gate_off")
CONDITION_TITLES = {
    "full": "Full model\nsame fitted checkpoint",
    "interaction_off": "Interaction OFF\nsame fitted checkpoint",
    "lr_gate_off": (
        "LR gate OFF → all-spatial candidates\n"
        "same checkpoint; edge count changes"
    ),
}
CONDITION_COLORS = {
    "interaction_off": "#D55E00",
    "lr_gate_off": "#7B3294",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-dir", required=True)
    parser.add_argument("--adata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-key", default=None)
    parser.add_argument("--spatial-key", default=None)
    parser.add_argument("--annotation-key", default=None)
    parser.add_argument("--top-cell-types", type=int, default=8)
    parser.add_argument("--density-bins", type=int, default=92)
    parser.add_argument("--arrow-cells", type=int, default=420)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def _load_ablation(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    endpoints: dict[str, np.ndarray] = {}
    starts: list[np.ndarray] = []
    for name in CONDITIONS:
        path = root / f"{name}.npz"
        with np.load(path) as saved:
            points = np.asarray(saved["points"], dtype=np.float32)
        if points.ndim != 3 or points.shape[0] < 2 or points.shape[2] < 2:
            raise ValueError(f"{path} has invalid points shape {points.shape}.")
        if not np.isfinite(points).all():
            raise ValueError(f"{path} contains non-finite values.")
        starts.append(points[0])
        endpoints[name] = points[-1]
    if not all(np.array_equal(starts[0], item) for item in starts[1:]):
        raise ValueError("The conditions do not share the exact initial cohort.")
    endpoints["start"] = starts[0]
    return endpoints, manifest


def _resolved_key(manifest: dict, supplied: str | None, field: str, fallback: str):
    if supplied:
        return supplied
    value = manifest.get("input", {}).get(field)
    return str(value) if value else fallback


def _plot_limits(*arrays: np.ndarray) -> tuple[float, float, float, float]:
    joined = np.vstack([np.asarray(item)[:, :2] for item in arrays])
    lo = np.nanquantile(joined, 0.001, axis=0)
    hi = np.nanquantile(joined, 0.999, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo -= 0.055 * span
    hi += 0.055 * span
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _inside(points: np.ndarray, limits) -> np.ndarray:
    xmin, xmax, ymin, ymax = limits
    return (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )


def _density(
    points: np.ndarray,
    *,
    limits,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = limits
    keep = _inside(points, limits)
    hist, xedges, yedges = np.histogram2d(
        points[keep, 0],
        points[keep, 1],
        bins=int(bins),
        range=((xmin, xmax), (ymin, ymax)),
    )
    hist = gaussian_filter(hist, sigma=1.15)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist, xedges, yedges


def _mass_threshold(density: np.ndarray, mass: float = 0.90) -> float:
    flat = np.asarray(density, dtype=float).ravel()
    if flat.size == 0 or not np.any(flat > 0):
        return np.nan
    ordered = np.sort(flat)[::-1]
    cumulative = np.cumsum(ordered)
    index = int(np.searchsorted(cumulative, mass * cumulative[-1], side="left"))
    return float(ordered[min(index, ordered.size - 1)])


def _draw_observed_boundary(
    ax,
    observed_density: np.ndarray,
    xedges: np.ndarray,
    yedges: np.ndarray,
):
    level = _mass_threshold(observed_density, 0.90)
    if not np.isfinite(level) or level <= 0:
        return
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    ax.contour(
        xc,
        yc,
        observed_density.T,
        levels=[level],
        colors="#111111",
        linewidths=1.15,
        linestyles="--",
        zorder=4,
    )


def _save(fig, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=260, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _render_density(
    *,
    observed: np.ndarray,
    endpoints: dict[str, np.ndarray],
    limits,
    output_dir: Path,
    bins: int,
) -> dict[str, int]:
    panels = [("observed", observed)] + [
        (name, endpoints[name][:, :2]) for name in CONDITIONS
    ]
    densities = {}
    edges = {}
    out_of_view = {}
    for name, points in panels:
        densities[name], xedges, yedges = _density(
            points,
            limits=limits,
            bins=bins,
        )
        edges[name] = (xedges, yedges)
        out_of_view[name] = int((~_inside(points, limits)).sum())
    vmax = max(float(np.nanquantile(value, 0.995)) for value in densities.values())
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(1, 4, figsize=(14.8, 4.05), sharex=True, sharey=True)
    xmin, xmax, ymin, ymax = limits
    image = None
    for ax, (name, points) in zip(axes, panels):
        image = ax.imshow(
            densities[name].T,
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        _draw_observed_boundary(
            ax,
            densities["observed"],
            *edges["observed"],
        )
        title = (
            "Observed t4\nreference tissue"
            if name == "observed"
            else CONDITION_TITLES[name]
        )
        ax.set_title(title, fontsize=10.2, pad=8)
        visible = int(_inside(points, limits).sum())
        note = f"{visible:,}/{len(points):,} cells in shared view"
        if out_of_view[name]:
            note += f"\n{out_of_view[name]:,} outside reference window"
        ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            fontsize=7.2,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2),
        )
        ax.set_xlabel("aligned spatial x")
        ax.set_aspect("equal")
    axes[0].set_ylabel("aligned spatial y")
    fig.suptitle(
        "Where does the fixed t3 cohort end at t4?\n"
        "Black dashed contour = observed t4 90% tissue-density region",
        fontsize=13,
        y=1.075,
    )
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
    cbar.set_label("relative cell density", fontsize=9)
    fig.text(
        0.5,
        -0.035,
        (
            "All model panels use the same checkpoint, start cells, seed and "
            "deterministic time grid. LR-gate OFF admits all within-cutoff "
            "spatial candidates, so edge density is not matched."
        ),
        ha="center",
        va="top",
        fontsize=8.4,
    )
    _save(fig, output_dir, "01_endpoint_tissue_density")
    return out_of_view


def _balanced_arrow_indices(
    points: np.ndarray,
    *,
    n: int,
    seed: int,
) -> np.ndarray:
    if len(points) <= n:
        return np.arange(len(points))
    rng = np.random.default_rng(seed)
    # Equal-probability spatial strata keep the arrows distributed across the
    # tissue instead of over-representing its densest region.
    qx = np.quantile(points[:, 0], np.linspace(0, 1, 13))
    qy = np.quantile(points[:, 1], np.linspace(0, 1, 13))
    bx = np.clip(np.digitize(points[:, 0], qx[1:-1]), 0, 11)
    by = np.clip(np.digitize(points[:, 1], qy[1:-1]), 0, 11)
    groups = bx * 12 + by
    chosen: list[int] = []
    per_group = max(1, int(np.ceil(n / max(1, len(np.unique(groups))))))
    for group in np.unique(groups):
        candidates = np.flatnonzero(groups == group)
        take = min(per_group, len(candidates))
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) > n:
        chosen = rng.choice(chosen, size=n, replace=False)
    elif len(chosen) < n:
        remaining = np.setdiff1d(np.arange(len(points)), chosen)
        chosen = np.concatenate(
            [chosen, rng.choice(remaining, size=n - len(chosen), replace=False)]
        )
    return np.sort(chosen)


def _render_displacement(
    *,
    endpoints: dict[str, np.ndarray],
    limits,
    output_dir: Path,
    arrow_cells: int,
    seed: int,
) -> None:
    full = endpoints["full"][:, :2]
    indices = _balanced_arrow_indices(full, n=arrow_cells, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharex=True, sharey=True)
    xmin, xmax, ymin, ymax = limits
    for ax, name in zip(axes, ("interaction_off", "lr_gate_off")):
        counterfactual = endpoints[name][:, :2]
        delta = counterfactual - full
        magnitude = np.linalg.norm(delta, axis=1)
        outside = int((~_inside(counterfactual, limits)).sum())
        ax.scatter(
            full[:, 0],
            full[:, 1],
            s=2.8,
            color="#A6A6A6",
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        ax.quiver(
            full[indices, 0],
            full[indices, 1],
            delta[indices, 0],
            delta[indices, 1],
            color=CONDITION_COLORS[name],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0022,
            headwidth=3.3,
            headlength=4.2,
            headaxislength=3.7,
            alpha=0.77,
            zorder=2,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(
            "\n".join(textwrap.wrap(CONDITION_TITLES[name].replace("\n", " — "), 42)),
            fontsize=10.4,
        )
        ax.text(
            0.02,
            0.02,
            (
                f"paired spatial shift\n"
                f"median {np.median(magnitude):.3g}; "
                f"95% {np.quantile(magnitude, 0.95):.3g}"
                + (
                    f"\n{outside:,} endpoints outside shared view"
                    if outside
                    else ""
                )
            ),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.86, edgecolor="none", pad=2.5),
        )
        ax.set_xlabel("aligned spatial x")
    axes[0].set_ylabel("aligned spatial y")
    fig.suptitle(
        "Which parts of the tissue move when a fitted component is disabled?\n"
        "Gray = full-model t4 endpoint; arrows = paired full → counterfactual",
        fontsize=12.6,
        y=1.035,
    )
    fig.text(
        0.5,
        -0.025,
        (
            f"A spatially balanced, fixed-seed subset of {len(indices):,} of "
            f"{len(full):,} paired cells is drawn; all cells are retained in "
            "the exported table. Arrow length is the actual aligned-coordinate shift."
        ),
        ha="center",
        fontsize=8.2,
    )
    _save(fig, output_dir, "02_paired_spatial_displacement")


def _cell_type_palette(labels: pd.Series, top_n: int):
    counts = labels.value_counts()
    top = counts.head(int(top_n)).index.astype(str).tolist()
    cmap = plt.get_cmap("tab10")
    palette = {name: cmap(i % 10) for i, name in enumerate(top)}
    palette["Other"] = (0.72, 0.72, 0.72, 0.45)
    return top, palette


def _collapse_labels(values, top: list[str]) -> np.ndarray:
    labels = pd.Series(values, dtype="string").fillna("Unknown").astype(str)
    return np.where(labels.isin(top), labels, "Other")


def _render_territories(
    *,
    observed: np.ndarray,
    observed_labels: np.ndarray,
    endpoints: dict[str, np.ndarray],
    cohort_labels: np.ndarray,
    limits,
    output_dir: Path,
    top_cell_types: int,
) -> tuple[list[str], dict[str, str]]:
    top, palette = _cell_type_palette(
        pd.Series(cohort_labels, dtype="string").fillna("Unknown"),
        top_cell_types,
    )
    panels = [("observed", observed, _collapse_labels(observed_labels, top))]
    panels.extend(
        (
            name,
            endpoints[name][:, :2],
            _collapse_labels(cohort_labels, top),
        )
        for name in CONDITIONS
    )
    xmin, xmax, ymin, ymax = limits
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.5), sharex=True, sharey=True)
    draw_order = ["Other"] + top
    for ax, (name, points, labels) in zip(axes, panels):
        for label in draw_order:
            keep = labels == label
            if not np.any(keep):
                continue
            ax.scatter(
                points[keep, 0],
                points[keep, 1],
                s=4.2 if label == "Other" else 5.5,
                color=palette[label],
                alpha=0.25 if label == "Other" else 0.72,
                linewidths=0,
                rasterized=True,
            )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(
            "Observed t4\nmeasured labels"
            if name == "observed"
            else CONDITION_TITLES[name],
            fontsize=10,
        )
        ax.set_xlabel("aligned spatial x")
    axes[0].set_ylabel("aligned spatial y")
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=palette[label],
            markeredgecolor="none",
            markersize=6,
            label=label,
        )
        for label in top + ["Other"]
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=min(5, len(legend)),
        frameon=False,
        fontsize=8.3,
    )
    fig.suptitle(
        "Do the same labelled t3 populations retain the same spatial territories at t4?",
        fontsize=12.8,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.19,
        (
            "For model panels, the original t3 Annotation is carried with each "
            "fixed cell; it is not reclassified at t4. Observed t4 uses its "
            "measured Annotation. “Other” pools less abundant t3 labels."
        ),
        ha="center",
        fontsize=8.2,
    )
    _save(fig, output_dir, "03_cell_type_territories")
    rgba = {
        name: matplotlib.colors.to_hex(color, keep_alpha=True)
        for name, color in palette.items()
    }
    return top, rgba


def _effect_table(
    *,
    endpoints: dict[str, np.ndarray],
    cohort_labels: np.ndarray,
) -> pd.DataFrame:
    full = endpoints["full"]
    labels = pd.Series(cohort_labels, dtype="string").fillna("Unknown").astype(str)
    records = []
    for condition in ("interaction_off", "lr_gate_off"):
        delta = endpoints[condition] - full
        spatial = np.linalg.norm(delta[:, :2], axis=1)
        state = np.linalg.norm(delta[:, 2:], axis=1)
        for cell_type in sorted(labels.unique()):
            keep = labels.to_numpy() == cell_type
            records.append(
                {
                    "condition": condition,
                    "cell_type": cell_type,
                    "n_cells": int(keep.sum()),
                    "spatial_displacement_median": float(np.median(spatial[keep])),
                    "spatial_displacement_p95": float(
                        np.quantile(spatial[keep], 0.95)
                    ),
                    "state_displacement_median": float(np.median(state[keep])),
                    "state_displacement_p95": float(np.quantile(state[keep], 0.95)),
                }
            )
    return pd.DataFrame.from_records(records)


def _write_guide(
    *,
    output_dir: Path,
    ablation_dir: Path,
    adata_path: Path,
    summary: dict,
) -> None:
    interaction = summary["paired_effects"]["interaction_off"]
    gate = summary["paired_effects"]["lr_gate_off"]
    interaction_top = summary["cell_type_highlights"]["interaction_off"][0]
    gate_top = summary["cell_type_highlights"]["lr_gate_off"][0]
    gate_outside = summary["out_of_reference_view"]["lr_gate_off"]
    text = f"""# Frozen-checkpoint ablation：读图说明

## 这次到底比较了什么

三种条件都从同一个 t3 细胞 cohort 出发，使用同一个已训练 checkpoint、同一个随机种子、同一个 deterministic Euler 时间网格，并在 t4 结束。没有重新训练、没有噪声、没有 cell splitting/resampling、没有 growth、没有 spatial warp。

- **Full**：原始 fitted model 和训练时的 LR-informed edge gate。
- **Interaction OFF**：同 checkpoint，仅在推理时不把 interaction velocity 加入总 velocity。
- **LR gate OFF / all-spatial gate counterfactual**：同 checkpoint、同一个已训练 interaction GNN，但推理时让 spatial cutoff 内所有 candidate edges 都通过。

最后一种不是“重新训练一个 no-LR model”，也不是 matched-edge-density shuffle。它会改变 edge density，因此只能回答“这个 fitted model 是否依赖原本的 LR gate”，不能把差异全部解释成某个具体 LR 的生物学因果作用。

## 三张主图怎么读

### 01_endpoint_tissue_density

直接看 t4 的组织轮廓和细胞密度。黑色虚线是 measured t4 中容纳 90% density mass 的区域；三个 model panels 与它使用同一显示范围。若 counterfactual 的形状或密度离开黑色轮廓，说明关闭该功能改变了组织级 endpoint。

### 02_paired_spatial_displacement

每个箭头连接**同一个细胞**在 Full endpoint 与 counterfactual endpoint 的位置，因此不是两群细胞之间的最近邻匹配。箭头从 Full 指向 counterfactual。Interaction OFF 的 median spatial shift 为 {interaction["spatial_median"]:.4g}；LR-gate OFF 为 {gate["spatial_median"]:.4g}。坐标窗由 observed t4、start t3 和 Full 定义，而不是被 counterfactual outliers 拉大；all-spatial 条件有 {gate_outside} 个 endpoint 落在这个共同窗口外。LR-gate OFF 的大变化同时混合了 LR identity 与 edge-density 改变，不能过度解释。

### 03_cell_type_territories

model panels 中的颜色是 t3 原始 Annotation 随固定细胞一起带到 t4，没有在 endpoint 重新分类；因此这张图问的是“哪些起始细胞群的 spatial territory 对 interaction/gate 最敏感”。左侧 measured t4 使用 measured Annotation，只作为组织位置参照。

按每个 cell type 内的 paired median spatial shift，Interaction OFF 中最明显的是 **{interaction_top["cell_type"]}**（n={interaction_top["n_cells"]}, median={interaction_top["spatial_displacement_median"]:.4g}）；all-spatial gate 中最明显的是 **{gate_top["cell_type"]}**（n={gate_top["n_cells"]}, median={gate_top["spatial_displacement_median"]:.4g}）。后者尤其不能解释成该 cell type 的 LR-specific mechanism，因为 edge density 同时改变，而且存在长尾 outliers（spatial max={gate["spatial_max"]:.4g}, state max={gate["state_max"]:.4g}）。

## 可以支持的结论

1. 这个**已拟合的 full model**在 t3→t4 推理时对 interaction term 有功能依赖；关闭 interaction 后，同一批细胞的 state 和 spatial endpoint 都发生了变化。
2. 把 fitted LR gate 换成 all-spatial gate 后变化更大，说明原 gate 对限制 GNN message passing 很重要。
3. 这些是 matched, frozen-checkpoint functional dependencies，不受重新训练差异或细胞重采样混淆。

## 不能支持的结论

1. 不能据此声称某个具体 LR pair 已被实验验证为因果机制。
2. 不能把 LR-gate OFF 的全部差异归因于 LR identity，因为 all-spatial counterfactual **没有匹配 edge density**。
3. 不能用这次 frozen-checkpoint 结果替代 retrained architecture ablation；两者回答的是不同问题。
4. 这里的 state 是模型 latent/PCA state，不是独立的 wet-lab readout。

## 可追溯输入

- Ablation run: `{ablation_dir}`
- Aligned H5AD: `{adata_path}`
- Checkpoint SHA256: `{summary["checkpoint"]["weight_sha256"]}`
- Score SHA256: `{summary["checkpoint"]["score_sha256"]}`
"""
    (output_dir / "START_HERE_CN.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.top_cell_types < 1:
        raise ValueError("--top-cell-types must be positive.")
    if args.density_bins < 20:
        raise ValueError("--density-bins must be at least 20.")

    ablation_dir = Path(args.ablation_dir).expanduser().resolve()
    adata_path = Path(args.adata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoints, manifest = _load_ablation(ablation_dir)
    adata = sc.read_h5ad(adata_path)

    time_key = _resolved_key(
        manifest,
        args.time_key,
        "resolved_time_key",
        "time_point_processed",
    )
    spatial_key = _resolved_key(
        manifest,
        args.spatial_key,
        "spatial_key",
        "spatial_aligned",
    )
    annotation_key = _resolved_key(
        manifest,
        args.annotation_key,
        "annotation_key",
        "Annotation",
    )
    for key in (time_key, annotation_key):
        if key not in adata.obs:
            raise KeyError(f"AnnData.obs is missing '{key}'.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"AnnData.obsm is missing '{spatial_key}'.")

    end_time = float(manifest["matched_controls"]["end_time"])
    times = pd.to_numeric(adata.obs[time_key], errors="raise").to_numpy(dtype=float)
    target_mask = np.isclose(times, end_time)
    if not np.any(target_mask):
        raise ValueError(f"No observed cells found at end_time={end_time}.")
    observed = np.asarray(adata.obsm[spatial_key], dtype=np.float32)[target_mask, :2]
    observed_labels = (
        adata.obs.loc[target_mask, annotation_key].astype("string").fillna("Unknown")
    ).to_numpy()

    cohort = pd.read_csv(ablation_dir / "initial_cohort.csv")
    if len(cohort) != len(endpoints["full"]):
        raise ValueError("initial_cohort.csv length does not match the rollout.")
    if "cell_type" not in cohort:
        raise KeyError("initial_cohort.csv is missing cell_type.")
    cohort_labels = cohort["cell_type"].astype("string").fillna("Unknown").to_numpy()

    limits = _plot_limits(
        observed,
        endpoints["start"][:, :2],
        endpoints["full"][:, :2],
    )
    out_of_view = _render_density(
        observed=observed,
        endpoints=endpoints,
        limits=limits,
        output_dir=output_dir,
        bins=args.density_bins,
    )
    _render_displacement(
        endpoints=endpoints,
        limits=limits,
        output_dir=output_dir,
        arrow_cells=args.arrow_cells,
        seed=args.seed,
    )
    top_cell_types, palette = _render_territories(
        observed=observed,
        observed_labels=observed_labels,
        endpoints=endpoints,
        cohort_labels=cohort_labels,
        limits=limits,
        output_dir=output_dir,
        top_cell_types=args.top_cell_types,
    )

    effects = _effect_table(
        endpoints=endpoints,
        cohort_labels=cohort_labels,
    )
    effects.to_csv(output_dir / "cell_type_paired_effects.csv", index=False)
    paired_effects = {}
    for condition in ("interaction_off", "lr_gate_off"):
        delta = endpoints[condition] - endpoints["full"]
        spatial = np.linalg.norm(delta[:, :2], axis=1)
        state = np.linalg.norm(delta[:, 2:], axis=1)
        paired_effects[condition] = {
            "spatial_median": float(np.median(spatial)),
            "spatial_mean": float(np.mean(spatial)),
            "spatial_p95": float(np.quantile(spatial, 0.95)),
            "spatial_max": float(np.max(spatial)),
            "state_median": float(np.median(state)),
            "state_mean": float(np.mean(state)),
            "state_p95": float(np.quantile(state, 0.95)),
            "state_max": float(np.max(state)),
        }
    cell_type_highlights = {}
    for condition in ("interaction_off", "lr_gate_off"):
        selected = (
            effects.loc[(effects["condition"] == condition) & (effects["n_cells"] >= 20)]
            .sort_values(
                ["spatial_displacement_median", "n_cells"],
                ascending=[False, False],
            )
            .head(3)
        )
        cell_type_highlights[condition] = [
            {
                "cell_type": str(row.cell_type),
                "n_cells": int(row.n_cells),
                "spatial_displacement_median": float(
                    row.spatial_displacement_median
                ),
                "state_displacement_median": float(row.state_displacement_median),
            }
            for row in selected.itertuples(index=False)
        ]
    summary = {
        "analysis": "frozen_checkpoint_ablation_biology_first_reader",
        "ablation_dir": str(ablation_dir),
        "adata": str(adata_path),
        "adata_sha256": _sha256(adata_path),
        "n_fixed_t3_cells": int(len(cohort)),
        "n_observed_t4_cells": int(target_mask.sum()),
        "display_limits_from_observed_start_and_full_only": list(limits),
        "out_of_reference_view": out_of_view,
        "paired_effects": paired_effects,
        "cell_type_highlights": cell_type_highlights,
        "top_t3_cell_types": top_cell_types,
        "palette": palette,
        "checkpoint": manifest["checkpoint"],
        "semantic_guardrails": {
            "lr_gate_off_formal_label": (
                "same_checkpoint_all_spatial_gate_counterfactual"
            ),
            "edge_density_matched": False,
            "weights_retrained": False,
            "fixed_cohort": True,
            "deterministic": True,
            "spatial_warp": False,
            "growth_or_resampling": False,
            "model_endpoint_labels": (
                "carried from t3 initial cohort; not reclassified at t4"
            ),
        },
    }
    (output_dir / "reader_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_guide(
        output_dir=output_dir,
        ablation_dir=ablation_dir,
        adata_path=adata_path,
        summary=summary,
    )
    print(f"Saved biology-first frozen-ablation reader to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
