#!/usr/bin/env python
"""
Spatial lineage transition plot (t0 -> t1) for a single source label.

Goal: mimic the "Nature Methods"-like Sankey styling, but on spatial coordinates:
- Left panel: real observed cells at t0 as gray background + source cells colored.
- Right panel: t1 background cells (real observed or split-SDE generated) as gray background
  + simulated t1 points colored by predicted label (restricted to trajectories from source label at t0).
- Draw centroid-to-centroid flow lines (width ~ transition fraction).

Run (from project root):
  python evaluation/mosta/code/mosta_cartilage_lineage_transition_0_1_local.py \\
    --config config/mosta_config.yaml \\
    --data-csv evaluation/mosta/data/mosta_four_time_with_celltype_refined.csv \\
    --color-h5ad evaluation/mosta/data/Mouse_embryo_all_stage.h5ad \\
    --annotation-key annotation \\
    --source-label "Cartilage primordium" \\
    --t0 0 --t1 1 --sde-n-samples 8000

Example (interpolated t0 via split-SDE background):
  python evaluation/mosta/code/mosta_cartilage_lineage_transition_0_1_local.py \\
    --source-label "Cartilage primordium" \\
    --t0 0.5 --t1 1 \\
    --t0-background split-sde --t0-bg-split-sde-n-samples 50000 \\
    --t0-source-selection classifier \\
    --fate-keep-source-cumfrac 0.85

Example (interpolated t1 via split-SDE background):
  python evaluation/mosta/code/mosta_cartilage_lineage_transition_0_1_local.py \\
    --source-label "Cartilage primordium" \\
    --t0 0 --t1 0.5 \\
    --t1-background split-sde --t1-bg-split-sde-n-samples 50000 \\
    --fate-keep-source-cumfrac 0.85
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_under_root(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None
    p = Path(str(path_str)).expanduser()
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return str(p)


def _require_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {ctx}: {missing}")


def load_label_to_color(
    labels: np.ndarray,
    label_color_json: Optional[str] = None,
    color_h5ad: Optional[str] = None,
    annotation_key: str = "Annotation",
) -> Dict[str, str]:
    if label_color_json and os.path.exists(label_color_json):
        with open(label_color_json, "r", encoding="utf-8") as f:
            return json.load(f)

    if color_h5ad and os.path.exists(color_h5ad):
        # Fast path: read Scanpy-style categorical palette from .h5ad using h5py.
        try:
            import h5py

            keys_to_try = [str(annotation_key), str(annotation_key).lower()]
            with h5py.File(color_h5ad, "r") as f:
                for key in keys_to_try:
                    colors_key = f"{key}_colors"
                    if "uns" not in f or colors_key not in f["uns"]:
                        continue
                    if "obs" not in f or "__categories" not in f["obs"]:
                        continue
                    if key not in f["obs"]["__categories"]:
                        continue
                    cats = f["obs"]["__categories"][key][()]
                    cols = f["uns"][colors_key][()]
                    cats = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in cats]
                    cols = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in cols]
                    if len(cats) and len(cats) == len(cols):
                        return {str(c): str(col) for c, col in zip(cats, cols)}
        except Exception as exc:
            print(f"[warn] Color map fast-load failed from {color_h5ad}: {exc}")

        # Fallback: try loading via anndata (slower, but can handle more variants).
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
            print(f"[warn] Color map load failed from {color_h5ad}: {exc}")
    elif color_h5ad:
        print(f"[warn] color_h5ad not found, fallback to matplotlib colormap: {color_h5ad}")

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


def to_valid_color(color_str: str, default_alpha: float = 1.0) -> str:
    if not isinstance(color_str, str):
        return f"rgba(136,136,136,{default_alpha})"
    c = color_str.strip()
    if c.startswith("#") and len(c) == 9:  # #RRGGBBAA
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        a = int(c[7:9], 16) / 255.0
        return f"rgba({r},{g},{b},{a:.3f})"
    if c.startswith("#") and len(c) == 7:
        if default_alpha < 1.0:
            r = int(c[1:3], 16)
            g = int(c[3:5], 16)
            b = int(c[5:7], 16)
            return f"rgba({r},{g},{b},{default_alpha})"
        return c
    return c


def _rgba_tuple(color_str: str, alpha: float) -> Tuple[float, float, float, float]:
    import re

    c = to_valid_color(color_str, default_alpha=alpha)
    if c.startswith("#") and len(c) == 7:
        r = int(c[1:3], 16) / 255.0
        g = int(c[3:5], 16) / 255.0
        b = int(c[5:7], 16) / 255.0
        return (r, g, b, alpha)
    if c.startswith("rgba(") and c.endswith(")"):
        parts = [p.strip() for p in c[5:-1].split(",")]
        if len(parts) == 4:
            return (float(parts[0]) / 255.0, float(parts[1]) / 255.0, float(parts[2]) / 255.0, float(parts[3]))
    if c.startswith("rgb(") and c.endswith(")"):
        nums = [float(x) for x in re.findall(r"[0-9\\.]+", c)]
        if len(nums) >= 3:
            return (nums[0] / 255.0, nums[1] / 255.0, nums[2] / 255.0, alpha)
    return (0.533, 0.533, 0.533, alpha)


def _sample_rows_unique(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0]
    if len(df) <= n:
        return df
    idx = rng.choice(len(df), size=int(n), replace=False)
    return df.iloc[idx]


@dataclass(frozen=True)
class Transition:
    target_label: str
    count: int
    frac: float
    centroid_xy: Tuple[float, float]


def _keep_targets_by_cumfrac(
    transitions: Sequence[Transition],
    cumfrac: Optional[float],
    top_k: Optional[int],
    min_frac: Optional[float],
    min_count: Optional[int],
) -> list[Transition]:
    out = list(transitions)
    if min_count is not None:
        out = [t for t in out if t.count >= int(min_count)]
    if min_frac is not None:
        out = [t for t in out if t.frac >= float(min_frac)]
    out.sort(key=lambda t: t.count, reverse=True)
    if top_k is not None and top_k > 0:
        out = out[: int(top_k)]
    if cumfrac is not None:
        cf = float(cumfrac)
        if not (0.0 < cf <= 1.0):
            raise ValueError("cumfrac must be in (0, 1].")
        kept = []
        running = 0.0
        for t in out:
            kept.append(t)
            running += float(t.frac)
            if running >= cf:
                break
        out = kept
    return out


def simulate_sde_points_from_x0(
    *,
    x0_np: np.ndarray,
    dim: int,
    f_net,
    score_net,
    ts_points: Sequence[float],
    dt: float,
    sigma: float,
    include_score: bool,
    interaction_m: int,
    device: str,
    verbose: bool = True,
):
    import torch
    from DeepRUOT.interaction import cal_interaction
    from DeepRUOT.utils import euler_sdeint

    x0 = torch.tensor(np.asarray(x0_np, dtype=np.float32), dtype=torch.float32, device=device)
    if include_score:
        x0 = x0.requires_grad_()
    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / x0.shape[0])
    initial_state = (x0, lnw0)

    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, ode_drift, g, score, interaction, sigma):
            super().__init__()
            self.drift = ode_drift
            self.score = score
            self.interaction = interaction
            self.g_net = g
            self.sigma = float(sigma)

        def f(self, t, y):
            z, lnw = y
            with torch.no_grad():
                drift = self.drift(t, z)
                dlnw = self.g_net(t, z)
                net_forces = cal_interaction(z, lnw, self.interaction, t, m=int(interaction_m))
            if include_score:
                t_expand = t.expand(z.shape[0], 1)
                with torch.enable_grad():
                    z_req = z.detach().requires_grad_(True)
                    drift = drift + self.score.compute_gradient(t_expand, z_req)
            return (drift + net_forces, dlnw)

        def g(self, t, y):
            return torch.ones_like(y) * self.sigma

    if verbose:
        t_min = float(min(ts_points))
        t_max = float(max(ts_points))
        est_steps = int(round((t_max - t_min) / dt)) if dt > 0 else None
        print(
            "[simulate_sde_points_from_x0] start | "
            f"n_init={x0.shape[0]}, ts_points={len(ts_points)}, dt={dt}, sigma={sigma}, "
            f"include_score={include_score}, t_range=({t_min},{t_max}), est_steps={est_steps}"
        )

    sde = SDE(f_net.v_net, f_net.g_net, score_net, f_net.interaction_net, sigma=sigma)
    ts_tensor = torch.tensor(list(ts_points), dtype=torch.float32, device=device)
    sde_point, traj_lnw = euler_sdeint(sde, initial_state, dt=float(dt), ts=ts_tensor)
    weight = torch.exp(traj_lnw)
    weight_normed = weight / weight.sum(dim=1, keepdim=True)

    sde_point_np = [p.detach().cpu().numpy() for p in sde_point]
    if verbose:
        print(
            "[simulate_sde_points_from_x0] done | "
            f"timepoints={len(sde_point_np)}, shape0={sde_point_np[0].shape if sde_point_np else None}"
        )
    return np.array(sde_point_np, dtype=object), weight_normed.detach().cpu().numpy()


def simulate_split_sde_points_from_x0(
    *,
    x0_np: np.ndarray,
    dim: int,
    f_net,
    score_net,
    ts_points: Sequence[float],
    dt: float,
    sigma: float,
    growth_alpha: float,
    interaction_m: int,
    device: str,
    verbose: bool = True,
):
    import torch
    from DeepRUOT.interaction import cal_interaction, euler_sdeint_split

    x0 = torch.tensor(np.asarray(x0_np, dtype=np.float32), dtype=torch.float32, device=device)
    lnw0 = torch.log(torch.ones(x0.shape[0], 1, device=device) / x0.shape[0])
    initial_state = (x0, lnw0)

    class SDE(torch.nn.Module):
        noise_type = "diagonal"
        sde_type = "ito"

        def __init__(self, ode_drift, g, score, interaction, sigma):
            super().__init__()
            self.drift = ode_drift
            self.score = score
            self.sigma = float(sigma)
            self.interaction = interaction
            self.g_net = g

        def f(self, t, y):
            z, lnw = y
            with torch.no_grad():
                drift = self.drift(t, z)
                dlnw = self.g_net(t, z) * float(growth_alpha)
                net_forces = cal_interaction(z, lnw, self.interaction, t, m=int(interaction_m))
            t_expand = t.expand(z.shape[0], 1)
            score_grad = self.score.compute_gradient(t_expand, z)
            return (drift + score_grad + net_forces, dlnw)

        def g(self, t, y):
            return torch.ones_like(y) * self.sigma

    if verbose:
        t_min = float(min(ts_points))
        t_max = float(max(ts_points))
        est_steps = int(round((t_max - t_min) / dt)) if dt > 0 else None
        print(
            "[simulate_split_sde_points_from_x0] start | "
            f"n_init={x0.shape[0]}, ts_points={len(ts_points)}, dt={dt}, sigma={sigma}, growth_alpha={growth_alpha}, "
            f"t_range=({t_min},{t_max}), est_steps={est_steps}"
        )

    sde = SDE(f_net.v_net, f_net.g_net, score_net, f_net.interaction_net, sigma=sigma)
    ts_tensor = torch.tensor(list(ts_points), dtype=torch.float32, device=device)
    sde_points, _ = euler_sdeint_split(sde, initial_state, dt=float(dt), ts=ts_tensor, noise_std=0.0)
    sde_point_np = [p.detach().cpu().numpy() for p in sde_points]
    if verbose:
        print(
            "[simulate_split_sde_points_from_x0] done | "
            f"timepoints={len(sde_point_np)}, shape0={sde_point_np[0].shape if sde_point_np else None}"
        )
    return np.array(sde_point_np, dtype=object)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MOSTA: single-source (0->1) lineage transition spatial plot")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/mosta_config.yaml"))
    parser.add_argument(
        "--data-csv",
        default=str(PROJECT_ROOT / "evaluation/mosta/data/mosta_four_time_with_celltype_refined.csv"),
    )
    parser.add_argument("--annotation-key", default="Annotation")
    parser.add_argument(
        "--color-h5ad",
        default=str(PROJECT_ROOT / "spatial_data/Mouse_embryo_all_stage.h5ad"),
    )
    parser.add_argument("--label-color-json", default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results/mosta_cartilage_transition_0_1"))

    parser.add_argument("--source-label", default="Cartilage primordium")
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=1.0)

    parser.add_argument(
        "--t0-background",
        choices=["auto", "real", "split-sde"],
        default="auto",
        help="Background (and source selection) at t0: real | split-sde | auto (real if observed else split-sde).",
    )
    parser.add_argument("--t0-bg-split-sde-n-samples", type=int, default=50000)
    parser.add_argument(
        "--t1-background",
        choices=["auto", "real", "split-sde"],
        default="auto",
        help="Background at t1: real | split-sde | auto (real if observed else split-sde).",
    )
    parser.add_argument("--t1-bg-split-sde-n-samples", type=int, default=50000)
    parser.add_argument("--split-sde-dt", type=float, default=0.05)
    parser.add_argument("--split-sde-sigma", type=float, default=0.03)
    parser.add_argument("--split-growth-alpha", type=float, default=1.0)
    parser.add_argument("--split-interaction-m", type=int, default=1024)
    parser.add_argument(
        "--t0-source-selection",
        choices=["classifier", "propagate", "auto"],
        default="classifier",
        help=(
            "When --t0-background is split-sde, how to select the source-label region at t0: "
            "classifier (predict labels at t0 and pick source-label points), "
            "propagate (take source-label at the earliest observed time and split-SDE to t0), "
            "auto (try classifier first, then propagate)."
        ),
    )
    parser.add_argument(
        "--t0-source-split-sde-n-samples",
        type=int,
        default=20000,
        help=(
            "When --t0-background is split-sde, number of source-label cells sampled at the earliest "
            "observed time to simulate into t0."
        ),
    )
    parser.add_argument(
        "--t0-source-min-count",
        type=int,
        default=100,
        help="Minimum number of source-label points required at t0; if fewer, may fallback depending on --t0-source-selection.",
    )

    parser.add_argument("--sde-dt", type=float, default=0.05)
    parser.add_argument("--sde-sigma", type=float, default=0.0)
    parser.add_argument("--include-score", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--interaction-m", type=int, default=512)
    parser.add_argument("--sde-n-samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling/training/simulation.")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Deprecated alias of --seed (kept for compatibility).",
    )

    parser.add_argument("--classifier-epochs", type=int, default=500)
    parser.add_argument("--classifier-hidden", type=int, default=128)
    parser.add_argument("--classifier-feature-dim", type=int, default=12)
    parser.add_argument(
        "--classifier-best-metric",
        choices=["accuracy", "bacc"],
        default="bacc",
        help="Metric used to keep the best classifier epoch (must match cache metadata to reuse cache).",
    )
    parser.add_argument(
        "--classifier-train-on-full-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train classifier on full data (must match cache metadata to reuse cache).",
    )
    parser.add_argument(
        "--classifier-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse/save classifier weights in a cache dir to avoid retraining every run.",
    )
    parser.add_argument("--classifier-cache-dir", default=None)
    parser.add_argument("--classifier-cache-tag", default=None)

    parser.add_argument("--knn-neighbors", type=int, default=10)

    parser.add_argument("--bg-max-cells", type=int, default=20000, help="Max background points per timepoint")
    parser.add_argument("--bg-color", type=str, default="#6f6f6f", help="Background point color (hex/rgb/rgba).")
    parser.add_argument("--bg-alpha", type=float, default=0.25)
    parser.add_argument("--bg-size", type=float, default=2.0)
    parser.add_argument("--source-vis-max-cells", type=int, default=20000, help="Max source points to render (visual only)")
    parser.add_argument("--pt-size", type=float, default=4.0)
    parser.add_argument("--pt-alpha", type=float, default=0.90)
    parser.add_argument("--panel-border-color-observed", type=str, default="#5f6a72")
    parser.add_argument("--panel-border-color-generated", type=str, default="#8c6d5a")
    parser.add_argument("--panel-fill-color-observed", type=str, default="#e6f0f6")
    parser.add_argument("--panel-fill-color-generated", type=str, default="#f6eee5")
    parser.add_argument("--panel-fill-opacity", type=float, default=0.0)
    parser.add_argument("--panel-border-width", type=float, default=5.0)
    parser.add_argument(
        "--panel-style",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw observed/generated panel borders/fills (off by default).",
    )
    parser.add_argument(
        "--plot-unselected-sim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, also plot non-selected simulated points at t1 using the same gray as background.",
    )

    parser.add_argument("--keep-target-cumfrac", type=float, default=None, help="Deprecated alias of --fate-keep-source-cumfrac.")
    parser.add_argument("--fate-keep-source-cumfrac", type=float, default=0.85)
    parser.add_argument("--keep-target-top-k", type=int, default=None)
    parser.add_argument("--keep-target-min-frac", type=float, default=0)
    parser.add_argument("--keep-target-min-count", type=int, default=None)

    parser.add_argument("--line-width-base", type=float, default=2.0)
    parser.add_argument("--line-width-scale", type=float, default=18.0)
    parser.add_argument("--line-alpha", type=float, default=0.35)
    parser.add_argument("--line-curvature", type=float, default=0.20)
    parser.add_argument("--centroid-size", type=float, default=80.0)

    parser.add_argument("--gap-frac", type=float, default=0.15, help="Gap (as fraction of x-span) between panels")
    parser.add_argument("--title", default=None)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.random_seed is not None:
        args.seed = int(args.random_seed)

    args.config = _resolve_under_root(args.config)
    args.data_csv = _resolve_under_root(args.data_csv)
    args.color_h5ad = _resolve_under_root(args.color_h5ad)
    args.label_color_json = _resolve_under_root(args.label_color_json)
    args.output_dir = _resolve_under_root(args.output_dir) or args.output_dir
    args.classifier_cache_dir = _resolve_under_root(args.classifier_cache_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    # Reproducibility
    import random

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    try:
        import torch

        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        from evaluation.arista_code.arista_helpers import (
            load_config,
            load_models,
            predict_labels_for_trajectories,
            train_mlp_classifier,
        )
    except ModuleNotFoundError as exc:
        msg = (
            f"Missing dependency: {exc.name!r}. "
            "Please run this script inside the DeepRUOTv2 environment (which should include required deps "
            "like torch_geometric)."
        )
        raise ModuleNotFoundError(msg) from exc

    rng = np.random.default_rng(int(args.seed))

    config = load_config(args.config)
    dim = int(config["data"]["dim"])
    feature_cols_full = [f"x{i}" for i in range(1, dim + 1)]
    feat_dim = int(args.classifier_feature_dim)
    feat_dim = max(2, min(int(dim), feat_dim))
    clf_cols = ["samples"] + [f"x{i}" for i in range(1, feat_dim + 1)]

    df = pd.read_csv(args.data_csv, low_memory=False)
    _require_columns(df, ["samples", args.annotation_key] + feature_cols_full, ctx=args.data_csv)
    df = df.copy()
    df["samples"] = df["samples"].astype(float)
    df[args.annotation_key] = df[args.annotation_key].astype(str)
    df = df.sort_values("samples").reset_index(drop=True)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    f_net, score_net, exp_dir, device = load_models(
        config,
        exp_name=config["exp"]["name"],
        device=device,
        model_tag="model_final",
        score_tag="score_model",
    )
    print("Project root:", PROJECT_ROOT)
    print("Device:", device)
    print("Experiment dir:", exp_dir)
    print("Data:", args.data_csv, "| rows:", len(df))

    # Colors
    label_to_color = load_label_to_color(
        df[args.annotation_key].astype(str).values,
        label_color_json=args.label_color_json,
        color_h5ad=args.color_h5ad,
        annotation_key=args.annotation_key,
    )
    missing = sorted(set(df[args.annotation_key].astype(str).unique()) - set(label_to_color.keys()))
    if missing:
        print(f"[warn] {len(missing)} labels missing in colormap (will fallback per-label): {missing[:10]}")

    observed_ts = sorted(float(x) for x in df["samples"].unique())

    # Background points for t0/t1 (real observed or split-SDE generated)
    df_t0 = df[df["samples"] == float(args.t0)]
    df_t1 = df[df["samples"] == float(args.t1)]
    t0_is_observed = len(df_t0) > 0
    t1_is_observed = len(df_t1) > 0

    t0_bg_mode = str(args.t0_background)
    if t0_bg_mode == "auto":
        t0_bg_mode = "real" if t0_is_observed else "split-sde"
    if t0_bg_mode == "real" and not t0_is_observed:
        raise ValueError(f"--t0-background real requested, but t0={args.t0} is not an observed timepoint.")

    t1_bg_mode = str(args.t1_background)
    if t1_bg_mode == "auto":
        t1_bg_mode = "real" if t1_is_observed else "split-sde"
    if t1_bg_mode == "real":
        if not t1_is_observed:
            raise ValueError(f"--t1-background real requested, but t1={args.t1} is not an observed timepoint.")
        df_t1_bg = _sample_rows_unique(df_t1, int(args.bg_max_cells), rng)
        xy_t1_bg = df_t1_bg[["x1", "x2"]].values.astype(np.float32)
    else:
        prev_candidates_t1 = [t for t in observed_ts if t <= float(args.t1)]
        if not prev_candidates_t1:
            raise ValueError(f"Cannot run split-SDE background for t1={args.t1}: no observed timepoint <= t1.")
        t_prev_t1 = float(prev_candidates_t1[-1])
        df_prev_t1 = df[df["samples"] == t_prev_t1]
        if len(df_prev_t1) == 0:
            raise ValueError(f"No rows found for t_prev={t_prev_t1} (needed for t1 split-SDE init).")
        n_init_t1 = min(int(args.t1_bg_split_sde_n_samples), len(df_prev_t1))
        if n_init_t1 <= 0:
            raise ValueError("--t1-bg-split-sde-n-samples must be > 0.")
        df_prev_t1_sampled = _sample_rows_unique(df_prev_t1, n_init_t1, rng).reset_index(drop=True)
        x_prev_t1 = df_prev_t1_sampled[feature_cols_full].values.astype(np.float32)
        if float(t_prev_t1) == float(args.t1):
            x_t1_bg = x_prev_t1
        else:
            ts_split_t1 = [t_prev_t1, float(args.t1)]
            sde_points_split_t1 = simulate_split_sde_points_from_x0(
                x0_np=x_prev_t1,
                dim=dim,
                f_net=f_net,
                score_net=score_net,
                ts_points=ts_split_t1,
                dt=float(args.split_sde_dt),
                sigma=float(args.split_sde_sigma),
                growth_alpha=float(args.split_growth_alpha),
                interaction_m=int(args.split_interaction_m),
                device=device,
                verbose=True,
            )
            x_t1_bg = np.asarray(sde_points_split_t1[1], dtype=np.float32)
        xy_t1_bg = x_t1_bg[:, :2]
        print(f"[t1 split-sde] t_prev={t_prev_t1} -> t1={args.t1} | bg_points={xy_t1_bg.shape[0]}")

    # Train/reuse classifier
    cache_dir = None
    if args.classifier_cache:
        cache_dir = args.classifier_cache_dir or os.path.join(args.output_dir, "classifier_cache")
    print("Training/loading classifier | features=", len(clf_cols), "| epochs=", args.classifier_epochs)
    model, label_encoder, acc = train_mlp_classifier(
        df,
        feature_cols=clf_cols,
        label_col=args.annotation_key,
        hidden_size=int(args.classifier_hidden),
        epochs=int(args.classifier_epochs),
        seed=int(args.seed),
        cache_dir=cache_dir,
        cache_tag=args.classifier_cache_tag,
        df_source_path=str(Path(args.data_csv).resolve()),
        reuse_if_possible=bool(args.classifier_cache),
        progress=True,
        device=device,
        best_epoch_metric=str(args.classifier_best_metric),
        train_on_full_data=bool(args.classifier_train_on_full_data),
    )
    print(f"Classifier ready | accuracy={acc:.4f}")

    # Resolve cumfrac alias
    fate_keep_cumfrac = args.fate_keep_source_cumfrac
    if args.keep_target_cumfrac is not None:
        fate_keep_cumfrac = args.keep_target_cumfrac

    # Build t0 background + pick x0 points for trajectories (source-only).
    if t0_bg_mode == "real":
        df_t0_bg = _sample_rows_unique(df_t0, int(args.bg_max_cells), rng)
        xy_t0_bg = df_t0_bg[["x1", "x2"]].values.astype(np.float32)
        df_src = df_t0[df_t0[args.annotation_key] == str(args.source_label)]
        if len(df_src) == 0:
            raise ValueError(f"No rows found for source-label='{args.source_label}' at t0={args.t0}.")
        # visual
        df_src_vis = _sample_rows_unique(df_src, int(args.source_vis_max_cells), rng)
        xy_src_vis = df_src_vis[["x1", "x2"]].values.astype(np.float32)
        # trajectories
        n_src = min(int(args.sde_n_samples), len(df_src))
        df_src_sampled = _sample_rows_unique(df_src, n_src, rng).reset_index(drop=True)
        x0 = df_src_sampled[feature_cols_full].values.astype(np.float32)
        src_centroid = (float(xy_src_vis[:, 0].mean()), float(xy_src_vis[:, 1].mean()))
        print(f"Source label='{args.source_label}' | available={len(df_src)} | traj_sampled={n_src} | vis={len(df_src_vis)}")
    else:
        # Split-SDE generate t0 background from the earliest observed time (e.g., 0.0 -> t0).
        t_start = float(observed_ts[0])
        if float(t_start) > float(args.t0):
            raise ValueError(
                f"Cannot run split-SDE background for t0={args.t0}: earliest observed time {t_start} is > t0."
            )
        df_prev = df[df["samples"] == t_start]
        if len(df_prev) == 0:
            raise ValueError(f"No rows found for t_start={t_start} (needed for split-SDE init).")
        # global background
        n_init = min(int(args.t0_bg_split_sde_n_samples), len(df_prev))
        df_prev_sampled = _sample_rows_unique(df_prev, n_init, rng).reset_index(drop=True)
        x_prev = df_prev_sampled[feature_cols_full].values.astype(np.float32)
        ts_split = [t_start, float(args.t0)]
        if float(t_start) == float(args.t0):
            x_t0 = x_prev
            sde_points_split = np.array([x_prev, x_t0], dtype=object)
        else:
            sde_points_split = simulate_split_sde_points_from_x0(
                x0_np=x_prev,
                dim=dim,
                f_net=f_net,
                score_net=score_net,
                ts_points=ts_split,
                dt=float(args.split_sde_dt),
                sigma=float(args.split_sde_sigma),
                growth_alpha=float(args.split_growth_alpha),
                interaction_m=int(args.split_interaction_m),
                device=device,
                verbose=True,
            )
            x_t0 = np.asarray(sde_points_split[1], dtype=np.float32)
        xy_t0_bg = x_t0[:, :2]
        selection = str(args.t0_source_selection)
        if selection == "auto":
            selection = "classifier"

        x0_all = None

        def _select_by_classifier() -> Optional[np.ndarray]:
            predicted_split = predict_labels_for_trajectories(
                sde_points=sde_points_split,
                ts_points=ts_split,
                model=model,
                label_encoder=label_encoder,
                feature_dim=int(feat_dim),
                device=device,
                knn_neighbors=int(args.knn_neighbors),
            )
            labels_t0_pred = np.asarray(predicted_split[1]).astype(str)
            mask_src = labels_t0_pred == str(args.source_label)
            if int(mask_src.sum()) < int(args.t0_source_min_count):
                return None
            return x_t0[mask_src]

        def _select_by_propagate() -> Optional[np.ndarray]:
            df_prev_src = df_prev[df_prev[args.annotation_key] == str(args.source_label)]
            if len(df_prev_src) == 0:
                return None
            n_init_src = min(int(args.t0_source_split_sde_n_samples), len(df_prev_src))
            df_prev_src_sampled = _sample_rows_unique(df_prev_src, n_init_src, rng).reset_index(drop=True)
            x_prev_src = df_prev_src_sampled[feature_cols_full].values.astype(np.float32)
            sde_src = simulate_split_sde_points_from_x0(
                x0_np=x_prev_src,
                dim=dim,
                f_net=f_net,
                score_net=score_net,
                ts_points=ts_split,
                dt=float(args.split_sde_dt),
                sigma=float(args.split_sde_sigma),
                growth_alpha=float(args.split_growth_alpha),
                interaction_m=int(args.split_interaction_m),
                device=device,
                verbose=False,
            )
            x_src_t0 = np.asarray(sde_src[1], dtype=np.float32)
            if x_src_t0.shape[0] < int(args.t0_source_min_count):
                return None
            return x_src_t0

        if selection == "classifier":
            x0_all = _select_by_classifier()
            if x0_all is None and args.t0_source_selection == "auto":
                x0_all = _select_by_propagate()
        elif selection == "propagate":
            x0_all = _select_by_propagate()
        else:
            raise ValueError("--t0-source-selection must be classifier|propagate|auto")

        if x0_all is None:
            raise ValueError(
                f"Could not select enough source-label='{args.source_label}' points at t0={args.t0} "
                f"(need >= {int(args.t0_source_min_count)}). "
                "Try increasing --t0-bg-split-sde-n-samples and/or adjusting classifier, "
                "or use --t0-source-selection propagate."
            )

        # visual: show more source points (optional cap)
        if x0_all.shape[0] > int(args.source_vis_max_cells):
            idx_vis = rng.choice(int(x0_all.shape[0]), size=int(args.source_vis_max_cells), replace=False)
            xy_src_vis = x0_all[idx_vis, :2]
        else:
            xy_src_vis = x0_all[:, :2]

        # trajectories: sample subset
        n_src = min(int(args.sde_n_samples), int(x0_all.shape[0]))
        if n_src <= 0:
            raise ValueError("No source points available at t0 for trajectories.")
        if x0_all.shape[0] > n_src:
            idx = rng.choice(int(x0_all.shape[0]), size=int(n_src), replace=False)
            x0 = x0_all[idx]
        else:
            x0 = x0_all
        src_centroid = (float(xy_src_vis[:, 0].mean()), float(xy_src_vis[:, 1].mean()))
        print(
            f"[t0 split-sde] t_start={t_start} -> t0={args.t0} | "
            f"bg_points={x_t0.shape[0]} | src_points={x0_all.shape[0]} | traj_sampled={x0.shape[0]} | "
            f"vis={xy_src_vis.shape[0]} | source_selection={args.t0_source_selection}"
        )

    # Simulate non-split SDE for just these trajectories
    ts_points = [float(args.t0), float(args.t1)]
    sde_points, _ = simulate_sde_points_from_x0(
        x0_np=x0,
        dim=dim,
        f_net=f_net,
        score_net=score_net,
        ts_points=ts_points,
        dt=float(args.sde_dt),
        sigma=float(args.sde_sigma),
        include_score=bool(args.include_score),
        interaction_m=int(args.interaction_m),
        device=device,
        verbose=True,
    )
    xy_t1_sim = np.asarray(sde_points[1], dtype=np.float32)[:, :2]

    # Predict labels at t1 (and t0, though t0 are known)
    predicted = predict_labels_for_trajectories(
        sde_points=sde_points,
        ts_points=ts_points,
        model=model,
        label_encoder=label_encoder,
        feature_dim=int(feat_dim),
        device=device,
        knn_neighbors=int(args.knn_neighbors),
    )
    labels_t1_pred = np.asarray(predicted[1]).astype(str)

    # Transition distribution (source is fixed)
    uniq, cnt = np.unique(labels_t1_pred, return_counts=True)
    total = float(cnt.sum()) if cnt.size else 0.0
    transitions_all: list[Transition] = []
    for lab, c in sorted(zip(uniq.tolist(), cnt.tolist()), key=lambda x: x[1], reverse=True):
        mask = labels_t1_pred == lab
        if not mask.any():
            continue
        centroid = (float(xy_t1_sim[mask, 0].mean()), float(xy_t1_sim[mask, 1].mean()))
        frac = float(c) / total if total > 0 else 0.0
        transitions_all.append(Transition(target_label=str(lab), count=int(c), frac=frac, centroid_xy=centroid))

    transitions = _keep_targets_by_cumfrac(
        transitions_all,
        cumfrac=fate_keep_cumfrac,
        top_k=args.keep_target_top_k,
        min_frac=args.keep_target_min_frac,
        min_count=args.keep_target_min_count,
    )
    kept_labels = {t.target_label for t in transitions}
    print("Kept targets:", ", ".join([f"{t.target_label} ({t.frac*100:.1f}%)" for t in transitions]))

    # Plot (single axis with x-shifted right panel so we can draw cross-panel lines)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    bg_rgba = _rgba_tuple(args.bg_color, alpha=float(args.bg_alpha))

    # Downsample t0 background for visualization only (split-SDE can be huge).
    if args.bg_max_cells is not None and int(args.bg_max_cells) > 0 and xy_t0_bg.shape[0] > int(args.bg_max_cells):
        idx_bg = rng.choice(int(xy_t0_bg.shape[0]), size=int(args.bg_max_cells), replace=False)
        xy_t0_bg_vis = xy_t0_bg[idx_bg]
    else:
        xy_t0_bg_vis = xy_t0_bg
    xy_t1_bg_vis = xy_t1_bg  # already sampled above

    x_min = float(min(xy_t0_bg[:, 0].min(), xy_t1_bg[:, 0].min(), xy_src_vis[:, 0].min(), xy_t1_sim[:, 0].min()))
    x_max = float(max(xy_t0_bg[:, 0].max(), xy_t1_bg[:, 0].max(), xy_src_vis[:, 0].max(), xy_t1_sim[:, 0].max()))
    x_span = max(1e-6, x_max - x_min)
    gap = float(args.gap_frac) * x_span
    shift = x_span + gap

    xy_t1_bg_shift = xy_t1_bg_vis.copy()
    xy_t1_bg_shift[:, 0] += shift
    xy_t1_sim_shift = xy_t1_sim.copy()
    xy_t1_sim_shift[:, 0] += shift

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Panel styling (optional)
    y_min = float(min(xy_t0_bg[:, 1].min(), xy_t1_bg[:, 1].min(), xy_src_vis[:, 1].min(), xy_t1_sim[:, 1].min()))
    y_max = float(max(xy_t0_bg[:, 1].max(), xy_t1_bg[:, 1].max(), xy_src_vis[:, 1].max(), xy_t1_sim[:, 1].max()))
    if args.panel_style:
        from matplotlib.patches import Rectangle

        t0_panel_kind = "generated" if t0_bg_mode == "split-sde" else "observed"
        t1_panel_kind = "generated" if t1_bg_mode == "split-sde" else "observed"

        def _panel_style(kind: str):
            if kind == "generated":
                border = args.panel_border_color_generated
                fill = args.panel_fill_color_generated
            else:
                border = args.panel_border_color_observed
                fill = args.panel_fill_color_observed
            return border, fill

        border0, fill0 = _panel_style(t0_panel_kind)
        border1, fill1 = _panel_style(t1_panel_kind)

        panel_h = y_max - y_min
        panel_w = x_span
        rect0 = Rectangle(
            (x_min, y_min),
            panel_w,
            panel_h,
            facecolor=_rgba_tuple(fill0, alpha=float(args.panel_fill_opacity)),
            edgecolor=to_valid_color(border0, default_alpha=1.0),
            linewidth=float(args.panel_border_width),
            zorder=0,
        )
        rect1 = Rectangle(
            (x_min + shift, y_min),
            panel_w,
            panel_h,
            facecolor=_rgba_tuple(fill1, alpha=float(args.panel_fill_opacity)),
            edgecolor=to_valid_color(border1, default_alpha=1.0),
            linewidth=float(args.panel_border_width),
            zorder=0,
        )
        ax.add_patch(rect0)
        ax.add_patch(rect1)

    # Background
    ax.scatter(
        xy_t0_bg_vis[:, 0],
        xy_t0_bg_vis[:, 1],
        s=float(args.bg_size),
        c=[bg_rgba],
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        xy_t1_bg_shift[:, 0],
        xy_t1_bg_shift[:, 1],
        s=float(args.bg_size),
        c=[bg_rgba],
        linewidths=0,
        rasterized=True,
    )

    # Source points at t0 (colored)
    src_color = label_to_color.get(str(args.source_label), "#bf024f")
    ax.scatter(
        xy_src_vis[:, 0],
        xy_src_vis[:, 1],
        s=float(args.pt_size),
        c=[_rgba_tuple(src_color, alpha=float(args.pt_alpha))],
        linewidths=0,
        rasterized=True,
        label=f"{args.source_label} (t={args.t0:g})",
    )

    # Simulated points at t1 (colored by predicted label; default: only plot selected targets)
    if args.plot_unselected_sim:
        colors_t1 = []
        for lab in labels_t1_pred:
            if lab in kept_labels:
                colors_t1.append(_rgba_tuple(label_to_color.get(lab, "#888888"), alpha=float(args.pt_alpha)))
            else:
                colors_t1.append(bg_rgba)
        ax.scatter(
            xy_t1_sim_shift[:, 0],
            xy_t1_sim_shift[:, 1],
            s=float(args.pt_size),
            c=colors_t1,
            linewidths=0,
            rasterized=True,
        )
    else:
        mask_keep = np.array([lab in kept_labels for lab in labels_t1_pred], dtype=bool)
        if mask_keep.any():
            labs_keep = labels_t1_pred[mask_keep]
            cols_keep = [_rgba_tuple(label_to_color.get(lab, "#888888"), alpha=float(args.pt_alpha)) for lab in labs_keep]
            ax.scatter(
                xy_t1_sim_shift[mask_keep, 0],
                xy_t1_sim_shift[mask_keep, 1],
                s=float(args.pt_size),
                c=cols_keep,
                linewidths=0,
                rasterized=True,
            )

    # Centroids
    ax.scatter(
        [src_centroid[0]],
        [src_centroid[1]],
        s=float(args.centroid_size),
        c=[_rgba_tuple(src_color, alpha=1.0)],
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )

    # Flow lines
    src_rgba = _rgba_tuple(src_color, alpha=float(args.line_alpha))
    for tr in transitions:
        tgt_cx, tgt_cy = tr.centroid_xy
        tgt_cx_shift = tgt_cx + shift
        ax.scatter(
            [tgt_cx_shift],
            [tgt_cy],
            s=float(args.centroid_size) * 0.75,
            c=[_rgba_tuple(label_to_color.get(tr.target_label, "#888888"), alpha=1.0)],
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        lw = float(args.line_width_base) + float(args.line_width_scale) * float(tr.frac)
        rad = float(args.line_curvature)
        patch = FancyArrowPatch(
            posA=(src_centroid[0], src_centroid[1]),
            posB=(tgt_cx_shift, tgt_cy),
            arrowstyle="-",
            mutation_scale=10,
            lw=lw,
            color=src_rgba,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
        ax.add_patch(patch)
        ax.text(
            tgt_cx_shift,
            tgt_cy,
            f"{tr.target_label}\n{tr.frac*100:.1f}%",
            fontsize=9,
            ha="left",
            va="center",
            color="#111111",
        )

    # Annotations: time labels and separator
    y_span = max(1e-6, y_max - y_min)
    y_text = y_min - 0.06 * y_span
    ax.text(x_min + 0.02 * x_span, y_text, f"t = {args.t0:g}", fontsize=13, weight="bold")
    ax.text(x_min + shift + 0.02 * x_span, y_text, f"t = {args.t1:g}", fontsize=13, weight="bold")
    ax.plot([x_min + shift - gap / 2, x_min + shift - gap / 2], [y_min, y_max], color="#dddddd", lw=2)

    title = args.title or f"{args.source_label}: lineage transition {args.t0:g} → {args.t1:g}"
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    stem = f"lineage_transition__{args.source_label}__t{args.t0:g}_to_t{args.t1:g}"
    stem = stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
    out_base = os.path.join(args.output_dir, stem)
    png_path = out_base + ".png"
    svg_path = out_base + ".svg"
    pdf_path = out_base + ".pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved:", png_path)
    if not args.skip_export:
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print("Saved:", svg_path)
        print("Saved:", pdf_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
