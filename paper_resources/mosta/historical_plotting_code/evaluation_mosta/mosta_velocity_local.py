#!/usr/bin/env python3
"""
Local runnable MOSTA velocity plotting script.

Goal:
- Use MOSTA velocity inputs (CSV + model checkpoints) and redraw the plots that
  `evaluation/mosta/code/mosta_velocity.ipynb` generates, but following the
  reusable logic in `evaluation/arista_code/mosta_ported/velocity.py`
  (i.e. the "velocity_migration" implementation style).

Notes:
- This script expects the DeepRUOT environment (torch, torch_geometric, scanpy,
  scvelo, etc.) to be available.
- Defaults target the files that exist in this repo:
  - data: `evaluation/mosta/data/mosta_four_time_with_celltype_refined.csv`
  - communication: `evaluation/mosta/data/all_timepoint_communications_merged.pkl`
  - config: `config/mosta_config.yaml`
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@dataclass(frozen=True)
class TimepointSpec:
    idx: float
    label: str


TELENCEPHALON_PALETTE: Dict[str, str] = {
    "Apical Progenitors (RG)": "#1f77b4",
    "Basal Progenitors (IP)": "#aec7e8",
    "Immature Neurons": "#2ca02c",
    "Excitatory Neurons": "#ffbb78",
    "Inhibitory Neurons": "#9467bd",
    "Cajal-Retzius Cells": "#e377c2",
    "Glioblasts": "#8c564b",
    "Choroid Plexus": "#7f7f7f",
    "Other": "#d9d9d9",
}


def _ensure_project_root_on_path() -> None:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def _parse_csv_list(value: str) -> List[str]:
    parts = [p.strip() for p in (value or "").split(",")]
    return [p for p in parts if p]


def _build_palette(categories: Sequence[str]) -> Dict[str, str]:
    import scanpy as sc

    categories = [str(c) for c in categories]
    if len(categories) <= 20:
        colors = list(sc.pl.palettes.vega_20)
    elif len(categories) <= 28:
        colors = list(sc.pl.palettes.zeileis_28)
    else:
        colors = list(sc.pl.palettes.godsnot_102)
    if len(categories) > len(colors):
        colors = colors * (len(categories) // len(colors) + 1)
    return {cat: colors[i] for i, cat in enumerate(categories)}


def _merge_palettes(base: Dict[str, str], extra: Dict[str, str]) -> Dict[str, str]:
    merged = dict(base)
    for k, v in (extra or {}).items():
        if k not in merged:
            merged[k] = v
    return merged


def _load_palette_from_h5ad(
    h5ad_path: str,
    obs_key_candidates: Sequence[str],
) -> Optional[Dict[str, str]]:
    if not h5ad_path or not os.path.exists(h5ad_path):
        return None

    try:
        import scanpy as sc
    except ModuleNotFoundError:
        return None

    # backed='r' avoids loading X into memory for large h5ad files
    adata = sc.read_h5ad(h5ad_path, backed="r")
    try:
        for obs_key in obs_key_candidates:
            if obs_key not in adata.obs:
                continue
            colors_key = f"{obs_key}_colors"
            if colors_key not in adata.uns:
                continue
            colors = list(adata.uns[colors_key])
            series = adata.obs[obs_key]
            if hasattr(series, "cat"):
                categories = [str(x) for x in series.cat.categories.tolist()]
            else:
                categories = sorted({str(x) for x in series.astype(str).unique().tolist()})
            if not categories or not colors:
                continue
            if len(colors) < len(categories):
                # Repeat if needed; common when palette list is shorter than categories
                colors = colors * (len(categories) // len(colors) + 1)
            return {str(cat): str(colors[i]) for i, cat in enumerate(categories)}
    finally:
        try:
            adata.file.close()
        except Exception:
            pass

    return None


def _load_palette_cache(cache_path: str) -> Optional[Dict[str, Dict[str, str]]]:
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _save_palette_cache(cache_path: str, payload: Dict[str, Dict[str, str]]) -> None:
    if not cache_path:
        return
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _save_palette_legend(palette: Dict[str, str], save_path: str) -> str:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

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
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return save_path


def _save_velocity_figs(
    figs,
    base_path_no_ext: str,
    bg_cell: Optional[str],
    focus_cell: Optional[str],
    zoom: bool,
    prefer_format: str = "pdf",
) -> List[str]:
    import matplotlib as mpl
    import matplotlib.collections as mcoll
    import matplotlib.patches as mpatches

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    bg_s = f"_bg-{bg_cell}" if bg_cell else ""
    focus_s = f"_focus-{focus_cell}" if focus_cell else ""
    zoom_s = "_zoom" if zoom else ""

    out_paths: List[str] = []

    def _rasterize_heavy_artists(fig) -> None:
        for ax in fig.axes:
            for artist in ax.get_children():
                if isinstance(artist, (mcoll.Collection, mpatches.Patch)):
                    try:
                        artist.set_rasterized(True)
                    except Exception:
                        pass

    def _save_one(fig, suffix: str) -> None:
        nonlocal out_paths
        os.makedirs(os.path.dirname(base_path_no_ext), exist_ok=True)

        def _path(ext: str) -> str:
            return f"{base_path_no_ext}{bg_s}{focus_s}{zoom_s}_{suffix}.{ext}"

        preferred = _path(prefer_format)
        fallback = _path("png")
        try:
            fig.savefig(preferred, dpi=300, bbox_inches="tight")
            out_paths.append(preferred)
        except ValueError as exc:
            # Common with PDF backend when streamlines contain non-finite vertices.
            msg = str(exc)
            if "finite numbers" in msg:
                if prefer_format == "pdf":
                    _rasterize_heavy_artists(fig)
                    try:
                        fig.savefig(preferred, dpi=300, bbox_inches="tight")
                        out_paths.append(preferred)
                        print("Info: PDF save succeeded after rasterizing heavy artists:", preferred)
                        return
                    except ValueError:
                        pass
                fig.savefig(fallback, dpi=300, bbox_inches="tight")
                out_paths.append(fallback)
                print(f"Warning: save as '{prefer_format}' failed; wrote PNG instead:", fallback)
            else:
                raise

    if len(figs) >= 1:
        _save_one(figs[0], "intrinsic")
    if len(figs) >= 2:
        _save_one(figs[1], "interaction")

    return out_paths


def _save_component_fig(
    fig,
    base_path_no_ext: str,
    suffix: str,
    bg_cell: Optional[str],
    focus_cell: Optional[str],
    zoom: bool,
    prefer_format: str = "pdf",
) -> str:
    import matplotlib as mpl
    import matplotlib.collections as mcoll
    import matplotlib.patches as mpatches

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    bg_s = f"_bg-{bg_cell}" if bg_cell else ""
    focus_s = f"_focus-{focus_cell}" if focus_cell else ""
    zoom_s = "_zoom" if zoom else ""

    def _rasterize_heavy_artists() -> None:
        for ax in fig.axes:
            for artist in ax.get_children():
                if isinstance(artist, (mcoll.Collection, mpatches.Patch)):
                    try:
                        artist.set_rasterized(True)
                    except Exception:
                        pass

    os.makedirs(os.path.dirname(base_path_no_ext), exist_ok=True)
    preferred = f"{base_path_no_ext}{bg_s}{focus_s}{zoom_s}_{suffix}.{prefer_format}"
    fallback = f"{base_path_no_ext}{bg_s}{focus_s}{zoom_s}_{suffix}.png"
    try:
        fig.savefig(preferred, dpi=300, bbox_inches="tight")
        return preferred
    except ValueError as exc:
        msg = str(exc)
        if "finite numbers" in msg and prefer_format == "pdf":
            _rasterize_heavy_artists()
            try:
                fig.savefig(preferred, dpi=300, bbox_inches="tight")
                print("Info: PDF save succeeded after rasterizing heavy artists:", preferred)
                return preferred
            except ValueError:
                pass
        fig.savefig(fallback, dpi=300, bbox_inches="tight")
        print(f"Warning: save as '{prefer_format}' failed; wrote PNG instead:", fallback)
        return fallback


def _load_df(data_csv: str):
    import pandas as pd

    df = pd.read_csv(data_csv, low_memory=False)
    if "samples" not in df.columns:
        raise ValueError(f"Missing required column 'samples' in {data_csv}")
    df = df.copy()
    df["samples"] = df["samples"].astype(float)

    # Ensure expected feature columns exist and are numeric.
    for i in range(1, 53):
        col = f"x{i}"
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {data_csv}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure common label columns are string categories with a stable "Other" fill.
    for col in ("Annotation", "celltype", "telencephalon"):
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), "Other").astype(str)
            df[col] = df[col].replace({"nan": "Other", "None": "Other"})

    # Guard against NaNs in model inputs.
    feature_cols = [f"x{i}" for i in range(1, 53)]
    n_bad = int(df[feature_cols].isna().any(axis=1).sum())
    if n_bad:
        raise ValueError(
            f"Found {n_bad} rows with NaNs in x1..x52 after numeric coercion. "
            "Please fix the input CSV (or regenerate it) before plotting."
        )

    return df


def _load_communications(path: str) -> Optional[Dict]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Communication pickle not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Communication pickle must be a dict, got {type(obj)}")
    return obj


def _subsample_df_by_timepoint(
    df,
    max_cells_per_timepoint: Optional[int],
    random_state: int,
    stratify_col: Optional[str] = "Annotation",
):
    import numpy as np
    import pandas as pd

    if not max_cells_per_timepoint or max_cells_per_timepoint <= 0:
        return df

    out = []
    rng = np.random.default_rng(random_state)

    for t in sorted(df["samples"].unique()):
        df_t = df[df["samples"] == t]
        n = len(df_t)
        if n <= max_cells_per_timepoint:
            out.append(df_t)
            continue

        if stratify_col and stratify_col in df_t.columns:
            groups = list(df_t.groupby(stratify_col, sort=False))
            sizes = np.array([len(g) for _, g in groups], dtype=int)
            total = sizes.sum()
            if total <= 0:
                out.append(df_t.sample(n=max_cells_per_timepoint, random_state=random_state))
                continue

            # Proportional allocation with at least 1 per non-empty group.
            raw = sizes / total * max_cells_per_timepoint
            base = np.floor(raw).astype(int)
            base = np.maximum(base, 1)
            base = np.minimum(base, sizes)
            remainder = raw - np.floor(raw)

            def current_sum() -> int:
                return int(base.sum())

            # Adjust to hit target sum.
            if current_sum() < max_cells_per_timepoint:
                order = np.argsort(-remainder)
                for idx in order:
                    if current_sum() >= max_cells_per_timepoint:
                        break
                    if base[idx] < sizes[idx]:
                        base[idx] += 1
            elif current_sum() > max_cells_per_timepoint:
                order = np.argsort(remainder)
                for idx in order:
                    if current_sum() <= max_cells_per_timepoint:
                        break
                    if base[idx] > 1:
                        base[idx] -= 1

            # If still off (rare due to caps), fix with random trimming/adding.
            while current_sum() > max_cells_per_timepoint:
                cand = np.where(base > 1)[0]
                if cand.size == 0:
                    break
                base[int(rng.choice(cand))] -= 1
            while current_sum() < max_cells_per_timepoint:
                cand = np.where(base < sizes)[0]
                if cand.size == 0:
                    break
                base[int(rng.choice(cand))] += 1

            sampled = []
            for (lab, g), k in zip(groups, base):
                if k <= 0:
                    continue
                # Different seed per (timepoint, label) for determinism.
                seed = abs(hash((float(t), str(lab), random_state))) % (2**32)
                sampled.append(g.sample(n=int(k), random_state=seed))
            out.append(pd.concat(sampled, ignore_index=False))
        else:
            out.append(df_t.sample(n=max_cells_per_timepoint, random_state=random_state))

    return pd.concat(out, ignore_index=False).reset_index(drop=True)


def _parse_timepoints(labels_csv: str, df_timepoints: Sequence[float]) -> List[TimepointSpec]:
    labels = _parse_csv_list(labels_csv)
    tps = list(sorted(set(float(x) for x in df_timepoints)))
    if labels and len(labels) != len(tps):
        raise ValueError(f"--timepoint-labels has {len(labels)} labels but df has {len(tps)} timepoints: {tps}")
    if not labels:
        # Notebook convention for the 4 MOSTA stages used in velocity plotting.
        default = ["E12.5", "E13.5", "E14.5", "E15.5"]
        labels = default[: len(tps)]
    return [TimepointSpec(idx=t, label=labels[i]) for i, t in enumerate(tps)]


def _run(args: argparse.Namespace) -> None:
    _ensure_project_root_on_path()
    os.environ.setdefault("MPLBACKEND", "Agg")

    try:
        from evaluation.arista_code import arista_helpers as helpers
        from evaluation.arista_code.mosta_ported.velocity import VelocityAnalyzer, ensure_valid_palette, plot_single_velocity_field
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing required python packages for DeepRUOT/velocity plotting.\n"
            "Make sure you are running in the project env (torch, torch_geometric, scanpy, scvelo, etc.).\n"
            f"Original import error: {exc}"
        ) from exc

    df = _load_df(args.data_csv)
    df = _subsample_df_by_timepoint(
        df,
        max_cells_per_timepoint=args.max_cells_per_timepoint,
        random_state=args.random_state,
        stratify_col=args.sample_stratify_col,
    )
    timepoint_specs = _parse_timepoints(args.timepoint_labels, df["samples"].unique())

    os.makedirs(args.output_dir, exist_ok=True)

    # Palettes (global, stable across timepoints)
    palette_cache = _load_palette_cache(args.palette_cache) if args.palette_cache else None

    annotation_palette = None
    if "Annotation" in df.columns:
        cats = sorted(df["Annotation"].astype(str).unique())
        annotation_palette = _build_palette(cats)

        h5ad_anno = None
        if palette_cache and isinstance(palette_cache.get("Annotation"), dict):
            h5ad_anno = palette_cache.get("Annotation")
        elif args.color_h5ad:
            h5ad_anno = _load_palette_from_h5ad(
                args.color_h5ad,
                obs_key_candidates=("annotation", "Annotation", "bin_annotation"),
            )

        if isinstance(h5ad_anno, dict) and h5ad_anno:
            # Prefer h5ad colors, but keep coverage for any categories absent from h5ad.
            annotation_palette = _merge_palettes(h5ad_anno, annotation_palette)

        _save_palette_legend(annotation_palette, os.path.join(args.output_dir, "annotation_legend.pdf"))

    celltype_palette = None
    if "celltype" in df.columns:
        cats = sorted(df["celltype"].astype(str).unique())
        celltype_palette = _build_palette(cats)

        h5ad_cell = None
        if palette_cache and isinstance(palette_cache.get("celltype"), dict):
            h5ad_cell = palette_cache.get("celltype")
        elif args.celltype_color_h5ad:
            h5ad_cell = _load_palette_from_h5ad(
                args.celltype_color_h5ad,
                obs_key_candidates=("celltype", "Celltype", "cell_type", "CellType"),
            )
        if isinstance(h5ad_cell, dict) and h5ad_cell:
            celltype_palette = _merge_palettes(h5ad_cell, celltype_palette)

        _save_palette_legend(celltype_palette, os.path.join(args.output_dir, "celltype_legend.pdf"))

    if args.palette_cache and (annotation_palette or celltype_palette):
        payload: Dict[str, Dict[str, str]] = {}
        if annotation_palette:
            payload["Annotation"] = annotation_palette
        if celltype_palette:
            payload["celltype"] = celltype_palette
        _save_palette_cache(args.palette_cache, payload)

    _save_palette_legend(TELENCEPHALON_PALETTE, os.path.join(args.output_dir, "telencephalon_legend.pdf"))

    comm = _load_communications(args.communication_pkl) if args.communication_pkl else None

    # Load config + models
    try:
        config = helpers.load_config(args.config)
        f_net, score_net, exp_dir, device = helpers.load_models(
            config,
            device=args.device,
            model_tag=args.model_tag,
            score_tag=args.score_tag,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "A required dependency is missing while loading DeepRUOT config/models.\n"
            "Common missing packages include: torch_geometric (PyG), scanpy/scvelo deps.\n"
            f"Original error: {exc}"
        ) from exc

    # Keep outputs separate from model dir by default.
    print("Project root:", PROJECT_ROOT)
    print("Model exp_dir:", exp_dir)
    print("Using device:", device)
    print("Output dir:", args.output_dir)

    analyzer = VelocityAnalyzer(df, f_net, score_net, dim=52)

    def out_base(name: str) -> str:
        return os.path.join(args.output_dir, name)

    # === Plots mirrored from mosta_velocity.ipynb ===
    spaces = tuple(_parse_csv_list(args.spaces)) if args.spaces else ("gene", "physical")
    if any(s not in ("gene", "physical") for s in spaces):
        raise ValueError(f"--spaces must be subset of gene,physical. Got: {spaces}")

    # (A) Communication overlay on all cells, focus on Brain
    if args.with_communication and comm is not None:
        for ti, spec in enumerate(timepoint_specs):
            if spec.label not in comm:
                print(f"Warning: communication dict missing key '{spec.label}'. Skipping comm overlay for t={spec.idx}.")
                continue
            if "gene" in spaces:
                _, _, figs = analyzer.plot_fingerprint(
                    adata=None,
                    timepoint=spec.idx,
                    timepoint_str=spec.label,
                    space="gene",
                    color="Annotation",
                    mode="default",
                    cell_type="Brain",
                    communication=True,
                    figsize=(16, 20),
                    all_time_communication=comm,
                    save_path=None,
                    label_to_color=annotation_palette,
                    scvelo_default_style=args.scvelo_default_style,
                    density=args.density,
                    interaction_m=args.interaction_m,
                    interaction_threshold=args.interaction_threshold,
                    device=device,
                )
                _save_velocity_figs(
                    figs,
                    base_path_no_ext=out_base(f"velocity_communication_gene_brain_t{ti}_combined"),
                    bg_cell=None,
                    focus_cell="Brain",
                    zoom=False,
                    prefer_format=args.save_format,
                )
            if "physical" in spaces and spec.idx == timepoint_specs[-1].idx:
                _, _, figs = analyzer.plot_fingerprint(
                    adata=None,
                    timepoint=spec.idx,
                    timepoint_str=spec.label,
                    space="physical",
                    color="Annotation",
                    mode="default",
                    cell_type="Brain",
                    communication=True,
                    figsize=(16, 20),
                    all_time_communication=comm,
                    save_path=None,
                    label_to_color=annotation_palette,
                    scvelo_default_style=args.scvelo_default_style,
                    density=args.density,
                    interaction_m=args.interaction_m,
                    interaction_threshold=args.interaction_threshold,
                    device=device,
                )
                _save_velocity_figs(
                    figs,
                    base_path_no_ext=out_base(f"velocity_communication_physical_brain_t{ti}_combined"),
                    bg_cell=None,
                    focus_cell="Brain",
                    zoom=False,
                    prefer_format=args.save_format,
                )

    # (B) Brain-only velocity plots (no communication edges)
    if args.with_brain_only:
        first = timepoint_specs[0]
        first_i = 0
        if "physical" in spaces:
            _, _, figs = analyzer.plot_fingerprint(
                adata=None,
                timepoint=first.idx,
                timepoint_str=first.label,
                space="physical",
                color="celltype",
                mode="default",
                cell_type="celltype",
                background_cell_type="Brain",
                figsize=(17.5, 12.5),
                communication=False,
                all_time_communication=comm,
                save_path=None,
                label_to_color=celltype_palette,
                scvelo_default_style=args.scvelo_default_style,
                density=max(args.density, 4.0),
                interaction_m=args.interaction_m,
                interaction_threshold=args.interaction_threshold,
                device=device,
            )
            _save_velocity_figs(
                figs,
                base_path_no_ext=out_base(f"velocity_physical_brain_t{first_i}_combined"),
                bg_cell="Brain",
                focus_cell=None,
                zoom=False,
                prefer_format=args.save_format,
            )

        last = timepoint_specs[-1]
        last_i = len(timepoint_specs) - 1
        if "gene" in spaces:
            _, _, figs = analyzer.plot_fingerprint(
                adata=None,
                timepoint=last.idx,
                timepoint_str=last.label,
                space="gene",
                color="celltype",
                mode="default",
                cell_type="celltype",
                background_cell_type="Brain",
                figsize=(17.5, 12.5),
                communication=False,
                all_time_communication=comm,
                save_path=None,
                label_to_color=celltype_palette,
                scvelo_default_style=args.scvelo_default_style,
                density=args.density,
                interaction_m=args.interaction_m,
                interaction_threshold=args.interaction_threshold,
                device=device,
            )
            _save_velocity_figs(
                figs,
                base_path_no_ext=out_base(f"velocity_gene_brain_t{last_i}_combined"),
                bg_cell="Brain",
                focus_cell=None,
                zoom=False,
                prefer_format=args.save_format,
            )

    # (C) Telencephalon zoom plots (brain-only, no communication)
    if args.with_telencephalon:
        import numpy as np
        import anndata as ad
        import scanpy as sc
        import scvelo as scv

        last = timepoint_specs[-1]
        last_i = len(timepoint_specs) - 1
        zoom_region = (-1.3, -0.5, 3.3, 4.2)
        tel_density = max(0.5, args.density / 2)
        tel_figsize = (7, 5)

        df_t = df[df["samples"] == float(last.idx)].copy()
        if "Annotation" in df_t.columns:
            df_t = df_t[df_t["Annotation"] == "Brain"].copy()
        if df_t.empty:
            print("Warning: Telencephalon plot skipped because Brain subset is empty at timepoint:", last.idx)
        else:
            all_data = df_t[[f"x{i}" for i in range(1, 53)]].values.astype(np.float32)
            coords = all_data[:, :2]
            X_expression = all_data[:, 2:]

            # Compute all velocity components once (drift, interaction, score, full)
            vel = helpers.compute_velocity_components(
                all_data,
                float(last.idx),
                f_net,
                score_net,
                interaction_m=args.interaction_m,
                interaction_threshold=args.interaction_threshold,
                device=device,
            )

            def _project_gene_velocity(V_high_dim: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
                tmp_ad = ad.AnnData(X=X_expression)
                tmp_ad.layers["Ms"] = X_expression.copy()
                tmp_ad.obsm["X_spatial"] = coords.copy()
                sc.pp.neighbors(tmp_ad, n_neighbors=n_neighbors, use_rep="X")
                tmp_ad.layers["velocity"] = V_high_dim.copy()
                scv.tl.velocity_graph(tmp_ad, vkey="velocity", xkey="Ms", n_jobs=-1)
                scv.tl.velocity_embedding(tmp_ad, basis="spatial", vkey="velocity")
                return tmp_ad.obsm["velocity_spatial"].copy()

            def _plot_tel_triplet(space: str) -> Tuple[object, object, object]:
                if space == "physical":
                    V_intr = vel["drift"][:, :2]
                    V_inter = vel["interaction"][:, :2]
                    V_full = vel["full"][:, :2]
                    space_title = "Physical Space"
                elif space == "gene":
                    print("  Telencephalon gene space: Projecting (intr/inter/full) to 2D using scVelo...")
                    V_intr = _project_gene_velocity(vel["drift"][:, 2:], n_neighbors=30)
                    V_inter = _project_gene_velocity(vel["interaction"][:, 2:], n_neighbors=30)
                    V_full = _project_gene_velocity(vel["full"][:, 2:], n_neighbors=30)
                    space_title = "Gene Space"
                else:
                    raise ValueError(f"Unknown space: {space}")

                adata_tel = ad.AnnData(X=all_data)
                adata_tel.obsm["X_spatial"] = coords
                adata_tel.obsm["velocity_intrinsic_spatial"] = V_intr
                adata_tel.obsm["velocity_interaction_spatial"] = V_inter
                adata_tel.obsm["velocity_full_spatial"] = V_full

                if "telencephalon" in df_t.columns:
                    tel = df_t["telencephalon"].astype(str).values
                    allowed = set(TELENCEPHALON_PALETTE.keys())
                    tel = np.array([x if x in allowed else "Other" for x in tel], dtype=object)
                    adata_tel.obs["telencephalon"] = tel
                else:
                    adata_tel.obs["telencephalon"] = "Other"
                adata_tel.obs["telencephalon"] = adata_tel.obs["telencephalon"].astype("category")

                pal = ensure_valid_palette(adata_tel, "telencephalon", TELENCEPHALON_PALETTE)

                common = dict(
                    density=tel_density,
                    figsize=tel_figsize,
                    flip_y=False,
                    flip_x=False,
                    mode="default",
                    remove_outliers=True,
                    timepoint_str=last.label,
                    plot_region=zoom_region,
                    palette=pal,
                    scvelo_default_style=args.scvelo_default_style,
                    alpha=0.4,
                    color_key="telencephalon",
                )

                fig_i, _ = plot_single_velocity_field(
                    adata_tel,
                    "velocity_intrinsic",
                    title=f"{space_title} - Intrinsic Velocity",
                    **common,
                )
                fig_x, _ = plot_single_velocity_field(
                    adata_tel,
                    "velocity_interaction",
                    title=f"{space_title} - Interaction Velocity",
                    **common,
                )

                if args.skip_telencephalon_full:
                    return fig_i, fig_x, None

                fig_f, _ = plot_single_velocity_field(
                    adata_tel,
                    "velocity_full",
                    title=f"{space_title} - Full Velocity",
                    **common,
                )
                return fig_i, fig_x, fig_f

        if "physical" in spaces:
            fig_i, fig_x, fig_f = _plot_tel_triplet("physical")
            _save_component_fig(
                fig_i,
                base_path_no_ext=out_base(f"velocity_physical_telencephalon_t{last_i}_combined"),
                bg_cell="Brain",
                zoom=True,
                focus_cell=None,
                prefer_format=args.save_format,
                suffix="intrinsic",
            )
            _save_component_fig(
                fig_x,
                base_path_no_ext=out_base(f"velocity_physical_telencephalon_t{last_i}_combined"),
                bg_cell="Brain",
                zoom=True,
                focus_cell=None,
                prefer_format=args.save_format,
                suffix="interaction",
            )
            if fig_f is not None:
                _save_component_fig(
                    fig_f,
                    base_path_no_ext=out_base(f"velocity_physical_telencephalon_t{last_i}_combined"),
                    bg_cell="Brain",
                    zoom=True,
                    focus_cell=None,
                    prefer_format=args.save_format,
                    suffix="full",
                )

        if "gene" in spaces:
            fig_i, fig_x, fig_f = _plot_tel_triplet("gene")
            _save_component_fig(
                fig_i,
                base_path_no_ext=out_base(f"velocity_gene_telencephalon_t{last_i}_combined"),
                bg_cell="Brain",
                zoom=True,
                focus_cell=None,
                prefer_format=args.save_format,
                suffix="intrinsic",
            )
            _save_component_fig(
                fig_x,
                base_path_no_ext=out_base(f"velocity_gene_telencephalon_t{last_i}_combined"),
                bg_cell="Brain",
                zoom=True,
                focus_cell=None,
                prefer_format=args.save_format,
                suffix="interaction",
            )
            if fig_f is not None:
                _save_component_fig(
                    fig_f,
                    base_path_no_ext=out_base(f"velocity_gene_telencephalon_t{last_i}_combined"),
                    bg_cell="Brain",
                    zoom=True,
                    focus_cell=None,
                    prefer_format=args.save_format,
                    suffix="full",
                )

    print("Done.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MOSTA velocity local plotting script")
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "config", "mosta_config.yaml"),
        help="Path to mosta_config.yaml",
    )
    parser.add_argument(
        "--data-csv",
        default=os.path.join(PROJECT_ROOT, "evaluation", "mosta", "data", "mosta_four_time_with_celltype_refined.csv"),
        help="MOSTA 4-timepoint CSV with Annotation/celltype/telencephalon columns",
    )
    parser.add_argument(
        "--communication-pkl",
        default=os.path.join(PROJECT_ROOT, "evaluation", "mosta", "data", "all_timepoint_communications_merged.pkl"),
        help="Pickle containing per-timepoint communication matrices keyed by stage string (e.g. E12.5)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "results", "mosta_velocity_local"),
        help="Directory to write plots",
    )
    parser.add_argument(
        "--color-h5ad",
        default=os.path.join(PROJECT_ROOT, "spatial_data", "Mouse_embryo_all_stage.h5ad"),
        help="h5ad used to source the Annotation colormap (reads *_colors from .uns, uses backed='r')",
    )
    parser.add_argument(
        "--celltype-color-h5ad",
        default=None,
        help="Optional h5ad used to source the celltype colormap (if it has celltype + celltype_colors in .uns)",
    )
    parser.add_argument(
        "--palette-cache",
        default=None,
        help="Optional JSON cache for palettes (avoids re-reading large h5ad every run)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device override (e.g. cpu or cuda). Default: auto-detect",
    )
    parser.add_argument(
        "--model-tag",
        default="model_final",
        help="Model checkpoint filename under the experiment directory (default: model_final)",
    )
    parser.add_argument(
        "--score-tag",
        default="score_model",
        help="Score model checkpoint filename under the experiment directory (default: score_model)",
    )
    parser.add_argument(
        "--spaces",
        default="gene,physical",
        help="Comma-separated spaces to render: gene,physical",
    )
    parser.add_argument(
        "--timepoint-labels",
        default="E12.5,E13.5,E14.5,E15.5",
        help="Comma-separated stage labels aligned to sorted df['samples'] timepoints",
    )
    parser.add_argument("--density", type=float, default=2.0, help="Stream density (scvelo)")
    parser.add_argument("--interaction-m", type=int, default=1024, help="Interaction sampling m")
    parser.add_argument("--interaction-threshold", type=int, default=1000, help="Interaction threshold")
    parser.add_argument(
        "--save-format",
        default="pdf",
        choices=("pdf", "png"),
        help="Preferred output format; may fall back to PNG on PDF backend errors",
    )
    parser.add_argument(
        "--max-cells-per-timepoint",
        type=int,
        default=None,
        help="Optional: downsample each timepoint to at most N rows (for quick local tests)",
    )
    parser.add_argument(
        "--sample-stratify-col",
        default="Annotation",
        help="Column to stratify sampling by when downsampling (default: Annotation; set '' to disable)",
    )
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for downsampling")
    parser.add_argument("--with-communication", action="store_true", help="Render communication overlays (focus Brain)")
    parser.add_argument("--with-brain-only", action="store_true", help="Render brain-only velocity plots")
    parser.add_argument("--with-telencephalon", action="store_true", help="Render telencephalon zoom plots")
    parser.add_argument(
        "--scvelo-default-style",
        action="store_true",
        help="Use scVelo default stream plot style (fewer custom parameters).",
    )
    parser.add_argument(
        "--skip-telencephalon-full",
        action="store_true",
        help="Do not render full velocity for telencephalon (saves time; default renders full).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.sample_stratify_col == "":
        args.sample_stratify_col = None

    # Default behavior matches the notebook: do all three groups.
    if not (args.with_communication or args.with_brain_only or args.with_telencephalon):
        args.with_communication = True
        args.with_brain_only = True
        args.with_telencephalon = True

    _run(args)


if __name__ == "__main__":
    main()
