"""Reusable high-level downstream workflows."""

from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
import time
from typing import Any, Callable, Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .attention import analyze_attention_by_celltype, save_interpolated_attention
from .classification import load_cached_mlp_classifier, predict_labels_for_trajectories
from .pipeline_utils import downsample_xy, find_single_classifier_cache, select_evenly_spaced
from .simulation import (
    apply_spatial_warp_to_segments,
    sample_observed_x0,
    simulate_piecewise_spatially_warped_split,
    simulate_sde_points,
    simulate_sde_points_split,
    simulate_sde_points_split_from_x0,
)

if TYPE_CHECKING:
    import anndata as ad
    from .runtime import DynamicalRuntime

__all__ = [
    "InterpolationResult",
    "compute_timepoint_communications",
    "plot_lineage_sankey",
    "plot_spatiotemporal_3d",
    "run_interpolation_workflow",
]


@dataclass(frozen=True)
class InterpolationResult:
    adata_dict: dict[str, "ad.AnnData"]
    ts_points: list[float]
    time_keys: list[str]
    observed_time_points: list[float]
    interp_points: list[float]
    plot_3d_ts_points: list[float]
    plot_3d_time_keys: list[str]
    predicted_labels_list: Optional[Sequence[np.ndarray]]
    predicted_labels_split: Optional[Sequence[np.ndarray]]
    predicted_labels_split_prewarp: Optional[Sequence[np.ndarray]]
    sde_points_split: Optional[np.ndarray]
    sde_points_split_prewarp: Optional[np.ndarray]
    piecewise_x0_by_observed: Optional[Dict[float, np.ndarray]]
    piecewise_labels_by_observed: Optional[Dict[float, np.ndarray]]
    piecewise_endpoint_by_observed: Optional[Dict[float, np.ndarray]]
    classifier_model: Optional[object]
    label_encoder: Optional[object]
    classifier_feature_dim: Optional[int]


def run_interpolation_workflow(
    *,
    df,
    dim: int,
    annotation_key: str,
    runtime: "DynamicalRuntime",
    device: str,
    output_dir: str,
    requested_plot_points: Optional[Sequence[float]] = None,
    interp_time_points: Sequence[float] = (),
    no_interp: bool = False,
    target_total_slices: Optional[int] = None,
    max_observed_timepoints: Optional[int] = None,
    use_real_for_observed: bool = True,
    classifier_cache_path: Optional[str] = None,
    classifier_cache_dir: Optional[str] = None,
    classifier_best_metric: str = "accuracy",
    classifier_n_pcs: Optional[int] = None,
    classifier_knn_neighbors: int = 10,
    sde_n_samples: Optional[int] = None,
    skip_nonsplit_sde: bool = False,
    sde_dt: float = 0.05,
    split_sde_dt: float = 0.05,
    split_sigma_scalar: float = 0.03,
    split_sigma_vector: Optional[Sequence[float]] = None,
    split_growth_alpha: float = 1.0,
    split_sde_piecewise: bool = False,
    split_sde_piecewise_include_end: bool = False,
    piecewise_observed_sample_mode: str = "t0_fixed",
    spatial_warp_to_observed: bool = False,
    spatial_warp_to_observed_piecewise: bool = False,
    spatial_warp_k: int = 8,
    spatial_warp_eps: float = 1e-6,
    slice_max_cells_per_timepoint: Optional[int] = None,
    random_seed: Optional[int] = 42,
) -> InterpolationResult:
    import anndata as ad

    f_net = runtime.f_net
    score_net = runtime.score_net

    observed_time_points = sorted(df["samples"].unique().tolist())
    required_obs_points = set(requested_plot_points) if requested_plot_points is not None else set()

    if max_observed_timepoints is not None:
        max_obs = int(max_observed_timepoints)
        if max_obs <= 0:
            raise ValueError("--max-observed-timepoints must be > 0")
        if max_obs < len(observed_time_points):
            required_obs = [t for t in observed_time_points if t in required_obs_points]
            if len(required_obs) > max_obs:
                raise ValueError(
                    f"--max-observed-timepoints={max_obs} is smaller than required observed plot points "
                    f"{sorted(required_obs)} from requested_plot_points={requested_plot_points}"
                )
            remaining = [t for t in observed_time_points if t not in set(required_obs)]
            keep_extra = max_obs - len(required_obs)
            extra = select_evenly_spaced(remaining, keep_extra) if keep_extra > 0 else []
            observed_time_points = sorted(set(required_obs + extra))
            print("Capped observed timepoints:", observed_time_points)

    interp_points = [] if no_interp else [float(t) for t in interp_time_points]
    interp_points = [float(t) for t in interp_points if float(t) not in observed_time_points]

    if split_sde_piecewise and len(interp_points) == 0:
        print("[warn] split_sde_piecewise has no effect without interpolation points; disabling it.")
        split_sde_piecewise = False
    if spatial_warp_to_observed_piecewise and len(interp_points) == 0:
        print("[warn] spatial_warp_to_observed_piecewise has no effect without interpolation points; disabling it.")
        spatial_warp_to_observed_piecewise = False
    if target_total_slices is not None and split_sde_piecewise:
        print("[warn] target_total_slices is ignored when split_sde_piecewise is enabled.")
    if target_total_slices is not None and spatial_warp_to_observed_piecewise:
        print("[warn] target_total_slices is ignored when spatial_warp_to_observed_piecewise is enabled.")
    if target_total_slices is not None and (not split_sde_piecewise) and (not spatial_warp_to_observed_piecewise):
        keep_observed = max(1, int(target_total_slices) - len(interp_points))
        if keep_observed < len(observed_time_points):
            observed_time_points = select_evenly_spaced(observed_time_points, keep_observed)
            print("Selected observed timepoints:", observed_time_points)

    ts_points = sorted(set(observed_time_points + interp_points))
    time_keys = [str(t) for t in ts_points]

    if requested_plot_points is None:
        plot_3d_ts_points = list(ts_points)
    else:
        ts_set = {float(t) for t in ts_points}
        missing = [float(t) for t in requested_plot_points if float(t) not in ts_set]
        if missing:
            raise ValueError(f"requested_plot_points contains values not in ts_points: {missing} (ts_points={ts_points})")
        plot_3d_ts_points = [float(t) for t in requested_plot_points]
    plot_3d_time_keys = [str(t) for t in plot_3d_ts_points]

    need_interp = len(interp_points) > 0
    piecewise_x0_by_observed: Optional[Dict[float, np.ndarray]] = None
    piecewise_labels_by_observed: Optional[Dict[float, np.ndarray]] = None
    piecewise_endpoint_by_observed: Optional[Dict[float, np.ndarray]] = None
    predicted_labels_list = None
    predicted_labels_split = None
    predicted_labels_split_prewarp = None
    sde_points_split = None
    sde_points_split_prewarp = None
    classifier_model = None
    label_encoder = None
    classifier_feature_dim = None

    feature_cols_full = [f"x{i}" for i in range(1, dim + 1)]
    if need_interp:
        print(
            "Interpolation enabled | interp_points=",
            interp_points,
            "| observed_time_points=",
            observed_time_points,
        )
        t_train0 = time.perf_counter()
        classifier_cache_resolved = find_single_classifier_cache(
            explicit_path=classifier_cache_path,
            cache_dir=classifier_cache_dir,
            output_dir=output_dir,
        )
        cached_classifier = load_cached_mlp_classifier(classifier_cache_resolved, device=device)
        classifier_model = cached_classifier.model
        label_encoder = cached_classifier.label_encoder
        classifier_feature_dim = int(cached_classifier.feature_dim)
        _ = (
            cached_classifier.balanced_accuracy
            if classifier_best_metric == "bacc"
            else cached_classifier.accuracy
        )
        print(
            f"Loaded classifier cache in {time.perf_counter() - t_train0:.1f}s | "
            f"path={classifier_cache_resolved} | "
            f"acc={cached_classifier.accuracy} | "
            f"bacc={cached_classifier.balanced_accuracy}"
        )

        t0 = float(min(observed_time_points))
        n_samples = int((df["samples"] == t0).sum())
        if sde_n_samples is not None:
            if sde_n_samples <= 0:
                raise ValueError("--sde-n-samples must be > 0")
            n_samples = min(n_samples, int(sde_n_samples))
        print("SDE n_samples (from t0):", n_samples)
        if split_sde_piecewise:
            print("Piecewise observed-start sampling mode:", piecewise_observed_sample_mode)

        sde_points = None
        if skip_nonsplit_sde:
            print("Skipping non-split SDE.")
        else:
            t_sde0 = time.perf_counter()
            print("Simulating non-split SDE...")
            sde_points, _ = simulate_sde_points(
                df=df,
                dim=dim,
                f_net=f_net,
                score_net=score_net,
                time_index=0,
                n_samples=n_samples,
                ts_points=ts_points,
                dt=sde_dt,
                sigma=0.0,
                include_score=False,
                device=device,
            )
            print(f"Non-split SDE done in {time.perf_counter() - t_sde0:.1f}s")

        t_sde_split0 = time.perf_counter()
        print("Simulating split SDE...")
        if spatial_warp_to_observed_piecewise:
            if spatial_warp_k <= 0:
                raise ValueError("--spatial-warp-k must be > 0")
            if spatial_warp_eps <= 0:
                raise ValueError("--spatial-warp-eps must be > 0")
            rng_warp_piecewise = np.random.default_rng(
                1 if random_seed is None else int(random_seed) + 1
            )
            x0_warp, _ = sample_observed_x0(
                df,
                time_value=t0,
                feature_cols=feature_cols_full,
                label_col=annotation_key,
                n_samples_cap=n_samples,
                rng=rng_warp_piecewise,
            )
            sde_points_split = simulate_piecewise_spatially_warped_split(
                x0=x0_warp,
                f_net=f_net,
                score_net=score_net,
                observed_time_points=observed_time_points,
                ts_points=ts_points,
                df=df,
                feature_cols_full=feature_cols_full,
                label_col=annotation_key,
                dt=split_sde_dt,
                sigma=split_sigma_scalar,
                sigma_by_dim=split_sigma_vector,
                growth_alpha=split_growth_alpha,
                interaction_m=1024,
                device=device,
                rng=rng_warp_piecewise,
                k=int(spatial_warp_k),
                eps=float(spatial_warp_eps),
            )
        elif split_sde_piecewise:
            rng_piecewise = np.random.default_rng(0 if random_seed is None else int(random_seed))
            x0_by_observed: Dict[float, np.ndarray] = {}
            labels0_by_observed: Dict[float, np.ndarray] = {}
            if piecewise_observed_sample_mode == "per_timepoint":
                piecewise_cap = None if sde_n_samples is None else int(sde_n_samples)
            else:
                piecewise_cap = int(n_samples)
            for t_obs in observed_time_points:
                x0_t, labels_t = sample_observed_x0(
                    df,
                    time_value=float(t_obs),
                    feature_cols=feature_cols_full,
                    label_col=annotation_key,
                    n_samples_cap=piecewise_cap,
                    rng=rng_piecewise,
                )
                x0_by_observed[float(t_obs)] = x0_t
                labels0_by_observed[float(t_obs)] = labels_t
            piecewise_x0_by_observed = x0_by_observed
            piecewise_labels_by_observed = labels0_by_observed
            piecewise_endpoint_by_observed = {}

            points_by_time: Dict[float, np.ndarray] = {}
            for t_obs in observed_time_points:
                points_by_time[float(t_obs)] = x0_by_observed[float(t_obs)]

            observed_set = {float(t) for t in observed_time_points}
            for t_start, t_end in zip(observed_time_points[:-1], observed_time_points[1:]):
                mids = sorted([t for t in interp_points if float(t_start) < float(t) < float(t_end)])
                if (not mids) and (not split_sde_piecewise_include_end):
                    continue
                seg_ts: list[float] = [float(t_start)] + [float(t) for t in mids]
                if split_sde_piecewise_include_end:
                    seg_ts.append(float(t_end))
                print(f"[piecewise split-SDE] segment {t_start}->{t_end} | targets={seg_ts}")
                seg_points = simulate_sde_points_split_from_x0(
                    x0=x0_by_observed[float(t_start)],
                    f_net=f_net,
                    score_net=score_net,
                    ts_points=seg_ts,
                    dt=split_sde_dt,
                    sigma=split_sigma_scalar,
                    sigma_by_dim=split_sigma_vector,
                    growth_alpha=split_growth_alpha,
                    interaction_m=1024,
                    device=device,
                    verbose=True,
                )
                for t_val, pts in zip(seg_ts, seg_points):
                    if float(t_val) in observed_set:
                        if split_sde_piecewise_include_end and float(t_val) == float(t_end):
                            piecewise_endpoint_by_observed[float(t_end)] = np.asarray(pts, dtype=np.float32)
                        continue
                    points_by_time[float(t_val)] = np.asarray(pts, dtype=np.float32)

            missing = [float(t) for t in ts_points if float(t) not in points_by_time]
            if missing:
                raise ValueError(f"Piecewise split-SDE missing timepoints: {missing}")
            sde_points_split = np.array([points_by_time[float(t)] for t in ts_points], dtype=object)
        else:
            # Keep legacy parity here: the original MOSTA workflow sampled x0
            # inside simulate_sde_points_split(...) via torch.randperm, not via
            # a separate NumPy-based sample_observed_x0(...) pre-pass.
            sde_points_split = simulate_sde_points_split(
                df=df,
                dim=dim,
                f_net=f_net,
                score_net=score_net,
                time_index=0,
                n_samples=n_samples,
                ts_points=ts_points,
                dt=split_sde_dt,
                sigma=split_sigma_scalar,
                sigma_by_dim=split_sigma_vector,
                growth_alpha=split_growth_alpha,
                device=device,
                verbose=True,
            )

        if spatial_warp_to_observed:
            sde_points_split_prewarp = np.array(
                [np.asarray(p, dtype=np.float32).copy() for p in sde_points_split],
                dtype=object,
            )
            if spatial_warp_k <= 0:
                raise ValueError("--spatial-warp-k must be > 0")
            if spatial_warp_eps <= 0:
                raise ValueError("--spatial-warp-eps must be > 0")
            rng_warp = np.random.default_rng(
                1 if random_seed is None else int(random_seed) + 1
            )
            sde_points_split = apply_spatial_warp_to_segments(
                sde_points_split=sde_points_split,
                ts_points=ts_points,
                observed_time_points=observed_time_points,
                df=df,
                feature_cols_full=feature_cols_full,
                label_col=annotation_key,
                rng=rng_warp,
                piecewise=bool(split_sde_piecewise),
                piecewise_include_end=bool(split_sde_piecewise_include_end),
                piecewise_endpoint_by_observed=piecewise_endpoint_by_observed,
                use_real_for_observed=bool(use_real_for_observed),
                k=int(spatial_warp_k),
                eps=float(spatial_warp_eps),
            )
        print(f"Split SDE done in {time.perf_counter() - t_sde_split0:.1f}s")

        if sde_points is not None:
            predicted_labels_list = predict_labels_for_trajectories(
                sde_points=sde_points,
                ts_points=ts_points,
                model=classifier_model,
                label_encoder=label_encoder,
                feature_dim=classifier_feature_dim,
                device=device,
                knn_neighbors=int(classifier_knn_neighbors),
            )

        predicted_labels_split = predict_labels_for_trajectories(
            sde_points=sde_points_split,
            ts_points=ts_points,
            model=classifier_model,
            label_encoder=label_encoder,
            feature_dim=classifier_feature_dim,
            device=device,
            knn_neighbors=int(classifier_knn_neighbors),
        )
        if sde_points_split_prewarp is not None:
            predicted_labels_split_prewarp = predict_labels_for_trajectories(
                sde_points=sde_points_split_prewarp,
                ts_points=ts_points,
                model=classifier_model,
                label_encoder=label_encoder,
                feature_dim=classifier_feature_dim,
                device=device,
                knn_neighbors=int(classifier_knn_neighbors),
            )

    adata_dict = {}
    rng = np.random.default_rng(0 if random_seed is None else int(random_seed))
    for t in ts_points:
        key = str(t)
        if use_real_for_observed and (t in observed_time_points):
            subset = df[df["samples"] == float(t)]
            X = subset[feature_cols_full].values.astype(np.float32)
            labels = subset[annotation_key].astype(str).values
        elif (not use_real_for_observed) and split_sde_piecewise and (t in observed_time_points):
            if sde_points_split is None or predicted_labels_split is None:
                raise ValueError("Piecewise split-SDE observed slice requested but split outputs are missing.")
            if spatial_warp_to_observed and float(t) != float(min(observed_time_points)):
                idx = ts_points.index(t)
                X = np.array(sde_points_split[idx], dtype=np.float32)
                labels = np.asarray(predicted_labels_split[idx]).astype(str)
            else:
                if piecewise_x0_by_observed is None or piecewise_labels_by_observed is None:
                    raise ValueError("Piecewise split-SDE enabled but observed x0/labels cache is missing.")
                X = np.asarray(piecewise_x0_by_observed[float(t)], dtype=np.float32)
                labels = np.asarray(piecewise_labels_by_observed[float(t)]).astype(str)
        else:
            if sde_points_split is None or predicted_labels_split is None:
                raise ValueError("Interpolation requested but split SDE outputs are missing.")
            idx = ts_points.index(t)
            X = np.array(sde_points_split[idx], dtype=np.float32)
            labels = np.asarray(predicted_labels_split[idx]).astype(str)

        X, labels = downsample_xy(X, labels, slice_max_cells_per_timepoint, rng)
        adata_t = ad.AnnData(X=X)
        adata_t.obs[annotation_key] = labels
        adata_t.obsm["spatial"] = X[:, :2]
        adata_dict[key] = adata_t

    return InterpolationResult(
        adata_dict=adata_dict,
        ts_points=list(ts_points),
        time_keys=time_keys,
        observed_time_points=list(observed_time_points),
        interp_points=list(interp_points),
        plot_3d_ts_points=list(plot_3d_ts_points),
        plot_3d_time_keys=plot_3d_time_keys,
        predicted_labels_list=predicted_labels_list,
        predicted_labels_split=predicted_labels_split,
        predicted_labels_split_prewarp=predicted_labels_split_prewarp,
        sde_points_split=sde_points_split,
        sde_points_split_prewarp=sde_points_split_prewarp,
        piecewise_x0_by_observed=piecewise_x0_by_observed,
        piecewise_labels_by_observed=piecewise_labels_by_observed,
        piecewise_endpoint_by_observed=piecewise_endpoint_by_observed,
        classifier_model=classifier_model,
        label_encoder=label_encoder,
        classifier_feature_dim=classifier_feature_dim,
    )


def compute_timepoint_communications(
    *,
    adata_dict,
    time_points: Sequence[float],
    annotation_key: str,
    f_net,
    device: str,
    out_dir: str,
    save_dense_attention_matrix: bool = False,
    remove_self_loop: bool = False,
    winsor_quantile: float = 0.995,
    save_pickle_path: Optional[str] = None,
) -> dict[str, dict]:
    os.makedirs(out_dir, exist_ok=True)
    all_time_communications = {}
    for t in time_points:
        key = str(t)
        adata_t = adata_dict[key]
        print("Time", key, "cells", adata_t.n_obs)

        attn_out = save_interpolated_attention(
            adata_t,
            time_value=float(t),
            f_net=f_net,
            device=device,
            out_dir=out_dir,
            save_dense_matrix=bool(save_dense_attention_matrix),
        )

        comm = analyze_attention_by_celltype(
            edge_index=attn_out["edge_index"],
            attn=attn_out["attn_mean"],
            labels=adata_t.obs[annotation_key].values,
            spatial_coord=adata_t.obsm["spatial"],
            time_title=key,
            remove_self_loop=remove_self_loop,
            winsor_quantile=winsor_quantile,
            distance_bins=None,
            n_permutations=0,
            plot=False,
        )
        all_time_communications[key] = comm

    if save_pickle_path is not None:
        with open(save_pickle_path, "wb") as handle:
            pickle.dump(all_time_communications, handle)
        print("Saved:", save_pickle_path)

    return all_time_communications


def plot_lineage_sankey(
    *,
    plot_fn: Callable[..., Any],
    predicted_labels_list: Sequence[np.ndarray],
    time_keys: Sequence[str],
    label_to_color: dict[str, str],
    out_html: str,
    min_flow: Optional[float] = None,
    keep_source_cumfrac: Optional[float] = None,
    normalize_mode: Optional[str] = None,
    style: str = "nature-methods",
    title: str = "Cell Fate Transitions",
    show_time_axis: bool = True,
):
    fig = plot_fn(
        predicted_labels_list=predicted_labels_list,
        out_html=out_html,
        time_keys=time_keys,
        show_time_axis=show_time_axis,
        min_flow=min_flow,
        keep_source_cumfrac=keep_source_cumfrac,
        normalize_mode=normalize_mode,
        label_to_color=label_to_color,
        style=style,
        title=title,
    )
    print("Saved:", out_html)
    return fig


def plot_spatiotemporal_3d(
    *,
    plot_fn: Callable[..., Any],
    adata_dict,
    all_time_communications,
    time_keys: Sequence[str],
    plot_time_points: Sequence[float],
    ts_points: Sequence[float],
    observed_time_points: Sequence[float],
    interp_points: Sequence[float],
    annotation_key: str,
    label_to_color: dict[str, str],
    out_html: str,
    predicted_labels_list: Optional[Sequence[np.ndarray]] = None,
    **plot_kwargs,
):
    adata_dict_3d = {k: adata_dict[k] for k in time_keys}
    comm_3d = {k: all_time_communications[k] for k in time_keys}

    if predicted_labels_list is not None:
        idxs = [ts_points.index(float(t)) for t in plot_time_points]
        predicted_labels_3d = [predicted_labels_list[i] for i in idxs]
    else:
        predicted_labels_3d = [
            np.asarray(adata_dict_3d[k].obs[annotation_key]).astype(str)
            for k in time_keys
        ]

    plot_time_point_set = set(float(t) for t in plot_time_points)
    observed_time_points_3d = [float(t) for t in observed_time_points if float(t) in plot_time_point_set]
    interp_points_3d = [float(t) for t in interp_points if float(t) in plot_time_point_set]

    fig = plot_fn(
        adata_dict=adata_dict_3d,
        all_time_communications=comm_3d,
        time_keys=time_keys,
        label_to_color=label_to_color,
        predicted_labels_list=predicted_labels_3d,
        observed_time_points=observed_time_points_3d,
        generated_time_points=interp_points_3d,
        out_html=out_html,
        annotation_key=annotation_key,
        **plot_kwargs,
    )
    print("Saved:", out_html)
    return fig
