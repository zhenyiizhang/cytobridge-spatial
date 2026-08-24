#!/usr/bin/env python3
"""Run shared interpolation, lineage, communication, and 3D APIs from AnnData."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _nested(config: dict, section: str, key: str, default=None):
    return config.get(section, {}).get(key, default)


def _resolve(cli_value, config: dict, section: str, key: str, default=None):
    return cli_value if cli_value is not None else _nested(config, section, key, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-agnostic CytoBridge spatiotemporal downstream workflow. "
            "Dataset-specific keys and figure styling live in YAML."
        )
    )
    parser.add_argument("--config", required=True, help="Dataset downstream YAML.")
    parser.add_argument("--aligned-h5ad", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-format", choices=("current", "legacy"), default="current")
    parser.add_argument("--edge-predictor-root", default=None, help="Required only for legacy checkpoints.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--annotation-key", default=None)
    parser.add_argument("--time-key", default=None)
    parser.add_argument("--obsm-key", default=None)
    parser.add_argument("--spatial-key", default=None)
    parser.add_argument("--concat-spatial", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--time-points", default=None, help="Comma-separated observed time values.")
    parser.add_argument("--interp-time-points", default=None, help="Comma-separated generated time values.")
    parser.add_argument(
        "--dense-time-step",
        type=float,
        default=None,
        help=(
            "Generate a dense interpolation grid instead of listing every "
            "--interp-time-points value."
        ),
    )
    parser.add_argument("--dense-time-min", type=float, default=None)
    parser.add_argument("--dense-time-max", type=float, default=None)
    parser.add_argument("--plot-3d-time-points", default=None, help="Comma-separated 3D subset.")
    parser.add_argument("--sde-n-samples", type=int, default=None)
    parser.add_argument("--sde-dt", type=float, default=None)
    parser.add_argument("--split-sde-dt", type=float, default=None)
    parser.add_argument("--split-sigma", type=float, default=None)
    parser.add_argument("--classifier-cache-dir", default=None)
    parser.add_argument("--classifier-cache-path", default=None)
    parser.add_argument("--classifier-epochs", type=int, default=None)
    parser.add_argument("--classifier-hidden-size", type=int, default=None)
    parser.add_argument("--classifier-lr", type=float, default=None)
    parser.add_argument("--classifier-knn-neighbors", type=int, default=None)
    parser.add_argument("--classifier-best-metric", choices=("accuracy", "bacc"), default=None)
    parser.add_argument(
        "--classifier-train-on-full-data",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--classifier-refit-on-full-data-after-selection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Select the epoch on the held-out validation split, then initialize "
            "a fresh classifier and refit it on all rows for exactly that many epochs."
        ),
    )
    parser.add_argument(
        "--classifier-strict-stratification",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fail instead of falling back when a stratified split is impossible.",
    )
    parser.add_argument(
        "--use-real-for-observed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--spatial-warp-to-observed-piecewise",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--spatial-warp-visualization-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use warped XY only for display while classification and "
            "communication consume prewarp dynamical states."
        ),
    )
    parser.add_argument("--spatial-warp-k", type=int, default=None)
    parser.add_argument("--spatial-warp-eps", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--snapshot-time-points",
        default=None,
        help=(
            "Optional comma-separated subset used only for snapshot/mosaic "
            "rendering. The simulation and video still use every requested time."
        ),
    )
    parser.add_argument(
        "--skip-nonsplit-sde",
        action="store_true",
        help=(
            "Skip the fixed-particle non-split trajectory. Useful for a "
            "display-only dense split-SDE mosaic/video run."
        ),
    )
    parser.add_argument("--skip-snapshots", action="store_true")
    parser.add_argument("--skip-communication", action="store_true")
    parser.add_argument("--skip-lineage", action="store_true")
    parser.add_argument("--skip-3d", action="store_true")
    parser.add_argument(
        "--render-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render the split-population trajectory as GIF and/or MP4.",
    )
    parser.add_argument("--video-formats", default="gif,mp4")
    parser.add_argument("--video-fps", type=int, default=4)
    return parser


def _export_plotly_static(fig, output_stem: Path) -> dict[str, str]:
    outputs = {}
    try:
        import plotly.io as pio

        for suffix in ("svg", "pdf", "png"):
            path = output_stem.with_suffix(f".{suffix}")
            pio.write_image(fig, str(path), scale=3 if suffix != "png" else 2)
            outputs[suffix] = str(path)
    except Exception as exc:
        outputs["error"] = str(exc)
    return outputs


def _git_revision() -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": revision, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def main() -> None:
    args = _parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    h5ad_path = Path(args.aligned_h5ad).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import anndata as ad
    import torch

    import CytoBridge as cb

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    annotation_key = _resolve(args.annotation_key, config, "dataset", "annotation_key", "Annotation")
    time_key = _resolve(args.time_key, config, "dataset", "time_key", None)
    obsm_key = _resolve(args.obsm_key, config, "dataset", "obsm_key", "X_latent")
    spatial_key = _resolve(args.spatial_key, config, "dataset", "spatial_key", "spatial_aligned")
    concat_spatial = _resolve(args.concat_spatial, config, "dataset", "concat_spatial", None)

    adata = ad.read_h5ad(h5ad_path)
    if annotation_key not in adata.obs.columns:
        raise KeyError(f"adata.obs is missing annotation column '{annotation_key}'.")
    df, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        annotation_key=annotation_key,
    )
    feature_cols = cb.tl.infer_feature_columns(df, annotation_column=annotation_key)
    dim = len(feature_cols)

    if args.model_format == "legacy":
        loaded = cb.tl.load_legacy_dynamical_model_from_dir(
            model_dir,
            device=device,
            edge_predictor_root=args.edge_predictor_root,
        )
    else:
        loaded = cb.tl.load_dynamical_model_from_dir(model_dir, dim=dim, device=device)
    runtime = cb.tl.build_dynamical_runtime(loaded)

    observed = _parse_floats(args.time_points)
    if observed is None:
        observed = [float(value) for value in _nested(config, "time", "observed", sorted(df["samples"].unique()))]
    interpolated = _parse_floats(args.interp_time_points)
    if args.dense_time_step is not None:
        if interpolated is not None:
            raise ValueError(
                "Use either --dense-time-step or --interp-time-points, not both."
            )
        if float(args.dense_time_step) <= 0.0:
            raise ValueError("--dense-time-step must be > 0.")
        dense_min = (
            float(args.dense_time_min)
            if args.dense_time_min is not None
            else float(min(observed))
        )
        dense_max = (
            float(args.dense_time_max)
            if args.dense_time_max is not None
            else float(max(observed))
        )
        if dense_max <= dense_min:
            raise ValueError("--dense-time-max must be greater than --dense-time-min.")
        dense_values = np.arange(
            dense_min,
            dense_max + float(args.dense_time_step) * 0.5,
            float(args.dense_time_step),
            dtype=np.float64,
        )
        dense_values = np.round(dense_values, decimals=9)
        interpolated = [
            float(value)
            for value in dense_values
            if not any(
                np.isclose(float(value), float(obs), rtol=0.0, atol=1e-9)
                for obs in observed
            )
        ]
    elif interpolated is None:
        interpolated = [float(value) for value in _nested(config, "time", "interpolated", [])]
    plot_3d_points = _parse_floats(args.plot_3d_time_points)
    if plot_3d_points is None:
        plot_3d_points = [float(value) for value in _nested(config, "time", "plot_3d", observed + interpolated)]
    requested_points = sorted(set(observed + interpolated))
    snapshot_points = _parse_floats(args.snapshot_time_points)

    classifier_cache_dir = args.classifier_cache_dir or str(output_dir / "classifier_cache")
    warp_piecewise = bool(
        _resolve(
            args.spatial_warp_to_observed_piecewise,
            config,
            "simulation",
            "spatial_warp_to_observed_piecewise",
            False,
        )
    )
    warp_visualization_only = bool(
        _resolve(
            args.spatial_warp_visualization_only,
            config,
            "simulation",
            "spatial_warp_visualization_only",
            True,
        )
    )
    result = cb.tl.run_interpolation_workflow(
        df=df,
        dim=dim,
        annotation_key=annotation_key,
        runtime=runtime,
        device=device,
        output_dir=str(output_dir),
        requested_plot_points=requested_points,
        interp_time_points=interpolated,
        max_observed_timepoints=len(observed),
        use_real_for_observed=bool(args.use_real_for_observed),
        classifier_cache_path=args.classifier_cache_path,
        classifier_cache_dir=classifier_cache_dir,
        classifier_adata=adata,
        classifier_time_key=resolved_time_key,
        classifier_obsm_key=obsm_key,
        classifier_spatial_key=spatial_key,
        classifier_concat_spatial=concat_spatial,
        classifier_epochs=int(_resolve(args.classifier_epochs, config, "classifier", "epochs", 500)),
        classifier_hidden_size=int(
            _resolve(args.classifier_hidden_size, config, "classifier", "hidden_size", 128)
        ),
        classifier_lr=float(
            _resolve(args.classifier_lr, config, "classifier", "lr", 1e-3)
        ),
        classifier_best_metric=str(
            _resolve(args.classifier_best_metric, config, "classifier", "best_metric", "bacc")
        ),
        classifier_train_on_full_data=bool(
            _resolve(
                args.classifier_train_on_full_data,
                config,
                "classifier",
                "train_on_full_data",
                False,
            )
        ),
        classifier_refit_on_full_data_after_selection=bool(
            _resolve(
                args.classifier_refit_on_full_data_after_selection,
                config,
                "classifier",
                "refit_on_full_data_after_selection",
                False,
            )
        ),
        classifier_strict_stratification=bool(
            _resolve(
                args.classifier_strict_stratification,
                config,
                "classifier",
                "strict_stratification",
                False,
            )
        ),
        classifier_knn_neighbors=int(
            _resolve(args.classifier_knn_neighbors, config, "classifier", "knn_neighbors", 10)
        ),
        sde_n_samples=int(_resolve(args.sde_n_samples, config, "simulation", "n_samples", 5000)),
        skip_nonsplit_sde=bool(args.skip_nonsplit_sde),
        sde_dt=float(_resolve(args.sde_dt, config, "simulation", "sde_dt", 0.05)),
        split_sde_dt=float(_resolve(args.split_sde_dt, config, "simulation", "split_sde_dt", 0.01)),
        split_sigma_scalar=float(_resolve(args.split_sigma, config, "simulation", "split_sigma", 0.03)),
        split_growth_alpha=float(_nested(config, "simulation", "split_growth_alpha", 1.0)),
        spatial_warp_to_observed_piecewise=warp_piecewise,
        spatial_warp_visualization_only=warp_visualization_only,
        spatial_warp_k=int(_resolve(args.spatial_warp_k, config, "simulation", "spatial_warp_k", 8)),
        spatial_warp_eps=float(
            _resolve(args.spatial_warp_eps, config, "simulation", "spatial_warp_eps", 1e-6)
        ),
        random_seed=int(args.random_seed),
    )

    labels = adata.obs[annotation_key].astype(str).to_numpy()
    label_to_color = cb.tl.load_label_to_color(
        labels,
        color_h5ad=str(h5ad_path),
        annotation_key=annotation_key,
    )
    (output_dir / "label_to_color.json").write_text(
        json.dumps(label_to_color, indent=2),
        encoding="utf-8",
    )

    video_outputs = {}
    if args.render_video:
        if result.sde_points_split is None or result.slice_labels_split is None:
            raise RuntimeError("Video rendering requires split-SDE points and labels.")
        trajectory_path = output_dir / "split_population_trajectory.npy"
        labels_path = output_dir / "split_population_labels.npy"
        np.save(trajectory_path, result.sde_points_split, allow_pickle=True)
        np.save(
            labels_path,
            np.asarray(result.slice_labels_split, dtype=object),
            allow_pickle=True,
        )
        video_outputs["trajectory_array"] = str(trajectory_path)
        video_outputs["label_array"] = str(labels_path)
        formats = [
            value.strip().lower()
            for value in str(args.video_formats).split(",")
            if value.strip()
        ]
        invalid_formats = sorted(set(formats) - {"gif", "mp4"})
        if invalid_formats:
            raise ValueError(f"Unsupported video formats: {invalid_formats}")
        for video_format in formats:
            video_path = output_dir / f"spatiotemporal_split_population.{video_format}"
            try:
                cb.pl.plot_trajectory_gif(
                    sde_points=result.sde_points_split,
                    time_values=result.ts_points,
                    labels_list=result.slice_labels_split,
                    label_to_color=label_to_color,
                    out_path=str(video_path),
                    dim_pair=(0, 1),
                    point_size=3.5,
                    alpha=0.75,
                    fps=int(args.video_fps),
                )
                video_outputs[video_format] = str(video_path)
            except (FileNotFoundError, RuntimeError) as exc:
                video_outputs[f"{video_format}_error"] = str(exc)
                print(f"[warn] {video_format} animation export failed: {exc}")

    if not args.skip_snapshots:
        if snapshot_points is None:
            snapshot_time_keys = list(result.time_keys)
        else:
            key_by_time = {float(key): key for key in result.time_keys}
            missing_snapshot_points = [
                float(value)
                for value in snapshot_points
                if float(value) not in key_by_time
            ]
            if missing_snapshot_points:
                raise ValueError(
                    "--snapshot-time-points contains values absent from the "
                    f"simulation grid: {missing_snapshot_points}"
                )
            snapshot_time_keys = [key_by_time[float(value)] for value in snapshot_points]
        snapshot_time_set = {float(key) for key in snapshot_time_keys}
        snapshot_adata_dict = {
            key: result.adata_dict[key]
            for key in snapshot_time_keys
        }
        observed_variants = {}
        if result.sde_points_split is not None and result.slice_labels_split is not None:
            for time_value in observed:
                if float(time_value) not in snapshot_time_set:
                    continue
                idx = result.ts_points.index(float(time_value))
                observed_df = df[np.isclose(df["samples"], float(time_value))]
                observed_variants[float(time_value)] = {
                    "Observed": (
                        observed_df[feature_cols].to_numpy(dtype=np.float32)[:, :2],
                        observed_df[annotation_key].astype(str).to_numpy(),
                    ),
                    "Generated": (
                        np.asarray(result.sde_points_split[idx], dtype=np.float32)[:, :2],
                        np.asarray(result.slice_labels_split[idx]).astype(str),
                    ),
                }
        cb.tl.save_timepoint_snapshots(
            adata_dict=snapshot_adata_dict,
            time_keys=snapshot_time_keys,
            annotation_key=annotation_key,
            label_to_color=label_to_color,
            observed_variants=observed_variants or None,
            snapshot_dir=str(output_dir / "snapshots"),
            background_color=None,
            font_color="#1a1a1a",
            snapshot_point_size=2.5,
            snapshot_alpha=0.9,
            mosaic_cols=4,
            mosaic_cell_size=2.2,
            mosaic_show_title=True,
            save_pdf=True,
        )

    lineage_labels = result.predicted_labels_list or result.predicted_labels_split
    fixed_particle_state = {
        "status": "not available",
        "identity_source": None,
        "labels_npz": None,
        "composition_csv": None,
        "particle_counts_by_time": None,
    }
    if result.predicted_labels_list is not None:
        import pandas as pd

        labels_path = output_dir / "fixed_particle_lineage_labels.npz"
        composition_path = output_dir / "fixed_particle_composition.csv"
        label_payload = {
            "time_points": np.asarray(result.ts_points, dtype=np.float64),
        }
        composition_rows = []
        particle_counts = {}
        for index, (time_value, labels_at_time) in enumerate(
            zip(result.ts_points, result.predicted_labels_list)
        ):
            labels_array = np.asarray(labels_at_time).astype("U")
            label_payload[f"labels_{index}"] = labels_array
            particle_counts[str(float(time_value))] = int(len(labels_array))
            values, counts = np.unique(labels_array, return_counts=True)
            for label, count in zip(values, counts):
                composition_rows.append(
                    {
                        "time": float(time_value),
                        "celltype": str(label),
                        "count": int(count),
                        "fraction": float(count / max(len(labels_array), 1)),
                    }
                )
        np.savez_compressed(labels_path, **label_payload)
        pd.DataFrame(composition_rows).to_csv(composition_path, index=False)
        fixed_particle_state = {
            "status": "completed",
            "identity_source": "non_split_fixed_particles",
            "labels_npz": str(labels_path),
            "composition_csv": str(composition_path),
            "particle_counts_by_time": particle_counts,
        }
    static_exports = {}
    if not args.skip_lineage and lineage_labels is not None:
        lineage_cfg = config.get("lineage", {})
        lineage_path = output_dir / "lineage_sankey.html"
        lineage_fig = cb.tl.plot_lineage_sankey(
            predicted_labels_list=lineage_labels,
            time_keys=result.time_keys,
            label_to_color=label_to_color,
            out_html=str(lineage_path),
            min_flow=lineage_cfg.get("min_flow"),
            keep_source_cumfrac=lineage_cfg.get("keep_source_cumfrac"),
            normalize_mode=lineage_cfg.get("normalize_mode"),
            style=str(lineage_cfg.get("style", "nature-methods")),
            title=str(lineage_cfg.get("title", "Cell Fate Transitions")),
        )
        static_exports["lineage"] = _export_plotly_static(
            lineage_fig,
            output_dir / "lineage_sankey",
        )

    all_communications = None
    if not args.skip_communication:
        communication_cfg = config.get("communication", {})
        all_communications = cb.tl.compute_timepoint_communications(
            adata_dict=result.communication_adata_dict,
            time_points=result.ts_points,
            annotation_key=annotation_key,
            f_net=runtime.f_net,
            device=device,
            out_dir=str(output_dir / "attention"),
            remove_self_loop=bool(communication_cfg.get("remove_self_loop", False)),
            winsor_quantile=float(communication_cfg.get("winsor_quantile", 0.995)),
            save_pickle_path=str(output_dir / "all_time_communications.pkl"),
        )

    plot_cfg = dict(config.get("plot_3d", {}))
    layout_cfg = dict(config.get("plot_3d_layout", {}))
    if not args.skip_3d:
        if all_communications is None:
            raise ValueError("3D communication plot requires communication analysis; remove --skip-communication.")
        plot_path = output_dir / "spatiotemporal_3d.html"
        plot_fig = cb.tl.plot_spatiotemporal_3d(
            adata_dict=result.adata_dict,
            all_time_communications=all_communications,
            time_keys=[str(value) for value in plot_3d_points],
            plot_time_points=plot_3d_points,
            ts_points=result.ts_points,
            observed_time_points=result.observed_time_points,
            interp_points=result.interp_points,
            annotation_key=annotation_key,
            label_to_color=label_to_color,
            out_html=str(plot_path),
            predicted_labels_list=lineage_labels,
            **plot_cfg,
        )
        if layout_cfg:
            plot_fig.update_layout(**layout_cfg)
            plot_fig.write_html(str(plot_path))
        static_exports["spatiotemporal_3d"] = _export_plotly_static(
            plot_fig,
            output_dir / "spatiotemporal_3d",
        )

    manifest = {
        "git": _git_revision(),
        "dataset": _nested(config, "dataset", "name", "unknown"),
        "config": str(config_path),
        "aligned_h5ad": str(h5ad_path),
        "model_dir": str(model_dir),
        "model_format": args.model_format,
        "model_weight_stage": loaded.weight_stage,
        "model_score_stage": loaded.score_stage,
        "device": device,
        "dim": int(dim),
        "time_key": resolved_time_key,
        "observed_time_points": result.observed_time_points,
        "interpolated_time_points": result.interp_points,
        "classifier_cache_dir": classifier_cache_dir,
        "classifier_knn_neighbors": int(
            _resolve(args.classifier_knn_neighbors, config, "classifier", "knn_neighbors", 10)
        ),
        "spatial_warp_to_observed_piecewise": warp_piecewise,
        "spatial_warp_visualization_only": warp_visualization_only,
        "skip_nonsplit_sde": bool(args.skip_nonsplit_sde),
        "snapshot_time_points": (
            [float(value) for value in snapshot_points]
            if snapshot_points is not None
            else [float(value) for value in result.ts_points]
        ),
        "dense_time_grid": (
            {
                "min": (
                    float(args.dense_time_min)
                    if args.dense_time_min is not None
                    else float(min(observed))
                ),
                "max": (
                    float(args.dense_time_max)
                    if args.dense_time_max is not None
                    else float(max(observed))
                ),
                "step": float(args.dense_time_step),
            }
            if args.dense_time_step is not None
            else None
        ),
        "classifier": {
            "cache_path": result.classifier_cache_path,
            "cache_sha256": (
                _sha256(Path(result.classifier_cache_path))
                if result.classifier_cache_path is not None
                and Path(result.classifier_cache_path).is_file()
                else None
            ),
            "validation_accuracy": result.classifier_accuracy,
            "validation_balanced_accuracy": result.classifier_balanced_accuracy,
            "metadata": result.classifier_metadata,
            "evaluation": result.classifier_evaluation,
            "knn_neighbors": int(
                _resolve(
                    args.classifier_knn_neighbors,
                    config,
                    "classifier",
                    "knn_neighbors",
                    10,
                )
            ),
        },
        "trajectory_semantics": {
            "lineage_identity_source": (
                None if args.skip_nonsplit_sde else "non_split_fixed_particles"
            ),
            "slice_population_source": "split_sde_birth_death",
            "slice_coordinate_source": (
                "piecewise_warped_spatial"
                if warp_piecewise
                else "split_sde_state"
            ),
            "slice_label_source": (
                "prewarp_split_state" if warp_visualization_only else "display_state"
            ),
            "communication_state_source": (
                None
                if args.skip_communication
                else (
                    "prewarp_split_state"
                    if warp_visualization_only
                    else "display_state"
                )
            ),
            "piecewise_segment_continuation_source": (
                "prewarp_split_state"
                if warp_piecewise and warp_visualization_only
                else "display_state"
            ),
            "display_boundary_contract": (
                "shared piecewise boundaries are preserved exactly after warp"
                if warp_piecewise
                else None
            ),
        },
        "fixed_particle_lineage_state": fixed_particle_state,
        "simulation_seeds": result.simulation_seeds,
        "video": {
            "enabled": bool(args.render_video),
            "fps": int(args.video_fps),
            "frames": int(len(result.ts_points)),
            "time_points": result.ts_points,
            "population_source": "split_sde_birth_death",
            "outputs": video_outputs,
        },
        "plot_3d": {
            "enabled": not bool(args.skip_3d),
            "time_points": plot_3d_points,
            "settings": plot_cfg,
            "layout": layout_cfg,
        },
        "static_exports": static_exports,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
