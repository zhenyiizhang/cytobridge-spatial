#!/usr/bin/env python3
"""Create the formal chicken-heart perturbation and paper figure bank.

The standard package workflow remains the source of the observed/generated
snapshots, velocity, growth, composition, communication, and ligand-receptor
outputs.  This runner adds the manuscript perturbation experiments that must
start once from the real D4 population (processed time 0) and evolve
continuously to D14.  It never re-anchors a generated trajectory at D7 or D10.

The experiments are fixed-population, single-seed model-sensitivity analyses:

* equal-particle removal of Endocardial, Valve, or Immature myocardial cells;
* paired interaction-on versus zero-interaction dynamics from the same D4 x0.

They are not causal knockout experiments and do not estimate uncertainty.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import CytoBridge as cb  # noqa: E402


TIME_POINTS = tuple(float(value) for value in np.arange(0.0, 3.0 + 0.5, 0.5))
DISPLAY_TIMES = (0.0, 1.0, 2.0, 3.0)
CELLTYPE_ABLATIONS = {
    "remove_endocardial": "Endocardial cells",
    "remove_valve": "Valve cells",
    "remove_immature_myocardial": "Immature myocardial cells",
}
RANDOM_SEED = 42
INTERACTION_SEED = 10_043
DT = 0.05
SIGMA = 0.03
INTERACTION_M = 1024


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _require_empty_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"Refusing to reuse non-empty output directory: {resolved}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _validate_standard_downstream_summary(summary: Mapping[str, object]) -> None:
    if summary.get("dataset") != "chicken_heart":
        raise RuntimeError("Standard downstream summary is not for chicken_heart.")
    required_analyses = (
        "velocity",
        "growth",
        "composition",
        "communication",
        "figures",
        "gene_dynamics",
        "ligand_receptor",
    )
    analyses = summary.get("analyses", {})
    if not isinstance(analyses, Mapping):
        raise RuntimeError("Standard downstream summary lacks analyses.")
    incomplete = {
        name: analyses.get(name, {}).get("status")
        for name in required_analyses
        if not isinstance(analyses.get(name), Mapping)
        or analyses.get(name, {}).get("status") != "completed"
    }
    if incomplete:
        raise RuntimeError(
            f"Standard package downstream analyses are incomplete: {incomplete}."
        )


def _coordinate_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(f"|{array.dtype.str}|".encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_workflow_aligned_input(adata) -> dict[str, object]:
    """Validate the formal package-preprocessed H5AD used by this runner."""

    for key in ("preprocess_info", "spatial_alignment_info"):
        if key not in adata.uns or not isinstance(adata.uns[key], Mapping):
            raise RuntimeError(
                "Chicken-heart paper downstream requires the formal package "
                f"preprocessing output with uns[{key!r}]."
            )
    required_obsm = {"spatial_aligned": 2, "X_latent": 50}
    for key, width in required_obsm.items():
        values = np.asarray(adata.obsm.get(key))
        if values.shape != (adata.n_obs, width) or not np.isfinite(values).all():
            raise RuntimeError(
                f"Formal chicken-heart obsm[{key!r}] must be finite "
                f"({adata.n_obs}, {width})."
            )
    if "counts" not in adata.layers:
        raise RuntimeError("Formal chicken-heart H5AD lacks layers['counts'].")
    for key in ("timepoint", "time_point_processed", "region", "celltype_prediction"):
        if key not in adata.obs:
            raise RuntimeError(f"Formal chicken-heart H5AD lacks obs[{key!r}].")
        values = adata.obs[key]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise RuntimeError(
                f"Formal chicken-heart obs[{key!r}] contains missing values."
            )
    legacy_annotation_alias_ignored = bool(
        "Annotation" in adata.obs
        and not np.array_equal(
            adata.obs["Annotation"].astype(str).to_numpy(),
            adata.obs["celltype_prediction"].astype(str).to_numpy(),
        )
    )
    observed_times = sorted(
        np.unique(
            adata.obs["time_point_processed"].to_numpy(dtype=np.float64)
        ).tolist()
    )
    if observed_times != [0.0, 1.0, 2.0, 3.0]:
        raise RuntimeError(
            "Formal chicken-heart processed times must be [0, 1, 2, 3], "
            f"received {observed_times}."
        )
    anatomical = cb.pp.chicken_heart_anatomical_orientation_qc(adata)
    if anatomical.get("status") != "pass":
        raise RuntimeError(
            "Formal chicken-heart aligned anatomy failed orientation QC: "
            f"{anatomical.get('failures')}."
        )
    return {
        "source_kind": "package_preprocessed_aligned_h5ad",
        "coordinate_sha256": _coordinate_sha256(adata.obsm["spatial_aligned"]),
        "downstream_annotation_key": "celltype_prediction",
        "legacy_annotation_alias_ignored": legacy_annotation_alias_ignored,
        "anatomical_orientation_qc": anatomical,
    }


def _label_colors(labels: Sequence[str]) -> dict[str, str]:
    order = tuple(sorted({str(label) for label in labels}))
    palette = mpl.colormaps["tab20"].resampled(len(order))
    return {
        label: mpl.colors.to_hex(palette(index)) for index, label in enumerate(order)
    }


def _composition_rows(
    labels_by_condition: Mapping[str, Sequence[np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, frames in labels_by_condition.items():
        for time_index, (time_value, labels) in enumerate(zip(TIME_POINTS, frames)):
            values = pd.Series(np.asarray(labels).astype(str))
            counts = values.value_counts().sort_index()
            total = int(counts.sum())
            for celltype, count in counts.items():
                rows.append(
                    {
                        "condition": str(condition),
                        "time_index": int(time_index),
                        "time": float(time_value),
                        "celltype": str(celltype),
                        "count": int(count),
                        "fraction": float(count / total),
                    }
                )
    return pd.DataFrame(rows)


def _plot_metric_summary(table: pd.DataFrame, output_path: Path, title: str) -> None:
    selected = table.loc[table["space"].isin(["spatial", "latent"])].copy()
    with mpl.rc_context(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
        colors = {"spatial": "#2C7FB8", "latent": "#D95F0E"}
        for ax, metric, label in zip(
            axes,
            ("w2", "centroid_shift"),
            ("Wasserstein-2", "Centroid shift"),
            strict=True,
        ):
            for (variant, space), group in selected.groupby(
                ["variant", "space"], sort=True
            ):
                display_variant = {
                    "interaction_off": "Without interaction",
                    "remove_endocardial": "Endocardial-cell removal",
                    "remove_valve": "Valve-cell removal",
                    "remove_immature_myocardial": "Immature-myocardial-cell removal",
                }.get(str(variant), str(variant).replace("_", " "))
                ax.plot(
                    group["time"],
                    group[metric],
                    marker="o",
                    markersize=3,
                    linewidth=1.1,
                    color=colors[str(space)],
                    alpha=0.72 if len(selected["variant"].unique()) > 1 else 1.0,
                    label=f"{display_variant} · {space}",
                )
            ax.set_xlabel("Processed developmental time")
            ax.set_ylabel(label)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        fig.suptitle(title, fontweight="bold")
        fig.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.27, wspace=0.32)
        fig.savefig(output_path, dpi=320, facecolor="white")
        plt.close(fig)


def _write_lr_attention_figures(
    standard_downstream: Path,
    output_dir: Path,
) -> list[Path]:
    """Render package LR/attention tables as manuscript-ready vector panels."""

    lr_dir = standard_downstream / "ligand_receptor"
    pair = pd.read_csv(lr_dir / "pair_timecourse.csv")
    pattern = pd.read_csv(lr_dir / "pattern_summary.csv")
    coverage = pd.read_csv(lr_dir / "coverage.csv")
    communication = pd.read_csv(
        standard_downstream / "communication" / "communication_by_celltype.csv"
    )
    required_pair = {"time", "pair", "score"}
    required_pattern = {"pair", "auc", "peak_time", "peak_score"}
    required_communication = {"time", "source", "target", "attention_per_source"}
    if not required_pair.issubset(pair.columns):
        raise KeyError("LR pair_timecourse.csv lacks the formal score columns.")
    if not required_pattern.issubset(pattern.columns):
        raise KeyError("LR pattern_summary.csv lacks the formal AUC columns.")
    if not required_communication.issubset(communication.columns):
        raise KeyError("Communication table lacks source/target attention columns.")
    if pair.empty or pattern.empty or communication.empty or coverage.empty:
        raise ValueError("LR/attention tables must be non-empty.")

    output_dir.mkdir(parents=True, exist_ok=False)
    top_pairs = (
        pattern.sort_values(["auc", "pair"], ascending=[False, True], kind="stable")
        .head(8)["pair"]
        .astype(str)
        .tolist()
    )
    selected = pair.loc[pair["pair"].astype(str).isin(top_pairs)].copy()
    selected["pair"] = pd.Categorical(
        selected["pair"].astype(str), categories=top_pairs, ordered=True
    )
    selected = selected.sort_values(["pair", "time"], kind="stable")
    selected_path = output_dir / "top_lr_pair_timecourses.csv"
    selected.to_csv(selected_path, index=False)

    figure_paths: list[Path] = [selected_path]
    with mpl.rc_context(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        colors = mpl.colormaps["tab10"].resampled(len(top_pairs))
        for index, lr_pair in enumerate(top_pairs):
            group = selected.loc[selected["pair"].astype(str) == lr_pair]
            ax.plot(
                group["time"],
                np.log1p(group["score"].to_numpy(float)),
                marker="o",
                markersize=3,
                linewidth=1.25,
                color=colors(index),
                label=lr_pair.replace("_", "–"),
            )
        ax.set_xlabel("Processed developmental time")
        ax.set_ylabel("log(1 + model-derived LR score)")
        ax.set_xticks(TIME_POINTS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.legend(
            bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=7
        )
        ax.set_title("Top conserved-symbol ligand–receptor trajectories")
        fig.tight_layout()
        for extension in ("pdf", "png"):
            path = output_dir / f"top_lr_pair_timecourses.{extension}"
            fig.savefig(path, dpi=320, facecolor="white", bbox_inches="tight")
            figure_paths.append(path)
        plt.close(fig)

        regions = sorted(
            set(communication["source"].astype(str))
            | set(communication["target"].astype(str))
        )
        selected_times = [
            time
            for time in DISPLAY_TIMES
            if np.isclose(communication["time"].to_numpy(float), time, atol=1e-8).any()
        ]
        matrices: list[np.ndarray] = []
        for time_value in selected_times:
            frame = communication.loc[
                np.isclose(communication["time"].to_numpy(float), time_value, atol=1e-8)
            ]
            matrix = (
                frame.pivot_table(
                    index="source",
                    columns="target",
                    values="attention_per_source",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .reindex(index=regions, columns=regions, fill_value=0.0)
                .to_numpy(float)
            )
            matrices.append(np.log1p(matrix))
        vmax = max(float(np.max(matrix)) for matrix in matrices)
        short_region = {
            "Compact LV and inter-ventricular septum": "Compact LV/septum",
            "Trabecular LV and endocardium": "Trabecular LV",
        }
        display_regions = [short_region.get(region, region) for region in regions]
        fig = plt.figure(figsize=(11.7, 4.6))
        grid = fig.add_gridspec(
            1,
            len(selected_times) + 1,
            width_ratios=[1.0] * len(selected_times) + [0.055],
            wspace=0.18,
        )
        axes = [
            fig.add_subplot(grid[0, column]) for column in range(len(selected_times))
        ]
        colorbar_axis = fig.add_subplot(grid[0, -1])
        image = None
        for column, (time_value, matrix) in enumerate(
            zip(selected_times, matrices, strict=True)
        ):
            ax = axes[column]
            image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=vmax, aspect="equal")
            ax.set_title(f"t={time_value:g}")
            ax.set_xticks(range(len(regions)), display_regions, rotation=90, fontsize=6)
            if column == 0:
                ax.set_yticks(range(len(regions)), display_regions, fontsize=6)
                ax.set_ylabel("Sender anatomical region")
            else:
                ax.set_yticks([])
            ax.set_xlabel("Receiver anatomical region")
        assert image is not None
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("log(1 + attention per source)")
        fig.suptitle("Learned communication-attention evolution", fontweight="bold")
        fig.subplots_adjust(left=0.12, right=0.96, top=0.82, bottom=0.31)
        for extension in ("pdf", "png"):
            path = output_dir / f"communication_attention_heatmaps.{extension}"
            fig.savefig(path, dpi=320, facecolor="white", bbox_inches="tight")
            figure_paths.append(path)
        plt.close(fig)

    coverage_summary = {
        "database_pairs": int(coverage["n_lr_pairs_database"].iloc[0]),
        "scored_complete_pairs": int(coverage["n_lr_pairs_scored"].iloc[0]),
        "active_lr_features": int(coverage["n_active_lr_features"].iloc[0]),
        "requested_lr_symbols": int(coverage["n_requested_lr_symbols"].iloc[0]),
        "scope": "human CellChatDB conserved-symbol proxy; not species-complete",
        "score_interpretation": "model-derived attention weighted by reconstructed expression",
    }
    coverage_path = output_dir / "lr_coverage_summary.json"
    coverage_path.write_text(
        json.dumps(coverage_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    figure_paths.append(coverage_path)
    return figure_paths


def _save_comparison_grid(
    *,
    trajectories: Mapping[str, Sequence[np.ndarray]],
    labels: Mapping[str, Sequence[np.ndarray]],
    colors: Mapping[str, str],
    output_stem: Path,
    title: str,
    display_names: Mapping[str, str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    display_names = {} if display_names is None else dict(display_names)
    plotted_trajectories = {
        display_names.get(condition, condition): values
        for condition, values in trajectories.items()
    }
    plotted_labels = {
        display_names.get(condition, condition): values
        for condition, values in labels.items()
    }
    for extension in ("pdf", "png"):
        path = output_stem.with_suffix(f".{extension}")
        figure = cb.pl.plot_trajectory_comparison_grid(
            plotted_trajectories,
            TIME_POINTS,
            out_path=str(path),
            labels_by_condition=plotted_labels,
            label_to_color=colors,
            selected_times=DISPLAY_TIMES,
            dim_pair=(0, 1),
            point_size=5.0,
            alpha=0.86,
            figsize_per_panel=(2.45, 2.3),
            shared_axis_limits=True,
            show_counts=True,
            show_legend=True,
            legend_title="Predicted cell type",
            title=title,
        )
        plt.close(figure)
        paths.append(path)
    return paths


def _trajectory_labeler(cache, device: str):
    def labeler(points, time_points):
        return cb.tl.predict_labels_for_trajectories(
            sde_points=points,
            ts_points=time_points,
            model=cache.model,
            label_encoder=cache.label_encoder,
            feature_dim=int(cache.feature_dim),
            device=device,
            knn_neighbors=1,
            include_time_feature=cache.include_time_feature,
            spatial_indices=(0, 1),
        )

    return labeler


def run(args: argparse.Namespace) -> dict[str, object]:
    run_root = args.run_root.expanduser().resolve()
    input_h5ad = args.input_h5ad.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    standard_downstream = args.standard_downstream.expanduser().resolve()
    output_dir = _require_empty_output(args.output_dir)
    for path, description in (
        (input_h5ad, "prepared chicken-heart H5AD"),
        (model_dir, "training directory"),
        (standard_downstream / "summary.json", "standard downstream summary"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")

    standard_summary = json.loads(
        (standard_downstream / "summary.json").read_text(encoding="utf-8")
    )
    _validate_standard_downstream_summary(standard_summary)

    adata = ad.read_h5ad(input_h5ad)
    input_contract = _validate_workflow_aligned_input(adata)
    dataframe, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key="time_point_processed",
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        annotation_key="celltype_prediction",
    )
    feature_columns = cb.tl.infer_feature_columns(
        dataframe, annotation_column="celltype_prediction"
    )
    loaded = cb.tl.load_dynamical_model_from_dir(
        model_dir,
        dim=len(feature_columns),
        device=args.device,
    )
    runtime = cb.tl.build_dynamical_runtime(loaded)

    classifier, classifier_path = cb.tl.train_cached_mlp_classifier_from_adata(
        adata,
        cache_dir=output_dir / "classifier_cache",
        cache_tag="chicken-heart-paper-celltype-spatial2-latent50",
        label_col="celltype_prediction",
        time_key=resolved_time_key,
        obsm_key="X_latent",
        spatial_key="spatial_aligned",
        concat_spatial=True,
        hidden_size=128,
        epochs=int(args.classifier_epochs),
        lr=1e-3,
        test_size=0.1,
        seed=RANDOM_SEED,
        device=args.device,
        include_time_feature=True,
        n_features=None,
        best_epoch_metric="bacc",
        train_on_full_data=True,
        strict_stratification=False,
    )
    if int(classifier.feature_dim) != len(feature_columns):
        raise RuntimeError(
            "Chicken-heart classifier feature width does not match model state."
        )
    labeler = _trajectory_labeler(classifier, args.device)
    labels_all = adata.obs["celltype_prediction"].astype(str).tolist()
    colors = _label_colors(labels_all)

    results_dir = output_dir / "perturbations"
    results_dir.mkdir(parents=True, exist_ok=False)
    generated_files: list[Path] = [classifier_path]
    generated_files.extend(
        _write_lr_attention_figures(
            standard_downstream,
            output_dir / "lr_attention",
        )
    )
    ablation_metrics: list[pd.DataFrame] = []
    composition_tables: list[pd.DataFrame] = []

    for variant, label in CELLTYPE_ABLATIONS.items():
        result = cb.tl.run_virtual_cell_type_ablation(
            adata,
            runtime,
            ablations={variant: [label]},
            time_points=TIME_POINTS,
            output_dir=results_dir / variant,
            time_index=0,
            n_samples=None,
            dt=DT,
            resample_dt=DT,
            sigma=SIGMA,
            growth_alpha=0.0,
            interaction_m=INTERACTION_M,
            max_particles=100_000,
            device=args.device,
            time_key=resolved_time_key,
            annotation_key="celltype_prediction",
            obsm_key="X_latent",
            spatial_key="spatial_aligned",
            concat_spatial=True,
            spatial_dim=2,
            random_seed=RANDOM_SEED,
            interaction_seed=INTERACTION_SEED,
            common_random_seed=True,
            max_ot_points=None,
            mass_control=True,
            trajectory_labeler=labeler,
            save_data=True,
            save_snapshots=False,
            label_to_color=colors,
            verbose=True,
        )
        metrics = result.metrics.copy()
        metrics.insert(0, "excluded_celltype", label)
        ablation_metrics.append(metrics)
        composition = result.label_composition.copy()
        composition.insert(0, "excluded_celltype", label)
        composition_tables.append(composition)
        generated_files.extend(result.files)
        generated_files.extend(
            _save_comparison_grid(
                trajectories={
                    "baseline": result.baseline_points,
                    variant: result.ablation_points[variant],
                },
                labels={
                    "baseline": result.baseline_labels,
                    variant: result.ablation_labels[variant],
                },
                colors=colors,
                output_stem=results_dir / variant / "paper_spatial_comparison",
                title=(f"D4-origin fixed-population sensitivity: {label} removal"),
                display_names={
                    "baseline": "Baseline",
                    variant: f"Without {label}",
                },
            )
        )

    all_ablation_metrics = pd.concat(ablation_metrics, ignore_index=True)
    metrics_path = results_dir / "celltype_ablation_metrics.csv"
    all_ablation_metrics.to_csv(metrics_path, index=False)
    composition_path = results_dir / "celltype_ablation_composition.csv"
    pd.concat(composition_tables, ignore_index=True).to_csv(
        composition_path, index=False
    )
    metric_figure = results_dir / "celltype_ablation_metric_summary.pdf"
    _plot_metric_summary(
        all_ablation_metrics,
        metric_figure,
        "D4-origin cell-type-removal sensitivity",
    )
    generated_files.extend([metrics_path, composition_path, metric_figure])

    t0 = dataframe.loc[
        np.isclose(dataframe["samples"].to_numpy(float), 0.0), feature_columns
    ].to_numpy(np.float32)
    interaction = cb.tl.run_virtual_interaction_ablation(
        t0,
        runtime,
        time_points=TIME_POINTS,
        output_dir=results_dir / "interaction_off",
        variant_name="interaction_off",
        dt=DT,
        resample_dt=DT,
        sigma=SIGMA,
        growth_alpha=0.0,
        interaction_m=INTERACTION_M,
        max_particles=100_000,
        spatial_dim=2,
        device=args.device,
        random_seed=RANDOM_SEED,
        save_data=True,
        save_snapshots=False,
        verbose=True,
    )
    interaction_labels = {
        "interaction_on": tuple(labeler(interaction.baseline_points, TIME_POINTS)),
        "interaction_off": tuple(labeler(interaction.ablated_points, TIME_POINTS)),
    }
    interaction_composition = _composition_rows(interaction_labels)
    interaction_composition_path = (
        results_dir / "interaction_off" / "interaction_ablation_composition.csv"
    )
    interaction_composition.to_csv(interaction_composition_path, index=False)
    generated_files.extend(interaction.files)
    generated_files.append(interaction_composition_path)
    generated_files.extend(
        _save_comparison_grid(
            trajectories={
                "interaction_on": interaction.baseline_points,
                "interaction_off": interaction.ablated_points,
            },
            labels=interaction_labels,
            colors=colors,
            output_stem=(results_dir / "interaction_off" / "paper_spatial_comparison"),
            title="D4-origin interaction-force sensitivity",
            display_names={
                "interaction_on": "With interaction",
                "interaction_off": "Without interaction",
            },
        )
    )
    interaction_metric_figure = (
        results_dir / "interaction_off" / "interaction_metric_summary.pdf"
    )
    _plot_metric_summary(
        interaction.metrics,
        interaction_metric_figure,
        "D4-origin interaction-on versus interaction-off sensitivity",
    )
    generated_files.append(interaction_metric_figure)

    caption_path = output_dir / "CAPTION.md"
    caption_path.write_text(
        "\n".join(
            (
                "# Chicken-heart perturbation figure bank",
                "",
                "All simulations begin once from the observed D4 population (processed ",
                "time 0) and evolve continuously to D14 without observed-slice ",
                "replacement, re-anchoring, or spatial warping. Cell-type removals use ",
                "separate equal-particle baseline and target-excluded cohorts with ",
                "growth disabled. The interaction experiment uses the same D4 state and ",
                "branch seed with the learned interaction force set to zero. These are ",
                "single-seed model-sensitivity analyses, not causal knockouts or ",
                "uncertainty estimates.",
                "The LR and communication panels summarize model-derived attention ",
                "and reconstructed-expression scores using the human CellChatDB ",
                "conserved-symbol proxy; they are not direct ligand-flux measurements ",
                "or a species-complete chicken interaction atlas.",
                "",
            )
        ),
        encoding="utf-8",
    )
    generated_files.append(caption_path)

    manifest_path = output_dir / "manifest.json"
    output_records = []
    for path in sorted({Path(path).resolve() for path in generated_files}):
        if not path.is_file():
            raise FileNotFoundError(f"Expected perturbation output is missing: {path}")
        output_records.append(
            {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "created_at": _utc_now(),
        "run_root": str(run_root),
        "input_h5ad": {
            "path": str(input_h5ad),
            "sha256": _sha256(input_h5ad),
            "coordinate_sha256": input_contract["coordinate_sha256"],
        },
        "model_dir": str(model_dir),
        "standard_downstream": {
            "path": str(standard_downstream),
            "summary_sha256": _sha256(standard_downstream / "summary.json"),
        },
        "contract": {
            "origin": "observed D4 / processed time 0",
            "time_points": list(TIME_POINTS),
            "continuous_global_t0": True,
            "reanchoring": False,
            "spatial_warp": False,
            "growth_alpha": 0.0,
            "mass_control": True,
            "single_seed_model_sensitivity": True,
            "causal_claim": False,
            "celltype_ablations": CELLTYPE_ABLATIONS,
            "interaction_ablation": "learned interaction force set to zero",
        },
        "classifier": {
            "path": str(classifier_path),
            "sha256": _sha256(classifier_path),
            "feature_dim": int(classifier.feature_dim),
            "accuracy": classifier.accuracy,
            "balanced_accuracy": classifier.balanced_accuracy,
        },
        "outputs": output_records,
    }
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--standard-downstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--classifier-epochs", type=int, default=500)
    args = parser.parse_args(argv)
    if args.classifier_epochs <= 0:
        parser.error("--classifier-epochs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args)
    print(json.dumps(_json_ready(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
