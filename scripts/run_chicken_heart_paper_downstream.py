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
            ("wasserstein_2", "centroid_shift"),
            ("Wasserstein-2", "Centroid shift"),
            strict=True,
        ):
            for (variant, space), group in selected.groupby(
                ["variant", "space"], sort=True
            ):
                ax.plot(
                    group["time"],
                    group[metric],
                    marker="o",
                    markersize=3,
                    linewidth=1.1,
                    color=colors[str(space)],
                    alpha=0.72 if len(selected["variant"].unique()) > 1 else 1.0,
                    label=f"{variant} · {space}",
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


def _save_comparison_grid(
    *,
    trajectories: Mapping[str, Sequence[np.ndarray]],
    labels: Mapping[str, Sequence[np.ndarray]],
    colors: Mapping[str, str],
    output_stem: Path,
    title: str,
) -> list[Path]:
    paths: list[Path] = []
    for extension in ("pdf", "png"):
        path = output_stem.with_suffix(f".{extension}")
        figure = cb.pl.plot_trajectory_comparison_grid(
            trajectories,
            TIME_POINTS,
            out_path=str(path),
            labels_by_condition=labels,
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
    input_contract = cb.pp.validate_prepared_chicken_heart_input(adata)
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
                title=(f"D4-origin fixed-population sensitivity: remove {label}"),
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
