"""Package-native execution of the standard CytoBridge workflow.

The command-line workflow deliberately stays small.  Dataset presets describe
scientific parameters and dataset schema, while numerical work is delegated to
the public preprocessing, training, and downstream APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path
from typing import Any, Mapping


WORKFLOW_PRESETS = ("zebrafish", "mosta", "arista", "admouse")
_PRESET_ALIASES = {
    "ad": "admouse",
    "ad-mouse": "admouse",
    "zfish": "zebrafish",
}


@dataclass(frozen=True)
class WorkflowOptions:
    """Runtime paths and explicitly selected workflow operations."""

    input_h5ad: Path | None = None
    aligned_h5ad: Path | None = None
    model_dir: Path | None = None
    output_dir: Path | None = None
    training_config: str | None = None
    interaction_cutoff: float | None = None
    edge_predictor_path: Path | None = None
    edge_predictor_threshold: float | None = None
    edge_predictor_root: Path | None = None
    device: str = "cuda"
    model_format: str | None = None
    reference_h5ad: Path | None = None
    gene_dynamics: bool = False
    lr_database: Path | None = None
    lr_complex_mode: str = "min"
    preferred_species_tag: str | None = None
    reconstruction_diagnostic: bool = False
    steps: tuple[str, ...] = ()
    train: bool = False


def available_workflow_configs() -> tuple[str, ...]:
    """Return the names of dataset presets shipped in the wheel."""

    return WORKFLOW_PRESETS


def _package_file(directory: str, filename: str):
    return resources.files("CytoBridge").joinpath(directory, filename)


def _parse_config_text(text: str, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Reading YAML workflow config {source!r} requires PyYAML. "
                "Install the CytoBridge core dependencies or use JSON."
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Workflow config {source!r} must contain a mapping.")
    return value


def load_workflow_config(config: str | Path) -> tuple[dict[str, Any], str]:
    """Load a custom config path or a packaged dataset preset.

    Packaged presets are resolved with :mod:`importlib.resources`, so this works
    from an installed wheel and does not depend on a source checkout.
    """

    path = Path(config).expanduser()
    if path.is_file():
        resolved = path.resolve()
        return (
            _parse_config_text(
                resolved.read_text(encoding="utf-8"), source=str(resolved)
            ),
            str(resolved),
        )

    name = _PRESET_ALIASES.get(str(config).strip().lower(), str(config).strip().lower())
    if name not in WORKFLOW_PRESETS:
        choices = ", ".join(WORKFLOW_PRESETS)
        raise FileNotFoundError(
            f"Workflow config {config!r} is not a file or packaged preset. "
            f"Available presets: {choices}."
        )
    resource = _package_file("workflow_configs", f"{name}.json")
    return (
        _parse_config_text(resource.read_text(encoding="utf-8"), source=name),
        f"packaged preset: {name}",
    )


def _selected_steps(
    config: Mapping[str, Any], options: WorkflowOptions
) -> tuple[str, ...]:
    if options.steps:
        selected = list(options.steps)
    else:
        selected = list(
            config.get("steps", {}).get("default", ("preprocess", "downstream"))
        )
    if options.train and "train" not in selected:
        insert_at = (
            selected.index("downstream") if "downstream" in selected else len(selected)
        )
        selected.insert(insert_at, "train")
    return tuple(dict.fromkeys(str(step) for step in selected))


def _output_paths(
    config: Mapping[str, Any], options: WorkflowOptions
) -> dict[str, Path | None]:
    dataset_name = str(config["dataset"]["name"])
    output_dir = options.output_dir
    aligned = options.aligned_h5ad
    if (
        aligned is None
        and output_dir is not None
        and "preprocess" in _selected_steps(config, options)
    ):
        aligned = output_dir / "preprocess" / f"{dataset_name}_aligned.h5ad"
    model_dir = options.model_dir
    if model_dir is None and output_dir is not None and options.train:
        model_dir = output_dir / "training"
    return {
        "output_dir": output_dir,
        "aligned_h5ad": aligned,
        "model_dir": model_dir,
    }


def build_workflow_plan(
    config: Mapping[str, Any],
    *,
    source: str,
    options: WorkflowOptions,
) -> dict[str, Any]:
    """Build the concise execution plan shown by ``--dry-run``."""

    selected = _selected_steps(config, options)
    paths = _output_paths(config, options)
    dataset = config["dataset"]
    scientific = config["scientific"]
    preprocess_config = config.get("preprocess", {})
    train_config = config.get("train", {})
    downstream_config = config.get("downstream", {})

    steps: list[dict[str, Any]] = []

    if "preprocess" not in selected:
        steps.append(
            {
                "name": "preprocess",
                "status": "skipped",
                "compute": "GPU for spatial alignment",
            }
        )
    elif not preprocess_config.get("enabled", True):
        steps.append(
            {
                "name": "preprocess",
                "status": "unavailable in this preset",
                "compute": "not scheduled",
                "note": preprocess_config.get("note", "Use an aligned H5AD input."),
            }
        )
    else:
        missing = []
        if options.input_h5ad is None:
            missing.append("--input-h5ad")
        if options.output_dir is None:
            missing.append("--output-dir")
        steps.append(
            {
                "name": "preprocess",
                "status": "ready" if not missing else "missing input",
                "compute": "GPU recommended for spatial alignment",
                "missing": missing,
                "output": None
                if paths["aligned_h5ad"] is None
                else str(paths["aligned_h5ad"]),
                "note": preprocess_config.get("note"),
            }
        )

    if not options.train:
        steps.append(
            {
                "name": "train",
                "status": "skipped; add --train to run",
                "compute": "GPU required for production training",
            }
        )
    else:
        missing = []
        if paths["aligned_h5ad"] is None:
            missing.append("--aligned-h5ad (or run preprocess)")
        if options.output_dir is None:
            missing.append("--output-dir")
        training_preset = options.training_config or train_config.get("config")
        if not training_preset:
            missing.append("--training-config")
        if (
            train_config.get("requires_edge_predictor", False)
            and options.edge_predictor_path is None
        ):
            missing.append("--edge-predictor-path")
        steps.append(
            {
                "name": "train",
                "status": "ready" if not missing else "missing input",
                "compute": "GPU required for production training",
                "missing": missing,
                "training_config": training_preset,
                "interaction_cutoff": (
                    float(options.interaction_cutoff)
                    if options.interaction_cutoff is not None
                    else train_config.get("interaction_cutoff")
                ),
                "edge_predictor_threshold": (
                    float(options.edge_predictor_threshold)
                    if options.edge_predictor_threshold is not None
                    else train_config.get("edge_predictor_threshold")
                ),
                "output": None
                if paths["model_dir"] is None
                else str(paths["model_dir"]),
            }
        )

    if "downstream" not in selected:
        steps.append(
            {"name": "downstream", "status": "skipped", "compute": "GPU recommended"}
        )
    elif not downstream_config.get("enabled", True):
        steps.append(
            {
                "name": "downstream",
                "status": "unavailable in this preset",
                "compute": "not scheduled",
                "note": downstream_config.get(
                    "note", "No downstream recipe is defined."
                ),
            }
        )
    else:
        missing = []
        if paths["aligned_h5ad"] is None:
            missing.append("--aligned-h5ad (or run preprocess)")
        if paths["model_dir"] is None:
            missing.append("--model-dir (or add --train)")
        if options.output_dir is None:
            missing.append("--output-dir")
        analyses = [
            {
                "name": "interpolation and classification",
                "status": "enabled",
            },
            {"name": "time-slice velocity", "status": "enabled"},
            {"name": "growth", "status": "enabled when present in the model"},
            {"name": "cell-type composition", "status": "enabled"},
            {"name": "sparse communication", "status": "enabled"},
            {
                "name": "standard figures",
                "status": "enabled",
                "note": (
                    "snapshots, mosaic, growth, composition, velocity, and 3D "
                    "communication; lineage only with an explicit persistent-ID contract"
                ),
            },
            {
                "name": "gene dynamics",
                "status": "requested" if options.gene_dynamics else "not requested",
                "note": (
                    "requires exact PCA loadings in varm['PCs'] and center in "
                    "var['pca_center'] of --reference-h5ad or the aligned H5AD"
                ),
            },
            {
                "name": "strict ligand-receptor projection",
                "status": "requested"
                if options.lr_database is not None
                else "not requested",
                "missing": (
                    [f"LR database file not found: {options.lr_database}"]
                    if options.lr_database is not None
                    and not options.lr_database.expanduser().is_file()
                    else []
                ),
                "note": (
                    "uses all required complex subunits and the selected min/geometric-mean rule; "
                    "requires exact PCA metadata for generated slices"
                ),
            },
            {
                "name": "fitted-model reconstruction diagnostic",
                "status": (
                    "requested"
                    if options.reconstruction_diagnostic
                    else "not requested"
                ),
                "note": (
                    "descriptive W2 diagnostic; not a training holdout or "
                    "cross-method benchmark"
                ),
            },
        ]
        if (
            options.reference_h5ad is not None
            and not options.reference_h5ad.expanduser().is_file()
        ):
            analyses.append(
                {
                    "name": "reference data",
                    "status": "missing input",
                    "missing": [f"reference H5AD not found: {options.reference_h5ad}"],
                }
            )
        steps.append(
            {
                "name": "downstream",
                "status": "ready" if not missing else "missing input",
                "compute": "GPU recommended for SDE simulation and classifier fitting",
                "missing": missing,
                "model_format": options.model_format
                or downstream_config.get("model_format", "current"),
                "output": (
                    None
                    if options.output_dir is None
                    else str(options.output_dir / "downstream")
                ),
                "simulation": {
                    "observed_time_points": [
                        float(value) for value in downstream_config.get("observed", [])
                    ],
                    "interpolated_time_points": [
                        float(value)
                        for value in downstream_config.get("interpolated", [])
                    ],
                    "initial_particles": (
                        "all observed t0 cells"
                        if downstream_config.get("sde_n_samples") is None
                        else int(downstream_config["sde_n_samples"])
                    ),
                    "split_dt": float(downstream_config.get("split_sde_dt", 0.01)),
                    "sigma": float(downstream_config.get("split_sigma", 0.03)),
                    "growth_alpha": float(
                        downstream_config.get("split_growth_alpha", 1.0)
                    ),
                },
                "analyses": analyses,
            }
        )

    return {
        "config": source,
        "dataset": {
            "name": dataset["name"],
            "display_name": dataset.get("display_name", dataset["name"]),
        },
        "scientific": {
            "alpha_express": float(scientific["alpha_express"]),
            "alpha_spatial": float(scientific.get("alpha_spatial", 10.0)),
            "classifier_k": int(scientific["classifier_k"]),
            "seed": int(scientific["seed"]),
        },
        "selected_steps": list(selected),
        "steps": steps,
    }


def render_workflow_plan(plan: Mapping[str, Any]) -> str:
    """Render a workflow plan as readable terminal text."""

    dataset = plan["dataset"]
    scientific = plan["scientific"]
    lines = [
        "CytoBridge workflow plan",
        f"dataset: {dataset['display_name']} ({dataset['name']})",
        f"config: {plan['config']}",
        (
            "scientific parameters: "
            f"alpha_spatial={scientific['alpha_spatial']:g}, "
            f"alpha_express={scientific['alpha_express']:g}, "
            f"seed={scientific['seed']}, classifier_k={scientific['classifier_k']}"
        ),
        "steps:",
    ]
    for step in plan["steps"]:
        lines.append(f"  {step['name']}: {step['status']} ({step['compute']})")
        if step.get("missing"):
            lines.append(f"    missing: {', '.join(step['missing'])}")
        if step.get("training_config"):
            lines.append(f"    training config: {step['training_config']}")
        if step.get("interaction_cutoff") is not None:
            lines.append(f"    interaction cutoff: {step['interaction_cutoff']}")
        if step.get("edge_predictor_threshold") is not None:
            lines.append(
                f"    edge predictor threshold: {step['edge_predictor_threshold']}"
            )
        if step.get("model_format"):
            lines.append(f"    model format: {step['model_format']}")
        if step.get("output"):
            lines.append(f"    output: {step['output']}")
        if step.get("note"):
            lines.append(f"    note: {step['note']}")
        if step.get("simulation"):
            simulation = step["simulation"]
            lines.append(
                "    simulation: "
                f"observed={simulation['observed_time_points']}, "
                f"interpolated={simulation['interpolated_time_points']}, "
                f"initial particles={simulation['initial_particles']}, "
                f"dt={simulation['split_dt']:g}, sigma={simulation['sigma']:g}, "
                f"growth alpha={simulation['growth_alpha']:g}"
            )
        for analysis in step.get("analyses", []):
            lines.append(f"    {analysis['name']}: {analysis['status']}")
            if analysis.get("missing"):
                lines.append(f"      missing: {', '.join(analysis['missing'])}")
            if analysis.get("note"):
                lines.append(f"      note: {analysis['note']}")
    return "\n".join(lines)


def plan_missing_inputs(plan: Mapping[str, Any]) -> list[str]:
    """Return selected steps that cannot run because an input was not supplied."""

    missing = [
        f"{step['name']}: {', '.join(step.get('missing', []))}"
        for step in plan["steps"]
        if step.get("status") == "missing input"
    ]
    for step in plan["steps"]:
        for analysis in step.get("analyses", []):
            if analysis.get("missing"):
                missing.append(f"{analysis['name']}: {', '.join(analysis['missing'])}")
    return missing


def _read_training_config(config_name: str) -> dict[str, Any]:
    path = Path(config_name).expanduser()
    if path.is_file():
        return _parse_config_text(path.read_text(encoding="utf-8"), source=str(path))
    filename = (
        config_name if str(config_name).endswith(".yaml") else f"{config_name}.yaml"
    )
    resource = _package_file("configs", filename)
    if not resource.is_file():
        raise FileNotFoundError(f"Training config not found: {config_name}")
    return _parse_config_text(resource.read_text(encoding="utf-8"), source=filename)


def _run_preprocess(
    config: Mapping[str, Any],
    options: WorkflowOptions,
    *,
    aligned_h5ad: Path,
) -> Path:
    from CytoBridge.pp import AlignConfig, preprocess_align_to_files

    preprocess_config = config["preprocess"]
    align_values = dict(preprocess_config.get("align", {}))
    if (
        "spatial_obs_keys" in align_values
        and align_values["spatial_obs_keys"] is not None
    ):
        align_values["spatial_obs_keys"] = tuple(align_values["spatial_obs_keys"])
    cfg = AlignConfig(**align_values)
    output_csv = aligned_h5ad.with_suffix(".csv")
    adata = preprocess_align_to_files(
        h5ad_path=str(options.input_h5ad),
        time_key=str(preprocess_config["time_key"]),
        output_csv=str(output_csv),
        output_h5ad=str(aligned_h5ad),
        cfg=cfg,
        batch_indices=preprocess_config.get("batch_indices"),
        device=options.device,
    )
    annotation_source = preprocess_config.get("annotation_source")
    annotation_key = str(config["dataset"].get("annotation_key", "Annotation"))
    if annotation_source and annotation_source in adata.obs:
        adata.obs[annotation_key] = (
            adata.obs[str(annotation_source)].astype(str).to_numpy()
        )
        adata.write_h5ad(aligned_h5ad)
    return aligned_h5ad


def _run_train(
    config: Mapping[str, Any],
    options: WorkflowOptions,
    *,
    aligned_h5ad: Path,
    model_dir: Path,
) -> Path:
    import CytoBridge as cb

    dataset = config["dataset"]
    train_config = config.get("train", {})
    config_name = options.training_config or str(train_config["config"])
    resolved = _read_training_config(config_name)
    scientific = config["scientific"]
    resolved["seed"] = int(scientific["seed"])
    resolved["ckpt_dir"] = str(model_dir)
    defaults = resolved.setdefault("training", {}).setdefault("defaults", {})
    defaults["alpha_spatial"] = float(scientific.get("alpha_spatial", 10.0))
    defaults["alpha_express"] = float(scientific["alpha_express"])
    interaction = resolved.setdefault("model", {}).setdefault("interaction_net", {})
    if options.edge_predictor_path is not None:
        interaction["edge_predictor_path"] = str(
            options.edge_predictor_path.expanduser().resolve()
        )
    threshold = options.edge_predictor_threshold
    if threshold is None:
        threshold = train_config.get("edge_predictor_threshold")
    if threshold is not None:
        interaction["edge_predictor_thre"] = float(threshold)
    cutoff = options.interaction_cutoff
    if cutoff is None:
        cutoff = train_config.get("interaction_cutoff")
    if cutoff is not None:
        interaction["cutoff"] = float(cutoff)
    cb.tl.fit(
        str(aligned_h5ad),
        config=resolved,
        device=options.device,
        time_key=str(dataset.get("time_key", "time_point_processed")),
        obsm_key=str(dataset.get("obsm_key", "X_latent")),
        is_spatial=(
            True
            if dataset.get("concat_spatial", True) is None
            else bool(dataset.get("concat_spatial", True))
        ),
        spatial_key=str(dataset.get("spatial_key", "spatial_aligned")),
        ckpt_dir=model_dir,
        interaction_cutoff=None if cutoff is None else float(cutoff),
        edge_predictor_path=(
            None
            if options.edge_predictor_path is None
            else str(options.edge_predictor_path.expanduser().resolve())
        ),
        edge_predictor_threshold=(None if threshold is None else float(threshold)),
        evaluate_after_training=bool(
            train_config.get("evaluate_after_training", False)
        ),
    )
    return model_dir


def _safe_time_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "minus_").replace(".", "p")


def _write_velocity_outputs(
    *,
    cb,
    adata,
    model,
    dataset: Mapping[str, Any],
    annotation_key: str,
    label_to_color: Mapping[str, str],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Evaluate the fitted vector field independently within each observed slice."""

    import numpy as np

    time_key = dataset.get("time_key")
    obsm_key = str(dataset.get("obsm_key", "X_latent"))
    spatial_key = str(dataset.get("spatial_key", "spatial_aligned"))
    concat_spatial = dataset.get("concat_spatial", True)
    interaction_net = getattr(model, "interaction_net", None)
    interaction_cutoff = float(getattr(interaction_net, "cutoff", 1000.0))
    components = cb.tl.compute_velocity_components_from_adata(
        adata,
        model,
        interaction_threshold=interaction_cutoff,
        device=device,
        time_key=time_key,
        obsm_key=obsm_key,
        spatial_key=spatial_key,
        concat_spatial=concat_spatial,
        write_to_adata=False,
        reuse_if_present=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "velocity_components.npz"
    np.savez_compressed(archive_path, **components)

    figures: list[str] = []
    use_spatial = (
        bool(concat_spatial)
        if concat_spatial is not None
        else spatial_key in adata.obsm
    )
    spatial_dim = (
        int(np.asarray(adata.obsm[spatial_key]).shape[1])
        if use_spatial and spatial_key in adata.obsm
        else 0
    )
    if spatial_dim >= 2:
        labels = adata.obs[annotation_key].astype(str).to_numpy()
        for time_value in sorted(np.unique(components["times"]).astype(float)):
            mask = np.isclose(components["times"], time_value)
            coords = components["features"][mask, :2]
            labels_at_time = labels[mask]
            for component_name, title in (
                ("drift", "Intrinsic velocity"),
                ("interaction", "Interaction velocity"),
                ("full", "Full velocity"),
            ):
                figure_path = output_dir / (
                    f"{component_name}_time_{_safe_time_name(time_value)}.pdf"
                )
                cb.pl.plot_velocity_component(
                    coords=coords,
                    velocity=components[component_name][mask],
                    feature_matrix=components["features"][mask],
                    labels=labels_at_time,
                    label_to_color=dict(label_to_color),
                    title=f"{title} (t={time_value:g})",
                    out_path=str(figure_path),
                    show_legend=False,
                )
                figures.append(str(figure_path))
    return {
        "status": "completed",
        "component_archive": str(archive_path),
        "interaction_cutoff": interaction_cutoff,
        "figures": figures,
    }


def _write_growth_outputs(
    *,
    cb,
    result,
    model,
    annotation_key: str,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Export per-cell growth values and the shared manuscript grid."""

    if "growth" not in set(getattr(model, "components", [])):
        return {"status": "not applicable", "reason": "model has no growth component"}

    output_dir.mkdir(parents=True, exist_ok=True)
    table = cb.tl.evaluate_growth_by_timepoint(
        result.communication_adata_dict,
        model,
        time_points=result.ts_points,
        time_keys=result.time_keys,
        annotation_key=annotation_key,
        spatial_key="spatial",
        value_key="growth_rate",
        device=device,
    )
    for key in result.time_keys:
        result.adata_dict[key].obs["growth_rate"] = (
            result.communication_adata_dict[key].obs["growth_rate"].to_numpy()
        )
    table_path = output_dir / "growth_by_cell.csv"
    table.to_csv(table_path, index=False)
    figure_path = output_dir / "growth_timepoint_grid.pdf"
    cb.pl.plot_growth_timepoint_grid(
        result.adata_dict,
        time_points=result.ts_points,
        time_keys=result.time_keys,
        out_path=str(figure_path),
        value_key="growth_rate",
        spatial_key="spatial",
        scale_mode="per_time_0_1",
        shared_colorbar=True,
        colorbar_label="Growth rate (per-time robust scale)",
        title="Growth across observed and generated slices",
    )
    return {
        "status": "completed",
        "state_source": "unwarped model state",
        "table": str(table_path),
        "figure": str(figure_path),
    }


def _write_composition_outputs(
    *,
    cb,
    result,
    annotation_key: str,
    label_to_color: Mapping[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    """Summarize real labels at observed slices and classifier labels elsewhere."""

    labels_by_time = [
        result.communication_adata_dict[key].obs[annotation_key].astype(str).to_numpy()
        for key in result.time_keys
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    table = cb.tl.summarize_label_composition(labels_by_time, result.ts_points)
    table_path = output_dir / "celltype_composition.csv"
    table.to_csv(table_path, index=False)
    figure_path = output_dir / "celltype_composition.pdf"
    cb.pl.plot_celltype_composition(
        table,
        out_path=figure_path,
        label_to_color=dict(label_to_color),
        title="Cell-type composition across time",
    )
    return {
        "status": "completed",
        "table": str(table_path),
        "figure": str(figure_path),
    }


def _write_communication_outputs(
    *,
    cb,
    result,
    runtime,
    annotation_key: str,
    output_dir: Path,
    device: str,
    downstream: Mapping[str, Any],
    seed: int,
) -> tuple[dict[str, Any], Mapping[str, Mapping[str, Any]]]:
    """Compute sparse edge attention and export readable type-to-type tables."""

    import pandas as pd

    communication_config = dict(downstream.get("communication", {}))
    output_dir.mkdir(parents=True, exist_ok=True)
    communications = cb.tl.compute_timepoint_communications(
        adata_dict=result.communication_adata_dict,
        time_points=result.ts_points,
        annotation_key=annotation_key,
        f_net=runtime.f_net,
        device=device,
        out_dir=str(output_dir / "sparse_attention"),
        save_dense_attention_matrix=False,
        remove_self_loop=bool(communication_config.get("remove_self_loop", False)),
        winsor_quantile=float(communication_config.get("winsor_quantile", 0.995)),
        max_cells_per_timepoint=communication_config.get("max_cells_per_timepoint"),
        random_seed=seed,
    )
    rows: list[dict[str, Any]] = []
    for time_value in result.ts_points:
        record = communications[str(time_value)]
        types = [str(value) for value in record["types"]]
        matrix = record["M_per_source"]
        for source_index, source in enumerate(types):
            for target_index, target in enumerate(types):
                rows.append(
                    {
                        "time": float(time_value),
                        "source": source,
                        "target": target,
                        "attention_per_source": float(
                            matrix[source_index, target_index]
                        ),
                    }
                )
    table_path = output_dir / "communication_by_celltype.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    return (
        {
            "status": "completed",
            "representation": "sparse model-edge attention",
            "remove_self_loop": bool(
                communication_config.get("remove_self_loop", False)
            ),
            "winsor_quantile": float(
                communication_config.get("winsor_quantile", 0.995)
            ),
            "table": str(table_path),
            "attention_directory": str(output_dir / "sparse_attention"),
        },
        communications,
    )


def _write_standard_figures(
    *,
    cb,
    result,
    communications,
    annotation_key: str,
    label_to_color: Mapping[str, str],
    output_dir: Path,
    lineage_enabled: bool,
) -> dict[str, Any]:
    """Render the shared paper-style snapshot, lineage, and 3D outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "spatial_snapshots"
    cb.tl.save_timepoint_snapshots(
        adata_dict=result.adata_dict,
        time_keys=result.time_keys,
        annotation_key=annotation_key,
        label_to_color=dict(label_to_color),
        snapshot_dir=str(snapshot_dir),
        background_color="white",
        font_color="#1a1a1a",
        snapshot_point_size=2.5,
        snapshot_alpha=0.9,
        mosaic_cols=4,
        mosaic_cell_size=2.2,
        mosaic_show_title=True,
        save_pdf=True,
    )
    outputs: dict[str, Any] = {
        "status": "completed",
        "snapshots": str(snapshot_dir),
    }

    lineage_labels = result.predicted_labels_list if lineage_enabled else None
    if lineage_labels is None:
        outputs["lineage"] = {
            "status": "not applicable",
            "reason": (
                "persistent fixed-particle lineage was not enabled for this dataset"
            ),
        }
    else:
        lineage_path = output_dir / "lineage_sankey.html"
        cb.tl.plot_lineage_sankey(
            predicted_labels_list=lineage_labels,
            time_keys=result.time_keys,
            label_to_color=dict(label_to_color),
            out_html=str(lineage_path),
            style="nature-methods",
            title="Cell Fate Transitions",
        )
        outputs["lineage"] = {"status": "completed", "figure": str(lineage_path)}

    plot_3d_path = output_dir / "spatiotemporal_communication_3d.html"
    cb.tl.plot_spatiotemporal_3d(
        adata_dict=result.adata_dict,
        all_time_communications=communications,
        time_keys=result.plot_3d_time_keys,
        plot_time_points=result.plot_3d_ts_points,
        ts_points=result.ts_points,
        observed_time_points=result.observed_time_points,
        interp_points=result.interp_points,
        annotation_key=annotation_key,
        label_to_color=dict(label_to_color),
        out_html=str(plot_3d_path),
        predicted_labels_list=lineage_labels,
        ribbon_render_mode="line" if lineage_labels is not None else "none",
        background_color="white",
        font_color="black",
        show_slice_border=True,
    )
    outputs["spatiotemporal_3d"] = {
        "status": "completed",
        "figure": str(plot_3d_path),
        "lineage_ribbons": "included" if lineage_labels is not None else "omitted",
    }
    return outputs


def _require_pca_reference(reference_adata) -> None:
    """Require the exact inverse-PCA transform used by gene and LR analyses."""

    if "PCs" not in reference_adata.varm:
        raise KeyError("Reference H5AD must contain exact PCA loadings in varm['PCs'].")
    if "pca_center" not in reference_adata.var:
        raise KeyError(
            "Reference H5AD must contain the fitted PCA center in var['pca_center']."
        )


def _write_gene_dynamics_outputs(
    *,
    cb,
    result,
    reference_adata,
    spatial_dim: int,
    preferred_species_tag: str | None,
    output_dir: Path,
    downstream: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct temporal gene programs from the retained PCA transform."""

    _require_pca_reference(reference_adata)
    gene_config = dict(downstream.get("gene_dynamics", {}))
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_result = cb.tl.summarize_temporal_gene_patterns(
        result.communication_adata_dict,
        reference_adata,
        time_points=result.ts_points,
        spatial_dim=spatial_dim,
        n_top_genes=int(gene_config.get("n_top_genes", 250)),
        n_cluster_genes=gene_config.get("n_cluster_genes"),
        n_clusters=int(gene_config.get("n_clusters", 2)),
        preferred_species_tag=preferred_species_tag,
        clip_min=gene_config.get("clip_min", 0.0),
    )
    gene_result.expression.to_csv(output_dir / "mean_expression.csv")
    gene_result.signed_expression.to_csv(output_dir / "signed_mean_expression.csv")
    gene_result.reconstruction_diagnostics.to_csv(
        output_dir / "reconstruction_diagnostics.csv", index=False
    )
    gene_result.top_variable_genes.to_csv(
        output_dir / "top_variable_genes.csv", index=False
    )
    gene_result.clustering.prototypes.to_csv(
        output_dir / "cluster_prototypes.csv", index=False
    )
    figure_path = output_dir / "temporal_gene_programs.pdf"
    cb.pl.plot_temporal_gene_heatmap(
        gene_result.expression,
        gene_result.top_variable_genes,
        out_path=figure_path,
        top_n=int(gene_config.get("plot_top_n", 60)),
        panel_columns=int(gene_config.get("panel_columns", 1)),
    )
    return {
        "status": "completed",
        "pca_loadings_key": "varm['PCs']",
        "pca_center_key": "var['pca_center']",
        "clip_min": gene_config.get("clip_min", 0.0),
        "preferred_species_tag": preferred_species_tag,
        "expression": str(output_dir / "mean_expression.csv"),
        "top_variable_genes": str(output_dir / "top_variable_genes.csv"),
        "figure": str(figure_path),
    }


def _write_lr_outputs(
    *,
    cb,
    result,
    reference_adata,
    communications,
    lr_database: Path,
    lr_complex_mode: str,
    preferred_species_tag: str | None,
    annotation_key: str,
    resolved_time_key: str,
    spatial_dim: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Project sparse communication through a strict, fully supported LR database."""

    _require_pca_reference(reference_adata)
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_result = cb.tl.project_communication_to_lr_timecourses(
        result.communication_adata_dict,
        reference_adata,
        communications,
        lr_database,
        time_points=result.ts_points,
        annotation_key=annotation_key,
        spatial_dim=spatial_dim,
        complex_mode=lr_complex_mode,
        expression_space="log1p",
        require_all_subunits=True,
        preferred_species_tag=preferred_species_tag,
        observed_adata=reference_adata,
        observed_time_key=resolved_time_key,
        observed_time_points=result.observed_time_points,
        observed_annotation_key=annotation_key,
        observed_expression_space="log1p",
    )
    tables = {
        "pair_timecourse": lr_result.pair_timecourse,
        "celltype_timecourse": lr_result.celltype_timecourse,
        "pattern_summary": lr_result.pattern_summary,
        "coverage": lr_result.coverage,
        "trajectory_coverage": lr_result.trajectory_coverage,
        "dropped_trajectories": lr_result.dropped_trajectories,
    }
    paths = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)
    return {
        "status": "completed",
        "database": str(lr_database),
        "complex_mode": lr_complex_mode,
        "require_all_subunits": True,
        "pca_loadings_key": "varm['PCs']",
        "pca_center_key": "var['pca_center']",
        "preferred_species_tag": preferred_species_tag,
        "tables": paths,
    }


def _write_reconstruction_diagnostic(
    *,
    cb,
    result,
    dataframe,
    feature_columns: list[str],
    spatial_dim: int,
    dataset_name: str,
    output_dir: Path,
    downstream: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare fitted-model reconstructions to observed slices without a holdout claim."""

    import numpy as np
    import pandas as pd

    quantitative_trajectory = getattr(result, "sde_points_split_prewarp", None)
    if quantitative_trajectory is None:
        quantitative_trajectory = result.sde_points_split
    if quantitative_trajectory is None:
        raise ValueError("Reconstruction diagnostics require split-SDE predictions.")
    observed_times = list(result.observed_time_points)
    if len(observed_times) < 2:
        raise ValueError(
            "Reconstruction diagnostics require at least two observed time points."
        )
    first_time = float(observed_times[0])
    train = dataframe[np.isclose(dataframe["samples"], first_time)][
        feature_columns
    ].to_numpy()
    train_spatial = train[:, :spatial_dim] if spatial_dim else None
    train_state = train[:, spatial_dim:] if spatial_dim else train
    transform = cb.tl.fit_frozen_benchmark_transform(train_state, train_spatial)
    diagnostic_config = dict(downstream.get("reconstruction_diagnostic", {}))
    rows = []
    for time_value in observed_times[1:]:
        observed = dataframe[np.isclose(dataframe["samples"], float(time_value))][
            feature_columns
        ].to_numpy()
        predicted = np.asarray(
            quantitative_trajectory[result.ts_points.index(float(time_value))],
            dtype=float,
        )
        metrics = cb.tl.evaluate_spatiotemporal_prediction(
            transform=transform,
            benchmark=f"{dataset_name}_fitted_reconstruction",
            split=f"observed_time_{float(time_value):g}",
            method="CytoBridge fitted model",
            predicted_state=(predicted[:, spatial_dim:] if spatial_dim else predicted),
            observed_state=(observed[:, spatial_dim:] if spatial_dim else observed),
            predicted_spatial=(predicted[:, :spatial_dim] if spatial_dim else None),
            observed_spatial=(observed[:, :spatial_dim] if spatial_dim else None),
            n_projections=int(diagnostic_config.get("n_projections", 256)),
            projection_repeats=int(diagnostic_config.get("projection_repeats", 5)),
            max_ot_points=int(diagnostic_config.get("max_ot_points", 1024)),
        )
        rows.append(metrics.drop(columns=["projection_sha256"]))
    table = pd.concat(rows, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "fitted_model_reconstruction_metrics.csv"
    table.to_csv(table_path, index=False)
    return {
        "status": "completed",
        "claim": "fitted-model reconstruction diagnostic; not a training holdout benchmark",
        "normalization_fit_time": first_time,
        "table": str(table_path),
    }


def _run_downstream(
    config: Mapping[str, Any],
    options: WorkflowOptions,
    *,
    aligned_h5ad: Path,
    model_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    import anndata as ad
    import CytoBridge as cb

    dataset = config["dataset"]
    scientific = config["scientific"]
    downstream = config["downstream"]
    adata = ad.read_h5ad(aligned_h5ad)
    annotation_key = str(dataset.get("annotation_key", "Annotation"))
    dataframe, resolved_time_key = cb.tl.adata_to_aligned_dataframe(
        adata,
        time_key=dataset.get("time_key"),
        obsm_key=str(dataset.get("obsm_key", "X_latent")),
        spatial_key=str(dataset.get("spatial_key", "spatial_aligned")),
        concat_spatial=dataset.get("concat_spatial", True),
        annotation_key=annotation_key,
    )
    feature_columns = cb.tl.infer_feature_columns(
        dataframe,
        annotation_column=annotation_key,
    )
    model_format = options.model_format or str(
        downstream.get("model_format", "current")
    )
    if model_format == "legacy":
        loaded = cb.tl.load_legacy_dynamical_model_from_dir(
            model_dir,
            device=options.device,
            edge_predictor_root=options.edge_predictor_root,
        )
    else:
        loaded = cb.tl.load_dynamical_model_from_dir(
            model_dir,
            dim=len(feature_columns),
            device=options.device,
            edge_predictor_path=options.edge_predictor_path,
        )
    runtime = cb.tl.build_dynamical_runtime(loaded)
    observed = [float(value) for value in downstream["observed"]]
    interpolated = [float(value) for value in downstream.get("interpolated", [])]
    time_points = sorted(set(observed + interpolated))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = cb.tl.run_interpolation_workflow(
        df=dataframe,
        dim=len(feature_columns),
        annotation_key=annotation_key,
        runtime=runtime,
        device=options.device,
        output_dir=str(output_dir),
        requested_plot_points=time_points,
        interp_time_points=interpolated,
        max_observed_timepoints=len(observed),
        classifier_cache_dir=str(output_dir / "classifier_cache"),
        classifier_adata=adata,
        classifier_time_key=resolved_time_key,
        classifier_obsm_key=str(dataset.get("obsm_key", "X_latent")),
        classifier_spatial_key=str(dataset.get("spatial_key", "spatial_aligned")),
        classifier_concat_spatial=dataset.get("concat_spatial", True),
        classifier_epochs=int(downstream.get("classifier_epochs", 500)),
        classifier_hidden_size=int(downstream.get("classifier_hidden_size", 128)),
        classifier_lr=float(downstream.get("classifier_lr", 1e-3)),
        classifier_best_metric=str(downstream.get("classifier_best_metric", "bacc")),
        classifier_strict_stratification=bool(
            downstream.get("classifier_strict_stratification", True)
        ),
        classifier_knn_neighbors=int(scientific["classifier_k"]),
        sde_n_samples=(
            None
            if downstream.get("sde_n_samples") is None
            else int(downstream["sde_n_samples"])
        ),
        skip_nonsplit_sde=not bool(downstream.get("lineage_enabled", False)),
        sde_dt=float(downstream.get("sde_dt", 0.05)),
        split_sde_dt=float(downstream.get("split_sde_dt", 0.01)),
        split_sigma_scalar=float(downstream.get("split_sigma", 0.03)),
        split_growth_alpha=float(downstream.get("split_growth_alpha", 1.0)),
        spatial_warp_to_observed_piecewise=False,
        spatial_warp_visualization_only=True,
        random_seed=int(scientific["seed"]),
    )

    snapshot_dir = output_dir / "slice_data"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for time_value, time_key in zip(result.ts_points, result.time_keys):
        path = snapshot_dir / f"time_{_safe_time_name(time_value)}.h5ad"
        result.adata_dict[time_key].write_h5ad(path)
        snapshots.append(str(path))

    labels = adata.obs[annotation_key].astype(str).to_numpy()
    label_to_color = cb.tl.load_label_to_color(
        labels,
        color_h5ad=str(aligned_h5ad),
        annotation_key=annotation_key,
    )
    (output_dir / "label_to_color.json").write_text(
        json.dumps(label_to_color, indent=2),
        encoding="utf-8",
    )

    analyses: dict[str, Any] = {}
    analyses["velocity"] = _write_velocity_outputs(
        cb=cb,
        adata=adata,
        model=loaded.model,
        dataset=dataset,
        annotation_key=annotation_key,
        label_to_color=label_to_color,
        output_dir=output_dir / "velocity",
        device=options.device,
    )
    analyses["growth"] = _write_growth_outputs(
        cb=cb,
        result=result,
        model=loaded.model,
        annotation_key=annotation_key,
        output_dir=output_dir / "growth",
        device=options.device,
    )
    analyses["composition"] = _write_composition_outputs(
        cb=cb,
        result=result,
        annotation_key=annotation_key,
        label_to_color=label_to_color,
        output_dir=output_dir / "composition",
    )
    communication_summary, communications = _write_communication_outputs(
        cb=cb,
        result=result,
        runtime=runtime,
        annotation_key=annotation_key,
        output_dir=output_dir / "communication",
        device=options.device,
        downstream=downstream,
        seed=int(scientific["seed"]),
    )
    analyses["communication"] = communication_summary
    analyses["figures"] = _write_standard_figures(
        cb=cb,
        result=result,
        communications=communications,
        annotation_key=annotation_key,
        label_to_color=label_to_color,
        output_dir=output_dir / "figures",
        lineage_enabled=bool(downstream.get("lineage_enabled", False)),
    )

    spatial_key = str(dataset.get("spatial_key", "spatial_aligned"))
    concat_spatial = dataset.get("concat_spatial", True)
    use_spatial = (
        bool(concat_spatial)
        if concat_spatial is not None
        else spatial_key in adata.obsm
    )
    spatial_dim = (
        int(adata.obsm[spatial_key].shape[1])
        if use_spatial and spatial_key in adata.obsm
        else 0
    )
    reference_adata = None
    if options.gene_dynamics or options.lr_database is not None:
        reference_adata = (
            adata
            if options.reference_h5ad is None
            else ad.read_h5ad(options.reference_h5ad.expanduser())
        )
    if options.gene_dynamics:
        analyses["gene_dynamics"] = _write_gene_dynamics_outputs(
            cb=cb,
            result=result,
            reference_adata=reference_adata,
            spatial_dim=spatial_dim,
            preferred_species_tag=options.preferred_species_tag,
            output_dir=output_dir / "gene_dynamics",
            downstream=downstream,
        )
    else:
        analyses["gene_dynamics"] = {"status": "not requested"}

    if options.lr_database is not None:
        analyses["ligand_receptor"] = _write_lr_outputs(
            cb=cb,
            result=result,
            reference_adata=reference_adata,
            communications=communications,
            lr_database=options.lr_database.expanduser().resolve(),
            lr_complex_mode=options.lr_complex_mode,
            preferred_species_tag=options.preferred_species_tag,
            annotation_key=annotation_key,
            resolved_time_key=resolved_time_key,
            spatial_dim=spatial_dim,
            output_dir=output_dir / "ligand_receptor",
        )
    else:
        analyses["ligand_receptor"] = {"status": "not requested"}

    if options.reconstruction_diagnostic:
        analyses["reconstruction_diagnostic"] = _write_reconstruction_diagnostic(
            cb=cb,
            result=result,
            dataframe=dataframe,
            feature_columns=list(feature_columns),
            spatial_dim=spatial_dim,
            dataset_name=str(dataset["name"]),
            output_dir=output_dir / "reconstruction_diagnostic",
            downstream=downstream,
        )
    else:
        analyses["reconstruction_diagnostic"] = {
            "status": "not requested",
            "claim": "not a training holdout or cross-method benchmark",
        }
    summary = {
        "dataset": dataset["name"],
        "seed": int(scientific["seed"]),
        "alpha_spatial": float(scientific.get("alpha_spatial", 10.0)),
        "alpha_express": float(scientific["alpha_express"]),
        "classifier_k": int(scientific["classifier_k"]),
        "reference_h5ad": str(options.reference_h5ad or aligned_h5ad),
        "model": {
            "format": model_format,
            "directory": str(model_dir),
            "weight_stage": loaded.weight_stage,
            "score_stage": loaded.score_stage,
        },
        "time_points": [float(value) for value in result.ts_points],
        "simulation": {
            "initial_particles": int(
                result.communication_adata_dict[result.time_keys[0]].n_obs
            ),
            "configured_particle_cap": downstream.get("sde_n_samples"),
            "split_dt": float(downstream.get("split_sde_dt", 0.01)),
            "sigma": float(downstream.get("split_sigma", 0.03)),
            "growth_alpha": float(downstream.get("split_growth_alpha", 1.0)),
            "non_split_lineage_rollout": bool(downstream.get("lineage_enabled", False)),
        },
        "classifier_accuracy": result.classifier_accuracy,
        "classifier_balanced_accuracy": result.classifier_balanced_accuracy,
        "snapshots": snapshots,
        "analyses": analyses,
    }
    summary_path = output_dir / "summary.json"
    summary["summary_file"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def run_workflow(
    config: Mapping[str, Any],
    *,
    options: WorkflowOptions,
) -> dict[str, Any]:
    """Execute selected steps after the caller has checked the dry-run plan."""

    selected = _selected_steps(config, options)
    paths = _output_paths(config, options)
    output_dir = paths["output_dir"]
    aligned_h5ad = paths["aligned_h5ad"]
    model_dir = paths["model_dir"]
    completed: list[str] = []
    outputs: dict[str, Any] = {}

    if "preprocess" in selected and config.get("preprocess", {}).get("enabled", True):
        assert aligned_h5ad is not None
        aligned_h5ad.parent.mkdir(parents=True, exist_ok=True)
        aligned_h5ad = _run_preprocess(config, options, aligned_h5ad=aligned_h5ad)
        completed.append("preprocess")
        outputs["aligned_h5ad"] = str(aligned_h5ad)

    if options.train:
        assert aligned_h5ad is not None and model_dir is not None
        model_dir.mkdir(parents=True, exist_ok=True)
        model_dir = _run_train(
            config,
            options,
            aligned_h5ad=aligned_h5ad,
            model_dir=model_dir,
        )
        completed.append("train")
        outputs["model_dir"] = str(model_dir)

    if "downstream" in selected and config.get("downstream", {}).get("enabled", True):
        assert (
            aligned_h5ad is not None
            and model_dir is not None
            and output_dir is not None
        )
        downstream_summary = _run_downstream(
            config,
            options,
            aligned_h5ad=aligned_h5ad,
            model_dir=model_dir,
            output_dir=output_dir / "downstream",
        )
        completed.append("downstream")
        outputs["downstream"] = downstream_summary

    return {"completed": completed, "outputs": outputs}
