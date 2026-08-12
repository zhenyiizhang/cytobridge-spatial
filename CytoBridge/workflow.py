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
    edge_predictor_path: Path | None = None
    edge_predictor_threshold: float | None = None
    edge_predictor_root: Path | None = None
    device: str = "cuda"
    model_format: str | None = None
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
            _parse_config_text(resolved.read_text(encoding="utf-8"), source=str(resolved)),
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


def _selected_steps(config: Mapping[str, Any], options: WorkflowOptions) -> tuple[str, ...]:
    if options.steps:
        selected = list(options.steps)
    else:
        selected = list(config.get("steps", {}).get("default", ("preprocess", "downstream")))
    if options.train and "train" not in selected:
        insert_at = selected.index("downstream") if "downstream" in selected else len(selected)
        selected.insert(insert_at, "train")
    return tuple(dict.fromkeys(str(step) for step in selected))


def _output_paths(config: Mapping[str, Any], options: WorkflowOptions) -> dict[str, Path | None]:
    dataset_name = str(config["dataset"]["name"])
    output_dir = options.output_dir
    aligned = options.aligned_h5ad
    if aligned is None and output_dir is not None and "preprocess" in _selected_steps(config, options):
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
        steps.append({"name": "preprocess", "status": "skipped", "compute": "GPU for spatial alignment"})
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
                "output": None if paths["aligned_h5ad"] is None else str(paths["aligned_h5ad"]),
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
        if train_config.get("requires_edge_predictor", False) and options.edge_predictor_path is None:
            missing.append("--edge-predictor-path")
        steps.append(
            {
                "name": "train",
                "status": "ready" if not missing else "missing input",
                "compute": "GPU required for production training",
                "missing": missing,
                "training_config": training_preset,
                "output": None if paths["model_dir"] is None else str(paths["model_dir"]),
            }
        )

    if "downstream" not in selected:
        steps.append({"name": "downstream", "status": "skipped", "compute": "GPU recommended"})
    elif not downstream_config.get("enabled", True):
        steps.append(
            {
                "name": "downstream",
                "status": "unavailable in this preset",
                "compute": "not scheduled",
                "note": downstream_config.get("note", "No downstream recipe is defined."),
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
        steps.append(
            {
                "name": "downstream",
                "status": "ready" if not missing else "missing input",
                "compute": "GPU recommended for SDE simulation and classifier fitting",
                "missing": missing,
                "model_format": options.model_format or downstream_config.get("model_format", "current"),
                "output": (
                    None
                    if options.output_dir is None
                    else str(options.output_dir / "downstream")
                ),
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
        if step.get("model_format"):
            lines.append(f"    model format: {step['model_format']}")
        if step.get("output"):
            lines.append(f"    output: {step['output']}")
        if step.get("note"):
            lines.append(f"    note: {step['note']}")
    return "\n".join(lines)


def plan_missing_inputs(plan: Mapping[str, Any]) -> list[str]:
    """Return selected steps that cannot run because an input was not supplied."""

    return [
        f"{step['name']}: {', '.join(step.get('missing', []))}"
        for step in plan["steps"]
        if step.get("status") == "missing input"
    ]


def _read_training_config(config_name: str) -> dict[str, Any]:
    path = Path(config_name).expanduser()
    if path.is_file():
        return _parse_config_text(path.read_text(encoding="utf-8"), source=str(path))
    filename = config_name if str(config_name).endswith(".yaml") else f"{config_name}.yaml"
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
    if "spatial_obs_keys" in align_values and align_values["spatial_obs_keys"] is not None:
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
        adata.obs[annotation_key] = adata.obs[str(annotation_source)].astype(str).to_numpy()
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

    train_config = config.get("train", {})
    config_name = options.training_config or str(train_config["config"])
    resolved = _read_training_config(config_name)
    scientific = config["scientific"]
    resolved["seed"] = int(scientific["seed"])
    resolved["ckpt_dir"] = str(model_dir)
    defaults = resolved.setdefault("training", {}).setdefault("defaults", {})
    defaults["alpha_express"] = float(scientific["alpha_express"])
    interaction = resolved.setdefault("model", {}).setdefault("interaction_net", {})
    if options.edge_predictor_path is not None:
        interaction["edge_predictor_path"] = str(options.edge_predictor_path.expanduser().resolve())
    threshold = options.edge_predictor_threshold
    if threshold is None:
        threshold = train_config.get("edge_predictor_threshold")
    if threshold is not None:
        interaction["edge_predictor_thre"] = float(threshold)
    cb.tl.fit(
        str(aligned_h5ad),
        config=resolved,
        device=options.device,
        ckpt_dir=model_dir,
        evaluate_after_training=bool(train_config.get("evaluate_after_training", False)),
    )
    return model_dir


def _safe_time_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "minus_").replace(".", "p")


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
    model_format = options.model_format or str(downstream.get("model_format", "current"))
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
        sde_n_samples=int(downstream.get("sde_n_samples", 5000)),
        sde_dt=float(downstream.get("sde_dt", 0.05)),
        split_sde_dt=float(downstream.get("split_sde_dt", 0.01)),
        split_sigma_scalar=float(downstream.get("split_sigma", 0.03)),
        split_growth_alpha=float(downstream.get("split_growth_alpha", 1.0)),
        spatial_warp_to_observed_piecewise=False,
        spatial_warp_visualization_only=True,
        random_seed=int(scientific["seed"]),
    )

    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for time_value, time_key in zip(result.ts_points, result.time_keys):
        path = snapshot_dir / f"time_{_safe_time_name(time_value)}.h5ad"
        result.adata_dict[time_key].write_h5ad(path)
        snapshots.append(str(path))
    summary = {
        "dataset": dataset["name"],
        "seed": int(scientific["seed"]),
        "alpha_express": float(scientific["alpha_express"]),
        "classifier_k": int(scientific["classifier_k"]),
        "time_points": [float(value) for value in result.ts_points],
        "classifier_accuracy": result.classifier_accuracy,
        "classifier_balanced_accuracy": result.classifier_balanced_accuracy,
        "snapshots": snapshots,
    }
    (output_dir / "summary.json").write_text(
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
        assert aligned_h5ad is not None and model_dir is not None and output_dir is not None
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
