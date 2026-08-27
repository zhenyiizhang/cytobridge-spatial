"""Package-native execution of the standard CytoBridge workflow.

The command-line workflow deliberately stays small.  Dataset presets describe
scientific parameters and dataset schema, while numerical work is delegated to
the public preprocessing, training, and downstream APIs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from importlib import resources
import json
import math
from numbers import Real
from pathlib import Path
import shutil
from typing import Any, Mapping

from .graph_database import (
    FORMAL_GRAPH_DATABASES,
    bundled_graph_database_path,
    match_graph_database_features,
    resolve_graph_database,
)


WORKFLOW_PRESETS = ("zebrafish", "mosta", "arista", "admouse", "chicken_heart")
_PRESET_ALIASES = {
    "ad": "admouse",
    "ad-mouse": "admouse",
    "zfish": "zebrafish",
    "chicken-heart": "chicken_heart",
    "chicken": "chicken_heart",
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
    graph_database: Path | None = None
    edge_predictor_path: Path | None = None
    edge_predictor_threshold: float | None = None
    edge_predictor_root: Path | None = None
    device: str = "cuda"
    model_format: str | None = None
    reference_h5ad: Path | None = None
    gene_dynamics: bool = False
    skip_gene_dynamics: bool = False
    lr_database: Path | None = None
    skip_lr: bool = False
    lr_complex_mode: str = "min"
    preferred_species_tag: str | None = None
    reconstruction_diagnostic: bool = False
    allow_complete_reference_pca_center_fallback: bool = False
    steps: tuple[str, ...] = ()
    train: bool = False


def _resolve_workflow_options(
    config: Mapping[str, Any], options: WorkflowOptions
) -> WorkflowOptions:
    """Resolve cross-step options once before planning or execution."""

    species_tag = options.preferred_species_tag
    if species_tag is None:
        species_tag = config.get("downstream", {}).get("preferred_species_tag")
    if species_tag is not None:
        species_tag = str(species_tag).strip() or None
    if species_tag == options.preferred_species_tag:
        return options
    return replace(options, preferred_species_tag=species_tag)


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
    if options.steps and "train" in options.steps and not options.train:
        raise ValueError(
            "Selecting --step train also requires the explicit --train flag."
        )
    if options.steps:
        selected = list(options.steps)
    else:
        selected = list(config.get("steps", {}).get("default", ("downstream",)))
        # A de novo training command starts from raw data unless the caller
        # explicitly supplies an aligned H5AD. Artifact reuse stays downstream
        # only, so newly fitted PCA coordinates can never be paired silently
        # with an unrelated existing checkpoint.
        if (
            options.train
            and options.aligned_h5ad is None
            and config.get("preprocess", {}).get("enabled", True)
            and "preprocess" not in selected
        ):
            selected.insert(0, "preprocess")
    if options.train and "train" not in selected:
        insert_at = (
            selected.index("downstream") if "downstream" in selected else len(selected)
        )
        selected.insert(insert_at, "train")
    if options.train:
        _validate_training_edge_prior_options(config, options)
    if (
        options.train
        and "preprocess" in selected
        and config.get("preprocess", {}).get("enabled", True)
        and options.edge_predictor_path is not None
    ):
        raise ValueError(
            "A preprocess+train run must fit a new edge predictor from the newly "
            "aligned PCA features and interaction graphs. Do not pass "
            "--edge-predictor-path for a raw-data run. Reuse an existing edge "
            "predictor only with its matched --aligned-h5ad and without the "
            "preprocess step."
        )
    if (
        options.train
        and "preprocess" in selected
        and options.training_config is not None
    ):
        expected_cutoff = (
            _read_training_config(options.training_config)
            .get("model", {})
            .get("interaction_net", {})
            .get("cutoff")
        )
        preset_cutoff = config.get("train", {}).get("interaction_cutoff")
        if (
            options.interaction_cutoff is None
            and expected_cutoff is not None
            and preset_cutoff is not None
            and not math.isclose(
                float(expected_cutoff),
                float(preset_cutoff),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "A custom training config changes the interaction cutoff while "
                "preprocessing is selected. Supply the matching workflow preset or "
                "an explicit --interaction-cutoff so graph construction, negative "
                "sampling, training, and downstream use one cutoff."
            )
    if not options.train and {"preprocess", "downstream"}.issubset(selected):
        raise ValueError(
            "Preprocessing and downstream inference cannot share one command "
            "without --train: a newly fitted PCA/alignment must not be paired "
            "with an existing checkpoint. Run preprocessing alone, add --train "
            "for a de novo workflow, or run downstream from matched artifacts."
        )
    return tuple(dict.fromkeys(str(step) for step in selected))


def _validate_training_edge_prior_options(
    config: Mapping[str, Any],
    options: WorkflowOptions,
) -> dict[str, Any]:
    """Reject predictor settings that cannot affect the selected training prior."""

    train_config = config.get("train", {})
    config_name = options.training_config or str(train_config["config"])
    resolved = _read_training_config(config_name)
    model = resolved.get("model", {})
    interaction = model.get("interaction_net", {})
    mode = str(interaction.get("edge_prior_mode", "learned")).strip().lower()
    if mode not in {"learned", "all_spatial"}:
        raise ValueError(
            "Training config model.interaction_net.edge_prior_mode must be "
            f"'learned' or 'all_spatial'; got {mode!r}."
        )
    uses_interaction = "interaction" in model.get("components", [])
    uses_gnn_interaction = bool(
        uses_interaction
        and str(model.get("interaction_type", "potential")).lower() == "gnn"
    )
    uses_learned = bool(uses_gnn_interaction and mode == "learned")
    if not uses_interaction and options.interaction_cutoff is not None:
        raise ValueError(
            "The selected training model has no interaction component, so "
            "--interaction-cutoff cannot affect it. Remove the inert option."
        )
    if not uses_learned and options.graph_database is not None:
        raise ValueError(
            "The selected training model does not construct a learned LR edge "
            "prior, so --graph-database cannot affect it. Remove the inert option."
        )
    if not uses_learned:
        stale_config_values = {
            key: interaction.get(key)
            for key in ("edge_predictor_path", "edge_predictor_thre")
            if interaction.get(key) is not None
        }
        supplied = {
            "--edge-predictor-path": options.edge_predictor_path,
            "--edge-predictor-threshold": options.edge_predictor_threshold,
        }
        supplied = {key: value for key, value in supplied.items() if value is not None}
        if stale_config_values or supplied:
            raise ValueError(
                f"edge_prior_mode={mode!r} does not use an edge predictor. Remove "
                "predictor path/threshold settings instead of recording ignored "
                f"scientific parameters; config={stale_config_values}, runtime={supplied}."
            )
    return resolved


def _training_uses_learned_edge_prior(
    config: Mapping[str, Any],
    options: WorkflowOptions,
) -> bool:
    """Return whether the effective training config needs a learned edge gate."""

    if options.training_config is None:
        # This fast path keeps dependency-free installed-wheel dry-runs usable.
        # Execution and train planning call _validate_builtin_training_contract,
        # which parses the YAML and fails closed on any duplicated-field drift.
        return bool(config.get("train", {}).get("requires_edge_predictor", False))
    resolved = _read_training_config(options.training_config)
    model = resolved.get("model", {})
    uses_learned = bool(
        "interaction" in model.get("components", [])
        and str(model.get("interaction_type", "potential")).lower() == "gnn"
        and str(
            model.get("interaction_net", {}).get("edge_prior_mode", "learned")
        ).lower()
        == "learned"
    )
    return uses_learned


def _validate_builtin_training_contract(config: Mapping[str, Any]) -> None:
    """Check matching fields in the workflow JSON and training YAML."""

    train_config = config.get("train", {})
    resolved = _read_training_config(str(train_config["config"]))
    model = resolved.get("model", {})
    interaction = model.get("interaction_net", {})
    yaml_mode = str(interaction.get("edge_prior_mode", "learned")).lower()
    yaml_learned = bool(
        "interaction" in model.get("components", [])
        and str(model.get("interaction_type", "potential")).lower() == "gnn"
        and yaml_mode == "learned"
    )
    json_learned = bool(train_config.get("requires_edge_predictor", False))
    if json_learned != yaml_learned:
        raise ValueError(
            "Workflow preset train.requires_edge_predictor does not match its "
            f"training YAML edge-prior mode: JSON={json_learned}, YAML={yaml_mode!r}."
        )
    pairs = (
        (
            "interaction cutoff",
            train_config.get("interaction_cutoff"),
            interaction.get("cutoff"),
        ),
        (
            "edge predictor threshold",
            train_config.get("edge_predictor_threshold"),
            interaction.get("edge_predictor_thre"),
        ),
    )
    for label, json_value, yaml_value in pairs:
        if label == "edge predictor threshold" and not yaml_learned:
            if json_value is None and yaml_value is None:
                continue
        if (
            label == "edge predictor threshold"
            and yaml_learned
            and config.get("preprocess", {}).get("enabled", True)
            and json_value is None
            and yaml_value is None
        ):
            # A de novo learned prior selects its operating threshold on the
            # freshly fitted predictor validation split.  Presets with a frozen
            # historical model record that model's threshold instead.
            continue
        if (
            json_value is None
            or yaml_value is None
            or not math.isclose(
                float(json_value), float(yaml_value), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError(
                f"Workflow preset {label} does not match its training YAML: "
                f"JSON={json_value!r}, YAML={yaml_value!r}."
            )


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
    edge_predictor = options.edge_predictor_path
    if (
        edge_predictor is None
        and output_dir is not None
        and options.train
        and "preprocess" in _selected_steps(config, options)
        and config.get("preprocess", {}).get("enabled", True)
        and _training_uses_learned_edge_prior(config, options)
    ):
        edge_predictor = (
            output_dir
            / "preprocess"
            / "edge_classifier"
            / f"{dataset_name}_edge_model.pt"
        )
    return {
        "output_dir": output_dir,
        "aligned_h5ad": aligned,
        "model_dir": model_dir,
        "edge_predictor_path": edge_predictor,
    }


def _effective_downstream_analyses(
    config: Mapping[str, Any],
    options: WorkflowOptions,
) -> dict[str, Any]:
    """Resolve optional downstream analyses after preset and CLI overrides.

    Packaged presets can make gene dynamics and strict LR projection part of
    their standard workflow.  Existing command-line flags remain explicit
    overrides, and the LR default deliberately reuses the same species-matched
    database as interaction-graph construction.
    """

    downstream = config.get("downstream", {})
    gene_default = bool(downstream.get("gene_dynamics_enabled", False))
    gene_dynamics = bool(
        not options.skip_gene_dynamics and (options.gene_dynamics or gene_default)
    )
    gene_source = None
    if options.gene_dynamics:
        gene_source = "explicit --gene-dynamics"
    elif gene_default:
        gene_source = "packaged preset default"

    lr_default = bool(downstream.get("lr_enabled", False))
    lr_enabled = bool(
        not options.skip_lr and (options.lr_database is not None or lr_default)
    )
    lr_database = None if options.skip_lr else options.lr_database
    lr_source = None
    if lr_enabled and options.lr_database is not None:
        lr_source = "explicit --lr-database override"
    elif lr_enabled and lr_default:
        lr_database = bundled_graph_database_path(
            str(config["dataset"]["name"]),
            filename=config.get("train", {}).get("graph_database"),
        )
        lr_source = (
            "bundled species-matched CellChatDB used for graph construction and "
            "downstream strict LR projection"
            if _training_uses_learned_edge_prior(config, options)
            else "bundled species-matched CellChatDB used only for downstream "
            "strict LR projection"
        )

    preferred_species_tag = options.preferred_species_tag
    if preferred_species_tag is None:
        configured_tag = downstream.get("preferred_species_tag")
        preferred_species_tag = None if configured_tag is None else str(configured_tag)

    return {
        "gene_dynamics": gene_dynamics,
        "gene_dynamics_source": gene_source,
        "lr_enabled": lr_enabled,
        "lr_database": lr_database,
        "lr_database_source": lr_source,
        "preferred_species_tag": preferred_species_tag,
    }


_OPERATIONAL_TRAINING_CONFIG_PATHS = frozenset(
    {
        ("ckpt_dir",),
        ("checkpoint_dir",),
        ("device",),
        ("log_dir",),
        ("output_dir",),
        ("spatial_dim",),
        ("training", "history_flush_every"),
        ("model", "interaction_net", "edge_predictor_path"),
        ("model", "interaction_net", "edge_predictor_model_path"),
        ("model", "interaction_net", "load_edge_predictor_from_path"),
        ("model", "spatial_dim"),
    }
)
_OPERATIONAL_TRAINING_CONFIG_KEYS = frozenset(
    {
        "checkpoint_dir",
        "ckpt_dir",
        "device",
        "history_flush_every",
        "log_dir",
        "output_dir",
    }
)
_OMIT_CONFIG_VALUE = object()
_MISSING_CONFIG_VALUE = object()


def _scientific_training_config(
    value: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> Any:
    """Remove only runtime locations and reporting controls from a config."""

    string_path = tuple(str(part) for part in path)
    if string_path in _OPERATIONAL_TRAINING_CONFIG_PATHS or (
        string_path and string_path[-1] in _OPERATIONAL_TRAINING_CONFIG_KEYS
    ):
        return _OMIT_CONFIG_VALUE
    if isinstance(value, Mapping):
        projected = {}
        for key, child in value.items():
            child_value = _scientific_training_config(
                child,
                path=(*path, str(key)),
            )
            if child_value is not _OMIT_CONFIG_VALUE:
                projected[str(key)] = child_value
        return projected
    if isinstance(value, (list, tuple)):
        return [
            _scientific_training_config(child, path=(*path, index))
            for index, child in enumerate(value)
        ]
    return value


def _config_path_text(path: tuple[str | int, ...]) -> str:
    text = ""
    for part in path:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += ("." if text else "") + part
    return text or "<root>"


def _config_values_match(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return (
            isinstance(actual, bool)
            and isinstance(expected, bool)
            and actual == expected
        )
    if isinstance(actual, Real) and isinstance(expected, Real):
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    if (
        isinstance(actual, str)
        and isinstance(expected, Real)
        and not isinstance(expected, bool)
    ):
        try:
            numeric_actual = float(actual)
        except ValueError:
            return False
        return math.isfinite(numeric_actual) and math.isclose(
            numeric_actual,
            float(expected),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    if (
        isinstance(expected, str)
        and isinstance(actual, Real)
        and not isinstance(actual, bool)
    ):
        try:
            numeric_expected = float(expected)
        except ValueError:
            return False
        return math.isfinite(numeric_expected) and math.isclose(
            float(actual),
            numeric_expected,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    return type(actual) is type(expected) and actual == expected


def _scientific_config_mismatches(
    actual: Any,
    expected: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    """Return every structural or scalar difference with its readable path."""

    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        differences = []
        for key in sorted(set(actual).union(expected)):
            differences.extend(
                _scientific_config_mismatches(
                    actual.get(key, _MISSING_CONFIG_VALUE),
                    expected.get(key, _MISSING_CONFIG_VALUE),
                    path=(*path, str(key)),
                )
            )
        return differences
    if isinstance(actual, list) and isinstance(expected, list):
        differences = []
        for index in range(max(len(actual), len(expected))):
            actual_value = (
                actual[index] if index < len(actual) else _MISSING_CONFIG_VALUE
            )
            expected_value = (
                expected[index] if index < len(expected) else _MISSING_CONFIG_VALUE
            )
            differences.extend(
                _scientific_config_mismatches(
                    actual_value,
                    expected_value,
                    path=(*path, index),
                )
            )
        return differences
    if (
        actual is _MISSING_CONFIG_VALUE
        or expected is _MISSING_CONFIG_VALUE
        or not _config_values_match(actual, expected)
    ):
        return [(path, actual, expected)]
    return []


def _display_config_value(value: Any) -> str:
    return "<missing>" if value is _MISSING_CONFIG_VALUE else repr(value)


def _loaded_model_scientific_contract(
    loaded,
    *,
    config: Mapping[str, Any],
    options: WorkflowOptions,
) -> dict[str, Any]:
    """Verify that a loaded checkpoint matches the requested scientific run.

    This check is intentionally about interpretable model parameters, not file
    identity. It prevents an existing baseline checkpoint from being labelled
    as an alpha=0.015 run merely because a workflow preset requested 0.015.
    """

    loaded_config = loaded.config
    if "legacy" in loaded_config:
        raise ValueError(
            "Legacy params.yml does not record the resolved alpha=0.015 "
            "six-stage contract. Use a dedicated historical compatibility "
            "comparison; do not label a legacy checkpoint as a formal packaged run."
        )
    source = deepcopy(loaded_config)
    expected_config = deepcopy(
        _read_training_config(options.training_config or str(config["train"]["config"]))
    )
    requested_scientific = config["scientific"]
    expected_config["seed"] = int(requested_scientific["seed"])
    expected_defaults = expected_config.setdefault("training", {}).setdefault(
        "defaults", {}
    )
    expected_defaults["alpha_express"] = float(requested_scientific["alpha_express"])
    expected_defaults["alpha_spatial"] = float(
        requested_scientific.get("alpha_spatial", 10.0)
    )
    expected_model = expected_config.setdefault("model", {})
    actual_model = source.get("model", {})
    expected_components = {
        str(value).strip().lower() for value in expected_model.get("components", [])
    }
    actual_components = {
        str(value).strip().lower() for value in actual_model.get("components", [])
    }
    expected_has_interaction = "interaction" in expected_components
    actual_has_interaction = "interaction" in actual_components
    if expected_has_interaction:
        expected_interaction = expected_model.get("interaction_net")
        if not isinstance(expected_interaction, dict) or not expected_interaction:
            raise ValueError(
                "Requested training config includes the interaction component but "
                "does not define model.interaction_net."
            )
        actual_interaction = actual_model.get("interaction_net")
        if not isinstance(actual_interaction, dict) or not actual_interaction:
            raise ValueError(
                "Loaded model includes the interaction component but does not "
                "record model.interaction_net."
            )
    else:
        if "interaction_net" in expected_model:
            raise ValueError(
                "Requested no-interaction training config contains an inert "
                "model.interaction_net section; remove it."
            )
        if "interaction_net" in actual_model:
            raise ValueError(
                "Loaded no-interaction model contains an inert "
                "model.interaction_net section; it is not a clean matched ablation."
            )
        if (
            options.interaction_cutoff is not None
            or options.edge_predictor_path is not None
            or options.edge_predictor_threshold is not None
            or options.graph_database is not None
        ):
            raise ValueError(
                "A no-interaction checkpoint does not accept interaction cutoff, "
                "edge-predictor, or graph-database overrides."
            )
        expected_interaction = {}
        actual_interaction = {}

    # Current GNN configs spell out the learned prior explicitly. Checkpoints
    # made before that field was added have identical runtime semantics because
    # ``learned`` has always been the GNN constructor default. Normalize only
    # that historical omission; do not invent an interaction config for models
    # that do not have one or weaken any other scientific field.
    if (
        expected_has_interaction
        and actual_has_interaction
        and isinstance(actual_interaction, dict)
        and actual_interaction
        and "edge_prior_mode" not in actual_interaction
        and str(expected_interaction.get("edge_prior_mode", "learned")).lower()
        == "learned"
    ):
        actual_interaction["edge_prior_mode"] = "learned"

    requested_cutoff = None
    if expected_has_interaction:
        requested_cutoff = options.interaction_cutoff
        if requested_cutoff is None and options.training_config is None:
            requested_cutoff = config.get("train", {}).get("interaction_cutoff")
        if requested_cutoff is not None:
            expected_interaction["cutoff"] = float(requested_cutoff)

    expected_edge_mode = (
        str(expected_interaction.get("edge_prior_mode", "learned")).strip().lower()
        if expected_has_interaction
        else "none"
    )
    actual_edge_mode = (
        str(actual_interaction.get("edge_prior_mode", "learned")).strip().lower()
        if actual_has_interaction
        else "none"
    )
    if actual_edge_mode != expected_edge_mode:
        raise ValueError(
            "Loaded model edge-prior mode does not match the requested training "
            f"config: loaded={actual_edge_mode!r}, expected={expected_edge_mode!r}."
        )
    if expected_edge_mode != "learned":
        stale_actual = {
            key: actual_interaction.get(key)
            for key in ("edge_predictor_path", "edge_predictor_thre")
            if actual_interaction.get(key) is not None
        }
        if (
            options.edge_predictor_path is not None
            or options.edge_predictor_threshold is not None
        ):
            raise ValueError(
                f"edge_prior_mode={expected_edge_mode!r} does not accept predictor "
                "path or threshold overrides."
            )
        if stale_actual:
            raise ValueError(
                f"Loaded {expected_edge_mode!r} model records predictor settings that "
                f"cannot affect its edge prior: {stale_actual}."
            )

    requested_threshold = options.edge_predictor_threshold
    threshold_source = None
    if requested_threshold is not None:
        threshold_source = "explicit workflow override"
    elif expected_edge_mode == "learned":
        # A de novo run selects this value on its edge-predictor validation
        # split, then records the effective value in the checkpoint config.
        # That recorded value is the scientific contract for later standalone
        # downstream runs; the packaged preset threshold describes only the
        # historical released artifact.
        requested_threshold = actual_interaction.get("edge_predictor_thre")
        if requested_threshold is not None:
            threshold_source = "loaded checkpoint recorded effective threshold"
    if requested_threshold is not None:
        requested_threshold = float(requested_threshold)
        if not math.isfinite(requested_threshold) or not 0 < requested_threshold < 1:
            raise ValueError(
                "Loaded model edge predictor threshold must lie strictly between "
                f"0 and 1; got {requested_threshold!r}."
            )
        expected_interaction["edge_predictor_thre"] = requested_threshold

    actual_spatial_dims = {
        "spatial_dim": source.get("spatial_dim"),
        "model.spatial_dim": source.get("model", {}).get("spatial_dim"),
    }
    recorded_spatial_dims = {
        int(value) for value in actual_spatial_dims.values() if value is not None
    }
    if len(recorded_spatial_dims) > 1:
        raise ValueError(
            "Loaded model records conflicting derived spatial dimensions: "
            + ", ".join(
                f"{path}={value!r}" for path, value in actual_spatial_dims.items()
            )
        )
    actual_spatial_dim = (
        next(iter(recorded_spatial_dims)) if recorded_spatial_dims else 2
    )
    expected_spatial_dim = expected_config.get("model", {}).get("spatial_dim")
    if expected_spatial_dim is None:
        uses_spatial = config.get("dataset", {}).get("concat_spatial", True)
        if uses_spatial:
            expected_spatial_dim = (
                config.get("preprocess", {}).get("align", {}).get("spatial_dim")
            )
        elif uses_spatial is False:
            expected_spatial_dim = 0
    if actual_spatial_dim is not None and expected_spatial_dim is not None:
        if int(actual_spatial_dim) != int(expected_spatial_dim):
            raise ValueError(
                "Loaded model derived spatial dimension does not match the "
                "requested workflow: "
                f"loaded model.spatial_dim={actual_spatial_dim!r}; "
                f"expected model.spatial_dim={int(expected_spatial_dim)!r}."
            )

    configured_stage_names = [
        str(stage.get("name"))
        for stage in source.get("training", {}).get("plan", [])
        if stage.get("name")
    ]
    expected_plan = expected_config.get("training", {}).get("plan", [])
    expected_stage_names = [
        str(stage.get("name")) for stage in expected_plan if stage.get("name")
    ]
    if configured_stage_names != expected_stage_names:
        raise ValueError(
            "Loaded model training stages do not match the requested packaged "
            f"plan: loaded={configured_stage_names}, expected={expected_stage_names}."
        )

    actual_scientific_config = _scientific_training_config(source)
    expected_scientific_config = _scientific_training_config(expected_config)
    mismatches = _scientific_config_mismatches(
        actual_scientific_config,
        expected_scientific_config,
    )
    if mismatches:
        shown = mismatches[:12]
        lines = [
            "Loaded model scientific config does not match the requested training config:"
        ]
        for path, actual, expected in shown:
            name = _config_path_text(path)
            lines.append(
                f"- loaded {name}={_display_config_value(actual)}; "
                f"expected {name}={_display_config_value(expected)}"
            )
        if len(mismatches) > len(shown):
            lines.append(f"- ... and {len(mismatches) - len(shown)} more differences")
        raise ValueError(
            "\n".join(lines)
            + "\nOnly runtime path, output, device, and history-flush fields are ignored."
        )

    expected_weight_stages = [
        str(stage["name"])
        for stage in expected_plan
        if stage.get("name")
        and str(stage.get("mode", "")).lower() != "score_matching"
        and str(stage.get("train_strategy", "")).lower() != "s"
    ]
    expected_weight_stage = (
        expected_weight_stages[-1] if expected_weight_stages else None
    )
    if expected_weight_stage and str(loaded.weight_stage) != expected_weight_stage:
        raise ValueError(
            f"Loaded model used weight stage {loaded.weight_stage!r}; "
            f"the requested training config requires {expected_weight_stage!r}."
        )
    expected_score_stages = [
        str(stage["name"])
        for stage in expected_plan
        if stage.get("name")
        if str(stage.get("mode", "")).lower() == "score_matching"
        or str(stage.get("train_strategy", "")).lower() == "s"
    ]
    if expected_score_stages and str(loaded.score_stage) != expected_score_stages[-1]:
        raise ValueError(
            f"Loaded score stage {loaded.score_stage!r} does not match the formal "
            f"final score stage {expected_score_stages[-1]!r}."
        )

    defaults = source.get("training", {}).get("defaults", {})
    interaction = source.get("model", {}).get("interaction_net", {})
    alpha_express = float(defaults["alpha_express"])
    alpha_spatial = float(defaults["alpha_spatial"])
    seed = int(source["seed"])
    cutoff = interaction.get("cutoff") if actual_has_interaction else None
    threshold = (
        interaction.get("edge_predictor_thre")
        if str(interaction.get("edge_prior_mode", "learned")).lower() == "learned"
        else None
    )
    interaction_group_size = (
        int(source.get("model", {}).get("interaction_group_size", 1024))
        if actual_has_interaction
        else None
    )
    return {
        "status": "matches requested preset",
        "alpha_express": alpha_express,
        "alpha_spatial": alpha_spatial,
        "seed": int(seed),
        "interaction_cutoff": None if cutoff is None else float(cutoff),
        "components": sorted(actual_components),
        "interaction_component": bool(actual_has_interaction),
        "edge_prior_mode": (
            str(interaction.get("edge_prior_mode", "learned")).lower()
            if actual_has_interaction
            else "none"
        ),
        "interaction_group_size": interaction_group_size,
        "interaction_group_max_size": (
            None if interaction_group_size is None else 2 * interaction_group_size - 1
        ),
        "interaction_group_remainder_policy": (
            None
            if interaction_group_size is None
            else "merge a nonzero remainder into the final base-size group"
        ),
        "edge_predictor_threshold": (None if threshold is None else float(threshold)),
        "edge_predictor_threshold_check": threshold_source,
        "weight_stage": loaded.weight_stage,
        "score_stage": loaded.score_stage,
    }


def _split_trajectory_contract(downstream: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit estimand contract for split-SDE outputs."""

    piecewise = bool(downstream.get("split_sde_piecewise", False))
    if not piecewise:
        return {
            "trajectory_mode": "global_t0_extrapolation",
            "split_sde_piecewise": False,
            "piecewise_observed_sample_mode": None,
            "piecewise_include_end": None,
            "trajectory_scope": (
                "Generated states belong to one split-SDE rollout initialized from t0; "
                "later states are global extrapolations rather than independently "
                "observed-anchored interval interpolations."
            ),
        }

    configured_sample_mode = downstream.get("piecewise_observed_sample_mode")
    sample_mode = (
        "t0_fixed" if configured_sample_mode is None else str(configured_sample_mode)
    )
    if sample_mode not in {"t0_fixed", "per_timepoint"}:
        raise ValueError(
            "piecewise_observed_sample_mode must be 't0_fixed' or "
            f"'per_timepoint'; got {sample_mode!r}."
        )
    include_end = bool(downstream.get("split_sde_piecewise_include_end", False))
    return {
        "trajectory_mode": ("piecewise_observed_anchored_interval_forward_simulation"),
        "split_sde_piecewise": True,
        "piecewise_observed_sample_mode": sample_mode,
        "piecewise_include_end": include_end,
        "trajectory_scope": (
            "Observed times use real cells. Each generated time is a one-sided, "
            "interval-local forward simulation initialized only from the "
            "immediately preceding observed anchor; it is not conditioned on the "
            "following observed endpoint. It is not global-t0 extrapolation and "
            "not a lineage-continuous population trajectory."
        ),
    }


def build_workflow_plan(
    config: Mapping[str, Any],
    *,
    source: str,
    options: WorkflowOptions,
) -> dict[str, Any]:
    """Build the concise execution plan shown by ``--dry-run``."""

    options = _resolve_workflow_options(config, options)
    if options.train and options.training_config is None:
        _validate_builtin_training_contract(config)
    selected = _selected_steps(config, options)
    paths = _output_paths(config, options)
    dataset = config["dataset"]
    scientific = config["scientific"]
    downstream_config = config.get("downstream", {})
    trajectory = _split_trajectory_contract(downstream_config)
    preprocess_config = config.get("preprocess", {})
    train_config = config.get("train", {})
    downstream_analyses = _effective_downstream_analyses(config, options)

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
        effective_learned_prior = _training_uses_learned_edge_prior(config, options)
        auto_edge_predictor = (
            effective_learned_prior
            and options.train
            and options.edge_predictor_path is None
            and "preprocess" in selected
            and preprocess_config.get("enabled", True)
        )
        bundled_database = None
        if auto_edge_predictor and options.graph_database is None:
            bundled_database = bundled_graph_database_path(
                str(dataset["name"]),
                filename=train_config.get("graph_database"),
            )
        if (
            auto_edge_predictor
            and options.graph_database is not None
            and not options.graph_database.expanduser().is_file()
        ):
            missing.append(f"graph database not found: {options.graph_database}")
        steps.append(
            {
                "name": "preprocess",
                "status": "ready" if not missing else "missing input",
                "compute": (
                    "CPU validation; fixed anatomy-reviewed alignment"
                    if preprocess_config.get("mode") == "fixed_aligned_input"
                    else "GPU recommended for spatial alignment"
                ),
                "missing": missing,
                "output": None
                if paths["aligned_h5ad"] is None
                else str(paths["aligned_h5ad"]),
                "edge_predictor": (
                    {
                        "status": "will be trained automatically",
                        "graph_database": (
                            str(options.graph_database)
                            if options.graph_database is not None
                            else str(bundled_database)
                        ),
                        "database_source": (
                            "custom --graph-database override"
                            if options.graph_database is not None
                            else "bundled formal CellChatDB resource"
                        ),
                        "interaction_cutoff": (
                            float(options.interaction_cutoff)
                            if options.interaction_cutoff is not None
                            else train_config.get("interaction_cutoff")
                        ),
                        "decision_threshold": (
                            float(options.edge_predictor_threshold)
                            if options.edge_predictor_threshold is not None
                            else None
                        ),
                        "decision_threshold_source": (
                            "explicit --edge-predictor-threshold"
                            if options.edge_predictor_threshold is not None
                            else "validation-selected during de novo training"
                        ),
                        "output": None
                        if paths["edge_predictor_path"] is None
                        else str(paths["edge_predictor_path"]),
                    }
                    if auto_edge_predictor
                    else {
                        "status": "not requested during preprocessing",
                    }
                ),
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
        effective_training = _read_training_config(
            options.training_config or str(train_config["config"])
        )
        effective_model = effective_training.get("model", {})
        effective_components = {
            str(value).strip().lower()
            for value in effective_model.get("components", [])
        }
        effective_has_interaction = "interaction" in effective_components
        effective_interaction = effective_model.get("interaction_net", {})
        planned_interaction_cutoff = (
            None
            if not effective_has_interaction
            else float(options.interaction_cutoff)
            if options.interaction_cutoff is not None
            else effective_interaction.get("cutoff")
            if options.training_config is not None
            else train_config.get("interaction_cutoff")
        )
        effective_learned_prior = _training_uses_learned_edge_prior(config, options)
        auto_edge_predictor = (
            effective_learned_prior
            and options.edge_predictor_path is None
            and "preprocess" in selected
            and preprocess_config.get("enabled", True)
        )
        if effective_learned_prior and not (
            options.edge_predictor_path is not None or auto_edge_predictor
        ):
            missing.append("--edge-predictor-path")
        if (
            options.edge_predictor_path is not None
            and not options.edge_predictor_path.expanduser().is_file()
        ):
            missing.append(f"edge predictor not found: {options.edge_predictor_path}")
        planned_edge_threshold = None
        if options.edge_predictor_threshold is not None:
            planned_edge_threshold = float(options.edge_predictor_threshold)
            threshold_source = "explicit --edge-predictor-threshold"
        elif auto_edge_predictor:
            threshold_source = "validation-selected during preprocessing"
        elif effective_learned_prior:
            planned_edge_threshold = effective_interaction.get(
                "edge_predictor_thre",
                train_config.get("edge_predictor_threshold"),
            )
            threshold_source = "packaged corrected artifact threshold"
        else:
            threshold_source = None
        steps.append(
            {
                "name": "train",
                "status": "ready" if not missing else "missing input",
                "compute": "GPU required for production training",
                "missing": missing,
                "training_config": training_preset,
                "interaction_cutoff": planned_interaction_cutoff,
                "edge_predictor_threshold": planned_edge_threshold,
                "edge_predictor_threshold_source": threshold_source,
                "edge_predictor_path": None
                if paths["edge_predictor_path"] is None
                else str(paths["edge_predictor_path"]),
                "edge_predictor_source": (
                    "explicit --edge-predictor-path"
                    if options.edge_predictor_path is not None
                    else "generated by preprocessing"
                    if auto_edge_predictor
                    else None
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
        if options.training_config is None:
            # Built-in dependency-free dry-runs use the packaged workflow JSON
            # as their lightweight planning contract. Training plans and real
            # execution still call _validate_builtin_training_contract above or
            # in run_workflow, so duplicated JSON/YAML science cannot drift.
            planned_has_interaction = bool(
                train_config.get("requires_edge_predictor", False)
            )
        else:
            planned_training = _read_training_config(options.training_config)
            planned_components = {
                str(value).strip().lower()
                for value in planned_training.get("model", {}).get("components", [])
            }
            planned_has_interaction = "interaction" in planned_components
        if not planned_has_interaction and options.lr_database is not None:
            raise ValueError(
                "--lr-database cannot affect a no-interaction downstream run: "
                "ligand-receptor projection requires model-derived communication."
            )
        analyses = [
            {
                "name": "interpolation and classification",
                "status": "enabled",
            },
            {"name": "time-slice velocity", "status": "enabled"},
            {"name": "growth", "status": "enabled when present in the model"},
            {"name": "cell-type composition", "status": "enabled"},
            {
                "name": "sparse communication",
                "status": "enabled" if planned_has_interaction else "not applicable",
                "note": (
                    None
                    if planned_has_interaction
                    else "selected checkpoint config has no interaction component"
                ),
            },
            {
                "name": "standard figures",
                "status": "enabled",
                "note": (
                    "snapshots, mosaic, growth, composition, and velocity; 3D "
                    "communication only when the model has an interaction component; "
                    "lineage only with an explicit persistent-ID contract"
                ),
            },
            {
                "name": "gene dynamics",
                "status": (
                    "enabled"
                    if downstream_analyses["gene_dynamics"]
                    else "not requested"
                ),
                "source": downstream_analyses["gene_dynamics_source"],
                "preferred_species_tag": downstream_analyses["preferred_species_tag"],
                "note": (
                    "requires PCA loadings in varm['PCs']; new preprocessing "
                    "saves var['pca_center']; for an older complete PCA-fit "
                    "reference without that field, use "
                    "--allow-complete-reference-pca-center-fallback; the inferred "
                    "center must reproduce the saved PCA coordinates"
                ),
            },
            {
                "name": "strict ligand-receptor projection",
                "status": (
                    "enabled"
                    if downstream_analyses["lr_enabled"] and planned_has_interaction
                    else "not applicable"
                    if not planned_has_interaction
                    else "not requested"
                ),
                "database": (
                    None
                    if not planned_has_interaction
                    else None
                    if downstream_analyses["lr_database"] is None
                    else str(downstream_analyses["lr_database"])
                ),
                "source": (
                    downstream_analyses["lr_database_source"]
                    if planned_has_interaction
                    else None
                ),
                "preferred_species_tag": (
                    downstream_analyses["preferred_species_tag"]
                    if planned_has_interaction
                    else None
                ),
                "missing": (
                    [
                        "LR database file not found: "
                        f"{downstream_analyses['lr_database']}"
                    ]
                    if planned_has_interaction
                    and downstream_analyses["lr_database"] is not None
                    and not downstream_analyses["lr_database"].expanduser().is_file()
                    else []
                ),
                "note": (
                    "model has no interaction component"
                    if not planned_has_interaction
                    else "uses all required complex subunits and the selected "
                    "min/geometric-mean rule; requires PCA loadings and the "
                    "matching complete reference H5AD"
                ),
            },
            {
                "name": "fitted-model reconstruction diagnostic",
                "status": (
                    "not applicable to observed-anchored interval simulation"
                    if options.reconstruction_diagnostic
                    and trajectory["split_sde_piecewise"]
                    else "requested"
                    if options.reconstruction_diagnostic
                    else "not requested"
                ),
                "note": (
                    "observed endpoints are interval anchors, so comparing them "
                    "to themselves is not a fitted global-t0 reconstruction "
                    "diagnostic"
                    if trajectory["split_sde_piecewise"]
                    else "descriptive W2 comparison using the fitted data"
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
                    "split_resample_dt": (
                        None
                        if downstream_config.get("split_resample_dt") is None
                        else float(downstream_config["split_resample_dt"])
                    ),
                    "split_max_particles": (
                        None
                        if downstream_config.get("split_max_particles") is None
                        else int(downstream_config["split_max_particles"])
                    ),
                    "sigma": float(downstream_config.get("split_sigma", 0.03)),
                    "daughter_noise_std": float(
                        downstream_config.get("split_daughter_noise_std", 0.0)
                    ),
                    "growth_alpha": float(
                        downstream_config.get("split_growth_alpha", 1.0)
                    ),
                    "trajectory_mode": trajectory["trajectory_mode"],
                    "piecewise_observed_sample_mode": trajectory[
                        "piecewise_observed_sample_mode"
                    ],
                    "piecewise_include_end": trajectory["piecewise_include_end"],
                    "trajectory_scope": trajectory["trajectory_scope"],
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

    def display_path(value: object) -> str:
        text = str(value)
        parts = Path(text).parts
        try:
            package_index = parts.index("CytoBridge")
        except ValueError:
            return text
        if parts[package_index + 1 : package_index + 2] == ("workflow_databases",):
            return "package: " + "/".join(parts[package_index:])
        return text

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
        if step.get("edge_predictor_threshold_source"):
            lines.append(
                "    edge predictor threshold source: "
                f"{step['edge_predictor_threshold_source']}"
            )
        if step.get("edge_predictor_path"):
            lines.append(f"    edge predictor: {step['edge_predictor_path']}")
        if step.get("edge_predictor_source"):
            lines.append(f"    edge predictor source: {step['edge_predictor_source']}")
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
                f"daughter noise={simulation['daughter_noise_std']:g}, "
                f"growth alpha={simulation['growth_alpha']:g}, "
                f"trajectory mode={simulation['trajectory_mode']}"
            )
            if simulation["piecewise_observed_sample_mode"] is not None:
                lines.append(
                    "      piecewise observed anchors: "
                    f"sample mode={simulation['piecewise_observed_sample_mode']}, "
                    f"include end={simulation['piecewise_include_end']}"
                )
            lines.append(f"      scope: {simulation['trajectory_scope']}")
        if step.get("edge_predictor"):
            edge = step["edge_predictor"]
            lines.append(f"    edge predictor: {edge['status']}")
            if edge.get("graph_database"):
                lines.append(
                    f"      graph database: {display_path(edge['graph_database'])}"
                )
            if edge.get("database_source"):
                lines.append(f"      database source: {edge['database_source']}")
            if edge.get("interaction_cutoff") is not None:
                lines.append(f"      interaction cutoff: {edge['interaction_cutoff']}")
            if edge.get("decision_threshold") is not None:
                lines.append(f"      decision threshold: {edge['decision_threshold']}")
            if edge.get("decision_threshold_source"):
                lines.append(
                    "      decision threshold source: "
                    f"{edge['decision_threshold_source']}"
                )
            if edge.get("output"):
                lines.append(f"      output: {edge['output']}")
        for analysis in step.get("analyses", []):
            lines.append(f"    {analysis['name']}: {analysis['status']}")
            if analysis.get("database"):
                lines.append(f"      database: {display_path(analysis['database'])}")
            if analysis.get("source"):
                lines.append(f"      source: {analysis['source']}")
            if analysis.get("preferred_species_tag"):
                lines.append(
                    "      preferred species tag: "
                    f"{analysis['preferred_species_tag']}"
                )
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
    import anndata as ad

    from CytoBridge.pp import (
        AlignConfig,
        preprocess_align_to_files,
        validate_prepared_chicken_heart_input,
    )

    preprocess_config = config["preprocess"]
    mode = str(preprocess_config.get("mode", "fit_spatial_alignment"))
    if mode == "fixed_aligned_input":
        if str(config["dataset"]["name"]) != "chicken_heart":
            raise ValueError(
                "preprocess.mode='fixed_aligned_input' is currently defined only "
                "for the anatomy-reviewed chicken_heart preset."
            )
        assert options.input_h5ad is not None
        input_h5ad = options.input_h5ad.expanduser().resolve()
        prepared = ad.read_h5ad(input_h5ad)
        fixed_contract = validate_prepared_chicken_heart_input(prepared)
        if aligned_h5ad.exists() and aligned_h5ad.resolve() != input_h5ad:
            raise FileExistsError(
                f"Refusing to overwrite fixed aligned H5AD: {aligned_h5ad}."
            )
        if aligned_h5ad.resolve() != input_h5ad:
            shutil.copy2(input_h5ad, aligned_h5ad)
        print(
            "Validated fixed chicken-heart alignment: "
            f"policy={fixed_contract['coordinate_policy']}."
        )
        return aligned_h5ad
    if mode != "fit_spatial_alignment":
        raise ValueError(f"Unknown preprocess.mode={mode!r}.")

    align_values = dict(preprocess_config.get("align", {}))
    if (
        "spatial_obs_keys" in align_values
        and align_values["spatial_obs_keys"] is not None
    ):
        align_values["spatial_obs_keys"] = tuple(align_values["spatial_obs_keys"])
    if (
        "observation_id_keys" in align_values
        and align_values["observation_id_keys"] is not None
    ):
        align_values["observation_id_keys"] = tuple(align_values["observation_id_keys"])

    dataset_name = str(config["dataset"]["name"])
    train_config = config.get("train", {})
    lr_feature_coverage = None
    should_match_lr_features = bool(
        options.graph_database is not None
        or train_config.get("graph_database")
        or dataset_name in FORMAL_GRAPH_DATABASES
    )
    if should_match_lr_features:
        graph_database = resolve_graph_database(
            dataset_name,
            options.graph_database,
            bundled_filename=train_config.get("graph_database"),
        )
        raw_adata = ad.read_h5ad(str(options.input_h5ad), backed="r")
        try:
            var_names = tuple(str(name) for name in raw_adata.var_names)
        finally:
            raw_adata.file.close()
        lr_features, lr_feature_coverage = match_graph_database_features(
            graph_database,
            var_names,
            preferred_species_tag=options.preferred_species_tag,
        )
        configured_required = tuple(
            str(name) for name in (align_values.get("required_latent_features") or ())
        )
        align_values["required_latent_features"] = tuple(
            dict.fromkeys((*configured_required, *lr_features))
        )
        lr_feature_coverage["database_source"] = (
            "custom --graph-database override"
            if options.graph_database is not None
            else "bundled formal CellChatDB resource"
        )
        print(
            "LR latent-feature coverage before HVG/PCA: "
            f"matched={lr_feature_coverage['n_matched_features']}/"
            f"{lr_feature_coverage['n_unique_database_subunits']}, "
            f"missing={lr_feature_coverage['n_missing_database_subunits']}, "
            f"ambiguous_skipped="
            f"{lr_feature_coverage['n_ambiguous_database_subunits']}."
        )

    cfg = AlignConfig(**align_values)
    output_csv = aligned_h5ad.with_suffix(".csv")
    adata = preprocess_align_to_files(
        h5ad_path=str(options.input_h5ad),
        time_key=str(preprocess_config["time_key"]),
        output_csv=str(output_csv),
        output_h5ad=None,
        cfg=cfg,
        batch_indices=preprocess_config.get("batch_indices"),
        batch_values=preprocess_config.get("batch_values"),
        drop_uns_keys=preprocess_config.get("drop_uns_keys"),
        device=options.device,
    )
    if lr_feature_coverage is not None:
        preprocess_info = dict(adata.uns.get("preprocess_info", {}))
        preprocess_info["lr_latent_feature_coverage"] = lr_feature_coverage
        adata.uns["preprocess_info"] = preprocess_info
    annotation_source = preprocess_config.get("annotation_source")
    annotation_key = str(config["dataset"].get("annotation_key", "Annotation"))
    if annotation_source and annotation_source in adata.obs:
        adata.obs[annotation_key] = (
            adata.obs[str(annotation_source)].astype(str).to_numpy()
        )
    adata.write_h5ad(aligned_h5ad)
    return aligned_h5ad


def _interaction_expression_layer(adata) -> str:
    """Return the raw expression layer selected by package preprocessing."""

    preprocess_info = adata.uns.get("preprocess_info", {})
    layer = str(
        preprocess_info.get(
            "raw_counts_layer",
            preprocess_info.get("counts_layer", "counts"),
        )
    )
    if layer not in adata.layers:
        raise KeyError(
            "The raw-expression layer recorded by preprocessing is "
            f"{layer!r}, but the aligned H5AD contains {list(adata.layers)}."
        )
    return layer


def _run_edge_predictor(
    config: Mapping[str, Any],
    options: WorkflowOptions,
    *,
    aligned_h5ad: Path,
    edge_predictor_path: Path,
) -> dict[str, Any]:
    """Build per-time interaction graphs and train their shared edge predictor."""

    import anndata as ad
    import pandas as pd

    import CytoBridge as cb

    dataset = config["dataset"]
    preprocess_config = config["preprocess"]
    train_config = config["train"]
    edge_config = dict(preprocess_config.get("edge_predictor", {}))
    data_name = str(dataset["name"])
    time_key = str(dataset.get("time_key", "time_point_processed"))
    spatial_key = str(dataset.get("spatial_key", "spatial_aligned"))
    latent_key = str(dataset.get("obsm_key", "X_latent"))
    interaction_cutoff = options.interaction_cutoff
    if interaction_cutoff is None:
        interaction_cutoff = train_config.get("interaction_cutoff")
    if interaction_cutoff is None:
        raise ValueError(
            "Automatic edge prediction requires train.interaction_cutoff or "
            "--interaction-cutoff."
        )
    decision_threshold = options.edge_predictor_threshold

    adata = ad.read_h5ad(aligned_h5ad)
    if time_key not in adata.obs:
        raise KeyError(f"Aligned H5AD is missing time column {time_key!r}.")
    time_points = sorted(
        pd.to_numeric(adata.obs[time_key], errors="raise").astype(float).unique()
    )
    expression_layer = _interaction_expression_layer(adata)
    graph_database = resolve_graph_database(
        data_name,
        options.graph_database,
        bundled_filename=train_config.get("graph_database"),
    )
    preprocess_dir = aligned_h5ad.parent
    graph_input_dir = preprocess_dir / "input_graph"
    metadata_dir = preprocess_dir / "metadata"
    graph_results = []
    spot_diameter = float(edge_config.get("spot_diameter", interaction_cutoff / 4.0))
    for time_index, time_value in enumerate(time_points):
        slice_name = f"{data_name}_t{time_index}"
        graph_results.append(
            cb.pp.generate_interaction_graph(
                data_name=slice_name,
                data_from=adata,
                data_to=str(graph_input_dir / slice_name),
                metadata_to=str(metadata_dir / slice_name),
                database_path=str(graph_database),
                split=int(edge_config.get("split", 0)),
                time_key=time_key,
                time_value=float(time_value),
                neighborhood_threshold=float(interaction_cutoff),
                spot_diameter=spot_diameter,
                spatial_key=spatial_key,
                expression_layer=expression_layer,
                auto_neighborhood_threshold=False,
                save_metadata=bool(edge_config.get("save_metadata", False)),
                save_quantile_matrix=bool(
                    edge_config.get("save_quantile_matrix", False)
                ),
                verbose=bool(edge_config.get("verbose", True)),
                use_tqdm=bool(edge_config.get("use_tqdm", True)),
                preferred_species_tag=options.preferred_species_tag,
            )
        )

    edge_predictor_path.parent.mkdir(parents=True, exist_ok=True)
    edge_result = cb.pp.train_edge_predictor(
        data_name=data_name,
        adata_or_h5ad=adata,
        graph_input_dir=str(graph_input_dir),
        output_model_path=str(edge_predictor_path),
        epochs=int(edge_config.get("epochs", 100)),
        batch_size=int(edge_config.get("batch_size", 1024)),
        learning_rate=float(edge_config.get("learning_rate", 1e-3)),
        spatial_dim=int(preprocess_config.get("align", {}).get("spatial_dim", 2)),
        distance_threshold=float(interaction_cutoff),
        device=options.device,
        time_key=time_key,
        latent_key=latent_key,
        spatial_key=spatial_key,
        train_sample_ratio_per_epoch=float(
            edge_config.get("train_sample_ratio_per_epoch", 1.0)
        ),
        max_train_edges_per_epoch=edge_config.get("max_train_edges_per_epoch"),
        num_workers=int(edge_config.get("num_workers", 4)),
        random_seed=int(config["scientific"]["seed"]),
        edge_predictor_threshold=(
            None if decision_threshold is None else float(decision_threshold)
        ),
        split_strategy=str(edge_config.get("split_strategy", "node_disjoint")),
    )
    cb.pp.sanitize_interaction_graph_uns(adata)
    adata.write_h5ad(aligned_h5ad)
    return {
        "model_path": str(edge_predictor_path),
        "meta_path": str(edge_result["meta_path"]),
        "graph_input_dir": str(graph_input_dir),
        "metadata_dir": str(metadata_dir),
        "graph_database": str(graph_database),
        "graph_database_source": (
            "custom --graph-database override"
            if options.graph_database is not None
            else "bundled formal CellChatDB resource"
        ),
        "time_points": [float(value) for value in time_points],
        "graph_slices": graph_results,
        "interaction_cutoff": float(interaction_cutoff),
        "edge_predictor_threshold": float(edge_result["edge_predictor_threshold"]),
        "edge_predictor_threshold_selected": float(
            edge_result["edge_predictor_threshold_selected"]
        ),
    }


def _run_train(
    config: Mapping[str, Any],
    options: WorkflowOptions,
    *,
    aligned_h5ad: Path,
    model_dir: Path,
    edge_predictor_path: Path | None = None,
    edge_predictor_threshold: float | None = None,
) -> Path:
    import CytoBridge as cb

    dataset = config["dataset"]
    train_config = config.get("train", {})
    resolved = _validate_training_edge_prior_options(config, options)
    scientific = config["scientific"]
    resolved["seed"] = int(scientific["seed"])
    resolved["ckpt_dir"] = str(model_dir)
    defaults = resolved.setdefault("training", {}).setdefault("defaults", {})
    defaults["alpha_spatial"] = float(scientific.get("alpha_spatial", 10.0))
    defaults["alpha_express"] = float(scientific["alpha_express"])
    model = resolved.setdefault("model", {})
    components = {
        str(component).strip().lower() for component in model.get("components", [])
    }
    uses_interaction = "interaction" in components
    interaction_type = str(model.get("interaction_type", "potential")).lower()
    interaction = model.get("interaction_net")
    if uses_interaction and interaction_type == "gnn" and interaction is None:
        interaction = {}
        model["interaction_net"] = interaction
    elif interaction is None:
        interaction = {}
    edge_prior_mode = str(interaction.get("edge_prior_mode", "learned")).lower()
    uses_learned_edge_prior = bool(
        uses_interaction and interaction_type == "gnn" and edge_prior_mode == "learned"
    )
    if not uses_learned_edge_prior and (
        edge_predictor_path is not None or edge_predictor_threshold is not None
    ):
        raise ValueError(
            "The selected training model does not use a learned edge prior and "
            "cannot receive a generated edge predictor path or threshold."
        )
    effective_edge_path = None
    if uses_learned_edge_prior:
        supplied_edge_path = edge_predictor_path or options.edge_predictor_path
        if supplied_edge_path is None:
            raise ValueError(
                "Training with edge_prior_mode='learned' requires an explicit "
                "edge predictor produced by this workflow or supplied with "
                "--edge-predictor-path. The relative path stored in a packaged "
                "training YAML is a resource placeholder and is never resolved "
                "against the current working directory."
            )
        effective_edge_path = Path(supplied_edge_path).expanduser().resolve()
        if not effective_edge_path.is_file():
            raise FileNotFoundError(
                "Learned edge predictor not found at the explicit run-bound path: "
                f"{effective_edge_path}"
            )
        interaction["edge_predictor_path"] = str(effective_edge_path)
    threshold = None
    if uses_learned_edge_prior:
        threshold = edge_predictor_threshold
        if threshold is None:
            threshold = options.edge_predictor_threshold
        if threshold is None and options.training_config is None:
            threshold = train_config.get("edge_predictor_threshold")
    if threshold is not None:
        interaction["edge_predictor_thre"] = float(threshold)
    cutoff = None
    if uses_interaction:
        cutoff = options.interaction_cutoff
        if cutoff is None and options.training_config is not None:
            cutoff = interaction.get("cutoff")
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
            None if effective_edge_path is None else str(effective_edge_path)
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
    model_components = {
        str(component).strip().lower() for component in getattr(model, "components", [])
    }
    has_interaction = "interaction" in model_components
    interaction_net = getattr(model, "interaction_net", None)
    if has_interaction and interaction_net is None:
        raise ValueError(
            "Model declares an interaction component but has no interaction_net."
        )
    interaction_cutoff = (
        float(getattr(interaction_net, "cutoff")) if has_interaction else None
    )
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

    spatial_figures: list[str] = []
    gene_figures: list[str] = []
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
        spatial_coordinates = np.asarray(adata.obsm[spatial_key], dtype=float)[:, :2]
        fitted_features = np.asarray(components["features"], dtype=float)
        if fitted_features.ndim != 2 or fitted_features.shape[1] <= spatial_dim:
            raise ValueError(
                "Velocity components do not contain expression-state dimensions "
                "after the spatial coordinates."
            )
        expression_features = fitted_features[:, spatial_dim:]
        gene_dir = output_dir / "gene"
        gene_dir.mkdir(parents=True, exist_ok=True)
        for time_value in sorted(np.unique(components["times"]).astype(float)):
            mask = np.isclose(components["times"], time_value)
            coords = spatial_coordinates[mask]
            labels_at_time = labels[mask]
            plotted_components = [
                ("drift", "Intrinsic velocity"),
                ("full", "Full velocity"),
            ]
            if has_interaction:
                plotted_components.insert(1, ("interaction", "Interaction velocity"))
            for component_name, title in plotted_components:
                figure_path = output_dir / (
                    f"{component_name}_time_{_safe_time_name(time_value)}.pdf"
                )
                cb.pl.plot_velocity_component(
                    coords=coords,
                    velocity=components[component_name][mask, :2],
                    feature_matrix=None,
                    labels=labels_at_time,
                    label_to_color=dict(label_to_color),
                    title=f"{title} (spatial direct, t={time_value:g})",
                    out_path=str(figure_path),
                    show_legend=False,
                )
                spatial_figures.append(str(figure_path))

                gene_figure_path = gene_dir / (
                    f"{component_name}_time_{_safe_time_name(time_value)}.pdf"
                )
                cb.pl.plot_velocity_component(
                    coords=coords,
                    velocity=components[component_name][mask, spatial_dim:],
                    feature_matrix=expression_features[mask],
                    labels=labels_at_time,
                    label_to_color=dict(label_to_color),
                    title=f"Gene {title.lower()} (t={time_value:g})",
                    out_path=str(gene_figure_path),
                    show_legend=False,
                )
                gene_figures.append(str(gene_figure_path))
    figures = [*spatial_figures, *gene_figures]
    return {
        "status": "completed",
        "component_archive": str(archive_path),
        "interaction_cutoff": interaction_cutoff,
        "interaction_component": has_interaction,
        "interaction_vector_status": (
            "evaluated"
            if has_interaction
            else "not applicable; zero sentinel retained in the component archive"
        ),
        "projection_contract": {
            "spatial": "direct_model_spatial_components",
            "latent": "scvelo_transition_graph_to_spatial_display",
            "gene": "scvelo_expression_state_to_observed_spatial_coordinates",
            "spatial_dim": spatial_dim,
        },
        "spatial_projection_mode": "direct_model_spatial_vector",
        "gene_projection_mode": (
            "scvelo_expression_state_to_observed_spatial_coordinates"
        ),
        "expression_dimensions": (
            int(expression_features.shape[1]) if spatial_dim >= 2 else 0
        ),
        "spatial_figures": spatial_figures,
        "gene_figures": gene_figures,
        # Backward-compatible name for consumers that predate the explicit
        # gene-panel terminology.  These are the same scVelo-projected
        # expression-state figures, not an additional rendering path.
        "latent_figures": gene_figures,
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
    interaction_net = getattr(getattr(runtime, "f_net", None), "interaction_net", None)
    edge_prior_mode = str(
        getattr(interaction_net, "edge_prior_mode", "learned")
    ).lower()
    edge_selection_by_time = {
        f"{float(time_value):g}": dict(
            communications[str(time_value)]["edge_selection"]
        )
        for time_value in result.ts_points
    }
    return (
        {
            "status": "completed",
            "representation": "sparse model-edge attention",
            "edge_prior_mode": edge_prior_mode,
            "interpretation": communication_config.get("interpretation"),
            "edge_selection_by_time": edge_selection_by_time,
            "structural_zero_interpretation": (
                "When candidate edges exist, a structural zero means no candidate "
                "edge passed the LR-informed learned edge-predictor gate at that "
                "time point; when the "
                "candidate count is zero, no edge was available within the spatial "
                "cutoff. Neither case establishes absence of all biological "
                "communication."
                if edge_prior_mode == "learned"
                else "A structural zero means no edge was available within the "
                "spatial cutoff at that time point; it does not establish absence "
                "of all biological communication."
            ),
            "attention_scope": (
                "full time-slice radius candidate graph"
                if communication_config.get("max_cells_per_timepoint") is None
                else "seeded time-slice subsample radius candidate graph"
            ),
            "training_interaction_scope": (
                "stochastic interaction groups with base size "
                f"{int(getattr(runtime.f_net, 'interaction_group_size', 1024))}; "
                "a nonzero remainder is merged into the final group (at most "
                f"{2 * int(getattr(runtime.f_net, 'interaction_group_size', 1024)) - 1}); "
                "radius candidates are evaluated only within each group during "
                "model training and dynamics"
            ),
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

    if communications:
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
            "lineage_ribbons": (
                "included" if lineage_labels is not None else "omitted"
            ),
        }
    else:
        outputs["spatiotemporal_3d"] = {
            "status": "not applicable",
            "reason": (
                "model has no interaction component, so no communication graph "
                "or communication ribbons are defined"
            ),
        }
    return outputs


def _require_pca_reference(
    reference_adata,
    *,
    allow_complete_reference_pca_center_fallback: bool = False,
) -> None:
    """Require PCA loadings needed by gene and LR reconstruction.

    New preprocessing persists ``var['pca_center']``. A historical object is
    accepted only with an explicit declaration that it is the complete
    original PCA-fit reference, followed by a saved-score consistency check.
    """

    if "PCs" not in reference_adata.varm:
        raise KeyError("Reference H5AD must contain exact PCA loadings in varm['PCs'].")
    if "pca_center" not in reference_adata.var:
        from CytoBridge.tl.downstream.temporal import infer_pca_center

        infer_pca_center(
            reference_adata,
            allow_complete_reference_pca_center_fallback=(
                allow_complete_reference_pca_center_fallback
            ),
        )


def _pca_center_source(reference_adata) -> str:
    if "pca_center" in reference_adata.var:
        return "reference var['pca_center']"
    return "explicit complete-reference X column mean matching saved PCA coordinates"


def _write_gene_dynamics_outputs(
    *,
    cb,
    result,
    reference_adata,
    spatial_dim: int,
    preferred_species_tag: str | None,
    output_dir: Path,
    downstream: Mapping[str, Any],
    allow_complete_reference_pca_center_fallback: bool = False,
) -> dict[str, Any]:
    """Reconstruct temporal gene programs from the retained PCA transform."""

    _require_pca_reference(
        reference_adata,
        allow_complete_reference_pca_center_fallback=(
            allow_complete_reference_pca_center_fallback
        ),
    )
    gene_config = dict(downstream.get("gene_dynamics", {}))
    output_dir.mkdir(parents=True, exist_ok=True)
    gene_result = cb.tl.summarize_temporal_gene_patterns(
        result.communication_adata_dict,
        reference_adata,
        time_points=result.ts_points,
        spatial_dim=spatial_dim,
        allow_complete_reference_pca_center_fallback=(
            allow_complete_reference_pca_center_fallback
        ),
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
        "pca_center_source": _pca_center_source(reference_adata),
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
    allow_complete_reference_pca_center_fallback: bool = False,
    analysis_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project sparse communication through a strict, fully supported LR database."""

    _require_pca_reference(
        reference_adata,
        allow_complete_reference_pca_center_fallback=(
            allow_complete_reference_pca_center_fallback
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    lr_result = cb.tl.project_communication_to_lr_timecourses(
        result.communication_adata_dict,
        reference_adata,
        communications,
        lr_database,
        time_points=result.ts_points,
        annotation_key=annotation_key,
        spatial_dim=spatial_dim,
        allow_complete_reference_pca_center_fallback=(
            allow_complete_reference_pca_center_fallback
        ),
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
        "pca_center_source": _pca_center_source(reference_adata),
        "preferred_species_tag": preferred_species_tag,
        "analysis_scope": None if analysis_scope is None else dict(analysis_scope),
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
    """Compare fitted-model reconstructions with observed slices."""

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


def _downstream_simulation_summary(
    *,
    downstream: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    result: Any,
    interaction_group_size: int | None,
) -> dict[str, Any]:
    """Build a readable simulation record from the slices actually emitted."""

    particle_counts = {
        str(float(time_value)): int(result.adata_dict[time_key].n_obs)
        for time_value, time_key in zip(result.ts_points, result.time_keys)
    }
    slice_origins = {
        str(float(time_value)): str(
            result.adata_dict[time_key].uns.get("slice_origin", "")
        )
        for time_value, time_key in zip(result.ts_points, result.time_keys)
    }
    source_anchor_times = {
        str(float(time_value)): float(
            result.adata_dict[time_key].uns["source_anchor_time"]
        )
        for time_value, time_key in zip(result.ts_points, result.time_keys)
    }
    observed = {float(value) for value in result.observed_time_points}
    observed_counts = {
        key: count for key, count in particle_counts.items() if float(key) in observed
    }
    generated_counts = {
        key: count
        for key, count in particle_counts.items()
        if float(key) not in observed
    }
    return {
        "initial_particles": int(result.adata_dict[result.time_keys[0]].n_obs),
        "configured_particle_cap": downstream.get("sde_n_samples"),
        "initial_particle_cap": downstream.get("sde_n_samples"),
        "split_particle_ceiling": downstream.get("split_max_particles"),
        "split_dt": float(downstream.get("split_sde_dt", 0.01)),
        "split_resample_dt": downstream.get("split_resample_dt"),
        "sigma": float(downstream.get("split_sigma", 0.03)),
        "daughter_noise_std": float(downstream.get("split_daughter_noise_std", 0.0)),
        "growth_alpha": float(downstream.get("split_growth_alpha", 1.0)),
        "interaction_group_size": (
            None if interaction_group_size is None else int(interaction_group_size)
        ),
        "non_split_lineage_rollout": bool(downstream.get("lineage_enabled", False)),
        "trajectory_mode": trajectory["trajectory_mode"],
        "split_sde_piecewise": bool(trajectory["split_sde_piecewise"]),
        "piecewise_observed_sample_mode": trajectory["piecewise_observed_sample_mode"],
        "piecewise_include_end": trajectory["piecewise_include_end"],
        "trajectory_scope": trajectory["trajectory_scope"],
        "particle_counts_by_time": particle_counts,
        "observed_particle_counts": observed_counts,
        "generated_particle_counts": generated_counts,
        "slice_origins_by_time": slice_origins,
        "source_anchor_times_by_time": source_anchor_times,
        "stochastic_streams": dict(getattr(result, "simulation_seeds", {})),
        "stochastic_stream_contract": (
            "Brownian diffusion, split/resampling decisions, and daughter noise "
            "share the declared split_population stream. Stochastic interaction "
            "grouping uses a separate seeded torch.Generator and therefore does "
            "not advance the population stream. Piecewise interaction grouping "
            "derives a distinct stable seed for each observed interval."
        ),
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
    trajectory = _split_trajectory_contract(downstream)
    if options.reconstruction_diagnostic and trajectory["split_sde_piecewise"]:
        raise ValueError(
            "The fitted-model reconstruction diagnostic is not defined for "
            "piecewise observed-anchored interval simulation: each observed endpoint "
            "is an input anchor, so the comparison would be self-reconstruction "
            "rather than global-t0 prediction."
        )
    downstream_analyses = _effective_downstream_analyses(config, options)
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
    if spatial_dim != 2:
        raise ValueError(
            "CytoBridge training records the actual spatial_dim, but the current "
            "package workflow's interpolation, snapshots, and communication "
            f"downstream require exactly 2 spatial dimensions; got {spatial_dim}. "
            "Use the training API without this downstream workflow for non-2D "
            "or nonspatial inputs."
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
    model_contract = _loaded_model_scientific_contract(
        loaded,
        config=config,
        options=options,
    )
    has_interaction = bool(model_contract["interaction_component"])
    if not has_interaction and options.lr_database is not None:
        raise ValueError(
            "--lr-database cannot affect a no-interaction downstream run: "
            "ligand-receptor projection requires model-derived communication."
        )
    if not has_interaction and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to write a no-interaction downstream result into a non-empty "
            f"directory: {output_dir}. Use a new condition-specific output root so "
            "stale communication, LR, or interaction figures cannot coexist with "
            "a not-applicable summary."
        )
    effective_edge_mode = str(model_contract["edge_prior_mode"]).lower()
    effective_downstream = dict(downstream)
    effective_communication = dict(downstream.get("communication", {}))
    effective_lr_scope = downstream.get("lr_scope")
    if effective_edge_mode == "all_spatial":
        effective_communication["interpretation"] = (
            "Spatial-attention summaries over within-cutoff candidates without a "
            "learned ligand-receptor edge gate. This is a no-LR-prior ablation, "
            "not the production main model and not a global cell-cell "
            "communication screen."
        )
        if str(dataset.get("name")) == "admouse":
            effective_lr_scope = {
                "strict_supported_pair_count": 7,
                "interpretation": (
                    "The downstream strict complete-subunit projection remains "
                    "limited to the seven CellChatDB pairs represented by the AD "
                    "expression panel; those labels did not gate the all-spatial "
                    "ablation model and do not form a global CCI screen."
                ),
            }
    if not has_interaction:
        effective_communication["interpretation"] = (
            "Not applicable: this matched ablation has no interaction component. "
            "No radius graph, learned-gate graph, zero communication matrix, or "
            "ligand-receptor projection is substituted."
        )
        effective_lr_scope = None
    effective_downstream["communication"] = effective_communication
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
        split_daughter_noise_std=float(downstream.get("split_daughter_noise_std", 0.0)),
        split_growth_alpha=float(downstream.get("split_growth_alpha", 1.0)),
        split_interaction_m=(
            1024
            if model_contract["interaction_group_size"] is None
            else int(model_contract["interaction_group_size"])
        ),
        split_resample_dt=(
            None
            if downstream.get("split_resample_dt") is None
            else float(downstream["split_resample_dt"])
        ),
        split_max_particles=(
            None
            if downstream.get("split_max_particles") is None
            else int(downstream["split_max_particles"])
        ),
        split_sde_piecewise=bool(trajectory["split_sde_piecewise"]),
        split_sde_piecewise_include_end=bool(trajectory["piecewise_include_end"]),
        piecewise_observed_sample_mode=(
            trajectory["piecewise_observed_sample_mode"] or "t0_fixed"
        ),
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
    if has_interaction:
        communication_summary, communications = _write_communication_outputs(
            cb=cb,
            result=result,
            runtime=runtime,
            annotation_key=annotation_key,
            output_dir=output_dir / "communication",
            device=options.device,
            downstream=effective_downstream,
            seed=int(scientific["seed"]),
        )
    else:
        communication_summary = {
            "status": "not applicable",
            "representation": None,
            "edge_prior_mode": "none",
            "interpretation": effective_communication["interpretation"],
            "reason": "model has no interaction component",
            "edge_selection_by_time": None,
            "table": None,
            "attention_directory": None,
        }
        communications = {}
    analyses["communication"] = communication_summary
    analyses["communication"]["trajectory_scope"] = trajectory["trajectory_scope"]
    analyses["figures"] = _write_standard_figures(
        cb=cb,
        result=result,
        communications=communications,
        annotation_key=annotation_key,
        label_to_color=label_to_color,
        output_dir=output_dir / "figures",
        lineage_enabled=bool(downstream.get("lineage_enabled", False)),
    )

    reference_adata = None
    if downstream_analyses["gene_dynamics"] or (
        downstream_analyses["lr_enabled"] and has_interaction
    ):
        reference_adata = (
            adata
            if options.reference_h5ad is None
            else ad.read_h5ad(options.reference_h5ad.expanduser())
        )
    if downstream_analyses["gene_dynamics"]:
        analyses["gene_dynamics"] = _write_gene_dynamics_outputs(
            cb=cb,
            result=result,
            reference_adata=reference_adata,
            spatial_dim=spatial_dim,
            preferred_species_tag=downstream_analyses["preferred_species_tag"],
            output_dir=output_dir / "gene_dynamics",
            downstream=downstream,
            allow_complete_reference_pca_center_fallback=(
                options.allow_complete_reference_pca_center_fallback
            ),
        )
    else:
        analyses["gene_dynamics"] = {"status": "not requested"}

    if downstream_analyses["lr_enabled"] and has_interaction:
        lr_database = downstream_analyses["lr_database"]
        if lr_database is None:
            raise ValueError("Strict LR projection is enabled without an LR database.")
        analyses["ligand_receptor"] = _write_lr_outputs(
            cb=cb,
            result=result,
            reference_adata=reference_adata,
            communications=communications,
            lr_database=lr_database.expanduser().resolve(),
            lr_complex_mode=options.lr_complex_mode,
            preferred_species_tag=downstream_analyses["preferred_species_tag"],
            annotation_key=annotation_key,
            resolved_time_key=resolved_time_key,
            spatial_dim=spatial_dim,
            output_dir=output_dir / "ligand_receptor",
            allow_complete_reference_pca_center_fallback=(
                options.allow_complete_reference_pca_center_fallback
            ),
            analysis_scope=effective_lr_scope,
        )
    elif not has_interaction:
        analyses["ligand_receptor"] = {
            "status": "not applicable",
            "reason": (
                "model has no interaction component; LR projection requires "
                "model-derived sparse communication attention"
            ),
            "analysis_scope": None,
        }
    else:
        analyses["ligand_receptor"] = {"status": "not requested"}
    analyses["ligand_receptor"]["trajectory_scope"] = trajectory["trajectory_scope"]

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
            "claim": (
                "not applicable as a global-t0 reconstruction diagnostic because "
                "observed endpoints are interval anchors"
                if trajectory["split_sde_piecewise"]
                else "not a training holdout or cross-method benchmark"
            ),
        }
    summary = {
        "dataset": dataset["name"],
        "annotation_key": annotation_key,
        "annotation_semantics": (
            "cell type"
            if str(dataset["name"]) == "chicken_heart"
            and annotation_key == "celltype_prediction"
            else "dataset annotation"
        ),
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
            "scientific_contract": model_contract,
        },
        "time_points": [float(value) for value in result.ts_points],
        "simulation": _downstream_simulation_summary(
            downstream=downstream,
            trajectory=trajectory,
            result=result,
            interaction_group_size=model_contract["interaction_group_size"],
        ),
        "classifier_accuracy": result.classifier_accuracy,
        "classifier_balanced_accuracy": result.classifier_balanced_accuracy,
        "classifier_split": (
            None
            if result.classifier_evaluation is None
            else {
                "cache_protocol_version": 8,
                "per_class_counts": result.classifier_evaluation.get(
                    "per_class_split_counts", {}
                ),
                "training_only_singleton_classes": result.classifier_evaluation.get(
                    "training_only_singleton_classes", []
                ),
                "singleton_class_policy": result.classifier_evaluation.get(
                    "singleton_class_policy"
                ),
                "stratification_repairs": result.classifier_evaluation.get(
                    "stratification_repairs", []
                ),
            }
        ),
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

    options = _resolve_workflow_options(config, options)
    if options.training_config is None:
        _validate_builtin_training_contract(config)
    selected = _selected_steps(config, options)
    paths = _output_paths(config, options)
    output_dir = paths["output_dir"]
    aligned_h5ad = paths["aligned_h5ad"]
    model_dir = paths["model_dir"]
    edge_predictor_path = paths["edge_predictor_path"]
    completed: list[str] = []
    outputs: dict[str, Any] = {}
    generated_edge_threshold: float | None = None

    if "preprocess" in selected and config.get("preprocess", {}).get("enabled", True):
        assert aligned_h5ad is not None
        aligned_h5ad.parent.mkdir(parents=True, exist_ok=True)
        aligned_h5ad = _run_preprocess(config, options, aligned_h5ad=aligned_h5ad)
        completed.append("preprocess")
        outputs["aligned_h5ad"] = str(aligned_h5ad)
        if (
            options.train
            and options.edge_predictor_path is None
            and _training_uses_learned_edge_prior(config, options)
        ):
            assert edge_predictor_path is not None
            edge_result = _run_edge_predictor(
                config,
                options,
                aligned_h5ad=aligned_h5ad,
                edge_predictor_path=edge_predictor_path,
            )
            generated_edge_threshold = float(edge_result["edge_predictor_threshold"])
            completed.append("edge_predictor")
            outputs["edge_predictor"] = edge_result

    if options.train:
        assert aligned_h5ad is not None and model_dir is not None
        model_dir.mkdir(parents=True, exist_ok=True)
        model_dir = _run_train(
            config,
            options,
            aligned_h5ad=aligned_h5ad,
            model_dir=model_dir,
            edge_predictor_path=edge_predictor_path,
            edge_predictor_threshold=generated_edge_threshold,
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
