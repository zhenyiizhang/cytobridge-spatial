"""Package-owned workflows for non-spatial temporal single-cell data.

The public names are resolved lazily so that ``cytobridge nonspatial plan``
and package-resource inspection remain available before optional scientific
dependencies are imported.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "PreparedNonSpatialData": (".preprocess", "PreparedNonSpatialData"),
    "prepare_scnt_nonspatial": (".preprocess", "prepare_scnt_nonspatial"),
    "prepare_weinreb_nonspatial": (".preprocess", "prepare_weinreb_nonspatial"),
    "NonSpatialPreset": (".workflow", "NonSpatialPreset"),
    "available_nonspatial_presets": (".workflow", "available_nonspatial_presets"),
    "build_nonspatial_lr_prior": (".workflow", "build_nonspatial_lr_prior"),
    "evaluate_nonspatial_pair": (".workflow", "evaluate_nonspatial_pair"),
    "nonspatial_plan": (".workflow", "nonspatial_plan"),
    "nonspatial_preset": (".workflow", "nonspatial_preset"),
    "packaged_training_config": (".workflow", "packaged_training_config"),
    "prepare_nonspatial_dataset": (".workflow", "prepare_nonspatial_dataset"),
    "train_nonspatial_condition": (".workflow", "train_nonspatial_condition"),
    "replay_nonspatial_figure": (".figures", "replay_nonspatial_figure"),
    "validate_historical_figure_bundle": (
        ".figures",
        "validate_historical_figure_bundle",
    ),
    "evaluate_weinreb_clone_fate": (
        ".weinreb_fate",
        "evaluate_weinreb_clone_fate",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
