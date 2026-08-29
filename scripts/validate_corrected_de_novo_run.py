#!/usr/bin/env python3
"""Validate the four corrected CytoBridge de novo production runs.

This is an artifact-level scientific acceptance check.  It reads the outputs
created by ``cytobridge workflow --train`` and verifies the shared protocol,
the complete six-stage fit, each dataset's declared edge prior, and the
numerical downstream products.  It does not modify the run directory.

Example
-------
python scripts/validate_corrected_de_novo_run.py \
    --run-root /path/to/corrected-de-novo-20260813-r2 \
    --report /path/to/corrected-de-novo-acceptance.json
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml


STAGES = (
    "Pretrain",
    "Refine",
    "Init_interaction",
    "Train_Score",
    "Finetune",
    "Score_Refine",
)

NO_INTERACTION_STAGES = (
    "Pretrain",
    "Refine",
    "Matched_stage_3_no_interaction",
    "Train_Score",
    "Finetune_no_interaction",
    "Score_Refine",
)


DATASETS = {
    "zebrafish": {
        "shape": (11999, 26628),
        "counts_layer": "counts",
        "annotation_key": "Annotation",
        "classifier_k": 10,
        "cutoff": 0.09606367405591873,
        "edge_prior_mode": "learned",
        "edge_predictor_threshold": 0.6063615679740906,
        "observed_counts": {0.0: 563, 1.0: 1036, 2.0: 2081, 3.0: 3048, 4.0: 5271},
        "interpolated": (0.5, 1.5, 2.5, 3.5),
        "score_epochs": 2001,
        "species": "zebrafish",
        "lr_database_name": "CellChatDB.ligrec.zebrafish.csv",
        "lr_database_sha256": "27fd0eb35da035a371ef68783d3e2dcf0729668fd58c2bb59f203173ea1b3f37",
    },
    "mosta": {
        "shape": (344603, 23761),
        "counts_layer": "count",
        "annotation_key": "Annotation",
        "classifier_k": 10,
        "cutoff": 0.02400244047956264,
        "edge_prior_mode": "learned",
        "edge_predictor_threshold": 0.1192110925912857,
        "observed_counts": {0.0: 51365, 1.0: 77369, 2.0: 102519, 3.0: 113350},
        "interpolated": (0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75),
        "score_epochs": 2001,
        "species": "mouse",
        "lr_database_name": "CellChatDB.ligrec.mouse.csv",
        "lr_database_sha256": "851c7ac12f4b2ba355eb991cda76646bf215fcab6e3819f7453bce2d3bc77673",
    },
    "arista": {
        "shape": (46209, 16379),
        "counts_layer": "counts",
        "annotation_key": "Annotation",
        "classifier_k": 10,
        "cutoff": 0.03154105148551745,
        "edge_prior_mode": "learned",
        "edge_predictor_threshold": 0.5884028673171997,
        "observed_counts": {0.0: 7668, 1.0: 8106, 2.0: 9440, 3.0: 9676, 4.0: 11319},
        "interpolated": (0.5, 1.5, 2.5, 3.5),
        "score_epochs": 2001,
        "species": "hs",
        "lr_database_name": "CellChatDB.ligrec.human.csv",
        "lr_database_sha256": "8bfb86da81206cc4c1d8ab15a0086e9fc6cd38a7206450a427e4e77bfb32731c",
    },
    "admouse": {
        "shape": (172092, 347),
        "counts_layer": "counts",
        "annotation_key": "major_annotation",
        "classifier_k": 1,
        "cutoff": 0.012106042891492197,
        "edge_prior_mode": "learned",
        "edge_predictor_threshold": 0.9956824779510498,
        "observed_counts": {0.0: 53615, 1.0: 58447, 2.0: 60030},
        "interpolated": tuple(
            round(value / 10, 1) for value in range(1, 26) if value not in (10, 20)
        ),
        "score_epochs": 3001,
        "species": "mouse",
        "lr_database_name": "CellChatDB.ligrec.mouse.csv",
        "lr_database_sha256": "851c7ac12f4b2ba355eb991cda76646bf215fcab6e3819f7453bce2d3bc77673",
        "strict_lr_pairs": 7,
    },
}

# The canonical 4d53ec9 Zebrafish downstream predates the explicit summary
# field below, but that immutable release passed daughter noise only as the
# then-hard-coded zero inside ``simulate_sde_points_split_from_x0``.  Do not
# generalize this exception: it is accepted only when the run root carries the
# exact final manifest and acceptance report that bind those historical bytes.
LEGACY_ZERO_DAUGHTER_NOISE_RELEASE = "4d53ec9ef29b5b5e41e76a22ca1e21900179a3c8"
LEGACY_ZERO_DAUGHTER_NOISE_ACCEPTANCE_SHA256 = (
    "48ac1924bc3fe10dd80d858f4bdf2100acc392811d184bc50d1bb5f1e161dd92"
)
LEGACY_ZERO_DAUGHTER_NOISE_MANIFEST_SHA256 = (
    "29aafb1b601b04d5853d11039e9c45d2c7258b3e0e658b34226ebcaab5b5295e"
)

_MATCHED_ABLATION_CONFIGS = {
    "zebrafish": {
        "no_lr_prior": "zebrafish_spatial_full_alpha_express_0015_no_lr_prior.yaml",
        "no_interaction": (
            "zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml"
        ),
    },
    "mosta": {
        "no_lr_prior": "mosta_spatial_full_alpha_express_0015_no_lr_prior.yaml",
        "no_interaction": ("mosta_spatial_full_alpha_express_0015_no_interaction.yaml"),
    },
    "arista": {
        "no_lr_prior": "arista_spatial_full_no_lr_prior.yaml",
        "no_interaction": "arista_spatial_full_no_interaction.yaml",
    },
    "admouse": {
        "no_lr_prior": ("admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml"),
        "no_interaction": (
            "admouse_spatial_full_alpha_express_0015_no_interaction.yaml"
        ),
    },
}

_MATCHED_FULL_CONFIGS = {
    "zebrafish": "zebrafish_spatial_full_alpha_express_0015.yaml",
    "mosta": "mosta_spatial_full_alpha_express_0015.yaml",
    "arista": "arista_spatial_full.yaml",
    "admouse": "admouse_spatial_full_alpha_express_0015.yaml",
}

ABLATION_PROFILES: dict[str, dict[str, Any]] = {}
for _dataset_name, _config_names in _MATCHED_ABLATION_CONFIGS.items():
    ABLATION_PROFILES[f"{_dataset_name}_no_lr_prior"] = {
        **DATASETS[_dataset_name],
        "artifact_dataset": _dataset_name,
        "run_role": "no-LR-prior ablation",
        "training_config": _config_names["no_lr_prior"],
        "interaction_component": True,
        "edge_prior_mode": "all_spatial",
        "edge_predictor_threshold": None,
        "stages": STAGES,
    }
    ABLATION_PROFILES[f"{_dataset_name}_no_interaction"] = {
        **DATASETS[_dataset_name],
        "artifact_dataset": _dataset_name,
        "run_role": "no-interaction ablation",
        "training_config": _config_names["no_interaction"],
        "interaction_component": False,
        "cutoff": None,
        "edge_prior_mode": "none",
        "edge_predictor_threshold": None,
        "strict_lr_pairs": None,
        "stages": NO_INTERACTION_STAGES,
    }
VALIDATION_PROFILES = {**DATASETS, **ABLATION_PROFILES}

MATCHED_ABLATION_ARMS = ("full", "no_lr_prior", "no_interaction")
MATCHED_ABLATION_PROTOCOL = "isolated-interaction-crn-v1"
MATCHED_ABLATION_SEED = 42
MATCHED_INTERACTION_SEED_OFFSET = 10_000
MATCHED_SCORE_ENERGY_OBJECTIVE = "velocity_score_cross_term"
TRAINING_IMPLEMENTATION_FILES = (
    "CytoBridge/tl/train/fit.py",
    "CytoBridge/tl/train/trainer.py",
    "CytoBridge/tl/core/models.py",
    "CytoBridge/tl/core/methods.py",
    "CytoBridge/tl/core/interaction.py",
    "CytoBridge/tl/core/losses.py",
    "CytoBridge/tl/core/flow_matching.py",
    "CytoBridge/tl/graph/spatial_gnn.py",
    "CytoBridge/utils/config.py",
    "CytoBridge/utils/utils.py",
)
MATCHED_CONDITIONS = {
    "full": ("full_learned", "learned"),
    "no_lr_prior": ("no_lr_all_spatial", "all_spatial"),
    "no_interaction": ("no_interaction", "none"),
}


@dataclass
class Audit:
    dataset: str
    checks: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def check(self, condition: bool, name: str, detail: str) -> None:
        self.checks.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}
        )

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def errors(self) -> list[dict[str, str]]:
        return [item for item in self.checks if item["status"] == "FAIL"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS" if not self.errors else "FAIL",
            "checks": self.checks,
            "warnings": self.warnings,
        }


def close_enough(actual: float, expected: float, atol: float = 1e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)


def safe_time_name(value: float) -> str:
    return f"{float(value):g}".replace("-", "minus_").replace(".", "p")


def h5_shape(node: h5py.Dataset | h5py.Group) -> tuple[int, ...]:
    if isinstance(node, h5py.Dataset):
        return tuple(int(value) for value in node.shape)
    return tuple(int(value) for value in node.attrs["shape"])


def numeric_node(node: h5py.Dataset | h5py.Group) -> h5py.Dataset:
    """Return the stored numeric values for a dense or sparse AnnData matrix."""

    return node if isinstance(node, h5py.Dataset) else node["data"]


def numeric_stats(node: h5py.Dataset | h5py.Group) -> tuple[bool, int, float, float]:
    """Stream a numeric HDF5 array and return finite/count/min/max statistics."""

    values = numeric_node(node)
    count = 0
    minimum = math.inf
    maximum = -math.inf
    finite = True
    block_rows = 100_000
    if values.ndim == 0:
        blocks = (np.asarray(values[()]),)
    else:
        blocks = (
            np.asarray(values[start : start + block_rows])
            for start in range(0, values.shape[0], block_rows)
        )
    for block in blocks:
        if block.size == 0:
            continue
        finite = finite and bool(np.isfinite(block).all())
        count += int(block.size)
        minimum = min(minimum, float(np.nanmin(block)))
        maximum = max(maximum, float(np.nanmax(block)))
    return finite, count, minimum, maximum


def h5_sum_identity(
    handle: h5py.File,
    total_key: str,
    part_keys: tuple[str, ...],
    *,
    atol: float = 1e-5,
) -> bool:
    total = numeric_node(handle[total_key])
    parts = [numeric_node(handle[key]) for key in part_keys]
    if any(part.shape != total.shape for part in parts):
        return False
    for start in range(0, total.shape[0], 100_000):
        stop = start + 100_000
        expected = sum(np.asarray(part[start:stop]) for part in parts)
        if not np.allclose(
            np.asarray(total[start:stop]), expected, rtol=1e-5, atol=atol
        ):
            return False
    return True


def read_metadata(path: Path):
    data = ad.read_h5ad(path, backed="r")
    try:
        return data.shape, data.obs.copy(), data.obs_names.copy(), dict(data.uns)
    finally:
        data.file.close()


def has_interaction_component(spec: Mapping[str, Any]) -> bool:
    """Return the profile's explicit component contract (main runs default true)."""

    return bool(spec.get("interaction_component", True))


def expected_stages(spec: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact ordered training stages declared by a profile."""

    return tuple(str(value) for value in spec.get("stages", STAGES))


def expected_epochs(spec: dict[str, Any]) -> dict[str, int]:
    stage_epochs = {
        "Pretrain": 100,
        "Refine": 100,
        "Init_interaction": 50,
        "Matched_stage_3_no_interaction": 50,
        "Train_Score": int(spec["score_epochs"]),
        "Finetune": 1000,
        "Finetune_no_interaction": 1000,
        "Score_Refine": int(spec["score_epochs"]),
    }
    return {stage: stage_epochs[stage] for stage in expected_stages(spec)}


def _mapping(value: Any) -> dict[str, Any]:
    """Return a plain mapping when possible, otherwise an empty mapping."""

    return dict(value) if isinstance(value, Mapping) else {}


def _finite_probability(value: Any) -> float | None:
    """Normalize a strict probability, returning ``None`` for invalid values."""

    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(normalized) or not 0.0 < normalized < 1.0:
        return None
    return normalized


def _is_not_applicable_number(value: Any) -> bool:
    """Accept only an explicit null or non-finite numeric N/A sentinel."""

    if value is None:
        return True
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _resolved_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _matched_profile_names(dataset: str) -> dict[str, str]:
    return {
        "full": dataset,
        "no_lr_prior": f"{dataset}_no_lr_prior",
        "no_interaction": f"{dataset}_no_interaction",
    }


def _expected_matched_ablation(dataset: str, arm: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "family": f"{dataset}-full-no-lr-no-interaction-v1",
        "dataset": dataset,
        "arm": arm,
        "protocol": MATCHED_ABLATION_PROTOCOL,
        "shared_seed": MATCHED_ABLATION_SEED,
        "interaction_grouping_seed_offset": MATCHED_INTERACTION_SEED_OFFSET,
        "input_contract": "exact-shared-aligned-h5ad",
        "implementation_contract": "exact-shared-training-code-sha256",
    }


def _matched_config_name(dataset: str, arm: str) -> str:
    if arm == "full":
        return _MATCHED_FULL_CONFIGS[dataset]
    return _MATCHED_ABLATION_CONFIGS[dataset][arm]


def _normalize_resolved_training_config(
    value: Mapping[str, Any],
    *,
    learned_edge_path: Path | None,
) -> dict[str, Any]:
    """Normalize only operational output/path fields and derived spatial dimension."""

    from copy import deepcopy

    normalized = deepcopy(dict(value))
    normalized["ckpt_dir"] = "__OPERATIONAL_CHECKPOINT_DIRECTORY__"
    normalized.pop("spatial_dim", None)
    model = _mapping(normalized.get("model"))
    model.pop("spatial_dim", None)
    normalized["model"] = model
    interaction = model.get("interaction_net")
    if isinstance(interaction, Mapping):
        interaction = deepcopy(dict(interaction))
        if str(interaction.get("edge_prior_mode", "learned")).lower() == "learned":
            if learned_edge_path is None:
                interaction["edge_predictor_path"] = "__MISSING_LEARNED_EDGE_PATH__"
            else:
                recorded = _resolved_optional_path(
                    interaction.get("edge_predictor_path")
                )
                interaction["edge_predictor_path"] = (
                    "__EXACT_LEARNED_EDGE_PREDICTOR__"
                    if recorded == learned_edge_path.expanduser().resolve()
                    else "__WRONG_LEARNED_EDGE_PREDICTOR__"
                )
        model["interaction_net"] = interaction
    return normalized


def canonical_matched_config_contract(
    dataset: str,
    arm: str,
    resolved_config: Mapping[str, Any],
    *,
    learned_edge_path: Path | None,
) -> tuple[bool, str]:
    """Compare an artifact config to the current package YAML, fail closed."""

    config_path = (
        Path(__file__).resolve().parents[1]
        / "CytoBridge"
        / "configs"
        / _matched_config_name(dataset, arm)
    )
    canonical = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = _normalize_resolved_training_config(
        canonical,
        learned_edge_path=learned_edge_path,
    )
    # The packaged learned path is a resource placeholder. Its only allowed
    # resolved replacement is the exact predictor file checked above.
    if arm == "full":
        expected["model"]["interaction_net"][
            "edge_predictor_path"
        ] = "__EXACT_LEARNED_EDGE_PREDICTOR__"
    actual = _normalize_resolved_training_config(
        resolved_config,
        learned_edge_path=learned_edge_path,
    )
    ok = actual == expected
    return ok, (
        f"artifact equals current {config_path.name} after only checkpoint/path/"
        "derived-spatial normalization"
        if ok
        else f"artifact differs from current {config_path.name}"
    )


def matched_config_only_delta_contract(
    configs: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, str]:
    """Independently assert the three arms differ only by the intervention."""

    from copy import deepcopy

    full = deepcopy(dict(configs["full"]))
    no_lr = deepcopy(dict(configs["no_lr_prior"]))
    no_interaction = deepcopy(dict(configs["no_interaction"]))
    for config in (full, no_lr, no_interaction):
        config.pop("spatial_dim", None)
        model = _mapping(config.get("model"))
        model.pop("spatial_dim", None)
        config["model"] = model

    def without(mapping: Mapping[str, Any], *keys: str) -> dict[str, Any]:
        result = deepcopy(dict(mapping))
        for key in keys:
            result.pop(key, None)
        return result

    no_lr_ok = without(no_lr, "model", "ckpt_dir", "matched_ablation") == without(
        full, "model", "ckpt_dir", "matched_ablation"
    )
    no_lr_ok = no_lr_ok and no_lr.get("training") == full.get("training")
    full_model = deepcopy(_mapping(full.get("model")))
    no_lr_model = deepcopy(_mapping(no_lr.get("model")))
    full_interaction = _mapping(full_model.pop("interaction_net", None))
    no_lr_interaction = _mapping(no_lr_model.pop("interaction_net", None))
    no_lr_ok = no_lr_ok and full_model == no_lr_model
    no_lr_ok = no_lr_ok and full_interaction.pop("edge_prior_mode", None) == "learned"
    no_lr_ok = (
        no_lr_ok and no_lr_interaction.pop("edge_prior_mode", None) == "all_spatial"
    )
    full_interaction.pop("edge_predictor_path", None)
    full_interaction.pop("edge_predictor_thre", None)
    no_lr_ok = no_lr_ok and full_interaction == no_lr_interaction

    none_ok = without(
        no_interaction, "model", "training", "ckpt_dir", "matched_ablation"
    ) == without(full, "model", "training", "ckpt_dir", "matched_ablation")
    no_interaction_model = _mapping(no_interaction.get("model"))
    full_model = _mapping(full.get("model"))
    none_ok = none_ok and no_interaction_model.get("components") == [
        component
        for component in full_model.get("components", [])
        if component != "interaction"
    ]
    none_ok = none_ok and set(no_interaction_model) == {
        "components",
        "velocity_net",
        "growth_net",
        "score_net",
    }
    none_ok = none_ok and all(
        no_interaction_model.get(network) == full_model.get(network)
        for network in ("velocity_net", "growth_net", "score_net")
    )
    full_training = _mapping(full.get("training"))
    none_training = _mapping(no_interaction.get("training"))
    none_ok = none_ok and none_training.get("defaults") == full_training.get("defaults")
    full_plan = full_training.get("plan", [])
    none_plan = none_training.get("plan", [])
    none_ok = none_ok and isinstance(full_plan, list) and isinstance(none_plan, list)
    none_ok = none_ok and len(full_plan) == len(none_plan) == 6
    expected_names = list(NO_INTERACTION_STAGES)
    if none_ok:
        for index, (full_stage, none_stage) in enumerate(
            zip(full_plan, none_plan, strict=True)
        ):
            full_stage = dict(full_stage)
            none_stage = dict(none_stage)
            full_strategy = str(full_stage.pop("train_strategy"))
            none_strategy = str(none_stage.pop("train_strategy"))
            full_stage.pop("name", None)
            none_name = none_stage.pop("name", None)
            none_interaction_use = none_stage.pop("interaction_use", None)
            full_stage.pop("interaction_use", None)
            none_ok = (
                none_ok
                and none_name == expected_names[index]
                and none_strategy == full_strategy.replace("+i", "")
                and none_stage == full_stage
                and (
                    none_interaction_use is False
                    if str(full_stage.get("mode", "")).lower() == "neural_ode"
                    else none_interaction_use is None
                )
            )
    return no_lr_ok and none_ok, f"full-vs-noLR={no_lr_ok}, full-vs-none={none_ok}"


def _training_implementation_identity() -> dict[str, Any]:
    """Independently hash the current package files that control training."""

    repository_root = Path(__file__).resolve().parents[1]
    files = {
        relative: _file_sha256(repository_root / relative)
        for relative in TRAINING_IMPLEMENTATION_FILES
    }
    aggregate = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "contract": "exact-shared-training-code-sha256",
        "hash_algorithm": "sha256",
        "files": files,
        "aggregate_sha256": aggregate,
        "unchanged_during_training": True,
    }


def _json_from_h5_scalar(node: h5py.Dataset) -> Any:
    value = node[()]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(f"expected embedded JSON string, found {type(value).__name__}")
    return json.loads(value)


def _embedded_training_run_summary(path: Path) -> dict[str, Any]:
    """Read only the embedded training summary without materializing AnnData."""

    with h5py.File(path, "r") as handle:
        node = handle.get("uns/training_run_summary/summary_json")
        if not isinstance(node, h5py.Dataset):
            raise KeyError("uns/training_run_summary/summary_json")
        value = _json_from_h5_scalar(node)
    if not isinstance(value, dict):
        raise TypeError("embedded training summary JSON must decode to an object")
    return value


def training_summary_embedding_contract(
    standalone_summary: Any,
    trained_h5ad: Path,
) -> tuple[bool, str]:
    """Require the standalone summary to be JSON-equivalent to AnnData metadata."""

    try:
        standalone = dict(standalone_summary)
        embedded = _embedded_training_run_summary(trained_h5ad)
        with h5py.File(trained_h5ad, "r") as handle:
            nested_node = handle.get("uns/all_model/training_run_summary/summary_json")
            if not isinstance(nested_node, h5py.Dataset):
                raise KeyError("uns/all_model/training_run_summary/summary_json")
            nested = _json_from_h5_scalar(nested_node)
        standalone_canonical = json.dumps(
            standalone, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        embedded_canonical = json.dumps(
            embedded, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        nested_canonical = json.dumps(
            nested, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        ok = standalone_canonical == embedded_canonical == nested_canonical
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"{type(error).__name__}: {error}"
    return ok, (
        "standalone, top-level embedded, and all_model embedded summaries are JSON-equivalent"
        if ok
        else "standalone or duplicate embedded summaries differ"
    )


def edge_predictor_artifact_contract(
    *,
    edge_path: Path,
    summary_provenance: Any,
    all_model: Any,
) -> tuple[bool, str]:
    """Bind a learned training run to the exact predictor bytes it consumed."""

    provenance = _mapping(summary_provenance)
    model = _mapping(all_model)
    resolved = edge_path.expanduser().resolve()
    size = int(resolved.stat().st_size) if resolved.is_file() else None
    digest = _file_sha256(resolved)
    recorded_path = _resolved_optional_path(provenance.get("path"))
    ok = (
        set(provenance)
        == {
            "applicable",
            "edge_prior_mode",
            "path",
            "size_bytes",
            "sha256",
            "not_applicable",
            "not_applicable_reason",
            "unchanged_during_training",
        }
        and size is not None
        and size > 0
        and digest is not None
        and provenance.get("applicable") is True
        and provenance.get("edge_prior_mode") == "learned"
        and provenance.get("not_applicable") is False
        and provenance.get("not_applicable_reason") is None
        and provenance.get("unchanged_during_training") is True
        and recorded_path == resolved
        and provenance.get("size_bytes") == size
        and provenance.get("sha256") == digest
        and model.get("edge_predictor_size_bytes") == size
        and model.get("edge_predictor_sha256") == digest
    )
    return ok, (
        f"path={resolved}, size={size}, sha256={digest}, "
        f"summary_size={provenance.get('size_bytes')!r}, "
        f"summary_sha256={provenance.get('sha256')!r}, "
        f"AnnData_size={model.get('edge_predictor_size_bytes')!r}, "
        f"AnnData_sha256={model.get('edge_predictor_sha256')!r}"
    )


def _cached_file_identity(
    path: Path,
    cache: dict[Path, tuple[int | None, str | None]],
) -> tuple[int | None, str | None]:
    resolved = path.expanduser().resolve()
    if resolved not in cache:
        cache[resolved] = (
            int(resolved.stat().st_size) if resolved.is_file() else None,
            _file_sha256(resolved),
        )
    return cache[resolved]


def _valid_array_identity(value: Any) -> bool:
    record = _mapping(value)
    shape = record.get("shape")
    return (
        set(record)
        == {
            "shape",
            "dtype",
            "nbytes",
            "canonical_order",
            "canonical_byte_order",
            "sha256",
        }
        and isinstance(shape, list)
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in shape
        )
        and isinstance(record.get("dtype"), str)
        and bool(record["dtype"])
        and isinstance(record.get("nbytes"), int)
        and not isinstance(record.get("nbytes"), bool)
        and int(record["nbytes"]) >= 0
        and record.get("canonical_order") == "C"
        and record.get("canonical_byte_order") == "little"
        and _is_sha256(record.get("sha256"))
    )


def _valid_obs_names_identity(value: Any) -> bool:
    record = _mapping(value)
    return (
        set(record) == {"count", "encoding", "length_prefix", "sha256"}
        and isinstance(record.get("count"), int)
        and not isinstance(record.get("count"), bool)
        and int(record["count"]) >= 0
        and record.get("encoding") == "utf-8"
        and record.get("length_prefix") == "uint64-big-endian"
        and _is_sha256(record.get("sha256"))
    )


def _valid_input_selection(value: Any) -> bool:
    selection = _mapping(value)
    return (
        set(selection)
        == {
            "time_key",
            "processed_time_key",
            "obsm_key",
            "resolved_latent_key",
            "spatial_key",
            "is_spatial",
        }
        and isinstance(selection.get("is_spatial"), bool)
        and all(
            isinstance(selection.get(key), str) and bool(selection[key].strip())
            for key in (
                "time_key",
                "processed_time_key",
                "obsm_key",
                "resolved_latent_key",
                "spatial_key",
            )
        )
    )


def _canonical_array_identity(values: Any) -> dict[str, Any]:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("cannot hash object-dtype training array")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "nbytes": int(canonical.nbytes),
        "canonical_order": "C",
        "canonical_byte_order": "little",
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _ordered_obs_names_identity(values: Sequence[Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
        count += 1
    return {
        "count": count,
        "encoding": "utf-8",
        "length_prefix": "uint64-big-endian",
        "sha256": digest.hexdigest(),
    }


def recompute_training_input_identities(
    h5ad_path: Path,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently reconstruct the arrays actually passed into formal training."""

    if not _valid_input_selection(selection):
        raise ValueError("invalid input_selection declaration")
    data = ad.read_h5ad(h5ad_path, backed="r")
    try:
        latent_key = str(selection["resolved_latent_key"])
        spatial_key = str(selection["spatial_key"])
        processed_time_key = str(selection["processed_time_key"])
        if latent_key not in data.obsm:
            raise KeyError(f"obsm/{latent_key}")
        if processed_time_key not in data.obs:
            raise KeyError(f"obs/{processed_time_key}")
        latent = np.asarray(data.obsm[latent_key], dtype=np.float32)
        if bool(selection["is_spatial"]):
            if spatial_key not in data.obsm:
                raise KeyError(f"obsm/{spatial_key}")
            spatial = np.asarray(data.obsm[spatial_key], dtype=np.float32)
            if spatial.shape[0] != latent.shape[0]:
                raise ValueError("spatial and latent rows differ")
            model_input = np.hstack((spatial, latent)).astype(np.float32)
        else:
            model_input = latent
        processed_time = data.obs[processed_time_key].to_numpy()
        obs_names = data.obs_names
        unique_timepoints = sorted(pd.unique(processed_time), key=float)
        return {
            "model_input": _canonical_array_identity(model_input),
            "processed_time": _canonical_array_identity(processed_time),
            "obs_names": _ordered_obs_names_identity(obs_names),
            "n_observations": int(data.n_obs),
            "n_timepoints": int(len(unique_timepoints)),
            "sample_counts_by_timepoint": [
                int(np.count_nonzero(processed_time == value))
                for value in unique_timepoints
            ],
        }
    finally:
        data.file.close()


def _matched_environment_signature(environment: Any) -> dict[str, Any] | None:
    """Normalize a same-software/same-device-model contract, ignoring GPU index."""

    value = _mapping(environment)
    required = (
        "device_type",
        "torch_version",
        "cuda_compiled_version",
        "cuda_available",
        "cuda_device_name",
        "cudnn_version",
        "python_version",
        "platform",
        "dependency_versions",
    )
    if any(key not in value for key in required):
        return None
    device_type = value.get("device_type")
    if device_type not in {"cpu", "cuda"}:
        return None
    if device_type == "cuda" and not (
        value.get("cuda_available") is True
        and isinstance(value.get("cuda_device_name"), str)
        and bool(value["cuda_device_name"].strip())
    ):
        return None
    dependencies = value.get("dependency_versions")
    if not isinstance(dependencies, Mapping):
        return None
    expected_dependencies = {
        "numpy",
        "pot",
        "torchdiffeq",
        "torch_geometric",
        "torch",
        "cuda",
        "cudnn",
        "python",
        "platform",
    }
    if set(dependencies) != expected_dependencies:
        return None
    package_specs = {
        "numpy": ("numpy", "numpy"),
        "pot": ("POT", "ot"),
        "torchdiffeq": ("torchdiffeq", "torchdiffeq"),
        "torch_geometric": ("torch-geometric", "torch_geometric"),
        "torch": ("torch", "torch"),
    }
    for key, (distribution, module) in package_specs.items():
        record = _mapping(dependencies.get(key))
        if (
            set(record) != {"distribution", "module", "version", "status"}
            or record.get("distribution") != distribution
            or record.get("module") != module
            or record.get("status") != "available"
            or not isinstance(record.get("version"), str)
            or not record["version"].strip()
        ):
            return None
    cuda = _mapping(dependencies.get("cuda"))
    cudnn = _mapping(dependencies.get("cudnn"))
    python = _mapping(dependencies.get("python"))
    dependency_platform = _mapping(dependencies.get("platform"))
    if set(cuda) != {"version", "status"} or cuda.get("status") not in {
        "compiled",
        "not_compiled",
    }:
        return None
    if (cuda.get("version") is None) is (cuda.get("status") == "compiled"):
        return None
    if cuda.get("version") is not None and not isinstance(cuda.get("version"), str):
        return None
    if set(cudnn) != {"version", "status"} or cudnn.get("status") not in {
        "available",
        "unavailable",
    }:
        return None
    if (cudnn.get("version") is None) is (cudnn.get("status") == "available"):
        return None
    if cudnn.get("version") is not None and (
        not isinstance(cudnn.get("version"), int)
        or isinstance(cudnn.get("version"), bool)
    ):
        return None
    if (
        set(python) != {"version", "implementation", "status"}
        or python.get("status") != "available"
        or not all(
            isinstance(python.get(key), str) and bool(python[key].strip())
            for key in ("version", "implementation")
        )
    ):
        return None
    if (
        set(dependency_platform) != {"value", "status"}
        or dependency_platform.get("status") != "available"
        or not isinstance(dependency_platform.get("value"), str)
        or not dependency_platform["value"].strip()
    ):
        return None
    if (
        value.get("torch_version") != dependencies["torch"]["version"]
        or value.get("cuda_compiled_version") != cuda.get("version")
        or value.get("cudnn_version") != cudnn.get("version")
        or value.get("python_version") != python.get("version")
        or value.get("platform") != dependency_platform.get("value")
    ):
        return None
    return {key: value[key] for key in required}


def _checkpoint_state_dict(path: Path) -> dict[str, Any]:
    """Load a plain or conventionally wrapped Torch state dictionary."""

    import torch

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older Torch.
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    for key in ("model_state_dict", "state_dict"):
        nested = loaded.get(key)
        if isinstance(nested, Mapping):
            loaded = nested
            break
    return {str(key): value for key, value in loaded.items()}


def _is_interaction_checkpoint_key(name: str) -> bool:
    parts = {
        part.strip().lower()
        for part in str(name).replace("/", ".").split(".")
        if part.strip()
    }
    return bool(
        parts
        & {
            "interaction",
            "interaction_net",
            "link_predictor",
            "edge_predictor",
        }
    )


def _retained_checkpoint_state(path: Path) -> dict[str, Any]:
    state = {
        key: value
        for key, value in _checkpoint_state_dict(path).items()
        if not _is_interaction_checkpoint_key(key)
    }
    if not state:
        raise ValueError(f"checkpoint has no retained non-interaction tensors: {path}")
    return state


def retained_checkpoint_contract(paths: Sequence[Path]) -> tuple[bool, str]:
    """Require identical non-interaction tensor state across three checkpoints."""

    import torch

    try:
        states = [_retained_checkpoint_state(path) for path in paths]
        key_sets = [set(state) for state in states]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            return (
                False,
                f"retained key sets differ: {[len(keys) for keys in key_sets]}",
            )
        for key in sorted(key_sets[0]):
            reference = states[0][key]
            if not isinstance(reference, torch.Tensor):
                return False, f"retained value is not a tensor: {key}"
            for state in states[1:]:
                candidate = state[key]
                exact = isinstance(candidate, torch.Tensor)
                if exact:
                    exact = (
                        reference.dtype == candidate.dtype
                        and tuple(reference.shape) == tuple(candidate.shape)
                        and torch.equal(
                            reference.detach()
                            .cpu()
                            .contiguous()
                            .reshape(-1)
                            .view(torch.uint8),
                            candidate.detach()
                            .cpu()
                            .contiguous()
                            .reshape(-1)
                            .view(torch.uint8),
                        )
                    )
                if not exact:
                    return False, f"retained tensor differs: {key}"
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return False, f"{type(error).__name__}: {error}"
    return True, f"{len(key_sets[0])} retained tensors are byte-exact"


def _stage_checkpoint_path(
    training_dir: Path, config: Mapping[str, Any], stage: str
) -> Path:
    defaults = _mapping(_mapping(config.get("training")).get("defaults"))
    plan = _mapping(config.get("training")).get("plan", [])
    stage_config = next(
        (
            item
            for item in plan
            if isinstance(item, Mapping) and str(item.get("name")) == stage
        ),
        None,
    )
    if not isinstance(stage_config, Mapping):
        return training_dir / stage / "__missing_stage__.pth"
    strategy = str(
        stage_config.get("save_strategy", defaults.get("save_strategy", "best"))
    )
    return training_dir / stage / f"{strategy}_model.pth"


def _configured_stage_interaction_active(
    config: Mapping[str, Any], stage_index: int
) -> bool:
    """Derive stage interaction activity from config, never from the summary."""

    model = _mapping(config.get("model"))
    has_interaction = "interaction" in {
        str(component).strip().lower() for component in model.get("components", [])
    }
    if not has_interaction:
        return False
    plan = _mapping(config.get("training")).get("plan", [])
    if not isinstance(plan, list) or not 0 <= stage_index < len(plan):
        return False
    stage = _mapping(plan[stage_index])
    mode = str(stage.get("mode", "neural_ode")).strip().lower()
    if mode == "score_matching":
        return True
    if mode != "neural_ode":
        return False
    if stage.get("interaction_use") is not None:
        return stage.get("interaction_use") is True
    return "i" in {
        token.strip().lower()
        for token in str(stage.get("train_strategy", "")).split("+")
    }


def _configured_stage_mode(config: Mapping[str, Any], stage_index: int) -> str | None:
    plan = _mapping(config.get("training")).get("plan")
    if not isinstance(plan, list) or not 0 <= stage_index < len(plan):
        return None
    stage = plan[stage_index]
    if not isinstance(stage, Mapping):
        return None
    mode = stage.get("mode")
    return str(mode).strip().lower() if isinstance(mode, str) and mode.strip() else None


def _expected_stage_execution_contract(
    config: Mapping[str, Any],
    n_timepoints: Any,
    stage_index: int,
) -> dict[str, Any] | None:
    training = _mapping(config.get("training"))
    defaults = _mapping(training.get("defaults"))
    plan = training.get("plan")
    if not isinstance(plan, list) or not 0 <= stage_index < len(plan):
        return None
    stage = plan[stage_index]
    if not isinstance(stage, Mapping):
        return None
    try:
        epochs = int(stage["epochs"])
        batch_size = int(stage.get("batch_size", defaults["batch_size"]))
        n_timepoints = int(n_timepoints)
    except (KeyError, TypeError, ValueError):
        return None
    if any(isinstance(value, bool) for value in (epochs, batch_size, n_timepoints)):
        return None
    if epochs <= 0 or batch_size <= 0 or n_timepoints < 2:
        return None
    mode = str(stage.get("mode", "")).strip().lower()
    if mode == "neural_ode":
        steps_per_epoch = (n_timepoints - 1) * (
            2 if config.get("reverse") is True else 1
        )
        optimizer_steps = epochs * steps_per_epoch
    elif mode == "score_matching":
        optimizer_steps = epochs
    else:
        return None
    return {
        "stage": str(stage.get("name", "")),
        "mode": mode,
        "configured_epochs": epochs,
        "recorded_epochs": epochs,
        "batch_size": batch_size,
        "optimizer_step_count": optimizer_steps,
    }


def _valid_global_rng_record(value: Any) -> bool:
    record = _mapping(value)
    cuda = _mapping(record.get("torch_cuda"))
    determinism = _mapping(record.get("determinism"))
    selected = cuda.get("selected_device")
    cuda_available = cuda.get("available")
    if cuda_available is True:
        selected_mapping = _mapping(selected)
        cuda_ok = (
            isinstance(cuda.get("visible_device_count"), int)
            and not isinstance(cuda.get("visible_device_count"), bool)
            and int(cuda["visible_device_count"]) > 0
            and isinstance(selected_mapping.get("index"), int)
            and not isinstance(selected_mapping.get("index"), bool)
            and isinstance(selected_mapping.get("name"), str)
            and bool(selected_mapping["name"].strip())
            and _is_sha256(cuda.get("state_sha256"))
        )
    elif cuda_available is False:
        cuda_ok = (
            cuda.get("visible_device_count") == 0
            and selected is None
            and cuda.get("state_sha256") is None
        )
    else:
        cuda_ok = False
    determinism_ok = (
        set(determinism)
        == {
            "deterministic_algorithms_enabled",
            "deterministic_algorithms_warn_only_enabled",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "bit_exact_cuda_determinism_claimed",
        }
        and all(
            determinism.get(key) is None or isinstance(determinism.get(key), bool)
            for key in determinism
        )
        and determinism.get("bit_exact_cuda_determinism_claimed") is False
    )
    expected_aggregate = hashlib.sha256(
        json.dumps(
            {key: item for key, item in record.items() if key != "aggregate_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return (
        _is_sha256(record.get("python_random_sha256"))
        and _is_sha256(record.get("numpy_legacy_sha256"))
        and _is_sha256(record.get("torch_cpu_sha256"))
        and set(cuda)
        == {
            "available",
            "visible_device_count",
            "selected_device",
            "state_sha256",
            "snapshot_scope",
        }
        and cuda.get("snapshot_scope") == "selected_training_device_only"
        and cuda_ok
        and determinism_ok
        and record.get("aggregate_sha256") == expected_aggregate
    )


def _comparable_global_rng(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop only the physical CUDA index, which may differ between matched jobs."""

    normalized = json.loads(json.dumps(value))
    normalized.pop("aggregate_sha256", None)
    selected = _mapping(_mapping(normalized.get("torch_cuda")).get("selected_device"))
    if selected:
        selected.pop("index", None)
        normalized["torch_cuda"]["selected_device"] = selected
    return normalized


def _valid_private_rng_record(value: Any, *, active: bool) -> bool:
    record = _mapping(value)
    if set(record) != {"active", "seed", "state_sha256"}:
        return False
    if active:
        return (
            record.get("active") is True
            and record.get("seed")
            == MATCHED_ABLATION_SEED + MATCHED_INTERACTION_SEED_OFFSET
            and _is_sha256(record.get("state_sha256"))
        )
    return record == {"active": False, "seed": None, "state_sha256": None}


def _valid_common_random_numbers(
    value: Any,
    *,
    condition: str,
    interaction_mode: str,
    components: Sequence[str],
) -> bool:
    record = _mapping(value)
    formal = _mapping(record.get("formal_data_contract"))
    global_streams = _mapping(record.get("global_streams"))
    constructor = _mapping(record.get("constructor_isolation"))
    optional_interaction = _mapping(constructor.get("optional_interaction_component"))
    frozen_predictor = _mapping(constructor.get("frozen_edge_predictor"))
    grouping = _mapping(record.get("interaction_grouping_stream"))
    inactive = _mapping(record.get("inactive_interaction"))
    has_interaction = condition != "no_interaction"
    expected_frozen = {
        "active": interaction_mode == "learned",
        "mechanism": (
            "nested torch.random.fork_rng(devices=[])"
            if interaction_mode == "learned"
            else "not_applicable"
        ),
        "trainable_backbone_initialization_isolated": interaction_mode
        in {"learned", "all_spatial"},
    }
    grouping_ok = (
        set(grouping)
        == {
            "active",
            "generator",
            "device",
            "seed_offset",
            "seed",
            "reset_between_stages",
            "shared_seed_for_full_and_no_lr",
            "advances_global_torch_stream",
        }
        and grouping.get("active") is has_interaction
        and grouping.get("generator")
        == ("private torch.Generator" if has_interaction else None)
        and (
            isinstance(grouping.get("device"), str)
            if has_interaction
            else grouping.get("device") is None
        )
        and grouping.get("seed_offset")
        == (MATCHED_INTERACTION_SEED_OFFSET if has_interaction else None)
        and grouping.get("seed")
        == (
            MATCHED_ABLATION_SEED + MATCHED_INTERACTION_SEED_OFFSET
            if has_interaction
            else None
        )
        and grouping.get("reset_between_stages") is False
        and grouping.get("shared_seed_for_full_and_no_lr")
        is (interaction_mode in {"learned", "all_spatial"})
        and grouping.get("advances_global_torch_stream") is False
    )
    return (
        set(record)
        == {
            "schema_version",
            "protocol",
            "strict_matched_entrypoint",
            "condition",
            "interaction_mode",
            "components",
            "formal_data_contract",
            "global_streams",
            "constructor_isolation",
            "interaction_grouping_stream",
            "inactive_interaction",
        }
        and record.get("schema_version") == 1
        and record.get("protocol") == MATCHED_ABLATION_PROTOCOL
        and record.get("strict_matched_entrypoint") is True
        and record.get("condition") == condition
        and record.get("interaction_mode") == interaction_mode
        and record.get("components") == list(components)
        and formal
        == {
            "matched_ablation_declared": True,
            "h5ad_and_exact_model_input_provenance_valid": True,
            "edge_predictor_provenance_valid": True,
        }
        and set(global_streams)
        == {
            "base_seed",
            "seed_application",
            "python_random",
            "numpy_legacy",
            "torch_cpu",
            "torch_cuda",
            "optional_interaction_advance_policy",
            "cuda_determinism_boundary",
        }
        and global_streams.get("base_seed") == MATCHED_ABLATION_SEED
        and global_streams.get("seed_application") == "once_before_model_construction"
        and global_streams.get("python_random")
        == {"api": "random.seed", "seed": MATCHED_ABLATION_SEED}
        and global_streams.get("numpy_legacy")
        == {"api": "numpy.random.seed", "seed": MATCHED_ABLATION_SEED}
        and global_streams.get("torch_cpu")
        == {"api": "torch.manual_seed", "seed": MATCHED_ABLATION_SEED}
        and _mapping(global_streams.get("torch_cuda")).get("api")
        == "torch.cuda.manual_seed_all"
        and _mapping(global_streams.get("torch_cuda")).get("seed")
        == MATCHED_ABLATION_SEED
        and isinstance(
            _mapping(global_streams.get("torch_cuda")).get("available"), bool
        )
        and global_streams.get("optional_interaction_advance_policy") == "forbidden"
        and isinstance(global_streams.get("cuda_determinism_boundary"), str)
        and bool(global_streams["cuda_determinism_boundary"])
        and set(constructor)
        == {"optional_interaction_component", "frozen_edge_predictor"}
        and optional_interaction
        == {
            "mechanism": "torch.random.fork_rng(devices=[])",
            "construction_device": "cpu_before_model.to(device)",
            "restores_global_torch_cpu_state": True,
        }
        and frozen_predictor == expected_frozen
        and grouping_ok
        and inactive
        == {
            "forward_compute_skipped": True,
            "private_stream_not_advanced": True,
            "no_interaction_constructor_skipped": not has_interaction,
            "private_generator_created": has_interaction,
            "no_interaction_arm_skips_constructor_and_generator": (
                condition == "no_interaction"
            ),
        }
    )


def legacy_zero_daughter_noise_provenance(run_root: Path) -> tuple[bool, str]:
    """Recognize only the immutable accepted 4d53ec9 summary omission."""

    manifest_path = run_root / "final-manifest-4d53ec9.json"
    acceptance_path = run_root / "acceptance-4d53ec9.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        declared_acceptance = str(manifest.get("acceptance_report", ""))
        declared_path = (run_root / declared_acceptance).resolve()
        acceptance_digest = _file_sha256(acceptance_path)
        manifest_digest = _file_sha256(manifest_path)
        ok = (
            manifest_digest == LEGACY_ZERO_DAUGHTER_NOISE_MANIFEST_SHA256
            and manifest.get("release_commit") == LEGACY_ZERO_DAUGHTER_NOISE_RELEASE
            and Path(str(manifest.get("canonical_run_root", ""))).resolve()
            == run_root.resolve()
            and declared_path == acceptance_path.resolve()
            and manifest.get("acceptance_report_sha256")
            == LEGACY_ZERO_DAUGHTER_NOISE_ACCEPTANCE_SHA256
            and acceptance_digest == LEGACY_ZERO_DAUGHTER_NOISE_ACCEPTANCE_SHA256
            and manifest.get("acceptance_status") == "PASS"
            and acceptance.get("status") == "PASS"
            and _mapping(acceptance.get("datasets")).get("zebrafish", {}).get("status")
            == "PASS"
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        ok = False
        acceptance_digest = None
        manifest_digest = None
    return ok, (
        f"manifest={manifest_path.name}, release={LEGACY_ZERO_DAUGHTER_NOISE_RELEASE}, "
        f"manifest_sha256={manifest_digest}, acceptance_sha256={acceptance_digest}"
    )


def edge_threshold_provenance_contract(
    predictor_metadata: Any,
    *,
    expected_threshold: float,
) -> tuple[bool, str]:
    """Require the validation-selected threshold frozen for a production run."""

    metadata = _mapping(predictor_metadata)
    effective = _finite_probability(metadata.get("edge_predictor_threshold"))
    selected = _finite_probability(metadata.get("edge_predictor_threshold_selected"))
    expected = _finite_probability(expected_threshold)
    ok = (
        metadata.get("selection_source") == "validation"
        and effective is not None
        and selected is not None
        and expected is not None
        and close_enough(effective, selected)
        and close_enough(effective, expected)
    )
    return ok, (
        f"effective={metadata.get('edge_predictor_threshold')!r}, "
        f"selected={metadata.get('edge_predictor_threshold_selected')!r}, "
        f"expected={expected_threshold!r}, source={metadata.get('selection_source')!r}"
    )


def lr_database_provenance_contract(
    *,
    graph_metadata: Any,
    graph_metadata_present: bool,
    downstream_analysis: Any,
    spec: Mapping[str, Any],
    allow_ignored_input_graph: bool = False,
) -> tuple[bool, str]:
    """Validate the exact LR database and its role in graph and projection steps."""

    graph = _mapping(graph_metadata)
    graph_present = bool(graph_metadata_present or graph_metadata is not None)
    downstream = _mapping(downstream_analysis)
    expected_name = str(spec["lr_database_name"])
    expected_digest = str(spec["lr_database_sha256"])
    expected_species = str(spec["species"])
    mode = str(spec["edge_prior_mode"]).strip().lower()

    downstream_path = _resolved_optional_path(downstream.get("database"))
    downstream_digest = _file_sha256(downstream_path)
    if mode == "none":
        downstream_reason = str(downstream.get("reason", "")).lower()
        downstream_ok = (
            str(downstream.get("status", "")).lower() == "not applicable"
            and downstream_path is None
            and downstream.get("analysis_scope") is None
            and "no interaction component" in downstream_reason
        )
    else:
        downstream_ok = (
            downstream_path is not None
            and downstream_path.name == expected_name
            and downstream_digest == expected_digest
            and str(downstream.get("complex_mode", "")).lower() in {"min", "minimum"}
            and downstream.get("require_all_subunits") is True
            and str(downstream.get("preferred_species_tag", "")) == expected_species
        )

    graph_path = _resolved_optional_path(graph.get("lr_database_path"))
    graph_digest = _file_sha256(graph_path)
    resolved_pairs = graph.get("lr_unique_resolved_pairs")
    if mode == "learned":
        try:
            pair_count_ok = int(resolved_pairs) > 0
            if spec.get("strict_lr_pairs") is not None:
                pair_count_ok = pair_count_ok and int(resolved_pairs) == int(
                    spec["strict_lr_pairs"]
                )
        except (TypeError, ValueError):
            pair_count_ok = False
        graph_ok = (
            graph_present
            and graph_path is not None
            and graph_path.name == expected_name
            and graph_digest == expected_digest
            and graph.get("lr_matching_rule")
            == "selected_symbol_exact_case_insensitive_all_complex_subunits"
            and graph.get("lr_complex_expression_rule") == "minimum"
            and str(graph.get("preferred_species_tag", "")) == expected_species
            and pair_count_ok
        )
    elif mode in {"all_spatial", "none"}:
        # A matched no-LR or no-interaction ablation may deliberately reuse the
        # main run's exact aligned H5AD. That input can retain immutable learned-
        # graph provenance, provided training records that it stripped/ignored
        # the metadata and the trained artifact itself is clean. It is evidence
        # about the shared input, not an edge prior used by the ablation.
        if not graph_present:
            graph_ok = True
        elif allow_ignored_input_graph:
            try:
                ignored_pair_count_ok = int(resolved_pairs) > 0
                if spec.get("strict_lr_pairs") is not None:
                    ignored_pair_count_ok = int(resolved_pairs) == int(
                        spec["strict_lr_pairs"]
                    )
            except (TypeError, ValueError):
                ignored_pair_count_ok = False
            graph_ok = (
                graph_path is not None
                and graph_path.name == expected_name
                and graph_digest == expected_digest
                and graph.get("lr_matching_rule")
                == "selected_symbol_exact_case_insensitive_all_complex_subunits"
                and graph.get("lr_complex_expression_rule") == "minimum"
                and str(graph.get("preferred_species_tag", "")) == expected_species
                and ignored_pair_count_ok
            )
        else:
            graph_ok = False
    else:
        graph_ok = False

    ok = graph_ok and downstream_ok
    return ok, (
        f"mode={mode!r}, graph_present={graph_present}, "
        f"allow_ignored_input_graph={allow_ignored_input_graph}, "
        f"graph_database={None if graph_path is None else graph_path.name!r}, "
        f"graph_sha256={graph_digest}, graph_pairs={resolved_pairs!r}, "
        f"downstream_database={None if downstream_path is None else downstream_path.name!r}, "
        f"downstream_sha256={downstream_digest}, expected={expected_name!r}/"
        f"{expected_digest}"
    )


def trained_edge_prior_contract(
    all_model: Any,
    interaction_graph: Any,
    *,
    expected_mode: str,
    expected_threshold: float | None,
    expected_edge_path: Path | None,
    predictor_metadata: Any = None,
    interaction_graph_present: bool | None = None,
) -> tuple[bool, str]:
    """Validate the edge-prior provenance serialized in a trained AnnData.

    A learned prior must retain one internally consistent predictor identity in
    ``all_model``, the aligned interaction-graph metadata, and the predictor
    sidecar. A radius-only model must serialize no inert predictor values. A
    no-interaction model must additionally omit every interaction-model field.
    Neither ablation may carry a stale ``interaction_graph`` in its trained
    artifact.
    """

    model_meta = _mapping(all_model)
    graph_meta = _mapping(interaction_graph)
    predictor_meta = _mapping(predictor_metadata)
    graph_present = bool(
        interaction_graph is not None
        or (interaction_graph_present is not None and bool(interaction_graph_present))
    )
    mode = str(model_meta.get("edge_prior_mode", "")).strip().lower()
    recorded_path = _resolved_optional_path(model_meta.get("edge_predictor_path"))
    recorded_threshold = _finite_probability(model_meta.get("edge_predictor_threshold"))
    detail = (
        f"mode={mode!r}, path={recorded_path}, "
        f"threshold={model_meta.get('edge_predictor_threshold')!r}, "
        f"interaction_graph_present={graph_present}"
    )

    if expected_mode == "all_spatial":
        clean = (
            mode == "all_spatial"
            and model_meta.get("edge_predictor_path") is None
            and model_meta.get("edge_predictor_threshold") is None
            and model_meta.get("edge_predictor_size_bytes") is None
            and model_meta.get("edge_predictor_sha256") is None
            and not graph_present
        )
        return clean, detail

    if expected_mode == "none":
        model_config = _mapping(model_meta.get("model_config"))
        components = {
            str(component).strip().lower()
            for component in model_config.get("components", [])
        }
        clean = (
            mode in {"", "none"}
            and model_meta.get("edge_predictor_path") is None
            and model_meta.get("edge_predictor_threshold") is None
            and model_meta.get("edge_predictor_size_bytes") is None
            and model_meta.get("edge_predictor_sha256") is None
            and _is_not_applicable_number(model_meta.get("interaction_cutoff"))
            and components == {"velocity", "growth", "score"}
            and "interaction_net" not in model_config
            and "interaction_type" not in model_config
            and "interaction_group_size" not in model_config
            and not graph_present
        )
        return clean, f"components={sorted(components)}, {detail}"

    if expected_mode != "learned":
        return False, f"unsupported expected mode={expected_mode!r}; {detail}"

    expected_path = (
        None
        if expected_edge_path is None
        else expected_edge_path.expanduser().resolve()
    )
    expected_probability = _finite_probability(expected_threshold)
    sidecar_threshold = _finite_probability(
        predictor_meta.get("edge_predictor_threshold")
    )
    sidecar_selected = _finite_probability(
        predictor_meta.get("edge_predictor_threshold_selected")
    )
    graph_path = _resolved_optional_path(graph_meta.get("edge_predictor_path"))
    graph_model_path = _resolved_optional_path(
        graph_meta.get("edge_predictor_model_path")
    )
    graph_threshold = _finite_probability(graph_meta.get("edge_predictor_threshold"))
    graph_selected = _finite_probability(
        graph_meta.get("edge_predictor_threshold_selected")
    )
    learned_ok = (
        mode == "learned"
        and expected_path is not None
        and expected_path.is_file()
        and graph_present
        and recorded_path == expected_path
        and graph_path == expected_path
        and graph_model_path == expected_path
        and expected_probability is not None
        and recorded_threshold is not None
        and sidecar_threshold is not None
        and sidecar_selected is not None
        and graph_threshold is not None
        and graph_selected is not None
        and close_enough(recorded_threshold, expected_probability)
        and close_enough(sidecar_threshold, expected_probability)
        and close_enough(sidecar_selected, expected_probability)
        and close_enough(graph_threshold, expected_probability)
        and close_enough(graph_selected, expected_probability)
    )
    return learned_ok, detail


def classifier_split_contract(
    split: Any,
    *,
    expected_class_counts: Mapping[str, int] | None = None,
) -> tuple[bool, str]:
    """Validate strict class support in the cached classifier holdout."""

    split_meta = _mapping(split)
    per_class = _mapping(split_meta.get("per_class_counts"))
    listed_raw = split_meta.get("training_only_singleton_classes", [])
    listed = (
        [str(value) for value in listed_raw] if isinstance(listed_raw, list) else []
    )
    singleton_policy = split_meta.get("singleton_class_policy")
    expected_singletons: list[str] = []
    class_errors: list[str] = []
    recorded_totals: dict[str, int] = {}
    for class_name, raw_counts in per_class.items():
        counts = _mapping(raw_counts)
        try:
            total = int(counts["total"])
            train = int(counts["train"])
            validation = int(counts["validation"])
        except (KeyError, TypeError, ValueError):
            class_errors.append(f"{class_name}:malformed")
            continue
        recorded_totals[str(class_name)] = total
        if min(total, train, validation) < 0 or train + validation != total:
            class_errors.append(
                f"{class_name}:total={total},train={train},validation={validation}"
            )
        elif total == 1:
            expected_singletons.append(str(class_name))
            if (train, validation) != (1, 0):
                class_errors.append(
                    f"{class_name}:singleton train={train},validation={validation}"
                )
        elif total >= 2 and (train <= 0 or validation <= 0):
            class_errors.append(
                f"{class_name}:missing support train={train},validation={validation}"
            )
        elif total == 0:
            class_errors.append(f"{class_name}:empty class")

    singleton_listing_ok = len(listed) == len(set(listed)) and set(listed) == set(
        expected_singletons
    )
    expected_totals = (
        None
        if expected_class_counts is None
        else {str(key): int(value) for key, value in expected_class_counts.items()}
    )
    cohort_matches = expected_totals is None or recorded_totals == expected_totals
    ok = (
        split_meta.get("cache_protocol_version") == 8
        and bool(per_class)
        and not class_errors
        and singleton_listing_ok
        and cohort_matches
        and isinstance(singleton_policy, str)
        and bool(singleton_policy.strip())
    )
    detail = (
        f"protocol={split_meta.get('cache_protocol_version')!r}, "
        f"classes={len(per_class)}, listed_singletons={listed}, "
        f"expected_singletons={expected_singletons}, errors={class_errors}"
        f", cohort_matches={cohort_matches}, recorded_total={sum(recorded_totals.values())}, "
        f"expected_total={None if expected_totals is None else sum(expected_totals.values())}"
    )
    return ok, detail


def _time_count_mapping(value: Any) -> dict[float, int] | None:
    mapping = _mapping(value)
    normalized: dict[float, int] = {}
    try:
        for key, count in mapping.items():
            time_value = float(key)
            normalized_count = int(count)
            if not math.isfinite(time_value) or normalized_count <= 0:
                return None
            if time_value in normalized:
                return None
            normalized[time_value] = normalized_count
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized


def _time_text_mapping(value: Any) -> dict[float, str] | None:
    mapping = _mapping(value)
    normalized: dict[float, str] = {}
    try:
        for key, text in mapping.items():
            time_value = float(key)
            normalized_text = str(text)
            if (
                not math.isfinite(time_value)
                or not normalized_text
                or time_value in normalized
            ):
                return None
            normalized[time_value] = normalized_text
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized


def _time_float_mapping(value: Any) -> dict[float, float] | None:
    mapping = _mapping(value)
    normalized: dict[float, float] = {}
    try:
        for key, raw_value in mapping.items():
            time_value = float(key)
            normalized_value = float(raw_value)
            if (
                not math.isfinite(time_value)
                or not math.isfinite(normalized_value)
                or time_value in normalized
            ):
                return None
            normalized[time_value] = normalized_value
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized


def _zebrafish_scope_contract(scope: Any) -> bool:
    """Require language that rules out endpoint-conditioned/global trajectory claims."""

    normalized = " ".join(str(scope).lower().replace("_", "-").split())
    observed_anchor = "observed" in normalized and "anchor" in normalized
    interval_local = "interval-local" in normalized or "interval local" in normalized
    one_sided = "one-sided" in normalized or "one sided" in normalized
    following_endpoint = (
        "not conditioned on the following" in normalized and "endpoint" in normalized
    )
    not_global = (
        "not global-t0" in normalized
        or "not a global-t0" in normalized
        or "not global t0" in normalized
        or "not a global t0" in normalized
        or (
            "not a single lineage-continuous" in normalized
            and "or global-t0" in normalized
        )
    )
    not_lineage = (
        "not lineage-continuous" in normalized
        or "not a lineage-continuous" in normalized
        or "not a single lineage-continuous" in normalized
        or "not lineage continuous" in normalized
        or "not a lineage continuous" in normalized
        or "not a single lineage continuous" in normalized
    )
    return all(
        (
            observed_anchor,
            interval_local,
            one_sided,
            following_endpoint,
            not_global,
            not_lineage,
        )
    )


def slice_provenance_summary_contract(
    simulation: Any,
    *,
    slice_provenance: Mapping[float, Mapping[str, Any]],
    expected_times: Sequence[float],
) -> tuple[bool, str]:
    """Require the summary's origin/anchor maps to equal the emitted slices."""

    simulation_meta = _mapping(simulation)
    actual_origins = {
        float(time): str(provenance.get("origin"))
        for time, provenance in slice_provenance.items()
    }
    try:
        actual_anchors: dict[float, float] | None = {
            float(time): float(provenance.get("anchor_time"))
            for time, provenance in slice_provenance.items()
        }
    except (TypeError, ValueError, OverflowError):
        actual_anchors = None
    declared_origins = _time_text_mapping(simulation_meta.get("slice_origins_by_time"))
    declared_anchors = _time_float_mapping(
        simulation_meta.get("source_anchor_times_by_time")
    )
    expected = {float(time) for time in expected_times}
    ok = (
        declared_origins == actual_origins
        and declared_anchors == actual_anchors
        and set(actual_origins) == expected
    )
    return ok, (
        f"origins_match={declared_origins == actual_origins}, "
        f"anchors_match={declared_anchors == actual_anchors}, "
        f"times_match={set(actual_origins) == expected}"
    )


def zebrafish_split_sde_contract(
    simulation: Any,
    *,
    slice_counts: Mapping[float, int],
    expected_observed_counts: Mapping[float, int],
    analyses: Any,
    allow_legacy_implicit_zero_daughter_noise: bool = False,
) -> tuple[bool, str]:
    simulation_meta = _mapping(simulation)
    resample_dt = simulation_meta.get("split_resample_dt")
    ceiling = simulation_meta.get("split_particle_ceiling")
    trajectory_mode = simulation_meta.get("trajectory_mode")
    sample_mode = simulation_meta.get("piecewise_observed_sample_mode")
    include_end = simulation_meta.get("piecewise_include_end")
    scope = str(simulation_meta.get("trajectory_scope", ""))
    actual_counts = {float(time): int(count) for time, count in slice_counts.items()}
    expected_observed = {
        float(time): int(count) for time, count in expected_observed_counts.items()
    }
    actual_observed = {time: actual_counts.get(time) for time in expected_observed}
    actual_generated = {
        time: count
        for time, count in actual_counts.items()
        if time not in expected_observed
    }
    declared_all = _time_count_mapping(simulation_meta.get("particle_counts_by_time"))
    declared_observed = _time_count_mapping(
        simulation_meta.get("observed_particle_counts")
    )
    declared_generated = _time_count_mapping(
        simulation_meta.get("generated_particle_counts")
    )
    analysis_meta = _mapping(analyses)
    communication_scope = _mapping(analysis_meta.get("communication")).get(
        "trajectory_scope"
    )
    lr_scope = _mapping(analysis_meta.get("ligand_receptor")).get("trajectory_scope")
    reconstruction = _mapping(analysis_meta.get("reconstruction_diagnostic"))
    reconstruction_claim = str(reconstruction.get("claim", ""))
    reconstruction_claim_normalized = " ".join(
        reconstruction_claim.lower().replace("_", "-").split()
    )
    daughter_noise = simulation_meta.get("daughter_noise_std")
    daughter_noise_ok = (
        daughter_noise is not None and close_enough(daughter_noise, 0.0)
    ) or (daughter_noise is None and bool(allow_legacy_implicit_zero_daughter_noise))
    try:
        normalized_ceiling = int(ceiling)
        bounded_generated = bool(actual_generated) and all(
            0 < int(count) <= normalized_ceiling for count in actual_generated.values()
        )
        ok = (
            close_enough(simulation_meta.get("split_dt"), 0.05)
            and close_enough(resample_dt, 0.05)
            and close_enough(simulation_meta.get("sigma"), 0.03)
            and daughter_noise_ok
            and close_enough(simulation_meta.get("growth_alpha"), 1.0)
            and normalized_ceiling == 100_000
            and simulation_meta.get("configured_particle_cap") is None
            and simulation_meta.get("initial_particle_cap") is None
            and int(simulation_meta.get("initial_particles")) == 563
            and simulation_meta.get("non_split_lineage_rollout") is False
            and trajectory_mode
            == "piecewise_observed_anchored_interval_forward_simulation"
            and simulation_meta.get("split_sde_piecewise") is True
            and sample_mode == "per_timepoint"
            and include_end is False
            and _zebrafish_scope_contract(scope)
            and communication_scope == scope
            and lr_scope == scope
            and "global-t0" in reconstruction_claim_normalized
            and "not applicable" in reconstruction_claim_normalized
            and actual_observed == expected_observed
            and bounded_generated
            and declared_all == actual_counts
            and declared_observed == expected_observed
            and declared_generated == actual_generated
        )
    except (TypeError, ValueError, OverflowError):
        ok = False
    return (
        ok,
        f"trajectory_mode={trajectory_mode!r}, sample_mode={sample_mode!r}, "
        f"include_end={include_end!r}, split_dt={simulation_meta.get('split_dt')!r}, "
        f"split_resample_dt={resample_dt!r}, sigma={simulation_meta.get('sigma')!r}, "
        f"daughter_noise_std={daughter_noise!r}, "
        f"legacy_implicit_zero_allowed={allow_legacy_implicit_zero_daughter_noise}, "
        f"growth_alpha={simulation_meta.get('growth_alpha')!r}, "
        f"split_particle_ceiling={ceiling!r}, observed={actual_observed}, "
        f"generated={actual_generated}, declared_matches={declared_all == actual_counts}, "
        f"scope_contract={_zebrafish_scope_contract(scope)}, "
        f"analysis_scopes_match={communication_scope == scope and lr_scope == scope}, "
        f"reconstruction_claim={reconstruction_claim!r}",
    )


def retained_top_level_lr_pairs(frame: pd.DataFrame) -> tuple[int | None, str]:
    """Count retained pair trajectories, excluding pair-cell-type children."""

    required = {"trajectory_kind", "retained", "pair_id"}
    if not required.issubset(frame.columns):
        missing = sorted(required.difference(frame.columns))
        return None, f"missing_columns={missing}"
    retained = frame["retained"].astype(str).str.strip().str.lower().eq("true")
    top_level = frame["trajectory_kind"].astype(str).eq("pair")
    rows = frame.loc[retained & top_level, "pair_id"].astype(str)
    unique = int(rows.nunique())
    if unique != len(rows):
        return None, f"retained_pair_rows={len(rows)}, unique_pair_ids={unique}"
    return unique, f"retained_pair_rows={len(rows)}, unique_pair_ids={unique}"


def complete_lr_pair_time_grid(
    trajectory_coverage: pd.DataFrame,
    pair_timecourse: pd.DataFrame,
    *,
    expected_times: Sequence[float],
) -> tuple[bool, str]:
    """Require identical retained pair IDs and one row per pair and time."""

    coverage_columns = {"trajectory_kind", "retained", "pair_id"}
    timecourse_columns = {"pair_id", "time"}
    if not coverage_columns.issubset(trajectory_coverage.columns) or not (
        timecourse_columns.issubset(pair_timecourse.columns)
    ):
        return False, "missing pair identity or time columns"
    retained_mask = (
        trajectory_coverage["retained"].astype(str).str.strip().str.lower().eq("true")
    )
    top_level_mask = trajectory_coverage["trajectory_kind"].astype(str).eq("pair")
    retained_ids = set(
        trajectory_coverage.loc[retained_mask & top_level_mask, "pair_id"].astype(str)
    )
    pair_ids = pair_timecourse["pair_id"].astype(str)
    times = pd.to_numeric(pair_timecourse["time"], errors="coerce")
    keys = list(zip(pair_ids, times))
    expected_keys = {
        (pair_id, float(time)) for pair_id in retained_ids for time in expected_times
    }
    actual_keys = {(pair_id, float(time)) for pair_id, time in keys if pd.notna(time)}
    pair_id_match = retained_ids == set(pair_ids)
    unique_rows = len(keys) == len(actual_keys)
    complete = pair_id_match and unique_rows and actual_keys == expected_keys
    return complete, (
        f"retained_ids={len(retained_ids)}, timecourse_ids={pair_ids.nunique()}, "
        f"rows={len(keys)}, expected_rows={len(expected_keys)}, "
        f"pair_id_match={pair_id_match}, unique_rows={unique_rows}"
    )


def downstream_scope_contract(
    summary: Any,
    *,
    expected_edge_mode: str,
    expected_ad_lr_pairs: int | None = None,
) -> tuple[bool, str]:
    """Check that human-readable communication and LR scope matches the model."""

    summary_meta = _mapping(summary)
    analyses = _mapping(summary_meta.get("analyses"))
    communication = _mapping(analyses.get("communication"))
    ligand_receptor = _mapping(analyses.get("ligand_receptor"))
    if expected_edge_mode == "none":
        communication_reason = str(communication.get("reason", "")).lower()
        communication_text = str(communication.get("interpretation", "")).lower()
        lr_reason = str(ligand_receptor.get("reason", "")).lower()
        communication_ok = (
            str(communication.get("status", "")).lower() == "not applicable"
            and str(communication.get("edge_prior_mode", "")).lower() == "none"
            and communication.get("representation") is None
            and communication.get("edge_selection_by_time") is None
            and communication.get("table") is None
            and communication.get("attention_directory") is None
            and "no interaction component" in communication_reason
            and "not applicable" in communication_text
            and "no interaction component" in communication_text
            and "no radius graph" in communication_text
            and "ligand-receptor projection" in communication_text
            and "substituted" in communication_text
        )
        lr_ok = (
            str(ligand_receptor.get("status", "")).lower() == "not applicable"
            and ligand_receptor.get("analysis_scope") is None
            and "no interaction component" in lr_reason
            and "model-derived" in lr_reason
        )
        return communication_ok and lr_ok, (
            f"communication={communication}, ligand_receptor={ligand_receptor}"
        )

    communication_interpretation = communication.get("interpretation")
    training_scope = communication.get("training_interaction_scope")
    attention_scope = communication.get("attention_scope")
    communication_ok = (
        str(communication.get("edge_prior_mode", "")).lower() == expected_edge_mode
        and attention_scope == "full time-slice radius candidate graph"
        and isinstance(training_scope, str)
        and "stochastic interaction groups" in training_scope
        and "only within each group" in training_scope
        and "base size 1024" in training_scope
        and "at most 2047" in training_scope
    )

    lr_scope = _mapping(_mapping(analyses.get("ligand_receptor")).get("analysis_scope"))
    lr_ok = True
    if expected_ad_lr_pairs is not None:
        lr_interpretation = lr_scope.get("interpretation")
        communication_text = (
            communication_interpretation.lower()
            if isinstance(communication_interpretation, str)
            else ""
        )
        lr_text = (
            lr_interpretation.lower() if isinstance(lr_interpretation, str) else ""
        )
        communication_ok = communication_ok and (
            "within-cutoff" in communication_text
            and "global cell-cell communication" in communication_text
        )
        if expected_edge_mode == "learned":
            communication_ok = (
                communication_ok
                and "learned-predictor-gated" in communication_text
                and "trained from seven" in communication_text
                and "no-lr-prior ablation" not in communication_text
            )
        elif expected_edge_mode == "all_spatial":
            communication_ok = (
                communication_ok
                and "without a learned ligand-receptor edge gate" in communication_text
                and "no-lr-prior ablation" in communication_text
                and "not the production main model" in communication_text
                and "trained from seven" not in communication_text
            )
        else:
            communication_ok = False
        try:
            pair_count_ok = (
                int(lr_scope.get("strict_supported_pair_count")) == expected_ad_lr_pairs
            )
        except (TypeError, ValueError):
            pair_count_ok = False
        lr_ok = (
            pair_count_ok
            and ("seven" in lr_text or "7" in lr_text)
            and "expression panel" in lr_text
            and "global cci screen" in lr_text
        )
        if expected_edge_mode == "learned":
            lr_ok = lr_ok and "not a global cci screen" in lr_text
        elif expected_edge_mode == "all_spatial":
            lr_ok = lr_ok and "did not gate the all-spatial ablation model" in lr_text
    ok = communication_ok and lr_ok
    detail = (
        f"communication_mode={communication.get('edge_prior_mode')!r}, "
        f"attention_scope={attention_scope!r}, training_scope={training_scope!r}, "
        f"lr_scope={lr_scope}"
    )
    return ok, detail


def required_files(
    run_dir: Path,
    dataset: str,
    spec: dict[str, Any],
) -> dict[str, Path]:
    artifact_dataset = str(spec.get("artifact_dataset", dataset))
    training = run_dir / "training"
    downstream = run_dir / "downstream"
    paths = {
        "aligned H5AD": run_dir / "preprocess" / f"{artifact_dataset}_aligned.h5ad",
        "resolved training config": training / "config.yaml",
        "training history": training / "training_history.csv",
        "training run summary": training / "training_run_summary.json",
        "trained AnnData": training / "adata.h5ad",
        "downstream summary": downstream / "summary.json",
    }
    if spec["edge_prior_mode"] == "learned":
        edge_model = (
            run_dir
            / "preprocess"
            / "edge_classifier"
            / f"{artifact_dataset}_edge_model.pt"
        )
        paths["generated edge model"] = edge_model
        paths["generated edge metadata"] = edge_model.with_suffix(
            edge_model.suffix + ".meta.json"
        )
    return paths


def validate_aligned(
    audit: Audit,
    paths: dict[str, Path],
    spec: dict[str, Any],
) -> tuple[int, int, dict[str, int]]:
    aligned_path = paths["aligned H5AD"]
    shape, obs, obs_names, uns = read_metadata(aligned_path)
    audit.check(
        tuple(shape) == tuple(spec["shape"]), "analyzed cohort", f"shape={shape}"
    )
    audit.check(
        bool(obs_names.is_unique),
        "stable observation identities",
        f"unique={obs_names.is_unique}",
    )

    time_values = pd.to_numeric(obs["time_point_processed"], errors="raise").astype(
        float
    )
    counts = {
        float(key): int(value)
        for key, value in time_values.value_counts().sort_index().items()
    }
    audit.check(
        counts == spec["observed_counts"],
        "observed slice membership",
        f"counts={counts}",
    )
    annotation = obs[spec["annotation_key"]]
    valid_annotation = bool(
        annotation.notna().all() and annotation.astype(str).str.len().gt(0).all()
    )
    audit.check(
        valid_annotation, "cell annotations", f"column={spec['annotation_key']!r}"
    )
    annotation_counts = {
        str(key): int(value)
        for key, value in annotation.astype(str).value_counts().items()
    }

    preprocess_info = uns.get("preprocess_info", {})
    raw_layer = str(preprocess_info.get("raw_counts_layer", ""))
    audit.check(
        raw_layer == spec["counts_layer"],
        "raw expression layer",
        f"recorded={raw_layer!r}, expected={spec['counts_layer']!r}",
    )

    with h5py.File(aligned_path, "r") as handle:
        latent = handle["obsm/X_latent"]
        spatial = handle["obsm/spatial_aligned"]
        pcs = handle["varm/PCs"]
        center = handle["var/pca_center"]
        counts_node = handle[f"layers/{spec['counts_layer']}"]
        latent_stats = numeric_stats(latent)
        spatial_stats = numeric_stats(spatial)
        pcs_stats = numeric_stats(pcs)
        center_stats = numeric_stats(center)
        counts_stats = numeric_stats(counts_node)
        audit.check(
            h5_shape(latent) == (shape[0], 50) and latent_stats[0],
            "aligned latent state",
            f"shape={h5_shape(latent)}, range=({latent_stats[2]:.6g}, {latent_stats[3]:.6g})",
        )
        audit.check(
            h5_shape(spatial) == (shape[0], 2) and spatial_stats[0],
            "aligned spatial state",
            f"shape={h5_shape(spatial)}, range=({spatial_stats[2]:.6g}, {spatial_stats[3]:.6g})",
        )
        audit.check(
            h5_shape(pcs) == (shape[1], 50) and pcs_stats[0] and center_stats[0],
            "retained PCA transform",
            f"PCs={h5_shape(pcs)}, center={h5_shape(center)}",
        )
        audit.check(
            counts_stats[0] and counts_stats[1] > 0 and counts_stats[2] >= 0,
            "nonnegative finite raw counts",
            f"stored_values={counts_stats[1]}, range=({counts_stats[2]:.6g}, {counts_stats[3]:.6g})",
        )
    return int(shape[0]), 52, annotation_counts


def validate_edge_predictor(
    audit: Audit,
    run_dir: Path,
    paths: dict[str, Path],
    spec: dict[str, Any],
) -> float:
    meta = json.loads(paths["generated edge metadata"].read_text(encoding="utf-8"))
    threshold = float(meta["edge_predictor_threshold"])
    threshold_ok, threshold_detail = edge_threshold_provenance_contract(
        meta,
        expected_threshold=float(spec["edge_predictor_threshold"]),
    )
    audit.check(
        threshold_ok,
        "validation-selected corrected production edge threshold",
        threshold_detail,
    )
    audit.check(
        0.0 <= threshold <= 1.0
        and close_enough(meta["distance_threshold"], spec["cutoff"]),
        "edge distance and decision thresholds",
        f"cutoff={float(meta['distance_threshold']):.17g}, decision={threshold:.9g}",
    )
    audit.check(
        int(meta.get("random_seed", -1)) == 42
        and meta.get("split", {}).get("strategy") == "node_disjoint_holdout",
        "edge predictor split and seed",
        f"split={meta.get('split', {}).get('strategy')!r}, seed={meta.get('random_seed')}",
    )
    universe = meta["candidate_universe"]
    candidate_ok = (
        universe.get("definition") == "all directed pairs with 1e-6 < distance < cutoff"
        and int(universe.get("positive_edges", 0)) > 0
        and int(universe.get("negative_edges", 0)) > 0
        and int(universe.get("training_balanced_edges", 0)) > 0
    )
    audit.check(
        candidate_ok,
        "edge candidate universe",
        f"positive={universe.get('positive_edges')}, negative={universe.get('negative_edges')}",
    )
    expected_times = tuple(spec["observed_counts"])
    meta_times = tuple(
        float(value) for value in meta["split"]["time_values_by_local_index"]
    )
    graph_files = sorted(
        (run_dir / "preprocess" / "input_graph").glob("*/*_adjacency_records")
    )
    audit.check(
        meta_times == expected_times and len(graph_files) == len(expected_times),
        "per-time LR interaction graphs",
        f"times={meta_times}, graph_files={len(graph_files)}",
    )
    validation = meta["validation_metrics_at_selected_threshold"]
    test = meta["test_metrics_at_validation_threshold"]
    quality = (
        f"validation AP={validation.get('average_precision')}, F1={validation.get('f1')}; "
        f"test AP={test.get('average_precision')}, F1={test.get('f1')}"
    )
    audit.check(
        int(validation.get("n_candidates", 0)) > 0
        and int(test.get("n_candidates", 0)) > 0,
        "natural-prevalence edge holdouts",
        quality,
    )
    if float(validation.get("f1", 0.0)) < 0.1:
        audit.warn(
            "Low edge-predictor validation F1 under natural prevalence: " + quality
        )
    return threshold


def validate_training(
    audit: Audit,
    paths: dict[str, Path],
    spec: dict[str, Any],
    threshold: float | None,
    model_dim: int,
) -> None:
    config = yaml.safe_load(
        paths["resolved training config"].read_text(encoding="utf-8")
    )
    defaults = config["training"]["defaults"]
    model_config = _mapping(config.get("model"))
    components = {
        str(component).strip().lower()
        for component in model_config.get("components", [])
    }
    interaction_expected = has_interaction_component(spec)
    expected_components = (
        {"velocity", "growth", "score", "interaction"}
        if interaction_expected
        else {"velocity", "growth", "score"}
    )
    interaction = _mapping(model_config.get("interaction_net"))
    audit.check(
        int(config["seed"]) == 42
        and close_enough(defaults["alpha_express"], 0.015)
        and close_enough(defaults["alpha_spatial"], 10.0),
        "shared training constants",
        f"seed={config['seed']}, alpha_express={defaults['alpha_express']}, alpha_spatial={defaults['alpha_spatial']}",
    )
    component_contract_ok = components == expected_components
    if interaction_expected:
        component_contract_ok = component_contract_ok and bool(interaction)
    else:
        component_contract_ok = component_contract_ok and all(
            key not in model_config
            for key in ("interaction_net", "interaction_type", "interaction_group_size")
        )
    audit.check(
        component_contract_ok,
        "declared model components",
        f"components={sorted(components)}, interaction_fields="
        f"{tuple(key for key in ('interaction_net', 'interaction_type', 'interaction_group_size') if key in model_config)}",
    )
    actual_edge_prior = (
        str(interaction.get("edge_prior_mode", "learned")).lower()
        if interaction_expected
        else "none"
    )
    if spec["edge_prior_mode"] == "learned":
        edge_path = paths["generated edge model"].resolve()
        configured_edge_path = Path(interaction["edge_predictor_path"]).resolve()
        edge_prior_ok = (
            threshold is not None
            and actual_edge_prior == "learned"
            and configured_edge_path == edge_path
            and close_enough(interaction["edge_predictor_thre"], threshold)
            and close_enough(interaction["cutoff"], spec["cutoff"])
        )
        edge_prior_detail = (
            f"mode={actual_edge_prior}, path={configured_edge_path}, "
            f"threshold={interaction['edge_predictor_thre']}"
        )
    elif spec["edge_prior_mode"] == "all_spatial":
        edge_prior_ok = (
            threshold is None
            and actual_edge_prior == "all_spatial"
            and close_enough(interaction["cutoff"], spec["cutoff"])
            and "edge_predictor_path" not in interaction
            and "edge_predictor_thre" not in interaction
        )
        edge_prior_detail = (
            f"mode={actual_edge_prior}, cutoff={interaction.get('cutoff')}, "
            f"predictor_path={interaction.get('edge_predictor_path')}, "
            f"predictor_threshold={interaction.get('edge_predictor_thre')}"
        )
    else:
        edge_prior_ok = (
            spec["edge_prior_mode"] == "none"
            and not interaction_expected
            and threshold is None
            and actual_edge_prior == "none"
            and "interaction_net" not in model_config
        )
        edge_prior_detail = (
            f"mode={actual_edge_prior}, interaction_component={interaction_expected}, "
            f"interaction_net_present={'interaction_net' in model_config}"
        )
    audit.check(
        edge_prior_ok,
        "declared edge prior wired into training",
        edge_prior_detail,
    )

    plan = config["training"]["plan"]
    plan_names = tuple(str(stage["name"]) for stage in plan)
    required_stages = expected_stages(spec)
    epochs = expected_epochs(spec)
    audit.check(
        plan_names == required_stages,
        "six-stage training plan",
        f"stages={plan_names}, expected={required_stages}",
    )
    plan_epochs = {str(stage["name"]): int(stage["epochs"]) for stage in plan}
    audit.check(
        plan_epochs == epochs, "configured stage lengths", f"epochs={plan_epochs}"
    )

    history = pd.read_csv(paths["training history"])
    history_names = tuple(history["stage"].drop_duplicates().astype(str))
    complete = history_names == required_stages
    details = []
    for stage_name, stage_epochs in epochs.items():
        stage_rows = history.loc[history["stage"] == stage_name]
        epoch_values = stage_rows["epoch"].astype(int).to_numpy()
        selected = (
            stage_rows["is_selected_checkpoint"].astype(str).str.lower().eq("true")
        )
        stage_complete = (
            len(stage_rows) == stage_epochs
            and np.array_equal(epoch_values, np.arange(1, stage_epochs + 1))
            and int(selected.sum()) == 1
        )
        complete = complete and stage_complete
        details.append(
            f"{stage_name}:{len(stage_rows)}/{stage_epochs},selected={int(selected.sum())}"
        )
    core_numeric = history[["loss", "checkpoint_value", "learning_rate"]].to_numpy(
        dtype=float
    )
    complete = complete and bool(np.isfinite(core_numeric).all())
    audit.check(complete, "complete finite training history", "; ".join(details))

    for stage in plan:
        name = str(stage["name"])
        is_score = str(stage.get("mode", "")).lower() == "score_matching"
        strategy = str(
            stage.get("save_strategy", defaults.get("save_strategy", "best"))
        )
        filename = "score_model.pth" if is_score else f"{strategy}_model.pth"
        checkpoint = paths["resolved training config"].parent / name / filename
        audit.check(
            checkpoint.is_file() and checkpoint.stat().st_size > 0,
            f"{name} checkpoint",
            str(checkpoint),
        )

    run_summary = json.loads(paths["training run summary"].read_text(encoding="utf-8"))
    stage_summaries = run_summary["stages"]
    summary_complete = (
        tuple(item["stage"] for item in stage_summaries) == required_stages
    )
    summary_complete = summary_complete and all(
        int(item["recorded_epochs"]) == epochs[item["stage"]]
        and int(item["configured_epochs"]) == epochs[item["stage"]]
        and item["selected_checkpoint_epoch"] is not None
        for item in stage_summaries
    )
    audit.check(
        summary_complete and float(run_summary["timing"]["run_wall_time_seconds"]) > 0,
        "measured training run summary",
        f"wall_seconds={run_summary['timing']['run_wall_time_seconds']:.3f}",
    )

    trained_shape, _, _, trained_uns = read_metadata(paths["trained AnnData"])
    audit.check(
        tuple(trained_shape) == tuple(spec["shape"]),
        "trained AnnData cohort",
        f"shape={trained_shape}",
    )
    embedded_ok, embedded_detail = training_summary_embedding_contract(
        run_summary,
        paths["trained AnnData"],
    )
    audit.check(
        embedded_ok,
        "standalone training summary bound to trained AnnData",
        embedded_detail,
    )
    predictor_metadata = None
    expected_edge_path = None
    if spec["edge_prior_mode"] == "learned":
        expected_edge_path = paths["generated edge model"]
        predictor_metadata = json.loads(
            paths["generated edge metadata"].read_text(encoding="utf-8")
        )
        predictor_bytes_ok, predictor_bytes_detail = edge_predictor_artifact_contract(
            edge_path=expected_edge_path,
            summary_provenance=_mapping(run_summary.get("data")).get("edge_predictor"),
            all_model=trained_uns.get("all_model"),
        )
        audit.check(
            predictor_bytes_ok,
            "exact learned edge-predictor bytes used by training",
            predictor_bytes_detail,
        )
    edge_artifact_ok, edge_artifact_detail = trained_edge_prior_contract(
        trained_uns.get("all_model"),
        trained_uns.get("interaction_graph"),
        expected_mode=spec["edge_prior_mode"],
        expected_threshold=threshold,
        expected_edge_path=expected_edge_path,
        predictor_metadata=predictor_metadata,
        interaction_graph_present="interaction_graph" in trained_uns,
    )
    audit.check(
        edge_artifact_ok,
        "trained AnnData edge-prior provenance",
        edge_artifact_detail,
    )
    if spec["edge_prior_mode"] in {"all_spatial", "none"}:
        all_model = _mapping(trained_uns.get("all_model"))
        with h5py.File(paths["aligned H5AD"], "r") as aligned_handle:
            aligned_has_graph = "uns/interaction_graph" in aligned_handle
        ignored_input_graph = all_model.get("ignored_input_interaction_graph_metadata")
        audit.check(
            isinstance(ignored_input_graph, (bool, np.bool_))
            and bool(ignored_input_graph) == aligned_has_graph,
            "matched-ablation aligned graph handling",
            f"aligned_interaction_graph_present={aligned_has_graph}, "
            "ignored_input_interaction_graph_metadata="
            f"{ignored_input_graph!r}",
        )
    with h5py.File(paths["trained AnnData"], "r") as handle:
        expected_vectors = [
            "obsm/velocity_model",
            "obsm/score_gradient_model",
            "obsm/full_drift_model",
            "obsm/growth_rate",
        ]
        if interaction_expected:
            expected_vectors.insert(1, "obsm/interaction_model")
        vectors_ok = all(key in handle for key in expected_vectors)
        if not interaction_expected:
            vectors_ok = vectors_ok and "obsm/interaction_model" not in handle
        vector_details = []
        if vectors_ok:
            for key in expected_vectors:
                stats = numeric_stats(handle[key])
                vectors_ok = vectors_ok and stats[0] and stats[1] > 0
                vector_details.append(
                    f"{key.rsplit('/', 1)[-1]}={h5_shape(handle[key])}"
                )
            vectors_ok = vectors_ok and h5_shape(handle["obsm/full_drift_model"]) == (
                trained_shape[0],
                model_dim,
            )
            vector_parts = ["obsm/velocity_model", "obsm/score_gradient_model"]
            if interaction_expected:
                vector_parts.insert(1, "obsm/interaction_model")
            vectors_ok = vectors_ok and h5_sum_identity(
                handle,
                "obsm/full_drift_model",
                tuple(vector_parts),
            )
        audit.check(
            vectors_ok, "finite fitted vector components", ", ".join(vector_details)
        )

    import CytoBridge as cb
    from CytoBridge.workflow import (
        WorkflowOptions,
        _loaded_model_scientific_contract,
        load_workflow_config,
    )

    load_options = {}
    if spec["edge_prior_mode"] == "learned":
        load_options["edge_predictor_path"] = paths["generated edge model"]
    loaded = cb.tl.load_dynamical_model_from_dir(
        paths["resolved training config"].parent,
        dim=model_dim,
        device="cpu",
        **load_options,
    )
    preset_name = str(spec.get("artifact_dataset", audit.dataset))
    preset, _ = load_workflow_config(preset_name)
    training_config = spec.get("training_config")
    contract = _loaded_model_scientific_contract(
        loaded,
        config=preset,
        options=WorkflowOptions(train=True, training_config=training_config),
    )
    parameters_finite = all(
        bool(np.isfinite(value.detach().cpu().numpy()).all())
        for value in loaded.model.state_dict().values()
    )
    expected_weight_stage = (
        "Finetune" if interaction_expected else "Finetune_no_interaction"
    )
    audit.check(
        loaded.weight_stage == expected_weight_stage
        and loaded.score_stage == "Score_Refine"
        and parameters_finite
        and contract["status"] == "matches requested preset",
        "strict final checkpoint load",
        f"weight={loaded.weight_stage}, expected_weight={expected_weight_stage}, "
        f"score={loaded.score_stage}, contract={contract['status']}",
    )


def validate_slice(
    audit: Audit,
    path: Path,
    annotation_key: str,
    model_dim: int,
    *,
    time_value: float,
    observed_time: bool,
    aligned: ad.AnnData | None = None,
    aligned_time_key: str = "time_point_processed",
) -> tuple[int, set[str], dict[str, Any]]:
    shape, obs, _, uns = read_metadata(path)
    annotations_ok = annotation_key in obs and obs[annotation_key].notna().all()
    with h5py.File(path, "r") as handle:
        state = handle["X"]
        spatial = handle["obsm/spatial"]
        state_stats = numeric_stats(state)
        spatial_stats = numeric_stats(spatial)
        consistent = h5_shape(state) == (shape[0], model_dim) and h5_shape(spatial) == (
            shape[0],
            2,
        )
        if consistent:
            state_values = numeric_node(state)
            spatial_values = numeric_node(spatial)
            for start in range(0, shape[0], 100_000):
                stop = start + 100_000
                if not np.allclose(
                    np.asarray(state_values[start:stop, :2]),
                    np.asarray(spatial_values[start:stop]),
                    rtol=0,
                    atol=0,
                ):
                    consistent = False
                    break
    audit.check(
        bool(
            shape[0] > 0
            and annotations_ok
            and state_stats[0]
            and spatial_stats[0]
            and consistent
        ),
        f"finite state {path.stem}",
        f"shape={shape}",
    )
    labels = set(obs[annotation_key].astype(str)) if annotation_key in obs else set()
    provenance = {
        "origin": uns.get("slice_origin"),
        "anchor_time": uns.get("source_anchor_time"),
    }
    if aligned is not None:
        if observed_time:
            aligned_times = pd.to_numeric(
                aligned.obs[aligned_time_key], errors="coerce"
            ).to_numpy(dtype=float)
            aligned_mask = np.isclose(
                aligned_times, float(time_value), rtol=0, atol=1e-12
            )
            aligned_subset = aligned[aligned_mask]
            expected_ids = aligned_subset.obs_names.astype(str).to_numpy()
            source_ids = (
                obs["source_obs_id"].astype(str).to_numpy()
                if "source_obs_id" in obs
                else np.asarray([], dtype=str)
            )
            expected_annotations = (
                aligned_subset.obs[annotation_key].astype(str).to_numpy()
            )
            actual_annotations = (
                obs[annotation_key].astype(str).to_numpy()
                if annotation_key in obs
                else np.asarray([], dtype=str)
            )
            expected_state = np.hstack(
                (
                    np.asarray(aligned_subset.obsm["spatial_aligned"]),
                    np.asarray(aligned_subset.obsm["X_latent"]),
                )
            )
            with h5py.File(path, "r") as handle:
                actual_state = np.asarray(numeric_node(handle["X"]))
            try:
                anchor_matches = close_enough(provenance["anchor_time"], time_value)
            except (TypeError, ValueError, OverflowError):
                anchor_matches = False
            observed_provenance_ok = (
                provenance["origin"] == "observed_real"
                and anchor_matches
                and "source_obs_id" in obs
                and np.array_equal(source_ids, expected_ids)
                and np.array_equal(actual_annotations, expected_annotations)
                and actual_state.shape == expected_state.shape
                and np.allclose(
                    actual_state,
                    expected_state,
                    rtol=0,
                    atol=1e-6,
                )
            )
            audit.check(
                observed_provenance_ok,
                f"exact observed slice provenance t={time_value:g}",
                f"origin={provenance['origin']!r}, anchor={provenance['anchor_time']!r}, "
                f"source_ids={len(source_ids)}, expected={len(expected_ids)}",
            )
        else:
            earlier_anchors = [
                float(value)
                for value in DATASETS["zebrafish"]["observed_counts"]
                if float(value) < float(time_value)
            ]
            expected_anchor = max(earlier_anchors) if earlier_anchors else None
            aligned_labels = set(aligned.obs[annotation_key].astype(str))
            source_ids_absent = "source_obs_id" not in obs or bool(
                obs["source_obs_id"].isna().all()
            )
            try:
                anchor_matches = expected_anchor is not None and close_enough(
                    provenance["anchor_time"], expected_anchor
                )
            except (TypeError, ValueError, OverflowError):
                anchor_matches = False
            generated_provenance_ok = (
                provenance["origin"] == "generated_interval_local"
                and anchor_matches
                and source_ids_absent
                and annotations_ok
                and bool(labels)
                and labels.issubset(aligned_labels)
            )
            audit.check(
                generated_provenance_ok,
                f"interval-local generated slice provenance t={time_value:g}",
                f"origin={provenance['origin']!r}, anchor={provenance['anchor_time']!r}, "
                f"expected_anchor={expected_anchor!r}, labels={len(labels)}",
            )
    return int(shape[0]), labels, provenance


def valid_color_map(
    value: Any,
    *,
    required_labels: set[str],
) -> tuple[bool, str]:
    """Reject missing, incomplete, invalid, or single-color label palettes."""

    mapping = _mapping(value)
    colors = [str(mapping.get(label, "")) for label in sorted(required_labels)]
    valid_hex = all(
        len(color) == 7
        and color.startswith("#")
        and all(character in "0123456789abcdefABCDEF" for character in color[1:])
        for color in colors
    )
    distinct = len({color.lower() for color in colors})
    ok = (
        len(required_labels) > 1
        and set(required_labels).issubset(mapping)
        and valid_hex
        and distinct > 1
    )
    return ok, (
        f"required_labels={len(required_labels)}, mapped={len(mapping)}, "
        f"distinct_required_colors={distinct}"
    )


def finite_numeric_frame(frame: pd.DataFrame, *, allow_na: bool = False) -> bool:
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return False
    values = numeric.to_numpy(dtype=float)
    if allow_na:
        return bool(
            np.isfinite(values[~np.isnan(values)]).all() and np.isfinite(values).any()
        )
    return bool(np.isfinite(values).all())


def velocity_component_contract(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    interaction_component: bool,
) -> tuple[bool, str]:
    """Validate the component archive, including an exact no-interaction zero."""

    if not path.is_file():
        return False, "missing"
    expected_keys = {"drift", "interaction", "score", "full", "times", "features"}
    try:
        with np.load(path) as velocity:
            if set(velocity.files) != expected_keys:
                return False, f"keys={tuple(sorted(velocity.files))}"
            arrays = {key: np.asarray(velocity[key]) for key in expected_keys}
    except (OSError, ValueError):
        return False, "unreadable archive"

    vector_keys = ("drift", "interaction", "score", "full")
    shapes_ok = all(arrays[key].shape == expected_shape for key in vector_keys)
    shapes_ok = (
        shapes_ok
        and arrays["features"].shape == expected_shape
        and arrays["times"].shape == (expected_shape[0],)
    )
    finite = all(np.isfinite(value).all() for value in arrays.values())
    identity = np.allclose(
        arrays["full"],
        arrays["drift"] + arrays["interaction"] + arrays["score"],
        rtol=1e-5,
        atol=1e-5,
    )
    nondegenerate = all(
        float(np.ptp(arrays[key])) > 0.0 for key in ("drift", "score", "full")
    )
    if interaction_component:
        interaction_ok = float(np.ptp(arrays["interaction"])) > 0.0
    else:
        interaction_ok = bool(
            np.array_equal(arrays["interaction"], np.zeros_like(arrays["interaction"]))
        )
    ok = shapes_ok and finite and identity and nondegenerate and interaction_ok
    return ok, (
        f"shape={arrays['full'].shape}, times={tuple(np.unique(arrays['times']))}, "
        f"interaction_component={interaction_component}, "
        f"interaction_nonzero={int(np.count_nonzero(arrays['interaction']))}"
    )


def no_interaction_downstream_artifact_cleanliness(
    downstream: Path,
) -> tuple[bool, str]:
    """Reject communication, LR, interaction-panel, and communication-3D files."""

    forbidden: list[Path] = []
    for directory_name in ("communication", "ligand_receptor"):
        directory = downstream / directory_name
        if directory.is_symlink():
            forbidden.append(directory)
        elif directory.is_dir():
            forbidden.extend(path for path in directory.rglob("*") if path.is_file())
    forbidden.extend(
        path for path in downstream.rglob("*interaction*.pdf") if path.is_file()
    )
    communication_3d = downstream / "figures" / "spatiotemporal_communication_3d.html"
    if communication_3d.is_file():
        forbidden.append(communication_3d)
    forbidden = sorted(set(forbidden))
    return (not forbidden), (
        "none"
        if not forbidden
        else ", ".join(str(path.relative_to(downstream)) for path in forbidden)
    )


def communication_edge_selection_contract(
    communication: pd.DataFrame,
    communication_summary: Any,
    sparse_attention_dir: Path,
    *,
    expected_times: Sequence[float],
    expected_node_counts: Mapping[float, int],
    expected_label_sets: Mapping[float, set[str]],
    expected_edge_mode: str,
) -> tuple[bool, str]:
    """Validate per-time sparse attention while allowing learned-gate zeros.

    An empty time point is a valid structural zero only when both sparse arrays
    have their canonical empty shapes.  The run as a whole must still contain
    selected edges and a nonzero cell-type communication summary.
    """

    required_table_columns = {"time", "source", "target", "attention_per_source"}
    table_ok = not communication.empty and required_table_columns.issubset(
        communication.columns
    )
    table_max = math.nan
    if table_ok:
        table_times = pd.to_numeric(communication["time"], errors="coerce").to_numpy(
            dtype=float
        )
        attention_per_source = pd.to_numeric(
            communication["attention_per_source"], errors="coerce"
        ).to_numpy(dtype=float)
        table_max = float(attention_per_source.max())
        source_values = communication["source"].astype("string")
        target_values = communication["target"].astype("string")
        complete_type_grids = True
        for time_value in expected_times:
            time_mask = np.isclose(table_times, float(time_value))
            time_sources = set(source_values[time_mask].tolist())
            time_targets = set(target_values[time_mask].tolist())
            expected_labels = expected_label_sets.get(float(time_value))
            complete_type_grids = complete_type_grids and (
                isinstance(expected_labels, set)
                and bool(expected_labels)
                and time_sources == expected_labels
                and time_targets == expected_labels
                and int(time_mask.sum()) == len(time_sources) ** 2
            )
        table_ok = (
            np.isfinite(table_times).all()
            and np.isfinite(attention_per_source).all()
            and bool((attention_per_source >= 0.0).all())
            and table_max > 0.0
            and not source_values.isna().any()
            and not target_values.isna().any()
            and bool(source_values.str.len().gt(0).all())
            and bool(target_values.str.len().gt(0).all())
            and not communication.duplicated(subset=["time", "source", "target"]).any()
            and set(table_times) == {float(value) for value in expected_times}
            and complete_type_grids
        )
    else:
        table_times = np.empty(0, dtype=float)
        attention_per_source = np.empty(0, dtype=float)

    summary_meta = _mapping(communication_summary)
    edge_selection_raw = summary_meta.get("edge_selection_by_time")
    edge_selection = _mapping(edge_selection_raw)
    expected_keys = tuple(f"{float(time_value):g}" for time_value in expected_times)
    mapping_ok = (
        isinstance(edge_selection_raw, Mapping)
        and len(expected_keys) == len(set(expected_keys))
        and set(edge_selection) == set(expected_keys)
    )

    interpretation_ok = False
    interpretation = summary_meta.get("structural_zero_interpretation")
    interpretation_text = (
        interpretation.lower() if isinstance(interpretation, str) else ""
    )
    if expected_edge_mode == "learned":
        interpretation_ok = (
            "no candidate edge passed the lr-informed learned edge-predictor gate"
            in interpretation_text
            and "no edge was available within the spatial cutoff" in interpretation_text
            and "neither case establishes absence of all biological communication"
            in interpretation_text
        )
    elif expected_edge_mode == "all_spatial":
        interpretation_ok = (
            "no edge was available within the spatial cutoff" in interpretation_text
            and "does not establish absence of all biological communication"
            in interpretation_text
        )

    arrays_ok = True
    records_ok = mapping_ok
    total_selected = 0
    sparse_details: list[str] = []
    for time_value, time_key in zip(expected_times, expected_keys):
        attention_path = sparse_attention_dir / f"attn_mean_interp_t{time_value}.npy"
        edge_path = sparse_attention_dir / f"edge_index_interp_t{time_value}.npy"
        if not attention_path.is_file() or not edge_path.is_file():
            arrays_ok = False
            sparse_details.append(f"t={time_value:g}:missing")
            continue
        try:
            attention = np.load(attention_path, mmap_mode="r")
            edge_index = np.load(edge_path, mmap_mode="r")
        except (OSError, ValueError):
            arrays_ok = False
            sparse_details.append(f"t={time_value:g}:unreadable")
            continue

        canonical = (
            edge_index.ndim == 2
            and edge_index.shape[0] == 2
            and np.issubdtype(edge_index.dtype, np.integer)
            and attention.ndim == 1
            and edge_index.shape[1] == attention.shape[0]
        )
        node_count = expected_node_counts.get(float(time_value))
        valid_indices = (
            canonical
            and isinstance(node_count, int)
            and not isinstance(node_count, bool)
            and node_count > 0
            and (
                edge_index.shape[1] == 0
                or (
                    bool((edge_index >= 0).all())
                    and int(edge_index.max()) < node_count
                    and bool((edge_index[0] != edge_index[1]).all())
                    and np.unique(edge_index.T, axis=0).shape[0] == edge_index.shape[1]
                )
            )
        )
        if valid_indices and expected_edge_mode == "all_spatial":
            directed_edges = {
                (int(source), int(target)) for source, target in edge_index.T
            }
            valid_indices = all(
                (target, source) in directed_edges for source, target in directed_edges
            )
        selected_count = int(attention.shape[0]) if attention.ndim == 1 else -1
        valid_values = (
            valid_indices
            and np.issubdtype(attention.dtype, np.floating)
            and np.isfinite(attention).all()
            and bool((attention >= 0.0).all())
        )
        arrays_ok = arrays_ok and valid_values
        if canonical:
            total_selected += selected_count

        record = _mapping(edge_selection.get(time_key)) if mapping_ok else {}
        candidate_value = record.get("candidate_count")
        selected_value = record.get("selected_count")
        candidate_count = (
            candidate_value
            if isinstance(candidate_value, int)
            and not isinstance(candidate_value, bool)
            else None
        )
        recorded_selected = (
            selected_value
            if isinstance(selected_value, int) and not isinstance(selected_value, bool)
            else None
        )
        fraction_value = record.get("selected_fraction")
        try:
            selected_fraction = (
                math.nan if isinstance(fraction_value, bool) else float(fraction_value)
            )
        except (TypeError, ValueError):
            selected_fraction = math.nan
        expected_fraction = (
            selected_count / candidate_count
            if candidate_count is not None and candidate_count > 0
            else 0.0
        )
        expected_status = (
            "selected_edges"
            if selected_count > 0
            else (
                "no_edges_within_cutoff"
                if candidate_count == 0
                else (
                    "no_edges_passed_learned_gate"
                    if expected_edge_mode == "learned"
                    else "no_edges_within_cutoff"
                )
            )
        )
        record_ok = (
            canonical
            and candidate_count is not None
            and candidate_count >= 0
            and candidate_count % 2 == 0
            and isinstance(node_count, int)
            and candidate_count <= node_count * (node_count - 1)
            and recorded_selected == selected_count
            and recorded_selected <= candidate_count
            and (
                expected_edge_mode != "all_spatial"
                or recorded_selected == candidate_count
            )
            and math.isfinite(selected_fraction)
            and selected_fraction == expected_fraction
            and record.get("status") == expected_status
            and (
                selected_count > 0
                or (
                    table_ok
                    and bool(
                        (
                            attention_per_source[
                                np.isclose(table_times, float(time_value))
                            ]
                            == 0.0
                        ).all()
                    )
                )
            )
        )
        records_ok = records_ok and record_ok
        sparse_details.append(
            f"t={time_value:g}:selected={selected_count},candidate={candidate_count},"
            f"status={record.get('status')!r}"
        )

    dense_attention = list(sparse_attention_dir.glob("attn_interp_t*.npy"))
    arrays_ok = arrays_ok and not dense_attention
    ok = (
        table_ok
        and arrays_ok
        and records_ok
        and total_selected > 0
        and interpretation_ok
    )
    return ok, (
        f"table_max={table_max:.6g}, total_selected={total_selected}, "
        f"summary_keys={tuple(sorted(edge_selection))}, "
        f"interpretation_ok={interpretation_ok}; " + ", ".join(sparse_details)
    )


def downstream_model_contract(
    summary_model: Any,
    simulation: Any,
    *,
    spec: Mapping[str, Any],
    expected_threshold: float | None,
) -> tuple[bool, str]:
    """Validate the exact loaded-model contract for one validation profile."""

    model = _mapping(summary_model)
    contract = _mapping(model.get("scientific_contract"))
    simulation_meta = _mapping(simulation)
    interaction_expected = has_interaction_component(spec)
    recorded_threshold = contract.get("edge_predictor_threshold")
    try:
        threshold_matches = (
            recorded_threshold is None
            if expected_threshold is None
            else recorded_threshold is not None
            and close_enough(recorded_threshold, expected_threshold)
        )
    except (TypeError, ValueError):
        threshold_matches = False
    try:
        if interaction_expected:
            group_size_matches = (
                int(contract.get("interaction_group_size")) == 1024
                and int(contract.get("interaction_group_max_size")) == 2047
                and int(simulation_meta.get("interaction_group_size"))
                == int(contract.get("interaction_group_size"))
                and "remainder"
                in str(contract.get("interaction_group_remainder_policy", ""))
            )
        else:
            group_size_matches = all(
                value is None
                for value in (
                    contract.get("interaction_group_size"),
                    contract.get("interaction_group_max_size"),
                    contract.get("interaction_group_remainder_policy"),
                    simulation_meta.get("interaction_group_size"),
                )
            )
    except (TypeError, ValueError):
        group_size_matches = False
    try:
        cutoff_matches = (
            contract.get("interaction_cutoff") is None
            if not interaction_expected
            else close_enough(contract.get("interaction_cutoff"), spec["cutoff"])
        )
    except (TypeError, ValueError):
        cutoff_matches = False
    expected_components = (
        {"velocity", "growth", "score", "interaction"}
        if interaction_expected
        else {"velocity", "growth", "score"}
    )
    contract_components = {
        str(component).strip().lower() for component in contract.get("components", [])
    }
    explicit_component_profile = "interaction_component" in spec
    component_flag_matches = (
        contract.get("interaction_component") == interaction_expected
        if explicit_component_profile or "interaction_component" in contract
        else True
    )
    components_match = (
        contract_components == expected_components
        if explicit_component_profile or "components" in contract
        else True
    )
    expected_weight_stage = (
        "Finetune" if interaction_expected else "Finetune_no_interaction"
    )
    no_interaction_na_matches = interaction_expected or (
        contract.get("edge_predictor_threshold_check") is None
    )
    ok = (
        contract.get("status") == "matches requested preset"
        and component_flag_matches
        and cutoff_matches
        and contract.get("edge_prior_mode") == spec["edge_prior_mode"]
        and components_match
        and group_size_matches
        and threshold_matches
        and no_interaction_na_matches
        and model.get("weight_stage") == expected_weight_stage
        and model.get("score_stage") == "Score_Refine"
        and contract.get("weight_stage") == expected_weight_stage
        and contract.get("score_stage") == "Score_Refine"
    )
    return ok, (
        f"components={sorted(contract_components)}, interaction_component="
        f"{contract.get('interaction_component')}, mode={contract.get('edge_prior_mode')}, "
        f"cutoff={contract.get('interaction_cutoff')}, "
        f"group_size={contract.get('interaction_group_size')}, "
        f"simulation_group_size={simulation_meta.get('interaction_group_size')}, "
        f"threshold={recorded_threshold}, weight={model.get('weight_stage')}"
    )


def downstream_analysis_status_contract(
    analyses: Any,
    *,
    interaction_component: bool,
) -> tuple[bool, str]:
    """Require productive retained analyses and exact N/A interaction outputs."""

    analysis_meta = _mapping(analyses)
    expected = {
        name: "completed"
        for name in ("velocity", "growth", "composition", "figures", "gene_dynamics")
    }
    expected.update(
        {
            "communication": (
                "completed" if interaction_component else "not applicable"
            ),
            "ligand_receptor": (
                "completed" if interaction_component else "not applicable"
            ),
        }
    )
    actual = {
        name: _mapping(analysis_meta.get(name)).get("status") for name in expected
    }
    metadata_ok = True
    if not interaction_component:
        velocity = _mapping(analysis_meta.get("velocity"))
        vector_status = str(velocity.get("interaction_vector_status", "")).lower()
        figure_3d = _mapping(
            _mapping(analysis_meta.get("figures")).get("spatiotemporal_3d")
        )
        figure_reason = str(figure_3d.get("reason", "")).lower()
        metadata_ok = (
            velocity.get("interaction_component") is False
            and velocity.get("interaction_cutoff") is None
            and "not applicable" in vector_status
            and "zero sentinel" in vector_status
            and str(figure_3d.get("status", "")).lower() == "not applicable"
            and "no interaction component" in figure_reason
        )
    return actual == expected and metadata_ok, (
        f"actual={actual}, expected={expected}, metadata_ok={metadata_ok}"
    )


def validate_downstream(
    audit: Audit,
    run_dir: Path,
    paths: dict[str, Path],
    spec: dict[str, Any],
    threshold: float | None,
    model_dim: int,
    aligned_class_counts: Mapping[str, int],
    allow_legacy_implicit_zero_daughter_noise: bool = False,
) -> None:
    downstream = run_dir / "downstream"
    summary = json.loads(paths["downstream summary"].read_text(encoding="utf-8"))
    observed = tuple(spec["observed_counts"])
    expected_times = tuple(sorted(set(observed + tuple(spec["interpolated"]))))
    actual_times = tuple(float(value) for value in summary["time_points"])
    interaction_expected = has_interaction_component(spec)
    audit.check(
        summary["dataset"] == str(spec.get("artifact_dataset", audit.dataset))
        and int(summary["seed"]) == 42
        and close_enough(summary["alpha_express"], 0.015)
        and int(summary["classifier_k"]) == int(spec["classifier_k"]),
        "downstream scientific constants",
        f"seed={summary['seed']}, alpha={summary['alpha_express']}, k={summary['classifier_k']}",
    )
    audit.check(
        actual_times == expected_times, "downstream time grid", f"times={actual_times}"
    )
    model_ok, model_detail = downstream_model_contract(
        summary.get("model"),
        summary.get("simulation"),
        spec=spec,
        expected_threshold=threshold,
    )
    audit.check(
        model_ok,
        "downstream loaded corrected checkpoint",
        model_detail,
    )
    split_ok, split_detail = classifier_split_contract(
        summary.get("classifier_split"),
        expected_class_counts=aligned_class_counts,
    )
    audit.check(split_ok, "classifier holdout class support", split_detail)
    classifier_metrics = (
        float(summary["classifier_accuracy"]),
        float(summary["classifier_balanced_accuracy"]),
    )
    audit.check(
        all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in classifier_metrics
        ),
        "classifier metrics",
        f"accuracy={classifier_metrics[0]:.6g}, balanced_accuracy={classifier_metrics[1]:.6g}",
    )

    analyses = summary["analyses"]
    analysis_ok, analysis_detail = downstream_analysis_status_contract(
        analyses,
        interaction_component=interaction_expected,
    )
    audit.check(
        analysis_ok,
        "complete standard downstream analyses",
        analysis_detail,
    )
    _, _, _, aligned_uns = read_metadata(paths["aligned H5AD"])
    trained_all_model = _mapping(
        read_metadata(paths["trained AnnData"])[3].get("all_model")
    )
    ignored_input_graph = spec["edge_prior_mode"] in {"all_spatial", "none"} and bool(
        trained_all_model.get("ignored_input_interaction_graph_metadata", False)
    )
    database_ok, database_detail = lr_database_provenance_contract(
        graph_metadata=aligned_uns.get("interaction_graph"),
        graph_metadata_present="interaction_graph" in aligned_uns,
        downstream_analysis=analyses.get("ligand_receptor"),
        spec=spec,
        allow_ignored_input_graph=ignored_input_graph,
    )
    audit.check(
        database_ok,
        "species-matched LR database provenance and role",
        database_detail,
    )
    scope_ok, scope_detail = downstream_scope_contract(
        summary,
        expected_edge_mode=spec["edge_prior_mode"],
        expected_ad_lr_pairs=(
            spec.get("strict_lr_pairs") if interaction_expected else None
        ),
    )
    audit.check(
        scope_ok,
        "readable downstream scientific scope",
        scope_detail,
    )

    slice_counts: dict[float, int] = {}
    slice_labels: dict[float, set[str]] = {}
    slice_provenance: dict[float, dict[str, Any]] = {}
    artifact_dataset = str(spec.get("artifact_dataset", audit.dataset))
    aligned_for_slice_provenance = (
        ad.read_h5ad(paths["aligned H5AD"], backed="r")
        if artifact_dataset == "zebrafish"
        else None
    )
    for time_value in expected_times:
        path = downstream / "slice_data" / f"time_{safe_time_name(time_value)}.h5ad"
        if not path.is_file():
            audit.check(False, f"slice file t={time_value:g}", str(path))
            continue
        (
            slice_counts[time_value],
            slice_labels[time_value],
            slice_provenance[time_value],
        ) = validate_slice(
            audit,
            path,
            spec["annotation_key"],
            model_dim,
            time_value=time_value,
            observed_time=time_value in spec["observed_counts"],
            aligned=aligned_for_slice_provenance,
        )
    if aligned_for_slice_provenance is not None:
        aligned_for_slice_provenance.file.close()
    observed_slice_counts = {time: slice_counts.get(time) for time in observed}
    audit.check(
        observed_slice_counts == spec["observed_counts"],
        "observed states retain real cells",
        f"counts={observed_slice_counts}",
    )
    observed_labels = set().union(*(slice_labels.get(time, set()) for time in observed))
    generated_times = tuple(
        time for time in expected_times if time not in spec["observed_counts"]
    )
    generated_labels = set().union(
        *(slice_labels.get(time, set()) for time in generated_times)
    )
    generated_label_counts = {
        time: len(slice_labels.get(time, set())) for time in generated_times
    }
    audit.check(
        len(observed_labels) > 1
        and bool(generated_label_counts)
        and all(count > 1 for count in generated_label_counts.values()),
        "noncollapsed observed and generated labels",
        f"observed_labels={len(observed_labels)}, generated_per_slice={generated_label_counts}",
    )
    if artifact_dataset == "zebrafish":
        provenance_ok, provenance_detail = slice_provenance_summary_contract(
            summary.get("simulation"),
            slice_provenance=slice_provenance,
            expected_times=expected_times,
        )
        audit.check(
            provenance_ok,
            "zebrafish summary matches per-slice provenance",
            provenance_detail,
        )
        split_sde_ok, split_sde_detail = zebrafish_split_sde_contract(
            summary.get("simulation"),
            slice_counts=slice_counts,
            expected_observed_counts=spec["observed_counts"],
            analyses=analyses,
            allow_legacy_implicit_zero_daughter_noise=(
                allow_legacy_implicit_zero_daughter_noise
            ),
        )
        audit.check(
            split_sde_ok,
            "observed-anchored zebrafish split-SDE contract",
            split_sde_detail,
        )
    color_path = downstream / "label_to_color.json"
    color_mapping = (
        json.loads(color_path.read_text(encoding="utf-8"))
        if color_path.is_file()
        else None
    )
    color_ok, color_detail = valid_color_map(
        color_mapping,
        required_labels=observed_labels | generated_labels,
    )
    audit.check(color_ok, "nondegenerate label color mapping", color_detail)

    velocity_path = downstream / "velocity" / "velocity_components.npz"
    velocity_ok, velocity_detail = velocity_component_contract(
        velocity_path,
        expected_shape=(sum(spec["observed_counts"].values()), model_dim),
        interaction_component=interaction_expected,
    )
    audit.check(
        velocity_ok,
        (
            "finite nondegenerate velocity decomposition"
            if interaction_expected
            else "finite velocity decomposition with exact zero interaction sentinel"
        ),
        velocity_detail,
    )

    growth = pd.read_csv(downstream / "growth" / "growth_by_cell.csv")
    growth_ok = (
        len(growth) == sum(slice_counts.values())
        and finite_numeric_frame(growth)
        and float(growth["growth"].max() - growth["growth"].min()) > 0.0
    )
    audit.check(
        growth_ok,
        "finite nondegenerate growth",
        f"rows={len(growth)}, range=({growth['growth'].min():.6g}, {growth['growth'].max():.6g})",
    )

    composition = pd.read_csv(downstream / "composition" / "celltype_composition.csv")
    fraction_sums = composition.groupby("time")["fraction"].sum()
    composition_ok = (
        not composition.empty
        and finite_numeric_frame(composition)
        and bool((composition["count"] > 0).all())
        and bool(np.allclose(fraction_sums.to_numpy(), 1.0, atol=1e-8))
    )
    audit.check(composition_ok, "cell-type composition", f"rows={len(composition)}")

    if interaction_expected:
        communication_dir = downstream / "communication"
        communication = pd.read_csv(communication_dir / "communication_by_celltype.csv")
        communication_ok, communication_detail = communication_edge_selection_contract(
            communication,
            analyses.get("communication"),
            communication_dir / "sparse_attention",
            expected_times=expected_times,
            expected_node_counts=slice_counts,
            expected_label_sets=slice_labels,
            expected_edge_mode=spec["edge_prior_mode"],
        )
        audit.check(
            communication_ok,
            "sparse nondegenerate communication",
            communication_detail,
        )

    gene_dir = downstream / "gene_dynamics"
    mean_expression = pd.read_csv(gene_dir / "mean_expression.csv", index_col=0)
    signed_expression = pd.read_csv(
        gene_dir / "signed_mean_expression.csv", index_col=0
    )
    diagnostics = pd.read_csv(gene_dir / "reconstruction_diagnostics.csv")
    top_genes = pd.read_csv(gene_dir / "top_variable_genes.csv")
    prototypes = pd.read_csv(gene_dir / "cluster_prototypes.csv", index_col=0)
    gene_ok = (
        not mean_expression.empty
        and mean_expression.shape[1] == len(expected_times)
        and finite_numeric_frame(mean_expression)
        and bool((mean_expression.to_numpy(dtype=float) >= 0).all())
        and finite_numeric_frame(signed_expression)
        and finite_numeric_frame(diagnostics)
        and not top_genes.empty
        and finite_numeric_frame(top_genes)
        and finite_numeric_frame(prototypes)
        and float(np.ptp(mean_expression.to_numpy(dtype=float))) > 0.0
    )
    audit.check(
        gene_ok,
        "finite nondegenerate gene dynamics",
        f"genes={mean_expression.shape[0]}, times={mean_expression.shape[1]}",
    )

    if interaction_expected:
        lr_dir = downstream / "ligand_receptor"
        lr_tables = {
            name: pd.read_csv(lr_dir / f"{name}.csv")
            for name in (
                "pair_timecourse",
                "celltype_timecourse",
                "pattern_summary",
                "coverage",
                "trajectory_coverage",
            )
        }
        lr_ok = all(not frame.empty for frame in lr_tables.values())
        lr_ok = lr_ok and all(
            finite_numeric_frame(frame, allow_na=True) for frame in lr_tables.values()
        )
        pair_times = tuple(
            sorted(
                pd.to_numeric(lr_tables["pair_timecourse"]["time"])
                .astype(float)
                .unique()
            )
        )
        lr_ok = lr_ok and pair_times == expected_times
        retained = lr_tables["trajectory_coverage"].get("retained")
        lr_ok = (
            lr_ok
            and retained is not None
            and bool(retained.astype(str).str.lower().eq("true").any())
        )
        retained_pair_count, retained_pair_detail = retained_top_level_lr_pairs(
            lr_tables["trajectory_coverage"]
        )
        if spec.get("strict_lr_pairs") is not None:
            lr_ok = lr_ok and retained_pair_count == int(spec["strict_lr_pairs"])
            complete_pair_times, pair_grid_detail = complete_lr_pair_time_grid(
                lr_tables["trajectory_coverage"],
                lr_tables["pair_timecourse"],
                expected_times=expected_times,
            )
            lr_ok = lr_ok and complete_pair_times
            retained_pair_detail += (
                f", complete_pair_times={complete_pair_times}, {pair_grid_detail}"
            )
        audit.check(
            lr_ok,
            "strict complete-trajectory LR dynamics",
            f"pair_rows={len(lr_tables['pair_timecourse'])}, "
            f"celltype_rows={len(lr_tables['celltype_timecourse'])}, "
            f"{retained_pair_detail}",
        )
    else:
        clean, clean_detail = no_interaction_downstream_artifact_cleanliness(downstream)
        audit.check(
            clean,
            "no communication, LR, interaction-panel, or communication-3D artifacts",
            clean_detail,
        )

    expected_visuals = [
        downstream / "growth" / "growth_timepoint_grid.pdf",
        downstream / "composition" / "celltype_composition.pdf",
        downstream / "gene_dynamics" / "temporal_gene_programs.pdf",
        downstream / "figures" / "spatial_snapshots" / "timepoint_mosaic.svg",
    ]
    if interaction_expected:
        expected_visuals.append(
            downstream / "figures" / "spatiotemporal_communication_3d.html"
        )
    visuals_ok = all(
        path.is_file() and path.stat().st_size > 0 for path in expected_visuals
    )
    audit.check(
        visuals_ok,
        "standard figure artifacts",
        ", ".join(path.name for path in expected_visuals),
    )


def validate_dataset(run_root: Path, dataset: str) -> Audit:
    audit = Audit(dataset)
    spec = VALIDATION_PROFILES[dataset]
    run_dir = run_root / dataset
    paths = required_files(run_dir, dataset, spec)
    missing = False
    for label, path in paths.items():
        present = path.is_file() and path.stat().st_size > 0
        audit.check(present, label, str(path))
        missing = missing or not present
    if spec["edge_prior_mode"] in {"all_spatial", "none"}:
        clean, cleanliness_detail = validate_ablation_artifact_cleanliness(run_dir)
        audit.check(
            clean,
            "no generated graph, metadata, or predictor artifacts for ablation",
            cleanliness_detail,
        )
    if missing:
        return audit

    n_cells, model_dim, aligned_class_counts = validate_aligned(audit, paths, spec)
    audit.check(
        n_cells == sum(spec["observed_counts"].values()),
        "cohort count total",
        f"n_cells={n_cells}",
    )
    threshold = None
    if spec["edge_prior_mode"] == "learned":
        threshold = validate_edge_predictor(audit, run_dir, paths, spec)
    validate_training(audit, paths, spec, threshold, model_dim)
    legacy_zero_ok = False
    if dataset == "zebrafish":
        legacy_zero_ok, legacy_zero_detail = legacy_zero_daughter_noise_provenance(
            run_root
        )
        if legacy_zero_ok:
            audit.warn(
                "The immutable accepted 4d53ec9 summary omitted "
                "daughter_noise_std; exact manifest/acceptance provenance proves "
                f"the then-hard-coded production value was 0.0 ({legacy_zero_detail})."
            )
    validate_downstream(
        audit,
        run_dir,
        paths,
        spec,
        threshold,
        model_dim,
        aligned_class_counts,
        allow_legacy_implicit_zero_daughter_noise=legacy_zero_ok,
    )
    return audit


def validate_ablation_artifact_cleanliness(run_dir: Path) -> tuple[bool, str]:
    """Reject generated graph/predictor artifacts in either matched ablation.

    A matched aligned H5AD may itself retain immutable main-run graph provenance;
    training must strip it and record that action in the trained AnnData. Separate
    graph, metadata, and predictor files in either ablation run remain forbidden.
    """

    preprocess = run_dir / "preprocess"
    artifacts: list[Path] = []
    for relative in ("edge_classifier", "input_graph", "metadata"):
        root = preprocess / relative
        if root.is_symlink():
            artifacts.append(root)
        elif root.is_dir():
            artifacts.extend(path for path in root.rglob("*") if path.is_file())
    artifacts = sorted(artifacts)
    return (not artifacts), (
        "none"
        if not artifacts
        else ", ".join(str(path.relative_to(preprocess)) for path in artifacts)
    )


# Backward-compatible public name used by earlier validator callers/tests.
validate_no_lr_ablation_artifact_cleanliness = validate_ablation_artifact_cleanliness


def validate_matched_family(
    run_root: Path,
    dataset: str,
    *,
    requested_profiles: Sequence[str],
    individual_audits: Mapping[str, Audit],
    file_hash_cache: dict[Path, tuple[int | None, str | None]] | None = None,
    input_identity_cache: dict[tuple[Path, str], dict[str, Any]] | None = None,
) -> Audit:
    """Accept a formal full/no-LR/no-interaction common-random-number family."""

    audit = Audit(dataset)
    profiles = _matched_profile_names(dataset)
    required_profiles = set(profiles.values())
    requested = set(requested_profiles)
    complete_request = required_profiles <= requested
    audit.check(
        complete_request,
        "complete matched three-arm request",
        f"required={sorted(required_profiles)}, requested={sorted(requested)}",
    )
    individual_pass = complete_request and all(
        profile in individual_audits and not individual_audits[profile].errors
        for profile in required_profiles
    )
    audit.check(
        individual_pass,
        "all matched arms individually accepted",
        ", ".join(
            f"{profile}="
            f"{'PASS' if profile in individual_audits and not individual_audits[profile].errors else 'FAIL/MISSING'}"
            for profile in sorted(required_profiles)
        ),
    )
    if not complete_request:
        return audit

    configs: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    paths_by_arm: dict[str, dict[str, Path]] = {}
    missing: list[str] = []
    for arm, profile in profiles.items():
        run_dir = run_root / profile
        paths = required_files(run_dir, profile, VALIDATION_PROFILES[profile])
        paths_by_arm[arm] = paths
        for label in (
            "aligned H5AD",
            "resolved training config",
            "training run summary",
            "trained AnnData",
        ):
            if not paths[label].is_file():
                missing.append(f"{profile}:{label}")
        for stage in ("Pretrain", "Refine"):
            # The concrete filename is resolved after the config is loaded below.
            if not (paths["resolved training config"].parent / stage).is_dir():
                missing.append(f"{profile}:{stage} checkpoint directory")
    audit.check(
        not missing,
        "matched-family source artifacts",
        "all present" if not missing else ", ".join(missing),
    )
    if missing:
        return audit

    for arm in MATCHED_ABLATION_ARMS:
        paths = paths_by_arm[arm]
        configs[arm] = yaml.safe_load(
            paths["resolved training config"].read_text(encoding="utf-8")
        )
        summaries[arm] = json.loads(
            paths["training run summary"].read_text(encoding="utf-8")
        )

    canonical_configs_ok = True
    canonical_config_details: list[str] = []
    for arm in MATCHED_ABLATION_ARMS:
        learned_edge_path = (
            paths_by_arm[arm].get("generated edge model") if arm == "full" else None
        )
        arm_ok, arm_detail = canonical_matched_config_contract(
            dataset,
            arm,
            configs[arm],
            learned_edge_path=learned_edge_path,
        )
        canonical_configs_ok = canonical_configs_ok and arm_ok
        canonical_config_details.append(f"{arm}: {arm_detail}")
    audit.check(
        canonical_configs_ok,
        "exact current canonical three-arm configs",
        "; ".join(canonical_config_details),
    )
    only_delta_ok, only_delta_detail = matched_config_only_delta_contract(configs)
    audit.check(
        only_delta_ok,
        "matched scientific only-delta config contract",
        only_delta_detail,
    )

    declaration_ok = True
    declaration_detail: list[str] = []
    crn_ok = True
    embedding_ok = True
    for arm in MATCHED_ABLATION_ARMS:
        expected = _expected_matched_ablation(dataset, arm)
        config = configs[arm]
        summary = summaries[arm]
        training = _mapping(summary.get("training"))
        declaration = _mapping(training.get("matched_ablation"))
        condition, interaction_mode = MATCHED_CONDITIONS[arm]
        components = [
            str(value).strip().lower()
            for value in _mapping(config.get("model")).get("components", [])
        ]
        arm_declaration_ok = (
            config.get("matched_ablation") == expected
            and config.get("seed") == MATCHED_ABLATION_SEED
            and declaration
            == {
                "declared": True,
                "config_declaration": expected,
                "normalized": expected,
                "actual_condition": condition,
            }
        )
        declaration_ok = declaration_ok and arm_declaration_ok
        declaration_detail.append(
            f"{arm}={'exact' if arm_declaration_ok else 'invalid'}"
        )
        crn_ok = crn_ok and _valid_common_random_numbers(
            training.get("common_random_numbers"),
            condition=condition,
            interaction_mode=interaction_mode,
            components=components,
        )
        crn_ok = (
            crn_ok
            and training.get("score_energy_objective_default")
            == MATCHED_SCORE_ENERGY_OBJECTIVE
        )
        arm_embedded_ok, _ = training_summary_embedding_contract(
            summary,
            paths_by_arm[arm]["trained AnnData"],
        )
        embedding_ok = embedding_ok and arm_embedded_ok
    audit.check(
        declaration_ok,
        "exact matched-ablation declarations",
        ", ".join(declaration_detail),
    )
    audit.check(
        crn_ok,
        "formal common-random-number protocol declarations",
        f"protocol={MATCHED_ABLATION_PROTOCOL}, seed={MATCHED_ABLATION_SEED}",
    )
    audit.check(
        embedding_ok,
        "standalone summaries bound to all trained AnnData artifacts",
        "JSON-equivalent in all three arms" if embedding_ok else "mismatch or missing",
    )

    hash_cache = file_hash_cache if file_hash_cache is not None else {}
    recompute_cache = input_identity_cache if input_identity_cache is not None else {}
    recorded_h5ad_identities: dict[str, tuple[int, str]] = {}
    selections: dict[str, dict[str, Any]] = {}
    recorded_training_inputs: dict[str, dict[str, Any]] = {}
    recomputed_data_contracts: dict[str, dict[str, Any]] = {}
    input_records_ok = True
    current_inputs_ok = True
    exact_training_inputs_ok = True
    predictor_records_ok = True
    for arm, profile in profiles.items():
        data_record = _mapping(summaries[arm].get("data"))
        h5ad = _mapping(data_record.get("input_h5ad"))
        selection = _mapping(data_record.get("input_selection"))
        model_input = _mapping(data_record.get("model_input"))
        processed_time = _mapping(data_record.get("processed_time"))
        obs_names = _mapping(data_record.get("obs_names"))
        expected_path = paths_by_arm[arm]["aligned H5AD"].resolve()
        recorded_path = _resolved_optional_path(h5ad.get("path"))
        declared_ok = (
            set(h5ad)
            == {
                "source_kind",
                "path",
                "size_bytes",
                "sha256",
                "not_applicable",
                "not_applicable_reason",
            }
            and h5ad.get("source_kind") == "h5ad_path"
            and h5ad.get("not_applicable") is False
            and h5ad.get("not_applicable_reason") is None
            and recorded_path == expected_path
            and isinstance(h5ad.get("size_bytes"), int)
            and not isinstance(h5ad.get("size_bytes"), bool)
            and int(h5ad["size_bytes"]) > 0
            and _is_sha256(h5ad.get("sha256"))
            and _valid_input_selection(selection)
            and _valid_array_identity(model_input)
            and _valid_array_identity(processed_time)
            and _valid_obs_names_identity(obs_names)
        )
        input_records_ok = input_records_ok and declared_ok
        if declared_ok:
            current_size, current_digest = _cached_file_identity(
                expected_path, hash_cache
            )
            bytes_ok = (
                current_size == h5ad["size_bytes"] and current_digest == h5ad["sha256"]
            )
            current_inputs_ok = current_inputs_ok and bytes_ok
            recorded_h5ad_identities[arm] = (
                int(h5ad["size_bytes"]),
                str(h5ad["sha256"]),
            )
            selections[arm] = selection
            recorded_training_inputs[arm] = {
                "model_input": model_input,
                "processed_time": processed_time,
                "obs_names": obs_names,
            }
            cache_key = (
                expected_path,
                json.dumps(selection, sort_keys=True, separators=(",", ":")),
            )
            try:
                if cache_key not in recompute_cache:
                    recompute_cache[cache_key] = recompute_training_input_identities(
                        expected_path,
                        selection,
                    )
                recomputed = recompute_cache[cache_key]
                recomputed_arrays = {
                    key: recomputed[key]
                    for key in ("model_input", "processed_time", "obs_names")
                }
                exact_training_inputs_ok = (
                    exact_training_inputs_ok
                    and recomputed_arrays == recorded_training_inputs[arm]
                )
                recomputed_data_contracts[arm] = {
                    "n_observations": recomputed["n_observations"],
                    "n_timepoints": recomputed["n_timepoints"],
                    "sample_counts_by_timepoint": recomputed[
                        "sample_counts_by_timepoint"
                    ],
                }
            except (OSError, KeyError, TypeError, ValueError) as error:
                exact_training_inputs_ok = False
                audit.warn(
                    f"could not recompute {profile} training input identity: "
                    f"{type(error).__name__}: {error}"
                )
        else:
            current_inputs_ok = False
            exact_training_inputs_ok = False

        edge = _mapping(data_record.get("edge_predictor"))
        edge_schema_ok = set(edge) == {
            "applicable",
            "edge_prior_mode",
            "path",
            "size_bytes",
            "sha256",
            "not_applicable",
            "not_applicable_reason",
            "unchanged_during_training",
        }
        if arm == "full":
            edge_path = paths_by_arm[arm].get("generated edge model")
            edge_size, edge_digest = (
                _cached_file_identity(edge_path.resolve(), hash_cache)
                if edge_path is not None
                else (None, None)
            )
            edge_ok = (
                edge_schema_ok
                and edge_path is not None
                and edge.get("applicable") is True
                and edge.get("edge_prior_mode") == "learned"
                and _resolved_optional_path(edge.get("path")) == edge_path.resolve()
                and edge.get("size_bytes") == edge_size
                and edge.get("sha256") == edge_digest
                and edge.get("not_applicable") is False
                and edge.get("not_applicable_reason") is None
                and edge.get("unchanged_during_training") is True
            )
        else:
            edge_ok = (
                edge_schema_ok
                and edge.get("applicable") is False
                and edge.get("edge_prior_mode") == MATCHED_CONDITIONS[arm][1]
                and edge.get("path") is None
                and edge.get("size_bytes") is None
                and edge.get("sha256") is None
                and edge.get("not_applicable") is True
                and isinstance(edge.get("not_applicable_reason"), str)
                and bool(edge["not_applicable_reason"].strip())
                and edge.get("unchanged_during_training") is True
            )
        predictor_records_ok = predictor_records_ok and edge_ok

    same_h5ad = (
        len(recorded_h5ad_identities) == 3
        and len(set(recorded_h5ad_identities.values())) == 1
    )
    same_selection = len(selections) == 3 and all(
        selection == selections["full"] for selection in selections.values()
    )
    same_training_arrays = len(recorded_training_inputs) == 3 and all(
        identities == recorded_training_inputs["full"]
        for identities in recorded_training_inputs.values()
    )
    summary_data_contract_ok = len(recomputed_data_contracts) == 3 and all(
        _mapping(summaries[arm].get("data")).get("n_observations")
        == recomputed_data_contracts[arm]["n_observations"]
        and _mapping(summaries[arm].get("data")).get("n_timepoints")
        == recomputed_data_contracts[arm]["n_timepoints"]
        and _mapping(summaries[arm].get("data")).get("sample_counts_by_timepoint")
        == recomputed_data_contracts[arm]["sample_counts_by_timepoint"]
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (
                _mapping(summaries[arm].get("data")).get("n_observations"),
                _mapping(summaries[arm].get("data")).get("n_timepoints"),
            )
        )
        and isinstance(
            _mapping(summaries[arm].get("data")).get("sample_counts_by_timepoint"),
            list,
        )
        for arm in MATCHED_ABLATION_ARMS
    )
    audit.check(
        input_records_ok and current_inputs_ok and same_h5ad,
        "exact shared current H5AD bytes",
        f"identities={recorded_h5ad_identities}",
    )
    audit.check(
        same_selection
        and same_training_arrays
        and exact_training_inputs_ok
        and summary_data_contract_ok,
        "exact shared reconstructed training arrays and observation order",
        f"selection_equal={same_selection}, recorded_equal={same_training_arrays}, "
        f"current_recompute={exact_training_inputs_ok}, "
        f"counts_exact={summary_data_contract_ok}",
    )
    audit.check(
        predictor_records_ok,
        "condition-specific edge-predictor byte provenance",
        "learned bytes exact; ablated arms explicitly not applicable",
    )

    expected_implementation = _training_implementation_identity()
    implementations = {
        arm: _mapping(_mapping(summaries[arm].get("training")).get("implementation"))
        for arm in MATCHED_ABLATION_ARMS
    }
    implementation_ok = all(
        implementation == expected_implementation
        for implementation in implementations.values()
    )
    audit.check(
        implementation_ok,
        "exact shared current training implementation",
        f"current_aggregate={expected_implementation['aggregate_sha256']}, "
        f"recorded={[value.get('aggregate_sha256') for value in implementations.values()]}",
    )

    environments = {
        arm: _matched_environment_signature(summaries[arm].get("environment"))
        for arm in MATCHED_ABLATION_ARMS
    }
    environment_ok = all(value is not None for value in environments.values()) and all(
        value == environments["full"] for value in environments.values()
    )
    audit.check(
        environment_ok,
        "shared software and device-model environment",
        "GPU indices ignored; device model and dependency versions exact",
    )

    stage_records_ok = True
    global_rng_equal = True
    private_rng_equal = True
    private_rng_activity_ok = True
    stage_boundary_continuity_ok = True
    stages_by_arm = {arm: summaries[arm].get("stages") for arm in MATCHED_ABLATION_ARMS}
    if any(
        not isinstance(stages, list) or len(stages) != 6
        for stages in stages_by_arm.values()
    ):
        stage_records_ok = False
        global_rng_equal = False
        private_rng_equal = False
        private_rng_activity_ok = False
        stage_boundary_continuity_ok = False
    else:
        for stage_index in range(6):
            stage_triplet = {
                arm: _mapping(stages_by_arm[arm][stage_index])
                for arm in MATCHED_ABLATION_ARMS
            }
            stage_records_ok = stage_records_ok and all(
                stage.get("stage_index") == stage_index
                for stage in stage_triplet.values()
            )
            expected_activity = {
                arm: _configured_stage_interaction_active(configs[arm], stage_index)
                for arm in MATCHED_ABLATION_ARMS
            }
            expected_execution = {
                arm: _expected_stage_execution_contract(
                    configs[arm],
                    recomputed_data_contracts.get(arm, {}).get("n_timepoints"),
                    stage_index,
                )
                for arm in MATCHED_ABLATION_ARMS
            }
            stage_records_ok = stage_records_ok and all(
                expected_execution[arm] is not None
                and all(
                    isinstance(stage_triplet[arm].get(field), int)
                    and not isinstance(stage_triplet[arm].get(field), bool)
                    and stage_triplet[arm].get(field) == expected_execution[arm][field]
                    for field in (
                        "configured_epochs",
                        "recorded_epochs",
                        "batch_size",
                        "optimizer_step_count",
                    )
                )
                and stage_triplet[arm].get("stage") == expected_execution[arm]["stage"]
                and str(stage_triplet[arm].get("mode", "")).lower()
                == expected_execution[arm]["mode"]
                for arm in MATCHED_ABLATION_ARMS
            )
            stage_records_ok = stage_records_ok and all(
                stage_triplet[arm].get("score_energy_objective")
                == (
                    MATCHED_SCORE_ENERGY_OBJECTIVE
                    if _configured_stage_mode(configs[arm], stage_index) == "neural_ode"
                    else None
                )
                and _configured_stage_mode(configs[arm], stage_index) is not None
                for arm in MATCHED_ABLATION_ARMS
            )
            stage_records_ok = stage_records_ok and all(
                stage_triplet[arm].get("interaction_active") is expected_activity[arm]
                and stage_triplet[arm].get("interaction_rng_action")
                == (
                    "consume_private_interaction_generator"
                    if expected_activity[arm]
                    else "inactive_skip_without_rng_advance"
                )
                for arm in MATCHED_ABLATION_ARMS
            )
            for boundary in ("stage_start", "stage_end"):
                boundary_records = {
                    arm: _mapping(
                        _mapping(stage_triplet[arm].get("rng_state_digests")).get(
                            boundary
                        )
                    )
                    for arm in MATCHED_ABLATION_ARMS
                }
                stage_records_ok = stage_records_ok and all(
                    set(_mapping(stage_triplet[arm].get("rng_state_digests")))
                    == {"stage_start", "stage_end"}
                    and set(boundary_records[arm])
                    == {"global", "private_interaction_grouping"}
                    for arm in MATCHED_ABLATION_ARMS
                )
                globals_by_arm = {
                    arm: _mapping(boundary_records[arm].get("global"))
                    for arm in MATCHED_ABLATION_ARMS
                }
                globals_valid = all(
                    _valid_global_rng_record(value) for value in globals_by_arm.values()
                )
                comparable = {
                    arm: _comparable_global_rng(value)
                    for arm, value in globals_by_arm.items()
                }
                global_rng_equal = (
                    global_rng_equal
                    and globals_valid
                    and all(
                        value == comparable["full"] for value in comparable.values()
                    )
                )
                private = {
                    arm: boundary_records[arm].get("private_interaction_grouping")
                    for arm in MATCHED_ABLATION_ARMS
                }
                private_rng_equal = (
                    private_rng_equal
                    and _valid_private_rng_record(private["full"], active=True)
                    and _valid_private_rng_record(private["no_lr_prior"], active=True)
                    and _valid_private_rng_record(
                        private["no_interaction"], active=False
                    )
                    and private["full"] == private["no_lr_prior"]
                )
            for arm in MATCHED_ABLATION_ARMS:
                rng = _mapping(stage_triplet[arm].get("rng_state_digests"))
                start_private = _mapping(
                    _mapping(rng.get("stage_start")).get("private_interaction_grouping")
                )
                end_private = _mapping(
                    _mapping(rng.get("stage_end")).get("private_interaction_grouping")
                )
                configured_epochs = stage_triplet[arm].get("configured_epochs")
                positive_epochs = (
                    isinstance(configured_epochs, int)
                    and not isinstance(configured_epochs, bool)
                    and configured_epochs > 0
                )
                if expected_activity[arm] and positive_epochs:
                    private_rng_activity_ok = (
                        private_rng_activity_ok
                        and start_private.get("state_sha256")
                        != end_private.get("state_sha256")
                    )
                else:
                    private_rng_activity_ok = (
                        private_rng_activity_ok and start_private == end_private
                    )
            if stage_index < 5:
                for arm in MATCHED_ABLATION_ARMS:
                    current_end = _mapping(
                        _mapping(stage_triplet[arm].get("rng_state_digests")).get(
                            "stage_end"
                        )
                    )
                    next_stage = _mapping(stages_by_arm[arm][stage_index + 1])
                    next_start = _mapping(
                        _mapping(next_stage.get("rng_state_digests")).get("stage_start")
                    )
                    stage_boundary_continuity_ok = (
                        stage_boundary_continuity_ok and current_end == next_start
                    )
    audit.check(
        stage_records_ok,
        "matched stage budgets and optimizer-step counts",
        "six positional stages with equal budgets and update counts",
    )
    audit.check(
        global_rng_equal,
        "identical global RNG boundary states across all arms",
        "Python, NumPy, Torch CPU, selected CUDA state, and determinism flags",
    )
    audit.check(
        private_rng_equal,
        "matched private interaction-grouping RNG states",
        "full=no-LR; no-interaction explicitly inactive",
    )
    audit.check(
        private_rng_activity_ok,
        "private RNG advances exactly in configured interaction-active stages",
        "active positive-epoch stages advance; inactive stages do not",
    )
    audit.check(
        stage_boundary_continuity_ok,
        "RNG continuity between adjacent stage boundaries",
        "each stage end equals the next stage start within every arm",
    )

    for stage in ("Pretrain", "Refine"):
        checkpoint_paths = [
            _stage_checkpoint_path(
                paths_by_arm[arm]["resolved training config"].parent,
                configs[arm],
                stage,
            )
            for arm in MATCHED_ABLATION_ARMS
        ]
        checkpoints_ok, checkpoints_detail = retained_checkpoint_contract(
            checkpoint_paths
        )
        audit.check(
            checkpoints_ok,
            f"{stage} retained checkpoint tensors exact across arms",
            checkpoints_detail,
        )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(VALIDATION_PROFILES),
        default=tuple(DATASETS),
        help=(
            "Datasets or explicit ablation profiles to validate "
            "(default: four production main runs)."
        ),
    )
    parser.add_argument(
        "--report", type=Path, help="Optional human-readable JSON report."
    )
    parser.add_argument(
        "--matched-family",
        action="append",
        choices=tuple(DATASETS),
        default=[],
        help=(
            "Repeat for each formal dataset family whose full, no-LR-prior, and "
            "no-interaction arms must pass strict cross-arm acceptance. All three "
            "profiles must also be listed in --datasets."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    audits = []
    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        try:
            audit = validate_dataset(run_root, dataset)
        except Exception as error:
            audit = Audit(dataset)
            audit.check(
                False, "acceptance execution", f"{type(error).__name__}: {error}"
            )
        audits.append(audit)
        for item in audit.checks:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")
        for warning in audit.warnings:
            print(f"[WARN] {warning}")

    audits_by_profile = {audit.dataset: audit for audit in audits}
    matched_audits = []
    file_hash_cache: dict[Path, tuple[int | None, str | None]] = {}
    input_identity_cache: dict[tuple[Path, str], dict[str, Any]] = {}
    for dataset in dict.fromkeys(args.matched_family):
        print(f"\n=== matched family: {dataset} ===")
        try:
            matched_audit = validate_matched_family(
                run_root,
                dataset,
                requested_profiles=args.datasets,
                individual_audits=audits_by_profile,
                file_hash_cache=file_hash_cache,
                input_identity_cache=input_identity_cache,
            )
        except Exception as error:
            matched_audit = Audit(dataset)
            matched_audit.check(
                False,
                "matched-family acceptance execution",
                f"{type(error).__name__}: {error}",
            )
        matched_audits.append(matched_audit)
        for item in matched_audit.checks:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")
        for warning in matched_audit.warnings:
            print(f"[WARN] {warning}")

    report = {
        "run_root": str(run_root),
        "status": (
            "PASS"
            if all(not audit.errors for audit in (*audits, *matched_audits))
            else "FAIL"
        ),
        "datasets": {audit.dataset: audit.as_dict() for audit in audits},
        "matched_families": {
            audit.dataset: audit.as_dict() for audit in matched_audits
        },
    }
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"\nReport: {report_path}")
    print(f"\nOverall: {report['status']}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
