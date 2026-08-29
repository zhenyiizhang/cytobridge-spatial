#!/usr/bin/env python3
"""Official-API adapters for stVCR, STORIES, and MIOFlow.

The adapters consume only immutable training artifacts from the shared input
manifest.  They never open truth artifacts and never derive particle count
from target data.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import inspect
import json
import pickle
import random
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import (  # type: ignore
        METHODS,
        PREDICTION_N,
        ContractError,
        SplitContract,
        TrainingData,
        add_source_to_path,
        canonical_adata,
        dynamic_adapter_implementation_identity,
        environment_provenance,
        git_identity,
        input_provenance,
        load_training_data,
        package_version,
        physical_to_encoded_time,
        read_split_contract,
        same_time,
        sha256_array,
        sha256_file,
        source_time_for_fit,
        stable_seed,
        validate_target,
        write_json,
    )
else:
    from .common import (
        METHODS,
        PREDICTION_N,
        ContractError,
        SplitContract,
        TrainingData,
        add_source_to_path,
        canonical_adata,
        dynamic_adapter_implementation_identity,
        environment_provenance,
        git_identity,
        input_provenance,
        load_training_data,
        package_version,
        physical_to_encoded_time,
        read_split_contract,
        same_time,
        sha256_array,
        sha256_file,
        source_time_for_fit,
        stable_seed,
        validate_target,
        write_json,
    )


DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "stvcr": {
        "n_epochs": 2000,
        "num_samples": 500,
        "use_alignment": False,
        "use_growth": True,
        "device": "cuda",
        "delta_t": 0.1,
        "weights_return_index": None,
    },
    "stories": {
        "max_iter": 100,
        "batch_size": 128,
        "restore": False,
        "keep_checkpoints": True,
    },
    "mioflow": {
        "n_epochs": 100,
        "hidden_dim": 64,
        "learning_rate": 1e-3,
        "lambda_ot": 1.0,
        "lambda_energy": 0.01,
        "sample_size": 512,
        "use_cuda": True,
        "n_bins": 101,
    },
}

OFFICIAL_APIS = {
    "stvcr": {
        "fit": "stvcr.training.train.train_stvcr",
        "infer": "stvcr.downstream.utils.evolution_forward_sim_rgb_data",
    },
    "stories": {
        "fit": "stories.SpaceTime.fit",
        "infer": "stories.SpaceTime.transform",
    },
    "mioflow": {
        "fit": "mioflow.mioflow.MIOFlow.fit",
        "infer": "torchdiffeq.odeint over fitted MIOFlow.ode_model",
    },
}

DEFAULT_PINS = Path(__file__).with_name("method_pins.json")


def official_import_strategy(method: str) -> str:
    """Describe how the exact pinned official API is imported."""

    if method == "stvcr":
        return (
            "exact pinned training/train.py and downstream/utils.py through "
            "minimal namespace packages; eager plotting/preprocessing initializers skipped"
        )
    if method == "mioflow":
        return (
            "exact pinned mioflow/mioflow.py and core modules through minimal "
            "namespace packages; eager unused GAGA/PHATE initializer skipped"
        )
    return "standard import from exact pinned checkout"


def load_method_pins(path: Path = DEFAULT_PINS) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods")
    if payload.get("schema_version") != "1.0.0" or not isinstance(methods, Mapping):
        raise ContractError(f"invalid dynamic method pin registry: {path}")
    if set(methods) != set(METHODS):
        raise ContractError("pin registry must contain exactly stvcr/stories/mioflow")
    for method, pin in methods.items():
        if not isinstance(pin, Mapping):
            raise ContractError(f"{method} pin must be an object")
        commit = str(pin.get("commit", ""))
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise ContractError(f"{method} pin must use a full lowercase git commit")
    return payload


def _pinned_source_identity(
    method: str, source_root: Path, pins_path: Path = DEFAULT_PINS
) -> dict[str, Any]:
    pins_path = Path(pins_path).expanduser().resolve()
    registry = load_method_pins(pins_path)
    pin = registry["methods"][method]
    source = git_identity(source_root)
    if source["source_git_commit"] != pin["commit"]:
        raise ContractError(
            f"{method} source commit mismatch: expected {pin['commit']}, "
            f"found {source['source_git_commit']}"
        )
    package_root = (
        Path(source_root).expanduser().resolve() / pin["package_subdir"]
    ).resolve()
    if not package_root.is_dir():
        raise ContractError(f"{method} pinned package root is missing: {package_root}")
    source.update(
        {
            "source_expected_git_commit": pin["commit"],
            "source_repository_pin": pin["repository"],
            "source_package_root": str(package_root),
            "method_pin_registry": str(pins_path),
            "method_pin_registry_sha256": sha256_file(pins_path),
            "method_pin": dict(pin),
        }
    )
    return source


def representation_contract(method: str, data: TrainingData) -> dict[str, Any]:
    state = {
        "h5ad_key": data.state_key,
        "dimension": data.state_dim,
        "coordinate_system": "original frozen shared state coordinates",
        "adapter_pca_refit": False,
        "gene_expression_used": False,
    }
    if method == "stvcr":
        return {
            "fit_input": {
                "state": state,
                "spatial": {
                    "h5ad_key": data.spatial_key,
                    "dimension": data.spatial_dim,
                    "coordinate_system": "original frozen shared spatial coordinates",
                },
            },
            "output_scope": "native_joint",
            "display_warp_applied": False,
            "alignment_refit": False,
        }
    if method == "stories":
        return {
            "fit_input": {
                "state": state,
                "spatial_for_training_loss": data.spatial_key,
            },
            "output_scope": "native_state",
            "spatial_output_invented": False,
            "display_warp_applied": False,
        }
    return {
        "fit_input": {"state": state},
        "output_scope": "native_state",
        "train_only_transform": "official MIOFlow z transform",
        "inverse_transform_before_export": True,
        "spatial_output_invented": False,
        "display_warp_applied": False,
    }


def _validate_params(method: str, params: Mapping[str, Any]) -> None:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS[method]))
    if unknown:
        raise ContractError(f"unknown {method} parameters: {unknown}")
    if method == "stvcr":
        if bool(params["use_alignment"]):
            raise ContractError(
                "stVCR use_alignment must remain false in the shared coordinate benchmark"
            )
        index = params.get("weights_return_index")
        if index is not None and not isinstance(index, int):
            raise ContractError("stVCR weights_return_index must be an integer or null")


def parse_params(method: str, value: str | None) -> dict[str, Any]:
    params = dict(DEFAULT_PARAMS[method])
    if value:
        candidate = Path(value).expanduser()
        overrides = (
            json.loads(candidate.read_text(encoding="utf-8"))
            if candidate.is_file()
            else json.loads(value)
        )
        if not isinstance(overrides, Mapping):
            raise ContractError("parameter overrides must be a JSON object")
        params.update(overrides)
    _validate_params(method, params)
    return params


def _install_minimal_packages(
    source_root: Path,
    *,
    package_root: Path,
    packages: Sequence[tuple[str, Path]],
    required: Sequence[Path],
    label: str,
) -> None:
    """Install namespace packages without executing unrelated eager initializers."""

    source_root = Path(source_root).expanduser().resolve()
    package_root = package_root.resolve()
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise ContractError(f"pinned {label} source is incomplete: {missing}")
    try:
        package_root.relative_to(source_root)
    except ValueError as exc:
        raise ContractError(
            f"{label} package root escapes pinned source checkout: {package_root}"
        ) from exc

    for name, raw_path in packages:
        path = raw_path.resolve()
        existing = sys.modules.get(name)
        if existing is not None:
            existing_paths = {
                str(Path(item).resolve()) for item in getattr(existing, "__path__", [])
            }
            if str(path) not in existing_paths:
                raise ContractError(
                    f"refusing mixed {label} import: {name} is already loaded from "
                    f"{sorted(existing_paths)} instead of {path}"
                )
            continue
        module = types.ModuleType(name)
        module.__file__ = str(path / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(path)]
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        module.__spec__ = spec
        module.__dict__["__cytobridge_minimal_import__"] = True
        sys.modules[name] = module
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            setattr(sys.modules[parent_name], child_name, module)


def _install_stvcr_minimal_packages(source_root: Path) -> None:
    source_root = Path(source_root).expanduser().resolve()
    candidate = source_root / "src" / "stvcr"
    package_root = candidate if candidate.is_dir() else source_root / "stvcr"
    _install_minimal_packages(
        source_root,
        package_root=package_root,
        packages=(
            ("stvcr", package_root),
            ("stvcr.training", package_root / "training"),
            ("stvcr.downstream", package_root / "downstream"),
        ),
        required=(
            package_root / "training" / "train.py",
            package_root / "downstream" / "utils.py",
        ),
        label="stVCR",
    )


def _import_stvcr(source_root: Path):
    _install_stvcr_minimal_packages(source_root)
    train_module = importlib.import_module("stvcr.training.train")
    downstream = importlib.import_module("stvcr.downstream.utils")
    package = sys.modules["stvcr"]
    return (
        package,
        train_module.default_config,
        train_module.train_stvcr,
        downstream.evolution_forward_sim_rgb_data,
    )


def _import_stories():
    return importlib.import_module("stories"), importlib.import_module("jax")


def _install_mioflow_minimal_packages(source_root: Path) -> None:
    source_root = Path(source_root).expanduser().resolve()
    package_root = source_root / "mioflow"
    _install_minimal_packages(
        source_root,
        package_root=package_root,
        packages=(
            ("mioflow", package_root),
            ("mioflow.core", package_root / "core"),
            ("mioflow.core.models", package_root / "core" / "models"),
        ),
        required=(
            package_root / "mioflow.py",
            package_root / "core" / "datasets.py",
            package_root / "core" / "models" / "ode_model.py",
        ),
        label="MIOFlow",
    )


def _import_mioflow(source_root: Path):
    _install_mioflow_minimal_packages(source_root)
    name = "mioflow.mioflow"
    module = importlib.import_module(name)
    return module.MIOFlow, name, module


def _source_file(callable_object: Any, source_root: Path, label: str) -> str:
    raw = inspect.getsourcefile(callable_object)
    if raw is None:
        raise ContractError(f"cannot locate official source file for {label}")
    path = Path(raw).resolve()
    root = Path(source_root).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContractError(
            f"official {label} resolved outside pinned source checkout: {path}"
        ) from exc
    return str(path)


def _official_identity(method: str, source_root: Path) -> dict[str, Any]:
    if method == "stvcr":
        package, _, fit_api, infer_api = _import_stvcr(source_root)
        version = package_version(package, ("stvcr",))
        signatures = {
            "fit": str(inspect.signature(fit_api)),
            "infer": str(inspect.signature(infer_api)),
        }
        source_files = {
            "fit": _source_file(fit_api, source_root, "stVCR fit API"),
            "infer": _source_file(infer_api, source_root, "stVCR infer API"),
        }
    elif method == "stories":
        package, _ = _import_stories()
        if not hasattr(package, "SpaceTime"):
            raise AttributeError("official STORIES package lacks SpaceTime")
        version = package_version(package, ("stories",))
        signatures = {
            "fit": str(inspect.signature(package.SpaceTime.fit)),
            "infer": str(inspect.signature(package.SpaceTime.transform)),
        }
        source_files = {
            "fit": _source_file(package.SpaceTime.fit, source_root, "STORIES fit API"),
            "infer": _source_file(
                package.SpaceTime.transform, source_root, "STORIES infer API"
            ),
        }
    else:
        cls, imported_as, module = _import_mioflow(source_root)
        importlib.import_module("torchdiffeq")
        version = package_version(module, ("mioflow", "MIOFlow"))
        signatures = {
            "fit": str(inspect.signature(cls.fit)),
            "imported_as": imported_as,
        }
        source_files = {"fit": _source_file(cls.fit, source_root, "MIOFlow fit API")}
    return {
        "official_api": OFFICIAL_APIS[method],
        "official_api_signatures": signatures,
        "official_import_strategy": official_import_strategy(method),
        "official_source_files": source_files,
        "method_version": version,
        "environment": environment_provenance(method, version),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _dump_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _new_output_dir(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_roster(
    output_dir: Path,
    data: TrainingData,
    source_time: float,
    split: SplitContract,
) -> tuple[Path, dict[str, Any]]:
    if sha256_file(split.source_roster_npz) != split.source_roster_declared_sha256:
        raise ContractError("canonical source roster SHA-256 differs from manifest")
    with np.load(split.source_roster_npz, allow_pickle=False) as archive:
        required = {"indices", "row_id", "source_time", "spatial", "state"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ContractError(f"canonical source roster lacks keys {missing}")
        indices = np.asarray(archive["indices"], dtype=np.int64)
        row_id = np.asarray(archive["row_id"]).astype(str)
        roster_time = float(np.asarray(archive["source_time"]).reshape(-1)[0])
        spatial = np.asarray(archive["spatial"], dtype=np.float32)
        state = np.asarray(archive["state"], dtype=np.float32)
    if (
        indices.shape != (PREDICTION_N,)
        or row_id.shape != (PREDICTION_N,)
        or spatial.shape != (PREDICTION_N, data.spatial_dim)
        or state.shape != (PREDICTION_N, data.state_dim)
        or np.any(indices < 0)
        or np.any(indices >= data.n_obs)
        or not same_time(roster_time, source_time)
    ):
        raise ContractError(
            "canonical source roster has invalid shape, indices, or time"
        )
    if not np.array_equal(row_id, data.row_id[indices]):
        raise ContractError("canonical source roster row IDs differ from training rows")
    if not np.all(np.isclose(data.time[indices], source_time, rtol=0.0, atol=1e-8)):
        raise ContractError("canonical source roster includes rows outside source time")
    if not np.allclose(spatial, data.spatial[indices], rtol=1e-6, atol=1e-6):
        raise ContractError(
            "canonical source roster spatial values differ from training rows"
        )
    if not np.allclose(state, data.state[indices], rtol=1e-6, atol=1e-6):
        raise ContractError(
            "canonical source roster state values differ from training rows"
        )
    path = output_dir / "source_roster.npz"
    shutil.copy2(split.source_roster_npz, path)
    return path, {
        "source_time": float(source_time),
        "initial_n": PREDICTION_N,
        "canonical_input_roster": str(split.source_roster_npz),
        "canonical_input_roster_sha256": split.source_roster_declared_sha256,
        "source_roster_sha256": sha256_file(path),
        "source_indices_sha256": sha256_array(indices),
        "source_row_id_sha256": sha256_array(row_id.astype("U")),
        "source_available_n": int(
            np.count_nonzero(np.isclose(data.time, source_time, rtol=0.0, atol=1e-8))
        ),
        "sampled_with_replacement": bool(len(np.unique(indices)) < len(indices)),
        "shared_across_dynamic_methods": True,
        "shared_across_all_benchmark_families": True,
    }


def _load_roster(fit_manifest: Mapping[str, Any], data: TrainingData) -> dict[str, Any]:
    artifact = fit_manifest["artifacts"]["source_roster"]
    path = Path(artifact["path"])
    if sha256_file(path) != artifact["sha256"]:
        raise ContractError("frozen source roster changed after fitting")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    indices = np.asarray(result["indices"], dtype=np.int64)
    if (
        indices.shape != (PREDICTION_N,)
        or np.any(indices < 0)
        or np.any(indices >= data.n_obs)
    ):
        raise ContractError("frozen source roster indices are invalid")
    source_time = float(np.asarray(result["source_time"]).reshape(-1)[0])
    checks = (
        ("row_id", np.asarray(result["row_id"]).astype(str), data.row_id[indices]),
        (
            "spatial",
            np.asarray(result["spatial"], dtype=np.float32),
            data.spatial[indices],
        ),
        ("state", np.asarray(result["state"], dtype=np.float32), data.state[indices]),
    )
    for label, observed, expected in checks:
        equal = (
            np.array_equal(observed, expected)
            if label == "row_id"
            else np.allclose(observed, expected, rtol=1e-6, atol=1e-6)
        )
        if observed.shape != expected.shape or not equal:
            raise ContractError(
                f"frozen source roster {label} differs from training input"
            )
    if not np.all(np.isclose(data.time[indices], source_time, rtol=0.0, atol=1e-8)):
        raise ContractError("frozen source roster spans more than one source stage")
    result["indices"] = indices
    result["source_time"] = source_time
    return result


def preflight(
    method: str,
    input_manifest: Path,
    split_id: str,
    source_root: Path,
    *,
    check_import: bool = True,
    pins_path: Path = DEFAULT_PINS,
) -> dict[str, Any]:
    split = read_split_contract(input_manifest, split_id)
    data = load_training_data(split)
    source = _pinned_source_identity(method, source_root, pins_path)
    add_source_to_path(source_root)
    official = (
        _official_identity(method, source_root)
        if check_import
        else {
            "official_api": OFFICIAL_APIS[method],
            "method_version": None,
            "import_check_skipped": True,
        }
    )
    source_time = source_time_for_fit(split)
    return {
        "status": "ok",
        "phase": "preflight",
        "method": method,
        "dataset": split.dataset_id,
        "split_id": split.split_id,
        "regime": split.regime,
        "track": split.regime,
        "observed_times": list(split.observed_times),
        "n_train": data.n_obs,
        "state_dim": data.state_dim,
        "spatial_dim": data.spatial_dim,
        "output_scope": representation_contract(method, data)["output_scope"],
        "primary_benchmark_eligible": True,
        "target_plan": [
            {
                "source_time": float(source_time),
                "target_time": float(target),
                "initial_n": PREDICTION_N,
                "expected_output_n": (
                    "native_growth" if method == "stvcr" else PREDICTION_N
                ),
            }
            for target in split.evaluation_targets
        ],
        "source_policy": (
            "one frozen initial-stage roster reused for every full-data target"
            if split.regime == "full_data"
            else "one frozen nearest-previous-stage roster for target-absent fold"
        ),
        "representation_contract": representation_contract(method, data),
        **source,
        **official,
        **input_provenance(split),
    }


def fit_method(
    method: str,
    input_manifest: Path,
    split_id: str,
    source_root: Path,
    output_dir: Path,
    *,
    seed: int,
    params: Mapping[str, Any],
    dry_run: bool = False,
    pins_path: Path = DEFAULT_PINS,
) -> dict[str, Any]:
    _validate_params(method, params)
    split = read_split_contract(input_manifest, split_id)
    data = load_training_data(split)
    source = _pinned_source_identity(method, source_root, pins_path)
    import_root = add_source_to_path(source_root)
    source_time = source_time_for_fit(split)
    fit_seed = stable_seed(seed, split.dataset_id, split.split_id, method, "fit")
    plan: dict[str, Any] = {
        "status": "dry-run" if dry_run else "complete",
        "phase": "fit",
        "method": method,
        "dataset": split.dataset_id,
        "split_id": split.split_id,
        "regime": split.regime,
        "observed_times": list(split.observed_times),
        "evaluation_targets": list(split.evaluation_targets),
        "holdout_time": split.holdout_time,
        "target_time": split.holdout_time if split.regime == "loto" else None,
        "n_train": data.n_obs,
        "initial_n": PREDICTION_N,
        "source_time": float(source_time),
        "source_policy": (
            "fixed t0/initial-stage bootstrap reused for all targets; no segment reset"
            if split.regime == "full_data"
            else "fixed nearest-previous-stage bootstrap for this target-removed fold"
        ),
        "seed_base": int(seed),
        "fit_seed": int(fit_seed),
        "params": dict(params),
        "official_api": OFFICIAL_APIS[method],
        "output_scope": representation_contract(method, data)["output_scope"],
        "primary_benchmark_eligible": True,
        "representation_contract": representation_contract(method, data),
        "adapter_implementation": dynamic_adapter_implementation_identity(),
        "source_import_root": str(import_root),
        **source,
        **input_provenance(split),
    }
    if dry_run:
        return plan

    output_dir = _new_output_dir(output_dir)
    roster_path, roster_summary = _save_roster(
        output_dir,
        data,
        source_time,
        split,
    )
    _seed_everything(fit_seed)
    adata = canonical_adata(data)
    if method == "stvcr":
        import torch

        package, default_config, train_stvcr, infer_api = _import_stvcr(source_root)
        method_version = package_version(package, ("stvcr",))
        api_signatures = {
            "fit": str(inspect.signature(train_stvcr)),
            "infer": str(inspect.signature(infer_api)),
        }
        official_source_files = {
            "fit": _source_file(train_stvcr, source_root, "stVCR fit API"),
            "infer": _source_file(infer_api, source_root, "stVCR infer API"),
        }
        config = dict(default_config)
        config.update(
            {
                "n_epochs": int(params["n_epochs"]),
                "num_samples": int(params["num_samples"]),
            }
        )
        requested = torch.device(str(params["device"]))
        device = (
            requested
            if requested.type == "cpu" or torch.cuda.is_available()
            else torch.device("cpu")
        )
        model_path = output_dir / "model.pt"
        rigid_path = output_dir / "rigid_transform.pt"
        train_stvcr(
            adata,
            model_path=str(model_path),
            rigid_transformation_path=str(rigid_path),
            config=config,
            use_alignment=False,
            use_growth=bool(params["use_growth"]),
            device=device,
        )
        fitted_path = output_dir / "fitted_train.h5ad"
        adata.write_h5ad(fitted_path)
        method_artifacts = {
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "rigid_transform": {
                "path": str(rigid_path),
                "sha256": sha256_file(rigid_path),
            },
            "fitted_train": {
                "path": str(fitted_path),
                "sha256": sha256_file(fitted_path),
            },
            "device": str(device),
        }
    elif method == "stories":
        stories, jax = _import_stories()
        method_version = package_version(stories, ("stories",))
        api_signatures = {
            "fit": str(inspect.signature(stories.SpaceTime.fit)),
            "infer": str(inspect.signature(stories.SpaceTime.transform)),
        }
        official_source_files = {
            "fit": _source_file(stories.SpaceTime.fit, source_root, "STORIES fit API"),
            "infer": _source_file(
                stories.SpaceTime.transform, source_root, "STORIES infer API"
            ),
        }
        model = stories.SpaceTime(proximal_step=stories.steps.ExplicitStep())
        checkpoint_dir = output_dir / "checkpoints"
        model.fit(
            adata,
            time_key="time",
            omics_key="X_stories",
            space_key="spatial",
            batch_size=int(params["batch_size"]),
            max_iter=int(params["max_iter"]),
            checkpoint_manager=str(checkpoint_dir.resolve()),
            key=jax.random.PRNGKey(fit_seed),
            restore=bool(params["restore"]),
        )
        model_path = output_dir / "model.pkl"
        _dump_pickle(model_path, model)
        if not bool(params["keep_checkpoints"]) and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        method_artifacts = {
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "checkpoint_dir": str(checkpoint_dir),
        }
    else:
        MIOFlow, imported_as, module = _import_mioflow(source_root)
        method_version = package_version(module, ("mioflow", "MIOFlow"))
        api_signatures = {
            "fit": str(inspect.signature(MIOFlow.fit)),
            "imported_as": imported_as,
        }
        official_source_files = {
            "fit": _source_file(MIOFlow.fit, source_root, "MIOFlow fit API")
        }
        model = MIOFlow(
            adata,
            gaga_model=None,
            gaga_input_key="X_pca",
            obs_time_key="time",
            hidden_dim=int(params["hidden_dim"]),
            use_cuda=bool(params["use_cuda"]),
            use_sde=False,
            n_epochs=int(params["n_epochs"]),
            lambda_ot=float(params["lambda_ot"]),
            lambda_energy=float(params["lambda_energy"]),
            learning_rate=float(params["learning_rate"]),
            sample_size=int(params["sample_size"]),
            exp_dir=str(output_dir),
            n_trajectories=PREDICTION_N,
            n_bins=int(params["n_bins"]),
        )
        model.fit()
        mean = np.asarray(model.mean_vals, dtype=np.float64).reshape(-1)
        std = np.asarray(model.std_vals, dtype=np.float64).reshape(-1)
        if (
            mean.shape != (data.state_dim,)
            or std.shape != (data.state_dim,)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or np.any(std <= 0)
        ):
            raise ContractError("MIOFlow exposed an invalid train-only state transform")
        transform_path = output_dir / "state_transform.npz"
        np.savez_compressed(
            transform_path,
            mean=mean,
            std=std,
            fitted_on=np.asarray(["training_rows_only"]),
        )
        model_path = output_dir / "model.pkl"
        _dump_pickle(model_path, model)
        method_artifacts = {
            "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
            "state_transform": {
                "path": str(transform_path),
                "sha256": sha256_file(transform_path),
                "semantics": "official train-only z transform; inverted before export",
            },
            "imported_as": imported_as,
        }

    plan["method_version"] = method_version
    plan["official_api_signatures"] = api_signatures
    plan["official_import_strategy"] = official_import_strategy(method)
    plan["official_source_files"] = official_source_files
    plan["environment"] = environment_provenance(method, method_version)
    plan["source_roster"] = roster_summary
    plan["artifacts"] = {
        "source_roster": {
            "path": str(roster_path),
            "sha256": sha256_file(roster_path),
        },
        **method_artifacts,
    }
    manifest_path = output_dir / "fit_manifest.json"
    summary_path = output_dir / "summary.json"
    plan["fit_manifest"] = str(manifest_path)
    plan["summary"] = str(summary_path)
    write_json(manifest_path, plan)
    write_json(summary_path, plan)
    return plan


def _load_fit_manifest(fit_dir: Path, method: str, split_id: str) -> dict[str, Any]:
    path = Path(fit_dir).expanduser().resolve() / "fit_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "complete"
        or payload.get("method") != method
        or payload.get("split_id") != split_id
    ):
        raise ContractError("fit manifest does not match requested method/split")
    return payload


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_weights(
    outputs: Sequence[Any], index: int | None, native_n: int
) -> np.ndarray | None:
    if index is None:
        return None
    if index < 0:
        index += len(outputs)
    if index < 0 or index >= len(outputs):
        raise ContractError(
            f"stVCR weights_return_index is outside simulator output length {len(outputs)}"
        )
    candidate = outputs[index]
    if isinstance(candidate, (list, tuple)):
        candidate = candidate[-1]
    weights = _to_numpy(candidate).reshape(-1).astype(np.float32)
    if (
        weights.shape != (native_n,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ContractError("explicit stVCR weights are invalid for native output rows")
    return weights


def infer_method(
    method: str,
    input_manifest: Path,
    split_id: str,
    source_root: Path,
    fit_dir: Path,
    target_time: float,
    output_dir: Path,
    *,
    seed: int,
    dry_run: bool = False,
    pins_path: Path = DEFAULT_PINS,
) -> dict[str, Any]:
    split = read_split_contract(input_manifest, split_id)
    target = validate_target(split, target_time)
    data = load_training_data(split)
    fit_dir = Path(fit_dir).expanduser().resolve()
    fit_manifest_path = fit_dir / "fit_manifest.json"
    fit_manifest = _load_fit_manifest(fit_dir, method, split_id)
    adapter_implementation = dynamic_adapter_implementation_identity()
    if fit_manifest.get("adapter_implementation") != adapter_implementation:
        raise ContractError("dynamic adapter implementation changed after fitting")
    expected_fit_seed = stable_seed(
        seed, split.dataset_id, split.split_id, method, "fit"
    )
    if (
        fit_manifest.get("dataset") != split.dataset_id
        or fit_manifest.get("regime") != split.regime
        or isinstance(fit_manifest.get("fit_seed"), bool)
        or fit_manifest.get("fit_seed") != expected_fit_seed
        or fit_manifest.get("seed_base") != int(seed)
    ):
        raise ContractError("fit manifest dataset/regime/seed contract changed")
    source = _pinned_source_identity(method, source_root, pins_path)
    if source["source_git_commit"] != fit_manifest["source_git_commit"]:
        raise ContractError("official source commit differs from fitted commit")
    if split.root_manifest_sha256 != fit_manifest["input_manifest_sha256"]:
        raise ContractError("root input manifest changed after fitting")
    if split.train_h5ad_declared_sha256 != fit_manifest["train_h5ad_sha256"]:
        raise ContractError("training H5AD reference changed after fitting")
    if (
        split.training_reference_declared_sha256
        != fit_manifest["training_reference_sha256"]
    ):
        raise ContractError("training reference changed after fitting")
    add_source_to_path(source_root)
    roster = _load_roster(fit_manifest, data)
    source_time = float(roster["source_time"])
    expected_source = source_time_for_fit(split)
    if not same_time(source_time, expected_source):
        raise ContractError("fit source roster uses the wrong stage")
    infer_seed = stable_seed(
        seed,
        split.dataset_id,
        split.split_id,
        method,
        "full_trajectory" if split.regime == "full_data" else "loto_trajectory",
    )
    params = fit_manifest["params"]
    plan: dict[str, Any] = {
        "status": "dry-run" if dry_run else "complete",
        "phase": "infer",
        "method": method,
        "dataset": split.dataset_id,
        "split_id": split.split_id,
        "regime": split.regime,
        "track": split.regime,
        "source_time": source_time,
        "target_time": target,
        "initial_n": PREDICTION_N,
        "prediction_n_contract": PREDICTION_N,
        "prediction_n_policy": "fixed in train contract; no truth/target_n access",
        "start_policy": (
            "same frozen t0/initial-stage roster for every target; direct inference"
            if split.regime == "full_data"
            else "frozen nearest-previous-stage roster from target-removed fold"
        ),
        "source_roster_sha256": fit_manifest["artifacts"]["source_roster"]["sha256"],
        "source_indices_sha256": fit_manifest["source_roster"]["source_indices_sha256"],
        "source_row_id_sha256": fit_manifest["source_roster"]["source_row_id_sha256"],
        "source_roster_shared_across_dynamic_methods": fit_manifest["source_roster"][
            "shared_across_dynamic_methods"
        ],
        "seed_base": int(seed),
        "infer_seed": int(infer_seed),
        "shared_full_trajectory_seed_across_targets": split.regime == "full_data",
        "params": params,
        "official_api": OFFICIAL_APIS[method],
        "primary_benchmark_eligible": True,
        "official_api_signatures": fit_manifest.get("official_api_signatures"),
        "official_import_strategy": fit_manifest.get("official_import_strategy"),
        "official_source_files": fit_manifest.get("official_source_files"),
        "method_version": fit_manifest.get("method_version"),
        "environment": fit_manifest.get("environment"),
        "representation_contract": representation_contract(method, data),
        "adapter_implementation": adapter_implementation,
        "fit_manifest": str(fit_manifest_path),
        "fit_manifest_sha256": sha256_file(fit_manifest_path),
        **source,
        **input_provenance(split),
    }
    if dry_run:
        return plan

    _seed_everything(infer_seed)
    source_state = np.asarray(roster["state"], dtype=np.float32)
    source_spatial = np.asarray(roster["spatial"], dtype=np.float32)
    weights: np.ndarray | None = None
    weight_semantics: str | None = None
    if method == "stvcr":
        import anndata as ad
        import torch

        package, _, _, simulator = _import_stvcr(source_root)
        current_method_version = package_version(package, ("stvcr",))
        artifacts = fit_manifest["artifacts"]
        fitted_path = Path(artifacts["fitted_train"]["path"])
        if sha256_file(fitted_path) != artifacts["fitted_train"]["sha256"]:
            raise ContractError("fitted stVCR training artifact changed")
        fitted = ad.read_h5ad(fitted_path)
        try:
            if "row_id" not in fitted.obs or not np.array_equal(
                fitted.obs["row_id"].astype(str).to_numpy(), data.row_id
            ):
                raise ContractError(
                    "fitted stVCR row order differs from training input"
                )
            indices = np.asarray(roster["indices"], dtype=np.int64)
            initial = np.concatenate(
                [
                    np.asarray(
                        fitted.obsm["X_spatial_aligned"][indices], dtype=np.float32
                    ),
                    np.asarray(fitted.obsm["X_gene_input"][indices], dtype=np.float32),
                ],
                axis=1,
            )
            if not np.allclose(
                initial[:, : data.spatial_dim], source_spatial, rtol=1e-6, atol=1e-6
            ) or not np.allclose(
                initial[:, data.spatial_dim :], source_state, rtol=1e-6, atol=1e-6
            ):
                raise ContractError(
                    "stVCR fitted source coordinates differ from the frozen shared roster"
                )
        finally:
            if getattr(fitted, "file", None) is not None:
                fitted.file.close()
        model_path = Path(artifacts["model"]["path"])
        if sha256_file(model_path) != artifacts["model"]["sha256"]:
            raise ContractError("fitted stVCR model changed")
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        outputs = simulator(
            torch.from_numpy(initial),
            model,
            source_time,
            target,
            spatial_dim=data.spatial_dim,
            delta_t=float(params["delta_t"]),
        )
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            raise ContractError("official stVCR simulator returned an invalid object")
        spatial_series, state_series = outputs[0], outputs[1]
        prediction_spatial = _to_numpy(spatial_series[-1]).astype(np.float32)
        prediction_state = _to_numpy(state_series[-1]).astype(np.float32)
        if prediction_state.ndim != 2 or prediction_spatial.ndim != 2:
            raise ContractError("stVCR prediction must contain 2D state/spatial arrays")
        native_n = int(prediction_state.shape[0])
        if (
            native_n <= 0
            or prediction_state.shape != (native_n, data.state_dim)
            or prediction_spatial.shape != (native_n, data.spatial_dim)
        ):
            raise ContractError("stVCR returned inconsistent native output shapes")
        weights = _extract_weights(
            outputs, params.get("weights_return_index"), native_n
        )
        if weights is None and bool(params["use_growth"]):
            weights = np.full(native_n, 1.0 / PREDICTION_N, dtype=np.float32)
            weight_semantics = (
                "uniform initial-particle mass 1/5000; sum(weights)=native_n/5000"
            )
        elif weights is not None:
            weight_semantics = (
                "explicit native weights from audited simulator return index"
            )
        elif native_n != PREDICTION_N:
            raise ContractError(
                "stVCR without growth/weights must preserve the 5000 input rows"
            )
        output_scope = "native_joint"
    elif method == "stories":
        stories, jax = _import_stories()
        current_method_version = package_version(stories, ("stories",))
        model_artifact = fit_manifest["artifacts"]["model"]
        model_path = Path(model_artifact["path"])
        if sha256_file(model_path) != model_artifact["sha256"]:
            raise ContractError("fitted STORIES model changed")
        model = _load_pickle(model_path)
        source_data = TrainingData(
            time=np.full(PREDICTION_N, source_time, dtype=np.float64),
            spatial=source_spatial,
            state=source_state,
            row_id=np.asarray([f"bootstrap-{index}" for index in range(PREDICTION_N)]),
            state_key=data.state_key,
            spatial_key=data.spatial_key,
            time_key=data.time_key,
            row_id_key=data.row_id_key,
            h5ad_contract=data.h5ad_contract,
        )
        prediction_state = np.asarray(
            model.transform(
                canonical_adata(source_data),
                omics_key="X_stories",
                tau=float(target - source_time),
                batch_size=int(params["batch_size"]),
                key=jax.random.PRNGKey(infer_seed),
            ),
            dtype=np.float32,
        )
        if prediction_state.shape != (PREDICTION_N, data.state_dim):
            raise ContractError(
                "official STORIES transform did not preserve the contracted state shape"
            )
        prediction_spatial = None
        output_scope = "native_state"
        native_n = int(prediction_state.shape[0])
    else:
        import torch
        from torchdiffeq import odeint

        _, _, mioflow_module = _import_mioflow(source_root)
        current_method_version = package_version(mioflow_module, ("mioflow", "MIOFlow"))
        model_artifact = fit_manifest["artifacts"]["model"]
        model_path = Path(model_artifact["path"])
        if sha256_file(model_path) != model_artifact["sha256"]:
            raise ContractError("fitted MIOFlow model changed")
        model = _load_pickle(model_path)
        if getattr(model, "use_sde", False):
            raise ContractError(
                "this adapter supports official deterministic MIOFlow only"
            )
        transform_artifact = fit_manifest["artifacts"]["state_transform"]
        transform_path = Path(transform_artifact["path"])
        if sha256_file(transform_path) != transform_artifact["sha256"]:
            raise ContractError("MIOFlow train-only transform changed")
        with np.load(transform_path, allow_pickle=False) as transform:
            mean = np.asarray(transform["mean"], dtype=np.float64)
            std = np.asarray(transform["std"], dtype=np.float64)
        if not np.allclose(mean, np.asarray(model.mean_vals), rtol=0, atol=1e-12):
            raise ContractError("MIOFlow model mean differs from saved train transform")
        if not np.allclose(std, np.asarray(model.std_vals), rtol=0, atol=1e-12):
            raise ContractError("MIOFlow model std differs from saved train transform")
        encoded_source = physical_to_encoded_time(split.observed_times, source_time)
        encoded_target = physical_to_encoded_time(split.observed_times, target)
        normalized = (source_state.astype(np.float64) - mean) / std
        device = getattr(model, "device", "cpu")
        initial = torch.as_tensor(normalized, dtype=torch.float32, device=device)
        times = torch.tensor(
            [encoded_source, encoded_target], dtype=torch.float32, device=device
        )
        model.ode_model.eval()
        if hasattr(model.ode_model, "reset_momentum"):
            model.ode_model.reset_momentum()
        with torch.no_grad():
            trajectory = odeint(model.ode_model, initial, times)
        prediction_state = (
            _to_numpy(trajectory[-1]).astype(np.float64) * std + mean
        ).astype(np.float32)
        prediction_spatial = None
        output_scope = "native_state"
        native_n = int(prediction_state.shape[0])
        plan.update(
            {
                "encoded_source_time": encoded_source,
                "encoded_target_time": encoded_target,
                "state_inverse_transform_applied": True,
                "state_transform_sha256": transform_artifact["sha256"],
            }
        )

    current_environment = environment_provenance(method, current_method_version)
    if current_method_version != fit_manifest.get("method_version"):
        raise ContractError("external method version differs between fit and inference")
    fitted_environment = fit_manifest.get("environment", {})
    if current_environment["environment_fingerprint_sha256"] != fitted_environment.get(
        "environment_fingerprint_sha256"
    ):
        raise ContractError("execution environment differs between fit and inference")
    plan["inference_environment"] = current_environment

    if method != "stvcr" and prediction_state.shape != (
        PREDICTION_N,
        data.state_dim,
    ):
        raise ContractError(
            f"{method} must return exactly {(PREDICTION_N, data.state_dim)}, "
            f"found {prediction_state.shape}"
        )
    if not np.isfinite(prediction_state).all():
        raise ContractError("prediction state contains NaN or infinity")
    arrays: dict[str, np.ndarray] = {"state": prediction_state.astype(np.float32)}
    if prediction_spatial is not None:
        if not np.isfinite(prediction_spatial).all():
            raise ContractError("prediction spatial contains NaN or infinity")
        arrays["spatial"] = prediction_spatial.astype(np.float32)
    if weights is not None:
        arrays["weights"] = weights.astype(np.float32)

    output_dir = _new_output_dir(output_dir)
    prediction_path = output_dir / "prediction.npz"
    np.savez_compressed(prediction_path, **arrays)
    plan.update(
        {
            "output_scope": output_scope,
            "native_output_n": int(native_n),
            "native_growth": bool(method == "stvcr" and bool(params.get("use_growth"))),
            "native_count_changed": bool(
                method == "stvcr" and native_n != PREDICTION_N
            ),
            "native_mass": weights is not None,
            "weights_are_unnormalised": bool(
                method == "stvcr"
                and weights is not None
                and weight_semantics
                == "uniform initial-particle mass 1/5000; sum(weights)=native_n/5000"
            ),
            "weight_semantics": weight_semantics,
            "weight_sum": float(weights.sum()) if weights is not None else None,
            "growth_ratio": (
                float(native_n / PREDICTION_N) if method == "stvcr" else None
            ),
            "prediction_npz": str(prediction_path),
            "prediction_npz_sha256": sha256_file(prediction_path),
            "prediction_keys": sorted(arrays),
            "prediction_shapes": {
                key: list(value.shape) for key, value in arrays.items()
            },
        }
    )
    manifest_path = output_dir / "run_manifest.json"
    summary_path = output_dir / "summary.json"
    plan["run_manifest"] = str(manifest_path)
    plan["summary"] = str(summary_path)
    write_json(manifest_path, plan)
    write_json(summary_path, plan)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def shared(command: argparse.ArgumentParser) -> None:
        command.add_argument("--method", choices=METHODS, required=True)
        command.add_argument("--input-manifest", type=Path, required=True)
        command.add_argument("--split-id", required=True)
        command.add_argument("--source-root", type=Path, required=True)
        command.add_argument("--pins", type=Path, default=DEFAULT_PINS)

    check = commands.add_parser("preflight")
    shared(check)
    check.add_argument("--skip-import", action="store_true")

    fit = commands.add_parser("fit")
    shared(fit)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=20260718)
    fit.add_argument("--params-json")
    fit.add_argument("--dry-run", action="store_true")

    infer = commands.add_parser("infer")
    shared(infer)
    infer.add_argument("--fit-dir", type=Path, required=True)
    infer.add_argument("--target-time", type=float, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--seed", type=int, default=20260718)
    infer.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        payload = preflight(
            args.method,
            args.input_manifest,
            args.split_id,
            args.source_root,
            check_import=not args.skip_import,
            pins_path=args.pins,
        )
    elif args.command == "fit":
        payload = fit_method(
            args.method,
            args.input_manifest,
            args.split_id,
            args.source_root,
            args.output_dir,
            seed=args.seed,
            params=parse_params(args.method, args.params_json),
            dry_run=args.dry_run,
            pins_path=args.pins,
        )
    else:
        payload = infer_method(
            args.method,
            args.input_manifest,
            args.split_id,
            args.source_root,
            args.fit_dir,
            args.target_time,
            args.output_dir,
            seed=args.seed,
            dry_run=args.dry_run,
            pins_path=args.pins,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
