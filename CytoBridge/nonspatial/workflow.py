"""Package-owned non-spatial preprocessing, prior, training, and evaluation.

The two supported presets intentionally model expression state only.  Clone,
cell-type, SPRING, and metabolic-label annotations are downstream evidence and
never enter model fitting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import resources
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .preprocess import PreparedNonSpatialData


@dataclass(frozen=True)
class NonSpatialPreset:
    """Frozen scientific choices for one supported non-spatial dataset."""

    name: str
    display_name: str
    expected_cells: int
    expected_latent_dim: int
    time_key: str
    cell_type_key: str
    gene_symbol_key: str
    full_config: str
    no_interaction_config: str
    interaction_cutoff: float
    edge_predictor_threshold: float


_PRESETS = {
    "weinreb": NonSpatialPreset(
        name="weinreb",
        display_name="Weinreb lineage tracing",
        expected_cells=49_302,
        expected_latent_dim=50,
        time_key="Time point",
        cell_type_key="Cell type annotation",
        gene_symbol_key="gene",
        full_config="weinreb_nonspatial_gnn_full.yaml",
        no_interaction_config="weinreb_nonspatial_gnn_no_interaction.yaml",
        interaction_cutoff=25.815367340408883,
        edge_predictor_threshold=0.34204426407814026,
    ),
    "scnt_cortex": NonSpatialPreset(
        name="scnt_cortex",
        display_name="scNT cortical KCl time course",
        expected_cells=20_547,
        expected_latent_dim=50,
        time_key="time_point_processed",
        cell_type_key="cell_type",
        gene_symbol_key="gene_symbol",
        full_config="scnt_cortex_nonspatial_gnn_full.yaml",
        no_interaction_config="scnt_cortex_nonspatial_gnn_no_interaction.yaml",
        interaction_cutoff=23.65247975535851,
        edge_predictor_threshold=0.27495816349983215,
    ),
}


def available_nonspatial_presets() -> tuple[str, ...]:
    """Return the stable names accepted by the non-spatial workflow."""

    return tuple(sorted(_PRESETS))


def nonspatial_preset(name: str) -> NonSpatialPreset:
    """Resolve one preset, accepting ``scnt`` as a convenience alias."""

    normalized = str(name).strip().lower().replace("-", "_")
    if normalized == "scnt":
        normalized = "scnt_cortex"
    try:
        return _PRESETS[normalized]
    except KeyError as exc:
        raise KeyError(
            f"Unknown non-spatial preset {name!r}; choose from "
            + ", ".join(available_nonspatial_presets())
        ) from exc


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _read_json(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = _require_file(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a JSON object.")
    return value, resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_preprocessing_manifest(
    preset: NonSpatialPreset, manifest: Mapping[str, Any]
) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError("Current non-spatial preprocessing requires schema_version=2.")
    expected_operation = {
        "weinreb": "prepare_weinreb_nonspatial",
        "scnt_cortex": "prepare_scnt_nonspatial",
    }[preset.name]
    if manifest.get("operation") != expected_operation:
        raise ValueError(
            f"{preset.name} requires operation={expected_operation!r}, got "
            f"{manifest.get('operation')!r}."
        )
    shape = manifest.get("shape_latent", manifest.get("model_shape"))
    if shape != [preset.expected_cells, preset.expected_latent_dim]:
        raise ValueError(
            f"{preset.name} requires prepared shape "
            f"[{preset.expected_cells}, {preset.expected_latent_dim}], got {shape!r}."
        )
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("Preprocessing manifest lacks its scientific contract.")
    if bool(preprocessing.get("uses_spatial_coordinates", True)):
        raise ValueError("Non-spatial preprocessing must not use spatial coordinates.")
    if preset.name == "weinreb":
        if bool(preprocessing.get("uses_clone_or_annotation_for_preprocessing", True)):
            raise ValueError("Weinreb clone or annotation leaked into preprocessing.")
    else:
        blinding = manifest.get("training_blinding")
        if not isinstance(blinding, Mapping) or any(
            bool(value) for value in blinding.values()
        ):
            raise ValueError("scNT training-blinding contract is absent or violated.")


def packaged_training_config(dataset: str, arm: str) -> Path:
    """Return the installed YAML for a Full or No-interaction run."""

    preset = nonspatial_preset(dataset)
    normalized_arm = str(arm).strip().lower().replace("-", "_")
    filename = {
        "full": preset.full_config,
        "no_interaction": preset.no_interaction_config,
    }.get(normalized_arm)
    if filename is None:
        raise ValueError("arm must be 'full' or 'no_interaction'.")
    resource = resources.files("CytoBridge").joinpath("configs", filename)
    if not resource.is_file():
        raise FileNotFoundError(f"Packaged training config is missing: {filename}")
    return Path(str(resource))


def prepare_nonspatial_dataset(
    dataset: str,
    input_h5ad: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> PreparedNonSpatialData:
    """Prepare a Weinreb or scNT dataset and write the processed files."""

    from .preprocess import prepare_scnt_nonspatial, prepare_weinreb_nonspatial

    preset = nonspatial_preset(dataset)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "output_h5ad": output / "model_input_50pc.h5ad",
        "expression_output_h5ad": output / "lr_expression.h5ad",
        "artifacts_npz": output / "pca_artifacts.npz",
        "manifest_json": output / "preprocess_manifest.json",
        "n_pcs": preset.expected_latent_dim,
        "overwrite": bool(overwrite),
    }
    if preset.name == "weinreb":
        return prepare_weinreb_nonspatial(
            input_h5ad,
            expected_cells=preset.expected_cells,
            **common,
        )
    return prepare_scnt_nonspatial(
        input_h5ad,
        expected_cells=preset.expected_cells,
        time_key=preset.time_key,
        cell_type_key=preset.cell_type_key,
        **common,
    )


def build_nonspatial_lr_prior(
    dataset: str,
    preprocess_manifest: str | Path,
    output_dir: str | Path,
    *,
    lr_database: str | Path | None = None,
    device: str = "auto",
    overwrite: bool = False,
    epochs: int = 50,
) -> dict[str, Any]:
    """Train the accepted directed LR edge prior from frozen preprocessing."""

    import numpy as np

    from CytoBridge.graph_database import bundled_graph_database_path
    from CytoBridge.pp.lr_edge_prior import LREdgePriorConfig, build_lr_edge_prior

    preset = nonspatial_preset(dataset)
    manifest, manifest_path = _read_json(preprocess_manifest)
    _validate_preprocessing_manifest(preset, manifest)
    expression = _require_file(str(manifest.get("expression_output_h5ad")))
    latent = _require_file(str(manifest.get("output_h5ad")))
    if _sha256(expression) != manifest.get("expression_output_sha256"):
        raise ValueError("LR expression H5AD no longer matches preprocessing.")
    if _sha256(latent) != manifest.get("output_sha256"):
        raise ValueError("Latent H5AD no longer matches preprocessing.")
    database = (
        _require_file(lr_database)
        if lr_database is not None
        else bundled_graph_database_path("mosta")
    )
    config = LREdgePriorConfig(
        time_key=preset.time_key,
        gene_symbol_key=preset.gene_symbol_key,
        latent_key="X_latent",
        epochs=int(epochs),
        expected_latent_dim=preset.expected_latent_dim,
        database_source="CellChatDB mouse ligand-receptor database",
        database_version="wheel-bundled exact CSV",
        database_commit=_sha256(database),
    )
    result = build_lr_edge_prior(
        expression,
        latent,
        database,
        output_dir,
        config=config,
        device=device,
        overwrite=bool(overwrite),
        implementation_paths=(Path(__file__), manifest_path),
    )
    observed_cutoff = float(result["pair_sampling"]["candidate_radius"])
    if not np.isclose(observed_cutoff, preset.interaction_cutoff, rtol=0, atol=1e-10):
        raise ValueError(
            f"{preset.name} state-space cutoff changed: {observed_cutoff!r}."
        )
    return result


def train_nonspatial_condition(
    dataset: str,
    arm: str,
    preprocess_manifest: str | Path,
    output_dir: str | Path,
    *,
    edge_prior_manifest: str | Path | None = None,
    device: str = "cuda",
    evaluate_after_training: bool = False,
) -> dict[str, Any]:
    """Train one corrected matched arm and write a workflow manifest."""

    import numpy as np
    import yaml

    from CytoBridge.tl.train import fit

    preset = nonspatial_preset(dataset)
    normalized_arm = str(arm).strip().lower().replace("-", "_")
    if normalized_arm not in {"full", "no_interaction"}:
        raise ValueError("arm must be 'full' or 'no_interaction'.")
    preprocessing, preprocessing_path = _read_json(preprocess_manifest)
    _validate_preprocessing_manifest(preset, preprocessing)
    prepared = _require_file(str(preprocessing.get("output_h5ad")))
    if _sha256(prepared) != preprocessing.get("output_sha256"):
        raise ValueError("Prepared model H5AD no longer matches its manifest.")
    if preprocessing.get("shape_latent", preprocessing.get("model_shape"))[1] != 50:
        raise ValueError("Formal non-spatial training requires exactly 50 PCs.")

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty training output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_path = packaged_training_config(preset.name, normalized_arm)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["ckpt_dir"] = str(output / "model")

    prior_record: dict[str, Any] | None = None
    predictor_path: Path | None = None
    predictor_threshold: float | None = None
    if normalized_arm == "full":
        if edge_prior_manifest is None:
            raise ValueError("Full training requires --edge-prior-manifest.")
        prior, prior_path = _read_json(edge_prior_manifest)
        predictor_path = _require_file(prior["artifacts"]["link_predictor.pt"]["path"])
        if _sha256(predictor_path) != prior["artifacts"]["link_predictor.pt"]["sha256"]:
            raise ValueError("LR edge predictor no longer matches its manifest.")
        predictor_threshold = float(
            prior["predictor"]["recommended_edge_predictor_threshold"]
        )
        if not np.isclose(
            predictor_threshold, preset.edge_predictor_threshold, rtol=0, atol=1e-10
        ):
            raise ValueError(
                f"{preset.name} predictor threshold changed: {predictor_threshold!r}."
            )
        prior_record = {
            "manifest": str(prior_path),
            "manifest_sha256": _sha256(prior_path),
            "predictor": str(predictor_path),
            "predictor_sha256": _sha256(predictor_path),
            "threshold": predictor_threshold,
        }
    elif edge_prior_manifest is not None:
        raise ValueError("No-interaction training must not receive an edge prior.")

    fit(
        prepared,
        config,
        device=device,
        time_key="time_point_processed",
        obsm_key="X_latent",
        is_spatial=False,
        interaction_cutoff=(
            preset.interaction_cutoff if normalized_arm == "full" else None
        ),
        edge_predictor_path=(str(predictor_path) if predictor_path else None),
        edge_predictor_threshold=predictor_threshold,
        ckpt_dir=output / "model",
        sigma=0.1,
        evaluate_after_training=bool(evaluate_after_training),
    )
    model_dir = output / "model"
    required = (
        model_dir / "adata.h5ad",
        model_dir / "config.yaml",
        model_dir / "training_run_summary.json",
    )
    for path in required:
        _require_file(path)
    run_manifest = {
        "schema_version": 2,
        "workflow": "CytoBridge package non-spatial matched training",
        "dataset": preset.name,
        "condition": normalized_arm,
        "seed": 42,
        "sigma": 0.1,
        "smoke": False,
        "prepared_h5ad": str(prepared),
        "prepared_sha256": _sha256(prepared),
        "preprocess_manifest": str(preprocessing_path),
        "preprocess_manifest_sha256": _sha256(preprocessing_path),
        "training_config_source": str(config_path),
        "training_config_source_sha256": _sha256(config_path),
        "model_dir": str(model_dir),
        "model_artifacts": {
            path.name: {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in required
        },
        "edge_prior": prior_record,
        "uses_spatial_coordinates_for_training": False,
        "uses_clone_or_cell_type_for_training": False,
        "uses_metabolic_velocity_for_training": False,
        "scientific_contract": {
            "matched_family": config["matched_ablation"]["family"],
            "arm": normalized_arm,
            "score_energy_objective": "velocity_score_cross_term",
            "intervention": "remove only learned interaction computation and force",
        },
    }
    run_manifest_path = output / "run_manifest.json"
    _write_json(run_manifest_path, run_manifest)
    return {**run_manifest, "run_manifest": str(run_manifest_path)}


def evaluate_nonspatial_pair(
    dataset: str,
    prepared_h5ad: str | Path,
    full_run_dir: str | Path,
    no_interaction_run_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    inference_seeds: Sequence[int] = (10_000,),
    n_samples: int = 2_048,
    sigma: float = 0.1,
    max_ot_points: int = 1_024,
) -> dict[str, Any]:
    """Run matched weighted W1/W2/TMV evaluation for both trained arms."""

    import anndata as ad
    import numpy as np
    import pandas as pd

    from CytoBridge.tl.downstream import (
        evaluate_model_distributions,
        load_dynamical_model_from_dir,
        save_distribution_evaluation,
    )

    preset = nonspatial_preset(dataset)
    prepared = _require_file(prepared_h5ad)
    adata = ad.read_h5ad(prepared)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    if latent.shape != (preset.expected_cells, preset.expected_latent_dim):
        raise ValueError(f"Unexpected {preset.name} prepared shape {latent.shape!r}.")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty evaluation output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    run_records = {}
    for condition, run_dir_value in (
        ("full", full_run_dir),
        ("no_interaction", no_interaction_run_dir),
    ):
        run_dir = Path(run_dir_value).expanduser().resolve()
        run_manifest, run_manifest_path = _read_json(run_dir / "run_manifest.json")
        if run_manifest.get("condition") != condition:
            raise ValueError(f"{run_manifest_path} is not condition={condition!r}.")
        if run_manifest.get("prepared_sha256") != _sha256(prepared):
            raise ValueError("Paired runs were not trained from this prepared H5AD.")
        model_dir = _require_file(run_dir / "model" / "config.yaml").parent
        loaded = load_dynamical_model_from_dir(
            model_dir,
            dim=preset.expected_latent_dim,
            device=device,
        )
        run_records[condition] = {
            "run_manifest": str(run_manifest_path),
            "run_manifest_sha256": _sha256(run_manifest_path),
            "weight_path": str(loaded.weight_path),
            "weight_sha256": _sha256(loaded.weight_path),
            "score_path": str(loaded.score_path),
            "score_sha256": _sha256(loaded.score_path),
        }
        for repeat, random_seed in enumerate(inference_seeds):
            result = evaluate_model_distributions(
                adata,
                loaded.model,
                n_samples=int(n_samples),
                sigma=float(sigma),
                concat_spatial=False,
                max_ot_points=int(max_ot_points),
                device=device,
                random_seed=int(random_seed),
                verbose=False,
            )
            table = result.metrics.copy()
            table.insert(0, "condition", condition)
            table.insert(1, "inference_repeat", int(repeat))
            table.insert(2, "inference_seed", int(random_seed))
            rows.append(table)
            if repeat == 0:
                save_distribution_evaluation(result, output / condition)
    combined = pd.concat(rows, ignore_index=True)
    metrics_path = output / "paired_distribution_metrics.csv"
    combined.to_csv(metrics_path, index=False)
    manifest = {
        "schema_version": 1,
        "operation": "evaluate_nonspatial_pair",
        "dataset": preset.name,
        "prepared_h5ad": str(prepared),
        "prepared_sha256": _sha256(prepared),
        "conditions": run_records,
        "settings": {
            "weighted_ot": True,
            "weights": "model-predicted relative particle masses",
            "metrics": ["w1", "w2", "tmv", "tmv_absolute"],
            "n_samples": int(n_samples),
            "sigma": float(sigma),
            "max_ot_points": int(max_ot_points),
            "inference_seeds": [int(value) for value in inference_seeds],
            "continuous_simulation_origin": "earliest observed time (t=0)",
        },
        "metrics": {
            "path": str(metrics_path),
            "sha256": _sha256(metrics_path),
            "n_rows": int(len(combined)),
        },
    }
    manifest_path = output / "evaluation_manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


def nonspatial_plan(dataset: str) -> dict[str, Any]:
    """Return the ordered steps for one non-spatial dataset."""

    preset = nonspatial_preset(dataset)
    return {
        "schema_version": 1,
        "preset": asdict(preset),
        "steps": [
            "prepare total-expression and 50-PC state",
            "build directed ligand-receptor edge prior",
            "train matched Full and No-interaction arms",
            "evaluate weighted W1/W2/TMV from t=0",
            "run dataset-specific clone-fate or scNT direction evaluation",
            "compute exact interaction attribution",
            f"draw Supplementary Figure {'S4' if preset.name == 'weinreb' else 'S5'}",
        ],
        "historical_replay_note": (
            "The included numerical files reproduce the published figure; use the "
            "steps above to analyze a new run."
        ),
    }


__all__ = [
    "NonSpatialPreset",
    "available_nonspatial_presets",
    "build_nonspatial_lr_prior",
    "evaluate_nonspatial_pair",
    "nonspatial_plan",
    "nonspatial_preset",
    "packaged_training_config",
    "prepare_nonspatial_dataset",
    "train_nonspatial_condition",
]
