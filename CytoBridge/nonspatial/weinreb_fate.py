"""Clone-fate evaluation for corrected non-spatial Weinreb matched models."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .clone_fate import (
    evaluate_clone_fate_agreement,
    paired_bootstrap_clone_metric_difference,
)
from .weinreb_simulation import simulate_sde_from_x0


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_run(run_dir: str | Path, condition: str, prepared_sha: str):
    from CytoBridge.tl.downstream import load_dynamical_model_from_dir

    directory = Path(run_dir).expanduser().resolve()
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("condition") != condition:
        raise ValueError(f"{manifest_path} is not condition={condition!r}.")
    if manifest.get("prepared_sha256") != prepared_sha:
        raise ValueError("Clone-fate runs do not match the supplied prepared H5AD.")
    if bool(manifest.get("uses_spatial_coordinates_for_training", True)):
        raise ValueError("Weinreb clone-fate models must be non-spatial.")
    if bool(manifest.get("uses_clone_or_cell_type_for_training", True)):
        raise ValueError("Clone/cell-type information leaked into model training.")
    loaded = load_dynamical_model_from_dir(directory / "model", dim=50)
    return loaded, {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "weight": str(loaded.weight_path),
        "weight_sha256": _sha256(loaded.weight_path),
        "score": str(loaded.score_path),
        "score_sha256": _sha256(loaded.score_path),
    }


def _save_evaluation(result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.predicted_distributions.to_csv(
        output_dir / "predicted_fate_distributions.csv"
    )
    result.observed_distributions.to_csv(output_dir / "observed_fate_distributions.csv")
    result.per_lineage.to_csv(output_dir / "per_lineage_metrics.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(dict(result.summary), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def evaluate_weinreb_clone_fate(
    prepared_h5ad: str | Path,
    full_run_dir: str | Path,
    no_interaction_run_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    source_time: float = 0.0,
    target_time: float = 2.0,
    sigma: float = 0.1,
    dt: float = 0.1,
    simulation_seeds: Sequence[int] = tuple(range(10)),
    classifier_neighbors: int = 20,
    n_bootstrap: int = 5_000,
) -> dict[str, Any]:
    """Compare lineage-level fate distributions from t=0 to the final day.

    All source-time cells are propagated continuously from ``source_time``;
    observed intermediate slices are not used to restart the simulation.
    Clone and cell-type labels are opened only after both model manifests have
    proven that those fields were excluded from training.
    """

    import anndata as ad
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    prepared_path = Path(prepared_h5ad).expanduser().resolve()
    if not prepared_path.is_file():
        raise FileNotFoundError(prepared_path)
    prepared_sha = _sha256(prepared_path)
    adata = ad.read_h5ad(prepared_path)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    if latent.shape != (49_302, 50):
        raise ValueError(f"Unexpected Weinreb prepared shape {latent.shape!r}.")
    for column in ("time_point_processed", "lineage_id", "Cell type annotation"):
        if column not in adata.obs:
            raise KeyError(f"Prepared Weinreb H5AD lacks {column!r}.")
    times = adata.obs["time_point_processed"].to_numpy(dtype=float)
    lineages = adata.obs["lineage_id"].astype(str).to_numpy()
    labels = adata.obs["Cell type annotation"].astype(str).to_numpy()
    source_mask = np.isclose(times, float(source_time))
    target_mask = np.isclose(times, float(target_time))
    if not source_mask.any() or not target_mask.any():
        raise ValueError("Weinreb source/target time selection is empty.")

    full, full_record = _read_run(full_run_dir, "full", prepared_sha)
    no_interaction, no_record = _read_run(
        no_interaction_run_dir, "no_interaction", prepared_sha
    )

    target_indices = np.flatnonzero(target_mask)
    train_idx, validation_idx = train_test_split(
        target_indices,
        test_size=0.25,
        random_state=42,
        stratify=labels[target_indices],
    )
    classifier = KNeighborsClassifier(
        n_neighbors=int(classifier_neighbors), weights="distance", n_jobs=1
    )
    classifier.fit(latent[train_idx], labels[train_idx])
    validation_prediction = classifier.predict(latent[validation_idx])
    classifier_record = {
        "training_population": "target-time cells only; clone is not a feature",
        "n_train": int(len(train_idx)),
        "n_validation": int(len(validation_idx)),
        "accuracy": float(
            accuracy_score(labels[validation_idx], validation_prediction)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels[validation_idx], validation_prediction)
        ),
    }

    source_indices = np.flatnonzero(source_mask)
    source_lineages = lineages[source_indices]
    target_lineages = lineages[target_indices]
    target_labels = labels[target_indices]
    target_lineage_set = set(target_lineages.tolist())
    evaluable = np.asarray(
        [lineage in target_lineage_set for lineage in source_lineages], dtype=bool
    )
    if not evaluable.any():
        raise ValueError("No source lineages are observed at the target time.")
    categories = tuple(sorted(set(labels.tolist())))

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty clone-fate output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    condition_results = {}
    condition_records = {}
    for condition, loaded, include_interaction in (
        ("full", full, True),
        ("no_interaction", no_interaction, False),
    ):
        endpoint_labels = []
        endpoint_weights = []
        for seed in simulation_seeds:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            points, weights, _ = simulate_sde_from_x0(
                x0=latent[source_indices],
                model=loaded.model.to(device),
                ts_points=[float(source_time), float(target_time)],
                dt=float(dt),
                sigma=float(sigma),
                include_score=True,
                include_interaction=include_interaction,
                interaction_m=16,
                device=device,
                noise_seed=int(seed),
                interaction_seed=10_042 + int(seed),
                verbose=False,
            )
            endpoint_labels.append(
                classifier.predict(np.asarray(points[-1])[evaluable])
            )
            endpoint_weights.append(np.asarray(weights[-1, :, 0])[evaluable])
        generated_labels = np.concatenate(endpoint_labels)
        generated_weights = np.concatenate(endpoint_weights)
        generated_lineages = np.tile(source_lineages[evaluable], len(simulation_seeds))
        result = evaluate_clone_fate_agreement(
            generated_lineages,
            generated_labels,
            target_lineages,
            target_labels,
            generated_endpoint_weights=generated_weights,
            categories=categories,
            min_source=1,
            min_target=1,
        )
        _save_evaluation(result, output / condition)
        condition_results[condition] = result
        condition_records[condition] = dict(result.summary)
        loaded.model.to("cpu")

    comparisons = []
    for metric in ("tv_agreement", "js_similarity", "dominant_fate_match"):
        result = paired_bootstrap_clone_metric_difference(
            condition_results["full"],
            condition_results["no_interaction"],
            metric=metric,
            n_bootstrap=int(n_bootstrap),
            confidence_level=0.95,
            seed=42,
        )
        record = asdict(result)
        record.pop("bootstrap_differences")
        comparisons.append(record)

    manifest = {
        "schema_version": 1,
        "operation": "evaluate_weinreb_clone_fate",
        "prepared_h5ad": str(prepared_path),
        "prepared_sha256": prepared_sha,
        "runs": {"full": full_record, "no_interaction": no_record},
        "simulation": {
            "source_time": float(source_time),
            "target_time": float(target_time),
            "continuous_from_t0_without_observed_restarts": True,
            "sigma": float(sigma),
            "dt": float(dt),
            "seeds": [int(seed) for seed in simulation_seeds],
            "interaction_grouping_seeds": [
                10_042 + int(seed) for seed in simulation_seeds
            ],
            "weighted_by_predicted_growth_mass": True,
        },
        "classifier": classifier_record,
        "condition_summaries": condition_records,
        "paired_bootstrap_full_minus_no_interaction": comparisons,
        "clone_or_annotation_used_for_training": False,
    }
    manifest_path = output / "clone_fate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest": str(manifest_path)}


__all__ = ["evaluate_weinreb_clone_fate"]
