#!/usr/bin/env python3
"""Evaluate corrected Weinreb no-interaction against frozen Full fate evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from CytoBridge.tl.downstream import (
    evaluate_clone_fate_agreement,
    load_dynamical_model_from_dir,
    simulate_sde_from_x0,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def load_historical_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("historical_clone_fate", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-h5ad", required=True, type=Path)
    parser.add_argument("--no-interaction-model-dir", required=True, type=Path)
    parser.add_argument("--historical-evaluator", required=True, type=Path)
    parser.add_argument("--frozen-full-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    historical_script = args.historical_evaluator.expanduser().resolve()
    helpers = load_historical_helpers(historical_script)

    prepared_path = args.prepared_h5ad.expanduser().resolve()
    adata = ad.read_h5ad(prepared_path)
    latent = np.asarray(adata.obsm["X_latent"], dtype=np.float32)
    times = np.asarray(adata.obs["time_point_processed"], dtype=float)
    lineages = adata.obs["lineage_id"].astype(str).to_numpy()
    clone_ids = np.asarray(adata.obs["clone"]).astype(int)
    cell_types = adata.obs["Cell type annotation"].astype(str).to_numpy()
    if latent.shape != (49302, 50) or sorted(np.unique(times).tolist()) != [0.0, 1.0, 2.0]:
        raise ValueError("Unexpected Weinreb prepared input contract")

    population = helpers.select_source_simulation_population(
        times=times,
        lineages=lineages,
        clone_ids=clone_ids,
        source_time=0.0,
        target_time=2.0,
        max_simulation_particles=0,
    )
    simulation_indices = population["simulation_indices"]
    evaluation_positions = population["evaluation_positions"]
    source_indices = population["evaluable_source_indices"]
    target_indices = population["target_indices"]
    classifier_target_indices = np.flatnonzero(np.isclose(times, 2.0))
    if len(simulation_indices) != 4638:
        raise ValueError(f"Expected all 4,638 Day-2 source cells, got {len(simulation_indices)}")

    classifier, classifier_report, classifier_confusion = helpers._fit_terminal_classifier(
        latent[classifier_target_indices],
        cell_types[classifier_target_indices],
        n_neighbors=20,
        n_jobs=8,
        seed=42,
    )
    classifier_confusion.to_csv(output / "classifier_confusion_matrix.csv")
    (output / "classifier_report.json").write_text(
        json.dumps(helpers._json_safe(classifier_report), indent=2, sort_keys=True) + "\n"
    )

    no_model_dir = args.no_interaction_model_dir.expanduser().resolve()
    loaded = load_dynamical_model_from_dir(
        no_model_dir,
        dim=50,
        device=args.device,
        stage="Finetune_no_interaction",
    )
    model = loaded.model.to(args.device)
    model.eval()
    seeds = list(range(10))
    x0 = latent[simulation_indices]
    endpoint_points = []
    endpoint_labels = []
    endpoint_weights = []
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        points, weights, _ = simulate_sde_from_x0(
            x0=x0,
            model=model,
            ts_points=[0.0, 2.0],
            dt=0.1,
            sigma=0.1,
            include_score=True,
            include_interaction=False,
            interaction_m=16,
            device=args.device,
            noise_seed=seed,
            verbose=True,
        )
        endpoints = np.asarray(points[-1])[evaluation_positions]
        final_weights = np.asarray(weights[-1])[evaluation_positions, 0]
        endpoint_points.append(endpoints)
        endpoint_labels.append(np.asarray(classifier.predict(endpoints), dtype=object))
        endpoint_weights.append(np.asarray(final_weights, dtype=float))

    npz_path = output / "clone_endpoint_predictions.npz"
    np.savez_compressed(
        npz_path,
        seeds=np.asarray(seeds, dtype=int),
        simulation_indices=simulation_indices,
        evaluation_positions=evaluation_positions,
        source_indices=source_indices,
        lineage_ids=lineages[source_indices],
        endpoints=np.stack(endpoint_points),
        endpoint_labels=np.stack(endpoint_labels),
        endpoint_weights=np.stack(endpoint_weights),
    )
    generated_labels = np.concatenate(endpoint_labels)
    generated_weights = np.concatenate(endpoint_weights)
    generated_lineages = np.tile(lineages[source_indices], len(seeds))
    target_labels = cell_types[target_indices]
    categories = list(
        dict.fromkeys(target_labels.tolist() + np.asarray(classifier.classes_, dtype=object).tolist())
    )
    result = evaluate_clone_fate_agreement(
        generated_lineages,
        generated_labels,
        lineages[target_indices],
        target_labels,
        generated_endpoint_weights=generated_weights,
        categories=categories,
        min_source=1,
        min_target=1,
    )
    result.predicted_distributions.to_csv(output / "predicted_fate_distributions.csv")
    result.observed_distributions.to_csv(output / "observed_fate_distributions.csv")
    result.per_lineage.to_csv(output / "per_lineage_metrics.csv")
    no_summary_path = output / "summary.json"
    no_summary_path.write_text(
        json.dumps(helpers._json_safe(dict(result.summary)), indent=2, sort_keys=True) + "\n"
    )

    frozen_root = args.frozen_full_root.expanduser().resolve()
    full_summary_path = frozen_root / "full_interaction_noise" / "all_fates" / "summary.json"
    full_lineage_path = frozen_root / "full_interaction_noise" / "all_fates" / "per_lineage_metrics.csv"
    full_summary = json.loads(full_summary_path.read_text())
    no_summary = dict(result.summary)
    metrics = {
        "tv_agreement": "clone_macro_tv_agreement",
        "js_similarity": "clone_macro_js_similarity",
        "dominant_fate_match": "clone_macro_dominant_fate_match",
    }
    comparison_rows = []
    for label, key in metrics.items():
        full_value = float(full_summary[key])
        no_value = float(no_summary[key])
        comparison_rows.append(
            {
                "metric": label,
                "full": full_value,
                "no_interaction": no_value,
                "delta_no_interaction_minus_full": no_value - full_value,
                "relative_change": (no_value - full_value) / full_value,
                "higher_is_better": True,
            }
        )
    comparison_path = output / "frozen_full_vs_corrected_no_interaction_summary.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

    full_lineage = pd.read_csv(full_lineage_path, index_col=0)
    no_lineage = result.per_lineage.copy()
    lineage = full_lineage.join(no_lineage, how="inner", lsuffix="_full", rsuffix="_no_interaction")
    for metric in ("tv_agreement", "js_similarity", "dominant_fate_match"):
        lineage[f"{metric}_delta_no_interaction_minus_full"] = (
            lineage[f"{metric}_no_interaction"].astype(float) - lineage[f"{metric}_full"].astype(float)
        )
    lineage_path = output / "paired_per_lineage_metrics.csv"
    lineage.to_csv(lineage_path)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "frozen existing Full checkpoint versus corrected retrained no-interaction clone fate",
        "claim_scope": "descriptive single-training-seed comparison; not strict CRN matched retraining",
        "settings": {
            "source_time": 0.0,
            "target_time": 2.0,
            "all_source_cells": 4638,
            "sigma": 0.1,
            "dt": 0.1,
            "simulation_seeds": seeds,
            "include_score": True,
            "include_interaction": False,
            "interaction_m": 16,
            "classifier_neighbors": 20,
            "classifier_seed": 42,
        },
        "inputs": {
            "prepared_h5ad": record(prepared_path),
            "frozen_full_summary": record(full_summary_path),
            "frozen_full_per_lineage": record(full_lineage_path),
            "corrected_no_interaction_weight": record(Path(loaded.weight_path)),
            "corrected_no_interaction_score": record(Path(loaded.score_path)),
            "historical_evaluator": record(historical_script),
        },
        "outputs": {
            "endpoint_predictions": record(npz_path),
            "no_interaction_summary": record(no_summary_path),
            "comparison": record(comparison_path),
            "paired_per_lineage": record(lineage_path),
        },
        "implementation": record(Path(__file__).resolve()),
    }
    manifest_path = output / "frozen_baseline_clone_fate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print(manifest_path)


if __name__ == "__main__":
    main()
