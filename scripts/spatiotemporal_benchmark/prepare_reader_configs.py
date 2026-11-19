#!/usr/bin/env python3
"""Bind downloaded data/current model recipes for a NEW LOTO benchmark.

No models or H5ADs are copied. Historical matched-run acceptance is not claimed
for the selected downloads; the unchanged input builder and model adapter still
enforce their scientific contracts. This is not a full-data checkpoint adapter.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.spatiotemporal_benchmark.build_inputs import inspect_input, sha256
from scripts.spatiotemporal_benchmark.cytobridge.common import validate_training_config

REPO = Path(__file__).resolve().parents[2]
DATASETS = ("zebrafish", "mosta", "arista", "admouse", "chicken_heart")


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size)


def check_training_data(source, config, summary, source_sha):
    """Compare genuine summary fields; missing legacy hashes are not invented."""
    from CytoBridge.tl.train.fit import _canonical_array_provenance, _obs_names_provenance
    recorded = summary["data"]
    times = source.obs[config["time_key"]].to_numpy(dtype=np.float64)
    states = np.column_stack((source.obsm[config["spatial_key"]],
                              source.obsm[config["state_key"]])).astype(np.float32)
    counts = [int(np.count_nonzero(times == time)) for time in np.unique(times)]
    expected = dict(n_observations=source.n_obs, n_timepoints=len(counts),
                    sample_counts_by_timepoint=counts, spatial_dim=config["spatial_dim"],
                    latent_dim=config["state_dim"], model_input_dim=states.shape[1])
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ValueError(f"Training summary {key} does not match selected data: {recorded.get(key)!r} != {value!r}")
    actual = dict(model_input=_canonical_array_provenance(states),
                  processed_time=_canonical_array_provenance(times),
                  obs_names=_obs_names_provenance(source.obs_names))
    verified = []
    for key, value in actual.items():
        if key in recorded:
            if recorded[key].get("sha256") != value["sha256"]:
                raise ValueError(f"Training summary {key} SHA-256 does not match selected data")
            verified.append(key)
    if recorded.get("input_h5ad", {}).get("sha256") is not None:
        if recorded["input_h5ad"]["sha256"] != source_sha:
            raise ValueError("Training summary input_h5ad SHA-256 does not match selected data")
        verified.append("input_h5ad")
    return dict(verified_summary_hashes=verified, actual_arrays=actual,
                summary_counts=expected,
                full_data_checkpoint_reuse_proven=False,
                legacy_missing_hashes=[key for key in (*actual, "input_h5ad") if key not in verified])


def prepare(data_root, output_dir, datasets=DATASETS):
    data_root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if output == data_root or data_root in output.parents:
        raise ValueError("Keep new benchmark configuration outside the downloaded data")
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - set(DATASETS):
        raise ValueError("Choose unique supported dataset names")
    prepared = []
    for name in datasets:
        directory = data_root / name
        h5ad, model = directory / "aligned.h5ad", directory / "model"
        workflow_path, config_path = directory / "workflow.json", model / "config.yaml"
        summary_path = model / "training_run_summary.json"
        for path in (h5ad, workflow_path, config_path, summary_path,
                     model / "Finetune/best_model.pth", model / "Score_Refine/score_model.pth"):
            if not path.is_file():
                raise FileNotFoundError(f"Extract the model and analysis downloads first: {path}")
        workflow = json.loads(workflow_path.read_text())
        if workflow["dataset"]["name"] != name:
            raise ValueError(f"workflow.json belongs to another dataset: {directory}")
        training = yaml.safe_load(config_path.read_text())
        profile = validate_training_config(training)
        if profile["interaction_mode"] != "learned":
            raise ValueError("This reader route selects the published full learned-prior model recipe")
        summary = json.loads(summary_path.read_text())
        if summary.get("schema_version") != 1:
            raise ValueError(f"Unsupported training summary: {summary_path}")
        template_path = REPO / "configs/unified_benchmark" / f"{name}.yaml"
        config = deepcopy(yaml.safe_load(template_path.read_text()))
        historical_audits = config["preprocess_contract"]["external_audits"]
        config["input_h5ad"] = str(h5ad)
        config["expected_source_sha256"] = sha256(h5ad)
        for target, key in (("annotation_key", "annotation_key"), ("time_key", "time_key"),
                            ("state_key", "obsm_key"), ("spatial_key", "spatial_key")):
            config[target] = workflow["dataset"][key]
        config["benchmark_times"] = sorted(target for _, target in config["time_map"])
        config["preprocess_contract"]["external_audits"] = [{
            "name": "selected_download_training_summary_not_historical_acceptance",
            "path": str(summary_path), "sha256": sha256(summary_path),
            "required_exact": {"schema_version": 1, "data": summary["data"]},
        }]
        source, _, _ = inspect_input(config)
        try:
            binding = check_training_data(source, config, summary, config["expected_source_sha256"])
        finally:
            source.file.close()
        config["benchmark"].update(formal_run_root=str(output / "formal" / name),
                                    training_config="training/config.yaml")
        config["reader_run"] = dict(purpose="new_loto_from_selected_downloads",
                                     paper_results_reproduced=False,
                                     model_recipe=str(config_path))
        record = dict(dataset=name, purpose="new_loto_from_selected_downloads",
                      source_h5ad=dict(path=str(h5ad), sha256=config["expected_source_sha256"]),
                      training_config=artifact(config_path), training_summary=artifact(summary_path),
                      workflow=artifact(workflow_path), template=artifact(template_path),
                      removed_historical_audits=historical_audits,
                      training_data_binding=binding,
                      scientific_profile=profile,
                      full_data_reuse_note="Downloaded final checkpoints omit earlier stages/saved training AnnData; retain the strict full-data validator.")
        prepared.append((name, model, config, record))
    (output / "configs").mkdir(parents=True)
    records = []
    for name, model, config, record in prepared:
        formal = output / "formal" / name
        formal.mkdir(parents=True)
        (formal / "training").symlink_to(model, target_is_directory=True)
        destination = output / "configs" / f"{name}.yaml"
        destination.write_text(yaml.safe_dump(config, sort_keys=False))
        record["generated_config"] = artifact(destination)
        records.append(record)
    (output / "reader_input_manifest.json").write_text(json.dumps(dict(
        workflow="portable_reader_loto_configuration", datasets=records,
        paper_results_reproduced=False, data_copied=False), indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="directory containing data/<dataset> entries; pass the data directory itself")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    print(prepare(args.data_root, args.output_dir, args.datasets))
