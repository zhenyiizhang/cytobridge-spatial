from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    CONTRACT_VERSION,
    PREDICTION_N,
    ContractError,
    bootstrap_indices,
    checkpoint_inventory,
    load_training_data,
    read_split_input,
    sha256_file,
    source_time,
    validate_training_config,
)
from run_cytobridge import (  # noqa: E402
    _atomic_prediction,
    inference_schedule,
    ordered_graph_plan,
)


def locked_config() -> dict:
    stages = [
        {
            "name": "Pretrain",
            "mode": "neural_ode",
            "epochs": 100,
            "train_strategy": "v+g",
            "global_mass": False,
        },
        {
            "name": "Refine",
            "mode": "neural_ode",
            "epochs": 100,
            "train_strategy": "v+g",
            "global_mass": True,
        },
        {
            "name": "Init_interaction",
            "mode": "neural_ode",
            "epochs": 50,
            "train_strategy": "v+g+i",
            "global_mass": True,
        },
        {
            "name": "Train_Score",
            "mode": "score_matching",
            "epochs": 2001,
            "train_strategy": "s",
            "sigma": 0.03,
            "save_strategy": "last",
        },
        {
            "name": "Finetune",
            "mode": "neural_ode",
            "epochs": 1000,
            "train_strategy": "v+g+i",
            "global_mass": True,
            "score_use": True,
            "save_strategy": "best",
        },
        {
            "name": "Score_Refine",
            "mode": "score_matching",
            "epochs": 2001,
            "train_strategy": "s",
            "sigma": 0.03,
            "save_strategy": "last",
        },
    ]
    return {
        "seed": 42,
        "reverse": True,
        "model": {
            "components": ["velocity", "growth", "score", "interaction"],
            "interaction_type": "gnn",
            "interaction_group_size": 1024,
            "velocity_net": {
                "hidden_dim": 256,
                "n_layers": 5,
                "residual": False,
                "activation": "leaky_relu",
                "use_spatial": True,
            },
            "growth_net": {
                "hidden_dim": 256,
                "n_layers": 3,
                "residual": False,
                "activation": "leaky_relu",
            },
            "score_net": {
                "hidden_dim": 400,
                "n_layers": 3,
                "activation": "leaky_relu",
            },
            "interaction_net": {
                "hidden_dim": 256,
                "num_heads": 8,
                "num_layers": 1,
                "activation": "leakyrelu",
                "num_rbf": 8,
                "use_spatial": True,
                "rbf_trainable": False,
            },
        },
        "training": {
            "defaults": {
                "alpha_express": 0.015,
                "alpha_spatial": 10.0,
                "sigma": 0.03,
                "global_mass": True,
            },
            "plan": stages,
        },
    }


def write_mock_inputs(
    root: Path,
    *,
    split_id: str = "loto_t1",
    times: np.ndarray | None = None,
) -> Path:
    split_dir = root / split_id
    split_dir.mkdir(parents=True)
    regime = "full_data" if split_id == "full_data" else "loto"
    holdout = None if regime == "full_data" else 1
    if times is None:
        times = (
            np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int16)
            if regime == "full_data"
            else np.array([0, 0, 2, 2, 3, 3, 4, 4], dtype=np.int16)
        )
    rng = np.random.default_rng(7)
    state = rng.normal(size=(times.size, 3)).astype(np.float32)
    spatial = rng.normal(size=(times.size, 2)).astype(np.float32)
    row_id = np.asarray([f"r{index}" for index in range(times.size)], dtype=str)
    obs = pd.DataFrame(
        {
            "benchmark_time": times,
            "row_id": row_id,
        },
        index=pd.Index(row_id, name="row_id"),
    )
    data = ad.AnnData(X=np.zeros((times.size, 2), dtype=np.float32), obs=obs)
    data.obsm["benchmark_state"] = state
    data.obsm["benchmark_spatial"] = spatial
    data.uns["cytobridge_benchmark_contract"] = {
        "dataset_id": "mock_dataset",
        "split": split_id,
        "role": "train_and_truth" if regime == "full_data" else "train",
        "state_key": "benchmark_state",
        "state_dim": 3,
        "spatial_key": "benchmark_spatial",
        "spatial_dim": 2,
        "time_key": "benchmark_time",
        "row_id_key": "row_id",
        "prediction_n": PREDICTION_N,
        "truth_cell_count_must_not_control_prediction_n": True,
        "transductive_frozen_representation": True,
        "representation_refit_per_fold": False,
        "target_removed": regime == "loto",
        "held_out_benchmark_time": "none" if holdout is None else holdout,
    }
    h5ad = split_dir / "train.h5ad"
    data.write_h5ad(h5ad)
    reference = split_dir / "training_reference.npz"
    np.savez_compressed(
        reference,
        state=state,
        spatial=spatial,
        time=times,
        row_id=row_id,
    )
    source_value = int(np.min(times)) if regime == "full_data" else int(
        max(value for value in np.unique(times) if value < holdout)
    )
    source_candidates = np.flatnonzero(times == source_value)
    source_indices = np.resize(source_candidates, PREDICTION_N)
    roster = split_dir / "source_roster.npz"
    np.savez_compressed(
        roster,
        indices=source_indices,
        row_id=row_id[source_indices],
        source_time=np.asarray([source_value], dtype=np.int16),
        state=state[source_indices],
        spatial=spatial[source_indices],
    )
    counts = {str(value): int(np.count_nonzero(times == value)) for value in range(5)}
    targets = [1, 2, 3, 4] if regime == "full_data" else [1]
    split = {
        "protocol": "full_data" if regime == "full_data" else "leave_one_timepoint_out",
        "held_out_benchmark_time": holdout,
        "evaluation_targets": targets,
        "target_rows_physically_removed_from_train": regime == "loto",
        "prediction_n": PREDICTION_N,
        "truth_cell_count_must_not_control_prediction_n": True,
        "transductive_frozen_representation": True,
        "representation_refit_per_fold": False,
        "contract_uns_key": "cytobridge_benchmark_contract",
        "train_time_counts": counts,
        "train": {
            "h5ad": {
                "relative_path": f"{split_id}/train.h5ad",
                "path": "/build/machine/path/must/not/be/used.h5ad",
                "sha256": sha256_file(h5ad),
            },
            "training_reference_npz": {
                "relative_path": f"{split_id}/training_reference.npz",
                "path": "/build/machine/path/must/not/be/used.npz",
                "sha256": sha256_file(reference),
            },
            "source_roster_npz": {
                "relative_path": f"{split_id}/source_roster.npz",
                "path": "/build/machine/path/must/not/be/used-roster.npz",
                "sha256": sha256_file(roster),
            },
        },
        # A deliberately missing truth artifact proves train-side loading does
        # not resolve or open truth.
        "truth_by_time_npz": {
            str(target): {"relative_path": "missing_truth.npz", "sha256": "0" * 64}
            for target in targets
        },
    }
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "dataset_id": "mock_dataset",
                "contract_uns_key": "cytobridge_benchmark_contract",
                "prediction_n": PREDICTION_N,
                "splits": {split_id: split},
            }
        ),
        encoding="utf-8",
    )
    manifest.with_suffix(".json.sha256").write_text(
        f"{sha256_file(manifest)}  manifest.json\n", encoding="utf-8"
    )
    return manifest


class ConfigContractTests(unittest.TestCase):
    def test_exact_six_stage_profile(self) -> None:
        report = validate_training_config(locked_config())
        self.assertEqual(report["alpha_express"], 0.015)
        self.assertEqual(
            [stage["epochs"] for stage in report["stage_profile"]],
            [100, 100, 50, 2001, 1000, 2001],
        )

    def test_wrong_alpha_and_epoch_are_rejected(self) -> None:
        config = locked_config()
        config["training"]["defaults"]["alpha_express"] = 0.05
        with self.assertRaisesRegex(ContractError, "alpha_express"):
            validate_training_config(config)
        config = locked_config()
        config["training"]["plan"][-1]["epochs"] = 2000
        with self.assertRaisesRegex(ContractError, "epochs"):
            validate_training_config(config)


class InputContractTests(unittest.TestCase):
    def test_loto_train_only_relative_artifacts_and_physical_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_mock_inputs(Path(temporary))
            split = read_split_input(manifest, "loto_t1")
            data = load_training_data(split)
            self.assertFalse(np.any(data.time == 1))
            self.assertEqual(source_time(split), 0)
            self.assertEqual(inference_schedule(split), (0.0, (0.0, 1.0)))
            self.assertEqual(
                ordered_graph_plan(split),
                (
                    (0, 0.0, "mock_dataset_t0"),
                    (1, 2.0, "mock_dataset_t1"),
                    (2, 3.0, "mock_dataset_t2"),
                    (3, 4.0, "mock_dataset_t3"),
                ),
            )

    def test_target_row_leak_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_mock_inputs(
                Path(temporary), times=np.array([0, 0, 1, 2, 3, 4], dtype=np.int16)
            )
            with self.assertRaisesRegex(ContractError, "held-out target appears"):
                read_split_input(manifest, "loto_t1")

    def test_full_schedule_is_one_t0_to_all_target_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_mock_inputs(Path(temporary), split_id="full_data")
            split = read_split_input(manifest, "full_data")
            data = load_training_data(split)
            self.assertEqual(data.n_obs, 10)
            self.assertEqual(
                inference_schedule(split),
                (0.0, (0.0, 1.0, 2.0, 3.0, 4.0)),
            )


class PopulationAndOutputTests(unittest.TestCase):
    def test_fixed_population_is_deterministic_and_source_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_mock_inputs(Path(temporary))
            split = read_split_input(manifest, "loto_t1")
            data = load_training_data(split)
            first = bootstrap_indices(data, 0.0)
            second = bootstrap_indices(data, 0.0)
            self.assertEqual(first.shape, (PREDICTION_N,))
            np.testing.assert_array_equal(first, second)
            self.assertTrue(np.all(data.time[first] == 0))

    def test_prediction_npz_keeps_raw_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            point = np.arange(PREDICTION_N * 5, dtype=np.float32).reshape(PREDICTION_N, 5)
            weights = np.linspace(0.1, 0.2, PREDICTION_N, dtype=np.float64)

            class Data:
                spatial_dim = 2

            path = Path(temporary) / "prediction.npz"
            _atomic_prediction(path, point, weights, Data(), 0.0, 1.0)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["spatial"].shape, (PREDICTION_N, 2))
                self.assertEqual(archive["state"].shape, (PREDICTION_N, 3))
                np.testing.assert_array_equal(archive["weights"], weights)


class CheckpointTests(unittest.TestCase):
    def test_all_six_stage_files_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.yaml").write_text(
                yaml.safe_dump(locked_config()), encoding="utf-8"
            )
            config = locked_config()
            for stage in config["training"]["plan"]:
                folder = model / stage["name"]
                folder.mkdir()
                filename = (
                    "score_model.pth"
                    if stage["mode"] == "score_matching"
                    else "best_model.pth"
                )
                (folder / filename).write_bytes(b"mock checkpoint")
            report = checkpoint_inventory(model)
            self.assertTrue(report["stage_complete"])
            self.assertEqual(report["stage_count"], 6)
            (model / "Score_Refine" / "score_model.pth").unlink()
            with self.assertRaisesRegex(ContractError, "incomplete six-stage"):
                checkpoint_inventory(model)


if __name__ == "__main__":
    unittest.main()
