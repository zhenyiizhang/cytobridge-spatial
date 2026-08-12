from __future__ import annotations

import copy
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
    _checkpoint_runtime_binding,
    bootstrap_indices,
    checkpoint_inventory,
    checkpoint_training_match,
    load_training_data,
    read_split_input,
    sha256_array,
    sha256_file,
    source_time,
    validate_training_config,
)
from run_cytobridge import (  # noqa: E402
    _atomic_prediction,
    inference_schedule,
    ordered_graph_plan,
)


PACKAGE_CONFIGS = {
    "zebrafish": "zebrafish_spatial_full_alpha_express_0015.yaml",
    "mosta": "mosta_spatial_full_alpha_express_0015.yaml",
    "arista": "arista_spatial_full.yaml",
    "admouse": "admouse_spatial_full_alpha_express_0015.yaml",
}


def package_config(dataset: str = "zebrafish") -> dict:
    path = HERE.parents[2] / "CytoBridge/configs" / PACKAGE_CONFIGS[dataset]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def locked_config() -> dict:
    return package_config("zebrafish")


def runtime_config() -> dict:
    config = copy.deepcopy(locked_config())
    config["ckpt_dir"] = "/audited/runtime/model"
    config["model"]["spatial_dim"] = 2
    interaction = config["model"]["interaction_net"]
    interaction["cutoff"] = 0.08
    interaction["edge_predictor_path"] = "/audited/runtime/edge.pt"
    interaction["edge_predictor_thre"] = 0.57
    for stage in config["training"]["plan"]:
        stage["sigma"] = 0.03
    return config


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
    original_obs_name = np.asarray(
        [f"legacy_{index}" for index in range(times.size)], dtype=str
    )
    obs = pd.DataFrame(
        {
            "benchmark_time": times,
            "row_id": row_id,
            "benchmark_original_obs_name": original_obs_name,
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
    source_value = (
        int(np.min(times))
        if regime == "full_data"
        else int(max(value for value in np.unique(times) if value < holdout))
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
    def test_all_package_dataset_profiles_share_the_core_contract(self) -> None:
        expected_score_epochs = {
            "zebrafish": 2001,
            "mosta": 2001,
            "arista": 2001,
            "admouse": 3001,
        }
        for dataset in PACKAGE_CONFIGS:
            with self.subTest(dataset=dataset):
                report = validate_training_config(package_config(dataset))
                self.assertEqual(report["alpha_express"], 0.015)
                self.assertEqual(
                    report["stage_profile"][3]["epochs"], expected_score_epochs[dataset]
                )

    def test_all_resolved_package_profiles_keep_their_dataset_recipe(self) -> None:
        for dataset in PACKAGE_CONFIGS:
            with self.subTest(dataset=dataset):
                source = package_config(dataset)
                resolved = copy.deepcopy(source)
                resolved["ckpt_dir"] = f"/runs/{dataset}/training"
                resolved["spatial_dim"] = 2
                resolved["model"]["spatial_dim"] = 2
                resolved["model"]["interaction_net"][
                    "edge_predictor_path"
                ] = f"/runs/{dataset}/preprocess/edge_classifier/{dataset}.pt"
                for stage in resolved["training"]["plan"]:
                    stage["sigma"] = 0.03
                report = validate_training_config(
                    resolved,
                    runtime_resolved=True,
                    reference=source,
                )
                self.assertTrue(report["runtime_resolved"])

    def test_wrong_shared_alpha_and_stage_role_are_rejected(self) -> None:
        config = locked_config()
        config["training"]["defaults"]["alpha_express"] = 0.05
        with self.assertRaisesRegex(ContractError, "alpha_express"):
            validate_training_config(config)
        config = locked_config()
        config["training"]["plan"][-1]["train_strategy"] = "v+g"
        with self.assertRaisesRegex(ContractError, "train_strategy"):
            validate_training_config(config)

    def test_resolved_fit_must_match_its_dataset_package_recipe(self) -> None:
        reference = locked_config()
        mutations = (
            ("default learning rate", ("training", "defaults", "lr"), 0.0002),
            ("default OT weight", ("training", "defaults", "lambda_ot"), 9.0),
            ("model heads", ("model", "interaction_net", "num_heads"), 4),
            ("stage OT loss", ("training", "plan", 0, "OT_loss"), "weighted_emd"),
            ("stage numeric bool", ("training", "plan", 0, "lambda_ot"), True),
            ("stage batch size", ("training", "plan", 3, "batch_size"), 256),
            (
                "stage scheduler metric",
                ("training", "plan", 4, "scheduler_metric"),
                "average_loss",
            ),
        )
        for label, path, replacement in mutations:
            with self.subTest(label=label):
                config = locked_config()
                target = config
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                config["model"]["spatial_dim"] = 2
                with self.assertRaisesRegex(ContractError, "package YAML"):
                    validate_training_config(
                        config,
                        runtime_resolved=True,
                        reference=reference,
                    )

    def test_missing_shared_contract_fields_are_rejected(self) -> None:
        mutations = (
            (
                "missing alpha",
                lambda value: value["training"]["defaults"].pop("alpha_express"),
            ),
            (
                "missing model field",
                lambda value: value["model"]["velocity_net"].pop("n_layers"),
            ),
            (
                "missing stage epochs",
                lambda value: value["training"]["plan"][0].pop("epochs"),
            ),
        )
        for label, mutation in mutations:
            with self.subTest(label=label):
                config = locked_config()
                mutation(config)
                with self.assertRaises(ContractError):
                    validate_training_config(config)

    def test_runtime_graph_fields_can_change_but_scientific_fields_cannot(self) -> None:
        config = runtime_config()
        report = validate_training_config(
            config,
            runtime_resolved=True,
            runtime_sigma=0.03,
            reference=locked_config(),
        )
        self.assertTrue(report["runtime_resolved"])
        self.assertEqual(report["runtime_resolved_fields"]["model.spatial_dim"], 2)

        wrong_sigma = runtime_config()
        wrong_sigma["training"]["plan"][0]["sigma"] = 0.05
        with self.assertRaisesRegex(ContractError, "sigma"):
            validate_training_config(
                wrong_sigma, runtime_resolved=True, runtime_sigma=0.03
            )

        scientific_change = runtime_config()
        scientific_change["model"]["interaction_net"]["num_heads"] = 4
        with self.assertRaisesRegex(ContractError, "package YAML"):
            validate_training_config(
                scientific_change,
                runtime_resolved=True,
                runtime_sigma=0.03,
                reference=locked_config(),
            )

        with self.assertRaisesRegex(ContractError, "runtime CLI sigma"):
            validate_training_config(
                runtime_config(), runtime_resolved=True, runtime_sigma=0.05
            )

    def test_embedded_predictor_does_not_require_the_recorded_external_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary).resolve()

            class Data:
                spatial_dim = 2

            inventory = {
                "training_profile": {
                    "runtime_resolved_fields": {
                        "ckpt_dir": str(model),
                        "model.spatial_dim": 2,
                        "model.interaction_net.cutoff": 0.08,
                        "model.interaction_net.edge_predictor_path": "/old/machine/missing.pt",
                        "model.interaction_net.edge_predictor_thre": 0.57,
                    }
                }
            }
            report = _checkpoint_runtime_binding(
                model,
                Data(),
                inventory,
                {"interaction_cutoff": 0.08, "edge_threshold": 0.57},
            )
            self.assertFalse(report["external_edge_predictor_required"])
            self.assertEqual(
                report["edge_predictor_source"], "embedded_finetune_checkpoint"
            )


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
            point = np.arange(PREDICTION_N * 5, dtype=np.float32).reshape(
                PREDICTION_N, 5
            )
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
    def test_fit_summary_must_carry_matching_row_identity_proof(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_mock_inputs(root / "inputs")
            split = read_split_input(manifest, "loto_t1")
            data = load_training_data(split)
            model = root / "model"
            model.mkdir()
            saved = ad.read_h5ad(split.train_h5ad)
            saved.write_h5ad(model / "adata.h5ad")
            edge = root / "edge.pt"
            edge.write_bytes(b"audited edge")
            prepare = root / "prepare_graph_summary.json"
            prepare.write_text("{}", encoding="utf-8")
            inventory = {
                "config_sha256": "a" * 64,
                "checkpoints": {"Finetune": {"sha256": "b" * 64}},
                "training_profile": {
                    "runtime_resolved_fields": {
                        "ckpt_dir": str(model),
                        "model.spatial_dim": 2,
                        "model.interaction_net.cutoff": 0.08,
                        "model.interaction_net.edge_predictor_path": str(edge),
                        "model.interaction_net.edge_predictor_thre": 0.57,
                    }
                },
            }
            payload = {
                "status": "complete",
                "training_reference_sha256": split.training_reference_sha256,
                "input_manifest_sha256": split.root_manifest_sha256,
                "split_id": split.split_id,
                "regime": split.regime,
                "saved_config_sha256": inventory["config_sha256"],
                "checkpoint_sha256": {"Finetune": "b" * 64},
                "interaction_cutoff": 0.08,
                "edge_threshold": 0.57,
                "edge_model": str(edge),
                "edge_model_sha256": sha256_file(edge),
                "prepare_graph_summary": str(prepare),
                "prepare_graph_summary_sha256": sha256_file(prepare),
                "training_reference_match": {
                    "proof": "saved_adata_exact_frozen_arrays",
                    "sha256": sha256_file(model / "adata.h5ad"),
                },
            }
            summary = model / "benchmark_fit_summary.json"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "row-identity proof"):
                checkpoint_training_match(model, split, data, inventory=inventory)

            payload["training_reference_match"].update(
                {
                    "row_identity_proof": "contracted_row_id_exact_order",
                    "array_sha256": {
                        "row_identity": sha256_array(data.row_id.astype("U"))
                    },
                }
            )
            summary.write_text(json.dumps(payload), encoding="utf-8")
            report = checkpoint_training_match(model, split, data, inventory=inventory)
            self.assertEqual(report["proof"], "benchmark_fit_summary")

    def test_all_six_stage_files_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "config.yaml").write_text(
                yaml.safe_dump(runtime_config()), encoding="utf-8"
            )
            config = runtime_config()
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
            self.assertTrue(report["training_profile"]["runtime_resolved"])
            (model / "Score_Refine" / "score_model.pth").unlink()
            with self.assertRaisesRegex(ContractError, "incomplete six-stage"):
                checkpoint_inventory(model)

    def test_contracted_row_id_proves_exact_identity_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_mock_inputs(root / "inputs", split_id="full_data")
            split = read_split_input(manifest, "full_data")
            data = load_training_data(split)
            model = root / "model"
            model.mkdir()
            saved = ad.read_h5ad(split.train_h5ad)
            saved.write_h5ad(model / "adata.h5ad")

            report = checkpoint_training_match(model, split, data)
            self.assertEqual(
                report["row_identity_proof"], "contracted_row_id_exact_order"
            )

            saved.obs[data.row_id_key] = np.roll(
                saved.obs[data.row_id_key].astype(str).to_numpy(), 1
            )
            saved.obs_names.name = None
            saved.write_h5ad(model / "adata.h5ad")
            with self.assertRaisesRegex(ContractError, "row identity/order"):
                checkpoint_training_match(model, split, data)

    def test_legacy_obs_names_prove_exact_identity_and_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_mock_inputs(root / "inputs", split_id="full_data")
            split = read_split_input(manifest, "full_data")
            data = load_training_data(split)
            self.assertIsNotNone(data.benchmark_original_obs_name)
            model = root / "model"
            model.mkdir()
            saved = ad.read_h5ad(split.train_h5ad)
            del saved.obs[data.row_id_key]
            saved.obs_names = pd.Index(
                saved.obs["benchmark_original_obs_name"].astype(str),
                name=None,
            )
            saved.write_h5ad(model / "adata.h5ad")

            report = checkpoint_training_match(model, split, data)
            self.assertEqual(
                report["row_identity_proof"],
                "legacy_obs_names_vs_benchmark_original_obs_name",
            )

            saved.obs_names = pd.Index(np.roll(saved.obs_names.astype(str), 1))
            saved.write_h5ad(model / "adata.h5ad")
            with self.assertRaisesRegex(ContractError, "row identity/order"):
                checkpoint_training_match(model, split, data)


if __name__ == "__main__":
    unittest.main()
