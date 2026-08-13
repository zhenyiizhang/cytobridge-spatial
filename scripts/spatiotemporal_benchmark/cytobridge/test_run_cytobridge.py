from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

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
    _full_runtime_expectation,
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
    _all_spatial_cutoff,
    _atomic_prediction,
    _compact_prediction_summaries,
    _resolve_interaction_cutoff,
    _load_graph_summary,
    _simulate,
    _validated_interaction_m,
    command_fit_loto,
    command_prepare_loto,
    inference_schedule,
    ordered_graph_plan,
)


PACKAGE_CONFIGS = {
    "zebrafish": "zebrafish_spatial_full_alpha_express_0015.yaml",
    "mosta": "mosta_spatial_full_alpha_express_0015.yaml",
    "arista": "arista_spatial_full.yaml",
    "admouse": "admouse_spatial_full_alpha_express_0015.yaml",
}
ALL_SPATIAL_CONFIG = (
    HERE.parents[2]
    / "CytoBridge/configs/admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml"
)
NO_INTERACTION_CONFIGS = {
    "zebrafish": "zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml",
    "mosta": "mosta_spatial_full_alpha_express_0015_no_interaction.yaml",
    "arista": "arista_spatial_full_no_interaction.yaml",
    "admouse": "admouse_spatial_full_alpha_express_0015_no_interaction.yaml",
}


def package_config(dataset: str = "zebrafish") -> dict:
    path = HERE.parents[2] / "CytoBridge/configs" / PACKAGE_CONFIGS[dataset]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def locked_config() -> dict:
    return package_config("zebrafish")


def all_spatial_config() -> dict:
    payload = yaml.safe_load(ALL_SPATIAL_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def no_interaction_config(dataset: str = "zebrafish") -> dict:
    path = HERE.parents[2] / "CytoBridge/configs" / NO_INTERACTION_CONFIGS[dataset]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def no_interaction_config_path(dataset: str = "zebrafish") -> Path:
    return HERE.parents[2] / "CytoBridge/configs" / NO_INTERACTION_CONFIGS[dataset]


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
                self.assertEqual(report["edge_prior_mode"], "learned")
                self.assertEqual(
                    report["stage_profile"][3]["epochs"], expected_score_epochs[dataset]
                )

    def test_real_no_interaction_profiles_have_the_matched_third_state(self) -> None:
        expected_score_epochs = {
            "zebrafish": 2001,
            "mosta": 2001,
            "arista": 2001,
            "admouse": 3001,
        }
        for dataset in NO_INTERACTION_CONFIGS:
            with self.subTest(dataset=dataset):
                source = no_interaction_config(dataset)
                report = validate_training_config(source)
                self.assertEqual(report["components"], ["velocity", "growth", "score"])
                self.assertEqual(report["interaction_mode"], "none")
                self.assertEqual(report["edge_prior_mode"], "none")
                self.assertFalse(report["uses_interaction"])
                self.assertIsNone(report["interaction_group_size"])
                self.assertEqual(
                    report["expected_weight_stage"], "Finetune_no_interaction"
                )
                self.assertEqual(report["expected_score_stage"], "Score_Refine")
                self.assertEqual(
                    report["stage_profile"][3]["epochs"],
                    expected_score_epochs[dataset],
                )

                resolved = copy.deepcopy(source)
                resolved["ckpt_dir"] = f"/runs/{dataset}/no_interaction"
                resolved["model"]["spatial_dim"] = 2
                for stage in resolved["training"]["plan"]:
                    stage["sigma"] = 0.03
                resolved_report = validate_training_config(
                    resolved,
                    runtime_resolved=True,
                    reference=source,
                )
                fields = resolved_report["runtime_resolved_fields"]
                self.assertEqual(fields["interaction_mode"], "none")
                self.assertEqual(fields["edge_prior_mode"], "none")
                self.assertFalse(any("interaction_net" in key for key in fields))

    def test_no_interaction_reference_validation_rejects_any_scientific_drift(
        self,
    ) -> None:
        reference = no_interaction_config()
        resolved = copy.deepcopy(reference)
        resolved["ckpt_dir"] = "/runs/zebrafish/no_interaction"
        resolved["model"]["spatial_dim"] = 2
        resolved["training"]["defaults"]["lambda_mass"] = 99.0
        with self.assertRaisesRegex(ContractError, "package YAML"):
            validate_training_config(
                resolved,
                runtime_resolved=True,
                reference=reference,
            )

        inert = no_interaction_config()
        inert["model"]["interaction_net"] = {"edge_prior_mode": "none"}
        with self.assertRaisesRegex(ContractError, "inert interaction model fields"):
            validate_training_config(inert)

    def test_all_resolved_package_profiles_keep_their_dataset_recipe(self) -> None:
        for dataset in PACKAGE_CONFIGS:
            with self.subTest(dataset=dataset):
                source = package_config(dataset)
                resolved = copy.deepcopy(source)
                resolved["ckpt_dir"] = f"/runs/{dataset}/training"
                resolved["spatial_dim"] = 2
                resolved["model"]["spatial_dim"] = 2
                interaction = resolved["model"]["interaction_net"]
                interaction["cutoff"] = 0.08
                if interaction.get("edge_prior_mode", "learned") == "learned":
                    interaction[
                        "edge_predictor_path"
                    ] = f"/runs/{dataset}/preprocess/edge_classifier/{dataset}.pt"
                    interaction["edge_predictor_thre"] = 0.57
                for stage in resolved["training"]["plan"]:
                    stage["sigma"] = 0.03
                report = validate_training_config(
                    resolved,
                    runtime_resolved=True,
                    reference=source,
                )
                self.assertTrue(report["runtime_resolved"])
                fields = report["runtime_resolved_fields"]
                self.assertEqual(
                    fields["model.interaction_net.edge_prior_mode"],
                    "learned",
                )
                self.assertEqual(
                    fields["model.interaction_net.edge_predictor_thre"], 0.57
                )

    def test_only_historical_missing_learned_mode_matches_explicit_reference(
        self,
    ) -> None:
        reference = locked_config()
        historical = runtime_config()
        historical["model"]["interaction_net"].pop("edge_prior_mode")

        report = validate_training_config(
            historical,
            runtime_resolved=True,
            reference=reference,
        )
        self.assertEqual(report["edge_prior_mode"], "learned")
        self.assertEqual(
            report["runtime_resolved_fields"]["model.interaction_net.edge_prior_mode"],
            "learned",
        )

        all_spatial_reference = all_spatial_config()
        with self.assertRaisesRegex(ContractError, "package YAML"):
            validate_training_config(
                historical,
                runtime_resolved=True,
                reference=all_spatial_reference,
            )

        implicit_reference = locked_config()
        implicit_reference["model"]["interaction_net"].pop("edge_prior_mode")
        explicit_actual = runtime_config()
        with self.assertRaisesRegex(ContractError, "package YAML"):
            validate_training_config(
                explicit_actual,
                runtime_resolved=True,
                reference=implicit_reference,
            )

    def test_edge_prior_mode_controls_predictor_fields(self) -> None:
        learned_mutations = (
            (
                "missing path",
                lambda value: value["model"]["interaction_net"].pop(
                    "edge_predictor_path"
                ),
                "edge_predictor_path",
            ),
            (
                "blank path",
                lambda value: value["model"]["interaction_net"].__setitem__(
                    "edge_predictor_path", "  "
                ),
                "edge_predictor_path",
            ),
            (
                "missing threshold",
                lambda value: value["model"]["interaction_net"].pop(
                    "edge_predictor_thre"
                ),
                "edge_predictor_thre",
            ),
            (
                "invalid threshold",
                lambda value: value["model"]["interaction_net"].__setitem__(
                    "edge_predictor_thre", 1.0
                ),
                "edge_predictor_thre",
            ),
            (
                "non-numeric threshold",
                lambda value: value["model"]["interaction_net"].__setitem__(
                    "edge_predictor_thre", "not-a-number"
                ),
                "edge_predictor_thre",
            ),
        )
        for label, mutation, match in learned_mutations:
            with self.subTest(mode="learned", case=label):
                config = package_config("zebrafish")
                mutation(config)
                with self.assertRaisesRegex(ContractError, match):
                    validate_training_config(config)

        for key, value in (
            ("edge_predictor_path", None),
            ("edge_predictor_thre", 0.57),
            ("edge_predictor_threshold", 0.57),
        ):
            with self.subTest(mode="all_spatial", key=key):
                config = all_spatial_config()
                config["model"]["interaction_net"][key] = value
                with self.assertRaisesRegex(ContractError, "inert predictor keys"):
                    validate_training_config(config)

        config = all_spatial_config()
        config["model"]["interaction_net"]["edge_prior_mode"] = "unsupported"
        with self.assertRaisesRegex(ContractError, "edge_prior_mode"):
            validate_training_config(config)

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

    def test_all_spatial_runtime_cutoff_must_match_reference_config(self) -> None:
        reference = all_spatial_config()
        resolved = copy.deepcopy(reference)
        resolved["ckpt_dir"] = "/runs/no_lr/model"
        resolved["model"]["spatial_dim"] = 2
        validate_training_config(resolved, runtime_resolved=True, reference=reference)
        resolved["model"]["interaction_net"]["cutoff"] *= 2
        with self.assertRaisesRegex(
            ContractError, "resolved all_spatial interaction cutoff"
        ):
            validate_training_config(
                resolved, runtime_resolved=True, reference=reference
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

    def test_all_spatial_checkpoint_binding_has_no_predictor_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary).resolve()

            class Data:
                spatial_dim = 2
                interaction_graph = {}

            inventory = {
                "training_profile": {
                    "runtime_resolved_fields": {
                        "ckpt_dir": str(model),
                        "model.spatial_dim": 2,
                        "model.interaction_net.cutoff": 0.08,
                        "model.interaction_net.edge_prior_mode": "all_spatial",
                    }
                }
            }
            expected = _full_runtime_expectation(Data(), inventory)
            self.assertEqual(
                expected,
                {
                    "interaction_cutoff": 0.08,
                    "interaction_mode": "all_spatial",
                    "edge_prior_mode": "all_spatial",
                },
            )
            report = _checkpoint_runtime_binding(model, Data(), inventory, expected)
            self.assertEqual(report["edge_prior_mode"], "all_spatial")
            self.assertIsNone(report["edge_threshold"])
            self.assertIsNone(report["recorded_edge_predictor_path"])
            self.assertFalse(report["external_edge_predictor_required"])

            inventory["training_profile"]["runtime_resolved_fields"][
                "model.interaction_net.edge_predictor_path"
            ] = "/inert/edge.pt"
            with self.assertRaisesRegex(ContractError, "inert predictor fields"):
                _checkpoint_runtime_binding(model, Data(), inventory, expected)

    def test_no_interaction_checkpoint_binding_has_no_graph_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary).resolve()

            class Data:
                spatial_dim = 2
                interaction_graph = {
                    "neighborhood_threshold": 123.0,
                    "edge_predictor_threshold": 0.5,
                }

            inventory = {
                "training_profile": {
                    "runtime_resolved_fields": {
                        "ckpt_dir": str(model),
                        "model.spatial_dim": 2,
                        "interaction_mode": "none",
                        "edge_prior_mode": "none",
                    }
                }
            }
            expected = _full_runtime_expectation(Data(), inventory)
            self.assertEqual(
                expected,
                {"interaction_mode": "none", "edge_prior_mode": "none"},
            )
            report = _checkpoint_runtime_binding(model, Data(), inventory, expected)
            self.assertEqual(report["interaction_mode"], "none")
            self.assertEqual(report["edge_prior_mode"], "none")
            self.assertFalse(report["include_interaction"])
            self.assertNotIn("interaction_cutoff", report)

            inventory["training_profile"]["runtime_resolved_fields"][
                "model.interaction_net.cutoff"
            ] = 0.08
            with self.assertRaisesRegex(ContractError, "inert interaction fields"):
                _checkpoint_runtime_binding(model, Data(), inventory, expected)


class ModeAwareExecutionTests(unittest.TestCase):
    def test_learned_cutoff_uses_real_matched_config_without_h5ad_graph_uns(
        self,
    ) -> None:
        config = package_config("zebrafish")
        adata = ad.AnnData(np.zeros((2, 2), dtype=np.float32))
        cutoff, source = _resolve_interaction_cutoff(adata, config, None)
        self.assertEqual(cutoff, config["model"]["interaction_net"]["cutoff"])
        self.assertEqual(source, "training_config.model.interaction_net.cutoff")

    def test_full_learned_binding_uses_real_saved_config_schema_without_graph_uns(
        self,
    ) -> None:
        config = package_config("zebrafish")
        interaction = config["model"]["interaction_net"]
        fields = {
            "ckpt_dir": "/accepted/zebrafish/training",
            "model.spatial_dim": 2,
            "model.interaction_net.cutoff": interaction["cutoff"],
            "model.interaction_net.edge_prior_mode": interaction["edge_prior_mode"],
            "model.interaction_net.edge_predictor_path": interaction[
                "edge_predictor_path"
            ],
            "model.interaction_net.edge_predictor_thre": interaction[
                "edge_predictor_thre"
            ],
        }
        data = SimpleNamespace(spatial_dim=2, interaction_graph={})
        inventory = {"training_profile": {"runtime_resolved_fields": fields}}
        expected = _full_runtime_expectation(data, inventory)
        self.assertEqual(expected["interaction_mode"], "learned")
        self.assertEqual(expected["interaction_cutoff"], interaction["cutoff"])
        self.assertEqual(expected["edge_threshold"], interaction["edge_predictor_thre"])
        bound = _checkpoint_runtime_binding(
            Path("/accepted/zebrafish/training"), data, inventory, expected
        )
        self.assertEqual(bound["edge_predictor_source"], "embedded_finetune_checkpoint")
        self.assertFalse(bound["external_edge_predictor_required"])

    def test_inference_group_size_must_match_loaded_model(self) -> None:
        model = SimpleNamespace(interaction_group_size=1024)
        self.assertEqual(_validated_interaction_m(model, 1024), 1024)
        with self.assertRaisesRegex(ContractError, "must exactly match"):
            _validated_interaction_m(model, 512)
        for invalid in (None, 0, -1, 1.5, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ContractError, "positive integer"
            ):
                _validated_interaction_m(
                    SimpleNamespace(interaction_group_size=invalid), 1024
                )

    def test_learned_prepare_requires_database_at_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_mock_inputs(root / "inputs")
            with self.assertRaisesRegex(ContractError, "--database is required"):
                command_prepare_loto(
                    SimpleNamespace(
                        input_manifest=manifest,
                        split="loto_t1",
                        training_config=(
                            HERE.parents[2]
                            / "CytoBridge/configs/zebrafish_spatial_full_alpha_express_0015.yaml"
                        ),
                        output_dir=root / "prior",
                        database=None,
                        repo=HERE.parents[2],
                        device="cpu",
                        expression_layer="counts",
                        interaction_cutoff=None,
                        spot_diameter=None,
                        edge_threshold=None,
                        edge_epochs=1,
                        edge_batch_size=8,
                        edge_learning_rate=0.001,
                        edge_train_sample_ratio=1.0,
                        edge_max_train_edges=None,
                        edge_num_workers=0,
                        seed=42,
                        quiet=True,
                    )
                )

    def test_all_spatial_prepare_writes_no_graph_or_predictor_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_mock_inputs(root / "inputs")
            output = root / "prior"
            args = SimpleNamespace(
                input_manifest=manifest,
                split="loto_t1",
                training_config=ALL_SPATIAL_CONFIG,
                output_dir=output,
                database=None,
                repo=HERE.parents[2],
                device="cpu",
                expression_layer="counts",
                interaction_cutoff=None,
                spot_diameter=None,
                edge_threshold=None,
                edge_epochs=100,
                edge_batch_size=1024,
                edge_learning_rate=0.001,
                edge_train_sample_ratio=1.0,
                edge_max_train_edges=None,
                edge_num_workers=0,
                seed=42,
                quiet=True,
            )
            command_prepare_loto(args)
            summary_path = output / "prepare_graph_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["edge_prior_mode"], "all_spatial")
            self.assertEqual(summary["artifact_sha256"], {})
            self.assertEqual(
                summary["interaction_cutoff_source"],
                "training_config.model.interaction_net.cutoff",
            )
            for key in (
                "database",
                "edge_model",
                "edge_meta",
                "edge_threshold",
            ):
                self.assertNotIn(key, summary)
            self.assertEqual(
                [path.name for path in output.iterdir()],
                ["prepare_graph_summary.json"],
            )

            split = read_split_input(manifest, "loto_t1")
            loaded = _load_graph_summary(
                output,
                split,
                edge_prior_mode="all_spatial",
                training_config_sha256=sha256_file(ALL_SPATIAL_CONFIG),
            )
            self.assertAlmostEqual(loaded["interaction_cutoff"], 0.012106042891492197)
            with self.assertRaisesRegex(ContractError, "training-config SHA differs"):
                _load_graph_summary(
                    output,
                    split,
                    edge_prior_mode="all_spatial",
                    training_config_sha256="f" * 64,
                )

            summary["edge_threshold"] = 0.5
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "inert learned predictor"):
                _load_graph_summary(
                    output,
                    split,
                    edge_prior_mode="all_spatial",
                    training_config_sha256=sha256_file(ALL_SPATIAL_CONFIG),
                )

    def test_no_interaction_prepare_writes_only_a_mode_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_mock_inputs(root / "inputs")
            output = root / "prior"
            config_path = no_interaction_config_path()
            args = SimpleNamespace(
                input_manifest=manifest,
                split="loto_t1",
                training_config=config_path,
                output_dir=output,
                database=None,
                repo=HERE.parents[2],
                device="cpu",
                expression_layer="counts",
                interaction_cutoff=None,
                spot_diameter=None,
                edge_threshold=None,
                edge_epochs=100,
                edge_batch_size=1024,
                edge_learning_rate=0.001,
                edge_train_sample_ratio=1.0,
                edge_max_train_edges=None,
                edge_num_workers=0,
                seed=42,
                quiet=True,
            )
            command_prepare_loto(args)
            summary_path = output / "prepare_graph_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["interaction_mode"], "none")
            self.assertEqual(summary["edge_prior_mode"], "none")
            self.assertEqual(summary["artifact_sha256"], {})
            self.assertEqual(
                summary["graph_generation"],
                "not_applicable_no_interaction_component",
            )
            for key in (
                "interaction_cutoff",
                "interaction_cutoff_source",
                "observed_stage_graphs",
                "database",
                "edge_model",
                "edge_meta",
                "edge_threshold",
            ):
                self.assertNotIn(key, summary)
            self.assertEqual(
                [path.name for path in output.iterdir()],
                ["prepare_graph_summary.json"],
            )

            split = read_split_input(manifest, "loto_t1")
            loaded = _load_graph_summary(
                output,
                split,
                edge_prior_mode="none",
                training_config_sha256=sha256_file(config_path),
            )
            self.assertEqual(loaded["interaction_mode"], "none")
            summary["interaction_cutoff"] = 0.1
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "inert graph fields"):
                _load_graph_summary(
                    output,
                    split,
                    edge_prior_mode="none",
                    training_config_sha256=sha256_file(config_path),
                )

    def test_all_spatial_cli_values_are_fail_closed(self) -> None:
        configured = all_spatial_config()["model"]["interaction_net"]["cutoff"]
        value, source = _all_spatial_cutoff(all_spatial_config(), configured + 1e-13)
        self.assertEqual(value, configured)
        self.assertEqual(source, "explicit_cli_matches_training_config")
        with self.assertRaisesRegex(ContractError, "must match"):
            _all_spatial_cutoff(all_spatial_config(), configured * 2)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_mock_inputs(root / "inputs")
            base = dict(
                input_manifest=manifest,
                split="loto_t1",
                training_config=ALL_SPATIAL_CONFIG,
                output_dir=root / "prior",
                database=None,
                repo=HERE.parents[2],
                device="cpu",
                expression_layer="counts",
                interaction_cutoff=None,
                edge_epochs=1,
                edge_batch_size=8,
                edge_learning_rate=0.001,
                edge_train_sample_ratio=1.0,
                edge_max_train_edges=None,
                edge_num_workers=0,
                seed=42,
                quiet=True,
            )
            for field, flag in (
                ("edge_threshold", "--edge-threshold"),
                ("spot_diameter", "--spot-diameter"),
            ):
                args = SimpleNamespace(
                    **base,
                    edge_threshold=0.5 if field == "edge_threshold" else None,
                    spot_diameter=0.1 if field == "spot_diameter" else None,
                )
                with self.subTest(option=flag), self.assertRaisesRegex(
                    ContractError, flag
                ):
                    command_prepare_loto(args)
                self.assertFalse((root / "prior").exists())

            mismatch = SimpleNamespace(
                **{
                    **base,
                    "interaction_cutoff": configured * 2,
                    "edge_threshold": None,
                    "spot_diameter": None,
                }
            )
            with self.assertRaisesRegex(ContractError, "must match"):
                command_prepare_loto(mismatch)
            self.assertFalse((root / "prior").exists())

    def test_all_spatial_fit_passes_no_predictor_config_or_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "model"
            split = SimpleNamespace(
                regime="loto",
                holdout_time=1.0,
                split_id="loto_t1",
                train_h5ad=root / "train.h5ad",
            )
            data = SimpleNamespace(
                spatial_dim=2,
                state_key="benchmark_state",
                spatial_key="benchmark_spatial",
                time_key="benchmark_time",
            )
            config_source = ALL_SPATIAL_CONFIG
            cutoff = all_spatial_config()["model"]["interaction_net"]["cutoff"]
            graph = {
                "edge_prior_mode": "all_spatial",
                "interaction_cutoff": cutoff,
                "_summary_path": str(root / "prior/prepare_graph_summary.json"),
                "_summary_sha256": "a" * 64,
            }
            inventory = {
                "config_sha256": "b" * 64,
                "stage_complete": True,
                "checkpoints": {},
            }
            captured_binding = {}

            def new_output(path):
                path.mkdir(parents=True)
                return path

            def match(*unused, **kwargs):
                captured_binding.update(kwargs["runtime_binding"])
                return {"proof": "mock"}

            fit = mock.Mock()
            fake_cytobridge = SimpleNamespace(tl=SimpleNamespace(fit=fit))
            with (
                mock.patch("run_cytobridge.read_split_input", return_value=split),
                mock.patch("run_cytobridge.load_training_data", return_value=data),
                mock.patch("run_cytobridge._input_report", return_value={}),
                mock.patch("run_cytobridge._load_graph_summary", return_value=graph),
                mock.patch("run_cytobridge.new_output_dir", side_effect=new_output),
                mock.patch(
                    "run_cytobridge.checkpoint_inventory", return_value=inventory
                ),
                mock.patch(
                    "run_cytobridge.checkpoint_training_match", side_effect=match
                ),
                mock.patch("run_cytobridge.input_provenance", return_value={}),
                mock.patch("run_cytobridge.environment_provenance", return_value={}),
                mock.patch("run_cytobridge.repo_identity", return_value={}),
                mock.patch.dict(sys.modules, {"CytoBridge": fake_cytobridge}),
            ):
                command_fit_loto(
                    SimpleNamespace(
                        input_manifest=root / "manifest.json",
                        split="loto_t1",
                        training_config=config_source,
                        graph_dir=root / "prior",
                        output_dir=output,
                        repo=HERE.parents[2],
                        device="cpu",
                        sigma=0.03,
                        seed=42,
                    )
                )

            fit_kwargs = fit.call_args.kwargs
            self.assertNotIn("edge_predictor_path", fit_kwargs)
            self.assertNotIn("edge_predictor_threshold", fit_kwargs)
            interaction = fit_kwargs["config"]["model"]["interaction_net"]
            self.assertEqual(interaction["edge_prior_mode"], "all_spatial")
            self.assertNotIn("edge_predictor_path", interaction)
            self.assertNotIn("edge_predictor_thre", interaction)
            self.assertEqual(
                captured_binding,
                {
                    "interaction_cutoff": cutoff,
                    "interaction_mode": "all_spatial",
                    "edge_prior_mode": "all_spatial",
                },
            )
            report = json.loads(
                (output / "benchmark_fit_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["edge_prior_mode"], "all_spatial")
            self.assertNotIn("edge_model", report)
            self.assertNotIn("edge_threshold", report)

    def test_no_interaction_fit_passes_no_graph_or_interaction_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "model"
            split = SimpleNamespace(
                regime="loto",
                holdout_time=1.0,
                split_id="loto_t1",
                train_h5ad=root / "train.h5ad",
            )
            data = SimpleNamespace(
                spatial_dim=2,
                state_key="benchmark_state",
                spatial_key="benchmark_spatial",
                time_key="benchmark_time",
            )
            config_source = no_interaction_config_path()
            prepared = {
                "interaction_mode": "none",
                "edge_prior_mode": "none",
                "_summary_path": str(root / "prior/prepare_graph_summary.json"),
                "_summary_sha256": "a" * 64,
            }
            inventory = {
                "config_sha256": "b" * 64,
                "stage_complete": True,
                "checkpoints": {},
            }
            captured_binding = {}

            def new_output(path):
                path.mkdir(parents=True)
                return path

            def match(*unused, **kwargs):
                captured_binding.update(kwargs["runtime_binding"])
                return {"proof": "mock"}

            fit = mock.Mock()
            fake_cytobridge = SimpleNamespace(tl=SimpleNamespace(fit=fit))
            with (
                mock.patch("run_cytobridge.read_split_input", return_value=split),
                mock.patch("run_cytobridge.load_training_data", return_value=data),
                mock.patch("run_cytobridge._input_report", return_value={}),
                mock.patch("run_cytobridge._load_graph_summary", return_value=prepared),
                mock.patch("run_cytobridge.new_output_dir", side_effect=new_output),
                mock.patch(
                    "run_cytobridge.checkpoint_inventory", return_value=inventory
                ),
                mock.patch(
                    "run_cytobridge.checkpoint_training_match", side_effect=match
                ),
                mock.patch("run_cytobridge.input_provenance", return_value={}),
                mock.patch("run_cytobridge.environment_provenance", return_value={}),
                mock.patch("run_cytobridge.repo_identity", return_value={}),
                mock.patch.dict(sys.modules, {"CytoBridge": fake_cytobridge}),
            ):
                command_fit_loto(
                    SimpleNamespace(
                        input_manifest=root / "manifest.json",
                        split="loto_t1",
                        training_config=config_source,
                        graph_dir=root / "prior",
                        output_dir=output,
                        repo=HERE.parents[2],
                        device="cpu",
                        sigma=0.03,
                        seed=42,
                    )
                )

            fit_kwargs = fit.call_args.kwargs
            self.assertNotIn("interaction_cutoff", fit_kwargs)
            self.assertNotIn("edge_predictor_path", fit_kwargs)
            self.assertNotIn("edge_predictor_threshold", fit_kwargs)
            self.assertNotIn("interaction_net", fit_kwargs["config"]["model"])
            self.assertEqual(
                fit_kwargs["config"]["model"]["components"],
                ["velocity", "growth", "score"],
            )
            self.assertEqual(
                captured_binding,
                {"interaction_mode": "none", "edge_prior_mode": "none"},
            )
            report = json.loads(
                (output / "benchmark_fit_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["interaction_mode"], "none")
            self.assertEqual(report["edge_prior_mode"], "none")
            self.assertNotIn("interaction_cutoff", report)
            self.assertNotIn("edge_model", report)
            self.assertNotIn("edge_threshold", report)

    def test_no_interaction_inference_runs_velocity_growth_and_score_only(self) -> None:
        source = no_interaction_config()
        resolved = copy.deepcopy(source)
        resolved["ckpt_dir"] = "/runs/zebrafish/no_interaction"
        resolved["model"]["spatial_dim"] = 2
        for stage in resolved["training"]["plan"]:
            stage["sigma"] = 0.03
        loaded = SimpleNamespace(
            config=resolved,
            model=SimpleNamespace(components=["velocity", "growth", "score"]),
            weight_stage="Finetune_no_interaction",
            score_stage="Score_Refine",
        )
        data = SimpleNamespace(
            joint_dim=5,
            time_key="benchmark_time",
            state_key="benchmark_state",
            spatial_key="benchmark_spatial",
        )
        captured = {}

        def simulate(
            *,
            adata,
            model,
            dim,
            time_index,
            n_samples,
            ts_points,
            dt,
            sigma,
            include_score,
            interaction_m,
            device,
            time_key,
            obsm_key,
            spatial_key,
            concat_spatial,
            interaction_seed,
            verbose,
        ):
            captured.update(
                {
                    "adata": adata,
                    "model": model,
                    "dim": dim,
                    "time_index": time_index,
                    "n_samples": n_samples,
                    "ts_points": ts_points,
                    "dt": dt,
                    "sigma": sigma,
                    "include_score": include_score,
                    "interaction_m": interaction_m,
                    "device": device,
                    "time_key": time_key,
                    "obsm_key": obsm_key,
                    "spatial_key": spatial_key,
                    "concat_spatial": concat_spatial,
                    "interaction_seed": interaction_seed,
                    "verbose": verbose,
                }
            )
            points = [
                np.zeros((PREDICTION_N, data.joint_dim), dtype=np.float32)
                for _ in ts_points
            ]
            weights = [np.ones(PREDICTION_N, dtype=np.float64) for _ in ts_points]
            return points, weights

        cytobridge_module = ModuleType("CytoBridge")
        tl_module = ModuleType("CytoBridge.tl")
        downstream_module = ModuleType("CytoBridge.tl.downstream")
        simulation_module = ModuleType("CytoBridge.tl.downstream.simulation")
        tl_module.load_dynamical_model_from_dir = mock.Mock(return_value=loaded)
        simulation_module.simulate_sde_points = simulate
        downstream_module.simulation = simulation_module
        tl_module.downstream = downstream_module
        cytobridge_module.tl = tl_module
        with mock.patch.dict(
            sys.modules,
            {
                "CytoBridge": cytobridge_module,
                "CytoBridge.tl": tl_module,
                "CytoBridge.tl.downstream": downstream_module,
                "CytoBridge.tl.downstream.simulation": simulation_module,
            },
        ):
            points, weights, report = _simulate(
                repo=HERE.parents[2],
                model_dir=Path("/runs/zebrafish/no_interaction"),
                data=data,
                bootstrap=object(),
                times=[0.0, 1.0],
                device="cpu",
                dt=0.01,
                interaction_m=999,
                expected_interaction_mode="none",
            )

        self.assertEqual(len(points), 2)
        self.assertEqual(len(weights), 2)
        self.assertTrue(captured["include_score"])
        self.assertEqual(captured["interaction_m"], 1)
        self.assertEqual(captured["interaction_seed"], 10_042)
        self.assertEqual(report["interaction_mode"], "none")
        self.assertEqual(report["edge_prior_mode"], "none")
        self.assertFalse(report["include_interaction"])
        self.assertFalse(report["edge_predictor_used"])
        self.assertIsNone(report["interaction_m"])
        self.assertIsNone(report["interaction_grouping_seed"])
        self.assertEqual(report["dynamics_components"], ["velocity", "growth", "score"])


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

    def test_run_summary_contains_one_compact_record_per_target(self) -> None:
        summaries = [
            {
                "target": target,
                "prediction_npz": f"t{target}/prediction.npz",
                "prediction_npz_sha256": str(target) * 64,
                "predicted_mass": float(target),
                "unrelated_detail": "not copied",
            }
            for target in (1, 2, 3, 4)
        ]
        compact = _compact_prediction_summaries(summaries)
        self.assertEqual([row["target"] for row in compact], [1, 2, 3, 4])
        self.assertEqual(len(compact), len(summaries))
        self.assertTrue(all("unrelated_detail" not in row for row in compact))
        with self.assertRaisesRegex(ContractError, "repeat target"):
            _compact_prediction_summaries([summaries[0], summaries[0]])


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

            payload["training_config_source_sha256"] = "c" * 64
            summary.write_text(json.dumps(payload), encoding="utf-8")
            report = checkpoint_training_match(
                model,
                split,
                data,
                inventory=inventory,
                reference_config_sha256="c" * 64,
            )
            self.assertEqual(report["proof"], "benchmark_fit_summary")
            with self.assertRaisesRegex(ContractError, "training-config SHA differs"):
                checkpoint_training_match(
                    model,
                    split,
                    data,
                    inventory=inventory,
                    reference_config_sha256="d" * 64,
                )

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
                    else (
                        "last_model.pth"
                        if str(stage.get("save_strategy", "best")).lower() == "last"
                        else "best_model.pth"
                    )
                )
                (folder / filename).write_bytes(b"mock checkpoint")
            report = checkpoint_inventory(model)
            self.assertTrue(report["stage_complete"])
            self.assertEqual(report["stage_count"], 6)
            self.assertTrue(report["training_profile"]["runtime_resolved"])
            (model / "Score_Refine" / "score_model.pth").unlink()
            with self.assertRaisesRegex(ContractError, "incomplete six-stage"):
                checkpoint_inventory(model)

    def test_all_spatial_checkpoint_cutoff_matches_reference_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            reference = all_spatial_config()
            resolved = copy.deepcopy(reference)
            resolved["ckpt_dir"] = str(model)
            resolved["model"]["spatial_dim"] = 2
            (model / "config.yaml").write_text(
                yaml.safe_dump(resolved), encoding="utf-8"
            )
            for stage in resolved["training"]["plan"]:
                folder = model / stage["name"]
                folder.mkdir()
                filename = (
                    "score_model.pth"
                    if stage["mode"] == "score_matching"
                    else (
                        "last_model.pth"
                        if str(stage.get("save_strategy", "best")).lower() == "last"
                        else "best_model.pth"
                    )
                )
                (folder / filename).write_bytes(b"mock checkpoint")
            checkpoint_inventory(model, reference_config=reference)
            resolved["model"]["interaction_net"]["cutoff"] *= 2
            (model / "config.yaml").write_text(
                yaml.safe_dump(resolved), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ContractError, "resolved all_spatial interaction cutoff"
            ):
                checkpoint_inventory(model, reference_config=reference)

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
