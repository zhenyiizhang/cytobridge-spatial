from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.spatiotemporal_benchmark.dynamic.common import (
    CONTRACT_UNS_KEY,
    PREDICTION_N,
    ContractError,
    load_training_data,
    read_split_contract,
    sha256_file,
)
from scripts.spatiotemporal_benchmark.dynamic.run_dynamic import (
    DEFAULT_PARAMS,
    fit_method,
    infer_method,
    load_method_pins,
    preflight,
)


GIT = {
    "source_root": "/mock/official",
    "source_git_commit": "a" * 40,
    "source_remote": "https://example.test/official.git",
    "source_tracked_tree_clean": True,
}


class FakeSTVCRModel(torch.nn.Module):
    def forward(self, value):
        return value


def fake_train_stvcr(
    adata,
    *,
    model_path,
    rigid_transformation_path,
    config,
    use_alignment,
    use_growth,
    device,
):
    del adata, config, use_growth, device
    assert use_alignment is False
    torch.save(FakeSTVCRModel(), model_path)
    Path(rigid_transformation_path).write_bytes(b"mock-rigid")


def fake_growth_simulator(initial, model, start, target, *, spatial_dim, delta_t):
    del model, start, delta_t
    values = initial.detach().cpu().numpy()[:4250]
    return (
        [values[:, :spatial_dim], values[:, :spatial_dim] + float(target)],
        [values[:, spatial_dim:], values[:, spatial_dim:] + float(target)],
        [],
        [],
        [target],
    )


FAKE_STVCR_PACKAGE = types.SimpleNamespace(__version__="mock-stvcr-1")


class FakeSpaceTime:
    fit_calls = 0

    def __init__(self, proximal_step=None):
        self.proximal_step = proximal_step

    def fit(self, adata, **kwargs):
        del adata, kwargs
        type(self).fit_calls += 1

    def transform(self, adata, *, omics_key, tau, batch_size, key):
        del batch_size, key
        return np.asarray(adata.obsm[omics_key]) + float(tau)


class FakeJaxRandom:
    @staticmethod
    def PRNGKey(seed):
        return int(seed)


FAKE_STORIES = types.SimpleNamespace(
    __version__="mock-stories-1",
    steps=types.SimpleNamespace(ExplicitStep=lambda: object()),
    SpaceTime=FakeSpaceTime,
)
FAKE_JAX = types.SimpleNamespace(random=FakeJaxRandom())


class FakeODE(torch.nn.Module):
    def forward(self, time, state):
        del time
        return torch.zeros_like(state)

    def reset_momentum(self):
        return None


class FakeMIOFlow:
    def __init__(self, adata, **kwargs):
        del kwargs
        state = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        self.mean_vals = state.mean(axis=0)
        std = state.std(axis=0)
        self.std_vals = np.where(std <= 0, 1.0, std)
        self.ode_model = FakeODE()
        self.device = "cpu"
        self.use_sde = False

    def fit(self):
        return self


FAKE_MIOFLOW_MODULE = types.SimpleNamespace(__version__="mock-mioflow-1")


class FixtureMixin:
    state_dim = 4
    spatial_dim = 2

    def _write_split(
        self,
        root: Path,
        split_id: str,
        times: list[int],
        holdout: int | None,
        rng: np.random.Generator,
    ) -> dict:
        rows = []
        for time in times:
            for cell in range(5):
                rows.append((time, f"{split_id}-{time}-{cell}"))
        time = np.asarray([value[0] for value in rows], dtype=np.int16)
        row_id = np.asarray([value[1] for value in rows])
        spatial = rng.normal(size=(len(rows), self.spatial_dim)).astype(np.float32)
        state = rng.normal(size=(len(rows), self.state_dim)).astype(np.float32)
        split_dir = root / split_id
        split_dir.mkdir()
        h5ad_path = split_dir / "train.h5ad"
        reference_path = split_dir / "training_reference.npz"
        roster_path = split_dir / "source_roster.npz"
        data = ad.AnnData(
            X=np.zeros((len(rows), 1), dtype=np.float32),
            obs=pd.DataFrame(
                {"benchmark_time": time, "row_id": row_id},
                index=pd.Index(row_id, name="row_id"),
            ),
        )
        data.obsm["benchmark_state_pca"] = state
        data.obsm["spatial_aligned"] = spatial
        data.uns[CONTRACT_UNS_KEY] = {
            "version": "test-v1",
            "dataset": "fixture",
            "split": split_id,
            "role": "train" if holdout is not None else "train_and_truth",
            "target_removed": holdout is not None,
            "held_out_benchmark_time": "none" if holdout is None else holdout,
            "prediction_n": PREDICTION_N,
            "time_key": "benchmark_time",
            "state_key": "benchmark_state_pca",
            "spatial_key": "spatial_aligned",
            "row_id_key": "row_id",
            "time_values": np.asarray(times, dtype=np.int16),
            "transductive_frozen_representation": True,
        }
        data.write_h5ad(h5ad_path)
        np.savez_compressed(
            reference_path,
            state=state,
            spatial=spatial,
            time=time,
            row_id=row_id.astype(str),
            annotation=np.asarray(["cell"] * len(rows)),
        )
        source_time = min(times) if holdout is None else max(
            value for value in times if value < holdout
        )
        candidates = np.flatnonzero(time == source_time)
        roster_indices = np.resize(candidates, PREDICTION_N)
        np.savez_compressed(
            roster_path,
            indices=roster_indices,
            row_id=row_id[roster_indices].astype(str),
            source_time=np.asarray([source_time], dtype=np.int16),
            state=state[roster_indices],
            spatial=spatial[roster_indices],
        )
        return {
            "protocol": "full_data" if holdout is None else "leave_one_timepoint_out",
            "held_out_benchmark_time": holdout,
            "evaluation_targets": [1, 2, 3, 4] if holdout is None else [holdout],
            "target_rows_physically_removed_from_train": holdout is not None,
            "prediction_n": PREDICTION_N,
            "train_time_counts": {
                str(value): int(np.count_nonzero(time == value)) for value in range(5)
            },
            "train": {
                "h5ad": {"path": str(h5ad_path), "sha256": sha256_file(h5ad_path)},
                "training_reference_npz": {
                    "path": str(reference_path),
                    "sha256": sha256_file(reference_path),
                },
                "source_roster_npz": {
                    "path": str(roster_path),
                    "sha256": sha256_file(roster_path),
                },
            },
            "truth_by_time_npz": {
                str(value): {"path": str(split_dir / f"DO_NOT_OPEN_truth_t{value}.npz")}
                for value in ([holdout] if holdout is not None else [1, 2, 3, 4])
            },
        }

    def make_manifest(self, root: Path) -> Path:
        rng = np.random.default_rng(7)
        splits = {
            "full_data": self._write_split(
                root, "full_data", [0, 1, 2, 3, 4], None, rng
            ),
            "loto_t2": self._write_split(root, "loto_t2", [0, 1, 3, 4], 2, rng),
        }
        path = root / "manifest.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": "fixture",
                    "prediction_n": PREDICTION_N,
                    "contract_uns_key": CONTRACT_UNS_KEY,
                    "loto_targets": [2],
                    "full_data_targets": [1, 2, 3, 4],
                    "splits": splits,
                }
            ),
            encoding="utf-8",
        )
        return path


class ContractTests(FixtureMixin, unittest.TestCase):
    def test_official_commits_are_full_and_fixed(self):
        methods = load_method_pins()["methods"]
        self.assertEqual(
            methods["stvcr"]["commit"],
            "26aa79a63eba7a5e21726b1eb95bf6bb61cfe699",
        )
        self.assertEqual(
            methods["stories"]["commit"],
            "7d8269b8cc940b3d85c34729dd1f715822e74a97",
        )
        self.assertEqual(
            methods["mioflow"]["commit"],
            "36365403d0f23cc3ad1065781c7331bf81debf4e",
        )

    def test_loto_target_is_physically_absent_and_reference_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.make_manifest(Path(temporary))
            split = read_split_contract(manifest, "loto_t2")
            data = load_training_data(split)
            self.assertEqual(split.observed_times, (0.0, 1.0, 3.0, 4.0))
            self.assertNotIn(2.0, set(data.time))
            self.assertEqual(data.state.shape[1], self.state_dim)

    def test_missing_cytobridge_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.make_manifest(Path(temporary))
            split = read_split_contract(manifest, "loto_t2")
            data = ad.read_h5ad(split.train_h5ad)
            del data.uns[CONTRACT_UNS_KEY]
            data.write_h5ad(split.train_h5ad)
            payload = json.loads(manifest.read_text())
            payload["splits"]["loto_t2"]["train"]["h5ad"]["sha256"] = sha256_file(
                split.train_h5ad
            )
            manifest.write_text(json.dumps(payload))
            split = read_split_contract(manifest, "loto_t2")
            with self.assertRaisesRegex(ContractError, "lacks uns"):
                load_training_data(split)

    def test_reference_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.make_manifest(Path(temporary))
            payload = json.loads(manifest.read_text())
            payload["splits"]["loto_t2"]["train"]["training_reference_npz"][
                "sha256"
            ] = ("0" * 64)
            manifest.write_text(json.dumps(payload))
            split = read_split_contract(manifest, "loto_t2")
            with self.assertRaisesRegex(ContractError, "SHA-256"):
                load_training_data(split)


class AdapterTests(FixtureMixin, unittest.TestCase):
    def _fit_patches(self, method: str):
        stack = ExitStack()
        stack.enter_context(
            patch(
                "scripts.spatiotemporal_benchmark.dynamic.run_dynamic._source_file",
                return_value="/mock/official/api.py",
            )
        )
        if method == "stvcr":
            stack.enter_context(patch(
                "scripts.spatiotemporal_benchmark.dynamic.run_dynamic._import_stvcr",
                return_value=(
                    FAKE_STVCR_PACKAGE,
                    {},
                    fake_train_stvcr,
                    fake_growth_simulator,
                ),
            ))
        elif method == "stories":
            stack.enter_context(patch(
                "scripts.spatiotemporal_benchmark.dynamic.run_dynamic._import_stories",
                return_value=(FAKE_STORIES, FAKE_JAX),
            ))
        else:
            stack.enter_context(patch(
                "scripts.spatiotemporal_benchmark.dynamic.run_dynamic._import_mioflow",
                return_value=(FakeMIOFlow, "fake.mioflow", FAKE_MIOFLOW_MODULE),
            ))
        return stack

    def _run(self, method: str, root: Path, manifest: Path, split: str, target: int):
        fit_dir = root / "runs" / method / split / "fit"
        pred_dir = root / "runs" / method / split / f"t{target}"
        with patch(
            "scripts.spatiotemporal_benchmark.dynamic.run_dynamic._pinned_source_identity",
            return_value=GIT,
        ), self._fit_patches(method):
            fit = fit_method(
                method,
                manifest,
                split,
                root,
                fit_dir,
                seed=13,
                params=dict(DEFAULT_PARAMS[method]),
            )
            self.assertTrue((fit_dir / "summary.json").is_file())
            prediction = infer_method(
                method,
                manifest,
                split,
                root,
                fit_dir,
                target,
                pred_dir,
                seed=13,
            )
        return fit, prediction, pred_dir

    def test_preflight_full_data_uses_t0_for_all_four_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_manifest(root)
            with patch(
                "scripts.spatiotemporal_benchmark.dynamic.run_dynamic._pinned_source_identity",
                return_value=GIT,
            ):
                result = preflight(
                    "stories", manifest, "full_data", root, check_import=False
                )
            self.assertEqual(
                [row["target_time"] for row in result["target_plan"]], [1, 2, 3, 4]
            )
            self.assertEqual(
                {row["source_time"] for row in result["target_plan"]}, {0.0}
            )

    def test_stvcr_preserves_native_growth_and_initial_mass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_manifest(root)
            fit, result, pred_dir = self._run("stvcr", root, manifest, "loto_t2", 2)
            with np.load(pred_dir / "prediction.npz") as archive:
                self.assertEqual(archive["state"].shape, (4250, self.state_dim))
                self.assertEqual(archive["spatial"].shape, (4250, self.spatial_dim))
                np.testing.assert_allclose(archive["weights"], 1 / PREDICTION_N)
                self.assertAlmostEqual(float(archive["weights"].sum()), 0.85, places=5)
            self.assertEqual(fit["initial_n"], PREDICTION_N)
            self.assertEqual(result["native_output_n"], 4250)
            self.assertTrue(result["native_growth"])
            self.assertEqual(result["source_time"], 1.0)
            self.assertFalse(result["truth_inputs_opened"])

    def test_stories_full_fit_once_and_no_segment_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_manifest(root)
            fit_dir = root / "stories-fit"
            FakeSpaceTime.fit_calls = 0
            with patch(
                "scripts.spatiotemporal_benchmark.dynamic.run_dynamic._pinned_source_identity",
                return_value=GIT,
            ), self._fit_patches("stories"):
                fit_method(
                    "stories",
                    manifest,
                    "full_data",
                    root,
                    fit_dir,
                    seed=17,
                    params=dict(DEFAULT_PARAMS["stories"]),
                )
                roster_hashes = set()
                infer_seeds = set()
                for target in (1, 2, 3, 4):
                    result = infer_method(
                        "stories",
                        manifest,
                        "full_data",
                        root,
                        fit_dir,
                        target,
                        root / f"stories-t{target}",
                        seed=17,
                    )
                    self.assertEqual(result["source_time"], 0.0)
                    self.assertEqual(result["native_output_n"], PREDICTION_N)
                    roster_hashes.add(result["source_roster_sha256"])
                    infer_seeds.add(result["infer_seed"])
            self.assertEqual(FakeSpaceTime.fit_calls, 1)
            self.assertEqual(len(roster_hashes), 1)
            self.assertEqual(len(infer_seeds), 1)

    def test_state_only_methods_return_5000_and_mioflow_inverts_transform(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_manifest(root)
            for method in ("stories", "mioflow"):
                with self.subTest(method=method):
                    _, result, pred_dir = self._run(
                        method, root, manifest, "loto_t2", 2
                    )
                    with np.load(pred_dir / "prediction.npz") as archive:
                        self.assertEqual(archive.files, ["state"])
                        self.assertEqual(
                            archive["state"].shape, (PREDICTION_N, self.state_dim)
                        )
                    self.assertEqual(result["output_scope"], "native_state")
                    if method == "mioflow":
                        self.assertTrue(result["state_inverse_transform_applied"])

    def test_source_row_selection_is_shared_across_dynamic_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_manifest(root)
            hashes = set()
            for method in ("stories", "mioflow"):
                with patch(
                    "scripts.spatiotemporal_benchmark.dynamic.run_dynamic._pinned_source_identity",
                    return_value=GIT,
                ), self._fit_patches(method):
                    result = fit_method(
                        method,
                        manifest,
                        "full_data",
                        root,
                        root / f"{method}-fit",
                        seed=29,
                        params=dict(DEFAULT_PARAMS[method]),
                    )
                hashes.add(result["source_roster"]["source_indices_sha256"])
                self.assertTrue(
                    result["source_roster"]["shared_across_dynamic_methods"]
                )
            self.assertEqual(len(hashes), 1)

    def test_full_data_rejects_non_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_manifest(root)
            fit_dir = root / "fit"
            with patch(
                "scripts.spatiotemporal_benchmark.dynamic.run_dynamic._pinned_source_identity",
                return_value=GIT,
            ), self._fit_patches("stories"):
                fit_method(
                    "stories",
                    manifest,
                    "full_data",
                    root,
                    fit_dir,
                    seed=1,
                    params=dict(DEFAULT_PARAMS["stories"]),
                )
                with self.assertRaisesRegex(ContractError, "evaluation_targets"):
                    infer_method(
                        "stories",
                        manifest,
                        "full_data",
                        root,
                        fit_dir,
                        9,
                        root / "bad",
                        seed=1,
                    )


if __name__ == "__main__":
    unittest.main()
