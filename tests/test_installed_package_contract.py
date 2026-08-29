"""Wheel-install contract tests.

These tests are skipped in a source-tree suite. ``smoke_installed_wheel.py``
runs this file directly inside a clean, dependency-free virtual environment
with ``CYTOBRIDGE_TEST_INSTALLED=1``.
"""

from __future__ import annotations

from importlib import metadata, resources
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_INSTALLED_TESTS = os.environ.get("CYTOBRIDGE_TEST_INSTALLED") == "1"


@unittest.skipUnless(RUN_INSTALLED_TESTS, "requires an isolated wheel installation")
class InstalledPackageContractTests(unittest.TestCase):
    def test_metadata_and_console_entry_point(self) -> None:
        import CytoBridge

        distribution = metadata.distribution("CytoBridge")
        self.assertEqual(distribution.metadata["Name"], "CytoBridge")
        self.assertEqual(
            {
                part.strip()
                for part in distribution.metadata["Requires-Python"].split(",")
            },
            {">=3.10", "<3.12"},
        )
        self.assertEqual(distribution.version, CytoBridge.__version__)
        self.assertNotIn(PROJECT_ROOT, Path(CytoBridge.__file__).resolve().parents)
        self.assertEqual(
            set(distribution.metadata.get_all("Provides-Extra") or []),
            {
                "all",
                "docs",
                "graph",
                "notebook",
                "plot",
                "preprocess",
                "spatial",
                "train",
                "velocity",
            },
        )
        requirements = distribution.requires or []
        # Core Metadata permits either quote style around marker values.
        normalized_requirements = [
            item.replace(" ", "").replace('"', "'").lower() for item in requirements
        ]
        self.assertTrue(
            any(item.startswith("numpy<2,>=1.24") for item in normalized_requirements)
        )
        self.assertTrue(
            any(
                "torch-geometric<3,>=2.4" in item and "extra=='graph'" in item
                for item in normalized_requirements
            )
        )
        self.assertFalse(any("torchvision" in item for item in normalized_requirements))
        self.assertFalse(any("torchaudio" in item for item in normalized_requirements))
        self.assertTrue(
            any(
                item.startswith("pypdf<7,>=5") and "extra=='plot'" in item
                for item in normalized_requirements
            )
        )

        entry_points = {
            entry_point.name: entry_point.value
            for entry_point in distribution.entry_points
            if entry_point.group == "console_scripts"
        }
        self.assertEqual(entry_points["cytobridge"], "CytoBridge.cli:main")

        executable = Path(sys.executable).with_name("cytobridge")
        completed = subprocess.run(
            [str(executable), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.stdout.strip(), f"cytobridge {CytoBridge.__version__}"
        )

    def test_top_level_import_and_doctor_need_no_installed_dependencies(self) -> None:
        import CytoBridge

        self.assertNotIn("scanpy", sys.modules)
        self.assertNotIn("matplotlib", sys.modules)

        executable = Path(sys.executable).with_name("cytobridge")
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [str(executable), "doctor", "--json"],
                cwd=temporary_directory,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

        report = json.loads(completed.stdout)
        self.assertEqual(report["package"]["installed_version"], CytoBridge.__version__)
        self.assertTrue(report["package"]["version_match"])
        self.assertTrue(
            all(value is False for value in report["dependencies"].values())
        )
        self.assertEqual(
            set(report["profiles"]),
            {
                "all",
                "core",
                "docs",
                "graph",
                "notebook",
                "plot",
                "preprocess",
                "spatial",
                "train",
                "velocity",
            },
        )
        self.assertTrue(
            all(
                profile["available"] is False for profile in report["profiles"].values()
            )
        )

    def test_config_package_data_is_present(self) -> None:
        configs = resources.files("CytoBridge").joinpath("configs")
        self.assertEqual(
            {path.name for path in configs.iterdir() if path.suffix == ".yaml"},
            {
                "admouse_spatial_full_alpha_express_0015.yaml",
                "admouse_spatial_full_alpha_express_0015_no_interaction.yaml",
                "admouse_spatial_full_alpha_express_0015_no_lr_prior.yaml",
                "arista_spatial_full.yaml",
                "arista_spatial_full_no_interaction.yaml",
                "arista_spatial_full_no_lr_prior.yaml",
                "mosta_spatial_full_alpha_express_0015.yaml",
                "mosta_spatial_full_alpha_express_0015_no_interaction.yaml",
                "mosta_spatial_full_alpha_express_0015_no_lr_prior.yaml",
                "zebrafish_spatial_full_alpha_express_0015.yaml",
                "zebrafish_spatial_full_alpha_express_0015_no_interaction.yaml",
                "zebrafish_spatial_full_alpha_express_0015_no_lr_prior.yaml",
                "chicken_heart_spatial_full_alpha_express_0015.yaml",
                "weinreb_nonspatial_gnn_full.yaml",
                "weinreb_nonspatial_gnn_no_interaction.yaml",
                "scnt_cortex_nonspatial_gnn_full.yaml",
                "scnt_cortex_nonspatial_gnn_no_interaction.yaml",
            },
        )
        workflow_configs = resources.files("CytoBridge").joinpath("workflow_configs")
        self.assertEqual(
            {
                path.name
                for path in workflow_configs.iterdir()
                if path.suffix == ".json"
            },
            {
                "zebrafish.json",
                "mosta.json",
                "arista.json",
                "admouse.json",
                "chicken_heart.json",
            },
        )

    def test_compact_result_data_is_present(self) -> None:
        result_data = resources.files("CytoBridge").joinpath("results", "data")
        expected = {
            "agist_figures",
            "arista_local_domains",
            "arista_supplementary_figures",
            "classifier_smoothing",
            "full_model_compute_cost",
            "interaction_evidence",
            "loto_benchmark",
            "lr_complex_aggregation",
            "main_figure_2",
            "main_figure_5",
            "training_histories",
            "zebrafish_attention",
        }
        packaged = {path.name for path in result_data.iterdir() if path.is_dir()}
        self.assertTrue(
            expected.issubset(packaged),
            f"missing packaged result directories: {sorted(expected - packaged)}",
        )
        for slug in expected:
            analysis_directory = result_data.joinpath(slug)
            manifest_path = analysis_directory.joinpath("manifest.json")
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            payloads = list(manifest.get("files", {}))
            for section in ("history", "checkpoint_summary"):
                filename = manifest.get(section, {}).get("file")
                if filename:
                    payloads.append(filename)
            self.assertTrue(payloads)
            for relative_path in payloads:
                self.assertTrue(
                    analysis_directory.joinpath(relative_path).is_file(),
                    f"missing packaged result file: {slug}/{relative_path}",
                )

    def test_mosta_reader_module_is_present(self) -> None:
        results_package = resources.files("CytoBridge").joinpath("results")
        self.assertTrue(results_package.joinpath("mosta_figures.py").is_file())

    def test_arista_compact_reader_does_not_package_the_formal_release(self) -> None:
        package_root = resources.files("CytoBridge")
        results_package = package_root.joinpath("results")
        self.assertTrue(
            results_package.joinpath("arista_supplementary_figures.py").is_file()
        )
        self.assertTrue(
            results_package.joinpath(
                "data", "arista_supplementary_figures", "manifest.json"
            ).is_file()
        )
        self.assertFalse(package_root.joinpath("release_artifacts").is_dir())

    def test_installed_workflow_dry_run_uses_packaged_resources(self) -> None:
        executable = Path(sys.executable).with_name("cytobridge")
        with tempfile.TemporaryDirectory() as temporary_directory:
            plans = {}
            for name in (
                "zebrafish",
                "mosta",
                "arista",
                "admouse",
                "chicken_heart",
            ):
                completed = subprocess.run(
                    [
                        str(executable),
                        "workflow",
                        "--config",
                        name,
                        "--dry-run",
                        "--json",
                    ],
                    cwd=temporary_directory,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                plans[name] = json.loads(completed.stdout)
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

        for name, plan in plans.items():
            self.assertEqual(plan["dataset"]["name"], name)
            self.assertEqual(plan["scientific"]["alpha_express"], 0.015)
            self.assertEqual(plan["scientific"]["seed"], 42)
            expected_k = 1 if name in {"admouse", "chicken_heart"} else 10
            self.assertEqual(plan["scientific"]["classifier_k"], expected_k)
            train = next(step for step in plan["steps"] if step["name"] == "train")
            self.assertEqual(train["status"], "skipped; add --train to run")

    def test_installed_nonspatial_plan_needs_no_scientific_dependencies(self) -> None:
        executable = Path(sys.executable).with_name("cytobridge")
        with tempfile.TemporaryDirectory() as temporary_directory:
            plans = {}
            for name in ("weinreb", "scnt_cortex"):
                completed = subprocess.run(
                    [
                        str(executable),
                        "nonspatial",
                        "plan",
                        "--dataset",
                        name,
                        "--json",
                    ],
                    cwd=temporary_directory,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                plans[name] = json.loads(completed.stdout)
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

        self.assertEqual(plans["weinreb"]["preset"]["expected_cells"], 49_302)
        self.assertEqual(plans["scnt_cortex"]["preset"]["expected_cells"], 20_547)
        for plan in plans.values():
            self.assertEqual(plan["preset"]["expected_latent_dim"], 50)
            self.assertEqual(plan["steps"][3], "evaluate weighted W1/W2/TMV from t=0")


if __name__ == "__main__":
    unittest.main()
