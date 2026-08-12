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
        self.assertEqual(distribution.metadata["Requires-Python"], ">=3.10")
        self.assertEqual(distribution.version, CytoBridge.__version__)
        self.assertNotIn(PROJECT_ROOT, Path(CytoBridge.__file__).resolve().parents)
        self.assertEqual(
            set(distribution.metadata.get_all("Provides-Extra") or []),
            {
                "all",
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
        normalized_requirements = [item.replace(" ", "").lower() for item in requirements]
        self.assertTrue(any(item.startswith("numpy<2,>=1.24") for item in normalized_requirements))
        self.assertTrue(
            any(
                "torch-geometric<3,>=2.4" in item
                and "extra=='graph'" in item
                for item in normalized_requirements
            )
        )
        self.assertFalse(any("torchvision" in item for item in normalized_requirements))
        self.assertFalse(any("torchaudio" in item for item in normalized_requirements))

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
        self.assertEqual(completed.stdout.strip(), f"cytobridge {CytoBridge.__version__}")

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
        self.assertTrue(all(value is False for value in report["dependencies"].values()))
        self.assertEqual(
            set(report["profiles"]),
            {
                "all",
                "core",
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
            all(profile["available"] is False for profile in report["profiles"].values())
        )

    def test_config_package_data_is_present(self) -> None:
        config = resources.files("CytoBridge").joinpath(
            "configs", "simulation_config.yaml"
        )
        self.assertTrue(config.is_file())
        workflow_configs = resources.files("CytoBridge").joinpath("workflow_configs")
        self.assertEqual(
            {path.name for path in workflow_configs.iterdir() if path.suffix == ".json"},
            {"zebrafish.json", "mosta.json", "arista.json", "admouse.json"},
        )

    def test_installed_workflow_dry_run_uses_packaged_resources(self) -> None:
        executable = Path(sys.executable).with_name("cytobridge")
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [
                    str(executable),
                    "workflow",
                    "--config",
                    "zebrafish",
                    "--dry-run",
                    "--json",
                ],
                cwd=temporary_directory,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

        plan = json.loads(completed.stdout)
        self.assertEqual(plan["dataset"]["name"], "zebrafish")
        self.assertEqual(plan["scientific"]["alpha_express"], 0.015)
        self.assertEqual(plan["scientific"]["seed"], 42)
        self.assertEqual(plan["scientific"]["classifier_k"], 10)
        train = next(step for step in plan["steps"] if step["name"] == "train")
        self.assertEqual(train["status"], "skipped; add --train to run")


if __name__ == "__main__":
    unittest.main()
