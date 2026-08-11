"""Wheel-install contract tests.

These tests are skipped in a source-tree suite.  ``smoke_installed_wheel.py``
runs this file directly inside a clean no-dependencies virtual environment with
``CYTOBRIDGE_TEST_INSTALLED=1`` and records its output hashes.
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

    def test_config_package_data_is_present(self) -> None:
        config = resources.files("CytoBridge").joinpath(
            "configs", "simulation_config.yaml"
        )
        self.assertTrue(config.is_file())


if __name__ == "__main__":
    unittest.main()
