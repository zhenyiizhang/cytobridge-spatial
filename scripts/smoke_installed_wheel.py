#!/usr/bin/env python3
"""Build, install, and test the CytoBridge wheel in a clean environment.

The smoke test deliberately stays small: it builds the current source tree,
installs the wheel without dependencies in a fresh virtual environment, and
runs the installed-package contract tests. Pip index access is disabled so the
test reports missing local build tools instead of downloading them implicitly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTALLED_TEST = PROJECT_ROOT / "tests" / "test_installed_package_contract.py"


def _venv_python(venv_directory: Path) -> Path:
    if os.name == "nt":  # pragma: no cover - Windows path convention
        return venv_directory / "Scripts" / "python.exe"
    return venv_directory / "bin" / "python"


def _build_command(
    source_directory: Path,
    wheel_directory: Path,
    python: Path,
) -> list[str]:
    return [
        str(python),
        "-m",
        "pip",
        "wheel",
        "--use-pep517",
        "--no-build-isolation",
        "--no-deps",
        "--no-index",
        "--no-cache-dir",
        "--wheel-dir",
        str(wheel_directory),
        str(source_directory),
    ]


def _install_command(python: Path, wheel: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
        "--no-cache-dir",
        str(wheel),
    ]


def _installed_test_command(python: Path, test_file: Path) -> list[str]:
    return [str(python), str(test_file), "-v"]


def _clean_environment() -> dict[str, str]:
    """Return an environment that cannot import CytoBridge from PYTHONPATH."""

    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PIP_NO_INDEX"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        check=True,
    )


def run_smoke(workspace: Path) -> dict[str, str]:
    wheel_directory = workspace / "dist"
    venv_directory = workspace / "venv"
    test_directory = workspace / "test-cwd"
    wheel_directory.mkdir()
    test_directory.mkdir()

    environment = _clean_environment()
    _run(
        _build_command(PROJECT_ROOT, wheel_directory, Path(sys.executable)),
        cwd=workspace,
        environment=environment,
    )

    wheels = sorted(wheel_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one built wheel, found {len(wheels)}")
    wheel = wheels[0]

    _run(
        [sys.executable, "-m", "venv", str(venv_directory)],
        cwd=workspace,
        environment=environment,
    )
    installed_python = _venv_python(venv_directory)
    _run(
        _install_command(installed_python, wheel),
        cwd=test_directory,
        environment=environment,
    )

    test_environment = dict(environment)
    test_environment["CYTOBRIDGE_TEST_INSTALLED"] = "1"
    _run(
        _installed_test_command(installed_python, INSTALLED_TEST),
        cwd=test_directory,
        environment=test_environment,
    )
    return {"status": "passed", "wheel": str(wheel), "workspace": str(workspace)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and test a clean, dependency-free CytoBridge wheel install."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="new directory in which to keep the wheel and virtual environment",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.work_dir is not None:
            workspace = arguments.work_dir.expanduser().resolve()
            if workspace.exists():
                raise RuntimeError(f"--work-dir must be a new path: {workspace}")
            workspace.mkdir(parents=True)
            result = run_smoke(workspace)
            print(json.dumps(result, indent=2))
            return 0

        with tempfile.TemporaryDirectory(
            prefix="cytobridge-wheel-smoke-"
        ) as temporary:
            result = run_smoke(Path(temporary))
            result["workspace"] = "temporary directory removed after success"
            print(json.dumps(result, indent=2))
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"CytoBridge wheel smoke failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
