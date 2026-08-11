"""Small, read-only command-line interface for package diagnostics."""

from __future__ import annotations

import argparse
from importlib import metadata, util
import json
import platform
from pathlib import Path
import sys
from typing import Sequence

from ._version import __version__


_DISTRIBUTION_NAME = "CytoBridge"
_PROBED_MODULES = ("anndata", "matplotlib", "numpy", "pandas", "scanpy", "torch")


def _installed_version() -> str | None:
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return None


def build_doctor_report() -> dict[str, object]:
    """Return diagnostics without importing dependencies or changing state."""

    installed_version = _installed_version()
    return {
        "dependencies": {
            module_name: util.find_spec(module_name) is not None
            for module_name in _PROBED_MODULES
        },
        "package": {
            "distribution": _DISTRIBUTION_NAME,
            "installed_version": installed_version,
            "source": str(Path(__file__).resolve().parent),
            "version": __version__,
            "version_match": (
                None if installed_version is None else installed_version == __version__
            ),
        },
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def _render_doctor_text(report: dict[str, object]) -> str:
    package = report["package"]
    python = report["python"]
    dependencies = report["dependencies"]
    assert isinstance(package, dict)
    assert isinstance(python, dict)
    assert isinstance(dependencies, dict)

    installed = package["installed_version"] or "not installed"
    lines = [
        "CytoBridge doctor (read-only)",
        f"package version: {package['version']}",
        f"installed metadata version: {installed}",
        f"package source: {package['source']}",
        f"python: {python['implementation']} {python['version']}",
        f"python executable: {python['executable']}",
        "dependency availability (modules are not imported):",
    ]
    lines.extend(
        f"  {name}: {'available' if available else 'missing'}"
        for name, available in dependencies.items()
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cytobridge",
        description="CytoBridge package information and read-only diagnostics.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser(
        "doctor", help="report package and dependency availability without importing them"
    )
    doctor.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CytoBridge command-line interface."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != "doctor":
        parser.print_help()
        return 0

    report = build_doctor_report()
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_render_doctor_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
