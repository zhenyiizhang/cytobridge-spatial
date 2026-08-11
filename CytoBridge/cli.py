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
_DEPENDENCY_PROFILES = {
    "core": ("anndata", "numpy", "pandas", "scipy", "sklearn", "tqdm", "yaml"),
    "preprocess": ("ot", "qnorm", "scanpy", "torch"),
    "train": (
        "geomloss",
        "matplotlib",
        "ot",
        "PIL",
        "scanpy",
        "seaborn",
        "torch",
        "torchdiffeq",
    ),
    "plot": (
        "joblib",
        "kaleido",
        "matplotlib",
        "PIL",
        "plotly",
        "seaborn",
        "umap",
    ),
    "velocity": (
        "cellrank",
        "geomloss",
        "joblib",
        "kaleido",
        "matplotlib",
        "ot",
        "PIL",
        "plotly",
        "scanpy",
        "scvelo",
        "seaborn",
        "torch",
        "torchdiffeq",
        "umap",
    ),
    "graph": (
        "geomloss",
        "matplotlib",
        "ot",
        "PIL",
        "scanpy",
        "seaborn",
        "torch",
        "torch_geometric",
        "torchdiffeq",
    ),
    "notebook": ("ipywidgets", "phate"),
}
_PROFILE_INCLUDES = {
    "core": ("core",),
    "preprocess": ("core", "preprocess"),
    "train": ("core", "train"),
    "plot": ("core", "plot"),
    "velocity": ("core", "velocity"),
    "graph": ("core", "graph"),
    "notebook": ("core", "notebook"),
    "spatial": ("core", "preprocess", "train", "graph"),
    "all": tuple(_DEPENDENCY_PROFILES),
}
_PROBED_MODULES = tuple(
    sorted({module for modules in _DEPENDENCY_PROFILES.values() for module in modules})
)


def _installed_version() -> str | None:
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return None


def build_doctor_report() -> dict[str, object]:
    """Return diagnostics without importing dependencies or changing state."""

    installed_version = _installed_version()
    dependencies = {
        module_name: util.find_spec(module_name) is not None
        for module_name in _PROBED_MODULES
    }
    profiles: dict[str, dict[str, object]] = {}
    for profile_name, included_profiles in _PROFILE_INCLUDES.items():
        required_modules = sorted(
            {
                module
                for included_profile in included_profiles
                for module in _DEPENDENCY_PROFILES[included_profile]
            }
        )
        missing_modules = [
            module for module in required_modules if not dependencies[module]
        ]
        profiles[profile_name] = {
            "available": not missing_modules,
            "install": (
                "pip install CytoBridge"
                if profile_name == "core"
                else f"pip install 'CytoBridge[{profile_name}]'"
            ),
            "missing_modules": missing_modules,
        }
    return {
        "dependencies": dependencies,
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
        "profiles": profiles,
    }


def _render_doctor_text(report: dict[str, object]) -> str:
    package = report["package"]
    python = report["python"]
    dependencies = report["dependencies"]
    profiles = report["profiles"]
    assert isinstance(package, dict)
    assert isinstance(python, dict)
    assert isinstance(dependencies, dict)
    assert isinstance(profiles, dict)

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
    lines.append("dependency profiles:")
    for name, profile in profiles.items():
        assert isinstance(profile, dict)
        if profile["available"]:
            lines.append(f"  {name}: available")
        else:
            missing = ", ".join(profile["missing_modules"])
            lines.append(f"  {name}: missing {missing}; {profile['install']}")
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
