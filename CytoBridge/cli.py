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
    "notebook": (
        "geomloss",
        "ipywidgets",
        "jupyterlab",
        "matplotlib",
        "ot",
        "phate",
        "PIL",
        "qnorm",
        "scanpy",
        "seaborn",
        "torch",
        "torch_geometric",
        "torchdiffeq",
    ),
    "docs": (
        "furo",
        "myst_parser",
        "nbsphinx",
        "sphinx",
        "sphinx_copybutton",
        "sphinx_design",
    ),
}
_PROFILE_INCLUDES = {
    "core": ("core",),
    "preprocess": ("core", "preprocess"),
    "train": ("core", "train"),
    "plot": ("core", "plot"),
    "velocity": ("core", "velocity"),
    "graph": ("core", "graph"),
    "notebook": ("core", "notebook"),
    "docs": ("core", "docs"),
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
        description="CytoBridge package diagnostics and scientific workflows.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command")
    doctor = commands.add_parser(
        "doctor",
        help="report package and dependency availability without importing them",
    )
    doctor.add_argument("--json", action="store_true", dest="as_json")
    workflow = commands.add_parser(
        "workflow",
        help="plan or run a package-native dataset workflow",
    )
    workflow.add_argument(
        "--config",
        help="packaged preset (zebrafish, mosta, arista, admouse) or JSON/YAML path",
    )
    workflow.add_argument("--list-configs", action="store_true")
    workflow.add_argument("--dry-run", action="store_true")
    workflow.add_argument("--json", action="store_true", dest="as_json")
    workflow.add_argument(
        "--step",
        action="append",
        choices=("preprocess", "downstream"),
        help="run only this step; repeat to select both",
    )
    workflow.add_argument(
        "--train",
        action="store_true",
        help="explicitly enable model training between preprocessing and downstream",
    )
    workflow.add_argument("--input-h5ad", type=Path)
    workflow.add_argument("--aligned-h5ad", type=Path)
    workflow.add_argument("--model-dir", type=Path)
    workflow.add_argument("--output-dir", type=Path)
    workflow.add_argument("--training-config")
    workflow.add_argument(
        "--interaction-cutoff",
        type=float,
        help="override the packaged formal spatial interaction cutoff",
    )
    workflow.add_argument("--edge-predictor-path", type=Path)
    workflow.add_argument("--edge-predictor-threshold", type=float)
    workflow.add_argument("--edge-predictor-root", type=Path)
    workflow.add_argument("--device", default="cuda")
    workflow.add_argument("--model-format", choices=("current", "legacy"))
    workflow.add_argument(
        "--reference-h5ad",
        type=Path,
        help=(
            "reference expression/PCA AnnData for optional gene and ligand-receptor "
            "analyses; defaults to --aligned-h5ad"
        ),
    )
    workflow.add_argument(
        "--gene-dynamics",
        action="store_true",
        help="reconstruct and plot temporal gene programs using exact retained PCA metadata",
    )
    workflow.add_argument(
        "--lr-database",
        type=Path,
        help="run strict ligand-receptor projection with this database",
    )
    workflow.add_argument(
        "--lr-complex-mode",
        choices=("min", "geometric_mean"),
        default="min",
        help="multi-subunit LR aggregation rule (default: strict minimum)",
    )
    workflow.add_argument(
        "--preferred-species-tag",
        help="optional exact species tag used when simplifying reference gene names",
    )
    workflow.add_argument(
        "--reconstruction-diagnostic",
        action="store_true",
        help=(
            "report fitted-model future-slice W2 reconstruction diagnostics; "
            "this is explicitly not a training holdout benchmark"
        ),
    )
    return parser


def _run_workflow_command(args: argparse.Namespace) -> int:
    from .workflow import (
        WorkflowOptions,
        available_workflow_configs,
        build_workflow_plan,
        load_workflow_config,
        plan_missing_inputs,
        render_workflow_plan,
        run_workflow,
    )

    if args.list_configs:
        print("\n".join(available_workflow_configs()))
        return 0
    if not args.config:
        print(
            "cytobridge workflow requires --config (or use --list-configs)",
            file=sys.stderr,
        )
        return 2

    options = WorkflowOptions(
        input_h5ad=args.input_h5ad,
        aligned_h5ad=args.aligned_h5ad,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        training_config=args.training_config,
        interaction_cutoff=args.interaction_cutoff,
        edge_predictor_path=args.edge_predictor_path,
        edge_predictor_threshold=args.edge_predictor_threshold,
        edge_predictor_root=args.edge_predictor_root,
        device=args.device,
        model_format=args.model_format,
        reference_h5ad=args.reference_h5ad,
        gene_dynamics=bool(args.gene_dynamics),
        lr_database=args.lr_database,
        lr_complex_mode=args.lr_complex_mode,
        preferred_species_tag=args.preferred_species_tag,
        reconstruction_diagnostic=bool(args.reconstruction_diagnostic),
        steps=tuple(args.step or ()),
        train=bool(args.train),
    )
    try:
        config, source = load_workflow_config(args.config)
        plan = build_workflow_plan(config, source=source, options=options)
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as error:
        print(f"CytoBridge workflow config error: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        if args.as_json:
            print(json.dumps(plan, indent=2))
        else:
            print(render_workflow_plan(plan))
            print("dry-run: no work executed")
        return 0

    missing = plan_missing_inputs(plan)
    if missing:
        print(render_workflow_plan(plan), file=sys.stderr)
        print("Cannot run until these inputs are supplied:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 2

    result = run_workflow(config, options=options)
    if args.as_json:
        print(json.dumps(result, indent=2))
    else:
        completed = ", ".join(result["completed"]) or "none"
        print(f"CytoBridge workflow completed: {completed}")
        for name, value in result["outputs"].items():
            if isinstance(value, str):
                print(f"  {name}: {value}")
            elif isinstance(value, dict) and value.get("summary_file"):
                print(f"  {name} summary: {value['summary_file']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CytoBridge command-line interface."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "workflow":
        return _run_workflow_command(args)
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
