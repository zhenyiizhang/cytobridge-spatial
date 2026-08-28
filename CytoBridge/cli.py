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
_FIGURE_COMMAND_HELP = {
    "agist": "redraw Supplementary Figures S2-S3",
    "nonspatial": "redraw Supplementary Figures S4-S5",
    "classifier-smoothing": "redraw Supplementary Figure S6 summaries",
    "arista-lr": "recalculate and redraw corrected Figures S23-S24",
    "lr-complex": "redraw Supplementary Figure S25",
    "zebrafish-si": "redraw Supplementary Figures S31-S38",
    "interaction-evidence": "redraw Supplementary Figure S39 summaries",
    "loto-benchmark": "redraw Supplementary Figure S40 summaries",
    "training-histories": "redraw Supplementary Figure S41",
    "arista-local-domains": "redraw Supplementary Figure S42 summaries",
    "zebrafish-attention": "redraw Supplementary Figure S43 summaries",
    "compute-cost": "write Supplementary Table 2",
    "main-figure-2": "assemble Main Figure 2 from packaged inputs",
    "main-figure-5-reference": "export the accepted Main Figure 5 reference page",
    "main-figure-4": "assemble Main Figure 4 from an external MOSTA release",
    "mosta-reference-pages": "export S11-S18 from an external MOSTA release",
}
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
        "fitz",
        "joblib",
        "kaleido",
        "matplotlib",
        "PIL",
        "plotly",
        "pypdf",
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
        "fitz",
        "geomloss",
        "ipywidgets",
        "jupyterlab",
        "matplotlib",
        "ot",
        "phate",
        "PIL",
        "pypdf",
        "qnorm",
        "scanpy",
        "seaborn",
        "torch",
        "torch_geometric",
        "torchdiffeq",
    ),
    "docs": (
        "furo",
        "IPython",
        "myst_parser",
        "nbsphinx",
        "sphinx",
        "sphinx_copybutton",
        "sphinx_design",
        "statsmodels",
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
        "optional libraries (modules are not imported):",
    ]
    lines.extend(
        f"  {name}: {'available' if available else 'missing'}"
        for name, available in dependencies.items()
    )
    lines.append("installation options:")
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
        help="preprocess, train, and analyze a dataset",
    )
    workflow.add_argument(
        "--config",
        help=(
            "example dataset name (zebrafish, mosta, arista, admouse, chicken_heart) "
            "or JSON/YAML path"
        ),
    )
    figure = commands.add_parser(
        "figure",
        help="run a paper-figure redraw, table, export, or assembly workflow",
    )
    figure_commands = figure.add_subparsers(dest="figure_command")
    figure_commands.add_parser("list", help="list figure workflows")
    figure_explain = figure_commands.add_parser(
        "explain",
        help="show how the inputs, calculations, and figure command connect",
    )
    figure_explain.add_argument("name", choices=tuple(_FIGURE_COMMAND_HELP))
    figure_explain.add_argument("--json", action="store_true", dest="as_json")
    for name, help_text in _FIGURE_COMMAND_HELP.items():
        figure_parser = figure_commands.add_parser(name, help=help_text)
        figure_parser.add_argument(
            "--results-dir",
            type=Path,
            default=None,
            help=(
                "compact result directory; defaults to package data. For MOSTA "
                "commands, supply the downloaded release directory"
            ),
        )
        figure_parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="new or empty output directory",
        )
    nonspatial = commands.add_parser(
        "nonspatial",
        help="run the Weinreb or scNT non-spatial analysis",
    )
    nonspatial_commands = nonspatial.add_subparsers(dest="nonspatial_command")
    nonspatial_commands.add_parser("list-datasets", help="list supported datasets")
    nonspatial_plan = nonspatial_commands.add_parser(
        "plan", help="show the steps for one dataset"
    )
    nonspatial_plan.add_argument("--dataset", required=True)
    nonspatial_plan.add_argument("--json", action="store_true", dest="as_json")

    nonspatial_prepare = nonspatial_commands.add_parser(
        "prepare", help="prepare expression and a 50-PC non-spatial model input"
    )
    nonspatial_prepare.add_argument("--dataset", required=True)
    nonspatial_prepare.add_argument("--input-h5ad", required=True, type=Path)
    nonspatial_prepare.add_argument("--output-dir", required=True, type=Path)
    nonspatial_prepare.add_argument("--overwrite", action="store_true")

    nonspatial_prior = nonspatial_commands.add_parser(
        "build-prior", help="train a directed ligand-receptor edge prior"
    )
    nonspatial_prior.add_argument("--dataset", required=True)
    nonspatial_prior.add_argument("--preprocess-manifest", required=True, type=Path)
    nonspatial_prior.add_argument("--output-dir", required=True, type=Path)
    nonspatial_prior.add_argument("--lr-database", type=Path)
    nonspatial_prior.add_argument("--device", default="auto")
    nonspatial_prior.add_argument("--epochs", type=int, default=50)
    nonspatial_prior.add_argument("--overwrite", action="store_true")

    nonspatial_train = nonspatial_commands.add_parser(
        "train", help="train either the Full or No-interaction model"
    )
    nonspatial_train.add_argument("--dataset", required=True)
    nonspatial_train.add_argument(
        "--arm", required=True, choices=("full", "no_interaction")
    )
    nonspatial_train.add_argument("--preprocess-manifest", required=True, type=Path)
    nonspatial_train.add_argument("--edge-prior-manifest", type=Path)
    nonspatial_train.add_argument("--output-dir", required=True, type=Path)
    nonspatial_train.add_argument("--device", default="cuda")

    nonspatial_evaluate = nonspatial_commands.add_parser(
        "evaluate", help="compare matched arms with weighted W1/W2/TMV"
    )
    nonspatial_evaluate.add_argument("--dataset", required=True)
    nonspatial_evaluate.add_argument("--prepared-h5ad", required=True, type=Path)
    nonspatial_evaluate.add_argument("--full-run-dir", required=True, type=Path)
    nonspatial_evaluate.add_argument(
        "--no-interaction-run-dir", required=True, type=Path
    )
    nonspatial_evaluate.add_argument("--output-dir", required=True, type=Path)
    nonspatial_evaluate.add_argument("--device", default="cuda")
    nonspatial_evaluate.add_argument(
        "--inference-seed", action="append", type=int, dest="inference_seeds"
    )
    nonspatial_evaluate.add_argument("--n-samples", type=int, default=2048)
    nonspatial_evaluate.add_argument("--sigma", type=float, default=0.1)
    nonspatial_evaluate.add_argument("--max-ot-points", type=int, default=1024)
    nonspatial_direction = nonspatial_commands.add_parser(
        "scnt-direction",
        help="compare scNT new-RNA direction after both models finish",
    )
    nonspatial_direction.add_argument("--source-h5ad", required=True, type=Path)
    nonspatial_direction.add_argument("--prepared-h5ad", required=True, type=Path)
    nonspatial_direction.add_argument("--pca-artifacts-npz", required=True, type=Path)
    nonspatial_direction.add_argument("--full-run-dir", required=True, type=Path)
    nonspatial_direction.add_argument(
        "--no-interaction-run-dir", required=True, type=Path
    )
    nonspatial_direction.add_argument("--output-dir", required=True, type=Path)
    nonspatial_direction.add_argument("--device", default="cuda")
    nonspatial_direction.add_argument("--overwrite", action="store_true")

    nonspatial_attribution = nonspatial_commands.add_parser(
        "attribution",
        help="compute exact GNN messages and LR-supported cell-type pathways",
    )
    nonspatial_attribution.add_argument("--dataset", required=True)
    nonspatial_attribution.add_argument("--expression-h5ad", required=True, type=Path)
    nonspatial_attribution.add_argument("--latent-h5ad", required=True, type=Path)
    nonspatial_attribution.add_argument(
        "--edge-prior-manifest", required=True, type=Path
    )
    nonspatial_attribution.add_argument("--training-run-dir", required=True, type=Path)
    nonspatial_attribution.add_argument("--output-dir", required=True, type=Path)
    nonspatial_attribution.add_argument("--device", default="cuda")
    nonspatial_attribution.add_argument("--overwrite", action="store_true")
    nonspatial_figure = nonspatial_commands.add_parser(
        "figure",
        help="draw the Weinreb or scNT figure from calculated panel data",
    )
    nonspatial_figure.add_argument("--dataset", required=True)
    nonspatial_figure.add_argument("--bundle-dir", required=True, type=Path)
    nonspatial_figure.add_argument("--output-dir", required=True, type=Path)
    nonspatial_figure.add_argument("--dpi", type=int, default=320)
    nonspatial_fate = nonspatial_commands.add_parser(
        "weinreb-clone-fate",
        help="evaluate lineage-level fate agreement from t=0 to Day 6",
    )
    nonspatial_fate.add_argument("--prepared-h5ad", required=True, type=Path)
    nonspatial_fate.add_argument("--full-run-dir", required=True, type=Path)
    nonspatial_fate.add_argument("--no-interaction-run-dir", required=True, type=Path)
    nonspatial_fate.add_argument("--output-dir", required=True, type=Path)
    nonspatial_fate.add_argument("--device", default="cuda")
    nonspatial_fate.add_argument(
        "--simulation-seed", action="append", type=int, dest="simulation_seeds"
    )
    nonspatial_fate.add_argument("--bootstrap", type=int, default=5000)
    workflow.add_argument("--list-configs", action="store_true")
    workflow.add_argument(
        "--export-config",
        type=Path,
        help="copy the selected example configuration to a new JSON file",
    )
    workflow.add_argument(
        "--check",
        action="store_true",
        dest="dry_run",
        help="show the inputs and output paths without starting the calculation",
    )
    workflow.add_argument(
        "--dry-run", action="store_true", dest="dry_run", help=argparse.SUPPRESS
    )
    workflow.add_argument("--json", action="store_true", dest="as_json")
    workflow.add_argument(
        "--step",
        action="append",
        choices=("preprocess", "train", "downstream"),
        help="run only this step; repeat to select multiple steps",
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
        help="override the spatial interaction cutoff in the configuration",
    )
    workflow.add_argument(
        "--graph-database",
        type=Path,
        help=(
            "ligand-receptor database used to build interaction graphs when "
            "preprocessing automatically trains an edge predictor; overrides "
            "the species-matched database included with each example dataset"
        ),
    )
    workflow.add_argument("--edge-predictor-path", type=Path)
    workflow.add_argument("--edge-predictor-threshold", type=float)
    workflow.add_argument("--edge-predictor-root", type=Path)
    workflow.add_argument("--device", default="cuda")
    workflow.add_argument(
        "--model-format",
        choices=("current",),
        help=(
            "the current workflows use the six-stage checkpoint format; "
            "use the dedicated legacy loader API for historical models"
        ),
    )
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
        help=(
            "enable temporal gene reconstruction for a custom config; the "
            "example dataset configurations already enable it"
        ),
    )
    workflow.add_argument(
        "--skip-gene-dynamics",
        action="store_true",
        help="skip gene dynamics, for example when an older model has no PCA reference",
    )
    workflow.add_argument(
        "--lr-database",
        type=Path,
        help=(
            "override the species-matched database used by example datasets for "
            "strict ligand-receptor projection"
        ),
    )
    workflow.add_argument(
        "--skip-lr",
        action="store_true",
        help="skip ligand-receptor projection while retaining graph attention outputs",
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
        "--allow-complete-reference-pca-center-fallback",
        action="store_true",
        help=(
            "explicitly declare that a historical reference without pca_center "
            "is the complete original PCA-fit population; its inferred mean must "
            "still reproduce saved PCA coordinates"
        ),
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
        graph_database=args.graph_database,
        edge_predictor_path=args.edge_predictor_path,
        edge_predictor_threshold=args.edge_predictor_threshold,
        edge_predictor_root=args.edge_predictor_root,
        device=args.device,
        model_format=args.model_format,
        reference_h5ad=args.reference_h5ad,
        gene_dynamics=bool(args.gene_dynamics),
        skip_gene_dynamics=bool(args.skip_gene_dynamics),
        lr_database=args.lr_database,
        skip_lr=bool(args.skip_lr),
        lr_complex_mode=args.lr_complex_mode,
        preferred_species_tag=args.preferred_species_tag,
        reconstruction_diagnostic=bool(args.reconstruction_diagnostic),
        allow_complete_reference_pca_center_fallback=bool(
            args.allow_complete_reference_pca_center_fallback
        ),
        steps=tuple(args.step or ()),
        train=bool(args.train),
    )
    try:
        config, source = load_workflow_config(args.config)
        if args.export_config is not None:
            destination = args.export_config.expanduser().resolve()
            if destination.exists():
                raise FileExistsError(
                    f"Refusing to replace an existing config: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote editable workflow config: {destination}")
            print(f"Starting configuration: {source}")
            return 0
        plan = build_workflow_plan(config, source=source, options=options)
    except (
        FileExistsError,
        FileNotFoundError,
        ModuleNotFoundError,
        ValueError,
    ) as error:
        print(f"CytoBridge workflow config error: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        if args.as_json:
            print(json.dumps(plan, indent=2))
        else:
            print(render_workflow_plan(plan))
            print("Check complete: no calculation was started.")
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


def _run_figure_command(args: argparse.Namespace) -> int:
    """Run a public paper-figure workflow."""

    if args.figure_command == "list":
        from .results.figure_workflows import list_figure_workflows

        print("command\tpaper location\tinput")
        for workflow in list_figure_workflows():
            source = "included example data" if workflow["wheel_runnable"] else "separate result directory"
            print(
                f"{workflow['name']}\t{workflow['paper_location']}\t{source}"
            )
        return 0
    if args.figure_command == "explain":
        from .results.figure_workflows import describe_figure_workflow
        from .results.reproduction_chains import describe_figure_steps

        route = describe_figure_workflow(args.name)
        chain = describe_figure_steps(args.name)
        if args.as_json:
            print(json.dumps({**route, "steps": chain}, indent=2, sort_keys=True))
        else:
            labels = (
                ("Paper location", "paper_location"),
                ("Input", "starts_from"),
                ("Figure command", "figure_command"),
                ("Scope", "scope"),
            )
            for label, key in labels:
                value = route[key]
                if value:
                    print(f"{label}: {value}")
            print("How the files connect:")
            for index, row in enumerate(chain, start=1):
                print(f"  {index}. {row['paper_part']} — {row['step']}")
                print(f"     command: {row['code_or_command']}")
                print(f"     input: {row['reads']}")
                print(f"     creates: {row['writes']}")
                print(f"     continue with: {row['next_step']}")
                if row.get("note"):
                    print(f"     note: {row['note']}")
        return 0
    if args.figure_command not in _FIGURE_COMMAND_HELP:
        print("cytobridge figure requires a subcommand; use --help", file=sys.stderr)
        return 2

    try:
        from .results.figure_workflows import run_figure_workflow

        summary = run_figure_workflow(
            args.figure_command,
            args.output_dir,
            results_dir=args.results_dir,
        )
    except ModuleNotFoundError as error:
        print(
            f"Figure dependencies are missing ({error.name}). "
            "Install them with: pip install 'CytoBridge[plot]'",
            file=sys.stderr,
        )
        return 2
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"CytoBridge figure input error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_nonspatial_command(args: argparse.Namespace) -> int:
    """Dispatch the package-owned non-spatial workflow lazily."""

    from .nonspatial import (
        available_nonspatial_presets,
        build_nonspatial_lr_prior,
        evaluate_nonspatial_pair,
        nonspatial_plan,
        prepare_nonspatial_dataset,
        train_nonspatial_condition,
        replay_nonspatial_figure,
    )

    command = args.nonspatial_command
    if command == "list-datasets":
        print("\n".join(available_nonspatial_presets()))
        return 0
    if command == "plan":
        plan = nonspatial_plan(args.dataset)
        if args.as_json:
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            print(f"CytoBridge non-spatial dataset: {plan['preset']['display_name']}")
            for index, step in enumerate(plan["steps"], start=1):
                print(f"  {index}. {step}")
            print(plan["historical_replay_note"])
        return 0
    try:
        if command == "prepare":
            result = prepare_nonspatial_dataset(
                args.dataset,
                args.input_h5ad,
                args.output_dir,
                overwrite=bool(args.overwrite),
            )
            payload = {
                "model_h5ad": str(result.model_h5ad),
                "expression_h5ad": str(result.expression_h5ad),
                "pca_artifacts": str(result.pca_artifacts),
                "manifest": str(result.manifest),
            }
        elif command == "build-prior":
            payload = build_nonspatial_lr_prior(
                args.dataset,
                args.preprocess_manifest,
                args.output_dir,
                lr_database=args.lr_database,
                device=args.device,
                overwrite=bool(args.overwrite),
                epochs=int(args.epochs),
            )
        elif command == "train":
            payload = train_nonspatial_condition(
                args.dataset,
                args.arm,
                args.preprocess_manifest,
                args.output_dir,
                edge_prior_manifest=args.edge_prior_manifest,
                device=args.device,
            )
        elif command == "evaluate":
            payload = evaluate_nonspatial_pair(
                args.dataset,
                args.prepared_h5ad,
                args.full_run_dir,
                args.no_interaction_run_dir,
                args.output_dir,
                device=args.device,
                inference_seeds=tuple(args.inference_seeds or (10_000,)),
                n_samples=int(args.n_samples),
                sigma=float(args.sigma),
                max_ot_points=int(args.max_ot_points),
            )
        elif command == "scnt-direction":
            from .nonspatial.scnt_direction import main as run_scnt_direction

            direction_argv = [
                "--source-h5ad",
                str(args.source_h5ad),
                "--prepared-h5ad",
                str(args.prepared_h5ad),
                "--pca-artifacts-npz",
                str(args.pca_artifacts_npz),
                "--full-run-dir",
                str(args.full_run_dir),
                "--no-interaction-run-dir",
                str(args.no_interaction_run_dir),
                "--output-dir",
                str(args.output_dir),
                "--device",
                str(args.device),
            ]
            if args.overwrite:
                direction_argv.append("--overwrite")
            run_scnt_direction(direction_argv)
            return 0
        elif command == "attribution":
            from .nonspatial.interaction_attribution import (
                main as run_interaction_attribution,
            )

            from .nonspatial.workflow import nonspatial_preset

            preset = nonspatial_preset(args.dataset)
            attribution_argv = [
                "--expression-h5ad",
                str(args.expression_h5ad),
                "--latent-h5ad",
                str(args.latent_h5ad),
                "--edge-prior-manifest",
                str(args.edge_prior_manifest),
                "--training-run-dir",
                str(args.training_run_dir),
                "--output-dir",
                str(args.output_dir),
                "--cell-type-key",
                preset.cell_type_key,
                "--time-key",
                "time_point_processed",
                "--device",
                str(args.device),
            ]
            if args.overwrite:
                attribution_argv.append("--overwrite")
            run_interaction_attribution(attribution_argv)
            return 0
        elif command == "figure":
            payload = replay_nonspatial_figure(
                args.dataset,
                args.bundle_dir,
                args.output_dir,
                dpi=int(args.dpi),
            )
        elif command == "weinreb-clone-fate":
            from .nonspatial.weinreb_fate import evaluate_weinreb_clone_fate

            payload = evaluate_weinreb_clone_fate(
                args.prepared_h5ad,
                args.full_run_dir,
                args.no_interaction_run_dir,
                args.output_dir,
                device=args.device,
                simulation_seeds=tuple(args.simulation_seeds or range(10)),
                n_bootstrap=int(args.bootstrap),
            )
        else:
            print(
                "cytobridge nonspatial requires a subcommand; use --help",
                file=sys.stderr,
            )
            return 2
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as error:
        print(f"CytoBridge non-spatial contract error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CytoBridge command-line interface."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "workflow":
        return _run_workflow_command(args)
    if args.command == "figure":
        return _run_figure_command(args)
    if args.command == "nonspatial":
        return _run_nonspatial_command(args)
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
