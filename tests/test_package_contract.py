from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MODULES = ("pl", "pp", "tl", "utils")
HEAVY_MODULES = ("matplotlib", "scanpy")
INSTALLED_SMOKE_RUNNER = PROJECT_ROOT / "scripts" / "smoke_installed_wheel.py"
EXTRA_NAMES = {
    "all",
    "docs",
    "graph",
    "notebook",
    "plot",
    "preprocess",
    "spatial",
    "train",
    "velocity",
}


def _source_environment() -> dict[str, str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    source_path = str(PROJECT_ROOT)
    env["PYTHONPATH"] = (
        source_path
        if not current_pythonpath
        else source_path + os.pathsep + current_pythonpath
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_source_python(
    source: str,
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=cwd or PROJECT_ROOT,
        env=_source_environment(),
        check=True,
        capture_output=True,
        text=True,
    )


def _authoritative_version() -> str:
    namespace: dict[str, object] = {}
    version_file = PROJECT_ROOT / "CytoBridge" / "_version.py"
    exec(
        compile(version_file.read_text(encoding="utf-8"), version_file, "exec"),
        namespace,
    )
    version = namespace["__version__"]
    assert isinstance(version, str)
    return version


def _installed_smoke_namespace() -> dict[str, object]:
    return runpy.run_path(str(INSTALLED_SMOKE_RUNNER), run_name="cytobridge_smoke_test")


def test_version_has_one_authoritative_source() -> None:
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = pyproject["project"]
    dynamic = pyproject["tool"]["setuptools"]["dynamic"]

    assert "version" not in project
    assert "version" in project["dynamic"]
    assert dynamic["version"] == {"attr": "CytoBridge._version.__version__"}

    setup_source = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")
    setup_tree = ast.parse(setup_source)
    setup_calls = [
        node
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]
    assert len(setup_calls) == 1
    assert setup_calls[0].args == []
    assert setup_calls[0].keywords == []

    expected = _authoritative_version()
    imported = _run_source_python("import CytoBridge; print(CytoBridge.__version__)")
    assert imported.stdout.strip() == expected

    legacy = subprocess.run(
        [sys.executable, "setup.py", "--version"],
        cwd=PROJECT_ROOT,
        env=_source_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert legacy.stdout.strip().splitlines()[-1] == expected


def test_top_level_import_is_lightweight() -> None:
    result = _run_source_python(
        """
import json
import sys
import CytoBridge

print(json.dumps({
    "heavy": sorted(
        name for name in sys.modules
        if name.split(".", 1)[0] in {"matplotlib", "scanpy"}
    ),
    "public_loaded": sorted(
        name for name in sys.modules
        if name in {"CytoBridge.pl", "CytoBridge.pp", "CytoBridge.tl", "CytoBridge.utils"}
    ),
    "public_in_dir": all(name in dir(CytoBridge) for name in ("pl", "pp", "tl", "utils")),
    "version": CytoBridge.__version__,
}))
"""
    )
    payload = json.loads(result.stdout)
    assert payload == {
        "heavy": [],
        "public_loaded": [],
        "public_in_dir": True,
        "version": _authoritative_version(),
    }


def test_public_namespaces_remain_from_import_compatible() -> None:
    result = _run_source_python(
        """
import json
import sys
from types import ModuleType
import CytoBridge

for short_name in ("pl", "pp", "tl", "utils"):
    sys.modules[f"CytoBridge.{short_name}"] = ModuleType(f"CytoBridge.{short_name}")

from CytoBridge import pl, pp, tl, utils
print(json.dumps([module.__name__ for module in (pl, pp, tl, utils)]))
"""
    )
    assert json.loads(result.stdout) == [
        "CytoBridge.pl",
        "CytoBridge.pp",
        "CytoBridge.tl",
        "CytoBridge.utils",
    ]


def test_config_submodule_import_is_independent_of_training_import_order() -> None:
    result = _run_source_python(
        "from CytoBridge.utils.config import load_config; print(load_config.__name__)"
    )
    assert result.stdout.strip() == "load_config"


def test_plot_namespace_is_monotonic_and_legacy_exports_are_lazy() -> None:
    result = _run_source_python(
        """
import json
import sys
import CytoBridge.pl as pl

print(json.dumps({
    "legacy_in_dir": "plot_growth" in dir(pl),
    "legacy_module_loaded": "CytoBridge.pl.plot" in sys.modules,
    "training_namespace_loaded": "CytoBridge.tl" in sys.modules,
}))
"""
    )
    assert json.loads(result.stdout) == {
        "legacy_in_dir": True,
        "legacy_module_loaded": False,
        "training_namespace_loaded": False,
    }

    actionable = _run_source_python(
        """
import CytoBridge.pl as pl

def missing_legacy_stack(_name):
    raise ModuleNotFoundError("blocked optional dependency", name="torchdiffeq")

pl._import_module = missing_legacy_stack
try:
    pl.plot_growth
except ModuleNotFoundError as exc:
    print(str(exc))
else:
    raise AssertionError("legacy plotting access unexpectedly succeeded")
"""
    )
    assert "pip install 'CytoBridge[velocity]'" in actionable.stdout


def test_graph_api_remains_actionable_without_torch_geometric() -> None:
    result = _run_source_python(
        """
import builtins
import os
import tempfile

real_import = builtins.__import__

def block_torch_geometric(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch_geometric" or name.startswith("torch_geometric."):
        raise ModuleNotFoundError(
            "blocked optional dependency", name="torch_geometric"
        )
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = block_torch_geometric
with tempfile.TemporaryDirectory(prefix="cytobridge-numba-") as cache_dir:
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
    from CytoBridge.tl.graph import GNNInteraction

assert callable(GNNInteraction)
try:
    GNNInteraction(
        in_out_dim=4,
        hidden_dim=4,
        num_heads=1,
        num_layers=1,
        edge_prior_mode="all_spatial",
    )
except ImportError as exc:
    print(str(exc))
else:
    raise AssertionError("GNNInteraction unexpectedly loaded without torch_geometric")
"""
    )
    assert "pip install 'CytoBridge[graph]'" in result.stdout


def test_cli_entry_point_version_and_read_only_doctor(tmp_path: Path) -> None:
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["scripts"] == {"cytobridge": "CytoBridge.cli:main"}

    version = subprocess.run(
        [sys.executable, "-m", "CytoBridge.cli", "--version"],
        cwd=tmp_path,
        env=_source_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert version.stdout.strip() == f"cytobridge {_authoritative_version()}"

    doctor = subprocess.run(
        [sys.executable, "-m", "CytoBridge.cli", "doctor", "--json"],
        cwd=tmp_path,
        env=_source_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(doctor.stdout)
    assert report["package"]["version"] == _authoritative_version()
    assert report["package"]["distribution"] == "CytoBridge"
    assert {"anndata", "matplotlib", "numpy", "pandas", "scanpy", "torch"} <= set(
        report["dependencies"]
    )
    assert set(report["profiles"]) == {"core", *EXTRA_NAMES}
    for profile_name, profile in report["profiles"].items():
        assert isinstance(profile["available"], bool)
        assert isinstance(profile["missing_modules"], list)
        expected_install = (
            "pip install CytoBridge"
            if profile_name == "core"
            else f"pip install 'CytoBridge[{profile_name}]'"
        )
        assert profile["install"] == expected_install
    assert list(tmp_path.iterdir()) == []

    import_probe = _run_source_python(
        """
import json
import sys
from CytoBridge.cli import build_doctor_report

build_doctor_report()
print(json.dumps(sorted(
    name for name in sys.modules
    if name.split(".", 1)[0] in {"matplotlib", "scanpy"}
)))
"""
    )
    assert json.loads(import_probe.stdout) == []


def test_dependency_profiles_are_bounded_and_all_is_the_union() -> None:
    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = pyproject["project"]
    dynamic = pyproject["tool"]["setuptools"]["dynamic"]

    assert set(project["dynamic"]) == {
        "dependencies",
        "optional-dependencies",
        "version",
    }
    assert dynamic["dependencies"] == {"file": ["requirements/core.txt"]}
    optional = dynamic["optional-dependencies"]
    assert set(optional) == EXTRA_NAMES
    assert optional == {
        name: {"file": [f"requirements/{name}.txt"]} for name in EXTRA_NAMES
    }

    def load_requirements(name: str) -> dict[str, Requirement]:
        requirements: dict[str, Requirement] = {}
        for raw_line in (
            (PROJECT_ROOT / "requirements" / f"{name}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = Requirement(line)
            assert parsed.name not in requirements
            specifiers = {specifier.operator for specifier in parsed.specifier}
            bounded_range = any(
                operator in specifiers for operator in {">", ">="}
            ) and any(operator in specifiers for operator in {"<", "<="})
            assert bounded_range or specifiers == {"=="}
            requirements[parsed.name] = parsed
        return requirements

    core = load_requirements("core")
    groups = {
        name: load_requirements(name)
        for name in EXTRA_NAMES
        if name not in {"all", "spatial"}
    }
    all_requirements = load_requirements("all")
    expected_all = {
        requirement.name: requirement
        for requirements in groups.values()
        for requirement in requirements.values()
    }
    assert set(all_requirements) == set(expected_all)
    for name, requirement in expected_all.items():
        assert str(all_requirements[name].specifier) == str(requirement.specifier)

    spatial = load_requirements("spatial")
    expected_spatial_names = {
        requirement.name
        for group_name in ("preprocess", "train", "graph")
        for requirement in groups[group_name].values()
    }
    assert set(spatial) == expected_spatial_names

    # The default spatial command continues into velocity and interactive
    # downstream figures. Keep those dependencies in their existing focused
    # extra, while ensuring the documented combination is complete.
    default_spatial_workflow = {**spatial, **groups["velocity"]}
    assert {"scvelo", "plotly"} <= set(default_spatial_workflow)
    assert {"scvelo", "plotly"}.isdisjoint(spatial)

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    installation = (PROJECT_ROOT / "docs" / "installation.md").read_text(
        encoding="utf-8"
    )
    assert ".[spatial,velocity]" in readme
    assert ".[notebook,velocity]" in readme
    assert "[spatial,velocity,notebook]" in installation

    compatibility_lines = {
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert compatibility_lines == {
        "-r requirements/core.txt",
        "-r requirements/all.txt",
    }
    forbidden = {"torchvision", "torchaudio", "torchsde", "imageio"}
    assert forbidden.isdisjoint(core)
    assert forbidden.isdisjoint(all_requirements)

    distribution_to_module = {
        "anndata": "anndata",
        "cellrank": "cellrank",
        "geomloss": "geomloss",
        "imageio-ffmpeg": "imageio_ffmpeg",
        "ipython": "IPython",
        "ipywidgets": "ipywidgets",
        "jupyterlab": "jupyterlab",
        "joblib": "joblib",
        "kaleido": "kaleido",
        "matplotlib": "matplotlib",
        "myst-parser": "myst_parser",
        "nbsphinx": "nbsphinx",
        "numpy": "numpy",
        "pandas": "pandas",
        "phate": "phate",
        "Pillow": "PIL",
        "plotly": "plotly",
        "POT": "ot",
        "PyMuPDF": "fitz",
        "pypdf": "pypdf",
        "PyYAML": "yaml",
        "qnorm": "qnorm",
        "scanpy": "scanpy",
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "scvelo": "scvelo",
        "seaborn": "seaborn",
        "sphinx": "sphinx",
        "sphinx-copybutton": "sphinx_copybutton",
        "sphinx-design": "sphinx_design",
        "statsmodels": "statsmodels",
        "furo": "furo",
        "torch": "torch",
        "torch-geometric": "torch_geometric",
        "torchdiffeq": "torchdiffeq",
        "tqdm": "tqdm",
        "umap-learn": "umap",
    }
    profile_probe = _run_source_python(
        """
import json
from CytoBridge.cli import _DEPENDENCY_PROFILES
print(json.dumps(_DEPENDENCY_PROFILES, sort_keys=True))
"""
    )
    dependency_profiles = json.loads(profile_probe.stdout)
    requirement_groups = {"core": core, **groups}
    for profile_name, modules in dependency_profiles.items():
        expected_modules = {
            distribution_to_module[requirement.name]
            for requirement in requirement_groups[profile_name].values()
        }
        assert set(modules) == expected_modules


def test_installed_wheel_smoke_runner_is_a_simple_install_check(tmp_path) -> None:
    runner = _installed_smoke_namespace()
    source = PROJECT_ROOT
    wheel_directory = PROJECT_ROOT / "dist"
    venv_directory = PROJECT_ROOT / ".venv-smoke"
    wheel = wheel_directory / "cytobridge-test.whl"

    build_command = runner["_build_command"](
        source, wheel_directory, Path(sys.executable)
    )
    assert build_command[:4] == [sys.executable, "-m", "pip", "wheel"]
    assert {
        "--use-pep517",
        "--no-build-isolation",
        "--no-deps",
        "--no-index",
        "--no-cache-dir",
    }.issubset(build_command)
    assert build_command[-1] == str(source)

    stale_build_config = PROJECT_ROOT / "build" / "lib" / "CytoBridge" / "configs"
    staged_source = runner["_stage_clean_source"](tmp_path / "staged-source")
    assert not (staged_source / "build").exists()
    assert not (staged_source / "CytoBridge.egg-info").exists()
    assert not (staged_source / stale_build_config.relative_to(PROJECT_ROOT)).exists()

    installed_python = runner["_venv_python"](venv_directory)
    install_command = runner["_install_command"](installed_python, wheel)
    assert install_command[:4] == [
        str(installed_python),
        "-m",
        "pip",
        "install",
    ]
    assert {"--no-deps", "--no-index", "--no-cache-dir"}.issubset(install_command)
    assert install_command[-1] == str(wheel)

    installed_test = PROJECT_ROOT / "tests" / "test_installed_package_contract.py"
    assert runner["_installed_test_command"](installed_python, installed_test) == [
        str(installed_python),
        str(installed_test),
        "-v",
    ]

    environment = runner["_clean_environment"]()
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_installed_wheel_smoke_rejects_workspace_inside_source_tree(tmp_path) -> None:
    runner = _installed_smoke_namespace()
    with pytest.raises(RuntimeError, match="outside the source tree"):
        runner["run_smoke"](PROJECT_ROOT / "wheel-smoke-inside-source")
