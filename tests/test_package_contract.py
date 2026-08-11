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


def _run_source_python(source: str, *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
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
    exec(compile(version_file.read_text(encoding="utf-8"), version_file, "exec"), namespace)
    version = namespace["__version__"]
    assert isinstance(version, str)
    return version


def _installed_smoke_namespace() -> dict[str, object]:
    return runpy.run_path(str(INSTALLED_SMOKE_RUNNER), run_name="cytobridge_smoke_test")


def test_version_has_one_authoritative_source() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
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
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"] == {
        "cytobridge": "CytoBridge.cli:main"
    }

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
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
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
        name: {"file": [f"requirements/{name}.txt"]}
        for name in EXTRA_NAMES
    }

    def load_requirements(name: str) -> dict[str, Requirement]:
        requirements: dict[str, Requirement] = {}
        for raw_line in (PROJECT_ROOT / "requirements" / f"{name}.txt").read_text(
            encoding="utf-8"
        ).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = Requirement(line)
            assert parsed.name not in requirements
            specifiers = {specifier.operator for specifier in parsed.specifier}
            bounded_range = (
                any(operator in specifiers for operator in {">", ">="})
                and any(operator in specifiers for operator in {"<", "<="})
            )
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

    compatibility_lines = {
        line.strip()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
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
        "ipywidgets": "ipywidgets",
        "joblib": "joblib",
        "kaleido": "kaleido",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "pandas": "pandas",
        "phate": "phate",
        "Pillow": "PIL",
        "plotly": "plotly",
        "POT": "ot",
        "PyYAML": "yaml",
        "qnorm": "qnorm",
        "scanpy": "scanpy",
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "scvelo": "scvelo",
        "seaborn": "seaborn",
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


def test_installed_wheel_smoke_runner_contract(tmp_path: Path) -> None:
    runner = _installed_smoke_namespace()
    staged_source = tmp_path / "source"
    wheel_directory = tmp_path / "dist"
    venv_directory = tmp_path / "venv"
    wheel = wheel_directory / "cytobridge-test.whl"
    test_file = PROJECT_ROOT / "tests" / "test_installed_package_contract.py"

    build_command = runner["_build_command"](
        staged_source, wheel_directory, Path(sys.executable)
    )
    assert build_command[:4] == [sys.executable, "-m", "pip", "wheel"]
    assert {
        "--use-pep517",
        "--no-build-isolation",
        "--no-deps",
        "--no-index",
        "--no-cache-dir",
    }.issubset(build_command)
    assert build_command[-1] == str(staged_source)

    venv_python = runner["_venv_python"](venv_directory)
    install_command = runner["_install_command"](venv_python, wheel)
    assert install_command[:4] == [str(venv_python), "-m", "pip", "install"]
    assert {"--no-deps", "--no-index", "--no-cache-dir"}.issubset(
        install_command
    )
    assert install_command[-1] == str(wheel)
    assert runner["_installed_test_command"](venv_python, test_file) == [
        str(venv_python),
        str(test_file),
        "-v",
    ]

    workspace = tmp_path / "private-workspace"
    workspace.mkdir(mode=0o700)
    environment, containment = runner["_contained_environment"](
        workspace,
        {
            "PATH": os.environ["PATH"],
            "CFLAGS": "-Dinjected",
            "DYLD_INSERT_LIBRARIES": "/invalid/injected.dylib",
            "HTTPS_PROXY": "https://invalid.example:443",
            "LD_PRELOAD": "/invalid/injected.so",
            "PIP_INDEX_URL": "https://invalid.example/simple",
            "PYTHONHOME": "/invalid/python-home",
            "PYTHONPATH": "/invalid/python-path",
            "SETUPTOOLS_SCM_PRETEND_VERSION": "999",
            "VIRTUAL_ENV": "/invalid/venv",
        },
    )
    for removed in (
        "CFLAGS",
        "DYLD_INSERT_LIBRARIES",
        "HTTPS_PROXY",
        "LD_PRELOAD",
        "PIP_INDEX_URL",
        "PYTHONHOME",
        "PYTHONPATH",
        "SETUPTOOLS_SCM_PRETEND_VERSION",
        "VIRTUAL_ENV",
    ):
        assert removed not in environment
    assert environment["PIP_NO_INDEX"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    for name, value in containment.items():
        assert environment[name] == value
        assert Path(value).resolve().is_relative_to(workspace.resolve())
    assert {environment[name] for name in ("TEMP", "TMP", "TMPDIR")} == {
        str(workspace / "runtime" / "tmp")
    }
    assert Path(environment["PIP_CONFIG_FILE"]).is_file()

    venv_directory.mkdir()
    configuration = venv_directory / "pyvenv.cfg"
    configuration.write_text("include-system-site-packages = false\n", encoding="utf-8")
    runner["_assert_non_system_venv"](venv_directory)
    configuration.write_text("include-system-site-packages = true\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not include system site-packages"):
        runner["_assert_non_system_venv"](venv_directory)


def test_installed_wheel_smoke_sha_record_detects_tampering(tmp_path: Path) -> None:
    runner = _installed_smoke_namespace()
    wheel = tmp_path / "cytobridge-test.whl"
    wheel.write_bytes(b"first wheel payload")

    first = runner["_wheel_identity"](wheel)
    assert first["sha256"] == runner["_sha256_bytes"](b"first wheel payload")

    wheel.write_bytes(b"tampered wheel payload")
    second = runner["_wheel_identity"](wheel)
    assert second["sha256"] != first["sha256"]


def test_installed_wheel_smoke_rejects_every_existing_work_leaf(
    tmp_path: Path,
) -> None:
    runner = _installed_smoke_namespace()

    existing_directory = tmp_path / "existing-directory"
    existing_directory.mkdir()
    with pytest.raises(RuntimeError, match="already exists or is a symlink"):
        runner["_create_explicit_workspace"](existing_directory)

    regular_leaf = tmp_path / "regular-leaf"
    regular_leaf.write_text("occupied", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already exists or is a symlink"):
        runner["_create_explicit_workspace"](regular_leaf)

    dangling_leaf = tmp_path / "dangling-leaf"
    dangling_leaf.symlink_to(tmp_path / "missing-target")
    assert not dangling_leaf.exists() and os.path.lexists(dangling_leaf)
    with pytest.raises(RuntimeError, match="already exists or is a symlink"):
        runner["_create_explicit_workspace"](dangling_leaf)

    external_target = tmp_path / "external-target"
    external_target.mkdir()
    external_link = tmp_path / "external-link"
    external_link.symlink_to(external_target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="already exists or is a symlink"):
        runner["_create_explicit_workspace"](external_link)

    created = runner["_create_explicit_workspace"](tmp_path / "new-workspace")
    assert created.parent == tmp_path.resolve()
    assert created.lstat().st_mode & 0o777 == 0o700
    assert not created.is_symlink()


def _write_minimal_package_source(root: Path) -> None:
    for name in ("LICENSE", "README.md", "pyproject.toml", "requirements.txt", "setup.py"):
        (root / name).write_text(f"fixture for {name}\n", encoding="utf-8")
    package = root / "CytoBridge"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    requirements = root / "requirements"
    requirements.mkdir()
    (requirements / "core.txt").write_text("numpy>=1.24,<2\n", encoding="utf-8")


def test_installed_wheel_smoke_rejects_source_symlinks_and_special_files(
    tmp_path: Path,
) -> None:
    runner = _installed_smoke_namespace()
    source = tmp_path / "source"
    source.mkdir()
    _write_minimal_package_source(source)
    external = tmp_path / "external.py"
    external.write_text("external = True\n", encoding="utf-8")

    package_link = source / "CytoBridge" / "linked.py"
    package_link.symlink_to(external)
    with pytest.raises(RuntimeError, match="symbolic link"):
        runner["_stage_source"](source, tmp_path / "stage-package-link")
    package_link.unlink()

    requirements_link = source / "requirements" / "linked.txt"
    requirements_link.symlink_to(external)
    with pytest.raises(RuntimeError, match="symbolic link"):
        runner["_stage_source"](source, tmp_path / "stage-requirements-link")
    requirements_link.unlink()

    requirements_directory = source / "requirements"
    original_requirements = source / "requirements-original"
    external_requirements = tmp_path / "external-requirements"
    external_requirements.mkdir()
    (external_requirements / "core.txt").write_text("external\n", encoding="utf-8")
    requirements_directory.rename(original_requirements)
    requirements_directory.symlink_to(external_requirements, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symbolic link: requirements"):
        runner["_stage_source"](source, tmp_path / "stage-requirements-root-link")
    requirements_directory.unlink()
    original_requirements.rename(requirements_directory)

    special = source / "CytoBridge" / "special.fifo"
    os.mkfifo(special)
    with pytest.raises(RuntimeError, match="special file"):
        runner["_stage_source"](source, tmp_path / "stage-special")
    special.unlink()

    setup_file = source / "setup.py"
    setup_file.unlink()
    setup_file.symlink_to(external)
    with pytest.raises(RuntimeError, match="ordinary file without symlinks"):
        runner["_stage_source"](source, tmp_path / "stage-root-link")


def test_installed_wheel_smoke_rejects_directory_swap_without_external_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _installed_smoke_namespace()
    source = tmp_path / "source"
    source.mkdir()
    _write_minimal_package_source(source)
    source_subdirectory = source / "CytoBridge" / "sub"
    source_subdirectory.mkdir()
    (source_subdirectory / "safe.py").write_text("safe = True\n", encoding="utf-8")

    external = tmp_path / "external"
    external.mkdir()
    injected_bytes = b"INJECTED_EXTERNAL_BYTES\n"
    (external / "injected.py").write_bytes(injected_bytes)
    staged = tmp_path / "staged"

    runner_os = runner["_stage_source"].__globals__["os"]
    original_open = runner_os.open
    swapped = {"done": False}

    def swap_before_child_directory_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if (
            not swapped["done"]
            and path == "sub"
            and dir_fd is not None
            and flags & runner_os.O_DIRECTORY
        ):
            source_subdirectory.rename(source_subdirectory.with_name("sub-original"))
            source_subdirectory.symlink_to(external, target_is_directory=True)
            swapped["done"] = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(runner_os, "open", swap_before_child_directory_open)
    with pytest.raises(RuntimeError, match="changed or became unsafe"):
        runner["_stage_source"](source, staged)
    assert swapped["done"]
    assert not (staged / "CytoBridge" / "sub" / "injected.py").exists()
    if staged.exists():
        for staged_file in staged.rglob("*"):
            assert not staged_file.is_symlink()
            if staged_file.is_file():
                assert injected_bytes not in staged_file.read_bytes()


def test_installed_wheel_smoke_late_failure_record_cannot_claim_pass(
    tmp_path: Path,
) -> None:
    runner = _installed_smoke_namespace()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    installed_test = tmp_path / "installed-test.py"
    installed_test.write_text("pass\n", encoding="utf-8")
    record_path = workspace / "wheel-smoke-evidence.json"
    record = runner["_new_evidence"](workspace, installed_test)
    record["wheel"]["pre_test"] = {
        "filename": "cytobridge-test.whl",
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    record["installed_tests"] = {"status": "pass"}
    record["installed_probe"] = {"status": "running"}
    runner["_write_evidence"](record_path, record)

    runner["_finalize_failure"](
        record_path,
        record,
        phase="installed_metadata_probe",
        error=RuntimeError("late probe failure"),
    )
    observed = json.loads(record_path.read_text(encoding="utf-8"))
    assert observed["status"] == "failed"
    assert observed["status"] != "pass"
    assert observed["ended_utc"] is not None
    assert observed["failure"] == {
        "message": "late probe failure",
        "phase": "installed_metadata_probe",
        "type": "RuntimeError",
    }
    assert observed["installed_tests"]["status"] == "pass"
    assert observed["installed_probe"]["status"] == "failed"
    with pytest.raises(RuntimeError, match="Only an in-progress record can become PASS"):
        runner["_complete_pass_record"](
            record_path,
            record,
            post_wheel=record["wheel"]["pre_test"],
            post_source_digest="",
            post_test_file=record["installed_test_file"]["pre_test"],
        )


def test_installed_wheel_smoke_final_pass_schema(tmp_path: Path) -> None:
    runner = _installed_smoke_namespace()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    installed_test = tmp_path / "installed-test.py"
    installed_test.write_text("pass\n", encoding="utf-8")
    wheel = tmp_path / "cytobridge-test.whl"
    wheel.write_bytes(b"wheel payload")
    record_path = workspace / "wheel-smoke-evidence.json"
    record = runner["_new_evidence"](workspace, installed_test)
    wheel_identity = runner["_wheel_identity"](wheel)
    source_inventory = [
        {"mode": "0644", "path": "CytoBridge/__init__.py", "sha256": "1" * 64, "size_bytes": 1}
    ]
    source_digest = runner["_inventory_digest"](source_inventory)
    record["staged_source"] = {
        "file_count": 1,
        "inventory": source_inventory,
        "inventory_sha256": source_digest,
    }
    record["runtime_containment"] = {
        "HOME": str(workspace / "runtime" / "home"),
        "PIP_CACHE_DIR": str(workspace / "runtime" / "cache" / "pip"),
        "TMPDIR": str(workspace / "runtime" / "tmp"),
    }
    record["wheel"]["pre_test"] = wheel_identity
    phases = (
        "wheel_build",
        "venv_creation",
        "wheel_install",
        "installed_tests",
        "installed_metadata_probe",
    )
    record["commands"] = [
        {
            "argv": ["python", "-m", "pip", "wheel"],
            "cwd": str(workspace),
            "exit_code": 0,
            "phase": phase,
            "sequence": sequence,
            "status": "pass",
            "stderr": {"sha256": "2" * 64},
            "stdout": {"sha256": "3" * 64},
        }
        for sequence, phase in enumerate(phases, start=1)
    ]
    record["installed_tests"] = {
        "status": "pass",
        "command_sequence": 4,
        "exit_code": 0,
        "stdout_sha256": "3" * 64,
        "stderr_sha256": "2" * 64,
    }
    record["installed_probe"] = {
        "status": "pass",
        "command_sequence": 5,
        "stdout_sha256": "3" * 64,
        "result": {
            "distribution_name": "CytoBridge",
            "entry_points": {"cytobridge": "CytoBridge.cli:main"},
        },
    }
    test_identity = record["installed_test_file"]["pre_test"]

    completed = runner["_complete_pass_record"](
        record_path,
        record,
        post_wheel=wheel_identity,
        post_source_digest=source_digest,
        post_test_file=test_identity,
    )
    observed = json.loads(record_path.read_text(encoding="utf-8"))
    assert observed == completed
    assert observed["schema_version"] == 2
    assert observed["status"] == "pass"
    assert observed["ended_utc"] is not None
    assert observed["failure"] is None
    assert observed["wheel"] == {
        "pre_test": wheel_identity,
        "post_test": wheel_identity,
    }
    assert observed["installed_test_file"]["post_test"] == test_identity
    assert observed["staged_source"]["inventory_sha256"] == source_digest
    assert "not authenticated provenance" in observed["evidence_notice"]
