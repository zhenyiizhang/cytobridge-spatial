#!/usr/bin/env python3
"""Build and test a CytoBridge wheel with pip's resolver kept offline.

Run from the project root with::

    python scripts/smoke_installed_wheel.py

The command stages verified regular files into a private workspace, disables
package-index resolution, builds through PEP 517 using the invoking Python's
installed build tools, creates a clean virtual environment, and runs the
installed-only contract tests outside the source tree.

This is resolver-offline, not a network sandbox: an arbitrary build backend
could still initiate its own network connection.  The retained
``wheel-smoke-evidence.json`` sidecar is local checksum and test evidence, not
authenticated provenance.  Pass ``--work-dir NEW_PATH`` to retain all inputs,
logs, the wheel, and the evidence sidecar.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STAGED_ROOT_FILES = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
)
_EVIDENCE_FILENAME = "wheel-smoke-evidence.json"
_REQUIRED_COMMAND_PHASES = (
    "wheel_build",
    "venv_creation",
    "wheel_install",
    "installed_tests",
    "installed_metadata_probe",
)
_ENVIRONMENT_REMOVE_NAMES = frozenset(
    {
        "ARCHFLAGS",
        "BASH_ENV",
        "CC",
        "CFLAGS",
        "CPPFLAGS",
        "CXX",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "ENV",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_PROXY_COMMAND",
        "LDFLAGS",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "MACOSX_DEPLOYMENT_TARGET",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "REQUESTS_CA_BUNDLE",
        "SETUPTOOLS_USE_DISTUTILS",
        "SOURCE_DATE_EPOCH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "VIRTUAL_ENV",
    }
)
_ENVIRONMENT_REMOVE_PREFIXES = (
    "CMAKE_",
    "CONAN_",
    "MESON_",
    "PIP_",
    "PYPA_BUILD_",
    "SETUPTOOLS_SCM_",
    "SKBUILD_",
)
_PROXY_NAMES = frozenset(
    {
        "all_proxy",
        "ftp_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "rsync_proxy",
    }
)


class SmokeCommandError(RuntimeError):
    """Raised after a failed subprocess has been recorded in the sidecar."""

    def __init__(self, phase: str, returncode: int) -> None:
        super().__init__(f"{phase} command failed with exit code {returncode}")
        self.phase = phase
        self.returncode = returncode


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _regular_file_record(path: Path, *, display_path: str | None = None) -> dict[str, object]:
    """Hash an ordinary file without following a symbolic link."""

    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required regular file is missing: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Expected an ordinary file without symlinks: {path}")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"Could not safely open ordinary file: {path}") from exc

    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"File changed to a non-regular object: {path}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError(f"File identity changed before hashing: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if (opened.st_dev, opened.st_ino, opened.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ) or size != after.st_size:
        raise RuntimeError(f"File changed while it was being hashed: {path}")
    return {
        "path": display_path if display_path is not None else str(path),
        "mode": f"{stat.S_IMODE(after.st_mode):04o}",
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def _wheel_identity(path: Path) -> dict[str, object]:
    record = _regular_file_record(path)
    return {
        "filename": path.name,
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }


def _inventory_digest(inventory: Sequence[Mapping[str, object]]) -> str:
    serialized = json.dumps(
        list(inventory),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(serialized)


def _directory_open_flags() -> int:
    missing = [name for name in ("O_DIRECTORY", "O_NOFOLLOW") if not hasattr(os, name)]
    if missing:
        raise RuntimeError(
            "Secure source staging requires directory-FD primitives: " + ", ".join(missing)
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _source_file_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Secure source staging requires O_NOFOLLOW")
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _destination_file_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("Secure source staging requires O_NOFOLLOW")
    return (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _open_root_directory(path: Path, *, label: str) -> int:
    try:
        descriptor = os.open(path, _directory_open_flags())
    except OSError as exc:
        raise RuntimeError(f"Could not safely open {label}: {path}") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise RuntimeError(f"{label} is not an ordinary directory: {path}")
    return descriptor


def _directory_names(directory_fd: int, *, label: str) -> list[str]:
    try:
        os.lseek(directory_fd, 0, os.SEEK_SET)
        names = os.listdir(directory_fd)
    except OSError as exc:
        raise RuntimeError(f"Could not list held directory FD for {label}") from exc
    if any(not isinstance(name, str) or "/" in name or name in {"", ".", ".."} for name in names):
        raise RuntimeError(f"Held directory FD returned an unsafe entry for {label}")
    return sorted(names)


def _relative_status(parent_fd: int, name: str, *, label: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(f"Could not lstat {label}: {name}") from exc


def _open_child_directory(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    label: str,
) -> int:
    try:
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise RuntimeError(f"{label} changed or became unsafe before directory open: {name}") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or not _same_object(expected, opened):
        os.close(descriptor)
        raise RuntimeError(f"{label} identity changed before directory traversal: {name}")
    return descriptor


def _create_destination_directory(parent_fd: int, name: str, *, label: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        created = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _directory_open_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise RuntimeError(f"Could not exclusively create destination directory for {label}: {name}") from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(created.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or not _same_object(created, opened)
    ):
        os.close(descriptor)
        raise RuntimeError(f"Destination directory identity changed for {label}: {name}")
    return descriptor


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise RuntimeError("Destination file write made no progress")
        remaining = remaining[written:]


def _record_held_source_file(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    relative: Path,
    destination_parent_fd: int | None,
) -> dict[str, object]:
    try:
        source_fd = os.open(name, _source_file_open_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise RuntimeError(
            f"Source file changed or became unsafe before open: {relative.as_posix()}"
        ) from exc
    opened = os.fstat(source_fd)
    if not stat.S_ISREG(opened.st_mode) or not _same_object(expected, opened):
        os.close(source_fd)
        raise RuntimeError(f"Source file identity changed before copy: {relative.as_posix()}")

    destination_fd: int | None = None
    if destination_parent_fd is not None:
        try:
            destination_fd = os.open(
                name,
                _destination_file_open_flags(),
                stat.S_IMODE(opened.st_mode),
                dir_fd=destination_parent_fd,
            )
        except OSError as exc:
            os.close(source_fd)
            raise RuntimeError(
                f"Could not exclusively create staged file: {relative.as_posix()}"
            ) from exc

    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if destination_fd is not None:
                _write_all(destination_fd, chunk)
        after = os.fstat(source_fd)
        if destination_fd is not None:
            os.fchmod(destination_fd, stat.S_IMODE(opened.st_mode))
            os.fsync(destination_fd)
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)

    if (
        not _same_object(opened, after)
        or opened.st_size != after.st_size
        or opened.st_mtime_ns != after.st_mtime_ns
        or size != after.st_size
    ):
        raise RuntimeError(f"Source file changed while copied: {relative.as_posix()}")
    return {
        "path": relative.as_posix(),
        "mode": f"{stat.S_IMODE(after.st_mode):04o}",
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def _walk_held_package_directory(
    source_directory_fd: int,
    *,
    relative_directory: Path,
    destination_directory_fd: int | None,
    include: bool,
    inventory: list[dict[str, object]],
) -> None:
    for name in _directory_names(
        source_directory_fd,
        label=relative_directory.as_posix(),
    ):
        relative = relative_directory / name
        entry_status = _relative_status(
            source_directory_fd,
            name,
            label=relative.as_posix(),
        )
        if stat.S_ISLNK(entry_status.st_mode):
            raise RuntimeError(f"Source tree contains a symbolic link: {relative.as_posix()}")
        if stat.S_ISDIR(entry_status.st_mode):
            source_child_fd = _open_child_directory(
                source_directory_fd,
                name,
                entry_status,
                label=relative.as_posix(),
            )
            child_include = include and name != "__pycache__"
            destination_child_fd: int | None = None
            try:
                if child_include and destination_directory_fd is not None:
                    destination_child_fd = _create_destination_directory(
                        destination_directory_fd,
                        name,
                        label=relative.as_posix(),
                    )
                _walk_held_package_directory(
                    source_child_fd,
                    relative_directory=relative,
                    destination_directory_fd=destination_child_fd,
                    include=child_include,
                    inventory=inventory,
                )
            finally:
                if destination_child_fd is not None:
                    os.close(destination_child_fd)
                os.close(source_child_fd)
            continue
        if not stat.S_ISREG(entry_status.st_mode):
            raise RuntimeError(f"Source tree contains a special file: {relative.as_posix()}")

        file_include = include and Path(name).suffix != ".pyc"
        record = _record_held_source_file(
            source_directory_fd,
            name,
            entry_status,
            relative=relative,
            destination_parent_fd=(
                destination_directory_fd if file_include else None
            ),
        )
        if file_include:
            inventory.append(record)


def _inventory_held_source_root(
    source_root_fd: int,
    *,
    destination_root_fd: int | None = None,
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for name in _STAGED_ROOT_FILES:
        entry_status = _relative_status(source_root_fd, name, label=name)
        if stat.S_ISLNK(entry_status.st_mode) or not stat.S_ISREG(entry_status.st_mode):
            raise RuntimeError(f"Expected an ordinary file without symlinks: {name}")
        inventory.append(
            _record_held_source_file(
                source_root_fd,
                name,
                entry_status,
                relative=Path(name),
                destination_parent_fd=destination_root_fd,
            )
        )

    package_status = _relative_status(source_root_fd, "CytoBridge", label="CytoBridge")
    if stat.S_ISLNK(package_status.st_mode):
        raise RuntimeError("Source tree contains a symbolic link: CytoBridge")
    if not stat.S_ISDIR(package_status.st_mode):
        raise RuntimeError("Expected an ordinary package directory: CytoBridge")
    source_package_fd = _open_child_directory(
        source_root_fd,
        "CytoBridge",
        package_status,
        label="CytoBridge",
    )
    destination_package_fd: int | None = None
    try:
        if destination_root_fd is not None:
            destination_package_fd = _create_destination_directory(
                destination_root_fd,
                "CytoBridge",
                label="CytoBridge",
            )
        _walk_held_package_directory(
            source_package_fd,
            relative_directory=Path("CytoBridge"),
            destination_directory_fd=destination_package_fd,
            include=True,
            inventory=inventory,
        )
    finally:
        if destination_package_fd is not None:
            os.close(destination_package_fd)
        os.close(source_package_fd)
    return sorted(inventory, key=lambda item: str(item["path"]))


def _source_inventory(project_root: Path) -> list[dict[str, object]]:
    """Inventory build inputs entirely through held, no-follow directory FDs."""

    root_fd = _open_root_directory(project_root, label="source root")
    try:
        return _inventory_held_source_root(root_fd)
    finally:
        os.close(root_fd)


def _stage_source(project_root: Path, staged_source: Path) -> list[dict[str, object]]:
    """Copy held source FDs into exclusively created files under held destination FDs."""

    if staged_source.name in {"", ".", ".."}:
        raise RuntimeError(f"Staged source must name a safe directory leaf: {staged_source}")
    source_root_fd = _open_root_directory(project_root, label="source root")
    destination_parent_fd = _open_root_directory(
        staged_source.parent,
        label="staging parent",
    )
    destination_root_fd: int | None = None
    try:
        destination_root_fd = _create_destination_directory(
            destination_parent_fd,
            staged_source.name,
            label="staged source root",
        )
        inventory = _inventory_held_source_root(
            source_root_fd,
            destination_root_fd=destination_root_fd,
        )
        staged_inventory = _inventory_held_source_root(destination_root_fd)
        if staged_inventory != inventory:
            raise RuntimeError(
                "Staged source inventory does not match verified source inputs"
            )
        return staged_inventory
    finally:
        if destination_root_fd is not None:
            os.close(destination_root_fd)
        os.close(destination_parent_fd)
        os.close(source_root_fd)


def _assert_ordinary_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        status = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"{label} must be an ordinary directory without symlinks: {path}")
    return status


def _assert_safe_parent(path: Path) -> None:
    status = _assert_ordinary_directory(path, label="Workspace parent")
    if hasattr(os, "geteuid") and status.st_uid not in {0, os.geteuid()}:
        raise RuntimeError(f"Workspace parent has an unexpected owner: {path}")
    shared_writable = status.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if shared_writable and not status.st_mode & stat.S_ISVTX:
        raise RuntimeError(f"Workspace parent is writable by others without sticky bit: {path}")
    if not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(f"Workspace parent is not writable and searchable: {path}")


def _assert_private_directory(path: Path, *, label: str) -> os.stat_result:
    status = _assert_ordinary_directory(path, label=label)
    if status.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise RuntimeError(f"{label} must not grant group or other permissions: {path}")
    if hasattr(os, "geteuid") and status.st_uid != os.geteuid():
        raise RuntimeError(f"{label} must be owned by the current user: {path}")
    return status


def _verify_created_directory(path: Path, created: os.stat_result, *, label: str) -> None:
    observed = _assert_private_directory(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identities = {
        (created.st_dev, created.st_ino),
        (observed.st_dev, observed.st_ino),
        (opened.st_dev, opened.st_ino),
    }
    if len(identities) != 1 or not stat.S_ISDIR(opened.st_mode):
        raise RuntimeError(f"{label} identity changed after creation: {path}")


def _create_explicit_workspace(requested: Path) -> Path:
    """Atomically create a new leaf after canonicalizing only its parent."""

    expanded = requested.expanduser()
    if expanded.name in {"", ".", ".."}:
        raise RuntimeError(f"--work-dir must name a new directory leaf: {requested}")
    candidate = expanded if expanded.is_absolute() else Path.cwd() / expanded
    try:
        canonical_parent = candidate.parent.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"--work-dir parent must resolve to an existing safe directory: {candidate.parent}"
        ) from exc
    _assert_safe_parent(canonical_parent)
    workspace = canonical_parent / candidate.name

    # lexists detects dangling links; mkdir below closes the check/create race.
    if os.path.lexists(os.fspath(workspace)):
        raise RuntimeError(f"--work-dir already exists or is a symlink: {workspace}")
    try:
        os.mkdir(workspace, mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError(f"--work-dir appeared during creation: {workspace}") from exc
    created = workspace.lstat()
    _verify_created_directory(workspace, created, label="Workspace")
    return workspace


def _create_private_child(parent: Path, name: str) -> Path:
    _assert_private_directory(parent, label="Runtime parent")
    child = parent / name
    if os.path.lexists(os.fspath(child)):
        raise RuntimeError(f"Contained runtime path already exists: {child}")
    os.mkdir(child, mode=0o700)
    created = child.lstat()
    _verify_created_directory(child, created, label="Contained runtime directory")
    return child


def _contained_environment(
    workspace: Path,
    source: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Create private temp/cache/home directories and a sanitized environment."""

    _assert_private_directory(workspace, label="Workspace")
    runtime = _create_private_child(workspace, "runtime")
    home = _create_private_child(runtime, "home")
    temporary = _create_private_child(runtime, "tmp")
    cache = _create_private_child(runtime, "cache")
    config = _create_private_child(runtime, "config")
    data = _create_private_child(runtime, "data")
    pip_cache = _create_private_child(cache, "pip")
    matplotlib_cache = _create_private_child(cache, "matplotlib")
    numba_cache = _create_private_child(cache, "numba")
    torch_cache = _create_private_child(cache, "torch")

    pip_configuration = config / "pip.conf"
    descriptor = os.open(
        pip_configuration,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    os.close(descriptor)
    _regular_file_record(pip_configuration)

    environment = dict(os.environ if source is None else source)
    for name in tuple(environment):
        upper = name.upper()
        if (
            upper in _ENVIRONMENT_REMOVE_NAMES
            or name.casefold() in _PROXY_NAMES
            or any(upper.startswith(prefix) for prefix in _ENVIRONMENT_REMOVE_PREFIXES)
        ):
            environment.pop(name)
    environment.update(
        {
            "HOME": str(home),
            "MPLCONFIGDIR": str(matplotlib_cache),
            "NUMBA_CACHE_DIR": str(numba_cache),
            "PIP_CACHE_DIR": str(pip_cache),
            "PIP_CONFIG_FILE": str(pip_configuration),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "TMPDIR": str(temporary),
            "TORCH_HOME": str(torch_cache),
            "XDG_CACHE_HOME": str(cache),
            "XDG_CONFIG_HOME": str(config),
            "XDG_DATA_HOME": str(data),
        }
    )
    containment = {
        name: environment[name]
        for name in (
            "HOME",
            "MPLCONFIGDIR",
            "NUMBA_CACHE_DIR",
            "PIP_CACHE_DIR",
            "PIP_CONFIG_FILE",
            "TEMP",
            "TMP",
            "TMPDIR",
            "TORCH_HOME",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
        )
    }
    return environment, containment


def _build_command(
    staged_source: Path,
    wheel_directory: Path,
    python_executable: Path,
) -> list[str]:
    return [
        str(python_executable),
        "-m",
        "pip",
        "wheel",
        "--use-pep517",
        "--no-build-isolation",
        "--no-deps",
        "--no-index",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--wheel-dir",
        str(wheel_directory),
        str(staged_source),
    ]


def _install_command(venv_python: Path, wheel: Path) -> list[str]:
    return [
        str(venv_python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
        "--no-cache-dir",
        "--disable-pip-version-check",
        str(wheel),
    ]


def _installed_test_command(venv_python: Path, test_file: Path) -> list[str]:
    return [str(venv_python), str(test_file), "-v"]


def _installed_probe_source() -> str:
    return """
import json
from importlib import metadata, resources
from pathlib import Path
import platform
import sys
import CytoBridge

distribution = metadata.distribution("CytoBridge")
entry_points = {
    item.name: item.value
    for item in distribution.entry_points
    if item.group == "console_scripts"
}
probed = ("anndata", "matplotlib", "numpy", "pandas", "scanpy", "torch")
print(json.dumps({
    "config_present": resources.files("CytoBridge").joinpath(
        "configs", "simulation_config.yaml"
    ).is_file(),
    "distribution_name": distribution.metadata["Name"],
    "distribution_version": distribution.version,
    "entry_points": entry_points,
    "imported_version": CytoBridge.__version__,
    "package_source": str(Path(CytoBridge.__file__).resolve()),
    "pip_version": metadata.version("pip"),
    "python_executable": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "requires_python": distribution.metadata["Requires-Python"],
    "scientific_modules_loaded": sorted(name for name in probed if name in sys.modules),
}, sort_keys=True))
""".strip()


def _tool_versions() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for distribution in ("pip", "setuptools", "wheel"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "distributions": versions,
    }


def _new_evidence(workspace: Path, installed_test: Path) -> dict[str, object]:
    return {
        "schema_version": 2,
        "evidence_kind": "local_resolver_offline_wheel_smoke",
        "evidence_notice": (
            "Local checksum and test evidence only; this is not authenticated provenance."
        ),
        "status": "in_progress",
        "started_utc": _utc_now(),
        "ended_utc": None,
        "network_policy": {
            "mode": "resolver_offline",
            "note": (
                "pip index resolution is disabled; arbitrary backend network access is not "
                "sandboxed"
            ),
        },
        "workspace": str(workspace),
        "build_environment": _tool_versions(),
        "runtime_containment": None,
        "staged_source": None,
        "installed_test_file": {
            "pre_test": _regular_file_record(installed_test),
            "post_test": None,
        },
        "wheel": {"pre_test": None, "post_test": None},
        "commands": [],
        "installed_tests": {"status": "pending"},
        "installed_probe": {"status": "pending"},
        "failure": None,
    }


def _write_evidence(record_path: Path, record: Mapping[str, object]) -> None:
    payload = json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    temporary = record_path.with_name(f".{record_path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, record_path)


def _emit_output(payload: bytes, *, stderr: bool) -> None:
    if not payload:
        return
    stream = sys.stderr if stderr else sys.stdout
    buffer = getattr(stream, "buffer", None)
    if buffer is not None:
        buffer.write(payload)
        buffer.flush()
    else:  # pragma: no cover - StringIO-like embedding
        stream.write(payload.decode("utf-8", errors="replace"))
        stream.flush()


def _run_recorded_command(
    phase: str,
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    workspace: Path,
    record_path: Path,
    record: dict[str, object],
) -> tuple[dict[str, object], bytes]:
    commands = record["commands"]
    assert isinstance(commands, list)
    sequence = len(commands) + 1
    logs = workspace / "logs"
    stdout_path = logs / f"{sequence:02d}-{phase}.stdout"
    stderr_path = logs / f"{sequence:02d}-{phase}.stderr"
    print(f"+ {shlex.join(command)}", flush=True)
    started = _utc_now()
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(environment),
            check=False,
            capture_output=True,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode: int | None = completed.returncode
        exception = None
    except OSError as exc:
        stdout = b""
        stderr = str(exc).encode("utf-8", errors="replace")
        returncode = None
        exception = {"type": type(exc).__name__, "message": str(exc)}

    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    _emit_output(stdout, stderr=False)
    _emit_output(stderr, stderr=True)
    command_record: dict[str, object] = {
        "sequence": sequence,
        "phase": phase,
        "argv": list(command),
        "cwd": str(cwd),
        "started_utc": started,
        "ended_utc": _utc_now(),
        "status": "pass" if returncode == 0 else "failed",
        "exit_code": returncode,
        "stdout": {
            "path": str(stdout_path.relative_to(workspace)),
            "sha256": _sha256_bytes(stdout),
            "size_bytes": len(stdout),
        },
        "stderr": {
            "path": str(stderr_path.relative_to(workspace)),
            "sha256": _sha256_bytes(stderr),
            "size_bytes": len(stderr),
        },
        "exception": exception,
    }
    commands.append(command_record)
    _write_evidence(record_path, record)
    if returncode != 0:
        raise SmokeCommandError(phase, -1 if returncode is None else returncode)
    return command_record, stdout


def _validate_installed_probe(
    probe: Mapping[str, object],
    *,
    venv_directory: Path,
    expected_version: str,
) -> None:
    expected = {
        "config_present": True,
        "distribution_name": "CytoBridge",
        "distribution_version": expected_version,
        "imported_version": expected_version,
        "requires_python": ">=3.10",
        "scientific_modules_loaded": [],
    }
    for key, value in expected.items():
        if probe.get(key) != value:
            raise RuntimeError(
                f"Installed metadata probe mismatch for {key}: "
                f"expected={value!r}, observed={probe.get(key)!r}"
            )
    entry_points = probe.get("entry_points")
    if not isinstance(entry_points, dict) or entry_points.get("cytobridge") != "CytoBridge.cli:main":
        raise RuntimeError("Installed metadata probe did not find the expected console entry point")
    source = Path(str(probe.get("package_source", ""))).resolve()
    if not source.is_relative_to(venv_directory.resolve()):
        raise RuntimeError(f"Installed metadata probe imported outside the smoke venv: {source}")


def _finalize_failure(
    record_path: Path,
    record: dict[str, object],
    *,
    phase: str,
    error: BaseException,
) -> None:
    if record.get("status") == "pass":
        raise RuntimeError("Cannot replace a completed PASS record with failure evidence")
    installed_tests = record.get("installed_tests")
    if isinstance(installed_tests, dict) and installed_tests.get("status") == "running":
        installed_tests["status"] = "failed"
    installed_probe = record.get("installed_probe")
    if isinstance(installed_probe, dict) and installed_probe.get("status") == "running":
        installed_probe["status"] = "failed"
    record["status"] = "failed"
    record["ended_utc"] = _utc_now()
    record["failure"] = {
        "phase": phase,
        "type": type(error).__name__,
        "message": str(error),
    }
    _write_evidence(record_path, record)


def _complete_pass_record(
    record_path: Path,
    record: dict[str, object],
    *,
    post_wheel: Mapping[str, object],
    post_source_digest: str,
    post_test_file: Mapping[str, object],
) -> dict[str, object]:
    """Apply the final PASS state only after every recorded gate is satisfied."""

    if record.get("status") != "in_progress":
        raise RuntimeError("Only an in-progress record can become PASS")
    if record.get("ended_utc") is not None or record.get("failure") is not None:
        raise RuntimeError("A completed or failed record cannot become PASS")
    if not record.get("started_utc"):
        raise RuntimeError("PASS evidence requires a start timestamp")
    if "not authenticated provenance" not in str(record.get("evidence_notice", "")):
        raise RuntimeError("PASS evidence must state that it is not authenticated provenance")

    build_environment = record.get("build_environment")
    if not isinstance(build_environment, dict):
        raise RuntimeError("PASS evidence requires build-environment versions")
    distributions = build_environment.get("distributions")
    if not isinstance(distributions, dict) or any(
        not distributions.get(name) for name in ("pip", "setuptools", "wheel")
    ):
        raise RuntimeError("PASS evidence requires pip, setuptools, and wheel versions")
    containment = record.get("runtime_containment")
    if not isinstance(containment, dict) or any(
        name not in containment for name in ("HOME", "TMPDIR", "PIP_CACHE_DIR")
    ):
        raise RuntimeError("PASS evidence requires contained home, temp, and cache paths")

    wheel = record.get("wheel")
    source = record.get("staged_source")
    test_file = record.get("installed_test_file")
    commands = record.get("commands")
    if not isinstance(wheel, dict) or wheel.get("pre_test") != dict(post_wheel):
        raise RuntimeError("Post-test wheel checksum does not match the pre-test checksum")
    if not isinstance(source, dict) or source.get("inventory_sha256") != post_source_digest:
        raise RuntimeError("Staged source inventory changed during the smoke test")
    source_inventory = source.get("inventory")
    if not isinstance(source_inventory, list) or not source_inventory:
        raise RuntimeError("PASS evidence requires the staged source inventory")
    if source.get("file_count") != len(source_inventory):
        raise RuntimeError("Staged source inventory count is inconsistent")
    if source.get("inventory_sha256") != _inventory_digest(source_inventory):
        raise RuntimeError("Stored source inventory digest is inconsistent")
    if not isinstance(test_file, dict) or test_file.get("pre_test") != dict(post_test_file):
        raise RuntimeError("Installed contract test file changed during the smoke test")
    if not isinstance(commands, list) or not commands or any(
        not isinstance(item, dict) or item.get("status") != "pass" for item in commands
    ):
        raise RuntimeError("Not every recorded smoke command passed")
    if tuple(item.get("phase") for item in commands) != _REQUIRED_COMMAND_PHASES:
        raise RuntimeError("Recorded smoke commands are incomplete or out of order")
    installed_tests = record.get("installed_tests")
    if (
        not isinstance(installed_tests, dict)
        or installed_tests.get("status") != "pass"
        or installed_tests.get("exit_code") != 0
    ):
        raise RuntimeError("Installed-wheel contract tests have not passed")
    test_command = commands[3]
    if (
        installed_tests.get("command_sequence") != test_command.get("sequence")
        or installed_tests.get("stdout_sha256") != test_command.get("stdout", {}).get("sha256")
        or installed_tests.get("stderr_sha256") != test_command.get("stderr", {}).get("sha256")
    ):
        raise RuntimeError("Installed-test result is not bound to its command outputs")
    installed_probe = record.get("installed_probe")
    if not isinstance(installed_probe, dict) or installed_probe.get("status") != "pass":
        raise RuntimeError("Installed metadata and entry-point probe has not passed")
    probe_command = commands[4]
    probe_result = installed_probe.get("result")
    if (
        installed_probe.get("command_sequence") != probe_command.get("sequence")
        or installed_probe.get("stdout_sha256") != probe_command.get("stdout", {}).get("sha256")
        or not isinstance(probe_result, dict)
        or probe_result.get("distribution_name") != "CytoBridge"
        or probe_result.get("entry_points", {}).get("cytobridge") != "CytoBridge.cli:main"
    ):
        raise RuntimeError("Installed probe is not bound to valid metadata output")

    wheel["post_test"] = dict(post_wheel)
    test_file["post_test"] = dict(post_test_file)
    record["status"] = "pass"
    record["ended_utc"] = _utc_now()
    record["failure"] = None
    _write_evidence(record_path, record)
    return record


def _execute_smoke(workspace: Path) -> dict[str, object]:
    _assert_private_directory(workspace, label="Workspace")
    staged_source = workspace / "source"
    wheel_directory = workspace / "dist"
    venv_directory = workspace / "venv"
    test_cwd = workspace / "test-cwd"
    logs = workspace / "logs"
    record_path = workspace / _EVIDENCE_FILENAME
    installed_test = PROJECT_ROOT / "tests" / "test_installed_package_contract.py"
    record = _new_evidence(workspace, installed_test)
    _write_evidence(record_path, record)
    phase = "runtime_containment"
    try:
        environment, containment = _contained_environment(workspace)
        record["runtime_containment"] = containment
        for directory in (wheel_directory, test_cwd, logs):
            _create_private_child(workspace, directory.name)

        phase = "source_staging"
        inventory = _stage_source(PROJECT_ROOT, staged_source)
        record["staged_source"] = {
            "file_count": len(inventory),
            "inventory": inventory,
            "inventory_sha256": _inventory_digest(inventory),
        }
        _write_evidence(record_path, record)

        phase = "wheel_build"
        build_command = _build_command(
            staged_source,
            wheel_directory,
            Path(sys.executable),
        )
        _run_recorded_command(
            phase,
            build_command,
            cwd=workspace,
            environment=environment,
            workspace=workspace,
            record_path=record_path,
            record=record,
        )
        wheels = sorted(wheel_directory.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected exactly one wheel, found {len(wheels)}")
        wheel_path = wheels[0]
        wheel_record = record["wheel"]
        assert isinstance(wheel_record, dict)
        wheel_record["pre_test"] = _wheel_identity(wheel_path)
        _write_evidence(record_path, record)

        phase = "venv_creation"
        _run_recorded_command(
            phase,
            [sys.executable, "-m", "venv", str(venv_directory)],
            cwd=workspace,
            environment=environment,
            workspace=workspace,
            record_path=record_path,
            record=record,
        )
        _assert_ordinary_directory(venv_directory, label="Smoke-test venv")
        os.chmod(venv_directory, 0o700, follow_symlinks=False)
        _assert_private_directory(venv_directory, label="Smoke-test venv")
        _assert_non_system_venv(venv_directory)
        venv_python = _venv_python(venv_directory)

        phase = "wheel_install"
        _run_recorded_command(
            phase,
            _install_command(venv_python, wheel_path),
            cwd=test_cwd,
            environment=environment,
            workspace=workspace,
            record_path=record_path,
            record=record,
        )

        phase = "installed_tests"
        record["installed_tests"] = {"status": "running"}
        _write_evidence(record_path, record)
        test_environment = dict(environment)
        test_environment["CYTOBRIDGE_TEST_INSTALLED"] = "1"
        test_command_record, _ = _run_recorded_command(
            phase,
            _installed_test_command(venv_python, installed_test),
            cwd=test_cwd,
            environment=test_environment,
            workspace=workspace,
            record_path=record_path,
            record=record,
        )
        record["installed_tests"] = {
            "status": "pass",
            "command_sequence": test_command_record["sequence"],
            "exit_code": test_command_record["exit_code"],
            "stdout_sha256": test_command_record["stdout"]["sha256"],
            "stderr_sha256": test_command_record["stderr"]["sha256"],
        }
        _write_evidence(record_path, record)

        phase = "installed_metadata_probe"
        record["installed_probe"] = {"status": "running"}
        _write_evidence(record_path, record)
        probe_command_record, probe_stdout = _run_recorded_command(
            phase,
            [str(venv_python), "-c", _installed_probe_source()],
            cwd=test_cwd,
            environment=environment,
            workspace=workspace,
            record_path=record_path,
            record=record,
        )
        probe = json.loads(probe_stdout.decode("utf-8"))
        expected_version = str(record["wheel"]["pre_test"]["filename"]).split("-")[1]
        _validate_installed_probe(
            probe,
            venv_directory=venv_directory,
            expected_version=expected_version,
        )
        record["installed_probe"] = {
            "status": "pass",
            "command_sequence": probe_command_record["sequence"],
            "stdout_sha256": probe_command_record["stdout"]["sha256"],
            "result": probe,
        }
        _write_evidence(record_path, record)

        phase = "final_integrity"
        post_wheel = _wheel_identity(wheel_path)
        post_inventory = _source_inventory(staged_source)
        post_test = _regular_file_record(installed_test)
        completed = _complete_pass_record(
            record_path,
            record,
            post_wheel=post_wheel,
            post_source_digest=_inventory_digest(post_inventory),
            post_test_file=post_test,
        )
        return {
            "evidence": str(record_path),
            "sha256": post_wheel["sha256"],
            "size_bytes": post_wheel["size_bytes"],
            "status": completed["status"],
            "test_cwd": str(test_cwd),
            "wheel": str(wheel_path),
        }
    except Exception as error:
        try:
            _finalize_failure(
                record_path,
                record,
                phase=phase,
                error=error,
            )
        except Exception as record_error:
            raise RuntimeError(
                f"Smoke failed during {phase}; failure evidence also failed: {record_error}"
            ) from error
        raise


def _venv_python(venv_directory: Path) -> Path:
    if os.name == "nt":  # pragma: no cover - Windows path convention
        return venv_directory / "Scripts" / "python.exe"
    return venv_directory / "bin" / "python"


def _assert_non_system_venv(venv_directory: Path) -> None:
    configuration = venv_directory / "pyvenv.cfg"
    record = _regular_file_record(configuration)
    del record
    values: dict[str, str] = {}
    for raw_line in configuration.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip().casefold()] = value.strip().casefold()
    if values.get("include-system-site-packages") != "false":
        raise RuntimeError("Smoke-test venv must not include system site-packages")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and test the installed CytoBridge wheel with pip index resolution "
            "disabled (not a general network sandbox)."
        )
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="new directory in which to retain wheel, logs, and checksum evidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    temporary_workspace = arguments.work_dir is None
    if temporary_workspace:
        workspace = Path(tempfile.mkdtemp(prefix="cytobridge-wheel-smoke-"))
        _assert_private_directory(workspace, label="Workspace")
    else:
        workspace = _create_explicit_workspace(arguments.work_dir)

    try:
        result = _execute_smoke(workspace)
    except Exception as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "evidence": str(workspace / _EVIDENCE_FILENAME),
                    "status": "failed",
                    "workspace_retained": str(workspace),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    if temporary_workspace:
        shutil.rmtree(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
