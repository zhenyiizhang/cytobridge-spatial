"""Fail-closed runtime and exact-source provenance for official methods."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import DependencyUnavailable
from .registry import get_method_spec


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError as exc:
        raise DependencyUnavailable(
            f"Cannot inspect official source checkout {root}: git is unavailable"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "git command failed").strip()
        raise DependencyUnavailable(
            f"Cannot inspect official source checkout {root}: {detail}"
        ) from exc


def _compatibility_versions(spec: dict[str, Any]) -> dict[str, str]:
    required = spec.get("compatibility_versions", {})
    if not isinstance(required, dict):
        raise DependencyUnavailable("method compatibility_versions must be an object")
    observed: dict[str, str] = {}
    for distribution, expected in required.items():
        try:
            value = importlib.metadata.version(str(distribution))
        except importlib.metadata.PackageNotFoundError as exc:
            raise DependencyUnavailable(
                f"Required compatibility dependency {distribution}=={expected} is missing"
            ) from exc
        if value != str(expected):
            raise DependencyUnavailable(
                f"Compatibility dependency mismatch: {distribution} must be {expected}, found {value}"
            )
        observed[str(distribution)] = value
    return observed


def prepare_source_root(method_name: str, source_root: Path | None) -> dict[str, Any]:
    """Validate and expose an exact clean official checkout when requested."""

    spec = get_method_spec(method_name)
    expected_commit = spec.get("reference_commit")
    if source_root is None:
        return {
            "requested_source_root": None,
            "source_mode": False,
            "expected_git_commit": expected_commit,
            "git_commit": None,
            "git_dirty": None,
            "git_toplevel": None,
            "source_commit_verified": False,
        }

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise DependencyUnavailable(
            f"Requested official source root does not exist: {root}"
        )
    if not expected_commit:
        raise DependencyUnavailable(
            f"Registry method {method_name!r} has no reference_commit; "
            f"refusing unverifiable --source-root {root}"
        )
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise DependencyUnavailable(
            f"--source-root must be the official git checkout root: requested {root}, "
            f"git toplevel is {top}"
        )
    commit = _git(root, "rev-parse", "HEAD")
    if commit != str(expected_commit):
        raise DependencyUnavailable(
            f"Official {method_name} checkout commit mismatch: expected "
            f"{expected_commit}, observed {commit}"
        )
    porcelain = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if porcelain:
        preview = "; ".join(porcelain.splitlines()[:5])
        raise DependencyUnavailable(
            f"Official {method_name} checkout is dirty: {preview}"
        )
    import_roots = [root / "src", root] if (root / "src").is_dir() else [root]
    for candidate in reversed(import_roots):
        text = str(candidate)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    importlib.invalidate_caches()
    return {
        "requested_source_root": str(root),
        "source_mode": True,
        "expected_git_commit": str(expected_commit),
        "git_commit": commit,
        "git_dirty": False,
        "git_toplevel": str(top),
        "source_commit_verified": True,
        "source_import_roots": [str(path) for path in import_roots],
        "distribution_metadata_required": False,
    }


def _git_info_near(path: Path | None) -> tuple[str | None, bool | None, str | None]:
    if path is None:
        return None, None, None
    candidate = path.resolve().parent if path.is_file() else path.resolve()
    try:
        root = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", root, "status", "--porcelain", "--untracked-files=normal"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty, root
    except (OSError, subprocess.CalledProcessError):
        return None, None, None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def dependency_probe(method_name: str, source_root: Path | None = None) -> dict[str, Any]:
    spec = get_method_spec(method_name)
    source = prepare_source_root(method_name, source_root)
    module_name = spec.get("module")
    if module_name is None:
        return {
            **source,
            "module": None,
            "observed_module": None,
            "available": True,
            "distribution": None,
            "version": None,
            "module_file": None,
            "compatibility_versions": {},
        }
    observed_module = next(
        (
            candidate
            for candidate in spec.get("module_candidates", [module_name])
            if importlib.util.find_spec(candidate) is not None
        ),
        None,
    )
    distribution = None
    version = None
    if not source["source_mode"]:
        for candidate in spec.get("distribution_candidates", []):
            try:
                version = importlib.metadata.version(candidate)
                distribution = candidate
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        expected_version = spec.get("reference_version")
        if observed_module is not None and expected_version and version != str(expected_version):
            raise DependencyUnavailable(
                f"Official {method_name} version mismatch: expected {expected_version}, found {version}"
            )
    compatibility = _compatibility_versions(spec)
    return {
        **source,
        "module": module_name,
        "observed_module": observed_module,
        "available": observed_module is not None,
        "distribution": distribution,
        "version": version,
        "module_file": None,
        "compatibility_versions": compatibility,
    }


def import_official(
    method_name: str, source_root: Path | None = None
) -> tuple[ModuleType, dict[str, Any]]:
    spec = get_method_spec(method_name)
    probe = dependency_probe(method_name, source_root)
    module_name = spec.get("module")
    if not module_name:
        raise DependencyUnavailable(
            f"{method_name} is a control and has no external official package"
        )
    if not probe["available"]:
        distributions = ", ".join(spec.get("distribution_candidates", [])) or module_name
        raise DependencyUnavailable(
            f"Official dependency for {method_name!r} is missing: cannot import "
            f"{module_name!r} ({distributions}); no surrogate will be run"
        )
    try:
        module = importlib.import_module(str(probe["observed_module"]))
    except Exception as exc:
        raise DependencyUnavailable(
            f"Official {method_name!r} module import failed: "
            f"{type(exc).__name__}: {exc}; no surrogate will be run"
        ) from exc
    raw_file = getattr(module, "__file__", None)
    module_file = Path(raw_file).resolve() if raw_file else None
    probe["module_file"] = str(module_file) if module_file else None
    if probe["source_mode"]:
        root = Path(str(probe["requested_source_root"]))
        if module_file is None or not _is_relative_to(module_file, root):
            raise DependencyUnavailable(
                f"Official {method_name} import did not come from --source-root "
                f"{root}: module_file={module_file}"
            )
        probe["module_from_requested_source"] = True
        probe["version"] = str(getattr(module, "__version__", "")) or None
        probe["version_source"] = "module attribute; exact git commit is authoritative"
    else:
        if probe["version"] is None:
            raw_version = getattr(module, "__version__", None)
            probe["version"] = str(raw_version) if raw_version is not None else None
        if probe.get("git_commit") is None:
            commit, dirty, root = _git_info_near(module_file)
            probe["git_commit"] = commit
            probe["git_dirty"] = dirty
            probe["detected_git_root"] = root
    return module, probe
