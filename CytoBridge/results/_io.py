"""Small file helpers shared by the results modules."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
from typing import Any


def resolve_results_dir(results_dir: str | Path | None, *, slug: str) -> Path:
    """Return an explicit results directory or the packaged example data."""

    if results_dir is None:
        resource = files("CytoBridge.results").joinpath("data", slug)
        path = Path(str(resource))
    else:
        path = Path(results_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Results directory not found: {path}")
    return path


def require_files(results_dir: Path, names: tuple[str, ...]) -> dict[str, Path]:
    """Resolve a fixed set of files below a results directory."""

    paths = {name: results_dir / name for name in names}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing processed result files:\n" + "\n".join(missing)
        )
    return paths


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_manifest(results_dir: Path) -> dict[str, Any]:
    """Read the optional public data manifest."""

    path = results_dir / "manifest.json"
    return read_json(path) if path.is_file() else {}


def prepare_output_dir(path: str | Path, *, require_empty: bool = False) -> Path:
    """Create an output directory and optionally require it to be empty."""

    output = Path(path).expanduser().resolve()
    if require_empty and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output
