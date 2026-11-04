"""Single-source method registry loader."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


REGISTRY_PATH = Path(__file__).with_name("method_registry.json")
SCHEMA_VERSION = "2.0.0"


def load_registry() -> dict[str, Any]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported registry schema: {registry.get('schema_version')!r}")
    return registry


def list_method_specs() -> dict[str, dict[str, Any]]:
    return deepcopy(load_registry()["methods"])


def get_method_spec(name: str) -> dict[str, Any]:
    methods = load_registry()["methods"]
    try:
        return deepcopy(methods[name])
    except KeyError as exc:
        raise KeyError(f"Unknown method {name!r}; choose from {sorted(methods)}") from exc
