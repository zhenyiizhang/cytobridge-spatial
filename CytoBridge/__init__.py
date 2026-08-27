"""CytoBridge public package entrypoint.

The public namespaces are loaded only when requested.  This keeps package
metadata and the command-line diagnostics usable without importing scientific
or plotting stacks as a side effect of ``import CytoBridge``.
"""

from importlib import import_module
from types import ModuleType

from ._version import __version__


_PUBLIC_MODULES = frozenset({"pl", "pp", "results", "tl", "utils"})
__all__ = ["pp", "tl", "pl", "results", "utils", "__version__"]


def __getattr__(name: str) -> ModuleType:
    """Load a public namespace on first attribute access."""

    if name not in _PUBLIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Include lazily exposed namespaces in interactive discovery."""

    return sorted(set(globals()) | _PUBLIC_MODULES)
