"""CytoBridge public package entrypoint."""

from . import pl
from . import tl

try:
    from . import utils
except ModuleNotFoundError as exc:
    if exc.name in {"anndata"}:
        utils = None
    else:
        raise

try:
    from . import pp
except ModuleNotFoundError as exc:
    # Allow downstream-only usage when optional preprocessing deps are missing.
    if exc.name in {"scanpy", "anndata", "squidpy", "qnorm"}:
        pp = None
    else:
        raise

__all__ = ["pp", "tl", "pl", "utils"]
