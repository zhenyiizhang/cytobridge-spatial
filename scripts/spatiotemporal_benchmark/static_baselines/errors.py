"""Errors raised by the audited static spatiotemporal benchmark adapters."""


class BaselineError(RuntimeError):
    """Base class for a benchmark adapter failure."""


class ContractError(BaselineError):
    """The training H5AD or benchmark contract is invalid."""


class LeakageError(BaselineError):
    """A leave-one-timepoint-out input exposes the held-out stage."""


class DependencyUnavailable(BaselineError):
    """An official external implementation is unavailable."""


class OfficialAPIError(BaselineError):
    """An official implementation failed or returned an invalid coupling."""
