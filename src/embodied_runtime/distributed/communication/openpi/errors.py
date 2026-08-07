"""Normalized OpenPI dependency, protocol, and transport failures."""


class OpenPIError(RuntimeError):
    """Base class for OpenPI endpoint failures."""


class OpenPIDependencyError(OpenPIError, ImportError):
    """An optional OpenPI transport dependency is not installed."""


class OpenPIProtocolError(OpenPIError):
    """The peer sent a response that does not satisfy the OpenPI contract."""


class OpenPIRemoteError(OpenPIProtocolError):
    """The policy server returned an explicit error response."""


class OpenPIUnavailableError(OpenPIError, ConnectionError):
    """The remote endpoint could not complete a connection or exchange."""


class OpenPITimeoutError(OpenPIUnavailableError, TimeoutError):
    """An OpenPI connection or request exceeded its configured deadline."""
