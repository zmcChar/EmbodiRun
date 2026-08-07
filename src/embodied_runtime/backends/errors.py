"""Failures owned by hardware execution backends."""

from embodied_runtime.errors import EmbodiedRuntimeError


class UnsupportedBackendError(EmbodiedRuntimeError):
    """A backend cannot execute the requested package on the target device."""


class BackendExecutionError(EmbodiedRuntimeError):
    """A loaded backend failed while executing model work."""


__all__ = ["BackendExecutionError", "UnsupportedBackendError"]
