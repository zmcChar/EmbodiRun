"""Failures owned by model packages and adapters."""

from embodied_runtime.errors import EmbodiedRuntimeError


class ModelPackageError(EmbodiedRuntimeError):
    """A model package or adapter violates its declared semantics."""


__all__ = ["ModelPackageError"]
