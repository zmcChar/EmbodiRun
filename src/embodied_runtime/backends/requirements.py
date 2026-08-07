"""Resource requirements declared by a model deployment."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceRequirements:
    minimum_memory_bytes: int | None = None
    required_capabilities: frozenset[str] = frozenset()
    preferred_dtype: str | None = None


__all__ = ["ResourceRequirements"]
