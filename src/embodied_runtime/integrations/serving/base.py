"""Common contract for local and external inference providers.

A provider is the Group-3 boundary seen by callers.  It deliberately reuses
the existing asynchronous inference endpoint instead of introducing another
request API.  Framework-specific operations such as OpenPI ``connect`` and
``reset`` remain optional extensions on their concrete providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from embodied_runtime.contracts import DeviceInfo, InferenceRequest, InferenceResult, ModelSpec
from embodied_runtime.distributed.communication.base import AsyncInferenceEndpoint


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Stable facts advertised by one loaded provider."""

    name: str
    runtime: str
    model: ModelSpec
    is_remote: bool
    device: DeviceInfo | None = None
    transport: str | None = None
    features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("provider capability name must not be empty")
        if not self.runtime.strip():
            raise ValueError("provider runtime must not be empty")
        if self.is_remote and not self.transport:
            raise ValueError("a remote provider must declare its transport")
        if not self.is_remote and self.transport is not None:
            raise ValueError("a local provider cannot declare a network transport")


@runtime_checkable
class InferenceProvider(
    AsyncInferenceEndpoint[InferenceRequest, InferenceResult],
    Protocol,
):
    """Minimal common surface implemented by every serving option."""

    @property
    def capabilities(self) -> ProviderCapabilities: ...

    async def aclose(self) -> None: ...


__all__ = ["InferenceProvider", "ProviderCapabilities"]
