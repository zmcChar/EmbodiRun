"""Common interface exposed by local and remote inference providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from embodied_runtime.backends.device import DeviceInfo
from embodied_runtime.models.spec import ModelSpec

from .request import InferenceRequest
from .result import InferenceResult


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
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
class InferenceProvider(Protocol):
    @property
    def capabilities(self) -> ProviderCapabilities: ...

    async def infer_async(self, request: InferenceRequest) -> InferenceResult: ...

    async def aclose(self) -> None: ...


__all__ = ["InferenceProvider", "ProviderCapabilities"]
