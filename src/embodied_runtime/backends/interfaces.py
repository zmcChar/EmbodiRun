"""Interfaces implemented by model-agnostic hardware backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from embodied_runtime.models.package import ModelPackage
from embodied_runtime.types import TensorTree

from .artifact import ArtifactVariant
from .compile import CompileOptions
from .device import DeviceInfo
from .memory import MemoryStats
from .support import SupportReport

if TYPE_CHECKING:
    from embodied_runtime.engine.context import ExecutionContext


@runtime_checkable
class BackendSession(Protocol):
    @property
    def package_id(self) -> str: ...

    @property
    def device(self) -> DeviceInfo: ...

    def submit(
        self,
        entrypoint: str,
        inputs: TensorTree,
        context: ExecutionContext,
    ) -> TensorTree: ...

    def add_scaled(
        self,
        state: TensorTree,
        update: TensorTree,
        scale: float,
        context: ExecutionContext,
    ) -> TensorTree: ...

    def memory_stats(self) -> MemoryStats: ...

    def close(self) -> None: ...


@runtime_checkable
class Backend(Protocol):
    name: str

    def probe(self) -> tuple[DeviceInfo, ...]: ...

    def supports(self, package: ModelPackage, device: DeviceInfo) -> SupportReport: ...

    def compile(
        self,
        package: ModelPackage,
        device: DeviceInfo,
        options: CompileOptions,
    ) -> ArtifactVariant: ...

    def load(self, artifact: ArtifactVariant) -> BackendSession: ...


__all__ = ["Backend", "BackendSession"]
