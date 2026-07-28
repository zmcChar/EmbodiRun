"""Hardware backend contract consumed by the execution engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .artifact import ArtifactVariant, CompileOptions
from .execution import ExecutionContext
from .model import ModelPackage
from .resource import DeviceInfo, MemoryStats
from .types import TensorTree


@dataclass(frozen=True, slots=True)
class SupportReport:
    supported: bool
    reasons: tuple[str, ...] = ()
    capabilities: frozenset[str] = frozenset()

    @classmethod
    def yes(cls, *capabilities: str) -> "SupportReport":
        return cls(True, capabilities=frozenset(capabilities))

    @classmethod
    def no(cls, *reasons: str) -> "SupportReport":
        return cls(False, reasons=tuple(reasons))


@runtime_checkable
class BackendSession(Protocol):
    """Loaded artifact on one concrete device."""

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
    ) -> TensorTree:
        """Execute the engine-owned state update on the backend device."""
        ...

    def memory_stats(self) -> MemoryStats: ...

    def close(self) -> None: ...


@runtime_checkable
class Backend(Protocol):
    """Group 4 boundary; implementations must remain model-family agnostic."""

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
