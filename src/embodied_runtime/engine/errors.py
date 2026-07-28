"""Engine-owned errors that do not leak a concrete backend or model."""

from __future__ import annotations

from embodied_runtime.contracts import RuntimePrototypeError


class EngineError(RuntimePrototypeError):
    """Base class for execution-engine failures."""


class EngineClosedError(EngineError):
    """The engine no longer accepts work."""


class QueueFullError(EngineError):
    """Admission was rejected because the bounded queue is full."""


class MemoryBudgetExceededError(EngineError):
    """Current device memory use violates the configured admission policy."""


class EnginePayloadError(EngineError):
    """The request payload cannot be consumed without a model adapter."""


class UnsupportedExecutionPlanError(EngineError):
    """No engine runner is registered for a model package's execution plan."""
