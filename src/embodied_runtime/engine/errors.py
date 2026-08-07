"""Failures owned by inference scheduling and request execution."""

from __future__ import annotations

from embodied_runtime.errors import EmbodiedRuntimeError


class EngineError(EmbodiedRuntimeError):
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


class RequestCancelledError(EngineError):
    """Execution stopped because the caller cancelled the request."""


class RequestDeadlineExceededError(EngineError):
    """Execution could not finish before the request deadline."""


__all__ = [
    "EngineClosedError",
    "EngineError",
    "EnginePayloadError",
    "MemoryBudgetExceededError",
    "QueueFullError",
    "RequestCancelledError",
    "RequestDeadlineExceededError",
    "UnsupportedExecutionPlanError",
]
