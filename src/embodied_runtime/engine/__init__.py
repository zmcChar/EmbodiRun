"""Group 3: hardware- and model-neutral single-node execution engine."""

from .config import EngineConfig
from .errors import (
    EngineClosedError,
    EngineError,
    EnginePayloadError,
    MemoryBudgetExceededError,
    QueueFullError,
    UnsupportedExecutionPlanError,
)
from .execution_engine import EngineState, ExecutionEngine
from .handle import RequestHandle
from .memory import MemoryBudgetPolicy
from .metrics import EngineMetrics, MetricsSnapshot
from .plan_runner import PlanRunner, PlanRunnerRegistry, RunnerHost, RunnerRequest

__all__ = [
    "EngineClosedError",
    "EngineConfig",
    "EngineError",
    "EngineMetrics",
    "EnginePayloadError",
    "EngineState",
    "ExecutionEngine",
    "MemoryBudgetExceededError",
    "MemoryBudgetPolicy",
    "MetricsSnapshot",
    "PlanRunner",
    "PlanRunnerRegistry",
    "QueueFullError",
    "RequestHandle",
    "RunnerHost",
    "RunnerRequest",
    "UnsupportedExecutionPlanError",
]
