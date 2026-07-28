"""Built-in execution-plan runners."""

from __future__ import annotations

from embodied_runtime.contracts import IterativeFlowPlan, SingleForwardPlan

from ..plan_runner import PlanRunnerRegistry
from .iterative_flow import IterativeFlowRunner
from .single_forward import SingleForwardRunner

DEFAULT_PLAN_RUNNERS = PlanRunnerRegistry()
DEFAULT_PLAN_RUNNERS.register(IterativeFlowPlan().kind, IterativeFlowRunner)
DEFAULT_PLAN_RUNNERS.register(SingleForwardPlan().kind, SingleForwardRunner)

__all__ = [
    "DEFAULT_PLAN_RUNNERS",
    "IterativeFlowRunner",
    "SingleForwardRunner",
]
