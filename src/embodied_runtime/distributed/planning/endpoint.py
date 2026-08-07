"""Planner transport boundary."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from embodied_runtime.tasks.planning import PlanEnvelope, PlanRequest


@runtime_checkable
class AsyncPlannerEndpoint(Protocol):
    """A task planner whose placement and transport are implementation details."""

    async def plan_async(self, request: PlanRequest) -> PlanEnvelope: ...
