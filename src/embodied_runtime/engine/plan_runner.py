"""Execution-plan runner protocol and registry."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from threading import Event
from typing import Protocol, runtime_checkable

from embodied_runtime.contracts import ExecutionPlan, InferenceRequest, TensorTree

from .errors import UnsupportedExecutionPlanError


class BatchAborted(Exception):
    """Internal control-flow signal: no request in a batch remains active."""


@dataclass(slots=True, kw_only=True)
class RunnerRequest:
    """Stable request view shared with plan runners.

    Queue futures, status mutation, and other engine lifecycle state remain
    private to the engine.
    """

    request: InferenceRequest
    cancellation: Event = field(default_factory=Event)


@runtime_checkable
class RunnerHost(Protocol):
    """Narrow engine capability surface available to execution-plan runners."""

    def preprocessed_payload(self, request: RunnerRequest) -> TensorTree: ...

    def ensure_active_sync(self, request: RunnerRequest) -> None: ...

    def has_active(self, requests: Sequence[RunnerRequest]) -> bool: ...

    def submit_sync(
        self,
        entrypoint: str,
        inputs: TensorTree,
        request: RunnerRequest,
        step_index: int | None,
        total_steps: int | None,
    ) -> TensorTree: ...

    async def submit_async(
        self,
        entrypoint: str,
        inputs: TensorTree,
        requests: Sequence[RunnerRequest],
        step_index: int | None,
        total_steps: int | None,
    ) -> TensorTree: ...

    def advance_sync(
        self,
        state: TensorTree,
        update: TensorTree,
        scale: float,
        request: RunnerRequest,
        step_index: int,
        total_steps: int,
    ) -> TensorTree: ...

    async def advance_async(
        self,
        state: TensorTree,
        update: TensorTree,
        scale: float,
        requests: Sequence[RunnerRequest],
        step_index: int,
        total_steps: int,
    ) -> TensorTree: ...

    def check_memory_safe_point_sync(self) -> None: ...

    async def check_memory_safe_point_async(self) -> None: ...


@runtime_checkable
class PlanRunner(Protocol):
    """Execute one plan through engine-owned lifecycle and backend helpers."""

    def run_sync(self, request: RunnerRequest) -> TensorTree: ...

    async def run_async(
        self,
        requests: Sequence[RunnerRequest],
        payload: TensorTree,
    ) -> TensorTree | None: ...


PlanRunnerFactory = Callable[[ExecutionPlan, RunnerHost], PlanRunner]


class PlanRunnerRegistry:
    """Map stable plan ``kind`` values to engine-owned runner factories."""

    def __init__(self) -> None:
        self._factories: dict[str, PlanRunnerFactory] = {}

    def register(
        self,
        kind: str,
        factory: PlanRunnerFactory,
        *,
        replace: bool = False,
    ) -> None:
        if not kind:
            raise ValueError("plan kind must not be empty")
        if kind in self._factories and not replace:
            raise ValueError(f"plan runner is already registered for kind {kind!r}")
        self._factories[kind] = factory

    def create(self, plan: ExecutionPlan, host: RunnerHost) -> PlanRunner:
        try:
            factory = self._factories[plan.kind]
        except KeyError as exc:
            raise UnsupportedExecutionPlanError(
                f"no engine runner is registered for execution plan kind {plan.kind!r}"
            ) from exc
        return factory(plan, host)

    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
