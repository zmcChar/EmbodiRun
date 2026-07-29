"""Non-blocking task-planning coordination for an edge control runtime."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from embodied_runtime.contracts.task import (
    PlanEnvelope,
    PlanFeedback,
    PlanRequest,
    PlanStep,
    PlanStepStatus,
)


@runtime_checkable
class AsyncPlannerEndpoint(Protocol):
    """A task planner whose placement and transport are implementation details."""

    async def plan_async(self, request: PlanRequest) -> PlanEnvelope: ...


class PlanManager:
    """Own active and pending task plans for one isolated robot session.

    Remote results are first validated and staged as ``pending_plan``. The
    control loop must explicitly identify a safe boundary before the pending
    plan can atomically replace the active plan.
    """

    def __init__(
        self,
        session_id: str,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        normalized_session_id = session_id.strip() if isinstance(session_id, str) else ""
        if not normalized_session_id:
            raise ValueError("session_id must not be empty")
        self.session_id = normalized_session_id
        self._clock = clock
        self._pending_plan: PlanEnvelope | None = None
        self._pending_observation_timestamp_s: float | None = None
        self._active_plan: PlanEnvelope | None = None
        self._active_step_index = 0
        self._latest_revision_by_task: dict[str, int] = {}
        self._feedback_history: list[PlanFeedback] = []
        self._lock = RLock()

    @property
    def pending_plan(self) -> PlanEnvelope | None:
        with self._lock:
            return self._pending_plan

    @property
    def active_plan(self) -> PlanEnvelope | None:
        with self._lock:
            return self._active_plan

    @property
    def active_step_index(self) -> int | None:
        with self._lock:
            if self._active_plan is None or self._active_step_index >= len(self._active_plan.steps):
                return None
            return self._active_step_index

    @property
    def current_step(self) -> PlanStep | None:
        with self._lock:
            plan = self._active_plan
            if plan is None or self._active_step_index >= len(plan.steps):
                return None
            return plan.steps[self._active_step_index]

    @property
    def feedback_history(self) -> tuple[PlanFeedback, ...]:
        with self._lock:
            return tuple(self._feedback_history)

    @property
    def plan_complete(self) -> bool:
        with self._lock:
            return self._active_plan is not None and self._active_step_index >= len(
                self._active_plan.steps
            )

    def latest_revision(self, task_id: str) -> int | None:
        with self._lock:
            return self._latest_revision_by_task.get(task_id)

    def offer_plan(
        self,
        plan: PlanEnvelope,
        *,
        request: PlanRequest,
    ) -> bool:
        """Validate a planner response and stage it for safe-boundary activation."""

        if not isinstance(plan, PlanEnvelope):
            raise TypeError("plan must be a PlanEnvelope")
        if not isinstance(request, PlanRequest):
            raise TypeError("request must be a PlanRequest")

        with self._lock:
            self._validate_binding(plan, request)
            now = self._now()
            if plan.expires_at_s <= now:
                raise ValueError("plan is already expired")

            latest_revision = self._latest_revision_by_task.get(plan.task_id, 0)
            if plan.revision <= latest_revision:
                raise ValueError(
                    f"plan revision {plan.revision} is not newer than {latest_revision}"
                )
            if request.active_revision is not None and plan.revision <= request.active_revision:
                raise ValueError("plan revision must be newer than the request's active revision")

            allowed_skills = set(request.goal.allowed_skills)
            if allowed_skills:
                for step in plan.steps:
                    if step.skill is None:
                        raise ValueError(
                            "every plan step must name a skill when the goal uses a closed skill set"
                        )
                    if step.skill not in allowed_skills:
                        raise ValueError(
                            f"plan step {step.step_id!r} uses disallowed skill {step.skill!r}"
                        )

            self._pending_plan = plan
            self._pending_observation_timestamp_s = request.observation_timestamp_s
            self._latest_revision_by_task[plan.task_id] = plan.revision
            return True

    def activate_pending(
        self,
        *,
        safe_boundary: bool,
        current_observation_timestamp_s: float | None = None,
        max_observation_staleness_s: float | None = None,
    ) -> PlanEnvelope | None:
        """Atomically activate a pending plan only at an explicit safe boundary."""

        if not isinstance(safe_boundary, bool):
            raise TypeError("safe_boundary must be a bool")
        current_timestamp, max_staleness = _validate_staleness_check(
            current_observation_timestamp_s,
            max_observation_staleness_s,
        )
        if not safe_boundary:
            return None

        with self._lock:
            pending = self._pending_plan
            if pending is None:
                return None
            if pending.expires_at_s <= self._now():
                self._discard_pending()
                return None
            if current_timestamp is not None:
                source_timestamp = self._pending_observation_timestamp_s
                if source_timestamp is None:
                    raise ValueError(
                        "pending plan has no observation timestamp for staleness validation"
                    )
                staleness_s = current_timestamp - source_timestamp
                if staleness_s < 0:
                    raise ValueError(
                        "current observation timestamp cannot precede the plan observation"
                    )
                assert max_staleness is not None
                if staleness_s > max_staleness:
                    self._discard_pending()
                    return None
            self._active_plan = pending
            self._active_step_index = 0
            self._discard_pending()
            return pending

    def record_feedback(
        self,
        status: PlanStepStatus,
        *,
        message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        timestamp_s: float | None = None,
    ) -> PlanFeedback:
        """Append status for the current step without changing plan position."""

        with self._lock:
            feedback = self._feedback_for_current_step(
                status,
                message=message,
                metadata=metadata,
                timestamp_s=timestamp_s,
            )
            self._feedback_history.append(feedback)
            return feedback

    def feedback(
        self,
        status: PlanStepStatus,
        *,
        message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        timestamp_s: float | None = None,
    ) -> PlanFeedback:
        """Compatibility-friendly spelling for :meth:`record_feedback`."""

        return self.record_feedback(
            status,
            message=message,
            metadata=metadata,
            timestamp_s=timestamp_s,
        )

    def advance(
        self,
        status: PlanStepStatus = PlanStepStatus.SUCCEEDED,
        *,
        message: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        timestamp_s: float | None = None,
    ) -> PlanFeedback:
        """Record a terminal status and advance to the next sequential step."""

        try:
            normalized_status = PlanStepStatus(status)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported plan step status: {status!r}") from error
        if normalized_status not in {
            PlanStepStatus.SUCCEEDED,
            PlanStepStatus.FAILED,
            PlanStepStatus.SKIPPED,
            PlanStepStatus.CANCELLED,
        }:
            raise ValueError("advance requires a terminal plan step status")

        with self._lock:
            feedback = self._feedback_for_current_step(
                normalized_status,
                message=message,
                metadata=metadata,
                timestamp_s=timestamp_s,
            )
            self._feedback_history.append(feedback)
            self._active_step_index += 1
            return feedback

    def _validate_binding(self, plan: PlanEnvelope, request: PlanRequest) -> None:
        if request.goal.session_id != self.session_id:
            raise ValueError("request session does not match PlanManager session")
        if plan.session_id != self.session_id:
            raise ValueError("plan session does not match PlanManager session")
        if plan.request_id != request.request_id:
            raise ValueError("plan request_id does not match the originating request")
        if plan.task_id != request.goal.task_id:
            raise ValueError("plan task_id does not match the request goal")
        if plan.based_on_observation_id != request.observation_id:
            raise ValueError("plan observation lineage does not match the request")

    def _feedback_for_current_step(
        self,
        status: PlanStepStatus,
        *,
        message: str | None,
        metadata: Mapping[str, Any] | None,
        timestamp_s: float | None,
    ) -> PlanFeedback:
        plan = self._active_plan
        if plan is None:
            raise RuntimeError("no active plan")
        if self._active_step_index >= len(plan.steps):
            raise RuntimeError("active plan has no remaining step")
        step = plan.steps[self._active_step_index]
        return PlanFeedback(
            task_id=plan.task_id,
            session_id=plan.session_id,
            plan_id=plan.plan_id,
            revision=plan.revision,
            step_id=step.step_id,
            status=status,
            timestamp_s=self._now() if timestamp_s is None else timestamp_s,
            message=message,
            metadata={} if metadata is None else metadata,
        )

    def _now(self) -> float:
        now = self._clock()
        if not isinstance(now, (int, float)):
            raise TypeError("clock must return a number")
        normalized = float(now)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError("clock must return a finite, non-negative timestamp")
        return normalized

    def _discard_pending(self) -> None:
        self._pending_plan = None
        self._pending_observation_timestamp_s = None


def _validate_staleness_check(
    current_observation_timestamp_s: float | None,
    max_observation_staleness_s: float | None,
) -> tuple[float | None, float | None]:
    if (current_observation_timestamp_s is None) != (max_observation_staleness_s is None):
        raise ValueError(
            "current_observation_timestamp_s and max_observation_staleness_s "
            "must be provided together"
        )
    if current_observation_timestamp_s is None:
        return None, None

    values: list[float] = []
    for name, value in (
        ("current_observation_timestamp_s", current_observation_timestamp_s),
        ("max_observation_staleness_s", max_observation_staleness_s),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        values.append(normalized)
    return values[0], values[1]


class AsyncPlanCoordinator:
    """Submit at most one planner request without blocking the control loop."""

    def __init__(
        self,
        *,
        planner: AsyncPlannerEndpoint,
        manager: PlanManager,
        request_timeout_s: float = 5.0,
    ) -> None:
        if (
            not isinstance(request_timeout_s, (int, float))
            or not math.isfinite(float(request_timeout_s))
            or request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be finite and greater than zero")
        self.planner = planner
        self.manager = manager
        self.request_timeout_s = float(request_timeout_s)
        self._request_task: asyncio.Task[None] | None = None
        self._last_result: PlanEnvelope | None = None
        self._last_error: Exception | None = None
        self._last_request_id: str | None = None
        self._closed = False

    @property
    def request_in_flight(self) -> bool:
        return self._request_task is not None and not self._request_task.done()

    @property
    def last_result(self) -> PlanEnvelope | None:
        return self._last_result

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    @property
    def last_request_id(self) -> str | None:
        return self._last_request_id

    def submit(self, request: PlanRequest) -> bool:
        """Admit a background planner call, returning immediately."""

        if self._closed:
            raise RuntimeError("plan coordinator is closed")
        if not isinstance(request, PlanRequest):
            raise TypeError("request must be a PlanRequest")
        if request.goal.session_id != self.manager.session_id:
            raise ValueError("request session does not match PlanManager session")
        if self.request_in_flight:
            return False

        self._last_result = None
        self._last_error = None
        self._last_request_id = request.request_id
        task = asyncio.create_task(
            self._run_request(request),
            name=f"task-planning-{request.request_id}",
        )
        self._request_task = task
        return True

    def request_plan(self, request: PlanRequest) -> bool:
        """Alias for :meth:`submit` at the domain boundary."""

        return self.submit(request)

    async def wait_for_idle(self) -> PlanEnvelope | None:
        """Wait outside the control path for the admitted planner call."""

        task = self._request_task
        if task is not None:
            await asyncio.shield(task)
        return self._last_result

    async def close(self) -> None:
        """Cancel coordinator-owned work without closing the planner endpoint."""

        if self._closed:
            return
        self._closed = True
        task = self._request_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._request_task = None

    async def aclose(self) -> None:
        await self.close()

    async def __aenter__(self) -> AsyncPlanCoordinator:  # noqa: PYI034
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def _run_request(self, request: PlanRequest) -> None:
        task = asyncio.current_task()
        try:
            plan = await asyncio.wait_for(
                self.planner.plan_async(request),
                timeout=self.request_timeout_s,
            )
            self.manager.offer_plan(plan, request=request)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - planner transports expose arbitrary failures
            self._last_error = error
        else:
            self._last_result = plan
        finally:
            if self._request_task is task:
                self._request_task = None


__all__ = [
    "AsyncPlanCoordinator",
    "AsyncPlannerEndpoint",
    "PlanManager",
]
