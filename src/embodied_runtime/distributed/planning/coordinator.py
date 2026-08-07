"""Asynchronous planner request coordination."""

from __future__ import annotations

import asyncio
import math

from embodied_runtime.tasks.planning import PlanEnvelope, PlanRequest

from .endpoint import AsyncPlannerEndpoint
from .manager import PlanManager


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
