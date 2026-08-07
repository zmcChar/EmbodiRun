"""Concrete backend and safe-point capabilities exposed to plan runners."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from threading import Event

from embodied_runtime.backends.errors import BackendExecutionError
from embodied_runtime.backends.interfaces import BackendSession
from embodied_runtime.types import TensorTree

from ._admission import (
    RequestLifecycle,
    WorkItem,
    preprocessed_payload,
    raise_if_inactive_sync,
)
from .context import ExecutionContext
from .errors import RequestCancelledError, RequestDeadlineExceededError
from .memory import MemoryBudgetPolicy
from .metrics import EngineMetrics
from .plan_runner import RunnerRequest


class ExecutionRunnerHost:
    """Execute runner operations through one backend session with metrics."""

    def __init__(
        self,
        session: BackendSession,
        metrics: EngineMetrics,
        memory_policy: MemoryBudgetPolicy,
        lifecycle: RequestLifecycle,
    ) -> None:
        self._session = session
        self._metrics = metrics
        self._memory_policy = memory_policy
        self._lifecycle = lifecycle

    def preprocessed_payload(self, request: RunnerRequest) -> TensorTree:
        return preprocessed_payload(request.request)

    def ensure_active_sync(self, request: RunnerRequest) -> None:
        raise_if_inactive_sync(request)

    def has_active(self, requests: Sequence[RunnerRequest]) -> bool:
        return bool(self._lifecycle.active(self._work_items(requests)))

    def submit_sync(
        self,
        entrypoint: str,
        inputs: TensorTree,
        request: RunnerRequest,
        step_index: int | None,
        total_steps: int | None,
    ) -> TensorTree:
        context = ExecutionContext(
            request_ids=(request.request.request_id,),
            stage=entrypoint,
            step_index=step_index,
            total_steps=total_steps,
            cancellation=request.cancellation,
            metadata=request.request.metadata,
        )
        started = time.monotonic()
        try:
            return self._session.submit(entrypoint, inputs, context)
        except (RequestCancelledError, RequestDeadlineExceededError):
            raise
        except Exception as error:
            raise BackendExecutionError(
                f"backend stage '{entrypoint}' failed for request {request.request.request_id}"
            ) from error
        finally:
            self._metrics.stage_finished(entrypoint, time.monotonic() - started)

    async def submit_async(
        self,
        entrypoint: str,
        inputs: TensorTree,
        requests: Sequence[RunnerRequest],
        step_index: int | None,
        total_steps: int | None,
    ) -> TensorTree:
        items = self._work_items(requests)
        cancellation = items[0].cancellation if len(items) == 1 else Event()
        context = ExecutionContext(
            request_ids=tuple(item.request.request_id for item in items),
            stage=entrypoint,
            step_index=step_index,
            total_steps=total_steps,
            cancellation=cancellation,
            metadata={"batch_size": len(items)},
        )
        started = time.monotonic()
        try:
            return await asyncio.to_thread(self._session.submit, entrypoint, inputs, context)
        except (RequestCancelledError, RequestDeadlineExceededError):
            raise
        except Exception as error:
            request_ids = ", ".join(context.request_ids)
            raise BackendExecutionError(
                f"backend stage '{entrypoint}' failed for requests [{request_ids}]"
            ) from error
        finally:
            self._metrics.stage_finished(entrypoint, time.monotonic() - started)

    def advance_sync(
        self,
        state: TensorTree,
        update: TensorTree,
        scale: float,
        request: RunnerRequest,
        step_index: int,
        total_steps: int,
    ) -> TensorTree:
        context = ExecutionContext(
            request_ids=(request.request.request_id,),
            stage="advance_state",
            step_index=step_index,
            total_steps=total_steps,
            cancellation=request.cancellation,
            metadata=request.request.metadata,
        )
        started = time.monotonic()
        try:
            return self._session.add_scaled(state, update, scale, context)
        except (RequestCancelledError, RequestDeadlineExceededError):
            raise
        except Exception as error:
            raise BackendExecutionError(
                f"backend state update failed for request {request.request.request_id}"
            ) from error
        finally:
            self._metrics.stage_finished("advance_state", time.monotonic() - started)

    async def advance_async(
        self,
        state: TensorTree,
        update: TensorTree,
        scale: float,
        requests: Sequence[RunnerRequest],
        step_index: int,
        total_steps: int,
    ) -> TensorTree:
        items = self._work_items(requests)
        cancellation = items[0].cancellation if len(items) == 1 else Event()
        context = ExecutionContext(
            request_ids=tuple(item.request.request_id for item in items),
            stage="advance_state",
            step_index=step_index,
            total_steps=total_steps,
            cancellation=cancellation,
            metadata={"batch_size": len(items)},
        )
        started = time.monotonic()
        try:
            return await asyncio.to_thread(
                self._session.add_scaled,
                state,
                update,
                scale,
                context,
            )
        except (RequestCancelledError, RequestDeadlineExceededError):
            raise
        except Exception as error:
            request_ids = ", ".join(context.request_ids)
            raise BackendExecutionError(
                f"backend state update failed for requests [{request_ids}]"
            ) from error
        finally:
            self._metrics.stage_finished("advance_state", time.monotonic() - started)

    def check_memory(self) -> None:
        self._memory_policy.check(self._session.memory_stats())

    def check_memory_safe_point_sync(self) -> None:
        if (
            self._memory_policy.maximum_reserved_bytes is not None
            or self._memory_policy.minimum_free_bytes > 0
        ):
            self.check_memory()

    async def check_memory_safe_point_async(self) -> None:
        await asyncio.to_thread(self.check_memory_safe_point_sync)

    @staticmethod
    def _work_items(requests: Sequence[RunnerRequest]) -> list[WorkItem]:
        items: list[WorkItem] = []
        for request in requests:
            if not isinstance(request, WorkItem):
                raise TypeError("asynchronous plan execution requires engine queue work items")
            items.append(request)
        return items


__all__ = ["ExecutionRunnerHost"]
