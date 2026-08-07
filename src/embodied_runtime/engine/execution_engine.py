"""Model-neutral orchestration over a plan runner and backend session.

The public engine owns construction, synchronous execution, async lifecycle,
and session closure. Queueing, request state, backend runner capabilities, and
worker-side batch delivery live in focused internal modules.
"""

from __future__ import annotations

import asyncio
import time
from threading import RLock
from typing import Any

from embodied_runtime._compat import StrEnum
from embodied_runtime.backends.interfaces import BackendSession
from embodied_runtime.models.package import ModelPackage
from embodied_runtime.types import TensorTree

from ._admission import RequestLifecycle, coerce_request, raise_if_inactive_sync
from ._batching import Batcher, Splitter
from ._runner_host import ExecutionRunnerHost
from ._tree import split_batch
from ._worker import AsyncExecutionWorker
from .config import EngineConfig
from .errors import (
    EngineClosedError,
    EnginePayloadError,
    RequestCancelledError,
    RequestDeadlineExceededError,
)
from .handle import RequestHandle
from .memory import MemoryBudgetPolicy
from .metrics import EngineMetrics
from .plan_runner import PlanRunnerRegistry, RunnerRequest
from .request import InferenceRequest
from .result import InferenceResult
from .runners import DEFAULT_PLAN_RUNNERS


class EngineState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CLOSED = "closed"


class ExecutionEngine:
    """Run one portable model package on one loaded backend session.

    Higher numeric request priority runs first. ``deadline_s`` on
    :class:`InferenceRequest` is interpreted as a duration relative to the
    request's ``created_at_s``.
    """

    def __init__(
        self,
        package: ModelPackage,
        session: BackendSession,
        config: EngineConfig | None = None,
        *,
        batcher: Batcher | None = None,
        splitter: Splitter | None = None,
        runner_registry: PlanRunnerRegistry | None = None,
    ) -> None:
        if session.package_id != package.package_id:
            raise ValueError(
                "backend session was loaded from a different ModelPackage: "
                f"session={session.package_id!r}, package={package.package_id!r}"
            )
        self.package = package
        self.session = session
        self.config = config or EngineConfig()
        self.batcher = batcher
        self.splitter = splitter or split_batch
        self.runner_registry = runner_registry or DEFAULT_PLAN_RUNNERS
        self.metrics = EngineMetrics()
        self.memory_policy = MemoryBudgetPolicy(
            maximum_reserved_bytes=self.config.memory_budget_bytes,
            minimum_free_bytes=self.config.minimum_free_bytes,
        )

        self._request_lifecycle = RequestLifecycle(self.metrics)
        self._runner_host = ExecutionRunnerHost(
            self.session,
            self.metrics,
            self.memory_policy,
            self._request_lifecycle,
        )
        self._plan_runner = self.runner_registry.create(package.plan, self._runner_host)
        self._worker = AsyncExecutionWorker(
            config=self.config,
            batcher=self.batcher,
            splitter=self.splitter,
            plan_runner=self._plan_runner,
            runner_host=self._runner_host,
            lifecycle=self._request_lifecycle,
            metrics=self.metrics,
            result_metadata=lambda batch_size: self._result_metadata(batch_size=batch_size),
        )

        self._state = EngineState.CREATED
        self._state_lock = RLock()
        self._session_closed = False

    @property
    def state(self) -> EngineState:
        with self._state_lock:
            return self._state

    def infer(
        self,
        request: InferenceRequest | TensorTree,
        **request_options: Any,
    ) -> InferenceResult:
        """Execute one preprocessed request in the caller's thread."""

        with self._state_lock:
            if self._state is EngineState.CLOSED:
                raise EngineClosedError("engine is closed")
            if self._state in (EngineState.RUNNING, EngineState.STOPPING):
                raise RuntimeError(
                    "synchronous infer cannot run while the asynchronous worker is active"
                )

        inference_request = coerce_request(request, request_options)
        self.metrics.submitted(queue_depth=0)
        item = RunnerRequest(request=inference_request)
        try:
            raise_if_inactive_sync(item)
            self._runner_host.check_memory()
        except RequestCancelledError:
            self.metrics.cancelled()
            raise
        except RequestDeadlineExceededError:
            self.metrics.deadline_exceeded()
            raise
        except Exception:
            self.metrics.failed()
            raise

        started = time.monotonic()
        queue_time = max(0.0, started - inference_request.created_at_s)
        self.metrics.batch_started(1, [queue_time])
        try:
            output = self._plan_runner.run_sync(item)
            outputs = list(self.splitter(output, 1))
            if len(outputs) != 1:
                raise EnginePayloadError(
                    f"result splitter returned {len(outputs)} outputs for batch size 1"
                )
        except RequestCancelledError:
            self.metrics.cancelled()
            raise
        except RequestDeadlineExceededError:
            self.metrics.deadline_exceeded()
            raise
        except Exception:
            self.metrics.failed()
            raise

        elapsed = time.monotonic() - started
        self.metrics.succeeded(1, elapsed)
        return InferenceResult(
            request_id=inference_request.request_id,
            output=outputs[0],
            queue_time_s=queue_time,
            execution_time_s=elapsed,
            metadata=self._result_metadata(batch_size=1),
        )

    async def start(self) -> None:
        """Start the bounded priority-queue worker; idempotent."""

        self._start_on_current_loop()

    def submit(
        self, request: InferenceRequest | TensorTree, **request_options: Any
    ) -> RequestHandle:
        """Queue work without waiting and return an awaitable handle."""

        self._start_on_current_loop()
        inference_request = coerce_request(request, request_options)
        return self._worker.submit(inference_request)

    async def infer_async(
        self,
        request: InferenceRequest | TensorTree,
        **request_options: Any,
    ) -> InferenceResult:
        """Queue one request and await its result."""

        return await self.submit(request, **request_options)

    async def stop(self, *, drain: bool = True) -> None:
        """Stop the async worker while keeping the backend session loaded."""

        with self._state_lock:
            if self._state in (EngineState.CREATED, EngineState.STOPPED):
                self._state = EngineState.STOPPED
                return
            if self._state is EngineState.CLOSED:
                return
            self._state = EngineState.STOPPING

        await self._worker.stop(drain=drain)
        with self._state_lock:
            if self._state is not EngineState.CLOSED:
                self._state = EngineState.STOPPED

    def close(self) -> None:
        """Close a stopped engine and its backend session."""

        with self._state_lock:
            if self._state in (EngineState.RUNNING, EngineState.STOPPING):
                raise RuntimeError("await engine.aclose() while the async worker is active")
            if self._state is EngineState.CLOSED:
                return
            self._close_session()
            self._state = EngineState.CLOSED

    async def aclose(self, *, drain: bool = True) -> None:
        await self.stop(drain=drain)
        with self._state_lock:
            self._close_session()
            self._state = EngineState.CLOSED

    async def __aenter__(self) -> ExecutionEngine:  # noqa: PYI034
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose(drain=exc is None)

    def _start_on_current_loop(self) -> None:
        loop = asyncio.get_running_loop()
        with self._state_lock:
            if self._state is EngineState.CLOSED:
                raise EngineClosedError("engine is closed")
            if self._state is EngineState.STOPPING:
                raise EngineClosedError("engine is stopping and not accepting requests")
            if self._state is EngineState.RUNNING:
                if loop is not self._worker.loop:
                    raise RuntimeError("engine is already bound to another event loop")
                return
            self._worker.start(
                loop,
                task_name=f"execution-engine-{self.package.spec.model_id}",
            )
            self._state = EngineState.RUNNING

    def _result_metadata(self, *, batch_size: int) -> dict[str, Any]:
        return {
            "model_id": self.package.spec.model_id,
            "backend": self.session.device.backend,
            "device_id": self.session.device.device_id,
            "batch_size": batch_size,
        }

    def _close_session(self) -> None:
        if not self._session_closed:
            self.session.close()
            self._session_closed = True
