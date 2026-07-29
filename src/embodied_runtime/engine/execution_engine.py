"""Model-neutral request lifecycle over an execution-plan runner.

The engine never imports a model implementation or concrete backend, and it
never invokes ``ModelPackage.entrypoints`` directly. A runner selected from the
package's :class:`ExecutionPlan` submits every stage through the loaded backend
session while the engine retains admission, batching, cancellation, memory,
metrics, and session lifecycle ownership.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from threading import Event, RLock
from typing import Any

from embodied_runtime._compat import StrEnum
from embodied_runtime.contracts import (
    BackendExecutionError,
    BackendSession,
    ExecutionContext,
    InferenceRequest,
    InferenceResult,
    ModelPackage,
    RawRequest,
    RequestCancelledError,
    RequestDeadlineExceededError,
    RequestStatus,
    TensorTree,
)

from ._tree import split_batch
from .config import EngineConfig
from .errors import EngineClosedError, EnginePayloadError, QueueFullError
from .handle import RequestHandle
from .memory import MemoryBudgetPolicy
from .metrics import EngineMetrics
from .plan_runner import BatchAborted, PlanRunnerRegistry, RunnerRequest
from .runners import DEFAULT_PLAN_RUNNERS

Batcher = Callable[[Sequence[TensorTree]], TensorTree]
Splitter = Callable[[TensorTree, int], Sequence[TensorTree]]
_QueueEntry = tuple[int, int, "_WorkItem"]


class EngineState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CLOSED = "closed"


@dataclass(slots=True, kw_only=True)
class _WorkItem(RunnerRequest):
    future: asyncio.Future[InferenceResult]
    status: RequestStatus = RequestStatus.QUEUED
    started_at_s: float | None = None


class ExecutionEngine:
    """Run one portable model package on one loaded backend session.

    Higher numeric request priority runs first.  ``deadline_s`` on
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
        self._runner_host = _ExecutionRunnerHost(self)
        self._plan_runner = self.runner_registry.create(package.plan, self._runner_host)
        self.metrics = EngineMetrics()
        self.memory_policy = MemoryBudgetPolicy(
            maximum_reserved_bytes=self.config.memory_budget_bytes,
            minimum_free_bytes=self.config.minimum_free_bytes,
        )

        self._state = EngineState.CREATED
        self._state_lock = RLock()
        self._queue: asyncio.PriorityQueue[_QueueEntry] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._sequence = itertools.count()
        self._items: dict[str, _WorkItem] = {}
        self._session_closed = False

    @property
    def state(self) -> EngineState:
        with self._state_lock:
            return self._state

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------
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

        inference_request = self._coerce_request(request, request_options)
        self.metrics.submitted(queue_depth=0)
        item = RunnerRequest(request=inference_request)
        try:
            self._raise_if_inactive_sync(item)
            self._check_memory()
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

    # ------------------------------------------------------------------
    # Asynchronous lifecycle and API
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the bounded priority-queue worker; idempotent."""

        self._start_on_current_loop()

    def submit(
        self, request: InferenceRequest | TensorTree, **request_options: Any
    ) -> RequestHandle:
        """Queue work without waiting and return an awaitable handle.

        The method must be called from a running event loop.  It lazily starts
        the worker, so callers may use either ``await start(); submit(...)`` or
        simply ``submit(...)``.
        """

        self._start_on_current_loop()
        assert self._queue is not None
        assert self._loop is not None

        inference_request = self._coerce_request(request, request_options)
        if inference_request.request_id in self._items:
            raise ValueError(f"request_id is already active: {inference_request.request_id}")

        future: asyncio.Future[InferenceResult] = self._loop.create_future()
        item = _WorkItem(request=inference_request, future=future)
        entry: _QueueEntry = (-inference_request.priority, next(self._sequence), item)
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull as exc:
            self.metrics.rejected()
            raise QueueFullError(
                f"engine queue is full (capacity={self.config.max_queue_size})"
            ) from exc

        self._items[inference_request.request_id] = item
        self.metrics.submitted(queue_depth=self._queue.qsize())
        return RequestHandle(
            inference_request.request_id,
            future,
            cancel_callback=lambda: self._cancel_item(item),
            status_callback=lambda: item.status,
        )

    async def infer_async(
        self,
        request: InferenceRequest | TensorTree,
        **request_options: Any,
    ) -> InferenceResult:
        """Queue one request and await its result."""

        return await self.submit(request, **request_options)

    async def stop(self, *, drain: bool = True) -> None:
        """Stop the async worker while keeping the backend session loaded.

        With ``drain=False`` queued and running requests are cooperatively
        cancelled.  The current backend stage is allowed to return; cancellation
        is observed at the next plan-defined safe point.
        """

        with self._state_lock:
            if self._state in (EngineState.CREATED, EngineState.STOPPED):
                self._state = EngineState.STOPPED
                return
            if self._state is EngineState.CLOSED:
                return
            self._state = EngineState.STOPPING

        if not drain:
            for item in tuple(self._items.values()):
                item.cancellation.set()

        if self._queue is not None:
            await self._queue.join()
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

        self._worker_task = None
        self._queue = None
        self._loop = None
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

    async def __aenter__(self) -> "ExecutionEngine":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose(drain=exc is None)

    # ------------------------------------------------------------------
    # Async queue worker
    # ------------------------------------------------------------------
    def _start_on_current_loop(self) -> None:
        loop = asyncio.get_running_loop()
        with self._state_lock:
            if self._state is EngineState.CLOSED:
                raise EngineClosedError("engine is closed")
            if self._state is EngineState.STOPPING:
                raise EngineClosedError("engine is stopping and not accepting requests")
            if self._state is EngineState.RUNNING:
                if loop is not self._loop:
                    raise RuntimeError("engine is already bound to another event loop")
                return
            self._loop = loop
            self._queue = asyncio.PriorityQueue(maxsize=self.config.max_queue_size)
            self._worker_task = loop.create_task(
                self._worker_loop(), name=f"execution-engine-{self.package.spec.model_id}"
            )
            self._state = EngineState.RUNNING

    async def _worker_loop(self) -> None:
        assert self._queue is not None
        while True:
            first = await self._queue.get()
            entries = await self._collect_batch(first)
            items = [entry[2] for entry in entries]
            try:
                active = [item for item in items if self._activate_or_complete(item)]
                if active:
                    await self._execute_batch(active)
            finally:
                for entry in entries:
                    self._queue.task_done()
                    self._items.pop(entry[2].request.request_id, None)
                self.metrics.queue_depth(self._queue.qsize())

    async def _collect_batch(self, first: _QueueEntry) -> list[_QueueEntry]:
        assert self._queue is not None
        if self.batcher is None or self.config.max_batch_size == 1:
            return [first]

        entries = [first]
        batch_key = self._batch_key(first[2].request)
        wait_s = self.config.max_wait_ms / 1000.0
        deadline = asyncio.get_running_loop().time() + wait_s

        while len(entries) < self.config.max_batch_size:
            try:
                if wait_s == 0:
                    candidate = self._queue.get_nowait()
                else:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    candidate = await asyncio.wait_for(self._queue.get(), remaining)
            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                break

            if self._batch_key(candidate[2].request) != batch_key:
                # This queue item was acquired, so balance unfinished_tasks before
                # placing the exact priority entry back.
                self._queue.task_done()
                self._queue.put_nowait(candidate)
                break
            entries.append(candidate)
        return entries

    def _activate_or_complete(self, item: _WorkItem) -> bool:
        now = time.monotonic()
        if item.cancellation.is_set():
            self._complete_cancelled(item)
            return False
        if self._deadline_exceeded(item.request, now):
            self._complete_deadline(item)
            return False
        item.status = RequestStatus.RUNNING
        item.started_at_s = now
        return True

    async def _execute_batch(self, items: list[_WorkItem]) -> None:
        started = time.monotonic()
        queue_times = [max(0.0, started - item.request.created_at_s) for item in items]
        self.metrics.batch_started(len(items), queue_times)
        original_size = len(items)

        try:
            await asyncio.to_thread(self._check_memory)
            payloads = [self._preprocessed_payload(item.request) for item in items]
            payload = payloads[0] if original_size == 1 else self.batcher(payloads)
            output = await self._plan_runner.run_async(items, payload)
            if output is None:
                return
            outputs = list(self.splitter(output, original_size))
            if len(outputs) != original_size:
                raise EnginePayloadError(
                    "result splitter returned "
                    f"{len(outputs)} outputs for batch size {original_size}"
                )
        except BatchAborted:
            return
        except RequestCancelledError:
            # A cooperative backend may observe the same cancellation Event
            # before the engine regains control at the next step boundary.
            for item in items:
                if item.cancellation.is_set():
                    self._complete_cancelled(item)
                elif not item.future.done():
                    item.status = RequestStatus.FAILED
                    item.future.set_exception(
                        BackendExecutionError(
                            "backend aborted a batch without identifying a cancelled request"
                        )
                    )
                    self.metrics.failed()
            return
        except RequestDeadlineExceededError:
            for item in items:
                if self._deadline_exceeded(item.request):
                    self._complete_deadline(item)
                elif not item.future.done():
                    item.status = RequestStatus.FAILED
                    item.future.set_exception(
                        BackendExecutionError(
                            "backend aborted a batch without identifying an expired request"
                        )
                    )
                    self.metrics.failed()
            return
        except Exception as exc:
            for item in items:
                if item.future.done():
                    continue
                item.status = RequestStatus.FAILED
                item.future.set_exception(exc)
                self.metrics.failed()
            return

        elapsed = time.monotonic() - started
        succeeded = 0
        for index, item in enumerate(items):
            if item.future.done():
                continue
            # Cancellation/deadline may arrive after finalize but before delivery.
            if not self._still_active(item):
                continue
            item.status = RequestStatus.SUCCEEDED
            item.future.set_result(
                InferenceResult(
                    request_id=item.request.request_id,
                    output=outputs[index],
                    queue_time_s=queue_times[index],
                    execution_time_s=elapsed,
                    metadata=self._result_metadata(batch_size=original_size),
                )
            )
            succeeded += 1
        if succeeded:
            self.metrics.succeeded(succeeded, elapsed)

    def _call_stage_sync(
        self,
        entrypoint: str,
        inputs: TensorTree,
        item: RunnerRequest,
        step_index: int | None,
        total_steps: int | None,
    ) -> TensorTree:
        context = ExecutionContext(
            request_ids=(item.request.request_id,),
            stage=entrypoint,
            step_index=step_index,
            total_steps=total_steps,
            cancellation=item.cancellation,
            metadata=item.request.metadata,
        )
        started = time.monotonic()
        try:
            return self.session.submit(entrypoint, inputs, context)
        except (RequestCancelledError, RequestDeadlineExceededError):
            raise
        except Exception as exc:
            raise BackendExecutionError(
                f"backend stage '{entrypoint}' failed for request {item.request.request_id}"
            ) from exc
        finally:
            self.metrics.stage_finished(entrypoint, time.monotonic() - started)

    async def _call_stage_async(
        self,
        entrypoint: str,
        inputs: TensorTree,
        items: list[_WorkItem],
        step_index: int | None,
        total_steps: int | None,
    ) -> TensorTree:
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
            return await asyncio.to_thread(self.session.submit, entrypoint, inputs, context)
        except (RequestCancelledError, RequestDeadlineExceededError):
            raise
        except Exception as exc:
            request_ids = ", ".join(context.request_ids)
            raise BackendExecutionError(
                f"backend stage '{entrypoint}' failed for requests [{request_ids}]"
            ) from exc
        finally:
            self.metrics.stage_finished(entrypoint, time.monotonic() - started)

    def _advance_state_sync(
        self,
        state: TensorTree,
        update: TensorTree,
        dt: float,
        item: RunnerRequest,
        step_index: int,
        total_steps: int,
    ) -> TensorTree:
        context = ExecutionContext(
            request_ids=(item.request.request_id,),
            stage="advance_state",
            step_index=step_index,
            total_steps=total_steps,
            cancellation=item.cancellation,
            metadata=item.request.metadata,
        )
        started = time.monotonic()
        try:
            return self.session.add_scaled(state, update, dt, context)
        except (RequestCancelledError, RequestDeadlineExceededError):
            raise
        except Exception as exc:
            raise BackendExecutionError(
                f"backend state update failed for request {item.request.request_id}"
            ) from exc
        finally:
            self.metrics.stage_finished("advance_state", time.monotonic() - started)

    async def _advance_state_async(
        self,
        state: TensorTree,
        update: TensorTree,
        dt: float,
        items: list[_WorkItem],
        step_index: int,
        total_steps: int,
    ) -> TensorTree:
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
                self.session.add_scaled,
                state,
                update,
                dt,
                context,
            )
        except (RequestCancelledError, RequestDeadlineExceededError):
            raise
        except Exception as exc:
            request_ids = ", ".join(context.request_ids)
            raise BackendExecutionError(
                f"backend state update failed for requests [{request_ids}]"
            ) from exc
        finally:
            self.metrics.stage_finished("advance_state", time.monotonic() - started)

    # ------------------------------------------------------------------
    # Admission, cancellation, and helpers
    # ------------------------------------------------------------------
    def _active_items(self, items: list[_WorkItem]) -> list[_WorkItem]:
        return [item for item in items if self._still_active(item)]

    def _still_active(self, item: _WorkItem) -> bool:
        if item.future.done():
            return False
        if item.cancellation.is_set():
            self._complete_cancelled(item)
            return False
        if self._deadline_exceeded(item.request):
            self._complete_deadline(item)
            return False
        return True

    def _complete_cancelled(self, item: _WorkItem) -> None:
        if item.future.done():
            return
        item.status = RequestStatus.CANCELLED
        item.future.set_exception(
            RequestCancelledError(f"request {item.request.request_id} was cancelled")
        )
        self.metrics.cancelled()

    def _complete_deadline(self, item: _WorkItem) -> None:
        if item.future.done():
            return
        item.status = RequestStatus.FAILED
        item.future.set_exception(
            RequestDeadlineExceededError(f"request {item.request.request_id} exceeded its deadline")
        )
        self.metrics.deadline_exceeded()

    @staticmethod
    def _cancel_item(item: _WorkItem) -> bool:
        if item.future.done():
            return False
        item.cancellation.set()
        return True

    @staticmethod
    def _deadline_exceeded(
        request: InferenceRequest,
        now: float | None = None,
    ) -> bool:
        if request.deadline_s is None:
            return False
        return (time.monotonic() if now is None else now) >= (
            request.created_at_s + request.deadline_s
        )

    @staticmethod
    def _batch_key(request: InferenceRequest) -> tuple[int | None, str | None]:
        """Only requests with identical execution controls may share a batch."""

        seed_isolation = request.request_id if request.seed is not None else None
        return request.num_steps, seed_isolation

    def _raise_if_inactive_sync(self, item: RunnerRequest) -> None:
        if item.cancellation.is_set():
            raise RequestCancelledError(f"request {item.request.request_id} was cancelled")
        if self._deadline_exceeded(item.request):
            raise RequestDeadlineExceededError(
                f"request {item.request.request_id} exceeded its deadline"
            )

    @staticmethod
    def _coerce_request(
        request: InferenceRequest | TensorTree,
        options: dict[str, Any],
    ) -> InferenceRequest:
        if isinstance(request, InferenceRequest):
            if options:
                names = ", ".join(sorted(options))
                raise TypeError(f"request options ({names}) cannot accompany an InferenceRequest")
            return request
        return InferenceRequest(payload=request, **options)

    @staticmethod
    def _preprocessed_payload(request: InferenceRequest) -> TensorTree:
        if isinstance(request.payload, RawRequest):
            raise EnginePayloadError(
                "RawRequest must be converted by a model adapter before entering "
                "the execution engine"
            )
        return request.payload

    def _check_memory(self) -> None:
        self.memory_policy.check(self.session.memory_stats())

    def _check_memory_if_configured(self) -> None:
        if self.config.memory_budget_bytes is not None or self.config.minimum_free_bytes > 0:
            self._check_memory()

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


class _ExecutionRunnerHost:
    """Internal facade implementing the stable capabilities exposed to runners."""

    def __init__(self, engine: ExecutionEngine) -> None:
        self._engine = engine

    def preprocessed_payload(self, request: RunnerRequest) -> TensorTree:
        return self._engine._preprocessed_payload(request.request)

    def ensure_active_sync(self, request: RunnerRequest) -> None:
        self._engine._raise_if_inactive_sync(request)

    def has_active(self, requests: Sequence[RunnerRequest]) -> bool:
        return bool(self._engine._active_items(self._work_items(requests)))

    def submit_sync(
        self,
        entrypoint: str,
        inputs: TensorTree,
        request: RunnerRequest,
        step_index: int | None,
        total_steps: int | None,
    ) -> TensorTree:
        return self._engine._call_stage_sync(
            entrypoint,
            inputs,
            request,
            step_index,
            total_steps,
        )

    async def submit_async(
        self,
        entrypoint: str,
        inputs: TensorTree,
        requests: Sequence[RunnerRequest],
        step_index: int | None,
        total_steps: int | None,
    ) -> TensorTree:
        return await self._engine._call_stage_async(
            entrypoint,
            inputs,
            self._work_items(requests),
            step_index,
            total_steps,
        )

    def advance_sync(
        self,
        state: TensorTree,
        update: TensorTree,
        scale: float,
        request: RunnerRequest,
        step_index: int,
        total_steps: int,
    ) -> TensorTree:
        return self._engine._advance_state_sync(
            state,
            update,
            scale,
            request,
            step_index,
            total_steps,
        )

    async def advance_async(
        self,
        state: TensorTree,
        update: TensorTree,
        scale: float,
        requests: Sequence[RunnerRequest],
        step_index: int,
        total_steps: int,
    ) -> TensorTree:
        return await self._engine._advance_state_async(
            state,
            update,
            scale,
            self._work_items(requests),
            step_index,
            total_steps,
        )

    def check_memory_safe_point_sync(self) -> None:
        self._engine._check_memory_if_configured()

    async def check_memory_safe_point_async(self) -> None:
        await asyncio.to_thread(self._engine._check_memory_if_configured)

    @staticmethod
    def _work_items(requests: Sequence[RunnerRequest]) -> list[_WorkItem]:
        items: list[_WorkItem] = []
        for request in requests:
            if not isinstance(request, _WorkItem):
                raise TypeError("asynchronous plan execution requires engine queue work items")
            items.append(request)
        return items
