"""Async execution worker for queued, dynamically batched requests."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Any

from embodied_runtime.backends.errors import BackendExecutionError

from ._admission import (
    RequestLifecycle,
    WorkItem,
    cancel_item,
    deadline_exceeded,
)
from ._batching import Batcher, PriorityBatchQueue, Splitter
from ._runner_host import ExecutionRunnerHost
from .config import EngineConfig
from .errors import (
    EnginePayloadError,
    QueueFullError,
    RequestCancelledError,
    RequestDeadlineExceededError,
)
from .handle import RequestHandle
from .metrics import EngineMetrics
from .plan_runner import BatchAborted, PlanRunner
from .request import InferenceRequest
from .result import InferenceResult
from .status import RequestStatus

ResultMetadataFactory = Callable[[int], dict[str, Any]]


class AsyncExecutionWorker:
    """Own one event-loop-bound queue, worker task, and batch delivery path."""

    def __init__(
        self,
        *,
        config: EngineConfig,
        batcher: Batcher | None,
        splitter: Splitter,
        plan_runner: PlanRunner,
        runner_host: ExecutionRunnerHost,
        lifecycle: RequestLifecycle,
        metrics: EngineMetrics,
        result_metadata: ResultMetadataFactory,
    ) -> None:
        self._config = config
        self._batcher = batcher
        self._splitter = splitter
        self._plan_runner = plan_runner
        self._runner_host = runner_host
        self._lifecycle = lifecycle
        self._metrics = metrics
        self._result_metadata = result_metadata
        self._queue: PriorityBatchQueue | None = None
        self._task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    def start(self, loop: asyncio.AbstractEventLoop, *, task_name: str) -> None:
        self._loop = loop
        self._queue = PriorityBatchQueue(
            self._config,
            batching_enabled=self._batcher is not None,
        )
        self._task = loop.create_task(self._run(), name=task_name)

    def submit(self, request: InferenceRequest) -> RequestHandle:
        assert self._queue is not None
        assert self._loop is not None
        if self._lifecycle.contains(request.request_id):
            raise ValueError(f"request_id is already active: {request.request_id}")

        future: asyncio.Future[InferenceResult] = self._loop.create_future()
        item = WorkItem(request=request, future=future)
        try:
            self._queue.put(item)
        except asyncio.QueueFull as error:
            self._metrics.rejected()
            raise QueueFullError(
                f"engine queue is full (capacity={self._config.max_queue_size})"
            ) from error

        self._lifecycle.register(item)
        self._metrics.submitted(queue_depth=self._queue.depth)
        return RequestHandle(
            request.request_id,
            future,
            cancel_callback=lambda: cancel_item(item),
            status_callback=lambda: item.status,
        )

    async def stop(self, *, drain: bool) -> None:
        if not drain:
            self._lifecycle.cancel_all()
        if self._queue is not None:
            await self._queue.join()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._queue = None
        self._loop = None

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            entries = await self._queue.next_batch()
            items = [entry[2] for entry in entries]
            try:
                active = [item for item in items if self._lifecycle.activate_or_complete(item)]
                if active:
                    await self._execute_batch(active)
            finally:
                self._queue.finish(entries)
                for item in items:
                    self._lifecycle.discard(item.request.request_id)
                self._metrics.queue_depth(self._queue.depth)

    async def _execute_batch(self, items: list[WorkItem]) -> None:
        started = time.monotonic()
        queue_times = [max(0.0, started - item.request.created_at_s) for item in items]
        self._metrics.batch_started(len(items), queue_times)
        original_size = len(items)

        try:
            await asyncio.to_thread(self._runner_host.check_memory)
            payloads = [self._runner_host.preprocessed_payload(item) for item in items]
            if original_size == 1:
                payload = payloads[0]
            else:
                assert self._batcher is not None
                payload = self._batcher(payloads)
            output = await self._plan_runner.run_async(items, payload)
            if output is None:
                return
            outputs = list(self._splitter(output, original_size))
            if len(outputs) != original_size:
                raise EnginePayloadError(
                    "result splitter returned "
                    f"{len(outputs)} outputs for batch size {original_size}"
                )
        except BatchAborted:
            return
        except RequestCancelledError:
            self._complete_aborted_batch(items, cancellation=True)
            return
        except RequestDeadlineExceededError:
            self._complete_aborted_batch(items, cancellation=False)
            return
        except Exception as error:  # noqa: BLE001
            for item in items:
                if item.future.done():
                    continue
                item.status = RequestStatus.FAILED
                item.future.set_exception(error)
                self._metrics.failed()
            return

        elapsed = time.monotonic() - started
        succeeded = 0
        for index, item in enumerate(items):
            if item.future.done():
                continue
            # Cancellation/deadline may arrive after finalize but before delivery.
            if not self._lifecycle.still_active(item):
                continue
            item.status = RequestStatus.SUCCEEDED
            item.future.set_result(
                InferenceResult(
                    request_id=item.request.request_id,
                    output=outputs[index],
                    queue_time_s=queue_times[index],
                    execution_time_s=elapsed,
                    metadata=self._result_metadata(original_size),
                )
            )
            succeeded += 1
        if succeeded:
            self._metrics.succeeded(succeeded, elapsed)

    def _complete_aborted_batch(
        self,
        items: list[WorkItem],
        *,
        cancellation: bool,
    ) -> None:
        for item in items:
            if cancellation and item.cancellation.is_set():
                self._lifecycle.complete_cancelled(item)
            elif not cancellation and deadline_exceeded(item.request):
                self._lifecycle.complete_deadline(item)
            elif not item.future.done():
                item.status = RequestStatus.FAILED
                reason = "cancelled" if cancellation else "expired"
                item.future.set_exception(
                    BackendExecutionError(
                        f"backend aborted a batch without identifying a {reason} request"
                    )
                )
                self._metrics.failed()


__all__ = ["AsyncExecutionWorker", "ResultMetadataFactory"]
