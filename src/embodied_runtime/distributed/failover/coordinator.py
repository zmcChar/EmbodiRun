"""Asynchronous edge/cloud inference lifecycle coordination."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import Generic, TypeVar, cast

from ..communication import AsyncInferenceEndpoint, AsyncRemoteInferenceEndpoint
from ._selection import CloudResult, edge_decision, select_available_cloud_result
from .types import FailoverConfig, FailoverDecision, FailoverMode, FallbackReason, ResultFuser

EdgeRequestT = TypeVar("EdgeRequestT")
CloudRequestT = TypeVar("CloudRequestT")
ResultT = TypeVar("ResultT")


class AsyncFailoverCoordinator(Generic[EdgeRequestT, CloudRequestT, ResultT]):
    """Keep edge execution hot while cloud inference proceeds independently.

    The edge endpoint runs for every call and is the real-time availability
    baseline. Cloud work runs in a background task. A fresh cloud result may
    preempt the *output authority* at a later decision point; model execution on
    the edge is never cancelled and a control tick never waits for cloud.

    A result fuser must be synchronous, lightweight, free of I/O, and must not
    mutate either input because one cached cloud result may be reused by
    multiple edge ticks.
    """

    def __init__(
        self,
        *,
        edge: AsyncInferenceEndpoint[EdgeRequestT, ResultT],
        cloud: AsyncRemoteInferenceEndpoint[CloudRequestT, ResultT],
        config: FailoverConfig | None = None,
        fuser: ResultFuser[ResultT] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.edge = edge
        self.cloud = cloud
        self.config = config or FailoverConfig()
        if self.config.mode is FailoverMode.ASYNC_BLEND and fuser is None:
            raise ValueError("async_blend mode requires a result fuser")
        self._fuser = fuser
        self._clock = clock
        self._sequence_id = 0
        self._observed_epoch = cloud.connection_epoch
        self._latest_cloud: CloudResult[ResultT] | None = None
        self._last_cloud_failure: tuple[int, FallbackReason] | None = None
        self._last_cloud_submit_s: float | None = None
        self._active_cloud_submit_s: float | None = None
        self._cloud_task: asyncio.Task[None] | None = None
        self._retired_tasks: set[asyncio.Task[None]] = set()
        self._step_lock = asyncio.Lock()
        self._closed = False

    @property
    def cloud_request_in_flight(self) -> bool:
        return self._cloud_task is not None and not self._cloud_task.done()

    async def infer_async(
        self,
        edge_request: EdgeRequestT,
        *,
        cloud_request: CloudRequestT | None = None,
        cloud_request_factory: Callable[[], CloudRequestT] | None = None,
    ) -> FailoverDecision[ResultT]:
        """Run one edge tick and opportunistically select a cloud result.

        ``cloud_request`` may differ from ``edge_request`` because two model
        adapters can require different preprocessed payloads. A factory defers
        request snapshotting until a cloud submission is actually admitted. If
        both are omitted, the exact edge request is reused.
        """

        if self._closed:
            raise RuntimeError("failover coordinator is closed")
        if cloud_request is not None and cloud_request_factory is not None:
            raise ValueError("provide cloud_request or cloud_request_factory, not both")

        async with self._step_lock:
            self._sequence_id += 1
            sequence_id = self._sequence_id
            self._observe_connection_epoch()
            self._expire_cloud_request()

            if cloud_request_factory is None:
                resolved_request = (
                    cast(CloudRequestT, edge_request) if cloud_request is None else cloud_request
                )

                def resolve_cloud_request() -> CloudRequestT:
                    return resolved_request

                request_factory = resolve_cloud_request
            else:
                request_factory = cloud_request_factory
            self._maybe_submit_cloud(request_factory, sequence_id)

            # This is the only inference awaited by the control tick.
            edge_result = await self.edge.infer_async(edge_request)
            return self._select(edge_result, sequence_id)

    async def wait_for_cloud_idle(self) -> None:
        """Wait for the current cloud request outside the real-time control path."""

        task = self._cloud_task
        if task is not None:
            await asyncio.shield(task)

    async def aclose(self) -> None:
        """Cancel coordinator-owned background work without closing endpoints."""

        if self._closed:
            return
        self._closed = True
        tasks = set(self._retired_tasks)
        if self._cloud_task is not None:
            tasks.add(self._cloud_task)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._cloud_task = None
        self._retired_tasks.clear()

    async def __aenter__(
        self,
    ) -> AsyncFailoverCoordinator[EdgeRequestT, CloudRequestT, ResultT]:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()

    def _observe_connection_epoch(self) -> None:
        epoch = self.cloud.connection_epoch
        if epoch == self._observed_epoch:
            return
        self._observed_epoch = epoch
        self._latest_cloud = None
        self._last_cloud_failure = None
        self._last_cloud_submit_s = None
        self._active_cloud_submit_s = None
        task = self._cloud_task
        if task is not None and not task.done():
            task.cancel()
            self._retire(task)
        self._cloud_task = None

    def _maybe_submit_cloud(
        self,
        request_factory: Callable[[], CloudRequestT],
        sequence_id: int,
    ) -> None:
        if self.config.mode is FailoverMode.EDGE_ONLY or not self.cloud.connected:
            return
        if self._cloud_task is not None and not self._cloud_task.done():
            return

        now = self._clock()
        if (
            self._last_cloud_submit_s is not None
            and now - self._last_cloud_submit_s < self.config.cloud_submit_interval_s
        ):
            return

        epoch = self.cloud.connection_epoch
        try:
            request = request_factory()
        except Exception:  # noqa: BLE001 - any snapshot failure must preserve edge authority
            self._last_cloud_failure = (
                epoch,
                FallbackReason.CLOUD_ERROR,
            )
            return
        self._last_cloud_submit_s = now
        task = asyncio.create_task(
            self._run_cloud(request, sequence_id, epoch, now),
            name=f"cloud-inference-{sequence_id}",
        )
        self._cloud_task = task
        self._active_cloud_submit_s = now

    async def _run_cloud(
        self,
        request: CloudRequestT,
        sequence_id: int,
        epoch: int,
        submitted_at_s: float,
    ) -> None:
        task = asyncio.current_task()
        try:
            result = await asyncio.wait_for(
                self.cloud.infer_async(request),
                timeout=self.config.cloud_request_timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            if self._cloud_task is task and epoch == self.cloud.connection_epoch:
                self._last_cloud_failure = (epoch, FallbackReason.CLOUD_TIMEOUT)
        except Exception:  # noqa: BLE001 - remote Providers expose arbitrary failures
            if self._cloud_task is task and epoch == self.cloud.connection_epoch:
                self._last_cloud_failure = (epoch, FallbackReason.CLOUD_ERROR)
        else:
            if (
                self._cloud_task is task
                and epoch == self.cloud.connection_epoch
                and self.cloud.connected
            ):
                self._latest_cloud = CloudResult(
                    result=result,
                    sequence_id=sequence_id,
                    connection_epoch=epoch,
                    submitted_at_s=submitted_at_s,
                )
                self._last_cloud_failure = None
        finally:
            if self._cloud_task is task:
                self._cloud_task = None
                self._active_cloud_submit_s = None

    def _expire_cloud_request(self) -> None:
        task = self._cloud_task
        submitted_at_s = self._active_cloud_submit_s
        if task is None or task.done() or submitted_at_s is None:
            return
        if self._clock() - submitted_at_s <= self.config.cloud_request_timeout_s:
            return

        task.cancel()
        self._retire(task)
        self._cloud_task = None
        self._active_cloud_submit_s = None
        self._last_cloud_failure = (
            self.cloud.connection_epoch,
            FallbackReason.CLOUD_TIMEOUT,
        )

    def _select(
        self,
        edge_result: ResultT,
        sequence_id: int,
    ) -> FailoverDecision[ResultT]:
        epoch = self.cloud.connection_epoch
        if self.config.mode is FailoverMode.EDGE_ONLY:
            return edge_decision(
                edge_result,
                sequence_id,
                epoch,
                FallbackReason.CLOUD_DISABLED,
            )
        if not self.cloud.connected:
            return edge_decision(
                edge_result,
                sequence_id,
                epoch,
                FallbackReason.CLOUD_DISCONNECTED,
            )

        outcome = select_available_cloud_result(
            edge_result=edge_result,
            sequence_id=sequence_id,
            epoch=epoch,
            cloud_result=self._latest_cloud,
            last_cloud_failure=self._last_cloud_failure,
            cloud_request_in_flight=(
                self.cloud_request_in_flight if self._latest_cloud is None else False
            ),
            config=self.config,
            fuser=self._fuser,
            clock=self._clock,
        )
        if outcome.discard_cloud_result:
            self._latest_cloud = None
        return outcome.decision

    def _retire(self, task: asyncio.Task[None]) -> None:
        self._retired_tasks.add(task)

        def forget(completed: asyncio.Task[None]) -> None:
            self._retired_tasks.discard(completed)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                completed.result()

        task.add_done_callback(forget)
