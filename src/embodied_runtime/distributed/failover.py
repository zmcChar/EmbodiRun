"""Asynchronous result-level failover between edge and cloud runtimes."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar, cast

from .communication import AsyncInferenceEndpoint, AsyncRemoteInferenceEndpoint

EdgeRequestT = TypeVar("EdgeRequestT")
CloudRequestT = TypeVar("CloudRequestT")
ResultT = TypeVar("ResultT")
ResultFuser = Callable[[ResultT, ResultT], ResultT]


class FailoverMode(StrEnum):
    """Supported result-selection policies."""

    ASYNC_CLOUD_PREFERRED = "async_cloud_preferred"
    ASYNC_BLEND = "async_blend"
    EDGE_ONLY = "edge_only"


class ResultSource(StrEnum):
    CLOUD = "cloud"
    EDGE = "edge"
    BLENDED = "blended"


class FallbackReason(StrEnum):
    CLOUD_DISABLED = "cloud_disabled"
    CLOUD_DISCONNECTED = "cloud_disconnected"
    CLOUD_PENDING = "cloud_pending"
    CLOUD_TIMEOUT = "cloud_timeout"
    CLOUD_ERROR = "cloud_error"
    BLEND_ERROR = "blend_error"
    NO_CLOUD_RESULT = "no_cloud_result"
    STALE_CLOUD_RESULT = "stale_cloud_result"


@dataclass(frozen=True, slots=True)
class FailoverConfig:
    """Policy knobs for non-blocking cloud-preferred inference.

    A control tick never waits for cloud completion. ``cloud_result_ttl_s`` and
    ``max_cloud_sequence_lag`` bound reuse of the most recent asynchronous cloud
    result. At most one cloud request is active, so a slow or disconnected cloud
    cannot create an unbounded request backlog.
    """

    mode: FailoverMode = FailoverMode.ASYNC_CLOUD_PREFERRED
    cloud_request_timeout_s: float = 5.0
    cloud_result_ttl_s: float = 1.0
    max_cloud_sequence_lag: int = 1
    cloud_submit_interval_s: float = 0.0

    def __post_init__(self) -> None:
        if self.cloud_request_timeout_s <= 0:
            raise ValueError("cloud_request_timeout_s must be greater than zero")
        if self.cloud_result_ttl_s <= 0:
            raise ValueError("cloud_result_ttl_s must be greater than zero")
        if self.max_cloud_sequence_lag < 0:
            raise ValueError("max_cloud_sequence_lag cannot be negative")
        if self.cloud_submit_interval_s < 0:
            raise ValueError("cloud_submit_interval_s cannot be negative")


@dataclass(frozen=True, slots=True)
class FailoverDecision(Generic[ResultT]):
    """One control-tick result plus the reason for its authority source."""

    result: ResultT
    source: ResultSource
    sequence_id: int
    source_sequence_id: int
    connection_epoch: int
    fallback_reason: FallbackReason | None = None
    cloud_result_age_s: float | None = None


@dataclass(frozen=True, slots=True)
class _CloudResult(Generic[ResultT]):
    result: ResultT
    sequence_id: int
    connection_epoch: int
    submitted_at_s: float


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
        self._latest_cloud: _CloudResult[ResultT] | None = None
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
    ) -> FailoverDecision[ResultT]:
        """Run one edge tick and opportunistically select a cloud result.

        ``cloud_request`` may differ from ``edge_request`` because two model
        adapters can require different preprocessed payloads. If omitted, the
        exact edge request is reused for endpoints with a shared request type.
        """

        if self._closed:
            raise RuntimeError("failover coordinator is closed")

        async with self._step_lock:
            self._sequence_id += 1
            sequence_id = self._sequence_id
            self._observe_connection_epoch()
            self._expire_cloud_request()

            if cloud_request is None:
                cloud_request = cast(CloudRequestT, edge_request)
            self._maybe_submit_cloud(cloud_request, sequence_id)

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

    def _maybe_submit_cloud(self, request: CloudRequestT, sequence_id: int) -> None:
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
            result = await self.cloud.infer_async(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            if self._cloud_task is task and epoch == self.cloud.connection_epoch:
                self._last_cloud_failure = (epoch, FallbackReason.CLOUD_ERROR)
        else:
            if (
                self._cloud_task is task
                and epoch == self.cloud.connection_epoch
                and self.cloud.connected
            ):
                self._latest_cloud = _CloudResult(
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
            return self._edge_decision(
                edge_result,
                sequence_id,
                epoch,
                FallbackReason.CLOUD_DISABLED,
            )
        if not self.cloud.connected:
            return self._edge_decision(
                edge_result,
                sequence_id,
                epoch,
                FallbackReason.CLOUD_DISCONNECTED,
            )

        cloud_result = self._latest_cloud
        if cloud_result is None:
            reason = (
                self._last_cloud_failure[1]
                if self._last_cloud_failure is not None and self._last_cloud_failure[0] == epoch
                else FallbackReason.CLOUD_PENDING
                if self.cloud_request_in_flight
                else FallbackReason.NO_CLOUD_RESULT
            )
            return self._edge_decision(edge_result, sequence_id, epoch, reason)

        age_s = self._clock() - cloud_result.submitted_at_s
        sequence_lag = sequence_id - cloud_result.sequence_id
        if (
            cloud_result.connection_epoch != epoch
            or age_s > self.config.cloud_result_ttl_s
            or sequence_lag > self.config.max_cloud_sequence_lag
        ):
            self._latest_cloud = None
            return self._edge_decision(
                edge_result,
                sequence_id,
                epoch,
                FallbackReason.STALE_CLOUD_RESULT,
            )

        if self.config.mode is FailoverMode.ASYNC_BLEND:
            assert self._fuser is not None
            try:
                selected_result = self._fuser(edge_result, cloud_result.result)
                if inspect.isawaitable(selected_result):
                    close = getattr(selected_result, "close", None)
                    if callable(close):
                        close()
                    raise TypeError("result fuser must be synchronous")
            except Exception:
                self._latest_cloud = None
                return self._edge_decision(
                    edge_result,
                    sequence_id,
                    epoch,
                    FallbackReason.BLEND_ERROR,
                )
            selected_source = ResultSource.BLENDED
        else:
            selected_result = cloud_result.result
            selected_source = ResultSource.CLOUD

        return FailoverDecision(
            result=selected_result,
            source=selected_source,
            sequence_id=sequence_id,
            source_sequence_id=cloud_result.sequence_id,
            connection_epoch=epoch,
            cloud_result_age_s=max(0.0, age_s),
        )

    @staticmethod
    def _edge_decision(
        result: ResultT,
        sequence_id: int,
        epoch: int,
        reason: FallbackReason,
    ) -> FailoverDecision[ResultT]:
        return FailoverDecision(
            result=result,
            source=ResultSource.EDGE,
            sequence_id=sequence_id,
            source_sequence_id=sequence_id,
            connection_epoch=epoch,
            fallback_reason=reason,
        )

    def _retire(self, task: asyncio.Task[None]) -> None:
        self._retired_tasks.add(task)

        def forget(completed: asyncio.Task[None]) -> None:
            self._retired_tasks.discard(completed)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                completed.result()

        task.add_done_callback(forget)


__all__ = [
    "AsyncFailoverCoordinator",
    "FailoverConfig",
    "FailoverDecision",
    "FailoverMode",
    "FallbackReason",
    "ResultFuser",
    "ResultSource",
]
