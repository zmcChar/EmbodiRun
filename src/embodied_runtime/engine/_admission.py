"""Request admission, cancellation, deadlines, and active-item tracking."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from embodied_runtime.models.request import RawRequest
from embodied_runtime.types import TensorTree

from .errors import (
    EnginePayloadError,
    RequestCancelledError,
    RequestDeadlineExceededError,
)
from .metrics import EngineMetrics
from .plan_runner import RunnerRequest
from .request import InferenceRequest
from .result import InferenceResult
from .status import RequestStatus


@dataclass(slots=True, kw_only=True)
class WorkItem(RunnerRequest):
    """One admitted async request and its user-visible completion state."""

    future: asyncio.Future[InferenceResult]
    status: RequestStatus = RequestStatus.QUEUED
    started_at_s: float | None = None


class RequestLifecycle:
    """Own active work-item identity, terminal completion, and metrics."""

    def __init__(self, metrics: EngineMetrics) -> None:
        self._metrics = metrics
        self._items: dict[str, WorkItem] = {}

    def contains(self, request_id: str) -> bool:
        return request_id in self._items

    def register(self, item: WorkItem) -> None:
        self._items[item.request.request_id] = item

    def discard(self, request_id: str) -> None:
        self._items.pop(request_id, None)

    def cancel_all(self) -> None:
        for item in tuple(self._items.values()):
            item.cancellation.set()

    def activate_or_complete(self, item: WorkItem) -> bool:
        now = time.monotonic()
        if item.cancellation.is_set():
            self.complete_cancelled(item)
            return False
        if deadline_exceeded(item.request, now):
            self.complete_deadline(item)
            return False
        item.status = RequestStatus.RUNNING
        item.started_at_s = now
        return True

    def active(self, items: list[WorkItem]) -> list[WorkItem]:
        return [item for item in items if self.still_active(item)]

    def still_active(self, item: WorkItem) -> bool:
        if item.future.done():
            return False
        if item.cancellation.is_set():
            self.complete_cancelled(item)
            return False
        if deadline_exceeded(item.request):
            self.complete_deadline(item)
            return False
        return True

    def complete_cancelled(self, item: WorkItem) -> None:
        if item.future.done():
            return
        item.status = RequestStatus.CANCELLED
        item.future.set_exception(
            RequestCancelledError(f"request {item.request.request_id} was cancelled")
        )
        self._metrics.cancelled()

    def complete_deadline(self, item: WorkItem) -> None:
        if item.future.done():
            return
        item.status = RequestStatus.FAILED
        item.future.set_exception(
            RequestDeadlineExceededError(f"request {item.request.request_id} exceeded its deadline")
        )
        self._metrics.deadline_exceeded()


def cancel_item(item: WorkItem) -> bool:
    """Request cooperative cancellation if delivery is not already terminal."""

    if item.future.done():
        return False
    item.cancellation.set()
    return True


def deadline_exceeded(
    request: InferenceRequest,
    now: float | None = None,
) -> bool:
    if request.deadline_s is None:
        return False
    return (time.monotonic() if now is None else now) >= (request.created_at_s + request.deadline_s)


def raise_if_inactive_sync(item: RunnerRequest) -> None:
    if item.cancellation.is_set():
        raise RequestCancelledError(f"request {item.request.request_id} was cancelled")
    if deadline_exceeded(item.request):
        raise RequestDeadlineExceededError(
            f"request {item.request.request_id} exceeded its deadline"
        )


def coerce_request(
    request: InferenceRequest | TensorTree,
    options: dict[str, Any],
) -> InferenceRequest:
    """Normalize the two public request spellings without silently merging them."""

    if isinstance(request, InferenceRequest):
        if options:
            names = ", ".join(sorted(options))
            raise TypeError(f"request options ({names}) cannot accompany an InferenceRequest")
        return request
    return InferenceRequest(payload=request, **options)


def preprocessed_payload(request: InferenceRequest) -> TensorTree:
    """Reject model-domain raw observations at the execution boundary."""

    if isinstance(request.payload, RawRequest):
        raise EnginePayloadError(
            "RawRequest must be converted by a model adapter before entering the execution engine"
        )
    return request.payload


__all__ = [
    "RequestLifecycle",
    "WorkItem",
    "cancel_item",
    "coerce_request",
    "deadline_exceeded",
    "preprocessed_payload",
    "raise_if_inactive_sync",
]
