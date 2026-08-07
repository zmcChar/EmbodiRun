"""User-facing handle for an asynchronously queued request."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from typing import Any

from .result import InferenceResult
from .status import RequestStatus


class RequestHandle:
    """Awaitable request handle with cooperative cancellation."""

    def __init__(
        self,
        request_id: str,
        future: asyncio.Future[InferenceResult],
        cancel_callback: Callable[[], bool],
        status_callback: Callable[[], RequestStatus],
    ) -> None:
        self.request_id = request_id
        self._future = future
        self._cancel_callback = cancel_callback
        self._status_callback = status_callback

    def cancel(self) -> bool:
        """Request cancellation at the next engine safe point."""

        if self._future.done():
            return False
        return self._cancel_callback()

    def done(self) -> bool:
        return self._future.done()

    @property
    def status(self) -> RequestStatus:
        return self._status_callback()

    async def result(self) -> InferenceResult:
        return await self._future

    def __await__(self) -> Generator[Any, None, InferenceResult]:
        return self.result().__await__()
