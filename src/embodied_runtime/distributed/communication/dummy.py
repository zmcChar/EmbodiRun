"""Controllable in-process link used to exercise remote-runtime behavior."""

from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

from .base import AsyncInferenceEndpoint

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


class RemoteUnavailableError(ConnectionError):
    """The simulated remote link cannot deliver this request or response."""


class DummyLink:
    """A connection switch and latency boundary with monotonically increasing epochs.

    Changing connectivity creates a new epoch. Work that entered the link in an
    older epoch is rejected even if the link has reconnected by the time it
    returns, which prevents an old response from taking control after recovery.
    """

    def __init__(self, *, connected: bool = True, one_way_latency_s: float = 0.0) -> None:
        if one_way_latency_s < 0:
            raise ValueError("one_way_latency_s cannot be negative")
        self._connected = connected
        self._one_way_latency_s = one_way_latency_s
        self._connection_epoch = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def connection_epoch(self) -> int:
        return self._connection_epoch

    @property
    def one_way_latency_s(self) -> float:
        return self._one_way_latency_s

    def set_connected(self, connected: bool) -> None:
        if connected != self._connected:
            self._connected = connected
            self._connection_epoch += 1

    def set_one_way_latency_s(self, latency_s: float) -> None:
        if latency_s < 0:
            raise ValueError("latency_s cannot be negative")
        self._one_way_latency_s = latency_s

    async def cross(self, epoch: int) -> None:
        """Cross one side of the link and validate the originating epoch."""

        self._validate(epoch)
        if self._one_way_latency_s:
            await asyncio.sleep(self._one_way_latency_s)
        self._validate(epoch)

    def _validate(self, epoch: int) -> None:
        if not self._connected:
            raise RemoteUnavailableError("dummy link is disconnected")
        if epoch != self._connection_epoch:
            raise RemoteUnavailableError(
                f"dummy link epoch changed from {epoch} to {self._connection_epoch}"
            )


class DummyRemoteInferenceEndpoint(Generic[RequestT, ResultT]):
    """Wrap an endpoint with a controllable request/response link."""

    def __init__(
        self,
        endpoint: AsyncInferenceEndpoint[RequestT, ResultT],
        link: DummyLink | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.link = link or DummyLink()

    @property
    def connected(self) -> bool:
        return self.link.connected

    @property
    def connection_epoch(self) -> int:
        return self.link.connection_epoch

    async def infer_async(self, request: RequestT) -> ResultT:
        epoch = self.connection_epoch
        await self.link.cross(epoch)
        result = await self.endpoint.infer_async(request)
        await self.link.cross(epoch)
        return result


__all__ = [
    "DummyLink",
    "DummyRemoteInferenceEndpoint",
    "RemoteUnavailableError",
]
