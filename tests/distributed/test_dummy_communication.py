from __future__ import annotations

import asyncio
from functools import wraps

import pytest

from embodied_runtime.distributed.communication import (
    DummyLink,
    DummyRemoteInferenceEndpoint,
    RemoteUnavailableError,
)


def async_test(function):
    @wraps(function)
    def wrapper():
        return asyncio.run(function())

    return wrapper


class _ImmediateEndpoint:
    async def infer_async(self, request: str) -> str:
        return f"remote:{request}"


class _GateEndpoint:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def infer_async(self, request: str) -> str:
        self.started.set()
        await self.release.wait()
        return f"remote:{request}"


@async_test
async def test_disconnected_dummy_link_fails_without_calling_remote() -> None:
    link = DummyLink(connected=False)
    endpoint = DummyRemoteInferenceEndpoint(_ImmediateEndpoint(), link)

    with pytest.raises(RemoteUnavailableError, match="disconnected"):
        await endpoint.infer_async("request")


@async_test
async def test_response_from_an_old_connection_epoch_is_rejected() -> None:
    remote = _GateEndpoint()
    link = DummyLink()
    endpoint = DummyRemoteInferenceEndpoint(remote, link)

    request = asyncio.create_task(endpoint.infer_async("old"))
    await asyncio.wait_for(remote.started.wait(), timeout=1.0)
    link.set_connected(False)
    link.set_connected(True)
    remote.release.set()

    with pytest.raises(RemoteUnavailableError, match="epoch changed"):
        await request
