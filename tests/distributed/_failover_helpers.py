from __future__ import annotations

import asyncio
from functools import wraps


def async_test(function):
    @wraps(function)
    def wrapper():
        return asyncio.run(function())

    return wrapper


class ImmediateEndpoint:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.calls = 0

    async def infer_async(self, request: str) -> str:
        self.calls += 1
        return f"{self.prefix}:{request}"


class GateEndpoint:
    def __init__(self, prefix: str = "cloud") -> None:
        self.prefix = prefix
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def infer_async(self, request: str) -> str:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return f"{self.prefix}:{request}"


class FailIfCalledEndpoint:
    def __init__(self) -> None:
        self.calls = 0

    async def infer_async(self, request: str) -> str:
        self.calls += 1
        raise AssertionError(f"cloud should not receive {request!r}")
