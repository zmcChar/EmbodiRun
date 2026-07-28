from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from embodied_runtime.contracts import Envelope

RequestT = TypeVar("RequestT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


@runtime_checkable
class Transport(Protocol):
    def send(self, envelope: Envelope) -> None: ...

    def receive(self, timeout_s: float | None = None) -> Envelope | None: ...

    def close(self) -> None: ...


@runtime_checkable
class AsyncInferenceEndpoint(Protocol[RequestT, ResultT]):
    """A model-serving endpoint without assumptions about its placement.

    ``ExecutionEngine`` satisfies this protocol structurally. Remote clients
    can implement the same method while hiding serialization and transport.
    """

    async def infer_async(self, request: RequestT) -> ResultT: ...


@runtime_checkable
class AsyncRemoteInferenceEndpoint(AsyncInferenceEndpoint[RequestT, ResultT], Protocol):
    """An asynchronous endpoint whose current link generation is observable."""

    @property
    def connected(self) -> bool: ...

    @property
    def connection_epoch(self) -> int: ...
