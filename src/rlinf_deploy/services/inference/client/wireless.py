"""Synchronous VVLA policy client over an asynchronous WirelessComm runtime."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..contracts import PolicyObservation, PolicyResult, Session

if TYPE_CHECKING:
    from wireless_comm import Comm, Peer, RuntimeConfig

RPC_SCHEMA = "vvla.policy.rpc.v1"
REQUEST_TAG = 0x56564C41
RESPONSE_TAG = 0x56564C42


class VvlaWirelessError(RuntimeError):
    """A WirelessComm policy request failed locally or remotely."""

    def __init__(
        self, message: str, *, status: int | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class WirelessTransport(Protocol):
    """Blocking RPC boundary used by :class:`VvlaWirelessClient`."""

    def request(
        self,
        method: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


class VvlaWirelessClient:
    """Session-oriented policy client with the same semantics as the HTTP client."""

    def __init__(
        self,
        transport: WirelessTransport,
        *,
        timeout_s: float = 5.0,
        owns_transport: bool = True,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.transport = transport
        self.timeout_s = float(timeout_s)
        self.owns_transport = owns_transport

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        server_node_id: str,
        token: str | None = None,
        timeout_s: float = 5.0,
    ) -> VvlaWirelessClient:
        """Create a client backed by a process-owned WirelessComm runtime."""

        transport = WirelessRpcTransport.from_config(
            config_path,
            server_node_id=server_node_id,
            token=token,
        )
        return cls(transport, timeout_s=timeout_s)

    def health(self) -> dict[str, Any]:
        return self._request("health", None)

    def capabilities(self) -> dict[str, Any]:
        return self._request("capabilities", None)

    def open_session(
        self,
        *,
        robot_id: str,
        action_space: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session:
        result = self._request(
            "open_session",
            {
                "schema": "vvla.policy.session.v1",
                "robot_id": robot_id,
                "action_space": action_space,
                "metadata": dict(metadata or {}),
            },
        )
        return Session(
            session_id=str(result["session_id"]),
            revision=int(result.get("session_revision", 0)),
        )

    def step(self, observation: PolicyObservation) -> PolicyResult:
        result = PolicyResult.from_payload(
            self._request(
                "step",
                {
                    "schema": "vvla.policy.step.v1",
                    "session_id": observation.session_id,
                    "request_id": observation.request_id,
                    "step_id": observation.step_id,
                    "instruction": observation.instruction,
                    "state": dict(observation.state),
                    "reset": observation.reset,
                    "metadata": dict(observation.metadata),
                    "images": [
                        {
                            "name": image.name,
                            "mime_type": image.mime_type,
                            "data": image.data,
                        }
                        for image in observation.images
                    ],
                },
            )
        )
        if result.request_id != observation.request_id:
            raise VvlaWirelessError(
                "step response request_id does not match the request"
            )
        if result.session_id != observation.session_id:
            raise VvlaWirelessError(
                "step response session_id does not match the request"
            )
        if result.step_id != observation.step_id:
            raise VvlaWirelessError("step response step_id does not match the request")
        return result

    def reset(self, session_id: str, *, request_id: str) -> Session:
        result = self._request(
            "reset",
            {"session_id": session_id, "body": {"request_id": request_id}},
        )
        return Session(
            session_id=str(result["session_id"]),
            revision=int(result["session_revision"]),
        )

    def close(self, session_id: str) -> None:
        self._request("close", {"session_id": session_id})

    def shutdown(self) -> None:
        """Release the process-owned WirelessComm runtime."""

        if self.owns_transport:
            self.transport.shutdown()

    def with_timeout(self, timeout_s: float) -> VvlaWirelessClient:
        """Share this connection through a client with a request-local timeout."""

        return VvlaWirelessClient(
            self.transport,
            timeout_s=timeout_s,
            owns_transport=False,
        )

    def _request(
        self,
        method: str,
        payload: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return self.transport.request(method, payload, timeout_s=self.timeout_s)


class WirelessRpcTransport:
    """Run WirelessComm on a private event loop and expose blocking RPC calls."""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        *,
        server_node_id: str,
        token: str | None = None,
    ) -> None:
        if not isinstance(server_node_id, str) or not server_node_id:
            raise ValueError("server_node_id must be non-empty")
        self._runtime_config = runtime_config
        self._server_node_id = server_node_id
        self._token = token
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="vvla-wireless-client",
            daemon=True,
        )
        self._comm: Comm | None = None
        self._server: Peer | None = None
        self._responses: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._failure: BaseException | None = None
        self._closed = False
        self._thread.start()
        try:
            self._submit(self._start()).result()
        except Exception:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop.close()
            raise

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        server_node_id: str,
        token: str | None = None,
    ) -> WirelessRpcTransport:
        from wireless_comm import load_runtime_config

        return cls(
            load_runtime_config(config_path),
            server_node_id=server_node_id,
            token=token,
        )

    def request(
        self,
        method: str,
        payload: Mapping[str, Any] | None,
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        if self._closed:
            raise VvlaWirelessError("wireless transport is closed")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        try:
            return self._submit(self._request(method, payload, timeout_s)).result()
        except VvlaWirelessError:
            raise
        except Exception as error:
            raise VvlaWirelessError(f"wireless request failed: {error}") from error

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._submit(self._shutdown()).result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join()
            self._loop.close()

    async def _start(self) -> None:
        from wireless_comm import Comm

        config = self._runtime_config
        self._comm = await Comm.create(
            local=config.local,
            peers=config.peers,
            bind_host=config.bind_host,
            config=config.comm,
        )
        try:
            self._server = self._comm.peer(self._server_node_id)
        except (KeyError, ValueError):
            await self._comm.close()
            raise
        self._responses = asyncio.create_task(
            self._response_loop(),
            name="vvla-wireless-responses",
        )

    async def _request(
        self,
        method: str,
        payload: Mapping[str, Any] | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        from wireless_comm import CommOptions, OperationTimeoutError

        if self._failure is not None:
            raise VvlaWirelessError(f"wireless response loop failed: {self._failure}")
        deadline = self._loop.time() + timeout_s
        rpc_id = uuid.uuid4().hex
        future = self._loop.create_future()
        self._pending[rpc_id] = future
        metadata = {
            "schema": RPC_SCHEMA,
            "kind": "request",
            "rpc_id": rpc_id,
            "method": method,
        }
        if self._token is not None:
            metadata["token"] = self._token
        try:
            await self._comm.send(
                dict(payload or {}),
                self._server,
                piggypayload=metadata,
                options=CommOptions(tag=REQUEST_TAG, timeout=timeout_s),
            )
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            return await asyncio.wait_for(asyncio.shield(future), remaining)
        except (asyncio.TimeoutError, OperationTimeoutError) as error:
            raise VvlaWirelessError(
                f"{method} timed out after {timeout_s:g} seconds",
                code="timeout",
            ) from error
        finally:
            self._pending.pop(rpc_id, None)

    async def _response_loop(self) -> None:
        from wireless_comm import CommError, CommOptions

        try:
            while True:
                payload, metadata = await self._comm.recv(
                    self._server,
                    CommOptions(tag=RESPONSE_TAG),
                )
                if not isinstance(metadata, Mapping):
                    raise VvlaWirelessError("wireless response is missing RPC metadata")
                if (
                    metadata.get("schema") != RPC_SCHEMA
                    or metadata.get("kind") != "response"
                ):
                    raise VvlaWirelessError(
                        "wireless response has an unsupported RPC envelope"
                    )
                rpc_id = metadata.get("rpc_id")
                if not isinstance(rpc_id, str):
                    raise VvlaWirelessError("wireless response is missing rpc_id")
                future = self._pending.get(rpc_id)
                if future is None:
                    continue
                status = metadata.get("status")
                if isinstance(status, bool) or not isinstance(status, int):
                    future.set_exception(
                        VvlaWirelessError("wireless response status is invalid")
                    )
                elif not 200 <= status < 300:
                    message = (
                        payload.get("message") if isinstance(payload, Mapping) else None
                    )
                    future.set_exception(
                        VvlaWirelessError(
                            str(message or "wireless policy request failed"),
                            status=status,
                            code=str(metadata.get("code") or "remote_error"),
                        )
                    )
                elif not isinstance(payload, Mapping):
                    future.set_exception(
                        VvlaWirelessError("wireless response payload must be an object")
                    )
                else:
                    future.set_result(dict(payload))
        except asyncio.CancelledError:
            raise
        except (CommError, VvlaWirelessError, ConnectionError, OSError) as error:
            self._failure = error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(
                        VvlaWirelessError(f"wireless response loop failed: {error}")
                    )

    async def _shutdown(self) -> None:
        if self._responses is not None:
            self._responses.cancel()
            await asyncio.gather(self._responses, return_exceptions=True)
        if self._comm is not None:
            await self._comm.close()

    def _submit(self, coroutine: Any) -> concurrent.futures.Future[Any]:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


__all__ = [
    "VvlaWirelessClient",
    "VvlaWirelessError",
    "WirelessRpcTransport",
    "WirelessTransport",
]
