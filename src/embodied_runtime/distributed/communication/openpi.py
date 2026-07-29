"""OpenPI-compatible WebSocket endpoint for remote robot-policy serving.

The wire dependencies are deliberately optional.  Importing
``embodied_runtime.distributed.communication`` does not require either
``openpi-client`` or ``websockets``; they are loaded only when the default
codec or connection factory is first used.  Tests and alternate transports can
inject the two small protocols defined below.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import math
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from embodied_runtime.contracts import (
    InferenceRequest,
    InferenceResult,
    RawRequest,
    RequestStatus,
)


class OpenPIError(RuntimeError):
    """Base class for OpenPI endpoint failures."""


class OpenPIDependencyError(OpenPIError, ImportError):
    """An optional OpenPI transport dependency is not installed."""


class OpenPIProtocolError(OpenPIError):
    """The peer sent a response that does not satisfy the OpenPI contract."""


class OpenPIRemoteError(OpenPIProtocolError):
    """The policy server returned an explicit error response."""


class OpenPIUnavailableError(OpenPIError, ConnectionError):
    """The remote endpoint could not complete a connection or exchange."""


class OpenPITimeoutError(OpenPIUnavailableError, TimeoutError):
    """An OpenPI connection or request exceeded its configured deadline."""


@runtime_checkable
class OpenPIWebSocketConnection(Protocol):
    """Minimal synchronous WebSocket surface used by the endpoint."""

    def send(self, message: bytes) -> Any: ...

    def recv(self, timeout: float | None = None) -> bytes | str: ...

    def close(self) -> Any: ...


@runtime_checkable
class OpenPIMessageCodec(Protocol):
    """Numpy-aware codec used for OpenPI frames."""

    def pack(self, value: Any) -> bytes: ...

    def unpack(self, value: bytes | str) -> Any: ...


OpenPIConnectionFactory = Callable[[str, float], OpenPIWebSocketConnection]


class _OpenPIClientCodec:
    """Adapter around the codec shipped by ``openpi-client``."""

    def __init__(self) -> None:
        try:
            from openpi_client import msgpack_numpy
        except ImportError as error:
            raise OpenPIDependencyError(
                "OpenPI WebSocket serialization requires the optional `openpi-client` package"
            ) from error
        self._module = msgpack_numpy
        self._packer = msgpack_numpy.Packer()

    def pack(self, value: Any) -> bytes:
        return self._packer.pack(value)

    def unpack(self, value: bytes | str) -> Any:
        if isinstance(value, str):
            # OpenPI inference and handshake frames are msgpack, while some
            # compatible servers acknowledge reset with a plain text frame.
            return value
        return self._module.unpackb(value)


def _default_connection_factory(
    uri: str,
    timeout_s: float,
) -> OpenPIWebSocketConnection:
    try:
        from websockets.sync.client import connect
    except ImportError as error:
        raise OpenPIDependencyError(
            "OpenPI WebSocket transport requires the optional `websockets` package"
        ) from error

    return connect(uri, **_websocket_connect_options(uri, timeout_s))


def _websocket_connect_options(uri: str, timeout_s: float) -> dict[str, Any]:
    options: dict[str, Any] = {
        "compression": None,
        "max_size": None,
        "open_timeout": timeout_s,
        "close_timeout": timeout_s,
        "ping_interval": 60.0,
        "ping_timeout": 600.0,
    }
    host = urlsplit(uri).hostname
    if host is None:
        return options
    normalized_host = host.rstrip(".").lower()
    loopback = normalized_host == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(normalized_host).is_loopback
        except ValueError:
            pass
    if loopback:
        # websockets 15+ enables automatic proxy discovery by default.  A
        # loopback robot-policy gateway must never leave the local host.
        options["proxy"] = None
    return options


def _load_numpy() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise OpenPIDependencyError(
            "OpenPI action validation requires the optional `numpy` package"
        ) from error
    return numpy


def _validate_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError("uri must be an absolute ws:// or wss:// URL")
    return uri


def _action_signature(array: Any, *, name: str, numpy: Any) -> tuple[int, int, int]:
    if not isinstance(array, numpy.ndarray):
        raise OpenPIProtocolError(
            f"action {name!r} must decode to numpy.ndarray, got {type(array).__name__}"
        )
    if array.dtype != numpy.dtype(numpy.float32):
        raise OpenPIProtocolError(f"action {name!r} must have dtype float32, got {array.dtype}")
    if array.ndim not in {2, 3}:
        raise OpenPIProtocolError(
            f"action {name!r} must have shape [horizon, dim] or "
            f"[batch, horizon, dim], got {array.shape}"
        )
    if any(int(size) <= 0 for size in array.shape):
        raise OpenPIProtocolError(f"action {name!r} has an empty dimension: {array.shape}")
    if not bool(numpy.isfinite(array).all()):
        raise OpenPIProtocolError(f"action {name!r} contains non-finite values")

    if array.ndim == 2:
        horizon, action_dim = (int(size) for size in array.shape)
        return 1, horizon, action_dim
    batch, horizon, action_dim = (int(size) for size in array.shape)
    return batch, horizon, action_dim


def validate_openpi_actions(
    actions: Any,
    *,
    expected_batch_size: int | None = 1,
    expected_action_horizon: int | None = None,
    expected_action_dim: int | None = None,
) -> Any:
    """Validate and return an OpenPI action array or named action mapping.

    Arrays may be unbatched ``[horizon, dim]`` or batched
    ``[batch, horizon, dim]``.  Every array in a multi-head action mapping must
    share the same normalized ``(batch, horizon)``.  ``expected_action_dim``
    denotes the final dimension for a single array and the sum of final
    dimensions for a named mapping.
    """

    numpy = _load_numpy()
    if expected_batch_size is not None and expected_batch_size <= 0:
        raise ValueError("expected_batch_size must be greater than zero or None")
    if expected_action_horizon is not None and expected_action_horizon <= 0:
        raise ValueError("expected_action_horizon must be greater than zero or None")
    if expected_action_dim is not None and expected_action_dim <= 0:
        raise ValueError("expected_action_dim must be greater than zero or None")

    if isinstance(actions, numpy.ndarray):
        action_items = (("actions", actions),)
        validated: Any = actions
    elif isinstance(actions, Mapping):
        if not actions:
            raise OpenPIProtocolError("action mapping must not be empty")
        copied: dict[str, Any] = {}
        for name, array in actions.items():
            if not isinstance(name, str) or not name:
                raise OpenPIProtocolError("action mapping keys must be non-empty strings")
            copied[name] = array
        action_items = tuple(copied.items())
        validated = copied
    else:
        raise OpenPIProtocolError(
            "OpenPI response actions must be a numpy.ndarray or a mapping of arrays"
        )

    common_batch_horizon: tuple[int, int] | None = None
    total_action_dim = 0
    for name, array in action_items:
        batch, horizon, action_dim = _action_signature(array, name=name, numpy=numpy)
        signature = (batch, horizon)
        if common_batch_horizon is None:
            common_batch_horizon = signature
        elif signature != common_batch_horizon:
            raise OpenPIProtocolError(
                "all action heads must share batch and horizon dimensions; "
                f"expected {common_batch_horizon}, action {name!r} has {signature}"
            )
        total_action_dim += action_dim

    assert common_batch_horizon is not None
    batch, horizon = common_batch_horizon
    if expected_batch_size is not None and batch != expected_batch_size:
        raise OpenPIProtocolError(f"expected action batch size {expected_batch_size}, got {batch}")
    if expected_action_horizon is not None and horizon != expected_action_horizon:
        raise OpenPIProtocolError(
            f"expected action horizon {expected_action_horizon}, got {horizon}"
        )
    if expected_action_dim is not None and total_action_dim != expected_action_dim:
        raise OpenPIProtocolError(
            f"expected total action dimension {expected_action_dim}, got {total_action_dim}"
        )
    return validated


class OpenPiWebSocketEndpoint:
    """An asynchronous, reconnecting OpenPI remote inference endpoint.

    The underlying WebSocket API is synchronous because that is the API used
    by the reference OpenPI client.  Blocking connect/send/receive/close calls
    run in worker threads and are serialized by one asynchronous lock, so a
    WebSocket connection is never accessed concurrently.
    """

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float = 10.0,
        session_id: str | None = None,
        reconnect_attempts: int = 1,
        expected_batch_size: int | None = 1,
        expected_action_horizon: int | None = None,
        expected_action_dim: int | None = None,
        connection_factory: OpenPIConnectionFactory | None = None,
        codec: OpenPIMessageCodec | None = None,
    ) -> None:
        if timeout_s <= 0 or not math.isfinite(timeout_s):
            raise ValueError("timeout_s must be a finite value greater than zero")
        if reconnect_attempts < 0:
            raise ValueError("reconnect_attempts cannot be negative")
        if not session_id:
            session_id = uuid.uuid4().hex

        self.url = _validate_uri(url)
        self.timeout_s = timeout_s
        self.reconnect_attempts = reconnect_attempts
        self.expected_batch_size = expected_batch_size
        self.expected_action_horizon = expected_action_horizon
        self.expected_action_dim = expected_action_dim
        self._session_id = session_id
        self._connection_factory = connection_factory or _default_connection_factory
        self._codec = codec
        self._connection: OpenPIWebSocketConnection | None = None
        self._server_metadata: dict[str, Any] | None = None
        self._connection_epoch = 0
        self._io_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._connection is not None

    @property
    def uri(self) -> str:
        """Compatibility alias for code that calls WebSocket URLs URIs."""

        return self.url

    @property
    def connection_epoch(self) -> int:
        return self._connection_epoch

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def server_metadata(self) -> Mapping[str, Any] | None:
        if self._server_metadata is None:
            return None
        return dict(self._server_metadata)

    def _message_codec(self) -> OpenPIMessageCodec:
        if self._codec is None:
            self._codec = _OpenPIClientCodec()
        return self._codec

    async def connect(self) -> Mapping[str, Any]:
        """Connect if necessary, consume the metadata handshake, and return it."""

        async with self._io_lock:
            deadline = time.monotonic() + self.timeout_s
            await self._ensure_connected_locked(deadline)
            assert self._server_metadata is not None
            return dict(self._server_metadata)

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        """Send one inference request and return a validated action result."""

        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        payload, session_id = self._inference_payload(request)
        deadline = self._request_deadline(request)
        started_at_s = time.monotonic()

        async with self._io_lock:
            try:
                response, epoch = await self._exchange_locked(payload, deadline)
                actions, response_metadata = self._decode_inference_response(response)
            except OpenPIProtocolError:
                await self._invalidate_connection_locked()
                raise

            metadata: dict[str, Any] = {
                "transport": "openpi_websocket",
                "session_id": session_id,
                "connection_epoch": epoch,
                "server_metadata": dict(self._server_metadata or {}),
            }
            if response_metadata:
                metadata["response_metadata"] = response_metadata
            return InferenceResult(
                request_id=request.request_id,
                output={"actions": actions},
                status=RequestStatus.SUCCEEDED,
                execution_time_s=time.monotonic() - started_at_s,
                metadata=metadata,
            )

    async def reset(
        self,
        reset_info: Mapping[str, Any] | None = None,
        *,
        session_id: str | None = None,
    ) -> str:
        """Reset server-side policy state and rotate the local session id."""

        payload = dict(reset_info or {})
        payload["endpoint"] = "reset"
        payload.setdefault("session_id", self._session_id)

        async with self._io_lock:
            deadline = time.monotonic() + self.timeout_s
            try:
                response, _ = await self._exchange_locked(payload, deadline)
                status = self._decode_reset_response(response)
            except OpenPIProtocolError:
                await self._invalidate_connection_locked()
                raise
            self._session_id = session_id or uuid.uuid4().hex
            return status

    async def aclose(self) -> None:
        """Close the active connection, if any."""

        async with self._io_lock:
            await self._invalidate_connection_locked()

    def _request_deadline(self, request: InferenceRequest) -> float:
        now = time.monotonic()
        endpoint_deadline = now + self.timeout_s
        if request.deadline_s is None:
            return endpoint_deadline
        if request.deadline_s <= 0 or not math.isfinite(request.deadline_s):
            raise ValueError("request deadline_s must be a finite value greater than zero")
        request_deadline = request.created_at_s + request.deadline_s
        if request_deadline <= now:
            raise OpenPITimeoutError(
                f"OpenPI request {request.request_id!r} deadline has already expired"
            )
        return min(endpoint_deadline, request_deadline)

    def _inference_payload(self, request: InferenceRequest) -> tuple[dict[str, Any], str]:
        request_payload = request.payload
        if isinstance(request_payload, RawRequest):
            payload = dict(request_payload.observation)
            if request_payload.prompt is not None:
                payload.setdefault("prompt", request_payload.prompt)
        elif isinstance(request_payload, Mapping):
            payload = dict(request_payload)
        else:
            raise TypeError("OpenPI request payload must be RawRequest or a mapping observation")

        metadata_session_id = request.metadata.get("session_id")
        if metadata_session_id is not None and not isinstance(metadata_session_id, str):
            raise TypeError("request metadata session_id must be a string")
        payload_session_id = payload.get("session_id")
        if payload_session_id is not None and not isinstance(payload_session_id, str):
            raise TypeError("OpenPI payload session_id must be a string")
        selected_session_id = payload_session_id or metadata_session_id or self._session_id
        if not selected_session_id:
            raise ValueError("OpenPI session_id must not be empty")

        payload["endpoint"] = "infer"
        payload["session_id"] = selected_session_id
        return payload, selected_session_id

    async def _exchange_locked(
        self,
        payload: Mapping[str, Any],
        deadline: float,
    ) -> tuple[Any, int]:
        codec = self._message_codec()
        try:
            packed = codec.pack(dict(payload))
        except OpenPIError:
            raise
        except Exception as error:
            raise OpenPIProtocolError(f"failed to encode OpenPI request: {error}") from error
        if not isinstance(packed, bytes):
            raise OpenPIProtocolError(
                f"OpenPI codec.pack must return bytes, got {type(packed).__name__}"
            )

        last_error: Exception | None = None
        for attempt in range(self.reconnect_attempts + 1):
            try:
                await self._ensure_connected_locked(deadline)
                connection = self._connection
                assert connection is not None
                epoch = self._connection_epoch
                response = await self._run_blocking(
                    lambda connection=connection: self._send_and_receive(
                        connection,
                        packed,
                        deadline,
                    ),
                    deadline,
                )
                try:
                    return codec.unpack(response), epoch
                except OpenPIError:
                    raise
                except Exception as error:
                    raise OpenPIProtocolError(
                        f"failed to decode OpenPI response: {error}"
                    ) from error
            except asyncio.CancelledError:
                await self._invalidate_connection_locked()
                raise
            except (OpenPIDependencyError, OpenPIProtocolError):
                raise
            # WebSocket implementations expose transport failures through
            # version-specific exception classes; keep the optional dependency
            # outside this module's import surface and normalize them here.
            except Exception as error:  # noqa: BLE001
                last_error = error
                await self._invalidate_connection_locked()
                if attempt >= self.reconnect_attempts:
                    break
                if time.monotonic() >= deadline:
                    break

        assert last_error is not None
        if isinstance(last_error, OpenPITimeoutError) or time.monotonic() >= deadline:
            raise OpenPITimeoutError(f"OpenPI exchange with {self.url!r} timed out") from last_error
        raise OpenPIUnavailableError(
            f"OpenPI exchange with {self.url!r} failed after "
            f"{self.reconnect_attempts + 1} attempt(s): {last_error}"
        ) from last_error

    async def _ensure_connected_locked(self, deadline: float) -> None:
        if self._connection is not None:
            return
        codec = self._message_codec()
        remaining_s = self._remaining_time(deadline)
        connection: OpenPIWebSocketConnection | None = None

        def connect_and_handshake() -> tuple[OpenPIWebSocketConnection, Any]:
            nonlocal connection
            connection = self._connection_factory(self.url, remaining_s)
            try:
                frame = connection.recv(timeout=remaining_s)
                return connection, codec.unpack(frame)
            except Exception:
                with contextlib.suppress(Exception):
                    connection.close()
                raise

        try:
            connection, metadata = await self._run_blocking(connect_and_handshake, deadline)
        except (OpenPIDependencyError, OpenPIProtocolError):
            raise
        except OpenPITimeoutError:
            raise
        except Exception as error:
            raise OpenPIUnavailableError(
                f"failed to connect to OpenPI endpoint {self.url!r}: {error}"
            ) from error
        if not isinstance(connection, OpenPIWebSocketConnection):
            with contextlib.suppress(Exception):
                connection.close()
            raise OpenPIProtocolError(
                "OpenPI connection factory returned an incompatible connection object"
            )
        if not isinstance(metadata, Mapping):
            with contextlib.suppress(Exception):
                connection.close()
            raise OpenPIProtocolError(
                f"OpenPI metadata handshake must be a mapping, got {type(metadata).__name__}"
            )

        self._connection = connection
        self._server_metadata = dict(metadata)
        self._connection_epoch += 1

    @staticmethod
    def _send_and_receive(
        connection: OpenPIWebSocketConnection,
        payload: bytes,
        deadline: float,
    ) -> bytes | str:
        connection.send(payload)
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise OpenPITimeoutError("OpenPI request deadline expired before receive")
        return connection.recv(timeout=remaining_s)

    async def _invalidate_connection_locked(self) -> None:
        connection = self._connection
        if connection is None:
            self._server_metadata = None
            return
        self._connection = None
        self._server_metadata = None
        self._connection_epoch += 1
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                asyncio.to_thread(connection.close),
                timeout=min(self.timeout_s, 1.0),
            )

    @staticmethod
    def _remaining_time(deadline: float) -> float:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise OpenPITimeoutError("OpenPI operation deadline expired")
        return remaining_s

    async def _run_blocking(self, function: Callable[[], Any], deadline: float) -> Any:
        remaining_s = self._remaining_time(deadline)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(function),
                timeout=remaining_s,
            )
        except asyncio.TimeoutError as error:
            raise OpenPITimeoutError(
                f"OpenPI operation exceeded {remaining_s:.3f}s timeout"
            ) from error

    def _decode_inference_response(self, response: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(response, Mapping) and (
            response.get("type") == "error" or "error" in response
        ):
            message = response.get("message", response.get("error", response))
            raise OpenPIRemoteError(f"OpenPI inference server error: {message}")

        response_metadata: dict[str, Any] = {}
        if isinstance(response, Mapping) and "actions" in response:
            actions = response["actions"]
            for key, value in response.items():
                if key != "actions":
                    response_metadata[str(key)] = value
        else:
            actions = response

        validated = validate_openpi_actions(
            actions,
            expected_batch_size=self.expected_batch_size,
            expected_action_horizon=self.expected_action_horizon,
            expected_action_dim=self.expected_action_dim,
        )
        return validated, response_metadata

    @staticmethod
    def _decode_reset_response(response: Any) -> str:
        if isinstance(response, Mapping) and (
            response.get("type") == "error" or "error" in response
        ):
            message = response.get("message", response.get("error", response))
            raise OpenPIRemoteError(f"OpenPI reset server error: {message}")
        if isinstance(response, str):
            status = response
        elif isinstance(response, Mapping) and isinstance(response.get("status"), str):
            status = response["status"]
        else:
            raise OpenPIProtocolError(f"unexpected OpenPI reset response: {response!r}")
        if status.strip().lower() not in {"reset successful", "reset", "ok"}:
            raise OpenPIProtocolError(f"OpenPI reset was not acknowledged: {status!r}")
        return status


# Retain the all-caps spelling as a compatibility alias while exposing the
# requested public class spelling used by provider wrappers.
OpenPIWebSocketEndpoint = OpenPiWebSocketEndpoint


__all__ = [
    "OpenPIConnectionFactory",
    "OpenPIDependencyError",
    "OpenPIError",
    "OpenPIMessageCodec",
    "OpenPIProtocolError",
    "OpenPIRemoteError",
    "OpenPITimeoutError",
    "OpenPIUnavailableError",
    "OpenPIWebSocketConnection",
    "OpenPIWebSocketEndpoint",
    "OpenPiWebSocketEndpoint",
    "validate_openpi_actions",
]
