"""Asynchronous OpenPI remote inference endpoint."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult
from embodied_runtime.engine.status import RequestStatus

from .connection import validate_uri
from .errors import OpenPIProtocolError
from .messages import (
    decode_inference_response,
    decode_reset_response,
    inference_payload,
    request_deadline,
)
from .protocols import OpenPIConnectionFactory, OpenPIMessageCodec
from .transport import OpenPITransport


class OpenPiWebSocketEndpoint:
    """An asynchronous, reconnecting OpenPI remote inference endpoint.

    The underlying WebSocket API is synchronous because that is the API used
    by the reference OpenPI client. Blocking connect/send/receive/close calls
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

        self.url = validate_uri(url)
        self.timeout_s = timeout_s
        self.reconnect_attempts = reconnect_attempts
        self.expected_batch_size = expected_batch_size
        self.expected_action_horizon = expected_action_horizon
        self.expected_action_dim = expected_action_dim
        self._session_id = session_id
        self._transport = OpenPITransport(
            self.url,
            timeout_s=timeout_s,
            reconnect_attempts=reconnect_attempts,
            connection_factory=connection_factory,
            codec=codec,
        )

    @property
    def connected(self) -> bool:
        return self._transport.connected

    @property
    def uri(self) -> str:
        """Compatibility alias for code that calls WebSocket URLs URIs."""

        return self.url

    @property
    def connection_epoch(self) -> int:
        return self._transport.connection_epoch

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def server_metadata(self) -> Mapping[str, Any] | None:
        return self._transport.server_metadata

    async def connect(self) -> Mapping[str, Any]:
        """Connect if necessary, consume the metadata handshake, and return it."""

        async with self._transport.io_lock:
            deadline = time.monotonic() + self.timeout_s
            await self._transport.ensure_connected_locked(deadline)
            metadata = self._transport.server_metadata
            assert metadata is not None
            return dict(metadata)

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        """Send one inference request and return a validated action result."""

        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")
        payload, session_id = inference_payload(request, self._session_id)
        deadline = request_deadline(request, self.timeout_s)
        started_at_s = time.monotonic()

        async with self._transport.io_lock:
            try:
                response, epoch = await self._transport.exchange_locked(payload, deadline)
                actions, response_metadata = decode_inference_response(
                    response,
                    expected_batch_size=self.expected_batch_size,
                    expected_action_horizon=self.expected_action_horizon,
                    expected_action_dim=self.expected_action_dim,
                )
            except OpenPIProtocolError:
                await self._transport.invalidate_connection_locked()
                raise

            metadata: dict[str, Any] = {
                "transport": "openpi_websocket",
                "session_id": session_id,
                "connection_epoch": epoch,
                "server_metadata": dict(self._transport.server_metadata or {}),
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

        async with self._transport.io_lock:
            deadline = time.monotonic() + self.timeout_s
            try:
                response, _ = await self._transport.exchange_locked(payload, deadline)
                status = decode_reset_response(response)
            except OpenPIProtocolError:
                await self._transport.invalidate_connection_locked()
                raise
            self._session_id = session_id or uuid.uuid4().hex
            return status

    async def aclose(self) -> None:
        """Close the active connection, if any."""

        async with self._transport.io_lock:
            await self._transport.invalidate_connection_locked()


# Retain the all-caps spelling as a compatibility alias while exposing the
# requested public class spelling used by provider wrappers.
OpenPIWebSocketEndpoint = OpenPiWebSocketEndpoint
