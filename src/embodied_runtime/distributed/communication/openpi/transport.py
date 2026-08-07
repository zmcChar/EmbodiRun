"""Serialized OpenPI WebSocket connection and exchange lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping
from typing import Any

from .codec import OpenPIClientCodec, pack_request, unpack_response
from .connection import (
    default_connection_factory,
    remaining_time,
    run_blocking,
    send_and_receive,
)
from .errors import (
    OpenPIDependencyError,
    OpenPIProtocolError,
    OpenPITimeoutError,
    OpenPIUnavailableError,
)
from .protocols import (
    OpenPIConnectionFactory,
    OpenPIMessageCodec,
    OpenPIWebSocketConnection,
)


class OpenPITransport:
    """Own one reconnecting synchronous WebSocket behind an async lock."""

    def __init__(
        self,
        url: str,
        *,
        timeout_s: float,
        reconnect_attempts: int,
        connection_factory: OpenPIConnectionFactory | None,
        codec: OpenPIMessageCodec | None,
    ) -> None:
        self.url = url
        self.timeout_s = timeout_s
        self.reconnect_attempts = reconnect_attempts
        self._connection_factory = connection_factory or default_connection_factory
        self._codec = codec
        self._connection: OpenPIWebSocketConnection | None = None
        self._server_metadata: dict[str, Any] | None = None
        self._connection_epoch = 0
        self.io_lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._connection is not None

    @property
    def connection_epoch(self) -> int:
        return self._connection_epoch

    @property
    def server_metadata(self) -> Mapping[str, Any] | None:
        if self._server_metadata is None:
            return None
        return dict(self._server_metadata)

    def message_codec(self) -> OpenPIMessageCodec:
        if self._codec is None:
            self._codec = OpenPIClientCodec()
        return self._codec

    async def exchange_locked(
        self,
        payload: Mapping[str, Any],
        deadline: float,
    ) -> tuple[Any, int]:
        codec = self.message_codec()
        packed = pack_request(codec, payload)

        last_error: Exception | None = None
        for attempt in range(self.reconnect_attempts + 1):
            try:
                await self.ensure_connected_locked(deadline)
                connection = self._connection
                assert connection is not None
                epoch = self._connection_epoch
                response = await run_blocking(
                    lambda connection=connection: send_and_receive(
                        connection,
                        packed,
                        deadline,
                    ),
                    deadline,
                )
                return unpack_response(codec, response), epoch
            except asyncio.CancelledError:
                await self.invalidate_connection_locked()
                raise
            except (OpenPIDependencyError, OpenPIProtocolError):
                raise
            # WebSocket implementations expose transport failures through
            # version-specific exception classes; keep the optional dependency
            # outside this package's import surface and normalize them here.
            except Exception as error:  # noqa: BLE001
                last_error = error
                await self.invalidate_connection_locked()
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

    async def ensure_connected_locked(self, deadline: float) -> None:
        if self._connection is not None:
            return
        codec = self.message_codec()
        remaining_s = remaining_time(deadline)
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
            connection, metadata = await run_blocking(connect_and_handshake, deadline)
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

    async def invalidate_connection_locked(self) -> None:
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
