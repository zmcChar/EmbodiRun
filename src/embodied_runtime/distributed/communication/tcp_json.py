"""Small length-prefixed JSON transport for real two-host prototype tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import struct
from collections.abc import Mapping
from typing import Any

_HEADER = struct.Struct("!I")
DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class TcpJsonProtocolError(RuntimeError):
    """A peer sent a malformed or oversized framed JSON message."""


def _reject_nonstandard_json_constant(value: str) -> None:
    raise TcpJsonProtocolError(f"non-standard JSON constant is not allowed: {value}")


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(ConnectionError, OSError):
        await writer.wait_closed()


async def read_json_message(
    reader: asyncio.StreamReader,
    *,
    timeout_s: float | None = None,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> dict[str, Any]:
    """Read one unsigned-length-prefixed UTF-8 JSON object."""

    if max_message_bytes <= 0:
        raise ValueError("max_message_bytes must be greater than zero")

    async def read() -> dict[str, Any]:
        header = await reader.readexactly(_HEADER.size)
        (size,) = _HEADER.unpack(header)
        if size > max_message_bytes:
            raise TcpJsonProtocolError(f"message size {size} exceeds limit {max_message_bytes}")
        encoded = await reader.readexactly(size)
        try:
            value = json.loads(
                encoded,
                parse_constant=_reject_nonstandard_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TcpJsonProtocolError(f"invalid JSON message: {error}") from error
        if not isinstance(value, dict):
            raise TcpJsonProtocolError("top-level JSON message must be an object")
        return value

    return await asyncio.wait_for(read(), timeout=timeout_s)


async def write_json_message(
    writer: asyncio.StreamWriter,
    payload: Mapping[str, Any],
    *,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> int:
    """Write one framed JSON object and return the encoded payload size."""

    if max_message_bytes <= 0:
        raise ValueError("max_message_bytes must be greater than zero")
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > max_message_bytes:
        raise TcpJsonProtocolError(f"message size {len(encoded)} exceeds limit {max_message_bytes}")
    writer.write(_HEADER.pack(len(encoded)))
    writer.write(encoded)
    await writer.drain()
    return len(encoded)


class TcpJsonRequestClient:
    """Open one TCP connection per request and exchange one JSON object."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_s: float = 5.0,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 0 < port < 65536:
            raise ValueError("port must be between 1 and 65535")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be greater than zero")
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.max_message_bytes = max_message_bytes

    async def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        reader: asyncio.StreamReader
        writer: asyncio.StreamWriter
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port),
            timeout=self.timeout_s,
        )
        try:
            await write_json_message(
                writer,
                payload,
                max_message_bytes=self.max_message_bytes,
            )
            return await read_json_message(
                reader,
                timeout_s=self.timeout_s,
                max_message_bytes=self.max_message_bytes,
            )
        finally:
            await _close_writer(writer)


__all__ = [
    "DEFAULT_MAX_MESSAGE_BYTES",
    "TcpJsonProtocolError",
    "TcpJsonRequestClient",
    "read_json_message",
    "write_json_message",
]
