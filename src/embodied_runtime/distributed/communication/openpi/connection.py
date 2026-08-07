"""Default WebSocket connection setup and deadline-aware blocking calls."""

from __future__ import annotations

import asyncio
import ipaddress
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from .errors import OpenPIDependencyError, OpenPITimeoutError
from .protocols import OpenPIWebSocketConnection


def default_connection_factory(
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
        # websockets 15+ enables automatic proxy discovery by default. A
        # loopback robot-policy gateway must never leave the local host.
        options["proxy"] = None
    return options


def validate_uri(uri: str) -> str:
    parsed = urlsplit(uri)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        raise ValueError("uri must be an absolute ws:// or wss:// URL")
    return uri


def remaining_time(deadline: float) -> float:
    remaining_s = deadline - time.monotonic()
    if remaining_s <= 0:
        raise OpenPITimeoutError("OpenPI operation deadline expired")
    return remaining_s


async def run_blocking(function: Callable[[], Any], deadline: float) -> Any:
    remaining_s = remaining_time(deadline)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(function),
            timeout=remaining_s,
        )
    except asyncio.TimeoutError as error:
        raise OpenPITimeoutError(f"OpenPI operation exceeded {remaining_s:.3f}s timeout") from error


def send_and_receive(
    connection: OpenPIWebSocketConnection,
    payload: bytes,
    deadline: float,
) -> bytes | str:
    connection.send(payload)
    remaining_s = deadline - time.monotonic()
    if remaining_s <= 0:
        raise OpenPITimeoutError("OpenPI request deadline expired before receive")
    return connection.recv(timeout=remaining_s)
