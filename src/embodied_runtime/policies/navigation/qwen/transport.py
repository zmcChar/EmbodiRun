"""Synchronous JSON-over-HTTP transport for Qwen navigation serving."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .errors import QwenNavigationError

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class JsonHttpTransport(Protocol):
    """Injectable transport boundary for tests and alternate HTTP stacks."""

    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """Send one request and return one decoded JSON object."""

    def close(self) -> None:
        """Release transport resources, if any."""


TransportFactory = Callable[[], JsonHttpTransport]


class QwenNavigationTransportError(QwenNavigationError):
    """The OpenAI-compatible endpoint could not complete a request."""


class UrllibJsonTransport:
    """Dependency-free JSON transport; construction performs no network I/O."""

    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
        timeout_s: float,
    ) -> Mapping[str, Any]:
        try:
            body = (
                None
                if payload is None
                else json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        except (TypeError, ValueError) as error:
            raise QwenNavigationTransportError(
                "Qwen HTTP payload must be JSON-compatible"
            ) from error

        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise QwenNavigationTransportError(f"Qwen HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise QwenNavigationTransportError(f"cannot reach Qwen endpoint: {error}") from error

        if len(response_body) > MAX_RESPONSE_BYTES:
            raise QwenNavigationTransportError(f"Qwen response exceeds {MAX_RESPONSE_BYTES} bytes")
        try:
            decoded = json.loads(response_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QwenNavigationTransportError("Qwen endpoint returned invalid JSON") from error
        if not isinstance(decoded, Mapping):
            raise QwenNavigationTransportError("Qwen endpoint returned a non-object response")
        return decoded

    def close(self) -> None:
        """urllib opens a fresh response context per request."""


__all__ = [
    "JsonHttpTransport",
    "QwenNavigationTransportError",
    "TransportFactory",
    "UrllibJsonTransport",
]
