"""Synchronous request orchestration for a Qwen navigation endpoint."""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from embodied_runtime.tasks.navigation import WaypointPlan

from .errors import QwenNavigationError, QwenNavigationValidationError
from .parsing import decode_strict_json_object, validate_waypoint_plan
from .request import build_chat_request, build_correction_request
from .transport import JsonHttpTransport, TransportFactory, UrllibJsonTransport

DEFAULT_BASE_URL = "http://127.0.0.1:15003/v1"
DEFAULT_MODEL = "qwen3.5-9b"


def _validated_base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url must be a non-empty HTTP(S) URL")
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query or fragment")
    return normalized


def _validated_timeout(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("timeout_s must be a positive finite number")
    return float(value)


class QwenNavigationClient:
    """Validation-first client for one metric waypoint-plan request.

    The transport is created lazily. Policies call this blocking client through
    ``asyncio.to_thread`` so HTTP never blocks the event loop.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        transport: JsonHttpTransport | None = None,
        transport_factory: TransportFactory | None = None,
    ) -> None:
        if transport is not None and transport_factory is not None:
            raise ValueError("transport and transport_factory are mutually exclusive")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("api_key must be a non-empty string or None")

        self.base_url = _validated_base_url(base_url)
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_s = _validated_timeout(timeout_s)
        if transport is not None:
            self._transport_factory: TransportFactory = lambda: transport
        else:
            self._transport_factory = transport_factory or UrllibJsonTransport
        self._transport: JsonHttpTransport | None = None
        self._transport_lock = threading.Lock()
        self._closed = False

    @property
    def connected(self) -> bool:
        """Whether the lazy transport has been materialized and remains open."""

        with self._transport_lock:
            return self._transport is not None and not self._closed

    def _get_transport(self) -> JsonHttpTransport:
        with self._transport_lock:
            if self._closed:
                raise RuntimeError("Qwen navigation client is closed")
            if self._transport is None:
                transport = self._transport_factory()
                if not callable(getattr(transport, "request_json", None)):
                    raise TypeError("transport must implement request_json")
                self._transport = transport
            return self._transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_request(
        self,
        prompt: str,
        image_data: bytes,
        media_type: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build one request without materializing the HTTP transport."""

        return build_chat_request(
            model=self.model,
            prompt=prompt,
            image_data=image_data,
            media_type=media_type,
            context=context,
        )

    def _request_json(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._get_transport().request_json(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            payload=payload,
            timeout_s=self.timeout_s,
        )

    @staticmethod
    def _assistant_content(response: Mapping[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise QwenNavigationError("Qwen response is missing assistant content") from error
        if not isinstance(content, str):
            raise QwenNavigationError("Qwen assistant content must be text")
        return content

    def plan(
        self,
        prompt: str,
        image_data: bytes,
        media_type: str,
        context: Mapping[str, Any],
        *,
        observation_sequence: int,
    ) -> WaypointPlan:
        """Request and locally validate one metric base-link waypoint plan."""

        request = self.build_request(prompt, image_data, media_type, context)
        last_error: QwenNavigationValidationError | None = None
        for attempt in range(2):
            response = self._request_json(request)
            if not isinstance(response, Mapping):
                raise QwenNavigationError("Qwen transport response must be an object")
            content = self._assistant_content(response)
            try:
                return validate_waypoint_plan(
                    decode_strict_json_object(content),
                    observation_sequence=observation_sequence,
                )
            except QwenNavigationValidationError as error:
                last_error = error
                if attempt:
                    break
                request = build_correction_request(
                    request,
                    assistant_content=content,
                    error=error,
                )
        assert last_error is not None
        raise QwenNavigationValidationError(
            f"Qwen returned two invalid waypoint plans: {last_error}"
        ) from last_error

    def close(self) -> None:
        """Close a materialized transport without forcing lazy construction."""

        with self._transport_lock:
            if self._closed:
                return
            self._closed = True
            transport, self._transport = self._transport, None
        if transport is not None:
            close = getattr(transport, "close", None)
            if callable(close):
                close()


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MODEL", "QwenNavigationClient"]
