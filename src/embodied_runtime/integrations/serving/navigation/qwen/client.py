"""Small OpenAI-compatible HTTP client used by the Qwen navigation provider."""

from __future__ import annotations

import base64
import json
import math
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib.parse import urlsplit

from embodied_runtime.contracts.navigation import WaypointPlan

from .schema import (
    SYSTEM_PROMPT,
    WAYPOINT_PLAN_SCHEMA,
    QwenNavigationError,
    QwenNavigationValidationError,
    validate_waypoint_plan,
)

DEFAULT_BASE_URL = "http://127.0.0.1:15003/v1"
DEFAULT_MODEL = "qwen3.5-9b"
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_CONTEXT_JSON_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


class JsonHttpTransport(Protocol):
    """Injectable synchronous transport boundary for tests and alternate HTTP stacks."""

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


def _validate_image(data: bytes, media_type: str) -> None:
    if not isinstance(data, bytes) or not data:
        raise QwenNavigationError("latest RGB frame must be non-empty bytes")
    if len(data) > MAX_IMAGE_BYTES:
        raise QwenNavigationError(f"latest RGB frame exceeds {MAX_IMAGE_BYTES} bytes")
    if media_type not in SUPPORTED_IMAGE_TYPES:
        raise QwenNavigationError(
            f"unsupported RGB media type {media_type!r}; expected JPEG, PNG, or WebP"
        )
    signatures = {
        "image/jpeg": data.startswith(b"\xff\xd8"),
        "image/png": data.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/webp": data.startswith(b"RIFF") and data[8:12] == b"WEBP",
    }
    if not signatures[media_type]:
        raise QwenNavigationError(
            f"latest RGB bytes do not match declared media type {media_type!r}"
        )


def _strict_json_object(content: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            content,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise QwenNavigationValidationError(
            f"Qwen assistant content is not strict JSON: {error}"
        ) from error
    if not isinstance(decoded, dict):
        raise QwenNavigationValidationError("Qwen assistant content must be one JSON object")
    return decoded


class QwenNavigationClient:
    """Synchronous, validation-first client for one Qwen waypoint-plan request.

    The transport is created on the first request.  The provider calls this
    blocking client through ``asyncio.to_thread`` so urllib never blocks the
    event loop.
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
        """Build one strict OpenAI-compatible multimodal chat request."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise QwenNavigationError("navigation prompt must be a non-empty string")
        if len(prompt) > 10_000:
            raise QwenNavigationError("navigation prompt exceeds 10000 characters")
        if not isinstance(context, Mapping):
            raise QwenNavigationError("navigation context must be a mapping")
        _validate_image(image_data, media_type)

        task = {"instruction": prompt, "observation": dict(context)}
        try:
            task_json = json.dumps(
                task,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            raise QwenNavigationError(
                "navigation context must contain only JSON-compatible values"
            ) from error
        if len(task_json.encode("utf-8")) > MAX_CONTEXT_JSON_BYTES:
            raise QwenNavigationError(
                f"navigation context exceeds {MAX_CONTEXT_JSON_BYTES} encoded bytes"
            )

        image_url = f"data:{media_type};base64,{base64.b64encode(image_data).decode('ascii')}"
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_json},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 1024,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "base_link_waypoint_plan",
                    "strict": True,
                    "schema": WAYPOINT_PLAN_SCHEMA,
                },
            },
        }

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
                    _strict_json_object(content),
                    observation_sequence=observation_sequence,
                )
            except QwenNavigationValidationError as error:
                last_error = error
                if attempt:
                    break
                request = {
                    **request,
                    "messages": [
                        *request["messages"],
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                f"上一输出不符合 waypoint schema：{error}。"
                                "只修正并返回一个 JSON 对象。"
                            ),
                        },
                    ],
                }
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


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "JsonHttpTransport",
    "QwenNavigationClient",
    "QwenNavigationTransportError",
    "TransportFactory",
    "UrllibJsonTransport",
]
