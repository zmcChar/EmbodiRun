"""Bounded HTTP client for the VVLA policy service."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import PolicyObservation, PolicyResult, Session


MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class VvlaHttpError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        maximum_bytes: int,
    ) -> HttpResponse: ...


class UrllibHttpTransport:
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
        maximum_bytes: int,
    ) -> HttpResponse:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                payload = response.read(maximum_bytes + 1)
                result = HttpResponse(
                    status=int(response.status),
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=payload,
                )
        except urllib.error.HTTPError as error:
            detail = error.read(min(maximum_bytes, 16 * 1024))
            error.close()
            raise VvlaHttpError(
                f"{method} {url} returned HTTP {error.code}: "
                f"{detail.decode('utf-8', errors='replace')}"
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise VvlaHttpError(f"{method} {url} failed: {error}") from error
        if len(result.body) > maximum_bytes:
            raise VvlaHttpError(f"{method} {url} response exceeded {maximum_bytes} bytes")
        if not 200 <= result.status < 300:
            raise VvlaHttpError(f"{method} {url} returned HTTP {result.status}")
        return result


class VvlaHttpClient:
    """Session-oriented client; it has no model or framework dependencies."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout_s: float = 5.0,
        transport: HttpTransport | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = float(timeout_s)
        self.transport = transport or UrllibHttpTransport()

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/healthz")

    def capabilities(self) -> dict[str, Any]:
        return self._request_json("GET", "/v1/capabilities")

    def open_session(
        self,
        *,
        robot_id: str,
        action_space: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Session:
        payload = self._request_json(
            "POST",
            "/v1/sessions",
            {
                "schema": "vvla.policy.session.v1",
                "robot_id": robot_id,
                "action_space": action_space,
                "metadata": dict(metadata or {}),
            },
        )
        return Session(
            session_id=str(payload["session_id"]),
            revision=int(payload.get("session_revision", 0)),
        )

    def step(self, observation: PolicyObservation) -> PolicyResult:
        body, content_type = _multipart(observation)
        path = f"/v1/sessions/{urllib.parse.quote(observation.session_id, safe='')}/steps"
        payload = self._request_json_bytes(
            "POST",
            path,
            body,
            content_type=content_type,
            extra_headers={"Idempotency-Key": observation.request_id},
        )
        result = PolicyResult.from_payload(payload)
        if result.request_id != observation.request_id:
            raise VvlaHttpError("step response request_id does not match the request")
        if result.session_id != observation.session_id:
            raise VvlaHttpError("step response session_id does not match the request")
        if result.step_id != observation.step_id:
            raise VvlaHttpError("step response step_id does not match the request")
        return result

    def reset(self, session_id: str, *, request_id: str) -> Session:
        path = f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}/reset"
        payload = self._request_json(
            "POST",
            path,
            {"request_id": request_id},
            extra_headers={"Idempotency-Key": request_id},
        )
        return Session(
            session_id=str(payload["session_id"]),
            revision=int(payload["session_revision"]),
        )

    def close(self, session_id: str) -> None:
        path = f"/v1/sessions/{urllib.parse.quote(session_id, safe='')}"
        self._request_json("DELETE", path)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        body = None
        if payload is not None:
            try:
                body = json.dumps(
                    dict(payload),
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError) as error:
                raise VvlaHttpError("request payload is not valid JSON") from error
        return self._request_json_bytes(
            method,
            path,
            body,
            content_type="application/json" if body is not None else None,
            extra_headers=extra_headers,
        )

    def _request_json_bytes(
        self,
        method: str,
        path: str,
        body: bytes | None,
        *,
        content_type: str | None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json", **dict(extra_headers or {})}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = self.transport.request(
            method,
            self.base_url + path,
            headers=headers,
            body=body,
            timeout_s=self.timeout_s,
            maximum_bytes=MAX_RESPONSE_BYTES,
        )
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VvlaHttpError(f"{method} {path} returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise VvlaHttpError(f"{method} {path} must return a JSON object")
        return payload


def _multipart(observation: PolicyObservation) -> tuple[bytes, str]:
    boundary = f"rlinf-{uuid.uuid4().hex}"
    image_metadata = [
        {"name": image.name, "mime_type": image.mime_type}
        for image in observation.images
    ]
    metadata = json.dumps(
        {
            "schema": "vvla.policy.step.v1",
            "session_id": observation.session_id,
            "request_id": observation.request_id,
            "step_id": observation.step_id,
            "instruction": observation.instruction,
            "state": dict(observation.state),
            "reset": observation.reset,
            "metadata": dict(observation.metadata),
            "images": image_metadata,
        },
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    parts = [
        _part(boundary, "metadata", metadata, "application/json", "metadata.json")
    ]
    parts.extend(
        _part(boundary, f"image_{index}", image.data, image.mime_type, image.name)
        for index, image in enumerate(observation.images)
    )
    parts.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _part(boundary: str, name: str, body: bytes, mime: str, filename: str) -> bytes:
    safe_filename = filename.replace('"', "_")
    return b"".join(
        (
            f"--{boundary}\r\n".encode("ascii"),
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{safe_filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime}\r\n\r\n".encode("ascii"),
            body,
            b"\r\n",
        )
    )


__all__ = [
    "HttpResponse",
    "HttpTransport",
    "UrllibHttpTransport",
    "VvlaHttpClient",
    "VvlaHttpError",
]
