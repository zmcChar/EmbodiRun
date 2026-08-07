"""Read-only health probe and response validation for a robot cloud."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embodied_runtime.distributed.communication import (
    MULTI_ROBOT_PROTOCOL_VERSION,
    TcpJsonRequestClient,
)


def _required_non_empty_string(response: Mapping[str, Any], key: str) -> str:
    value = response.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"health response requires non-empty {key}")
    return value


def _optional_string(response: Mapping[str, Any], key: str) -> str | None:
    value = response.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"health response {key} must be a string or null")
    return value


def _optional_positive_integer(response: Mapping[str, Any], key: str) -> int | None:
    value = response.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"health response {key} must be a positive integer or null")
    return value


def validate_multi_robot_health_response(
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the read-only cloud ``ping`` response."""

    if response.get("ok") is not True:
        error_type = str(response.get("error_type") or "CloudHealthError")
        error = str(response.get("error") or "cloud reported an unhealthy status")
        raise RuntimeError(f"{error_type}: {error}")
    if response.get("kind") != "pong":
        raise ValueError("health response kind must be 'pong'")
    if response.get("protocol_version") != MULTI_ROBOT_PROTOCOL_VERSION:
        raise ValueError(f"health response protocol_version must be {MULTI_ROBOT_PROTOCOL_VERSION}")

    embodiments = response.get("supported_embodiments")
    if (
        not isinstance(embodiments, list)
        or not embodiments
        or any(not isinstance(value, str) or not value.strip() for value in embodiments)
    ):
        raise ValueError("health response supported_embodiments must be a non-empty string array")
    registered_sessions = response.get("registered_sessions")
    if (
        isinstance(registered_sessions, bool)
        or not isinstance(registered_sessions, int)
        or registered_sessions < 0
    ):
        raise ValueError("health response registered_sessions must be a non-negative integer")

    return {
        "healthy": True,
        "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
        "provider": {
            "name": _required_non_empty_string(response, "provider"),
            "runtime": _required_non_empty_string(response, "provider_runtime"),
            "backend": _optional_string(response, "backend"),
            "device": _optional_string(response, "device"),
        },
        "model": {
            "id": _required_non_empty_string(response, "model_id"),
            "family": _required_non_empty_string(response, "model_family"),
            "revision": _optional_string(response, "model_revision"),
        },
        "action_contract": {
            "action_space_id": _required_non_empty_string(response, "action_space_id"),
            "supported_embodiments": sorted(embodiments),
            "action_dim": _optional_positive_integer(response, "action_dim"),
            "action_horizon": _optional_positive_integer(response, "action_horizon"),
        },
        "registered_sessions": registered_sessions,
    }


async def probe_multi_robot_cloud_health(
    host: str,
    port: int,
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Probe one cloud endpoint without registering a robot or running inference."""

    response = await TcpJsonRequestClient(host, port, timeout_s=timeout_s).request(
        {"kind": "ping", "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION}
    )
    return validate_multi_robot_health_response(response)
