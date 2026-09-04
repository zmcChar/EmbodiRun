"""Create an inference client from transport-neutral service routing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .http import VvlaHttpClient
from .wireless import VvlaWirelessClient


def build_inference_client(
    transport: str,
    endpoint: str,
    options: Mapping[str, Any],
    *,
    timeout_s: float,
) -> Any:
    """Build one HTTP or WirelessComm client without depending on its caller."""

    token = options.get("token")
    if token is not None and not isinstance(token, str):
        raise ValueError("inference token must be a string")
    if transport == "http":
        return VvlaHttpClient(endpoint, token=token, timeout_s=timeout_s)
    if transport != "wireless":
        raise ValueError(f"unsupported inference transport {transport!r}")
    return VvlaWirelessClient.from_config(
        _string(options, "comm_config"),
        server_node_id=_string(options, "server_node_id"),
        token=token,
        timeout_s=timeout_s,
    )


def _string(options: Mapping[str, Any], name: str) -> str:
    value = options.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"inference option {name!r} must be non-empty")
    return value


__all__ = ["build_inference_client"]
