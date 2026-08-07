"""Validation helpers used by the navigation application configuration."""

from __future__ import annotations

import math
from pathlib import Path


def nonempty(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        suffix = " or omitted" if optional else ""
        raise ValueError(f"{name} must be a non-empty string{suffix}")
    return str(value).strip()


def positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def network_port(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def robot_service_url(host: str, port: int) -> str:
    normalized_host = str(nonempty(host, "robot_host"))
    if "://" in normalized_host or any(character in normalized_host for character in "/?#@"):
        raise ValueError("robot_host must be a hostname or IP address, not a URL")
    port = network_port(port, "robot service port")
    authority = (
        f"[{normalized_host}]"
        if ":" in normalized_host and not normalized_host.startswith("[")
        else normalized_host
    )
    return f"http://{authority}:{port}"


__all__ = ["network_port", "nonempty", "positive", "robot_service_url"]
