"""Shared fail-closed OpenPI client helpers for navigation policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_VLLM_OMNI_URL = "ws://127.0.0.1:8000/v1/realtime/robot/openpi"
NAVIGATION_PROTOCOL_NAME = "embodied-runtime.navigation.openpi"
NAVIGATION_PROTOCOL_VERSION = 1
MAX_OPENPI_IMAGE_PAYLOAD_BYTES = 60 * 1024 * 1024
SUPPORTED_IMPLEMENTATIONS = frozenset({"transformers_wrapper", "vllm_native", "vllm_hybrid"})


def navigation_handshake_metadata(
    model_family: str,
    *,
    implementation: str,
) -> dict[str, Any]:
    """Build the exact metadata frame expected from a matching model server."""

    if not isinstance(implementation, str) or implementation not in SUPPORTED_IMPLEMENTATIONS:
        choices = ", ".join(sorted(SUPPORTED_IMPLEMENTATIONS))
        raise ValueError(f"implementation must be one of {choices}")
    common = {
        "protocol_name": NAVIGATION_PROTOCOL_NAME,
        "protocol_version": NAVIGATION_PROTOCOL_VERSION,
        "model_family": model_family,
        "implementation": implementation,
        "needs_session_id": True,
    }
    if model_family == "streamvln":
        return {
            **common,
            "input_schema": "rgb_uint8_hwc",
            "action_head": "discrete",
            "action_horizon": 4,
            "action_dim": 1,
            "padding_id": -1,
        }
    if model_family == "navila":
        return {
            **common,
            "input_schema": "rgb_frames_uint8_list_hwc",
            "action_head": "navigation",
            "action_horizon": 1,
            "action_dim": 3,
        }
    raise ValueError(f"unsupported vLLM-Omni navigation model family {model_family!r}")


def validate_navigation_handshake(
    metadata: Mapping[str, Any],
    *,
    model_family: str,
    input_schema: str,
    action_head: str,
    action_horizon: int,
    action_dim: int,
    padding_id: int | None = None,
) -> None:
    """Reject a healthy OpenPI server exposing a different policy contract."""

    exact_fields: tuple[tuple[str, object], ...] = (
        ("protocol_name", NAVIGATION_PROTOCOL_NAME),
        ("protocol_version", NAVIGATION_PROTOCOL_VERSION),
        ("model_family", model_family),
        ("needs_session_id", True),
        ("input_schema", input_schema),
        ("action_head", action_head),
    )
    for name, expected in exact_fields:
        value = metadata.get(name)
        if isinstance(expected, bool):
            matches = value is expected
        elif isinstance(expected, int) and isinstance(value, bool):
            matches = False
        else:
            matches = value == expected
        if not matches:
            raise RuntimeError(
                f"vLLM-Omni navigation handshake requires {name}={expected!r}, got {value!r}"
            )
    implementation = metadata.get("implementation")
    if not isinstance(implementation, str) or implementation not in SUPPORTED_IMPLEMENTATIONS:
        choices = ", ".join(sorted(SUPPORTED_IMPLEMENTATIONS))
        raise RuntimeError(
            "vLLM-Omni navigation handshake requires implementation to be one of "
            f"{choices}; got {implementation!r}"
        )
    for name, expected in (
        ("action_horizon", action_horizon),
        ("action_dim", action_dim),
    ):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise RuntimeError(
                f"vLLM-Omni {model_family} handshake requires {name}={expected}, got {value!r}"
            )
    if padding_id is not None:
        value = metadata.get("padding_id")
        if isinstance(value, bool) or not isinstance(value, int) or value != padding_id:
            raise RuntimeError(
                f"vLLM-Omni {model_family} handshake requires padding_id={padding_id}, "
                f"got {value!r}"
            )


def validate_image_payload(images: tuple[Any, ...]) -> int:
    """Validate decoded RGB arrays and cap one msgpack request below 64 MiB."""

    if not images:
        raise ValueError("vLLM-Omni navigation request requires at least one RGB frame")
    total_bytes = 0
    for image in images:
        shape = getattr(image, "shape", ())
        if (
            getattr(image, "ndim", None) != 3
            or len(shape) != 3
            or int(shape[-1]) != 3
            or str(getattr(image, "dtype", "")) != "uint8"
        ):
            raise ValueError("vLLM-Omni navigation RGB frames must be HWC uint8 arrays")
        total_bytes += int(getattr(image, "nbytes", 0))
    if total_bytes > MAX_OPENPI_IMAGE_PAYLOAD_BYTES:
        raise ValueError("vLLM-Omni navigation RGB payload exceeds the 60 MiB client safety limit")
    return total_bytes


__all__ = [
    "DEFAULT_VLLM_OMNI_URL",
    "MAX_OPENPI_IMAGE_PAYLOAD_BYTES",
    "NAVIGATION_PROTOCOL_NAME",
    "NAVIGATION_PROTOCOL_VERSION",
    "SUPPORTED_IMPLEMENTATIONS",
    "navigation_handshake_metadata",
    "validate_image_payload",
    "validate_navigation_handshake",
]
