"""Create an inference service client for a supported backend/protocol pair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .backends.sglang import SglangHttpClient
from .backends.vvla import VvlaHttpClient, VvlaWirelessClient
from .client import InferenceClient


def build_inference_client(
    protocol: str,
    endpoint: str,
    options: Mapping[str, Any],
    *,
    backend: str = "vvla",
    timeout_s: float,
) -> InferenceClient:
    """Build one of the explicitly supported backend/protocol clients."""

    token = options.get("token")
    if token is not None and not isinstance(token, str):
        raise ValueError("inference token must be a string")
    if backend == "sglang":
        if protocol != "http":
            raise ValueError(
                "SGLang action serving currently requires the http protocol"
            )
        return SglangHttpClient(
            endpoint,
            token=token,
            timeout_s=timeout_s,
            image_keys=_mapping(options, "image_keys"),
            state_fields=_strings(options, "state_fields"),
            action_feature_names=_strings(options, "action_feature_names"),
            output_action_dim=_optional_positive_integer(
                options,
                "output_action_dim",
            ),
            parameters=_mapping(options, "parameters"),
            runtime=_mapping(options, "runtime"),
        )
    if backend != "vvla":
        raise ValueError(f"unsupported inference backend {backend!r}")
    if protocol == "http":
        return VvlaHttpClient(endpoint, token=token, timeout_s=timeout_s)
    if protocol != "wireless":
        raise ValueError(f"unsupported inference protocol {protocol!r}")
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


def _mapping(options: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = options.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"inference option {name!r} must be an object")
    return dict(value)


def _strings(options: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = options.get(name, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(f"inference option {name!r} must be a list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"inference option {name!r} must contain non-empty strings")
    return tuple(value)


def _optional_positive_integer(
    options: Mapping[str, Any],
    name: str,
) -> int | None:
    value = options.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"inference option {name!r} must be a positive integer")
    return value


__all__ = ["build_inference_client"]
