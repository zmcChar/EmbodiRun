"""OpenAI-compatible request assembly for Qwen navigation."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from typing import Any

from .errors import QwenNavigationError, QwenNavigationValidationError
from .prompt import SYSTEM_PROMPT
from .schema import WAYPOINT_PLAN_SCHEMA

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_CONTEXT_JSON_BYTES = 64 * 1024
SUPPORTED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})


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


def build_chat_request(
    *,
    model: str,
    prompt: str,
    image_data: bytes,
    media_type: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one strict multimodal chat-completions request."""

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
        "model": model,
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


def build_correction_request(
    request: Mapping[str, Any],
    *,
    assistant_content: str,
    error: QwenNavigationValidationError,
) -> dict[str, Any]:
    """Append one schema-correction turn after an invalid assistant response."""

    return {
        **request,
        "messages": [
            *request["messages"],
            {"role": "assistant", "content": assistant_content},
            {
                "role": "user",
                "content": f"上一输出不符合 waypoint schema：{error}。只修正并返回一个 JSON 对象。",
            },
        ],
    }


__all__ = ["build_chat_request", "build_correction_request"]
