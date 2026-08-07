"""Prepare task-owned navigation requests for the Qwen client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from embodied_runtime.tasks.navigation import (
    EncodedDepthFrame,
    NavigationObservation,
    NavigationRequest,
)


@dataclass(frozen=True, slots=True)
class PreparedNavigationInput:
    prompt: str
    image_data: bytes
    media_type: str
    observation_sequence: int
    episode_id: str
    context: Mapping[str, Any]


def _depth_context(depth: EncodedDepthFrame | None) -> dict[str, Any] | None:
    if depth is None:
        return None
    return {
        "sequence": depth.sequence,
        "captured_at_s": depth.captured_at_s,
        "width": depth.width,
        "height": depth.height,
        "scale_m": depth.scale_m,
        "registered_to_rgb": depth.registered_to_rgb,
        "encoding": depth.encoding,
        "metadata": dict(depth.metadata),
    }


def _observation_input(
    prompt: str,
    observation: NavigationObservation,
) -> PreparedNavigationInput:
    latest = observation.latest_rgb
    context = {
        "episode_id": observation.episode_id,
        "sequence": observation.sequence,
        "reset": observation.reset,
        "rgb": {
            "sequence": latest.sequence,
            "captured_at_s": latest.captured_at_s,
            "media_type": latest.media_type,
            "width": latest.width,
            "height": latest.height,
            "context_frame_count": len(observation.rgb_frames),
        },
        "depth": _depth_context(observation.depth),
        "robot_state": dict(observation.robot_state),
        "metadata": dict(observation.metadata),
    }
    return PreparedNavigationInput(
        prompt=prompt,
        image_data=latest.data,
        media_type=latest.media_type,
        observation_sequence=observation.sequence,
        episode_id=observation.episode_id,
        context=context,
    )


def prepare_navigation_request(request: NavigationRequest) -> PreparedNavigationInput:
    if not isinstance(request, NavigationRequest):
        raise TypeError("request must be a NavigationRequest")
    return _observation_input(request.instruction, request.observation)


__all__ = [
    "PreparedNavigationInput",
    "prepare_navigation_request",
]
