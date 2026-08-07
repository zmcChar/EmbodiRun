"""Synthetic DROID observations for GR00T smoke tests."""

from __future__ import annotations

import numpy as np

from ...request import RawRequest


def synthetic_droid_request(
    prompt: str = "pick up the object",
    *,
    image_height: int = 180,
    image_width: int = 320,
) -> RawRequest:
    """Create one correctly shaped zero-valued DROID request."""

    if not prompt:
        raise ValueError("prompt must not be empty")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be greater than zero")
    frame_history = np.zeros(
        (2, image_height, image_width, 3),
        dtype=np.uint8,
    )
    eef_9d = np.zeros((1, 9), dtype=np.float32)
    eef_9d[..., 3:] = np.asarray((1, 0, 0, 0, 1, 0), dtype=np.float32)
    return RawRequest(
        observation={
            "video": {
                "exterior_image_1_left": frame_history.copy(),
                "wrist_image_left": frame_history.copy(),
            },
            "state": {
                "eef_9d": eef_9d,
                "gripper_position": np.zeros((1, 1), dtype=np.float32),
                "joint_position": np.zeros((1, 7), dtype=np.float32),
            },
        },
        prompt=prompt,
    )
