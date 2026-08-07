"""StreamVLN's normalized native discrete action vocabulary.

The real-world StreamVLN checkpoint emits four integer actions: stop, move
forward, and turn left or right. This model-owned module validates only the
native checkpoint output; policies own spatial interpretation.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from enum import IntEnum

MAX_FUTURE_ACTIONS = 4


class StreamVLNAction(IntEnum):
    """Native action ids used by the official real-world checkpoint."""

    STOP = 0
    FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class StreamVLNNativeOutputError(ValueError):
    """The checkpoint returned an invalid native action sequence."""


def normalize_native_actions(
    actions: Sequence[object],
    *,
    max_actions: int = MAX_FUTURE_ACTIONS,
) -> tuple[int, ...]:
    """Validate an evaluator result without accepting lossy numeric coercion."""

    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise StreamVLNNativeOutputError("StreamVLN actions must be a sequence")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
        raise ValueError("max_actions must be a positive integer")
    if not actions:
        raise StreamVLNNativeOutputError("StreamVLN returned no parseable actions")

    normalized: list[int] = []
    for index, action in enumerate(actions):
        if isinstance(action, bool):
            raise StreamVLNNativeOutputError(f"StreamVLN action {index} must be an integer")
        try:
            action_id = operator.index(action)
        except TypeError as error:
            raise StreamVLNNativeOutputError(
                f"StreamVLN action {index} must be an integer"
            ) from error
        try:
            StreamVLNAction(action_id)
        except ValueError as error:
            raise StreamVLNNativeOutputError(
                f"unsupported StreamVLN action {action_id} at index {index}"
            ) from error
        normalized.append(action_id)
        if len(normalized) > max_actions:
            raise StreamVLNNativeOutputError(
                f"StreamVLN returned more than {max_actions} reachable actions"
            )
        if action_id == StreamVLNAction.STOP:
            # The official controller ignores any suffix after STOP.  Truncate
            # here as well so downstream consumers never see unreachable work.
            break
    return tuple(normalized)


__all__ = [
    "MAX_FUTURE_ACTIONS",
    "StreamVLNAction",
    "StreamVLNNativeOutputError",
    "normalize_native_actions",
]
