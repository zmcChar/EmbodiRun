"""Official full-episode sampling cadence for NaVILA's eight image slots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

NAVILA_FRAME_COUNT = 8

FrameT = TypeVar("FrameT")


def sample_episode_frames(frames: Sequence[FrameT]) -> tuple[FrameT, ...]:
    """Uniformly select seven historical positions plus the latest frame."""

    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence) or not frames:
        raise ValueError("frames must be a non-empty sequence")
    if len(frames) <= NAVILA_FRAME_COUNT:
        return tuple(frames)
    last_index = len(frames) - 1
    indices = tuple((index * last_index) // 7 for index in range(7)) + (last_index,)
    return tuple(frames[index] for index in indices)


__all__ = ["NAVILA_FRAME_COUNT", "sample_episode_frames"]
