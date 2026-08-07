"""Pure aligned-history compaction for recurrent StreamVLN inference."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CompactedHistory:
    """A bounded set of globally spaced frames and aligned payloads."""

    frame_ids: list[int]
    aligned: tuple[list[Any], ...]

    @property
    def count(self) -> int:
        return len(self.frame_ids)


def compact_aligned_history(
    frame_ids: Sequence[int],
    aligned: Sequence[Sequence[Any]],
    *,
    next_step_id: int,
    num_history: int,
) -> CompactedHistory:
    """Select the nearest available frames for uniformly spaced global targets."""

    if not frame_ids:
        return CompactedHistory(frame_ids=[], aligned=tuple([] for _ in aligned))

    stride = max(1, next_step_id // num_history)
    targets = tuple(range(0, next_step_id, stride))[:num_history]
    available = set(range(len(frame_ids)))
    selected: list[int] = []
    for target in targets:
        if not available:
            break
        index = min(
            available,
            key=lambda candidate: (
                abs(frame_ids[candidate] - target),
                frame_ids[candidate] > target,
                frame_ids[candidate],
            ),
        )
        selected.append(index)
        available.remove(index)
    selected.sort(key=frame_ids.__getitem__)
    return CompactedHistory(
        frame_ids=[frame_ids[index] for index in selected],
        aligned=tuple([values[index] for index in selected] for values in aligned),
    )


__all__ = ["CompactedHistory", "compact_aligned_history"]
