"""Model-local prediction values returned by StreamVLN."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamVLNPrediction:
    """Normalized native checkpoint output without task-domain interpretation."""

    actions: tuple[int, ...]
    raw_output: str
    generation_time_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "actions": list(self.actions),
            "raw_output": self.raw_output,
            "generation_time_s": self.generation_time_s,
        }


__all__ = ["StreamVLNPrediction"]
