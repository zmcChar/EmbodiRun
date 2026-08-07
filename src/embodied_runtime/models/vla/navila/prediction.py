"""Model-native value returned by a NaVILA inference."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import NaVILAAction


@dataclass(frozen=True, slots=True)
class NaVILAPrediction:
    action: NaVILAAction
    raw_output: str
    token_ids: tuple[int, ...]
    generation_time_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "primitive": self.action.primitive.value,
            "magnitude": self.action.magnitude,
            "unit": self.action.unit,
            "raw_output": self.raw_output,
            "token_ids": list(self.token_ids),
            "generation_time_s": self.generation_time_s,
        }


__all__ = ["NaVILAPrediction"]
