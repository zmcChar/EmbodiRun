"""Recurrent episode ordering for stateful navigation policies."""

from __future__ import annotations

from dataclasses import dataclass

from embodied_runtime.tasks.navigation import NavigationObservation

from .errors import NavigationPolicyError


@dataclass(slots=True)
class EpisodeCursor:
    """Validate one recurrent model's episode and observation ordering."""

    episode_id: str | None = None
    sequence: int | None = None

    def requires_reset(self, observation: NavigationObservation) -> bool:
        if self.episode_id is None:
            return True
        if observation.episode_id != self.episode_id:
            if not observation.reset:
                raise NavigationPolicyError("a new episode_id must set observation.reset=true")
            return True
        if observation.reset:
            return True
        if self.sequence is not None and observation.sequence <= self.sequence:
            raise NavigationPolicyError(
                f"observation sequence must increase beyond {self.sequence}"
            )
        return False

    def commit(self, observation: NavigationObservation) -> None:
        self.episode_id = observation.episode_id
        self.sequence = observation.sequence

    def clear(self) -> None:
        self.episode_id = None
        self.sequence = None


__all__ = ["EpisodeCursor"]
