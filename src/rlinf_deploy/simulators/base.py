"""Dependency-free values and protocol for simulator-backed robot episodes."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol, TypeAlias, runtime_checkable

from rlinf_deploy.robots.adapter import RobotAction, RobotObservation
from rlinf_deploy.types import Metadata

ObservationMapper: TypeAlias = Callable[[Any], RobotObservation]
ActionMapper: TypeAlias = Callable[[RobotAction], Any]
SuccessMapper: TypeAlias = Callable[[Mapping[str, Any]], bool | None]
SubgoalMapper: TypeAlias = Callable[[Mapping[str, Any]], float | None]


@dataclass(frozen=True, slots=True)
class SimulatorCapabilities:
    """Stable facts advertised by a simulator endpoint."""

    name: str
    environment_id: str
    embodiment: str | None = None
    action_space_id: str | None = None
    supports_seed: bool = True
    supports_observe: bool = True
    features: frozenset[str] = frozenset()
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("name", "environment_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"simulator {field_name} must be a string")
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"simulator {field_name} must not be empty")
            object.__setattr__(self, field_name, normalized)

        for field_name in ("embodiment", "action_space_id"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(f"simulator {field_name} must be a string or None")
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"simulator {field_name} must not be empty")
            object.__setattr__(self, field_name, normalized)

        for field_name in ("supports_seed", "supports_observe"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"simulator {field_name} must be a bool")

        if isinstance(self.features, str):
            raise TypeError("simulator features must be an iterable of strings")
        try:
            features = frozenset(self.features)
        except TypeError as error:
            raise TypeError("simulator features must be an iterable of strings") from error
        if any(not isinstance(feature, str) or not feature.strip() for feature in features):
            raise ValueError("simulator features must contain non-empty strings")
        object.__setattr__(self, "features", frozenset(feature.strip() for feature in features))

        if not isinstance(self.metadata, Mapping):
            raise TypeError("simulator metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    """One normalized transition returned by a simulator endpoint."""

    observation: RobotObservation
    reward: float
    terminated: bool
    truncated: bool
    success: bool | None = None
    subgoal_progress: float | None = None
    info: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.observation, RobotObservation):
            raise TypeError("episode step observation must be a RobotObservation")
        if isinstance(self.reward, bool) or not isinstance(self.reward, Real):
            raise TypeError("episode step reward must be a real number")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("episode step reward must be finite")
        object.__setattr__(self, "reward", reward)

        for field_name in ("terminated", "truncated"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"episode step {field_name} must be a bool")
        if self.success is not None and not isinstance(self.success, bool):
            raise TypeError("episode step success must be a bool or None")

        if self.subgoal_progress is not None:
            progress = self.subgoal_progress
            if isinstance(progress, bool) or not isinstance(progress, Real):
                raise TypeError("episode step subgoal_progress must be a real number or None")
            normalized_progress = float(progress)
            if not math.isfinite(normalized_progress) or not 0.0 <= normalized_progress <= 1.0:
                raise ValueError("episode step subgoal_progress must be between zero and one")
            object.__setattr__(self, "subgoal_progress", normalized_progress)

        if not isinstance(self.info, Mapping):
            raise TypeError("episode step info must be a mapping")
        object.__setattr__(self, "info", dict(self.info))

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@runtime_checkable
class SimulatorEndpoint(Protocol):
    """Hardware-neutral surface used by evaluation and policy code."""

    @property
    def capabilities(self) -> SimulatorCapabilities: ...

    def reset(
        self,
        task: str | None = None,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> RobotObservation: ...

    def observe(self) -> RobotObservation: ...

    def step(self, action: RobotAction) -> EpisodeStep: ...

    def close(self) -> None: ...


__all__ = [
    "ActionMapper",
    "EpisodeStep",
    "ObservationMapper",
    "SimulatorCapabilities",
    "SimulatorEndpoint",
    "SubgoalMapper",
    "SuccessMapper",
]
