"""Dependency-free boundary for simulator-backed robot episodes."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Protocol, TypeAlias, runtime_checkable

from embodied_runtime.contracts import Metadata, RobotAction, RobotObservation

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


class GymLikeSimulatorAdapter:
    """Adapt a Gym-style object without importing Gym or a simulator package.

    The wrapped object only needs ``reset`` and ``step`` methods. An optional
    ``observe`` method is used when present; otherwise the most recent mapped
    observation is returned.
    """

    def __init__(
        self,
        environment: Any,
        capabilities: SimulatorCapabilities,
        *,
        observation_mapper: ObservationMapper | None = None,
        action_mapper: ActionMapper | None = None,
        success_mapper: SuccessMapper | None = None,
        subgoal_mapper: SubgoalMapper | None = None,
    ) -> None:
        if not callable(getattr(environment, "reset", None)):
            raise TypeError("Gym-like environment must provide reset()")
        if not callable(getattr(environment, "step", None)):
            raise TypeError("Gym-like environment must provide step()")
        if not isinstance(capabilities, SimulatorCapabilities):
            raise TypeError("capabilities must be SimulatorCapabilities")
        observation_mapper = observation_mapper or _identity_observation
        action_mapper = action_mapper or _identity_action
        for mapper_name, mapper in (
            ("observation_mapper", observation_mapper),
            ("action_mapper", action_mapper),
        ):
            if not callable(mapper):
                raise TypeError(f"{mapper_name} must be callable")
        for mapper_name, mapper in (
            ("success_mapper", success_mapper),
            ("subgoal_mapper", subgoal_mapper),
        ):
            if mapper is not None and not callable(mapper):
                raise TypeError(f"{mapper_name} must be callable or None")

        self._environment = environment
        self._capabilities = capabilities
        self._observation_mapper = observation_mapper
        self._action_mapper = action_mapper
        self._success_mapper = success_mapper or _success_from_info
        self._subgoal_mapper = subgoal_mapper or _subgoal_from_info
        self._last_observation: RobotObservation | None = None
        self._last_reset_info: dict[str, Any] = {}
        self._closed = False

    @property
    def capabilities(self) -> SimulatorCapabilities:
        return self._capabilities

    @property
    def last_reset_info(self) -> Mapping[str, Any]:
        return dict(self._last_reset_info)

    def reset(
        self,
        task: str | None = None,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> RobotObservation:
        self._ensure_open()
        if task is not None:
            if not isinstance(task, str) or not task.strip():
                raise ValueError("simulator task must be a non-empty string or None")
            task = task.strip()
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("simulator seed must be an int or None")
            if not self._capabilities.supports_seed:
                raise ValueError("simulator endpoint does not advertise seeded resets")
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("simulator reset options must be a mapping or None")

        reset_options = dict(options or {})
        if task is not None:
            configured_task = reset_options.get("task")
            if configured_task is not None and configured_task != task:
                raise ValueError("task conflicts with reset options")
            reset_options["task"] = task

        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if reset_options:
            kwargs["options"] = reset_options
        result = self._environment.reset(**kwargs)
        raw_observation, info = _unpack_reset(result)
        observation = self._map_observation(raw_observation)
        self._last_observation = observation
        self._last_reset_info = dict(info)
        return observation

    def observe(self) -> RobotObservation:
        self._ensure_open()
        observe = getattr(self._environment, "observe", None)
        if callable(observe):
            observation = self._map_observation(observe())
            self._last_observation = observation
            return observation
        if self._last_observation is None:
            raise RuntimeError("simulator must be reset before a cached observation is available")
        return self._last_observation

    def step(self, action: RobotAction) -> EpisodeStep:
        self._ensure_open()
        if not isinstance(action, RobotAction):
            raise TypeError("simulator action must be a RobotAction")
        raw_action = self._action_mapper(action)
        raw_observation, reward, terminated, truncated, info = _unpack_step(
            self._environment.step(raw_action)
        )
        observation = self._map_observation(raw_observation)
        success = self._success_mapper(info)
        subgoal_progress = self._subgoal_mapper(info)
        outcome = EpisodeStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=success,
            subgoal_progress=subgoal_progress,
            info=info,
        )
        self._last_observation = observation
        return outcome

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self._environment, "close", None)
        if callable(close):
            close()
        self._closed = True

    def _map_observation(self, raw_observation: Any) -> RobotObservation:
        observation = self._observation_mapper(raw_observation)
        if not isinstance(observation, RobotObservation):
            raise TypeError("observation_mapper must return a RobotObservation")
        return observation

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("simulator endpoint is closed")


def _identity_observation(value: Any) -> RobotObservation:
    if not isinstance(value, RobotObservation):
        raise TypeError("raw observation requires an observation_mapper")
    return value


def _identity_action(action: RobotAction) -> RobotAction:
    return action


def _unpack_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], result[1]
    return result, {}


def _unpack_step(
    result: Any,
) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    if not isinstance(result, tuple):
        raise TypeError("Gym-like step() must return a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        if not isinstance(info, Mapping):
            raise TypeError("Gym-like step info must be a mapping")
        return observation, reward, bool(terminated), bool(truncated), dict(info)
    if len(result) == 4:
        observation, reward, done, info = result
        if not isinstance(info, Mapping):
            raise TypeError("Gym-like step info must be a mapping")
        normalized_info = dict(info)
        truncated = bool(normalized_info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
        return observation, reward, terminated, truncated, normalized_info
    raise ValueError("Gym-like step() must return four or five values")


def _success_from_info(info: Mapping[str, Any]) -> bool | None:
    value = info.get("success", info.get("is_success"))
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    raise TypeError("simulator success info must be boolean, zero, one, or None")


def _subgoal_from_info(info: Mapping[str, Any]) -> float | None:
    value = info.get("subgoal_progress")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("simulator subgoal progress info must be numeric or None")
    return float(value)


__all__ = [
    "ActionMapper",
    "EpisodeStep",
    "GymLikeSimulatorAdapter",
    "ObservationMapper",
    "SimulatorCapabilities",
    "SimulatorEndpoint",
    "SubgoalMapper",
    "SuccessMapper",
]
