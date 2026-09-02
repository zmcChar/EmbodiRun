"""Dependency-free adapter for Gym-shaped simulator environments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlinf_deploy.robots.adapter import RobotAction, RobotObservation

from ._gym_normalization import (
    identity_action,
    identity_observation,
    subgoal_from_info,
    success_from_info,
    unpack_reset,
    unpack_step,
)
from .base import (
    ActionMapper,
    EpisodeStep,
    ObservationMapper,
    SimulatorCapabilities,
    SubgoalMapper,
    SuccessMapper,
)


class GymLikeSimulatorAdapter:
    """Adapt reset/step environments without importing Gym or a simulator package.

    An optional ``observe`` method is used when present; otherwise the most
    recent mapped observation is returned.
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
        observation_mapper = observation_mapper or identity_observation
        action_mapper = action_mapper or identity_action
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
        self._success_mapper = success_mapper or success_from_info
        self._subgoal_mapper = subgoal_mapper or subgoal_from_info
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
        raw_observation, info = unpack_reset(self._environment.reset(**kwargs))
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
        raw_observation, reward, terminated, truncated, info = unpack_step(
            self._environment.step(raw_action)
        )
        observation = self._map_observation(raw_observation)
        outcome = EpisodeStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=self._success_mapper(info),
            subgoal_progress=self._subgoal_mapper(info),
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


__all__ = ["GymLikeSimulatorAdapter"]
