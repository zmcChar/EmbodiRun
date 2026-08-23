"""Gym-style reset, step, and outcome normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

from rlinf_deploy.robots.action import RobotAction
from rlinf_deploy.robots.observation import RobotObservation


def identity_observation(value: Any) -> RobotObservation:
    if not isinstance(value, RobotObservation):
        raise TypeError("raw observation requires an observation_mapper")
    return value


def identity_action(action: RobotAction) -> RobotAction:
    return action


def unpack_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], result[1]
    return result, {}


def unpack_step(
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


def success_from_info(info: Mapping[str, Any]) -> bool | None:
    value = info.get("success", info.get("is_success"))
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    raise TypeError("simulator success info must be boolean, zero, one, or None")


def subgoal_from_info(info: Mapping[str, Any]) -> float | None:
    value = info.get("subgoal_progress")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("simulator subgoal progress info must be numeric or None")
    return float(value)


__all__ = [
    "identity_action",
    "identity_observation",
    "subgoal_from_info",
    "success_from_info",
    "unpack_reset",
    "unpack_step",
]
