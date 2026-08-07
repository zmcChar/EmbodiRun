"""Deterministic endpoint-level card placement controller."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from embodied_runtime.evaluation import ExperimentCondition
from embodied_runtime.robots.action import RobotAction
from embodied_runtime.robots.observation import RobotObservation
from embodied_runtime.simulators import EpisodeStep

from .scene import agent_state, place_target_robot_frame
from .settings import EndpointPlaceControllerConfig


@dataclass(frozen=True, slots=True)
class PlaceControllerResult:
    observation: RobotObservation
    outcome: EpisodeStep | None
    steps: int
    events: tuple[dict[str, Any], ...]


class EndpointPlaceController:
    """Move a currently grasped card to the placemat through endpoint actions."""

    def __init__(self, config: EndpointPlaceControllerConfig) -> None:
        if not isinstance(config, EndpointPlaceControllerConfig):
            raise TypeError("config must be an EndpointPlaceControllerConfig")
        self.config = config

    def execute(
        self,
        *,
        endpoint: Any,
        observation: RobotObservation,
        available_steps: int,
        condition: ExperimentCondition,
        prompt: str,
        grasped_poker: str,
        placement_index: int,
    ) -> PlaceControllerResult:
        if not isinstance(observation, RobotObservation):
            raise TypeError("observation must be a RobotObservation")
        if isinstance(available_steps, bool) or not isinstance(available_steps, int):
            raise TypeError("available_steps must be an integer")
        if available_steps < 0:
            raise ValueError("available_steps must be non-negative")
        if placement_index < 0:
            raise ValueError("placement_index must be non-negative")
        if available_steps == 0:
            return PlaceControllerResult(observation, None, 0, ())

        target = place_target_robot_frame(
            endpoint,
            placement_index=placement_index,
            config=self.config,
        )
        state = agent_state(observation)
        waypoints = build_place_waypoints(state, target, config=self.config)
        events: list[dict[str, Any]] = []
        outcome: EpisodeStep | None = None
        current_observation = observation
        for waypoint_index, (phase, action_values) in enumerate(waypoints[:available_steps]):
            outcome = endpoint.step(
                RobotAction(
                    timestamp_s=time.time(),
                    values={"action": action_values},
                    metadata={
                        "condition": condition.value,
                        "prompt": prompt,
                        "controller": "shared_endpoint_place",
                        "phase": phase,
                        "grasped_poker": grasped_poker,
                        "placement_index": placement_index,
                        "waypoint_index": waypoint_index,
                    },
                )
            )
            current_observation = outcome.observation
            events.append(
                {
                    "event": "place_controller_step",
                    "phase": phase,
                    "grasped_poker": grasped_poker,
                    "placement_index": placement_index,
                    "waypoint_index": waypoint_index,
                    "success": bool(outcome.success),
                    "physics_error": bool(outcome.info.get("physics_error", False)),
                }
            )
            if outcome.done:
                break
        return PlaceControllerResult(
            observation=current_observation,
            outcome=outcome,
            steps=len(events),
            events=tuple(events),
        )


def build_place_waypoints(
    state: Any,
    target_position: Any,
    *,
    config: EndpointPlaceControllerConfig,
) -> list[tuple[str, Any]]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("the VLABench place controller requires NumPy") from error

    state = np.asarray(state, dtype=np.float64).reshape(7)
    target = np.asarray(target_position, dtype=np.float64).reshape(3)
    current = state[:3].copy()
    euler = state[3:6].copy()
    safe_z = max(current[2] + config.lift_height_m, target[2] + config.clearance_m)
    lift = np.array([current[0], current[1], safe_z], dtype=np.float64)
    above = np.array([target[0], target[1], safe_z], dtype=np.float64)
    retract = np.array(
        [target[0], target[1], max(safe_z, target[2] + config.retract_height_m)],
        dtype=np.float64,
    )
    result: list[tuple[str, Any]] = []

    def add_segment(phase: str, destination: Any, gripper: float) -> None:
        nonlocal current
        destination = np.asarray(destination, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(destination - current))
        count = max(1, math.ceil(distance / config.max_translation_step_m))
        start = current.copy()
        for fraction in np.linspace(1.0 / count, 1.0, count):
            position = start + (destination - start) * float(fraction)
            result.append(
                (
                    phase,
                    np.concatenate([position, euler, np.array([gripper])]).astype(np.float32),
                )
            )
        current = destination

    add_segment("lift", lift, 0.0)
    add_segment("translate", above, 0.0)
    add_segment("descend", target, 0.0)
    open_action = np.concatenate([target, euler, np.array([1.0])]).astype(np.float32)
    result.extend(("release", open_action.copy()) for _ in range(config.open_steps))
    add_segment("retract", retract, 1.0)
    return result


__all__ = ["EndpointPlaceController"]
