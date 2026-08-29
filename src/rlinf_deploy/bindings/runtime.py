"""Robot-independent execution loop for a remote policy session."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Protocol

from rlinf_deploy.inference import (
    ImagePayload,
    PolicyObservation,
    PolicyResult,
    VvlaHttpClient,
)
from rlinf_deploy.robots import RobotAction, RobotAdapter


class ActionMappingError(RuntimeError):
    """A policy result cannot be represented in a robot action space."""


class PolicyActionMapper(Protocol):
    def map_result(self, result: PolicyResult) -> RobotAction: ...


class PolicyRobotRuntime:
    """Own session ordering while delegating policy and hardware semantics."""

    def __init__(
        self,
        robot: RobotAdapter,
        client: VvlaHttpClient,
        *,
        instruction: str,
        policy_action_space: str,
        mapper: PolicyActionMapper,
        mapping_error: Callable[[ActionMappingError], Exception] | None = None,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        if not policy_action_space.strip():
            raise ValueError("policy_action_space must not be empty")
        self.robot = robot
        self.client = client
        self.instruction = instruction
        self.policy_action_space = policy_action_space
        self.mapper = mapper
        self._mapping_error = mapping_error
        self.session = client.open_session(
            robot_id=robot.robot_id,
            action_space=policy_action_space,
        )
        self.step_id = 0

    def step(
        self, images: Sequence[ImagePayload], *, reset: bool = False
    ) -> PolicyResult:
        if reset:
            self.reset()
        observation = self.robot.observe()
        request = PolicyObservation(
            session_id=self.session.session_id,
            request_id=f"step-{self.step_id}-{uuid.uuid4().hex}",
            step_id=self.step_id,
            instruction=self.instruction,
            state=dict(observation.values),
            images=images,
            reset=False,
            metadata={"robot_timestamp_s": observation.timestamp_s},
        )
        result = self.client.step(request)
        try:
            action = self.mapper.map_result(result)
        except ActionMappingError as error:
            if self._mapping_error is None:
                raise
            raise self._mapping_error(error) from error
        self.robot.execute(action)
        self.step_id += 1
        return result

    def reset(self) -> None:
        self.robot.stop()
        self.session = self.client.reset(
            self.session.session_id,
            request_id=f"reset-{uuid.uuid4().hex}",
        )
        self.step_id = 0

    def close(self) -> None:
        try:
            self.robot.stop()
        finally:
            self.client.close(self.session.session_id)


__all__ = [
    "ActionMappingError",
    "PolicyActionMapper",
    "PolicyRobotRuntime",
]
