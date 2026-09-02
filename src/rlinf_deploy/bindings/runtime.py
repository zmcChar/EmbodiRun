"""Generic request-response control loop for a robot-policy binding."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from rlinf_deploy.inference import InferenceClient, PolicyResult
from rlinf_deploy.robots import RobotAdapter
from rlinf_deploy.robots.sensors.cameras import CameraFrame

from . import BindingMapper


class BindingRuntime:
    """Own one policy session and execute mapped actions on one robot."""

    def __init__(
        self,
        robot: RobotAdapter,
        client: InferenceClient,
        *,
        instruction: str,
        mapper: BindingMapper,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        self.robot = robot
        self.client = client
        self.instruction = instruction
        self.mapper = mapper
        self.session = client.open_session(
            robot_id=robot.robot_id,
            action_space=mapper.policy_action_space,
        )
        self.step_id = 0

    def step(
        self,
        frames: Sequence[CameraFrame],
        *,
        reset: bool = False,
    ) -> PolicyResult:
        if reset:
            self.reset()
        observation = self.robot.observe()
        request = self.mapper.map_observation(
            observation,
            session_id=self.session.session_id,
            request_id=f"step-{self.step_id}-{uuid.uuid4().hex}",
            step_id=self.step_id,
            instruction=self.instruction,
            frames=tuple(frames),
        )
        result = self.client.step(request)
        self.robot.execute(self.mapper.map_result(result))
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


__all__ = ["BindingRuntime"]
