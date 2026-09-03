"""Generic request-response control loop for a robot-policy binding."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Sequence

from rlinf_deploy.inference import InferenceClient, PolicyResult
from rlinf_deploy.robots import RobotAction, RobotAdapter
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
        chunk_steps: int,
        control_hz: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        if (
            isinstance(chunk_steps, bool)
            or not isinstance(chunk_steps, int)
            or chunk_steps <= 0
        ):
            raise ValueError("chunk_steps must be a positive integer")
        if (
            isinstance(control_hz, bool)
            or not isinstance(control_hz, (int, float))
            or not math.isfinite(control_hz)
            or control_hz <= 0
        ):
            raise ValueError("control_hz must be a finite positive number")
        self.robot = robot
        self.client = client
        self.instruction = instruction
        self.mapper = mapper
        self.chunk_steps = chunk_steps
        self.action_period_s = 1.0 / control_hz
        self.monotonic = monotonic
        self.sleep = sleep
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
        actions = tuple(self.mapper.map_result(result))
        if not actions:
            raise RuntimeError("binding returned an empty action chunk")
        if any(not isinstance(action, RobotAction) for action in actions):
            raise TypeError("binding action chunk must contain RobotAction values")
        if len(actions) < self.chunk_steps:
            raise RuntimeError(
                f"binding returned {len(actions)} action(s), fewer than requested "
                f"chunk_steps={self.chunk_steps}"
            )
        actions = actions[: self.chunk_steps]
        deadline_s = self.monotonic()
        for index, action in enumerate(actions):
            self.robot.execute(action)
            if index + 1 == len(actions):
                continue
            deadline_s += self.action_period_s
            remaining_s = deadline_s - self.monotonic()
            if remaining_s > 0:
                self.sleep(remaining_s)
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
