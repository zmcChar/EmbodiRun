"""Connect a Franka adapter to a remote VVLA policy session."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from rlinf_deploy.inference import (
    ImagePayload,
    PolicyClient,
    PolicyObservation,
    PolicyResult,
)
from rlinf_deploy.robots.franka.fr3.adapter import FR3Adapter, FR3AdapterError

from .action import Pi05ActionMapper, Pi05ActionMapperError
from .contract import POLICY_ACTION_SPACE


class Pi05FR3Runtime:
    def __init__(
        self,
        robot: FR3Adapter,
        client: PolicyClient,
        *,
        instruction: str,
        mapper: Pi05ActionMapper | None = None,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        self.robot = robot
        self.client = client
        self.instruction = instruction
        self.mapper = mapper or Pi05ActionMapper()
        self.session = client.open_session(
            robot_id=robot.robot_id,
            action_space=POLICY_ACTION_SPACE,
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
        except Pi05ActionMapperError as error:
            raise FR3AdapterError(f"action mapping failed: {error}") from error
        self.robot.execute(action)
        self.step_id += 1
        return result

    def reset(self) -> None:
        self.robot.stop()
        request_id = f"reset-{uuid.uuid4().hex}"
        self.session = self.client.reset(
            self.session.session_id,
            request_id=request_id,
        )
        self.step_id = 0

    def close(self) -> None:
        self.robot.stop()
        self.client.close(self.session.session_id)


__all__ = ["Pi05FR3Runtime"]
