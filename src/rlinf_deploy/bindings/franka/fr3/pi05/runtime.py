"""Connect a Franka adapter to a remote VVLA policy session."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence

from rlinf_deploy.inference import ImagePayload, PolicyObservation, PolicyResult, VvlaHttpClient

from ..action import RobotAction
from rlinf_deploy.robots.franka.fr3.adapter import FR3_ACTION_SPACE, FR3Adapter, FR3AdapterError


class Pi05FR3Runtime:
    def __init__(
        self,
        robot: FR3Adapter,
        client: VvlaHttpClient,
        *,
        instruction: str,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        self.robot = robot
        self.client = client
        self.instruction = instruction
        self.session = client.open_session(
            robot_id=robot.robot_id,
            action_space=FR3_ACTION_SPACE,
        )
        self.step_id = 0

    def step(self, images: Sequence[ImagePayload], *, reset: bool = False) -> PolicyResult:
        observation = self.robot.observe()
        request = PolicyObservation(
            session_id=self.session.session_id,
            request_id=f"step-{self.step_id}-{uuid.uuid4().hex}",
            step_id=self.step_id,
            instruction=self.instruction,
            state=dict(observation.values),
            images=images,
            reset=reset,
            metadata={"robot_timestamp_s": observation.timestamp_s},
        )
        result = self.client.step(request)
        if result.action_space != FR3_ACTION_SPACE:
            raise FR3AdapterError(
                f"VVLA returned {result.action_space!r}, expected {FR3_ACTION_SPACE!r}"
            )
        for policy_action in result.actions:
            self.robot.execute(
                RobotAction(
                    timestamp_s=time.time(),
                    values={"type": policy_action.kind, **dict(policy_action.values)},
                    metadata={
                        "action_space": result.action_space,
                        "request_id": result.request_id,
                        "step_id": result.step_id,
                        "session_revision": result.session_revision,
                    },
                )
            )
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
