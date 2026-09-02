"""Connect an SO-101 follower to a remote VVLA Pi0.5 session."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Protocol

from rlinf_deploy.bindings import ActionMapper
from rlinf_deploy.inference import (
    ImagePayload,
    PolicyClient,
    PolicyObservation,
    PolicyResult,
)
from rlinf_deploy.robots.lerobot.so101 import SO101Adapter

from .mapper import POLICY_ACTION_SPACE, Pi05SO101ActionMapper


class _EncodedImage(Protocol):
    name: str
    mime_type: str
    data: bytes


class Pi05SO101Runtime:
    """Own one SO-101 policy session and execute one validated action per step."""

    def __init__(
        self,
        robot: SO101Adapter,
        client: PolicyClient,
        *,
        instruction: str,
        mapper: ActionMapper | None = None,
    ) -> None:
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        self.robot = robot
        self.client = client
        self.instruction = instruction
        self.mapper = mapper if mapper is not None else Pi05SO101ActionMapper()
        self.session = client.open_session(
            robot_id=robot.robot_id,
            action_space=POLICY_ACTION_SPACE,
        )
        self.step_id = 0

    def step(
        self,
        images: Sequence[_EncodedImage],
        *,
        reset: bool = False,
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
            images=tuple(
                ImagePayload(image.name, image.mime_type, image.data)
                for image in images
            ),
            reset=False,
            metadata={"robot_timestamp_s": observation.timestamp_s},
        )
        result = self.client.step(request)
        action = self.mapper.map_result(result)
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


__all__ = ["Pi05SO101Runtime"]
