"""Connect an SO-101 follower to a remote VVLA Pi0.5 session."""

from __future__ import annotations

from rlinf_deploy.inference import VvlaHttpClient
from rlinf_deploy.robots.lerobot.so101 import SO101Adapter
from rlinf_deploy.runtime import PolicyRobotRuntime

from .action import Pi05SO101ActionMapper
from .contract import POLICY_ACTION_SPACE


class Pi05SO101Runtime(PolicyRobotRuntime):
    def __init__(
        self,
        robot: SO101Adapter,
        client: VvlaHttpClient,
        *,
        instruction: str,
        mapper: Pi05SO101ActionMapper | None = None,
    ) -> None:
        super().__init__(
            robot,
            client,
            instruction=instruction,
            policy_action_space=POLICY_ACTION_SPACE,
            mapper=(
                mapper
                if mapper is not None
                else Pi05SO101ActionMapper(position_mode=robot.config.position_mode)
            ),
        )


__all__ = ["Pi05SO101Runtime"]
