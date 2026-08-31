"""Connect a Franka adapter to a remote VVLA policy session."""

from __future__ import annotations

from rlinf_deploy.inference import VvlaHttpClient
from rlinf_deploy.robots.franka.fr3.adapter import FR3Adapter, FR3AdapterError
from rlinf_deploy.runtime import PolicyRobotRuntime

from .action import Pi05ActionMapper
from .contract import POLICY_ACTION_SPACE


class Pi05FR3Runtime(PolicyRobotRuntime):
    def __init__(
        self,
        robot: FR3Adapter,
        client: VvlaHttpClient,
        *,
        instruction: str,
        mapper: Pi05ActionMapper | None = None,
    ) -> None:
        super().__init__(
            robot,
            client,
            instruction=instruction,
            policy_action_space=POLICY_ACTION_SPACE,
            mapper=mapper or Pi05ActionMapper(),
            mapping_error=lambda error: FR3AdapterError(
                f"action mapping failed: {error}"
            ),
        )


__all__ = ["Pi05FR3Runtime"]
