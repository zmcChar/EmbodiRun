"""Map VVLA pi0.5 action chunks to FR3 joint commands."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from rlinf_deploy.bindings import ActionMappingError
from rlinf_deploy.bindings.franka.fr3.pi05.contract import POLICY_ACTION_SPACE
from rlinf_deploy.inference import PolicyResult
from rlinf_deploy.robots import RobotAction
from rlinf_deploy.robots.franka.fr3 import FR3_ACTION_SPACE


class Pi05ActionMapperError(ActionMappingError):
    pass


@dataclass(frozen=True)
class Pi05ActionMapperConfig:
    joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    gripper_index: int | None = None

    def __post_init__(self) -> None:
        for index in self.joint_indices:
            if not isinstance(index, int) or index < 0 or index > 31:
                raise Pi05ActionMapperError("joint_indices must be ints in [0,31]")
        if self.gripper_index is not None and (
            not isinstance(self.gripper_index, int)
            or self.gripper_index < 0
            or self.gripper_index > 31
        ):
            raise Pi05ActionMapperError(
                "gripper_index must be an int in [0,31] or None"
            )


class Pi05ActionMapper:
    """Convert a single pi05 action chunk to one FR3 joint command."""

    def __init__(self, *, config: Pi05ActionMapperConfig | None = None) -> None:
        self.config = config or Pi05ActionMapperConfig()

    def map_result(self, result: PolicyResult) -> RobotAction:
        if result.action_space != POLICY_ACTION_SPACE:
            raise Pi05ActionMapperError(
                f"policy action_space mismatch: got {result.action_space!r}, expected {POLICY_ACTION_SPACE!r}"
            )
        if not result.actions:
            raise Pi05ActionMapperError("result contains no action rows")
        raw = result.actions[0]
        if raw.kind != "action_chunk":
            raise Pi05ActionMapperError(f"unsupported action kind {raw.kind!r}")
        data = raw.values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise Pi05ActionMapperError("action_chunk values.data must be a sequence")
        if not data:
            raise Pi05ActionMapperError("action_chunk values.data must not be empty")
        first = data[0]
        if isinstance(first, (str, bytes)) or not isinstance(first, Sequence):
            raise Pi05ActionMapperError(
                "action_chunk first row must be a numeric sequence"
            )

        row: list[float] = []
        for item in first:
            if not isinstance(item, (int, float)):
                if isinstance(item, bool):
                    raise Pi05ActionMapperError("action values must be numeric")
                try:
                    item = float(item)
                except (TypeError, ValueError):
                    raise Pi05ActionMapperError(
                        "action values must be numeric"
                    ) from None
            row.append(float(item))
            if not math.isfinite(row[-1]):
                raise Pi05ActionMapperError("action values must be finite")
        max_index = max(
            self.config.joint_indices
            + (
                (self.config.gripper_index,)
                if self.config.gripper_index is not None
                else ()
            )
        )
        if len(row) <= max_index:
            raise Pi05ActionMapperError(
                "action_chunk row is too short for configured indices"
            )

        robot = {
            "type": "joint_position",
            "joint_positions_rad": [row[index] for index in self.config.joint_indices],
        }
        if self.config.gripper_index is not None:
            robot["gripper_width_m"] = row[self.config.gripper_index]

        return RobotAction(
            timestamp_s=time.time(),
            values=robot,
            metadata={
                "action_space": FR3_ACTION_SPACE,
                "request_id": result.request_id,
                "session_id": result.session_id,
                "step_id": result.step_id,
                "session_revision": result.session_revision,
            },
        )


__all__ = ["Pi05ActionMapper", "Pi05ActionMapperConfig", "Pi05ActionMapperError"]
