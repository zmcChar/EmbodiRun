"""Compose the calibrated SO-101 drivers as one twelve-dimensional robot."""

from __future__ import annotations

from collections.abc import Mapping

from rlinf_deploy.robots import RobotAction, RobotAdapter, RobotObservation

from ..so101 import (
    SO101_ACTION_SPACE,
    SO101_POSITION_FEATURES,
    SO101Adapter,
    SO101AdapterError,
)
from .config import BiSO101Config

BI_SO101_ACTION_SPACE = "lerobot.bi_so101.position.v1"
BI_SO101_POSITION_FEATURES = tuple(
    f"{side}_arm_{feature}"
    for side in ("left", "right")
    for feature in SO101_POSITION_FEATURES
)


class BiSO101Adapter(RobotAdapter):
    """One control owner for two arms; writes to separate buses are sequential."""

    def __init__(
        self,
        config: BiSO101Config,
        *,
        left: SO101Adapter | None = None,
        right: SO101Adapter | None = None,
    ) -> None:
        self.config = config
        self.robot_id = config.robot_id
        self.left = left if left is not None else SO101Adapter(config.left)
        self.right = right if right is not None else SO101Adapter(config.right)

    def _call_both(self, method: str) -> None:
        errors = []
        for side, arm in (("left", self.left), ("right", self.right)):
            try:
                getattr(arm, method)()
            except Exception as error:
                errors.append(f"{side}: {error}")
        if errors:
            raise SO101AdapterError(f"dual SO-101 {method} failed: {'; '.join(errors)}")

    def connect(self) -> None:
        try:
            self.left.connect()
            self.right.connect()
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise SO101AdapterError(
                    f"dual SO-101 connect failed ({error}); cleanup failed ({cleanup_error})"
                ) from error
            raise

    def observe(self) -> RobotObservation:
        left = self.left.observe()
        right = self.right.observe()
        positions = (
            *left.values["joint_positions_deg"], left.values["gripper_position"],
            *right.values["joint_positions_deg"], right.values["gripper_position"],
        )
        return RobotObservation(
            timestamp_s=min(left.timestamp_s, right.timestamp_s),
            values=dict(zip(BI_SO101_POSITION_FEATURES, positions)),
            metadata={
                "robot_id": self.robot_id,
                "robot_type": "bi_so101_follower",
                "action_space": BI_SO101_ACTION_SPACE,
                "position_units": "degrees_and_normalized_gripper",
            },
        )

    def execute(self, action: RobotAction) -> None:
        space = action.metadata.get("action_space")
        if space is not None and space != BI_SO101_ACTION_SPACE:
            raise SO101AdapterError(f"unsupported dual SO-101 action space: {space!r}")
        if not isinstance(action.values, Mapping):
            raise SO101AdapterError("dual SO-101 action values must be an object")
        if action.values == {"type": "stop"}:
            self.stop()
            return
        if action.values.get("type") != "joint_position" or set(action.values) != {
            "type", "left", "right"
        }:
            raise SO101AdapterError("dual SO-101 joint_position requires left and right targets")
        actions = []
        for side in ("left", "right"):
            values = action.values[side]
            if not isinstance(values, Mapping) or set(values) != {
                "joint_positions_deg", "gripper_position"
            }:
                raise SO101AdapterError(f"{side} target requires joints and gripper")
            actions.append(RobotAction(
                timestamp_s=action.timestamp_s,
                values={"type": "joint_position", **values},
                metadata={**action.metadata, "action_space": SO101_ACTION_SPACE},
            ))
        # Reject an invalid target on either arm before writing to either bus.
        self.left.validate_action(actions[0])
        self.right.validate_action(actions[1])
        try:
            self.left.execute(actions[0])
            self.right.execute(actions[1])
        except Exception as error:
            try:
                self.stop()
            except Exception as stop_error:
                raise SO101AdapterError(
                    f"dual SO-101 execute failed ({error}); hold failed ({stop_error})"
                ) from error
            raise

    def stop(self) -> None:
        """Attempt a measured hold on both arms, even when one bus fails."""
        self._call_both("stop")

    def close(self) -> None:
        self._call_both("close")
