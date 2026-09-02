"""Fail-closed SO-101 follower execution through the LeRobot API."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ...action import RobotAction
from ...observation import RobotObservation
from .config import SO101Config

SO101_ACTION_SPACE = "lerobot.so101.position.v1"
SO101_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
SO101_MOTORS = (*SO101_JOINTS, "gripper")
SO101_POSITION_FEATURES = tuple(f"{name}.pos" for name in SO101_MOTORS)


class SO101AdapterError(RuntimeError):
    pass


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise SO101AdapterError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise SO101AdapterError(f"{name} must be numeric") from None
    if not math.isfinite(result):
        raise SO101AdapterError(f"{name} must be finite")
    return result


def _numbers(value: object, name: str, length: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SO101AdapterError(f"{name} must be a sequence")
    if len(value) != length:
        raise SO101AdapterError(f"{name} must contain {length} values")
    return tuple(_number(item, f"{name}[{index}]") for index, item in enumerate(value))


class SO101Adapter:
    """Synchronous adapter for one calibrated SO-101 follower arm."""

    def __init__(
        self, config: SO101Config, *, lerobot_robot: Any | None = None
    ) -> None:
        self.config = config
        self.robot_id = config.robot_id
        if lerobot_robot is None:
            try:
                from lerobot.robots.so_follower import (
                    SO101Follower,
                    SO101FollowerConfig,
                )
            except ImportError as error:
                raise SO101AdapterError(
                    "SO-101 support requires LeRobot with Feetech support; "
                    "run `uv sync --python 3.12 --frozen --no-dev "
                    "--group robot-so101`"
                ) from error
            options: dict[str, object] = {
                "port": config.port,
                "id": config.calibration_id or config.robot_id,
                "disable_torque_on_disconnect": config.disable_torque_on_disconnect,
                "use_degrees": True,
                # This adapter rejects oversized steps instead of allowing the SDK to clip them.
                "max_relative_target": None,
                "cameras": {},
            }
            if config.calibration_dir is not None:
                options["calibration_dir"] = config.calibration_dir
            lerobot_robot = SO101Follower(SO101FollowerConfig(**options))
        self.robot = lerobot_robot
        try:
            self.robot.connect(calibrate=False)
        except BaseException:
            if self.robot.is_connected:
                self.robot.disconnect()
            raise
        if not self.robot.is_calibrated:
            self.robot.disconnect()
            raise SO101AdapterError(
                "SO-101 is not calibrated; run lerobot-calibrate with --robot.id "
                f"{config.calibration_id or config.robot_id!r}"
            )

    def _read_positions(self) -> tuple[float, ...]:
        raw = self.robot.get_observation()
        if not isinstance(raw, Mapping):
            raise SO101AdapterError("SO-101 observation must be an object")
        missing = set(SO101_POSITION_FEATURES) - raw.keys()
        if missing:
            raise SO101AdapterError(
                f"SO-101 observation is missing features: {sorted(missing)!r}"
            )
        return tuple(_number(raw[name], name) for name in SO101_POSITION_FEATURES)

    def observe(self) -> RobotObservation:
        positions = self._read_positions()
        return RobotObservation(
            timestamp_s=time.time(),
            values={
                "joint_positions_deg": list(positions[:-1]),
                "gripper_position": positions[-1],
            },
            metadata={
                "robot_id": self.robot_id,
                "robot_type": "so101_follower",
                "action_space": SO101_ACTION_SPACE,
                "position_units": "degrees_and_normalized_gripper",
            },
        )

    def execute(self, action: RobotAction) -> None:
        declared_space = action.metadata.get("action_space")
        if declared_space is not None and declared_space != SO101_ACTION_SPACE:
            raise SO101AdapterError(
                f"unsupported action space {declared_space!r}; expected {SO101_ACTION_SPACE!r}"
            )
        if not isinstance(action.values, Mapping):
            raise SO101AdapterError("SO-101 action values must be an object")
        values = dict(action.values)
        kind = values.pop("type", None)
        if kind == "stop":
            if values:
                raise SO101AdapterError("stop action must not contain parameters")
            self.stop()
            return
        if kind != "joint_position":
            raise SO101AdapterError(f"unsupported SO-101 action type: {kind!r}")
        expected_fields = {"joint_positions_deg", "gripper_position"}
        if set(values) != expected_fields:
            missing = expected_fields - values.keys()
            unexpected = values.keys() - expected_fields
            raise SO101AdapterError(
                "SO-101 joint_position fields do not match the contract; "
                f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
            )
        target_joints = _numbers(
            values["joint_positions_deg"], "joint_positions_deg", 5
        )
        target_gripper = _number(values["gripper_position"], "gripper_position")
        if not 0.0 <= target_gripper <= 100.0:
            raise SO101AdapterError("gripper_position must be in [0, 100]")
        current = self._read_positions()
        maximum_joint_step = max(
            abs(target - present)
            for target, present in zip(target_joints, current[:-1])
        )
        if maximum_joint_step > self.config.max_joint_step_deg:
            raise SO101AdapterError(
                f"joint step {maximum_joint_step:.6f} exceeds "
                f"{self.config.max_joint_step_deg:.6f} degrees"
            )
        gripper_step = abs(target_gripper - current[-1])
        if gripper_step > self.config.max_gripper_step:
            raise SO101AdapterError(
                f"gripper step {gripper_step:.6f} exceeds "
                f"{self.config.max_gripper_step:.6f}"
            )
        command = {
            feature: value
            for feature, value in zip(
                SO101_POSITION_FEATURES,
                (*target_joints, target_gripper),
            )
        }
        self.robot.send_action(command)

    def stop(self) -> None:
        """Hold the measured pose; SO-101 exposes no separate stop primitive."""
        positions = self._read_positions()
        self.robot.send_action(dict(zip(SO101_POSITION_FEATURES, positions)))

    def close(self) -> None:
        self.robot.disconnect()


__all__ = [
    "SO101_ACTION_SPACE",
    "SO101_JOINTS",
    "SO101_MOTORS",
    "SO101_POSITION_FEATURES",
    "SO101Adapter",
    "SO101AdapterError",
]
