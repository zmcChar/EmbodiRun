"""Map named Pi0.5 action chunks to SO-101 position commands."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from rlinf_deploy.inference import (
    ImagePayload,
    PolicyObservation,
    PolicyResult,
)
from rlinf_deploy.robots import RobotAction, RobotObservation
from rlinf_deploy.robots.lerobot.so101 import (
    SO101_ACTION_SPACE,
    SO101_POSITION_FEATURES,
)
from rlinf_deploy.robots.sensors.cameras import CameraFrame


POLICY_ACTION_SPACE = "pi05.action_chunk.v1"


class Pi05SO101MapperError(RuntimeError):
    pass


class Pi05SO101Mapper:
    """Translate between SO-101 observations and Pi0.5 policy values."""

    policy_action_space = POLICY_ACTION_SPACE

    def map_observation(
        self,
        observation: RobotObservation,
        *,
        session_id: str,
        request_id: str,
        step_id: int,
        instruction: str,
        frames: Sequence[CameraFrame],
    ) -> PolicyObservation:
        return PolicyObservation(
            session_id=session_id,
            request_id=request_id,
            step_id=step_id,
            instruction=instruction,
            state=dict(observation.values),
            images=tuple(
                ImagePayload(frame.name, frame.mime_type, frame.data)
                for frame in frames
            ),
            reset=False,
            metadata={"robot_timestamp_s": observation.timestamp_s},
        )

    def map_result(self, result: PolicyResult) -> tuple[RobotAction, ...]:
        if result.action_space != self.policy_action_space:
            raise Pi05SO101MapperError(
                f"policy action_space mismatch: got {result.action_space!r}, "
                f"expected {self.policy_action_space!r}"
            )
        if len(result.actions) != 1:
            raise Pi05SO101MapperError("SO-101 expects exactly one policy action")
        raw = result.actions[0]
        if raw.kind != "action_chunk":
            raise Pi05SO101MapperError(f"unsupported action kind {raw.kind!r}")
        data = raw.values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise Pi05SO101MapperError(
                "action_chunk values.data must be a sequence"
            )
        if not data:
            raise Pi05SO101MapperError("action_chunk values.data must not be empty")
        declared = raw.values.get("feature_names")
        if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
            raise Pi05SO101MapperError(
                "action_chunk feature_names must be a sequence"
            )
        names = tuple(declared)
        if any(not isinstance(name, str) or not name for name in names):
            raise Pi05SO101MapperError(
                "action_chunk feature_names must be strings"
            )
        if len(names) != len(set(names)):
            raise Pi05SO101MapperError(
                "action_chunk feature_names must be unique"
            )
        actual_names = set(names)
        expected_names = set(SO101_POSITION_FEATURES)
        if actual_names != expected_names:
            missing = expected_names - actual_names
            unexpected = actual_names - expected_names
            raise Pi05SO101MapperError(
                "action features do not match SO-101; "
                f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
            )
        actions: list[RobotAction] = []
        for row_index, row in enumerate(data):
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise Pi05SO101MapperError(
                    f"action_chunk row {row_index} must be a numeric sequence"
                )
            if len(row) != len(names):
                raise Pi05SO101MapperError(
                    f"action_chunk row {row_index} and feature_names lengths differ"
                )
            by_name: dict[str, float] = {}
            for name, item in zip(names, row):
                if isinstance(item, bool):
                    raise Pi05SO101MapperError("action values must be numeric")
                try:
                    value = float(item)
                except (TypeError, ValueError):
                    raise Pi05SO101MapperError(
                        "action values must be numeric"
                    ) from None
                if not math.isfinite(value):
                    raise Pi05SO101MapperError("action values must be finite")
                by_name[name] = value
            positions = [by_name[name] for name in SO101_POSITION_FEATURES]
            actions.append(
                RobotAction(
                    timestamp_s=time.time(),
                    values={
                        "type": "joint_position",
                        "joint_positions_deg": positions[:-1],
                        "gripper_position": positions[-1],
                    },
                    metadata={
                        "action_space": SO101_ACTION_SPACE,
                        "request_id": result.request_id,
                        "session_id": result.session_id,
                        "step_id": result.step_id,
                        "session_revision": result.session_revision,
                        "chunk_index": row_index,
                        "chunk_size": len(data),
                    },
                )
            )
        return tuple(actions)


__all__ = [
    "POLICY_ACTION_SPACE",
    "Pi05SO101Mapper",
    "Pi05SO101MapperError",
]
