"""Map named Pi0.5 action chunks to SO-101 position commands."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from rlinf_deploy.inference import PolicyResult
from rlinf_deploy.robots import RobotAction
from rlinf_deploy.robots.lerobot.so101 import (
    SO101_ACTION_SPACE,
    SO101_POSITION_FEATURES,
)


POLICY_ACTION_SPACE = "pi05.action_chunk.v1"


class Pi05SO101ActionMapperError(RuntimeError):
    pass


class Pi05SO101ActionMapper:
    """Map by declared feature name so checkpoint order cannot move the wrong joint."""

    policy_action_space = POLICY_ACTION_SPACE

    def map_result(self, result: PolicyResult) -> RobotAction:
        if result.action_space != self.policy_action_space:
            raise Pi05SO101ActionMapperError(
                f"policy action_space mismatch: got {result.action_space!r}, "
                f"expected {self.policy_action_space!r}"
            )
        if len(result.actions) != 1:
            raise Pi05SO101ActionMapperError("SO-101 expects exactly one policy action")
        raw = result.actions[0]
        if raw.kind != "action_chunk":
            raise Pi05SO101ActionMapperError(f"unsupported action kind {raw.kind!r}")
        data = raw.values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence):
            raise Pi05SO101ActionMapperError(
                "action_chunk values.data must be a sequence"
            )
        if len(data) != 1:
            raise Pi05SO101ActionMapperError(
                "SO-101 runtime expects exactly one action row"
            )
        first = data[0]
        if isinstance(first, (str, bytes)) or not isinstance(first, Sequence):
            raise Pi05SO101ActionMapperError(
                "action_chunk first row must be a numeric sequence"
            )
        declared = raw.values.get("feature_names")
        if isinstance(declared, (str, bytes)) or not isinstance(declared, Sequence):
            raise Pi05SO101ActionMapperError(
                "action_chunk feature_names must be a sequence"
            )
        names = tuple(declared)
        if any(not isinstance(name, str) or not name for name in names):
            raise Pi05SO101ActionMapperError(
                "action_chunk feature_names must be strings"
            )
        if len(names) != len(set(names)):
            raise Pi05SO101ActionMapperError(
                "action_chunk feature_names must be unique"
            )
        if len(first) != len(names):
            raise Pi05SO101ActionMapperError(
                "action row and feature_names lengths differ"
            )
        actual_names = set(names)
        expected_names = set(SO101_POSITION_FEATURES)
        if actual_names != expected_names:
            missing = expected_names - actual_names
            unexpected = actual_names - expected_names
            raise Pi05SO101ActionMapperError(
                "action features do not match SO-101; "
                f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
            )
        by_name: dict[str, float] = {}
        for name, item in zip(names, first):
            if isinstance(item, bool):
                raise Pi05SO101ActionMapperError("action values must be numeric")
            try:
                value = float(item)
            except (TypeError, ValueError):
                raise Pi05SO101ActionMapperError(
                    "action values must be numeric"
                ) from None
            if not math.isfinite(value):
                raise Pi05SO101ActionMapperError("action values must be finite")
            by_name[name] = value
        positions = [by_name[name] for name in SO101_POSITION_FEATURES]
        return RobotAction(
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
            },
        )


__all__ = [
    "POLICY_ACTION_SPACE",
    "Pi05SO101ActionMapper",
    "Pi05SO101ActionMapperError",
]
