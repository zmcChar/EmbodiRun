"""Map named Pi0.5 action chunks to SO-101 position commands."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from rlinf_deploy.bindings.runtime import ActionMappingError
from rlinf_deploy.inference import PolicyResult
from rlinf_deploy.robots import RobotAction
from rlinf_deploy.robots.lerobot.so101 import (
    SO101_ACTION_SPACE,
    SO101_POSITION_FEATURES,
)

from .contract import POLICY_ACTION_SPACE


class Pi05SO101ActionMapperError(ActionMappingError):
    pass


@dataclass(frozen=True, slots=True)
class Pi05SO101ActionMapperConfig:
    feature_names: tuple[str, ...] = SO101_POSITION_FEATURES

    def __post_init__(self) -> None:
        if len(self.feature_names) != 6:
            raise Pi05SO101ActionMapperError("feature_names must contain six names")
        if any(
            not isinstance(name, str) or not name.strip() for name in self.feature_names
        ):
            raise Pi05SO101ActionMapperError(
                "feature_names must contain non-empty strings"
            )
        if len(set(self.feature_names)) != len(self.feature_names):
            raise Pi05SO101ActionMapperError("feature_names must be unique")


class Pi05SO101ActionMapper:
    """Map by declared feature name so checkpoint order cannot move the wrong joint."""

    def __init__(self, *, config: Pi05SO101ActionMapperConfig | None = None) -> None:
        self.config = config or Pi05SO101ActionMapperConfig()

    def map_result(self, result: PolicyResult) -> RobotAction:
        if result.action_space != POLICY_ACTION_SPACE:
            raise Pi05SO101ActionMapperError(
                f"policy action_space mismatch: got {result.action_space!r}, "
                f"expected {POLICY_ACTION_SPACE!r}"
            )
        raw = result.actions[0]
        if raw.kind != "action_chunk":
            raise Pi05SO101ActionMapperError(f"unsupported action kind {raw.kind!r}")
        data = raw.values.get("data")
        if isinstance(data, (str, bytes)) or not isinstance(data, Sequence) or not data:
            raise Pi05SO101ActionMapperError(
                "action_chunk values.data must be a non-empty sequence"
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
        missing = set(self.config.feature_names) - set(names)
        if missing:
            raise Pi05SO101ActionMapperError(
                f"action chunk is missing required features: {sorted(missing)!r}"
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
        positions = [by_name[name] for name in self.config.feature_names]
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
    "Pi05SO101ActionMapper",
    "Pi05SO101ActionMapperConfig",
    "Pi05SO101ActionMapperError",
]
