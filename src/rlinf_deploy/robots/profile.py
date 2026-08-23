from __future__ import annotations

from dataclasses import dataclass, field

from rlinf_deploy.types import Metadata


@dataclass(frozen=True, slots=True)
class RobotProfile:
    robot_id: str
    robot_type: str
    observation_keys: tuple[str, ...]
    action_space: str
    control_hz: float
    safety_capabilities: frozenset[str] = frozenset()
    metadata: Metadata = field(default_factory=dict)
