from __future__ import annotations

from dataclasses import dataclass, field

from embodirun.types import Metadata


@dataclass(frozen=True, slots=True)
class RobotProfile:
    """What a robot adapter declares about itself.

    A profile names the robot and its type, the observation keys it publishes, the action
    space it accepts, and the rate it is controlled at, so a deployment can be checked
    against the hardware before anything moves. ``safety_capabilities`` is where an adapter
    lists the safety features it implements, and ``metadata`` carries adapter-specific
    detail.
    """

    robot_id: str
    robot_type: str
    observation_keys: tuple[str, ...]
    action_space: str
    control_hz: float
    safety_capabilities: frozenset[str] = frozenset()
    metadata: Metadata = field(default_factory=dict)
