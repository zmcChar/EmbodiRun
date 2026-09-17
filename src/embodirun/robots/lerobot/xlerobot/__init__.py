"""XLeRobot external-owner proxy."""

from ... import RobotDefinition
from .adapter import XLeRobotAdapter, XLeRobotAdapterError, XLEROBOT_ACTION_SPACE
from .config import XLeRobotConfig

ROBOT_DEFINITION = RobotDefinition(
    kind="lerobot.xlerobot",
    config_factory=XLeRobotConfig.from_mapping,
    adapter_type=XLeRobotAdapter,
    environment_group="robot-xlerobot-external-owner",
    python=None,
)

__all__ = ["ROBOT_DEFINITION", "XLeRobotAdapter", "XLeRobotAdapterError", "XLeRobotConfig", "XLEROBOT_ACTION_SPACE"]
