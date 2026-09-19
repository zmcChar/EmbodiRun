"""Dual SO-101 follower using the existing calibrated single-arm drivers."""

from embodirun.robots import RobotDefinition

from .adapter import BI_SO101_ACTION_SPACE, BI_SO101_POSITION_FEATURES, BiSO101Adapter
from .config import BiSO101Config

ROBOT_DEFINITION = RobotDefinition(
    kind="lerobot.bi_so101",
    config_factory=BiSO101Config.from_mapping,
    adapter_type=BiSO101Adapter,
    environment_group="robot-so101",
    python="3.12",
    port_fields=("left_port", "right_port"),
)

__all__ = [
    "BI_SO101_ACTION_SPACE", "BI_SO101_POSITION_FEATURES", "BiSO101Adapter",
    "BiSO101Config", "ROBOT_DEFINITION",
]
