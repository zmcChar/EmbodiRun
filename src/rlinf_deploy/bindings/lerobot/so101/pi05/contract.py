"""Compatibility identity for the Pi0.5-to-SO-101 binding."""

POLICY_FAMILY = "pi05"
POLICY_ACTION_SPACE = "pi05.action_chunk.v1"
ROBOT_MODEL = "so101_follower"
ROBOT_ACTION_SPACE = "lerobot.so101.position.v1"

ACTION_SPACE = ROBOT_ACTION_SPACE

__all__ = [
    "ACTION_SPACE",
    "POLICY_ACTION_SPACE",
    "POLICY_FAMILY",
    "ROBOT_ACTION_SPACE",
    "ROBOT_MODEL",
]
