"""Compatibility identity for the Pi0.5-to-FR3 binding."""

POLICY_FAMILY = "pi0.5"
ROBOT_MODEL = "fr3"
ROBOT_ACTION_SPACE = "franka.fr3.control.v1"
POLICY_ACTION_SPACE = "pi05.action_chunk.v1"
ACTION_SPACE = ROBOT_ACTION_SPACE

__all__ = [
    "POLICY_ACTION_SPACE",
    "POLICY_FAMILY",
    "ROBOT_ACTION_SPACE",
    "ACTION_SPACE",
    "ROBOT_MODEL",
]
