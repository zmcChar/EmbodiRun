"""Pi0.5 inference-output binding for the Franka Research 3."""

from .runtime import Pi05FR3Runtime

POLICY_FAMILY = "pi0.5"
ROBOT_MODEL = "fr3"

__all__ = ["POLICY_FAMILY", "ROBOT_MODEL", "Pi05FR3Runtime"]
