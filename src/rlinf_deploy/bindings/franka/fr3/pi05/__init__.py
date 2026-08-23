"""Pi0.5 inference-output binding for the Franka Research 3."""

from .runtime import Pi05FR3Runtime
from .action import Pi05ActionMapper, Pi05ActionMapperConfig, Pi05ActionMapperError

POLICY_FAMILY = "pi0.5"
ROBOT_MODEL = "fr3"
POLICY_ACTION_SPACE = "pi05.action_chunk.v1"
ROBOT_ACTION_SPACE = "franka.fr3.control.v1"

__all__ = [
    "POLICY_FAMILY",
    "POLICY_ACTION_SPACE",
    "ROBOT_ACTION_SPACE",
    "ROBOT_MODEL",
    "Pi05FR3Runtime",
    "Pi05ActionMapper",
    "Pi05ActionMapperConfig",
    "Pi05ActionMapperError",
]
