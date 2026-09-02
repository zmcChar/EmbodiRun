"""Pi0.5 inference-output binding for the Franka Research 3."""

from .... import BindingDefinition
from .action import Pi05ActionMapper, Pi05ActionMapperConfig, Pi05ActionMapperError
from .runtime import Pi05FR3Runtime

POLICY_FAMILY = "pi0.5"
ROBOT_MODEL = "fr3"
POLICY_ACTION_SPACE = "pi05.action_chunk.v1"
ROBOT_ACTION_SPACE = "franka.fr3.control.v1"

BINDING_DEFINITION = BindingDefinition(
    kind="franka.fr3.pi05",
    robot_kind="franka.fr3",
    model_kind="pi05",
)

__all__ = [
    "BINDING_DEFINITION",
    "POLICY_ACTION_SPACE",
    "POLICY_FAMILY",
    "ROBOT_ACTION_SPACE",
    "ROBOT_MODEL",
    "Pi05ActionMapper",
    "Pi05ActionMapperConfig",
    "Pi05ActionMapperError",
    "Pi05FR3Runtime",
]
