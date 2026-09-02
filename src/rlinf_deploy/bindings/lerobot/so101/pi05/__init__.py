"""Pi0.5 value mapping for the SO-101 follower."""

from .... import BindingDefinition
from .mapper import (
    POLICY_ACTION_SPACE,
    Pi05SO101Mapper,
    Pi05SO101MapperError,
)


BINDING_DEFINITION = BindingDefinition(
    kind="lerobot.so101.pi05",
    robot_kind="lerobot.so101",
    model_kind="pi05",
    mapper_factory=Pi05SO101Mapper,
)

__all__ = [
    "BINDING_DEFINITION",
    "POLICY_ACTION_SPACE",
    "Pi05SO101Mapper",
    "Pi05SO101MapperError",
]
