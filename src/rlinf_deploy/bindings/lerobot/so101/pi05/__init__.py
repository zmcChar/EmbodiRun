"""Pi0.5 value mapping for the SO-101 follower."""

from .... import BindingDefinition, BindingRun, BindingRunRequest
from .mapper import (
    POLICY_ACTION_SPACE,
    Pi05SO101Mapper,
    Pi05SO101MapperError,
)


def _build_run(request: BindingRunRequest) -> BindingRun:
    from ..request import build_run

    return build_run(request)


BINDING_DEFINITION = BindingDefinition(
    kind="lerobot.so101.pi05",
    robot_kind="lerobot.so101",
    model_kind="pi05",
    worker_module=f"{__name__.rsplit('.', 1)[0]}.worker",
    build_run=_build_run,
    mapper_factory=Pi05SO101Mapper,
)

__all__ = [
    "BINDING_DEFINITION",
    "POLICY_ACTION_SPACE",
    "Pi05SO101Mapper",
    "Pi05SO101MapperError",
]
