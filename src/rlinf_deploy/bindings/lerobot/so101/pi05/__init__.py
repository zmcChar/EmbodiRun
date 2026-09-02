"""Pi0.5 inference-output binding for the SO-101 follower."""

from .... import BindingDefinition, BindingRun, BindingRunRequest
from .action import (
    Pi05SO101ActionMapper,
    Pi05SO101ActionMapperError,
)
from .contract import (
    ACTION_SPACE,
    POLICY_ACTION_SPACE,
    POLICY_FAMILY,
    ROBOT_ACTION_SPACE,
    ROBOT_MODEL,
)
from .runtime import Pi05SO101Runtime


def _build_run(request: BindingRunRequest) -> BindingRun:
    from .runner import build_run

    return build_run(request)


BINDING_DEFINITION = BindingDefinition(
    kind="lerobot.so101.pi05",
    robot_kind="lerobot.so101",
    model_kind="pi05",
    runner_module=f"{__name__}.runner",
    build_run=_build_run,
)

__all__ = [
    "ACTION_SPACE",
    "BINDING_DEFINITION",
    "POLICY_ACTION_SPACE",
    "POLICY_FAMILY",
    "ROBOT_ACTION_SPACE",
    "ROBOT_MODEL",
    "Pi05SO101ActionMapper",
    "Pi05SO101ActionMapperError",
    "Pi05SO101Runtime",
]
