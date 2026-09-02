"""Pi0.5 inference-output binding for the SO-101 follower."""

from .... import BindingDefinition, BindingRun, BindingRunRequest
from .mapper import (
    POLICY_ACTION_SPACE,
    Pi05SO101ActionMapper,
    Pi05SO101ActionMapperError,
)
from .runtime import Pi05SO101Runtime


def _build_run(request: BindingRunRequest) -> BindingRun:
    from .spec import build_run

    return build_run(request)


BINDING_DEFINITION = BindingDefinition(
    kind="lerobot.so101.pi05",
    robot_kind="lerobot.so101",
    model_kind="pi05",
    worker_module=f"{__name__}.worker",
    build_run=_build_run,
)

__all__ = [
    "BINDING_DEFINITION",
    "POLICY_ACTION_SPACE",
    "Pi05SO101ActionMapper",
    "Pi05SO101ActionMapperError",
    "Pi05SO101Runtime",
]
