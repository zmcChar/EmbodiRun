"""Pi0.5 mapping for LIBERO's relative Panda end-effector contract."""

from rlinf_deploy.bindings.franka.panda.pi05.mapper import (
    POLICY_ACTION_SPACE,
    Pi05FrankaPandaMapper,
)

from .... import BindingDefinition


MAXIMUM_CHUNK_STEPS = 50
ADAPTER_CONFIG = {
    "state_fields": ("observation.state",),
    "image_fields": (
        "observation.images.image",
        "observation.images.wrist_image",
    ),
}


BINDING_DEFINITION = BindingDefinition(
    kind="libero.panda.pi05",
    robot_kind="libero.panda.delta_eef",
    model_kind="pi05",
    mapper_factory=Pi05FrankaPandaMapper,
    maximum_chunk_steps=MAXIMUM_CHUNK_STEPS,
    adapter_config=ADAPTER_CONFIG,
)


__all__ = [
    "ADAPTER_CONFIG",
    "BINDING_DEFINITION",
    "MAXIMUM_CHUNK_STEPS",
    "POLICY_ACTION_SPACE",
    "Pi05FrankaPandaMapper",
]
