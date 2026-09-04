"""LIBERO simulator adapter."""

from .. import SimulatorDefinition
from .adapter import LIBERO_ACTION_DIM, LiberoAdapter
from .config import LIBERO_SUITES, LiberoConfig


SIMULATOR_DEFINITION = SimulatorDefinition(
    kind="libero",
    embodiment_kind="libero.panda.delta_eef",
    config_factory=LiberoConfig.from_mapping,
    adapter_type=LiberoAdapter,
    environment_group="sim-libero",
    python="3.12",
)


__all__ = [
    "LIBERO_ACTION_DIM",
    "LIBERO_SUITES",
    "SIMULATOR_DEFINITION",
    "LiberoAdapter",
    "LiberoConfig",
]
