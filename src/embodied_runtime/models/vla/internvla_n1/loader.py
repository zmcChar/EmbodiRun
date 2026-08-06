"""Lazy construction of the official InternVLA-N1 real-world agent.

This module deliberately contains no eager InternNav, Torch, or CUDA imports.
The deployment environment may therefore import and inspect the model package
before the optional InternNav stack is installed or a GPU has been selected.
"""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

VARIANT_DUALVLN = "dualvln"
VARIANT_NAVDP = "navdp"
VARIANTS = (VARIANT_DUALVLN, VARIANT_NAVDP)

DEFAULT_MODEL_IDS = {
    VARIANT_DUALVLN: "InternRobotics/InternVLA-N1-DualVLN",
    VARIANT_NAVDP: "InternRobotics/InternVLA-N1-w-NavDP",
}

DEFAULT_CAMERA_INTRINSIC = (
    (386.5, 0.0, 328.9, 0.0),
    (0.0, 386.5, 244.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


class InternVLALoadError(RuntimeError):
    """The optional official implementation could not be constructed."""


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """Static sensor and System-1 facts for one published checkpoint."""

    name: str
    model_id: str
    depth_required: bool
    system1: str


VARIANT_SPECS = {
    VARIANT_DUALVLN: VariantSpec(
        name=VARIANT_DUALVLN,
        model_id=DEFAULT_MODEL_IDS[VARIANT_DUALVLN],
        depth_required=False,
        system1="nextdit_async_rgb",
    ),
    VARIANT_NAVDP: VariantSpec(
        name=VARIANT_NAVDP,
        model_id=DEFAULT_MODEL_IDS[VARIANT_NAVDP],
        depth_required=True,
        system1="navdp_rgbd",
    ),
}


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _camera_matrix(value: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(float(item) for item in row) for row in value)
    except (TypeError, ValueError) as error:
        raise ValueError("camera_intrinsic must be a finite 4x4 matrix") from error
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("camera_intrinsic must be a finite 4x4 matrix")
    if any(not math.isfinite(item) for row in rows for item in row):
        raise ValueError("camera_intrinsic must be a finite 4x4 matrix")
    return rows


@dataclass(frozen=True, slots=True)
class InternVLAConfig:
    """Arguments shared by the loader and the official real-world agent."""

    variant: str = VARIANT_DUALVLN
    model_path: str | None = None
    device: str = "cuda:0"
    resize_w: int = 384
    resize_h: int = 384
    num_history: int = 8
    plan_step_gap: int = 4
    camera_intrinsic: Sequence[Sequence[float]] = DEFAULT_CAMERA_INTRINSIC
    internnav_root: str | Path | None = None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}")
        if self.model_path is not None and not str(self.model_path).strip():
            raise ValueError("model_path must not be empty")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        for name in ("resize_w", "resize_h", "num_history", "plan_step_gap"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        object.__setattr__(self, "camera_intrinsic", _camera_matrix(self.camera_intrinsic))
        if self.internnav_root is not None:
            object.__setattr__(self, "internnav_root", Path(self.internnav_root).expanduser())

    @property
    def spec(self) -> VariantSpec:
        return VARIANT_SPECS[self.variant]

    @property
    def resolved_model_path(self) -> str:
        return self.model_path or self.spec.model_id


AgentFactory = Callable[[object], object]


def resolve_official_agent_factory(
    internnav_root: str | Path | None = None,
) -> AgentFactory:
    """Resolve ``InternVLAN1AsyncAgent`` only when loading is requested."""

    if internnav_root is not None:
        root = str(Path(internnav_root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        module = importlib.import_module("internnav.agent.internvla_n1_agent_realworld")
        factory = module.InternVLAN1AsyncAgent
    except Exception as error:
        raise InternVLALoadError(
            "cannot import the InternNav real-world agent; install InternNav in "
            f"this model environment: {error}"
        ) from error
    if not callable(factory):
        raise InternVLALoadError("InternNav's InternVLAN1AsyncAgent is not callable")
    return factory


def build_official_agent_args(config: InternVLAConfig) -> SimpleNamespace:
    """Translate stable runtime configuration into InternNav's argument object."""

    try:
        numpy = importlib.import_module("numpy")
    except ImportError as error:
        raise InternVLALoadError("InternVLA-N1 loading requires NumPy") from error
    return SimpleNamespace(
        device=config.device,
        model_path=config.resolved_model_path,
        resize_w=config.resize_w,
        resize_h=config.resize_h,
        num_history=config.num_history,
        plan_step_gap=config.plan_step_gap,
        camera_intrinsic=numpy.asarray(config.camera_intrinsic, dtype=numpy.float32),
    )


def load_internvla_agent(
    config: InternVLAConfig,
    *,
    agent_factory: AgentFactory | None = None,
) -> Any:
    """Construct and reset the official stateful agent.

    ``agent_factory`` is an injection seam for CPU-only tests. Production uses
    the lazily imported official class.
    """

    factory = agent_factory or resolve_official_agent_factory(config.internnav_root)
    try:
        agent = factory(build_official_agent_args(config))
        if not callable(getattr(agent, "reset", None)):
            raise TypeError("the official agent does not expose reset()")
        if not callable(getattr(agent, "step", None)):
            raise TypeError("the official agent does not expose step()")
        agent.reset()
    except InternVLALoadError:
        raise
    except Exception as error:
        raise InternVLALoadError(f"cannot load InternVLA-N1: {error}") from error
    return agent


__all__ = [
    "DEFAULT_CAMERA_INTRINSIC",
    "DEFAULT_MODEL_IDS",
    "VARIANTS",
    "VARIANT_DUALVLN",
    "VARIANT_NAVDP",
    "VARIANT_SPECS",
    "AgentFactory",
    "InternVLAConfig",
    "InternVLALoadError",
    "VariantSpec",
    "build_official_agent_args",
    "load_internvla_agent",
    "resolve_official_agent_factory",
]
