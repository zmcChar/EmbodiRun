from __future__ import annotations

from dataclasses import dataclass, field

from embodied_runtime.types import Metadata, TensorTree


@dataclass(slots=True)
class RobotAction:
    """Timestamped command in one robot profile's declared action space."""

    timestamp_s: float
    values: TensorTree
    metadata: Metadata = field(default_factory=dict)


__all__ = ["RobotAction"]
