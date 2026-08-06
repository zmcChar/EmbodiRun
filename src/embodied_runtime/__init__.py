"""Hardware-neutral runtime for embodied models and robot deployment.

Public symbols are imported lazily so the small robot-resident agents can run
without importing the host-side model stack (and its newer Python features).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ActionChunk",
    "ExecutionPlan",
    "FlowRecipe",
    "InferenceRequest",
    "IterativeFlowPlan",
    "ModelPackage",
    "ModelSpec",
    "SingleForwardPlan",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".contracts", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
