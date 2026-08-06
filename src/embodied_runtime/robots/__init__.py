"""Group 5: the last mile between model actions and physical robots."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ActionMapper", "ObservationMapper", "RobotProfile"]

_SYMBOL_MODULES = {
    "ActionMapper": ".action",
    "ObservationMapper": ".observation",
    "RobotProfile": ".profile",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
