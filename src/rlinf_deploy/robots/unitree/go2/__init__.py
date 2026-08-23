"""Canonical Unitree Go2 hardware bindings and robot-resident agent.

The host-only bindings stay lazy because ``agent`` is deployed to the Go2's
Python 3.8 environment and must be importable without the host task stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Go2CameraClient",
    "Go2CameraError",
    "Go2ClientError",
    "Go2ControlClient",
]

_SYMBOL_MODULES = {
    "Go2CameraClient": ".camera",
    "Go2CameraError": ".camera",
    "Go2ClientError": ".client",
    "Go2ControlClient": ".client",
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
