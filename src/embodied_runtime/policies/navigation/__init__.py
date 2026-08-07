"""Canonical visual-navigation policy implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from embodied_runtime.tasks.navigation.interfaces import NavigationPolicy

__all__ = [
    "EpisodeCursor",
    "InternVLANavigationPolicy",
    "NavigationPolicy",
    "NavigationPolicyError",
    "QwenNavigationPolicy",
    "StreamVLNNavigationPolicy",
]

_SYMBOL_MODULES = {
    "EpisodeCursor": ".episode",
    "NavigationPolicyError": ".errors",
    "InternVLANavigationPolicy": ".internvla.policy",
    "QwenNavigationPolicy": ".qwen.policy",
    "StreamVLNNavigationPolicy": ".streamvln.policy",
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
