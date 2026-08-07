"""Lifecycle extension required from policies owned by this application."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from embodied_runtime.tasks.navigation import NavigationPolicy


@runtime_checkable
class ManagedNavigationPolicy(NavigationPolicy, Protocol):
    async def prepare(self) -> None: ...

    async def aclose(self) -> None: ...


__all__ = ["ManagedNavigationPolicy"]
