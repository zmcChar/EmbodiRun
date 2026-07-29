"""Lazy in-process registry for inference-provider factories.

This registry selects a serving implementation inside one process.  Network
node registration, leases, addresses, and routing remain Group-2 concerns.
Factories are stored instead of loaded providers so listing available options
never allocates model weights or opens a remote connection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import RLock

from .base import InferenceProvider

ProviderFactory = Callable[[], InferenceProvider]


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("provider name must not be empty")
    return normalized


class ProviderRegistry:
    """Register and lazily construct local or external inference providers."""

    def __init__(
        self,
        factories: Iterable[tuple[str, ProviderFactory]] = (),
    ) -> None:
        self._lock = RLock()
        self._factories: dict[str, ProviderFactory] = {}
        for name, factory in factories:
            self.register(name, factory)

    def register(
        self,
        name: str,
        factory: ProviderFactory,
        *,
        replace: bool = False,
    ) -> None:
        normalized = _normalize_name(name)
        if not callable(factory):
            raise TypeError("provider factory must be callable")
        with self._lock:
            if normalized in self._factories and not replace:
                raise ValueError(f"provider {normalized!r} is already registered")
            self._factories[normalized] = factory

    def unregister(self, name: str) -> ProviderFactory:
        normalized = _normalize_name(name)
        with self._lock:
            try:
                return self._factories.pop(normalized)
            except KeyError as error:
                raise KeyError(f"unknown provider {name!r}") from error

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))

    def create(self, name: str) -> InferenceProvider:
        normalized = _normalize_name(name)
        with self._lock:
            try:
                factory = self._factories[normalized]
            except KeyError as error:
                available = ", ".join(sorted(self._factories)) or "<none>"
                raise KeyError(
                    f"unknown provider {name!r}; registered providers: {available}"
                ) from error

        provider = factory()
        if not isinstance(provider, InferenceProvider):
            raise TypeError(
                f"factory for {normalized!r} returned {type(provider).__name__}, "
                "which does not satisfy InferenceProvider"
            )
        return provider


__all__ = ["ProviderFactory", "ProviderRegistry"]
