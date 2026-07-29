"""Group-3 provider layer above model, engine, and backend boundaries."""

from __future__ import annotations

from typing import Any

from .base import InferenceProvider, ProviderCapabilities
from .multitenant import (
    MultiTenantInferenceService,
    SessionOverloadedError,
    SessionRegistrationError,
    SessionSequenceError,
    SessionSnapshot,
    SharedModelContract,
)
from .registry import ProviderFactory, ProviderRegistry

__all__ = [
    "InferenceProvider",
    "LocalBackendProvider",
    "MultiTenantInferenceService",
    "ProviderCapabilities",
    "ProviderFactory",
    "ProviderRegistry",
    "SessionOverloadedError",
    "SessionRegistrationError",
    "SessionSequenceError",
    "SessionSnapshot",
    "SharedModelContract",
]


def __getattr__(name: str) -> Any:
    if name == "LocalBackendProvider":
        from .local import LocalBackendProvider

        return LocalBackendProvider
    raise AttributeError(name)
