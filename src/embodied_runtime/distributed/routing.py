"""Routing policy boundary; concrete discovery systems plug in later."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from embodied_runtime.engine.request import InferenceRequest

from .discovery import RuntimeEndpoint


@runtime_checkable
class EndpointRouter(Protocol):
    def select(
        self,
        request: InferenceRequest,
        candidates: Sequence[RuntimeEndpoint],
    ) -> RuntimeEndpoint: ...
