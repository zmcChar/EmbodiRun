"""Execution context passed from the engine to hardware backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event

from embodied_runtime.types import Metadata


@dataclass(slots=True)
class ExecutionContext:
    request_ids: tuple[str, ...]
    stage: str
    step_index: int | None = None
    total_steps: int | None = None
    cancellation: Event = field(default_factory=Event)
    metadata: Metadata = field(default_factory=dict)

    @property
    def cancelled(self) -> bool:
        return self.cancellation.is_set()


__all__ = ["ExecutionContext"]
