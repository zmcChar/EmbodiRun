"""Input envelope accepted by inference endpoints."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from embodied_runtime.types import Metadata

PayloadT = TypeVar("PayloadT")


@dataclass(slots=True)
class InferenceRequest(Generic[PayloadT]):
    payload: PayloadT
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    priority: int = 0
    deadline_s: float | None = None
    num_steps: int | None = None
    seed: int | None = None
    metadata: Metadata = field(default_factory=dict)
    created_at_s: float = field(default_factory=time.monotonic)


__all__ = ["InferenceRequest"]
