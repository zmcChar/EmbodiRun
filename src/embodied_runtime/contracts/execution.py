"""Request lifecycle and backend execution context."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Event

from embodied_runtime._compat import StrEnum

from .model import RawRequest
from .types import Metadata, TensorTree


class RequestStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class InferenceRequest:
    """One engine request before model-specific preprocessing."""

    payload: RawRequest | TensorTree
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    priority: int = 0
    deadline_s: float | None = None
    num_steps: int | None = None
    seed: int | None = None
    metadata: Metadata = field(default_factory=dict)
    created_at_s: float = field(default_factory=time.monotonic)


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


@dataclass(slots=True)
class InferenceResult:
    request_id: str
    output: TensorTree
    status: RequestStatus = RequestStatus.SUCCEEDED
    queue_time_s: float = 0.0
    execution_time_s: float = 0.0
    metadata: Metadata = field(default_factory=dict)
