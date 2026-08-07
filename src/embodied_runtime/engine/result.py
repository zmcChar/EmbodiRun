"""Output envelope returned by inference endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from embodied_runtime.types import Metadata

from .status import RequestStatus

OutputT = TypeVar("OutputT")


@dataclass(slots=True)
class InferenceResult(Generic[OutputT]):
    request_id: str
    output: OutputT
    status: RequestStatus = RequestStatus.SUCCEEDED
    queue_time_s: float = 0.0
    execution_time_s: float = 0.0
    metadata: Metadata = field(default_factory=dict)


__all__ = ["InferenceResult"]
