"""Status feedback for one task-plan step."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from embodied_runtime.types import Metadata

from ._validation import identifier, metadata, optional_text, timestamp
from .status import PlanStepStatus


@dataclass(frozen=True, slots=True)
class PlanFeedback:
    task_id: str
    session_id: str
    plan_id: str
    revision: int
    step_id: str
    status: PlanStepStatus
    timestamp_s: float = field(default_factory=time.time)
    message: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("task_id", "session_id", "plan_id", "step_id"):
            object.__setattr__(self, name, identifier(name, getattr(self, name)))
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("revision must be an integer")
        if self.revision <= 0:
            raise ValueError("revision must be greater than zero")
        try:
            status = PlanStepStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported plan step status: {self.status!r}") from error
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "timestamp_s", timestamp("timestamp_s", self.timestamp_s))
        object.__setattr__(self, "message", optional_text("message", self.message))
        object.__setattr__(self, "metadata", metadata(self.metadata))


__all__ = ["PlanFeedback"]
