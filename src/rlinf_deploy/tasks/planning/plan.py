"""Versioned task-plan response with explicit lineage."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from rlinf_deploy.types import Metadata

from ._validation import identifier, metadata, timestamp
from .step import PlanStep


@dataclass(frozen=True, slots=True)
class PlanEnvelope:
    request_id: str
    task_id: str
    session_id: str
    revision: int
    steps: tuple[PlanStep, ...]
    created_at_s: float
    expires_at_s: float
    based_on_observation_id: str | None = None
    metadata: Metadata = field(default_factory=dict)
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        for name in ("plan_id", "request_id", "task_id", "session_id"):
            object.__setattr__(self, name, identifier(name, getattr(self, name)))
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("revision must be an integer")
        if self.revision <= 0:
            raise ValueError("revision must be greater than zero")
        if isinstance(self.steps, (str, bytes)) or not isinstance(self.steps, Sequence):
            raise TypeError("steps must be a sequence of PlanStep values")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("a plan must contain at least one step")
        if any(not isinstance(step, PlanStep) for step in steps):
            raise TypeError("steps must contain only PlanStep values")
        if len({step.step_id for step in steps}) != len(steps):
            raise ValueError("plan step_id values must be unique")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "created_at_s", timestamp("created_at_s", self.created_at_s))
        object.__setattr__(self, "expires_at_s", timestamp("expires_at_s", self.expires_at_s))
        if self.expires_at_s <= self.created_at_s:
            raise ValueError("expires_at_s must be greater than created_at_s")
        if self.based_on_observation_id is not None:
            object.__setattr__(
                self,
                "based_on_observation_id",
                identifier("based_on_observation_id", self.based_on_observation_id),
            )
        object.__setattr__(self, "metadata", metadata(self.metadata))


__all__ = ["PlanEnvelope"]
