"""Planner input with observation and execution lineage."""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from embodied_runtime.types import Metadata, TensorTree

from ._validation import identifier, metadata, observation_snapshot, timestamp
from .feedback import PlanFeedback
from .goal import TaskGoal


@dataclass(frozen=True, slots=True)
class PlanRequest:
    goal: TaskGoal
    observation: TensorTree | None = None
    observation_id: str | None = None
    observation_timestamp_s: float | None = None
    active_plan_id: str | None = None
    active_revision: int | None = None
    feedback: tuple[PlanFeedback, ...] = ()
    requested_at_s: float = field(default_factory=time.time)
    metadata: Metadata = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if not isinstance(self.goal, TaskGoal):
            raise TypeError("goal must be a TaskGoal")
        object.__setattr__(self, "request_id", identifier("request_id", self.request_id))
        self._validate_observation()
        self._validate_active_plan()
        self._validate_feedback()
        object.__setattr__(self, "requested_at_s", timestamp("requested_at_s", self.requested_at_s))
        object.__setattr__(self, "metadata", metadata(self.metadata))

    def _validate_observation(self) -> None:
        has_id = self.observation_id is not None
        has_timestamp = self.observation_timestamp_s is not None
        if has_id != has_timestamp:
            raise ValueError("observation_id and observation_timestamp_s must be provided together")
        if self.observation is not None and not has_id:
            raise ValueError("an observation requires observation_id and observation_timestamp_s")
        if self.observation is not None:
            object.__setattr__(self, "observation", observation_snapshot(self.observation))
        if self.observation_id is not None:
            object.__setattr__(
                self, "observation_id", identifier("observation_id", self.observation_id)
            )
            assert self.observation_timestamp_s is not None
            object.__setattr__(
                self,
                "observation_timestamp_s",
                timestamp("observation_timestamp_s", self.observation_timestamp_s),
            )

    def _validate_active_plan(self) -> None:
        has_id = self.active_plan_id is not None
        has_revision = self.active_revision is not None
        if has_id != has_revision:
            raise ValueError("active_plan_id and active_revision must be provided together")
        if self.active_plan_id is None:
            return
        object.__setattr__(
            self, "active_plan_id", identifier("active_plan_id", self.active_plan_id)
        )
        if not isinstance(self.active_revision, int) or isinstance(self.active_revision, bool):
            raise TypeError("active_revision must be an integer")
        if self.active_revision <= 0:
            raise ValueError("active_revision must be greater than zero")

    def _validate_feedback(self) -> None:
        if isinstance(self.feedback, (str, bytes)) or not isinstance(self.feedback, Sequence):
            raise TypeError("feedback must be a sequence of PlanFeedback values")
        feedback = tuple(self.feedback)
        for item in feedback:
            if not isinstance(item, PlanFeedback):
                raise TypeError("feedback must contain only PlanFeedback values")
            if item.task_id != self.goal.task_id or item.session_id != self.goal.session_id:
                raise ValueError("feedback task and session must match the request goal")
        object.__setattr__(self, "feedback", feedback)


__all__ = ["PlanRequest"]
