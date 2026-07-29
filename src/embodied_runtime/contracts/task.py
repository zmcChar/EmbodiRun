"""Task-level planning contracts shared by edge and planner runtimes."""

from __future__ import annotations

import copy
import math
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from embodied_runtime._compat import StrEnum

from .types import Metadata, TensorTree

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _identifier(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{name} must be 1-128 characters using letters, digits, '.', '_', ':', or '-'"
        )
    return normalized


def _text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _optional_text(name: str, value: str | None) -> str | None:
    return None if value is None else _text(name, value)


def _timestamp(name: str, value: float) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _metadata(value: Metadata) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return dict(value)


def _observation_snapshot(value: TensorTree) -> TensorTree:
    try:
        return copy.deepcopy(value)
    except Exception as error:
        raise TypeError("observation must support a safe snapshot") from error


class PlanStepStatus(StrEnum):
    """Lifecycle states reported for one task-level plan step."""

    PENDING = "pending"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TaskGoal:
    """One episode-scoped objective submitted to a task planner."""

    task_id: str
    session_id: str
    instruction: str
    allowed_skills: tuple[str, ...] = ()
    created_at_s: float = field(default_factory=time.time)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _identifier("task_id", self.task_id))
        object.__setattr__(self, "session_id", _identifier("session_id", self.session_id))
        object.__setattr__(self, "instruction", _text("instruction", self.instruction))
        if isinstance(self.allowed_skills, (str, bytes)) or not isinstance(
            self.allowed_skills, Sequence
        ):
            raise TypeError("allowed_skills must be a sequence of skill identifiers")
        skills = tuple(_identifier("allowed skill", skill) for skill in self.allowed_skills)
        if len(set(skills)) != len(skills):
            raise ValueError("allowed_skills must not contain duplicates")
        object.__setattr__(self, "allowed_skills", skills)
        object.__setattr__(
            self,
            "created_at_s",
            _timestamp("created_at_s", self.created_at_s),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class PlanStep:
    """A sequential, low-level instruction selected by a task planner."""

    step_id: str
    instruction: str
    skill: str | None = None
    success_criteria: str | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _identifier("step_id", self.step_id))
        object.__setattr__(self, "instruction", _text("instruction", self.instruction))
        if self.skill is not None:
            object.__setattr__(self, "skill", _identifier("skill", self.skill))
        object.__setattr__(
            self,
            "success_criteria",
            _optional_text("success_criteria", self.success_criteria),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class PlanFeedback:
    """A traceable status report for one step of one plan revision."""

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
        object.__setattr__(self, "task_id", _identifier("task_id", self.task_id))
        object.__setattr__(self, "session_id", _identifier("session_id", self.session_id))
        object.__setattr__(self, "plan_id", _identifier("plan_id", self.plan_id))
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise TypeError("revision must be an integer")
        if self.revision <= 0:
            raise ValueError("revision must be greater than zero")
        object.__setattr__(self, "step_id", _identifier("step_id", self.step_id))
        try:
            status = PlanStepStatus(self.status)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported plan step status: {self.status!r}") from error
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "timestamp_s",
            _timestamp("timestamp_s", self.timestamp_s),
        )
        object.__setattr__(self, "message", _optional_text("message", self.message))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class PlanRequest:
    """A planner input snapshot, including observation and execution lineage."""

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
        object.__setattr__(self, "request_id", _identifier("request_id", self.request_id))

        has_observation_lineage = self.observation_id is not None
        has_observation_timestamp = self.observation_timestamp_s is not None
        if has_observation_lineage != has_observation_timestamp:
            raise ValueError("observation_id and observation_timestamp_s must be provided together")
        if self.observation is not None and not has_observation_lineage:
            raise ValueError("an observation requires observation_id and observation_timestamp_s")
        if self.observation is not None:
            object.__setattr__(
                self,
                "observation",
                _observation_snapshot(self.observation),
            )
        if self.observation_id is not None:
            object.__setattr__(
                self,
                "observation_id",
                _identifier("observation_id", self.observation_id),
            )
            assert self.observation_timestamp_s is not None
            object.__setattr__(
                self,
                "observation_timestamp_s",
                _timestamp("observation_timestamp_s", self.observation_timestamp_s),
            )

        has_active_plan = self.active_plan_id is not None
        has_active_revision = self.active_revision is not None
        if has_active_plan != has_active_revision:
            raise ValueError("active_plan_id and active_revision must be provided together")
        if self.active_plan_id is not None:
            object.__setattr__(
                self,
                "active_plan_id",
                _identifier("active_plan_id", self.active_plan_id),
            )
            if not isinstance(self.active_revision, int) or isinstance(self.active_revision, bool):
                raise TypeError("active_revision must be an integer")
            if self.active_revision <= 0:
                raise ValueError("active_revision must be greater than zero")

        if isinstance(self.feedback, (str, bytes)) or not isinstance(self.feedback, Sequence):
            raise TypeError("feedback must be a sequence of PlanFeedback values")
        feedback = tuple(self.feedback)
        for item in feedback:
            if not isinstance(item, PlanFeedback):
                raise TypeError("feedback must contain only PlanFeedback values")
            if item.task_id != self.goal.task_id or item.session_id != self.goal.session_id:
                raise ValueError("feedback task and session must match the request goal")
        object.__setattr__(self, "feedback", feedback)
        object.__setattr__(
            self,
            "requested_at_s",
            _timestamp("requested_at_s", self.requested_at_s),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class PlanEnvelope:
    """A versioned plan response with explicit request and observation lineage."""

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
        object.__setattr__(self, "plan_id", _identifier("plan_id", self.plan_id))
        object.__setattr__(self, "request_id", _identifier("request_id", self.request_id))
        object.__setattr__(self, "task_id", _identifier("task_id", self.task_id))
        object.__setattr__(self, "session_id", _identifier("session_id", self.session_id))
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
        step_ids = tuple(step.step_id for step in steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("plan step_id values must be unique")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(
            self,
            "created_at_s",
            _timestamp("created_at_s", self.created_at_s),
        )
        object.__setattr__(
            self,
            "expires_at_s",
            _timestamp("expires_at_s", self.expires_at_s),
        )
        if self.expires_at_s <= self.created_at_s:
            raise ValueError("expires_at_s must be greater than created_at_s")
        if self.based_on_observation_id is not None:
            object.__setattr__(
                self,
                "based_on_observation_id",
                _identifier(
                    "based_on_observation_id",
                    self.based_on_observation_id,
                ),
            )
        object.__setattr__(self, "metadata", _metadata(self.metadata))


__all__ = [
    "PlanEnvelope",
    "PlanFeedback",
    "PlanRequest",
    "PlanStep",
    "PlanStepStatus",
    "TaskGoal",
]
