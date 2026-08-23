"""Episode-scoped objective submitted to a task planner."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from rlinf_deploy.types import Metadata

from ._validation import identifier, metadata, text, timestamp


@dataclass(frozen=True, slots=True)
class TaskGoal:
    task_id: str
    session_id: str
    instruction: str
    allowed_skills: tuple[str, ...] = ()
    created_at_s: float = field(default_factory=time.time)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier("task_id", self.task_id))
        object.__setattr__(self, "session_id", identifier("session_id", self.session_id))
        object.__setattr__(self, "instruction", text("instruction", self.instruction))
        if isinstance(self.allowed_skills, (str, bytes)) or not isinstance(
            self.allowed_skills, Sequence
        ):
            raise TypeError("allowed_skills must be a sequence of skill identifiers")
        skills = tuple(identifier("allowed skill", skill) for skill in self.allowed_skills)
        if len(set(skills)) != len(skills):
            raise ValueError("allowed_skills must not contain duplicates")
        object.__setattr__(self, "allowed_skills", skills)
        object.__setattr__(self, "created_at_s", timestamp("created_at_s", self.created_at_s))
        object.__setattr__(self, "metadata", metadata(self.metadata))


__all__ = ["TaskGoal"]
